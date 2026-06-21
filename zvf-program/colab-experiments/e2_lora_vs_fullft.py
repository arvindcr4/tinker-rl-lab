"""E2: LoRA vs FULL fine-tuning under identical GRPO.

Colab-only: Tinker is LoRA-only by construction, so the full-FT arm is
physically impossible there. We hold task, data, seed, compute fixed and flip
ONLY the LoRA<->full axis, then compare:
  * ZVF / GU trajectory  (does full-FT escape cold-start collapse differently?)
  * mean training reward trajectory
  * held-out accuracy delta (pre vs post) on a fixed eval set.

Run:  colab run --gpu T4 --timeout 1200 e2_lora_vs_fullft.py
"""
import json, re, random, statistics, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft"], check=False)
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
G, BATCH, MAX_NEW, LR_LORA, LR_FULL, STEPS = 6, 4, 40, 1e-4, 2e-6, 16
HELDOUT_N = 24
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem():            # fixed "medium" regime where ZVF dynamics are live
    a, b = random.randint(11, 60), random.randint(11, 60)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

def gen_group(model, prompt, gold):
    model.eval()
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [1.0 if parse(t) == gold else 0.0 for t in tok.batch_decode(gens, skip_special_tokens=True)]
    return enc.input_ids[0], gens, rewards

def seq_logprob(model, pids, gen_row):
    model.train()
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0:
        return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, pids.shape[0] - 1:].sum()

@torch.no_grad()
def heldout_acc(model, evalset):
    model.eval()
    correct = 0
    for q, gold in evalset:
        enc = tok([prompt_of(q)], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW, pad_token_id=PAD)
        if parse(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)) == gold:
            correct += 1
    return correct / len(evalset)

def run(mode):
    random.seed(SEED); torch.manual_seed(SEED)
    evalset = [problem() for _ in range(HELDOUT_N)]   # same set both modes (seed reset)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    if mode == "lora":
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg); lr = LR_LORA
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = base; lr = LR_FULL
        n_train = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    pre = heldout_acc(model, evalset)
    traj = []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        rewards_all, losses, zv = [], [], 0
        for _ in range(BATCH):
            q, gold = problem()
            pids, gens, rewards = gen_group(model, prompt_of(q), gold)
            rewards_all += rewards
            m = sum(rewards) / G; v = statistics.pvariance(rewards); s = v ** 0.5
            if v == 0.0:
                zv += 1; continue
            for i in range(G):
                adv = (rewards[i] - m) / (s + 1e-6)
                if adv:
                    lp = seq_logprob(model, pids, gens[i])
                    if lp is not None:
                        losses.append(-adv * lp)
        if losses:
            torch.stack(losses).sum().backward(); opt.step()
        zvf = zv / BATCH
        traj.append({"step": step, "p": round(sum(rewards_all) / len(rewards_all), 3),
                     "zvf": round(zvf, 3), "gu": round(1 - zvf, 3)})
        print(f"[e2:{mode}] step={step:2d} p={traj[-1]['p']:.2f} ZVF={zvf:.2f}", flush=True)
    post = heldout_acc(model, evalset)
    out = {"mode": mode, "trainable_params": n_train, "heldout_pre": round(pre, 3),
           "heldout_post": round(post, 3), "heldout_delta": round(post - pre, 3),
           "mean_zvf": round(statistics.mean(t["zvf"] for t in traj), 3),
           "first3_p": round(statistics.mean(t["p"] for t in traj[:3]), 3),
           "last3_p": round(statistics.mean(t["p"] for t in traj[-3:]), 3)}
    del base, model, opt; torch.cuda.empty_cache()
    return out

results = [run("lora"), run("full")]
summary = {"experiment": "E2_lora_vs_fullft", "model": MODEL, "seed": SEED,
           "steps": STEPS, "arms": results}
print("E2_RESULT " + json.dumps(summary), flush=True)
