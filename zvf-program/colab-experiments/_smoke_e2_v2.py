#!/usr/bin/env -S colab run --gpu A100 --session e2-v2-smoke
"""Smoke test for E2 production v2 (5 seeds, harder task).

Same as e2_lora_vs_fullft_4b_v2.py but with:
  - 1 seed, 2 steps, 4 heldout problems
  - goal: catch setup issues in <2 min
  - NOT a real experiment — results are not persisted

Run:
  colab run --gpu A100 --session e2-v2-smoke _smoke_e2_v2.py
"""
import json, re, random, statistics, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "torchao>=0.16"], check=False)
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEEDS = [0]
G, BATCH, MAX_NEW = 8, 2, 96
LR_LORA, LR_FULL = 1e-4, 1e-6
STEPS = 2
HELDOUT_N = 4
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[smoke-v2] MODEL={MODEL} STEPS={STEPS} HELDOUT_N={HELDOUT_N}", flush=True)
print(f"[smoke-v2] LORA_TARGETS={LORA_TARGETS}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem():
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content":
          f"Compute {q}. Show your reasoning, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.findall(r"-?\d+", t.split("####")[-1])
    return int(m[0]) if m else None

def gen_group(model, prompt, gold):
    model.eval()
    enc = tok([prompt] * G, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
    gens = out[:, enc.input_ids.shape[1]:]
    rewards = [1.0 if parse(tok.decode(g, skip_special_tokens=True)) == gold
               else 0.0 for g in gens]
    return enc.input_ids[0], gens, rewards

def seq_logprob(model, pids, gen_row):
    model.train()
    gen_row = gen_row[gen_row != PAD]
    if gen_row.numel() == 0: return None
    ids = torch.cat([pids, gen_row]).unsqueeze(0)
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, pids.shape[0] - 1:].sum()

@torch.no_grad()
def heldout_acc(model, evalset):
    model.eval(); correct = 0
    for q, gold in evalset:
        enc = tok([prompt_of(q)], return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                             pad_token_id=PAD)
        pred = parse(tok.decode(out[0, enc.input_ids.shape[1]:],
                                 skip_special_tokens=True))
        if pred == gold: correct += 1
    return correct / len(evalset)

def run(mode, seed):
    print(f"[smoke-v2] === mode={mode} seed={seed} ===", flush=True)
    random.seed(seed); torch.manual_seed(seed)
    evalset = [problem() for _ in range(HELDOUT_N)]
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    print(f"[smoke-v2] model loaded, mem={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    if mode == "lora":
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg); lr = LR_LORA
    else:
        model = base; lr = LR_FULL
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke-v2] mode={mode} trainable_params={n_train:,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    pre = heldout_acc(model, evalset)
    print(f"[smoke-v2] mode={mode} pre={pre:.3f}", flush=True)
    for step in range(1, STEPS+1):
        opt.zero_grad(set_to_none=True)
        losses, zv, rewards_all = [], 0, []
        for _ in range(BATCH):
            q, gold = problem()
            pids, gens, rewards = gen_group(model, prompt_of(q), gold)
            rewards_all += rewards
            m = sum(rewards)/G; v = statistics.pvariance(rewards); s = v**0.5
            if v == 0.0: zv += 1; continue
            for i in range(G):
                adv = (rewards[i] - m)/(s + 1e-6)
                if adv:
                    lp = seq_logprob(model, pids, gens[i])
                    if lp is not None: losses.append(-adv * lp)
        if losses:
            torch.stack(losses).sum().backward(); opt.step()
        print(f"[smoke-v2] mode={mode} step={step} mem={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    post = heldout_acc(model, evalset)
    print(f"[smoke-v2] mode={mode} pre={post:.3f} delta={post-pre:+.3f}", flush=True)
    del base, model, opt; torch.cuda.empty_cache()
    return {"mode": mode, "pre": pre, "post": post, "delta": post - pre}

results = []
for mode in ("lora", "full"):
    for seed in SEEDS:
        results.append(run(mode, seed))

print("SMOKE_RESULT_V2 " + json.dumps(results), flush=True)