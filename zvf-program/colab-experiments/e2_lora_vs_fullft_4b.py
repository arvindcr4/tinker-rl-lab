#!/usr/bin/env -S colab run --gpu A100 --session e2-prod-4b
"""E2 production run: LoRA vs full-FT on Qwen3-4B-Instruct-2507.

Production-grade follow-up to the 0.5B pilot in e2_lora_vs_fullft.py.
Differences from pilot:
  - MODEL: Qwen3-4B-Instruct-2507 (4B, text-only, non-thinking mode)
    * not Qwen3.5-4B: that's multimodal (Qwen3_5ForConditionalGeneration)
      and only loads via vLLM with TRL's GRPOTrainer colocate mode
      (known issue vllm-project/vllm#36275). Raw transformers + AutoModelForCausalLM
      breaks on Qwen3.5 multimodal checkpoints.
  - 3 seeds (0, 1, 2) for statistical signal
  - STEPS=40 (was 16): 4B needs more steps to show ZVF/held-out divergence
  - HELDOUT_N=50 (was 24): tighter CI on heldout_delta
  - BATCH=2 (was 4): full-FT on 4B in bf16 + AdamW ~32GB VRAM, tight on A100 40GB
  - MAX_NEW=64 (was 40): Qwen3-4B is more verbose, allow thinking-room
  - LR_FULL=1e-6 (was 2e-6): full-FT on 4B is unstable at higher LR
  - Per-step log lines (matching persist_e2_e7.py regex)
  - Final E2_RESULT {json} line for persist_e2_e7.py to harvest

Tinker contrast: Tinker is LoRA-only, so this script's full-FT arm is the
experiment Tinker structurally can't run. The paper future-work note is
addressed by THIS arm.

Run:
  colab run --gpu A100 --session e2-prod-4b e2_lora_vs_fullft_4b.py

Output: prints E2_RESULT {json} on stdout at end; persist_e2_e7.py reads
this back into results/e2_lora_vs_fullft_4b.json + W&B.
"""
import json, re, random, statistics, subprocess, sys

# peft pulls torchao as a dep; 0.10.0 is incompatible with Colab's torch.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "torchao>=0.16"], check=False)

import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEEDS = [0, 1, 2]
G, BATCH, MAX_NEW = 6, 2, 64
LR_LORA, LR_FULL = 1e-4, 1e-6
STEPS = 40
HELDOUT_N = 50
DEV = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[e2-prod] MODEL={MODEL} SEEDS={SEEDS} STEPS={STEPS} HELDOUT_N={HELDOUT_N}", flush=True)
print(f"[e2-prod] BATCH={BATCH} G={G} MAX_NEW={MAX_NEW} LR_LORA={LR_LORA} LR_FULL={LR_FULL}", flush=True)
print(f"[e2-prod] device={DEV}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem():
    """Mixed-difficulty synthetic arithmetic. a+b in [22,119] = mid range
    where ZVF dynamics are live for 4B (cold-start at extremes)."""
    a, b = random.randint(11, 60), random.randint(11, 60)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    """Qwen3-4B-2507 is non-thinking mode (no <think> blocks), but be defensive."""
    # Strip think block just in case
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
    rewards = [1.0 if parse(tok.decode(g, skip_special_tokens=True)) == gold else 0.0
               for g in gens]
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
        pred = parse(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
        if pred == gold:
            correct += 1
    return correct / len(evalset)

def run(mode, seed):
    print(f"[e2-prod] === seed={seed} mode={mode} ===", flush=True)
    random.seed(seed); torch.manual_seed(seed)
    evalset = [problem() for _ in range(HELDOUT_N)]

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)

    if mode == "lora":
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg); lr = LR_LORA
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model = base; lr = LR_FULL
        n_train = sum(p.numel() for p in model.parameters())

    print(f"[e2-prod] seed={seed} mode={mode} trainable_params={n_train:,}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    pre = heldout_acc(model, evalset)
    print(f"[e2-prod] seed={seed} mode={mode} heldout_pre={pre:.3f}", flush=True)

    traj = []
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        rewards_all, losses, zv = [], [], 0
        for _ in range(BATCH):
            q, gold = problem()
            pids, gens, rewards = gen_group(model, prompt_of(q), gold)
            rewards_all += rewards
            m = sum(rewards) / G
            v = statistics.pvariance(rewards)
            s = v ** 0.5
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
        p_mean = sum(rewards_all) / len(rewards_all) if rewards_all else 0.0
        traj.append({"step": step, "p": round(p_mean, 3),
                     "zvf": round(zvf, 3), "gu": round(1 - zvf, 3)})
        print(f"[e2:{mode}:s{seed}] step={step:2d}/{STEPS} p={p_mean:.2f} ZVF={zvf:.2f}",
              flush=True)

    post = heldout_acc(model, evalset)
    print(f"[e2-prod] seed={seed} mode={mode} heldout_post={post:.3f} delta={post-pre:+.3f}", flush=True)

    out = {"mode": mode, "seed": seed, "trainable_params": n_train,
           "heldout_pre": round(pre, 3), "heldout_post": round(post, 3),
           "heldout_delta": round(post - pre, 3),
           "mean_zvf": round(statistics.mean(t["zvf"] for t in traj), 3),
           "first3_p": round(statistics.mean(t["p"] for t in traj[:3]), 3),
           "last3_p": round(statistics.mean(t["p"] for t in traj[-3:]), 3),
           "trajectory": traj}
    del base, model, opt; torch.cuda.empty_cache()
    return out

# Run all 6 arms: {lora, full} x {0, 1, 2}
results = []
for mode in ("lora", "full"):
    for seed in SEEDS:
        results.append(run(mode, seed))

# Aggregate per-mode
def aggregate(arm_results):
    deltas = [r["heldout_delta"] for r in arm_results]
    zvfs = [r["mean_zvf"] for r in arm_results]
    return {
        "n_seeds": len(arm_results),
        "mean_heldout_delta": round(statistics.mean(deltas), 3),
        "std_heldout_delta": round(statistics.stdev(deltas), 3) if len(deltas) > 1 else 0.0,
        "mean_zvf": round(statistics.mean(zvfs), 3),
        "per_seed": arm_results,
    }

lora_arms = [r for r in results if r["mode"] == "lora"]
full_arms = [r for r in results if r["mode"] == "full"]

summary = {
    "experiment": "E2_lora_vs_fullft_4b",
    "model": MODEL,
    "seeds": SEEDS,
    "steps": STEPS,
    "batch": BATCH,
    "group_size": G,
    "lr_lora": LR_LORA,
    "lr_full": LR_FULL,
    "max_new_tokens": MAX_NEW,
    "heldout_n": HELDOUT_N,
    "lora": aggregate(lora_arms),
    "full": aggregate(full_arms),
    "delta_lora_minus_full": round(
        aggregate(lora_arms)["mean_heldout_delta"] - aggregate(full_arms)["mean_heldout_delta"], 3),
}
print("E2_RESULT " + json.dumps(summary), flush=True)