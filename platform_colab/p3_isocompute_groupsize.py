"""P3 Deep Review 1ed8fa8c follow-up: iso-compute GRPO group-size sweep on Colab.

Same minimal GRPO as experiments/modal/modal_groupsize_zvf_sweep.py (Qwen2.5-0.5B,
LoRA r16 on q/v, Adam 1e-4, T=1.0, sequence-mean log-prob, no KL/clip, one
step per batch), with three changes requested by the review:
  * iso-compute: ROLLOUTS completions per step for every G (prompts = ROLLOUTS/G),
    so all arms see the same generated rollouts and optimizer steps;
  * an unsaturated task (TASK=mul2: a*b with a,b in [10,99]) so G can matter;
  * WHITEN=0 arms use A = r - mean (no 1/sigma), isolating whitening.
Per step it logs per-prompt pass rates, so ZVF can be compared with the
heterogeneous closed form mean_x[p_x^G + (1-p_x)^G].

Env vars: MODE=probe|train, TASK, GS (comma list), SEEDS, WHITEN (0/1),
STEPS, ROLLOUTS, OUT (dir). One JSON per (G, seed, whiten) in OUT.
"""
import json
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B"
TASK = os.environ.get("TASK", "mul2")
MODE = os.environ.get("MODE", "train")
GS = [int(g) for g in os.environ.get("GS", "2,4,8,16,32").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123,456").split(",")]
WHITEN = os.environ.get("WHITEN", "1") == "1"
STEPS = int(os.environ.get("STEPS", "60"))
ROLLOUTS = int(os.environ.get("ROLLOUTS", "128"))
OUT = os.environ.get("OUT", "/content/p3_out")
MAX_NEW = 12
EPS = 1e-6
N_HELDOUT = 400
device = "cuda"


def problem(rng):
    if TASK == "add2":
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        return f"What is {a} + {b}? Answer with just the number.", a + b
    if TASK == "add3":
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        return f"What is {a} + {b}? Answer with just the number.", a + b
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    return f"What is {a} * {b}? Answer with just the number.", a * b


def parse(text):
    nums = re.findall(r"-?\d+", text)
    return int(nums[-1]) if nums else None


def load(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device)
    return tok, model


@torch.no_grad()
def heldout_acc(tok, model, seed):
    rng = random.Random(10_000 + seed)
    probs = [problem(rng) for _ in range(N_HELDOUT)]
    correct = 0
    for i in range(0, N_HELDOUT, 64):
        chunk = probs[i:i + 64]
        enc = tok([q for q, _ in chunk], return_tensors="pt", padding=True).to(device)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        correct += sum(parse(t) == ans for t, (_, ans) in zip(texts, chunk))
    return correct / N_HELDOUT


def run(group_size, seed):
    tok, model = load(seed)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    rng = random.Random(seed)
    n_prompts = ROLLOUTS // group_size
    pre = heldout_acc(tok, model, seed)
    log, t0 = [], time.time()
    for step in range(STEPS):
        model.eval()
        probs = [problem(rng) for _ in range(n_prompts)]
        prompts = [q for q, _ in probs for _ in range(group_size)]
        answers = [a for _, a in probs for _ in range(group_size)]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                                 top_p=1.0, pad_token_id=tok.pad_token_id)
        comp = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        r = np.array([1.0 if parse(t) == a else 0.0 for t, a in zip(texts, answers)], dtype=np.float32)
        R = r.reshape(n_prompts, group_size)
        gmean, gstd = R.mean(1, keepdims=True), R.std(1, keepdims=True)
        A = (R - gmean) / (gstd + EPS) if WHITEN else (R - gmean)
        adv = torch.tensor(A.reshape(-1), device=device)
        model.train()
        full = torch.cat([enc["input_ids"], comp], dim=1)
        attn = (full != tok.pad_token_id).long()
        plen, B = enc["input_ids"].shape[1], full.shape[0]
        opt.zero_grad()
        for cs in range(0, B, 16):
            f, a = full[cs:cs + 16], attn[cs:cs + 16]
            logits = model(input_ids=f, attention_mask=a).logits[:, :-1, :].float()
            tgt = f[:, 1:]
            cm = torch.zeros_like(tgt, dtype=torch.float32)
            cm[:, plen - 1:] = 1.0
            cm = cm * (tgt != tok.pad_token_id).float()
            tok_logp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            seq_logp = (tok_logp * cm).sum(1) / cm.sum(1).clamp(min=1.0)
            (-(adv[cs:cs + 16] * seq_logp).sum() / B).backward()
            del logits, tok_logp
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        p_x = R.mean(1)
        log.append({"step": step, "mean_reward": float(r.mean()),
                    "zvf": float((gstd[:, 0] <= EPS).mean()),
                    "zvf_closed_form_hetero": float(np.mean(p_x ** group_size + (1 - p_x) ** group_size)),
                    "grad_norm": gnorm, "p_x": [round(float(v), 4) for v in p_x]})
    model.eval()
    post = heldout_acc(tok, model, seed)
    out = {"experiment": "p3_isocompute_groupsize", "task": TASK, "model": MODEL, "group_size": group_size,
           "seed": seed, "whiten": WHITEN, "steps": STEPS, "rollouts_per_step": ROLLOUTS,
           "prompts_per_step": n_prompts, "heldout_pre": pre, "heldout_post": post,
           "last10_avg": float(np.mean([s["mean_reward"] for s in log[-10:]])),
           "mean_zvf": float(np.mean([s["zvf"] for s in log])),
           "elapsed_s": time.time() - t0, "gpu": torch.cuda.get_device_name(0), "step_log": log}
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{TASK}_G{group_size}_s{seed}_w{int(WHITEN)}.json", "w") as fh:
        json.dump(out, fh)
    print(f"G={group_size} seed={seed} whiten={WHITEN} pre={pre:.3f} post={post:.3f} "
          f"last10={out['last10_avg']:.3f} zvf={out['mean_zvf']:.3f} t={out['elapsed_s']:.0f}s", flush=True)
    del model, opt
    torch.cuda.empty_cache()


if MODE == "probe":
    tok, model = load(0)
    for task in ("add2", "add3", "mul2"):
        TASK = task
        print(task, "greedy heldout acc", heldout_acc(tok, model, 0), flush=True)
else:
    for seed in SEEDS:
        for g in GS:
            if os.path.exists(f"{OUT}/{TASK}_G{g}_s{seed}_w{int(WHITEN)}.json"):
                continue
            run(g, seed)
