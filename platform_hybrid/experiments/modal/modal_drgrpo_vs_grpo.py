"""Modal A/B: vanilla GRPO vs Dr. GRPO, isolating the two Dr. GRPO fixes.

Both arms share EVERYTHING: model (Qwen2.5-0.5B), verifiable arithmetic
correctness reward, generation pipeline, PPO-style token-level clipped surrogate,
K inner epochs, group size, compute budget, evaluation. They differ ONLY in the
two changes that define Dr. GRPO ("GRPO Done Right", Liu et al. 2025):

  (1) advantage std-normalization  (TRL `scale_rewards`):
        GRPO    : A_i = (r_i - mean_g) / (std_g + eps)      # divide by group std
        Dr.GRPO : A_i =  r_i - mean_g                       # no std divisor
  (2) loss length-normalization    (TRL `loss_type`):
        GRPO    : mean_i [ (1/|o_i|) * sum_t L_it ]         # per-response length
        Dr.GRPO : (1/(B*L_const)) * sum_i sum_t L_it        # constant normalizer

So any difference is attributable to the Dr. GRPO formulation, not the stack.
5 seeds -> paired test on held-out accuracy; we also report mean completion
length (Dr. GRPO is designed to remove the length bias).

Usage:
  modal run experiments/modal/modal_drgrpo_vs_grpo.py
"""
import modal
import json
import os
import time

app = modal.App("tinkerrl-drgrpo-vs-grpo")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.3.0", "transformers>=4.46.0", "peft>=0.13.0",
                 "numpy>=1.26.0,<2.0.0", "accelerate>=1.0.0", "safetensors>=0.4.0",
                 "huggingface-hub>=0.26.0", "wandb>=0.16.0")
)

SEEDS = [42, 123, 456, 789, 1024]
ALGOS = ["grpo", "dr_grpo"]
MODEL = "Qwen/Qwen2.5-0.5B"
N_STEPS = 40
N_PROMPTS = 16
GROUP = 8
K_EPOCHS = 2
CLIP = 0.2
MAX_NEW = 10
EPS = 1e-6
CHUNK = 8


@app.function(image=image, gpu="A10G", timeout=3600, volumes={RESULTS_DIR: results_vol}, retries=1, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")])
def run_arm(algo: str, seed: int) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import random
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import wandb
    wandb.init(project="tinkerrl-drgrpo-vs-grpo", name=f"{algo}_s{seed}", config={"algo": algo, "seed": seed, "model": MODEL})

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(device),
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)

    mp = lambda a, b: f"What is {a} + {b}? Answer with just the number."
    def parse(t):
        n = re.findall(r"-?\d+", t)
        return int(n[-1]) if n else None
    sample = lambda n: [(random.randint(1, 99), random.randint(1, 99)) for _ in range(n)]

    def token_logps(full, attn, plen, want_grad):
        """Chunked per-token completion logprobs -> (tok_logp[B,Tc], mask[B,Tc])."""
        B = full.shape[0]
        outs_lp, outs_m = [], []
        ctx = torch.enable_grad() if want_grad else torch.no_grad()
        with ctx:
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                f, a = full[cs:ce], attn[cs:ce]
                logits = model(input_ids=f, attention_mask=a).logits[:, :-1, :].float()
                tgt = f[:, 1:]
                lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                cm = torch.zeros_like(tgt, dtype=torch.float32)
                cm[:, plen - 1:] = 1.0
                cm = cm * (tgt != tok.pad_token_id).float()
                outs_lp.append(lp[:, plen - 1:])
                outs_m.append(cm[:, plen - 1:])
                del logits
        return torch.cat(outs_lp), torch.cat(outs_m)

    step_log = []
    t0 = time.time()
    for step in range(N_STEPS):
        probs = sample(N_PROMPTS)
        prompts, meta = [], []
        for (a, b) in probs:
            for _ in range(GROUP):
                prompts.append(mp(a, b)); meta.append((a, b))
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                                 top_p=1.0, pad_token_id=tok.pad_token_id)
        comp_ids = gen[:, plen:]
        comp_txt = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array([1.0 if (parse(t) is not None and parse(t) == a + b) else 0.0
                            for t, (a, b) in zip(comp_txt, meta)], dtype=np.float32)

        R = rewards.reshape(N_PROMPTS, GROUP)
        gmean = R.mean(1, keepdims=True)
        gstd = R.std(1, keepdims=True)
        zvf = float((gstd[:, 0] <= EPS).mean())
        if algo == "grpo":
            A = (R - gmean) / (gstd + EPS)               # std-normalized
        else:  # dr_grpo
            A = (R - gmean)                              # NO std divisor
        adv = torch.tensor(A.reshape(-1), dtype=torch.float32, device=device)

        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        B = full.shape[0]
        with torch.no_grad():
            old_lp, mask = token_logps(full, attn, plen, want_grad=False)
        comp_len = float(mask.sum(1).mean().item())

        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            # chunked token-level clipped surrogate with arm-specific normalization
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                f, a = full[cs:ce], attn[cs:ce]
                import torch.nn.functional as F2
                logits = model(input_ids=f, attention_mask=a).logits[:, :-1, :].float()
                tgt = f[:, 1:]
                lp = F2.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[:, plen - 1:]
                m = mask[cs:ce]
                old = old_lp[cs:ce]
                a_i = adv[cs:ce].unsqueeze(1)             # [chunk,1] broadcast over tokens
                ratio = torch.exp(lp - old)
                ptl = -torch.min(ratio * a_i, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * a_i) * m
                if algo == "grpo":
                    # per-response length norm, then mean over all B sequences
                    seqloss = ptl.sum(1) / m.sum(1).clamp(min=1.0)
                    loss_chunk = seqloss.sum() / B
                else:
                    # constant normalizer (B * MAX_NEW)
                    loss_chunk = ptl.sum() / (B * MAX_NEW)
                loss_chunk.backward()
                del logits, lp, ptl
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        step_log.append({"step": step, "mean_reward": float(rewards.mean()),
                         "zvf": zvf, "mean_comp_len": comp_len})
        wandb.log(step_log[-1])

    # held-out greedy eval
    model.eval()
    correct, total = 0, 200
    held = sample(total)
    for i in range(0, total, 32):
        ch = held[i:i + 32]
        e = tok([mp(a, b) for (a, b) in ch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            g = model.generate(**e, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for t, (a, b) in zip(tok.batch_decode(g[:, e["input_ids"].shape[1]:], skip_special_tokens=True), ch):
            if parse(t) is not None and parse(t) == a + b:
                correct += 1
    import numpy as _np
    res = {"experiment": "drgrpo_vs_grpo", "algo": algo, "seed": seed, "model": MODEL,
           "heldout_acc": correct / total, "last10_avg": float(_np.mean([s["mean_reward"] for s in step_log[-10:]])),
           "mean_comp_len": float(_np.mean([s["mean_comp_len"] for s in step_log])),
           "mean_zvf": float(_np.mean([s["zvf"] for s in step_log])),
           "elapsed_seconds": time.time() - t0, "step_log": step_log}
    os.makedirs(f"{RESULTS_DIR}/drgrpo", exist_ok=True)
    with open(f"{RESULTS_DIR}/drgrpo/{algo}_s{seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    results_vol.commit()
    print(f"[{algo} seed={seed}] heldout={res['heldout_acc']:.3f} last10={res['last10_avg']:.3f} len={res['mean_comp_len']:.2f}")

    try:
        if "HF_TOKEN" in os.environ:
            repo_id = f"arvindcr4/tinkerrl-drgrpo-vs-grpo-{algo}-s{seed}"
            model.push_to_hub(repo_id, token=os.environ["HF_TOKEN"])
            tok.push_to_hub(repo_id, token=os.environ["HF_TOKEN"])
            print(f"Pushed model to HF Hub: {repo_id}")
        else:
            print("HF_TOKEN not found in environment, skipping push_to_hub")
    except Exception as e:
        print(f"Failed to push to HF Hub: {e}")

    wandb.finish()
    return res


@app.local_entrypoint()
def main():
    import numpy as np
    import math
    jobs = [(al, s) for al in ALGOS for s in SEEDS]
    print(f"GRPO vs Dr.GRPO A/B: {len(jobs)} runs ({ALGOS} x {SEEDS})")
    results = [r for r in run_arm.starmap(jobs) if r]

    by = {al: [r for r in results if r["algo"] == al] for al in ALGOS}
    summ = {}
    for al in ALGOS:
        rs = by[al]
        summ[al] = {"n_seeds": len(rs),
                    "heldout_mean": float(np.mean([r["heldout_acc"] for r in rs])),
                    "heldout_se": float(np.std([r["heldout_acc"] for r in rs], ddof=1) / np.sqrt(len(rs))),
                    "last10_mean": float(np.mean([r["last10_avg"] for r in rs])),
                    "mean_comp_len": float(np.mean([r["mean_comp_len"] for r in rs]))}
    # paired test (dr_grpo - grpo) on held-out
    g = {r["seed"]: r["heldout_acc"] for r in by["grpo"]}
    p = {r["seed"]: r["heldout_acc"] for r in by["dr_grpo"]}
    common = sorted(set(g) & set(p))
    diffs = [p[s] - g[s] for s in common]
    paired = {}
    if len(diffs) >= 2:
        md = float(np.mean(diffs)); sd = float(np.std(diffs, ddof=1))
        t = md / (sd / math.sqrt(len(diffs))) if sd > 0 else float("inf")
        df = len(diffs) - 1
        def betacf(a, b, x):
            MAXIT, E, FP = 200, 3e-12, 1e-300
            qab, qap, qam = a + b, a + 1, a - 1
            c = 1.0; d = 1 - qab * x / qap
            d = 1 / (FP if abs(d) < FP else d); h = d
            for m in range(1, MAXIT):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d
                d = 1 / (FP if abs(d) < FP else d); c = 1 + aa / c
                if abs(c) < FP: c = FP
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d
                d = 1 / (FP if abs(d) < FP else d); c = 1 + aa / c
                if abs(c) < FP: c = FP
                de = d * c; h *= de
                if abs(de - 1) < E: break
            return h
        def betai(a, b, x):
            if x <= 0 or x >= 1: return 0.0 if x <= 0 else 1.0
            bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
            return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b
        pv = betai(df / 2.0, 0.5, df / (df + t * t)) if t != float("inf") else 0.0
        paired = {"metric": "heldout_acc", "n_seeds": len(diffs),
                  "mean_diff_drgrpo_minus_grpo": md, "t": t, "df": df, "p_two_sided": pv}

    out = {"summary": summ, "paired_drgrpo_vs_grpo": paired, "runs": results}
    with open("experiments/results/drgrpo_vs_grpo.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== GRPO vs Dr.GRPO (measured) ===")
    for al in ALGOS:
        s = summ[al]
        print(f"  {al:8s}: heldout={s['heldout_mean']:.3f}+/-{s['heldout_se']:.3f} "
              f"last10={s['last10_mean']:.3f} comp_len={s['mean_comp_len']:.2f} (n={s['n_seeds']})")
    if paired:
        print(f"  paired dr_grpo-grpo heldout diff={paired['mean_diff_drgrpo_minus_grpo']:+.3f} "
              f"t={paired['t']:.3f} df={paired['df']} p={paired['p_two_sided']:.3f}")
    print("Saved experiments/results/drgrpo_vs_grpo.json")
