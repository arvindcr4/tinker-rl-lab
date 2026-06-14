"""Modal GRPO group-size sweep with per-step ZVF / entropy / advantage-variance
instrumentation and a measured held-out evaluation.

Purpose (closes two adversarial-review gaps):
  * Pillar 3 (trainability / group size): produces a *measured* multi-seed
    G-sweep with held-out accuracy, replacing the hardcoded FALLBACK_ROWS in
    experiments/group_size_token_normalized.py.
  * Pillar 2 (ZVF): logs per-step Zero-Variance Fraction together with the
    confounders the withdrawn partial-correlation table needs (batch mean
    reward, policy entropy, advantage variance), so the partial correlation can
    be computed from real per-group rollout logs.

Self-contained minimal GRPO with the canonical group-relative advantage
A = (r - mean_g)/(std_g + eps) (i.e. WITH the std normalization; this is vanilla
GRPO, NOT Dr. GRPO -- Dr. GRPO drops the std divisor), no KL/clip, on the ungated
Qwen2.5-0.5B with a verifiable binary correctness reward on a+b arithmetic, so
rewards have real within-group variance (meaningful ZVF).

Usage:
  modal run experiments/modal/modal_groupsize_zvf_sweep.py
"""
import modal
import json
import os
import time

app = modal.App("tinkerrl-groupsize-zvf")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0",
        "transformers>=4.46.0",
        "peft>=0.13.0",
        "numpy>=1.26.0,<2.0.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.26.0",
    )
)

GROUP_SIZES = [2, 4, 8, 16]
SEEDS = [42, 123, 456]
MODEL = "Qwen/Qwen2.5-0.5B"
N_STEPS = 40
N_PROMPTS = 16          # prompts per optimizer step
MAX_NEW = 10
EPS = 1e-6


@app.function(image=image, gpu="A10G", timeout=3600, volumes={RESULTS_DIR: results_vol}, retries=1)
def run_one(group_size: int, seed: int) -> dict:
    import os as _os
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import random
    import re
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def make_prompt(a, b):
        return f"What is {a} + {b}? Answer with just the number."

    def parse(text):
        nums = re.findall(r"-?\d+", text)
        return int(nums[-1]) if nums else None

    def sample_batch(n):
        return [(random.randint(1, 99), random.randint(1, 99)) for _ in range(n)]

    step_log = []
    t0 = time.time()
    for step in range(N_STEPS):
        probs = sample_batch(N_PROMPTS)
        # build G copies of each prompt
        prompts, meta = [], []
        for (a, b) in probs:
            for _ in range(group_size):
                prompts.append(make_prompt(a, b))
                meta.append((a, b))
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                top_p=1.0, pad_token_id=tok.pad_token_id,
            )
        comp_ids = gen[:, enc["input_ids"].shape[1]:]
        comp_texts = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array([
            1.0 if (parse(t) is not None and parse(t) == a + b) else 0.0
            for t, (a, b) in zip(comp_texts, meta)
        ], dtype=np.float32)

        # group-relative advantages + ZVF
        R = rewards.reshape(N_PROMPTS, group_size)
        gmean = R.mean(axis=1, keepdims=True)
        gstd = R.std(axis=1, keepdims=True)
        zvf = float((gstd[:, 0] <= EPS).mean())          # fraction of zero-variance groups
        adv = (R - gmean) / (gstd + EPS)
        adv_flat = torch.tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        nz = adv.reshape(-1)[np.abs(adv.reshape(-1)) > EPS]
        adv_var = float(np.var(nz)) if nz.size > 0 else 0.0

        # chunked forward/backward (gradient accumulation) to bound peak memory:
        # log_softmax over the full vocab (~151k) would OOM at large G, so we
        # process the batch in chunks and accumulate gradients.
        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        plen = enc["input_ids"].shape[1]
        B = full.shape[0]
        CHUNK = 8
        opt.zero_grad()
        ent_sum, ent_count = 0.0, 0.0
        for cs in range(0, B, CHUNK):
            ce = min(cs + CHUNK, B)
            f, a = full[cs:ce], attn[cs:ce]
            logits = model(input_ids=f, attention_mask=a).logits[:, :-1, :].float()
            tgt = f[:, 1:]
            cm = torch.zeros_like(tgt, dtype=torch.float32)
            cm[:, plen - 1:] = 1.0
            cm = cm * (tgt != tok.pad_token_id).float()
            logp = F.log_softmax(logits, dim=-1)
            tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            seq_logp = (tok_logp * cm).sum(dim=1) / cm.sum(dim=1).clamp(min=1.0)
            loss_chunk = -(adv_flat[cs:ce] * seq_logp).sum() / B   # sum of chunks = full mean
            loss_chunk.backward()
            with torch.no_grad():
                ent_tok = -(logp.exp() * logp).sum(dim=-1)
                ent_sum += float((ent_tok * cm).sum().item())
                ent_count += float(cm.sum().item())
            del logits, logp, tok_logp, ent_tok, seq_logp
        entropy = ent_sum / max(ent_count, 1.0)
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0))
        opt.step()

        step_log.append({
            "step": step,
            "zvf": zvf,
            "mean_reward": float(rewards.mean()),
            "entropy": entropy,
            "advantage_variance": adv_var,
            "grad_norm": gnorm,
        })

    # measured held-out eval (greedy on fresh problems)
    model.eval()
    correct, total = 0, 200
    heldout = sample_batch(total)
    for i in range(0, total, 32):
        chunk = heldout[i:i + 32]
        enc = tok([make_prompt(a, b) for (a, b) in chunk], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for t, (a, b) in zip(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True), chunk):
            if parse(t) is not None and parse(t) == a + b:
                correct += 1
    heldout_acc = correct / total
    last10 = float(np.mean([s["mean_reward"] for s in step_log[-10:]]))
    mean_zvf = float(np.mean([s["zvf"] for s in step_log]))

    result = {
        "experiment": "groupsize_zvf_sweep",
        "model": MODEL, "task": "arithmetic_correctness",
        "group_size": group_size, "seed": seed,
        "n_steps": N_STEPS, "n_prompts": N_PROMPTS,
        "heldout_acc": heldout_acc, "last10_avg": last10, "mean_zvf": mean_zvf,
        "elapsed_seconds": time.time() - t0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "step_log": step_log,
    }
    os.makedirs(f"{RESULTS_DIR}/groupsize_zvf", exist_ok=True)
    with open(f"{RESULTS_DIR}/groupsize_zvf/G{group_size}_s{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    print(f"[G={group_size} seed={seed}] heldout={heldout_acc:.3f} last10={last10:.3f} meanZVF={mean_zvf:.3f}")
    return result


@app.local_entrypoint()
def main():
    import numpy as np
    jobs = [(g, s) for g in GROUP_SIZES for s in SEEDS]
    print(f"Launching {len(jobs)} GRPO runs: G in {GROUP_SIZES} x seeds {SEEDS}")
    results = list(run_one.starmap(jobs))

    # aggregate per G
    by_g = {}
    for r in results:
        by_g.setdefault(r["group_size"], []).append(r)
    summary = {}
    print("\n=== GROUP-SIZE SWEEP (measured) ===")
    for g in sorted(by_g):
        rs = by_g[g]
        accs = [r["heldout_acc"] for r in rs]
        zvfs = [r["mean_zvf"] for r in rs]
        l10 = [r["last10_avg"] for r in rs]
        summary[g] = {
            "n_seeds": len(rs),
            "heldout_acc_mean": float(np.mean(accs)),
            "heldout_acc_se": float(np.std(accs) / np.sqrt(len(accs))),
            "last10_mean": float(np.mean(l10)),
            "mean_zvf": float(np.mean(zvfs)),
        }
        print(f"  G={g}: heldout={np.mean(accs):.3f}±{np.std(accs)/np.sqrt(len(accs)):.3f} "
              f"last10={np.mean(l10):.3f} meanZVF={np.mean(zvfs):.3f} (n={len(rs)})")

    out = {"summary": summary, "runs": results}
    with open("experiments/results/groupsize_zvf_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved experiments/results/groupsize_zvf_sweep.json")
