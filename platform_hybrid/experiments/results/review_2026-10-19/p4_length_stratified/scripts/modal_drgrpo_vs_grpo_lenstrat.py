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

app = modal.App("tinkerrl-drgrpo-vs-grpo-lenstrat")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.3.0",
    "transformers>=4.46.0",
    "peft>=0.13.0",
    "numpy>=1.26.0,<2.0.0",
    "accelerate>=1.0.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.26.0",
    "wandb>=0.16.0",
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={RESULTS_DIR: results_vol},
    retries=1,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
)
def run_arm(algo: str, seed: int, max_new: int) -> dict:
    MAX_NEW = max_new
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

    import wandb

    wandb.init(
        project="tinkerrl-drgrpo-lenstrat",
        name=f"{algo}_s{seed}",
        config={"algo": algo, "seed": seed, "model": MODEL, "max_new": MAX_NEW},
    )

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device),
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
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
                cm[:, plen - 1 :] = 1.0
                cm = cm * (tgt != tok.pad_token_id).float()
                outs_lp.append(lp[:, plen - 1 :])
                outs_m.append(cm[:, plen - 1 :])
                del logits
        return torch.cat(outs_lp), torch.cat(outs_m)

    step_log = []
    t0 = time.time()
    for step in range(N_STEPS):
        probs = sample(N_PROMPTS)
        prompts, meta = [], []
        for a, b in probs:
            for _ in range(GROUP):
                prompts.append(mp(a, b))
                meta.append((a, b))
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.pad_token_id,
            )
        comp_ids = gen[:, plen:]
        comp_txt = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array(
            [
                1.0 if (parse(t) is not None and parse(t) == a + b) else 0.0
                for t, (a, b) in zip(comp_txt, meta, strict=False)
            ],
            dtype=np.float32,
        )

        R = rewards.reshape(N_PROMPTS, GROUP)
        gmean = R.mean(1, keepdims=True)
        gstd = R.std(1, keepdims=True)
        zvf = float((gstd[:, 0] <= EPS).mean())
        if algo == "grpo":
            A = (R - gmean) / (gstd + EPS)  # std-normalized
        else:  # dr_grpo
            A = R - gmean  # NO std divisor
        adv = torch.tensor(A.reshape(-1), dtype=torch.float32, device=device)

        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()
        B = full.shape[0]
        with torch.no_grad():
            old_lp, mask = token_logps(full, attn, plen, want_grad=False)
        comp_len = float(mask.sum(1).mean().item())
        _L = mask.sum(1).float().cpu().numpy()
        _c = np.asarray(rewards) > 0.5
        lenstrat = {
            "len_correct_mean": float(_L[_c].mean()) if _c.any() else None,
            "len_incorrect_mean": float(_L[~_c].mean()) if (~_c).any() else None,
            "n_correct": int(_c.sum()),
            "n_incorrect": int((~_c).sum()),
            "frac_at_cap": float((_L >= MAX_NEW).mean()),
            "frac_at_cap_correct": float((_L[_c] >= MAX_NEW).mean()) if _c.any() else None,
            "frac_at_cap_incorrect": float((_L[~_c] >= MAX_NEW).mean()) if (~_c).any() else None,
            "lengths": _L.astype(int).tolist(),
            "rewards": _c.astype(int).tolist(),
        }

        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            # chunked token-level clipped surrogate with arm-specific normalization
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                f, a = full[cs:ce], attn[cs:ce]
                import torch.nn.functional as F2

                logits = model(input_ids=f, attention_mask=a).logits[:, :-1, :].float()
                tgt = f[:, 1:]
                lp = (
                    F2.log_softmax(logits, dim=-1)
                    .gather(-1, tgt.unsqueeze(-1))
                    .squeeze(-1)[:, plen - 1 :]
                )
                m = mask[cs:ce]
                old = old_lp[cs:ce]
                a_i = adv[cs:ce].unsqueeze(1)  # [chunk,1] broadcast over tokens
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

        step_log.append(
            {
                "step": step,
                "mean_reward": float(rewards.mean()),
                "zvf": zvf,
                "mean_comp_len": comp_len,
                **lenstrat,
            }
        )
        wandb.log({k: v for k, v in step_log[-1].items() if not isinstance(v, list)})

    # held-out greedy eval
    model.eval()
    correct, total = 0, 200
    held = sample(total)
    for i in range(0, total, 32):
        ch = held[i : i + 32]
        e = tok([mp(a, b) for (a, b) in ch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            g = model.generate(
                **e, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id
            )
        for t, (a, b) in zip(
            tok.batch_decode(g[:, e["input_ids"].shape[1] :], skip_special_tokens=True),
            ch,
            strict=False,
        ):
            if parse(t) is not None and parse(t) == a + b:
                correct += 1
    import numpy as _np

    res = {
        "experiment": "drgrpo_vs_grpo",
        "algo": algo,
        "seed": seed,
        "model": MODEL,
        "max_new": MAX_NEW,
        "heldout_acc": correct / total,
        "last10_avg": float(_np.mean([s["mean_reward"] for s in step_log[-10:]])),
        "mean_comp_len": float(_np.mean([s["mean_comp_len"] for s in step_log])),
        "mean_zvf": float(_np.mean([s["zvf"] for s in step_log])),
        "elapsed_seconds": time.time() - t0,
        "step_log": step_log,
    }
    os.makedirs(f"{RESULTS_DIR}/drgrpo_lenstrat_cap{MAX_NEW}", exist_ok=True)
    with open(f"{RESULTS_DIR}/drgrpo_lenstrat_cap{MAX_NEW}/{algo}_s{seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    results_vol.commit()
    print(
        f"[{algo} seed={seed}] heldout={res['heldout_acc']:.3f} last10={res['last10_avg']:.3f} len={res['mean_comp_len']:.2f}"
    )

    try:
        if False:  # review rerun: no HF push
            repo_id = f"arvindcr4/tinkerrl-drgrpo-vs-grpo-{algo}-s{seed}"
            model.push_to_hub(repo_id, token=os.environ["HF_TOKEN"], private=True)
            tok.push_to_hub(repo_id, token=os.environ["HF_TOKEN"], private=True)
            print(f"Pushed model to HF Hub: {repo_id}")
        else:
            print("HF_TOKEN not found in environment, skipping push_to_hub")
    except Exception as e:
        print(f"Failed to push to HF Hub: {e}")

    wandb.finish()
    return res


@app.local_entrypoint()
def main(max_new: int = 0, out: str = ""):
    mn = max_new or MAX_NEW
    jobs = [(al, s, mn) for al in ALGOS for s in SEEDS]
    print(f"{len(jobs)} runs ({ALGOS} x {SEEDS}), MAX_NEW={mn}")
    results = [r for r in run_arm.starmap(jobs, return_exceptions=True) if isinstance(r, dict)]
    with open(out, "w") as f:
        json.dump({"max_new": mn, "model": MODEL, "runs": results}, f)
    print("saved", out, len(results), "runs")
