"""Modal same-stack PPO vs GRPO comparison (de-confounds pillar 1).

The paper's headline PPO-vs-GRPO numbers are stack-confounded (GRPO on the
Tinker API, PPO on Modal H100). This experiment removes that confound: PPO and
GRPO run in the SAME minimal framework, on the SAME model (Qwen2.5-0.5B), SAME
task (arithmetic correctness reward), SAME generation pipeline, SAME evaluation,
SAME compute budget (identical generations/step, steps, inner epochs, clipped
surrogate). The ONLY difference is the advantage estimator:
  * GRPO: group-relative baseline  A = (r - mean_g)/(std_g + eps), no value fn.
  * PPO : learned value-head baseline A = r - V_phi(prompt), with a value loss.
Both use the same PPO-style clipped surrogate and K inner epochs, so any
difference is attributable to the algorithm, not the stack.

Multi-seed (5) so a paired test across seeds is valid.

Usage:
  modal run experiments/modal/modal_samestack_ppo_grpo.py
"""
import modal
import json
import os
import time

app = modal.App("tinkerrl-samestack-ppo-grpo")
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

SEEDS = [42, 123, 456, 789, 1024]
ALGOS = ["grpo", "ppo"]
MODEL = "Qwen/Qwen2.5-0.5B"
N_STEPS = 40
N_GEN = 128       # total generations per optimizer step (identical for both arms)
GROUP = 8         # GRPO group size  -> N_GEN/GROUP = 16 prompts x 8 completions
K_EPOCHS = 2      # inner PPO/GRPO epochs over each rollout batch
CLIP = 0.2
MAX_NEW = 10
EPS = 1e-6
CHUNK = 8


@app.function(image=image, gpu="A10G", timeout=3600, volumes={RESULTS_DIR: results_vol}, retries=1)
def run_arm(algo: str, seed: int) -> dict:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import random
    import re
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    model.train()
    H = model.config.hidden_size
    params = [p for p in model.parameters() if p.requires_grad]
    vhead = None
    if algo == "ppo":
        vhead = nn.Linear(H, 1).to(device=device, dtype=torch.bfloat16)
        params = params + list(vhead.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)

    def make_prompt(a, b):
        return f"What is {a} + {b}? Answer with just the number."

    def parse(text):
        nums = re.findall(r"-?\d+", text)
        return int(nums[-1]) if nums else None

    def sample_batch(n):
        return [(random.randint(1, 99), random.randint(1, 99)) for _ in range(n)]

    def forward_logp(full, attn, plen, want_grad, want_value):
        """Chunked: returns (seq_logp[B], value[B] or None, entropy_scalar)."""
        B = full.shape[0]
        logps, vals = [], []
        ent_sum, ent_cnt = 0.0, 0.0
        ctx = torch.enable_grad() if want_grad else torch.no_grad()
        with ctx:
            for cs in range(0, B, CHUNK):
                ce = min(cs + CHUNK, B)
                f, a = full[cs:ce], attn[cs:ce]
                out = model(input_ids=f, attention_mask=a, output_hidden_states=want_value)
                logits = out.logits[:, :-1, :].float()
                tgt = f[:, 1:]
                cm = torch.zeros_like(tgt, dtype=torch.float32)
                cm[:, plen - 1:] = 1.0
                cm = cm * (tgt != tok.pad_token_id).float()
                logp = F.log_softmax(logits, dim=-1)
                tlp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                slp = (tlp * cm).sum(dim=1) / cm.sum(dim=1).clamp(min=1.0)
                logps.append(slp)
                if want_value:
                    h_last = out.hidden_states[-1][:, plen - 1, :]   # state = prompt-end hidden
                    vals.append(vhead(h_last).squeeze(-1).float())
                with torch.no_grad():
                    et = -(logp.exp() * logp).sum(dim=-1)
                    ent_sum += float((et * cm).sum().item()); ent_cnt += float(cm.sum().item())
                del out, logits, logp, tlp
        seq_logp = torch.cat(logps)
        value = torch.cat(vals) if want_value else None
        return seq_logp, value, ent_sum / max(ent_cnt, 1.0)

    step_log = []
    t0 = time.time()
    for step in range(N_STEPS):
        if algo == "grpo":
            P = N_GEN // GROUP
            probs = sample_batch(P)
            prompts, meta = [], []
            for (a, b) in probs:
                for _ in range(GROUP):
                    prompts.append(make_prompt(a, b)); meta.append((a, b))
        else:
            meta = sample_batch(N_GEN)
            prompts = [make_prompt(a, b) for (a, b) in meta]

        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                                 top_p=1.0, pad_token_id=tok.pad_token_id)
        comp_ids = gen[:, plen:]
        comp_texts = tok.batch_decode(comp_ids, skip_special_tokens=True)
        rewards = np.array([1.0 if (parse(t) is not None and parse(t) == a + b) else 0.0
                            for t, (a, b) in zip(comp_texts, meta)], dtype=np.float32)
        rew_t = torch.tensor(rewards, device=device)
        full = torch.cat([enc["input_ids"], comp_ids], dim=1)
        attn = (full != tok.pad_token_id).long()

        # old logp (+ old value) under the rollout policy
        old_logp, V_old, _ = forward_logp(full, attn, plen, want_grad=False, want_value=(algo == "ppo"))

        # advantage
        if algo == "grpo":
            R = rewards.reshape(-1, GROUP)
            gstd = R.std(axis=1, keepdims=True)
            zvf = float((gstd[:, 0] <= EPS).mean())
            A = (R - R.mean(axis=1, keepdims=True)) / (gstd + EPS)
            adv = torch.tensor(A.reshape(-1), device=device, dtype=torch.float32)
        else:
            adv = rew_t - V_old.detach()
            adv = (adv - adv.mean()) / (adv.std() + EPS)
            zvf = float('nan')

        # inner clipped-surrogate epochs (identical objective for both arms)
        for _ep in range(K_EPOCHS):
            opt.zero_grad()
            new_logp, V_new, entropy = forward_logp(full, attn, plen, want_grad=True, want_value=(algo == "ppo"))
            ratio = torch.exp(new_logp - old_logp.detach())
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv)
            ploss = -surr.mean()
            loss = ploss + (0.5 * F.mse_loss(V_new, rew_t) if algo == "ppo" else 0.0)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()

        step_log.append({"step": step, "mean_reward": float(rewards.mean()),
                         "zvf": zvf, "entropy": entropy, "grad_norm": gnorm})

    # measured held-out eval (greedy)
    model.eval()
    correct, total = 0, 200
    held = sample_batch(total)
    for i in range(0, total, 32):
        ch = held[i:i + 32]
        e = tok([make_prompt(a, b) for (a, b) in ch], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            g = model.generate(**e, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
        for t, (a, b) in zip(tok.batch_decode(g[:, e["input_ids"].shape[1]:], skip_special_tokens=True), ch):
            if parse(t) is not None and parse(t) == a + b:
                correct += 1
    heldout = correct / total
    last10 = float(np.mean([s["mean_reward"] for s in step_log[-10:]]))

    res = {"experiment": "samestack_ppo_grpo", "algo": algo, "seed": seed, "model": MODEL,
           "n_steps": N_STEPS, "n_gen": N_GEN, "k_epochs": K_EPOCHS,
           "heldout_acc": heldout, "last10_avg": last10,
           "elapsed_seconds": time.time() - t0, "step_log": step_log}
    os.makedirs(f"{RESULTS_DIR}/samestack", exist_ok=True)
    with open(f"{RESULTS_DIR}/samestack/{algo}_s{seed}.json", "w") as f:
        json.dump(res, f, indent=2)
    results_vol.commit()
    print(f"[{algo} seed={seed}] heldout={heldout:.3f} last10={last10:.3f}")
    return res


@app.local_entrypoint()
def main():
    import numpy as np
    jobs = [(algo, s) for algo in ALGOS for s in SEEDS]
    print(f"Same-stack PPO vs GRPO: {len(jobs)} runs ({ALGOS} x {SEEDS})")
    results = list(run_arm.starmap(jobs))

    by = {a: [r for r in results if r["algo"] == a] for a in ALGOS}
    summ = {}
    for a in ALGOS:
        rs = by[a]
        summ[a] = {
            "n_seeds": len(rs),
            "heldout_mean": float(np.mean([r["heldout_acc"] for r in rs])),
            "heldout_se": float(np.std([r["heldout_acc"] for r in rs]) / np.sqrt(len(rs))),
            "last10_mean": float(np.mean([r["last10_avg"] for r in rs])),
            "last10_se": float(np.std([r["last10_avg"] for r in rs]) / np.sqrt(len(rs))),
        }
    # paired comparison by seed on held-out accuracy
    paired = {}
    g = {r["seed"]: r["heldout_acc"] for r in by["grpo"]}
    p = {r["seed"]: r["heldout_acc"] for r in by["ppo"]}
    common = sorted(set(g) & set(p))
    diffs = [g[s] - p[s] for s in common]
    if len(diffs) >= 2:
        import math
        md = float(np.mean(diffs)); sd = float(np.std(diffs, ddof=1))
        t = md / (sd / math.sqrt(len(diffs))) if sd > 0 else float('inf')
        df = len(diffs) - 1
        # exact two-sided Student-t tail
        def betacf(aa, bb, x):
            MAXIT, E, FP = 200, 3e-12, 1e-300
            qab, qap, qam = aa + bb, aa + 1, aa - 1
            c = 1.0; d = 1 - qab * x / qap
            d = 1 / (FP if abs(d) < FP else d); h = d
            for m in range(1, MAXIT):
                m2 = 2 * m
                aA = m * (bb - m) * x / ((qam + m2) * (aa + m2)); d = 1 + aA * d
                d = 1 / (FP if abs(d) < FP else d); c = 1 + aA / c
                if abs(c) < FP: c = FP
                h *= d * c
                aA = -(aa + m) * (qab + m) * x / ((aa + m2) * (qap + m2)); d = 1 + aA * d
                d = 1 / (FP if abs(d) < FP else d); c = 1 + aA / c
                if abs(c) < FP: c = FP
                de = d * c; h *= de
                if abs(de - 1) < E: break
            return h
        def betai(aa, bb, x):
            if x <= 0 or x >= 1:
                return 0.0 if x <= 0 else 1.0
            bt = math.exp(math.lgamma(aa + bb) - math.lgamma(aa) - math.lgamma(bb)
                          + aa * math.log(x) + bb * math.log(1 - x))
            return bt * betacf(aa, bb, x) / aa if x < (aa + 1) / (aa + bb + 2) else 1 - bt * betacf(bb, aa, 1 - x) / bb
        pval = betai(df / 2.0, 0.5, df / (df + t * t)) if t != float('inf') else 0.0
        paired = {"metric": "heldout_acc", "n_seeds": len(diffs),
                  "mean_diff_grpo_minus_ppo": md, "t": t, "df": df, "p_two_sided": pval}

    out = {"summary": summ, "paired_grpo_vs_ppo": paired, "runs": results}
    with open("experiments/results/samestack_ppo_grpo.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SAME-STACK PPO vs GRPO (measured) ===")
    for a in ALGOS:
        print(f"  {a.upper()}: heldout={summ[a]['heldout_mean']:.3f}+/-{summ[a]['heldout_se']:.3f} "
              f"last10={summ[a]['last10_mean']:.3f}+/-{summ[a]['last10_se']:.3f} (n={summ[a]['n_seeds']})")
    if paired:
        print(f"  paired GRPO-PPO heldout diff={paired['mean_diff_grpo_minus_ppo']:+.3f} "
              f"t={paired['t']:.3f} df={paired['df']} p={paired['p_two_sided']:.3f}")
    print("Saved experiments/results/samestack_ppo_grpo.json")
