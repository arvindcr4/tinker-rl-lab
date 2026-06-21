"""E4 (Pillar 1, empirical sweep): does measured ZVF follow the closed form
ZVF(p,K) = p^K + (1-p)^K across a group-size sweep, and does numerical PRECISION
move the audit number?

Codex-review fixes vs the first draft:
  * POWER: 16 prompt-groups cannot resolve the 0.008 worked example (resolution
    1/16). We GENERATE ONCE (N_POOL rollouts/prompt) then SUBSAMPLE >=N_GROUPS
    size-K groups per prompt -> >=128 groups/K with bootstrap CIs, no re-gen.
  * The only clean Tinker-blocked lever here is PRECISION (Tinker pins it), so it
    is a small K=8 side-check, not the main grid.

Colab-only because: (a) fp32-vs-bf16 training/inference precision is fixed on
Tinker; (b) the closed-form check needs the full per-prompt reward matrix at
controlled large K, which the closed loss/sampler does not expose cleanly.

Run:  colab run --gpu T4 --timeout 1500 e4_scaling_law.py
"""
import json, re, random, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
N_PROMPTS = 40        # pool of prompts, difficulty tuned for p ~ 0.5
N_POOL = 64           # rollouts generated per prompt (once, then subsampled)
KS = [2, 4, 8, 16, 32]
N_GROUPS = 256        # bootstrap groups per prompt per K  (>=128, powered)
MAX_NEW = 24
PREC_PROMPTS = 10     # subset re-generated at fp32 for the precision side-check
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

def problem():
    # two 2-digit addends with a carry-heavy range -> base 0.5B lands near p~0.5
    a, b = random.randint(25, 95), random.randint(25, 95)
    return f"{a} + {b}", a + b

def prompt_of(q):
    return tok.apply_chat_template(
        [{"role": "user", "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}],
        tokenize=False, add_generation_prompt=True)

def parse(text):
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

@torch.no_grad()
def reward_pool(model, prompt, gold, n):
    """Generate n rollouts for one prompt; return list of 0/1 rewards."""
    rewards = []
    B = 32
    for i in range(0, n, B):
        k = min(B, n - i)
        enc = tok([prompt] * k, return_tensors="pt", padding=True).to(DEV)
        out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                             max_new_tokens=MAX_NEW, pad_token_id=PAD)
        gens = out[:, enc.input_ids.shape[1]:]
        rewards += [1.0 if parse(t) == gold else 0.0
                    for t in tok.batch_decode(gens, skip_special_tokens=True)]
    return rewards

def empirical_zvf(pool, K, rng, n_groups):
    """Fraction of size-K groups (subsampled w/ replacement from pool) with zero reward variance."""
    z = 0
    npool = len(pool)
    for _ in range(n_groups):
        idx = [rng.randrange(npool) for _ in range(K)]
        vals = [pool[i] for i in idx]
        if min(vals) == max(vals):     # all identical -> zero variance -> ZVF group
            z += 1
    return z / n_groups

def predicted_zvf(p, K):
    return p ** K + (1 - p) ** K

def build_pools(model, prompts):
    pools, ps = [], []
    for q, gold in prompts:
        pool = reward_pool(model, prompt_of(q), gold, N_POOL)
        pools.append(pool); ps.append(sum(pool) / len(pool))
    return pools, ps

def r2(emp, pred):
    mu = statistics.mean(emp)
    ss_res = sum((e - p) ** 2 for e, p in zip(emp, pred))
    ss_tot = sum((e - mu) ** 2 for e in emp) or 1e-12
    return 1 - ss_res / ss_tot

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED)
    prompts = [problem() for _ in range(N_PROMPTS)]

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)
    pools, ps = build_pools(model, prompts)
    pbar = statistics.mean(ps)
    print(f"[e4] mean p_hat over {N_PROMPTS} prompts = {pbar:.3f}", flush=True)

    by_K = {}
    emp_curve, pred_curve = [], []
    for K in KS:
        per_prompt_emp = [empirical_zvf(pool, K, rng, N_GROUPS) for pool in pools]
        per_prompt_pred = [predicted_zvf(p, K) for p in ps]
        emp = statistics.mean(per_prompt_emp)
        pred = statistics.mean(per_prompt_pred)
        # bootstrap 95% CI over prompts
        boots = []
        for _ in range(500):
            samp = [per_prompt_emp[rng.randrange(N_PROMPTS)] for _ in range(N_PROMPTS)]
            boots.append(statistics.mean(samp))
        boots.sort()
        ci = [round(boots[12], 4), round(boots[-13], 4)]
        by_K[K] = {"emp_zvf": round(emp, 4), "pred_zvf": round(pred, 4),
                   "emp_ci95": ci, "abs_err": round(abs(emp - pred), 4)}
        emp_curve.append(emp); pred_curve.append(pred)
        print(f"[e4] K={K:2d} emp={emp:.4f} pred={pred:.4f} ci={ci}", flush=True)

    fit_r2 = r2(emp_curve, pred_curve)

    # ---- precision side-check at K=8 (the Tinker-blocked lever) ----
    sub = prompts[:PREC_PROMPTS]
    bf16_pools = pools[:PREC_PROMPTS]
    del model; torch.cuda.empty_cache()
    random.seed(SEED); torch.manual_seed(SEED)
    model32 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
    fp32_pools, fp32_ps = build_pools(model32, sub)
    bf16_z8 = statistics.mean(empirical_zvf(p, 8, rng, N_GROUPS) for p in bf16_pools)
    fp32_z8 = statistics.mean(empirical_zvf(p, 8, rng, N_GROUPS) for p in fp32_pools)
    bf16_p = statistics.mean(ps[:PREC_PROMPTS]); fp32_p = statistics.mean(fp32_ps)
    del model32; torch.cuda.empty_cache()

    result = {
        "experiment": "E4_scaling_law", "model": MODEL, "seed": SEED,
        "n_prompts": N_PROMPTS, "n_pool": N_POOL, "n_groups_per_K": N_GROUPS,
        "mean_p_hat": round(pbar, 4), "ks": KS, "by_K": by_K,
        "closed_form_r2": round(fit_r2, 4),
        "worked_example_K8_pred_at_p0.5": round(predicted_zvf(0.5, 8), 5),
        "precision_side_check_K8": {
            "bf16_zvf": round(bf16_z8, 4), "fp32_zvf": round(fp32_z8, 4),
            "delta_zvf_fp32_minus_bf16": round(fp32_z8 - bf16_z8, 4),
            "bf16_p_hat": round(bf16_p, 4), "fp32_p_hat": round(fp32_p, 4)},
    }
    print("E4_RESULT " + json.dumps(result), flush=True)

main()
