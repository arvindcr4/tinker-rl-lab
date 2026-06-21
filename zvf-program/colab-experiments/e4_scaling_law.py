"""E4 (Pillar 1, empirical sweep): does measured ZVF follow the closed form
ZVF(p,K) = p^K + (1-p)^K across a group-size sweep, probed near p ~ 0.5, and does
numerical precision move the audit number?

To probe the p=0.5,K=8 -> 0.0078 crossing (and avoid a format-gated p collapse):
  * Few-shot scaffold removes the parser/format confound.
  * Calibrate: sample candidate prompts across digit regimes, measure p_hat from a
    pilot, keep only prompts with p_hat in [0.4,0.6]; flag the run if the band
    cannot be populated.
  * Generate once, then subsample >=256 size-K groups/prompt for bootstrap CIs.
Precision (fp32 vs bf16, the one cleanly Tinker-blocked lever here) is a K=8
side-check.

Run:  colab run --gpu T4 --timeout 1800 e4_scaling_law.py
"""
import json, re, random, statistics
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
N_PROMPTS = 40        # calibrated prompts kept (p_hat in band)
N_PILOT = 24          # rollouts to estimate p_hat during calibration
N_POOL = 64           # rollouts per kept prompt for the scaling law
N_CANDIDATES = 110    # candidates screened to find in-band prompts
BAND = (0.4, 0.6)     # target p_hat band (centered on 0.5)
KS = [2, 4, 8, 16, 32]
N_GROUPS = 256        # bootstrap groups per prompt per K
MAX_NEW = 128         # unchoke reasoning: 24 truncated traces -> no '####' -> spurious p=0
PREC_PROMPTS = 12     # subset re-generated at fp32 for the precision side-check
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
PAD = tok.pad_token_id

FEWSHOT = [
    {"role": "user", "content": "Compute 3 + 4. Reason briefly, then put the final integer after '####'."},
    {"role": "assistant", "content": "3 + 4 = 7.\n#### 7"},
]

def candidate(rng):
    # mixed digit regimes so p_hat spans low..high; calibration keeps the middle
    r = rng.random()
    if r < 0.34:            # 2-digit (usually easy -> high p)
        a, b = rng.randint(10, 99), rng.randint(10, 99)
    elif r < 0.67:          # mixed 2+3 digit (the p~0.5 sweet spot)
        a, b = rng.randint(10, 99), rng.randint(100, 999)
    else:                   # 3-digit (hard -> low p)
        a, b = rng.randint(100, 999), rng.randint(100, 999)
    return f"{a} + {b}", a + b

def prompt_of(q):
    msgs = FEWSHOT + [{"role": "user",
                       "content": f"Compute {q}. Reason briefly, then put the final integer after '####'."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def parse(text):
    if "####" not in text:            # no marker -> not parseable (don't grab question digits)
        return None
    m = re.findall(r"-?\d+", text.split("####")[-1])
    return int(m[0]) if m else None

@torch.no_grad()
def reward_pool(model, prompt, gold, n):
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
    z, npool = 0, len(pool)
    for _ in range(n_groups):
        vals = [pool[rng.randrange(npool)] for _ in range(K)]
        if min(vals) == max(vals):
            z += 1
    return z / n_groups

def predicted_zvf(p, K):
    return p ** K + (1 - p) ** K

def r2(emp, pred):
    mu = statistics.mean(emp)
    ss_res = sum((e - p) ** 2 for e, p in zip(emp, pred))
    ss_tot = sum((e - mu) ** 2 for e in emp) or 1e-12
    return 1 - ss_res / ss_tot

def calibrate(model, rng):
    """Screen candidates; return prompts with p_hat in BAND (fallback: closest to 0.5)."""
    scored = []
    for _ in range(N_CANDIDATES):
        q, gold = candidate(rng)
        ph = statistics.mean(reward_pool(model, prompt_of(q), gold, N_PILOT))
        scored.append((abs(ph - 0.5), ph, q, gold))
        if sum(1 for s in scored if BAND[0] <= s[1] <= BAND[1]) >= N_PROMPTS:
            break
    in_band = [s for s in scored if BAND[0] <= s[1] <= BAND[1]]
    ok = len(in_band) >= N_PROMPTS
    chosen = (in_band if ok else sorted(scored))[:N_PROMPTS]
    print(f"[e4] calibration: {len(in_band)}/{len(scored)} in band {BAND}; ok={ok}", flush=True)
    return [(q, gold) for _, _, q, gold in chosen], ok

def build_pools(model, prompts):
    pools, ps = [], []
    for q, gold in prompts:
        pool = reward_pool(model, prompt_of(q), gold, N_POOL)
        pools.append(pool); ps.append(sum(pool) / len(pool))
    return pools, ps

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV)

    prompts, calib_ok = calibrate(model, rng)
    pools, ps = build_pools(model, prompts)
    pbar = statistics.mean(ps)
    print(f"[e4] mean p_hat over {len(prompts)} calibrated prompts = {pbar:.3f}", flush=True)

    by_K, emp_curve, pred_curve = {}, [], []
    for K in KS:
        per_emp = [empirical_zvf(pool, K, rng, N_GROUPS) for pool in pools]
        per_pred = [predicted_zvf(p, K) for p in ps]
        emp, pred = statistics.mean(per_emp), statistics.mean(per_pred)
        boots = sorted(statistics.mean([per_emp[rng.randrange(len(per_emp))] for _ in per_emp])
                       for _ in range(500))
        by_K[K] = {"emp_zvf": round(emp, 4), "pred_zvf": round(pred, 4),
                   "emp_ci95": [round(boots[12], 4), round(boots[-13], 4)],
                   "abs_err": round(abs(emp - pred), 4)}
        emp_curve.append(emp); pred_curve.append(pred)
        print(f"[e4] K={K:2d} emp={emp:.4f} pred={pred:.4f} ci={by_K[K]['emp_ci95']}", flush=True)

    # ---- precision side-check at K=8 ----
    sub = prompts[:PREC_PROMPTS]
    bf16_pools = pools[:PREC_PROMPTS]
    del model; torch.cuda.empty_cache()
    random.seed(SEED); torch.manual_seed(SEED)
    model32 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV)
    fp32_pools, fp32_ps = build_pools(model32, sub)
    bf16_z8 = statistics.mean(empirical_zvf(p, 8, rng, N_GROUPS) for p in bf16_pools)
    fp32_z8 = statistics.mean(empirical_zvf(p, 8, rng, N_GROUPS) for p in fp32_pools)
    del model32; torch.cuda.empty_cache()

    result = {
        "experiment": "E4_scaling_law", "model": MODEL, "seed": SEED, "few_shot": True,
        "calibrated_to_p0.5": calib_ok, "band": list(BAND),
        "n_prompts": len(prompts), "n_pool": N_POOL, "n_groups_per_K": N_GROUPS,
        "mean_p_hat": round(pbar, 4), "ks": KS, "by_K": by_K,
        "closed_form_r2": round(r2(emp_curve, pred_curve), 4),
        "worked_example_K8_pred_at_p0.5": round(predicted_zvf(0.5, 8), 5),
        "emp_zvf_at_K8": by_K[8]["emp_zvf"],
        "precision_side_check_K8": {
            "bf16_zvf": round(bf16_z8, 4), "fp32_zvf": round(fp32_z8, 4),
            "delta_zvf_fp32_minus_bf16": round(fp32_z8 - bf16_z8, 4),
            "bf16_p_hat": round(statistics.mean(ps[:PREC_PROMPTS]), 4),
            "fp32_p_hat": round(statistics.mean(fp32_ps), 4)},
    }
    print("E4_RESULT " + json.dumps(result), flush=True)

main()
