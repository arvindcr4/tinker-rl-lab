#!/usr/bin/env python3
"""
Lean-STaR / STaR (SP25 L10, Sean Welleck) -> Pillar 3 group-size mining.

Idea: STaR / rejection-sampling fine-tuning (RFT) keeps only the *correct*
rollouts from a group of G samples and SFTs on them. That is exactly GRPO's
positive-advantage branch. So STaR extracts a training signal from a prompt iff
>=1 of its G rollouts is correct (k>=1), whereas GRPO produces a *nonzero
advantage* only when the group is mixed (0<k<G). This yields a clean identity
on our Zero-Variance-Fraction (ZVF) pillar:

    ZVF(G) = P(k=0) + P(k=G) = ZVF_lo + ZVF_hi
    Y_GRPO(G) = P(0<k<G)            (contrastive yield)
    Y_STaR(G) = P(k>=1) = 1 - P(k=0) = Y_GRPO(G) + ZVF_hi

=> STaR recovers exactly the all-correct tail ZVF_hi that GRPO discards as
   zero-advantage. The "irrecoverable" waste is only ZVF_lo (all-wrong groups).

We test this on REAL rollout data: 3 seeds x 200 GSM8K problems x G=8 rollouts
(Qwen3-8B), using EXACT hypergeometric subsampling to get counterfactual group
sizes G'<=8 with no parametric assumption, and per-prompt p_x extrapolation for
G'>8. No target leakage; every number traces to measured rewards.

Citations (verified 2026-07-04 via arxiv.org/abs):
  Lean-STaR: Lin, Sun, Welleck, Yang. arXiv:2407.10040 (2024).
  STaR: Zelikman, Wu, Mu, Goodman. arXiv:2203.14465 (NeurIPS 2022).
"""
import json, math, glob, os
from itertools import combinations

RES = "experiments/results"
OUT = "experiments/results/berkeley"
os.makedirs(OUT, exist_ok=True)

def comb(n, r):
    if r < 0 or r > n: return 0.0
    return float(math.comb(n, r))

# ---- load real per-prompt rollout counts (k out of n=8) ----
counts = []  # (k, n, seed)
for f in sorted(glob.glob(f"{RES}/tinker_gsm8k_zvf_s*.json")):
    if "summary" in f: continue
    d = json.load(open(f))
    G = d["group_size"]; seed = d.get("seed")
    for p in d["per_problem"]:
        r = p["rewards"]
        counts.append((int(round(sum(r))), len(r), seed))
N = len(counts)
n0 = counts[0][1]
assert all(n == n0 for _, n, _ in counts), "ragged group sizes"
print(f"loaded {N} prompts, native G={n0}")

def sub_probs(k, n, g):
    """Exact P(all-correct), P(all-wrong) for a random g-subset (no replacement)."""
    denom = comb(n, g)
    p_all_correct = comb(k, g) / denom
    p_all_wrong = comb(n - k, g) / denom
    return p_all_correct, p_all_wrong

def yields_at_G(sample, g, parametric=False):
    """Return (Y_STaR, Y_GRPO, zvf_hi, zvf_lo) averaged over prompts."""
    hi = lo = 0.0
    for (k, n, _) in sample:
        if parametric or g > n:
            p = k / n
            pc = p ** g
            pw = (1 - p) ** g
        else:
            pc, pw = sub_probs(k, n, g)
        hi += pc; lo += pw
    hi /= len(sample); lo /= len(sample)
    y_star = 1 - lo
    y_grpo = 1 - hi - lo
    return y_star, y_grpo, hi, lo

def boot_ci(sample, g, fn, B=2000, seed=7):
    import random
    rng = random.Random(seed)
    vals = []
    m = len(sample)
    idx = list(range(m))
    for _ in range(B):
        bs = [sample[rng.choice(idx)] for _ in range(m)]
        vals.append(fn(bs, g))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B)]

# ======================================================================
# H1: STaR yield > GRPO yield at every G; gap == ZVF_hi (identity).
# ======================================================================
GS = [2, 4, 8, 16, 32]
rows_g = []
for g in GS:
    param = g > n0
    ys, yg, hi, lo = yields_at_G(counts, g, parametric=param)
    gap = ys - yg
    recoverable = hi / (hi + lo) if (hi + lo) > 0 else float("nan")
    gap_ci = boot_ci(counts, g,
                     lambda s, gg: (lambda a: a[0] - a[1])(yields_at_G(s, gg, parametric=(gg > n0))))
    identity_err = abs(gap - hi)  # must be ~0
    rows_g.append(dict(G=g, source=("subsample" if not param else "param_px"),
                       Y_STaR=ys, Y_GRPO=yg, gap=gap, gap_ci_lo=gap_ci[0], gap_ci_hi=gap_ci[1],
                       ZVF=hi + lo, ZVF_hi=hi, ZVF_lo=lo,
                       recoverable_frac=recoverable, identity_err=identity_err))
h1_gap_pos = all(r["gap_ci_lo"] > 0 for r in rows_g)
h1_identity = all(r["identity_err"] < 1e-9 for r in rows_g)
H1 = h1_gap_pos and h1_identity

# ======================================================================
# H2: STaR-exclusive signal lives in the easy tail, not the frontier.
#     Stratify by p_x; gap (all-correct prob) at G=8, easy vs frontier.
# ======================================================================
def strat(sample, lo_p, hi_p, g=8):
    sub = [c for c in sample if lo_p <= c[0] / c[1] < hi_p]
    if not sub: return float("nan"), 0
    _, _, hi, _ = yields_at_G(sub, g, parametric=False)
    return hi, len(sub)  # hi == STaR-exclusive prob within stratum
strata = [("hard(0-0.3)", 0.0, 0.3), ("frontier(0.3-0.7)", 0.3, 0.7),
          ("easy(0.7-1.0001)", 0.7, 1.0001)]
rows_strat = []
for name, a, b in strata:
    gapv, nn = strat(counts, a, b, 8)
    rows_strat.append(dict(stratum=name, n=nn, frac_prompts=nn / N, star_exclusive_gap=gapv))
easy_gap = [r for r in rows_strat if r["stratum"].startswith("easy")][0]["star_exclusive_gap"]
front_gap = [r for r in rows_strat if r["stratum"].startswith("frontier")][0]["star_exclusive_gap"]
# decisive: STaR-exclusive signal concentrated on easy prompts (>=5x frontier)
H2_ratio = (easy_gap / front_gap) if front_gap > 1e-9 else float("inf")
H2 = H2_ratio >= 5.0

# ======================================================================
# H3: recoverable fraction of GRPO's wasted signal (ZVF_hi / ZVF) is
#     the MAJORITY at every G -> most zero-advantage groups are all-correct,
#     not all-wrong, so STaR reclaims most of GRPO's dead weight.
# ======================================================================
rec = [r["recoverable_frac"] for r in rows_g]
H3 = all(x > 0.5 for x in rec)   # majority recoverable at every G

# ======================================================================
# H4 (honest null-guard): the STaR-exclusive tail = already-solved prompts.
#   Does a larger STaR gap track higher heldout accuracy across the G-sweep?
#   Expect NULL / negative: extra STaR signal is low-value (saturated prompts).
# ======================================================================
sweep = {}
with open(f"{RES}/groupsize_zvf_sweep.tsv") as f:
    hdr = f.readline().strip().split("\t")
    for line in f:
        p = line.strip().split("\t")
        d = dict(zip(hdr, p))
        sweep[int(d["G"])] = float(d["heldout_acc_mean"])
gaps_by_g = {r["G"]: r["gap"] for r in rows_g}
xs, ys_acc = [], []
for g in [2, 4, 8, 16]:
    if g in sweep and g in gaps_by_g:
        xs.append(gaps_by_g[g]); ys_acc.append(sweep[g])
def pearson(a, b):
    m = len(a); ma = sum(a) / m; mb = sum(b) / m
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a)); db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")
rho_acc = pearson(xs, ys_acc) if len(xs) >= 3 else float("nan")
# H4 is a guard: we PREDICT no positive link (extra signal is saturated prompts).
H4_nolink = (not (rho_acc > 0.5))  # passes if not a strong positive link

# ======================================================================
# H5: iso-yield compute. Smallest G s.t. Y_STaR(G) >= Y_GRPO(native G=8).
#     Rejection sampling exposes G=8's contrastive-yield signal at lower G.
# ======================================================================
target = [r for r in rows_g if r["G"] == 8][0]["Y_GRPO"]
g_star = None
for g in range(1, 9):
    ys, _, _, _ = yields_at_G(counts, g, parametric=False)
    if ys >= target:
        g_star = g; break
H5 = (g_star is not None and g_star < 8)

# ---- write outputs ----
def wtsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

wtsv(f"{OUT}/leanstar_yield_by_g.tsv", rows_g,
     ["G", "source", "Y_STaR", "Y_GRPO", "gap", "gap_ci_lo", "gap_ci_hi",
      "ZVF", "ZVF_hi", "ZVF_lo", "recoverable_frac", "identity_err"])
wtsv(f"{OUT}/leanstar_difficulty_strata.tsv", rows_strat,
     ["stratum", "n", "frac_prompts", "star_exclusive_gap"])
wtsv(f"{OUT}/leanstar_acc_bridge.tsv",
     [dict(G=g, star_gap=gaps_by_g[g], heldout_acc=sweep[g]) for g in [2, 4, 8, 16]],
     ["G", "star_gap", "heldout_acc"])

summary = dict(
    citation="Lean-STaR arXiv:2407.10040 (2024); STaR arXiv:2203.14465 (NeurIPS 2022)",
    n_prompts=N, native_G=n0,
    H1_starexceeds_and_identity=dict(passed=bool(H1), gap_pos=bool(h1_gap_pos),
        identity_max_err=max(r["identity_err"] for r in rows_g),
        gap_at_G8=gaps_by_g[8]),
    H2_easytail_concentration=dict(passed=bool(H2), easy_over_frontier_ratio=H2_ratio,
        easy_gap=easy_gap, frontier_gap=front_gap),
    H3_majority_recoverable=dict(passed=bool(H3), recoverable_by_G={r["G"]: r["recoverable_frac"] for r in rows_g}),
    H4_nolink_guard=dict(passed=bool(H4_nolink), rho_gap_vs_heldout=rho_acc,
        note="extra STaR signal = already-solved prompts; expected low value"),
    H5_iso_yield=dict(passed=bool(H5), G_star=g_star, target_Y_GRPO_G8=target),
    n_decisive=sum([H1, H2, H3, H5]),  # H4 is a guard, not a decisive claim
)
json.dump(summary, open(f"{OUT}/leanstar_summary.json", "w"), indent=2)

print("\n=== Lean-STaR / STaR rejection-yield on Pillar-3 group-size ===")
for r in rows_g:
    print(f" G={r['G']:>2} [{r['source']:>10}]  Y_STaR={r['Y_STaR']:.3f}  Y_GRPO={r['Y_GRPO']:.3f}  "
          f"gap={r['gap']:.3f} (CI[{r['gap_ci_lo']:.3f},{r['gap_ci_hi']:.3f}])  "
          f"recoverable={r['recoverable_frac']:.3f}")
print(f"\nH1 STaR>GRPO everywhere & identity gap==ZVF_hi: {H1}")
print(f"H2 easy-tail concentration (easy/frontier={H2_ratio:.1f}x >=5): {H2}")
print(f"H3 majority of ZVF recoverable at every G: {H3}")
print(f"H4 guard: no positive gap->heldout link (rho={rho_acc:.3f}): {H4_nolink}")
print(f"H5 iso-yield G*={g_star} < 8 matches GRPO G=8 contrastive yield: {H5}")
print(f"\nDECISIVE: {summary['n_decisive']}/4  (H4 is an honest guard)")
