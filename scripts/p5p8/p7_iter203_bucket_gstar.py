#!/usr/bin/env python3
"""P7 Iter 203 — Bucket-Resolved Optimal Group Size on Real N2 Tensors.

Vein (a) of the iter-203 brief: ``when would [the adaptive-G controller]
have fired, what G would it have chosen, what contrast would it have
restored?'' — answered at the **per-k-bucket** resolution.

Unit of analysis: every (method, step, prompt) observation is reduced to
its k-bucket k = sum of G_BASE = 8 observed binary rewards.  k ∈
{0, 1, 2, 3, 4, 5, 6, 7, 8}.  For each k-bucket we answer three
counterfactual questions simultaneously:

  (i)   Empirical frequency:  how often does this k-bucket occur?
  (ii)  Optimal G*:  smallest G' ∈ {2, 4, 8, 12, 16, 24, 32} that
        achieves GU_iid(p̂, G') ≥ τ under the Beta-Binomial posterior
        predictive with prior Beta(1, 1) (uniform).
  (iii) Bucket-conditional contrast restoration: GU_iid(p̂, G*) versus
        GU_iid(p̂, G_BASE), averaged across the bucket.

This is the **bucket-resolved Pareto frontier** of the adaptive-G
controller.  Prior Iso-G work (iter-83) aggregated buckets into
``saturated'' (k ∈ {0,8}) vs ``non-saturated'' (k ∈ {1..7}) and
showed nothing can help saturated prompts.  The bucket-resolved view
keeps the 7 non-saturated buckets separate, which exposes whether the
optimal G* and the contrast-restoration reward vary with k, and
whether the controller should use k as its state signal instead of
zvf_step.

Hypotheses (declared, then tested):
  H1_resolvable: bucket-resolved G* differs by ≥ 1 rung from the
    single fixed-G bucket-mean optimal G* on ≥ 5 of the 7
    non-saturated buckets.
  H2_pareto: bucket-conditional optimal G* Pareto-dominates the
    single fixed-G=8 baseline on (cost, contrast_restored) for
    ≥ 5 of the 7 non-saturated buckets.
  H3_uniformity: the bucket-resolved Pareto frontier is uniform across
    the 4 N2 methods (no method is Pareto-worse than another).
  H4_signal_efficiency: per-bucket optimal G* ≤ G_BASE for ≥ 7 of the
    9 buckets — the controller can save rollouts on most buckets
    without losing contrast.

Stdlib only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import pathlib
import statistics

WORKTREE = pathlib.Path("/home/claude/tinker-rl-lab-minimax")
N2 = WORKTREE / "experiments" / "results" / "n2_reward_tensor_resume"
OUT = WORKTREE / "experiments" / "results" / "p5p8"
METHODS = ("grpo", "aero", "gift", "areal")
G_BASE = 8
G_CANDIDATES = (2, 4, 8, 12, 16, 24, 32)
N_PROMPTS = 16
N_STEPS = 40
TARGET_GU = 0.85  # iid GU threshold for "useful contrast"
N_BOOT = 2000
SEED = 20260706


def lbeta(a: float, b: float) -> float:
    """log Beta(a,b) via lgamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_binom_cdf(k: int, n: int, alpha: float, beta: float) -> float:
    """P(K <= k) for Beta-Binomial(n, alpha, beta)."""
    s = 0.0
    for j in range(k + 1):
        log_p = (lbeta(alpha + j, beta + (n - j)) - lbeta(j + 1, n - j + 1)
                 - lbeta(alpha, beta))
        s += math.exp(log_p)
    return min(max(s, 0.0), 1.0)


def gu_iid(p: float, g: int) -> float:
    """Useful-GU under iid binomial: P(0 < K < g)."""
    if g <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - p**g - (1.0 - p) ** g))


def gu_post(p_hat: float, g_base: int, g_new: int,
            alpha0: float = 1.0, beta0: float = 1.0) -> float:
    """Predictive P(0 < K' < g_new) integrating over Beta-Binomial posterior
    with prior Beta(alpha0, beta0), having observed k = p_hat * g_base successes.

    Computed via mixture over n1 ∈ {0..g_new} of P(K' = n1) =
    ∫ Bin(n1; g_new, p) · Beta(p; k+alpha, G-k+beta) dp
    """
    a_post = p_hat * g_base + alpha0
    b_post = (g_base - p_hat * g_base) + beta0
    # Truncate / clamp
    a_post = min(max(a_post, 1e-6), 1e6)
    b_post = min(max(b_post, 1e-6), 1e6)
    lo, hi = 1, g_new - 1
    if lo > hi:
        return 0.0
    p_lo = beta_binom_cdf(lo - 1, g_new, a_post, b_post) if lo > 0 else 0.0
    p_hi = 1.0 - beta_binom_cdf(g_new, g_new, a_post, b_post)
    return max(0.0, min(1.0, p_hi - p_lo))


def optimal_gstar_post(k: int, g_base: int = G_BASE,
                       target: float = TARGET_GU) -> int:
    """Smallest G' ∈ G_CANDIDATES such that GU_post >= target.  Returns
    g_base if no candidate qualifies or if k is saturated (k=0 or k=g_base).
    """
    if k == 0 or k == g_base:
        return g_base
    p_hat = k / g_base
    for g in G_CANDIDATES:
        if gu_post(p_hat, g_base, g) >= target:
            return g
    return g_base


def contrast_restored_bucket(k: int, g_base: int = G_BASE) -> float:
    """Contrast restored by switching from g_base to optimal G*:
    GU_post(p_hat, g_base, G*) - GU_post(p_hat, g_base, g_base).
    """
    if k == 0 or k == g_base:
        return 0.0
    p_hat = k / g_base
    g_star = optimal_gstar_post(k, g_base)
    return gu_post(p_hat, g_base, g_star) - gu_post(p_hat, g_base, g_base)


def load_rewards():
    """Load {method: [(step, list of 16 prompt vectors of length 8)]}."""
    out = {m: [] for m in METHODS}
    for m in METHODS:
        fpath = N2 / f"{m}_s0_tensors.jsonl"
        rows = []
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                if rec["method"] == m and rec["step"] < N_STEPS:
                    rows.append((rec["step"], rec["rewards"]))
        rows.sort(key=lambda x: x[0])
        out[m] = [r[1] for r in rows]
    return out


def all_k_buckets(rewards_2d, n_prompts: int = N_PROMPTS,
                  g_base: int = G_BASE) -> list[int]:
    """Return the list of k values (one per prompt) for the 2D rewards tensor."""
    out = []
    for prompt_vec in rewards_2d[:n_prompts]:
        out.append(int(round(sum(prompt_vec[:g_base]))))
    return out


def bootstrap_ci_diff(xs_a, xs_b, n_boot: int = N_BOOT, seed: int = SEED,
                      alpha: float = 0.05):
    """Bootstrap CI on mean(xs_a) - mean(xs_b) at 1-alpha level, percentile method."""
    import random
    rng = random.Random(seed)
    n = min(len(xs_a), len(xs_b))
    if n == 0:
        return (0.0, 0.0, 0.0)
    a = xs_a[:n]
    b = xs_b[:n]
    diffs = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        ma = sum(a[i] for i in idxs) / n
        mb = sum(b[i] for i in idxs) / n
        diffs.append(ma - mb)
    diffs.sort()
    lo = diffs[int(alpha / 2 * n_boot)]
    hi = dffs = diffs[int((1 - alpha / 2) * n_boot)]
    point = sum(a) / n - sum(b) / n
    return (point, lo, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gu", type=float, default=TARGET_GU)
    parser.add_argument("--out-tag", type=str, default="p7_iter203")
    args = parser.parse_args()

    target = args.target_gu
    rewards_by_method = load_rewards()

    # Per-(method, step, prompt) accumulate k, g_star, contrast_restored
    rows_per_obs = []
    for m in METHODS:
        steps = rewards_by_method[m]
        for step_idx, prompt_vecs in enumerate(steps[:N_STEPS]):
            for prompt_idx, vec in enumerate(prompt_vecs[:N_PROMPTS]):
                k = int(round(sum(vec[:G_BASE])))
                g_star = optimal_gstar_post(k, G_BASE, target)
                cr = contrast_restored_bucket(k, G_BASE) if g_star != G_BASE else 0.0
                rows_per_obs.append({
                    "method": m, "step": step_idx,
                    "prompt_index": prompt_idx, "k": k,
                    "g_star": g_star, "contrast_restored": cr,
                    "cost_ratio": g_star / G_BASE,
                })

    # Bucket-level distribution: per (method, k), how often, mean g*, mean cost
    bucket_table = {}  # (method, k) -> {n, g_star_dist, mean_g_star, mean_contrast_restored}
    for r in rows_per_obs:
        key = (r["method"], r["k"])
        bucket_table.setdefault(key, []).append(r)

    per_bucket_rows = []
    for (m, k), entries in sorted(bucket_table.items()):
        n = len(entries)
        g_stars = [e["g_star"] for e in entries]
        crs = [e["contrast_restored"] for e in entries]
        per_bucket_rows.append({
            "method": m, "k": k, "n_obs": n,
            "bucket_freq": n / sum(1 for r in rows_per_obs if r["method"] == m),
            "mean_g_star": sum(g_stars) / n,
            "mean_contrast_restored": sum(crs) / n,
            "frac_down_shift": sum(1 for g in g_stars if g < G_BASE) / n,
            "frac_up_shift": sum(1 for g in g_stars if g > G_BASE) / n,
        })

    # Method-level summary: average G*, contrast restored per G_BASE=8 prompt
    method_summary = {}
    for m in METHODS:
        m_rows = [r for r in rows_per_obs if r["method"] == m]
        n = len(m_rows)
        method_summary[m] = {
            "n_prompt_steps": n,
            "bucket_freq_dist": {k: sum(1 for r in m_rows if r["k"] == k) / n
                                  for k in range(G_BASE + 1)},
            "mean_g_star": sum(r["g_star"] for r in m_rows) / n,
            "mean_contrast_restored": sum(r["contrast_restored"] for r in m_rows) / n,
            "mean_cost_ratio": sum(r["cost_ratio"] for r in m_rows) / n,
            "n_buckets_with_benefit": len(set(
                k for r in m_rows
                if r["contrast_restored"] > 1e-9
            )),
        }

    # Test H1: bucket-resolved mean G* vs single fixed-G mean G* across non-sat
    # Single fixed-G mean G* = 8 (no switching). Bucket-resolved is per k.
    non_sat_buckets = [k for k in range(1, G_BASE)]
    h1_pass_count = 0
    for k in non_sat_buckets:
        # mean g* across (method, step, prompt) with this k
        ks = [r["g_star"] for r in rows_per_obs if r["k"] == k]
        if not ks:
            continue
        mgs = sum(ks) / len(ks)
        if abs(mgs - G_BASE) >= 1:
            h1_pass_count += 1
    h1_pass = h1_pass_count >= 5

    # H2: bucket-conditional optimal G* Pareto-dominates fixed-G=8 baseline on
    # (cost, contrast_restored).  Pareto-dominates = mean_g_star <= G_BASE AND
    # mean_contrast_restored >= 0  AND  not equal to (G_BASE, 0).
    h2_passes = {}
    h2_pass_count = 0
    for k in non_sat_buckets:
        ks = [r for r in rows_per_obs if r["k"] == k]
        if not ks:
            continue
        mg = sum(r["g_star"] for r in ks) / len(ks)
        cr = sum(r["contrast_restored"] for r in ks) / len(ks)
        # Pareto-dominates fixed-G=8 means: strictly less cost AND non-negative contrast
        if mg < G_BASE and cr >= -1e-9 and not (abs(mg - G_BASE) < 1e-9 and abs(cr) < 1e-9):
            h2_passes[k] = (mg, cr)
            h2_pass_count += 1
    h2_pass = h2_pass_count >= 5

    # H3: uniformity — no method Pareto-worse than another.
    # Operationally: range of mean_g_star across methods is < G_BASE unit.
    method_mean_gs = {m: method_summary[m]["mean_g_star"] for m in METHODS}
    spread = max(method_mean_gs.values()) - min(method_mean_gs.values())
    h3_pass = spread < G_BASE  # different methods should not pick wildly different G*

    # H4: signal efficiency — per-bucket optimal G* <= G_BASE for >=7 of 9 buckets
    h4_buckets_le_base = 0
    for k in range(G_BASE + 1):
        ks = [r for r in rows_per_obs if r["k"] == k]
        if not ks:
            continue
        mg = sum(r["g_star"] for r in ks) / len(ks)
        if mg <= G_BASE:
            h4_buckets_le_base += 1
    h4_pass = h4_buckets_le_base >= 7

    # Bootstrap CI: bucket-resolved G* vs fixed G=8 cost ratio difference
    cost_ratios_ctrl = [r["cost_ratio"] for r in rows_per_obs]
    cost_ratios_base = [1.0] * len(cost_ratios_ctrl)
    cost_diff = bootstrap_ci_diff(cost_ratios_ctrl, cost_ratios_base)

    cr_ctrl = [r["contrast_restored"] for r in rows_per_obs]
    cr_base = [0.0] * len(cr_ctrl)
    cr_diff = bootstrap_ci_diff(cr_ctrl, cr_base)

    # SAVE
    OUT.mkdir(parents=True, exist_ok=True)
    out_prefix = OUT / args.out_tag

    # Per-observation TSV
    obs_tsv = out_prefix.with_name(out_prefix.name + "_per_obs.tsv")
    with open(obs_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_per_obs[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in rows_per_obs:
            w.writerow(r)

    # Per-bucket TSV
    bucket_tsv = out_prefix.with_name(out_prefix.name + "_per_bucket.tsv")
    with open(bucket_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bucket_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in per_bucket_rows:
            w.writerow(r)

    summary = {
        "iter": 203,
        "pillar": "P7",
        "vein": "(a) bucket-resolved counterfactual G* evaluation",
        "settings": {
            "g_base": G_BASE,
            "g_candidates": list(G_CANDIDATES),
            "target_gu": target,
            "n_steps": N_STEPS,
            "n_prompts": N_PROMPTS,
            "n_methods": len(METHODS),
            "beta_binomial_prior": "Beta(1,1)",
            "n_boot": N_BOOT,
            "seed": SEED,
        },
        "method_summary": method_summary,
        "method_mean_g_star": method_mean_gs,
        "global": {
            "n_prompt_steps": len(rows_per_obs),
            "mean_g_star": sum(r["g_star"] for r in rows_per_obs) / len(rows_per_obs),
            "mean_contrast_restored": sum(r["contrast_restored"] for r in rows_per_obs) / len(rows_per_obs),
            "mean_cost_ratio": sum(r["cost_ratio"] for r in rows_per_obs) / len(rows_per_obs),
            "n_buckets_with_benefit_global": len(set(
                k for r in rows_per_obs if r["contrast_restored"] > 1e-9
            )),
        },
        "verdicts": {
            "H1_resolvable_pass": bool(h1_pass),
            "H1_n_buckets_with_diff_gs": h1_pass_count,
            "H2_pareto_pass": bool(h2_pass),
            "H2_n_pareto_buckets": h2_pass_count,
            "H3_uniformity_pass": bool(h3_pass),
            "H3_method_spread": spread,
            "H4_signal_efficiency_pass": bool(h4_pass),
            "H4_n_buckets_le_base": h4_buckets_le_base,
        },
        "bootstrap_ci": {
            "cost_ratio_diff_ctrl_minus_base": {
                "point": cost_diff[0], "ci_lo": cost_diff[1], "ci_hi": cost_diff[2],
                "excl_zero": 1 if (cost_diff[1] > 0 or cost_diff[2] < 0) else 0,
            },
            "contrast_restored_diff_ctrl_minus_zero": {
                "point": cr_diff[0], "ci_lo": cr_diff[1], "ci_hi": cr_diff[2],
                "excl_zero": 1 if (cr_diff[1] > 0 or cr_diff[2] < 0) else 0,
            },
        },
    }
    summary_tsv = out_prefix.with_name(out_prefix.name + "_summary.json")
    with open(summary_tsv, "w") as f:
        json.dump(summary, f, indent=2)

    # Plain stdout report
    print("=" * 70)
    print("ITER 203 — Bucket-Resolved Optimal G* on Real N2 Tensors")
    print("=" * 70)
    print()
    print(f"Predictive model: Beta-Binomial posterior predictive, prior Beta(1,1).")
    print(f"Target GU ≥ {target}, G candidates {G_CANDIDATES}, G_BASE = {G_BASE}.")
    print(f"Total prompt-step decisions: {summary['global']['n_prompt_steps']}.")
    print(f"Mean G* (pooled): {summary['global']['mean_g_star']:.4f}")
    print(f"Mean contrast restored: {summary['global']['mean_contrast_restored']:.6f}")
    print(f"Mean cost ratio: {summary['global']['mean_cost_ratio']:.4f}")
    print()
    print("Per-method mean G*:")
    for m in METHODS:
        s = method_summary[m]
        print(f"  {m:>6s}: G*={s['mean_g_star']:.4f}, "
              f"contrast_restored={s['mean_contrast_restored']:.6f}, "
              f"cost_ratio={s['mean_cost_ratio']:.4f}, "
              f"beneficial_buckets={s['n_buckets_with_benefit']}")
    print()
    print("Per-bucket distribution (freq per method):")
    print(f"  {'k':>2s}  ", end="")
    for m in METHODS:
        print(f"{m:>9s}", end="")
    print()
    for k in range(G_BASE + 1):
        print(f"  {k:>2d}  ", end="")
        for m in METHODS:
            f = method_summary[m]["bucket_freq_dist"].get(k, 0.0)
            print(f"{f:9.3f}", end="")
        print()
    print()
    print("V E R D I C T S")
    print(f"  H1_resolvable:        {'PASS' if h1_pass else 'FAIL'}"
          f"   ({h1_pass_count}/{len(non_sat_buckets)} non-saturated buckets"
          f" with mean G* != G_BASE)")
    print(f"  H2_pareto:            {'PASS' if h2_pass else 'FAIL'}"
          f"   ({h2_pass_count}/{len(non_sat_buckets)} non-saturated buckets"
          f" Pareto-dominate G=8)")
    print(f"  H3_uniformity:        {'PASS' if h3_pass else 'FAIL'}"
          f"   (method-mean G* spread = {spread:.4f}; < {G_BASE} = uniform)")
    print(f"  H4_signal_efficiency: {'PASS' if h4_pass else 'FAIL'}"
          f"   ({h4_buckets_le_base}/{G_BASE + 1} buckets with mean G* ≤ G_BASE)")
    print()
    print(f"Bootstrap CI (n_boot={N_BOOT}, seed={SEED})")
    print(f"  cost_ratio (ctrl - base): "
          f"{cost_diff[0]:+.4f}  [{cost_diff[1]:+.4f}, {cost_diff[2]:+.4f}]")
    print(f"  contrast_restored (ctrl - 0): "
          f"{cr_diff[0]:+.6f}  [{cr_diff[1]:+.6f}, {cr_diff[2]:+.6f}]")
    print()
    print(f"Outputs:")
    print(f"  {obs_tsv}")
    print(f"  {bucket_tsv}")
    print(f"  {summary_tsv}")


if __name__ == "__main__":
    main()
