"""
Iter 89 (P5 — Pillar 1) — Bootstrap CIs + leave-one-method-out stability
on the N2 four-method same-stack panel.

Closes brief veins (b) + (c): "quantify stack-conditioning with the N2
four-method same-stack tensors and the berkeley unpacking_dpo_ppo
factorization (algorithm-axis eta^2 vs stack axes)" + "bootstrap CIs
on every P5 headline number".

Reuses the `axis_variance_fraction` helper from
scripts/berkeley/unpacking_dpo_ppo_factorization.py verbatim. Extends
the iter 85 row 101 point estimates with bootstrap step-resampled 95%
CIs and a leave-one-method-out (LOMO) stability audit on the same N2
panel (40 steps × 4 methods × 1 seed = 160 rows on Qwen2.5-0.5B-MATH).

Falsifiable hypotheses on the N2 four-method panel:
  H1 (CI on algorithm-axis eta^2): the 95% bootstrap CI on every channel
      that passed Ivison strict (eta^2 <= 0.05) in iter 85 stays below
      0.05 on the upper bound. Specifically: `zvf` UB <= 0.08, `pcd` UB
      <= 0.07, `larq` UB <= 0.03, `reward_mean` UB <= 0.04. `mean_len`
      UB <= 0.10 (loose threshold).
  H2 (pair-wise eta^2 stability): when restricted to a single pair of
      methods (6 pairs total), eta^2_pair on the 4 strict-passing
      channels stays <= 0.04 on every pair. The algorithm-axis signal
      is not driven by one outlier pair.
  H3 (LOMO stability on `zvf`): removing any one method from the 4,
      pooled eta^2(zvf|remaining 3 methods) stays in [0.025, 0.075].
      "GIFT is the lone driver" would yield LOMO_grpo / LOMO_aero /
      LOMO_areal close to 0 and LOMO_gift close to 0.20+ -- rejected
      by the measured stability.
  H4 (CI on GIFT dominance): the 95% bootstrap CI on Cohen's d on
      `zvf` last-10-step pooled (GIFT vs other 3) excludes 0 with
      d >= 1.0. The GIFT signal is the strongest single channel-level
      effect and is statistically distinguishable from noise.

Outputs (≤300 LoC, stdlib only):
  - experiments/results/p5p8/p5_n2_unpacking_boot.tsv (per-metric CI table)
  - experiments/results/p5p8/p5_n2_unpacking_pair.tsv (6-pair decomposition)
  - experiments/results/p5p8/p5_n2_unpacking_lomo.tsv (LOMO stability)
  - experiments/results/p5p8/p5_n2_unpacking_boot_summary.json (machine-readable)
"""
from __future__ import annotations
import json, math, os, random
from collections import defaultdict
from itertools import combinations
from statistics import fmean, pstdev

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES  = os.path.join(ROOT, "experiments", "results")
N2   = os.path.join(RES, "n2_reward_tensor_resume", "n2_metrics.tsv")
OUT  = os.path.join(RES, "p5p8")
os.makedirs(OUT, exist_ok=True)

METRICS = ["zvf", "pcd", "larq", "reward_mean", "mean_len", "cv_len", "loss"]
METHODS = ["grpo", "aero", "areal", "gift"]
SEED = 20260705
B = 4000   # bootstrap resamples (paired-step)
ALPHA = 0.05


# ----------------------- helpers -----------------------

def load_rows(path):
    rows = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            d = dict(zip(header, parts))
            for col in ("step", "group_size", "seed"):
                if col in d:
                    d[col] = int(d[col])
            for col in METRICS + ["frac_all_zero", "frac_all_one", "lag1_autocorr"]:
                if col in d and d[col] not in ("nan", "", "None"):
                    try:
                        d[col] = float(d[col])
                    except ValueError:
                        d[col] = float("nan")
            rows.append(d)
    return rows


def axis_eta2(rows, axis_key, value_key):
    """Reuse the Berkeley unpacking machinery. Returns SS_axis/SS_total."""
    grand, by_axis = [], defaultdict(list)
    for r in rows:
        v = r.get(value_key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        grand.append(v)
        by_axis[r[axis_key]].append(v)
    if not grand or len(by_axis) < 2:
        return None
    gm = fmean(grand)
    ss_total = sum((x - gm) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (fmean(vs) - gm) ** 2 for vs in by_axis.values())
    if ss_total <= 1e-12:
        return None
    return ss_axis / ss_total


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = fmean(a), fmean(b)
    va, vb = pstdev(a), pstdev(b)
    pooled = math.sqrt(((len(a) - 1) * va * va + (len(b) - 1) * vb * vb) / (len(a) + len(b) - 2))
    if pooled <= 1e-12:
        return 0.0
    return (ma - mb) / pooled


def paired_step_bootstrap(rows, fn, b=B, seed=SEED):
    """Resample steps with replacement; for each resample build a new
    160-row panel by sampling (step s, all methods at step s) jointly.
    """
    rng = random.Random(seed)
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    steps = sorted(by_step.keys())
    n_steps = len(steps)
    out = []
    for _ in range(b):
        pick = [rng.choice(steps) for _ in range(n_steps)]
        sample = []
        for s in pick:
            sample.extend(by_step[s])
        v = fn(sample)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(v)
    return out


def ci(arr, alpha=ALPHA):
    if not arr:
        return None, None, None
    s = sorted(arr)
    n = len(s)
    lo_i = max(0, int(math.floor((alpha / 2) * n)))
    hi_i = min(n - 1, int(math.ceil((1 - alpha / 2) * n)) - 1)
    return s[lo_i], fmean(arr), s[hi_i]


def reject_in_null(ci_lo, ci_hi, null_value, direction="two-sided"):
    if direction == "two-sided":
        return ci_lo > null_value or ci_hi < null_value
    if direction == "greater":
        return ci_lo > null_value
    if direction == "less":
        return ci_hi < null_value
    raise ValueError(direction)


# ----------------------- main -----------------------

def main():
    rows = load_rows(N2)
    print(f"== Iter 89 P5 N2 unpacking bootstrap (B={B}, seed={SEED}) ==")
    print(f"loaded {len(rows)} rows ({len(METHODS)} methods x {len({r['step'] for r in rows})} steps x 1 seed)")

    # ------------------- H1: bootstrap CI on algorithm-axis eta^2 per metric -------------------
    print("\n--- H1: bootstrap CI on algorithm-axis eta^2 per metric ---")
    h1_table = []
    for metric in METRICS:
        point = axis_eta2(rows, "method", metric)
        boots = paired_step_bootstrap(rows, lambda s, m=metric: axis_eta2(s, "method", m))
        lo, mu, hi = ci(boots)
        ub_lt_005 = hi is not None and hi <= 0.05
        ub_lt_010 = hi is not None and hi <= 0.10
        h1_table.append({
            "metric": metric,
            "n_rows": len(rows),
            "eta2_point": round(point, 4),
            "eta2_boot_mean": round(mu, 4),
            "eta2_boot_lo": round(lo, 4),
            "eta2_boot_hi": round(hi, 4),
            "ub_le_0p05": bool(ub_lt_005),
            "ub_le_0p10": bool(ub_lt_010),
            "H1_ub_le_0p05_pass": bool(ub_lt_005 and metric in ("zvf", "pcd", "larq", "reward_mean", "cv_len")),
            "H1_ub_le_0p10_pass_strict_or_loose": bool(ub_lt_010),
            "n_boot": len(boots),
        })
        tag = "✓ ≤0.05 UB" if ub_lt_005 else ("· ≤0.10 UB" if ub_lt_010 else "✗ >0.10 UB")
        print(f"  {metric:11s}  pt={point:.4f}  boot=[{lo:.4f}, {hi:.4f}]  {tag}")

    # ------------------- H2: pair-wise algorithm-pair eta^2 (binary factor) -------------------
    print("\n--- H2: 6-pair algorithm-pair eta^2 on each strict-passing channel ---")
    strict_metrics = [m for m in METRICS if m != "loss"]  # exclude positive control
    pairs = list(combinations(METHODS, 2))
    h2_table = []
    for metric in strict_metrics:
        for (a, b) in pairs:
            sub = [r for r in rows if r["method"] in (a, b)]
            if len({r["method"] for r in sub}) < 2:
                continue
            point = axis_eta2(sub, "method", metric)
            boots = paired_step_bootstrap(sub, lambda s, m=metric: axis_eta2(s, "method", m), b=B)
            lo, mu, hi = ci(boots)
            ub_lt_004 = hi is not None and hi <= 0.04
            h2_table.append({
                "metric": metric,
                "pair": f"{a}_vs_{b}",
                "eta2_pair_point": round(point, 4),
                "eta2_pair_boot_lo": round(lo, 4),
                "eta2_pair_boot_hi": round(hi, 4),
                "ub_le_0p04": bool(ub_lt_004),
                "n_boot": len(boots),
            })
    # H2 verdict: every strict-passing metric's 6 pairs have UB <= 0.04
    h2_pass = {}
    for metric in strict_metrics:
        pairs_for_m = [h for h in h2_table if h["metric"] == metric]
        all_pass = all(h["ub_le_0p04"] for h in pairs_for_m)
        h2_pass[metric] = all_pass
    h2_pass_all = all(h2_pass.values())
    print(f"  H2 verdict: pair-wise UB<=0.04 on every strict-passing metric = {h2_pass_all}")
    for metric in strict_metrics:
        max_ub = max(h["eta2_pair_boot_hi"] for h in h2_table if h["metric"] == metric)
        print(f"    {metric:11s}  max UB over 6 pairs = {max_ub:.4f}  {'✓' if h2_pass[metric] else '✗'}")

    # ------------------- H3: leave-one-method-out (LOMO) stability on `zvf` -------------------
    print("\n--- H3: LOMO stability on zvf (algorithm-axis eta^2 with one method removed) ---")
    h3_table = []
    for omit in METHODS:
        sub = [r for r in rows if r["method"] != omit]
        if len({r["method"] for r in sub}) < 2:
            continue
        point = axis_eta2(sub, "method", "zvf")
        boots = paired_step_bootstrap(sub, lambda s: axis_eta2(s, "method", "zvf"), b=B)
        lo, mu, hi = ci(boots)
        h3_table.append({
            "omit_method": omit,
            "remaining_methods": ",".join(sorted({r["method"] for r in sub})),
            "eta2_zvf_point": round(point, 4),
            "eta2_zvf_boot_lo": round(lo, 4),
            "eta2_zvf_boot_hi": round(hi, 4),
            "in_band_0p025_0p075": bool(lo is not None and hi is not None and 0.025 <= lo and hi <= 0.075),
        })
    h3_in_band = [h for h in h3_table if h["in_band_0p025_0p075"]]
    h3_pass = len(h3_in_band) == len(h3_table)
    for h in h3_table:
        tag = "✓ in [0.025, 0.075]" if h["in_band_0p025_0p075"] else "✗ outside band"
        print(f"  omit {h['omit_method']:6s}  pt={h['eta2_zvf_point']:.4f}  boot=[{h['eta2_zvf_boot_lo']:.4f}, {h['eta2_zvf_boot_hi']:.4f}]  {tag}")

    # ------------------- H4: bootstrap CI on GIFT Cohen's d (last-10 step pooled) -------------------
    print("\n--- H4: bootstrap CI on GIFT vs others Cohen's d (last-10 step) ---")
    last10 = [r for r in rows if r["step"] >= 30]
    h4_table = []
    for metric in ("zvf", "reward_mean", "pcd"):
        # GIFT vs the rest, paired by step
        rng = random.Random(SEED)
        by_step_gift = {r["step"]: r[metric] for r in last10 if r["method"] == "gift"}
        by_step_rest = defaultdict(list)
        for r in last10:
            if r["method"] != "gift":
                by_step_rest[r["step"]].append(r[metric])
        common_steps = sorted(set(by_step_gift.keys()) & set(by_step_rest.keys()))
        if not common_steps:
            continue
        gift_pt = [by_step_gift[s] for s in common_steps]
        rest_pt = [fmean(by_step_rest[s]) for s in common_steps]
        d_point = cohens_d(gift_pt, rest_pt)
        # paired-step bootstrap: resample steps, recompute d
        boots = []
        for _ in range(B):
            pick = [rng.choice(common_steps) for _ in range(len(common_steps))]
            g_b = [by_step_gift[s] for s in pick]
            r_b = [fmean(by_step_rest[s]) for s in pick]
            d_b = cohens_d(g_b, r_b)
            if not (isinstance(d_b, float) and math.isnan(d_b)):
                boots.append(d_b)
        lo, mu, hi = ci(boots)
        ci_excludes_zero = lo is not None and hi is not None and (lo > 0 or hi < 0)
        ci_lower_ge_1 = lo is not None and lo >= 1.0
        h4_table.append({
            "metric": metric,
            "cohens_d_point": round(d_point, 4),
            "cohens_d_boot_lo": round(lo, 4),
            "cohens_d_boot_hi": round(hi, 4),
            "ci_excludes_zero": bool(ci_excludes_zero),
            "ci_lower_ge_1p0": bool(ci_lower_ge_1),
            "n_boot": len(boots),
        })
        tag = "✓ ≥1.0 LB" if ci_lower_ge_1 else ("✓ excludes 0" if ci_excludes_zero else "✗ includes 0")
        print(f"  {metric:11s}  d_pt={d_point:+.3f}  boot=[{lo:+.3f}, {hi:+.3f}]  {tag}")
    h4_zvf_pass = next((h["ci_lower_ge_1p0"] for h in h4_table if h["metric"] == "zvf"), False)

    # ------------------- summary -------------------
    summary = {
        "iter": 89,
        "vein": "(b)+(c) Bootstrap CIs + leave-one-method-out stability on the N2 four-method same-stack panel",
        "data": {"path": "experiments/results/n2_reward_tensor_resume/n2_metrics.tsv",
                 "n_rows": len(rows), "n_methods": 4, "n_steps": 40, "n_seeds": 1,
                 "methods": METHODS},
        "bootstrap": {"B": B, "seed": SEED, "alpha": ALPHA,
                       "method": "paired-step resample (sample 40 steps w/ replacement; keep all 4 methods at each resampled step)"},
        "h1_pooled_algo_axis_eta2_with_boot_ci": h1_table,
        "h2_pairwise_eta2_with_boot_ci": h2_table,
        "h2_pass_all_pairs_ub_le_0p04": h2_pass_all,
        "h3_lomo_zvf_with_boot_ci": h3_table,
        "h3_pass_all_in_band": h3_pass,
        "h4_gift_dominance_cohens_d_with_boot_ci": h4_table,
        "h4_zvf_lower_bound_ge_1p0": h4_zvf_pass,
    }

    # ------------------- write outputs -------------------
    # 1) bootstrap CI per metric
    with open(os.path.join(OUT, "p5_n2_unpacking_boot.tsv"), "w") as f:
        f.write("metric\tn_rows\teta2_point\teta2_boot_lo\teta2_boot_mean\teta2_boot_hi\tub_le_0p05\tub_le_0p10\tn_boot\n")
        for h in h1_table:
            f.write(f"{h['metric']}\t{h['n_rows']}\t{h['eta2_point']}\t{h['eta2_boot_lo']}\t{h['eta2_boot_mean']}\t"
                    f"{h['eta2_boot_hi']}\t{int(h['ub_le_0p05'])}\t{int(h['ub_le_0p10'])}\t{h['n_boot']}\n")
    # 2) pair-wise eta^2
    with open(os.path.join(OUT, "p5_n2_unpacking_pair.tsv"), "w") as f:
        f.write("metric\tpair\teta2_pair_point\teta2_pair_boot_lo\teta2_pair_boot_hi\tub_le_0p04\tn_boot\n")
        for h in h2_table:
            f.write(f"{h['metric']}\t{h['pair']}\t{h['eta2_pair_point']}\t{h['eta2_pair_boot_lo']}\t"
                    f"{h['eta2_pair_boot_hi']}\t{int(h['ub_le_0p04'])}\t{h['n_boot']}\n")
    # 3) LOMO
    with open(os.path.join(OUT, "p5_n2_unpacking_lomo.tsv"), "w") as f:
        f.write("omit_method\tremaining_methods\teta2_zvf_point\teta2_zvf_boot_lo\teta2_zvf_boot_hi\tin_band_0p025_0p075\n")
        for h in h3_table:
            f.write(f"{h['omit_method']}\t{h['remaining_methods']}\t{h['eta2_zvf_point']}\t"
                    f"{h['eta2_zvf_boot_lo']}\t{h['eta2_zvf_boot_hi']}\t{int(h['in_band_0p025_0p075'])}\n")
    # 4) summary json
    with open(os.path.join(OUT, "p5_n2_unpacking_boot_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nwrote:")
    print(f"  {os.path.join(OUT, 'p5_n2_unpacking_boot.tsv')}")
    print(f"  {os.path.join(OUT, 'p5_n2_unpacking_pair.tsv')}")
    print(f"  {os.path.join(OUT, 'p5_n2_unpacking_lomo.tsv')}")
    print(f"  {os.path.join(OUT, 'p5_n2_unpacking_boot_summary.json')}")
    print()
    print(f"H1 verdict: bootstrap UB <= 0.05 on zvf/pcd/larq/reward_mean/cv_len (strict channels) = {all(h['H1_ub_le_0p05_pass'] for h in h1_table)}")
    print(f"H2 verdict: pair-wise UB <= 0.04 on every strict channel = {h2_pass_all}")
    print(f"H3 verdict: LOMO all in [0.025, 0.075] = {h3_pass}")
    print(f"H4 verdict: zvf CI lower bound >= 1.0 = {h4_zvf_pass}")


if __name__ == "__main__":
    main()