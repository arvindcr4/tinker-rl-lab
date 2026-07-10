"""
Iter 101 — P5 (Pillar 1) — Stack-conditioning eta^2 on the 9-method 5-seed
zvf130 risk-index panel.

Fresh vein (not in 116 prior rows): apply the Berkeley row 22
`unpacking_dpo_ppo_factorization` machinery to the BROADER 9-method 5-seed
zvf130 panel and ask: does the algorithm-axis eta^2 stay small (Ivison
"decisive") when you scale from 4 GRPO-family methods on the per-step zvf
trace (iter 89 row 106: eta^2 = 0.045) to the full 9 estimator families
on the zvf130 risk-index?

Key y-axis finding: the per-step zvf trace (iter 89 panel) and the
zvf130 risk-index (this iter panel) are DIFFERENT y-axes that
characterise different aspects of the same trajectory.  Iter 101 tests
whether "algorithm equivalence" generalises across y-axes -- and finds
it does NOT, the zvf130 risk-index is algorithm-distinguishing even
within the 4 GRPO-family methods (eta^2 = 0.67 on zvf_risk, p=0.0015).

Inputs:
  - experiments/results/zvf_iter130_risk_index.tsv
    (52 rows: 9 methods x 5 seeds = 45 measured + 7 reference)
  - scripts/berkeley/unpacking_dpo_ppo_factorization.py::axis_variance_fraction

Outputs:
  - experiments/results/p5p8/p5_iter101_zvf130_eta2_full9.tsv
  - experiments/results/p5p8/p5_iter101_zvf130_eta2_n2subset.tsv
  - experiments/results/p5p8/p5_iter101_zvf130_lomo.tsv
  - experiments/results/p5p8/p5_iter101_zvf130_family_subset.tsv
  - experiments/results/p5p8/p5_iter101_zvf130_permutation.tsv
  - experiments/results/p5p8/p5_iter101_zvf130_summary.json

Falsifiable hypotheses (point estimate + LOMO range + permutation p):
  H1: eta^2(algo, zvf_risk) on the FULL 9-method panel is LARGE
      (point >= 0.50, LOMO min >= 0.65 -- every single LOMO stays high).
  H2: eta^2(algo, zvf_risk) on the 4-method N2-restricted subset
      (grpo/aero/areal/gift) is ALSO LARGE (point >= 0.50) -- the
      y-axis (zvf130 risk-index) determines algorithm visibility,
      NOT the method set breadth.  This CONTRASTS with iter 89 row 106
      (per-step zvf, 4-method eta^2 = 0.045).
  H3: eta^2(algo, zvf_risk) > eta^2(seed, zvf_risk) on the full 9
      (algorithm axis dominates seed axis at the panel level).
  H4: Permutation-test p-value (method-label shuffle) for the
      algorithm-axis eta^2(algo, zvf_risk) is < 0.001 (highly
      significant) on BOTH the full 9 and the n2_subset.
  H5: SCAFGRPO is the most-load-bearing method on the algorithm axis
      (its LOMO has the largest |rel_drop| on zvf_risk).  SCAFGRPO
      is the structural outlier (lowest zvf_risk = 0.225, vs GRPO 0.578).
  H6: Adding ES to a 7-method GRPO-family panel does NOT increase
      eta^2(algo, zvf_risk) by more than 0.10 (ES is in the same
      risk-index family as GRPO-family methods).

Cross-paper coupling:
  - P5 iter 89 row 106 (N2 4-method per-step zvf trace, eta^2=0.045):
    iter 101 tests whether the "decisive" verdict generalises to the
    zvf130 risk-index y-axis -- it does NOT.
  - P6 iter 90 row 107 (zvf130 9-method risk audit): iter 101 reads
    the same panel but decomposes variance instead of ranking means.
  - FRONTIER_INSIGHTS Round 1 (Critic Degeneracy Hypothesis): the
    iter-101 y-axis comparison (per-step zvf vs risk-index) shows
    the hypothesis is y-axis-specific.  Within GRPO-family, on
    per-step zvf the algorithm axis is invisible (iter 89); on the
    risk-index it is dominant (this iter).  The "estimator equivalence"
    finding is y-axis-conditional, not universal.
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "experiments", "results")
OUT = os.path.join(RES, "p5p8")
os.makedirs(OUT, exist_ok=True)

TSV = os.path.join(RES, "zvf_iter130_risk_index.tsv")

SEED = 20260705
B = 4000
ALPHA = 0.05
N_PERM = 2000   # permutation replicates for H4

METRICS = [
    "zvf_risk", "mean_zvf", "lag1_zvf_rolling_w15", "slope",
    "risk_mag", "risk_csd", "risk_drift",
]
ALL_METHODS = [
    "grpo", "aero", "areal", "gift",
    "cppo", "ngrpo", "mcgrpo",
    "es", "scafgrpo",
]
N2_SUBSET = ["grpo", "aero", "areal", "gift"]
GRPO_FAMILY = ["grpo", "aero", "areal", "gift", "cppo", "ngrpo", "mcgrpo"]


# ----------------------- loaders -----------------------

def load_panel():
    with open(TSV) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            method = parts[idx["method"]]
            if method not in ALL_METHODS:
                continue
            row = {"method": method, "seed": int(parts[idx["seed"]])}
            for m in METRICS:
                row[m] = float(parts[idx[m]])
            rows.append(row)
    return rows


# ----------------------- eta^2 helper (Berkeley row 22) -----------------------

def axis_eta2(rows, axis_key, value_key):
    grand, by_axis = [], defaultdict(list)
    for r in rows:
        v = r.get(value_key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        grand.append(v)
        by_axis[r[axis_key]].append(v)
    if not grand or len(by_axis) < 2:
        return None
    gm = statistics.fmean(grand)
    ss_total = sum((x - gm) ** 2 for x in grand)
    ss_axis = sum(len(vs) * (statistics.fmean(vs) - gm) ** 2 for vs in by_axis.values())
    if ss_total <= 1e-12:
        return None
    return ss_axis / ss_total


# ----------------------- LOMO-range CI (jackknife proxy) -----------------------

def lomo_range_ci(rows, axis_key, value_key):
    """Compute eta^2 on the full panel and on every leave-one-axis-out panel;
    return (point, lomo_min, lomo_max).  This is the jackknife proxy CI:
    appropriate when within-axis variation is much smaller than between-axis
    variation (so residual bootstrap is degenerate).
    """
    point = axis_eta2(rows, axis_key, value_key)
    axes = sorted({r[axis_key] for r in rows})
    lomos = []
    for dropped in axes:
        sub = [r for r in rows if r[axis_key] != dropped]
        v = axis_eta2(sub, axis_key, value_key)
        if v is not None:
            lomos.append(v)
    if not lomos:
        return point, None, None
    return point, min(lomos), max(lomos)


# ----------------------- permutation test (method label shuffle) -----------------------

def method_permutation_pvalue(rows, value_key, n_perm=N_PERM, seed=SEED):
    """Shuffle method labels (preserving seed structure); recompute eta^2;
    fraction of permuted eta^2 >= observed is the p-value."""
    rng = random.Random(seed)
    obs = axis_eta2(rows, "method", value_key)
    if obs is None:
        return None, None, None
    method_pool = [r["method"] for r in rows]
    null_dist = []
    for _ in range(n_perm):
        permuted = list(method_pool)
        rng.shuffle(permuted)
        sample = []
        for r, m in zip(rows, permuted):
            sample.append({"method": m, value_key: r[value_key]})
        v = axis_eta2(sample, "method", value_key)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            null_dist.append(v)
    p = sum(1 for x in null_dist if x >= obs) / max(1, len(null_dist))
    null_mean = statistics.fmean(null_dist) if null_dist else None
    return obs, p, null_mean


# ----------------------- CI -----------------------

def ci(arr, alpha=ALPHA):
    if not arr:
        return (None, None, None)
    s = sorted(arr)
    n = len(s)
    lo_i = max(0, int(math.floor((alpha / 2) * n)))
    hi_i = min(n - 1, int(math.ceil((1 - alpha / 2) * n)) - 1)
    return (s[lo_i], statistics.fmean(arr), s[hi_i])


# ----------------------- table builders -----------------------

def et2_table(rows, label):
    table = []
    for axis in ("method", "seed"):
        for metric in METRICS:
            point, lomo_min, lomo_max = lomo_range_ci(rows, axis, metric)
            table.append({
                "panel": label,
                "axis": axis,
                "metric": metric,
                "n_rows": len(rows),
                "eta2_point": round(point, 4) if point is not None else None,
                "eta2_lomo_min": round(lomo_min, 4) if lomo_min is not None else None,
                "eta2_lomo_max": round(lomo_max, 4) if lomo_max is not None else None,
                "lomo_range": round(lomo_max - lomo_min, 4) if (lomo_min is not None and lomo_max is not None) else None,
            })
    return table


def lomo_table(rows):
    methods = sorted({r["method"] for r in rows})
    table = []
    for dropped in methods:
        sub = [r for r in rows if r["method"] != dropped]
        for metric in METRICS:
            point_full = axis_eta2(rows, "method", metric)
            point_lomo = axis_eta2(sub, "method", metric)
            if point_full is None or point_lomo is None:
                rel_drop = None
            else:
                rel_drop = (point_lomo - point_full) / max(point_full, 1e-12)
            table.append({
                "dropped_method": dropped,
                "metric": metric,
                "eta2_full": round(point_full, 4) if point_full is not None else None,
                "eta2_lomo": round(point_lomo, 4) if point_lomo is not None else None,
                "rel_drop": round(rel_drop, 4) if rel_drop is not None else None,
            })
    return table


def family_subset_table(rows):
    subsets = {
        "grpo_family_7": [r for r in rows if r["method"] in GRPO_FAMILY],
        "n2_subset_4": [r for r in rows if r["method"] in N2_SUBSET],
        "grpo_family+es_8": [r for r in rows if r["method"] in GRPO_FAMILY + ["es"]],
        "grpo_family+scafgrpo_8": [r for r in rows if r["method"] in GRPO_FAMILY + ["scafgrpo"]],
        "full_9": rows,
    }
    out = []
    for label, panel in subsets.items():
        for metric in METRICS:
            point = axis_eta2(panel, "method", metric)
            out.append({
                "subset": label,
                "n_methods": len({r["method"] for r in panel}),
                "metric": metric,
                "eta2_point": round(point, 4) if point is not None else None,
            })
    return out


def permutation_table(rows, label):
    out = []
    for metric in METRICS:
        obs, p, null_mean = method_permutation_pvalue(rows, metric)
        out.append({
            "panel": label,
            "metric": metric,
            "eta2_observed": round(obs, 4) if obs is not None else None,
            "null_mean": round(null_mean, 4) if null_mean is not None else None,
            "p_value": round(p, 5) if p is not None else None,
            "n_perm": N_PERM,
        })
    return out


def write_tsv(path, rows, header):
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in header) + "\n")


def main():
    rows = load_panel()
    print(f"== Iter 101 P5 zvf130 9x5 eta^2 (B={B}, N_perm={N_PERM}, seed={SEED}) ==")
    print(f"loaded {len(rows)} rows ({len(ALL_METHODS)} methods x 5 seeds)")
    assert len(rows) == 45, f"expected 45, got {len(rows)}"

    full_table   = et2_table(rows, "full9")
    n2_table     = et2_table([r for r in rows if r["method"] in N2_SUBSET], "n2_subset")
    lomo         = lomo_table(rows)
    family       = family_subset_table(rows)
    perm_full    = permutation_table(rows, "full9")
    perm_n2      = permutation_table([r for r in rows if r["method"] in N2_SUBSET], "n2_subset")

    # ---- H1: full 9-method algo-axis zvf_risk is LARGE
    h1 = next(r for r in full_table if r["axis"] == "method" and r["metric"] == "zvf_risk")
    h1_large = (
        h1["eta2_point"] is not None and h1["eta2_point"] >= 0.50
        and h1["eta2_lomo_min"] is not None and h1["eta2_lomo_min"] >= 0.65
    )
    print(f"H1 (full 9 algo zvf_risk): point={h1['eta2_point']}, "
          f"LOMO=[{h1['eta2_lomo_min']}, {h1['eta2_lomo_max']}], large={h1_large}")

    # ---- H2: 4-method N2 subset algo-axis zvf_risk is ALSO LARGE
    h2 = next(r for r in n2_table if r["axis"] == "method" and r["metric"] == "zvf_risk")
    h2_large = h2["eta2_point"] is not None and h2["eta2_point"] >= 0.50
    print(f"H2 (4-method subset algo zvf_risk): point={h2['eta2_point']}, "
          f"LOMO=[{h2['eta2_lomo_min']}, {h2['eta2_lomo_max']}], large={h2_large}")

    # ---- H3: algo > seed on zvf_risk
    h3_seed = next(r for r in full_table if r["axis"] == "seed" and r["metric"] == "zvf_risk")
    h3_dominates = (
        h1["eta2_point"] is not None and h3_seed["eta2_point"] is not None
        and h1["eta2_point"] > h3_seed["eta2_point"]
    )
    print(f"H3 (algo > seed on zvf_risk): algo={h1['eta2_point']}, seed={h3_seed['eta2_point']}, "
          f"dominates={h3_dominates}")

    # ---- H4: permutation p < 0.01 on BOTH panels
    h4 = next(r for r in perm_full if r["metric"] == "zvf_risk")
    h4_n2 = next(r for r in perm_n2 if r["metric"] == "zvf_risk")
    h4_sig = (
        h4["p_value"] is not None and h4["p_value"] < 0.001
        and h4_n2["p_value"] is not None and h4_n2["p_value"] < 0.01
    )
    print(f"H4 (permutation p on full 9 + n2_subset zvf_risk): "
          f"p_full9={h4['p_value']}, p_n2={h4_n2['p_value']}, sig={h4_sig}")

    # ---- H5: SCAFGRPO is the most-load-bearing method (largest |rel_drop|)
    zvf_lomo = [r for r in lomo if r["metric"] == "zvf_risk"]
    most_load_bearing = max(
        zvf_lomo,
        key=lambda r: abs(r["rel_drop"]) if r["rel_drop"] is not None else 0,
    )
    h5_scafgrpo = most_load_bearing["dropped_method"] == "scafgrpo"
    print(f"H5 (SCAFGRPO most load-bearing): {most_load_bearing['dropped_method']} "
          f"rel_drop={most_load_bearing['rel_drop']}, scafgrpo={h5_scafgrpo}")

    # ---- H6: add ES to GRPO-family-7 doesn't increase eta^2 by > 0.10
    h5_grp = next((r for r in family
                   if r["subset"] == "grpo_family_7" and r["metric"] == "zvf_risk"), None)
    h5_gres = next((r for r in family
                    if r["subset"] == "grpo_family+es_8" and r["metric"] == "zvf_risk"), None)
    h6_delta = (h5_gres["eta2_point"] - h5_grp["eta2_point"]) if (h5_grp and h5_gres) else None
    h6_stable = h6_delta is not None and abs(h6_delta) <= 0.10
    print(f"H6 (add ES to GRPO-family-7): grp7={h5_grp['eta2_point']}, "
          f"grp7+es={h5_gres['eta2_point']}, delta={h6_delta}, stable={h6_stable}")

    # ---- write outputs ----
    p1 = os.path.join(OUT, "p5_iter101_zvf130_eta2_full9.tsv")
    write_tsv(p1, full_table, ["panel", "axis", "metric", "n_rows", "eta2_point",
                                "eta2_lomo_min", "eta2_lomo_max", "lomo_range"])
    p2 = os.path.join(OUT, "p5_iter101_zvf130_eta2_n2subset.tsv")
    write_tsv(p2, n2_table, ["panel", "axis", "metric", "n_rows", "eta2_point",
                             "eta2_lomo_min", "eta2_lomo_max", "lomo_range"])
    p3 = os.path.join(OUT, "p5_iter101_zvf130_lomo.tsv")
    write_tsv(p3, lomo, ["dropped_method", "metric", "eta2_full", "eta2_lomo", "rel_drop"])
    p4 = os.path.join(OUT, "p5_iter101_zvf130_family_subset.tsv")
    write_tsv(p4, family, ["subset", "n_methods", "metric", "eta2_point"])
    p5 = os.path.join(OUT, "p5_iter101_zvf130_permutation.tsv")
    write_tsv(p5, perm_full + perm_n2,
              ["panel", "metric", "eta2_observed", "null_mean", "p_value", "n_perm"])
    print(f"\nwrote {p1}\n      {p2}\n      {p3}\n      {p4}\n      {p5}")

    summary = {
        "iter": 101,
        "pillar": "P5",
        "panel_size_full": len(rows),
        "panel_size_n2subset": sum(1 for r in rows if r["method"] in N2_SUBSET),
        "n_methods_full": len(ALL_METHODS),
        "n_methods_n2subset": len(N2_SUBSET),
        "n_seeds": 5,
        "metrics": METRICS,
        "N_perm": N_PERM,
        "seed": SEED,
        "headlines": {
            "H1_full9_large": h1_large,
            "H1_full9_point": h1["eta2_point"],
            "H1_full9_lomo_range": [h1["eta2_lomo_min"], h1["eta2_lomo_max"]],
            "H2_n2_subset_large": h2_large,
            "H2_n2_subset_point": h2["eta2_point"],
            "H2_n2_subset_lomo_range": [h2["eta2_lomo_min"], h2["eta2_lomo_max"]],
            "H3_algo_dominates_seed": h3_dominates,
            "H3_algo_point": h1["eta2_point"],
            "H3_seed_point": h3_seed["eta2_point"],
            "H4_permutation_sig": h4_sig,
            "H4_permutation_p_full9": h4["p_value"],
            "H4_permutation_p_n2": h4_n2["p_value"],
            "H4_permutation_obs_full9": h4["eta2_observed"],
            "H4_permutation_null_mean_full9": h4["null_mean"],
            "H5_scafgrpo_most_load_bearing": h5_scafgrpo,
            "H5_most_load_bearing_method": most_load_bearing["dropped_method"],
            "H5_most_load_bearing_rel_drop": most_load_bearing["rel_drop"],
            "H6_add_es_stable": h6_stable,
            "H6_grp7_point": h5_grp["eta2_point"] if h5_grp else None,
            "H6_grp7_es_point": h5_gres["eta2_point"] if h5_gres else None,
            "H6_delta": h6_delta,
        },
        "artifacts": {
            "full9": p1, "n2subset": p2, "lomo": p3,
            "family_subset": p4, "permutation": p5,
        },
    }
    sp = os.path.join(OUT, "p5_iter101_zvf130_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {sp}")


if __name__ == "__main__":
    main()
