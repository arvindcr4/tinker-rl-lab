#!/usr/bin/env python3
"""P8 JOB A (iter 204): per-V_mean-decile cost-asymmetric savings.

Fresh vein, NOT in any of the 32 prior P8 rows (72, 80, 84, 88, 96, 100, 104,
108, 112, 116, 120, 124, 128, 132, 136, 140, 148, 156, 160, 164, 168, 172,
176, 180, 184, 188, 192, 196, 200).

Iter-192 stratified by V_mean decile and measured Brier calibration lift.
Iter-188 measured cost-asymmetric savings on aggregate cost-per-tx.
Iter-200 stress-tested base rates.

NO prior iter asked: in WHICH V_mean decile does the cost-savings lift
concentrate, and at WHICH cost ratio?

Pipeline:
  1. Train XGB on fraud_data.csv (50K rows, 24 feats) for 3 fsets
     {20raw, 24full, 4sensor} x 5 seeds.
  2. Stratify test_data.csv (10K rows) into 10 V_mean deciles
     (iter-192's stratification).
  3. For each (decile, fset, cost_ratio c, seed), threshold-sweep the cost
     curve cost(t) = (FN(t)*c + FP(t))/N over the decile's transactions.
  4. Per (decile, cost_ratio): paired bootstrap CI (B=2000, seed 20260706)
     on the 24full-20raw gap and the 4sensor-20raw gap.
  5. 5 falsifiable hypotheses:
     H1: Aggregate cost-savings lift ($-X/tx at c=100) matches iter-188
         within $0.001/tx (reproducibility check).
     H2: Per-decile 24full-20raw gap is NON-MONOTONE: some deciles have
         a STRONGER lift than the aggregate (lift concentration).
     H3: Per-decile lift varies by cost ratio: at c=10000, extreme-decile
         lift is strictly larger than aggregate.
     H4: Per-decile lift attribution: top-1 decile accounts for >= 30% of
         total positive lift at c=100 (lift-concentration share).
     H5: 4sensor per-decile lift is strictly negative at every decile
         (no decile where 4sensor alone beats 20raw in cost).

Outputs (under platform_hybrid/experiments/results/p5p8/):
  p8_iter204_decile_cost_curve.tsv   3 fsets x 5 c x 10 deciles x 5 seeds = 750 rows
  p8_iter204_decile_gap.tsv         2 contrasts x 5 c x 10 deciles = 100 rows
  p8_iter204_decile_summary.json    H1..H5 verdicts + lift attribution
  p8_iter204_decile_attribution.tsv per-decile lift share at each c

Cost ratios: c in {1, 10, 100, 1000, 10000} (extends iter-188's {1,10,100,1000}).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RES = ROOT / "experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"

RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
COL_IDX = {c: i for i, c in enumerate(ALL24)}

FEATURE_SETS = {
    "20raw": RAW20,
    "24full": ALL24,
    "4sensor": AGG4,
}
SEEDS = [42, 179, 316, 7, 911]  # 5 seeds for paired bootstrap reproducibility
COST_RATIOS = [1, 10, 100, 1000, 10000]
N_DECILES = 10
N_BOOT = 2000
N_TH = 100
N_REPS = 5  # 5 reps per (fset, c, decile, seed) for paired bootstrap


def load(path):
    X, y = [], []
    with path.open() as f:
        rdr = csv.reader(f)
        header = next(rdr)
        col_idx = {name: i for i, name in enumerate(header)}
        for line in rdr:
            X.append([float(line[col_idx[c]]) for c in ALL24])
            y.append(int(float(line[col_idx["Class"]])))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)


def fit_xgb(Xtr, ytr, Xte, feats, seed):
    import xgboost as xgb
    cols = [COL_IDX[c] for c in feats]
    spw = float((ytr == 0).sum()) / max(1.0, float((ytr == 1).sum()))
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="logloss", random_state=seed,
        tree_method="hist", n_jobs=4)
    m.fit(Xtr[:, cols], ytr, verbose=False)
    return m.predict_proba(Xte[:, cols])[:, 1]


def assign_deciles(v_mean, n_deciles=10):
    """Stratify by V_mean decile on the test set. Returns int array of decile
    indices in [0, n_deciles).
    """
    qs = np.quantile(v_mean, np.linspace(0, 1, n_deciles + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    dec = np.zeros(len(v_mean), dtype=np.int32)
    for i in range(n_deciles):
        mask = (v_mean >= qs[i]) & (v_mean < qs[i + 1])
        dec[mask] = i
    return dec, qs


def min_cost_on_mask(s, y, mask, c):
    """Min cost-per-tx within the masked subset."""
    if mask.sum() == 0 or y[mask].sum() == 0:
        return np.nan, np.nan
    s_sub = s[mask]
    y_sub = y[mask]
    n_pos = int((y_sub == 1).sum())
    n_neg = int((y_sub == 0).sum())
    n_total = n_pos + n_neg
    if n_total == 0:
        return np.nan, np.nan
    th = np.linspace(0.0, 1.0, N_TH + 1)[1:]
    best_cost = float("inf")
    best_th = float("nan")
    for t in th:
        flagged = (s_sub >= t)
        tp = int((flagged & (y_sub == 1)).sum())
        fp = int((flagged & (y_sub == 0)).sum())
        fn = n_pos - tp
        cost = (fn * c + fp * 1.0) / n_total
        if cost < best_cost:
            best_cost = cost
            best_th = float(t)
    return float(best_cost), float(best_th)


def paired_bootstrap_ci(diff, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = diff[idx].mean()
    return {
        "mean": float(diff.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
    }


def main():
    print("Loading data...", flush=True)
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    v_mean_te = Xte[:, COL_IDX["V_mean"]]
    deciles, qs = assign_deciles(v_mean_te, N_DECILES)
    n_per_decile = [int((deciles == i).sum()) for i in range(N_DECILES)]
    pos_per_decile = [int(((deciles == i) & (yte == 1)).sum()) for i in range(N_DECILES)]
    print(f"  train: {Xtr.shape} pos={int((ytr==1).sum())}", flush=True)
    print(f"  test:  {Xte.shape} pos={int((yte==1).sum())}", flush=True)
    for i in range(N_DECILES):
        print(f"  decile {i}: n={n_per_decile[i]} pos={pos_per_decile[i]}", flush=True)

    print("Fitting XGB on 3 fsets x 5 seeds...", flush=True)
    probas = {}
    for fset, feats in FEATURE_SETS.items():
        for sd in SEEDS:
            probas[(fset, sd)] = fit_xgb(Xtr, ytr, Xte, feats, sd)
            print(f"  {fset} s{sd} done", flush=True)

    print("Computing per-(decile, fset, c, seed) min-cost...", flush=True)
    # minc[decile][c][fset][seed] = cost
    minc = {d: {c: {f: {} for f in FEATURE_SETS} for c in COST_RATIOS}
            for d in range(N_DECILES)}
    opt_th = {d: {c: {f: {} for f in FEATURE_SETS} for c in COST_RATIOS}
              for d in range(N_DECILES)}
    for d in range(N_DECILES):
        mask_d = deciles == d
        for c in COST_RATIOS:
            for fset in FEATURE_SETS:
                for sd in SEEDS:
                    s = probas[(fset, sd)]
                    mc, ot = min_cost_on_mask(s, yte, mask_d, c)
                    minc[d][c][fset][sd] = mc
                    opt_th[d][c][fset][sd] = ot

    # ---- Per-(decile, fset, c, seed) cost table (single value per cell) ----
    out_curve = RES / "p8_iter204_decile_cost_curve.tsv"
    with out_curve.open("w") as f:
        f.write("decile\tfset\tcost_ratio\tseed\tn_decile\tpos_decile\tmin_cost\topt_threshold\n")
        for d in range(N_DECILES):
            for fset in FEATURE_SETS:
                for c in COST_RATIOS:
                    for sd in SEEDS:
                        f.write(f"{d}\t{fset}\t{c}\t{sd}\t"
                                f"{n_per_decile[d]}\t{pos_per_decile[d]}\t"
                                f"{minc[d][c][fset][sd]:.6f}\t"
                                f"{opt_th[d][c][fset][sd]:.4f}\n")

    # ---- Per-(decile, c) gap with paired bootstrap CI ----
    # Contrast: 24full-20raw, 4sensor-20raw
    gap_rows = []
    for d in range(N_DECILES):
        for c in COST_RATIOS:
            for contrast in ["24full-20raw", "4sensor-20raw"]:
                target = contrast.split("-")[0]
                diffs = np.array([
                    minc[d][c][target][sd] - minc[d][c]["20raw"][sd]
                    for sd in SEEDS
                ])
                # Filter out NaN diffs (empty decile)
                diffs = diffs[~np.isnan(diffs)]
                if len(diffs) == 0:
                    continue
                ci = paired_bootstrap_ci(diffs, N_BOOT, seed=20260706 + d * 100 + c)
                mean_target = float(np.mean([minc[d][c][target][sd] for sd in SEEDS
                                             if not np.isnan(minc[d][c][target][sd])]))
                mean_baseline = float(np.mean([minc[d][c]["20raw"][sd] for sd in SEEDS
                                               if not np.isnan(minc[d][c]["20raw"][sd])]))
                gap_rows.append({
                    "decile": d, "cost_ratio": c, "contrast": contrast,
                    "mean_gap": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
                    "mean_target": mean_target, "mean_20raw": mean_baseline,
                    "ci_negative": bool(ci["hi"] < 0),
                    "ci_positive": bool(ci["lo"] > 0),
                })

    out_gap = RES / "p8_iter204_decile_gap.tsv"
    with out_gap.open("w") as f:
        f.write("decile\tcost_ratio\tcontrast\tmean_gap\tlo\thi\t"
                "mean_target\tmean_20raw\tci_negative\tci_positive\n")
        for r in gap_rows:
            f.write(f"{r['decile']}\t{r['cost_ratio']}\t{r['contrast']}\t"
                    f"{r['mean_gap']:.6f}\t{r['lo']:.6f}\t{r['hi']:.6f}\t"
                    f"{r['mean_target']:.6f}\t{r['mean_20raw']:.6f}\t"
                    f"{int(r['ci_negative'])}\t{int(r['ci_positive'])}\n")

    # ---- Per-(decile, c) attribution: each decile's share of total negative lift ----
    # At each c, compute 24full-20raw gap per decile, sum positive contributions
    # (we treat cost reductions as positive lift); report share of total.
    attrib_rows = []
    for c in COST_RATIOS:
        per_decile_lift = []
        for d in range(N_DECILES):
            g = next((r for r in gap_rows
                      if r["decile"] == d and r["cost_ratio"] == c
                      and r["contrast"] == "24full-20raw"), None)
            per_decile_lift.append({
                "decile": d,
                "gap": g["mean_gap"] if g else np.nan,
            })
        # Share of total negative gap (cost reduction = -gap, larger is better)
        gaps = np.array([x["gap"] for x in per_decile_lift])
        valid = ~np.isnan(gaps)
        if valid.sum() == 0:
            continue
        total = float(np.nansum(gaps))  # negative if 24full beats 20raw overall
        for x in per_decile_lift:
            if np.isnan(x["gap"]) or total == 0:
                share = np.nan
            else:
                share = float(x["gap"] / total)
            attrib_rows.append({
                "decile": x["decile"], "cost_ratio": c,
                "gap": x["gap"], "share": share,
            })

    out_attr = RES / "p8_iter204_decile_attribution.tsv"
    with out_attr.open("w") as f:
        f.write("decile\tcost_ratio\tgap\tshare\n")
        for r in attrib_rows:
            f.write(f"{r['decile']}\t{r['cost_ratio']}\t"
                    f"{r['gap']:.6f}\t{r['share']:.6f}\n")

    # ---- Hypotheses ----
    def gap_for(contrast, decile, c):
        for r in gap_rows:
            if r["contrast"] == contrast and r["decile"] == decile and r["cost_ratio"] == c:
                return r
        return None

    # ---- Also compute GLOBAL aggregate min-cost for H1 reproducibility check ----
    # (iter-188 used ONE global threshold; per-decile min-cost uses per-decile
    # thresholds which are a different operational regime)
    print("Computing GLOBAL aggregate min-cost for H1 reproducibility...", flush=True)
    global_minc = {c: {f: {} for f in FEATURE_SETS} for c in COST_RATIOS}
    global_th = {c: {f: {} for f in FEATURE_SETS} for c in COST_RATIOS}
    th_grid = np.linspace(0.0, 1.0, N_TH + 1)[1:]
    for c in COST_RATIOS:
        for fset in FEATURE_SETS:
            for sd in SEEDS:
                s = probas[(fset, sd)]
                n_pos = int((yte == 1).sum())
                n_total = len(yte)
                best_cost = float("inf")
                best_th = float("nan")
                for t in th_grid:
                    flagged = (s >= t)
                    tp = int((flagged & (yte == 1)).sum())
                    fp = int((flagged & (yte == 0)).sum())
                    fn = n_pos - tp
                    cost = (fn * c + fp * 1.0) / n_total
                    if cost < best_cost:
                        best_cost = cost
                        best_th = float(t)
                global_minc[c][fset][sd] = float(best_cost)
                global_th[c][fset][sd] = best_th

    # Global aggregate 24full-20raw gap at c=100 (5-seed paired bootstrap CI)
    global_diffs_c100 = np.array([
        global_minc[100]["24full"][sd] - global_minc[100]["20raw"][sd]
        for sd in SEEDS
    ])
    global_ci_c100 = paired_bootstrap_ci(global_diffs_c100, N_BOOT, seed=20260706 + 999)
    global_gap_c100 = global_ci_c100["mean"]

    # H1: Global aggregate 24full-20raw gap at c=100 within $0.001/tx of iter-188.
    # This is a strict reproducibility check vs iter-188's headline number.
    ITER_188_GAP_C100 = -0.01116
    h1_pass = bool(abs(global_gap_c100 - ITER_188_GAP_C100) < 0.001)

    # Decile-N-weighted average of per-decile gaps (a DIFFERENT quantity than
    # global min-cost — uses per-decile optimal thresholds).
    agg_gap_c100 = 0.0
    agg_total_n = 0
    for d in range(N_DECILES):
        g = gap_for("24full-20raw", d, 100)
        if g is not None:
            agg_gap_c100 += g["mean_gap"] * n_per_decile[d]
            agg_total_n += n_per_decile[d]
    if agg_total_n > 0:
        agg_gap_c100 /= agg_total_n

    # H2: Per-decile 24full-20raw gap at c=100 is NON-MONOTONE in decile index:
    # at least one decile has strictly stronger (more negative) gap than the
    # aggregate AND at least one has strictly weaker.
    gaps_c100 = np.array([
        gap_for("24full-20raw", d, 100)["mean_gap"] if gap_for("24full-20raw", d, 100)
        else np.nan
        for d in range(N_DECILES)
    ])
    valid_g = gaps_c100[~np.isnan(gaps_c100)]
    has_stronger = bool(np.any(valid_g < agg_gap_c100))
    has_weaker = bool(np.any(valid_g > agg_gap_c100))
    h2_pass = has_stronger and has_weaker

    # H3: Per-decile lift attribution at c=10000 — top-1 decile (by most-negative
    # gap) accounts for >= 30% of total positive lift share.
    gaps_c10000 = np.array([
        gap_for("24full-20raw", d, 10000)["mean_gap"] if gap_for("24full-20raw", d, 10000)
        else np.nan
        for d in range(N_DECILES)
    ])
    valid_c10k = gaps_c10000[~np.isnan(gaps_c10000)]
    total_c10k = float(np.nansum(valid_c10k))
    if total_c10k < 0:  # only meaningful if 24full beats 20raw overall
        top1_share = float(np.nanmin(valid_c10k) / total_c10k)  # most-negative share
        h3_pass = bool(abs(top1_share) >= 0.30)
    else:
        top1_share = np.nan
        h3_pass = False

    # H4: Per-decile lift concentration share at c=100: top-1 decile accounts
    # for >= 30% of total positive lift (negative gap share).
    if agg_gap_c100 < 0:
        top1_share_c100 = float(np.nanmin(valid_g) / float(np.nansum(valid_g)))
        h4_pass = bool(abs(top1_share_c100) >= 0.30)
    else:
        top1_share_c100 = np.nan
        h4_pass = False

    # H5: 4sensor per-decile lift is strictly negative at every decile (c=100).
    h5_pass_count = 0
    h5_total = 0
    for d in range(N_DECILES):
        g = gap_for("4sensor-20raw", d, 100)
        if g is None:
            continue
        h5_total += 1
        if g["mean_gap"] > 0:  # 4sensor worse than 20raw
            h5_pass_count += 1
    h5_pass = (h5_pass_count == h5_total) and h5_total == N_DECILES

    summary = {
        "iter": 204,
"vein": ("P8 per-V_mean-decile cost-asymmetric savings (extends iter-192 "
                 "+ iter-188 with 5 cost ratios and 10 V_mean deciles on the same pool)"),
        "n_fsets": len(FEATURE_SETS),
        "n_seeds": len(SEEDS),
        "n_deciles": N_DECILES,
        "cost_ratios": COST_RATIOS,
        "n_per_decile": n_per_decile,
        "pos_per_decile": pos_per_decile,
        "h1_global_gap_within_001_of_iter188": h1_pass,
        "h1_global_gap_c100": global_gap_c100,
        "h1_global_ci_c100": global_ci_c100,
        "h1_iter188_reference": ITER_188_GAP_C100,
        "h1_decile_weighted_gap_c100": agg_gap_c100,
        "h2_per_decile_non_monotone": h2_pass,
        "h3_top1_share_c10000_ge_30pct": h3_pass,
        "h3_top1_share_c10000": top1_share,
        "h4_top1_share_c100_ge_30pct": h4_pass,
        "h4_top1_share_c100": top1_share_c100,
        "h5_4sensor_strictly_negative_per_decile": h5_pass,
        "h5_pass_count": h5_pass_count,
        "h5_total_deciles": h5_total,
        "per_decile_gaps_c100": gaps_c100.tolist(),
        "per_decile_gaps_c10000": gaps_c10000.tolist(),
        "verdict_counts": {
            "PASS": sum([h1_pass, h2_pass, h3_pass, h4_pass, h5_pass]),
            "FAIL": 5 - sum([h1_pass, h2_pass, h3_pass, h4_pass, h5_pass]),
        },
    }

    out_sum = RES / "p8_iter204_decile_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_curve}", flush=True)
    print(f"Wrote {out_gap}", flush=True)
    print(f"Wrote {out_attr}", flush=True)
    print(f"Wrote {out_sum}", flush=True)
    print(f"H1..H5: {summary['verdict_counts']}", flush=True)
    print(f"  aggregate gap c=100: {agg_gap_c100:.6f} (iter-188: {ITER_188_GAP_C100})", flush=True)
    print(f"  top-1 share c=100: {top1_share_c100}", flush=True)
    print(f"  top-1 share c=10000: {top1_share}", flush=True)
    print(f"  per-decile gaps c=100: {gaps_c100.tolist()}", flush=True)


if __name__ == "__main__":
    main()