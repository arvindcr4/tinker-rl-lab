#!/usr/bin/env python3
"""P8 JOB A (iter 200): base-rate stress test on V-stat feature lift.

Fresh vein: not in any prior P8 row. Prior P8 work measured the cost-optimal
threshold on the held-out test_data.csv with FIXED base rate 1.44% (iter-188,
iter-196). No prior iter asked: does the V-stat feature lift HOLD when the
operational base rate differs from the training distribution?

Real fraud systems see base rates that shift (e.g. new merchant category,
geographic expansion, seasonal pattern). The operational generalization
question is: at what base rate does V_max (the dominant single feature from
iter-196) stop paying for itself?

Pipeline:
  1. Train on fraud_data.csv (50K rows, 1.44% fraud) with 5 feature sets
     {20raw, 24full, 23_noVmean, 23_noVmax, 4sensor} x 3 seeds.
  2. Test on test_data.csv (10K rows, 144 frauds).
  3. For each base-rate target in {0.5%, 1.0%, 1.44% (orig), 2.0%, 3.0%, 5.0%},
     sub-sample negatives to achieve target rate; repeat B=5 resamples.
  4. For each (rate, fset, seed, resample): compute min-cost-per-tx at c=100
     and catch-rate at 1%-FP budget.
  5. Per-(rate, fset): paired bootstrap CI on the gap vs 20raw baseline.
  6. 4 falsifiable hypotheses:
     H1: 24full-20raw cost gap at the lowest rate (0.5%) is still CI-negative.
     H2: 24full-20raw cost gap at the highest rate (5.0%) is still CI-negative.
     H3: V_max dominates at every rate (noVmax retention <= 75% at every rate).
     H4: the cost-gap magnitude INCREASES with base rate (|gap(5.0%)| >
         |gap(0.5%)|) — higher fraud density makes V-stat lift more valuable.

Outputs:
  p8_iter200_base_rate_curve.tsv     5 fsets x 6 rates x 5 resamples x 3 seeds = 450 rows
  p8_iter200_base_rate_min_cost.tsv  same shape aggregated to min-cost-per-tx
  p8_iter200_base_rate_gap.tsv       4 contrasts x 6 rates = 24 rows
  p8_iter200_base_rate_catch.tsv     5 fsets x 6 rates x 5 resamples x 3 seeds = 450 rows
  p8_iter200_base_rate_summary.json  H1..H4 verdicts + retention table

Cost ratio: 100 (headline from iter-188).
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

# 5 feature sets: 20raw baseline, 24full, two key LOO sets from iter-196,
# 4sensor (no raw) — V_mean and V_min/V_std were minor so we drop them
FEATURE_SETS = {
    "20raw": RAW20,
    "24full": ALL24,
    "23_noVmean": [c for c in ALL24 if c != "V_mean"],
    "23_noVmax": [c for c in ALL24 if c != "V_max"],
    "4sensor": AGG4,
}
SEEDS = [42, 179, 316]  # 3 seeds for speed (iter-196 used 5)
BASE_RATES = [0.005, 0.010, 0.0144, 0.020, 0.030, 0.050]  # 6 rates
N_RESAMP = 5  # 5 resamples per (rate, seed)
N_BOOT = 2000
N_TH = 100
C_DEFAULT = 100


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


def make_subsample(Xte, yte, target_rate, rng):
    """Sub-sample negatives to achieve target_rate; keep all positives.

    Returns (X_sub, y_sub, idx_sub) where idx_sub are the chosen row indices
    into the original test set.
    """
    pos_idx = np.where(yte == 1)[0]
    n_pos = len(pos_idx)
    n_total_target = int(round(n_pos / target_rate))
    n_neg_target = n_total_target - n_pos
    if n_neg_target <= 0:
        raise ValueError(f"target_rate={target_rate} too high for {n_pos} positives")
    neg_idx_pool = np.where(yte == 0)[0]
    if n_neg_target >= len(neg_idx_pool):
        # use all negatives + repeat as needed
        chosen_neg = rng.choice(neg_idx_pool, size=n_neg_target, replace=True)
    else:
        chosen_neg = rng.choice(neg_idx_pool, size=n_neg_target, replace=False)
    chosen = np.concatenate([pos_idx, chosen_neg])
    rng.shuffle(chosen)
    return Xte[chosen], yte[chosen], chosen


def cost_curve(s, y, c):
    th = np.linspace(0.0, 1.0, N_TH + 1)[1:]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    N = n_pos + n_neg
    costs = np.empty(N_TH)
    for i, t in enumerate(th):
        flagged = (s >= t).astype(np.int32)
        tp = int(((flagged == 1) & (y == 1)).sum())
        fp = int(((flagged == 1) & (y == 0)).sum())
        fn = n_pos - tp
        costs[i] = (fn * c + fp * 1.0) / N
    return th, costs


def min_cost(s, y, c):
    th, costs = cost_curve(s, y, c)
    i = int(np.argmin(costs))
    return float(costs[i]), float(th[i])


def catch_at_fp(s, y, fp_budget=0.01):
    n = s.shape[0]
    k = max(1, int(round(n * fp_budget)))
    top_idx = np.argsort(-s)[:k]
    return float(y[top_idx].sum()) / max(1, int((y == 1).sum()))


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
    n_pos_orig = int((yte == 1).sum())
    print(f"  train: {Xtr.shape} pos={int((ytr==1).sum())}", flush=True)
    print(f"  test:  {Xte.shape} pos={n_pos_orig} base_rate={n_pos_orig/yte.shape[0]:.4f}", flush=True)

    print("Fitting XGB...", flush=True)
    probas = {}
    for fset, feats in FEATURE_SETS.items():
        for sd in SEEDS:
            s_full = fit_xgb(Xtr, ytr, Xte, feats, sd)
            probas[(fset, sd)] = s_full
            print(f"  {fset} s{sd} mean={float(s_full.mean()):.4f}", flush=True)

    rng_master = np.random.default_rng(20260706)

    # Per-(fset, rate, resample, seed) compute min-cost and catch-rate
    print("Computing subsample min-cost and catch-rate...", flush=True)
    minc = {}  # (fset, rate, seed, resample) -> (min_cost, threshold)
    catch = {}  # (fset, rate, seed, resample) -> catch_rate
    rates_eff = {}  # effective rate
    n_used = {}  # n_used
    for ri, rate in enumerate(BASE_RATES):
        for rr in range(N_RESAMP):
            seed_sub = 1000 * (ri + 1) + rr
            rng = np.random.default_rng(seed_sub)
            Xs, ys, chosen = make_subsample(Xte, yte, rate, rng)
            eff_rate = float(ys.mean())
            rates_eff[(rate, rr)] = eff_rate
            n_used[(rate, rr)] = int(ys.shape[0])
            for fset in FEATURE_SETS:
                for sd in SEEDS:
                    s_full = probas[(fset, sd)]
                    s = s_full[chosen]  # restrict to subsample rows
                    mc, ot = min_cost(s, ys, C_DEFAULT)
                    minc[(fset, rate, sd, rr)] = (mc, ot)
                    catch[(fset, rate, sd, rr)] = catch_at_fp(s, ys, 0.01)
            print(f"  rate={rate:.4f} rr={rr} eff_rate={eff_rate:.4f} n={int(ys.shape[0])}", flush=True)

    # ----- TSV: per-(fset, rate, seed, resample) min-cost + catch -----
    out_curve = RES / "p8_iter200_base_rate_curve.tsv"
    with out_curve.open("w") as f:
        f.write("fset\trate\tresample\tseed\teff_rate\tn\tmin_cost\topt_threshold\tcatch_at_1pct\n")
        for fset in FEATURE_SETS:
            for rate in BASE_RATES:
                for rr in range(N_RESAMP):
                    for sd in SEEDS:
                        mc, ot = minc[(fset, rate, sd, rr)]
                        f.write(f"{fset}\t{rate:.4f}\t{rr}\t{sd}\t"
                                f"{rates_eff[(rate, rr)]:.4f}\t{n_used[(rate, rr)]}\t"
                                f"{mc:.6f}\t{ot:.4f}\t{catch[(fset, rate, sd, rr)]:.4f}\n")

    # ----- Aggregate to (fset, rate) by averaging over (seed, resample) -----
    by_fset_rate = {}  # fset, rate -> list of min_cost values
    catch_by_fset_rate = {}
    for fset in FEATURE_SETS:
        for rate in BASE_RATES:
            vals = [minc[(fset, rate, sd, rr)][0]
                    for sd in SEEDS for rr in range(N_RESAMP)]
            cvals = [catch[(fset, rate, sd, rr)]
                     for sd in SEEDS for rr in range(N_RESAMP)]
            by_fset_rate[(fset, rate)] = vals
            catch_by_fset_rate[(fset, rate)] = cvals

    # ----- TSV: per-(fset, rate) aggregate min-cost -----
    out_minc = RES / "p8_iter200_base_rate_min_cost.tsv"
    with out_minc.open("w") as f:
        f.write("fset\trate\tmean_min_cost\tmedian_min_cost\tmin_min_cost\tmax_min_cost\t"
                "mean_catch_at_1pct\n")
        for fset in FEATURE_SETS:
            for rate in BASE_RATES:
                vals = by_fset_rate[(fset, rate)]
                cvals = catch_by_fset_rate[(fset, rate)]
                f.write(f"{fset}\t{rate:.4f}\t{np.mean(vals):.6f}\t"
                        f"{np.median(vals):.6f}\t{np.min(vals):.6f}\t{np.max(vals):.6f}\t"
                        f"{np.mean(cvals):.4f}\n")

    # ----- Per-(contrast, rate) gap with paired bootstrap CI -----
    # Contrast: 24full-20raw, 23_noVmean-20raw, 23_noVmax-20raw, 4sensor-20raw
    CONTRASTS = ["24full-20raw", "23_noVmean-20raw", "23_noVmax-20raw", "4sensor-20raw"]
    CONTRAST_MAP = {
        "24full-20raw": "24full",
        "23_noVmean-20raw": "23_noVmean",
        "23_noVmax-20raw": "23_noVmax",
        "4sensor-20raw": "4sensor",
    }

    gap_rows = []
    for rate in BASE_RATES:
        for contrast in CONTRASTS:
            target_fset = CONTRAST_MAP[contrast]
            diffs = np.array([
                by_fset_rate[(target_fset, rate)][i]
                - by_fset_rate[("20raw", rate)][i]
                for i in range(len(SEEDS) * N_RESAMP)
            ])
            ci = paired_bootstrap_ci(diffs, N_BOOT, seed=20260706 + int(rate * 10000))
            mean_target = float(np.mean(by_fset_rate[(target_fset, rate)]))
            mean_baseline = float(np.mean(by_fset_rate[("20raw", rate)]))
            gap_rows.append({
                "contrast": contrast,
                "rate": rate,
                "mean_gap": ci["mean"],
                "lo": ci["lo"],
                "hi": ci["hi"],
                "mean_target": mean_target,
                "mean_20raw": mean_baseline,
                "ci_negative": bool(ci["hi"] < 0),
                "ci_positive": bool(ci["lo"] > 0),
            })

    out_gap = RES / "p8_iter200_base_rate_gap.tsv"
    with out_gap.open("w") as f:
        f.write("contrast\trate\tmean_gap\tlo\thi\tmean_target\tmean_20raw\tci_negative\tci_positive\n")
        for r in gap_rows:
            f.write(f"{r['contrast']}\t{r['rate']:.4f}\t{r['mean_gap']:.6f}\t"
                    f"{r['lo']:.6f}\t{r['hi']:.6f}\t{r['mean_target']:.6f}\t"
                    f"{r['mean_20raw']:.6f}\t{int(r['ci_negative'])}\t{int(r['ci_positive'])}\n")

    # ----- TSV: per-(fset, rate, seed, resample) catch rate -----
    out_catch = RES / "p8_iter200_base_rate_catch.tsv"
    with out_catch.open("w") as f:
        f.write("fset\trate\tresample\tseed\tcatch_at_1pct\n")
        for fset in FEATURE_SETS:
            for rate in BASE_RATES:
                for rr in range(N_RESAMP):
                    for sd in SEEDS:
                        f.write(f"{fset}\t{rate:.4f}\t{rr}\t{sd}\t"
                                f"{catch[(fset, rate, sd, rr)]:.4f}\n")

    # ----- Hypotheses -----
    def gap_for(contrast, rate):
        for r in gap_rows:
            if r["contrast"] == contrast and abs(r["rate"] - rate) < 1e-9:
                return r
        return None

    # H1: 24full-20raw CI-negative at lowest rate (0.5%)
    h1 = gap_for("24full-20raw", 0.005)
    h1_pass = bool(h1 and h1["ci_negative"])

    # H2: 24full-20raw CI-negative at highest rate (5.0%)
    h2 = gap_for("24full-20raw", 0.050)
    h2_pass = bool(h2 and h2["ci_negative"])

    # H3: V_max dominates at every rate — 23_noVmax gap at every rate has
    # |gap_noVmax| >= |gap_24full| * 0.50 (i.e. retention <= 75% — close to iter-196's
    # 57% at c=100). Define retention = (gap_noVmax) / (gap_24full) since both are negative.
    h3_pass_count = 0
    h3_total = 0
    for rate in BASE_RATES:
        g_full = gap_for("24full-20raw", rate)
        g_noVmax = gap_for("23_noVmax-20raw", rate)
        if g_full is None or g_noVmax is None:
            continue
        h3_total += 1
        # Both should be negative; retention = |gap_noVmax| / |gap_full|
        if g_full["mean_gap"] >= 0 or g_noVmax["mean_gap"] >= 0:
            continue
        ret = abs(g_noVmax["mean_gap"]) / abs(g_full["mean_gap"])
        if ret <= 0.75:
            h3_pass_count += 1
    h3_pass = (h3_pass_count == h3_total) and h3_total == len(BASE_RATES)

    # H4: |gap_24full| at 5.0% > |gap_24full| at 0.5% (lift grows with rate)
    h4 = gap_for("24full-20raw", 0.050)
    h4_low = gap_for("24full-20raw", 0.005)
    h4_pass = bool(h4 and h4_low and abs(h4["mean_gap"]) > abs(h4_low["mean_gap"]))

    # Retention table for V_max (noVmax) at every rate
    retention = []
    for rate in BASE_RATES:
        g_full = gap_for("24full-20raw", rate)
        g_noVmax = gap_for("23_noVmax-20raw", rate)
        if g_full and g_noVmax and g_full["mean_gap"] < 0 and g_noVmax["mean_gap"] < 0:
            ret = abs(g_noVmax["mean_gap"]) / abs(g_full["mean_gap"])
            retention.append({"rate": rate, "retention_vmax": float(ret)})
        else:
            retention.append({"rate": rate, "retention_vmax": None})

    summary = {
        "iter": 200,
        "vein": "P8 base-rate stress test on V-stat feature lift (c=100, 6 rates, 5 fsets x 3 seeds x 5 resamples)",
        "n_fsets": len(FEATURE_SETS),
        "n_rates": len(BASE_RATES),
        "n_seeds": len(SEEDS),
        "n_resamples": N_RESAMP,
        "cost_ratio": C_DEFAULT,
        "h1_24full_lifts_at_lowest_rate_0_5pct": h1_pass,
        "h2_24full_lifts_at_highest_rate_5_0pct": h2_pass,
        "h3_vmax_dominates_at_every_rate": h3_pass,
        "h3_pass_count": h3_pass_count,
        "h3_total_rates": h3_total,
        "h4_lift_grows_with_rate": h4_pass,
        "h1_gap_low": h1["mean_gap"] if h1 else None,
        "h1_ci_low": h1["ci_negative"] if h1 else None,
        "h2_gap_high": h2["mean_gap"] if h2 else None,
        "h2_ci_high": h2["ci_negative"] if h2 else None,
        "h4_gap_low_rate": h4_low["mean_gap"] if h4_low else None,
        "h4_gap_high_rate": h4["mean_gap"] if h4 else None,
        "vmax_retention_by_rate": retention,
        "verdict_counts": {"PASS": sum([h1_pass, h2_pass, h3_pass, h4_pass]),
                            "FAIL": 4 - sum([h1_pass, h2_pass, h3_pass, h4_pass])},
    }

    out_sum = RES / "p8_iter200_base_rate_summary.json"
    with out_sum.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {out_curve}", flush=True)
    print(f"Wrote {out_minc}", flush=True)
    print(f"Wrote {out_gap}", flush=True)
    print(f"Wrote {out_catch}", flush=True)
    print(f"Wrote {out_sum}", flush=True)
    print(f"H1..H4: {summary['verdict_counts']}", flush=True)


if __name__ == "__main__":
    main()