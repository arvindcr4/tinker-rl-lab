#!/usr/bin/env python3
"""P8 JOB A (iter 196): V-stat feature leave-one-out (LOO) ablation on
held-out test_data.csv with paired bootstrap CIs on cost, catch-rate,
Brier, and AUC gaps.

Fresh vein: not in any prior P8 row. Prior P8 work measured AGGREGATE
4-sensor features (iter-176 sensor/scribe/scorer; iter-188 cost-asymmetric
transfer with all 4 V-stat features) but NEVER asked which individual
feature drives the lift. Iter-184 stratified BY V_std as a covariate.
Iter-188 dropped V-stat entirely (4sensor vs 24full vs 20raw).

This iter surgically removes each V-stat feature in turn (23-feature sets)
and measures the cost gap, catch-rate gap, Brier gap, and AUC gap with
paired bootstrap CI across 5 seeds.

Pipeline:
  1. Train on fraud_data.csv (50K rows, 719 frauds).
  2. Test on test_data.csv (10K rows, 144 frauds).
  3. For each fset in {20raw, 24full, 23-V_mean, 23-V_std, 23-V_max,
     23-V_min, 4sensor}, fit XGB-200 with 5 seeds.
  4. For each (fset, c, seed) compute min-cost-per-tx (c in {1, 10, 100,
     1000}) and per-fset recall at total flag budget 1% + Brier + AUC.
  5. Per-(c, seed), bootstrap B=2000 paired-seed CI on the gap
        gap_f = min_cost(23_minus_f, c) - min_cost(20raw, c)
     and compare to the FULL 24full-20raw baseline gap.
  6. 6 falsifiable hypotheses:
       H1: dropping V_mean alone costs >= 50% of the full 24full lift.
       H2: dropping V_std alone costs >= 50% of the full 24full lift.
       H3: dropping V_max alone costs >= 50% of the full 24full lift.
       H4: dropping V_min alone costs >= 50% of the full 24full lift.
       H5: dropping V_std alone produces the LARGEST AUC drop
           (V_std is the most discriminative aggregate).
       H6: the sum-of-LOO contributions approximates the full 24full lift
           (linearity check on aggregate 4-feature marginality).

Outputs:
  p8_iter196_loo_cost_curve.tsv     7 fsets x 4 c x 100 thresholds x 5 seeds
  p8_iter196_loo_min_cost.tsv       7 fsets x 4 c x 5 seeds = 140 rows
  p8_iter196_loo_min_cost_gap.tsv   6 contrasts x 4 c = 24 rows
  p8_iter196_loo_catch.tsv          7 fsets x 4 c x 5 seeds = 140 rows
  p8_iter196_loo_brier.tsv          7 fsets x 5 seeds = 35 rows
  p8_iter196_loo_auc.tsv            7 fsets x 5 seeds = 35 rows
  p8_iter196_loo_summary.json       H1..H6 verdicts + linearity check

Cost ratios: 1, 10, 100, 1000. Default cost-optimal headline at c=100.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # platform_modal/scripts, for _paths
from _review02_contracts import stable_seed, finite_json
from _analysis_common import auc_rank
from _paths import REPO_ROOT
from _p8_common import (
    RAW20,
    AGG4,
    ALL24,
    TRAIN,
    TEST,
    load_v2 as load,
    fit_xgb,
    paired_bootstrap_ci,
)

ROOT = REPO_ROOT
RES = ROOT / "platform_hybrid/experiments" / "results" / "p5p8"
RES.mkdir(parents=True, exist_ok=True)


# 7 feature sets: 20raw, 24full, four 23-feature LOO sets, 4sensor
FEATURE_SETS = {
    "20raw": RAW20,
    "24full": ALL24,
    "23_noVmean": [c for c in ALL24 if c != "V_mean"],
    "23_noVstd": [c for c in ALL24 if c != "V_std"],
    "23_noVmax": [c for c in ALL24 if c != "V_max"],
    "23_noVmin": [c for c in ALL24 if c != "V_min"],
    "4sensor": AGG4,
}
# Mapping from feature name (with underscore) to LOO fset key (no underscore)
LOO_KEY = {
    "V_mean": "23_noVmean",
    "V_std": "23_noVstd",
    "V_max": "23_noVmax",
    "V_min": "23_noVmin",
}
SEEDS = [42, 179, 316, 453, 590]
COST_RATIOS = [1, 10, 100, 1000]
N_BOOT = 2000
N_TH = 100


def cost_curve(s, y, c):
    th = np.linspace(0.0, 1.0, N_TH + 1)[1:]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    N = n_pos + n_neg
    costs = np.empty(N_TH)
    flagged_frac = np.empty(N_TH)
    for i, t in enumerate(th):
        flagged = (s >= t).astype(np.int32)
        tp = int(((flagged == 1) & (y == 1)).sum())
        fp = int(((flagged == 1) & (y == 0)).sum())
        fn = n_pos - tp
        costs[i] = (fn * c + fp * 1.0) / N
        flagged_frac[i] = (tp + fp) / N
    return th, costs, flagged_frac


def operating_rates(s, y, threshold):
    s, y = np.asarray(s), np.asarray(y)
    flags = s >= threshold
    n = len(y)
    fp = int((flags & (y == 0)).sum())
    negatives = int((y == 0).sum())
    return dict(
        flag_rate=float(flags.mean()) if n else float("nan"),
        fp_per_transaction=fp / n if n else float("nan"),
        false_positive_rate=fp / negatives if negatives else float("nan"),
    )


def min_cost(s, y, c):
    """Empirical oracle minimum over all breakpoints; never prospective test risk.

    For a deployed threshold, call this ONLY on validation labels, then evaluate
    the frozen threshold on test labels using cost_at_threshold.
    """
    s, y = np.asarray(s, float), np.asarray(y)
    if (
        len(s) != len(y)
        or not len(s)
        or not np.isfinite(s).all()
        or not np.isin(y, [0, 1]).all()
        or c < 0
    ):
        raise ValueError("finite scores, binary labels, and nonnegative costs required")
    thresholds = np.r_[np.unique(s), np.nextafter(s.max(), np.inf)]
    costs = [cost_at_threshold(s, y, c, t) for t in thresholds]
    i = int(np.argmin(costs))
    return float(costs[i]), float(thresholds[i])


def cost_at_threshold(s, y, c, threshold):
    flagged = np.asarray(s) >= threshold
    y = np.asarray(y)
    return float((c * ((~flagged) & (y == 1)).sum() + (flagged & (y == 0)).sum()) / len(y))


def recall_at_flag_budget(s, y, flag_budget=0.01):
    """Recall under a total flag-count budget; ties use stable input order."""
    s, y = np.asarray(s), np.asarray(y)
    n_pos = int((y == 1).sum())
    if not len(y) or n_pos == 0:
        return float("nan")
    k = min(len(s), max(0, int(np.floor(len(s) * flag_budget))))
    top_idx = np.argsort(-s, kind="stable")[:k]
    return float(y[top_idx].sum()) / n_pos


def most_damaging_features(auc_full, loo_aucs):
    if (
        not np.isfinite(auc_full)
        or not loo_aucs
        or not all(np.isfinite(x) for x in loo_aucs.values())
    ):
        return []
    drops = {key: auc_full - value for key, value in loo_aucs.items()}
    maximum = max(drops.values())
    return sorted(
        key for key, value in drops.items() if np.isclose(value, maximum, rtol=0, atol=1e-12)
    )


def brier(s, y):
    return float(((s - y) ** 2).mean())


def auc(s, y):
    return auc_rank(np.asarray(y).tolist(), np.asarray(s).tolist())


def main():
    print("Loading data...", flush=True)
    Xtr, ytr = load(TRAIN)
    Xte, yte = load(TEST)
    print(f"  train: {Xtr.shape} pos={int((ytr == 1).sum())}", flush=True)
    print(f"  test:  {Xte.shape} pos={int((yte == 1).sum())}", flush=True)

    # Fit all 7 fsets × 5 seeds
    print("Fitting XGB...", flush=True)
    probas = {}
    for fset, feats in FEATURE_SETS.items():
        for sd in SEEDS:
            s = fit_xgb(Xtr, ytr, Xte, feats, sd)
            probas[(fset, sd)] = s
            print(
                f"  {fset} s{sd} mean={float(s.mean()):.4f} top1%={float(np.partition(s, -int(0.01 * s.shape[0]))[-int(0.01 * s.shape[0]) :].mean()):.4f}",
                flush=True,
            )

    # Per-(fset, c, seed) min-cost and threshold
    print("Computing cost curves and metrics...", flush=True)
    minc = {}
    optt = {}
    catches = {}
    briers = {}
    aucs = {}
    for (fset, sd), s in probas.items():
        briers[(fset, sd)] = brier(s, yte)
        aucs[(fset, sd)] = auc(s, yte)
        for c in COST_RATIOS:
            mc, t_opt = min_cost(s, yte, c)
            minc[(fset, sd, c)] = mc
            optt[(fset, sd, c)] = t_opt
            catches[(fset, sd, c)] = recall_at_flag_budget(s, yte, flag_budget=0.01)

    # Per-(fset, c) mean across seeds
    minc_mean = {
        (fset, c): float(np.mean([minc[(fset, sd, c)] for sd in SEEDS]))
        for fset in FEATURE_SETS
        for c in COST_RATIOS
    }

    # Baseline: full 24full cost at c=100
    full_lift_c100 = minc_mean[("20raw", 100)] - minc_mean[("24full", 100)]

    # ----- TSV: per-cell min_cost -----
    out_cell = RES / "p8_iter196_loo_min_cost.tsv"
    with out_cell.open("w") as f:
        f.write("fset\tseed\tc\tmin_cost\topt_threshold\tfraud_base_rate\n")
        for fset, sd, c in sorted(minc.keys()):
            t_opt = optt[(fset, sd, c)]
            f.write(
                f"{fset}\t{sd}\t{c}\t{minc[(fset, sd, c)]:.6f}\t{t_opt:.4f}\t"
                f"{float((yte == 1).mean()):.4f}\n"
            )

    # ----- TSV: cost curves -----
    out_curve = RES / "p8_iter196_loo_cost_curve.tsv"
    with out_curve.open("w") as f:
        f.write(
            "fset\tc\tseed\tthreshold\tcost_per_tx\tflagged_frac\tfp_per_transaction\tfalse_positive_rate\n"
        )
        for fset, sd in sorted(probas.keys()):
            s = probas[(fset, sd)]
            for c in COST_RATIOS:
                th, costs, fps = cost_curve(s, yte, c)
                for i in range(len(th)):
                    rates = operating_rates(s, yte, th[i])
                    f.write(
                        f"{fset}\t{c}\t{sd}\t{th[i]:.4f}\t{costs[i]:.6f}\t{fps[i]:.4f}\t{rates['fp_per_transaction']:.4f}\t{rates['false_positive_rate']:.4f}\n"
                    )

    # ----- TSV: catches -----
    out_catch = RES / "p8_iter196_loo_catch.tsv"
    with out_catch.open("w") as f:
        f.write("fset\tc\tseed\tcatch_frac\n")
        for fset, sd, c in sorted(catches.keys()):
            f.write(f"{fset}\t{c}\t{sd}\t{catches[(fset, sd, c)]:.4f}\n")

    # ----- TSV: Brier & AUC -----
    out_brier = RES / "p8_iter196_loo_brier.tsv"
    with out_brier.open("w") as f:
        f.write("fset\tseed\tbrier\n")
        for fset, sd in sorted(briers.keys()):
            f.write(f"{fset}\t{sd}\t{briers[(fset, sd)]:.6f}\n")
    out_auc = RES / "p8_iter196_loo_auc.tsv"
    with out_auc.open("w") as f:
        f.write("fset\tseed\tauc\n")
        for fset, sd in sorted(aucs.keys()):
            f.write(f"{fset}\t{sd}\t{aucs[(fset, sd)]:.6f}\n")

    # ----- LOO gap at c=100 vs 20raw (paired-seed bootstrap) -----
    out_gap = RES / "p8_iter196_loo_min_cost_gap.tsv"
    loo_specs = [
        ("24full-20raw", "24full"),
        ("23_noVmean-20raw", "23_noVmean"),
        ("23_noVstd-20raw", "23_noVstd"),
        ("23_noVmax-20raw", "23_noVmax"),
        ("23_noVmin-20raw", "23_noVmin"),
        ("4sensor-20raw", "4sensor"),
    ]
    gaps_c100 = {}  # fset -> 5-vector
    gaps_per_c = {}
    with out_gap.open("w") as f:
        f.write("contrast\tc\tmean_gap\tlo\tmean_20raw\tmean_loo\th_pass\tlift_frac\n")
        for label, fset in loo_specs:
            gaps_per_c[label] = {}
            for c in COST_RATIOS:
                d = np.array([minc[(fset, sd, c)] - minc[("20raw", sd, c)] for sd in SEEDS])
                ci = paired_bootstrap_ci(d, N_BOOT, seed=20260706 + c + stable_seed(0, label))
                m_20 = float(np.mean([minc[("20raw", sd, c)] for sd in SEEDS]))
                m_loo = float(np.mean([minc[(fset, sd, c)] for sd in SEEDS]))
                # lift fraction: how much of the 24full-20raw gap is preserved?
                full_lift_c = m_20 - minc_mean[("24full", c)]
                if abs(full_lift_c) < 1e-9:
                    lift_frac = float("nan")
                else:
                    lift_frac = float((m_20 - m_loo) / full_lift_c)
                h_pass = ci["hi"] < 0.0  # gap negative means LOO is better than 20raw
                gaps_per_c[label][c] = d
                if c == 100:
                    gaps_c100[fset] = d
                f.write(
                    f"{label}\t{c}\t{ci['mean']:+.6f}\t{ci['lo']:+.6f}\t"
                    f"{m_20:.6f}\t{m_loo:.6f}\t{int(h_pass)}\t{lift_frac:+.4f}\n"
                )

    # ----- LOO contributions at c=100 -----
    # per-feature lift retained = (cost_20raw - cost_23_min_f) / (cost_20raw - cost_24full)
    contributions = {}
    for f in AGG4:
        fset_loo = LOO_KEY[f]
        if fset_loo in FEATURE_SETS:
            lift_loo = minc_mean[("20raw", 100)] - minc_mean[(fset_loo, 100)]
            lift_full = minc_mean[("20raw", 100)] - minc_mean[("24full", 100)]
            frac = float(lift_loo / lift_full) if abs(lift_full) > 1e-9 else float("nan")
            contributions[f] = {
                "lift_loo_minus_full": float(lift_full - lift_loo),  # how much lifting is lost
                "lift_retention_fraction": frac,  # how much of full lift remains
                "cost_23_minus_f_at_c100": minc_mean[(fset_loo, 100)],
            }

    # ----- Linear-additivity check -----
    # full_lift - sum(LOO losses) ≈ 0 if features are additive
    sum_loo_losses = sum(v["lift_loo_minus_full"] for v in contributions.values())
    linearity_gap = full_lift_c100 - sum_loo_losses  # positive = super-additive

    # ----- Hypotheses -----
    # H1: dropping V_mean alone preserves <= 50% of full 24full lift at c=100
    h1 = contributions["V_mean"]["lift_retention_fraction"] <= 0.50
    # H2: dropping V_std alone preserves <= 50% of full 24full lift at c=100
    h2 = contributions["V_std"]["lift_retention_fraction"] <= 0.50
    # H3: dropping V_max alone preserves <= 50% of full 24full lift at c=100
    h3 = contributions["V_max"]["lift_retention_fraction"] <= 0.50
    # H4: dropping V_min alone preserves <= 50% of full 24full lift at c=100
    h4 = contributions["V_min"]["lift_retention_fraction"] <= 0.50

    # H5: dropping V_std produces the LARGEST AUC drop among LOO sets
    # AUC gap: per-(LOO set) mean AUC vs 20raw AUC, the most negative
    auc_per_fset = {
        fset: float(np.mean([aucs[(fset, sd)] for sd in SEEDS])) for fset in FEATURE_SETS
    }
    loo_auc_drops = {f: auc_per_fset["24full"] - auc_per_fset[LOO_KEY[f]] for f in AGG4}
    worst = most_damaging_features(
        auc_per_fset["24full"], {f: auc_per_fset[LOO_KEY[f]] for f in AGG4}
    )
    h5 = worst == ["V_std"]

    # H6: linearity gap <= 50% of full lift (features are roughly additive)
    h6 = abs(linearity_gap) <= 0.5 * abs(full_lift_c100)

    summary = {
        "analysis_protocol": "pro-review-02-v2-exploratory",
        "selection": "test-label empirical oracle; no prospective performance claim",
        "uncertainty": "paired training seeds conditional on fixed reused test set; not example-population uncertainty",
        "auc_most_damaging_tied_set": worst,
        "resampling_seeds": {
            label: {str(c): 20260706 + c + stable_seed(0, label) for c in COST_RATIOS}
            for label, _ in loo_specs
        },
        "data": {
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "n_pos_train": int((ytr == 1).sum()),
            "n_pos_test": int((yte == 1).sum()),
            "fraud_base_rate_test": float((yte == 1).mean()),
        },
        "feature_set_cardinalities": {k: len(v) for k, v in FEATURE_SETS.items()},
        "headline_at_c100": {
            "20raw_mean_cost": minc_mean[("20raw", 100)],
            "24full_mean_cost": minc_mean[("24full", 100)],
            "full_lift_c100": full_lift_c100,
            "23_noVmean": minc_mean[("23_noVmean", 100)],
            "23_noVstd": minc_mean[("23_noVstd", 100)],
            "23_noVmax": minc_mean[("23_noVmax", 100)],
            "23_noVmin": minc_mean[("23_noVmin", 100)],
            "4sensor": minc_mean[("4sensor", 100)],
        },
        "contributions_at_c100": contributions,
        "linearity_check_c100": {
            "full_lift": full_lift_c100,
            "sum_loo_losses": sum_loo_losses,
            "linearity_gap": linearity_gap,
        },
        "auc_drops_per_loo": loo_auc_drops,
        "auc_per_fset": auc_per_fset,
        "brier_per_fset": {
            fset: float(np.mean([briers[(fset, sd)] for sd in SEEDS])) for fset in FEATURE_SETS
        },
        "hypotheses": {
            "H1_drop_Vmean_preserves_le_50pct_full_lift": bool(h1),
            "H2_drop_Vstd_preserves_le_50pct_full_lift": bool(h2),
            "H3_drop_Vmax_preserves_le_50pct_full_lift": bool(h3),
            "H4_drop_Vmin_preserves_le_50pct_full_lift": bool(h4),
            "H5_Vstd_drop_largest_AUC_drop": bool(h5),
            "H6_linearity_gap_le_50pct_full_lift": bool(h6),
        },
        "ci_per_loo_per_c": {
            label: {
                str(c): paired_bootstrap_ci(
                    gaps_per_c[label][c], N_BOOT, seed=20260706 + c + stable_seed(0, label)
                )
                for c in COST_RATIOS
            }
            for label, _ in loo_specs
        },
    }
    summary["descriptive_hypothesis_count"] = sum(int(v) for v in summary["hypotheses"].values())
    summary["contribution_ranking"] = sorted(
        contributions.keys(), key=lambda f: contributions[f]["lift_loo_minus_full"], reverse=True
    )
    out_sum = RES / "p8_iter196_loo_summary.json"
    out_sum.write_text(json.dumps(finite_json(summary), indent=2, default=float, allow_nan=False))
    print(f"Wrote {out_sum}", flush=True)
    print(f"H1..H6: {summary['hypotheses']}", flush=True)
    print(f"Headline at c=100: {summary['headline_at_c100']}", flush=True)
    print(f"Contributions: {contributions}", flush=True)
    print(
        f"Linearity gap: {linearity_gap} (sum_loo_losses={sum_loo_losses}, full_lift={full_lift_c100})",
        flush=True,
    )


if __name__ == "__main__":
    main()
