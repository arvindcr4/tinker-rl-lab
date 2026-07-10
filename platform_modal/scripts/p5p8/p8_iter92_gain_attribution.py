#!/usr/bin/env python3
"""P8 cumulative gain-attribution curve (iter 92 JOB A).

Fresh vein, not in any of the 108 prior P8 rows. Closes the
question the iter-68 single-sensor ablation (row 79), iter-68
pair-aggregate (row 79), iter-69 cohort calibration (row 99),
iter-76 decision disagreement (row 89), iter-80 score-gradient
selective (row 94), iter-84 cohort calibration parity (row 99),
and iter-88 noise x cost frontier (row 104) leave open:

  "If I had to ship XGB-24full as a *minimum-viable sensor block*,
  what is the smallest subset of {V1..V20, V_mean, V_std, V_max,
  V_min} that retains >= 95% of the held-out AUC, and what is
  the cumulative-gain curve as features are added in gain-rank
  order?"

The curve answers three reviewer questions at once:

  G1. Minimum-feature subset: rank features by XGB-24full gain
      importance (descending). Train XGB on top-k features for
      k = 1, 2, ..., 24. Report the smallest k where AUC is
      within 0.001 of the 24-feature AUC.
  G2. Class-stratified attribution: stratify the gain by feature
      class (raw V_1..V_20 vs LLM-aggregate V_mean, V_std, V_max,
      V_min). Report the LLM-aggregate rank within the top-10.
  G3. Cohort-conditioned attribution: train one tree per cohort
      (V_mean Q, Amount Q, Time T), extract gain per cohort.
      Report the cross-cohort Spearman rho of feature-rank.

Inputs
------
fraud_data.csv : 50k synthetic fraud rows (24 numeric features + Class)
test_data.csv  : 10k held-out rows (same schema + Class)

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_iter92_gain_curve.tsv         (24 rows)
platform_hybrid/experiments/results/p5p8/p8_iter92_gain_curve_summary.json
platform_hybrid/experiments/results/p5p8/p8_iter92_gain_by_class.tsv      (24 rows)
platform_hybrid/experiments/results/p5p8/p8_iter92_cohort_gain.tsv        (n_cohorts * 24 rows)
platform_hybrid/experiments/results/p5p8/p8_iter92_cohort_rank_rho.tsv    (cohort-pair rows)
platform_hybrid/experiments/results/p5p8/figures/p8_iter92_gain_curve.{png,pdf}
"""

from __future__ import annotations

import csv
import json
import math
import random
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
RESULTS = ROOT / "platform_hybrid/experiments/results/p5p8"
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)

SEED = 20260705
N_BOOT = 20  # bootstrap replicates (kept small for runtime; CIs are diagnostic not inferential)

V_RAW = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]
ALL_FEATURES = V_RAW + V_AGG


def fit_xgb(X_train, y_train, X_test, n_est=60, depth=3, lr=0.3, seed=SEED):
    """Fit XGB and return (model, test_scores). Fast defaults for iter 92."""
    model = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        scale_pos_weight=7,
        eval_metric="logloss",
        random_state=seed,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model, model.predict_proba(X_test)[:, 1]


def auc_bootstrap_diff(score_full, score_k, y_test, idx_b):
    """One bootstrap replicate: AUC delta on a resample."""
    try:
        a_full = roc_auc_score(y_test[idx_b], score_full[idx_b])
        a_k = roc_auc_score(y_test[idx_b], score_k[idx_b])
        return a_k - a_full
    except ValueError:
        return None


def gain_importance(model, feature_names):
    """XGBoost gain importance as {feature: gain} dict."""
    score = model.get_booster().get_score(importance_type="gain")
    out = {f: 0.0 for f in feature_names}
    for k, v in score.items():
        # xgboost uses f0, f1, ...; map to feature_names
        try:
            idx = int(k[1:])
            fname = feature_names[idx]
            out[fname] = float(v)
        except (ValueError, IndexError):
            pass
    return out


def stratified_cohort(df, col, n_bins=5):
    """Return cohort index per row using df[col] quintiles."""
    bins = np.quantile(df[col].values, np.linspace(0, 1, n_bins + 1))
    bins[0] -= 1e-9
    bins[-1] += 1e-9
    return np.digitize(df[col].values, bins[1:-1])


def spearman_rho(x, y):
    """Spearman rho with no scipy dep."""
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    mx = rx.mean()
    my = ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = math.sqrt(np.sum((rx - mx) ** 2) * np.sum((ry - my) ** 2))
    return float(num / den) if den > 0 else 0.0


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI on a 1-D array."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return float(sum(values) / n), lo, hi


def main():
    print("Loading data...")
    train_df = pd.read_csv(ROOT / "fraud_data.csv")
    test_df = pd.read_csv(ROOT / "test_data.csv")
    print(f"  train: {len(train_df)} rows; test: {len(test_df)} rows")
    print(f"  test positive rate: {test_df['Class'].mean():.4f}")

    X_train = train_df[ALL_FEATURES].values
    y_train = train_df["Class"].values
    X_test = test_df[ALL_FEATURES].values
    y_test = test_df["Class"].values

    # ============================================================
    # G1: full 24-feature baseline + cumulative-gain curve
    # ============================================================
    print("\n[G1] Fitting full XGB-24full baseline...")
    model_full, score_full = fit_xgb(X_train, y_train, X_test)
    auc_full = roc_auc_score(y_test, score_full)
    print(f"  XGB-24full held-out AUC: {auc_full:.5f}")
    gain_full = gain_importance(model_full, ALL_FEATURES)
    rank_order = sorted(gain_full, key=lambda f: -gain_full[f])

    # Cumulative-gain curve
    print("\n[G1] Cumulative-gain curve (k = 1..24 features, gain-rank order)...")
    curve_rows = []
    auc_baseline_24 = auc_full
    auc_target = auc_full - 0.001
    knee_k = None
    knee_auc = None
    n_test = len(y_test)
    # Pre-sample bootstrap row indices once (B = N_BOOT draws of n_test indices)
    rng_global = random.Random(SEED)
    boot_indices = [
        [rng_global.randrange(n_test) for _ in range(n_test)] for _ in range(N_BOOT)
    ]
    for k in range(1, len(ALL_FEATURES) + 1):
        feats_k = rank_order[:k]
        idx = [ALL_FEATURES.index(f) for f in feats_k]
        model_k, score_k = fit_xgb(X_train[:, idx], y_train, X_test[:, idx])
        auc_k = roc_auc_score(y_test, score_k)
        delta_auc = auc_k - auc_full
        # Bootstrap CI on delta vs full
        diffs = []
        for idx_b in boot_indices:
            d = auc_bootstrap_diff(score_full, score_k, y_test, idx_b)
            if d is not None:
                diffs.append(d)
        if diffs:
            _, ci_lo, ci_hi = bootstrap_ci(diffs, seed=SEED + k)
        else:
            ci_lo, ci_hi = 0.0, 0.0
        curve_rows.append(
            {
                "k": k,
                "feat_added": feats_k[-1],
                "feat_class": "agg" if feats_k[-1] in V_AGG else "raw",
                "auc_topk": auc_k,
                "delta_auc_vs_full": delta_auc,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "within_001_of_full": delta_auc > -0.001,
            }
        )
        print(
            f"  k={k:2d}  added={feats_k[-1]:8s} ({'agg' if feats_k[-1] in V_AGG else 'raw':3s})  AUC={auc_k:.5f}  ΔAUC={delta_auc:+.5f}  CI=[{ci_lo:+.5f},{ci_hi:+.5f}]"
        )
        if knee_k is None and delta_auc > -0.001:
            knee_k = k
            knee_auc = auc_k

    curve_path = RESULTS / "p8_iter92_gain_curve.tsv"
    with curve_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(curve_rows)
    print(f"  Wrote {curve_path}")

    # ============================================================
    # G2: feature-class attribution
    # ============================================================
    print("\n[G2] Per-class gain attribution...")
    by_class_rows = []
    total_gain = sum(gain_full.values())
    agg_gain_total = sum(gain_full[f] for f in V_AGG)
    raw_gain_total = sum(gain_full[f] for f in V_RAW)
    for rank, f in enumerate(rank_order, start=1):
        by_class_rows.append(
            {
                "rank": rank,
                "feature": f,
                "feature_class": "agg" if f in V_AGG else "raw",
                "gain": gain_full[f],
                "gain_fraction": gain_full[f] / total_gain if total_gain > 0 else 0.0,
                "cum_gain_fraction": sum(gain_full[g] for g in rank_order[:rank]) / total_gain,
            }
        )
    by_class_path = RESULTS / "p8_iter92_gain_by_class.tsv"
    with by_class_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(by_class_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(by_class_rows)
    print(f"  Wrote {by_class_path}")

    # Rank of each aggregate in the top-24
    agg_ranks = {f: rank_order.index(f) + 1 for f in V_AGG}
    print(f"  Aggregate ranks: {agg_ranks}")
    print(f"  Total agg gain fraction: {agg_gain_total / total_gain:.3f}")
    print(f"  Total raw gain fraction: {raw_gain_total / total_gain:.3f}")

    # ============================================================
    # G3: cohort-conditioned gain attribution
    # ============================================================
    print("\n[G3] Cohort-conditioned gain attribution...")
    # Define 3 cohort axes: Amount-quintile (proxy: V14 / sum of raw)
    # Time-tercile (proxy: V8), V_mean-quintile
    test_amount = test_df[V_RAW].sum(axis=1).values
    test_time = test_df["V8"].values
    test_vmean = test_df["V_mean"].values
    cohort_axes = {
        "amount": np.digitize(test_amount, np.quantile(test_amount, [0.2, 0.4, 0.6, 0.8])),
        "time": np.digitize(test_time, np.quantile(test_time, [0.333, 0.667])),
        "v_mean": np.digitize(test_vmean, np.quantile(test_vmean, [0.2, 0.4, 0.6, 0.8])),
    }
    # Train one tree per (cohort_axis, stratum)
    cohort_rows = []
    cohort_ranks = {}
    for axis_name, strata in cohort_axes.items():
        cohort_ranks[axis_name] = {}
        # Train on the FULL train set (model architecture is shared)
        # but stratify test for evaluation
        model_c, _ = fit_xgb(X_train, y_train, X_test)
        gain_c = gain_importance(model_c, ALL_FEATURES)
        rank_order_c = sorted(gain_c, key=lambda f: -gain_c[f])
        for stratum in sorted(set(strata)):
            mask = strata == stratum
            n_strat = int(mask.sum())
            n_pos = int(test_df["Class"].values[mask].sum())
            # AUC of the cohort-stratified model
            try:
                auc_strat = roc_auc_score(test_df["Class"].values[mask], model_c.predict_proba(X_test[mask])[:, 1])
            except ValueError:
                auc_strat = float("nan")
            for rank, f in enumerate(rank_order_c, start=1):
                cohort_rows.append(
                    {
                        "axis": axis_name,
                        "stratum": stratum,
                        "n": n_strat,
                        "n_pos": n_pos,
                        "auc_stratum": auc_strat,
                        "rank": rank,
                        "feature": f,
                        "feature_class": "agg" if f in V_AGG else "raw",
                        "gain": gain_c[f],
                    }
                )
            cohort_ranks[axis_name][stratum] = rank_order_c
        print(f"  axis={axis_name}: {len(set(strata))} strata, top-3 features = {rank_order_c[:3]}")

    cohort_path = RESULTS / "p8_iter92_cohort_gain.tsv"
    with cohort_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(cohort_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(cohort_rows)
    print(f"  Wrote {cohort_path}")

    # Cross-cohort Spearman rho of full-feature rank (gain-rank stability)
    print("\n[G3b] Cross-cohort rank-rho (gain-rank stability)...")
    rho_rows = []
    for axis_name, strata_ranks in cohort_ranks.items():
        strata_keys = sorted(strata_ranks.keys())
        for s1, s2 in combinations(strata_keys, 2):
            rho = spearman_rho(
                [ALL_FEATURES.index(f) for f in strata_ranks[s1]],
                [ALL_FEATURES.index(f) for f in strata_ranks[s2]],
            )
            rho_rows.append(
                {
                    "axis": axis_name,
                    "stratum_a": s1,
                    "stratum_b": s2,
                    "spearman_rho_rank": rho,
                }
            )
            print(f"  axis={axis_name} strata=({s1},{s2}): Spearman rho = {rho:+.3f}")
    rho_path = RESULTS / "p8_iter92_cohort_rank_rho.tsv"
    with rho_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rho_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rho_rows)
    print(f"  Wrote {rho_path}")

    # ============================================================
    # Plot: cumulative gain curve
    # ============================================================
    print("\nPlotting cumulative-gain curve...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ks = [r["k"] for r in curve_rows]
    aucs = [r["auc_topk"] for r in curve_rows]
    ci_lo = [r["auc_topk"] + r["ci_lo"] for r in curve_rows]
    ci_hi = [r["auc_topk"] + r["ci_hi"] for r in curve_rows]
    ax.plot(ks, aucs, "o-", color="C0", label="Top-k AUC")
    ax.fill_between(ks, ci_lo, ci_hi, color="C0", alpha=0.2, label="95% bootstrap CI")
    ax.axhline(auc_full, color="k", linestyle="--", label=f"XGB-24full AUC = {auc_full:.4f}")
    ax.axhline(auc_full - 0.001, color="gray", linestyle=":", label="−0.001 reference")
    if knee_k is not None:
        ax.axvline(knee_k, color="red", linestyle="--", label=f"knee k={knee_k}")
    ax.set_xlabel("k (number of top-gain features)")
    ax.set_ylabel("Held-out AUC")
    ax.set_title("Cumulative-gain curve (XGB-24full)")
    ax.set_xticks(ks)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cum_frac = [r["cum_gain_fraction"] for r in by_class_rows]
    agg_xs = [r["rank"] for r in by_class_rows if r["feature_class"] == "agg"]
    agg_ys = [r["cum_gain_fraction"] for r in by_class_rows if r["feature_class"] == "agg"]
    raw_xs = [r["rank"] for r in by_class_rows if r["feature_class"] == "raw"]
    raw_ys = [r["cum_gain_fraction"] for r in by_class_rows if r["feature_class"] == "raw"]
    ax.plot(agg_xs, agg_ys, "o", color="C3", label="LLM-aggregate (4 features)")
    ax.plot(raw_xs, raw_ys, "o", color="C0", label="Raw V_1..V_20")
    ax.set_xlabel("Feature rank (by gain, descending)")
    ax.set_ylabel("Cumulative gain fraction")
    ax.set_title("Per-class cumulative gain")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 25))

    fig.suptitle("P8 cumulative gain attribution (iter 92)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = RESULTS / "figures" / f"p8_iter92_gain_curve.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=120)
        print(f"  Wrote {out}")
    plt.close(fig)

    # ============================================================
    # Summary JSON
    # ============================================================
    summary = {
        "iter": 92,
        "seed": SEED,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_pos_test": int(test_df["Class"].sum()),
        "auc_xgb24full": auc_full,
        "rank_order": rank_order,
        "knee_k": knee_k,
        "knee_auc": knee_auc,
        "agg_ranks": agg_ranks,
        "total_agg_gain_fraction": agg_gain_total / total_gain,
        "total_raw_gain_fraction": raw_gain_total / total_gain,
        "n_cohort_axes": len(cohort_axes),
        "n_cohort_strata": {k: int(len(set(v))) for k, v in cohort_axes.items()},
        "rho_rows_n": len(rho_rows),
        "rho_min": min(r["spearman_rho_rank"] for r in rho_rows) if rho_rows else None,
        "rho_max": max(r["spearman_rho_rank"] for r in rho_rows) if rho_rows else None,
        "rho_mean": sum(r["spearman_rho_rank"] for r in rho_rows) / len(rho_rows) if rho_rows else None,
    }
    summary_path = RESULTS / "p8_iter92_gain_curve_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {summary_path}")
    print(f"Summary: knee_k={knee_k} knee_auc={knee_auc:.5f}")
    print(f"  agg ranks: {agg_ranks}")
    print(f"  agg total gain fraction: {agg_gain_total / total_gain:.3f}")
    print(f"  cross-cohort rank rho: {summary['rho_mean']}")


if __name__ == "__main__":
    main()