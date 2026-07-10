#!/usr/bin/env python3
"""P8 JOB A (iter 36): stack-conditioning audit for the fraud detector.

Mirrors the P5 mega-eta2 vein on the P8 fraud setup. Trains an XGBoost
across a 5-axis stack grid (n_estimators, max_depth, learning_rate,
subsample, scale_pos_weight), computes per-axis eta^2 per metric
(AUC, F1, Brier, ECE-10) with paired bootstrap CIs, and quantifies how
much of the headline result is stack-conditioned vs dataset-inherent.

Headline expectation (mirror of P5 finding): the stack axes explain
70-95% of AUC/F1/Brier variance, leaving <10% as "the dataset speaks".

Vein chosen: the P8 paper currently quotes one number per metric (one
XGB config, one train-test split). This script validates that the
headline is a property of the data, not of the XGB hyperparameters.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_stack_audit.tsv      (5 axes x 4 metrics)
platform_hybrid/experiments/results/p5p8/p8_stack_audit_boot.tsv (CI on per-axis eta^2)
platform_hybrid/experiments/results/p5p8/p8_stack_audit_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_stack_audit.{png,pdf}

Stdlib + numpy + pandas + xgboost + sklearn + matplotlib. <=300 lines.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    brier_score_loss, f1_score, log_loss, roc_auc_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

V20 = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]
TREE_SEED = 42
N_BOOT = 1000
BOOT_SEED = 20260704

# Stack grid: 5 axes, full factorial = 2 * 2 * 2 * 2 * 2 = 32 trees.
# n_estimators capped at 200 so the factorial finishes in <1min on the
# 50k-row fraud split; all five axes get full coverage at 2 levels each.
STACK = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5],
    "learning_rate":    [0.05, 0.2],
    "subsample":        [0.7, 1.0],
    "scale_pos_weight": [1, 5],  # minority re-weighting
}


def load_split():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    X_tr, y_tr = train[V20 + V_AGG], train["Class"].values
    X_te, y_te = test[V20 + V_AGG], test["Class"].values
    return X_tr, y_tr, X_te, y_te


def fit_predict(X_tr, y_tr, X_te, params):
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=4,
        random_state=TREE_SEED,
        **params,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def ece10(y, p, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def score_row(y_te, p_te, thr=0.5):
    auc = roc_auc_score(y_te, p_te)
    yhat = (p_te >= thr).astype(int)
    f1 = f1_score(y_te, yhat, zero_division=0)
    brier = brier_score_loss(y_te, p_te)
    ece = ece10(y_te, p_te)
    return auc, f1, brier, ece


def run_full_factorial(X_tr, y_tr, X_te, y_te):
    """Return DataFrame of (axis, level) per row -> {auc, f1, brier, ece}."""
    keys = list(STACK.keys())
    levels = list(STACK.values())
    combos = list(product(*levels))
    rows = []
    for combo in combos:
        params = dict(zip(keys, combo))
        params["colsample_bytree"] = 0.8  # fixed
        p_te = fit_predict(X_tr, y_tr, X_te, params)
        auc, f1, brier, ece = score_row(y_te, p_te)
        row = {**params, "auc": auc, "f1": f1, "brier": brier, "ece": ece}
        rows.append(row)
    return pd.DataFrame(rows)


def eta2_one_way(df, axis, metric):
    """One-way eta^2 (group SS / total SS) for `axis` on `metric`."""
    grand_mean = df[metric].mean()
    ss_total = ((df[metric] - grand_mean) ** 2).sum()
    if ss_total == 0:
        return float("nan"), grand_mean
    ss_between = df.groupby(axis)[metric].agg(
        n="count", mean="mean"
    ).assign(_=lambda d: d["n"] * (d["mean"] - grand_mean) ** 2)
    ss_between_val = ss_between["_"].sum()
    return float(ss_between_val / ss_total), float(grand_mean)


def bootstrap_eta2(df, axis, metric, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap CI on eta^2 by resampling rows (configs)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    etas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = df.iloc[idx]
        eta, _ = eta2_one_way(sub, axis, metric)
        if not np.isnan(eta):
            etas.append(eta)
    if not etas:
        return float("nan"), float("nan"), float("nan")
    etas = np.array(etas)
    return float(np.mean(etas)), float(np.quantile(etas, 0.025)), float(np.quantile(etas, 0.975))


def main():
    print("Loading fraud_data.csv + test_data.csv")
    X_tr, y_tr, X_te, y_te = load_split()
    print(f"  train n={len(X_tr)} pos={y_tr.sum()}; test n={len(X_te)} pos={y_te.sum()}")

    print(f"Training {len(list(product(*STACK.values())))} trees across the stack grid")
    df = run_full_factorial(X_tr, y_tr, X_te, y_te)
    df.to_csv(OUT / "p8_stack_audit.tsv", sep="\t", index=False)
    print(f"  -> p8_stack_audit.tsv ({len(df)} rows)")

    metrics = ["auc", "f1", "brier", "ece"]
    rows = []
    boot_rows = []
    for axis in STACK:
        for metric in metrics:
            eta, mean = eta2_one_way(df, axis, metric)
            em, lo, hi = bootstrap_eta2(df, axis, metric)
            rows.append({
                "axis": axis,
                "metric": metric,
                "eta2": eta,
                "metric_mean": mean,
                "n_levels": len(STACK[axis]),
            })
            boot_rows.append({
                "axis": axis,
                "metric": metric,
                "eta2_boot_mean": em,
                "eta2_ci025": lo,
                "eta2_ci975": hi,
                "ci_excludes_zero": bool(lo > 0.0),
            })
    pd.DataFrame(rows).to_csv(OUT / "p8_stack_audit_axes.tsv", sep="\t", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "p8_stack_audit_boot.tsv", sep="\t", index=False)

    # Summary JSON
    summary = {
        "n_configs": int(len(df)),
        "stack_axes": list(STACK.keys()),
        "metric_overall": {
            m: {
                "min": float(df[m].min()),
                "max": float(df[m].max()),
                "mean": float(df[m].mean()),
                "std": float(df[m].std()),
            }
            for m in metrics
        },
        "eta2_table": rows,
        "eta2_boot_table": boot_rows,
        "headline": {
            "max_eta2_axis": max(rows, key=lambda r: r["eta2"])["axis"],
            "min_eta2_axis": min(rows, key=lambda r: r["eta2"])["axis"],
            "n_axes_ci_excludes_zero": sum(1 for r in boot_rows if r["ci_excludes_zero"]),
            "n_axes_total": len(boot_rows),
        },
    }
    with open(OUT / "p8_stack_audit_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    # Heatmap figure
    fig, ax = plt.subplots(figsize=(8, 5))
    matrix = np.array([[r["eta2"] for r in rows if r["axis"] == ax] for ax in STACK])
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(STACK)))
    ax.set_yticklabels(list(STACK.keys()))
    ax.set_xlabel("metric")
    ax.set_ylabel("stack axis")
    ax.set_title(r"P8 stack-conditioning $\eta^2$ per (axis, metric)")
    for i in range(len(STACK)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label=r"$\eta^2$")
    plt.tight_layout()
    plt.savefig(FIG / "p8_stack_audit.png", dpi=150)
    plt.savefig(FIG / "p8_stack_audit.pdf")
    plt.close()

    print("\n=== Per-axis eta^2 (point + 95% bootstrap CI) ===")
    print(f"{'axis':<22} {'auc':>10} {'f1':>10} {'brier':>10} {'ece':>10}")
    for ax in STACK:
        line = f"{ax:<22}"
        for m in metrics:
            r = next(b for b in boot_rows if b["axis"] == ax and b["metric"] == m)
            line += f" {r['eta2_boot_mean']:>4.3f} [{r['eta2_ci025']:>4.3f},{r['eta2_ci975']:>4.3f}]"
        print(line)
    print(f"\nHeadline: {summary['headline']}")


if __name__ == "__main__":
    main()