#!/usr/bin/env python3
"""P8 JOB A (iter 48): cost-per-fraud-caught ($/fraud_caught) accounting on
the iter-40 (C_inv x L) grid.

iter-28 measured cost/decision ($/dec); iter-40 swept (C_inv x L) on $/dec;
iter-32 measured noisy-sensor $/dec; iter-49 measured paired bootstrap on
the same axis. All four used $/dec -- the cost per *stream event* --
which is dominated by alert count. Fraud-ops reports a complementary
metric: $/fraud_caught = total_cost / true_positives at the cost-optimal
threshold. A high-precision tree is "cheap per stream but expensive per
catch"; a high-recall tree is "expensive per stream but cheap per catch".
The ratio is a clean precision proxy that no prior P8 item separated.

Outputs
-------
experiments/results/p5p8/p8_cost_per_caught.tsv         (per cell)
experiments/results/p5p8/p8_cost_per_caught_boot.tsv    (per cell CI)
experiments/results/p5p8/p8_cost_per_caught_summary.json
experiments/results/p5p8/figures/p8_cost_per_caught.{png,pdf}

Stdlib + numpy + pandas + xgboost + matplotlib. <=300 lines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
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
C_SENSE = 0.0035  # USD/dec for sensor feature set (24full, 4sensor)
N_BOOT = 1000
BOOT_SEED = 20260704
TREE_SEED = 42

# Same grid as iter-40 -- reproducible across P8 metrics.
C_INV_GRID = [0.10, 0.50, 1.00, 2.50, 5.00]
L_GRID = [5.0, 25.0, 100.0, 250.0, 1000.0]
MODELS = ["XGB-20raw", "XGB-24full", "XGB-4sensor"]
SENSES = {"XGB-20raw": 0.0, "XGB-24full": C_SENSE, "XGB-4sensor": C_SENSE}
FEATS = {
    "XGB-20raw": V20,
    "XGB-24full": V20 + V_AGG,
    "XGB-4sensor": V_AGG,
}


def fit_tree(X_tr, y_tr):
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", random_state=TREE_SEED, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    return clf


def load_split():
    """Return dict of model -> test scores and y_te."""
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    y_tr = train["Class"].to_numpy(np.int32)
    y_te = test["Class"].to_numpy(np.int32)
    scores = {}
    for name, cols in FEATS.items():
        clf = fit_tree(train[cols].to_numpy(np.float64), y_tr)
        scores[name] = clf.predict_proba(test[cols].to_numpy(np.float64))[:, 1]
    return scores, y_te


def argmin_cost(p, y, C_inv, L, c_sense):
    """Find the rank cutoff tau minimising E[cost/dec] and return (cost_per_dec, cost_per_caught, tau, tp)."""
    if L <= 0 or C_inv <= 0:
        return float("nan"), float("nan"), 0.5, 0
    # Sort by predicted score descending -- tau is a rank cutoff.
    pos_mask = y == 1
    n = len(y)
    n_pos = int(pos_mask.sum())
    # Pre-sort.
    order = np.argsort(-p)
    sorted_p = p[order]
    sorted_y = y[order]
    cum_tp = np.cumsum(sorted_y == 1)
    cum_fp = np.cumsum(sorted_y == 0)
    # tau_k = number of alerts at cutoff k = #predicted positives = k
    k = np.arange(1, n + 1)
    tp = cum_tp
    fp = cum_fp
    fn = n_pos - tp
    cost = c_sense + (C_inv * (tp + fp) + L * fn) / n
    best = int(np.argmin(cost))
    tau = sorted_p[best]
    best_tp = int(cum_tp[best])
    cost_per_dec = float(cost[best])
    if best_tp == 0:
        # Cost-per-caught is undefined when no fraud caught: return +inf for ranking.
        cost_per_caught = float("inf")
    else:
        # total_cost_USD = (C_inv * (tp+fp) + L * fn) + c_sense * N
        total = C_inv * (tp[best] + fp[best]) + L * fn[best] + c_sense * n
        cost_per_caught = float(total / best_tp)
    return cost_per_dec, cost_per_caught, float(tau), best_tp


def grid_eval(scores, y):
    """Per-cell evaluation, returns list of dict rows."""
    rows = []
    for c_inv in C_INV_GRID:
        for L in L_GRID:
            for name in MODELS:
                cpd, cpc, tau, tp = argmin_cost(scores[name], y, c_inv, L, SENSES[name])
                rows.append({
                    "model": name,
                    "C_inv": c_inv,
                    "L": L,
                    "rho": L / c_inv,
                    "tau_star": tau,
                    "tp_at_tau_star": tp,
                    "cost_per_dec_usd": cpd,
                    "cost_per_caught_usd": cpc if np.isfinite(cpc) else None,
                })
    return rows


def boot_grid(scores, y, n_boot=N_BOOT):
    """Paired bootstrap CIs for 20full-20raw and 4sensor-20raw cost-per-caught."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    rows = []
    for c_inv in C_INV_GRID:
        for L in L_GRID:
            deltas_vs_raw = []
            deltas_4s_vs_raw = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                y_b = y[idx]
                # filter rows where at least 1 positive exists
                if y_b.sum() == 0:
                    continue
                _, cpc_raw, _, tp_raw = argmin_cost(scores["XGB-20raw"][idx], y_b, c_inv, L, SENSES["XGB-20raw"])
                _, cpc_full, _, tp_full = argmin_cost(scores["XGB-24full"][idx], y_b, c_inv, L, SENSES["XGB-24full"])
                _, cpc_4s, _, tp_4s = argmin_cost(scores["XGB-4sensor"][idx], y_b, c_inv, L, SENSES["XGB-4sensor"])
                # If any tree caught 0 positives on the bootstrap, treat that cell as missing
                if tp_raw == 0 or tp_full == 0 or tp_4s == 0:
                    continue
                if not (np.isfinite(cpc_raw) and np.isfinite(cpc_full) and np.isfinite(cpc_4s)):
                    continue
                deltas_vs_raw.append(cpc_full - cpc_raw)
                deltas_4s_vs_raw.append(cpc_4s - cpc_raw)
            if deltas_vs_raw:
                arr = np.array(deltas_vs_raw)
                rows.append({
                    "C_inv": c_inv,
                    "L": L,
                    "rho": L / c_inv,
                    "delta_24full_minus_20raw_mean": float(arr.mean()),
                    "delta_24full_minus_20raw_ci_low": float(np.percentile(arr, 2.5)),
                    "delta_24full_minus_20raw_ci_high": float(np.percentile(arr, 97.5)),
                    "delta_24full_minus_20raw_excl0": bool(arr.mean() > 0 and
                                                            np.percentile(arr, 2.5) > 0),
                    "delta_4sensor_minus_20raw_mean": float(np.mean(deltas_4s_vs_raw)),
                    "delta_4sensor_minus_20raw_ci_low": float(np.percentile(deltas_4s_vs_raw, 2.5)),
                    "delta_4sensor_minus_20raw_ci_high": float(np.percentile(deltas_4s_vs_raw, 97.5)),
                    "delta_4sensor_minus_20raw_excl0": bool(np.mean(deltas_4s_vs_raw) > 0 and
                                                            np.percentile(deltas_4s_vs_raw, 2.5) > 0),
                    "n_boot_used": len(deltas_vs_raw),
                })
    return rows


def plot_grid(grid_rows, out_png, out_pdf):
    """3-panel heatmap: cost_per_caught for each model."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, name in zip(axes, MODELS):
        z = np.full((len(C_INV_GRID), len(L_GRID)), np.nan)
        for r in grid_rows:
            if r["model"] != name or r["cost_per_caught_usd"] is None:
                continue
            i = C_INV_GRID.index(r["C_inv"])
            j = L_GRID.index(r["L"])
            z[i, j] = r["cost_per_caught_usd"]
        im = ax.imshow(z, aspect="auto", origin="lower")
        ax.set_title(f"{name}  $/fraud_caught")
        ax.set_xticks(range(len(L_GRID)))
        ax.set_xticklabels([f"${L:g}" for L in L_GRID])
        ax.set_yticks(range(len(C_INV_GRID)))
        ax.set_yticklabels([f"${c:g}" for c in C_INV_GRID])
        ax.set_xlabel("L (USD/miss)")
        ax.set_ylabel("C_inv (USD/alert)")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isfinite(z[i, j]):
                    ax.text(j, i, f"{z[i, j]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if z[i, j] > z[~np.isnan(z)].mean() else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Wrote {out_png}")


def main():
    print("Fitting trees on V20 / V20+V_AGG / V_AGG feature sets ...")
    scores, y = load_split()
    print(f"test n = {len(y)}, positives = {int(y.sum())} ({y.mean()*100:.3f}%)")
    print("\nEvaluating cost-per-caught on (C_inv x L) grid ...")
    grid_rows = grid_eval(scores, y)
    pd.DataFrame(grid_rows).to_csv(OUT / "p8_cost_per_caught.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_cost_per_caught.tsv'} ({len(grid_rows)} cells)")
    print("\nPaired bootstrap (B=1000, over test rows) ...")
    boot_rows = boot_grid(scores, y)
    pd.DataFrame(boot_rows).to_csv(OUT / "p8_cost_per_caught_boot.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_cost_per_caught_boot.tsv'} ({len(boot_rows)} cells)")
    plot_grid(grid_rows, FIG / "p8_cost_per_caught.png", FIG / "p8_cost_per_caught.pdf")
    summary = {
        "n_test": int(len(y)),
        "n_pos_test": int(y.sum()),
        "n_grid_cells": len(grid_rows),
        "n_boot_cells": len(boot_rows),
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "c_inv_grid": C_INV_GRID,
        "l_grid": L_GRID,
        "models": MODELS,
        "c_sense": C_SENSE,
        # Cost/Caught at the canonical c_inv=0.50, L=100: which tree is sharpest.
        "canonical_cinv_05_l100": {
            name: next((r["cost_per_caught_usd"] for r in grid_rows
                         if r["model"] == name and r["C_inv"] == 0.50 and r["L"] == 100.0), None)
            for name in MODELS
        },
        "cells_with_24full_minus_20raw_ci_excl0_positive": sum(
            1 for r in boot_rows if r["delta_24full_minus_20raw_excl0"]
        ),
        "cells_with_4sensor_minus_20raw_ci_excl0_positive": sum(
            1 for r in boot_rows if r["delta_4sensor_minus_20raw_excl0"]
        ),
    }
    with open(OUT / "p8_cost_per_caught_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT/'p8_cost_per_caught_summary.json'}")
    print("\nHeadline (canonical $0.50 / $100):")
    for name, v in summary["canonical_cinv_05_l100"].items():
        print(f"  {name:12s}  $/fraud_caught = ${v:.2f}")
    print(f"\nCells where adding the sensor augments the tree (24full vs 20raw cost-per-caught) have CI excluding 0 (positive direction): "
          f"{summary['cells_with_24full_minus_20raw_ci_excl0_positive']}/{len(boot_rows)}")
    print(f"Cells where the sensor-only surrogate (4sensor) loses CI-excluding-0 to raw: "
          f"{summary['cells_with_4sensor_minus_20raw_ci_excl0_positive']}/{len(boot_rows)}")


if __name__ == "__main__":
    main()
