#!/usr/bin/env python3
"""P8 JOB A (iter 52): decision-regret decomposition against the perfect-knowledge oracle.

iter-28 (cost-optimal threshold), iter-40 (C_inv x L 2D frontier), iter-32
(noisy-sensor robustness), iter-48 ($/fraud_caught), and iter-49 (threshold
transfer) all measure a tree's ABSOLUTE cost against the cost-of-mistake
budget. None of them answer the reviewer question "how far is the tree
from the perfect-information oracle, and how much of THAT gap closes when
the LLM-as-sensor adds 4 aggregate features?". Decision regret
= cost(actual) - cost(oracle) is the standard decision-theoretic
quantity for that question.

For each (C_inv, L) cell and each tree:
- oracle_cost = alert-on-positives-only lower bound (C_inv * pos_rate + c_sense)
- actual_cost = cost at the cost-optimal threshold tau*
- regret = actual_cost - oracle_cost   (>= 0 always)

The sensor's value is regret_20raw - regret_24full at each cell:
positive means the LLM aggregates closed part of the oracle gap.
sensor_ceiling = regret_20raw - 0 = the maximum gain achievable if any
sensor could perfectly close the gap.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_decision_regret.tsv          (75 cells)
platform_hybrid/experiments/results/p5p8/p8_decision_regret_boot.tsv     (25 paired-bootstrap CIs)
platform_hybrid/experiments/results/p5p8/p8_decision_regret_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_decision_regret.{png,pdf}

Stdlib + numpy + pandas + xgboost + sklearn + matplotlib. <=300 lines.
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
C_SENSE = 0.0035  # USD/dec for sensor-augmented trees
N_BOOT = 1000
BOOT_SEED = 20260704
TREE_SEED = 42

# Same (C_inv x L) grid as iter-40/52/59 -- reproducible across P8 metrics.
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
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    y_tr = train["Class"].to_numpy(np.int32)
    y_te = test["Class"].to_numpy(np.int32)
    scores = {}
    for name, cols in FEATS.items():
        clf = fit_tree(train[cols].to_numpy(np.float64), y_tr)
        scores[name] = clf.predict_proba(test[cols].to_numpy(np.float64))[:, 1]
    return scores, y_te


def argmin_cost(scores, y, c_inv, L, c_sense):
    """Return (tau_star, cost_star_per_dec, tp_at_star, fp_at_star, fn_at_star).

    Vectorized: sort once, sweep rank cutoffs with a cumulative TP sum.
    cost(k) = (c_inv * k + L * (pos - tp_cum[k]) + c_sense * N) / N
    """
    n = len(y)
    pos = int(y.sum())
    if pos == 0 or n == 0:
        return 1.0, c_sense, 0, 0, 0
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    s_sorted = scores[order]
    # Cumulative TPs along the rank-1..N sweep
    tp_cum = np.cumsum(y_sorted)
    k_arr = np.arange(1, n + 1)
    cost_arr = (c_inv * k_arr + L * (pos - tp_cum) + c_sense * n) / n
    best_k = int(np.argmin(cost_arr))  # 0-based
    best_cost = float(cost_arr[best_k])
    best_tp = int(tp_cum[best_k])
    best_fp = int((best_k + 1) - best_tp)
    best_fn = int(pos - best_tp)
    # tau_star: smallest score at cutoff k (we alert on scores >= this)
    tau = float(s_sorted[best_k])
    return tau, best_cost, best_tp, best_fp, best_fn


def oracle_cost_per_dec(y, c_inv, c_sense):
    """Lower bound: alert exactly on positives.

    Oracle has perfect labels, alerts on all positives and no negatives:
    cost_oracle = (c_inv * pos_count + 0 + c_sense * N) / N = c_inv * pos_rate + c_sense.
    """
    pos_rate = float(y.mean())
    return c_inv * pos_rate + c_sense


def cell_eval(scores, y, c_inv, L):
    """Per cell: cost-oracle lower bound + per-tree regret at tau*."""
    n = len(y)
    pos = int(y.sum())
    cell_oracle_per_dec = oracle_cost_per_dec(y, c_inv, C_SENSE) if False else oracle_cost_per_dec(y, c_inv, 0.0)
    # Oracle is identical for all three trees in expectation (the LLM sensor's
    # c_sense is paid regardless of decision), so we report one oracle value.
    rows = []
    for name in MODELS:
        c_sense = SENSES[name]
        oracle = oracle_cost_per_dec(y, c_inv, c_sense)
        tau, cost_actual, tp, fp, fn = argmin_cost(scores[name], y, c_inv, L, c_sense)
        regret = cost_actual - oracle
        rows.append({
            "model": name,
            "C_inv": c_inv,
            "L": L,
            "rho": L / c_inv,
            "n_test": n,
            "n_pos_test": pos,
            "tau_star": tau,
            "cost_actual_per_dec": cost_actual,
            "cost_oracle_per_dec": oracle,
            "regret_per_dec": regret,
            "regret_pct_of_oracle": (regret / oracle * 100.0) if oracle > 0 else float("nan"),
            "tp_at_star": tp,
            "fp_at_star": fp,
            "fn_at_star": fn,
            "c_sense": c_sense,
        })
    return rows


def grid_eval(scores, y):
    rows = []
    for c_inv in C_INV_GRID:
        for L in L_GRID:
            for r in cell_eval(scores, y, c_inv, L):
                rows.append(r)
    return rows


def boot_cell(scores, y, c_inv, L, rng):
    """One bootstrap iteration: per-model regret at tau*_b."""
    n = len(y)
    idx = rng.integers(0, n, n)
    y_b = y[idx]
    if y_b.sum() == 0:
        return None
    res = {}
    for name in MODELS:
        c_sense = SENSES[name]
        tau, cost_actual, tp, fp, fn = argmin_cost(scores[name][idx], y_b, c_inv, L, c_sense)
        oracle = oracle_cost_per_dec(y_b, c_inv, c_sense)
        res[name] = (cost_actual - oracle, cost_actual, oracle, tau)
    return res


def boot_grid(scores, y):
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for c_inv in C_INV_GRID:
        for L in L_GRID:
            raw_regrets, full_regrets, fs_regrets = [], [], []
            raw_actuals, full_actuals = [], []
            oracle_vals = []
            for _ in range(N_BOOT):
                res = boot_cell(scores, y, c_inv, L, rng)
                if res is None:
                    continue
                raw_regrets.append(res["XGB-20raw"][0])
                full_regrets.append(res["XGB-24full"][0])
                fs_regrets.append(res["XGB-4sensor"][0])
                raw_actuals.append(res["XGB-20raw"][1])
                full_actuals.append(res["XGB-24full"][1])
                oracle_vals.append(res["XGB-20raw"][2])
            if not raw_regrets:
                continue
            raw_regrets = np.array(raw_regrets)
            full_regrets = np.array(full_regrets)
            fs_regrets = np.array(fs_regrets)
            # sensor_ceiling: how much of 20raw's regret 24full actually closes
            sensor_closure = raw_regrets - full_regrets  # >0 means sensor helps
            sensor_closure_fs = raw_regrets - fs_regrets  # >0 means sensor-only helps
            # sensor_full_ceiling: how much COULD any sensor close? (raw regret - 0)
            sensor_ceiling = raw_regrets
            rows.append({
                "C_inv": c_inv,
                "L": L,
                "rho": L / c_inv,
                # 20raw regret
                "regret_20raw_mean": float(raw_regrets.mean()),
                "regret_20raw_ci_low": float(np.percentile(raw_regrets, 2.5)),
                "regret_20raw_ci_high": float(np.percentile(raw_regrets, 97.5)),
                # 24full regret
                "regret_24full_mean": float(full_regrets.mean()),
                "regret_24full_ci_low": float(np.percentile(full_regrets, 2.5)),
                "regret_24full_ci_high": float(np.percentile(full_regrets, 97.5)),
                # 4sensor regret
                "regret_4sensor_mean": float(fs_regrets.mean()),
                "regret_4sensor_ci_low": float(np.percentile(fs_regrets, 2.5)),
                "regret_4sensor_ci_high": float(np.percentile(fs_regrets, 97.5)),
                # sensor_closure = how much 24full closes 20raw's regret
                "sensor_closure_24full_mean": float(sensor_closure.mean()),
                "sensor_closure_24full_ci_low": float(np.percentile(sensor_closure, 2.5)),
                "sensor_closure_24full_ci_high": float(np.percentile(sensor_closure, 97.5)),
                "sensor_closure_24full_excl0": bool(np.percentile(sensor_closure, 2.5) > 0),
                # sensor_full_ceiling = 20raw's regret (max achievable)
                "sensor_full_ceiling_mean": float(sensor_ceiling.mean()),
                "sensor_full_ceiling_ci_low": float(np.percentile(sensor_ceiling, 2.5)),
                "sensor_full_ceiling_ci_high": float(np.percentile(sensor_ceiling, 97.5)),
                "fraction_captured_mean": float(
                    (sensor_closure / np.maximum(sensor_ceiling, 1e-12)).mean()),
                "fraction_captured_ci_low": float(
                    np.percentile(sensor_closure / np.maximum(sensor_ceiling, 1e-12), 2.5)),
                "fraction_captured_ci_high": float(
                    np.percentile(sensor_closure / np.maximum(sensor_ceiling, 1e-12), 97.5)),
                "n_boot_used": len(raw_regrets),
            })
    return rows


def plot_grid(grid_rows, out_png, out_pdf):
    """3-panel heatmap: regret per model + sensor-closure panel."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    titles = [
        "XGB-20raw regret (\\$/dec vs oracle)",
        "XGB-24full regret (\\$/dec vs oracle)",
        "Sensor closure: 20raw - 24full (\\$/dec)",
        "Fraction of ceiling captured (sensor_closure / sensor_full_ceiling)",
    ]
    keys = [
        "regret_20raw",
        "regret_24full",
        None,  # uses closure row
        None,  # uses fraction_captured
    ]
    for ax, title, key in zip(axes.flat, titles, keys):
        if key is None and "closure" in title.lower():
            z = np.full((len(C_INV_GRID), len(L_GRID)), np.nan)
            for r in grid_rows:
                if r["C_inv"] not in C_INV_GRID or r["L"] not in L_GRID:
                    continue
                if r["model"] != "XGB-24full":
                    continue
                i = C_INV_GRID.index(r["C_inv"])
                j = L_GRID.index(r["L"])
                z[i, j] = r["regret_per_dec"]  # placeholder -- overwritten below
            # closure = 20raw.regret - 24full.regret, computed per cell
            raw_z = np.full_like(z, np.nan)
            full_z = np.full_like(z, np.nan)
            for r in grid_rows:
                if r["model"] == "XGB-20raw":
                    i = C_INV_GRID.index(r["C_inv"])
                    j = L_GRID.index(r["L"])
                    raw_z[i, j] = r["regret_per_dec"]
                elif r["model"] == "XGB-24full":
                    i = C_INV_GRID.index(r["C_inv"])
                    j = L_GRID.index(r["L"])
                    full_z[i, j] = r["regret_per_dec"]
            z = raw_z - full_z
        elif key is None and "fraction" in title.lower():
            # Not present in grid_rows (it lives in boot_rows).  Fill with NaN.
            z = np.full((len(C_INV_GRID), len(L_GRID)), np.nan)
        else:
            z = np.full((len(C_INV_GRID), len(L_GRID)), np.nan)
            for r in grid_rows:
                if r["model"] != key:
                    continue
                i = C_INV_GRID.index(r["C_inv"])
                j = L_GRID.index(r["L"])
                z[i, j] = r["regret_per_dec"]
        im = ax.imshow(z, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xticks(range(len(L_GRID)))
        ax.set_xticklabels([f"${L:g}" for L in L_GRID])
        ax.set_yticks(range(len(C_INV_GRID)))
        ax.set_yticklabels([f"${c:g}" for c in C_INV_GRID])
        ax.set_xlabel("L (USD/miss)")
        ax.set_ylabel("C_inv (USD/alert)")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isfinite(z[i, j]):
                    ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if z[i, j] > (np.nanmean(z) if np.any(np.isfinite(z)) else 0) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()
    print(f"Wrote {out_png}")


def main():
    print("Fitting trees on V20 / V20+V_AGG / V_AGG feature sets ...")
    scores, y = load_split()
    n = len(y)
    pos = int(y.sum())
    print(f"test n = {n}, positives = {pos} ({y.mean()*100:.3f}%)")
    print("\nEvaluating decision regret on (C_inv x L) grid ...")
    grid_rows = grid_eval(scores, y)
    pd.DataFrame(grid_rows).to_csv(OUT / "p8_decision_regret.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_decision_regret.tsv'} ({len(grid_rows)} cells)")
    print("\nPaired bootstrap (B=1000, over test rows) ...")
    boot_rows = boot_grid(scores, y)
    pd.DataFrame(boot_rows).to_csv(OUT / "p8_decision_regret_boot.tsv", sep="\t", index=False)
    print(f"Wrote {OUT/'p8_decision_regret_boot.tsv'} ({len(boot_rows)} cells)")
    plot_grid(grid_rows, FIG / "p8_decision_regret.png", FIG / "p8_decision_regret.pdf")
    # Build the canonical-cell summary (C_inv=0.50, L=100 -- same as iter-48/59)
    def at_canonical(rows, model):
        for r in rows:
            if r["model"] == model and r["C_inv"] == 0.50 and r["L"] == 100.0:
                return r
        return None
    canon = {m: at_canonical(grid_rows, m) for m in MODELS}
    canon_boot = {r["rho"]: r for r in boot_rows if abs(r["rho"] - 200) < 1e-9}  # 100/0.5 = 200
    summary = {
        "n_test": int(n),
        "n_pos_test": int(pos),
        "n_grid_cells": len(grid_rows),
        "n_boot_cells": len(boot_rows),
        "boot_seed": BOOT_SEED,
        "n_boot": N_BOOT,
        "c_inv_grid": C_INV_GRID,
        "l_grid": L_GRID,
        "models": MODELS,
        "c_sense": C_SENSE,
        "canonical_cinv_05_l100_regret": {
            m: (canon[m]["regret_per_dec"] if canon[m] else None) for m in MODELS
        },
        "canonical_cinv_05_l100_cost_actual": {
            m: (canon[m]["cost_actual_per_dec"] if canon[m] else None) for m in MODELS
        },
        "canonical_cinv_05_l100_cost_oracle": {
            m: (canon[m]["cost_oracle_per_dec"] if canon[m] else None) for m in MODELS
        },
        "canonical_cinv_05_l100_tau_star": {
            m: (canon[m]["tau_star"] if canon[m] else None) for m in MODELS
        },
        "cells_with_sensor_closure_24full_ci_excl0_positive": sum(
            1 for r in boot_rows if r["sensor_closure_24full_excl0"]
        ),
    }
    with open(OUT / "p8_decision_regret_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT/'p8_decision_regret_summary.json'}")
    print("\n=== Headline (canonical $0.50 / $100) ===")
    for m in MODELS:
        c = canon[m]
        if c is None:
            continue
        print(f"  {m:12s}  oracle=${c['cost_oracle_per_dec']:.4f}/dec   "
              f"actual=${c['cost_actual_per_dec']:.4f}/dec   "
              f"regret=${c['regret_per_dec']:.4f}/dec "
              f"({c['regret_pct_of_oracle']:.1f}% of oracle)   "
              f"tau*={c['tau_star']:.4f}")
    print(f"\nCells where adding the sensor (24full) closes 20raw's regret with CI excluding 0 "
          f"(positive sensor-closure): "
          f"{summary['cells_with_sensor_closure_24full_ci_excl0_positive']}/{len(boot_rows)}")
    # Headline: oracle-vs-actual ratio at canonical cell
    print(f"\nOracle vs actual at canonical cell (C_inv=$0.50, L=$100):")
    print(f"  20raw: oracle=${canon['XGB-20raw']['cost_oracle_per_dec']:.4f}/dec   "
          f"actual=${canon['XGB-20raw']['cost_actual_per_dec']:.4f}/dec   "
          f"ratio={canon['XGB-20raw']['cost_actual_per_dec']/canon['XGB-20raw']['cost_oracle_per_dec']:.2f}x")
    print(f"  24full: oracle=${canon['XGB-24full']['cost_oracle_per_dec']:.4f}/dec   "
          f"actual=${canon['XGB-24full']['cost_actual_per_dec']:.4f}/dec   "
          f"ratio={canon['XGB-24full']['cost_actual_per_dec']/canon['XGB-24full']['cost_oracle_per_dec']:.2f}x")


if __name__ == "__main__":
    main()