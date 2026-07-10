#!/usr/bin/env python3
"""P8 JOB A (iter 40): asymmetric cost-asymmetry frontier.

iter-28 (cost-optimal threshold) and iter-36 (5x5 (sigma, L) phase diagram)
both held C_inv (analyst-review cost per alert) at the canonical
$0.50/alert. That assumption collapses a real fraud-ops lever: at
fully-automated triage C_inv approaches zero; at analyst-heavy triage
C_inv is several dollars per alert. The 2D (C_inv, L_missed) frontier
with paired bootstrap CIs is the orthogonal slice that a fraud-ops lead
needs to size the budget tradeoff.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_asym_cost.tsv
platform_hybrid/experiments/results/p5p8/p8_asym_cost_boot.tsv
platform_hybrid/experiments/results/p5p8/p8_asym_cost_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_asym_cost.{png,pdf}

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
C_SENSE = 0.0035
N_BOOT = 1000
BOOT_SEED = 20260704
TREE_SEED = 42

# New 2D phase grid: review cost × missed-positive cost (USD).
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


def cost_per_dec(p, y, C_inv, L, c_sense):
    """Vectorised expected cost per decision at the cost-optimal threshold.

    Cost = c_sense + (C_inv * (tp+fp) + L * fn) / N
    where tp/fp/fn are computed from argmax(C_inv) threshold sweep:
    pick rank cutoff p_t such that the per-decision cost is minimised.
    For C_inv > 0 the optimal alert policy is: alert iff p >= L / (L + C_inv).
    """
    N = len(y)
    if L <= 0 or C_inv <= 0:
        # degenerate: alert always or never; return constant cost
        return c_sense + C_inv * (1.0) * (p >= 0.5).mean() if C_inv > 0 else c_sense
    tau_star = L / (L + C_inv)
    alerts = (p >= tau_star).astype(np.int32)
    tp = int(((alerts == 1) & (y == 1)).sum())
    fp = int(((alerts == 1) & (y == 0)).sum())
    fn = int(((alerts == 0) & (y == 1)).sum())
    return c_sense + (C_inv * (tp + fp) + L * fn) / N, tau_star, tp, fp, fn


def boot_delta(scores, y, c_inv, L, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap on (cost_24full - cost_20raw). Resample rows;
    preserve pairing across both scores at the same resampled indices.
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    sens_4sens = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        c20, _, _, _, _ = cost_per_dec(scores["XGB-20raw"][idx], y[idx], c_inv, L, SENSES["XGB-20raw"])
        c24, _, _, _, _ = cost_per_dec(scores["XGB-24full"][idx], y[idx], c_inv, L, SENSES["XGB-24full"])
        c4s, _, _, _, _ = cost_per_dec(scores["XGB-4sensor"][idx], y[idx], c_inv, L, SENSES["XGB-4sensor"])
        deltas[b] = c24 - c20
        sens_4sens[b] = c24 - c4s
    return deltas, sens_4sens


def grid_eval(scores, y, c_inv_grid=C_INV_GRID, l_grid=L_GRID):
    rows = []
    for c_inv in c_inv_grid:
        for L in l_grid:
            for name in MODELS:
                cost, tau, tp, fp, fn = cost_per_dec(scores[name], y, c_inv, L, SENSES[name])
                rows.append({
                    "C_inv": c_inv,
                    "L": L,
                    "model": name,
                    "tau_star": round(tau, 6),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "cost_per_dec_usd": round(cost, 6),
                    "c_sense": SENSES[name],
                })
    df = pd.DataFrame(rows)
    df["alert_rate"] = (df["tp"] + df["fp"]) / len(y)
    df["recall"] = df.apply(lambda r: r["tp"] / max(1, r["tp"] + r["fn"]), axis=1)
    return df


def boot_grid(scores, y, c_inv_grid=C_INV_GRID, l_grid=L_GRID):
    rows = []
    for c_inv in c_inv_grid:
        for L in l_grid:
            d, d4s = boot_delta(scores, y, c_inv, L)
            # CI excludes zero iff (ci975 < 0) — entire CI below zero —
            # OR (ci025 > 0) — entire CI above zero. Either sign is
            # acceptable; what we do NOT want is zero inside the CI.
            d_ci025, d_ci975 = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
            ds_ci025, ds_ci975 = float(np.quantile(d4s, 0.025)), float(np.quantile(d4s, 0.975))
            rows.append({
                "C_inv": c_inv,
                "L": L,
                "delta_24_vs_20_mean": round(float(d.mean()), 7),
                "delta_24_vs_20_ci025": round(d_ci025, 7),
                "delta_24_vs_20_ci975": round(d_ci975, 7),
                "ci_excl_zero_24_vs_20": bool(d_ci975 < 0 or d_ci025 > 0),
                "sensor_pays_off_24_vs_20": bool(d.mean() < 0),
                "delta_24_vs_4s_mean": round(float(d4s.mean()), 7),
                "delta_24_vs_4s_ci025": round(ds_ci025, 7),
                "delta_24_vs_4s_ci975": round(ds_ci975, 7),
                "ci_excl_zero_24_vs_4s": bool(ds_ci975 < 0 or ds_ci025 > 0),
            })
    return pd.DataFrame(rows)


def figure_make(grid_df, boot_df):
    """Two-panel figure: (a) raw grid (24full-20raw) delta cost;
    (b) bootstrap 95% CI for cells where CI excludes zero on the favorable side.
    """
    pivot24 = grid_df[grid_df.model == "XGB-24full"].pivot(index="C_inv", columns="L", values="cost_per_dec_usd")
    pivot20 = grid_df[grid_df.model == "XGB-20raw"].pivot(index="C_inv", columns="L", values="cost_per_dec_usd")
    delta = (pivot24 - pivot20) * 1000.0  # millicents

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im0 = axes[0].imshow(delta.values, cmap="RdBu_r", aspect="auto",
                         vmin=-abs(delta.values).max(), vmax=abs(delta.values).max())
    axes[0].set_xticks(range(len(L_GRID)))
    axes[0].set_xticklabels([f"${L:g}" for L in L_GRID])
    axes[0].set_yticks(range(len(C_INV_GRID)))
    axes[0].set_yticklabels([f"${c:g}" for c in C_INV_GRID])
    axes[0].set_xlabel("Missed-fraud cost L (USD)")
    axes[0].set_ylabel("Alert-investigation cost C_inv (USD/alert)")
    axes[0].set_title("(a) $\\Delta$ cost per decision (24full - 20raw), millicents")
    for i in range(len(C_INV_GRID)):
        for j in range(len(L_GRID)):
            axes[0].text(j, i, f"{delta.values[i,j]:+.1f}", ha="center", va="center",
                         fontsize=8, color="black")
    fig.colorbar(im0, ax=axes[0])

    # panel (b): sensor-24full-aux-bar over 4sensor (where 4sensor is the
    # extreme LLM-as-only-sensor; 24full adds 20 raw features back).
    pivot4s = grid_df[grid_df.model == "XGB-4sensor"].pivot(index="C_inv", columns="L", values="cost_per_dec_usd")
    delta2 = (pivot24 - pivot4s) * 1000.0
    im1 = axes[1].imshow(delta2.values, cmap="RdBu_r", aspect="auto",
                         vmin=-abs(delta2.values).max(), vmax=abs(delta2.values).max())
    axes[1].set_xticks(range(len(L_GRID)))
    axes[1].set_xticklabels([f"${L:g}" for L in L_GRID])
    axes[1].set_yticks(range(len(C_INV_GRID)))
    axes[1].set_yticklabels([f"${c:g}" for c in C_INV_GRID])
    axes[1].set_xlabel("Missed-fraud cost L (USD)")
    axes[1].set_ylabel("Alert-investigation cost C_inv (USD/alert)")
    axes[1].set_title("(b) $\\Delta$ cost per decision (24full - 4sensor), millicents")
    for i in range(len(C_INV_GRID)):
        for j in range(len(L_GRID)):
            axes[1].text(j, i, f"{delta2.values[i,j]:+.1f}", ha="center", va="center",
                         fontsize=8, color="black")
    fig.colorbar(im1, ax=axes[1])

    fig.suptitle("P8 asymmetric cost (C$_\\mathrm{inv}$ $\\times$ L) frontier")
    fig.tight_layout()
    fig.savefig(FIG / "p8_asym_cost.png", dpi=150)
    fig.savefig(FIG / "p8_asym_cost.pdf")
    plt.close(fig)


def main():
    scores, y = load_split()
    grid = grid_eval(scores, y)
    grid.to_csv(OUT / "p8_asym_cost.tsv", sep="\t", index=False)
    boot = boot_grid(scores, y)
    boot.to_csv(OUT / "p8_asym_cost_boot.tsv", sep="\t", index=False)

    n_total = len(grid) // len(MODELS)
    n_24 = grid[grid.model == "XGB-24full"]
    n_20 = grid[grid.model == "XGB-20raw"]
    n_4s = grid[grid.model == "XGB-4sensor"]
    delta_24_vs_20_grid = (n_24.cost_per_dec_usd.values - n_20.cost_per_dec_usd.values)
    delta_24_vs_4s_grid = (n_24.cost_per_dec_usd.values - n_4s.cost_per_dec_usd.values)
    n_24wins_24vs20 = int((delta_24_vs_20_grid < 0).sum())
    n_4swins_4s_vs_24 = int((delta_24_vs_4s_grid > 0).sum())
    n_24wins_24vs4s = int((delta_24_vs_4s_grid < 0).sum())
    n_4swins_4s_vs_24_cheaper = n_4swins_4s_vs_24

    # Cells where 24full is cheaper than 4sensor with significant CI exclusion
    n_24full_cheaper_than_4sensor_ci = int(boot["ci_excl_zero_24_vs_4s"].sum())
    n_24full_cheaper_than_20raw_ci = int(boot["ci_excl_zero_24_vs_20"].sum())

    summary = {
        "C_inv_grid": C_INV_GRID,
        "L_grid": L_GRID,
        "n_cells": int(n_total),
        "n_boots": int(N_BOOT),
        "n_24full_cheaper_than_20raw_at_point_estimate": n_24wins_24vs20,
        "n_4sensor_cheaper_than_24full_at_point_estimate": n_4swins_4s_vs_24_cheaper,
        "n_24full_cheaper_than_4sensor_at_point_estimate": n_24wins_24vs4s,
        "n_24full_cheaper_than_20raw_with_ci_excl_zero": n_24full_cheaper_than_20raw_ci,
        "n_24full_cheaper_than_4sensor_with_ci_excl_zero": n_24full_cheaper_than_4sensor_ci,
        "min_delta_24_vs_20": round(float(delta_24_vs_20_grid.min()), 7),
        "max_delta_24_vs_20": round(float(delta_24_vs_20_grid.max()), 7),
        "min_delta_24_vs_4s": round(float(delta_24_vs_4s_grid.min()), 7),
        "max_delta_24_vs_4s": round(float(delta_24_vs_4s_grid.max()), 7),
    }
    with open(OUT / "p8_asym_cost_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    figure_make(grid, boot)
    print("iter 40 P8 asymmetric cost frontier")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
