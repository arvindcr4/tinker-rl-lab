#!/usr/bin/env python3
"""P8 JOB B (iter 36): cost-per-decision × sensor-noise phase diagram.

Closes the iter-32 synthesis note: "a 2-point sensor-noise sweep would
test whether L* scales with c_sense". This script builds the FULL
phase diagram: 5 sensor-noise levels x 5 fraud-loss values x 3 trees
on the released 10k held-out split. Computes expected cost per decision
at the noise-aware cost-optimal threshold tau*, paired bootstrap CI on
the (24full - 20raw) cost delta.

Headline question: at what (sigma, L) does the LLM sensor pay for itself?

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_cost_phase_diagram.tsv
platform_hybrid/experiments/results/p5p8/p8_cost_phase_diagram_boot.tsv
platform_hybrid/experiments/results/p5p8/p8_cost_phase_diagram_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_cost_phase_diagram.{png,pdf}

Stdlib + numpy + pandas + xgboost + sklearn + matplotlib. <=300 lines.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
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
C_INV = 0.50
C_SENSE = 0.0035
N_BOOT = 1000
BOOT_SEED = 20260704
TREE_SEED = 42
# Phase grid
SIGMA_GRID = [0.0, 0.005, 0.01, 0.02, 0.05]
# Fraud-loss L per missed fraud ($). Lower L = cost-saving matters more.
L_GRID = [1.0, 5.0, 25.0, 100.0, 500.0]


def load_split():
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    X_tr, y_tr = train[V20 + V_AGG], train["Class"].values
    X_te, y_te = test[V20 + V_AGG], test["Class"].values
    return X_tr, y_tr, X_te, y_te


def fit_tree(X_tr, y_tr):
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", random_state=TREE_SEED, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    return clf


def expected_cost(p, y, L, c_inv=C_INV, c_sense=C_SENSE):
    """Expected cost per decision = E[c_inv * alert + c_sense * p * n_alert + L * miss].

    For fair comparison across trees, c_sense is applied to the "LLM sensor"
    pipeline ONLY for trees that use V_AGG (24full, 4sensor); XGB-20raw
    pays 0 for sensing.
    """
    # Cost-optimal threshold: alert if (p >= L / (L + c_inv)) i.e. p >= tau_star
    tau = L / (L + c_inv)
    alert = p >= tau
    n = len(p)
    n_alerts = int(alert.sum())
    n_misses = int(((y == 1) & (~alert)).sum())
    cost_alerts = c_inv * n_alerts
    cost_sense = c_sense * n_alerts  # cost is per-row-of-input whether alert or not;
    # for simplicity apply only when alerted (LLM is queried at decision time
    # only for rows under review).
    cost_misses = L * n_misses
    return cost_alerts + cost_sense + cost_misses


def per_bootstrap_diff(y_te, p_te_20, p_te_24, L, c_sense=C_SENSE):
    """Paired bootstrap CI on (cost_24full - cost_20raw) per L value."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y_te)
    tau = L / (L + C_INV)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        y_b = y_te[idx]
        a_20 = p_te_20[idx] >= tau
        a_24 = p_te_24[idx] >= tau
        n_alerts_20 = int(a_20.sum())
        n_alerts_24 = int(a_24.sum())
        n_miss_20 = int(((y_b == 1) & (~a_20)).sum())
        n_miss_24 = int(((y_b == 1) & (~a_24)).sum())
        cost_20 = C_INV * n_alerts_20 + 0 * n_alerts_20 + L * n_miss_20
        cost_24 = C_INV * n_alerts_24 + c_sense * n_alerts_24 + L * n_miss_24
        diffs.append((cost_24 - cost_20) / n)  # per-decision cost delta
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def add_noise(X_agg, sigma, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return X_agg + rng.normal(0, sigma, size=X_agg.shape)


def main():
    print("Loading + training baseline trees")
    X_tr, y_tr, X_te, y_te = load_split()
    # Train 20raw tree once (no sensor)
    clf_20 = fit_tree(X_tr[V20], y_tr)
    p_te_20 = clf_20.predict_proba(X_te[V20])[:, 1]
    print(f"  XGB-20raw: AUC={roc_auc_score(y_te, p_te_20):.4f}")

    print("Phase diagram: 5 sigmas x 5 L values")
    rows = []
    boot_rows = []
    for sigma in SIGMA_GRID:
        if sigma > 0:
            X_te_24 = X_te.copy()
            X_te_24[V_AGG] = add_noise(X_te[V_AGG], sigma)
            X_tr_24 = X_tr.copy()
            X_tr_24[V_AGG] = add_noise(X_tr[V_AGG], sigma, seed=BOOT_SEED + 1)
        else:
            X_te_24 = X_te.copy()
            X_tr_24 = X_tr.copy()
        clf_24 = fit_tree(X_tr_24, y_tr)
        p_te_24 = clf_24.predict_proba(X_te_24)[:, 1]
        auc_24 = roc_auc_score(y_te, p_te_24)
        print(f"  sigma={sigma:.3f}: XGB-24full AUC={auc_24:.4f}")
        for L in L_GRID:
            tau = L / (L + C_INV)
            cost_20 = expected_cost(p_te_20, y_te, L, c_sense=0.0)
            cost_24 = expected_cost(p_te_24, y_te, L, c_sense=C_SENSE)
            rows.append({
                "sigma": sigma,
                "L": L,
                "tau_star": tau,
                "cost_20raw_per_dec": cost_20 / len(y_te),
                "cost_24full_per_dec": cost_24 / len(y_te),
                "delta_cost_per_dec": (cost_24 - cost_20) / len(y_te),
                "auc_20raw": float(roc_auc_score(y_te, p_te_20)),
                "auc_24full": float(auc_24),
            })
            m, lo, hi = per_bootstrap_diff(y_te, p_te_20, p_te_24, L)
            boot_rows.append({
                "sigma": sigma,
                "L": L,
                "delta_boot_mean": m,
                "delta_ci025": lo,
                "delta_ci975": hi,
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
                "sensor_pays_off": bool(hi < 0.0),  # cost delta < 0 = sensor cheaper
            })

    df = pd.DataFrame(rows)
    df_boot = pd.DataFrame(boot_rows)
    df.to_csv(OUT / "p8_cost_phase_diagram.tsv", sep="\t", index=False)
    df_boot.to_csv(OUT / "p8_cost_phase_diagram_boot.tsv", sep="\t", index=False)
    print(f"  -> p8_cost_phase_diagram.tsv ({len(df)} rows)")
    print(f"  -> p8_cost_phase_diagram_boot.tsv ({len(df_boot)} rows)")

    # Headline: which (sigma, L) cells has the sensor strictly paying off?
    pays_off = df_boot[df_boot["sensor_pays_off"]]
    n_pays = len(pays_off)
    n_total = len(df_boot)
    summary = {
        "sigma_grid": SIGMA_GRID,
        "L_grid": L_GRID,
        "n_pays_off": int(n_pays),
        "n_total": int(n_total),
        "pay_off_cells": pays_off[["sigma", "L"]].to_dict(orient="records"),
        "headline": {
            "sensor_pays_off_at_sigma0": bool(
                df_boot[(df_boot["sigma"] == 0.0) & (df_boot["sensor_pays_off"])].shape[0] > 0
            ),
            "sensor_never_pays_off": bool(n_pays == 0),
            "min_delta_at_sigma05": float(
                df_boot[df_boot["sigma"] == 0.05]["delta_boot_mean"].min()
            ),
            "max_delta_at_sigma0": float(
                df_boot[df_boot["sigma"] == 0.0]["delta_boot_mean"].max()
            ),
        },
        "boot_table": boot_rows,
        "rows": rows,
    }
    with open(OUT / "p8_cost_phase_diagram_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    # Phase diagram figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pivot_delta = df.pivot(index="L", columns="sigma", values="delta_cost_per_dec")
    im0 = axes[0].pcolormesh(
        np.arange(len(SIGMA_GRID)), np.arange(len(L_GRID)),
        pivot_delta.values, cmap="RdBu_r", vmin=-0.02, vmax=0.02,
    )
    axes[0].set_xticks(np.arange(len(SIGMA_GRID)) + 0.5)
    axes[0].set_xticklabels([f"{s:.3f}" for s in SIGMA_GRID], rotation=45)
    axes[0].set_yticks(np.arange(len(L_GRID)) + 0.5)
    axes[0].set_yticklabels([f"${int(L)}" for L in L_GRID])
    axes[0].set_xlabel(r"sensor noise $\sigma$")
    axes[0].set_ylabel("fraud loss $L$")
    axes[0].set_title(r"$\Delta$ cost/dec (24full $-$ 20raw)")
    plt.colorbar(im0, ax=axes[0])

    pivot_payoff = df_boot.pivot(index="L", columns="sigma", values="delta_boot_mean")
    im1 = axes[1].pcolormesh(
        np.arange(len(SIGMA_GRID)), np.arange(len(L_GRID)),
        pivot_payoff.values, cmap="RdBu_r", vmin=-0.02, vmax=0.02,
    )
    axes[1].set_xticks(np.arange(len(SIGMA_GRID)) + 0.5)
    axes[1].set_xticklabels([f"{s:.3f}" for s in SIGMA_GRID], rotation=45)
    axes[1].set_yticks(np.arange(len(L_GRID)) + 0.5)
    axes[1].set_yticklabels([f"${int(L)}" for L in L_GRID])
    axes[1].set_xlabel(r"sensor noise $\sigma$")
    axes[1].set_ylabel("fraud loss $L$")
    axes[1].set_title(r"Bootstrap-mean $\Delta$ cost/dec")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(FIG / "p8_cost_phase_diagram.png", dpi=150)
    plt.savefig(FIG / "p8_cost_phase_diagram.pdf")
    plt.close()

    print("\n=== Phase diagram: bootstrap-mean delta cost/dec (24full - 20raw) ===")
    print(f"{'sigma':>8} {'L=$1':>10} {'L=$5':>10} {'L=$25':>10} {'L=$100':>10} {'L=$500':>10}")
    for sigma in SIGMA_GRID:
        line = f"{sigma:>8.3f}"
        for L in L_GRID:
            r = df_boot[(df_boot["sigma"] == sigma) & (df_boot["L"] == L)].iloc[0]
            star = "*" if r["sensor_pays_off"] else " "
            line += f" {r['delta_boot_mean']:>+7.4f}{star}"
        print(line)
    print(f"\n  * = sensor strictly cheaper (CI excludes 0, upper bound < 0)")
    print(f"  cells where sensor pays off: {n_pays}/{n_total}")


if __name__ == "__main__":
    main()