#!/usr/bin/env python3
"""P8 JOB A (iter 72): Calibration drift under sensor noise.

Fresh vein: the iter-8 sensor-noise sweep measured AUC degradation
under sigma_mult ∈ {0.05, ..., 2.0} noise on the four aggregate
features (V_mean, V_std, V_max, V_min). Iter-60 measured operational
calibration at top-K alerts (clean features only).

The intersection — **operational calibration drift under sensor noise** —
has not been measured. This iter closes that gap. At each
sigma_mult ∈ {0.0, 0.05, 0.10, 0.20, 0.50, 1.00} on the 4 sensor
aggregates, we compute:
  - mean_predicted_topK (model's average predicted prob among top-K alerts)
  - observed_pos_rate_topK (actual positive rate among top-K alerts)
  - operational calibration gap = mean_predicted_topK - observed_pos_rate_topK
  - brier_topK = mean((pred - label)^2) over the top-K alerts

at K ∈ {0.5, 1.0, 2.0, 5.0}% on the 10k test split. We run a paired
bootstrap (B=400, seed 20260705) on Δ(gap), Δ(brier) at each (sigma, K).

Operational meaning: a fraud-ops team adopting the LLM-as-sensor stack
needs to know "if my sensor-extracted features have noise σ, what is
the worst-case over-confidence in my alert queue?". This iter answers
it directly.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_calib_noise.tsv               (6 sigma × 4 K × 4 trees)
platform_hybrid/experiments/results/p5p8/p8_calib_noise_boot.tsv          (paired bootstrap)
platform_hybrid/experiments/results/p5p8/p8_calib_noise_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_calib_noise.{png,pdf}

Stdlib + numpy + pandas + xgboost + matplotlib. <=290 lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------- paths
ROOT = Path("/home/claude/tinker-rl-lab-minimax")
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"

# ----------------------------------------------------------- experiment
SEED = 20260705
N_BOOT = 400
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
K_PCTS = [0.5, 1.0, 2.0, 5.0]   # alert budget as % of test
RAW20 = [f"V{i}" for i in range(1, 21)]
AGG4 = ["V_mean", "V_std", "V_max", "V_min"]
ALL24 = RAW20 + AGG4
N_TEST = 10000

# Pre-defined sensor std scales for reproducible sigma sweeps
# (V1..V20 are z-score-like already; aggregates are scaled by their empirical stds)
SENSOR_SCALE = {"V_mean": 1.0, "V_std": 0.5, "V_max": 1.0, "V_min": 1.0}


# ------------------------------------------------------------- helpers
def fit_tree(X_tr, y_tr):
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    ).fit(X_tr, y_tr)


def operational_calibration(probs, y_true, k_pct):
    """For top-k_pct% highest-prob rows, return (mean_pred, obs_rate, brier, gap)."""
    n = len(probs)
    k = max(1, int(round(n * k_pct / 100.0)))
    idx = np.argsort(-probs)[:k]
    p_top = probs[idx]
    y_top = y_true[idx]
    mean_pred = float(p_top.mean())
    obs_rate = float(y_top.mean())
    brier = float(((p_top - y_top) ** 2).mean())
    return mean_pred, obs_rate, brier, mean_pred - obs_rate


def add_noise(X, sigma_mult, rng):
    """Add Gaussian noise to the 4 aggregate columns scaled by SENSOR_SCALE."""
    Xn = X.copy()
    for c in AGG4:
        s = SENSOR_SCALE[c] * sigma_mult
        if s > 0:
            Xn[c] = Xn[c] + rng.normal(0.0, s, size=len(Xn))
    return Xn


# ------------------------------------------------------------- main
def main():
    print("[p8_calib_noise] loading...")
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    # derive aggregates for both
    for df in (train, test):
        df["V_mean"] = df[RAW20].mean(axis=1)
        df["V_std"] = df[RAW20].std(axis=1)
        df["V_max"] = df[RAW20].max(axis=1)
        df["V_min"] = df[RAW20].min(axis=1)

    X_tr_24 = train[ALL24].values
    y_tr = train["Class"].values
    X_te_24 = test[ALL24].values
    y_te = test["Class"].values
    X_te_20 = test[RAW20].values

    rng_global = np.random.default_rng(SEED)
    sigma_rng = np.random.default_rng(SEED)

    # Pre-compute noisy test sets for each sigma
    print("[p8_calib_noise] precomputing noisy variants...")
    Xte_24_noisy = {s: add_noise(pd.DataFrame(X_te_24, columns=ALL24),
                                  s, sigma_rng).values for s in SIGMAS}
    Xte_20_clean = X_te_20  # 20raw has no sensor features, so noise does not apply

    # Fit 4 trees ONCE on the clean training set
    print("[p8_calib_noise] fitting 4 trees (20raw, 24full, 4sensor, 20raw+agg)...")
    tree_20raw = fit_tree(train[RAW20].values, y_tr)
    tree_24full = fit_tree(X_tr_24, y_tr)
    tree_4sensor = fit_tree(train[AGG4].values, y_tr)
    tree_20raw_plus = fit_tree(X_tr_24, y_tr)  # alias for clarity

    # Predict on each (sigma, K) — store prob arrays for bootstrap
    pred_cache = {}
    for s in SIGMAS:
        # 24full uses noisy aggregates
        p24 = tree_24full.predict_proba(Xte_24_noisy[s])[:, 1]
        # 4sensor uses noisy aggregates
        p4 = tree_4sensor.predict_proba(Xte_24_noisy[s][:, 20:])[:, 1]
        # 20raw has no aggregates, so unaffected by sigma
        p20 = tree_20raw.predict_proba(X_te_20)[:, 1]
        pred_cache[("20raw", s)] = p20
        pred_cache[("24full", s)] = p24
        pred_cache[("4sensor", s)] = p4

    # ----------------------------------------------------------- point estimates
    print("[p8_calib_noise] computing operational calibration point estimates...")
    rows = []
    for s in SIGMAS:
        for k_pct in K_PCTS:
            for tree in ("20raw", "24full", "4sensor"):
                probs = pred_cache[(tree, s)]
                mp, obs, br, gap = operational_calibration(probs, y_te, k_pct)
                rows.append({
                    "sigma": s,
                    "k_pct": k_pct,
                    "tree": tree,
                    "mean_pred": mp,
                    "obs_rate": obs,
                    "brier_topk": br,
                    "calib_gap": gap,
                    "n_alerts": int(round(N_TEST * k_pct / 100.0)),
                })
    pt_df = pd.DataFrame(rows)
    pt_df.to_csv(OUT / "p8_calib_noise.tsv", sep="\t", index=False,
                 float_format="%.6f")
    print(f"[p8_calib_noise] point estimates written: {len(pt_df)} rows")

    # --------------------------------------------------------- paired bootstrap
    print("[p8_calib_noise] running paired bootstrap (B=%d)..." % N_BOOT)
    boot_rng = np.random.default_rng(SEED)
    idx_pool = np.arange(N_TEST)

    boot_rows = []
    for s in SIGMAS:
        for k_pct in K_PCTS:
            for tree_a, tree_b in (("24full", "20raw"),
                                    ("24full", "4sensor")):
                pa = pred_cache[(tree_a, s)]
                pb = pred_cache[(tree_b, s)]
                # bootstrap paired samples
                gap_a = np.empty(N_BOOT)
                gap_b = np.empty(N_BOOT)
                brier_a = np.empty(N_BOOT)
                brier_b = np.empty(N_BOOT)
                k_n = max(1, int(round(N_TEST * k_pct / 100.0)))
                for b in range(N_BOOT):
                    ii = boot_rng.choice(idx_pool, size=N_TEST, replace=True)
                    probs_a = pa[ii]
                    probs_b = pb[ii]
                    y_b = y_te[ii]
                    # top-k per resample
                    top_a = np.argsort(-probs_a)[:k_n]
                    top_b = np.argsort(-probs_b)[:k_n]
                    gap_a[b] = probs_a[top_a].mean() - y_b[top_a].mean()
                    gap_b[b] = probs_b[top_b].mean() - y_b[top_b].mean()
                    brier_a[b] = ((probs_a[top_a] - y_b[top_a]) ** 2).mean()
                    brier_b[b] = ((probs_b[top_b] - y_b[top_b]) ** 2).mean()

                d_gap = gap_a - gap_b
                d_brier = brier_a - brier_b
                boot_rows.append({
                    "sigma": s,
                    "k_pct": k_pct,
                    "tree_a": tree_a,
                    "tree_b": tree_b,
                    "delta_gap": float(d_gap.mean()),
                    "gap_lo": float(np.percentile(d_gap, 2.5)),
                    "gap_hi": float(np.percentile(d_gap, 97.5)),
                    "excl_zero_gap": bool(
                        np.percentile(d_gap, 2.5) > 0
                        or np.percentile(d_gap, 97.5) < 0
                    ),
                    "delta_brier": float(d_brier.mean()),
                    "brier_lo": float(np.percentile(d_brier, 2.5)),
                    "brier_hi": float(np.percentile(d_brier, 97.5)),
                    "excl_zero_brier": bool(
                        np.percentile(d_brier, 2.5) > 0
                        or np.percentile(d_brier, 97.5) < 0
                    ),
                })

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT / "p8_calib_noise_boot.tsv", sep="\t", index=False,
                   float_format="%.6f")
    print(f"[p8_calib_noise] bootstrap written: {len(boot_df)} rows")

    # ------------------------------------------------------------ summary
    summary = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "sigmas": SIGMAS,
        "k_pcts": K_PCTS,
        "n_boot": N_BOOT,
        "seed": SEED,
        "headlines": {},
    }
    # Sharpest finding — at sigma=0.20 (4×scribe noise floor), K=2%, what is
    # the calib gap on the 24-full tree vs the 4-sensor tree?
    for k_pct in K_PCTS:
        for s in SIGMAS:
            pt = pt_df[(pt_df.sigma == s) & (pt_df.k_pct == k_pct)]
            row_24 = pt[pt.tree == "24full"].iloc[0]
            row_4 = pt[pt.tree == "4sensor"].iloc[0]
            row_20 = pt[pt.tree == "20raw"].iloc[0]
            summary["headlines"][f"sigma={s}_k={k_pct}"] = {
                "calib_gap_24full": row_24.calib_gap,
                "calib_gap_4sensor": row_4.calib_gap,
                "calib_gap_20raw": row_20.calib_gap,
                "obs_rate_24full": row_24.obs_rate,
                "obs_rate_4sensor": row_4.obs_rate,
            }

    with open(OUT / "p8_calib_noise_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("[p8_calib_noise] summary written")

    # ------------------------------------------------------------- figure
    print("[p8_calib_noise] rendering figure...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"20raw": "#4c72b0", "24full": "#dd8452", "4sensor": "#55a868"}

    # Panel A: calib_gap vs sigma at K=2%
    ax = axes[0]
    sub = pt_df[pt_df.k_pct == 2.0]
    for tree in ("20raw", "24full", "4sensor"):
        ss = sub[sub.tree == tree]
        ax.plot(ss.sigma, ss.calib_gap, "-o", color=colors[tree],
                label=f"XGB-{tree}", linewidth=2)
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel(r"$\sigma_{\text{mult}}$ (sensor noise scaling)")
    ax.set_ylabel(r"Calib gap = $\bar p_{\text{top-K}} - \pi_{\text{top-K}}$ at $K{=}2\%$")
    ax.set_title("Operational calibration gap vs sensor noise ($K{=}2\%$)")
    ax.legend(fontsize=8)

    # Panel B: calib_gap vs K at sigma=0.10
    ax = axes[1]
    sub = pt_df[pt_df.sigma == 0.10]
    for tree in ("20raw", "24full", "4sensor"):
        ss = sub[sub.tree == tree]
        ax.plot(ss.k_pct, ss.calib_gap, "-o", color=colors[tree],
                label=f"XGB-{tree}", linewidth=2)
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_xlabel(r"$K$ budget (\% of test alerted)")
    ax.set_ylabel(r"Calib gap at $\sigma{=}0.10$")
    ax.set_title("Operational calibration gap vs alert budget ($σ{=}0.10$)")
    ax.legend(fontsize=8)

    fig.suptitle("P8 calibration under sensor noise (iter 72)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "p8_calib_noise.png", dpi=150)
    fig.savefig(FIG / "p8_calib_noise.pdf")
    plt.close(fig)
    print("[p8_calib_noise] done")


if __name__ == "__main__":
    main()