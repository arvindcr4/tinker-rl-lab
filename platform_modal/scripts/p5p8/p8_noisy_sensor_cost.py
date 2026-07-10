#!/usr/bin/env python3
"""P8 JOB A (iter 32): noisy-sensor cost-optimal robustness.

Prior P8 cost work (item 35, iter 28) used an oracle 4-aggregate sensor:
the LLM emits the exact V_mean/std/max/min per row. Item 10 (iter 4) gave
a sensor-noise budget of sigma <= 0.02 (the LLM cannot emit the exact
aggregates without drift). We close the loop: does the cost-optimal
thesis (sensor is not a scorer) survive the noise budget?

Concretely, for each sensor-noise sigma in {0.0, 0.005, 0.010, 0.020, 0.050},
add Gaussian noise N(0, sigma) to the four aggregate columns on the held-out
split, refit XGB-24full (the noisy 24-feature tree), and recompute the
cost-optimal frame at rho in {2, 5, 10, 20, 50, 100, 200, 500}.

Outputs
-------
platform_hybrid/experiments/results/p5p8/p8_noisy_sensor.tsv        (sigma x rho cost-optimal table)
platform_hybrid/experiments/results/p5p8/p8_noisy_sensor_boot.tsv   (paired bootstrap CI, sigma x rho)
platform_hybrid/experiments/results/p5p8/p8_noisy_sensor_breakeven.tsv  (per-sigma break-even L*)
platform_hybrid/experiments/results/p5p8/p8_noisy_sensor_summary.json
platform_hybrid/experiments/results/p5p8/figures/p8_noisy_sensor.{png,pdf}

Stdlib + numpy + pandas + xgboost. <=300 lines. Real data only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "fraud_data.csv"
TEST = ROOT / "test_data.csv"
OUT = ROOT / "experiments" / "results" / "p5p8"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

V20 = [f"V{i}" for i in range(1, 21)]
V_AGG = ["V_mean", "V_std", "V_max", "V_min"]
SENSES = {"XGB-20raw": 0.0, "XGB-24full": 0.0035}
C_INV = 0.50
RHO_GRID = [2, 5, 10, 20, 50, 100, 200, 500]
SIGMA_GRID = [0.0, 0.005, 0.010, 0.020, 0.050]
N_BOOT = 400
BOOT_SEED = 20260704
TREE_SEED = 42


def fit_tree(X_tr, y_tr):
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", random_state=TREE_SEED, n_jobs=4,
    )
    clf.fit(X_tr, y_tr)
    return clf


def optimal_cut(score, y, C_inv, L):
    """argmin over all rank cutoffs of C_inv*alerts + L*missed."""
    N = len(y)
    P = int(y.sum())
    order = np.argsort(-score, kind="mergesort")
    ys = y[order]
    ss = score[order]
    cum_tp = np.concatenate([[0], np.cumsum(ys)])
    k = np.arange(N + 1)
    alerts = k
    tp = cum_tp
    fn = P - tp
    det_cost = C_inv * alerts + L * fn
    kstar = int(np.argmin(det_cost))
    if kstar == 0:
        tau = float(ss[0]) + 1e-9
    elif kstar >= N:
        tau = float(ss[-1])
    else:
        tau = float(ss[kstar - 1])
    tp_star = int(tp[kstar])
    alert_pct = kstar / N
    recall = tp_star / max(P, 1)
    return {
        "tau": tau, "kstar": kstar, "tp": tp_star, "fn": int(fn[kstar]),
        "alert_pct": float(alert_pct), "recall": float(recall),
        "cost_per_dec": float(det_cost[kstar] / N),
    }


def add_noise(df, cols, sigma, seed):
    """Add Gaussian noise to specified columns with given sigma."""
    if sigma <= 0:
        return df.copy()
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(np.float64) + rng.normal(0.0, sigma, size=len(out))
    return out


def fit_noisy_scores(df_tr, df_te, y_tr, y_te, sigma, seed):
    """Fit XGB-24full on noisy training aggregates; return test scores."""
    df_tr_n = add_noise(df_tr, V_AGG, sigma, seed)
    df_te_n = add_noise(df_te, V_AGG, sigma, seed)
    cols = V20 + V_AGG
    clf = fit_tree(df_tr_n[cols].to_numpy(np.float64), y_tr)
    return clf.predict_proba(df_te_n[cols].to_numpy(np.float64))[:, 1]


def paired_bootstrap_diff(score_a, score_b, y, C_inv, L, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired bootstrap on the cost-per-decision difference (a - b)."""
    N = len(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        d_a = optimal_cut(score_a[idx], y[idx], C_inv, L)["cost_per_dec"]
        d_b = optimal_cut(score_b[idx], y[idx], C_inv, L)["cost_per_dec"]
        diffs[b] = d_a - d_b
    mean = float(np.mean(diffs))
    lo, hi = (float(x) for x in np.quantile(diffs, [0.025, 0.975]))
    return {"mean": mean, "lo": lo, "hi": hi, "sig_pos": hi < 0, "sig_neg": lo > 0}


def breakeven_L(score_20, score_24, y, sigma_tag):
    """Find L* where 24full cost-advantage over 20raw crosses 0.

    Cost for 24full includes the sensing charge c_sense=$0.0035/row.
    For L < L*, sensor charging dominates (raw cheaper);
    for L > L*, sensor recall-restore dominates (24full cheaper).
    """
    c_sense = 0.0035
    Ls = np.linspace(0.5, 500.0, 1000)
    diffs = []
    for L in Ls:
        c20 = optimal_cut(score_20, y, C_INV, L)["cost_per_dec"]
        c24 = optimal_cut(score_24, y, C_INV, L)["cost_per_dec"] + c_sense
        diffs.append(c24 - c20)
    diffs = np.array(diffs)
    if not (np.any(diffs < 0) and np.any(diffs > 0)):
        # Monotone: return sentinel
        return {"L_star": None, "monotone": diffs[-1] < diffs[0]}
    # Find first crossing
    cross = np.where(np.diff(np.sign(diffs)) != 0)[0]
    if len(cross) == 0:
        return {"L_star": None, "monotone": False}
    i = int(cross[0])
    # Linear interpolate
    L1, L2 = float(Ls[i]), float(Ls[i + 1])
    d1, d2 = float(diffs[i]), float(diffs[i + 1])
    L_star = L1 - d1 * (L2 - L1) / (d2 - d1)
    return {"L_star": float(L_star), "monotone": False, "range": (L1, L2)}


def main():
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)
    y_tr = df_tr["Class"].to_numpy(np.int32)
    y_te = df_te["Class"].to_numpy(np.int32)

    # Baseline: XGB-20raw (no noise, no aggregates)
    clf_raw = fit_tree(df_tr[V20].to_numpy(np.float64), y_tr)
    score_raw = clf_raw.predict_proba(df_te[V20].to_numpy(np.float64))[:, 1]

    # Per-sigma XGB-24full
    noisy_scores = {}
    for sigma in SIGMA_GRID:
        noisy_scores[sigma] = fit_noisy_scores(df_tr, df_te, y_tr, y_te, sigma, BOOT_SEED)

    # Build cost-optimal table
    rows_cost = []
    for sigma in SIGMA_GRID:
        s24 = noisy_scores[sigma]
        for rho in RHO_GRID:
            L = rho * C_INV
            c20 = optimal_cut(score_raw, y_te, C_INV, L)
            c24 = optimal_cut(s24, y_te, C_INV, L)
            # Add sensing charge for 24full only
            cost24_full = c24["cost_per_dec"] + 0.0035
            cost20_full = c20["cost_per_dec"]
            rows_cost.append({
                "sigma": sigma, "rho": rho,
                "raw_tau": c20["tau"], "raw_alert_pct": c20["alert_pct"],
                "raw_recall": c20["recall"], "raw_cost_per_dec": cost20_full,
                "full_tau": c24["tau"], "full_alert_pct": c24["alert_pct"],
                "full_recall": c24["recall"], "full_cost_per_dec": cost24_full,
                "delta_cost": cost24_full - cost20_full,
            })

    df_cost = pd.DataFrame(rows_cost)
    df_cost.to_csv(OUT / "p8_noisy_sensor.tsv", sep="\t", index=False)

    # Per-(sigma, rho) paired bootstrap CI on the cost difference (with sensing charge)
    # delta_cost = cost_24full - cost_20raw, positive = 24full costlier.
    # CI excludes 0 -> certifiably different.
    # If ci_hi < 0: 24full certifiably cheaper (positive finding for sensor).
    # If ci_lo > 0: 24full certifiably costlier (sensor pays for nothing at low L).
    rows_boot = []
    for sigma in SIGMA_GRID:
        s24 = noisy_scores[sigma]
        for rho in RHO_GRID:
            L = rho * C_INV
            ci = paired_bootstrap_diff(s24 + 0.0, score_raw, y_te, C_INV, L)
            # offset by the constant sensing charge: full - raw includes +0.0035 always
            # So we shift the mean by +0.0035 in the direction (full - raw).
            mean_shift = ci["mean"] + 0.0035
            lo_shift = ci["lo"] + 0.0035
            hi_shift = ci["hi"] + 0.0035
            rows_boot.append({
                "sigma": sigma, "rho": rho,
                "delta_cost_per_dec": mean_shift,
                "ci_lo": lo_shift, "ci_hi": hi_shift,
                "sensor_wins_sig": hi_shift < 0,   # 24full - raw < 0 -> sensor wins
                "sensor_loses_sig": lo_shift > 0,  # 24full - raw > 0 -> sensor loses
            })

    df_boot = pd.DataFrame(rows_boot)
    df_boot.to_csv(OUT / "p8_noisy_sensor_boot.tsv", sep="\t", index=False)

    # Break-even L* per sigma
    rows_be = []
    for sigma in SIGMA_GRID:
        s24 = noisy_scores[sigma]
        be = breakeven_L(score_raw, s24, y_te, sigma)
        rows_be.append({
            "sigma": sigma,
            "L_star": be["L_star"],
            "monotone": be["monotone"],
        })
    df_be = pd.DataFrame(rows_be)
    df_be.to_csv(OUT / "p8_noisy_sensor_breakeven.tsv", sep="\t", index=False)

    # Summary JSON
    summary = {
        "n_test": int(len(y_te)),
        "n_pos_test": int(y_te.sum()),
        "n_boot": N_BOOT,
        "boot_seed": BOOT_SEED,
        "sigma_grid": SIGMA_GRID,
        "rho_grid": RHO_GRID,
        "c_inv": C_INV,
        "c_sense": 0.0035,
        # Key headline: at sigma=0.02 (item-10 noise budget), which rho certify sensor wins?
        "headline_at_item10_budget": {
            "sigma": 0.02,
            "rho_sensor_wins_sig": df_boot.query("sigma == 0.02 and sensor_wins_sig").shape[0],
            "rho_sensor_loses_sig": df_boot.query("sigma == 0.02 and sensor_loses_sig").shape[0],
            "rho_neither_sig": df_boot.query("sigma == 0.02 and not sensor_wins_sig and not sensor_loses_sig").shape[0],
            "rho_total": len(RHO_GRID),
        },
        "headline_at_oracle_sigma0": {
            "sigma": 0.0,
            "rho_sensor_wins_sig": df_boot.query("sigma == 0.0 and sensor_wins_sig").shape[0],
            "rho_sensor_loses_sig": df_boot.query("sigma == 0.0 and sensor_loses_sig").shape[0],
            "rho_neither_sig": df_boot.query("sigma == 0.0 and not sensor_wins_sig and not sensor_loses_sig").shape[0],
        },
        "headline_at_high_noise_sigma05": {
            "sigma": 0.05,
            "rho_sensor_wins_sig": df_boot.query("sigma == 0.05 and sensor_wins_sig").shape[0],
            "rho_sensor_loses_sig": df_boot.query("sigma == 0.05 and sensor_loses_sig").shape[0],
            "rho_neither_sig": df_boot.query("sigma == 0.05 and not sensor_wins_sig and not sensor_loses_sig").shape[0],
        },
        "break_even_L_star": {
            str(row["sigma"]): row["L_star"] for _, row in df_be.iterrows()
        },
    }
    (OUT / "p8_noisy_sensor_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure: delta_cost vs rho, one curve per sigma
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        for sigma in SIGMA_GRID:
            sub = df_boot[df_boot["sigma"] == sigma].sort_values("rho")
            ax.plot(sub["rho"], sub["delta_cost_per_dec"] * 1000,
                    marker="o", label=f"$\\sigma={sigma}$")
            ax.fill_between(sub["rho"], sub["ci_lo"] * 1000, sub["ci_hi"] * 1000,
                            alpha=0.15)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_xlabel(r"cost ratio $\rho = L/C_{\mathrm{inv}}$")
        ax.set_ylabel(r"$\Delta$ cost / decision (milli-USD), positive = 24full costlier")
        ax.set_title("Noisy-sensor cost advantage: 24full $-$ 20raw\n"
                     r"(with $c_{\rm sense}=\$0.0035$/row for 24full)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "p8_noisy_sensor.png", dpi=130)
        fig.savefig(FIG / "p8_noisy_sensor.pdf")
        plt.close(fig)
    except Exception as e:
        summary["figure_error"] = str(e)
        (OUT / "p8_noisy_sensor_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()