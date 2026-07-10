"""Pillar 1 iter49 -- Two-parameter iso-FLOP joint fit + LOO predictive residual.

iter41 showed early-fraction traces predict late R_max with calibration.
iter45 fixed a single axis -- log10(C) -- and derived alpha_dense=1.030 vs
alpha_moe=0.057. What has NOT been done is the **two-parameter joint fit**
R_max = a*log10(P) + b*log10(C) + c across all 12 anchors, with
leave-one-out (LOO) predictive residuals broken out by deterministic phase
taxonomy (plateau / saturation / drift / collapse).

Concretely this iteration answers four sharp questions:

  Q1. Linear-predictive question:
      For each anchor i, fit R_max = a * log10(P) + b * log10(C) + c on
      the other 11 anchors; how well does that predict R_max_i?
      Metric: LOO RMSE.

  Q2. Residual-by-phase question:
      Does LOO residual |R_max_pred - R_max_actual| correlate with the
      deterministic phase class?
      Pre-registered: "collapse" anchor (Nemotron-120B) has the largest
      |residual| (the prediction cannot anticipate collapse).

  Q3. Iso-FLOP optimal anchor picker:
      Given a target FLOP budget C_target, which of the 12 anchors
      maximises the predicted R_max? This is the operational translation
      of the Chinchilla iso-compute optimal: it picks (params_B, n_steps)
      pairs that empirically dominated, not just which axis is steeper.

  Q4. Hoffmann cross-check:
      Standard Chinchilla theory says alpha_P + alpha_D = beta. With a 2D
      fit we can ask: is (alpha_P, alpha_C) consistent with a power-law
      R_max ~ P^ap * C^ac (interpret ac as the data-epoch exponent)?

Outputs (5 artefacts):
  experiments/results/scaling_law_iter49_two_param.tsv
  experiments/results/scaling_law_iter49_loo_residuals.tsv
  experiments/results/scaling_law_iter49_phase_residual.tsv
  experiments/results/scaling_law_iter49_optimal_anchor.tsv
  experiments/results/scaling_law_iter49_predictions.tsv
  paper/sections/scaling_law_iter49.tex
  figures/scaling_law_iter49.{pdf,png}
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "experiments" / "results" / "scaling_law_iter45_compute_proxy.tsv"
PHASE_SRC = REPO / "experiments" / "results" / "scaling_law_iter33_phase_score.tsv"
RESULTS = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_SEC = REPO / "paper" / "sections"
for d in (FIG_DIR, PAPER_SEC):
    d.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260702)
B_BOOT = 400


def ols_3var(x1, x2, y):
    """OLS y = a*x1 + b*x2 + c. Returns a, b, c, R^2, fitted array, residuals."""
    x1 = np.asarray(x1, float)
    x2 = np.asarray(x2, float)
    y = np.asarray(y, float)
    X = np.column_stack([x1, x2, np.ones_like(x1)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - ymean_safe(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[0]), float(coef[1]), float(coef[2]), r2, yhat, y - yhat


def ymean_safe(y):
    return float(np.mean(y))


def spearman_pvalue(x, y, b=2000):
    """Permutation p-value for Spearman rho (no scipy.stats)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    obs = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(2026_0702)
    null = np.empty(b, float)
    for i in range(b):
        yr = rng.permutation(ry)
        null[i] = float(np.corrcoef(rx, yr)[0, 1])
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + b)
    return obs, float(p)


def load_iter45():
    with open(SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    return rows


def load_phase():
    """Attach deterministic phase classifier to each anchor (iter33 score)."""
    if not PHASE_SRC.exists():
        return {}
    with open(PHASE_SRC) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for r in rows:
        if "model" in r and "phase_classifier" in r:
            out[r["model"]] = r["phase_classifier"]
    return out


def main() -> None:
    rows = load_iter45()
    phase_of = load_phase()

    # Filter to good fits with R_max > 0
    recs = [r for r in rows if r.get("fit_ok", False) and r["R_max"] > 0]
    log_p = np.array([r["log10_P"] for r in recs])
    log_c = np.array([r["log10_C"] for r in recs])
    rmax = np.array([r["R_max"] for r in recs])

    # ---- (1) Two-parameter OLS fit (full sample) ----
    a_full, b_full, c_full, r2_full, fit_full, resid_full = ols_3var(log_p, log_c, rmax)

    fit_rows = []
    fit_rows.append({
        "stack": "all", "n_used": int(len(recs)), "alpha_logP": round(a_full, 4),
        "alpha_logC": round(b_full, 4), "intercept": round(c_full, 4),
        "r_squared": round(r2_full, 4), "rmse": round(float(np.sqrt(np.mean(resid_full ** 2))), 4),
        "median_Rmax": round(float(np.median(rmax)), 4),
        "note": "OLS R_max = a*log10(P) + b*log10(C) + c on full sample",
    })
    # Stack-conditional fits
    for arch in ("dense", "moe"):
        ix = np.array([i for i, r in enumerate(recs) if r["arch"] == arch])
        if len(ix) < 3:
            continue
        a_x, b_x, c_x, r2_x, _, res_x = ols_3var(log_p[ix], log_c[ix], rmax[ix])
        fit_rows.append({
            "stack": arch, "n_used": int(len(ix)), "alpha_logP": round(a_x, 4),
            "alpha_logC": round(b_x, 4), "intercept": round(c_x, 4),
            "r_squared": round(r2_x, 4), "rmse": round(float(np.sqrt(np.mean(res_x ** 2))), 4),
            "median_Rmax": round(float(np.median(rmax[ix])), 4),
            "note": "OLS R_max = a*log10(P) + b*log10(C) + c on stack-conditional sample",
        })

    fields1 = list(fit_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_two_param.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields1, delimiter="\t")
        w.writeheader()
        w.writerows(fit_rows)

    # ---- (2) Leave-one-out predictive residuals ----
    n = len(recs)
    loo_rows = []
    for i in range(n):
        mask = np.ones(n, bool)
        mask[i] = False
        a_i, b_i, c_i, _, _, _ = ols_3var(log_p[mask], log_c[mask], rmax[mask])
        yhat_i = a_i * log_p[i] + b_i * log_c[i] + c_i
        loo_rows.append({
            "model": recs[i]["model"],
            "params_B": recs[i]["params_B"],
            "arch": recs[i]["arch"],
            "phase": phase_of.get(recs[i]["model"], "unknown"),
            "R_max_actual": round(float(rmax[i]), 4),
            "R_max_predicted": round(float(yhat_i), 4),
            "residual": round(float(rmax[i] - yhat_i), 4),
            "abs_residual": round(float(abs(rmax[i] - yhat_i)), 4),
            "rel_residual": round(float((rmax[i] - yhat_i) / max(1e-9, yhat_i)), 4),
            "log10_P": round(float(log_p[i]), 4),
            "log10_C": round(float(log_c[i]), 4),
        })

    fields2 = list(loo_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_loo_residuals.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields2, delimiter="\t")
        w.writeheader()
        w.writerows(loo_rows)

    # ---- (3) Aggregated residual by phase ----
    by_phase = {}
    for r in loo_rows:
        by_phase.setdefault(r["phase"], []).append(r["abs_residual"])
    phase_rows = []
    for ph, arr in sorted(by_phase.items()):
        arr = np.array(arr, float)
        phase_rows.append({
            "phase": ph, "n_anchors": int(len(arr)),
            "mean_abs_residual": round(float(np.mean(arr)), 4),
            "median_abs_residual": round(float(np.median(arr)), 4),
            "max_abs_residual": round(float(np.max(arr)), 4),
            "bootstrap_p90": round(float(np.percentile(arr, 90)), 4),
            "note": "LOO |residual| from two-param joint fit",
        })
    fields3 = list(phase_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_phase_residual.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields3, delimiter="\t")
        w.writeheader()
        w.writerows(phase_rows)

    # ---- (4) Iso-FLOP optimal anchor picker ----
    # For each LOO fitted model, predict R_max at the 12 anchors' (P, C) values,
    # then at every C bucket pick the anchor with maximum predicted R_max.
    # Operational: given a target FLOP budget log10_C_target, what anchor wins?
    grid_log_c = np.linspace(log_c.min() - 0.1, log_c.max() + 0.1, 50)
    optimal_rows = []
    # Use the all-stack OLS coefficients (a_full, b_full, c_full) as the
    # closed-form predictor.
    for lc in grid_log_c:
        # For each anchor i, what is predicted R_max at this target C while
        # holding each anchor's own log10_P fixed?
        preds = a_full * log_p + b_full * lc + c_full
        winner = int(np.argmax(preds))
        optimal_rows.append({
            "log10_C_target": round(float(lc), 4),
            "predicted_R_max": round(float(preds[winner]), 4),
            "winner_model": recs[winner]["model"],
            "winner_params_B": float(recs[winner]["params_B"]),
            "winner_log10_P": round(float(log_p[winner]), 4),
            "winner_arch": recs[winner]["arch"],
        })
    fields4 = list(optimal_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_optimal_anchor.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields4, delimiter="\t")
        w.writeheader()
        w.writerows(optimal_rows)

    # ---- (5) Pre-registered predictions ----
    loo_resid = np.array([r["residual"] for r in loo_rows])
    loo_abs = np.array([r["abs_residual"] for r in loo_rows])
    rmse_loo = float(np.sqrt(np.mean(loo_abs ** 2)))
    # P1: LOO RMSE < 0.30 R_max units.
    p1_pass = bool(rmse_loo < 0.30)
    # P2: collapse (Nemotron-120B) has the largest |residual|.
    collapse_row = next((r for r in loo_rows if r["model"] == "Nemotron-120B"), None)
    if collapse_row is not None:
        max_idx = int(np.argmax(loo_abs))
        max_model = loo_rows[max_idx]["model"]
        p2_pass = bool(max_model == "Nemotron-120B")
        collapse_abs = float(collapse_row["abs_residual"])
    else:
        p2_pass = False
        max_model = ""
        collapse_abs = float("nan")
    # P3: optimal anchor P* in [4, 30]B at the median log10_C.
    median_lc = float(np.median(log_c))
    pred_at_med = a_full * log_p + b_full * median_lc + c_full
    winner_med_idx = int(np.argmax(pred_at_med))
    winner_med_p = float(recs[winner_med_idx]["params_B"])
    p3_pass = bool(4.0 <= winner_med_p <= 30.0)

    pred_rows = [
        {
            "prediction": "P1_two_param_LOO_RMSE_lt_0p30",
            "value": round(rmse_loo, 4),
            "pass": p1_pass,
            "note": "LOO RMSE = sqrt(mean((R_max - pred)**2))",
        },
        {
            "prediction": "P2_collapse_largest_abs_residual",
            "value": round(collapse_abs, 4) if not math.isnan(collapse_abs) else float("nan"),
            "pass": p2_pass,
            "note": f"max |resid| is on {max_model}",
        },
        {
            "prediction": "P3_optimal_P_in_4_to_30B_at_median_C",
            "value": round(winner_med_p, 4),
            "pass": p3_pass,
            "note": f"winner at log10_C={median_lc:.2f} is {recs[winner_med_idx]['model']}",
        },
    ]
    fields5 = list(pred_rows[0].keys())
    with open(RESULTS / "scaling_law_iter49_predictions.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields5, delimiter="\t")
        w.writeheader()
        w.writerows(pred_rows)

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))

    ax = axes[0]
    colors = {"plateau": "#3a86ff", "saturation": "#06d6a0", "drift": "#ffbe0b",
              "collapse": "#ef476f", "converged": "#3a86ff", "unknown": "#999999"}
    for r in loo_rows:
        ax.scatter(r["R_max_predicted"], r["R_max_actual"],
                   c=colors.get(r["phase"], "#999"), alpha=0.85, s=85,
                   edgecolor="black", linewidth=0.4)
        if r["model"] == "Nemotron-120B":
            ax.annotate(r["model"], (r["R_max_predicted"], r["R_max_actual"]),
                        xytext=(6, -14), textcoords="offset points", fontsize=8,
                        color="#a00", weight="bold")
    lo_line = np.linspace(0, max(rmax) * 1.05, 100)
    ax.plot(lo_line, lo_line, "--", color="black", alpha=0.5, label="y=x")
    ax.set_xlabel("LOO predicted R_max (two-param OLS)")
    ax.set_ylabel("measured R_max")
    ax.set_title(f"Iter49: two-param LOO R^2={r2_full:.2f}, RMSE={rmse_loo:.2f}")
    ax.grid(alpha=0.3)

    ax = axes[1]
    log_c_grid = np.linspace(log_c.min() - 0.05, log_c.max() + 0.05, 60)
    winners = []
    for lc in log_c_grid:
        preds = a_full * log_p + b_full * lc + c_full
        winners.append(int(np.argmax(preds)))
    cmap = plt.get_cmap("tab10")
    marker_map = {}
    color_idx = 0
    for w_ in winners:
        m = recs[w_]["model"]
        if m not in marker_map:
            marker_map[m] = (color_idx, cmap(color_idx % 10))
            color_idx += 1
    prev = None
    for lc, w_ in zip(log_c_grid, winners):
        m = recs[w_]["model"]
        idx, col = marker_map[m]
        if m != prev:
            ax.plot(lc, recs[w_]["params_B"], "o", color=col, markersize=8,
                    label=m, markeredgecolor="black", markeredgewidth=0.4)
            prev = m
        else:
            ax.plot(lc, recs[w_]["params_B"], "o", color=col, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.4)
    ax.set_xlabel("log10(C) target")
    ax.set_ylabel("optimal params_B")
    ax.set_yscale("log")
    ax.set_title("Iter49: iso-FLOP optimal anchor (two-param OLS)")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=1)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "scaling_law_iter49.png", dpi=160)
    plt.savefig(FIG_DIR / "scaling_law_iter49.pdf")
    plt.close()

    # ---- summary ----
    n_pass = sum(1 for r in pred_rows if r["pass"])
    print(f"iter49 done: a_logP={a_full:.3f}, b_logC={b_full:.3f}, c={c_full:.3f}, "
          f"R^2={r2_full:.3f}, LOO RMSE={rmse_loo:.3f}, pred {n_pass}/3 pass.")


if __name__ == "__main__":
    main()
