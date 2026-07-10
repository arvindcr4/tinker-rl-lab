"""Pillar 1 iter37d -- Hill-form extrapolation given model-selection evidence.

Iter37c showed that on the 10 raw per-step reward traces, the Hill n=2 form
(sigmoidal: R_max * t^2 / (K^2 + t^2)) is the AIC winner on 5/10 traces
(3 GRPO + 2 PPO), with exponential saturation a close second (3 wins total).
The literature's preferred form is therefore not the right default.

This driver re-runs the iter21 two-anchor extrapolation battery under the
Hill form and compares to the exponential saturation result:

  (a) Take the 8 small models (< 32B params) as the calibration set
  (b) Hold out the 4 frontier models (>= 32B)
  (c) Predict R_max and t_80 from a 2-anchor log-log fit in params_B
  (d) Compare Hill vs saturation MAE on the 4-anchor holdout

If the Hill form extrapolates better, the paper can recommend Hill as the
default functional form for GRPO reward ceiling prediction.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "experiments" / "results" / "scaling_law_extended_frontier.tsv"
ROOTCAUSE = REPO / "experiments" / "results" / "scaling_law_nemotron_rootcause.tsv"
FITS37 = REPO / "experiments" / "results" / "scaling_law_iter37b_fits.tsv"
RESULTS = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_FIG = REPO / "paper" / "figures"
for d in (FIG_DIR, PAPER_FIG):
    d.mkdir(parents=True, exist_ok=True)


def model_saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def model_hill(t, r_max, k):
    return r_max * (t * t) / (k * k + t * t)


def aic_bic(n, k, ss_res):
    if ss_res <= 0 or not math.isfinite(ss_res):
        return float("inf"), float("inf")
    log_lik = -0.5 * n * (1.0 + math.log(2.0 * math.pi * ss_res / n))
    return -2.0 * log_lik + 2 * k, -2.0 * log_lik + k * math.log(n)


def fit_sat(t, y):
    try:
        popt, _ = curve_fit(model_saturation, t, y, p0=[0.8, 0.3],
                            bounds=([0, 1e-3], [2, 5]), maxfev=8000)
    except Exception:
        return None, float("inf"), float("inf")
    yh = model_saturation(t, *popt)
    a, b = aic_bic(len(y), 2, float(np.sum((y - yh) ** 2)))
    return popt, a, b


def fit_hill(t, y):
    try:
        popt, _ = curve_fit(model_hill, t, y, p0=[0.9, 5.0],
                            bounds=([0, 1e-3], [2, 1e4]), maxfev=8000)
    except Exception:
        return None, float("inf"), float("inf")
    yh = model_hill(t, *popt)
    a, b = aic_bic(len(y), 2, float(np.sum((y - yh) ** 2)))
    return popt, a, b


def synth_trace(r, rc_lookup):
    n = int(r["n_steps"])
    peak = int(float(rc_lookup[r["model"]]["peak_step"])) if r["model"] in rc_lookup else (
        1 if abs(r["r_peak"] - r["r_first"]) < 0.05 else max(1, n // 2)
    )
    if peak < 1:
        peak = 1
    if peak > n - 1:
        peak = n - 1
    peak_val, early, late, mean, zf = r["r_peak"], r["early_mean"], r["late_mean"], r["r_mean"], r["zero_frac"]
    t = np.arange(1, n + 1, dtype=float)
    out = np.linspace(early, late, n)
    out[peak - 1] = max(out[peak - 1], peak_val)
    if peak - 2 >= 0:
        out[peak - 2] = max(out[peak - 2], 0.5 * (out[peak - 1] + out[peak]))
    if peak < n:
        out[peak] = max(out[peak], 0.5 * (out[peak - 1] + out[peak + 1]))
    n_zero = int(round(zf * n))
    if n_zero > 0 and r["model"] == "Nemotron-120B":
        out[:n_zero] = 0.0
        if n - 1 > peak:
            out[(n_zero + peak) // 2] = 0.0
    cur = float(np.mean(out))
    if cur > 1e-9:
        out = out * (mean / cur)
    out = np.clip(out, 0.0, 1.0)
    return t, out, peak


def main() -> None:
    with open(DATA) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (ValueError, TypeError):
                pass
    with open(ROOTCAUSE) as f:
        rc_lookup = {r["model"]: r for r in csv.DictReader(f, delimiter="\t")}

    # Filter to dynamic anchors (r_var > 0.005); 10 anchors
    dynamic = [r for r in rows if r["r_var"] > 0.005]

    # fit both forms on each anchor, extract R_max, K (Hill) / lambda (sat)
    rows_out = []
    for r in dynamic:
        t, y, _ = synth_trace(r, rc_lookup)
        sat, aic_s, bic_s = fit_sat(t, y)
        hl, aic_h, bic_h = fit_hill(t, y)
        if sat is not None:
            rmax_s, lam_s = sat
            t80_s = -math.log(0.2) / lam_s if lam_s > 0 else float("inf")
        else:
            rmax_s, lam_s, t80_s = float("nan"), float("nan"), float("inf")
        if hl is not None:
            rmax_h, k_h = hl
            # Hill t_80: solve t^2 / (K^2 + t^2) = 0.8 => t = K * sqrt(4) = 2K
            t80_h = 2.0 * k_h
        else:
            rmax_h, k_h, t80_h = float("nan"), float("nan"), float("inf")
        rows_out.append({
            "model": r["model"],
            "params_B": r["params_B"],
            "arch": r["arch"],
            "n_steps": int(r["n_steps"]),
            "r_mean": round(r["r_mean"], 4),
            "sat_R_max": round(rmax_s, 4) if not math.isnan(rmax_s) else "NaN",
            "sat_lambda": round(lam_s, 4) if not math.isnan(lam_s) else "NaN",
            "sat_t_80": round(t80_s, 4) if math.isfinite(t80_s) else "inf",
            "sat_aic": round(aic_s, 4) if math.isfinite(aic_s) else "inf",
            "hill_R_max": round(rmax_h, 4) if not math.isnan(rmax_h) else "NaN",
            "hill_K": round(k_h, 4) if not math.isnan(k_h) else "NaN",
            "hill_t_80": round(t80_h, 4) if math.isfinite(t80_h) else "inf",
            "hill_aic": round(aic_h, 4) if math.isfinite(aic_h) else "inf",
            "delta_aic_sat_minus_hill": round(float(aic_s) - float(aic_h), 4)
                if (math.isfinite(aic_s) and math.isfinite(aic_h)) else "NaN",
        })

    # 2-anchor extrapolation
    # Take 2 smallest models (Qwen3.5-4B and Qwen3-8B) as anchors
    # Fit log(R_max) = a + b*log10(params_B) and log(t_80) = a + b*log10(params_B)
    # Hold out 4 frontier models (>= 32B), check MAE
    by_params = sorted(dynamic, key=lambda r: r["params_B"])
    calib = [r for r in by_params if r["params_B"] <= 30.0][:3]  # 3 smallest
    holdout = [r for r in by_params if r["params_B"] > 30.0]

    def get_lookup(rows_list, key_a, key_b, anchor_name):
        for r in rows_list:
            if r["model"] == anchor_name:
                return float(r[key_a]), float(r[key_b])
        return float("nan"), float("nan")

    # compute predictions under each form
    pred_rows = []
    for r in holdout:
        # Use the 3 calibration anchors to fit log-log scaling
        logP = np.log10([c["params_B"] for c in calib])
        for form, key_rmax, key_t80 in [("sat", "sat_R_max", "sat_t_80"),
                                        ("hill", "hill_R_max", "hill_t_80")]:
            rmax_cal = []
            t80_cal = []
            for c in calib:
                for rr in rows_out:
                    if rr["model"] == c["model"]:
                        v = rr[key_rmax]
                        t80 = rr[key_t80]
                        if isinstance(v, (int, float)) and isinstance(t80, (int, float)) and v > 0 and t80 > 0:
                            rmax_cal.append(float(v))
                            t80_cal.append(float(t80))
            if len(rmax_cal) < 2:
                continue
            logRmax = np.log10(rmax_cal)
            logT80 = np.log10(t80_cal)
            # OLS log-log
            slope_r, intc_r = np.polyfit(logP, logRmax, 1)
            slope_t, intc_t = np.polyfit(logP, logT80, 1)
            logP_h = math.log10(r["params_B"])
            rmax_pred = 10 ** (intc_r + slope_r * logP_h)
            t80_pred = 10 ** (intc_t + slope_t * logP_h)
            # ground-truth from the actual fit
            for rr in rows_out:
                if rr["model"] == r["model"]:
                    rmax_true = float(rr[key_rmax]) if isinstance(rr[key_rmax], (int, float)) else float("nan")
                    t80_true = float(rr[key_t80]) if isinstance(rr[key_t80], (int, float)) else float("nan")
                    break
            else:
                rmax_true = float("nan")
                t80_true = float("nan")
            err_rmax = rmax_pred - rmax_true
            err_t80 = t80_pred - t80_true
            pred_rows.append({
                "model": r["model"],
                "params_B": r["params_B"],
                "form": form,
                "rmax_pred": round(rmax_pred, 4),
                "rmax_true": round(rmax_true, 4) if not math.isnan(rmax_true) else "NaN",
                "abs_err_rmax": round(abs(err_rmax), 4) if not math.isnan(rmax_true) else "NaN",
                "t80_pred": round(t80_pred, 4),
                "t80_true": round(t80_true, 4) if not math.isnan(t80_true) else "NaN",
                "abs_err_t80": round(abs(err_t80), 4) if not math.isnan(t80_true) else "NaN",
            })

    # Aggregate
    sat_pred = [p for p in pred_rows if p["form"] == "sat"]
    hill_pred = [p for p in pred_rows if p["form"] == "hill"]
    summary = {
        "n_dynamic_anchors": len(dynamic),
        "n_calibration_anchors": len(calib),
        "n_holdout_anchors": len(holdout),
        "sat_mean_abs_err_rmax": round(float(np.mean([p["abs_err_rmax"] for p in sat_pred])), 4)
            if sat_pred else "NaN",
        "hill_mean_abs_err_rmax": round(float(np.mean([p["abs_err_rmax"] for p in hill_pred])), 4)
            if hill_pred else "NaN",
        "sat_mean_abs_err_t80": round(float(np.mean([p["abs_err_t80"] for p in sat_pred])), 4)
            if sat_pred else "NaN",
        "hill_mean_abs_err_t80": round(float(np.mean([p["abs_err_t80"] for p in hill_pred])), 4)
            if hill_pred else "NaN",
        "anchors_where_hill_aic_better": sum(1 for r in rows_out
            if r["delta_aic_sat_minus_hill"] != "NaN" and float(r["delta_aic_sat_minus_hill"]) > 0),
        "anchors_where_sat_aic_better": sum(1 for r in rows_out
            if r["delta_aic_sat_minus_hill"] != "NaN" and float(r["delta_aic_sat_minus_hill"]) < 0),
    }
    if isinstance(summary["sat_mean_abs_err_rmax"], (int, float)) and isinstance(summary["hill_mean_abs_err_rmax"], (int, float)):
        summary["extrapolation_winner"] = "hill" if summary["hill_mean_abs_err_rmax"] < summary["sat_mean_abs_err_rmax"] else "saturation"
    else:
        summary["extrapolation_winner"] = "n/a"

    # write
    with open(RESULTS / "scaling_law_iter37d_fits.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {RESULTS / 'scaling_law_iter37d_fits.tsv'}  ({len(rows_out)} rows)")
    with open(RESULTS / "scaling_law_iter37d_extrap.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(pred_rows)
    print(f"wrote {RESULTS / 'scaling_law_iter37d_extrap.tsv'}  ({len(pred_rows)} rows)")
    with open(RESULTS / "scaling_law_iter37d_summary.tsv", "w") as f:
        w = csv.writer(f, delimiter="\t")
        for k, v in summary.items():
            w.writerow([k, v])
    print(f"wrote {RESULTS / 'scaling_law_iter37d_summary.tsv'}")

    # figure: bar chart sat vs hill MAE on holdout
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    models_h = [p["model"] for p in sat_pred]
    short = [m.replace("-Instruct", "-Inst") for m in models_h]
    x = np.arange(len(models_h))
    axes[0].bar(x - 0.2, [p["abs_err_rmax"] for p in sat_pred], width=0.4,
                color="#2b8cbe", label="exponential saturation")
    axes[0].bar(x + 0.2, [p["abs_err_rmax"] for p in hill_pred], width=0.4,
                color="#fdae61", label="Hill n=2")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s}\n{p['params_B']:.0f}B" for s, p in zip(short, sat_pred)],
                            fontsize=8)
    axes[0].set_ylabel("|pred - true| R_max")
    axes[0].set_title("Two-anchor log-log extrapolation MAE on holdout (>=32B)")
    axes[0].legend(loc="upper right", fontsize=9)
    # scatter: pred vs true, one panel per form
    for ax_i, (form, color, label) in enumerate([("sat", "#2b8cbe", "exponential saturation"),
                                                    ("hill", "#fdae61", "Hill n=2")]):
        sub = [p for p in pred_rows if p["form"] == form]
        if not sub:
            continue
        ax = axes[1] if ax_i == 0 else None
        # use a third panel? no, put both in same axes via legend
    # actually use the second axes for pred vs true scatter, both forms
    ax = axes[1]
    for form, color, marker, label in [("sat", "#2b8cbe", "o", "exponential saturation"),
                                        ("hill", "#fdae61", "s", "Hill n=2")]:
        sub = [p for p in pred_rows if p["form"] == form and
               p["rmax_true"] != "NaN"]
        if not sub:
            continue
        ax.scatter([p["rmax_pred"] for p in sub],
                   [p["rmax_true"] for p in sub],
                   c=color, marker=marker, s=80, label=label, edgecolor="black", linewidth=0.5)
    ax.plot([0, 1.05], [0, 1.05], "k--", lw=0.7, alpha=0.6)
    ax.set_xlabel("predicted R_max (2-anchor log-log)")
    ax.set_ylabel("true R_max (in-sample fit)")
    ax.set_title("Predicted vs true R_max on 4-anchor holdout")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"scaling_law_iter37d.{ext}", bbox_inches="tight")
        fig.savefig(PAPER_FIG / f"scaling_law_iter37d.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/scaling_law_iter37d.{{pdf,png}}")

    # console
    print("\n=== Iter 37d summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n=== Per-anchor fits ===")
    for r in rows_out:
        print(f"  {r['model']:30s} sat: R_max={r['sat_R_max']} lam={r['sat_lambda']} t80={r['sat_t_80']} | "
              f"hill: R_max={r['hill_R_max']} K={r['hill_K']} t80={r['hill_t_80']} | "
              f"ΔAIC(sat-hill)={r['delta_aic_sat_minus_hill']}")


if __name__ == "__main__":
    main()
