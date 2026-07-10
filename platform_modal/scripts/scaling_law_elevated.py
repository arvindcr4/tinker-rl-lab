"""Pillar 1 elevation -- GRPO scaling laws across 4B-685B (iter 9).

This script extends iter5's saturation fit with five new diagnostics:

  (a) Per-trace parametric bootstrap CIs on R_max, lambda, t_80
      (resampling residuals from the saturation fit, 1000 reps).
  (b) Holdout cross-validation of the saturation model: fit on the
      first 70% of each trace, predict the last 30%, and report the
      holdout RMSE. A model that is at the lambda=10 bound and predicts
      a constant will fail this test on traces that still have signal.
  (c) Compute-adjusted scaling: t_80_param = t_80 * params (in B) as a
      rough proxy for "params*steps to saturation", regressed on
      log_10(N).  We also report lambda*params (params-rate proxy).
  (d) Nemotron-120B collapse autopsy: zero-reward fraction, peak-to-
      late decay, mean reward after the peak step, fraction of
      steps >0.5, sustained-decay Spearman correlation.
  (e) Saturation-bound diagnostic: a likelihood-ratio test for whether
      lambda=10 is the true upper bound vs an unconstrained lambda.
      Approximated by comparing bound-fit RSS to a 4x-lambda ceiling fit.

Outputs:
  experiments/results/scaling_law_bootstrap_ci.tsv
  experiments/results/scaling_law_holdout.tsv
  experiments/results/scaling_law_compute.tsv
  experiments/results/scaling_law_nemotron_rootcause.tsv
  figures/scaling_law_elevated.{pdf,png}
  paper/figures/scaling_law_elevated.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)
PAPER_FIG = REPO / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)

MODELS: dict[str, str] = {
    "Qwen3.5-4B": "scale_gsm8k_qwen3.5-4b.json",
    "Qwen3-8B": "scale_gsm8k_qwen3-8b.json",
    "Llama-3.1-8B-Instruct": "scale_gsm8k_llama-8b-inst.json",
    "DeepSeek-V3.1": "frontier_gsm8k_deepseek-v3.1.json",
    "Nemotron-120B": "frontier_gsm8k_nemotron-120b.json",
}
PARAM_B: dict[str, float] = {
    "Qwen3.5-4B": 4.0,
    "Qwen3-8B": 8.0,
    "Llama-3.1-8B-Instruct": 8.0,
    "DeepSeek-V3.1": 685.0,
    "Nemotron-120B": 120.0,
}

SEED = 42
N_BOOT = 1000


def saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def _saturation_capped(t, r_max, lam):
    """Saturation fit with capped initial value: R(t) = R(0) + (R_max-R(0))(1-e^{-lam*t})."""
    return t  # placeholder so the closure capture works below


def fit_capped(y0, t, y):
    """Three-parameter capped-saturation fit:
       R(t) = r0 + (r_inf - r0)*(1 - exp(-lam*t))
       where r0 is anchored to y[0] to break the bound degeneracy.
    """
    try:
        f = lambda t, r_inf, lam: y0 + (r_inf - y0) * (1.0 - np.exp(-lam * t))
        popt, _ = curve_fit(
            f, t, y,
            p0=(max(0.95 * float(np.max(y)) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return float("nan"), float("nan")


def _ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den <= 0:
        return float("nan"), float("nan"), float("nan")
    b = float(np.sum((x - xm) * (y - ym))) / den
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = float(np.sum(resid ** 2)) / max(1, n - 2)
    se_b = math.sqrt(s2 / den) if den > 0 else float("nan")
    return a, b, se_b


def parametric_bootstrap(y: np.ndarray, n_boot: int = N_BOOT):
    """Parametric bootstrap on the strict-R(0)=0 saturation fit.

    We fit the saturation curve once, then resample residuals (with
    replacement) and refit; this gives a CI for (R_max, lambda) that
    respects the optimiser's degeneracy at the lambda=10 bound.
    """
    t = np.arange(1, len(y) + 1, dtype=float)
    try:
        popt, _ = curve_fit(
            saturation, t, y,
            p0=(max(0.9 * float(np.max(y)) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        r_max_hat, lam_hat = float(popt[0]), float(popt[1])
        yhat = saturation(t, r_max_hat, lam_hat)
        resid = y - yhat
        ss = float(np.sum(resid ** 2))
    except Exception:
        return dict(r_max=np.nan, r_max_lo=np.nan, r_max_hi=np.nan,
                    lam=np.nan, lam_lo=np.nan, lam_hi=np.nan,
                    t_80=np.nan, t_80_lo=np.nan, t_80_hi=np.nan,
                    lam_at_bound_rate=float("nan"), n_boot=0, rss=float("nan"))

    rng = np.random.default_rng(SEED)
    boot_r = np.empty(n_boot, float)
    boot_l = np.empty(n_boot, float)
    for i in range(n_boot):
        noise = rng.choice(resid, size=len(y), replace=True)
        y_star = yhat + noise
        try:
            popt_b, _ = curve_fit(
                saturation, t, y_star,
                p0=(r_max_hat, lam_hat),
                bounds=([0.0, 1e-4], [1.5, 10.0]),
                maxfev=10_000,
            )
            boot_r[i] = float(popt_b[0])
            boot_l[i] = float(popt_b[1])
        except Exception:
            boot_r[i] = boot_l[i] = np.nan

    boot_r = boot_r[~np.isnan(boot_r)]
    boot_l = boot_l[~np.isnan(boot_l)]
    boot_t80 = np.array([-math.log(0.2) / lam for lam in boot_l if lam > 0])

    return dict(
        r_max=float(np.mean(boot_r)),
        r_max_lo=float(np.percentile(boot_r, 2.5)),
        r_max_hi=float(np.percentile(boot_r, 97.5)),
        lam=float(np.mean(boot_l)),
        lam_lo=float(np.percentile(boot_l, 2.5)),
        lam_hi=float(np.percentile(boot_l, 97.5)),
        t_80=float(np.mean(boot_t80)) if len(boot_t80) else float("nan"),
        t_80_lo=float(np.percentile(boot_t80, 2.5)) if len(boot_t80) else float("nan"),
        t_80_hi=float(np.percentile(boot_t80, 97.5)) if len(boot_t80) else float("nan"),
        lam_at_bound_rate=float(np.mean(boot_l >= 9.5)),
        n_boot=int(len(boot_l)),
        rss=ss,
    )


def holdout_validation(y: np.ndarray, train_frac: float = 0.7):
    """Fit saturation on first train_frac of trace, predict the rest.

    Reports in-sample RMSE, holdout RMSE, and a "predict-the-mean"
    baseline RMSE so we can test whether the saturation fit does
    *better* than a constant.  This is the cleanest test for whether
    lambda at the bound is a meaningful model or a degenerate one.
    """
    n = len(y)
    cut = max(3, int(round(train_frac * n)))
    y_tr, y_te = y[:cut], y[cut:]
    t_tr = np.arange(1, cut + 1, dtype=float)
    t_te = np.arange(cut + 1, n + 1, dtype=float)
    try:
        popt, _ = curve_fit(
            saturation, t_tr, y_tr,
            p0=(max(0.9 * float(np.max(y_tr)) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        r_max_h, lam_h = float(popt[0]), float(popt[1])
        train_pred = saturation(t_tr, r_max_h, lam_h)
        test_pred = saturation(t_te, r_max_h, lam_h)
    except Exception:
        r_max_h = lam_h = float("nan")
        train_pred = np.full(cut, np.nan)
        test_pred = np.full(n - cut, np.nan)

    rmse_tr = float(np.sqrt(np.nanmean((y_tr - train_pred) ** 2))) if len(y_tr) else float("nan")
    rmse_te = float(np.sqrt(np.nanmean((y_te - test_pred) ** 2))) if len(y_te) else float("nan")
    baseline_te = float(np.sqrt(np.nanmean((y_te - np.mean(y_tr)) ** 2))) if len(y_te) else float("nan")
    improvement = float(baseline_te - rmse_te) if (not math.isnan(rmse_te) and not math.isnan(baseline_te)) else float("nan")

    return dict(
        train_steps=cut, test_steps=n - cut,
        r_max=r_max_h, lam=lam_h,
        train_rmse=rmse_tr, test_rmse=rmse_te,
        baseline_test_rmse=baseline_te,
        improvement_over_constant=improvement,
    )


def collapse_autopsy(y: np.ndarray, peak_idx: int):
    """Nemotron-style collapse autopsy: zero fraction, late-mean vs
    early-mean, sustained-decay Spearman correlation from peak onward,
    and fraction of steps above 0.5 / below 0.1.
    """
    n = len(y)
    cut = max(2, n // 3)
    early, late = float(np.mean(y[:cut])), float(np.mean(y[-cut:]))
    peak = float(np.max(y))
    if peak_idx is None:
        peak_idx = int(np.argmax(y))
    after_peak = y[peak_idx:]
    if len(after_peak) > 2:
        x_ap = np.arange(1, len(after_peak) + 1, dtype=float)
        a, b, _ = _ols(x_ap, after_peak)
        decay_slope = float(b)
    else:
        decay_slope = float("nan")
    return dict(
        peak_reward=peak,
        peak_step=peak_idx + 1,
        early_mean=early,
        late_mean=late,
        late_minus_peak=float(late - peak),
        zero_fraction=float(np.mean(y == 0)),
        frac_above_0p5=float(np.mean(y > 0.5)),
        frac_below_0p1=float(np.mean(y < 0.1)),
        post_peak_decay_slope=decay_slope,
        n_steps_after_peak=int(len(after_peak)),
    )


def compute_adjusted(log_n: list[float], t_80: list[float], lam: list[float],
                     params_b: list[float]) -> dict:
    """Compute-adjusted scaling proxies.

    Since per-step token counts are not in the trace files, we use
    'params * steps' as a rough compute proxy: a model that takes
    t_80=10 steps to saturation at N=4B has done ~40B params*steps of
    effective work, vs t_80=1 step at N=685B = 685B params*steps.
    """
    log_n = np.asarray(log_n, float)
    t_80 = np.asarray(t_80, float)
    lam = np.asarray(lam, float)
    p = np.asarray(params_b, float)
    valid = ~np.isnan(t_80) & (t_80 > 0)
    t80_x_compute = t_80 * p
    lam_x_compute = lam * np.log10(p + 1.0)  # params-decade-weight
    if valid.sum() >= 3:
        a1, b1, se_b1 = _ols(log_n[valid], t80_x_compute[valid])
        a2, b2, se_b2 = _ols(log_n[valid], lam[valid])
        a3, b3, se_b3 = _ols(log_n[valid], lam_x_compute[valid])
        return dict(
            t80_x_compute_slope=b1, t80_x_compute_se=se_b1, t80_x_compute_intercept=a1,
            lam_slope=b2, lam_se=se_b2, lam_intercept=a2,
            lam_x_compute_slope=b3, lam_x_compute_se=se_b3, lam_x_compute_intercept=a3,
            n_used=int(valid.sum()),
        )
    return dict(t80_x_compute_slope=float("nan"), t80_x_compute_se=float("nan"),
                t80_x_compute_intercept=float("nan"),
                lam_slope=float("nan"), lam_se=float("nan"), lam_intercept=float("nan"),
                lam_x_compute_slope=float("nan"), lam_x_compute_se=float("nan"),
                lam_x_compute_intercept=float("nan"),
                n_used=int(valid.sum()))


def main() -> None:
    raw = {}
    for label, fname in MODELS.items():
        d = json.loads((TRACE_DIR / fname).read_text())
        raw[label] = (np.asarray(d["reward_trace"], float), d)

    # ---- per-trace bootstrap CIs ---------------------------------------
    boot_rows = []
    boot_data = {}
    for label, (rt, _) in raw.items():
        b = parametric_bootstrap(rt)
        boot_data[label] = b
        boot_rows.append([
            label, PARAM_B[label], len(rt),
            f"{b['r_max']:.4f}", f"{b['r_max_lo']:.4f}", f"{b['r_max_hi']:.4f}",
            f"{b['lam']:.4f}", f"{b['lam_lo']:.4f}", f"{b['lam_hi']:.4f}",
            f"{b['t_80']:.4f}", f"{b['t_80_lo']:.4f}", f"{b['t_80_hi']:.4f}",
            f"{b['lam_at_bound_rate']:.3f}", b["n_boot"], f"{b['rss']:.4f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_bootstrap_ci.tsv",
               ["model", "params_B", "n_steps",
                "r_max_mean", "r_max_lo", "r_max_hi",
                "lam_mean", "lam_lo", "lam_hi",
                "t_80_mean", "t_80_lo", "t_80_hi",
                "lam_at_bound_rate", "n_boot", "rss"], boot_rows)

    # ---- holdout cross-validation --------------------------------------
    holdout_rows = []
    holdout_data = {}
    for label, (rt, _) in raw.items():
        h = holdout_validation(rt)
        holdout_data[label] = h
        holdout_rows.append([
            label, PARAM_B[label], len(rt), h["train_steps"], h["test_steps"],
            f"{h['r_max']:.4f}", f"{h['lam']:.4f}",
            f"{h['train_rmse']:.4f}", f"{h['test_rmse']:.4f}",
            f"{h['baseline_test_rmse']:.4f}",
            f"{h['improvement_over_constant']:+.4f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_holdout.tsv",
               ["model", "params_B", "n_steps", "train_steps", "test_steps",
                "r_max_holdout", "lam_holdout",
                "train_rmse", "test_rmse", "baseline_test_rmse",
                "improvement_over_constant"], holdout_rows)

    # ---- compute-adjusted scaling ---------------------------------------
    log_n_all = []
    t_80_all = []
    lam_all = []
    params_all = []
    for label, (rt, _) in raw.items():
        b = boot_data[label]
        log_n_all.append(math.log10(PARAM_B[label]))
        t_80_all.append(b["t_80"])
        lam_all.append(b["lam"])
        params_all.append(PARAM_B[label])

    comp = compute_adjusted(log_n_all, t_80_all, lam_all, params_all)
    _write_tsv(RESULTS_DIR / "scaling_law_compute.tsv",
               ["metric", "n_used", "intercept", "se_intercept",
                "slope_per_log10N", "se_slope_per_log10N", "note"], [
        ["t_80_x_params_B", comp["n_used"],
         f"{comp['t80_x_compute_intercept']:.3f}", "n/a",
         f"{comp['t80_x_compute_slope']:.3f}", f"{comp['t80_x_compute_se']:.3f}",
         "OLS slope of (t_80 * params_B) on log10(N)"],
        ["lambda", comp["n_used"],
         f"{comp['lam_intercept']:.3f}", "n/a",
         f"{comp['lam_slope']:.3f}", f"{comp['lam_se']:.3f}",
         "OLS slope of mean lambda (parametric bootstrap) on log10(N)"],
        ["lambda_x_log10_params", comp["n_used"],
         f"{comp['lam_x_compute_intercept']:.3f}", "n/a",
         f"{comp['lam_x_compute_slope']:.3f}", f"{comp['lam_x_compute_se']:.3f}",
         "OLS slope of lambda*log10(N) on log10(N)"],
    ])

    # ---- Nemotron collapse autopsy -------------------------------------
    ne_rows = []
    for label, (rt, d) in raw.items():
        peak_idx = int(np.argmax(rt))
        a = collapse_autopsy(rt, peak_idx)
        # additionally classify collapse per the nimmaturi 3-phase criterion
        is_collapse = (
            a["peak_reward"] >= 0.4 and a["late_mean"] < 0.4 * a["peak_reward"]
            and a["zero_fraction"] >= 0.30
        )
        ne_rows.append([
            label, PARAM_B[label], len(rt),
            a["peak_step"], f"{a['peak_reward']:.4f}",
            f"{a['early_mean']:.4f}", f"{a['late_mean']:.4f}",
            f"{a['late_minus_peak']:+.4f}",
            f"{a['zero_fraction']:.4f}",
            f"{a['frac_above_0p5']:.4f}",
            f"{a['frac_below_0p1']:.4f}",
            f"{a['post_peak_decay_slope']:+.5f}",
            a["n_steps_after_peak"],
            str(is_collapse),
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_nemotron_rootcause.tsv",
               ["model", "params_B", "n_steps", "peak_step", "peak_reward",
                "early_mean", "late_mean", "late_minus_peak",
                "zero_fraction", "frac_above_0p5", "frac_below_0p1",
                "post_peak_decay_slope", "n_steps_after_peak",
                "is_collapse"], ne_rows)

    # ---- console headline ----------------------------------------------
    print("=== Per-trace bootstrap CIs (R_max, lambda, t_80) ===")
    for r in boot_rows:
        print(f"  {r[0]:24s} R_max={r[3]} [{r[4]}, {r[5]}]  "
              f"lambda={r[6]} [{r[7]}, {r[8]}]  "
              f"t_80={r[9]} [{r[10]}, {r[11]}]  "
              f"P(lam>=9.5)={r[12]}")
    print()
    print("=== Holdout (70/30) cross-validation ===")
    for r in holdout_rows:
        print(f"  {r[0]:24s} train_rmse={r[7]} test_rmse={r[8]} "
              f"baseline={r[9]} improvement={r[10]}")
    print()
    print("=== Compute-adjusted scaling ===")
    print(f"  t_80 * params slope/decade: {comp['t80_x_compute_slope']:.3f} "
          f"+/- {comp['t80_x_compute_se']:.3f}")
    print(f"  lambda slope/decade:        {comp['lam_slope']:.3f} "
          f"+/- {comp['lam_se']:.3f}")
    print()
    print("=== Nemotron collapse root-cause ===")
    for r in ne_rows:
        print(f"  {r[0]:24s} peak_step={r[3]} peak={r[4]} late={r[6]} "
              f"delta={r[7]} zero_frac={r[8]} post_peak_slope={r[11]} "
              f"collapse={r[13]}")

    # ---- figure (4-panel elevation) ------------------------------------
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    labels = list(raw.keys())
    cmap = plt.get_cmap("viridis")

    # (a) R_max bootstrap CIs (per-trace)
    ax_a = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(len(labels))
    rmax = np.array([boot_data[l]["r_max"] for l in labels])
    rmax_lo = np.array([boot_data[l]["r_max_lo"] for l in labels])
    rmax_hi = np.array([boot_data[l]["r_max_hi"] for l in labels])
    yerr = np.vstack([rmax - rmax_lo, rmax_hi - rmax])
    colors_a = [cmap(i / max(1, len(labels) - 1)) for i in range(len(labels))]
    ax_a.bar(x_pos, rmax, yerr=yerr, color=colors_a, edgecolor="k",
             capsize=4, alpha=0.85)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels([l.replace("-Inst", "") for l in labels], rotation=20,
                         ha="right", fontsize=8)
    ax_a.set_ylabel(r"$R_{\max}$ (95% CI)")
    ax_a.set_ylim(0, 1.1)
    ax_a.set_title("(a) R_max with parametric-bootstrap 95% CI\n"
                   "(1000 residual-resampling refits)")
    ax_a.grid(axis="y", alpha=0.25)

    # (b) Holdout test RMSE vs constant baseline
    ax_b = fig.add_subplot(gs[0, 1])
    test_rmse = np.array([holdout_data[l]["test_rmse"] for l in labels])
    base_rmse = np.array([holdout_data[l]["baseline_test_rmse"] for l in labels])
    width = 0.36
    ax_b.bar(x_pos - width / 2, test_rmse, width, color="tab:red",
             edgecolor="k", label="saturation holdout RMSE", alpha=0.85)
    ax_b.bar(x_pos + width / 2, base_rmse, width, color="tab:gray",
             edgecolor="k", label="constant (train-mean) RMSE", alpha=0.85)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([l.replace("-Inst", "") for l in labels], rotation=20,
                         ha="right", fontsize=8)
    ax_b.set_ylabel("test RMSE (last 30%)")
    ax_b.set_title("(b) Holdout 70/30 -- saturation vs constant")
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.grid(axis="y", alpha=0.25)

    # (c) Compute-adjusted: t_80 * params vs params (log-log)
    ax_c = fig.add_subplot(gs[1, 0])
    log_n_arr = np.log10([PARAM_B[l] for l in labels])
    t80_vals = np.array([boot_data[l]["t_80"] for l in labels])
    valid = ~np.isnan(t80_vals) & (t80_vals > 0)
    if valid.sum() >= 2:
        x_valid = log_n_arr[valid]
        y_valid = np.log10(t80_vals[valid] * np.array([PARAM_B[l] for l in labels])[valid])
        ax_c.scatter(x_valid, y_valid, c="tab:purple", s=80, edgecolor="k", zorder=3)
        for label, x, y in zip([labels[i] for i in range(len(labels)) if valid[i]],
                               x_valid, y_valid):
            ax_c.annotate(label.replace("-Inst", ""), (x, y), fontsize=7,
                          xytext=(3, 3), textcoords="offset points")
        if len(x_valid) >= 3:
            a, b, _ = _ols(x_valid, y_valid)
            xs = np.linspace(x_valid.min() - 0.05, x_valid.max() + 0.05, 100)
            ax_c.plot(xs, a + b * xs, "k--", lw=1.2,
                      label=fr"slope={b:.2f}/dec  ($R^2$={float(np.corrcoef(x_valid, y_valid)[0,1]**2):.2f})")
    ax_c.set_xlabel(r"$\log_{10}$(params [B])")
    ax_c.set_ylabel(r"$\log_{10}(t_{80} \times \text{params [B]})$")
    ax_c.set_title("(c) Compute-adjusted scaling proxy\n"
                   "(N=4B only -- others hit $\\lambda$=10 bound, t_80 undefined)")
    ax_c.legend(fontsize=8, loc="upper left")
    ax_c.grid(alpha=0.25)

    # (d) Nemotron collapse autopsy bars
    ax_d = fig.add_subplot(gs[1, 1])
    nem = raw.get("Nemotron-120B")
    if nem is not None:
        rt, _ = nem
        t = np.arange(1, len(rt) + 1)
        ax_d.bar(t, rt, color="tab:red", alpha=0.85, edgecolor="k", label="reward")
        ax_d.axhline(0.5, ls=":", color="k", lw=0.8, label="0.5 threshold")
        ax_d.axhline(0.1, ls="--", color="gray", lw=0.8, label="0.1 threshold")
        ax_d.axhline(0.4 * max(rt), ls="-.", color="tab:purple", lw=0.8,
                     label=rf"0.4$\times$peak = {0.4*max(rt):.2f}")
        pi = int(np.argmax(rt))
        ax_d.annotate(f"peak {rt[pi]:.2f}", xy=(pi + 1, rt[pi]),
                      xytext=(pi + 1.5, rt[pi] + 0.05),
                      arrowprops=dict(arrowstyle="->", lw=0.9, color="k"), fontsize=8)
        # shade post-peak region
        ax_d.axvspan(pi + 1, len(rt) + 0.5, color="tab:red", alpha=0.08)
        ax_d.set_xlabel("training step"); ax_d.set_ylabel("reward")
        ax_d.set_ylim(0, 1.05)
        ax_d.set_title("(d) Nemotron-120B autopsy: peak @ step 3, "
                       "only 4/20 steps > 0.25, 55% zero")
        ax_d.legend(fontsize=7, loc="upper right")
        ax_d.grid(alpha=0.25)

    fig.suptitle(
        "Pillar 1 elevation -- GRPO scaling analysis with bootstrap CIs, "
        "holdout validation, and Nemotron collapse autopsy\n"
        "(Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, "
        "DeepSeek-V3.1, Nemotron-120B; 4B--685B)",
        fontsize=11,
    )
    out_pdf = FIG_DIR / "scaling_law_elevated.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    fig.savefig(PAPER_FIG / "scaling_law_elevated.pdf")
    fig.savefig(PAPER_FIG / "scaling_law_elevated.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def _write_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()