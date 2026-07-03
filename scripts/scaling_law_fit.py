"""Pillar 1 -- honest GRPO scaling analysis across 4B-685B.

Reports canonical saturation R(t)=R_max(1-e^{-lambda t}) as a baseline
(flagging when lambda hits the bound), per-trace OLS slopes, and a
cross-scale OLS regression metric ~ a + b*log10(N) with bootstrap CIs.
Classifies each run into the three-phase partition
(slow_start -> rapid_improvement -> plateau / collapse),
per nimmaturi2025predictive (arXiv:2507.18014).

Outputs:
  experiments/results/scaling_law_fits.tsv        (per-trace fits + stats)
  experiments/results/scaling_law_three_phase.tsv (phase partition)
  experiments/results/scaling_law_cross_scale.tsv (cross-scale regression)
  figures/scaling_law_fit.{pdf,png}
  paper/figures/scaling_law_fit.{pdf,png}
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
    "DeepSeek-V3.1": 685.0,  # HF sidebar total
    "Nemotron-120B": 120.0,
}

SEED = 42
N_BOOT = 5000


def saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def _ols(x, y):
    """OLS y = a + b*x; returns (a, b, se_b)."""
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
    s2 = float(np.sum(resid ** 2)) / (n - 2)
    se_b = math.sqrt(s2 / den) if den > 0 else float("nan")
    return a, b, se_b


def fit_per_trace(y: np.ndarray) -> dict:
    """Per-trace fits: canonical saturation (may be degenerate) +
    honest per-trace statistics + three-parameter growth diagnostic."""
    t = np.arange(1, len(y) + 1, dtype=float)
    y = y.astype(float)
    n = len(y)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    # canonical saturation
    try:
        popt, _ = curve_fit(
            saturation, t, y,
            p0=(max(0.9 * float(np.max(y)) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        r_max, lam = float(popt[0]), float(popt[1])
        yhat = saturation(t, r_max, lam)
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 0 else float("nan")
        lam_at_bound = lam >= 9.999
    except Exception:
        r_max = lam = rmse = r2 = float("nan")
        lam_at_bound = False

    # three-parameter growth-from-initial (diagnostic)
    try:
        popt3, _ = curve_fit(
            lambda t, r0, ri, lg: r0 + (ri - r0) * (1 - np.exp(-lg * t)),
            t, y, p0=(float(y[0]), float(np.max(y)), 0.3),
            bounds=([0.0, 0.0, 1e-4], [1.5, 1.5, 10.0]),
            maxfev=20_000,
        )
        r0_fit, rinf_fit, lam_g = map(float, popt3)
    except Exception:
        r0_fit = rinf_fit = lam_g = float("nan")

    # honest per-trace statistics
    cut = max(2, n // 3)
    early = float(np.mean(y[:cut]))
    late = float(np.mean(y[-cut:]))
    peak = float(np.max(y))
    trough = float(np.min(y))
    var = float(np.var(y))
    delta = late - early
    _, ols_b, ols_se = _ols(t, y)
    sign = "increase" if ols_b > 0 else ("flat" if abs(ols_b) < 1e-3 else "decrease")

    # three-phase partition (nimmaturi2025predictive: arXiv:2507.18014)
    if peak >= 0.4 and late < 0.4 * peak:
        phase = "collapse"
    elif early < 0.15 and late > 0.4 * peak:
        phase = "three_phase"
    elif abs(delta) <= 0.05:
        phase = "plateau"
    elif delta > 0.05:
        phase = "saturation"
    else:
        phase = "drift"

    t_80 = float(-math.log(0.2) / lam) if (lam and not math.isnan(lam) and lam > 0) else float("nan")

    return dict(
        n_steps=n, mean_reward=float(y.mean()), var_reward=var,
        peak=peak, trough=trough, early_mean=early, late_mean=late,
        delta_late_minus_early=delta,
        ols_slope_per_step=ols_b, ols_slope_se=ols_se, slope_direction=sign,
        R_max=r_max, lam=lam, t_80=t_80, rmse=rmse, r2=r2, lam_at_bound=lam_at_bound,
        r0=r0_fit, r_inf=rinf_fit, lambda_growth=lam_g, phase=phase,
    )


def bootstrap_slope(log_n, metric, n_boot=N_BOOT):
    """Block-bootstrap CI on the OLS slope metric ~ a + b*log10(N)."""
    log_n = np.asarray(log_n, float)
    metric = np.asarray(metric, float)
    n = len(log_n)
    rng = np.random.default_rng(SEED)
    bs = np.empty(n_boot, float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            _, b, _ = _ols(log_n[idx], metric[idx])
            bs[i] = b
        except Exception:
            bs[i] = np.nan
    bs = bs[~np.isnan(bs)]
    return dict(
        slope=float(np.mean(bs)),
        lo=float(np.percentile(bs, 2.5)),
        hi=float(np.percentile(bs, 97.5)),
        n=int(len(bs)),
    )


def load_traces():
    out = {}
    for label, fname in MODELS.items():
        d = json.loads((TRACE_DIR / fname).read_text())
        out[label] = (np.asarray(d["reward_trace"], float), d)
    return out


def _write_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def main() -> None:
    raw = load_traces()
    fits = {label: fit_per_trace(rt) for label, (rt, _) in raw.items()}

    # ---- per-trace TSV ---------------------------------------------------
    cols = [
        "model", "params_B", "n_steps", "mean_reward", "var_reward",
        "peak", "trough", "early_mean", "late_mean",
        "delta_late_minus_early", "ols_slope_per_step", "ols_slope_se",
        "slope_direction", "R_max", "lambda", "t_80", "rmse", "r2",
        "lam_at_bound", "r0", "r_inf", "lambda_growth", "phase", "trace_file",
    ]
    rows = []
    for label, f in fits.items():
        rows.append([
            label, PARAM_B[label], f["n_steps"],
            f"{f['mean_reward']:.4f}", f"{f['var_reward']:.4f}",
            f"{f['peak']:.4f}", f"{f['trough']:.4f}",
            f"{f['early_mean']:.4f}", f"{f['late_mean']:.4f}",
            f"{f['delta_late_minus_early']:.4f}",
            f"{f['ols_slope_per_step']:.5f}", f"{f['ols_slope_se']:.5f}",
            f["slope_direction"],
            f"{f['R_max']:.4f}", f"{f['lam']:.4f}", f"{f['t_80']:.4f}",
            f"{f['rmse']:.4f}", f"{f['r2']:.4f}", f["lam_at_bound"],
            f"{f['r0']:.4f}", f"{f['r_inf']:.4f}", f"{f['lambda_growth']:.4f}",
            f["phase"], MODELS[label],
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_fits.tsv", cols, rows)

    # ---- three-phase TSV -------------------------------------------------
    cols = ["model", "params_B", "phase", "early_mean", "late_mean",
            "peak", "delta_late_minus_early", "mean_reward", "var_reward",
            "ols_slope_per_step", "R_max", "lambda", "t_80",
            "r_inf", "lambda_growth"]
    rows = []
    for label, f in fits.items():
        rows.append([
            label, PARAM_B[label], f["phase"],
            f"{f['early_mean']:.4f}", f"{f['late_mean']:.4f}",
            f"{f['peak']:.4f}", f"{f['delta_late_minus_early']:.4f}",
            f"{f['mean_reward']:.4f}", f"{f['var_reward']:.4f}",
            f"{f['ols_slope_per_step']:.5f}",
            f"{f['R_max']:.4f}", f"{f['lam']:.4f}", f"{f['t_80']:.4f}",
            f"{f['r_inf']:.4f}", f"{f['lambda_growth']:.4f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_three_phase.tsv", cols, rows)

    # ---- cross-scale TSV (regression metric ~ a + b * log10(N)) ----------
    log_n = np.log10([PARAM_B[l] for l in fits])
    cross_cols = ["metric", "n_models", "intercept", "se_intercept",
                  "slope_per_log10N", "se_slope_per_log10N",
                  "boot_slope_mean", "boot_slope_lo", "boot_slope_hi",
                  "n_boot", "corr_logN_metric"]
    cross_rows = []
    for metric in ("mean_reward", "peak", "var_reward"):
        vals = np.array([fits[l][metric] for l in fits])
        a, b, se_b = _ols(log_n, vals)
        se_a = math.sqrt(float(np.sum((vals - (a + b * log_n)) ** 2)) / (len(vals) - 2)
                         * (1.0 / len(vals) + (log_n.mean()) ** 2 / float(np.sum((log_n - log_n.mean()) ** 2))))
        boot = bootstrap_slope(log_n, vals)
        r = float(np.corrcoef(log_n, vals)[0, 1])
        cross_rows.append([
            metric, len(vals), f"{a:.6f}", f"{se_a:.6f}",
            f"{b:.6f}", f"{se_b:.6f}",
            f"{boot['slope']:.6f}", f"{boot['lo']:.6f}", f"{boot['hi']:.6f}",
            boot["n"], f"{r:.6f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_cross_scale.tsv", cross_cols, cross_rows)

    # ---- print headline numbers for the autolog -------------------------
    for r in cross_rows:
        print(
            f"cross-scale metric={r[0]:>14s} slope/decade={float(r[4]):+.4f} "
            f"95% CI=[{float(r[7]):+.4f}, {float(r[8]):+.4f}] corr={float(r[10]):+.3f}"
        )
    for label, f in fits.items():
        print(
            f"  {label:24s} mean={f['mean_reward']:.3f} peak={f['peak']:.3f} "
            f"ols_slope={f['ols_slope_per_step']:+.4f} R_max={f['R_max']:.3f} "
            f"lam={f['lam']:.3f} {'(at bound)' if f['lam_at_bound'] else ''} phase={f['phase']}"
        )

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    cmap = plt.get_cmap("viridis")
    labels = list(fits.keys())

    # (a) raw traces
    ax_a = fig.add_subplot(gs[0, 0])
    for i, (label, (rt, _)) in enumerate(raw.items()):
        ax_a.plot(np.arange(1, len(rt) + 1), rt, "o",
                  color=cmap(i / max(1, len(raw) - 1)),
                  markersize=4, alpha=0.7,
                  label=f"{label} ({PARAM_B[label]:.0f}B)")
    ax_a.set_xlabel("training step"); ax_a.set_ylabel("reward")
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.set_title("(a) Raw reward traces -- 5 frontier-scale anchors")
    ax_a.grid(alpha=0.25); ax_a.legend(fontsize=7, loc="lower right", ncol=2)

    # (b) cross-scale scatter
    ax_b = fig.add_subplot(gs[0, 1])
    log_n_arr = np.log10([PARAM_B[l] for l in labels])
    means = np.array([fits[l]["mean_reward"] for l in labels])
    peaks = np.array([fits[l]["peak"] for l in labels])
    a_m, b_m, _ = _ols(log_n_arr, means)
    a_p, b_p, _ = _ols(log_n_arr, peaks)
    xs = np.linspace(log_n_arr.min() - 0.05, log_n_arr.max() + 0.05, 100)
    ax_b.scatter(log_n_arr, means, c="tab:blue", s=70, edgecolor="k",
                 label="mean R", zorder=3)
    ax_b.scatter(log_n_arr, peaks, c="tab:red", s=50, marker="^",
                 edgecolor="k", label="peak R", zorder=3)
    ax_b.plot(xs, a_m + b_m * xs, "b--", lw=1.5, label=fr"$\bar R$ slope={b_m:.3f}/dec")
    ax_b.plot(xs, a_p + b_p * xs, "r:", lw=1.5, label=fr"$\hat R$ slope={b_p:.3f}/dec")
    for label, x, y in zip(labels, log_n_arr, means):
        ax_b.annotate(label.replace("-Inst", ""), (x, y),
                      fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax_b.set_xlabel(r"$\log_{10}$(params [B])"); ax_b.set_ylabel("reward (0-1)")
    ax_b.set_ylim(0, 1.1); ax_b.grid(alpha=0.25)
    ax_b.set_title("(b) Cross-scale law: mean & peak reward vs params")
    ax_b.legend(fontsize=7, loc="upper right")

    # (c) phase partition bar chart
    ax_c = fig.add_subplot(gs[1, 0])
    pcol = {"plateau": "tab:gray", "three_phase": "tab:green",
            "saturation": "tab:blue", "drift": "tab:orange", "collapse": "tab:red"}
    heights = [fits[l]["mean_reward"] for l in labels]
    errs = [math.sqrt(fits[l]["var_reward"]) for l in labels]
    colors = [pcol[fits[l]["phase"]] for l in labels]
    ax_c.bar(labels, heights, yerr=errs, color=colors, edgecolor="k",
             alpha=0.9, capsize=4)
    ax_c.set_ylim(0, 1.1); ax_c.set_ylabel("mean reward (sd bar)")
    ax_c.set_title("(c) Three-phase partition (colour) with mean +/- sd bars")
    ax_c.tick_params(axis="x", rotation=20, labelsize=8)
    ax_c.set_xticklabels([l.replace("-Inst", "") for l in labels],
                         rotation=20, ha="right")
    from matplotlib.patches import Patch
    seen = sorted({fits[l]["phase"] for l in labels}, key=lambda p: list(pcol).index(p))
    handles = [Patch(facecolor=pcol[p], edgecolor="k", label=p) for p in seen]
    ax_c.legend(handles=handles, fontsize=7, loc="upper right",
                title="phase", title_fontsize=7)
    ax_c.grid(axis="y", alpha=0.25)

    # (d) Nemotron collapse zoom
    ax_d = fig.add_subplot(gs[1, 1])
    nem = raw.get("Nemotron-120B")
    if nem is not None:
        rt, _ = nem
        t = np.arange(1, len(rt) + 1)
        ax_d.bar(t, rt, color="tab:red", alpha=0.85, edgecolor="k")
        ax_d.axhline(rt.mean(), ls="--", color="k", lw=0.9,
                     label=f"mean={rt.mean():.3f}")
        pi = int(np.argmax(rt))
        ax_d.annotate(f"peak {rt[pi]:.2f} @ step {pi+1}",
                      xy=(pi + 1, rt[pi]), xytext=(pi + 1.5, rt[pi] + 0.05),
                      arrowprops=dict(arrowstyle="->", lw=0.9, color="k"), fontsize=8)
        ax_d.set_xlabel("training step"); ax_d.set_ylabel("reward")
        ax_d.set_ylim(0, 1.05); ax_d.set_title(
            "(d) Nemotron-120B collapse: peak 0.875 not retained")
        ax_d.legend(fontsize=7); ax_d.grid(alpha=0.25)

    fig.suptitle(
        "Pillar 1 -- GRPO scaling analysis across 4B-685B (Qwen3.5-4B, "
        "Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-V3.1, Nemotron-120B)",
        fontsize=12,
    )
    out_pdf = FIG_DIR / "scaling_law_fit.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    fig.savefig(PAPER_FIG / "scaling_law_fit.pdf")
    fig.savefig(PAPER_FIG / "scaling_law_fit.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
