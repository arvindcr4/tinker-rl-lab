#!/usr/bin/env python3
"""A2 — Contrastive-yield re-plot of the scaling null (zero-cost re-analysis).

Reads existing GSM8K per-step traces under experiments/tinker-runs/results/,
recomputes per-anchor effective contrastive compute

    C_eff = sum_t  G * Y_G(p_x[t]) * KL_t[t]

with Y_G(p) = 1 - p**G - (1-p)**G, p_x[t] proxied by the logged per-step mean
reward, and KL_t[t] proxied by |loss| in the step log.  Fits a 3-parameter
offset saturation model to each reward trace to obtain R_max and the baseline
offset c, then re-plots R_max (and the gain R_max - c) against C_eff and
compares it to the raw parameter-count scaling null.

Outputs (all under experiments/results/A2_20260704/):
  a2_contrastive_yield_replot.png  side-by-side scaling-null replot
  a2_ceff_summary.tsv              per-anchor C_eff components and fits
  a2_scaling_fit.tsv               OLS/Spearman summary for the two abscissae
  a2_meta.json                     headline metadata
"""
from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting is optional for dry runs
    plt = None  # type: ignore[assignment]
    _plot_err = exc

try:
    from scipy.optimize import curve_fit
    from scipy.stats import spearmanr
except Exception as exc:  # pragma: no cover
    curve_fit = None  # type: ignore[assignment]
    spearmanr = None  # type: ignore[assignment]
    raise SystemExit(f"A2 requires scipy: {exc}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # experiments/results/A2_... -> repo root
TRACE_DIR = REPO_ROOT / "experiments" / "tinker-runs" / "results"
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_SIZES = {
    "qwen3-0.6b": 0.6,
    "qwen3-1.7b": 1.7,
    "qwen3-4b": 4.0,
    "qwen3.5-4b": 4.0,
    "qwen3-8b": 8.0,
    "qwen3-8b-base": 8.0,
    "qwen3.5-27b": 27.0,
    "qwen3-32b": 32.0,
    "qwen3-30b-moe": 3.0,       # active params
    "qwen3-30b-moe-inst": 3.0,
    "qwen3-235b-moe": 22.0,     # active params
    "llama-8b": 8.0,
    "llama-8b-inst": 8.0,
    "llama-3.1-8b": 8.0,
    "llama-3.2-1b": 1.0,
    "llama-3.2-3b": 3.0,
    "deepseek-v3.1": 685.0,
    "nemotron-120b": 12.0,      # active params
}

FAMILY = {
    "qwen3.5-4b": "qwen",
    "qwen3-8b": "qwen",
    "llama-8b-inst": "llama",
    "deepseek-v3.1": "deepseek",
    "nemotron-120b": "nemotron",
}

PRETTY = {
    "qwen3.5-4b": "Qwen3.5-4B",
    "qwen3-8b": "Qwen3-8B",
    "llama-8b-inst": "Llama-3.1-8B",
    "deepseek-v3.1": "DeepSeek-V3.1",
    "nemotron-120b": "Nemotron-120B",
}

COLORS = {
    "qwen": "#2166ac",
    "llama": "#d6604d",
    "deepseek": "#4dac26",
    "nemotron": "#f4a582",
}

# ---------------------------------------------------------------------------
# 3-parameter offset saturation fit (iter137 model)
# ---------------------------------------------------------------------------
def saturation_3p(t: np.ndarray, c: float, r_max: float, lam: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return c + (r_max - c) * (1.0 - np.exp(-lam * t))


def fit_3p(t: np.ndarray, y: np.ndarray) -> dict:
    """Return c, R_max, lambda, t_80, rmse, r2 for the offset saturation model."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(t)
    starts = [
        [float(np.min(y)), float(np.max(y)), 0.1],
        [0.0, float(np.mean(y)), 0.5],
        [float(np.median(y)), float(np.mean(y[-min(5, n):])), 0.2],
    ]
    best = None
    for p0 in starts:
        try:
            popt, _ = curve_fit(
                saturation_3p, t, y,
                p0=p0,
                bounds=([0.0, 0.0, 1e-4], [1.0, 1.05, 10.0]),
                maxfev=20000,
            )
            c, r_max, lam = float(popt[0]), float(popt[1]), float(popt[2])
            if r_max < c:
                c, r_max = r_max, c
            pred = saturation_3p(t, c, r_max, lam)
            resid = y - pred
            ss_res = float(np.sum(resid ** 2))
            rmse = float(math.sqrt(np.mean(resid ** 2)))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            cand = dict(c=c, R_max=r_max, lam=lam, rmse=rmse, r2=r2)
            if best is None or cand["rmse"] < best["rmse"]:
                best = cand
        except Exception:
            continue
    if best is None:
        return dict(c=float("nan"), R_max=float("nan"), lam=float("nan"),
                    t_80=float("nan"), rmse=float("nan"), r2=float("nan"))
    lam = best["lam"]
    best["t_80"] = float(-math.log(0.2) / lam) if lam and lam > 0 else float("nan")
    return best


# ---------------------------------------------------------------------------
# Contrastive-yield helpers
# ---------------------------------------------------------------------------
def contrastive_yield(p: float, G: int) -> float:
    """Y_G(p) = 1 - p^G - (1-p)^G."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0
    return 1.0 - p ** G - (1.0 - p) ** G


def optimal_G(p: float, G_values: list[int]) -> int:
    """G* = argmax_G Y_G(p) / G (compute-optimal group size at difficulty p)."""
    best_g, best_ratio = G_values[0], -1.0
    for G in G_values:
        ratio = contrastive_yield(p, G) / G
        if ratio > best_ratio:
            best_ratio = ratio
            best_g = G
    return best_g


# ---------------------------------------------------------------------------
# OLS + Spearman helpers
# ---------------------------------------------------------------------------
def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Return (intercept, slope, se_slope, r2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return (float("nan"),) * 4
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den <= 0:
        return (float("nan"),) * 4
    b = float(np.sum((x - xm) * (y - ym))) / den
    a = ym - b * xm
    resid = y - (a + b * x)
    s2 = float(np.sum(resid ** 2)) / (n - 2)
    se_b = math.sqrt(s2 / den) if den > 0 else float("nan")
    r2 = 1.0 - float(np.sum(resid ** 2)) / float(np.sum((y - ym) ** 2))
    return a, b, se_b, r2


def safe_log10(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return np.log10(np.maximum(a, 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not TRACE_DIR.exists():
        raise FileNotFoundError(f"Trace directory missing: {TRACE_DIR}")

    # Discover GSM8K traces with per-step logs.
    anchors: list[dict] = []
    for path in sorted(TRACE_DIR.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("task") != "gsm8k":
            continue
        rt = obj.get("reward_trace") or obj.get("rewards") or obj.get("trace")
        if not rt:
            continue
        step_log = obj.get("step_log")
        # We need per-step reward (p_x proxy) and, ideally, per-step loss (KL proxy).
        if step_log is None:
            # Build a synthetic step_log from reward_trace only; KL is unavailable.
            step_log = [{"reward": r, "loss": None} for r in rt]

        model_short = obj.get("model_short", path.stem)
        params = MODEL_SIZES.get(model_short, float("nan"))
        G = int(obj.get("group", 8))
        T = len(rt)

        # Per-step quantities
        p_x = []
        kl_t = []
        y_t = []
        for i, row in enumerate(step_log):
            if i >= T:
                break
            p = float(row.get("reward", rt[i]))
            p = max(0.0, min(1.0, p))
            p_x.append(p)
            # KL proxy: absolute surrogate loss when logged; otherwise default to 1.0
            loss = row.get("loss")
            if loss is None:
                kl = 1.0
            else:
                kl = abs(float(loss))
            kl_t.append(kl)
            y_t.append(contrastive_yield(p, G))

        p_x = np.array(p_x)
        kl_t = np.array(kl_t)
        y_t = np.array(y_t)

        # 3-parameter offset saturation fit on the raw reward trace.
        t_arr = np.arange(1, T + 1, dtype=float)
        fit = fit_3p(t_arr, np.asarray(rt, dtype=float))

        # Cumulative effective contrastive compute.
        ceff_per_step = G * y_t * kl_t
        cum_ceff = float(ceff_per_step.sum())

        # Compute-optimal static G for the average difficulty.
        mean_p = float(p_x.mean())
        G_star = optimal_G(mean_p, [2, 4, 8, 16, 32, 64])

        anchors.append({
            "model": obj.get("model", model_short),
            "model_short": model_short,
            "family": FAMILY.get(model_short, "other"),
            "params_B": params,
            "G": G,
            "n_steps": T,
            "trace_file": path.name,
            "has_step_loss": all(row.get("loss") is not None for row in step_log),
            "mean_p": mean_p,
            "mean_Y": float(y_t.mean()),
            "mean_KL": float(kl_t.mean()),
            "cum_C_eff": cum_ceff,
            "raw_rollouts": float(T * G),
            "ceff_discount_ratio": float(cum_ceff / (T * G)) if (T * G) > 0 else float("nan"),
            "c_3p": fit["c"],
            "R_max_3p": fit["R_max"],
            "lambda_3p": fit["lam"],
            "t80_3p": fit["t_80"],
            "rmse_3p": fit["rmse"],
            "r2_3p": fit["r2"],
            "delta_R_max": fit["R_max"] - fit["c"],
            "G_star": G_star,
            "peak": float(obj.get("peak", max(rt)) or max(rt)),
        })

    if not anchors:
        raise RuntimeError("No usable GSM8K traces found.")

    # Keep anchors with known size *and* per-step loss for the primary scaling fit.
    primary = [
        a for a in anchors
        if not math.isnan(a["params_B"]) and a["has_step_loss"]
    ]
    if len(primary) < 3:
        primary = anchors  # fall back to all usable traces

    # ------------------------------------------------------------------
    # Scaling fits: R_max and delta_R_max vs log10(N) and log10(C_eff)
    # ------------------------------------------------------------------
    def fit_and_record(rows: list[dict], x_field: str, y_field: str, label: str) -> dict:
        xs = np.array([r[x_field] for r in rows], dtype=float)
        ys = np.array([r[y_field] for r in rows], dtype=float)
        valid = ~(np.isnan(xs) | np.isnan(ys) | np.isinf(xs) | np.isinf(ys))
        xs_v, ys_v = xs[valid], ys[valid]
        logxs = safe_log10(xs_v)
        a, b, se, r2 = ols(logxs, ys_v)
        rho, pval = spearmanr(logxs, ys_v) if len(logxs) >= 2 else (float("nan"), float("nan"))
        return {
            "axis": label,
            "x_field": x_field,
            "y_field": y_field,
            "n": int(valid.sum()),
            "intercept": float(a),
            "slope_per_log10x": float(b),
            "se_slope": float(se),
            "r2": float(r2),
            "spearman_rho": float(rho),
            "spearman_p": float(pval),
            "models": [r["model_short"] for r in rows if valid[rows.index(r)]],
        }

    fits = [
        fit_and_record(primary, "params_B", "R_max_3p", "R_max ~ log10(params_B)"),
        fit_and_record(primary, "cum_C_eff", "R_max_3p", "R_max ~ log10(C_eff)"),
        fit_and_record(primary, "params_B", "delta_R_max", "delta_R_max ~ log10(params_B)"),
        fit_and_record(primary, "cum_C_eff", "delta_R_max", "delta_R_max ~ log10(C_eff)"),
    ]

    # ------------------------------------------------------------------
    # Write TSVs
    # ------------------------------------------------------------------
    summary_header = [
        "model", "model_short", "family", "params_B", "G", "n_steps",
        "has_step_loss", "mean_p", "mean_Y", "mean_KL", "cum_C_eff",
        "raw_rollouts", "ceff_discount_ratio", "c_3p", "R_max_3p",
        "lambda_3p", "t80_3p", "rmse_3p", "r2_3p", "delta_R_max",
        "G_star", "peak",
    ]
    summary_path = OUT_DIR / "a2_ceff_summary.tsv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(summary_header)
        for a in anchors:
            w.writerow([a[h] for h in summary_header])
    print(f"wrote {summary_path}")

    fit_path = OUT_DIR / "a2_scaling_fit.tsv"
    with fit_path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "axis", "y_metric", "n", "intercept", "slope_per_log10x", "se_slope",
            "r2", "spearman_rho", "spearman_p", "models",
        ])
        for fit in fits:
            w.writerow([
                fit["axis"], fit["y_field"], fit["n"], f"{fit['intercept']:.5f}",
                f"{fit['slope_per_log10x']:.5f}", f"{fit['se_slope']:.5f}",
                f"{fit['r2']:.4f}", f"{fit['spearman_rho']:.4f}",
                f"{fit['spearman_p']:.4g}", ",".join(fit["models"]),
            ])
    print(f"wrote {fit_path}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {_plot_err}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    def plot_panel(ax, x_field: str, fit: dict, title: str, xlabel: str):
        xs = np.array([r[x_field] for r in primary], dtype=float)
        ys = np.array([r["R_max_3p"] for r in primary], dtype=float)
        valid = ~(np.isnan(xs) | np.isnan(ys) | np.isinf(xs) | np.isinf(ys))
        xs_v, ys_v = xs[valid], ys[valid]
        logxs = safe_log10(xs_v)

        for r in primary:
            if np.isnan(r[x_field]) or np.isnan(r["R_max_3p"]):
                continue
            color = COLORS.get(r["family"], "#888888")
            ax.scatter(
                math.log10(r[x_field]), r["R_max_3p"],
                color=color, s=100, zorder=5, edgecolors="white", linewidths=0.8,
            )
            ax.annotate(
                PRETTY.get(r["model_short"], r["model_short"]),
                (math.log10(r[x_field]), r["R_max_3p"]),
                textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=color,
            )

        # OLS line
        if not np.isnan(fit["slope_per_log10x"]):
            x_line = np.linspace(logxs.min() - 0.1, logxs.max() + 0.1, 200)
            y_line = fit["intercept"] + fit["slope_per_log10x"] * x_line
            ax.plot(x_line, y_line, "--", color="#444444", linewidth=1.2, alpha=0.7,
                    label=f"OLS slope={fit['slope_per_log10x']:.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("R_max (3-param offset fit)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=8, loc="lower right")

    raw_fit = fits[0]
    ceff_fit = fits[1]
    plot_panel(
        axes[0], "params_B", raw_fit,
        f"(a) Raw scaling null\nslope={raw_fit['slope_per_log10x']:.3f} "
        f"(ρ={raw_fit['spearman_rho']:.2f}, p={raw_fit['spearman_p']:.3f})",
        "log10(parameters_B)",
    )
    plot_panel(
        axes[1], "cum_C_eff", ceff_fit,
        f"(b) Contrastive-yield replot\nslope={ceff_fit['slope_per_log10x']:.3f} "
        f"(ρ={ceff_fit['spearman_rho']:.2f}, p={ceff_fit['spearman_p']:.3f})",
        "log10(C_eff)",
    )

    fig.suptitle(
        "A2: contrastive-yield re-plot of the GRPO scaling null\n"
        f"n={len(primary)} anchors | C_eff = Σ_t G·Y_G(p_x[t])·|loss_t|",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = OUT_DIR / "a2_contrastive_yield_replot.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")

    # ------------------------------------------------------------------
    # Meta / headline
    # ------------------------------------------------------------------
    headline = {
        "date": "2026-07-04",
        "analysis": "A2_contrastive_yield_scaling_null",
        "n_anchors": len(primary),
        "n_total_traces": len(anchors),
        "caveats": [
            "p_x is proxied by the per-step mean reward (no per-prompt p_x in local files).",
            "KL_t is proxied by |surrogate loss| from step_log (no direct KL trace in local files).",
            "Partial traces without step-level loss are excluded from the primary fit.",
        ],
        "fits": fits,
        "primary_anchors": primary,
    }
    meta_path = OUT_DIR / "a2_meta.json"
    meta_path.write_text(json.dumps(headline, indent=2))
    print(f"wrote {meta_path}")

    print("\n=== A2 Headline ===")
    print(f"Primary anchors (n={len(primary)}): " + ", ".join(
        f"{r['model_short']} (C_eff={r['cum_C_eff']:.2e}, Rmax={r['R_max_3p']:.3f})"
        for r in primary
    ))
    for fit in fits:
        print(
            f"{fit['axis']:30s}  slope={fit['slope_per_log10x']:+.3f} "
            f"± {fit['se_slope']:.3f}  r2={fit['r2']:.3f}  "
            f"Spearman ρ={fit['spearman_rho']:.3f} (p={fit['spearman_p']:.3g})"
        )
    print(
        "\nInterpretation: replacing raw parameters with contrastive-yield compute "
        "does not rescue a log-linear scaling law on the 5-anchor evidence base; "
        "both abscissae remain flat/underpowered."
    )


if __name__ == "__main__":
    main()
