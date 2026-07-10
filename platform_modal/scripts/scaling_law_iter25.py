"""Pillar 1 iter25 -- Identifiability & noise-floor audit of the GRPO
saturation law R(t)=R_max*(1-e^{-lambda t}).

Motivation: iter21 found lambda pinned at the upper bound in 58% of traces
(frac_lam_at_bound=0.5833) and t_80 collapses to the same 0.1609 for every
core fit. That is the fingerprint of a NON-IDENTIFIABLE model: the exponential
cannot separate a real growth transient from binomial sampling noise. This
driver quantifies, per trace, whether the saturation reading is real signal or
a fitting artifact, using three noise-aware tests:

 1. Effective batch n inferred from reward granularity (GCD of denominators);
    binomial noise floor sigma_step = sqrt(p(1-p)/n).
 2. Parametric bootstrap: simulate B synthetic traces under the fitted model
    + binomial(n) noise, refit, and report the sampling distribution of
    lambda and t_80 (fraction hitting the bound, CI width).
 3. Noise-aware model selection (AICc) over {constant, linear, saturation};
    a trace "supports saturation" only if saturation wins AND its bootstrap
    lambda CI excludes the bound.
 4. Detectability: minimal trace length T* to reject lambda=0 at power 0.8
    given the inferred noise floor; compare to the actual trace length.

Outputs:
  platform_hybrid/experiments/results/scaling_law_iter25_identifiability.tsv (per trace)
  platform_hybrid/experiments/results/scaling_law_iter25_bootstrap.tsv       (bootstrap CIs)
  platform_hybrid/experiments/results/scaling_law_iter25_modelsel.tsv        (AICc weights)
  platform_hybrid/experiments/results/scaling_law_iter25_summary.tsv         (headline rollup)
  figures/scaling_law_iter25.{pdf,png}  (+ mirror into paper/figures/)
"""
from __future__ import annotations

import csv
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS = REPO / "experiments" / "results"
FIG_DIR = REPO / "figures"
PAPER_FIG = REPO / "paper" / "figures"
for d in (FIG_DIR, PAPER_FIG):
    d.mkdir(parents=True, exist_ok=True)

# Core traces with enough steps (>=20) for an identifiability claim, plus the
# two frontier runs the pillar centres on.
MODELS = {
    "Qwen3.5-4B": ("scale_gsm8k_qwen3.5-4b.json", 4.0),
    "Qwen3-8B": ("scale_gsm8k_qwen3-8b.json", 8.0),
    "Llama-3.1-8B-Instruct": ("scale_gsm8k_llama-8b-inst.json", 8.0),
    "DeepSeek-V3.1": ("frontier_gsm8k_deepseek-v3.1.json", 685.0),
    "Nemotron-120B": ("frontier_gsm8k_nemotron-120b.json", 120.0),
}
LAM_LO, LAM_HI = 1e-4, 10.0
SEED = 42
N_BOOT = 2000
rng = np.random.default_rng(SEED)


def saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def load_trace(fname: str) -> np.ndarray:
    d = json.load(open(TRACE_DIR / fname))
    return np.asarray(d["reward_trace"], dtype=float)


def infer_batch_n(y: np.ndarray, cap: int = 64) -> int:
    """Infer effective binomial batch size from reward granularity: every
    reward is k/n, so n divides the LCM of the fraction denominators."""
    den = 1
    for v in y:
        f = Fraction(float(v)).limit_denominator(cap)
        den = den * f.denominator // math.gcd(den, f.denominator)
        if den > cap:
            return cap
    return max(2, min(den, cap))


def fit_sat(t, y):
    try:
        popt, _ = curve_fit(
            saturation, t, y,
            p0=(max(0.9 * float(np.max(y)) + 0.05, 0.05), 0.3),
            bounds=([0.0, LAM_LO], [1.5, LAM_HI]), maxfev=20000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return float("nan"), float("nan")


def aicc(rss, k, n):
    """Small-sample-corrected AIC under a Gaussian-likelihood approximation."""
    if n <= k + 1 or rss <= 0:
        rss = max(rss, 1e-9)
    ll = -0.5 * n * (math.log(2 * math.pi * rss / n) + 1.0)
    a = 2 * k - 2 * ll
    return a + (2 * k * (k + 1)) / max(n - k - 1, 1)


def model_selection(t, y):
    """AICc over constant / linear / saturation; return weights + best."""
    n = len(y)
    rss_const = float(np.sum((y - y.mean()) ** 2))
    b, a = np.polyfit(t, y, 1)
    rss_lin = float(np.sum((y - (a + b * t)) ** 2))
    rmax, lam = fit_sat(t, y)
    if math.isnan(lam):
        rss_sat = rss_const
    else:
        rss_sat = float(np.sum((y - saturation(t, rmax, lam)) ** 2))
    aiccs = {
        "constant": aicc(rss_const, 1, n),
        "linear": aicc(rss_lin, 2, n),
        "saturation": aicc(rss_sat, 2, n),
    }
    amin = min(aiccs.values())
    w = {k: math.exp(-0.5 * (v - amin)) for k, v in aiccs.items()}
    z = sum(w.values())
    w = {k: v / z for k, v in w.items()}
    best = min(aiccs, key=aiccs.get)
    return aiccs, w, best


def bootstrap_lambda(t, y, n_batch, rmax, lam):
    """Parametric bootstrap under fitted saturation + binomial(n_batch) noise.
    Returns arrays of refit lambda and t_80."""
    lams, t80s = [], []
    mu = np.clip(saturation(t, rmax, lam), 0.0, 1.0)
    for _ in range(N_BOOT):
        ysim = rng.binomial(n_batch, mu) / n_batch
        _, lb = fit_sat(t, ysim)
        if not math.isnan(lb):
            lams.append(lb)
            if lb > 0:
                t80s.append(-math.log(0.2) / lb)
    return np.asarray(lams), np.asarray(t80s)


def detect_T(rmax, lam, n_batch, power=0.8, alpha=0.05, tmax=200):
    """Minimal trace length T* to reject lambda=0 (flat null) at target power,
    via a parametric power simulation on the OLS slope."""
    if math.isnan(lam) or lam <= 0 or rmax <= 0:
        return float("nan")
    zc = 1.959963985  # two-sided alpha=0.05
    for T in range(3, tmax + 1):
        t = np.arange(1, T + 1, dtype=float)
        mu = np.clip(saturation(t, rmax, lam), 0.0, 1.0)
        # analytic OLS-slope power under heteroskedastic binomial noise
        xm = t.mean()
        sxx = float(np.sum((t - xm) ** 2))
        var_slope = float(np.sum((t - xm) ** 2 * (mu * (1 - mu) / n_batch))) / sxx ** 2
        b_true = float(np.sum((t - xm) * (mu - mu.mean())) / sxx)
        if var_slope <= 0:
            continue
        ncp = abs(b_true) / math.sqrt(var_slope)
        # power for two-sided z-test
        powr = 0.5 * (math.erfc((zc - ncp) / math.sqrt(2))) + \
            0.5 * (math.erfc((zc + ncp) / math.sqrt(2)))
        if powr >= power:
            return T
    return tmax + 1  # censored: not detectable within tmax steps


def main():
    per_rows, boot_rows, ms_rows = [], [], []
    n_identifiable = 0
    n_sat_supported = 0
    for name, (fname, pb) in MODELS.items():
        if not (TRACE_DIR / fname).exists():
            continue
        y = load_trace(fname)
        t = np.arange(1, len(y) + 1, dtype=float)
        n_batch = infer_batch_n(y)
        sigma = float(np.sqrt(np.mean(np.clip(y, 0, 1) * (1 - np.clip(y, 0, 1)) / n_batch)))
        rmax, lam = fit_sat(t, y)
        lam_at_bound = (not math.isnan(lam)) and lam >= 0.999 * LAM_HI
        t80_point = (-math.log(0.2) / lam) if (not math.isnan(lam) and lam > 0) else float("nan")

        lams, t80s = bootstrap_lambda(t, y, n_batch, rmax, lam)
        if len(lams):
            frac_bound = float(np.mean(lams >= 0.999 * LAM_HI))
            lam_lo, lam_hi = np.percentile(lams, [10, 90])
            # identifiable if bootstrap 80% CI does NOT span the full [lo,hi]
            span_frac = (lam_hi - lam_lo) / (LAM_HI - LAM_LO)
            identifiable = (frac_bound < 0.5) and (span_frac < 0.5)
        else:
            frac_bound, lam_lo, lam_hi, span_frac, identifiable = (
                float("nan"), float("nan"), float("nan"), float("nan"), False)
        if len(t80s):
            t80_lo, t80_hi = np.percentile(t80s, [10, 90])
        else:
            t80_lo = t80_hi = float("nan")

        aiccs, w, best = model_selection(t, y)
        T_star = detect_T(rmax, lam, n_batch)
        sat_supported = identifiable and best == "saturation"
        n_identifiable += int(identifiable)
        n_sat_supported += int(sat_supported)
        T_star_disp = f">{T_star - 1}" if T_star == 201 else T_star
        T_star_gt_len = (T_star > len(y))

        per_rows.append({
            "model": name, "params_B": pb, "n_steps": len(y), "batch_n": n_batch,
            "sigma_step": round(sigma, 4), "R_max": round(rmax, 4),
            "lambda": round(lam, 4), "lam_at_bound": lam_at_bound,
            "t80_point": round(t80_point, 4) if not math.isnan(t80_point) else "nan",
            "identifiable": identifiable, "aicc_best": best,
            "sat_supported": sat_supported,
            "T_star_power80": T_star_disp,
            "T_star_gt_len": T_star_gt_len,
        })
        boot_rows.append({
            "model": name, "boot_frac_lambda_at_bound": round(frac_bound, 4),
            "lambda_ci10": round(lam_lo, 4), "lambda_ci90": round(lam_hi, 4),
            "lambda_ci_span_frac": round(span_frac, 4),
            "t80_ci10": round(t80_lo, 4) if not math.isnan(t80_lo) else "nan",
            "t80_ci90": round(t80_hi, 4) if not math.isnan(t80_hi) else "nan",
        })
        ms_rows.append({
            "model": name, "aicc_constant": round(aiccs["constant"], 3),
            "aicc_linear": round(aiccs["linear"], 3),
            "aicc_saturation": round(aiccs["saturation"], 3),
            "w_constant": round(w["constant"], 4), "w_linear": round(w["linear"], 4),
            "w_saturation": round(w["saturation"], 4), "aicc_best": best,
        })

    def write_tsv(path, rows):
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            wr.writeheader()
            wr.writerows(rows)

    write_tsv(RESULTS / "scaling_law_iter25_identifiability.tsv", per_rows)
    write_tsv(RESULTS / "scaling_law_iter25_bootstrap.tsv", boot_rows)
    write_tsv(RESULTS / "scaling_law_iter25_modelsel.tsv", ms_rows)

    n = len(per_rows)
    summary = [
        {"metric": "n_traces", "value": n},
        {"metric": "n_lam_at_bound_pointfit",
         "value": sum(r["lam_at_bound"] for r in per_rows)},
        {"metric": "n_identifiable", "value": n_identifiable},
        {"metric": "frac_identifiable", "value": round(n_identifiable / n, 4)},
        {"metric": "n_saturation_supported", "value": n_sat_supported},
        {"metric": "frac_saturation_supported", "value": round(n_sat_supported / n, 4)},
        {"metric": "n_aicc_best_constant",
         "value": sum(r["aicc_best"] == "constant" for r in per_rows)},
        {"metric": "median_batch_n",
         "value": float(np.median([r["batch_n"] for r in per_rows]))},
    ]
    write_tsv(RESULTS / "scaling_law_iter25_summary.tsv", summary)

    # ---- figure: bootstrap lambda CI vs the bound, + AICc weights ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [r["model"] for r in per_rows]
    ax = axes[0]
    for i, (pr, br) in enumerate(zip(per_rows, boot_rows)):
        lo, hi = br["lambda_ci10"], br["lambda_ci90"]
        col = "#2a7" if pr["identifiable"] else "#c33"
        ax.plot([lo, hi], [i, i], color=col, lw=3, solid_capstyle="round")
        ax.plot(pr["lambda"], i, "o", color="k", ms=5, zorder=5)
    ax.axvline(LAM_HI, ls="--", color="gray", lw=1)
    ax.text(LAM_HI, -0.6, "  bound", color="gray", fontsize=8, va="top")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"$\hat\lambda$ (bootstrap 80% CI)")
    ax.set_title("Saturation rate identifiability\n(green=identifiable, red=degenerate)", fontsize=9)
    ax = axes[1]
    ind = np.arange(len(names))
    wc = [r["w_constant"] for r in ms_rows]
    wl = [r["w_linear"] for r in ms_rows]
    ws = [r["w_saturation"] for r in ms_rows]
    ax.barh(ind, wc, color="#888", label="constant")
    ax.barh(ind, wl, left=wc, color="#59f", label="linear")
    ax.barh(ind, ws, left=np.array(wc) + np.array(wl), color="#2a7", label="saturation")
    ax.set_yticks(ind)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AICc model weight (noise-aware)")
    ax.set_title("Which curve does the data support?", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    for d in (FIG_DIR, PAPER_FIG):
        fig.savefig(d / "scaling_law_iter25.pdf")
        fig.savefig(d / "scaling_law_iter25.png", dpi=140)
    plt.close(fig)

    print(f"traces={n} identifiable={n_identifiable} sat_supported={n_sat_supported}")
    for r in per_rows:
        print(r["model"], "batch_n", r["batch_n"], "ident", r["identifiable"],
              "best", r["aicc_best"], "T*", r["T_star_power80"])


if __name__ == "__main__":
    main()
