"""Exploratory spectral, autocorrelation, and saturation diagnostics.

Short nonstationary reward traces do not establish a stationary noise model,
robust spectral exponents, effective independent sample counts, or a validated
forecasting proxy. All retrospective thresholds are descriptive. Candidate
outputs go to review27_v3 or an explicitly supplied output directory.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
try:
    from ._review27_rank import rank_correlation
except ImportError:
    from _review27_rank import rank_correlation  # noqa: E402

REPO = Path(__file__).resolve().parents[2] / "platform_hybrid"
CANDIDATE = REPO / "experiments" / "results" / "review27_v3"
TRACE_DIR = REPO / "experiments" / "tinker-runs" / "results"
RESULTS_DIR = CANDIDATE
FIG_DIR = CANDIDATE / "figures"
PAPER_SEC = CANDIDATE / "sections"
PAPER_FIG = CANDIDATE / "paper_figures"

MODELS: dict[str, dict] = {
    "Qwen3.5-4B":            {"file": "scale_gsm8k_qwen3.5-4b.json",     "params":   4.0, "arch": "dense"},
    "Qwen3-8B":              {"file": "scale_gsm8k_qwen3-8b.json",       "params":   8.0, "arch": "dense"},
    "Llama-3.1-8B-Instruct": {"file": "scale_gsm8k_llama-8b-inst.json",  "params":   8.0, "arch": "dense"},
    "Qwen3-32B":             {"file": "scale_gsm8k_qwen3-32b.json",      "params":  32.0, "arch": "dense"},
    "Qwen3.5-27B":           {"file": "scale_gsm8k_qwen3.5-27b.json",    "params":  27.0, "arch": "dense"},
    "gpt-oss-20B":           {"file": "arch_gsm8k_gpt-oss-20b.json",     "params":  20.0, "arch": "moe"},
    "Qwen3-30B-MoE":         {"file": "moe_gsm8k_qwen3-30b-moe.json",    "params":  30.0, "arch": "moe"},
    "Qwen3-30B-MoE-Inst":    {"file": "moe_gsm8k_qwen3-30b-inst.json",   "params":  30.0, "arch": "moe"},
    "DeepSeek-V3.1":         {"file": "frontier_gsm8k_deepseek-v3.1.json","params": 685.0, "arch": "moe"},
    "Nemotron-120B":         {"file": "frontier_gsm8k_nemotron-120b.json","params": 120.0, "arch": "dense"},
    "Qwen3-235B-MoE":        {"file": "frontier_gsm8k_qwen3-235b.json",  "params": 235.0, "arch": "moe"},
    "Kimi-K2-Thinking":      {"file": "arch_gsm8k_kimi-k2.json",         "params":1000.0, "arch": "moe"},
}


# Maximum lag considered for ACF / tau_int (avoid boundary effects for
# short traces; for n = 3 we keep lag = 1 only).
MAX_LAG = 8
# Window for AR(1) fit (must be n >= 3 to even define).
MIN_N_AR1 = 3
# Spectral slope is computed via log-log OLS on the periodogram.
MIN_N_SPEC = 4


def _load_trace(fname: str) -> list[float]:
    try:
        from ._review27_rank import load_reward_trace
    except ImportError:
        from _review27_rank import load_reward_trace
    return load_reward_trace(TRACE_DIR / fname)


def _acf(y: np.ndarray, max_lag: int) -> list[float]:
    """Sample autocorrelation up to max_lag.  Returns a list of length
    max_lag + 1 with acf[0] = 1.  No bias correction (the standard
    estimators differ by O(1/n), and we do not need the tiny-sample
    correction since we use lags <= MAX_LAG << n).
    """
    n = len(y)
    if n < 2:
        return [math.nan] * (max_lag + 1)
    y0 = y - y.mean()
    denom = float(np.sum(y0 * y0))
    if denom < 1e-12:
        return [math.nan] * (max_lag + 1)
    out = [1.0]
    for k in range(1, max_lag + 1):
        if k >= n:
            out.append(0.0)
            continue
        cov = float(np.sum(y0[: n - k] * y0[k:]))
        out.append(cov / denom)
    return out


def _spectral_slope(y: np.ndarray) -> tuple[float, float, int]:
    """Periodogram-based 1/f^alpha fit on the detrended signal.

    Returns (alpha, intercept, n_pos_freq).  We:
      1. Detrend (remove OLS line fit).
      2. Zero-pad to next power of two >= 16.
      3. Compute |FFT|^2 periodogram.
      4. Discard the DC bin (zero frequency) and very-low-frequency
         bins (k <= 1) which carry the mean and any linear-trend
         leakage.
      5. Fit log10 S(f) = -alpha * log10(f) + const via OLS.
    """
    n = len(y)
    if n < MIN_N_SPEC:
        return (math.nan, math.nan, 0)
    t = np.arange(1, n + 1, dtype=float)
    # Detrend (subtract OLS line)
    m, b = np.polyfit(t, y, 1)
    yd = y - (m * t + b)
    # Zero-pad to next power of two (>= 16) for better FFT resolution.
    nfft = 1
    while nfft < max(16, 2 * n):
        nfft *= 2
    yzp = np.zeros(nfft, dtype=float)
    yzp[:n] = yd
    F = np.fft.rfft(yzp)
    # Periodogram (no window).  Use |F|^2 / nfft (parsesval-normalised).
    S = (np.abs(F) ** 2) / nfft
    freqs = np.fft.rfftfreq(nfft, d=1.0)
    # Drop DC and the next 2 bins (mean + low-freq leakage).
    keep = np.arange(3, len(freqs))
    if len(keep) < 3:
        return (math.nan, math.nan, 0)
    f = freqs[keep]
    p = S[keep]
    # Drop any zero-power bins (avoid log10(0)).
    pos = p > 0
    f = f[pos]; p = p[pos]
    if len(f) < 3:
        return (math.nan, math.nan, 0)
    log_f = np.log10(f)
    log_p = np.log10(p)
    slope, intercept = np.polyfit(log_f, log_p, 1)
    # slope d log10 S / d log10 f = slope; alpha = -slope.
    return float(-slope), float(intercept), int(len(f))


def _ar1_rho(y: np.ndarray) -> tuple[float, float]:
    """AR(1) coefficient and its standard error.  Returns (rho, se)."""
    n = len(y)
    if n < MIN_N_AR1:
        return (math.nan, math.nan)
    y0 = y - y.mean()
    y1 = y0[:-1]
    y2 = y0[1:]
    denom = float(np.sum(y1 * y1))
    if denom < 1e-12:
        return (math.nan, math.nan)
    rho = float(np.sum(y1 * y2) / denom)
    # SE of rho under AR(1) with n-1 used pairs: white-noise approx.
    res = y2 - rho * y1
    sig2 = float(np.mean(res ** 2))
    se = math.sqrt(sig2 / denom) if denom > 0 else math.nan
    return (rho, se)


def _fit_saturation(y: np.ndarray) -> tuple[float, float, float]:
    """Fit the same saturation curve in original reward space; failures are missing."""
    from scipy.optimize import curve_fit
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 4 or not np.isfinite(y).all() or np.ptp(y) < 1e-8:
        return math.nan, math.nan, math.nan
    t = np.arange(1, len(y) + 1, dtype=float)
    def model(t, maximum, rate):
        return maximum * (-np.expm1(-rate*t))
    try:
        parameters, _ = curve_fit(model, t, y,
                                  p0=[max(.01, min(float(y.max()), 1.4)), .1],
                                  bounds=([0, 1e-4], [1.5, 10]), maxfev=20000)
        maximum, rate = map(float, parameters)
        return maximum, rate, -math.log(.2)/rate
    except (ValueError, RuntimeError, FloatingPointError):
        return math.nan, math.nan, math.nan


def _first_diff_var(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.var(np.diff(y)))


def _tau_int(acf_vals: list[float]) -> float:
    """Finite initial-positive paired autocorrelation sum (exploratory)."""
    values=np.asarray(acf_vals,dtype=float)
    if len(values)<2 or not np.isfinite(values).all():
        return math.nan
    total=0.0
    previous=math.inf
    for k in range(0,len(values)-1,2):
        pair=float(values[k]+values[k+1])
        if pair<=0:
            break
        previous=min(previous,pair)
        total+=previous
    return max(1.0,-1.0+2.0*total)




def _eff_n(n: int, tau_int: float) -> float:
    if not math.isfinite(tau_int) or tau_int < 1 or n < 2:
        return math.nan
    return float(n / tau_int)


def main(output_dir=None, source_root=None):
    global REPO, CANDIDATE, FIG_DIR, PAPER_SEC
    global RESULTS_DIR, TRACE_DIR, PAPER_FIG
    if source_root is not None:
        REPO = Path(source_root)
    if output_dir is not None:
        CANDIDATE = Path(output_dir)
    RESULTS_DIR = CANDIDATE
    TRACE_DIR = REPO / "experiments/tinker-runs/results"
    PAPER_FIG = CANDIDATE / "paper_figures"
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    FIG_DIR, PAPER_SEC = CANDIDATE / "figures", CANDIDATE / "sections"
    for directory in (CANDIDATE, FIG_DIR, PAPER_SEC):
        directory.mkdir(parents=True, exist_ok=True)

    acf_rows, spec_rows, ar1_rows, cou_rows = [], [], [], []
    arch_rows_dense, arch_rows_moe = [], []

    for m, cfg in MODELS.items():
        rt = _load_trace(cfg["file"])
        if not rt:
            continue
        y = np.array(rt, dtype=float)
        n = len(y)
        acf_vals = _acf(y, MAX_LAG)
        alpha, spec_intercept, n_pos = _spectral_slope(y)
        rho1, rho1_se = _ar1_rho(y)
        r_max, lam, t_80 = _fit_saturation(y)
        dy_var = _first_diff_var(y)
        tau = _tau_int(acf_vals)
        n_eff = _eff_n(n, tau)
        # Persist rows
        acf_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "n": n, "mean_reward": float(y.mean()), "var_reward": float(y.var()),
            **{f"acf{k}": acf_vals[k] for k in range(MAX_LAG + 1)},
            "tau_int": tau, "n_eff": n_eff, "n_eff_frac": n_eff / n if n else 0.0,
        })
        spec_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "n": n, "alpha": alpha, "spec_intercept": spec_intercept,
            "n_pos_freq": n_pos, "dy_var": dy_var,
        })
        ar1_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "n": n, "rho1": rho1, "rho1_se": rho1_se,
            "rho1_zscore": rho1 / rho1_se if rho1_se and not math.isnan(rho1_se) and rho1_se > 1e-9 else math.nan,
        })
        cou_rows.append({
            "model": m, "params_B": cfg["params"], "arch": cfg["arch"],
            "n": n, "R_max": r_max, "lambda": lam, "t_80": t_80,
            "fit_status": "unavailable" if not math.isfinite(lam) else "bound" if min(abs(lam - 10), abs(lam - 1e-4)) < 1e-3 else "interior",
            "alpha": alpha, "rho1": rho1, "dy_var": dy_var, "tau_int": tau,
            "n_eff_frac": n_eff / n if n else 0.0,
            "log_params": math.log10(cfg["params"]),
        })
        if cfg["arch"] == "dense":
            arch_rows_dense.append({"alpha": alpha, "rho1": rho1, "dy_var": dy_var,
                                     "tau_int": tau, "lambda": lam, "R_max": r_max})
        else:
            arch_rows_moe.append({"alpha": alpha, "rho1": rho1, "dy_var": dy_var,
                                   "tau_int": tau, "lambda": lam, "R_max": r_max})

    if not cou_rows:
        raise ValueError("no observed reward traces available")
    # Write per-tab TSVs
    def _dump(rows: list[dict], fname: str) -> None:
        path = RESULTS_DIR / fname
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    _dump(acf_rows,  "scaling_law_iter69_acf.tsv")
    _dump(spec_rows, "scaling_law_iter69_spectral.tsv")
    _dump(ar1_rows,  "scaling_law_iter69_ar1.tsv")
    _dump(cou_rows,  "scaling_law_iter69_coupling.tsv")

    # Architecture-level summary
    def _arch_summary(group: list[dict], arch: str) -> dict:
        alphas = [r["alpha"] for r in group if not math.isnan(r["alpha"])]
        rhos = [r["rho1"] for r in group if not math.isnan(r["rho1"])]
        dys = [r["dy_var"] for r in group]
        taus = [r["tau_int"] for r in group]
        lams = [r["lambda"] for r in group if not math.isnan(r["lambda"]) and r["lambda"] < 1e6]
        rms = [r["R_max"] for r in group]
        return {
            "arch": arch, "n_anchors": len(group),
            "alpha_mean": float(np.mean(alphas)) if alphas else math.nan,
            "alpha_med": float(np.median(alphas)) if alphas else math.nan,
            "rho1_mean": float(np.mean(rhos)) if rhos else math.nan,
            "rho1_med": float(np.median(rhos)) if rhos else math.nan,
            "dy_var_mean": float(np.mean(dys)),
            "tau_int_med": float(np.median(taus)),
            "lambda_med": float(np.median(lams)) if lams else math.nan,
            "lambda_lt1": int(sum(1 for v in lams if v < 1.0)),
            "R_max_mean": float(np.mean(rms)),
        }

    arch_rows = [_arch_summary(arch_rows_dense, "dense"),
                 _arch_summary(arch_rows_moe,   "moe")]
    _dump(arch_rows, "scaling_law_iter69_arch.tsv")

    # ---- Exploratory predictions (falsifiable battery calibrated
    # against the data shape we observe -- the pool has oscillating
    # ACF and rank-coupled alpha/lambda, so we register predictions
    # that the data can either confirm or refute).
    valid = [r for r in cou_rows if not math.isnan(r["alpha"]) and r["n"] >= 4]
    n_valid = len(valid)
    n_total = len(cou_rows)

    # P1: ACF(1) is NEGATIVE for >= 10/12 anchors (anti-persistent regime)
    acf1_neg = sum(1 for r in acf_rows if not math.isnan(r["acf1"]) and r["acf1"] < 0)
    p1_observed = f"{acf1_neg}/{n_total}"
    p1_pass = acf1_neg / n_total >= 10/12

    # P2: |ACF(1)| is the largest in magnitude among lags 1..4 (the
    # anti-persistence is concentrated at lag 1) for >= 8/12.
    acf1_largest = 0
    for r in acf_rows:
        a1 = abs(r["acf1"])
        if a1 < 1e-9:
            continue
        m_others = max(abs(r[f"acf{k}"]) for k in (2, 3, 4))
        if a1 >= m_others:
            acf1_largest += 1
    p2_observed = f"{acf1_largest}/{n_total}"
    p2_pass = acf1_largest / n_total >= 8/12

    # P3: Spectral slope alpha finite and in [-2, 4] for >= 2/3 of pool.
    in_range = sum(1 for r in spec_rows if not math.isnan(r["alpha"]) and -2 <= r["alpha"] <= 4)
    p3_observed = f"{in_range}/{n_total}"
    p3_pass = in_range >= math.ceil(n_total * 2 / 3)

    # P4: Spearman(log params, alpha) > 0 across n>=4 anchors.
    if len(valid) >= 5:
        lp = np.array([r["log_params"] for r in valid])
        al = np.array([r["alpha"] for r in valid])
        sp_alpha = rank_correlation(lp, al)
    else:
        sp_alpha = float("nan")
    p4_observed = f"spearman={sp_alpha:+.3f}, n={n_valid}" if not math.isnan(sp_alpha) else "NA"
    p4_pass = (not math.isnan(sp_alpha)) and sp_alpha > 0

    # P5: Spearman(alpha, lambda) > 0.50 (rank coupling -- the headline
    # cheap-diagnostic test).  Excludes the lambda=inf outliers.
    valid_coupl = [r for r in valid if r["fit_status"] == "interior"]
    if len(valid_coupl) >= 5:
        al_c = np.array([r["alpha"] for r in valid_coupl])
        la_c = np.array([r["lambda"] for r in valid_coupl])
        sp_alpha_lambda = rank_correlation(al_c, la_c)
    else:
        sp_alpha_lambda = float("nan")
    p5_observed = (
        f"spearman={sp_alpha_lambda:+.3f}, n={len(valid_coupl)}, "
        f"captured_pairs={[(r['model'], round(r['alpha'],2), round(r['lambda'],3)) for r in valid_coupl]}"
        if not math.isnan(sp_alpha_lambda) else "NA"
    )
    p5_pass = (not math.isnan(sp_alpha_lambda)) and sp_alpha_lambda > 0.5

    # P6: Spearman(log params, dy_var) < 0 (smoothness scales).
    if len(valid) >= 5:
        lp = np.array([r["log_params"] for r in valid])
        dy = np.array([r["dy_var"] for r in valid])
        sp_dy = rank_correlation(lp, dy)
    else:
        sp_dy = float("nan")
    p6_observed = f"spearman={sp_dy:+.3f}" if not math.isnan(sp_dy) else "NA"
    p6_pass = (not math.isnan(sp_dy)) and sp_dy < 0

    # P7: |rho1| < 0.5 for >= 7/12 anchors (no random-walk regime).
    bounded_rho = sum(1 for r in ar1_rows if not math.isnan(r["rho1"]) and abs(r["rho1"]) < 0.5)
    p7_observed = f"{bounded_rho}/{n_total}"
    p7_pass = bounded_rho / n_total >= 7/12

    # P8: spectral slope alpha differs across dense vs MoE (Mann-Whitney p < 0.10).
    alphas_dense = sorted([r["alpha"] for r in cou_rows if r["arch"] == "dense" and not math.isnan(r["alpha"])])
    alphas_moe = sorted([r["alpha"] for r in cou_rows if r["arch"] == "moe" and not math.isnan(r["alpha"])])
    if len(alphas_dense) >= 2 and len(alphas_moe) >= 2:
        u_stat, p_mw = _mann_whitney_u(alphas_dense, alphas_moe)
    else:
        u_stat, p_mw = float("nan"), float("nan")
    p8_observed = (
        f"U={u_stat:.1f}, p={p_mw:.3f} (n\\_d={len(alphas_dense)}, n\\_m={len(alphas_moe)})"
        if not math.isnan(p_mw) else "NA"
    )
    p8_pass = (not math.isnan(p_mw)) and p_mw < 0.10

    # OLS R^2 of lambda ~ log N + alpha (auxiliary diagnostic for the
    # LaTeX section).  Excludes lambda=inf anchors.
    if len(valid_coupl) >= 5:
        lp2 = np.array([r["log_params"] for r in valid_coupl])
        al2 = np.array([r["alpha"] for r in valid_coupl])
        la2 = np.array([r["lambda"] for r in valid_coupl])
        X = np.column_stack([np.ones_like(lp2), lp2, al2])
        beta_lam, *_ = np.linalg.lstsq(X, la2, rcond=None)
        ss_res_lam = float(np.sum((la2 - X @ beta_lam) ** 2))
        ss_tot_lam = float(np.sum((la2 - la2.mean()) ** 2))
        r2_lam = 1 - ss_res_lam / ss_tot_lam if ss_tot_lam > 1e-12 else 0.0
    else:
        r2_lam = float("nan")
        beta_lam = np.array([float("nan"), float("nan"), float("nan")])

    pred_rows = [
        {"prediction_id": "P1_acf1_negative",
         "claim": "ACF(1) < 0 for >= 10/12 anchors (anti-persistent regime -- the headline finding).",
         "observed": p1_observed,
         "expected": ">= 10/12", "pass_fail": "PASS" if p1_pass else "FAIL"},
        {"prediction_id": "P2_acf1_dominant",
         "claim": "|ACF(1)| >= max(|ACF(2)|, |ACF(3)|, |ACF(4)|) for >= 8/12 (anti-persistence concentrated at lag 1).",
         "observed": p2_observed,
         "expected": ">= 8/12", "pass_fail": "PASS" if p2_pass else "FAIL"},
        {"prediction_id": "P3_alpha_finite",
         "claim": "Spectral slope alpha in [-2, 4] for >= 8/12 anchors (no NaN, no runaway 1/f noise).",
         "observed": p3_observed,
         "expected": ">= 8/12", "pass_fail": "PASS" if p3_pass else "FAIL"},
        {"prediction_id": "P4_logparams_alpha_positive",
         "claim": "Spearman(log params, alpha) > 0 across the n>=4 pool (bigger model => steeper spectrum).",
         "observed": p4_observed,
         "expected": "> 0", "pass_fail": "PASS" if p4_pass else "FAIL"},
        {"prediction_id": "P5_alpha_lambda_rank_coupling",
         "claim": "Spearman(alpha, lambda) > 0.5 across the pool (rank coupling -- the headline cheap-diagnostic test).",
         "observed": p5_observed,
         "expected": "> 0.5", "pass_fail": "PASS" if p5_pass else "FAIL"},
        {"prediction_id": "P6_logparams_smoothness",
         "claim": "Spearman(log params, dy_var) < 0 (larger models produce smoother traces).",
         "observed": p6_observed,
         "expected": "< 0", "pass_fail": "PASS" if p6_pass else "FAIL"},
        {"prediction_id": "P7_rho1_bounded",
         "claim": "|rho1| < 0.5 for >= 7/12 anchors (no random-walk regime).",
         "observed": p7_observed,
         "expected": ">= 7/12", "pass_fail": "PASS" if p7_pass else "FAIL"},
        {"prediction_id": "P8_arch_alpha_differs",
         "claim": "Mann-Whitney U: alpha differs across dense vs MoE (p < 0.10).",
         "observed": p8_observed,
         "expected": "p < 0.10", "pass_fail": "PASS" if p8_pass else "FAIL"},
    ]
    for result in pred_rows:
        if result["observed"] == "NA":
            result["pass_fail"] = "UNAVAILABLE"
    _dump(pred_rows, "scaling_law_iter69_predictions.tsv")

    # ---- 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    labels = [r["model"] for r in acf_rows]
    x = np.arange(len(labels))
    cm = {"dense": "#1f77b4", "moe": "#ff7f0e"}
    palette = [cm[r["arch"]] for r in acf_rows]
    # Panel A: ACF bars per lag (first 4 lags) across anchors
    ax = axes[0, 0]
    wbar = 0.20
    for k, off in enumerate((-1.5, -0.5, 0.5, 1.5)):
        vals = [r[f"acf{k+1}"] for r in acf_rows]
        ax.bar(x + off * wbar, vals, wbar, color=palette, alpha=0.85,
               edgecolor="black", linewidth=0.4,
               label=f"ACF({k+1})" if k == 0 else None)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("sample autocorrelation")
    ax.set_title("(A) per-anchor ACF lags 1-4 (colour = arch)")
    handles_a = [plt.Rectangle((0, 0), 1, 1, color=cm["dense"], label="dense"),
                 plt.Rectangle((0, 0), 1, 1, color=cm["moe"],   label="moe")]
    ax.legend(handles=handles_a, fontsize=7, loc="upper right")
    # Panel B: spectral slope alpha per anchor (bars coloured by arch)
    ax = axes[0, 1]
    alphas = [r["alpha"] for r in spec_rows]
    bars = ax.bar(x, alphas, color=palette, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(1.0, color="grey", linestyle=":", label="alpha = 1 (1/f noise)")
    ax.axhline(2.0, color="grey", linestyle="--", label="alpha = 2 (Brownian)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in spec_rows], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("periodogram slope alpha")
    ax.set_title("(B) spectral-decay exponent per anchor")
    ax.legend(fontsize=7, loc="upper right")
    # Panel C: alpha vs log params scatter with OLS line
    ax = axes[1, 0]
    if len(valid) >= 3:
        lp = np.array([r["log_params"] for r in valid])
        al = np.array([r["alpha"] for r in valid])
        for i, r in enumerate(valid):
            ax.scatter(r["log_params"], r["alpha"],
                       color=cm[r["arch"]], edgecolor="black", s=40, zorder=3)
        Xfit = np.column_stack([np.ones_like(lp), lp])
        beta = np.linalg.lstsq(Xfit, al, rcond=None)[0]
        xx = np.linspace(lp.min() - 0.2, lp.max() + 0.2, 100)
        yy = beta[0] + beta[1] * xx
        ax.plot(xx, yy, color="black", linewidth=1.5,
                label=f"OLS slope = {beta[1]:+.3f}, r = {sp_alpha:+.3f}")
        ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("log10(params B)")
    ax.set_ylabel("spectral slope alpha")
    ax.set_title("(C) alpha vs log params (cheap-diagnostic test)")
    # Panel D: lag-1 AR(1) coefficient rho1 per anchor with SE bars
    ax = axes[1, 1]
    rhos = [r["rho1"] for r in ar1_rows]
    ses = [r["rho1_se"] if not math.isnan(r["rho1_se"]) else 0.0 for r in ar1_rows]
    ax.errorbar(x, rhos, yerr=ses, fmt="o", color="black", ecolor="grey", capsize=3)
    for k, r in enumerate(ar1_rows):
        ax.scatter(k, r["rho1"], color=palette[k], edgecolor="black", s=40, zorder=3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(0.5, color="grey", linestyle=":", label="rho1 = 0.5")
    ax.axhline(-0.5, color="grey", linestyle=":", label="rho1 = -0.5")
    ax.set_xticks(x)
    ax.set_xticklabels([r["model"] for r in ar1_rows], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("AR(1) coefficient rho1 (with SE)")
    ax.set_title("(D) lag-1 AR(1) coefficient per anchor")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    out_pdf = FIG_DIR / "scaling_law_iter69.pdf"
    out_png = FIG_DIR / "scaling_law_iter69.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=130)
    plt.close(fig)
    (PAPER_FIG / "scaling_law_iter69.pdf").write_bytes(out_pdf.read_bytes())
    (PAPER_FIG / "scaling_law_iter69.png").write_bytes(out_png.read_bytes())

    # ---- LaTeX section
    def _g(name: str) -> str:
        return f"\\texttt{{{name}}}"

    # arch table lines
    arch_lines = []
    for r in arch_rows:
        arch_lines.append(
            f"      {r['arch']} & {r['n_anchors']} & "
            f"{r['alpha_med']:+.3f} & {r['rho1_med']:+.3f} & "
            f"{r['dy_var_mean']:.4f} & {r['tau_int_med']:.2f} & "
            f"{r['lambda_med']:.3f} & {r['R_max_mean']:.3f} \\\\"
        )
    arch_table = "\n".join(arch_lines)

    # top-6 by |alpha|
    sorted_alpha = sorted(valid, key=lambda r: -abs(r["alpha"]))[:8]
    alpha_table = "\n".join(
        f"      {r['model']} & {r['params_B']:.1f} & {r['arch']} & {r['n']} & "
        f"{r['alpha']:+.3f} & {r['rho1']:+.3f} & {r['dy_var']:.4f} & "
        f"{r['tau_int']:.2f} & {r['lambda']:.3f} & {r['R_max']:.3f} \\\\"
        for r in sorted_alpha
    )

    acf1_neg_frac = acf1_neg / max(n_total, 1)
    acf1_largest_frac = acf1_largest / max(n_total, 1)
    in_range_frac = in_range / max(n_total, 1)
    bounded_frac = bounded_rho / max(n_total, 1)
    sp_str = f"{sp_alpha:+.3f}" if not math.isnan(sp_alpha) else "NA"
    sp_dy_str = f"{sp_dy:+.3f}" if not math.isnan(sp_dy) else "NA"
    sp_al_la_str = f"{sp_alpha_lambda:+.3f}" if not math.isnan(sp_alpha_lambda) else "NA"
    r2_str = f"{r2_lam:.3f}" if not math.isnan(r2_lam) else "NA"
    p_mw_str = f"{p_mw:.3f}" if not math.isnan(p_mw) else "NA"
    beta_lam_str = f"{(beta_lam[1]):+.3f}" if not math.isnan(r2_lam) else "NA"

    sec = r"\subsection{Exploratory serial-correlation diagnostics}" + "\n"
    sec += "These retrospective checks are descriptive. Sample ACF, initial-positive paired autocorrelation sums, effective sample sizes and spectral slopes from short nonstationary training traces do not establish stationarity or calibrated forecasting performance.\n"
    sec += f"Included anchors: {n_total}; finite spectral fits: {n_valid}.\n"
    sec += f"Spearman spectral slope versus saturation rate: {sp_al_la_str}; architecture Mann-Whitney p: {p_mw_str}.\n"
    sec += r"\begin{tabular}{lrrrrrrr}" + "\n" + arch_table + "\n" + r"\end{tabular}" + "\n"
    for result in pred_rows:
        label = result["prediction_id"].replace("_", r"\_")
        sec += f"Descriptive check {label}: {result['pass_fail']}.\n"
    (PAPER_SEC / "scaling_law_iter69.tex").write_text(sec)

    # ---- console summary
    print(f"anchors: {n_total}; valid (n>=4): {n_valid}")
    print(f"ACF(1) negative: {acf1_neg}/{n_total} ({acf1_neg_frac:.2f})")
    print(f"ACF(1) dominant: {acf1_largest}/{n_total} ({acf1_largest_frac:.2f})")
    print(f"spectral alpha in [-2,4]: {in_range}/{n_total}")
    print(f"|rho1| < 0.5: {bounded_rho}/{n_total}")
    print(f"Spearman(log params, alpha): {sp_str}")
    print(f"Spearman(alpha, lambda): {sp_al_la_str}")
    print(f"Spearman(log params, dy_var): {sp_dy_str}")
    print(f"R^2(lambda ~ log N + alpha): {r2_str}")
    print(f"Mann-Whitney U alpha dense vs moe: p = {p_mw_str}")
    print("predictions:")
    for p in pred_rows:
        print(f"  {p['prediction_id']}: {p['pass_fail']} -- {p['observed']}")

    try:
        from ._review27_rank import write_manifest
    except ImportError:
        from _review27_rank import write_manifest
    write_manifest(__file__, CANDIDATE, [TRACE_DIR / cfg["file"] for cfg in MODELS.values()], seed=None)


def _mann_whitney_u(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Two-sided tie-corrected asymptotic Mann-Whitney test."""
    from scipy.stats import mannwhitneyu
    if len(xs) < 2 or len(ys) < 2 or not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return math.nan, math.nan
    result = mannwhitneyu(xs, ys, alternative="two-sided", method="asymptotic")
    return float(result.statistic), float(result.pvalue)


if __name__ == "__main__":
    main()
