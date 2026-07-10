"""Pillar 1 iter-17 elevation: triple-orthogonal diagnostics on the
canonical 5-anchor + extended 12-anchor frontier GRPO reward traces.

Four orthogonal axes beyond the saturation/holdout/bootstrap
diagnostics already in iter5/iter9/iter13:

  (A) Multi-model AIC/BIC likelihood comparison (profile table).
      For each trace we fit four functional forms --
        constant       R(t) = c                                    (k=1)
        linear         R(t) = a + b*t                              (k=2)
        saturation     R(t) = R_max*(1 - exp(-lambda*t))           (k=2)
        logistic       R(t) = L/(1 + exp(-k*(t - t0)))             (k=3)
      -- and report AIC, BIC, residual sum-of-squares, and best-AIC
      winner per trace.  We additionally report DELTA-AIC relative to
      the best model, so that on traces where "constant" wins by less
      than 4 units of AIC (Burnham-Anderson "essentially equivalent"),
      a second model is competitive.  The key diagnostic:
      "what functional form does the data best support?"

  (B) Changepoint detection with permutation-test significance.
      Binary-segmentation brute force on per-step reward trace to
      locate tau = argmax_tau |mean(y_pre) - mean(y_post)|.  Report
      (i)  tau_hat,
      (ii) PERMUTATION-TEST p-value: probability of seeing
           |contrast| >= |contrast_tau_hat| under a permutation null
           that the order of rewards is exchangeable (no real
           changepoint).  This tests whether the changepoint is
           statistically distinct from null.
      (iii) block-bootstrap 95% CI on tau (block_size=n//6 to
           preserve local autocorrelation).

  (C) Effective plateau horizon T_eps (model-free saturation time).
      For each trace define T_eps as the earliest step T such that
      for every subsequent step t' >= T the rolling-window mean
      (window=5) satisfies |y(t') - mean(window) | < eps.  T_eps is
      the operational saturation time that does not depend on the
      functional-form's lambda bound.  Reported at eps in
      {0.05, 0.10, 0.20}.

  (D) Phase-label agreement test (Cohen's kappa).
      Re-derive phase label three ways:
        (1) heuristic:  hand-cut early-vs-late mean delta (iter5)
        (2) changepoint: sign of post-tau vs pre-tau contrast
        (3) AIC-refined: best-AIC model, with delta-AIC tiebreaks
            (constant+delta>=4 -> plateau; linear sign sets
            drift/saturation; logistic -> "logistic_curve")
      Report pair-wise Cohen's kappa on both 5-anchor and 12-anchor
      sets and a confusion-matrix dump for qualitative review.

Outputs:
  experiments/results/scaling_law_iter17_aic.tsv         (per-(trace,model) AIC profile)
  experiments/results/scaling_law_iter17_changepoint.tsv (changepoint tau + perm-p + CI)
  experiments/results/scaling_law_iter17_t_eps.tsv       (effective plateau horizon)
  experiments/results/scaling_law_iter17_phase_kappa.tsv (per-trace phase labels)
  experiments/results/scaling_law_iter17_phase_kappa_only.tsv (Cohen's kappa)
  figures/scaling_law_iter17.{pdf,png}
  paper/figures/scaling_law_iter17.{pdf,png}
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

MODELS_5: dict[str, str] = {
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
ARCH: dict[str, str] = {
    "Qwen3.5-4B": "dense",
    "Qwen3-8B": "dense",
    "Llama-3.1-8B-Instruct": "dense",
    "DeepSeek-V3.1": "MoE",
    "Nemotron-120B": "dense",
}
MODELS_12: dict[str, str] = {
    **MODELS_5,
    "Qwen3.5-27B": "scale_gsm8k_qwen3.5-27b.json",
    "Qwen3-32B": "scale_gsm8k_qwen3-32b.json",
    "Qwen3-30B-MoE": "moe_gsm8k_qwen3-30b-moe.json",
    "Qwen3-30B-MoE-Inst": "moe_gsm8k_qwen3-30b-inst.json",
    "Qwen3-235B": "frontier_gsm8k_qwen3-235b.json",
    "Kimi-K2-Thinking": "arch_gsm8k_kimi-k2.json",
    "GPT-OSS-20B": "arch_gsm8k_gpt-oss-20b.json",
}
PARAM_B.update({
    "Qwen3.5-27B": 27.0,
    "Qwen3-32B": 32.0,
    "Qwen3-30B-MoE": 30.0,
    "Qwen3-30B-MoE-Inst": 30.0,
    "Qwen3-235B": 235.0,
    "Kimi-K2-Thinking": 1000.0,
    "GPT-OSS-20B": 20.0,
})
ARCH.update({
    "Qwen3.5-27B": "dense",
    "Qwen3-32B": "dense",
    "Qwen3-30B-MoE": "MoE",
    "Qwen3-30B-MoE-Inst": "MoE",
    "Qwen3-235B": "MoE",
    "Kimi-K2-Thinking": "MoE",
    "GPT-OSS-20B": "MoE",
})

SEED = 42
N_BOOT = 2000
N_PERM = 5000


# --------------------------------------------------------------------------
# Functional forms
# --------------------------------------------------------------------------

def _linear(t, a, b):
    return a + b * t


def _saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def _logistic(t, L, k, t0):
    return L / (1.0 + np.exp(-k * (t - t0)))


# --------------------------------------------------------------------------
# Model-fitting helpers
# --------------------------------------------------------------------------

def aic_bic(n: int, k: int, rss: float) -> tuple[float, float]:
    """AIC and BIC under Gaussian-likelihood assumption.
    AIC = n*ln(RSS/n) + 2*k;  BIC = n*ln(RSS/n) + k*ln(n).
    """
    if rss <= 0 or n <= k + 1:
        return float("nan"), float("nan")
    base = n * math.log(rss / n)
    return base + 2 * k, base + k * math.log(n)


def _safe_curve_fit(f, x, y, p0, bounds, maxfev=20_000):
    try:
        popt, _ = curve_fit(f, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
        return popt
    except Exception:
        return None


def fit_models(y: np.ndarray) -> list[dict]:
    """Fit 4 functional forms and report RSS, AIC, BIC, parameters per model."""
    n = len(y)
    t = np.arange(1, n + 1, dtype=float)
    y = np.asarray(y, float)
    rows: list[dict] = []

    # constant
    c = float(y.mean())
    rss = float(np.sum((y - c) ** 2))
    a, b = aic_bic(n, 1, rss)
    rows.append(dict(model="constant", n_params=1, rss=rss, aic=a, bic=b,
                     converged=True, params=dict(c=c)))

    # linear
    p = _safe_curve_fit(
        _linear, t, y,
        p0=(float(y.mean()), 0.0),
        bounds=([-2.0, -1.0], [3.0, 1.0]),
    )
    if p is not None:
        yhat = _linear(t, *p)
        rss = float(np.sum((y - yhat) ** 2))
        a, b = aic_bic(n, 2, rss)
        rows.append(dict(model="linear", n_params=2, rss=rss, aic=a, bic=b,
                         converged=True,
                         params=dict(a=float(p[0]), b=float(p[1]))))
    else:
        rows.append(dict(model="linear", n_params=2, rss=float("nan"),
                         aic=float("nan"), bic=float("nan"),
                         converged=False, params=dict()))

    # saturation
    p = _safe_curve_fit(
        _saturation, t, y,
        p0=(max(0.9 * float(np.max(y)) + 0.05, 0.05), 0.3),
        bounds=([0.0, 1e-4], [1.5, 10.0]),
    )
    if p is not None:
        yhat = _saturation(t, *p)
        rss = float(np.sum((y - yhat) ** 2))
        a, b = aic_bic(n, 2, rss)
        rows.append(dict(model="saturation", n_params=2, rss=rss, aic=a, bic=b,
                         converged=True,
                         params=dict(R_max=float(p[0]), lam=float(p[1]))))
    else:
        rows.append(dict(model="saturation", n_params=2, rss=float("nan"),
                         aic=float("nan"), bic=float("nan"),
                         converged=False, params=dict()))

    # logistic (k=3)
    p = _safe_curve_fit(
        _logistic, t, y,
        p0=(float(np.max(y) + 0.05), 0.5, float(n / 2.0)),
        bounds=([0.0, 1e-3, 0.0], [2.0, 5.0, float(n) + 5]),
    )
    if p is not None:
        yhat = _logistic(t, *p)
        rss = float(np.sum((y - yhat) ** 2))
        a, b = aic_bic(n, 3, rss)
        rows.append(dict(model="logistic", n_params=3, rss=rss, aic=a, bic=b,
                         converged=True,
                         params=dict(L=float(p[0]), k=float(p[1]), t0=float(p[2]))))
    else:
        rows.append(dict(model="logistic", n_params=3, rss=float("nan"),
                         aic=float("nan"), bic=float("nan"),
                         converged=False, params=dict()))

    return rows


# --------------------------------------------------------------------------
# Binary-segmentation changepoint + permutation-test p-value
# --------------------------------------------------------------------------

def _contrast_at(y: np.ndarray, tau: int) -> float:
    n = len(y)
    if tau <= 0 or tau >= n:
        return 0.0
    pre = float(np.mean(y[:tau]))
    post = float(np.mean(y[tau:]))
    return abs(post - pre)


def best_changepoint(y: np.ndarray) -> int | None:
    n = len(y)
    if n < 4:
        return None
    y = np.asarray(y, float)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    prefix_means = csum[1:] / np.arange(1, n + 1)
    best_tau, best_d = None, -1.0
    for tau in range(2, n - 1):
        pre = float(prefix_means[tau - 1])
        post = (csum[n] - csum[tau]) / (n - tau)
        d = abs(pre - post)
        if d > best_d:
            best_tau, best_d = tau, d
    return best_tau


def changepoint_significance(y: np.ndarray, n_perm: int = N_PERM,
                              rng: np.random.Generator | None = None) -> tuple[int, float, float]:
    """Permutation test: probability of seeing |contrast| >= observed under
    the exchangeability null.  Returns (tau_hat, observed_contrast, perm_p).
    """
    n = len(y)
    if n < 4:
        return (-1, float("nan"), float("nan"))
    y = np.asarray(y, float)
    tau_hat = best_changepoint(y)
    if tau_hat is None:
        return (-1, float("nan"), float("nan"))
    obs = _contrast_at(y, tau_hat)
    if rng is None:
        rng = np.random.default_rng(SEED)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(y)
        pcp = best_changepoint(perm)
        if pcp is None:
            continue
        pc = _contrast_at(perm, pcp)
        if pc >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return (int(tau_hat), float(obs), float(p))


def changepoint_bootstrap_ci(y: np.ndarray, n_boot: int = N_BOOT,
                              block_size: int | None = None,
                              rng: np.random.Generator | None = None
                              ) -> tuple[int, int, int, float, float]:
    """Circular block-bootstrap CI on tau. block_size default n//6."""
    n = len(y)
    if rng is None:
        rng = np.random.default_rng(SEED)
    if block_size is None:
        block_size = max(3, n // 6)
    y = np.asarray(y, float)
    tau_hat = best_changepoint(y)
    if tau_hat is None:
        return (-1, -1, -1, float("nan"), float("nan"))

    n_blocks = math.ceil(n / block_size)
    taus = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        # circular block sampling: wrap indices modulo n
        sample = np.array([y[(s + i) % n] for s in starts
                           for i in range(block_size)])[:n]
        if len(sample) < 4:
            continue
        bcp = best_changepoint(sample)
        if bcp is not None:
            taus.append(bcp)
    if not taus:
        return (tau_hat, -1, -1, float("nan"), float("nan"))
    pre = float(np.mean(y[:tau_hat]))
    post = float(np.mean(y[tau_hat:]))
    lo, hi = np.percentile(taus, [2.5, 97.5])
    return (int(tau_hat), int(lo), int(hi), pre, post)


# --------------------------------------------------------------------------
# Effective plateau horizon T_eps
# --------------------------------------------------------------------------

def t_eps(y: np.ndarray, eps: float, window: int = 5,
          min_step: int = 5) -> int | None:
    """Earliest step T such that |y(t') - rolling_window_mean| < eps for
    all t' >= T.  Returns None if unbounded within trace."""
    n = len(y)
    if n < window + 1:
        return None
    y = np.asarray(y, float)
    csum = np.concatenate(([0.0], np.cumsum(y)))
    T_candidates = []
    for T in range(max(min_step, window + 1), n + 1):
        ok = True
        for tp in range(T, n + 1):
            a = max(0, tp - window)
            b = tp
            m = (csum[b] - csum[a]) / (b - a)
            if math.isnan(m) or abs(y[tp - 1] - m) >= eps:
                ok = False
                break
        if ok:
            return T
    return None


# --------------------------------------------------------------------------
# Phase label by 3 methods
# --------------------------------------------------------------------------

def phase_heuristic(y: np.ndarray) -> str:
    """Reproduce iter5's hand-cut early->late mean contrast."""
    n = len(y)
    cut = max(1, n // 3)
    early = float(np.mean(y[:cut]))
    late = float(np.mean(y[cut:]))
    d = late - early
    if d <= -0.30:
        return "collapse"
    if -0.30 < d < -0.10:
        return "drift"
    if d > 0.30:
        return "saturation"
    return "plateau"


def phase_changepoint(y: np.ndarray) -> str:
    n = len(y)
    if n < 6:
        return "plateau"
    y = np.asarray(y, float)
    tau = best_changepoint(y)
    if tau is None:
        return "plateau"
    pre = float(np.mean(y[:tau]))
    post = float(np.mean(y[tau:]))
    d = post - pre
    # the changepoint convention: d > 0 means later is higher
    if d <= -0.20:
        return "collapse"
    if -0.20 < d < -0.05:
        return "drift"
    if d > 0.20:
        return "saturation"
    return "plateau"


def phase_aic(y: np.ndarray) -> str:
    """Phase label by best-AIC model, mapping to the same phase ontology
    as the heuristic and changepoint methods:

      constant       -> 'plateau'
      linear (b>0)   -> 'saturation'
      linear (b<0)   -> 'drift'
      logistic       -> 'logistic_curve' (mapped to 'saturation' for kappa)
      saturation     -> 'saturation' if lambda<5 else 'plateau'

    The 'plateau' label for constant IS informative: it says the trace
    is statistically indistinguishable from a flat line at its mean.
    """
    fits = fit_models(y)
    fits_ok = [f for f in fits if f["converged"]
               and math.isfinite(f["aic"])]
    if not fits_ok:
        return "plateau"
    fits_ok.sort(key=lambda f: f["aic"])
    best = fits_ok[0]

    if best["model"] == "constant":
        return "plateau"
    if best["model"] == "linear":
        b = float(best["params"].get("b", 0.0))
        return "saturation" if b > 0 else "drift"
    if best["model"] == "logistic":
        # logistic = S-shaped curve; map to saturation since the
        # final state is a plateau either way
        return "saturation"
    if best["model"] == "saturation":
        lam = float(best["params"].get("lam", 0.0))
        return "plateau" if lam >= 5.0 else "saturation"
    return "plateau"


def cohens_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    n = len(a)
    po = float(np.mean([1.0 if ai == bj else 0.0 for ai, bj in zip(a, b)]))
    labels = sorted(set(a) | set(b))
    pe = 0.0
    for l in labels:
        pa = sum(1 for x in a if x == l) / n
        pb = sum(1 for x in b if x == l) / n
        if pa > 0 and pb > 0:
            pe += pa * pb
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1 - pe)


# --------------------------------------------------------------------------
# Trace loading
# --------------------------------------------------------------------------

def load_reward_per_step(path: Path) -> np.ndarray | None:
    """Extract the per-step mean reward from a tinker-trace JSON file.

    Canonical layout (scripts/tinker_run.py writes this):
        {..., "reward_trace": [r_1, r_2, ..., r_T], ...}

    We also fall back to per_step_rewards / step_log[].reward / etc.
    """
    try:
        with open(path, "r") as f:
            d = json.load(f)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None

    if "reward_trace" in d and isinstance(d["reward_trace"], list):
        try:
            arr = np.asarray(d["reward_trace"], float)
            if arr.size > 0:
                return arr
        except Exception:
            pass
    for key in ("per_step_rewards", "rewards_per_step", "reward_per_step"):
        if key in d and isinstance(d[key], list):
            arr = np.asarray(d[key], float)
            if arr.size > 0:
                return arr
    if "step_log" in d and isinstance(d["step_log"], list):
        try:
            rs = [s.get("reward") for s in d["step_log"]
                  if isinstance(s, dict) and "reward" in s]
            rs = [r for r in rs if isinstance(r, (int, float))]
            if rs:
                return np.asarray(rs, float)
        except Exception:
            pass
    if "steps" in d and isinstance(d["steps"], list):
        try:
            means = []
            for s in d["steps"]:
                if isinstance(s, dict):
                    if "rewards" in s and isinstance(s["rewards"], list) and s["rewards"]:
                        means.append(float(np.mean(s["rewards"])))
                    elif "mean_reward" in s:
                        means.append(float(s["mean_reward"]))
            if means:
                return np.asarray(means, float)
        except Exception:
            pass
    return None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)

    aic_rows: list[list] = []
    cp_rows: list[list] = []
    t_eps_rows: list[list] = []

    for label, fname in MODELS_5.items():
        path = TRACE_DIR / fname
        y = load_reward_per_step(path)
        if y is None or len(y) < 4:
            print(f"  [skip] {label}: no usable per-step reward trace")
            continue
        print(f"  [fit ] {label}: n={len(y)}, params_B={PARAM_B[label]}, arch={ARCH[label]}")

        # (A) multi-model AIC
        fits = fit_models(y)
        fits_ok = [f for f in fits if f["converged"]
                   and math.isfinite(f["aic"])]
        fits_ok.sort(key=lambda f: f["aic"])
        best_aic = fits_ok[0]["aic"]
        for fit in fits:
            if not fit["converged"] or not math.isfinite(fit["aic"]):
                continue
            aic_rows.append([
                label, PARAM_B[label], ARCH[label], len(y),
                fit["model"], fit["n_params"], fit["rss"], fit["aic"],
                fit["bic"], fit["aic"] - best_aic,
                1 if fit["model"] == fits_ok[0]["model"] else 0,
                json.dumps({k: float(v) for k, v in fit["params"].items()},
                           sort_keys=True),
            ])

        # (B) changepoint with permutation-test p-value
        tau_hat, tau_lo, tau_hi, pre_m, post_m = changepoint_bootstrap_ci(
            y, n_boot=N_BOOT, rng=rng)
        tau_sig, contrast_obs, perm_p = changepoint_significance(
            y, n_perm=N_PERM, rng=rng)
        early_cut = max(1, len(y) // 3)
        delta_early_late = float(np.mean(y[early_cut:]) - np.mean(y[:early_cut]))
        peak_step = int(np.argmax(y)) + 1
        peak_val = float(np.max(y))
        peak_minus_late = peak_val - float(np.mean(y[early_cut:]))
        zero_frac = float(np.mean(y == 0.0))
        cp_rows.append([
            label, PARAM_B[label], ARCH[label], len(y),
            tau_hat, tau_lo, tau_hi, pre_m, post_m, post_m - pre_m,
            delta_early_late, peak_step, peak_val, peak_minus_late,
            zero_frac, contrast_obs, perm_p,
        ])

        # (C) T_eps
        for eps in (0.05, 0.10, 0.20):
            t = t_eps(y, eps)
            t_eps_rows.append([
                label, PARAM_B[label], ARCH[label], len(y),
                eps, t if t is not None else -1, peak_step,
            ])

    with open(RESULTS_DIR / "scaling_law_iter17_aic.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "params_B", "arch", "n_steps",
                    "model_form", "n_params", "rss", "aic", "bic",
                    "delta_aic_vs_best", "is_best_aic", "params_json"])
        w.writerows(aic_rows)

    with open(RESULTS_DIR / "scaling_law_iter17_changepoint.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "params_B", "arch", "n_steps",
                    "tau_hat", "tau_lo_2p5", "tau_hi_97p5",
                    "pre_mean", "post_mean", "tau_contrast",
                    "early_minus_late_delta", "peak_step", "peak_val",
                    "peak_minus_late", "zero_frac",
                    "contrast_obs", "perm_p"])
        w.writerows(cp_rows)

    with open(RESULTS_DIR / "scaling_law_iter17_t_eps.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["model", "params_B", "arch", "n_steps",
                    "eps", "t_eps", "peak_step"])
        w.writerows(t_eps_rows)

    # (D) phase-label agreement
    def phase_table(rows: list[tuple[str, str]]) -> list[dict]:
        out = []
        for label, fname in rows:
            y = load_reward_per_step(TRACE_DIR / fname)
            if y is None or len(y) < 3:
                continue
            out.append({
                "model": label,
                "params_B": PARAM_B[label],
                "arch": ARCH[label],
                "n_steps": len(y),
                "phase_heuristic": phase_heuristic(y),
                "phase_changepoint": phase_changepoint(y),
                "phase_aic": phase_aic(y),
            })
        return out

    p5 = phase_table(list(MODELS_5.items()))
    p12 = phase_table(list(MODELS_12.items()))

    # per-trace phase labels
    with open(RESULTS_DIR / "scaling_law_iter17_phase_kappa.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["set", "model", "params_B", "arch", "n_steps",
                    "phase_heuristic", "phase_changepoint", "phase_aic"])
        for r in p5:
            w.writerow(["5anchor", r["model"], r["params_B"], r["arch"],
                        r["n_steps"], r["phase_heuristic"],
                        r["phase_changepoint"], r["phase_aic"]])
        for r in p12:
            w.writerow(["12anchor", r["model"], r["params_B"], r["arch"],
                        r["n_steps"], r["phase_heuristic"],
                        r["phase_changepoint"], r["phase_aic"]])

    # Cohen's kappa
    def kappa_block(p_set: list[dict], tag: str) -> list[list]:
        if not p_set:
            return []
        a = [r["phase_heuristic"] for r in p_set]
        b = [r["phase_changepoint"] for r in p_set]
        c = [r["phase_aic"] for r in p_set]
        return [
            [tag, "heuristic-vs-changepoint", cohens_kappa(a, b), len(p_set)],
            [tag, "heuristic-vs-aic", cohens_kappa(a, c), len(p_set)],
            [tag, "changepoint-vs-aic", cohens_kappa(b, c), len(p_set)],
        ]

    with open(RESULTS_DIR / "scaling_law_iter17_phase_kappa_only.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["set", "pair", "cohens_kappa", "n"])
        for r in kappa_block(p5, "5anchor") + kappa_block(p12, "12anchor"):
            w.writerow(r)

    # ----- figure -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Pillar 1 iter-17 elevation: AIC profile, changepoint significance, "
                 "plateau horizon, phase-label agreement", fontsize=11)

    # Panel A: delta-AIC stacked bars for each trace
    axA = axes[0, 0]
    model_order = ["constant", "linear", "saturation", "logistic"]
    bar_width = 0.18
    x = np.arange(len(p5))
    for i, mname in enumerate(model_order):
        vals = []
        for r in p5:
            rows_for_model = [row for row in aic_rows
                              if row[0] == r["model"] and row[4] == mname]
            if rows_for_model and math.isfinite(rows_for_model[0][9]):
                vals.append(rows_for_model[0][9])
            else:
                vals.append(float("nan"))
        axA.bar(x + i * bar_width, vals, bar_width, label=mname)
    axA.set_xticks(x + 1.5 * bar_width)
    axA.set_xticklabels([r["model"].replace("-Instruct", "") for r in p5],
                        rotation=15, ha="right")
    axA.set_ylabel("delta-AIC vs best (lower = better)")
    axA.set_title("(A) Multi-model AIC profile, 5 anchors")
    axA.legend(fontsize=8, loc="upper right")
    axA.axhline(2, color="gray", lw=0.5, ls="--", label="_2-unit line")

    # Panel B: changepoint tau with perm p-value
    axB = axes[0, 1]
    cps = []
    for row in cp_rows:
        label = row[0]
        tau = row[4]
        lo = row[5]
        hi = row[6]
        perm_p = row[16]
        if tau is None or tau < 0:
            continue
        cps.append((label, tau, lo, hi, max(0, tau - lo), max(1, hi - tau), perm_p))
    if cps:
        cpx = np.arange(len(cps))
        labels = [c[0].replace("-Instruct", "") for c in cps]
        taus = [c[1] for c in cps]
        errs_lo = [c[4] for c in cps]
        errs_hi = [c[5] for c in cps]
        axB.bar(cpx, taus, yerr=[errs_lo, errs_hi], capsize=4,
                color=["red" if c[6] < 0.05 else "C0" for c in cps])
        axB.set_xticks(cpx)
        axB.set_xticklabels(labels, rotation=15, ha="right")
    axB.set_ylabel("Changepoint tau [step]")
    axB.set_title("(B) Changepoint tau (red = perm p<0.05)")

    # Panel C: T_eps
    axC = axes[1, 0]
    eps_levels = (0.05, 0.10, 0.20)
    width = 0.25
    cx = np.arange(len(p5))
    for i, eps in enumerate(eps_levels):
        vals = []
        for r in p5:
            rows = [t for t in t_eps_rows
                    if t[0] == r["model"] and t[4] == eps and t[5] >= 0]
            vals.append(rows[0][5] if rows else float("nan"))
        axC.bar(cx + i * width, vals, width, label=f"eps={eps}")
    axC.set_xticks(cx + width)
    axC.set_xticklabels([r["model"].replace("-Instruct", "") for r in p5],
                        rotation=15, ha="right")
    axC.set_ylabel("T_eps [step]")
    axC.set_title("(C) Effective plateau horizon (model-free)")
    axC.legend(fontsize=8)

    # Panel D: kappa agreement
    axD = axes[1, 1]
    kappas5 = (cohens_kappa([r["phase_heuristic"] for r in p5],
                            [r["phase_changepoint"] for r in p5]),
               cohens_kappa([r["phase_heuristic"] for r in p5],
                            [r["phase_aic"] for r in p5]),
               cohens_kappa([r["phase_changepoint"] for r in p5],
                            [r["phase_aic"] for r in p5]))
    kappas12 = (cohens_kappa([r["phase_heuristic"] for r in p12],
                             [r["phase_changepoint"] for r in p12]),
                cohens_kappa([r["phase_heuristic"] for r in p12],
                             [r["phase_aic"] for r in p12]),
                cohens_kappa([r["phase_changepoint"] for r in p12],
                             [r["phase_aic"] for r in p12]))
    labels4d = ["heur-vs-cp", "heur-vs-aic", "cp-vs-aic"]
    cx4 = np.arange(3)
    axD.bar(cx4 - 0.18, kappas5, 0.35, label="5-anchor set")
    axD.bar(cx4 + 0.18, kappas12, 0.35, label="12-anchor set")
    axD.axhline(0, color="k", lw=0.5)
    axD.axhline(1, color="gray", lw=0.5, ls="--")
    axD.set_xticks(cx4)
    axD.set_xticklabels(labels4d, rotation=15, ha="right")
    axD.set_ylim(-0.5, 1.2)
    axD.set_ylabel("Cohen's kappa (-1..1)")
    axD.set_title("(D) Phase-label agreement")
    axD.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "scaling_law_iter17.pdf", bbox_inches="tight")
    fig.savefig(PAPER_FIG / "scaling_law_iter17.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "scaling_law_iter17.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ----- summary print -----
    print("\n=== Phase-label agreement (Cohen's kappa) ===")
    print(f"  5-anchor:    heur-vs-cp  = {kappas5[0]:+.3f}")
    print(f"  5-anchor:    heur-vs-aic = {kappas5[1]:+.3f}")
    print(f"  5-anchor:    cp-vs-aic   = {kappas5[2]:+.3f}")
    print(f"  12-anchor:   heur-vs-cp  = {kappas12[0]:+.3f}")
    print(f"  12-anchor:   heur-vs-aic = {kappas12[1]:+.3f}")
    print(f"  12-anchor:   cp-vs-aic   = {kappas12[2]:+.3f}")

    print("\n=== Best-AIC model + delta-AIC top runners (5 anchors) ===")
    winners = {}
    for row in aic_rows:
        if row[10] == 1:
            winners[row[0]] = (row[4], row[9])
    for r in p5:
        m, da = winners.get(r["model"], ("?", float("nan")))
        runner_up = sorted([(row[4], row[9]) for row in aic_rows
                            if row[0] == r["model"] and row[10] == 0],
                           key=lambda x: x[1])[0]
        print(f"  {r['model']:30s}: best={m:12s}  delta_2nd={runner_up[1]:.2f} "
              f"(runner-up = {runner_up[0]})")

    print("\n=== Changepoint tau + perm-test p-value (5 anchors) ===")
    for row in cp_rows:
        print(f"  {row[0]:30s}: tau={row[4]:>3d}  contrast={row[15]:.3f}  "
              f"perm_p={row[16]:.4f}")


if __name__ == "__main__":
    main()
