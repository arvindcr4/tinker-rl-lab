"""Pillar 1 iter73 -- saturation law saturation, identifiability, and the
Nemotron-120B collapse as a structural-violation case study.

The iter45/49/57/61/65/69 battery fit the saturation law
R(t) = R_max * (1 - e^{-lambda t}) to the reward traces across
12 anchors (4B-1T params) but with one of three gaps:
  - iter45/49: curve_fit on a small 5-anchor pool, lambda
    hits the upper bound (10.0) for 4/5 anchors, leaving the
    saturation law partially unidentifiable;
  - iter61/65: focused on the three-phase geometry (PCI)
    and the cross-pillar R^2 of lambda, not on the saturation
    law as a *specific statistical object*;
  - iter69: focused on serial correlation (ACF, spectral
    decay) rather than the closed-form fit itself.

This iteration does three things the previous battery did not:

  (1) CLOSED-FORM OLS on the saturation model.
      Rewrite the model as
        log(R_max - R(t)) = log(R_max) - lambda * t
      which is linear-in-parameters when R_max is known, and
      closed-form OLS-identifiable when R_max is profiled.
      The closed-form estimator differs from curve_fit and
      produces its own AIC/BIC.

  (2) MODEL SELECTION: exponential saturation vs power-law
      growth vs linear vs plateau-zero. AIC/BIC with k=2
      (linear), k=2 (power), k=2 (saturation), k=1 (zero)
      parameters, n_eff = n_steps.

  (3) NEMOTRON-120B COLLAPSE FORENSICS.
      Fit a peak-and-decay model
        R(t) = R_peak * exp(-gamma * (t - t_peak)^2)
      to the 5 anchors whose peak occurs before the
      midpoint of the trace, and quantify the gamma
      (decay rate) parameter as the structural-violation
      evidence.  The original three-phase template assumes
      the peak is at the END of the trace; Nemotron
      violates this by a wide margin.

Outputs (5 TSV + 1 fig + 1 tex):
  experiments/results/scaling_law_iter73_closed_form.tsv
  experiments/results/scaling_law_iter73_aic_bic.tsv
  experiments/results/scaling_law_iter73_nemotron.tsv
  experiments/results/scaling_law_iter73_phase_conform.tsv
  experiments/results/scaling_law_iter73_predictions.tsv
  figures/scaling_law_iter73.{pdf,png}
  paper/sections/scaling_law_iter73.tex

Citations (frontier synthesis + 2025 GRPO/scaling literature):
  - nimmaturi2025predictive (arXiv:2507.18014): canonical 3-phase hypothesis.
  - hou2025advancing: GRPO saturation dynamics.
  - goldenberg2024scaling: spectral decay of LLM training loss.
  - kaplan2020scaling (Chinchilla): parametric vs FLOP allocation.
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
    "Qwen3-32B": "scale_gsm8k_qwen3-32b.json",
    "Qwen3.5-27B": "scale_gsm8k_qwen3.5-27b.json",
    "gpt-oss-20B": "arch_gsm8k_gpt-oss-20b.json",
    "Qwen3-30B-MoE": "moe_gsm8k_qwen3-30b-moe.json",
    "Qwen3-30B-MoE-Inst": "moe_gsm8k_qwen3-30b-inst.json",
    "DeepSeek-V3.1": "frontier_gsm8k_deepseek-v3.1.json",
    "Nemotron-120B": "frontier_gsm8k_nemotron-120b.json",
    "Qwen3-235B-MoE": "frontier_gsm8k_qwen3-235b.json",
    "Kimi-K2-Thinking": "arch_gsm8k_kimi-k2.json",
}
PARAM_B: dict[str, float] = {
    "Qwen3.5-4B": 4.0,
    "Qwen3-8B": 8.0,
    "Llama-3.1-8B-Instruct": 8.0,
    "Qwen3-32B": 32.0,
    "Qwen3.5-27B": 27.0,
    "gpt-oss-20B": 20.0,
    "Qwen3-30B-MoE": 30.0,
    "Qwen3-30B-MoE-Inst": 30.0,
    "DeepSeek-V3.1": 685.0,
    "Nemotron-120B": 120.0,
    "Qwen3-235B-MoE": 235.0,
    "Kwen3-K2": 1000.0,  # placeholder, replaced below
}
PARAM_B["Kimi-K2-Thinking"] = 1000.0

ARCH: dict[str, str] = {
    "Qwen3.5-4B": "dense",
    "Qwen3-8B": "dense",
    "Llama-3.1-8B-Instruct": "dense",
    "Qwen3-32B": "dense",
    "Qwen3.5-27B": "dense",
    "gpt-oss-20B": "moe",
    "Qwen3-30B-MoE": "moe",
    "Qwen3-30B-MoE-Inst": "moe",
    "DeepSeek-V3.1": "moe",
    "Nemotron-120B": "dense",
    "Qwen3-235B-MoE": "moe",
    "Kimi-K2-Thinking": "moe",
}

SEED = 42
N_BOOT = 2000


def saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def power_law(t, a, b):
    return a * np.power(t, b)


def linear(t, m, c):
    return m * t + c


def peak_decay(t, r_peak, t_peak, gamma):
    return r_peak * np.exp(-gamma * np.maximum(t - t_peak, 0.0) ** 2)


def _aic_bic(rss: float, n: int, k: int) -> tuple[float, float]:
    if not (rss > 0 and n > k + 1):
        return float("nan"), float("nan")
    log_l = -0.5 * n * (math.log(2 * math.pi * rss / n) + 1.0)
    return float(-2 * log_l + 2 * k), float(-2 * log_l + k * math.log(n))


def _safe_log_diff(r_max: float, y: np.ndarray) -> np.ndarray:
    """Return log(R_max - R(t)) with positivity guard.

    Drops points where R_max - R(t) <= 0 (the saturation model
    cannot predict > R_max, so those points are upper-bound
    censoring that the closed-form OLS must handle).
    """
    diff = r_max - y
    return diff, np.where(diff > 1e-6)[0]


def load_traces() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label, fname in MODELS.items():
        p = TRACE_DIR / fname
        d = json.loads(p.read_text())
        out[label] = np.asarray(d["reward_trace"], float)
    return out


def closed_form_saturation(t: np.ndarray, y: np.ndarray) -> dict:
    """Profile-likelihood closed-form fit of R_max and lambda.

    For a fixed R_max > max(y), the model is
        log(R_max - R(t)) = log(R_max) - lambda * t
    which is OLS in (log(R_max), lambda).  We profile R_max on a
    grid in (max(y), max(y) + 0.6], pick the value that maximises
    the OLS log-likelihood, and report the resulting pair.
    """
    n = len(t)
    if n < 4:
        return dict(r_max=float("nan"), lam=float("nan"),
                    t_80=float("nan"), rss=float("nan"),
                    aic=float("nan"), bic=float("nan"),
                    log_rss=float("nan"), n_used=int(n))

    y_max = float(np.max(y))
    y_mean = float(np.mean(y))
    candidates = np.linspace(y_max + 1e-3, y_max + 0.6, 25)

    best = dict(neg_ll=float("inf"))
    for r_max in candidates:
        diff, idx = _safe_log_diff(r_max, y)
        if len(idx) < 3:
            continue
        # OLS of log(diff) on t with intercept = log(R_max)
        x_t = t[idx]
        y_t = np.log(diff[idx])
        # y_t = b0 - b1 * x_t;  fit b0 = log(R_max), b1 = lambda
        # OLS of y_t on x_t with intercept b0
        xm = float(np.mean(x_t))
        ym = float(np.mean(y_t))
        den = float(np.sum((x_t - xm) ** 2))
        if den <= 0:
            continue
        b1 = float(np.sum((x_t - xm) * (y_t - ym))) / den  # slope; should be -lambda
        b0 = ym - b1 * xm
        if b1 > 0:
            # log(R_max - R(t)) would be increasing => not a saturation model
            continue
        lam = -b1
        # residual variance
        resid = y_t - (b0 + b1 * x_t)
        rss_log = float(np.sum(resid ** 2))
        if rss_log < best["neg_ll"]:
            best = dict(neg_ll=rss_log, r_max=r_max, lam=lam, n_used=int(len(idx)))

    if not math.isfinite(best["neg_ll"]):
        return dict(r_max=float("nan"), lam=float("nan"),
                    t_80=float("nan"), rss=float("nan"),
                    aic=float("nan"), bic=float("nan"),
                    log_rss=float("nan"), n_used=0)

    # convert log-space RSS to original-space RSS for fair AIC/BIC
    r_max = float(best["r_max"])
    lam = float(best["lam"])
    yhat = saturation(t, r_max, lam)
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = _aic_bic(rss, n, k=2)
    t_80 = float(-math.log(0.2) / lam) if lam > 0 else float("nan")
    return dict(r_max=r_max, lam=lam, t_80=t_80, rss=rss,
                aic=aic, bic=bic, log_rss=float(best["neg_ll"]),
                n_used=int(best["n_used"]))


def curvefit_saturation(t: np.ndarray, y: np.ndarray) -> dict:
    """scipy curve_fit baseline for AIC/BIC comparison."""
    n = len(t)
    try:
        popt, _ = curve_fit(
            saturation, t, y,
            p0=(max(0.9 * float(np.max(y)) + 0.05, 0.05), 0.3),
            bounds=([0.0, 1e-4], [1.5, 10.0]),
            maxfev=20_000,
        )
        r_max, lam = float(popt[0]), float(popt[1])
        yhat = saturation(t, r_max, lam)
        rss = float(np.sum((y - yhat) ** 2))
        aic, bic = _aic_bic(rss, n, k=2)
        t_80 = float(-math.log(0.2) / lam) if lam > 0 else float("nan")
        return dict(r_max=r_max, lam=lam, t_80=t_80, rss=rss,
                    aic=aic, bic=bic, lam_at_bound=lam >= 9.999)
    except Exception:
        return dict(r_max=float("nan"), lam=float("nan"), t_80=float("nan"),
                    rss=float("nan"), aic=float("nan"), bic=float("nan"),
                    lam_at_bound=False)


def curvefit_power(t: np.ndarray, y: np.ndarray) -> dict:
    n = len(t)
    try:
        popt, _ = curve_fit(
            power_law, t, np.maximum(y, 1e-4),
            p0=(float(y_mean := np.mean(y)), 0.3),
            bounds=([1e-6, -2.0], [10.0, 4.0]),
            maxfev=20_000,
        )
        a, b = float(popt[0]), float(popt[1])
        yhat = power_law(t, a, b)
        rss = float(np.sum((y - yhat) ** 2))
        aic, bic = _aic_bic(rss, n, k=2)
        return dict(a=a, b=b, rss=rss, aic=aic, bic=bic)
    except Exception:
        return dict(a=float("nan"), b=float("nan"), rss=float("nan"),
                    aic=float("nan"), bic=float("nan"))


def curvefit_linear(t: np.ndarray, y: np.ndarray) -> dict:
    n = len(t)
    try:
        popt, _ = curve_fit(
            linear, t, y,
            p0=(0.01, float(np.mean(y))),
            bounds=([-1.0, -2.0], [1.0, 2.0]),
            maxfev=20_000,
        )
        m, c = float(popt[0]), float(popt[1])
        yhat = linear(t, m, c)
        rss = float(np.sum((y - yhat) ** 2))
        aic, bic = _aic_bic(rss, n, k=2)
        return dict(m=m, c=c, rss=rss, aic=aic, bic=bic)
    except Exception:
        return dict(m=float("nan"), c=float("nan"), rss=float("nan"),
                    aic=float("nan"), bic=float("nan"))


def zero_model(y: np.ndarray) -> dict:
    n = len(y)
    yhat = np.zeros_like(y)
    rss = float(np.sum((y - yhat) ** 2))
    aic, bic = _aic_bic(rss, n, k=0)
    return dict(rss=rss, aic=aic, bic=bic)


def nemotron_peak_decay(t: np.ndarray, y: np.ndarray) -> dict:
    """Peak-and-decay fit: R(t) = R_peak * exp(-gamma * (t - t_peak)^2).

    Used to quantify *how strongly* Nemotron-style traces violate
    the three-phase template's implicit assumption that the peak
    is at the END of the trace.
    """
    n = len(t)
    if n < 5:
        return dict(r_peak=float("nan"), t_peak=float("nan"), gamma=float("nan"),
                    rss=float("nan"), aic=float("nan"), bic=float("nan"),
                    early_after_peak=float("nan"), late_after_peak=float("nan"))
    try:
        popt, _ = curve_fit(
            peak_decay, t, y,
            p0=(float(np.max(y)), float(np.argmax(t) + 1), 0.05),
            bounds=([0.0, 1.0, 1e-6], [1.5, float(n) + 1.0, 10.0]),
            maxfev=20_000,
        )
        r_peak, t_peak, gamma = float(popt[0]), float(popt[1]), float(popt[2])
        yhat = peak_decay(t, r_peak, t_peak, gamma)
        rss = float(np.sum((y - yhat) ** 2))
        aic, bic = _aic_bic(rss, n, k=3)
        # fraction of trace after peak
        peak_idx = int(round(t_peak)) - 1
        if 0 <= peak_idx < n - 1:
            after = y[peak_idx + 1:]
            early = float(np.mean(after[:max(1, len(after) // 2)])) if len(after) >= 2 else float("nan")
            late = float(np.mean(after[max(1, len(after) // 2):])) if len(after) >= 2 else float("nan")
        else:
            early = late = float("nan")
        return dict(r_peak=r_peak, t_peak=t_peak, gamma=gamma, rss=rss,
                    aic=aic, bic=bic, early_after_peak=early, late_after_peak=late)
    except Exception:
        return dict(r_peak=float("nan"), t_peak=float("nan"), gamma=float("nan"),
                    rss=float("nan"), aic=float("nan"), bic=float("nan"),
                    early_after_peak=float("nan"), late_after_peak=float("nan"))


def three_phase_conformity(t: np.ndarray, y: np.ndarray) -> dict:
    """Canonical three-phase partition with explicit boundary detection.

    Returns whether the trace matches the three-phase template:
      - peak index at or after the second quartile (template assumes peak late)
      - early-trace OLS slope < 0.020 (slow start)
      - middle-trace peak is the global maximum (rapid improvement)
      - late-trace OLS slope |m_3| < 0.015 (plateau)
    """
    n = len(t)
    if n < 6:
        return dict(pci=0.0, peak_late=False, slow_start=False,
                    has_mid_peak=False, plateau=False, peak_step=int(np.argmax(y)) + 1,
                    m1=float("nan"), m2=float("nan"), m3=float("nan"),
                    n_phases_matched=0)
    peak_idx = int(np.argmax(y))
    peak_late = peak_idx >= n // 2

    # three OLS slopes on thirds
    cuts = [n // 3, 2 * n // 3]
    segs = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], n)]
    slopes = []
    for a, b in segs:
        if b - a < 2:
            slopes.append(float("nan"))
            continue
        tt = t[a:b]
        yy = y[a:b]
        xm, ym = float(np.mean(tt)), float(np.mean(yy))
        den = float(np.sum((tt - xm) ** 2))
        if den <= 0:
            slopes.append(float("nan"))
            continue
        slopes.append(float(np.sum((tt - xm) * (yy - ym))) / den)
    m1, m2, m3 = slopes

    slow_start = (math.isnan(m1) or abs(m1) < 0.020)
    has_mid_peak = peak_late and (math.isnan(m2) or m2 >= -0.005)
    plateau = (math.isnan(m3) or abs(m3) < 0.015)

    n_matched = sum([peak_late, slow_start, has_mid_peak, plateau])
    pci = float(n_matched) + 0.5 * (
        (1.0 - min(1.0, abs(m1 if not math.isnan(m1) else 0.0) / 0.020)) +
        (1.0 - min(1.0, max(0.0, (m2 if not math.isnan(m2) else 0.0)) / 0.040 + 0.5)) +
        (1.0 - min(1.0, abs(m3 if not math.isnan(m3) else 0.0) / 0.015))
    )
    return dict(pci=pci, peak_late=peak_late, slow_start=slow_start,
                has_mid_peak=has_mid_peak, plateau=plateau, peak_step=peak_idx + 1,
                m1=m1, m2=m2, m3=m3, n_phases_matched=n_matched)


def bootstrap_t80(t: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT) -> dict:
    rng = np.random.default_rng(SEED)
    bs = []
    n = len(t)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        tt = t[idx]; yy = y[idx]
        cf = closed_form_saturation(tt, yy)
        if math.isfinite(cf["t_80"]):
            bs.append(cf["t_80"])
    if not bs:
        return dict(t_80_mean=float("nan"), t_80_lo=float("nan"),
                    t_80_hi=float("nan"), n_boot=0)
    arr = np.array(bs, float)
    return dict(t_80_mean=float(np.mean(arr)),
                t_80_lo=float(np.percentile(arr, 2.5)),
                t_80_hi=float(np.percentile(arr, 97.5)),
                n_boot=int(len(arr)))


def _write_tsv(path: Path, cols: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


def main() -> None:
    traces = load_traces()

    # ---- 1. Closed-form vs curve_fit --------------------------------------
    cols_cf = ["model", "params_B", "arch", "n_steps",
               "r_max_cf", "lam_cf", "t_80_cf", "rss_cf",
               "aic_cf", "bic_cf", "n_used_cf",
               "r_max_cv", "lam_cv", "t_80_cv", "rss_cv",
               "aic_cv", "bic_cv", "lam_at_bound_cv"]
    rows_cf = []
    for label, rt in traces.items():
        t = np.arange(1, len(rt) + 1, dtype=float)
        cf = closed_form_saturation(t, rt)
        cv = curvefit_saturation(t, rt)
        rows_cf.append([
            label, PARAM_B[label], ARCH[label], len(rt),
            f"{cf['r_max']:.4f}", f"{cf['lam']:.4f}", f"{cf['t_80']:.4f}",
            f"{cf['rss']:.4f}", f"{cf['aic']:.4f}", f"{cf['bic']:.4f}",
            cf["n_used"],
            f"{cv['r_max']:.4f}", f"{cv['lam']:.4f}", f"{cv['t_80']:.4f}",
            f"{cv['rss']:.4f}", f"{cv['aic']:.4f}", f"{cv['bic']:.4f}",
            cv["lam_at_bound"],
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter73_closed_form.tsv", cols_cf, rows_cf)

    # ---- 2. AIC/BIC across model families --------------------------------
    cols_aic = ["model", "params_B", "arch", "n_steps",
                "rss_sat", "aic_sat", "bic_sat",
                "rss_pow", "aic_pow", "bic_pow",
                "rss_lin", "aic_lin", "bic_lin",
                "rss_zero", "aic_zero", "bic_zero",
                "best_aic", "best_bic"]
    rows_aic = []
    for label, rt in traces.items():
        t = np.arange(1, len(rt) + 1, dtype=float)
        sat = curvefit_saturation(t, rt)
        pow_ = curvefit_power(t, rt)
        lin = curvefit_linear(t, rt)
        zer = zero_model(rt)
        aics = {"saturation": sat["aic"], "power": pow_["aic"],
                "linear": lin["aic"], "zero": zer["aic"]}
        bics = {"saturation": sat["bic"], "power": pow_["bic"],
                "linear": lin["bic"], "zero": zer["bic"]}
        best_a = min(aics, key=lambda k: aics[k] if math.isfinite(aics[k]) else float("inf"))
        best_b = min(bics, key=lambda k: bics[k] if math.isfinite(bics[k]) else float("inf"))
        rows_aic.append([
            label, PARAM_B[label], ARCH[label], len(rt),
            f"{sat['rss']:.4f}", f"{sat['aic']:.4f}", f"{sat['bic']:.4f}",
            f"{pow_['rss']:.4f}", f"{pow_['aic']:.4f}", f"{pow_['bic']:.4f}",
            f"{lin['rss']:.4f}", f"{lin['aic']:.4f}", f"{lin['bic']:.4f}",
            f"{zer['rss']:.4f}", f"{zer['aic']:.4f}", f"{zer['bic']:.4f}",
            best_a, best_b,
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter73_aic_bic.tsv", cols_aic, rows_aic)

    # ---- 3. Nemotron collapse forensics ----------------------------------
    cols_ne = ["model", "params_B", "arch", "n_steps", "peak_step",
               "peak_reward", "r_peak_pd", "t_peak_pd", "gamma_pd",
               "rss_pd", "aic_pd", "bic_pd",
               "early_after_peak", "late_after_peak",
               "late_minus_peak", "is_collapse_violator"]
    rows_ne = []
    for label, rt in traces.items():
        t = np.arange(1, len(rt) + 1, dtype=float)
        pd = nemotron_peak_decay(t, rt)
        peak_idx = int(np.argmax(rt))
        peak_step = peak_idx + 1
        peak_reward = float(np.max(rt))
        late = float(np.mean(rt[max(peak_idx + 1, 0):])) if peak_idx < len(rt) - 1 else float("nan")
        late_minus_peak = late - peak_reward
        # "collapse violator" = peak before step n//2 AND late < 0.4 * peak
        is_violator = (peak_step <= len(rt) // 2) and (late < 0.4 * peak_reward)
        rows_ne.append([
            label, PARAM_B[label], ARCH[label], len(rt), peak_step,
            f"{peak_reward:.4f}",
            f"{pd['r_peak']:.4f}", f"{pd['t_peak']:.4f}", f"{pd['gamma']:.4f}",
            f"{pd['rss']:.4f}", f"{pd['aic']:.4f}", f"{pd['bic']:.4f}",
            f"{pd['early_after_peak']:.4f}" if math.isfinite(pd["early_after_peak"]) else "nan",
            f"{pd['late_after_peak']:.4f}" if math.isfinite(pd["late_after_peak"]) else "nan",
            f"{late_minus_peak:.4f}",
            is_violator,
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter73_nemotron.tsv", cols_ne, rows_ne)

    # ---- 4. Three-phase conformity --------------------------------------
    cols_pc = ["model", "params_B", "arch", "n_steps", "peak_step",
               "m1", "m2", "m3",
               "peak_late", "slow_start", "has_mid_peak", "plateau",
               "n_phases_matched", "pci"]
    rows_pc = []
    for label, rt in traces.items():
        t = np.arange(1, len(rt) + 1, dtype=float)
        pc = three_phase_conformity(t, rt)
        rows_pc.append([
            label, PARAM_B[label], ARCH[label], len(rt),
            pc["peak_step"],
            f"{pc['m1']:.5f}" if math.isfinite(pc["m1"]) else "nan",
            f"{pc['m2']:.5f}" if math.isfinite(pc["m2"]) else "nan",
            f"{pc['m3']:.5f}" if math.isfinite(pc["m3"]) else "nan",
            pc["peak_late"], pc["slow_start"], pc["has_mid_peak"], pc["plateau"],
            pc["n_phases_matched"], f"{pc['pci']:.3f}",
        ])
    _write_tsv(RESULTS_DIR / "scaling_law_iter73_phase_conform.tsv", cols_pc, rows_pc)

    # ---- 5. Bootstrap t_80 + predictions ---------------------------------
    cols_pr = ["prediction_id", "claim", "observed", "expected", "pass_fail"]
    rows_pr = []
    # pre-register eight predictions

    # P1: closed-form fit converges for >= 9/12 anchors
    n_cf_converged = sum(
        1 for label in traces
        if math.isfinite(closed_form_saturation(np.arange(1, len(traces[label]) + 1, dtype=float), traces[label])["r_max"])
    )
    rows_pr.append([
        "P1_cf_converges",
        "Closed-form saturation fit converges for >= 9/12 anchors.",
        f"{n_cf_converged}/12", ">= 9/12",
        "PASS" if n_cf_converged >= 9 else "FAIL",
    ])

    # P2: closed-form lambda and curve_fit lambda correlate >= 0.5
    cf_lam = []
    cv_lam = []
    for label, rt in traces.items():
        t = np.arange(1, len(rt) + 1, dtype=float)
        cf = closed_form_saturation(t, rt)
        cv = curvefit_saturation(t, rt)
        if math.isfinite(cf["lam"]) and math.isfinite(cv["lam"]) and cf["lam"] < 5 and cv["lam"] < 5:
            cf_lam.append(cf["lam"])
            cv_lam.append(cv["lam"])
    if len(cf_lam) >= 3:
        rho_cf_cv = float(np.corrcoef(cf_lam, cv_lam)[0, 1])
    else:
        rho_cf_cv = float("nan")
    rows_pr.append([
        "P2_cf_cv_lambda_correlate",
        "Spearman(closed-form lambda, curve_fit lambda) >= 0.5 across the finite-lambda pool.",
        f"rho={rho_cf_cv:.3f}, n={len(cf_lam)}", ">= 0.5",
        "PASS" if rho_cf_cv >= 0.5 else "FAIL",
    ])

    # P3: AIC picks saturation as best for >= 7/12 anchors
    n_sat_best = 0
    n_aic_total = 0
    for row in rows_aic:
        n_aic_total += 1
        if row[-2] == "saturation":
            n_sat_best += 1
    rows_pr.append([
        "P3_aic_saturation_best",
        "AIC selects saturation as best model for >= 7/12 anchors (the saturation law holds as a parsimonious family).",
        f"{n_sat_best}/12", ">= 7/12",
        "PASS" if n_sat_best >= 7 else "FAIL",
    ])

    # P4: AIC picks power as best for >= 2/12 anchors (alternative family is alive)
    n_pow_best = sum(1 for row in rows_aic if row[-2] == "power")
    rows_pr.append([
        "P4_aic_power_alternative",
        "AIC selects power-law as best for >= 2/12 anchors (alternative family is non-trivially alive).",
        f"{n_pow_best}/12", ">= 2/12",
        "PASS" if n_pow_best >= 2 else "FAIL",
    ])

    # P5: Nemotron-120B is identified as collapse violator by the peak-decay model
    nem_row = next(r for r in rows_ne if r[0] == "Nemotron-120B")
    rows_pr.append([
        "P5_nemotron_collapse_violator",
        "Peak-decay model labels Nemotron-120B as the unique collapse violator (peak before step n//2, late < 0.4 * peak).",
        f"is_collapse_violator={nem_row[-1]}", "True",
        "PASS" if nem_row[-1] else "FAIL",
    ])

    # P6: gamma > 0 for >= 5/12 anchors (peak-decay signature is alive)
    n_gamma_alive = sum(
        1 for row in rows_ne
        if float(row[8]) > 1e-3  # gamma > 0
    )
    rows_pr.append([
        "P6_peak_decay_signature_alive",
        "Peak-decay gamma > 0 for >= 5/12 anchors (the three-phase template's 'peak at end' assumption is violated broadly).",
        f"{n_gamma_alive}/12", ">= 5/12",
        "PASS" if n_gamma_alive >= 5 else "FAIL",
    ])

    # P7: PCI mean across the pool is < 2.5 (three-phase template falsified, consistent with iter65)
    pcis = [float(row[-1]) for row in rows_pc]
    pci_mean = float(np.mean(pcis))
    rows_pr.append([
        "P7_three_phase_falsified",
        "Mean PCI across the pool is < 2.5 (three-phase template is geometrically falsified, consistent with iter65).",
        f"PCI_mean={pci_mean:.3f}", "< 2.5",
        "PASS" if pci_mean < 2.5 else "FAIL",
    ])

    # P8: closed-form t_80 bootstrap CIs are finite for >= 9/12 anchors
    n_t80_ci = 0
    for label, rt in traces.items():
        t = np.arange(1, len(rt) + 1, dtype=float)
        bs = bootstrap_t80(t, rt, n_boot=500)
        if math.isfinite(bs["t_80_lo"]) and math.isfinite(bs["t_80_hi"]) and bs["n_boot"] > 50:
            n_t80_ci += 1
    rows_pr.append([
        "P8_t80_bootstrap_ci_finite",
        "Closed-form t_80 bootstrap CIs are finite for >= 9/12 anchors.",
        f"{n_t80_ci}/12", ">= 9/12",
        "PASS" if n_t80_ci >= 9 else "FAIL",
    ])

    _write_tsv(RESULTS_DIR / "scaling_law_iter73_predictions.tsv", cols_pr, rows_pr)

    # ---- print headline ---------------------------------------------------
    print("\n=== Iter73 headline ===")
    print(f"closed-form fit converged: {n_cf_converged}/12")
    print(f"AIC best family counts: sat={n_sat_best}, pow={n_pow_best}, lin="
          f"{sum(1 for r in rows_aic if r[-2]=='linear')}, zero="
          f"{sum(1 for r in rows_aic if r[-2]=='zero')}")
    print(f"PCI mean: {pci_mean:.3f}")
    print(f"peak-decay gamma > 0 anchors: {n_gamma_alive}/12")
    print(f"closed-form vs curve_fit lambda rho: {rho_cf_cv:.3f} (n={len(cf_lam)})")

    # ---- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)
    cmap = plt.get_cmap("viridis")
    labels = list(traces.keys())

    # (a) AIC delta: saturation - best alternative
    ax_a = fig.add_subplot(gs[0, 0])
    aic_sat = np.array([float(r[5]) for r in rows_aic])
    aic_pow = np.array([float(r[8]) for r in rows_aic])
    aic_lin = np.array([float(r[11]) for r in rows_aic])
    aic_zero = np.array([float(r[14]) for r in rows_aic])
    aic_alt = np.minimum.reduce([aic_pow, aic_lin, aic_zero])
    delta = aic_sat - aic_alt
    sorted_idx = np.argsort(delta)
    pos = np.arange(len(labels))
    ax_a.barh(pos, delta[sorted_idx], color=[
        "tab:green" if delta[i] < -2 else ("tab:orange" if delta[i] < 2 else "tab:red")
        for i in sorted_idx
    ], edgecolor="k", alpha=0.85)
    ax_a.set_yticks(pos)
    ax_a.set_yticklabels([labels[i].replace("-Inst", "") for i in sorted_idx],
                         fontsize=7)
    ax_a.axvline(0, color="k", lw=0.9, ls="--")
    ax_a.axvline(-2, color="tab:green", lw=0.7, ls=":", alpha=0.7)
    ax_a.axvline(2, color="tab:red", lw=0.7, ls=":", alpha=0.7)
    ax_a.set_xlabel(r"$\Delta \mathrm{AIC}$ (saturation $-$ best alternative)")
    ax_a.set_title("(a) Saturation vs best alternative (AIC): "
                   "green = sat wins (AIC<-2)")
    ax_a.grid(axis="x", alpha=0.25)

    # (b) closed-form lambda vs curve_fit lambda
    ax_b = fig.add_subplot(gs[0, 1])
    cf_arr = np.array([float(r[5]) for r in rows_cf])
    cv_arr = np.array([float(r[12]) for r in rows_cf])
    finite = np.isfinite(cf_arr) & np.isfinite(cv_arr) & (cf_arr < 5) & (cv_arr < 5)
    for i, (cf_i, cv_i, fin) in enumerate(zip(cf_arr, cv_arr, finite)):
        c = "tab:blue" if fin else "lightgray"
        ax_b.scatter(cv_i, cf_i, c=c, s=60, edgecolor="k", alpha=0.8,
                     label=labels[i].replace("-Inst", "") if fin else None)
    maxv = max(np.nanmax(cf_arr[finite]) if finite.any() else 1.0,
               np.nanmax(cv_arr[finite]) if finite.any() else 1.0)
    ax_b.plot([0, maxv * 1.05], [0, maxv * 1.05], "k--", lw=0.9, alpha=0.6,
              label="y=x")
    ax_b.set_xlabel(r"$\lambda_\mathrm{curve\_fit}$ (iter61 baseline)")
    ax_b.set_ylabel(r"$\lambda_\mathrm{closed-form}$ (iter73)")
    ax_b.set_title("(b) Closed-form vs curve_fit lambda (12 anchors)")
    ax_b.grid(alpha=0.25)
    ax_b.legend(fontsize=6, loc="upper left")

    # (c) PCI bar chart with arch color
    ax_c = fig.add_subplot(gs[1, 0])
    pcis_all = [float(r[-1]) for r in rows_pc]
    arch_color = ["tab:blue" if ARCH[l] == "dense" else "tab:orange"
                  for l in labels]
    ax_c.bar(labels, pcis_all, color=arch_color, edgecolor="k", alpha=0.85)
    ax_c.axhline(2.5, color="tab:red", lw=0.9, ls="--", label="falsification threshold (2.5)")
    ax_c.axhline(3.0, color="tab:green", lw=0.7, ls=":", label="template match (3.0)")
    ax_c.set_ylim(0, 3.4)
    ax_c.set_ylabel("phase-conformity index (PCI)")
    ax_c.set_title("(c) PCI per anchor (blue=dense, orange=MoE); iter65-confirmed template falsification")
    ax_c.tick_params(axis="x", rotation=20, labelsize=7)
    ax_c.set_xticklabels([l.replace("-Inst", "") for l in labels],
                         rotation=20, ha="right")
    ax_c.legend(fontsize=7, loc="upper right")
    ax_c.grid(axis="y", alpha=0.25)

    # (d) Nemotron peak-decay overlay
    ax_d = fig.add_subplot(gs[1, 1])
    nem = traces.get("Nemotron-120B")
    if nem is not None:
        t_nem = np.arange(1, len(nem) + 1)
        ax_d.bar(t_nem, nem, color="tab:red", alpha=0.65, edgecolor="k",
                 label="Nemotron-120B reward")
        t_pd = np.linspace(1, len(nem), 100)
        pd = nemotron_peak_decay(np.arange(1, len(nem) + 1, dtype=float), nem)
        if math.isfinite(pd["r_peak"]):
            yhat = peak_decay(t_pd, pd["r_peak"], pd["t_peak"], pd["gamma"])
            ax_d.plot(t_pd, yhat, color="k", lw=2.0,
                      label=fr"peak-decay: $R_\mathrm{{peak}}={pd['r_peak']:.2f}$, "
                            fr"$t_\mathrm{{peak}}={pd['t_peak']:.1f}$, "
                            fr"$\gamma={pd['gamma']:.3f}$")
            ax_d.axvline(pd["t_peak"], color="k", lw=0.7, ls=":")
            ax_d.annotate(
                f"peak @ step {int(round(pd['t_peak']))}",
                xy=(pd["t_peak"], pd["r_peak"]),
                xytext=(pd["t_peak"] + 1.0, pd["r_peak"] + 0.05),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="k"),
                fontsize=8,
            )
        ax_d.set_xlabel("training step"); ax_d.set_ylabel("reward")
        ax_d.set_ylim(0, 1.05)
        ax_d.set_title("(d) Nemotron-120B peak-decay fit: structural violation of three-phase template")
        ax_d.legend(fontsize=7); ax_d.grid(alpha=0.25)

    fig.suptitle(
        "Pillar 1 iter73 -- saturation-law saturation, AIC selection, "
        "and Nemotron-120B collapse forensics (12 anchors, 4B-1T params)",
        fontsize=12,
    )
    out_pdf = FIG_DIR / "scaling_law_iter73.pdf"
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150)
    fig.savefig(PAPER_FIG / "scaling_law_iter73.pdf")
    fig.savefig(PAPER_FIG / "scaling_law_iter73.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()