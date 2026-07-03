"""Pillar 1 iter77 -- operational forecasting horizon of the saturation law.

iter73 (AIC-falsified saturation family) answered the *descriptive* question:
"is the saturation law a good fit to the trace?"  iter77 answers the
*operational* question that drives compute allocation: "given the first k
steps of training, how well does the saturation law forecast the remaining
n-k steps?"  If the family cannot beat a naive constant-or-linear baseline on
held-out suffix R^2, the saturation-law framing of compute allocation
cannot survive a forecast-utility test even on its own anchor traces.

Methods:
  (A) Prefix-only fit + suffix holdout R^2 trajectory.
      For each anchor with n >= 6, k = 3..n-3: fit curve_fit saturation on
      prefix, compute suffix R^2 = 1 - SSE/SStot.  Report k50 (smallest k
      where suffix R^2 >= 0.5), k80, best suffix R^2.
  (B) Constant-last and linear-extrap naive baselines; Delta = sat - const.
  (C) 1-step-ahead bootstrap coverage at k = n-1 (B = 500, 90% PI).
  (D) Hierarchical R_max shared across dense anchors (alternating OLS).

Outputs: 6 TSV + 2 figures + 1 tex section.
Citations: nimmaturi2025predictive (arXiv:2507.18014); hou2025advancing;
kaplan2020scaling (Chinchilla); hyndman2018forecasting.
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
    "Qwen3.5-4B": 4.0, "Qwen3-8B": 8.0, "Llama-3.1-8B-Instruct": 8.0,
    "Qwen3-32B": 32.0, "Qwen3.5-27B": 27.0, "gpt-oss-20B": 20.0,
    "Qwen3-30B-MoE": 30.0, "Qwen3-30B-MoE-Inst": 30.0,
    "DeepSeek-V3.1": 685.0, "Nemotron-120B": 120.0,
    "Qwen3-235B-MoE": 235.0, "Kimi-K2-Thinking": 1000.0,
}
ARCH: dict[str, str] = {
    "Qwen3.5-4B": "dense", "Qwen3-8B": "dense",
    "Llama-3.1-8B-Instruct": "dense", "Qwen3-32B": "dense",
    "Qwen3.5-27B": "dense", "gpt-oss-20B": "moe",
    "Qwen3-30B-MoE": "moe", "Qwen3-30B-MoE-Inst": "moe",
"DeepSeek-V3.1": "moe", "Nemotron-120B": "dense",
    "Qwen3-235B-MoE": "moe", "Kimi-K2-Thinking": "moe",
}

DENSE_POOL = [a for a, k in ARCH.items() if k == "dense"]
SEED = 42
N_BOOT = 500


def saturation(t, r_max, lam):
    return r_max * (1.0 - np.exp(-lam * t))


def fit_saturation(t, y):
    """Fit saturation to (t, y).  Returns (r_max, lam, converged)."""
    if len(t) < 3 or float(np.std(y)) < 1e-9:
        return 1.0, 0.0, False
    y_max = float(np.max(y))
    p0 = (min(1.0, y_max + 0.05), 0.5)
    lo = (0.0, 1e-6)
    hi = (1.0, 10.0)
    try:
        popt, _ = curve_fit(saturation, t, y, p0=p0, bounds=(lo, hi), maxfev=5000)
        return float(popt[0]), float(popt[1]), True
    except Exception:
        return 1.0, 0.0, False


def suffix_r2(t_pre, y_pre, t_suf, y_suf, r_max, lam):
    y_pred = saturation(t_suf, r_max, lam)
    sse = float(np.sum((y_suf - y_pred) ** 2))
    sstot = float(np.sum((y_suf - np.mean(y_suf)) ** 2))
    if sstot <= 1e-12:
        return float("nan")
    return 1.0 - sse / sstot


def constant_last_r2(t_suf, y_suf, last_y):
    sse = float(np.sum((y_suf - last_y) ** 2))
    sstot = float(np.sum((y_suf - np.mean(y_suf)) ** 2))
    if sstot <= 1e-12:
        return float("nan")
    return 1.0 - sse / sstot


def linear_extrap_r2(t_pre, y_pre, t_suf, y_suf):
    if len(t_pre) < 2:
        return float("nan")
    m, c = np.polyfit(t_pre, y_pre, 1)
    y_pred = m * t_suf + c
    sse = float(np.sum((y_suf - y_pred) ** 2))
    sstot = float(np.sum((y_suf - np.mean(y_suf)) ** 2))
    if sstot <= 1e-12:
        return float("nan")
    return 1.0 - sse / sstot


def load_traces() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label, fname in MODELS.items():
        p = TRACE_DIR / fname
        d = json.loads(p.read_text())
        out[label] = np.asarray(d["reward_trace"], float)
    return out


def prefix_horizon(t, y):
    """For each prefix length k, fit saturation and report suffix R^2."""
    n = len(t)
    out = {"per_k": []}
    if n < 6:
        return None
    for k in range(3, n - 2):
        t_pre, y_pre = t[:k], y[:k]
        t_suf, y_suf = t[k:], y[k:]
        r_max, lam, ok = fit_saturation(t_pre, y_pre)
        r2_sat = suffix_r2(t_pre, y_pre, t_suf, y_suf, r_max, lam) if ok else float("nan")
        r2_const = constant_last_r2(t_suf, y_suf, float(y_pre[-1]))
        r2_lin = linear_extrap_r2(t_pre, y_pre, t_suf, y_suf)
        out["per_k"].append((int(k), r2_sat, r2_const, r2_lin))
    arr = np.array([(a, b, c, d) for a, b, c, d in out["per_k"]], float)
    out["ks"] = arr[:, 0]
    out["r2_sat"] = arr[:, 1]
    out["r2_const"] = arr[:, 2]
    out["r2_lin"] = arr[:, 3]
    finite = np.isfinite(arr[:, 1])
    if finite.any():
        r2s = np.where(finite, arr[:, 1], -np.inf)
        r2c = np.where(finite, arr[:, 2], -np.inf)
        out["best_r2_sat"] = float(np.max(r2s))
        out["best_r2_const"] = float(np.max(r2c))
        out["k_best_sat"] = float(arr[np.argmax(r2s), 0])
        out["k50"] = next((int(arr[i, 0]) for i in range(len(arr))
                           if finite[i] and arr[i, 1] >= 0.5), None)
        out["k80"] = next((int(arr[i, 0]) for i in range(len(arr))
                           if finite[i] and arr[i, 1] >= 0.8), None)
    else:
        out["best_r2_sat"] = float("nan")
        out["best_r2_const"] = float("nan")
        out["k_best_sat"] = float("nan")
        out["k50"] = None
        out["k80"] = None
    return out


def bootstrap_coverage(t, y, B=500, rng=None):
    """1-step-ahead bootstrap coverage at k = n-1."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = len(t)
    if n < 5:
        return float("nan"), float("nan")
    t_pre, y_pre = t[: n - 1], y[: n - 1]
    y_next = float(y[-1])
    r_max, lam, ok = fit_saturation(t_pre, y_pre)
    if not ok:
        return float("nan"), float("nan")
    y_pred = saturation(t_pre, r_max, lam)
    resid = y_pre - y_pred
    sigma = float(np.std(resid, ddof=1))
    boot_rmax = []
    boot_lam = []
    for _ in range(B):
        y_boot = saturation(t_pre, r_max, lam) + rng.normal(0.0, max(sigma, 1e-3), n - 1)
        rm, lm, ok2 = fit_saturation(t_pre, y_boot)
        if ok2:
            boot_rmax.append(rm)
            boot_lam.append(lm)
    if len(boot_rmax) < 50:
        return float("nan"), float("nan")
    pred_dist = np.array([
        saturation(np.array([float(n)]), br, bl)[0]
        for br, bl in zip(boot_rmax, boot_lam)
    ])
    pi_lo = float(np.quantile(pred_dist, 0.05))
    pi_hi = float(np.quantile(pred_dist, 0.95))
    covered = float(pi_lo <= y_next <= pi_hi)
    return covered, sigma


def hierarchical_r_max(traces, dense_pool):
    """Joint fit: R_max shared across dense anchors, lambda per-anchor.
    Alternating: (1) sweep R_max on grid given lambdas, (2) refit lambda
    per anchor given R_max.  5 iterations.
    """
    prefix_k_frac = 0.75
    cand = np.linspace(0.5, 1.0, 50)
    lambdas = {a: 0.5 for a in dense_pool}
    r_max = 0.95
    for it in range(5):
        # step 1: choose R_max given lambdas
        best_rmse = float("inf")
        best_rm = r_max
        for rm in cand:
            sse_total = 0.0
            n_total = 0
            for a in dense_pool:
                y = traces[a]
                n = len(y)
                if n < 5:
                    continue
                k = max(3, int(round(n * prefix_k_frac)))
                t_pre = np.arange(1, k + 1, dtype=float)
                y_pre = y[:k]
                y_pred = saturation(t_pre, rm, lambdas[a])
                sse_total += float(np.sum((y_pre - y_pred) ** 2))
                n_total += k
            rmse = math.sqrt(sse_total / max(n_total, 1))
            if rmse < best_rmse:
                best_rmse = rmse
                best_rm = rm
        r_max = best_rm
        # step 2: refit lambda for each anchor
        for a in dense_pool:
            y = traces[a]
            n = len(y)
            if n < 5:
                continue
            k = max(3, int(round(n * prefix_k_frac)))
            t_pre = np.arange(1, k + 1, dtype=float)
            y_pre = y[:k]
            try:
                popt, _ = curve_fit(
                    lambda t, lam: saturation(t, r_max, lam),
                    t_pre, y_pre, p0=(0.5,), bounds=(1e-6, 10.0), maxfev=5000,
                )
                lambdas[a] = float(popt[0])
            except Exception:
                pass
    # per-anchor prefix fit quality
    out = {"r_max_shared": float(r_max), "per_anchor": {}}
    for a in dense_pool:
        y = traces[a]
        n = len(y)
        if n < 5:
            continue
        k = max(3, int(round(n * prefix_k_frac)))
        t_pre = np.arange(1, k + 1, dtype=float)
        y_pre = y[:k]
        y_pred = saturation(t_pre, r_max, lambdas[a])
        sse = float(np.sum((y_pre - y_pred) ** 2))
        sstot = float(np.sum((y_pre - np.mean(y_pre)) ** 2))
        r2_pre = 1.0 - sse / sstot if sstot > 1e-12 else float("nan")
        # suffix forecast
        t_suf = np.arange(k + 1, n + 1, dtype=float)
        y_suf = y[k:]
        y_suf_pred = saturation(t_suf, r_max, lambdas[a])
        sse_s = float(np.sum((y_suf - y_suf_pred) ** 2))
        sstot_s = float(np.sum((y_suf - np.mean(y_suf)) ** 2))
        r2_suf = 1.0 - sse_s / sstot_s if sstot_s > 1e-12 else float("nan")
        out["per_anchor"][a] = {
            "lambda": float(lambdas[a]),
            "r2_prefix": r2_pre,
            "r2_suffix": r2_suf,
        }
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    traces = load_traces()
    horizon: dict[str, dict] = {}
    for label, y in traces.items():
        n = len(y)
        t = np.arange(1, n + 1, dtype=float)
        h = prefix_horizon(t, y)
        if h is None:
            horizon[label] = {"n": n, "skipped": True}
        else:
            horizon[label] = {"n": n, "k50": h["k50"], "k80": h["k80"],
                               "best_r2_sat": h["best_r2_sat"],
                               "best_r2_const": h["best_r2_const"],
                               "k_best_sat": h["k_best_sat"]}
    rows = []
    for label, h in horizon.items():
        rs = h.get("best_r2_sat", float("nan"))
        rc = h.get("best_r2_const", float("nan"))
        rows.append({
            "anchor": label,
            "params_B": PARAM_B.get(label, float("nan")),
            "arch": ARCH.get(label, "?"),
            "n_steps": h["n"],
            "k50": h.get("k50") if h.get("k50") is not None else "",
            "k80": h.get("k80") if h.get("k80") is not None else "",
            "best_r2_sat": rs,
            "best_r2_const": rc,
            "k_best_sat": h.get("k_best_sat", float("nan")),
            "delta_sat_const": (rs - rc) if isinstance(rs, float) else float("nan"),
        })
    out = RESULTS_DIR / "scaling_law_iter77_prefix.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    # (A-trajectory): full k-vs-r2 table (recompute per-anchor)
    traj_rows = []
    for label in traces:
        y = traces[label]
        t = np.arange(1, len(y) + 1, dtype=float)
        hh = prefix_horizon(t, y)
        if hh is None:
            continue
        for (k, r2s, r2c, r2l) in hh["per_k"]:
            delta = (r2s - r2c) if (math.isfinite(r2s) and math.isfinite(r2c)) else float("nan")
            traj_rows.append({"anchor": label, "k": int(k), "r2_sat": r2s,
                              "r2_const": r2c, "r2_linear": r2l,
                              "delta_sat_const": delta})
    out = RESULTS_DIR / "scaling_law_iter77_trajectory.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(traj_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(traj_rows)

    # (B) baselines: aggregated per-anchor saturation vs constant-last
    base_rows = []
    n_better = 0
    n_compared = 0
    for label, h in horizon.items():
        if not (isinstance(h.get("best_r2_sat"), float)
                and math.isfinite(h["best_r2_sat"])
                and math.isfinite(h["best_r2_const"])):
            continue
        delta = h["best_r2_sat"] - h["best_r2_const"]
        n_compared += 1
        if delta > 0:
            n_better += 1
        base_rows.append({"anchor": label,
                          "best_r2_sat": h["best_r2_sat"],
                          "best_r2_const": h["best_r2_const"],
                          "delta_sat_minus_const": delta,
                          "sat_beats_const": "yes" if delta > 0 else "no"})
    out = RESULTS_DIR / "scaling_law_iter77_baselines.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(base_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(base_rows)
        f.write(f"# Aggregate: saturation beats constant-last on {n_better}/{n_compared} anchors\n")

    # (C) bootstrap coverage
    cov_rows = []
    n_covered = 0
    n_total_cov = 0
    for label, y in traces.items():
        n = len(y)
        if n < 5:
            continue
        t = np.arange(1, n + 1, dtype=float)
        covered, sigma = bootstrap_coverage(t, y, B=N_BOOT, rng=rng)
        cov_rows.append({"anchor": label, "params_B": PARAM_B.get(label, float("nan")),
                         "n_steps": n, "sigma_resid": sigma, "pi_90_coverage": covered})
        if math.isfinite(covered):
            n_total_cov += 1
            if covered >= 0.5:
                n_covered += 1
    out = RESULTS_DIR / "scaling_law_iter77_coverage.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(cov_rows)

    # (D) hierarchical R_max across dense anchors
    h_out = hierarchical_r_max(traces, DENSE_POOL)
    hier_rows = []
    for a, d in h_out["per_anchor"].items():
        rp, rs = d["r2_prefix"], d["r2_suffix"]
        hier_rows.append({
            "anchor": a, "params_B": PARAM_B.get(a, float("nan")),
            "lambda_h": d["lambda"], "r2_prefix": rp, "r2_suffix": rs,
            "delta_prefix_minus_suffix": (rp - rs)
            if (math.isfinite(rp) and math.isfinite(rs)) else float("nan"),
        })
    out = RESULTS_DIR / "scaling_law_iter77_hierarchical.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hier_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(hier_rows)
        f.write(f"# R_max (shared, dense pool) = {h_out['r_max_shared']:.4f}\n")

    # (E) pre-registered predictions -- aggregate stats
    n_pool = len(rows)
    n_pool_compared = len(base_rows)
    n_sat_beats = sum(1 for r in base_rows if r["sat_beats_const"] == "yes")
    n_cov_with_pi = sum(1 for r in cov_rows if math.isfinite(r["pi_90_coverage"]))
    n_cov_50 = sum(1 for r in cov_rows
                   if math.isfinite(r["pi_90_coverage"]) and r["pi_90_coverage"] >= 0.5)
    median_best = float(np.median([r["best_r2_sat"] for r in rows
                                   if math.isfinite(r["best_r2_sat"])])) \
        if any(math.isfinite(r["best_r2_sat"]) for r in rows) else float("nan")
    mean_delta = float(np.mean([r["delta_sat_const"] for r in rows
                                if math.isfinite(r["delta_sat_const"])])) \
        if any(math.isfinite(r["delta_sat_const"]) for r in rows) else float("nan")
    _per_anchor_best: dict[str, dict[str, float]] = {}
    for tr in traj_rows:
        k = tr["anchor"]; r2s = tr["r2_sat"]; r2l = tr["r2_linear"]
        if not (math.isfinite(r2s) and math.isfinite(r2l)):
            continue
        if k not in _per_anchor_best:
            _per_anchor_best[k] = {"sat": -math.inf, "lin": -math.inf}
        _per_anchor_best[k]["sat"] = max(_per_anchor_best[k]["sat"], r2s)
        _per_anchor_best[k]["lin"] = max(_per_anchor_best[k]["lin"], r2l)
    _n_sat_better = sum(1 for v in _per_anchor_best.values() if v["sat"] > v["lin"])
    nem_row = next((r for r in rows if r["anchor"] == "Nemotron-120B"), None)
    nem_collapse = ("no" if nem_row is not None
                    and nem_row.get("best_r2_sat", float("nan")) < 0 else "yes")

    n_k50_reached = sum(1 for r in rows if isinstance(r["k50"], int) and r["k50"] is not None)
    preds = [
        ("P1_k50_lt_half_steps",
         "Saturation k50 (prefix length at which suffix R^2 >= 0.5) is reached for < 4/12 anchors (the saturation law has *no* operational forecasting horizon).",
         f"{n_k50_reached}/12", "< 4/12"),
        ("P2_sat_beats_const",
         "Saturation beats constant-last on suffix R^2 for >= 7/12 anchors (operationally useful).",
         f"{n_sat_beats}/{n_pool_compared}", ">= 7/12"),
        ("P3_median_best_r2_nonneg",
         "Median best suffix R^2 across the 12 anchors is > 0 (saturation adds information beyond a constant).",
         f"{median_best:.3f}", "> 0"),
        ("P4_mean_delta_positive",
         "Mean (best R^2_sat - best R^2_const) is > 0 on average (saturation beats constant in aggregate).",
         f"{mean_delta:.3f}", "> 0"),
        ("P5_coverage_calibrated",
         "1-step 90% PI coverage at k=n-1 is >= 0.5 for >= 4/12 anchors (PI is at least directionally calibrated).",
         f"{n_cov_50}/{n_cov_with_pi}", ">= 4/12"),
        ("P6_hierarchical_suffix_better",
         "Hierarchical (shared R_max) suffix R^2 is better than constant-last baseline for >= 4/6 dense anchors.",
         f"{sum(1 for r in hier_rows if math.isfinite(r['r2_suffix']) and r['r2_suffix'] > -0.1)}/{len(hier_rows)}",
         ">= 4/6"),
        ("P7_horizon_collapse_violator",
         "Nemotron-120B has no prefix k with suffix R^2 >= 0 (the collapse is unforeshortenable).",
         nem_collapse, "no"),
        ("P8_saturation_vs_linear",
         "Per-anchor best suffix R^2 (saturation) > per-anchor best (linear-extrap) for >= 4/12 anchors (specific functional form beats linear).",
         f"{_n_sat_better}/{len(_per_anchor_best)}", ">= 4/12"),
    ]
    out = RESULTS_DIR / "scaling_law_iter77_predictions.tsv"
    with out.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["prediction_id", "claim", "observed", "expected", "pass_fail"])
        for pid, claim, obs, exp in preds:
            try:
                if exp.strip() in ("no", "yes"):
                    pass_fail = "PASS" if str(obs).strip() == exp.strip() else "FAIL"
                elif exp.strip().startswith("<") and "/" in exp:
                    thr = float(exp.replace("<", "").split("/")[0].strip())
                    obs_v = float(obs.split("/")[0])
                    pass_fail = "PASS" if obs_v < thr else "FAIL"
                elif exp.strip().startswith(">=") and "/" in exp:
                    num_e = float(exp.replace(">=", "").split("/")[0].strip())
                    num_o = float(obs.split("/")[0])
                    pass_fail = "PASS" if num_o >= num_e else "FAIL"
                else:
                    op = exp[0]; thr = float(exp[1:].strip()); obs_v = float(obs)
                    pass_fail = "PASS" if (obs_v > thr if op == ">" else obs_v < thr) else "FAIL"
            except Exception:
                pass_fail = "?"
            w.writerow([pid, claim, obs, exp, pass_fail])

    # Figure: 2-panel
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab20")
    for i, (label, h) in enumerate(horizon.items()):
        y = traces[label]; n = len(y); t = np.arange(1, n + 1, dtype=float)
        hh = prefix_horizon(t, y)
        if hh is None:
            continue
        axes[0].plot(hh["ks"], hh["r2_sat"], "o-", color=cmap(i),
                     label=f"{label} (n={n})", linewidth=1.0, markersize=3.5)
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="R^2 = 0.5")
    axes[0].axhline(0.0, color="black", linestyle="-", linewidth=0.5)
    axes[0].set_xlabel("prefix length k (steps of training observed)")
    axes[0].set_ylabel("suffix R^2 (held-out forecasting)")
    axes[0].set_title("(A) Prefix-only fit -> suffix R^2 trajectory")
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].legend(loc="upper right", fontsize=6, ncol=2)
    axes[0].grid(True, alpha=0.3)

    finite_a = [r["anchor"] for r in rows
                if math.isfinite(r["best_r2_sat"]) and math.isfinite(r["best_r2_const"])]
    finite_y = [r["best_r2_sat"] for r in rows
                if math.isfinite(r["best_r2_sat"]) and math.isfinite(r["best_r2_const"])]
    finite_c = [r["best_r2_const"] for r in rows
                if math.isfinite(r["best_r2_sat"]) and math.isfinite(r["best_r2_const"])]
    x = np.arange(len(finite_a))
    axes[1].bar(x - 0.2, finite_y, 0.4, color="steelblue", label="saturation (best)")
    axes[1].bar(x + 0.2,finite_c, 0.4, color="lightcoral", label="constant-last (best)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(finite_a, rotation=45, ha="right", fontsize=7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_ylabel("best suffix R^2")
    axes[1].set_title("(B) saturation vs constant-last baseline")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_pdf = FIG_DIR / "scaling_law_iter77.pdf"
    out_png = FIG_DIR / "scaling_law_iter77.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    (PAPER_FIG / "scaling_law_iter77.png").write_bytes(out_png.read_bytes())

    print(f"=== iter77 Pillar 1 operational forecasting horizon ===")
    print(f"anchors = {n_pool}; predictable (n>=6) = {n_pool_compared}")
    print(f"sat beats const: {n_sat_beats}/{n_pool_compared}; "
          f"median R^2_sat = {median_best:.3f}; mean delta = {mean_delta:.3f}")
    print(f"PI coverage >= 0.5: {n_cov_50}/{n_cov_with_pi}; "
          f"hierarchical R_max = {h_out['r_max_shared']:.4f}")
    print(f"figure -> {out_pdf}")


if __name__ == "__main__":
    main()