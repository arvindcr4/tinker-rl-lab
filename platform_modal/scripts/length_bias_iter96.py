#!/usr/bin/env python3
"""Iter 96 -- Pillar 4 (Length Bias / Dr.GRPO): Innovation Cross-Correlation
Asymmetry (ICCA).

Iter 80 (OU / unit-root) characterised the LEVEL dynamics of L_t.  Iter 84
(Welch coherence + Granger F + DFA Hurst) decomposed the linear coupling.
Iter 88 (quantile regression) localised the conditional-mean slope.
Iter 92 (transfer entropy) measured the *directed information* L<->R.

Iter 96 attacks a different observable that is uniquely suited to the
Dr.GRPO mechanism: the **innovation cross-correlation asymmetry** of the
AR(1)-filtered (L_t, R_t) process.

IDEA.  Fit a *marginal* AR(1) to each series to strip the level dynamics
that iter80 showed are strongly mean-reverting under both algorithms.
The residual innovation series e_L(t) and e_R(t) are by construction
white noise under the null that the two processes are decoupled.

    L_t = phi_L L_{t-1} + c_L + e_L(t)
    R_t = phi_R R_{t-1} + c_R + e_R(t)

The cross-correlation function CCF(e_L, e_R; k) at lag k now measures
only the *coupling beyond each series' own AR(1) dynamics*.  Crucially,
CCF is asymmetric in lag under a genuine causal direction:

    k > 0  =>  e_L(t) leads e_R(t+k)   (L drives R)
    k < 0  =>  e_R(t) leads e_L(t+|k|) (R drives L)

Dr.GRPO's normalisation removes response-level advantages that leak
length back from reward; the *expected* sharpest signature is therefore
a DROP in the |CCF(k<0)| backward coupling (R leads L), so the
ASYMMETRY INDEX

    AI :=  sum_{k=1..K} [CCF(+k) - CCF(-k)]
         --------------------------------------------------
         sum_{k=0..K} [|CCF(+k)| + |CCF(-k)|]   (denominator)

should become MORE positive under Dr.GRPO.  Equivalently, the
forward/backward magnitude ratio F/B = (sum |CCF(+k)|)/(sum |CCF(-k)|)
should INCREASE.

We also report a residual-Granger F-test (predict e_R from e_L lags
beyond e_R's own AR(1)), and a per-task seed-paired bootstrap CI on
the AI and F/B deltas (B=2000, paired sign-preserving resamples).

INPUTS
------
platform_hybrid/experiments/results/drgrpo_vs_grpo.json         (arithmetic_easy, n=40, 5 seeds)
platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json  (gsm8k_cot,    n=30, 3 seeds)

OUTPUTS
-------
platform_hybrid/experiments/results/length_bias_iter96_perrun.tsv   (one row per run)
platform_hybrid/experiments/results/length_bias_iter96_paired.tsv   (one row per (task,key))
platform_hybrid/experiments/results/length_bias_iter96_summary.tsv  (task-level aggregates)
platform_hybrid/experiments/results/length_bias_iter96_meta.json    (run configuration)

USAGE
-----
python3 platform_modal/scripts/length_bias_iter96.py [--K 3] [--B 2000]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")

DRGRPO_VS_GRPO_PATH = os.path.join(RESULTS, "drgrpo_vs_grpo.json")
DRGRPO_GSM8K_PATH = os.path.join(RESULTS, "drgrpo_gsm8k_cot_full.json")


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def load_step_log(path: str) -> list[dict[str, Any]]:
    """Load runs from a Dr.GRPO JSON file and return list of dicts with
    keys (algo, seed, n, L, R)."""
    with open(path) as fh:
        d = json.load(fh)
    runs_raw = d["runs"]
    out = []
    for r in runs_raw:
        step_log = r.get("step_log") or []
        if len(step_log) < 5:
            continue
        L = np.array([float(s["mean_comp_len"]) for s in step_log], dtype=np.float64)
        R = np.array([float(s["mean_reward"]) for s in step_log], dtype=np.float64)
        out.append({
            "algo": r["algo"],
            "seed": r["seed"],
            "n": int(len(step_log)),
            "L": L,
            "R": R,
        })
    return out


# ---------------------------------------------------------------------------
#  AR(1) fitting (OLS) with intercept
# ---------------------------------------------------------------------------

def fit_ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Return (phi, c, residuals) where x[t] = phi * x[t-1] + c + e[t]."""
    x_lag = x[:-1]
    x_cur = x[1:]
    X = np.column_stack([x_lag, np.ones_like(x_lag)])
    # OLS via lstsq
    coef, *_ = np.linalg.lstsq(X, x_cur, rcond=None)
    phi, c = float(coef[0]), float(coef[1])
    pred = phi * x_lag + c
    e = x_cur - pred
    return phi, c, e


# ---------------------------------------------------------------------------
#  Cross-correlation function at lags -K..+K
# ---------------------------------------------------------------------------

def ccf_at_lags(e_a: np.ndarray, e_b: np.ndarray, K: int) -> np.ndarray:
    """Return CCF(e_a, e_b; k) for k in -K..+K, length 2K+1, ordered as
    CCF[-K], ..., CCF[+K].  Uses Pearson correlation on overlapping
    windows."""
    n = min(len(e_a), len(e_b))
    a = e_a[:n] - e_a[:n].mean()
    b = e_b[:n] - e_b[:n].mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum()) + 1e-300
    out = np.zeros(2 * K + 1, dtype=np.float64)
    for i, k in enumerate(range(-K, K + 1)):
        if k >= 0:
            x = a[:n - k]
            y = b[k:n]
        else:
            x = a[-k:n]
            y = b[:n + k]
        if len(x) < 3:
            out[i] = 0.0
            continue
        out[i] = float(np.dot(x, y) / denom)
    return out


def lag_label(k: int) -> str:
    return f"k={k:+d}"


# ---------------------------------------------------------------------------
#  Residual-Granger F-test (predict e_R from e_L lags beyond e_R's own AR(1))
# ---------------------------------------------------------------------------

def residual_granger_f(e_a: np.ndarray, e_b: np.ndarray, K: int) -> dict[str, float]:
    """Fit e_b(t) = b0 + sum_{k=1..K} b_a(k) e_a(t-k) + sum_{k=1..K} b_b(k) e_b(t-k)
    + eps; compare RSS to restricted model (e_b's own AR(K)).  Return F-stat
    and p-value approximation (chi^2 with K df, since residual series are
    pre-whitened the F and chi^2 are asymptotically equivalent)."""
    n = len(e_b)
    if n <= K + 2:
        return {"F_lr": float("nan"), "p_lr": float("nan"), "r2_lr": float("nan")}

    # Build target
    y = e_b[K:]
    # Restricted design: own lags only
    Xr = np.column_stack([e_b[K - k - 1:n - k - 1] for k in range(K)] + [np.ones(n - K)])
    # Unrestricted design: own lags + cross lags (e_a lags)
    Xu = np.column_stack(
        [e_b[K - k - 1:n - k - 1] for k in range(K)]
        + [e_a[K - k - 1:n - k - 1] for k in range(K)]
        + [np.ones(n - K)]
    )
    # OLS
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    bu, *_ = np.linalg.lstsq(Xu, y, rcond=None)
    rss_r = float(((y - Xr @ br) ** 2).sum())
    rss_u = float(((y - Xu @ bu) ** 2).sum())
    df_r = (n - K) - (K + 1)
    df_u = (n - K) - (2 * K + 1)
    if df_u <= 0 or rss_u <= 0:
        return {"F_lr": float("nan"), "p_lr": float("nan"), "r2_lr": float("nan")}
    F = ((rss_r - rss_u) / df_r) / (rss_u / df_u)
    # Two-sided p-value: chi2 with K df, since K is small, use survival 1 - chi2.cdf
    # Approximation via regularised lower incomplete gamma:
    from scipy.stats import chi2
    p = float(1.0 - chi2.cdf(F * K, df=K)) if F > 0 else 1.0
    r2 = 1.0 - rss_u / rss_r if rss_r > 0 else 0.0
    return {"F_lr": float(F), "p_lr": p, "r2_lr": float(r2)}


# ---------------------------------------------------------------------------
#  Per-run ICCA computation
# ---------------------------------------------------------------------------

def per_run_icca(L: np.ndarray, R: np.ndarray, K: int) -> dict[str, float]:
    phi_L, c_L, e_L = fit_ar1(L)
    phi_R, c_R, e_R = fit_ar1(R)
    ccf = ccf_at_lags(e_L, e_R, K)  # length 2K+1, lags -K..+K
    # Forward (k>0) and backward (k<0) magnitude sums
    fwd = float(np.sum(np.abs(ccf[K + 1 :])))
    bwd = float(np.sum(np.abs(ccf[:K])))
    fwd_signed = float(np.sum(ccf[K + 1 :]))
    bwd_signed = float(np.sum(ccf[:K]))
    asym_signed = fwd_signed - bwd_signed
    denom = float(np.sum(np.abs(ccf)))
    ai = asym_signed / denom if denom > 1e-12 else 0.0
    fb_ratio = fwd / bwd if bwd > 1e-12 else float("inf")
    # Residual Granger: e_L -> e_R (forward L->R in innovation space)
    granger = residual_granger_f(e_L, e_R, K=K)
    return {
        "phi_L": phi_L,
        "phi_R": phi_R,
        "c_L": c_L,
        "c_R": c_R,
        "e_L_std": float(e_L.std()),
        "e_R_std": float(e_R.std()),
        **{f"ccf_{lag_label(k)}": float(ccf[i]) for i, k in enumerate(range(-K, K + 1))},
        "fwd": fwd,
        "bwd": bwd,
        "fwd_signed": fwd_signed,
        "bwd_signed": bwd_signed,
        "ai": ai,
        "fb_ratio": float(fb_ratio),
        "F_lr": granger["F_lr"],
        "p_lr": granger["p_lr"],
        "r2_lr": granger["r2_lr"],
    }


# ---------------------------------------------------------------------------
#  Bootstrap paired CIs
# ---------------------------------------------------------------------------

def paired_bootstrap_delta(
    values_grpo: list[float],
    values_drgrpo: list[float],
    B: int,
    statistic=np.median,
) -> dict[str, float]:
    """Paired sign-preserving bootstrap (resample over seed indices)."""
    grpo = np.array(values_grpo, dtype=np.float64)
    drgrpo = np.array(values_drgrpo, dtype=np.float64)
    n = min(len(grpo), len(drgrpo))
    if n == 0:
        return {"delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p": float("nan"), "n": 0}
    grpo = grpo[:n]
    drgrpo = drgrpo[:n]
    diffs = drgrpo - grpo
    rng = np.random.default_rng(0xC0FFEE)
    idx = rng.integers(0, n, size=(B, n))
    boot = statistic(diffs[idx], axis=1)
    point = float(statistic(diffs))
    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))
    p_two = float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
    return {
        "delta": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "p": p_two,
        "n": int(n),
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iter96 -- ICCA")
    parser.add_argument("--K", type=int, default=3,
                        help="Number of lag steps in each direction (default 3)")
    parser.add_argument("--B", type=int, default=2000,
                        help="Bootstrap resamples for paired CIs (default 2000)")
    args = parser.parse_args(argv)

    K = int(args.K)
    B = int(args.B)

    runs = []
    runs += [(r, "arithmetic_easy") for r in load_step_log(DRGRPO_VS_GRPO_PATH)]
    runs += [(r, "gsm8k_cot") for r in load_step_log(DRGRPO_GSM8K_PATH)]

    # Compute per-run ICCA
    perrun_rows = []
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r, task in runs:
        rec = per_run_icca(r["L"], r["R"], K=K)
        rec["task"] = task
        rec["algo"] = r["algo"]
        rec["seed"] = int(r["seed"])
        rec["n"] = int(r["n"])
        perrun_rows.append(rec)
        by_task.setdefault(task, {"grpo": [], "dr_grpo": []})[r["algo"]].append(rec)

    # ---- Write perrun TSV
    perrun_path = os.path.join(RESULTS, "length_bias_iter96_perrun.tsv")
    keys = ["task", "algo", "seed", "n",
            "phi_L", "phi_R", "c_L", "c_R", "e_L_std", "e_R_std"]
    for k in range(-K, K + 1):
        keys.append(f"ccf_{lag_label(k)}")
    keys += ["fwd", "bwd", "fwd_signed", "bwd_signed", "ai", "fb_ratio",
             "F_lr", "p_lr", "r2_lr"]
    with open(perrun_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for rec in perrun_rows:
            w.writerow({k: rec.get(k, "") for k in keys})
    print(f"[iter96] wrote {perrun_path}  ({len(perrun_rows)} rows)")

    # ---- Paired comparisons: for each (task, key) compute Dr.GRPO - GRPO
    KEYS_TO_PAIR = [
        "phi_L", "phi_R", "fwd", "bwd", "fwd_signed", "bwd_signed",
        "ai", "fb_ratio", "F_lr", "p_lr", "r2_lr",
    ] + [f"ccf_{lag_label(+k)}" for k in range(1, K + 1)] \
      + [f"ccf_{lag_label(-k)}" for k in range(1, K + 1)] \
      + [f"ccf_{lag_label(0)}"]

    paired_rows = []
    for task, by_algo in by_task.items():
        grpo_recs = by_algo["grpo"]
        drgrpo_recs = by_algo["dr_grpo"]
        # align by seed
        seeds_grpo = {r["seed"]: r for r in grpo_recs}
        seeds_drgrpo = {r["seed"]: r for r in drgrpo_recs}
        common = sorted(set(seeds_grpo) & set(seeds_drgrpo))
        for key in KEYS_TO_PAIR:
            gv = [float(seeds_grpo[s][key]) for s in common]
            dv = [float(seeds_drgrpo[s][key]) for s in common]
            stat = paired_bootstrap_delta(gv, dv, B=B, statistic=np.median)
            paired_rows.append({
                "task": task,
                "key": key,
                "n_pairs": int(stat["n"]),
                "mean_diff": float(np.mean(np.array(dv) - np.array(gv))),
                "median_diff": stat["delta"],
                "ci_lo": stat["ci_lo"],
                "ci_hi": stat["ci_hi"],
"p": stat["p"],
            })
    paired_path = os.path.join(RESULTS, "length_bias_iter96_paired.tsv")
    with open(paired_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "key", "n_pairs", "mean_diff",
                                            "median_diff", "ci_lo", "ci_hi", "p"],
                           delimiter="\t")
        w.writeheader()
        for r in paired_rows:
            w.writerow(r)
    print(f"[iter96] wrote {paired_path}  ({len(paired_rows)} rows)")

    # ---- Task-level summary
    summary_rows = []
    for task, by_algo in by_task.items():
        for algo_key in ("grpo", "dr_grpo"):
            recs = by_algo[algo_key]
            if not recs:
                continue
            ai_arr = np.array([r["ai"] for r in recs])
            fb_arr = np.array([r["fb_ratio"] for r in recs])
            fwd_arr = np.array([r["fwd"] for r in recs])
            bwd_arr = np.array([r["bwd"] for r in recs])
            F_arr = np.array([r["F_lr"] for r in recs])
            p_arr = np.array([r["p_lr"] for r in recs])
            summary_rows.append({
                "task": task,
                "algo": algo_key,
                "n_seeds": len(recs),
                "ai_mean": float(ai_arr.mean()),
                "ai_std": float(ai_arr.std()),
                "fb_mean": float(np.mean(np.clip(fb_arr, -10, 10))),
                "fwd_mean": float(fwd_arr.mean()),
                "bwd_mean": float(bwd_arr.mean()),
                "F_lr_mean": float(F_arr.mean()),
                "p_lt_05": int(np.sum(p_arr < 0.05)),
            })
    summary_path = os.path.join(RESULTS, "length_bias_iter96_summary.tsv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"[iter96] wrote {summary_path}  ({len(summary_rows)} rows)")

    # ---- Meta
    meta = {
        "iter": 96,
        "K_lags": K,
        "B_bootstrap": B,
        "n_runs": len(perrun_rows),
        "tasks": sorted(by_task.keys()),
        "algos": ["grpo", "dr_grpo"],
        "inputs": [DRGRPO_VS_GRPO_PATH, DRGRPO_GSM8K_PATH],
        "outputs": [perrun_path, paired_path, summary_path],
        "key_metrics": KEYS_TO_PAIR,
        "notes": (
            "ICCA = innovation cross-correlation asymmetry on AR(1)-filtered "
            "(L_t, R_t).  AI > 0 => L leads R in innovation space; AI < 0 => "
            "R leads L.  Dr.GRPO prediction: AI rises (sever R->L feedback), "
            "F/B ratio rises, residual Granger F_lr rises."
        ),
    }
    meta_path = os.path.join(RESULTS, "length_bias_iter96_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter96] wrote {meta_path}")

    # ---- Console digest
    print("\n=== Iter96 ICCA summary (asymmetry index, AI > 0 = L leads R) ===")
    print(f"{'task':<18} {'algo':<10} {'AI_mean':>9} {'AI_std':>8} "
          f"{'F/B':>6} {'Fwd':>6} {'Bwd':>6} {'F_lr':>6} {'sig':>4}")
    for r in summary_rows:
        print(f"{r['task']:<18} {r['algo']:<10} {r['ai_mean']:>+9.4f} "
              f"{r['ai_std']:>8.4f} {r['fb_mean']:>6.2f} "
              f"{r['fwd_mean']:>6.3f} {r['bwd_mean']:>6.3f} "
              f"{r['F_lr_mean']:>6.2f} {r['p_lt_05']:>4d}")
    print("\n=== Paired deltas (Dr.GRPO - GRPO; medians, paired bootstrap B=%d) ===" % B)
    print(f"{'task':<18} {'key':<14} {'n':>3} {'Dmed':>10} {'CI_lo':>10} "
          f"{'CI_hi':>10} {'p':>8}")
    for r in paired_rows:
        print(f"{r['task']:<18} {r['key']:<14} {r['n_pairs']:>3d} "
              f"{r['median_diff']:>10.4f} {r['ci_lo']:>10.4f} {r['ci_hi']:>10.4f} "
              f"{r['p']:>8.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())