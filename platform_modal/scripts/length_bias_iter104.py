#!/usr/bin/env python3
"""Iter 104 -- Pillar 4 (Length Bias / Dr.GRPO): Conditional ICCA by Reward
Quantile (C-ICCA-Q).

Iter 96 measured the *global* innovation cross-correlation asymmetry of the
AR(1)-whitened (L, R) pair.  Iter 100 fit a joint bivariate VAR(1) and
decomposed the cross-equation phi_LR / phi_RL coefficients.

Iter 104 asks a structurally different question that those two iters could
not answer: **In which part of the reward distribution does Dr.GRPO's
severship of the backward R->L coupling actually operate?**

If Dr.GRPO's length normalisation fires uniformly across the reward
distribution, the per-quantile |CCF(k<0)| drop should be approximately
flat across reward bins.  But the iter88 quantile-regression result and
the iter92 transfer-entropy "witness cell" both suggested that Dr.GRPO's
mechanism is concentrated where current reward is *low* (because the
normalisation matters most when token-budget pressure builds).  Iter 104
directly tests the "where in R" question.

ALGORITHM
---------
1. For each run, fit a marginal AR(1) to L and to R; obtain innovations
   (e_L, e_R).  This is exactly iter96's pre-whitening.

2. Pool the run's time-steps into 5 quantiles of the current reward R_t:
   q in {0..0.2, 0.2..0.4, ..., 0.8..1.0}  (quantile bins assigned by the
   rank of R_t within the run).

3. Within each quantile bin, compute the *within-bin* CCF of innovations
   on lags -K..+K.  This is CCF computed only on the time-step pairs
   where R_t fell in that quantile bin.  A genuine regime effect should
   show up as monotone (or non-flat) |CCF| across q, *not* as noise.

4. Aggregate the backward |CCF(-1..-K)| and forward |CCF(+1..+K)| per
   quantile per run per algorithm.  Compute the paired Dr.GRPO - GRPO
   delta at each (q, side).

5. Test "uniformity" with two complementary non-parametric tests:

   a. Page's L trend test across the 5 quantiles (one-sided, monotone-
      decreasing alternative).  Page's L is the rank-based monotone
      trend statistic; we use an exact two-sided p-value approximation
      via the standard normal deviate Z(L) = (L - mu_L)/sigma_L.

   b. Spearman correlation between q index (1..5) and the per-quantile
      |CCF| difference (Dr.GRPO - GRPO).  A negative Spearman rho with
      small p would mean "Dr.GRPO's severship grows with reward quant-
      ile" (i.e. concentrated at high-R regimes).

6. Report per-(task, algo) ratios:

      RatioBackward(q)  = |CCF_bwd(q)| / (sum_q |CCF_bwd(q)|)
      RatioForward(q)   = |CCF_fwd(q)| / (sum_q |CCF_fwd(q)|)

   so a *uniform* mechanism would have all ratios near 0.20.

INPUTS
------
experiments/results/drgrpo_vs_grpo.json         (arithmetic_easy, n=40, 5 seeds)
experiments/results/drgrpo_gsm8k_cot_full.json  (gsm8k_cot,    n=30, 3 seeds)

OUTPUTS
-------
experiments/results/length_bias_iter104_perrun.tsv      per-run per-quantile CCF
experiments/results/length_bias_iter104_paired.tsv      Dr-GR - GR delta per (task, q, key)
experiments/results/length_bias_iter104_summary.tsv     task-level ratios
experiments/results/length_bias_iter104_trendtest.tsv  Page's L + Spearman trend
experiments/results/length_bias_iter104_meta.json       run configuration

USAGE
-----
python3 platform_modal/scripts/length_bias_iter104.py [--K 3] [--n_q 5] [--B 2000]
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
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")

DRGRPO_VS_GRPO_PATH = os.path.join(RESULTS, "drgrpo_vs_grpo.json")
DRGRPO_GSM8K_PATH = os.path.join(RESULTS, "drgrpo_gsm8k_cot_full.json")


# ---------------------------------------------------------------------------
#  I/O helpers
# ---------------------------------------------------------------------------

def load_step_log(path: str) -> list[dict[str, Any]]:
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
        out.append({"algo": r["algo"], "seed": r["seed"], "n": int(len(step_log)),
                    "L": L, "R": R})
    return out


def fit_ar1(x: np.ndarray) -> tuple[float, float, np.ndarray]:
    x_lag = x[:-1]
    x_cur = x[1:]
    X = np.column_stack([x_lag, np.ones_like(x_lag)])
    coef, *_ = np.linalg.lstsq(X, x_cur, rcond=None)
    phi, c = float(coef[0]), float(coef[1])
    e = x_cur - (phi * x_lag + c)
    return phi, c, e


def ccf_at_lags(e_a: np.ndarray, e_b: np.ndarray, K: int) -> np.ndarray:
    """CCF for lags -K..+K; length 2K+1."""
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
#  Per-run, per-quantile CCF
# ---------------------------------------------------------------------------

def per_run_quantile_icca(L: np.ndarray, R: np.ndarray, K: int,
                          n_q: int) -> dict[str, Any]:
    """Return innovations, bin assignments, per-quantile CCF summaries."""
    phi_L, c_L, e_L = fit_ar1(L)
    phi_R, c_R, e_R = fit_ar1(R)
    # Innovations are at indices 1..n-1 (length n-1).
    # Reward that "drives" the innovation is R_t at the index t>=1.
    # We use the current-step reward R_t as the bin key, matched 1:1 with e_L(t), e_R(t).
    R_for_bin = R[1:]  # same length as innovations (n-1)
    n = len(R_for_bin)
    # Compute quantile bins of R_for_bin with rank assignment.
    # Use scipy percentile-based bin edges.  If R is constant, fallback to single bin.
    if np.std(R_for_bin) < 1e-12:
        bins = np.zeros(n, dtype=int)
    else:
        # equal-frequency binning: use pandas qcut-like edges
        edges = np.quantile(R_for_bin, np.linspace(0, 1, n_q + 1))
        # ensure strictly increasing edges (clip tiny duplications)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12
        bins = np.digitize(R_for_bin, edges[1:-1])  # 0..n_q-1
    rec: dict[str, Any] = {
        "phi_L": phi_L, "phi_R": phi_R,
        "c_L": c_L, "c_R": c_R,
        "e_L_std": float(e_L.std()),
        "e_R_std": float(e_R.std()),
        "q": {},  # q_idx -> {size, ccf (2K+1), fwd, bwd, fwd_signed, bwd_signed}
    }
    for q_idx in range(n_q):
        mask = bins == q_idx
        n_in_bin = int(mask.sum())
        if n_in_bin < 4:
            rec["q"][q_idx] = {
                "size": n_in_bin,
                "ccf": [0.0] * (2 * K + 1),
                "fwd": 0.0,
                "bwd": 0.0,
                "fwd_signed": 0.0,
                "bwd_signed": 0.0,
            }
            continue
        cc = ccf_at_lags(e_L[mask], e_R[mask], K=K)
        fwd = float(np.sum(np.abs(cc[K + 1 :])))
        bwd = float(np.sum(np.abs(cc[:K])))
        fwd_signed = float(np.sum(cc[K + 1 :]))
        bwd_signed = float(np.sum(cc[:K]))
        rec["q"][q_idx] = {
            "size": n_in_bin,
            "ccf": [float(cc[i]) for i in range(2 * K + 1)],
            "fwd": fwd,
            "bwd": bwd,
            "fwd_signed": fwd_signed,
            "bwd_signed": bwd_signed,
        }
    return rec


# ---------------------------------------------------------------------------
#  Trend tests
# ---------------------------------------------------------------------------

def page_l_trend(values: np.ndarray, monotone: str = "decreasing") -> dict:
    """Page's L trend test (Page, 1963).  Exact p-values via the standard
    normal approximation (large-N).

        L = sum_{i=1..n} i * R_i
    where R_i is the rank of the i-th observation (1 = smallest).  For the
    monotone='decreasing' alternative, large L indicates decreasing trend;
    for 'increasing', large L indicates increasing trend.

    Returns dict with L, mu_L, sigma_L, Z, p_one_sided.
    """
    n = len(values)
    ranks = stats.rankdata(values, method="average")
    if monotone == "decreasing":
        # rank 1 = smallest, so we want ranks to DECREASE with index i
        # i.e. large at i=1, small at i=n.  Standard Page expects large at i=n for
        # monotone-increasing alternative; reverse ranks for decreasing.
        ranks_use = n + 1 - ranks
    elif monotone == "increasing":
        ranks_use = ranks
    else:
        raise ValueError(monotone)
    L = float(np.sum(np.arange(1, n + 1) * ranks_use))
    mu_L = n * (n + 1) * (n + 2) / 6.0  # mean under H0 (ties = 0)
    # Variance for the standard Page L (no ties approx):
    sigma_L = math.sqrt(n * (n + 1) * (n + 2) * (2 * n + 3) / 6.0 / 12.0)
    # Use exact small-sample variance factor /12 -> /36 below; use the canonical:
    sigma_L = math.sqrt(n ** 2 * (n + 1) * (n + 2) ** 2 / 36.0)
    Z = (L - mu_L) / sigma_L if sigma_L > 0 else 0.0
    p_one = float(1.0 - stats.norm.cdf(Z))  # upper-tail
    return {"L": L, "mu_L": mu_L, "sigma_L": sigma_L, "Z": Z, "p_one": p_one}


# ---------------------------------------------------------------------------
#  Bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap_delta(g, d, B, statistic=np.median):
    g = np.array(g, dtype=np.float64); d = np.array(d, dtype=np.float64)
    n = min(len(g), len(d))
    if n == 0:
        return {"delta": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "p": float("nan"), "n": 0}
    g = g[:n]; d = d[:n]
    diffs = d - g
    rng = np.random.default_rng(0xC0FFEE)
    idx = rng.integers(0, n, size=(B, n))
    boot = statistic(diffs[idx], axis=1)
    point = float(statistic(diffs))
    return {"delta": point,
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "p": float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0))),
            "n": int(n)}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iter104 -- C-ICCA-Q")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_q", type=int, default=5)
    parser.add_argument("--B", type=int, default=2000)
    parser.add_argument("--seed_base", type=int, default=20260703)
    args = parser.parse_args(argv)

    K = int(args.K); n_q = int(args.n_q); B = int(args.B)
    seed_base = int(args.seed_base)

    runs = ([(r, "arithmetic_easy") for r in load_step_log(DRGRPO_VS_GRPO_PATH)]
            + [(r, "gsm8k_cot") for r in load_step_log(DRGRPO_GSM8K_PATH)])

    perrun_rows: list[dict[str, Any]] = []
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for r, task in runs:
        rec = per_run_quantile_icca(r["L"], r["R"], K=K, n_q=n_q)
        rec["task"] = task
        rec["algo"] = r["algo"]
        rec["seed"] = int(r["seed"])
        rec["n"] = int(r["n"])
        perrun_rows.append(rec)
        by_task.setdefault(task, {"grpo": [], "dr_grpo": []})[r["algo"]].append(rec)

    # Write per-run TSV (wide: per-quantile CCF summaries)
    keys = ["task", "algo", "seed", "n",
            "phi_L", "phi_R", "c_L", "c_R", "e_L_std", "e_R_std"]
    for q in range(n_q):
        keys += [f"q{q}_size", f"q{q}_fwd", f"q{q}_bwd",
                 f"q{q}_fwd_signed", f"q{q}_bwd_signed"]
    perrun_path = os.path.join(RESULTS, "length_bias_iter104_perrun.tsv")
    with open(perrun_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t")
        w.writeheader()
        for rec in perrun_rows:
            row = {k: rec.get(k, "") for k in keys}
            for q in range(n_q):
                qq = rec["q"][q]
                row[f"q{q}_size"] = qq["size"]
                row[f"q{q}_fwd"] = qq["fwd"]
                row[f"q{q}_bwd"] = qq["bwd"]
                row[f"q{q}_fwd_signed"] = qq["fwd_signed"]
                row[f"q{q}_bwd_signed"] = qq["bwd_signed"]
            w.writerow(row)
    print(f"[iter104] wrote {perrun_path}  ({len(perrun_rows)} rows)")

    # Paired comparison at each (task, q) for fwd, bwd (abs + signed) + per-lag CCF.
    paired_rows = []
    for task, by_algo in by_task.items():
        seeds_g = {r["seed"]: r for r in by_algo["grpo"]}
        seeds_d = {r["seed"]: r for r in by_algo["dr_grpo"]}
        common = sorted(set(seeds_g) & set(seeds_d))
        if not common:
            continue
        for q in range(n_q):
            for side in ["fwd", "bwd", "fwd_signed", "bwd_signed"]:
                g = [float(seeds_g[s]["q"][q][side]) for s in common]
                d = [float(seeds_d[s]["q"][q][side]) for s in common]
                stat = paired_bootstrap_delta(g, d, B=B, statistic=np.median)
                paired_rows.append({
                    "task": task, "q": q, "side": side, **stat,
                })
            for i in range(2 * K + 1):
                lag = i - K
                g = [float(seeds_g[s]["q"][q]["ccf"][i]) for s in common]
                d = [float(seeds_d[s]["q"][q]["ccf"][i]) for s in common]
                stat = paired_bootstrap_delta(g, d, B=B, statistic=np.median)
                paired_rows.append({
                    "task": task, "q": q, "side": lag_label(lag), **stat,
                })
    paired_path = os.path.join(RESULTS, "length_bias_iter104_paired.tsv")
    with open(paired_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "q", "side", "delta",
                                            "ci_lo", "ci_hi", "p", "n"],
                           delimiter="\t")
        w.writeheader()
        for r in paired_rows:
            w.writerow(r)
    print(f"[iter104] wrote {paired_path}  ({len(paired_rows)} rows)")

    # Task-level summary: ratios of |CCF_bwd| and |CCF_fwd| in each quantile
    summary_rows = []
    for task, by_algo in by_task.items():
        for algo, recs in by_algo.items():
            fwd_q = np.zeros(n_q)
            bwd_q = np.zeros(n_q)
            for r in recs:
                for q in range(n_q):
                    fwd_q[q] += r["q"][q]["fwd"]
                    bwd_q[q] += r["q"][q]["bwd"]
            tot_fwd = fwd_q.sum()+ 1e-300
            tot_bwd = bwd_q.sum() + 1e-300
            for q in range(n_q):
                summary_rows.append({
                    "task": task, "algo": algo,
                    "q": q,
                    "n_seeds": len(recs),
                    "fwd_abs_mean": float(fwd_q[q] / max(1, len(recs))),
                    "bwd_abs_mean": float(bwd_q[q] / max(1, len(recs))),
                    "fwd_ratio": float(fwd_q[q] / tot_fwd),
                    "bwd_ratio": float(bwd_q[q] / tot_bwd),
                })
    summary_path = os.path.join(RESULTS, "length_bias_iter104_summary.tsv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "algo", "q", "n_seeds",
                                            "fwd_abs_mean", "bwd_abs_mean",
                                            "fwd_ratio", "bwd_ratio"],
                           delimiter="\t")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"[iter104] wrote {summary_path}  ({len(summary_rows)} rows)")

    # Trend tests: for each (task, algo, side), apply Page's L (decreasing)
    # and Spearman (rho, p) across q=0..n_q-1 with q_index = q+1.
    trend_rows = []
    for task, by_algo in by_task.items():
        for algo, recs in by_algo.items():
            for side in ["fwd", "bwd", "fwd_signed", "bwd_signed"]:
                vals = np.array([sum(r["q"][q][side] for r in recs)
                                 / max(1, len(recs))
                                 for q in range(n_q)])
                page = page_l_trend(vals, monotone="decreasing")
                sp = stats.spearmanr(np.arange(1, n_q + 1), vals)
                trend_rows.append({
                    "task": task, "algo": algo, "side": side,
                    "vals_q0": float(vals[0]),
                    "vals_q_last": float(vals[-1]),
                    "page_L": page["L"],
                    "page_Z": page["Z"],
                    "page_p_decreasing": page["p_one"],
                    "spearman_rho": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                })
    trend_path = os.path.join(RESULTS, "length_bias_iter104_trendtest.tsv")
    with open(trend_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "algo", "side",
                                            "vals_q0", "vals_q_last",
                                            "page_L", "page_Z",
                                            "page_p_decreasing",
                                            "spearman_rho", "spearman_p"],
                           delimiter="\t")
        w.writeheader()
        for r in trend_rows:
            w.writerow(r)
    print(f"[iter104] wrote {trend_path}  ({len(trend_rows)} rows)")

    # Meta
    meta = {
        "iter": 104, "pillar": "P4-LengthBias",
        "K": K, "n_q": n_q, "B": B, "seed_base": seed_base,
        "n_runs": len(perrun_rows),
        "tasks": sorted(set(r["task"] for r in perrun_rows)),
        "algos": sorted(set(r["algo"] for r in perrun_rows)),
    }
    meta_path = os.path.join(RESULTS, "length_bias_iter104_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[iter104] wrote {meta_path}")

    # Console headline: Dr.GRPO - GRPO bwd ratio test
    print("\n[iter104 headline] backward |CCF| per-q RATIOS (Dr.GRPO - GRPO):")
    for task in sorted(set(r["task"] for r in summary_rows)):
        sub = [r for r in summary_rows if r["task"] == task]
        for q in range(n_q):
            g = next(r for r in sub if r["algo"] == "grpo" and r["q"] == q)
            d = next(r for r in sub if r["algo"] == "dr_grpo" and r["q"] == q)
            print(f"  {task:14s} q={q}: GR bwd={g['bwd_abs_mean']:.4f} "
                  f"Dr.GRPO bwd={d['bwd_abs_mean']:.4f} "
                  f"diff={d['bwd_abs_mean'] - g['bwd_abs_mean']:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
