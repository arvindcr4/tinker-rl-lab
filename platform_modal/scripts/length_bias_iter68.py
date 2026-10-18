"""length_bias_iter68.py — Iter 68 Pillar 4 (Length Bias / Dr.GRPO).

Trajectory-divergence decomposition. Three paired diagnostics on real per-step
trajectories from drgrpo_gsm8k_cot_full.json and drgrpo_vs_grpo.json:

1. First-divergence step T*: smallest t where |L_grpo(t) - L_drgrpo(t)| exceeds
   a length-rescaled threshold (0.5 * sigma of pooled L across the two algos).
   Descriptive seed-bootstrap CI for mean pair-level divergence time; no p-value.
2. Length-volatility ratio V: sigma(Delta L) per run, paired ratio Dr.GRPO/GRPO.
   Bootstrap CI on the per-seed ratio and the log-ratio.
3. Reversal rate rho: fraction of steps where sign(Delta L_t) flips relative to
   sign(Delta L_{t-1}). Paired diff with bootstrap CI.

Negative control: arithmetic_easy should show no meaningful divergence (both
algos converge to L ~ 4 by step 5).

Outputs:
  platform_hybrid/experiments/results/length_bias_iter68_first_div_v2.tsv
  platform_hybrid/experiments/results/length_bias_iter68_volatility_v2.tsv
  platform_hybrid/experiments/results/length_bias_iter68_reversals_v2.tsv
  platform_hybrid/experiments/results/length_bias_iter68_summary_v2.tsv
  platform_hybrid/experiments/results/length_bias_iter68_meta_v2.json
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GSM_PATH = os.path.join(ROOT, "experiments", "results", "drgrpo_gsm8k_cot_full.json")
ARITH_PATH = os.path.join(ROOT, "experiments", "results", "drgrpo_vs_grpo.json")
OUT_DIR = os.path.join(ROOT, "experiments", "results")

# bootstrap params
B = 2000
RNG = random.Random(20260703)
Z975 = 1.959963984540054  # two-sided 95% normal


def load_runs(path: str, algo_map: Dict[str, str] | None = None) -> Dict[Tuple[str, int], List[dict]]:
    """Return {(algo, seed): [step_dict, ...]} keyed by (algo, seed)."""
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for r in d.get("runs", []):
        algo = r.get("algo")
        seed = r.get("seed")
        if algo_map and algo in algo_map:
            algo = algo_map[algo]
        out[(algo, seed)] = list(r.get("step_log", []))
    return out


def trajectory_features(steps: List[dict]) -> dict:
    """Compute per-trajectory features: deltas, sigma_dL, reversal count."""
    if len(steps) < 2:
        return dict(dL=[], sigma_dL=float("nan"), n_reversals=0, reversal_rate=float("nan"))
    L = [s["mean_comp_len"] for s in steps]
    R = [s["mean_reward"] for s in steps]
    dL = [L[t] - L[t - 1] for t in range(1, len(L))]
    dR = [R[t] - R[t - 1] for t in range(1, len(R))]
    # sigma of dL (sample std, ddof=1)
    n = len(dL)
    mean = sum(dL) / n
    var = sum((x - mean) ** 2 for x in dL) / max(n - 1, 1)
    sigma = math.sqrt(var)
    # reversal rate: number of t where sign(dL[t]) != sign(dL[t-1])
    signs = [1 if x > 0 else (-1 if x < 0 else 0) for x in dL]
    rev = 0
    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
            rev += 1
    rev_rate = rev / max(n - 1, 1)
    return dict(L=L, R=R, dL=dL, dR=dR, sigma_dL=sigma, n_reversals=rev, reversal_rate=rev_rate)


def first_divergence_step(L_a: List[float], L_b: List[float], frac: float = 0.5, min_hold: int = 2) -> int:
    """First step index t where |L_a[t] - L_b[t]| > frac * sigma(L_pooled) for >= min_hold steps.

    If never satisfied, returns len(L_a) - 1.
    Returns the index in step_log (0..n-1).
    """
    if not L_a or not L_b:
        return -1
    n = min(len(L_a), len(L_b))
    pool = L_a[:n] + L_b[:n]
    mean = sum(pool) / len(pool)
    var = sum((x - mean) ** 2 for x in pool) / max(len(pool) - 1, 1)
    sd = math.sqrt(var)
    thr = frac * sd
    diffs = [abs(L_a[t] - L_b[t]) for t in range(n)]
    # scan for first run of length min_hold where diffs > thr
    run = 0
    for t in range(n):
        if diffs[t] > thr:
            run += 1
            if run >= min_hold:
                return t - min_hold + 1
        else:
            run = 0
    return n - 1


def bootstrap_paired_diff(values_a: List[float], values_b: List[float], statistic="mean") -> Tuple[float, float, float, float]:
    """Return (mean_diff, ci_lo, ci_hi, p_le0) via paired bootstrap.

    statistic='mean' tests H0: mean(B) - mean(A) <= 0 vs H1: > 0.
    """
    n = len(values_a)
    assert n == len(values_b)
    diffs = [values_b[i] - values_a[i] for i in range(n)]
    obs = sum(diffs) / n
    # bootstrap distribution of the mean of resampled diffs
    boot = []
    for _ in range(B):
        idx = [RNG.randrange(n) for _ in range(n)]
        boot.append(sum(diffs[i] for i in idx) / n)
    boot.sort()
    lo = boot[int(0.025 * B)]
    hi = boot[int(0.975 * B)]
    # two-sided p-value for diff <= 0 vs > 0:
    # fraction of bootstrap means <= 0
    p_le0 = sum(1 for x in boot if x <= 0) / B
    return obs, lo, hi, p_le0


def bootstrap_paired_ratio(values_a: List[float], values_b: List[float]) -> Tuple[float, float, float, float]:
    """Return (ratio B/A, log-ratio CI, p_le0 for log-ratio <= 0).

    A zero-safe ratio: replace zeros in A with mean(A) (only if all-zero fallback).
    """
    n = len(values_a)
    eps = 1e-9
    a = [max(x, eps) for x in values_a]
    b = [max(x, eps) for x in values_b]
    ratios = [b[i] / a[i] for i in range(n)]
    obs = sum(ratios) / n
    log_ratios = [math.log(r) for r in ratios]
    boot = []
    for _ in range(B):
        idx = [RNG.randrange(n) for _ in range(n)]
        boot.append(sum(log_ratios[i] for i in idx) / n)
    boot.sort()
    lo = math.exp(boot[int(0.025 * B)])
    hi = math.exp(boot[int(0.975 * B)])
    p_le0 = sum(1 for x in boot if x <= 0) / B
    return obs, lo, hi, p_le0


def write_tsv(path: str, rows: List[dict]) -> None:
    if not rows:
        with open(path, "w") as fh:
            fh.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for row in rows:
            fh.write("\t".join(_fmt(row[k]) for k in keys) + "\n")


def _fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6f}"
    return str(v)


def reversal_rate_conditional(steps: List[dict], reward_sign: int) -> float:
    """Reversal rate restricted to steps where dR has the requested sign.

    Computed over dL[t] for t where dR[t] has the requested sign; the sign-flip
    is relative to the previous dL step (any reward sign).
    """
    if len(steps) < 3:
        return float("nan")
    L = [s["mean_comp_len"] for s in steps]
    R = [s["mean_reward"] for s in steps]
    dL = [L[t] - L[t - 1] for t in range(1, len(L))]
    dR = [R[t] - R[t - 1] for t in range(1, len(R))]
    signs_dL = [1 if x > 0 else (-1 if x < 0 else 0) for x in dL]
    signs_dR = [1 if x > 0 else (-1 if x < 0 else 0) for x in dR]
    eligible = [i for i in range(len(dL)) if signs_dR[i] == reward_sign]
    if not eligible:
        return float("nan")
    rev = 0
    for i in eligible:
        j = i - 1  # previous dL index
        if j < 0:
            continue
        if signs_dL[i] != 0 and signs_dL[j] != 0 and signs_dL[i] != signs_dL[j]:
            rev += 1
    return rev / len(eligible)


def trajectory_auc(L_a: List[float], L_b: List[float], steps=None) -> float:
    """Signed trapezoidal area, review25-v2; optional strictly increasing step grid."""
    if len(L_a) != len(L_b):
        raise ValueError("paired trajectories must have equal lengths")
    n = len(L_a)
    steps = list(range(n)) if steps is None else list(steps)
    if len(steps) != n or any(not math.isfinite(x) for x in [*steps, *L_a, *L_b]):
        raise ValueError("finite aligned trajectories and steps required")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("steps must increase strictly")
    if n < 2:
        return float("nan")
    return sum(.5 * ((L_a[t]-L_b[t]) + (L_a[t+1]-L_b[t+1])) *
               (steps[t+1]-steps[t]) for t in range(n-1))


def signed_auc_summary(values):
    """Dr.GRPO-minus-GRPO area: negate once, never compare AUC to its negative."""
    return bootstrap_paired_diff(values, [0.0] * len(values))


def analyse(experiment: str, runs: Dict[Tuple[str, int], List[dict]], algo_a: str, algo_b: str) -> dict:
    """Run all three paired diagnostics for one experiment."""
    # group seeds by algo
    seeds_a = sorted({seed for (a, seed) in runs if a == algo_a})
    seeds_b = sorted({seed for (a, seed) in runs if a == algo_b})
    seeds = sorted(set(seeds_a) & set(seeds_b))
    first_div_rows = []
    vol_rows = []
    rev_rows = []
    raw_rows = []
    auc_rows = []
    rev_cond_rows = []
    for seed in seeds:
        feat_a = trajectory_features(runs[(algo_a, seed)])
        feat_b = trajectory_features(runs[(algo_b, seed)])
        L_a = feat_a["L"]
        L_b = feat_b["L"]
        fds = first_divergence_step(L_a, L_b)
        log_a, log_b = runs[(algo_a, seed)], runs[(algo_b, seed)]
        has_steps = any("step" in row for row in log_a + log_b)
        grid = None
        if has_steps:
            if not all("step" in row for row in log_a + log_b):
                raise ValueError("incomplete step grid")
            grid = [row["step"] for row in log_a]
            if grid != [row["step"] for row in log_b]:
                raise ValueError("paired step grids must match")
        auc = trajectory_auc(L_a, L_b, grid)
        raw_rows.append(dict(
            experiment=experiment,
            seed=seed,
            algo=algo_a,
            sigma_dL=feat_a["sigma_dL"],
            reversal_rate=feat_a["reversal_rate"],
            n_reversals=feat_a["n_reversals"],
            n_steps=len(L_a),
        ))
        raw_rows.append(dict(
            experiment=experiment,
            seed=seed,
            algo=algo_b,
            sigma_dL=feat_b["sigma_dL"],
            reversal_rate=feat_b["reversal_rate"],
            n_reversals=feat_b["n_reversals"],
            n_steps=len(L_b),
        ))
        first_div_rows.append(dict(
            experiment=experiment,
            seed=seed,
            fds=fds,
            n_steps=len(L_a),
        ))
        vol_rows.append(dict(
            experiment=experiment,
            seed=seed,
            sigma_dL_grpo=feat_a["sigma_dL"],
            sigma_dL_drgrpo=feat_b["sigma_dL"],
        ))
        rev_rows.append(dict(
            experiment=experiment,
            seed=seed,
            rev_rate_grpo=feat_a["reversal_rate"],
            rev_rate_drgrpo=feat_b["reversal_rate"],
        ))
        auc_rows.append(dict(
            experiment=experiment,
            seed=seed,
            auc_grpo_minus_drgrpo=auc,
            L_mean_grpo=sum(L_a) / len(L_a),
            L_mean_drgrpo=sum(L_b) / len(L_b),
        ))
        # conditional reversals on (pos_dR, neg_dR) steps
        steps_a = runs[(algo_a, seed)]
        steps_b = runs[(algo_b, seed)]
        rev_cond_rows.append(dict(
            experiment=experiment, seed=seed,
            cell="rev_on_pos_dR",
            rev_grpo=reversal_rate_conditional(steps_a, 1),
            rev_drgrpo=reversal_rate_conditional(steps_b, 1),
        ))
        rev_cond_rows.append(dict(
            experiment=experiment, seed=seed,
            cell="rev_on_neg_dR",
            rev_grpo=reversal_rate_conditional(steps_a, -1),
            rev_drgrpo=reversal_rate_conditional(steps_b, -1),
        ))
    # paired bootstrap stats
    fds_vals = [r["fds"] for r in first_div_rows]
    sigma_a = [r["sigma_dL_grpo"] for r in vol_rows]
    sigma_b = [r["sigma_dL_drgrpo"] for r in vol_rows]
    rev_a = [r["rev_rate_grpo"] for r in rev_rows]
    rev_b = [r["rev_rate_drgrpo"] for r in rev_rows]
    # FDS is one descriptive statistic per trajectory pair, not an algorithm contrast.
    fds_mean = sum(fds_vals) / len(fds_vals)
    # bootstrap CI on per-seed FDS:
    boot = []
    for _ in range(B):
        idx = [RNG.randrange(len(fds_vals)) for _ in range(len(fds_vals))]
        boot.append(sum(fds_vals[i] for i in idx) / len(fds_vals))
    boot.sort()
    fds_ci_lo = boot[int(0.025 * B)]
    fds_ci_hi = boot[int(0.975 * B)]
    # volatility: paired ratio Dr.GRPO / GRPO
    vol_obs, vol_lo, vol_hi, vol_p = bootstrap_paired_ratio(sigma_a, sigma_b)
    # reversals: paired diff Dr.GRPO - GRPO
    rev_obs, rev_lo, rev_hi, rev_p = bootstrap_paired_diff(rev_a, rev_b)
    summary_rows = [
        dict(experiment=experiment, metric="fds_mean",
             n_pairs=len(seeds), mean_grpo=float("nan"), mean_drgrpo=float("nan"),
             mean_diff=float("nan"), ci_lo=fds_ci_lo, ci_hi=fds_ci_hi,
             p_le0=float("nan"), interpretation=f"descriptive pair-level mean FDS={fds_mean}; percentile CI; no algorithm contrast or p-value"),
        dict(experiment=experiment, metric="volatility_ratio_drgrpo_over_grpo",
             n_pairs=len(seeds), mean_grpo=sum(sigma_a) / len(sigma_a),
             mean_drgrpo=sum(sigma_b) / len(sigma_b),
             mean_diff=vol_obs, ci_lo=vol_lo, ci_hi=vol_hi,
             p_le0=vol_p, interpretation="ratio > 1 means Dr.GRPO length trajectory is more volatile"),
        dict(experiment=experiment, metric="reversal_rate_diff_drgrpo_minus_grpo",
             n_pairs=len(seeds), mean_grpo=sum(rev_a) / len(rev_a),
             mean_drgrpo=sum(rev_b) / len(rev_b),
             mean_diff=rev_obs, ci_lo=rev_lo, ci_hi=rev_hi,
             p_le0=rev_p, interpretation="positive = Dr.GRPO flips sign(dL) more often"),
    ]
    for row in summary_rows:
        row["estimate"] = fds_mean if row["metric"] == "fds_mean" else row["mean_diff"]
    return dict(
        first_div=first_div_rows,
        vol=vol_rows,
        rev=rev_rows,
        raw=raw_rows,
        auc=auc_rows,
        rev_cond=rev_cond_rows,
        summary=summary_rows,
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_paths = dict(
        first_div=os.path.join(OUT_DIR, "length_bias_iter68_first_div_v2.tsv"),
        vol=os.path.join(OUT_DIR, "length_bias_iter68_volatility_v2.tsv"),
        rev=os.path.join(OUT_DIR, "length_bias_iter68_reversals_v2.tsv"),
        raw=os.path.join(OUT_DIR, "length_bias_iter68_per_run_v2.tsv"),
        summary=os.path.join(OUT_DIR, "length_bias_iter68_summary_v2.tsv"),
    )
    meta = {"experiments": []}
    # GSM8K CoT
    runs_gsm = load_runs(GSM_PATH)
    res_gsm = analyse("drgrpo_gsm8k_cot", runs_gsm, "grpo", "dr_grpo")
    # arithmetic easy
    runs_arith = load_runs(ARITH_PATH)
    res_arith = analyse("drgrpo_vs_grpo", runs_arith, "grpo", "dr_grpo")
    # write
    write_tsv(out_paths["first_div"], res_gsm["first_div"] + res_arith["first_div"])
    write_tsv(out_paths["vol"], res_gsm["vol"] + res_arith["vol"])
    write_tsv(out_paths["rev"], res_gsm["rev"] + res_arith["rev"])
    write_tsv(out_paths["raw"], res_gsm["raw"] + res_arith["raw"])
    for row in res_gsm["summary"] + res_arith["summary"]:
        row.setdefault("estimate", row["mean_diff"])
    write_tsv(out_paths["summary"], res_gsm["summary"] + res_arith["summary"])
    out_paths["auc"] = os.path.join(OUT_DIR, "length_bias_iter68_auc_v2.tsv")
    out_paths["rev_cond"] = os.path.join(OUT_DIR, "length_bias_iter68_rev_cond_v2.tsv")
    write_tsv(out_paths["auc"], res_gsm["auc"] + res_arith["auc"])
    write_tsv(out_paths["rev_cond"], res_gsm["rev_cond"] + res_arith["rev_cond"])
    # summary rows for AUC and conditional reversals
    for res in (res_gsm, res_arith):
        auc_a = [r["auc_grpo_minus_drgrpo"] for r in res["auc"]]
        # Negate the GRPO-minus-Dr.GRPO area exactly once.
        auc_obs, auc_lo, auc_hi, auc_p = signed_auc_summary(auc_a)
        res["summary"].append(dict(
            experiment=res["summary"][0]["experiment"],
            metric="auc_drgrpo_minus_grpo",
            n_pairs=len(auc_a),
            mean_grpo=float("nan"),
            mean_drgrpo=float("nan"),
            mean_diff=auc_obs, ci_lo=auc_lo, ci_hi=auc_hi,
            p_le0=auc_p,
            interpretation="signed area (Dr.GRPO - GRPO); positive = Dr.GRPO trace lies above GRPO",
        ))
        # conditional reversals: pooled over (pos_dR, neg_dR)
        for cell in ("rev_on_pos_dR", "rev_on_neg_dR"):
            sub = [r for r in res["rev_cond"] if r["cell"] == cell]
            a_vals = [r["rev_grpo"] for r in sub]
            b_vals = [r["rev_drgrpo"] for r in sub]
            if not any(math.isnan(x) for x in a_vals + b_vals):
                obs, lo, hi, p = bootstrap_paired_diff(a_vals, b_vals)
                res["summary"].append(dict(
                    experiment=res["summary"][0]["experiment"],
                    metric=f"{cell}_diff_drgrpo_minus_grpo",
                    n_pairs=len(sub),
                    mean_grpo=sum(a_vals) / len(a_vals),
                    mean_drgrpo=sum(b_vals) / len(b_vals),
                    mean_diff=obs, ci_lo=lo, ci_hi=hi, p_le0=p,
                    interpretation="positive = Dr.GRPO flips dL direction more on this reward-sign stratum",
                ))
    for row in res_gsm["summary"] + res_arith["summary"]:
        row.setdefault("estimate", row["mean_diff"])
    write_tsv(out_paths["summary"], res_gsm["summary"] + res_arith["summary"])
    meta["experiments"].append(dict(
        name="drgrpo_gsm8k_cot",
        summary_rows=res_gsm["summary"],
    ))
    meta["experiments"].append(dict(
        name="drgrpo_vs_grpo",
        summary_rows=res_arith["summary"],
    ))
    meta["outputs"] = out_paths
    with open(os.path.join(OUT_DIR, "length_bias_iter68_meta_v2.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("[iter68] wrote:", json.dumps(out_paths, indent=2))
    # print headline
    for row in res_gsm["summary"] + res_arith["summary"]:
        print(f"  {row['experiment']:24s} {row['metric']:48s} grpo={row['mean_grpo']:.4f} drgrpo={row['mean_drgrpo']:.4f} diff={row['mean_diff']:.4f} CI=[{row['ci_lo']:.4f},{row['ci_hi']:.4f}] p_le0={row['p_le0']:.4f}")


if __name__ == "__main__":
    main()