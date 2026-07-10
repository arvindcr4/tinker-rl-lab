#!/usr/bin/env python3
"""
length_bias_iter40.py — Pillar 4 / Iter 40 stage-stratified length–reward coupling.

GOAL.  Iters 28/32/36 studied the *aggregate* L(t) and R(t) trajectories:
decomposition into (trend, ZVF-coupling, R-coupling) on iter 28; temporal
lag cross-correlation on iter 32; joint saturation fitting on iter 36.
None asks the *phase-conditional* question.  Does the length–reward
coupling change across training?  Specifically:

  (A) STAGE-STRATIFIED E[R|L] SLOPE.  Split each run into K=3 training
      phases (early / mid / late, equal-width steps).  Within each phase
      compute the OLS slope beta_{R|L}^{(phase)}.  Under the verbosity
      trap literature, beta_{R|L} should be POSITIVE (longer -> higher
      reward) and GROW over training (late phase steeper than early).
      Under the anti-trap regime, beta_{R|L} should be NEGATIVE
      (shorter -> higher reward) or flat.

  (B) REWARD-PER-TOKEN TRAJECTORY.  The R_t / L_t ratio measures
      "reward density per generated token" (a proxy for compute
      efficiency).  Report its first-vs-last value, the per-step drift,
      and a paired bootstrap on the late-vs-early delta.  Dr.GRPO is
      hypothesised to RAISE R/L (efficiency) by normalising out the
      length-induced advantage variance.

  (C) ANTI-TRAP QUANTIFICATION.  Pool all (L_t, R_t) pairs and compute
      a single Pearson slope as a per-run "anti-trap magnitude"
      (negative = anti-trap, positive = trap).  Paired bootstrap
      GRPO-vs-Dr.GRPO on this slope.

Inputs (existing):
  platform_hybrid/experiments/results/drgrpo_vs_grpo.json
  platform_hybrid/experiments/results/drgrpo_gsm8k_cot_full.json

Outputs:
  platform_hybrid/experiments/results/length_bias_iter40_per_run.tsv          (per-run slopes + ratios)
  platform_hybrid/experiments/results/length_bias_iter40_phases.tsv           (per-(task,algo,phase) aggregates)
  platform_hybrid/experiments/results/length_bias_iter40_summary.tsv          (per-(task,algo) headline numbers)
  platform_hybrid/experiments/results/length_bias_iter40_grpo_vs_drgrpo.tsv   (paired bootstrap)
  platform_hybrid/experiments/results/length_bias_iter40_findings.tsv         (2 rows aggregate)

Usage:
  python platform_modal/scripts/length_bias_iter40.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "results"
DRGRPO = ROOT / "experiments" / "results" / "drgrpo_vs_grpo.json"
GSM8K = ROOT / "experiments" / "results" / "drgrpo_gsm8k_cot_full.json"


def load_runs(path):
    with open(path) as f:
        return json.load(f)["runs"]


def ols_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    dx, dy = x - xm, y - ym
    denom = float((dx * dx).sum())
    if denom <= 0:
        return float("nan"), float("nan")
    b = float((dx * dy).sum() / denom)
    a = ym - b * xm
    resid = y - (a + b * x)
    rho = float(np.corrcoef(x, y)[0, 1]) if denom > 0 and float((dy * dy).sum()) > 0 else float("nan")
    return b, rho


def phase_split(n, k=3):
    """Return list of (lo, hi) index slices for k equal-width phases."""
    edges = [int(round(i * n / k)) for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k)]


def paired_bootstrap_diff(a, b, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = a - b
    obs = float(diffs.mean()) if len(diffs) else float("nan")
    n = len(diffs)
    if n < 2:
        return dict(obs=obs, lo=float("nan"), hi=float("nan"), p_le0=float("nan"))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = diffs[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p_le0 = float((boots <= 0).mean())
    return dict(obs=obs, lo=float(lo), hi=float(hi), p_le0=p_le0)


def main():
    out_per_run = []
    out_phases = []
    out_summary = []
    out_cross = []
    sources = [
        ("arithmetic_easy", DRGRPO),
        ("gsm8k_cot", GSM8K),
    ]
    K = 3
    for task, path in sources:
        runs = load_runs(path)
        per_algo = {"grpo": {"slopes": [], "r_over_l_first": [], "r_over_l_last": [],
                              "r_over_l_drift": [], "antitrap_slope": [], "late_minus_early": [],
                              "late_minus_mid": [], "mid_minus_early": []},
                    "dr_grpo": {"slopes": [], "r_over_l_first": [], "r_over_l_last": [],
                                 "r_over_l_drift": [], "antitrap_slope": [], "late_minus_early": [],
                                 "late_minus_mid": [], "mid_minus_early": []}}
        for run in runs:
            algo = run["algo"]; seed = run["seed"]
            sl = run["step_log"]
            t = np.array([s["step"] for s in sl], dtype=float)
            R = np.array([s["mean_reward"] for s in sl], dtype=float)
            L = np.array([s["mean_comp_len"] for s in sl], dtype=float)
            n = len(t)
            # (A) per-phase E[R|L] slopes
            phases = phase_split(n, k=K)
            slopes = []
            for ph_i, (lo, hi) in enumerate(phases):
                if hi - lo < 3:
                    slopes.append(float("nan"))
                    continue
                Lp, Rp = L[lo:hi], R[lo:hi]
                beta, rho = ols_slope(Lp, Rp)
                slopes.append(beta)
                out_phases.append(dict(
                    task=task, algo=algo, seed=seed, phase=ph_i,
                    n_steps=hi - lo, beta_R_on_L=round(beta, 6),
                    pearson_r=round(rho, 4),
                ))
            # (B) R/L trajectory
            r_over_l = R / np.maximum(L, 1e-6)
            r_over_l_first = float(r_over_l[: max(1, n // 10)].mean())
            r_over_l_last = float(r_over_l[-max(1, n // 10):].mean())
            r_over_l_drift = r_over_l_last - r_over_l_first
            # (C) pooled anti-trap slope
            antitrap, antitrap_r = ols_slope(L, R)
            # Phase-delta metrics
            late_minus_early = slopes[-1] - slopes[0] if (not math.isnan(slopes[-1]) and not math.isnan(slopes[0])) else float("nan")
            late_minus_mid = slopes[-1] - slopes[1] if (not math.isnan(slopes[-1]) and not math.isnan(slopes[1])) else float("nan")
            mid_minus_early = slopes[1] - slopes[0] if (not math.isnan(slopes[1]) and not math.isnan(slopes[0])) else float("nan")
            out_per_run.append(dict(
                task=task, algo=algo, seed=seed, n_steps=n,
                beta_early=round(slopes[0], 6) if not math.isnan(slopes[0]) else "NA",
                beta_mid=round(slopes[1], 6) if not math.isnan(slopes[1]) else "NA",
                beta_late=round(slopes[2], 6) if not math.isnan(slopes[2]) else "NA",
                late_minus_early=round(late_minus_early, 6) if not math.isnan(late_minus_early) else "NA",
                late_minus_mid=round(late_minus_mid, 6) if not math.isnan(late_minus_mid) else "NA",
                mid_minus_early=round(mid_minus_early, 6) if not math.isnan(mid_minus_early) else "NA",
                r_over_l_first=round(r_over_l_first, 6),
                r_over_l_last=round(r_over_l_last, 6),
                r_over_l_drift=round(r_over_l_drift, 6),
                antitrap_slope=round(antitrap, 6),
                antitrap_pearson_r=round(antitrap_r, 4),
            ))
            per_algo[algo]["slopes"].append(slopes)
            per_algo[algo]["r_over_l_first"].append(r_over_l_first)
            per_algo[algo]["r_over_l_last"].append(r_over_l_last)
            per_algo[algo]["r_over_l_drift"].append(r_over_l_drift)
            per_algo[algo]["antitrap_slope"].append(antitrap)
            per_algo[algo]["late_minus_early"].append(late_minus_early)
            per_algo[algo]["late_minus_mid"].append(late_minus_mid)
            per_algo[algo]["mid_minus_early"].append(mid_minus_early)
        # per-(task, algo) summary
        for algo in ("grpo", "dr_grpo"):
            seeds = per_algo[algo]
            slopes_arr = np.array(seeds["slopes"], dtype=float)  # shape (n_seeds, K)
            n_eff = int(np.sum(~np.isnan(slopes_arr[:, 0])))
            out_summary.append(dict(
                task=task, algo=algo, n_seeds=len(seeds["r_over_l_first"]),
                n_slopes_identified=n_eff,
                mean_beta_early=round(float(np.nanmedian(slopes_arr[:, 0])), 6) if n_eff else float("nan"),
                mean_beta_mid=round(float(np.nanmedian(slopes_arr[:, 1])), 6) if n_eff else float("nan"),
                mean_beta_late=round(float(np.nanmedian(slopes_arr[:, 2])), 6) if n_eff else float("nan"),
                median_late_minus_early=round(float(np.nanmedian(seeds["late_minus_early"])), 6),
                median_late_minus_mid=round(float(np.nanmedian(seeds["late_minus_mid"])), 6),
                median_mid_minus_early=round(float(np.nanmedian(seeds["mid_minus_early"])), 6),
                median_r_over_l_first=round(float(np.median(seeds["r_over_l_first"])), 6),
                median_r_over_l_last=round(float(np.median(seeds["r_over_l_last"])), 6),
                median_r_over_l_drift=round(float(np.median(seeds["r_over_l_drift"])), 6),
                median_antitrap_slope=round(float(np.median(seeds["antitrap_slope"])), 6),
            ))
        # paired bootstrap GRPO vs Dr.GRPO on multiple metrics
        for metric_name, key in [
            ("beta_early", "slopes"),  # use mean of per-seed K=3 slopes
            ("beta_late_minus_early", "late_minus_early"),
            ("r_over_l_drift", "r_over_l_drift"),
            ("r_over_l_last", "r_over_l_last"),
            ("antitrap_slope", "antitrap_slope"),
        ]:
            if key == "slopes":
                a = [float(np.nanmean(s)) for s in per_algo["grpo"]["slopes"]]
                b = [float(np.nanmean(s)) for s in per_algo["dr_grpo"]["slopes"]]
            else:
                a = per_algo["grpo"][key]
                b = per_algo["dr_grpo"][key]
            boot = paired_bootstrap_diff(a, b)
            out_cross.append(dict(
                task=task, metric=metric_name,
                grpo_mean=round(float(np.mean(a)), 6),
                drgrpo_mean=round(float(np.mean(b)), 6),
                diff_grpo_minus_drgrpo=round(boot["obs"], 6),
                diff_lo=round(boot["lo"], 6),
                diff_hi=round(boot["hi"], 6),
                p_le0=round(boot["p_le0"], 4),
            ))
    # write outputs
    p1 = OUT_DIR / "length_bias_iter40_per_run.tsv"
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_per_run[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out_per_run)
    p2 = OUT_DIR / "length_bias_iter40_phases.tsv"
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_phases[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out_phases)
    p3 = OUT_DIR / "length_bias_iter40_summary.tsv"
    with open(p3, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_summary[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out_summary)
    p4 = OUT_DIR / "length_bias_iter40_grpo_vs_drgrpo.tsv"
    with open(p4, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_cross[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(out_cross)
    # findings table
    findings = []
    for task in ("arithmetic_easy", "gsm8k_cot"):
        s_grpo = [s for s in out_summary if s["task"] == task and s["algo"] == "grpo"][0]
        s_drg = [s for s in out_summary if s["task"] == task and s["algo"] == "dr_grpo"][0]
        c_drift = [c for c in out_cross if c["task"] == task and c["metric"] == "r_over_l_drift"][0]
        c_anti = [c for c in out_cross if c["task"] == task and c["metric"] == "antitrap_slope"][0]
        # sign of late_minus_early median
        late_minus_early_g = s_grpo["median_late_minus_early"]
        late_minus_early_d = s_drg["median_late_minus_early"]
        # Determine phase-stratified trap direction
        # If late_beta > 0 -> late-stage positive coupling (trap-like); if late_beta < 0 -> anti-trap persists
        late_beta_g = s_grpo["mean_beta_late"]
        late_beta_d = s_drg["mean_beta_late"]
        anti_g = s_grpo["median_antitrap_slope"]
        anti_d = s_drg["median_antitrap_slope"]
        # verbosity-trap signature: late_beta > 0 AND anti > 0
        if late_beta_g > 0 and anti_g > 0:
            verdict_g = "trap-like"
        elif late_beta_g < 0 and anti_g < 0:
            verdict_g = "anti-trap"
        else:
            verdict_g = "mixed"
        if late_beta_d > 0 and anti_d > 0:
            verdict_d = "trap-like"
        elif late_beta_d < 0 and anti_d < 0:
            verdict_d = "anti-trap"
        else:
            verdict_d = "mixed"
        findings.append(dict(
            task=task,
            GRPO_late_beta=round(late_beta_g, 6),
            DrGRPO_late_beta=round(late_beta_d, 6),
            GRPO_antitrap_slope=round(anti_g, 6),
            DrGRPO_antitrap_slope=round(anti_d, 6),
            GRPO_r_over_l_drift=round(s_grpo["median_r_over_l_drift"], 6),
            DrGRPO_r_over_l_drift=round(s_drg["median_r_over_l_drift"], 6),
            drift_diff_p_le0=round(c_drift["p_le0"], 4),
            antitrap_diff_p_le0=round(c_anti["p_le0"], 4),
            GRPO_verdict=verdict_g,
            DrGRPO_verdict=verdict_d,
        ))
    p5 = OUT_DIR / "length_bias_iter40_findings.tsv"
    with open(p5, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(findings[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(findings)
    print(f"Wrote {p1.name} ({len(out_per_run)} rows)")
    print(f"Wrote {p2.name} ({len(out_phases)} rows)")
    print(f"Wrote {p3.name} ({len(out_summary)} rows)")
    print(f"Wrote {p4.name} ({len(out_cross)} rows)")
    print(f"Wrote {p5.name} ({len(findings)} rows)")
    print()
    print("== per-(task, algo) summary ==")
    for s in out_summary:
        print(s)
    print()
    print("== GRPO vs Dr.GRPO paired bootstrap ==")
    for s in out_cross:
        print(s)
    print()
    print("== findings ==")
    for f in findings:
        print(f)


if __name__ == "__main__":
    main()
