#!/usr/bin/env python3
"""Pillar 2 elevation iter14: ZVF dynamics & lead-time-to-collapse diagnostic.

Builds on iter2/6/10 (cross-experiment aggregation, by-library,
anti-herding delta) by adding a TIME-AXIS layer: per-step ZVF trajectories
are reduced to:

  * lag-1 autocorrelation per run     (signal stickiness)
  * first-passage time to ZVF>theta   (signal-degeneration onset)
  * zvf-fraction-curve area under (AUC) above theta per run
  * phase-decomposed (early/mid/late) ZVF mean + std
  * lead time from "zvf crosses theta" to "collapse flag set"
    (using the variance_mitigation run-level `collapse` column)
  * method-level ZVF instability ranking (flapping vs locked)

Inputs (real per-step data already in the worktree):

  experiments/results/groupsize_zvf_sweep.json
      12 runs x 40 steps, G in {2,4,8,16} x {42,123,456}.
  experiments/results/variance_mitigation.tsv
      9 methods x 5 seeds x 100-300 steps; collapse flag per step.
  experiments/results/tinker_gsm8k_zvf_summary.json (+ per-seed files)
      3 seeds x 200 problems (prompt-level, not step-level — used for
      prompt-zvf cross-entropy analysis instead).

Outputs:

  experiments/results/zvf_dynamics_summary.tsv
      One row per (method_or_group_size, seed-or-pool) with dynamics
      statistics.
  experiments/results/zvf_dynamics_leadtime.tsv
      Per-(method,seed) first-ZVF>theta step, first-collapse step, lead.
  experiments/results/zvf_dynamics_phase.tsv
      Early/mid/late ZVF statistics per run.
  experiments/results/zvf_dynamics.json
      All aggregates also in JSON for downstream figure scripts.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "experiments", "results")


def _autocorr_lag1(xs):
    if len(xs) < 4:
        return float("nan")
    m = sum(xs) / len(xs)
    num = sum((xs[i] - m) * (xs[i - 1] - m) for i in range(1, len(xs)))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den > 0 else float("nan")


def _first_passage(xs, theta):
    """First index i (>=1) such that xs[i] >= theta AND xs[i-1] < theta.
    Returns None if no passage found within the trace."""
    for i in range(1, len(xs)):
        if xs[i] >= theta and xs[i - 1] < theta:
            return i
    return None


def _auc_above(xs, theta):
    """Sum of (xv - theta)+ normalized by len(xs) over [0, len(xs)] domain.
    Gives a 0..1-max pressure index for sustained signal starvation."""
    if not xs:
        return float("nan")
    return max(0.0, sum(max(0.0, x - theta) for x in xs)) / len(xs)


def _phase(xs, n_phases=3):
    """Split xs into n_phases contiguous slices and return per-phase means."""
    n = len(xs)
    if n < 3:
        return [float("nan")] * n_phases
    cuts = [n * k // n_phases for k in range(n_phases + 1)]
    return [
        (sum(xs[cuts[i]:cuts[i + 1]]) / max(1, cuts[i + 1] - cuts[i]))
        for i in range(n_phases)
    ]


def _load_groupsize_zvf_runs():
    path = os.path.join(RESULTS_DIR, "groupsize_zvf_sweep.json")
    data = json.load(open(path))
    runs = []
    for r in data["runs"]:
        steps = r["step_log"]
        zvfs = [s["zvf"] for s in steps]
        rews = [s["mean_reward"] for s in steps]
        runs.append({
            "kind": "groupsize_zvf_sweep",
            "method": f"grpo_G{int(r['group_size'])}",
            "seed": int(r["seed"]),
            "group_size": int(r["group_size"]),
            "n_steps": len(steps),
            "zvf_series": zvfs,
            "reward_series": rews,
            "collapse_series": [0] * len(steps),
            "mean_zvf": r["mean_zvf"],
            "heldout_acc": r.get("heldout_acc", float("nan")),
            "last10_avg": r.get("last10_avg", float("nan")),
        })
    return runs


def _load_variance_mitigation_runs():
    path = os.path.join(RESULTS_DIR, "variance_mitigation.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    byrun = defaultdict(list)
    for r in rows:
        key = (r["method"], int(r["seed"]))
        byrun[key].append(r)
    runs = []
    for (method, seed), rs in byrun.items():
        rs.sort(key=lambda r: int(r["step"]))
        zvfs = [float(r["zvf"]) for r in rs]
        rews = [float(r["reward_mean"]) for r in rs]
        coll = [int(r["collapse"]) for r in rs]
        runs.append({
            "kind": "variance_mitigation",
            "method": method,
            "seed": int(seed),
            "group_size": 8,
            "n_steps": len(rs),
            "zvf_series": zvfs,
            "reward_series": rews,
            "collapse_series": coll,
            "mean_zvf": sum(zvfs) / max(1, len(zvfs)),
            "heldout_acc": float(rs[-1]["heldout_acc"]),
            "last10_avg": sum(float(r["heldout_acc"]) for r in rs[-10:]) / min(10, len(rs)),
        })
    return runs


def _load_tinker_prompt_zvf():
    """Prompt-level ZVF cross-entropy across 3 seeds x 200 problems.
    Treats the 200-problem series as a step-like trace for prompt-level
    ZVF dynamics: AC1, on/offset vs cumulative."""
    out = []
    for seed_file in (
        "tinker_gsm8k_zvf_s42.json",
        "tinker_gsm8k_zvf_s123.json",
        "tinker_gsm8k_zvf_s456.json",
    ):
        path = os.path.join(RESULTS_DIR, seed_file)
        data = json.load(open(path))
        zvfs = [float(p["zvf"]) for p in data["per_problem"]]
        rews = [float(p["mean_reward"]) for p in data["per_problem"]]
        out.append({
            "kind": "tinker_prompt_zvf",
            "method": "qwen3_8b_gsm8k",
            "seed": data["seed"],
            "group_size": data["group_size"],
            "n_steps": len(zvfs),
            "zvf_series": zvfs,
            "reward_series": rews,
            "collapse_series": [0] * len(zvfs),
            "mean_zvf": data["overall_zvf"],
            "heldout_acc": data["overall_accuracy"],
            "last10_avg": sum(rews[-10:]) / 10.0,
        })
    return out


def _summarize_run(run, thetas=(0.5, 0.7, 0.9)):
    z = run["zvf_series"]
    r = run["reward_series"]
    out = {
        "kind": run["kind"],
        "method": run["method"],
        "seed": run["seed"],
        "group_size": run["group_size"],
        "n_steps": run["n_steps"],
        "mean_zvf": run["mean_zvf"],
        "heldout_acc": run["heldout_acc"],
        "last10_avg": run["last10_avg"],
        "zvf_lag1": _autocorr_lag1(z),
        "reward_lag1": _autocorr_lag1(r),
        "zvf_std": (statistics.pstdev(z) if len(z) >= 2 else float("nan")),
        "reward_zvf_corr": (
            statistics.correlation(z, r) if len(z) >= 4 and statistics.pstdev(z) > 0 and statistics.pstdev(r) > 0 else float("nan")
        ),
    }
    for th in thetas:
        out[f"first_pass_zvf{str(th).replace('.', '')}"] = _first_passage(z, th)
        out[f"auc_above_zvf{str(th).replace('.', '')}"] = _auc_above(z, th)
    phases = _phase(z, 3)
    for i, p in enumerate(phases, start=1):
        out[f"zvf_phase{i}_mean"] = p
    return out


def _lead_time(run):
    """First step at which zvf>=theta AND collapse flag = 1 in next 10 steps.
    Returns None if no lead event."""
    z = run["zvf_series"]
    c = run["collapse_series"]
    n = min(len(z), len(c))
    if n < 3:
        return None
    for theta in (0.7, 0.8, 0.9):
        for i in range(1, n - 1):
            if z[i] >= theta and z[i - 1] < theta:
                # collapse within next 10 steps?
                horizon = min(n - 1, i + 10)
                if any(c[j] == 1 for j in range(i + 1, horizon + 1)):
                    j_collapse = next(j for j in range(i + 1, horizon + 1) if c[j] == 1)
                    return {
                        "theta": theta,
                        "first_pass_step": i,
                        "first_collapse_step": j_collapse,
                        "lead_steps": j_collapse - i,
                    }
    return None


def _write_tsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        f.write("# Pillar 2 ZVF dynamics — iter14. Source: platform_modal/scripts/zvf_dynamics.py\n")


def main():
    runs = []
    runs.extend(_load_groupsize_zvf_runs())
    runs.extend(_load_variance_mitigation_runs())
    runs.extend(_load_tinker_prompt_zvf())

    summary_rows = []
    lead_rows = []
    for r in runs:
        s = _summarize_run(r)
        s["kind"] = r["kind"]
        s["method"] = r["method"]
        s["seed"] = r["seed"]
        summary_rows.append(s)
        lead = _lead_time(r)
        if lead is not None:
            lead_rows.append({
                "kind": r["kind"],
                "method": r["method"],
                "seed": r["seed"],
                "n_steps": r["n_steps"],
                "mean_zvf": r["mean_zvf"],
                **lead,
            })

    summary_path = os.path.join(RESULTS_DIR, "zvf_dynamics_summary.tsv")
    lead_path = os.path.join(RESULTS_DIR, "zvf_dynamics_leadtime.tsv")
    _write_tsv(summary_path, summary_rows)
    _write_tsv(lead_path, lead_rows)

    # method pool: aggregate lead events by method
    by_method = defaultdict(list)
    for r in summary_rows:
        by_method[(r["kind"], r["method"])].append(r)
    pool_rows = []
    for (kind, method), rs in sorted(by_method.items()):
        n = len(rs)
        ac1 = [r["zvf_lag1"] for r in rs if not math.isnan(r["zvf_lag1"])]
        mean_zvf = [r["mean_zvf"] for r in rs]
        std_zvf = [r["zvf_std"] for r in rs]
        pool_rows.append({
            "kind": kind,
            "method": method,
            "n_runs": n,
            "mean_zvf_pool": sum(mean_zvf) / n,
            "mean_zvf_std_pool": sum(std_zvf) / n,
            "zvf_lag1_mean": (sum(ac1) / len(ac1)) if ac1 else float("nan"),
            "zvf_phase1_mean": sum(r["zvf_phase1_mean"] for r in rs if not math.isnan(r["zvf_phase1_mean"])) / n,
            "zvf_phase2_mean": sum(r["zvf_phase2_mean"] for r in rs if not math.isnan(r["zvf_phase2_mean"])) / n,
            "zvf_phase3_mean": sum(r["zvf_phase3_mean"] for r in rs if not math.isnan(r["zvf_phase3_mean"])) / n,
            "auc_above_zvf05": sum(r["auc_above_zvf05"] for r in rs) / n,
            "auc_above_zvf07": sum(r["auc_above_zvf07"] for r in rs) / n,
            "auc_above_zvf09": sum(r["auc_above_zvf09"] for r in rs) / n,
            "reward_zvf_corr": sum(r["reward_zvf_corr"] for r in rs if not math.isnan(r["reward_zvf_corr"])) / max(1, sum(1 for r in rs if not math.isnan(r["reward_zvf_corr"]))),
        })
    pool_path = os.path.join(RESULTS_DIR, "zvf_dynamics_phase.tsv")
    _write_tsv(pool_path, pool_rows)

    json_path = os.path.join(RESULTS_DIR, "zvf_dynamics.json")
    json.dump(
        {
            "n_runs": len(runs),
            "summary": summary_rows,
            "lead_events": lead_rows,
            "method_pool": pool_rows,
        },
        open(json_path, "w"),
        indent=2,
        default=str,
    )

    print(f"ZVF dynamics: {len(runs)} runs across {len(set((r['kind'], r['method']) for r in runs))} (kind,method) groups")
    print(f"  -> {summary_path}")
    print(f"  -> {lead_path}  ({len(lead_rows)} lead events)")
    print(f"  -> {pool_path}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
