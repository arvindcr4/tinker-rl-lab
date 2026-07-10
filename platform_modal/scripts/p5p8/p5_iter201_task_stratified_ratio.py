#!/usr/bin/env python3
"""P5 task-slice stratified algorithm-vs-stack variance ratio (iter 201).

Fresh P5 vein, not in 210 prior P5 rows. Iter-193 reported a single
mega-corpus stack:label ratio (60.6x reward, 10.3x zvf, 4.8x len) and
iter-197 (paired bootstrap + worst-axis stress + jackknife + composite)
robustness-audited that headline at the corpus level. Both treated
the 98 mega cells as ONE bag.

Iter-201 closes a deeper gap: **stratify the iter-193 headline by
task_slice** (humaneval_subset / gsm8k_easy / gsm8k_hard) and ask
whether the stack-vs-label dominance SURVIVES within each task_slice.
The natural concern is that iter-193's headline is dominated by between-
task_slice variance, and within a single task_slice the algorithm
label might regain explanatory power.

5 falsifiable hypotheses (set BEFORE measurement)
-------------------------------------------------
H1 (STRATIFIED-RATIO STRICT)   Within each (task_slice, channel) cell,
   the point estimate of eta^2_stack / eta^2_algo is strictly > 1.0.
H2 (BOOTSTRAP CI WITHIN TASK)  For at least 2/3 task_slices and 3/3
   channels, the 95% stratified bootstrap CI on the ratio EXCLUDES 1.0.
H3 (BETWEEN-TASK VARIANCE)     Between-task_slice variance of the
   point ratio exceeds 0.5 on every channel (i.e., the ratio is
   measurably task-conditional).
H4 (DOMINANT AXIS CONSISTENCY) The top stack axis (model_family / G /
   temperature) is the SAME for at least 2/3 task_slices on every
   channel.
H5 (DOMINANT AXIS DEPENDS ON TASK FOR ZVF)  On the zvf channel, the
   top stack axis VARIES across task_slices (within-task top axis
   differs from the aggregate iter-193 top axis on at least 1 task).

Method
------
- Load 98 mega cells (cells.tsv) with stack fields model_family /
  task_slice / G / temperature / seed.
- Load N2 4-method panel (n2_metrics.tsv) for algorithm-axis eta^2
  (same as iter-141 / iter-193).
- For each (channel, task_slice): compute eta^2 on stack field
  family: model_family (single level per task_slice -- so within-task
  collapse) / G / temperature / seed -- and on algorithm axis (4
  methods).
- Stratified bootstrap (B=2000, seed 20260706): resample within
  (task_slice, stack_level) cells with replacement.
- Report ratios and CIs.

Outputs
-------
- platform_hybrid/experiments/results/p5p8/p5_iter201_per_task_ratio.tsv
  (channels x task_slices: point ratios + bootstrap CI)
- platform_hybrid/experiments/results/p5p8/p5_iter201_dominant_axis_per_task.tsv
  (channels x task_slices: top stack axis per (channel, task_slice))
- platform_hybrid/experiments/results/p5p8/p5_iter201_within_task_summary.tsv
  (3 channels x 3 task_slices: aggregate summary)
- platform_hybrid/experiments/results/p5p8/p5_iter201_summary.json
  (H1-H5 verdicts + per-task findings + bootstrap CIs)
"""
from __future__ import annotations
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

random.seed(20260706)
N_BOOT = 2000
SEED = 20260706

WORKTREE = Path("/home/claude/tinker-rl-lab-minimax")
CELLS_TSV = WORKTREE / "platform_hybrid/experiments/results/mega_20260704/cells.tsv"
N2_METRICS = WORKTREE / "platform_hybrid/experiments/results/n2_reward_tensor_resume/metrics.tsv"
OUT_DIR = WORKTREE / "platform_hybrid/experiments/results/p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = ["zvf", "mean_reward", "mean_completion_len"]
STACK_AXES = ["model_family", "G", "temperature", "seed"]
TASK_SLICES = ["humaneval_subset", "gsm8k_easy", "gsm8k_hard"]
ALGO_AXIS = "method"  # grpo / aero / gift / areal


def load_mega_cells() -> list[dict]:
    """Load 98 mega cells from cells.tsv."""
    rows = []
    with CELLS_TSV.open() as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            try:
                rows.append({
                    "cell_id": row["cell_id"],
                    "model_family": row["model_family"],
                    "task_slice": row["task_slice"],
                    "G": int(row["G"]),
                    "temperature": float(row["temperature"]),
                    "seed": int(row["seed"]),
                    "zvf": float(row["zvf"]),
                    "mean_reward": float(row["mean_reward"]),
                    "mean_completion_len": float(row["mean_completion_len"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def load_n2_metrics() -> list[dict]:
    """Load N2 4-method panel (160 cells: 4 methods x 40 steps)."""
    rows = []
    if not N2_METRICS.exists():
        # fallback: try alternative name
        for alt in ["zvf_iter130_method_risk.tsv", "n2_metrics.tsv"]:
            cand = WORKTREE / "platform_hybrid/experiments/results/n2_reward_tensor_resume" / alt
            if cand.exists():
                with cand.open() as fh:
                    rd = csv.DictReader(fh, delimiter="\t")
                    for row in rd:
                        if "method" in row:
                            try:
                                rows.append({
                                    "method": row["method"],
                                    "zvf": float(row.get("zvf", 0)),
                                    "reward_mean": float(row.get("reward_mean", row.get("mean_reward", 0))),
                                })
                            except (ValueError, KeyError):
                                continue
                break
    else:
        with N2_METRICS.open() as fh:
            rd = csv.DictReader(fh, delimiter="\t")
            for row in rd:
                try:
                    rows.append({
                        "method": row["method"],
                        "zvf": float(row.get("zvf", 0)),
                        "reward_mean": float(row.get("reward_mean", row.get("mean_reward", 0))),
                    })
                except (ValueError, KeyError):
                    continue
    return rows


def eta_squared(groups: list[float], values: list[float]) -> float:
    """One-way eta^2 = SS_between / SS_total.

    groups[i] = group label for values[i].
    Returns 0.0 if SS_total == 0.
    """
    if len(values) < 2:
        return 0.0
    grand_mean = sum(values) / len(values)
    ss_total = sum((v - grand_mean) ** 2 for v in values)
    if ss_total <= 0:
        return 0.0
    by_group = defaultdict(list)
    for g, v in zip(groups, values):
        by_group[g].append(v)
    ss_between = 0.0
    for g, vals in by_group.items():
        gm = sum(vals) / len(vals)
        ss_between += len(vals) * (gm - grand_mean) ** 2
    return ss_between / ss_total


def stratified_bootstrap_ratio(
    groups: list[str], values: list[float], algo_groups: list[str],
    algo_values: list[float], n_boot: int = N_BOOT, seed: int = SEED,
) -> dict:
    """Bootstrap the ratio eta^2_stack / eta^2_algo.

    - Stack axis: resample within (group, level)strata with replacement.
    - Algo axis: resample within (method) strata with replacement.
    """
    rng = random.Random(seed)
    by_stack_group = defaultdict(list)
    for g, v in zip(groups, values):
        by_stack_group[g].append(v)
    by_algo_group = defaultdict(list)
    for g, v in zip(algo_groups, algo_values):
        by_algo_group[g].append(v)

    point_stack = eta_squared(groups, values)
    point_algo = eta_squared(algo_groups, algo_values)
    if point_algo <= 0:
        point_ratio = float("inf") if point_stack > 0 else 0.0
    else:
        point_ratio = point_stack / point_algo

    boot_ratios = []
    for _ in range(n_boot):
        # resample stack within levels
        new_groups = []
        new_values = []
        for g, vals in by_stack_group.items():
            sample = [rng.choice(vals) for _ in range(len(vals))]
            new_groups.extend([g] * len(sample))
            new_values.extend(sample)
        # resample algo within methods
        new_algo_groups = []
        new_algo_values = []
        for g, vals in by_algo_group.items():
            sample = [rng.choice(vals) for _ in range(len(vals))]
            new_algo_groups.extend([g] * len(sample))
            new_algo_values.extend(sample)
        b_stack = eta_squared(new_groups, new_values)
        b_algo = eta_squared(new_algo_groups, new_algo_values)
        if b_algo > 1e-12:
            boot_ratios.append(b_stack / b_algo)
    boot_ratios.sort()
    if not boot_ratios:
        return {
            "point_stack": point_stack,
            "point_algo": point_algo,
            "point_ratio": point_ratio,
            "boot_lo": float("nan"),
            "boot_hi": float("nan"),
            "boot_median": float("nan"),
            "ci_excludes_1": False,
        }
    lo = boot_ratios[int(0.025 * len(boot_ratios))]
    hi = boot_ratios[int(0.975 * len(boot_ratios)) - 1]
    med = boot_ratios[len(boot_ratios) // 2]
    return {
        "point_stack": point_stack,
        "point_algo": point_algo,
        "point_ratio": point_ratio,
        "boot_lo": lo,
        "boot_hi": hi,
        "boot_median": med,
        "ci_excludes_1": (lo > 1.0) or (hi < 1.0),
    }


def main():
    mega = load_mega_cells()
    n2 = load_n2_metrics()

    if not mega:
        raise SystemExit("No mega cells loaded")
    if not n2:
        raise SystemExit("No N2 metrics loaded")

    # detect available task_slices
    seen_tasks = sorted(set(c["task_slice"] for c in mega))
    print(f"[iter201] {len(mega)} mega cells across {len(seen_tasks)} task slices: {seen_tasks}")
    print(f"[iter201] {len(n2)} N2 method-step cells")

    # restrict N2 to 4 methods
    methods = sorted(set(r["method"] for r in n2))
    algo_groups = [r["method"] for r in n2]

    # Per-task stratified ratios
    per_task_results = []  # rows for p5_iter201_per_task_ratio.tsv
    dominant_axis_rows = []  # rows for p5_iter201_dominant_axis_per_task.tsv
    summary_rows = []

    # For each channel, for each task_slice, for each stack axis
    for ch in CHANNELS:
        # build N2 values
        algo_values = [r.get(ch, r.get("zvf" if ch == "zvf" else "reward_mean", 0)) for r in n2]

        for task in seen_tasks:
            if task not in TASK_SLICES:
                continue  # only test the 3 declared task slices
            task_cells = [c for c in mega if c["task_slice"] == task]
            if len(task_cells) < 5:
                continue

            per_axis_eta2 = {}
            for ax in STACK_AXES:
                groups = [str(c[ax]) for c in task_cells]
                vals = [c[ch] for c in task_cells]
                eta2 = eta_squared(groups, vals)
                per_axis_eta2[ax] = eta2

            # top axis per task
            top_axis = max(per_axis_eta2, key=per_axis_eta2.get)
            top_eta2 = per_axis_eta2[top_axis]

            # ratio on top axis
            groups = [str(c[top_axis]) for c in task_cells]
            vals = [c[ch] for c in task_cells]
            boot = stratified_bootstrap_ratio(
                groups, vals, algo_groups, algo_values
            )

            per_task_results.append({
                "channel": ch,
                "task_slice": task,
                "top_axis": top_axis,
                "top_axis_eta2": f"{top_eta2:.6f}",
                "algo_eta2": f"{boot['point_algo']:.6f}",
                "point_ratio": f"{boot['point_ratio']:.4f}",
                "boot_lo": f"{boot['boot_lo']:.4f}",
                "boot_hi": f"{boot['boot_hi']:.4f}",
                "boot_median": f"{boot['boot_median']:.4f}",
                "ci_excludes_1": "yes" if boot["ci_excludes_1"] else "no",
                "n_task_cells": len(task_cells),
            })
            dominant_axis_rows.append({
                "channel": ch,
                "task_slice": task,
                "model_family_eta2": f"{per_axis_eta2['model_family']:.6f}",
                "G_eta2": f"{per_axis_eta2['G']:.6f}",
                "temperature_eta2": f"{per_axis_eta2['temperature']:.6f}",
                "seed_eta2": f"{per_axis_eta2['seed']:.6f}",
                "top_axis": top_axis,
                "top_eta2": f"{top_eta2:.6f}",
            })

            print(f"  {ch:20s} {task:18s} top={top_axis:14s} eta2={top_eta2:.4f} "
                  f"algo={boot['point_algo']:.4f} ratio={boot['point_ratio']:.2f}x "
                  f"CI=[{boot['boot_lo']:.2f}, {boot['boot_hi']:.2f}]")

    # write per-task TSV
    with (OUT_DIR / "p5_iter201_per_task_ratio.tsv").open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_task_results[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(per_task_results)
    with (OUT_DIR / "p5_iter201_dominant_axis_per_task.tsv").open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dominant_axis_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(dominant_axis_rows)

    # H1: point ratio > 1 on every (channel, task_slice) cell
    h1 = all(float(r["point_ratio"]) > 1.0 for r in per_task_results)

    # H2: bootstrap CI excludes 1 on >= 2/3 task_slices and 3/3 channels
    by_chan_task = defaultdict(list)
    for r in per_task_results:
        by_chan_task[r["channel"]].append(r["ci_excludes_1"] == "yes")
    h2_per_chan = {ch: sum(v) for ch, v in by_chan_task.items()}
    h2 = all(v >= 2 for v in h2_per_chan.values())

    # H3: between-task_slice ratio variance > 0.5 on every channel
    by_chan = defaultdict(list)
    for r in per_task_results:
        by_chan[r["channel"]].append(float(r["point_ratio"]))
    h3_per_chan = {}
    for ch, ratios in by_chan.items():
        if len(ratios) >= 2:
            v = statistics.variance(ratios)
        else:
            v = 0.0
        h3_per_chan[ch] = v
    h3 = all(v > 0.5 for v in h3_per_chan.values())

    # H4: top axis consistent for >= 2/3 task_slices on every channel
    by_chan_top = defaultdict(list)
    for r in per_task_results:
        by_chan_top[r["channel"]].append(r["top_axis"])
    h4_per_chan = {}
    for ch, tops in by_chan_top.items():
        if tops:
            most_common = max(set(tops), key=tops.count)
            consistency = tops.count(most_common) / len(tops)
        else:
            consistency = 0.0
        h4_per_chan[ch] = consistency
    h4 = all(v >= 2 / 3 for v in h4_per_chan.values())

    # H5: zvf top axis varies across tasks (NOT consistent)
    zvf_tops = by_chan_top.get("zvf", [])
    if zvf_tops:
        most_common_zvf = max(set(zvf_tops), key=zvf_tops.count)
        h5_consistency = zvf_tops.count(most_common_zvf) / len(zvf_tops)
    else:
        h5_consistency = 1.0
    h5 = h5_consistency < 1.0  # i.e., at least one zvf task has a different top axis

    summary = {
        "n_mega_cells": len(mega),
        "n_n2_cells": len(n2),
        "task_slices_seen": seen_tasks,
        "channels": CHANNELS,
        "stack_axes": STACK_AXES,
        "n_boot": N_BOOT,
        "seed": SEED,
        "h1_strict_per_task_ratio": h1,
        "h2_boot_ci_excludes_1_per_chan": h2_per_chan,
        "h2_pass": h2,
        "h3_between_task_ratio_var": h3_per_chan,
        "h3_pass": h3,
        "h4_top_axis_consistency": h4_per_chan,
        "h4_pass": h4,
        "h5_zvf_consistency": h5_consistency,
        "h5_pass": h5,
        "per_task": per_task_results,
    }

    with (OUT_DIR / "p5_iter201_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    print()
    print(f"H1 (point ratio > 1 every cell): {'PASS' if h1 else 'FAIL'}")
    print(f"H2 (CI excludes 1 >= 2/3 tasks per chan): {h2_per_chan} -> {'PASS' if h2 else 'FAIL'}")
    print(f"H3 (between-task variance > 0.5 per chan): {h3_per_chan} -> {'PASS' if h3 else 'FAIL'}")
    print(f"H4 (top axis consistent >= 2/3 per chan): {h4_per_chan} -> {'PASS' if h4 else 'FAIL'}")
    print(f"H5 (zvf top axis varies): consistency={h5_consistency:.2f} -> {'PASS' if h5 else 'FAIL'}")


if __name__ == "__main__":
    main()