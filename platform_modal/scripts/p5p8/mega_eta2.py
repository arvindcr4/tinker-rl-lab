#!/usr/bin/env python3
"""P5-11 — Stack-axis eta^2 decomposition on the live 98-cell mega corpus.

Loads platform_hybrid/experiments/results/mega_20260704/cells.tsv (98 completed cells =
2 models * 3 task_slices * 5 G * 2 temperatures * 2 seeds, minus a few that
are still running) and decomposes the variance in {zvf, mean_reward, pcd,
mean_completion_len, std_completion_len} into five "stack axes":

  A  model_family   (meta-llama/Llama-3.2-3B vs Qwen/Qwen3.5-4B)
  B  task_slice     (humaneval_subset, gsm8k_easy, gsm8k_hard)
  C  G              (2, 4, 8, 16, 32)            -- the canonical stack axis
  D  temperature    (0.6, 1.0)
  E  seed           (0, 1)                        -- the noise axis (control)

eta^2 = SS_between / SS_total (one-way ANOVA per axis). When an axis is a
true stack axis, its eta^2 for stack-driven telemetry (ZVF, mean_reward,
mean_completion_len) should dwarf eta^2(seed); that is the quantitative
counterpart to the Pillar-1 "stack conditions everything" claim, now on the
98-cell live mega corpus rather than the 4-method N2 tensors.

Outputs:
  platform_hybrid/experiments/results/p5p8/mega_eta2.tsv
  platform_hybrid/experiments/results/p5p8/mega_eta2.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8" / "review16_v2"
# Versioned protocol outputs; historical artifacts are not overwritten.


AXES = (
    ("model_family", 4),
    ("task_slice", 5),
    ("G", 6),
    ("temperature", 7),
    ("seed", 8),
)
METRICS = ("mean_reward", "zvf", "pcd", "mean_completion_len", "std_completion_len")


def load_cells():
    rows = []
    with CELLS.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            d = dict(zip(header, parts, strict=False))
            try:
                row = {
                    "model_family": d["model_family"],
                    "task_slice": d["task_slice"],
                    "G": int(d["G"]),
                    "temperature": float(d["temperature"]),
                    "seed": int(d["seed"]),
                    "mean_reward": float(d["mean_reward"]),
                    "zvf": float(d["zvf"]),
                    "pcd": float(d["pcd"]),
                    "mean_completion_len": float(d["mean_completion_len"]),
                    "std_completion_len": float(d["std_completion_len"]),
                }
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def eta2_by_axis(rows, axis_name, metric_name):
    """One-way eta^2 for `metric_name` split by `axis_name` values."""
    pooled = [r[metric_name] for r in rows]
    n_total = len(pooled)
    if n_total < 2:
        return float("nan"), [], 0, n_total
    grand = sum(pooled) / n_total
    ss_total = sum((v - grand) ** 2 for v in pooled)
    if ss_total <= 0:
        return 0.0, [], 0, n_total
    groups = {}
    for r in rows:
        groups.setdefault(r[axis_name], []).append(r[metric_name])
    ss_between = 0.0
    ss_within = 0.0
    group_stats = []
    for k, g in groups.items():
        gm = sum(g) / len(g)
        ss_between += len(g) * (gm - grand) ** 2
        ss_within += sum((v - gm) ** 2 for v in g)
        group_stats.append(
            {
                "value": k,
                "n": len(g),
                "mean": round(gm, 6),
                "var_within": round(sum((v - gm) ** 2 for v in g) / max(1, len(g) - 1), 6),
            }
        )
    f_ratio = (
        (ss_between / max(1, len(groups) - 1)) / (ss_within / max(1, n_total - len(groups)))
        if ss_within > 0
        else float("inf")
    )
    return ss_between / ss_total, group_stats, len(groups), n_total


def omega2_by_axis(rows, axis_name, metric_name):
    """Standard one-way omega squared; negative estimates are retained."""
    pooled = [r[metric_name] for r in rows]
    n_total = len(pooled)
    if n_total < 2:
        return float("nan")
    grand = sum(pooled) / n_total
    ss_total = sum((v - grand) ** 2 for v in pooled)
    if ss_total <= 0:
        return 0.0
    groups = {}
    for r in rows:
        groups.setdefault(r[axis_name], []).append(r[metric_name])
    k = len(groups)
    if k < 2:
        return 0.0
    ss_within = 0.0
    for g in groups.values():
        gm = sum(g) / len(g)
        ss_within += sum((v - gm) ** 2 for v in g)
    if n_total <= k:
        return float("nan")
    ms_within = ss_within / (n_total - k)
    omega2 = (ss_total - ss_within - (k - 1) * ms_within) / (ss_total + ms_within)
    return omega2


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_cells()
    if not rows:
        print(f"No rows loaded from {CELLS}", file=sys.stderr)
        return 1
    print(f"Loaded n = {len(rows)} cells from {CELLS}")

    out_rows = []
    detail = {}
    for axis, _ in AXES:
        for metric in METRICS:
            eta2, group_stats, k, n = eta2_by_axis(rows, axis, metric)
            omega2 = omega2_by_axis(rows, axis, metric)
            out_rows.append(
                {
                    "axis": axis,
                    "metric": metric,
                    "n_cells": n,
                    "k_levels": k,
                    "eta2": round(eta2, 4) if not math.isnan(eta2) else "n/a",
                    "omega2": round(omega2, 4) if not math.isnan(omega2) else "n/a",
                    "verdict": (
                        "DOMINANT"
                        if isinstance(eta2, float) and eta2 >= 0.20
                        else "MODERATE"
                        if isinstance(eta2, float) and eta2 >= 0.05
                        else "SMALL"
                    ),
                }
            )
            detail[f"{axis}|{metric}"] = group_stats

    out_tsv = OUT_DIR / "mega_eta2.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["axis", "metric", "n_cells", "k_levels", "eta2", "omega2", "verdict"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    # Per-task G-axis decomposition: does G dominate ZVF everywhere or only on
    # the informative tasks? This sharpens the Pillar-1 claim by saying WHERE
    # the G axis does and does not matter.
    per_task_g = []
    for task in sorted({r["task_slice"] for r in rows}):
        sub = [r for r in rows if r["task_slice"] == task]
        if len(sub) < 4:
            continue
        for metric in ("zvf", "mean_reward"):
            eta2_g, _, k, n = eta2_by_axis(sub, "G", metric)
            per_task_g.append(
                {
                    "task_slice": task,
                    "metric": metric,
                    "n_cells": n,
                    "k_G": k,
                    "eta2_G": round(eta2_g, 4) if not math.isnan(eta2_g) else "n/a",
                    "verdict": (
                        "DOMINANT"
                        if isinstance(eta2_g, float) and eta2_g >= 0.20
                        else "MODERATE"
                        if isinstance(eta2_g, float) and eta2_g >= 0.05
                        else "SMALL"
                    ),
                }
            )

    # Headline: for each metric, compare "stack" eta^2 (model+task+G+temp) to
    # "noise" eta^2 (seed). Sum-of-SS form: stack_dominance ratio = SS_stack /
    # SS_total divided by SS_seed / SS_total.
    headline = [
        {
            "metric": metric,
            "marginal_eta2": {axis: eta2_by_axis(rows, axis, metric)[0] for axis, _ in AXES},
            "interpretation": "Marginal associations overlap and are not additive; no stack/seed ratio.",
        }
        for metric in METRICS
    ]

    summary = {
        "n_cells": len(rows),
        "axes": [a for a, _ in AXES],
        "metrics": list(METRICS),
        "rows": out_rows,
        "headline": headline,
        "per_task_G": per_task_g,
        "per_axis_groups": detail,
        "protocol_version": 2,
        "interpretation": [
            "Separate marginal associations only; correlated factors cannot be summed."
        ],
    }
    with (OUT_DIR / "mega_eta2.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("eta^2 table:")
    print(f"  {'axis':>14} | {'metric':>20} | {'eta2':>6} | {'omega2':>6} | verdict")
    for r in out_rows:
        print(
            f"  {r['axis']:>14} | {r['metric']:>20} | "
            f"{r['eta2']!s:>6} | {r['omega2']!s:>6} | {r['verdict']}"
        )
    print("\nMarginal eta squared values are non-additive; no aggregate ratio reported.")

    print("\nper-task G-axis eta^2 (where does G dominate ZVF/reward?):")
    for p in per_task_g:
        print(
            f"  {p['task_slice']:>20} | {p['metric']:>12} | "
            f"eta^2(G) = {p['eta2_G']!s:>6} | verdict={p['verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
