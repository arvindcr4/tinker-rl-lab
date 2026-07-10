#!/usr/bin/env python3
"""
P5 — Stack-conditioning generalization to the mega-cell corpus
(iter 49). Reads platform_hybrid/experiments/results/mega_20260704/cells.tsv
(98 cells of GRPO on a 2 model × 3 task × 5 G × 2 temperature
× 2 seed design) and decomposes per-cell mean_reward variance
into eta^2(model) / eta^2(task) / eta^2(G) / eta^2(temperature) /
eta^2(seed) with bootstrap 95% CIs (B=2000, cell-level resample).

Falsifiable headline (predicted): "stack axes dominate" — the sum
of the four stack eta^2 must exceed 0.50 (P5 thesis operationalised
at the 98-cell scale). For cross-paper coupling, this generalises
the iter-45 four-method same-stack eta^2(method)=0.0005 result
(algorithmic axis near-zero) into the larger claim: when only the
stack varies, the stack explains >50% of the outcome variance.

Output: platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_mega.tsv
        platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_mega_boot.tsv
        platform_hybrid/experiments/results/p5p8/p5_stack_conditioning_mega_summary.json
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/claude/tinker-rl-lab-minimax")
CELLS = ROOT / "platform_hybrid/experiments/results/mega_20260704/cells.tsv"
OUT = ROOT / "platform_hybrid/experiments/results/p5p8"
OUT.mkdir(parents=True, exist_ok=True)


def read_cells() -> list[dict]:
    """Return list of dicts for every cell with key stack axes + outcome."""
    cells = []
    with CELLS.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for row in rdr:
            try:
                cells.append(
                    dict(
                        cell_id=row["cell_id"],
                        model=row["model"],
                        task=row["task_slice"],
                        G=int(row["G"]),
                        temperature=float(row["temperature"]),
                        seed=int(row["seed"]),
                        mean_reward=float(row["mean_reward"]),
                        zvf=float(row["zvf"]),
                        pcd=float(row["pcd"]),
                        mean_completion_len=float(row["mean_completion_len"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return cells


def eta_squared(values: list[float], group_labels: list[str]) -> float:
    """One-way eta^2 = SS_between / SS_total."""
    n = len(values)
    if n < 2:
        return float("nan")
    grand = sum(values) / n
    ss_total = sum((v - grand) ** 2 for v in values)
    if ss_total == 0:
        return float("nan")
    by_group = defaultdict(list)
    for v, g in zip(values, group_labels):
        by_group[g].append(v)
    ss_between = 0.0
    for g, vs in by_group.items():
        if not vs:
            continue
        mean_g = sum(vs) / len(vs)
        ss_between += len(vs) * (mean_g - grand) ** 2
    return ss_between / ss_total


def k_groups(group_labels: list[str]) -> int:
    return len(set(group_labels))


def bootstrap_ci(
    values: list[float],
    group_labels: list[str],
    B: int = 2000,
    seed: int = 20260704,
) -> tuple[float, float]:
    """Percentile bootstrap CI on eta^2 by cell-level resample (n=len(values))."""
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        bv = [values[i] for i in idx]
        bl = [group_labels[i] for i in idx]
        boots.append(eta_squared(bv, bl))
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    return lo, hi


def main() -> None:
    cells = read_cells()
    if not cells:
        print(f"FATAL: no cells read from {CELLS}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(cells)} mega cells")

    # Per-cell mean_reward is the headline outcome
    values = [c["mean_reward"] for c in cells]

    axes = {
        "model": [c["model"] for c in cells],
        "task": [c["task"] for c in cells],
        "G": [str(c["G"]) for c in cells],
        "temperature": [str(c["temperature"]) for c in cells],
        "seed": [str(c["seed"]) for c in cells],
    }

    # Also include zvf as an outcome for cross-check
    zvf_values = [c["zvf"] for c in cells]

    # ZVF as outcome may be more useful because many mean_reward cells are 0
    # (35 cells at mean_reward=0 from the humaneval_subset failure mode).
    # Compute eta^2 for both outcomes.
    rows: list[dict] = []
    boot_rows: list[dict] = []

    def axis_pass(outcome: str, vals: list[float]) -> None:
        print(f"\n=== eta^2 decomposition on per-cell {outcome} ===")
        for axis_name, group_labels in axes.items():
            e2 = eta_squared(vals, group_labels)
            n_groups = k_groups(group_labels)
            ci_lo, ci_hi = bootstrap_ci(vals, group_labels, B=2000)
            rows.append(
                dict(
                    outcome=outcome,
                    axis=axis_name,
                    k=n_groups,
                    n_cells=len(vals),
                    eta_sq=round(e2, 6),
                    ci_lo=round(ci_lo, 6),
                    ci_hi=round(ci_hi, 6),
                    ci_excludes_zero=(ci_lo > 0.0),
                )
            )
            boot_rows.append(
                dict(
                    outcome=outcome,
                    axis=axis_name,
                    eta_sq=round(e2, 6),
                    ci_lo=round(ci_lo, 6),
                    ci_hi=round(ci_hi, 6),
                )
            )
            print(
                f"  eta^2({axis_name:11s}, k={n_groups:2d}) = {e2:.4f}  "
                f"95% CI [{ci_lo:.4f}, {ci_hi:.4f}]"
            )

    axis_pass("mean_reward", values)
    axis_pass("zvf", zvf_values)

    # Falsifiable headline: sum of stack-eta^2 (with each clamped at >=0)
    # must exceed 0.50 for "stack dominates" to hold at the 98-cell scale.
    stack_sum_mean = sum(
        max(0.0, r["eta_sq"])
        for r in rows
        if r["outcome"] == "mean_reward"
    )
    stack_sum_zvf = sum(
        max(0.0, r["eta_sq"])
        for r in rows
        if r["outcome"] == "zvf"
    )
    print("\n=== falsifiable headline ===")
    print(f"  sum(eta^2 stack axes) on mean_reward = {stack_sum_mean:.4f}")
    print(f"  sum(eta^2 stack axes) on zvf         = {stack_sum_zvf:.4f}")
    print(f"  threshold (P5 thesis)                = 0.5000")
    print(
        f"  verdict: mean_reward {'PASS' if stack_sum_mean > 0.5 else 'FAIL'}, "
        f"zvf {'PASS' if stack_sum_zvf > 0.5 else 'FAIL'}"
    )

    # Decompose the mean_reward=0 cluster (35 cells) — these are
    # trivially-explained by task (humaneval_subset). Quantify.
    zero_cells = [c for c in cells if c["mean_reward"] == 0.0]
    n_zero = len(zero_cells)
    zero_task_counts = defaultdict(int)
    for c in zero_cells:
        zero_task_counts[c["task"]] += 1
    print(f"\n=== zero-mean_reward cluster (n={n_zero}) ===")
    for task, count in sorted(zero_task_counts.items(), key=lambda x: -x[1]):
        print(f"  {task}: {count}")

    # Save TSVs
    out_main = OUT / "p5_stack_conditioning_mega.tsv"
    with out_main.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "outcome",
                "axis",
                "k",
                "n_cells",
                "eta_sq",
                "ci_lo",
                "ci_hi",
                "ci_excludes_zero",
            ],
            delimiter="\t",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_boot = OUT / "p5_stack_conditioning_mega_boot.tsv"
    with out_boot.open("w") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["outcome", "axis", "eta_sq", "ci_lo", "ci_hi"],
            delimiter="\t",
        )
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)

    summary = {
        "n_cells": len(cells),
        "axes": sorted(axes.keys()),
        "outcomes": ["mean_reward", "zvf"],
        "per_axis": [
            dict(
                outcome=r["outcome"],
                axis=r["axis"],
                k=r["k"],
                n_cells=r["n_cells"],
                eta_sq=r["eta_sq"],
                ci_lo=r["ci_lo"],
                ci_hi=r["ci_hi"],
                ci_excludes_zero=r["ci_excludes_zero"],
            )
            for r in rows
        ],
        "sum_stack_eta_sq": {
            "mean_reward": round(stack_sum_mean, 6),
            "zvf": round(stack_sum_zvf, 6),
        },
        "headline": {
            "falsifiable_claim": (
                "P5 thesis operationalised at 98-cell mega scale: "
                "sum(eta^2 stack axes) on mean_reward > 0.50 "
                "(when only the stack varies, the stack explains > 50% "
                "of outcome variance)."
            ),
            "mean_reward_sum_eta_sq": round(stack_sum_mean, 6),
            "zvf_sum_eta_sq": round(stack_sum_zvf, 6),
            "threshold": 0.5000,
            "verdict_mean_reward": "PASS" if stack_sum_mean > 0.5 else "FAIL",
            "verdict_zvf": "PASS" if stack_sum_zvf > 0.5 else "FAIL",
            "n_zero_mean_reward_cells": n_zero,
            "zero_task_counts": dict(zero_task_counts),
        },
    }
    out_summary = OUT / "p5_stack_conditioning_mega_summary.json"
    with out_summary.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {out_main}")
    print(f"Wrote {out_boot}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()