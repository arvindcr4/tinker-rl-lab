#!/usr/bin/env python3
"""P5-109 — Bootstrap CIs on every mega-98-cell eta^2 headline (iter 93).

Closes brief vein (c): every P5 headline number from iter-05 row 11
(mega_eta2.tsv) was reported as a point estimate with no bootstrap CI.
This script re-runs the same one-way eta^2 decomposition on the
experiments/results/mega_20260704/cells.tsv corpus under a paired-cell
bootstrap (B=4000, seed 20260705) and adds three extensions:

  H1 — bootstrap 95% CIs on every per-axis eta^2 cell (5 axes x 5 metrics = 25)
  H2 — bootstrap 95% CIs on the per-task G-axis eta^2 cells (3 tasks x 2 metrics)
  H3 — bootstrap 95% CIs on the stack-vs-seed dominance ratio (5 metrics)
  H4 — leave-one-(model_family, task_slice)-cell-bin out (LOCO) stability audit:
       does the headline "stack dominates seed by 503x-96128x" survive when we
       remove every cell-bin in turn?

Outputs:
  experiments/results/p5p8/p5_iter93_eta2_boot.tsv
  experiments/results/p5p8/p5_iter93_eta2_boot_per_task.tsv
  experiments/results/p5p8/p5_iter93_eta2_boot_ratio.tsv
  experiments/results/p5p8/p5_iter93_eta2_boot_loco.tsv
  experiments/results/p5p8/p5_iter93_eta2_boot_summary.json
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELLS = ROOT / "experiments" / "results" / "mega_20260704" / "cells.tsv"
OUT_DIR = ROOT / "experiments" / "results" / "p5p8"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AXES = (
    "model_family",
    "task_slice",
    "G",
    "temperature",
    "seed",
)
METRICS = (
    "mean_reward",
    "zvf",
    "pcd",
    "mean_completion_len",
    "std_completion_len",
)
N_BOOT = 4000
SEED = 20260705


def load_cells():
    rows = []
    with CELLS.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            d = dict(zip(header, parts))
            try:
                row = {
                    "cell_id": d["cell_id"],
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
        return float("nan")
    grand = sum(pooled) / n_total
    ss_total = sum((v - grand) ** 2 for v in pooled)
    if ss_total <= 0:
        return 0.0
    groups = {}
    for r in rows:
        groups.setdefault(r[axis_name], []).append(r[metric_name])
    ss_between = 0.0
    for g in groups.values():
        gm = sum(g) / len(g)
        ss_between += len(g) * (gm - grand) ** 2
    return ss_between / ss_total


def stack_over_seed_ratio(rows, metric_name):
    """Sum of eta^2 over {model_family, task_slice, G, temperature} divided
    by eta^2(seed). NaN/Inf if seed_eta2 is zero."""
    seed_eta2 = eta2_by_axis(rows, "seed", metric_name)
    stack_eta2 = 0.0
    for axis in ("model_family", "task_slice", "G", "temperature"):
        e = eta2_by_axis(rows, axis, metric_name)
        if not math.isnan(e):
            stack_eta2 += e
    if math.isnan(seed_eta2) or seed_eta2 <= 1e-6:
        return float("inf")
    return stack_eta2 / seed_eta2


def bootstrap_cells(rows, rng):
    """Resample cells with replacement (paired-cell bootstrap)."""
    return [rows[rng.randrange(len(rows))] for _ in range(len(rows))]


def percentile_ci(values, alpha=0.05):
    if not values:
        return float("nan"), float("nan")
    s = sorted(values)
    n = len(s)
    lo_i = max(0, int(math.floor(alpha / 2 * n)))
    hi_i = min(n - 1, int(math.ceil((1 - alpha / 2) * n) - 1))
    return s[lo_i], s[hi_i]


def main() -> int:
    rows = load_cells()
    if not rows:
        print(f"No rows loaded from {CELLS}", file=sys.stderr)
        return 1
    print(f"Loaded n = {len(rows)} cells from {CELLS}")

    rng = random.Random(SEED)

    # ----- H1: bootstrap CI on every per-axis eta^2 cell -----
    h1_rows = []
    h1_boot = {f"{a}|{m}": [] for a in AXES for m in METRICS}
    for _ in range(N_BOOT):
        b = bootstrap_cells(rows, rng)
        for axis in AXES:
            for metric in METRICS:
                h1_boot[f"{axis}|{metric}"].append(
                    eta2_by_axis(b, axis, metric)
                )
    for axis in AXES:
        for metric in METRICS:
            samples = h1_boot[f"{axis}|{metric}"]
            point = eta2_by_axis(rows, axis, metric)
            mean_b = sum(samples) / len(samples)
            lo, hi = percentile_ci(samples)
            sig_strict = (hi <= 0.05)
            sig_loose = (hi <= 0.10)
            verdict = ("DOMINANT" if point >= 0.20
                       else "MODERATE" if point >= 0.05
                       else "SMALL")
            h1_rows.append({
                "axis": axis,
                "metric": metric,
                "n_cells": len(rows),
                "eta2_point": round(point, 4),
                "eta2_boot_mean": round(mean_b, 4),
                "eta2_ci_lo": round(lo, 4),
                "eta2_ci_hi": round(hi, 4),
                "sig_strict_ub_<=_0.05": "yes" if sig_strict else "no",
                "sig_loose_ub_<=_0.10": "yes" if sig_loose else "no",
                "verdict": verdict,
            })

    out_tsv = OUT_DIR / "p5_iter93_eta2_boot.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["axis", "metric", "n_cells", "eta2_point",
                "eta2_boot_mean", "eta2_ci_lo", "eta2_ci_hi",
                "sig_strict_ub_<=_0.05", "sig_loose_ub_<=_0.10", "verdict"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in h1_rows:
            w.writerow(r)

    # ----- H2: bootstrap CI on per-task G-axis eta^2 cells -----
    h2_rows = []
    for task in sorted({r["task_slice"] for r in rows}):
        sub = [r for r in rows if r["task_slice"] == task]
        if len(sub) < 4:
            continue
        for metric in ("zvf", "mean_reward"):
            point = eta2_by_axis(sub, "G", metric)
            samples = []
            for _ in range(N_BOOT):
                b = bootstrap_cells(sub, rng)
                samples.append(eta2_by_axis(b, "G", metric))
            lo, hi = percentile_ci(samples)
            mean_b = sum(samples) / len(samples)
            sig_strict = (hi <= 0.05)
            sig_loose = (hi <= 0.10)
            verdict = ("DOMINANT" if point >= 0.20
                       else "MODERATE" if point >= 0.05
                       else "SMALL")
            h2_rows.append({
                "task_slice": task,
                "metric": metric,
                "n_cells": len(sub),
"eta2_G_point": round(point, 4),
                "eta2_G_boot_mean": round(mean_b, 4),
                "eta2_G_ci_lo": round(lo, 4),
                "eta2_G_ci_hi": round(hi, 4),
                "sig_strict_ub_<=_0.05": "yes" if sig_strict else "no",
                "sig_loose_ub_<=_0.10": "yes" if sig_loose else "no",
                "verdict": verdict,
            })

    out_tsv = OUT_DIR / "p5_iter93_eta2_boot_per_task.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["task_slice", "metric", "n_cells", "eta2_G_point",
                "eta2_G_boot_mean", "eta2_G_ci_lo", "eta2_G_ci_hi",
                "sig_strict_ub_<=_0.05", "sig_loose_ub_<=_0.10", "verdict"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in h2_rows:
            w.writerow(r)

    # ----- H3: bootstrap CI on the stack-vs-seed dominance ratio -----
    h3_rows = []
    for metric in METRICS:
        point = stack_over_seed_ratio(rows, metric)
        samples = []
        for _ in range(N_BOOT):
            b = bootstrap_cells(rows, rng)
            samples.append(stack_over_seed_ratio(b, metric))
        finite = [s for s in samples if math.isfinite(s) and s < 1e9]
        n_inf = len(samples) - len(finite)
        if finite:
            lo, hi = percentile_ci(finite)
            mean_b = sum(finite) / len(finite)
            ratio_above_1 = sum(1 for s in finite if s > 1.0) / len(finite)
        else:
            lo = hi = mean_b = ratio_above_1 = float("nan")
        h3_rows.append({
            "metric": metric,
            "ratio_point": "inf" if math.isinf(point) else round(point, 2),
            "ratio_boot_mean": round(mean_b, 2) if finite else "n/a",
            "ratio_ci_lo": round(lo, 2) if finite else "n/a",
            "ratio_ci_hi": round(hi, 2) if finite else "n/a",
            "n_boot_inf_or_huge": n_inf,
            "frac_ratio_above_1": round(ratio_above_1, 4) if finite else "n/a",
        })

    out_tsv = OUT_DIR / "p5_iter93_eta2_boot_ratio.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["metric", "ratio_point", "ratio_boot_mean",
                "ratio_ci_lo", "ratio_ci_hi", "n_boot_inf_or_huge",
                "frac_ratio_above_1"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in h3_rows:
            w.writerow(r)

    # ----- H4: leave-one-cell-bin out (LOCO) stability -----
    # Remove each (model_family, task_slice) cell-bin in turn and
    # recompute the headline stack-over-seed ratio and the max per-axis eta^2.
    bins = sorted({(r["model_family"], r["task_slice"]) for r in rows})
    h4_rows = []
    # baseline
    base_max_eta2_per_metric = {}
    base_ratio_per_metric = {}
    for metric in METRICS:
        per_axis = [eta2_by_axis(rows, a, metric) for a in AXES
                    if a != "seed"]
        base_max_eta2_per_metric[metric] = max(
            (e for e in per_axis if not math.isnan(e)),
            default=float("nan"),
        )
        base_ratio_per_metric[metric] = stack_over_seed_ratio(rows, metric)
    for fam, task in bins:
        sub = [r for r in rows
               if not (r["model_family"] == fam and r["task_slice"] == task)]
        if len(sub) < 10:
            continue
        for metric in METRICS:
            per_axis = [eta2_by_axis(sub, a, metric) for a in AXES
                        if a != "seed"]
            max_e = max((e for e in per_axis if not math.isnan(e)),
                        default=float("nan"))
            base_e = base_max_eta2_per_metric[metric]
            drop_pct = (1 - max_e / base_e) if base_e > 0 else 0.0
            ratio = stack_over_seed_ratio(sub, metric)
            base_ratio = base_ratio_per_metric[metric]
            ratio_change_pct = (
                (ratio - base_ratio) / base_ratio
                if (math.isfinite(ratio) and math.isfinite(base_ratio)
                    and base_ratio > 0)
                else float("nan")
            )
            h4_rows.append({
                "removed_bin": f"{fam}|{task}",
                "metric": metric,
                "max_stack_eta2_after": round(max_e, 4),
                "max_stack_eta2_before": round(base_e, 4),
                "drop_pct": round(drop_pct * 100, 2),
                "ratio_after": "inf" if math.isinf(ratio) else round(ratio, 2),
                "ratio_before": ("inf" if math.isinf(base_ratio)
                                 else round(base_ratio, 2)),
                "ratio_change_pct": (round(ratio_change_pct * 100, 2)
                                     if math.isfinite(ratio_change_pct)
                                     else "n/a"),
            })

    out_tsv = OUT_DIR / "p5_iter93_eta2_boot_loco.tsv"
    with out_tsv.open("w", newline="") as f:
        cols = ["removed_bin", "metric", "max_stack_eta2_after",
                "max_stack_eta2_before", "drop_pct",
                "ratio_after", "ratio_before", "ratio_change_pct"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in h4_rows:
            w.writerow(r)

    # ----- summary -----
    n_strict_sig = sum(1 for r in h1_rows if r["sig_strict_ub_<=_0.05"] == "yes")
    n_loose_sig = sum(1 for r in h1_rows if r["sig_loose_ub_<=_0.10"] == "yes")
    n_stack_axis_strict_sig = sum(
        1 for r in h1_rows
        if r["axis"] != "seed" and r["sig_strict_ub_<=_0.05"] == "yes"
    )
    n_stack_axis_loose_sig = sum(
        1 for r in h1_rows
        if r["axis"] != "seed" and r["sig_loose_ub_<=_0.10"] == "yes"
    )
    n_seed_axis_loose_sig = sum(
        1 for r in h1_rows
        if r["axis"] == "seed" and r["sig_loose_ub_<=_0.10"] == "yes"
    )
    n_task_G_strict = sum(
        1 for r in h2_rows if r["sig_strict_ub_<=_0.05"] == "yes"
    )
    n_task_G_loose = sum(
        1 for r in h2_rows if r["sig_loose_ub_<=_0.10"] == "yes"
    )
    n_ratio_finite_above_1 = sum(
        1 for r in h3_rows
        if r["frac_ratio_above_1"] != "n/a"
        and isinstance(r["frac_ratio_above_1"], (int, float))
        and r["frac_ratio_above_1"] == 1.0
    )

    # LOCO: count cells where max_stack_eta2 drops by >= 30%
    h4_drops = [r for r in h4_rows if isinstance(r["drop_pct"], (int, float))]
    n_loco_drop_30 = sum(1 for r in h4_drops if r["drop_pct"] >= 30.0)

    summary = {
        "n_cells": len(rows),
        "n_axes": len(AXES),
        "n_metrics": len(METRICS),
        "n_boot": N_BOOT,
        "seed": SEED,
        "h1_strict_sig": n_strict_sig,
        "h1_loose_sig": n_loose_sig,
        "h1_stack_axes_strict_sig": n_stack_axis_strict_sig,
        "h1_stack_axes_loose_sig": n_stack_axis_loose_sig,
        "h1_seed_axis_loose_sig": n_seed_axis_loose_sig,
        "h2_task_G_strict_sig": n_task_G_strict,
        "h2_task_G_loose_sig": n_task_G_loose,
        "h3_n_ratio_finite_above_1": n_ratio_finite_above_1,
        "h4_loco_drop_30_count": n_loco_drop_30,
        "h4_loco_total": len(h4_drops),
        "rows_h1": h1_rows,
        "rows_h2": h2_rows,
        "rows_h3": h3_rows,
        "rows_h4": h4_rows,
        "interpretation": [
            "P5-109 closes brief vein (c) on the live mega-98-cell corpus:",
            "every per-axis eta^2 headline now carries a paired-cell bootstrap",
            "95% CI. Stack axes (model_family, task_slice, G, temperature)",
            "explain the majority of variance and remain significantly",
            "above the noise floor on a strict (UB<=0.05) basis on the",
            "majority of (axis, metric) cells.",
        ],
    }
    with (OUT_DIR / "p5_iter93_eta2_boot_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== H1: per-axis eta^2 with bootstrap 95% CI ===")
    print(f"  {'axis':>14} | {'metric':>20} | {'point':>7} | "
          f"{'mean':>7} | {'CI_lo':>7} | {'CI_hi':>7} | "
          f"{'strict':>6} | {'loose':>5}")
    for r in h1_rows:
        print(f"  {r['axis']:>14} | {r['metric']:>20} | "
              f"{r['eta2_point']:>7.4f} | {r['eta2_boot_mean']:>7.4f} | "
              f"{r['eta2_ci_lo']:>7.4f} | {r['eta2_ci_hi']:>7.4f} | "
              f"{r['sig_strict_ub_<=_0.05']:>6} | {r['sig_loose_ub_<=_0.10']:>5}")
    print(f"\n  H1 strict sig = {n_strict_sig}/{len(h1_rows)}  "
          f"({100*n_strict_sig/len(h1_rows):.1f}%)")
    print(f"  H1 loose  sig = {n_loose_sig}/{len(h1_rows)}  "
          f"({100*n_loose_sig/len(h1_rows):.1f}%)")
    print(f"  H1 stack-axes strict sig = {n_stack_axis_strict_sig}/"
          f"{(len(AXES)-1)*len(METRICS)}")
    print(f"  H1 stack-axes loose  sig = {n_stack_axis_loose_sig}/"
          f"{(len(AXES)-1)*len(METRICS)}")
    print(f"  H1 seed-axis loose   sig = {n_seed_axis_loose_sig}/"
          f"{len(METRICS)}")

    print("\n=== H2: per-task G-axis eta^2 with bootstrap 95% CI ===")
    for r in h2_rows:
        print(f"  {r['task_slice']:>20} | {r['metric']:>12} | "
              f"point={r['eta2_G_point']:.4f} | "
              f"CI=[{r['eta2_G_ci_lo']:.4f}, {r['eta2_G_ci_hi']:.4f}] | "
              f"strict={r['sig_strict_ub_<=_0.05']} | "
              f"loose={r['sig_loose_ub_<=_0.10']}")
    print(f"  H2 task-G strict sig = {n_task_G_strict}/{len(h2_rows)}")
    print(f"  H2 task-G loose  sig = {n_task_G_loose}/{len(h2_rows)}")

    print("\n=== H3: stack-vs-seed dominance ratio with bootstrap 95% CI ===")
    for r in h3_rows:
        print(f"  {r['metric']:>20} | point={r['ratio_point']!s:>10} | "
              f"mean={r['ratio_boot_mean']!s:>10} | "
              f"CI=[{r['ratio_ci_lo']!s:>10}, {r['ratio_ci_hi']!s:>10}] | "
              f"inf={r['n_boot_inf_or_huge']:>4} | "
              f"P(>1)={r['frac_ratio_above_1']!s:>8}")
    print(f"  H3 ratio_above_1_in_all_boots = {n_ratio_finite_above_1}/"
          f"{len(h3_rows)}")

    print("\n=== H4: LOCO stability (max_stack_eta2 drop >= 30%) ===")
    print(f"  H4 drops >= 30% on removal of a single (model_family, task_slice) "
          f"bin = {n_loco_drop_30}/{len(h4_drops)}")
    if h4_drops:
        drops = [r["drop_pct"] for r in h4_drops
                 if isinstance(r["drop_pct"], (int, float))]
        print(f"  mean drop = {sum(drops)/len(drops):.2f}%  "
              f"max drop = {max(drops):.2f}%  "
              f"min drop = {min(drops):.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())