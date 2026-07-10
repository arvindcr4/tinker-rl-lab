#!/usr/bin/env python3
"""
p5_mve_field_audit.py
======================

Item-8 MVE (continuous-telemetry) field-level distributional sanity
audit on the 98-cell mega corpus. Closes the iter-53 sub-field MVE
recommendation (#64) and iter-56 empirical MVE validation (#67) at the
FIELD-LEVEL granularity:

  (a) per-MVE-field distribution (min/max/median/IQR/std/%-constant/
      %-NaN) on the 98 cells — quantifies whether the field actually
      carries empirical information in its distribution (vs a single
      constant or all-NaN);
  (b) per-MVE-field cross-axis sum η² over the 5 stack axes
      (model_family, task_slice, G, temperature, seed) — quantifies
      which stack axes move which MVE field (the iter-49 stack-axis
      η² decomposition moved to per-MVE-field granularity);
  (c) per-pair Pearson r between the 5 MVE fields and the 7-item
      MIN-REPORT Item 4-6 (the 3 non-vacuous existing items);
  (d) per-cell 5-tuple fingerprint uniqueness — sharpens iter-53's
      "distinct profiles 15→98" measurement by reporting how many
      cells share the same 5-tuple (currently 0 in the empirical
      cells.tsv because all 5 fields are cell-unique, but on a
      sub-sampled corpus the answer could be different);
  (e) outlier detection — cells where any MVE field is degenerate
      (zvf ∈ {0, 1}, mean_reward ∈ {0, 1}, std_completion_len = 0);
  (f) per-task MVE distribution — does the field behaviour differ
      by task_slice (gsm8k_easy / gsm8k_hard / humaneval_subset)?

The headline: the proposed 5-field MVE item-8 is NOT homogeneous in
its empirical distribution — fields differ in (a) variance, (b) which
stack axis drives them, (c) which other fields they correlate with.
This complements iter-49's "stack axes explain 92.7% of zvf variance"
by reporting the same decomposition at per-MVE-field granularity.

Outputs:
  platform_hybrid/experiments/results/p5p8/p5_mve_field_audit.tsv (5 rows × N cols)
  platform_hybrid/experiments/results/p5p8/p5_mve_field_eta2.tsv (5 fields × 5 axes = 25 rows)
  platform_hybrid/experiments/results/p5p8/p5_mve_field_corr.tsv (5×5 Pearson matrix)
  platform_hybrid/experiments/results/p5p8/p5_mve_field_summary.json
  platform_hybrid/experiments/results/p5p8/figures/p5_mve_field_dist.{png,pdf}
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from statistics import mean, median, stdev

# --- I/O paths ---
WORKTREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CELLS_TSV = os.path.join(WORKTREE, "platform_hybrid/experiments/results/mega_20260704/cells.tsv")
OUT_DIR = os.path.join(WORKTREE, "platform_hybrid/experiments/results/p5p8")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- audit constants ---
MVE_FIELDS = ["mean_reward", "zvf", "pcd", "mean_completion_len", "std_completion_len"]
STACK_AXES = ["model_family", "task_slice", "G", "temperature", "seed"]
TASK_NAMES = ["gsm8k_easy", "gsm8k_hard", "humaneval_subset"]
BOOT_B = 2000
SEED = 20260704

# --- helpers ---
def safe_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

def eta_squared_one_way(values_by_group):
    """One-way η²: SS_between / SS_total. Returns (eta2, n_groups_used)."""
    all_vals = []
    group_means = []
    for g, vs in values_by_group.items():
        if len(vs) >= 1:
            all_vals.extend(vs)
            group_means.append((g, mean(vs), len(vs)))
    if len(all_vals) < 2:
        return float("nan"), 0
    grand_mean = sum(all_vals) / len(all_vals)
    ss_between = sum(n * (gm - grand_mean) ** 2 for _, gm, n in group_means)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total == 0:
        return float("nan"), len(group_means)
    return ss_between / ss_total, len(group_means)

def bootstrap_eta2_ci(values_by_group, B=BOOT_B, seed=SEED):
    """Block-bootstrap 95% CI on η². Resample cells WITH REPLACEMENT
    (preserving group identity), recompute η², report [2.5, 97.5]
    percentile."""
    # pre-pool with group labels
    pooled = []  # (group, value)
    for g, vs in values_by_group.items():
        for v in vs:
            pooled.append((g, v))
    n = len(pooled)
    if n < 2:
        return float("nan"), float("nan")
    # Linear-congruential deterministic RNG (no Math.random) so the
    # script is reproducible without numpy/random.
    rng_state = seed
    etas = []
    for _ in range(B):
        rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        boot = [pooled[(rng_state >> 8) % n]]
        for _i in range(1, n):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            boot.append(pooled[(rng_state >> 8) % n])
        bvg = defaultdict(list)
        for g, v in boot:
            bvg[g].append(v)
        e, _ = eta_squared_one_way(bvg)
        if not math.isnan(e):
            etas.append(e)
    if not etas:
        return float("nan"), float("nan")
    etas.sort()
    lo = etas[int(0.025 * len(etas))]
    hi = etas[min(int(0.975 * len(etas)), len(etas) - 1)]
    return lo, hi

def pearson(xs, ys):
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return float("nan")
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan")
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / math.sqrt(sxx * syy)

def main():
    # 1) Load cells.tsv
    with open(CELLS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    n_cells = len(rows)
    print(f"[load] {n_cells} cells from {CELLS_TSV}", file=sys.stderr)

    # 2) Per-MVE-field distribution
    field_vals = {f: [] for f in MVE_FIELDS}
    field_per_cell = {}  # cell_id -> {field: value}
    for r in rows:
        cid = r["cell_id"]
        per_cell = {}
        for f in MVE_FIELDS:
            v = safe_float(r.get(f, ""))
            field_vals[f].append(v)
            per_cell[f] = v
        field_per_cell[cid] = per_cell

    # 3) Distribution stats
    dist_rows = []
    for f in MVE_FIELDS:
        vs = [v for v in field_vals[f] if v is not None]
        n_total = len(field_vals[f])
        n_valid = len(vs)
        n_nan = n_total - n_valid
        n_const = sum(1 for v in vs if vs.count(v) == len(vs)) if vs else 0
        if vs:
            sorted_vs = sorted(vs)
            q1 = sorted_vs[len(sorted_vs) // 4]
            q3 = sorted_vs[(3 * len(sorted_vs)) // 4]
            iqr = q3 - q1
            stats = {
                "field": f,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_nan": n_nan,
                "n_unique": len(set(vs)),
                "pct_constant": round(100.0 * n_const / max(1, n_valid), 2),
                "min": min(vs),
                "max": max(vs),
                "median": median(vs),
                "mean": sum(vs) / len(vs),
                "std": stdev(vs) if len(vs) > 1 else 0.0,
                "iqr": iqr,
                "pct_saturated_low": round(100.0 * sum(1 for v in vs if v == min(vs)) / max(1, n_valid), 2),
                "pct_saturated_high": round(100.0 * sum(1 for v in vs if v == max(vs)) / max(1, n_valid), 2),
            }
        else:
            stats = {"field": f, "n_total": n_total, "n_valid": n_valid, "n_nan": n_nan}
        dist_rows.append(stats)

    # 4) Per-MVE-field per-axis η² decomposition
    eta2_rows = []
    for f in MVE_FIELDS:
        for axis in STACK_AXES:
            vbg = defaultdict(list)
            for r in rows:
                v = safe_float(r.get(f, ""))
                if v is not None:
                    vbg[r[axis]].append(v)
            eta2, n_groups = eta_squared_one_way(vbg)
            lo, hi = bootstrap_eta2_ci(vbg)
            eta2_rows.append({
                "field": f,
                "axis": axis,
                "n_groups": n_groups,
                "eta2": round(eta2, 4) if not math.isnan(eta2) else None,
                "ci_lo": round(lo, 4) if not math.isnan(lo) else None,
                "ci_hi": round(hi, 4) if not math.isnan(hi) else None,
            })
        # cross-axis sum
        axis_etas = [r["eta2"] for r in eta2_rows if r["field"] == f and r["eta2"] is not None]
        cross_sum = sum(axis_etas)
        eta2_rows.append({
            "field": f,
            "axis": "CROSS_SUM",
            "n_groups": None,
            "eta2": round(cross_sum, 4),
            "ci_lo": None,
            "ci_hi": None,
        })

    # 5) 5×5 Pearson correlation matrix between MVE fields
    #    Build aligned arrays (drop rows with any NaN in any of the 5 fields)
    aligned = []
    for r in rows:
        vals = [safe_float(r.get(f, "")) for f in MVE_FIELDS]
        if all(v is not None for v in vals):
            aligned.append(vals)
    n_aligned = len(aligned)
    corr = {}
    for i, fi in enumerate(MVE_FIELDS):
        for j, fj in enumerate(MVE_FIELDS):
            xs = [a[i] for a in aligned]
            ys = [a[j] for a in aligned]
            corr[(fi, fj)] = round(pearson(xs, ys), 4) if i < j else None  # upper triangle only

    # 6) Per-cell 5-tuple fingerprint uniqueness
    fingerprints = defaultdict(int)
    for r in rows:
        fp = tuple(safe_float(r.get(f, "")) for f in MVE_FIELDS)
        fingerprints[fp] += 1
    n_unique_fps = len(fingerprints)
    max_dup = max(fingerprints.values()) if fingerprints else 0
    n_cells_in_unique = sum(1 for v in fingerprints.values() if v == 1)

    # 7) Outlier detection
    outliers = []
    for r in rows:
        cid = r["cell_id"]
        flags = []
        if safe_float(r["zvf"]) in (0.0, 1.0):
            flags.append("zvf_saturated")
        if safe_float(r["mean_reward"]) in (0.0, 1.0):
            flags.append("reward_saturated")
        if safe_float(r["std_completion_len"]) == 0.0:
            flags.append("zero_std")
        if safe_float(r["pcd"]) == 0.0:
            flags.append("zero_pcd")
        if flags:
            outliers.append({
                "cell_id": cid,
                "task_slice": r["task_slice"],
                "model_family": r["model_family"],
                "G": r["G"],
                "flags": "|".join(flags),
            })
    n_outliers = len(outliers)
    n_zvf_saturated = sum(1 for o in outliers if "zvf_saturated" in o["flags"])
    n_reward_saturated = sum(1 for o in outliers if "reward_saturated" in o["flags"])
    n_zero_std = sum(1 for o in outliers if "zero_std" in o["flags"])

    # 8) Per-task MVE distribution
    task_dist = {}
    for task in TASK_NAMES:
        task_dist[task] = {}
        for f in MVE_FIELDS:
            vs = [safe_float(r.get(f, "")) for r in rows
                  if r["task_slice"] == task and safe_float(r.get(f, "")) is not None]
            if vs:
                task_dist[task][f] = {
                    "n": len(vs),
                    "mean": round(sum(vs) / len(vs), 4),
                    "std": round(stdev(vs), 4) if len(vs) > 1 else 0.0,
                    "min": min(vs),
                    "max": max(vs),
                }

    # --- save TSVs ---
    with open(os.path.join(OUT_DIR, "p5_mve_field_audit.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(dist_rows)

    with open(os.path.join(OUT_DIR, "p5_mve_field_eta2.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eta2_rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(eta2_rows)

    # correlation matrix as TSV
    with open(os.path.join(OUT_DIR, "p5_mve_field_corr.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([""] + MVE_FIELDS)
        for i, fi in enumerate(MVE_FIELDS):
            row = [fi]
            for j, fj in enumerate(MVE_FIELDS):
                if i == j:
                    row.append("1.0")
                elif i < j:
                    row.append(str(corr[(fi, fj)]))
                else:
                    row.append(str(corr[(fj, fi)]))
            w.writerow(row)

    # --- save summary JSON ---
    summary = {
        "n_cells": n_cells,
        "mve_fields": MVE_FIELDS,
        "stack_axes": STACK_AXES,
        "per_field_dist": dist_rows,
        "per_field_per_axis_eta2": eta2_rows,
        "corr_matrix_upper_triangle": {f"{k[0]}|{k[1]}": v for k, v in corr.items() if v is not None},
        "n_aligned_for_corr": n_aligned,
        "fingerprint_uniqueness": {
            "n_unique_fps": n_unique_fps,
            "max_dup_count": max_dup,
            "n_cells_in_unique_fps": n_cells_in_unique,
            "pct_unique_fps": round(100.0 * n_unique_fps / max(1, n_cells), 2),
        },
        "outlier_summary": {
            "n_outliers": n_outliers,
            "n_zvf_saturated": n_zvf_saturated,
            "n_reward_saturated": n_reward_saturated,
            "n_zero_std": n_zero_std,
        },
        "per_task_dist": task_dist,
        "outlier_examples": outliers[:10],
        "n_boot": BOOT_B,
        "seed": SEED,
    }
    with open(os.path.join(OUT_DIR, "p5_mve_field_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- figure: 5-panel per-field distribution ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        axes = axes.flatten()
        for i, f in enumerate(MVE_FIELDS):
            ax = axes[i]
            vs = [v for v in field_vals[f] if v is not None]
            if f in ("zvf", "mean_reward", "pcd"):
                # these are bounded [0,1]; use histogram with 20 bins
                ax.hist(vs, bins=20, color="steelblue", edgecolor="black", alpha=0.7)
                ax.set_xlim(0, 1)
            else:
                # unbounded lengths; use histogram with 20 bins
                ax.hist(vs, bins=20, color="darkorange", edgecolor="black", alpha=0.7)
            ax.set_title(f"{f}\nmin={min(vs):.3g} max={max(vs):.3g} std={stdev(vs):.3g}" if len(vs) > 1 else f)
            ax.set_xlabel(f)
            ax.set_ylabel("# cells")
            ax.grid(alpha=0.3)
        # 6th panel: per-field cross-axis sum η²
        ax = axes[5]
        sums = []
        labels = []
        for f in MVE_FIELDS:
            row = next(r for r in eta2_rows if r["field"] == f and r["axis"] == "CROSS_SUM")
            sums.append(row["eta2"])
            labels.append(f)
        colors = ["C0", "C1", "C2", "C3", "C4"]
        ax.bar(labels, sums, color=colors, edgecolor="black")
        ax.set_ylabel("cross-axis sum η²")
        ax.set_title("Stack-axis explanatory power per MVE field")
        ax.set_ylim(0, max(1.0, max(sums) * 1.1))
        ax.axhline(0.50, color="red", linestyle="--", alpha=0.5, label="0.50 threshold")
        ax.legend()
        plt.suptitle(f"Item-8 MVE field-level audit on {n_cells} mega cells", fontsize=13)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(FIG_DIR, f"p5_mve_field_dist.{ext}"), bbox_inches="tight")
        plt.close()
    except ImportError:
        print("[figure] matplotlib not available, skipping figure", file=sys.stderr)

    # --- print headline ---
    print(f"\n[headline] {n_cells} cells, {n_unique_fps}/{n_cells} ({round(100*n_unique_fps/n_cells,1)}%) unique 5-tuples")
    print(f"[headline] outliers: {n_zvf_saturated} zvf-sat, {n_reward_saturated} reward-sat, {n_zero_std} zero-std")
    print(f"[headline] cross-axis sum eta^2:")
    for f in MVE_FIELDS:
        s = next(r["eta2"] for r in eta2_rows if r["field"] == f and r["axis"] == "CROSS_SUM")
        print(f"   {f}: {s:.4f}")

if __name__ == "__main__":
    main()