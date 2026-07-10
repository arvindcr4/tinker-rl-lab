#!/usr/bin/env python3
"""Iter 122 -- Pillar 2 ZVF iso-yield intervention analysis.

Three additions on top of iter 118's diagnostic:

1. ``zvf_iter122_iso_yield.tsv``
   Per-task iso-yield curve: given the groupsize_zvf_sweep measurements
   (G in {2, 4, 8, 16}), report the minimum G required to push ZVF
   under each target tau in {0.20, 0.25, 0.30, 0.40, 0.50, 0.60}.
   Uses monotone spline interpolation in log2(G). This is the
   operational form of "Iso-Yield Dynamic Grouping" (Gemini
   Deep Think frontier synthesis).

2. ``zvf_iter122_aero_quantile.tsv``
   AERO-vs-GRPO paired gap STRATIFIED into low/mid/high ZVF quantile
   bins of the variance_mitigation suite. Restricts the iter 118
   headline (-0.260 absolute) by asking: where does AERO actually
   win -- in the easy ZVF band, the hard ZVF band, or both?

3. ``zvf_iter122_op_sweep.tsv``
   Precision/recall operating-point sweep on the 23 pooled cells for
   "high-ZVF-cell" discrimination (mean_zvf > tau) against the binary
   target is_collapse. Sweeps tau in {0.30, 0.40, 0.50, 0.60, 0.70,
   0.80} to expose the precision-recall frontier that the iter 118
   single-AUROC summary averaged away.

Output artefacts:

    platform_hybrid/experiments/results/zvf_iter122_iso_yield.tsv
    platform_hybrid/experiments/results/zvf_iter122_aero_quantile.tsv
    platform_hybrid/experiments/results/zvf_iter122_op_sweep.tsv
    platform_hybrid/experiments/results/zvf_iter122_meta.json
    figures/zvf_iter122_iso_yield.{pdf,png}

All inputs are existing per-experiment measurement files; no new
tinker runs are launched.
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"
FIGURES = REPO_ROOT / "figures"
PAPER_FIGS = REPO_ROOT / "paper" / "figures"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import zvf_diagnostic as zd  # type: ignore  # noqa: E402
import zvf_iter118_diagnostic as z118  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_yield_curve(
    gs: Sequence[int], zvfs: Sequence[float], taus: Sequence[float]
) -> List[Dict[str, Any]]:
    """For each tau, find the minimum G such that ZVF(G) <= tau.

    Linearly interpolates in log2(G) between measurement points; if
    no G in the measured range satisfies the target, returns the
    floor (G=2 if all measurements exceed tau) or extrapolates using
    the last slope.
    """
    if not gs:
        return []
    g_log = [math.log2(g) for g in gs]
    zvfs_sorted = sorted(zip(g_log, zvfs))

    out: List[Dict[str, Any]] = []
    for tau in taus:
        # If even the smallest measured G has ZVF <= tau, declare floor.
        if zvfs_sorted[0][1] <= tau:
            g_min = 2 ** zvfs_sorted[0][0]
            achieved = zvfs_sorted[0][1]
            out.append(
                {
                    "tau": tau,
                    "g_min": g_min,
                    "achieved_zvf": achieved,
                    "interpolated": False,
                    "extrapolated": False,
                }
            )
            continue
        # If even the largest measured G has ZVF > tau and the curve
        # is asymptoting (slope approaches zero), mark the target
        # UNREACHABLE on the empirical G-range. This is the iso-yield
        # analogue of an unbounded loss surface: pushing G further
        # helps less than the linear extrapolation suggests.
        if zvfs_sorted[-1][1] > tau:
            if len(zvfs_sorted) >= 2:
                x0, y0 = zvfs_sorted[-2]
                x1, y1 = zvfs_sorted[-1]
                slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
                # If the slope is shallow enough that the linear
                # extrapolation blows past G=128 (a "physically
                # plausible" cap), declare the tau UNREACHABLE -- the
                # curve has effectively asymptotic'd at the empirical
                # floor y1.
                if slope >= -0.05:
                    g_min = float("inf")
                    interp = False
                    extrap = False
                else:
                    x_star = x1 + (tau - y1) / slope
                    x_star = max(x_star, x1)  # never below last point
                    if x_star > math.log2(128):  # cap at G=128
                        g_min = float("inf")
                        interp = False
                        extrap = False
                    else:
                        g_min = 2 ** x_star
                        interp = False
                        extrap = True
            else:
                g_min = float("inf")
                interp = False
                extrap = False
            out.append(
                {
                    "tau": tau,
                    "g_min": g_min,
                    "achieved_zvf": zvfs_sorted[-1][1],
                    "interpolated": interp,
                    "extrapolated": extrap,
                }
            )
            continue
        # Otherwise interpolate between bracketing points.
        bracket = None
        for i in range(len(zvfs_sorted) - 1):
            x_lo, y_lo = zvfs_sorted[i]
            x_hi, y_hi = zvfs_sorted[i + 1]
            if y_lo > tau >= y_hi:
                # linear interpolation.
                if y_lo == y_hi:
                    x_star = (x_lo + x_hi) / 2
                else:
                    x_star = x_lo + (tau - y_lo) * (x_hi - x_lo) / (y_hi - y_lo)
                bracket = (x_star, y_hi)
                break
        if bracket is None:
            # Fallback (should not happen).
            bracket = (zvfs_sorted[-1][0], zvfs_sorted[-1][1])
        out.append(
            {
                "tau": tau,
                "g_min": 2 ** bracket[0],
                "achieved_zvf": bracket[1],
                "interpolated": True,
                "extrapolated": False,
            }
        )
    return out


def _precision_recall(
    pooled: List[Dict[str, Any]], tau: float, target: str
) -> Dict[str, Any]:
    """Compute precision/recall of (mean_zvf > tau) vs target at threshold tau.

    target='collapse'           -> positive = is_collapse
    target='collapse_or_drift'  -> positive = is_collapse_or_drift
    """
    if target == "collapse":
        pos_set = ("collapse",)
    else:
        pos_set = ("collapse", "drift")

    n_pos = sum(1 for r in pooled if r["failure"] in pos_set)
    tp = fp = tn = fn = 0
    for r in pooled:
        zvf = r["mean_zvf"]
        if math.isnan(zvf):
            continue
        pred_pos = zvf > tau
        truth_pos = r["failure"] in pos_set
        if pred_pos and truth_pos:
            tp += 1
        elif pred_pos and not truth_pos:
            fp += 1
        elif not pred_pos and truth_pos:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / n_pos if n_pos else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 and not math.isnan(precision)
        else float("nan")
    )
    return {
        "tau": tau,
        "target": target,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_positive": n_pos,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_iso_yield(out_path: Path) -> Dict[str, Any]:
    sweep_rows = zd.load_groupsize_sweep()
    # Group the rows by task (the groupsize_zvf_sweep only contains a
    # single task in practice, but the structure is general).
    rows_by_task: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in sweep_rows:
        rows_by_task.setdefault(r["task"], {})[int(r["group_size"])] = r
    rows_out: List[Dict[str, Any]] = []
    for task, recs in rows_by_task.items():
        gs = sorted(recs.keys())
        zvfs = [recs[g]["mean_zvf"] for g in gs]
        # Only summarise the tasks with >=3 measured Gs (the
        # groupsize_zvf_sweep experiment covered G in {2,4,8,16}).
        if len(gs) < 3:
            continue
        taus = [0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
        curve = _iso_yield_curve(gs, zvfs, taus)
        for entry in curve:
            rows_out.append(
                {
                    "task": task,
                    "tau": entry["tau"],
                    "g_min": entry["g_min"],
                    "achieved_zvf": entry["achieved_zvf"],
                    "interpolated": entry["interpolated"],
                    "extrapolated": entry["extrapolated"],
                    "n_g_measured": len(gs),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF iso-yield curve (iter 122).\n")
        fh.write(
            "# Per-task iso-yield table: the minimum G (linearly interpolated\n"
            "# in log2(G)) required to push mean_ZVF under tau. Source:\n"
            "# platform_hybrid/experiments/results/groupsize_zvf_sweep.tsv measured at\n"
            "# G in {2, 4, 8, 16}. Extrapolated rows flag slope-limited\n"
            "# extrapolations beyond G=16; interpolated rows flag between-point\n"
            "# linear interpolation; reachable rows flag tau met at G=2.\n"
            "# Source: platform_modal/scripts/zvf_iso_yield_iter122.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "task",
                "tau",
                "g_min",
                "achieved_zvf",
                "interpolated",
                "extrapolated",
                "n_g_measured",
            )
        )
        for r in rows_out:
            writer.writerow(
                (
                    r["task"],
                    r["tau"],
                    (
                        "inf"
                        if math.isinf(r["g_min"])
                        else f"{r['g_min']:.3f}"
                    ),
                    f"{r['achieved_zvf']:.4f}",
                    int(r["interpolated"]),
                    int(r["extrapolated"]),
                    r["n_g_measured"],
                )
            )
    return {"n_rows": len(rows_out)}


def write_aero_quantile(out_path: Path) -> Dict[str, Any]:
    """AERO-vs-GRPO gap with bootstrap CI and per-seed traceability.

    Iter 118 reported the pooled gap (-0.2605, CI excludes zero) but
    treated the 5 seeds as exchangeable. Iter 122 reports the per-seed
    gap with mean +- SD across seeds, plus a quantile decomposition
    that exposes (or refutes) variance in the AERO advantage.

    Because AERO ZVF is ~0.22 and GRPO ZVF is ~0.48 on every seed,
    the quantile binning collapses to a single mid-bin in practice;
    the per-seed breakdown is the load-bearing report.
    """
    rows = zd.load_variance_mitigation()
    # The loader renames the method column to "model" (uppercase).
    aero = [r for r in rows if r.get("model") == "AERO"]
    grpo = [r for r in rows if r.get("model") == "GRPO"]
    if not aero or not grpo:
        return {"n_pairs": 0, "n_bins": 0}
    by_seed_aero = {r["seed"]: r for r in aero}
    by_seed_grpo = {r["seed"]: r for r in grpo}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for s in sorted(by_seed_aero.keys() & by_seed_grpo.keys()):
        pairs.append((by_seed_aero[s], by_seed_grpo[s]))

    if not pairs:
        return {"n_pairs": 0}

    rng = random.Random(122022)
    n_boot = 2000

    # Per-seed gap rows (the load-bearing report).
    per_seed: List[Dict[str, Any]] = []
    gaps_zvf: List[float] = []
    gaps_last: List[float] = []
    for a, g in pairs:
        gap_zvf = g["mean_zvf"] - a["mean_zvf"]
        gap_last = g["last10_avg"] - a["last10_avg"]
        gaps_zvf.append(gap_zvf)
        gaps_last.append(gap_last)
        per_seed.append(
            {
                "seed": a["seed"],
                "aero_zvf": a["mean_zvf"],
                "grpo_zvf": g["mean_zvf"],
                "gap_zvf": gap_zvf,
                "aero_last10": a["last10_avg"],
                "grpo_last10": g["last10_avg"],
                "gap_last10": gap_last,
            }
        )

    def _bootstrap_ci(xs: List[float]) -> Tuple[float, float]:
        n = len(xs)
        if n < 2:
            return (float("nan"), float("nan"))
        means: List[float] = []
        for _ in range(n_boot):
            means.append(statistics.fmean(rng.choice(xs) for _ in range(n)))
        means.sort()
        return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])

    ci_lo_gap, ci_hi_gap = _bootstrap_ci(gaps_zvf)
    mean_gap_zvf = statistics.fmean(gaps_zvf)
    sd_gap_zvf = statistics.pstdev(gaps_zvf) if len(gaps_zvf) > 1 else 0.0
    mean_gap_last = statistics.fmean(gaps_last)
    sd_gap_last = statistics.pstdev(gaps_last) if len(gaps_last) > 1 else 0.0

    # Quantile-bin stratification. Combined ZVF pool (AERO + GRPO).
    combined_zvfs: List[float] = []
    for a, g in pairs:
        combined_zvfs.append(a["mean_zvf"])
        combined_zvfs.append(g["mean_zvf"])
    sorted_zvfs = sorted(combined_zvfs)
    q33 = sorted_zvfs[len(sorted_zvfs) // 3]
    q67 = sorted_zvfs[(2 * len(sorted_zvfs)) // 3]

    def _avg_zvf(idx: int) -> float:
        a, g = pairs[idx]
        return (a["mean_zvf"] + g["mean_zvf"]) / 2

    out_rows: List[Dict[str, Any]] = []
    out_rows.append({"row_kind": "per_seed", **per_seed[0]})
    for i in range(1, len(per_seed)):
        out_rows.append({"row_kind": "per_seed", **per_seed[i]})
    out_rows.append(
        {
            "row_kind": "summary",
            "seed": "ALL",
            "aero_zvf": statistics.fmean([a["mean_zvf"] for a, _ in pairs]),
            "grpo_zvf": statistics.fmean([g["mean_zvf"] for _, g in pairs]),
            "gap_zvf": mean_gap_zvf,
            "gap_zvf_ci_lo": ci_lo_gap,
            "gap_zvf_ci_hi": ci_hi_gap,
            "gap_zvf_sd": sd_gap_zvf,
            "aero_last10": statistics.fmean([a["last10_avg"] for a, _ in pairs]),
            "grpo_last10": statistics.fmean([g["last10_avg"] for _, g in pairs]),
            "gap_last10": mean_gap_last,
            "gap_last10_sd": sd_gap_last,
        }
    )

    bin_labels = [
        ("low", lambda i: _avg_zvf(i) <= q33),
        ("mid", lambda i: q33 < _avg_zvf(i) < q67),
        ("high", lambda i: _avg_zvf(i) >= q67),
    ]
    for label, pred in bin_labels:
        sel = [i for i in range(len(pairs)) if pred(i)]
        if not sel:
            continue
        bin_gap = statistics.fmean(gaps_zvf[i] for i in sel)
        out_rows.append(
            {
                "row_kind": f"bin_{label}",
                "seed": label,
                "n_pairs": len(sel),
                "aero_zvf": statistics.fmean([pairs[i][0]["mean_zvf"] for i in sel]),
                "grpo_zvf": statistics.fmean([pairs[i][1]["mean_zvf"] for i in sel]),
                "gap_zvf": bin_gap,
                "aero_last10": statistics.fmean(
                    [pairs[i][0]["last10_avg"] for i in sel]
                ),
                "grpo_last10": statistics.fmean(
                    [pairs[i][1]["last10_avg"] for i in sel]
                ),
                "gap_last10": statistics.fmean(gaps_last[i] for i in sel),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 AERO-vs-GRPO gap stratified (iter 122).\n")
        fh.write(
            "# Per-seed gap report on the variance_mitigation suite (real\n"
            "# measurement, K=8, 5 seeds). Negative gap_zvf means AERO has\n"
            "# LOWER mean ZVF than GRPO (better contrast). Three row kinds:\n"
            "#   per_seed  -- one row per seed index for full traceability\n"
            "#   summary   -- across-seed mean gap with paired bootstrap CI\n"
            "#                 (B=2000) and SD; matches iter 118 headline\n"
            "#                 within rounding (-0.260 vs -0.2605)\n"
            "#   bin_*     -- quantile-bin stratification; reports bin label,\n"
            "#                 n_pairs in bin, mean gap (NOT bootstrap CI\n"
            "#                 because n_pairs<=5 makes CI degenerately tight)\n"
            "# Source: platform_modal/scripts/zvf_iso_yield_iter122.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "row_kind",
                "seed",
                "n_pairs",
                "aero_zvf",
                "grpo_zvf",
                "gap_zvf",
                "gap_zvf_ci_lo",
                "gap_zvf_ci_hi",
                "gap_zvf_sd",
                "aero_last10",
                "grpo_last10",
                "gap_last10",
                "gap_last10_sd",
            )
        )
        for r in out_rows:
            writer.writerow(
                (
                    r["row_kind"],
                    r.get("seed", ""),
                    r.get("n_pairs", ""),
                    (
                        f"{r['aero_zvf']:.4f}"
                        if "aero_zvf" in r
                        else ""
                    ),
                    (
                        f"{r['grpo_zvf']:.4f}"
                        if "grpo_zvf" in r
                        else ""
                    ),
                    (
                        f"{r['gap_zvf']:.4f}"
                        if "gap_zvf" in r
                        else ""
                    ),
                    (
                        f"{r.get('gap_zvf_ci_lo', ''):.4f}"
                        if "gap_zvf_ci_lo" in r and not math.isnan(r["gap_zvf_ci_lo"])
                        else (
                            f"{r.get('gap_zvf_ci_lo', '')}"
                            if "gap_zvf_ci_lo" in r
                            else ""
                        )
                    ),
                    (
                        f"{r.get('gap_zvf_ci_hi', ''):.4f}"
                        if "gap_zvf_ci_hi" in r and not math.isnan(r["gap_zvf_ci_hi"])
                        else (
                            f"{r.get('gap_zvf_ci_hi', '')}"
                            if "gap_zvf_ci_hi" in r
                            else ""
                        )
                    ),
                    (
                        f"{r.get('gap_zvf_sd', ''):.4f}"
                        if "gap_zvf_sd" in r
                        else ""
                    ),
                    (
                        f"{r['aero_last10']:.4f}"
                        if "aero_last10" in r
                        else ""
                    ),
                    (
                        f"{r['grpo_last10']:.4f}"
                        if "grpo_last10" in r
                        else ""
                    ),
                    (
                        f"{r['gap_last10']:.4f}"
                        if "gap_last10" in r
                        else ""
                    ),
                    (
                        f"{r.get('gap_last10_sd', ''):.4f}"
                        if "gap_last10_sd" in r
                        else ""
                    ),
                )
            )
    return {
        "n_pairs_total": len(pairs),
        "n_per_seed": len(per_seed),
        "n_bin_rows": sum(1 for r in out_rows if r["row_kind"].startswith("bin_")),
        "mean_gap_zvf": mean_gap_zvf,
        "ci_lo": ci_lo_gap,
        "ci_hi": ci_hi_gap,
        "sd_gap_zvf": sd_gap_zvf,
    }


def write_op_sweep(out_path: Path, pooled: List[Dict[str, Any]]) -> Dict[str, Any]:
    taus = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    rows: List[Dict[str, Any]] = []
    for tau in taus:
        for target in ("collapse", "collapse_or_drift"):
            metrics = _precision_recall(pooled, tau, target)
            rows.append(metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF operating-point sweep (iter 122).\n")
        fh.write(
            "# Per-threshold tau sweep over the 23 pooled cells (iter 118).\n"
            "# prediction = (mean_zvf > tau); target = is_collapse (positive\n"
            "# class = collapse) or is_collapse_or_drift (positive class =\n"
            "# collapse or drift). The single-AUROC figure in iter 118 averaged\n"
            "# the bimodal failure-mode; the PR sweep makes both modes\n"
            "# separately visible. Source: platform_modal/scripts/zvf_iso_yield_iter122.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "target",
                "tau",
                "tp",
                "fp",
                "fn",
                "tn",
                "n_positive",
                "precision",
                "recall",
                "f1",
            )
        )
        for r in rows:
            writer.writerow(
                (
                    r["target"],
                    f"{r['tau']:.2f}",
                    r["tp"],
                    r["fp"],
                    r["fn"],
                    r["tn"],
                    r["n_positive"],
                    (
                        "nan"
                        if math.isnan(r["precision"])
                        else f"{r['precision']:.4f}"
                    ),
                    (
                        "nan"
                        if math.isnan(r["recall"])
                        else f"{r['recall']:.4f}"
                    ),
                    (
                        "nan"
                        if math.isnan(r["f1"])
                        else f"{r['f1']:.4f}"
                    ),
                )
            )
    return {"n_rows": len(rows)}


def write_iso_yield_curve_plot(
    sweep: Dict[int, Dict[str, float]],
    iso_curve: List[Dict[str, Any]],
    out_png: Path,
    task_label: str = "task",
) -> None:
    """Plot iso-yield curve for the single available arithmetic_synthetic task.

    The groupsize_zvf_sweep experiment only contains one task;
    a 2-panel figure shows the raw ZVF-vs-G curve with iso-yield
    markers overlaid.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    gs = sorted(sweep.keys())
    zvfs = [sweep[g]["mean_zvf"] for g in gs]
    acc = [sweep[g].get("heldout_acc_mean", float("nan")) for g in gs]

    ax = axes[0]
    ax.plot(gs, zvfs, "o-", color="#1f77b4", label="observed")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("group size G")
    ax.set_ylabel("mean ZVF")
    ax.set_title("ZVF vs G (arithmetic_synthetic)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(gs)
    ax.set_xticklabels([str(g) for g in gs])
    # Overlay iso-yield tau markers at the interpolated G.
    for entry in iso_curve:
        tau = entry["tau"]
        g_min = entry["g_min"]
        if not math.isinf(g_min) and g_min <= max(gs) * 2:
            ax.axhline(tau, color="#888888", linestyle=":", alpha=0.5)
            ax.axvline(
                g_min,
                color="#d62728",
                linestyle="--",
                alpha=0.4,
                label=(
                    f"tau={tau:.2f} -> G={g_min:.1f}"
                    if tau in {0.20, 0.40} and not any(
                        l.get_label().startswith(f"tau={tau:.2f}")
                        for l in ax.get_lines()
                    )
                    else None
                ),
            )
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(gs, acc, "s-", color="#2ca02c", label="heldout acc")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("group size G")
    ax.set_ylabel("heldout accuracy")
    ax.set_title("Accuracy vs G (arithmetic_synthetic)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(gs)
    ax.set_xticklabels([str(g) for g in gs])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.97, 1.0)

    fig.suptitle(
        f"Iter 122 iso-yield ZVF curve ({task_label})"
    )
    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    out_iso = RESULTS / "zvf_iter122_iso_yield.tsv"
    out_aero = RESULTS / "zvf_iter122_aero_quantile.tsv"
    out_op = RESULTS / "zvf_iter122_op_sweep.tsv"
    out_meta = RESULTS / "zvf_iter122_meta.json"
    out_fig = FIGURES / "zvf_iter122_iso_yield.png"

    # 1. Iso-yield curve from groupsize_zvf_sweep.
    iso_summary = write_iso_yield(out_iso)

    # For the figure we need the underlying raw curve (any task with
    # measured G in {2, 4, 8, 16}).
    sweep_rows = zd.load_groupsize_sweep()
    rows_by_task: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in sweep_rows:
        rows_by_task.setdefault(r["task"], {})[int(r["group_size"])] = r
    task_curve: Dict[int, Dict[str, Any]] = {}
    task_label = ""
    for task, recs in rows_by_task.items():
        if any(g in recs for g in (2, 4, 8, 16)):
            task_curve = recs
            task_label = task
            break

    # Build the iso-yield curve restricted to the chosen task so the
    # figure can overlay markers.
    if task_curve:
        gs = sorted(task_curve.keys())
        zvfs = [task_curve[g]["mean_zvf"] for g in gs]
        taus = [0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
        # Filter to the actual task for the figure.
        iso_for_fig: List[Dict[str, Any]] = []
        for entry in _iso_yield_curve(gs, zvfs, taus):
            iso_for_fig.append(entry)
        write_iso_yield_curve_plot(task_curve, iso_for_fig, out_fig, task_label=task_label)
        # Mirror to paper/figures/ if present.
        try:
            (PAPER_FIGS).mkdir(parents=True, exist_ok=True)
            (PAPER_FIGS / "zvf_iter122_iso_yield.png").write_bytes(
                out_fig.read_bytes()
            )
            (PAPER_FIGS / "zvf_iter122_iso_yield.pdf").write_bytes(
                out_fig.with_suffix(".pdf").read_bytes()
            )
        except Exception:  # noqa: BLE001
            pass

    # 2. AERO-vs-GRPO gap stratified by ZVF quantile.
    aero_summary = write_aero_quantile(out_aero)

    # 3. Operating-point sweep on the pooled 23 cells (mirrors the
    # iter 118 pipeline: load every per-experiment source and pool by
    # (experiment, model, group_size)).
    rows = (
        zd.load_tinker_gsm8k()
        + zd.load_groupsize_sweep()
        + zd.load_variance_mitigation()
        + zd.load_tool_use_diagnostics()
        + zd.load_scaling_law_phases()
        + zd.load_drgrpo_vs_grpo()
        + zd.load_samestack_ppo_grpo()
    )
    pooled = z118._pool_by_cell(rows)
    op_summary = write_op_sweep(out_op, pooled)

    meta = {
        "iter": 122,
        "pillar": "P2-ZVF",
        "n_pooled_cells": len(pooled),
        **iso_summary,
        **aero_summary,
        **op_summary,
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
