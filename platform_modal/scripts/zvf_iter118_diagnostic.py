#!/usr/bin/env python3
"""Iter118 -- first-class ZVF diagnostic vs AERO.

Builds three additions on top of platform_modal/scripts/zvf_diagnostic.py:

1. zvf_iter118_auroc.tsv
   AUROC (with 95% bootstrap CI, B=2000) of mean_zvf vs terminal
   failure binary, computed on the pooled (experiment, model,
   group_size) cells.  Two targets: (a) is_collapse (cleanest),
   (b) is_collapse_or_drift (matches platform_modal/scripts/zvf_diagnostic).

2. zvf_iter118_aero_grpo_gap.tsv
   Effect-size gap (AERO - GRPO) on mean_zvf and on
   collapse-rate, with bootstrap CIs over the per-seed cells of
   variance_mitigation (real measurement, n=5 seeds each).

3. zvf_iter118_calibration.tsv
   Predictive calibration: bin pooled rows by their mean_zvf
   decile, then report the empirical fraction of "failure" rows in
   each bin plus Wilson 95% CIs.  This is the "first-class
   diagnostic" the paper can paste into a figure caption.

Output artefacts:

    platform_hybrid/experiments/results/zvf_iter118_auroc.tsv
    platform_hybrid/experiments/results/zvf_iter118_aero_grpo_gap.tsv
    platform_hybrid/experiments/results/zvf_iter118_calibration.tsv
    platform_hybrid/experiments/results/zvf_iter118_narrative.json
    figures/zvf_iter118_calibration.pdf

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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import zvf_diagnostic as zd  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


def _rankdata(vs: Sequence[float]) -> List[float]:
    n = len(vs)
    order = sorted(range(n), key=lambda i: vs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and vs[order[j]] == vs[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _auc_mann_whitney(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Mann-Whitney / Wilcoxon rank-sum AUC = P(X_pos > X_neg)."""
    if not pos or not neg:
        return float("nan")
    combined = list(pos) + list(neg)
    n = len(combined)
    ranks = _rankdata(combined)
    n_pos = len(pos)
    sum_pos = sum(ranks[:n_pos])
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    n_neg = n - n_pos
    return float(u / (n_pos * n_neg))


def _bootstrap_auc(
    pos_pool: Sequence[float],
    neg_pool: Sequence[float],
    B: int = 2000,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    rng = random.Random(seed)
    n_pos = len(pos_pool)
    n_neg = len(neg_pool)
    if n_pos < 2 or n_neg < 2:
        return float("nan"), (float("nan"), float("nan"))
    point = _auc_mann_whitney(pos_pool, neg_pool)
    samples: List[float] = []
    all_pos = list(pos_pool)
    all_neg = list(neg_pool)
    for _ in range(B):
        bs_pos = [rng.choice(all_pos) for _ in range(n_pos)]
        bs_neg = [rng.choice(all_neg) for _ in range(n_neg)]
        v = _auc_mann_whitney(bs_pos, bs_neg)
        if not math.isnan(v):
            samples.append(v)
    if not samples:
        return point, (float("nan"), float("nan"))
    samples.sort()
    lo_idx = max(0, int(0.025 * len(samples)))
    hi_idx = min(len(samples) - 1, int(0.975 * len(samples)) - 1)
    return point, (samples[lo_idx], samples[hi_idx])


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def _pool_by_cell(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reduce per-seed rows to one row per (experiment, model, group_size)."""
    keys: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (
            r["experiment"],
            str(r.get("model", "")),
            int(r.get("group_size", 0) or 0),
        )
        keys.setdefault(key, []).append(r)
    out: List[Dict[str, Any]] = []
    for key, recs in keys.items():
        mz_vals = [
            r["mean_zvf"]
            for r in recs
            if r.get("mean_zvf") is not None
            and not (isinstance(r["mean_zvf"], float) and math.isnan(r["mean_zvf"]))
        ]
        la_vals = [
            r["last10_avg"]
            for r in recs
            if r.get("last10_avg") is not None
            and not (isinstance(r["last10_avg"], float) and math.isnan(r["last10_avg"]))
        ]
        pk_vals = [
            r["peak"]
            for r in recs
            if r.get("peak") is not None
            and not (isinstance(r["peak"], float) and math.isnan(r["peak"]))
        ]
        if not la_vals or not pk_vals:
            continue
        mz = statistics.fmean(mz_vals) if mz_vals else float("nan")
        la = statistics.fmean(la_vals)
        pk = statistics.fmean(pk_vals)
        out.append(
            {
                "experiment": key[0],
                "model": key[1],
                "group_size": key[2],
                "mean_zvf": mz,
                "last10_avg": la,
                "peak": pk,
                "failure": zd.classify({"peak": pk, "last10_avg": la}),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_auroc(pooled: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    """AUROC of mean_zvf vs is_failure, stratified by experiment_kind.

    Two binary targets:
      (a) is_collapse            positive = collapse
      (b) is_collapse_or_drift   positive = collapse or drift
    """
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for r in pooled:
        kind = r["experiment"]
        if kind in ("drgrpo_vs_grpo", "samestack_ppo_grpo"):
            kind = "ppo_vs_grpo_family"
        by_kind.setdefault(kind, []).append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: List[Tuple[str, str, int, int, int, str, str, str]] = []
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF vs failure AUROC (iter118).\n")
        fh.write(
            "# AUROC of mean_zvf vs binary failure targets:\n"
            "#   is_collapse            positive class = collapse only\n"
            "#   is_collapse_or_drift   positive class = collapse or drift\n"
            "# Targets both reference platform_modal/scripts/zvf_diagnostic.py:zd.classify.\n"
            "# Bootstrap CIs: B=2000 percentile resamples over per-cell pooled\n"
            "# rows (NOT per-step rows, which are autocorrelated).\n"
            "# AUROC > 0.5 = higher ZVF is more failure-like.\n"
            "# The pooled 'all' AUROC collapses both bimodal failure modes\n"
            "# (drift at LOW ZVF, collapse at HIGH ZVF) into a single number,\n"
            "# so it sits near 0.5; the per-stratum CIs expose the bimodality.\n"
            "# Source: platform_modal/scripts/zvf_iter118_diagnostic.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "stratum",
                "target",
                "n_cells",
                "n_positive",
                "n_negative",
                "auroc",
                "ci_lo",
                "ci_hi",
            )
        )

        def _auroc_for(group: List[Dict[str, Any]], target: str) -> Tuple[float, Tuple[float, float]]:
            if target == "collapse_or_drift":
                pos_set = ("collapse", "drift")
                neg_set = ("converged", "plateau")
            elif target == "collapse":
                pos_set = ("collapse",)
                neg_set = ("converged", "plateau", "drift")
            else:
                raise ValueError(target)
            pos = [r["mean_zvf"] for r in group if r["failure"] in pos_set and not math.isnan(r["mean_zvf"])]
            neg = [r["mean_zvf"] for r in group if r["failure"] in neg_set and not math.isnan(r["mean_zvf"])]
            return _bootstrap_auc(pos, neg, B=2000, seed=7)

        for kind in sorted(by_kind.keys()):
            group = by_kind[kind]
            if len(group) < 4:
                continue
            for target in ("collapse_or_drift", "collapse"):
                au, ci = _auroc_for(group, target)
                if target == "collapse_or_drift":
                    n_p = sum(1 for r in group if r["failure"] in ("collapse", "drift"))
                    n_n = sum(1 for r in group if r["failure"] in ("converged", "plateau"))
                else:
                    n_p = sum(1 for r in group if r["failure"] == "collapse")
                    n_n = sum(1 for r in group if r["failure"] != "collapse")
                rows_out.append((kind, target, len(group), n_p, n_n, au, ci[0], ci[1]))
                writer.writerow(
                    (
                        kind,
                        target,
                        len(group),
                        n_p,
                        n_n,
                        f"{au:.4f}" if not math.isnan(au) else "NA",
                        f"{ci[0]:.4f}" if not math.isnan(ci[0]) else "NA",
                        f"{ci[1]:.4f}" if not math.isnan(ci[1]) else "NA",
                    )
                )

        for target in ("collapse_or_drift", "collapse"):
            au_all, ci_all = _auroc_for(pooled, target)
            if target == "collapse_or_drift":
                n_p = sum(1 for r in pooled if r["failure"] in ("collapse", "drift"))
                n_n = sum(1 for r in pooled if r["failure"] in ("converged", "plateau"))
            else:
                n_p = sum(1 for r in pooled if r["failure"] == "collapse")
                n_n = sum(1 for r in pooled if r["failure"] != "collapse")
            rows_out.append(("all", target, len(pooled), n_p, n_n, au_all, ci_all[0], ci_all[1]))
            writer.writerow(
                (
                    "all",
                    target,
                    len(pooled),
                    n_p,
                    n_n,
                    f"{au_all:.4f}" if not math.isnan(au_all) else "NA",
                    f"{ci_all[0]:.4f}" if not math.isnan(ci_all[0]) else "NA",
                    f"{ci_all[1]:.4f}" if not math.isnan(ci_all[1]) else "NA",
                )
            )

    au_collapse, ci_collapse = _auroc_for(pooled, "collapse")
    au_drift, ci_drift = _auroc_for(pooled, "collapse_or_drift")
    return {
        "rows": rows_out,
        "all_auroc_collapse": au_collapse,
        "all_ci_collapse": ci_collapse,
        "all_auroc_drift": au_drift,
        "all_ci_drift": ci_drift,
    }


def write_aero_grpo_gap(rows: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    """Effect-size gap (AERO - GRPO) on mean_zvf and on failure-rate."""
    for r in rows:
        if "failure" not in r:
            r["failure"] = zd.classify(r)

    aero = [r for r in rows if r["experiment"] == "variance_mitigation" and str(r["model"]).upper() == "AERO"]
    grpo = [r for r in rows if r["experiment"] == "variance_mitigation" and str(r["model"]).upper() == "GRPO"]
    if not aero or not grpo:
        return {"gap_zvf": float("nan"), "gap_failure": float("nan")}

    aero_zvfs = [r["mean_zvf"] for r in aero]
    grpo_zvfs = [r["mean_zvf"] for r in grpo]
    aero_fail = [1.0 if r["failure"] in ("collapse", "drift") else 0.0 for r in aero]
    grpo_fail = [1.0 if r["failure"] in ("collapse", "drift") else 0.0 for r in grpo]
    aero_last = [r["last10_avg"] for r in aero]
    grpo_last = [r["last10_avg"] for r in grpo]

    gap_zvf = statistics.fmean(aero_zvfs) - statistics.fmean(grpo_zvfs)
    gap_fail = statistics.fmean(aero_fail) - statistics.fmean(grpo_fail)
    gap_last = statistics.fmean(aero_last) - statistics.fmean(grpo_last)

    rng = random.Random(11)
    n = min(len(aero), len(grpo))
    if n >= 2:
        zs_diff: List[float] = []
        fs_diff: List[float] = []
        ls_diff: List[float] = []
        for _ in range(2000):
            idx = [rng.randrange(n) for _ in range(n)]
            z_a = statistics.fmean([aero_zvfs[i] for i in idx])
            z_g = statistics.fmean([grpo_zvfs[i] for i in idx])
            f_a = statistics.fmean([aero_fail[i] for i in idx])
            f_g = statistics.fmean([grpo_fail[i] for i in idx])
            l_a = statistics.fmean([aero_last[i] for i in idx])
            l_g = statistics.fmean([grpo_last[i] for i in idx])
            zs_diff.append(z_a - z_g)
            fs_diff.append(f_a - f_g)
            ls_diff.append(l_a - l_g)
        zs_diff.sort()
        fs_diff.sort()
        ls_diff.sort()
        z_lo = zs_diff[max(0, int(0.025 * len(zs_diff)))]
        z_hi = zs_diff[min(len(zs_diff) - 1, int(0.975 * len(zs_diff)) - 1)]
        f_lo = fs_diff[max(0, int(0.025 * len(fs_diff)))]
        f_hi = fs_diff[min(len(fs_diff) - 1, int(0.975 * len(fs_diff)) - 1)]
        l_lo = ls_diff[max(0, int(0.025 * len(ls_diff)))]
        l_hi = ls_diff[min(len(ls_diff) - 1, int(0.975 * len(ls_diff)) - 1)]
    else:
        z_lo = z_hi = f_lo = f_hi = l_lo = l_hi = float("nan")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 AERO-vs-GRPO effect size (iter118).\n")
        fh.write(
            "# Paired bootstrap (B=2000) over the n=5 per-seed cells of\n"
            "# variance_mitigation (real measurement).  Negative gap_zvf means\n"
            "# AERO has LOWER mean ZVF than GRPO (better contrast).\n"
            "# gap_failure is the gap in fraction-of-(seed)-cells labelled\n"
            "# collapse-or-drift; gap_last10 is the gap in mean last-10 accuracy.\n"
            "# Source: platform_modal/scripts/zvf_iter118_diagnostic.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "metric",
                "aero_mean",
                "grpo_mean",
                "gap",
                "ci_lo",
                "ci_hi",
                "n_aero",
                "n_grpo",
            )
        )
        for name, av, gv, gap, lo, hi in [
            (
                "mean_zvf",
                statistics.fmean(aero_zvfs),
                statistics.fmean(grpo_zvfs),
                gap_zvf,
                z_lo,
                z_hi,
            ),
            (
                "failure_rate",
                statistics.fmean(aero_fail),
                statistics.fmean(grpo_fail),
                gap_fail,
                f_lo,
                f_hi,
            ),
            (
                "last10_avg",
                statistics.fmean(aero_last),
                statistics.fmean(grpo_last),
                gap_last,
                l_lo,
                l_hi,
            ),
        ]:
            writer.writerow(
                (
                    name,
                    f"{av:.4f}",
                    f"{gv:.4f}",
                    f"{gap:+.4f}",
                    f"{lo:+.4f}" if not math.isnan(lo) else "NA",
                    f"{hi:+.4f}" if not math.isnan(hi) else "NA",
                    len(aero),
                    len(grpo),
                )
            )

    return {
        "gap_zvf": gap_zvf,
        "gap_failure": gap_fail,
        "gap_last10": gap_last,
        "ci_zvf": (z_lo, z_hi),
        "ci_failure": (f_lo, f_hi),
        "ci_last10": (l_lo, l_hi),
        "aero_n": len(aero),
        "grpo_n": len(grpo),
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def write_calibration(pooled: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    """Predictive calibration: bin mean_zvf, report failure rate per bin."""
    valid = [r for r in pooled if not math.isnan(r["mean_zvf"])]
    if not valid:
        return {}
    valid.sort(key=lambda r: r["mean_zvf"])
    n = len(valid)
    n_bins = min(8, max(4, n // 3))
    bin_size = max(1, n // n_bins)
    bins: List[List[Dict[str, Any]]] = []
    for i in range(0, n, bin_size):
        bins.append(valid[i : i + bin_size])
    while len(bins) > n_bins:
        bins[n_bins - 2].extend(bins[-1])
        bins = bins[:-1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF predictive calibration (iter118).\n")
        fh.write(
            "# Pooled (experiment, model, G) cells sorted by mean_zvf and\n"
            "# split into n_bins even-width bins.  failure_rate is the\n"
            "# fraction of cells in the bin labelled collapse or drift per\n"
            "# zvf_diagnostic.classify.  wilson_lo / wilson_hi are Wilson\n"
            "# 95% CIs on the bin failure rate.\n"
            "# Source: platform_modal/scripts/zvf_iter118_diagnostic.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "bin",
                "n_cells",
                "zvf_lo",
                "zvf_hi",
                "zvf_mean",
                "n_failure",
                "n_converged_or_plateau",
                "failure_rate",
                "wilson_lo",
                "wilson_hi",
            )
        )
        rates = []
        for i, b in enumerate(bins):
            n_f = sum(1 for r in b if r["failure"] in ("collapse", "drift"))
            n_c = sum(1 for r in b if r["failure"] in ("converged", "plateau"))
            zvfs = [r["mean_zvf"] for r in b]
            lo, hi = _wilson_ci(n_f, len(b))
            rate = n_f / len(b) if b else 0.0
            rates.append(rate)
            writer.writerow(
                (
                    i,
                    len(b),
                    f"{min(zvfs):.4f}",
                    f"{max(zvfs):.4f}",
                    f"{statistics.fmean(zvfs):.4f}",
                    n_f,
                    n_c,
                    f"{rate:.4f}",
                    f"{lo:.4f}",
                    f"{hi:.4f}",
                )
            )

        # Bimodality check: split into LOW (zvf<0.5) and HIGH (zvf>=0.5).
        low = [r for r in valid if r["mean_zvf"] < 0.5]
        high = [r for r in valid if r["mean_zvf"] >= 0.5]
        low_fail = sum(1 for r in low if r["failure"] in ("collapse", "drift")) / max(1, len(low))
        high_fail = sum(1 for r in high if r["failure"] in ("collapse", "drift")) / max(1, len(high))
    return {
        "n_bins": len(bins),
        "rates": rates,
        "low_failure_rate": low_fail,
        "high_failure_rate": high_fail,
    }


def _maybe_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def write_calibration_figure(
    pooled: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[str]:
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid = [r for r in pooled if not math.isnan(r["mean_zvf"])]
    valid.sort(key=lambda r: r["mean_zvf"])
    n = len(valid)
    n_bins = min(8, max(4, n // 3))
    bin_size = max(1, n // n_bins)
    bins = []
    for i in range(0, n, bin_size):
        bins.append(valid[i : i + bin_size])
    while len(bins) > n_bins:
        bins[n_bins - 2].extend(bins[-1])
        bins = bins[:-1]
    zvf_means = [statistics.fmean([r["mean_zvf"] for r in b]) for b in bins]
    fail_rates = [
        sum(1 for r in b if r["failure"] in ("collapse", "drift")) / len(b) for b in bins
    ]
    cis = [
        _wilson_ci(sum(1 for r in b if r["failure"] in ("collapse", "drift")), len(b))
        for b in bins
    ]
    # Identify the high-zvf tool-use cluster and the low-zvf drift cluster.
    tool_use = [r for r in valid if r["experiment"].startswith("cross_tool")]
    drift = [
        r
        for r in valid
        if r["failure"] == "drift"
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(
        zvf_means,
        fail_rates,
        yerr=[
            [r - lo for r, (lo, _) in zip(fail_rates, cis)],
            [hi - r for r, (_, hi) in zip(fail_rates, cis)],
        ],
        fmt="o-",
        color="#c0392b",
        ecolor="#c0392b",
        alpha=0.85,
        capsize=4,
        markersize=8,
        linewidth=1.5,
        label="Empirical failure rate (Wilson 95%)",
    )
    # Scatter all underlying cells.
    label_added = {"tool_use": False, "drift": False, "plateau": False, "converged": False}
    for r in valid:
        if r["experiment"].startswith("cross_tool"):
            color = "#c0392b"
            tag = "tool_use collapse"
        elif r["failure"] == "drift":
            color = "#e67e22"
            tag = "variance_mitigation drift"
        elif r["failure"] == "plateau":
            color = "#7f8c8d"
            tag = "variance_mitigation plateau"
        else:
            color = "#27ae60"
            tag = "converged"
        show = not label_added.get(tag.split()[0], False)
        ax.scatter(
            r["mean_zvf"],
            1.0 if r["failure"] in ("collapse", "drift") else 0.0,
            s=40,
            color=color,
            alpha=0.6,
            edgecolor="white",
            linewidth=0.4,
            label=tag if show else None,
        )
        label_added[tag.split()[0]] = True
    ax.axhline(0.5, linestyle=":", color="#34495e", alpha=0.4)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.10, 1.15)
    ax.set_xlabel("Mean ZVF across training trajectory (n=%d cells)" % n)
    ax.set_ylabel("Fraction of cells labelled collapse or drift")
    ax.set_title(
        "ZVF as first-class diagnostic: bimodal failure mode\n"
        "(low-ZVF drift in variance_mitigation; high-ZVF collapse in tool-use / scaling_law)"
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def write_narrative(
    pooled: List[Dict[str, Any]],
    auroc: Dict[str, Any],
    gap: Dict[str, Any],
    calib: Dict[str, Any],
    out_path: Path,
) -> None:
    """One-stop JSON summary for downstream paper-section writers."""
    summary = {
        "n_pooled_cells": len(pooled),
        "auroc_collapse": {
            "point": auroc.get("all_auroc_collapse"),
            "ci_lo": auroc.get("all_ci_collapse", (None, None))[0],
            "ci_hi": auroc.get("all_ci_collapse", (None, None))[1],
        },
        "auroc_drift": {
            "point": auroc.get("all_auroc_drift"),
            "ci_lo": auroc.get("all_ci_drift", (None, None))[0],
            "ci_hi": auroc.get("all_ci_drift", (None, None))[1],
        },
        "aero_grpo_gap": {
            "zvf_gap": gap.get("gap_zvf"),
            "zvf_ci_lo": gap.get("ci_zvf", (None, None))[0],
            "zvf_ci_hi": gap.get("ci_zvf", (None, None))[1],
            "last10_gap": gap.get("gap_last10"),
            "last10_ci_lo": gap.get("ci_last10", (None, None))[0],
            "last10_ci_hi": gap.get("ci_last10", (None, None))[1],
        },
        "calibration": {
            "n_bins": calib.get("n_bins"),
            "low_failure_rate": calib.get("low_failure_rate"),
            "high_failure_rate": calib.get("high_failure_rate"),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    rows: List[Dict[str, Any]] = []
    rows += zd.load_tinker_gsm8k()
    rows += zd.load_groupsize_sweep()
    rows += zd.load_variance_mitigation()
    rows += zd.load_tool_use_diagnostics()
    rows += zd.load_scaling_law_phases()
    rows += zd.load_drgrpo_vs_grpo()
    if (RESULTS / "samestack_ppo_grpo.json").exists():
        rows += zd.load_samestack_ppo_grpo()

    pooled = _pool_by_cell(rows)
    print(f"[zvf-iter118] pooled {len(pooled)} (experiment, model, G) cells")

    out_auroc = RESULTS / "zvf_iter118_auroc.tsv"
    out_gap = RESULTS / "zvf_iter118_aero_grpo_gap.tsv"
    out_calib = RESULTS / "zvf_iter118_calibration.tsv"
    out_fig = FIGURES / "zvf_iter118_calibration.pdf"
    out_narr = RESULTS / "zvf_iter118_narrative.json"

    auroc = write_auroc(pooled, out_auroc)
    gap = write_aero_grpo_gap(rows, out_gap)
    calib = write_calibration(pooled, out_calib)
    fig_path = write_calibration_figure(pooled, out_fig)
    write_narrative(pooled, auroc, gap, calib, out_narr)

    print(
        f"[zvf-iter118] all-stratum AUROC(is_collapse)            = "
        f"{auroc['all_auroc_collapse']:.3f} "
        f"[{auroc['all_ci_collapse'][0]:.3f}, {auroc['all_ci_collapse'][1]:.3f}]"
    )
    print(
        f"[zvf-iter118] all-stratum AUROC(is_collapse_or_drift)   = "
        f"{auroc['all_auroc_drift']:.3f} "
        f"[{auroc['all_ci_drift'][0]:.3f}, {auroc['all_ci_drift'][1]:.3f}]"
    )
    print(
        f"[zvf-iter118] AERO - GRPO gap on mean_zvf               = "
        f"{gap['gap_zvf']:+.3f} "
        f"[{gap['ci_zvf'][0]:+.3f}, {gap['ci_zvf'][1]:+.3f}]"
    )
    print(
        f"[zvf-iter118] AERO - GRPO gap on last10_avg             = "
        f"{gap['gap_last10']:+.3f} "
        f"[{gap['ci_last10'][0]:+.3f}, {gap['ci_last10'][1]:+.3f}]"
    )
    print(
        f"[zvf-iter118] calibration: {calib.get('n_bins', '?')} bins, "
        f"low-ZVF failure_rate={calib.get('low_failure_rate', float('nan')):.2f}, "
        f"high-ZVF failure_rate={calib.get('high_failure_rate', float('nan')):.2f}"
    )
    if fig_path:
        print(f"[zvf-iter118] wrote figure {fig_path}")
    print(
        f"[zvf-iter118] wrote {out_auroc.name}, {out_gap.name}, "
        f"{out_calib.name}, {out_narr.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())