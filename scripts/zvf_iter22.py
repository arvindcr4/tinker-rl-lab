#!/usr/bin/env python3
"""Iter 22 Pillar 2 ZVF elevation.

Two new empirical questions that the existing zvf_diagnostic.py does not
answer:

  (A) Per-library iterated bootstrap CIs on mean_ZVF and collapse_rate.
      zvf_diagnostic.py only prints point estimates. For NeurIPS we need
      95% CIs around each library's mean ZVF and collapse_rate so the
      AERO vs GRPO contrast is properly bracketed. We do paired bootstrap
      across seeds (n=5 within library).

  (B) ZVF lead-time: does the ZVF time-series PREDICT the collapse step?
      Existing zvf_dynamics_leadtime.tsv only covers grpo. We extend the
      lead-time analysis to every variance_mitigation library that has a
      per-step collapse flag in variance_mitigation.tsv, and add a
      "ZVF spike precedes collapse by k steps?" statistical test (signed
      rank, paired per (method, seed)).

Inputs (real, measured):
    experiments/results/variance_mitigation.tsv        -- per-step zvf, collapse
    experiments/results/groupsize_zvf_sweep.tsv        -- G-sweep point
    experiments/results/zvf_summary.tsv                -- pooled rows
    experiments/results/zvf_by_library.tsv             -- per-library pooled

Outputs (real, computed this session):
    experiments/results/zvf_library_bootstrap_ci.tsv   -- (A)
    experiments/results/zvf_leadtime_all.tsv           -- (B) extended table
    experiments/results/zvf_leadtime_summary.tsv       -- (B) aggregated
    figures/zvf_leadtime.pdf                           -- (B) figure
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


# ---------------------------------------------------------------------------
# (A) Per-library iterated bootstrap CIs on mean_ZVF and collapse_rate
# ---------------------------------------------------------------------------


def load_variance_mitigation_per_seed() -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Per-step rows grouped by (method, seed)."""
    path = RESULTS / "variance_mitigation.tsv"
    out: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            method = r["method"]
            seed = int(r["seed"])
            out.setdefault(method, {}).setdefault(seed, []).append(
                {
                    "step": int(r["step"]),
                    "zvf": float(r["zvf"]),
                    "heldout_acc": float(r["heldout_acc"]),
                    "reward_mean": float(r["reward_mean"]),
                    "collapse": int(r["collapse"]),
                }
            )
    return out


def per_seed_mean_zvf(method: str, seed: int, rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return statistics.fmean(r["zvf"] for r in rows)


def per_seed_collapse_rate(method: str, seed: int, rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    return statistics.fmean(r["collapse"] for r in rows)


def bootstrap_mean_ci(
    samples: Sequence[float], B: int = 2000, seed: int = 7
) -> Tuple[float, Tuple[float, float]]:
    """Percentile bootstrap CI on the mean."""
    samples = [float(s) for s in samples if s is not None and not (isinstance(s, float) and math.isnan(s))]
    n = len(samples)
    if n < 2:
        return (float("nan"), (float("nan"), float("nan")))
    rng = random.Random(seed)
    point = statistics.fmean(samples)
    boot = []
    for _ in range(B):
        rs = [samples[rng.randrange(n)] for _ in range(n)]
        boot.append(statistics.fmean(rs))
    boot.sort()
    lo = boot[int(0.025 * len(boot))]
    hi = boot[int(0.975 * len(boot)) - 1]
    return (point, (lo, hi))


def compute_library_bootstrap_ci(
    per_method: Dict[str, Dict[int, List[Dict[str, Any]]]], out_path: Path
) -> None:
    """For each method, compute mean_ZVF CI and collapse_rate CI from seeds."""
    rows: List[Dict[str, Any]] = []
    for method, by_seed in per_method.items():
        mz_by_seed = []
        cr_by_seed = []
        for seed, rows_ in by_seed.items():
            mz = per_seed_mean_zvf(method, seed, rows_)
            cr = per_seed_collapse_rate(method, seed, rows_)
            if not math.isnan(mz):
                mz_by_seed.append(mz)
            if not math.isnan(cr):
                cr_by_seed.append(cr)
        mz_pt, (mz_lo, mz_hi) = bootstrap_mean_ci(mz_by_seed, B=2000, seed=11)
        cr_pt, (cr_lo, cr_hi) = bootstrap_mean_ci(cr_by_seed, B=2000, seed=22)
        rows.append(
            {
                "library": method,
                "n_seeds": len(mz_by_seed),
                "mean_zvf": mz_pt,
                "mean_zvf_ci_lo": mz_lo,
                "mean_zvf_ci_hi": mz_hi,
                "collapse_rate": cr_pt,
                "collapse_rate_ci_lo": cr_lo,
                "collapse_rate_ci_hi": cr_hi,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 per-library iterated bootstrap CIs (iter 22)\n")
        fh.write(
            "# For each variance_mitigation method, B=2000 percentile bootstrap\n"
            "# CIs on mean_ZVF and on collapse_rate, with seed-level resampling.\n"
            "# Source: scripts/zvf_iter22.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "library",
                "n_seeds",
                "mean_zvf",
                "mean_zvf_ci_lo",
                "mean_zvf_ci_hi",
                "collapse_rate",
                "collapse_rate_ci_lo",
                "collapse_rate_ci_hi",
            )
        )
        for r in rows:
            writer.writerow(
                (
                    r["library"],
                    r["n_seeds"],
                    f"{r['mean_zvf']:.4f}",
                    f"{r['mean_zvf_ci_lo']:.4f}",
                    f"{r['mean_zvf_ci_hi']:.4f}",
                    f"{r['collapse_rate']:.4f}",
                    f"{r['collapse_rate_ci_lo']:.4f}",
                    f"{r['collapse_rate_ci_hi']:.4f}",
                )
            )


# ---------------------------------------------------------------------------
# (B) ZVF lead-time extended to every variance_mitigation method
# ---------------------------------------------------------------------------


def find_first_collapse(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, float]:
    """Return (peak_pass_step, first_collapse_step, lead_steps, mean_zvf_at_pass).

    "peak_pass_step" = argmax(heldout_acc) over the entire trajectory. This
    is the step where the policy is most "alive" -- not a fixed 0.5
    threshold (the variance_mitigation study peaks well below 0.5).

    "first_collapse_step" = first step AFTER the peak where collapse=1.
    A trajectory whose peak occurs after its first collapse gets lead=-1
    and is dropped downstream.

    "zvf_at_pass" = ZVF at the peak step.
    """
    if not rows:
        return (-1, -1, -1, float("nan"))
    peak_row = max(rows, key=lambda r: r["heldout_acc"])
    pass_step = peak_row["step"]
    zvf_at_pass = peak_row["zvf"]
    peak_val = peak_row["heldout_acc"]
    collapse_step = -1
    for r in rows:
        if r["step"] <= pass_step:
            continue
        if r["collapse"] == 1:
            collapse_step = r["step"]
            break
    if collapse_step < 0:
        # No collapse after peak; could be a non-collapsing run.
        return (pass_step, -1, -1, zvf_at_pass)
    return (pass_step, collapse_step, collapse_step - pass_step, zvf_at_pass)


def find_max_zvf_in_window(
    rows: List[Dict[str, Any]], t_start: int, t_end: int
) -> Tuple[int, float]:
    """Return (argmax step, max zvf) in [t_start, t_end]. Returns NaN if empty."""
    best_step = -1
    best_val: float = float("nan")
    found = False
    for r in rows:
        if t_start <= r["step"] <= t_end:
            if not found or r["zvf"] > best_val:
                best_val = r["zvf"]
                best_step = r["step"]
                found = True
    if not found:
        return (-1, float("nan"))
    return (best_step, best_val)


def compute_leadtime(
    per_method: Dict[str, Dict[int, List[Dict[str, Any]]]],
    lead_window: int = 30,
) -> List[Dict[str, Any]]:
    """For each (method, seed), compute collapse lead-time and ZVF-at-pass.

    Also computes the local ZVF max in the [pass, pass+lead_window] window,
    so we can test whether ZVF spikes PRECEDE the collapse step.

    If a (method, seed) never sets collapse=1, we still emit a row with
    peak_step = argmax(heldout_acc) but lead_steps = -1, so downstream
    aggregation can count non-collapsing runs explicitly.
    """
    out: List[Dict[str, Any]] = []
    for method, by_seed in per_method.items():
        for seed, rows in by_seed.items():
            pass_step, collapse_step, lead_steps, zvf_at_pass = find_first_collapse(rows)
            if pass_step < 0:
                continue
            local_max_step, local_max_zvf = find_max_zvf_in_window(
                rows, pass_step, min(pass_step + lead_window, (rows[-1]["step"] if rows else pass_step) + 1)
            )
            # If we have a real collapse, the "5-step before collapse" window
            # is well-defined. If not, fall back to the last 5 steps of
            # trajectory as a stand-in for terminal-ZVF.
            if lead_steps > 0:
                pre_window_end = pass_step + min(5, lead_steps)
                pre_zvf_window = [r["zvf"] for r in rows if pass_step <= r["step"] < pre_window_end]
                post_window_start = max(pass_step, collapse_step - 5)
                post_zvf_window = [r["zvf"] for r in rows if post_window_start < r["step"] <= collapse_step]
            else:
                pre_zvf_window = [r["zvf"] for r in rows if pass_step <= r["step"] < pass_step + 5]
                post_zvf_window = [r["zvf"] for r in rows if rows and r["step"] >= rows[-1]["step"] - 5]
            pre_zvf = statistics.fmean(pre_zvf_window) if pre_zvf_window else float("nan")
            post_zvf = statistics.fmean(post_zvf_window) if post_zvf_window else float("nan")
            out.append(
                {
                    "method": method,
                    "seed": seed,
                    "n_steps": len(rows),
                    "first_pass_step": pass_step,
                    "first_collapse_step": collapse_step,
                    "lead_steps": lead_steps,
                    "zvf_at_pass": zvf_at_pass,
                    "pre_zvf_5": pre_zvf,
                    "post_zvf_5": post_zvf,
                    "local_max_zvf": local_max_zvf,
                    "local_max_step": local_max_step,
                    "collapsed": 1 if collapse_step >= 0 else 0,
                }
            )
    return out


def write_leadtime_table(
    leadtime_rows: List[Dict[str, Any]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF lead-time extended table (iter 22)\n")
        fh.write(
            "# For every (method, seed) we compute (peak_step, first_collapse_step,\n"
            "# lead_steps) plus the mean ZVF in the 5-step window AFTER the peak.\n"
            "# lead_steps = -1 means the run never set collapse=1 (i.e. did NOT\n"
            "# collapse -- a positive outcome for mitigation libraries).\n"
            "# Source: scripts/zvf_iter22.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "method",
                "seed",
                "n_steps",
                "first_pass_step",
                "first_collapse_step",
                "lead_steps",
                "collapsed",
                "zvf_at_pass",
                "pre_zvf_5",
                "post_zvf_5",
                "local_max_zvf",
                "local_max_step",
            )
        )
        for r in leadtime_rows:
            writer.writerow(
                (
                    r["method"],
                    r["seed"],
                    r["n_steps"],
                    r["first_pass_step"],
                    r["first_collapse_step"],
                    r["lead_steps"],
                    r["collapsed"],
                    f"{r['zvf_at_pass']:.4f}",
                    f"{r['pre_zvf_5']:.4f}" if not math.isnan(r["pre_zvf_5"]) else "NA",
                    f"{r['post_zvf_5']:.4f}" if not math.isnan(r["post_zvf_5"]) else "NA",
                    f"{r['local_max_zvf']:.4f}",
                    r["local_max_step"],
                )
            )


def aggregate_leadtime(leadtime_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in leadtime_rows:
        by_method.setdefault(r["method"], []).append(r)
    summary: List[Dict[str, Any]] = []
    for method, recs in by_method.items():
        n_runs = len(recs)
        n_collapsed = sum(1 for r in recs if r["collapsed"] == 1)
        leads = [r["lead_steps"] for r in recs if r["lead_steps"] >= 0]
        zvf_at_pass = [r["zvf_at_pass"] for r in recs if not math.isnan(r["zvf_at_pass"])]
        pre = [r["pre_zvf_5"] for r in recs if not math.isnan(r["pre_zvf_5"])]
        post = [r["post_zvf_5"] for r in recs if not math.isnan(r["post_zvf_5"])]
        lmax = [r["local_max_zvf"] for r in recs if not math.isnan(r["local_max_zvf"])]
        summary.append(
            {
                "method": method,
                "n_runs": n_runs,
                "n_collapsed": n_collapsed,
                "collapse_rate": n_collapsed / n_runs if n_runs else float("nan"),
                "mean_lead": statistics.fmean(leads) if leads else float("nan"),
                "median_lead": statistics.median(leads) if leads else float("nan"),
                "min_lead": min(leads) if leads else float("nan"),
                "max_lead": max(leads) if leads else float("nan"),
                "mean_zvf_at_pass": statistics.fmean(zvf_at_pass) if zvf_at_pass else float("nan"),
                "mean_pre_zvf_5": statistics.fmean(pre) if pre else float("nan"),
                "mean_post_zvf_5": statistics.fmean(post) if post else float("nan"),
                "mean_local_max_zvf": statistics.fmean(lmax) if lmax else float("nan"),
            }
        )
    return summary


def write_leadtime_summary(
    summary: List[Dict[str, Any]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF lead-time aggregated by method (iter 22)\n")
        fh.write(
            "# n_runs = total (method, seed) pairs (5 seeds per method by default).\n"
            "# n_collapsed = number of those that set collapse=1 at any step.\n"
            "# collapse_rate = n_collapsed / n_runs (mitigation libraries should be\n"
            "# near 0; non-mitigation grpo was 3/5 = 0.6 in this dataset).\n"
            "# mean_pre_zvf_5 vs mean_post_zvf_5: 5-step ZVF average\n"
            "# immediately AFTER peak vs immediately BEFORE collapse (or terminal).\n"
            "# Source: scripts/zvf_iter22.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "method",
                "n_runs",
                "n_collapsed",
                "collapse_rate",
                "mean_lead",
                "median_lead",
                "min_lead",
                "max_lead",
                "mean_zvf_at_pass",
                "mean_pre_zvf_5",
                "mean_post_zvf_5",
                "mean_local_max_zvf",
            )
        )
        for r in summary:
            writer.writerow(
                (
                    r["method"],
                    r["n_runs"],
                    r["n_collapsed"],
                    f"{r['collapse_rate']:.4f}" if not math.isnan(r["collapse_rate"]) else "NA",
                    f"{r['mean_lead']:.2f}" if not math.isnan(r["mean_lead"]) else "NA",
                    f"{r['median_lead']:.1f}" if not math.isnan(r["median_lead"]) else "NA",
                    f"{r['min_lead']}" if not math.isnan(r["min_lead"]) else "NA",
                    f"{r['max_lead']}" if not math.isnan(r["max_lead"]) else "NA",
                    f"{r['mean_zvf_at_pass']:.4f}" if not math.isnan(r["mean_zvf_at_pass"]) else "NA",
                    f"{r['mean_pre_zvf_5']:.4f}" if not math.isnan(r["mean_pre_zvf_5"]) else "NA",
                    f"{r['mean_post_zvf_5']:.4f}" if not math.isnan(r["mean_post_zvf_5"]) else "NA",
                    f"{r['mean_local_max_zvf']:.4f}" if not math.isnan(r["mean_local_max_zvf"]) else "NA",
                )
            )


# ---------------------------------------------------------------------------
# (B') Statistical test: does pre-collapse ZVF exceed pre-pass ZVF?
# ---------------------------------------------------------------------------


def wilcoxon_signed_rank(deltas: Sequence[float]) -> Tuple[float, float]:
    """Returns (W+, p_two_sided) by a normal approximation.

    For n < 10 we just return the sum of positive ranks and a Gaussian
    approximation p-value (which is fine for our n in [3, 5]).
    """
    deltas = [d for d in deltas if not math.isnan(d)]
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"))
    abs_d = sorted(((abs(d), 1 if d > 0 else -1) for d in deltas), key=lambda x: x[0])
    # Assign average ranks for ties.
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and abs_d[j][0] == abs_d[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    w_plus = sum(ranks[k] for k in range(n) if abs_d[k][1] > 0)
    w_minus = sum(ranks[k] for k in range(n) if abs_d[k][1] < 0)
    # Normal approximation (no ties correction needed at n<=5).
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return (w_plus, float("nan"))
    z = (w_plus - mu) / sigma
    # Two-sided p-value (no continuity correction at n<=5).
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))
    return (w_plus, p)


def pre_post_test(
    leadtime_rows: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """For each method, test paired (pre_zvf_5, post_zvf_5) deltas.

    A positive mean_delta means ZVF RISES in the 5-step window ending
    at the collapse (or terminal) step -- the precursor signal.
    """
    out: Dict[str, Dict[str, float]] = {}
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in leadtime_rows:
        by_method.setdefault(r["method"], []).append(r)
    for method, recs in by_method.items():
        deltas = [r["post_zvf_5"] - r["pre_zvf_5"] for r in recs if not math.isnan(r["pre_zvf_5"]) and not math.isnan(r["post_zvf_5"])]
        w_plus, p_two = wilcoxon_signed_rank(deltas)
        out[method] = {
            "n": len(recs),
            "n_with_data": len(deltas),
            "mean_pre_zvf_5": statistics.fmean([r["pre_zvf_5"] for r in recs]) if recs else float("nan"),
            "mean_post_zvf_5": statistics.fmean([r["post_zvf_5"] for r in recs]) if recs else float("nan"),
            "mean_delta": statistics.fmean(deltas) if deltas else float("nan"),
            "W_plus": w_plus,
            "p_two_sided": p_two,
        }
    return out


def write_pre_post_test(
    test: Dict[str, Dict[str, float]], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 pre/post-collapse ZVF signed-rank test (iter 22)\n")
        fh.write(
            "# For each method, paired Wilcoxon signed-rank test on\n"
            "# (post_zvf_5 - pre_zvf_5). Positive mean_delta = ZVF rises\n"
            "# in the 5-step window immediately BEFORE collapse (or terminal).\n"
            "# Source: scripts/zvf_iter22.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "method",
                "n",
                "n_with_data",
                "mean_pre_zvf_5",
                "mean_post_zvf_5",
                "mean_delta",
                "W_plus",
                "p_two_sided",
            )
        )
        for method, t in test.items():
            writer.writerow(
                (
                    method,
                    t["n"],
                    t["n_with_data"],
                    f"{t['mean_pre_zvf_5']:.4f}",
                    f"{t['mean_post_zvf_5']:.4f}",
                    f"{t['mean_delta']:.4f}",
                    f"{t['W_plus']:.2f}" if not math.isnan(t["W_plus"]) else "NA",
                    f"{t['p_two_sided']:.4f}" if not math.isnan(t["p_two_sided"]) else "NA",
                )
            )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _maybe_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


METHOD_COLOR = {
    "grpo": "#2c3e50",
    "aero": "#8e44ad",
    "cppo": "#16a085",
    "ngrpo": "#d35400",
    "scafgrpo": "#7f8c8d",
    "mcgrpo": "#c0392b",
    "gift": "#2980b9",
    "areal": "#f39c12",
    "es": "#27ae60",
}


def write_leadtime_figure(
    summary: List[Dict[str, Any]],
    test: Dict[str, Dict[str, float]],
    out_path: Path,
) -> Optional[str]:
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    methods = [r["method"] for r in summary]
    mean_lead = [r["mean_lead"] for r in summary]
    mean_pre = [r["mean_pre_zvf_5"] for r in summary]
    mean_post = [r["mean_post_zvf_5"] for r in summary]
    pvals = [test.get(m, {}).get("p_two_sided", float("nan")) for m in methods]

    colors = [METHOD_COLOR.get(m, "#34495e") for m in methods]
    bars = axL.bar(methods, mean_lead, color=colors, edgecolor="white", linewidth=0.6)
    axL.set_ylabel("Mean lead-steps (collapse_step - pass_step)")
    axL.set_title("ZVF lead-time: collapse lag per library\n(only runs that collapsed)")
    axL.tick_params(axis="x", rotation=45, labelsize=8)
    for bar, val in zip(bars, mean_lead):
        if math.isnan(val):
            continue
        axL.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    import numpy as np  # local; figure-only dependency

    x = np.arange(len(methods))
    width = 0.38
    axR.bar(x - width / 2.0, mean_pre, width, color="#bdc3c7", edgecolor="white", label="pre (5-step)")
    axR.bar(x + width / 2.0, mean_post, width, color="#c0392b", edgecolor="white", label="pre-collapse (5-step)")
    axR.set_xticks(x)
    axR.set_xticklabels(methods, rotation=45, fontsize=8)
    axR.set_ylabel("Mean ZVF in 5-step window")
    axR.set_title("Pre vs pre-collapse ZVF (paired)\nW+ test reported in zvf_leadtime_summary.tsv")
    axR.set_ylim(0, 1.05)
    axR.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle("ZVF as a collapse precursor (iter 22)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def write_bootstrap_figure(
    bootstrap_rows: List[Dict[str, Any]], out_path: Path
) -> Optional[str]:
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    methods = [r["library"] for r in bootstrap_rows]
    means = [r["mean_zvf"] for r in bootstrap_rows]
    los = [r["mean_zvf_ci_lo"] for r in bootstrap_rows]
    his = [r["mean_zvf_ci_hi"] for r in bootstrap_rows]
    x = list(range(len(methods)))
    colors = [METHOD_COLOR.get(m, "#34495e") for m in methods]
    ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.6)
    err_lo = [m - lo for m, lo in zip(means, los)]
    err_hi = [hi - m for m, hi in zip(means, his)]
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=3, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, fontsize=9)
    ax.set_ylabel("Mean ZVF across training trajectory")
    ax.set_title("Mean ZVF per library with 95% bootstrap CIs (B=2000, seed-resampled)")
    ax.set_ylim(0, max(0.6, max(his) * 1.10) if his else 0.6)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    per_method = load_variance_mitigation_per_seed()

    # (A) Bootstrap CIs
    boot_path = RESULTS / "zvf_library_bootstrap_ci.tsv"
    compute_library_bootstrap_ci(per_method, boot_path)
    bootstrap_rows: List[Dict[str, Any]] = []
    with boot_path.open() as fh:
        lines = [l for l in fh.readlines() if not l.startswith("#")]
        reader = csv.DictReader(lines, delimiter="\t")
        for r in reader:
            bootstrap_rows.append(
                {
                    "library": r["library"],
                    "mean_zvf": float(r["mean_zvf"]),
                    "mean_zvf_ci_lo": float(r["mean_zvf_ci_lo"]),
                    "mean_zvf_ci_hi": float(r["mean_zvf_ci_hi"]),
                }
            )

    # (B) Leadtime
    leadtime_rows = compute_leadtime(per_method, lead_window=30)
    write_leadtime_table(leadtime_rows, RESULTS / "zvf_leadtime_all.tsv")
    summary = aggregate_leadtime(leadtime_rows)
    write_leadtime_summary(summary, RESULTS / "zvf_leadtime_summary.tsv")
    test = pre_post_test(leadtime_rows)
    write_pre_post_test(test, RESULTS / "zvf_pre_post_test.tsv")

    # Figures
    fig_leadtime = write_leadtime_figure(
        summary, test, FIGURES / "zvf_leadtime.pdf"
    )
    fig_boot = write_bootstrap_figure(bootstrap_rows, FIGURES / "zvf_library_bootstrap.pdf")

    # Console summary
    print(f"[zvf-iter22] wrote {len(bootstrap_rows)} library bootstrap CIs -> {boot_path}")
    for r in bootstrap_rows:
        print(
            f"  {r['library']:>10}  mean_zvf={r['mean_zvf']:.3f}  "
            f"CI=[{r['mean_zvf_ci_lo']:.3f},{r['mean_zvf_ci_hi']:.3f}]"
        )
    print()
    print(f"[zvf-iter22] wrote {len(leadtime_rows)} (method, seed) leadtime rows")
    for r in summary:
        p = test.get(r["method"], {}).get("p_two_sided", float("nan"))
        p_str = f"{p:.3f}" if not math.isnan(p) else "NA"
        print(
            f"  {r['method']:>10}  n={r['n_runs']:>2}  "
            f"mean_lead={r['mean_lead']:>6.2f}  "
            f"pre={r['mean_pre_zvf_5']:.3f}  post={r['mean_post_zvf_5']:.3f}  "
            f"p_two={p_str}"
        )
    if fig_leadtime:
        print(f"[zvf-iter22] wrote figure {fig_leadtime}")
    if fig_boot:
        print(f"[zvf-iter22] wrote figure {fig_boot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())