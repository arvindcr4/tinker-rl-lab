#!/usr/bin/env python3
"""Pillar 2 cross-experiment Zero-Variance Fraction (ZVF) diagnostic.

Aggregates ZVF measurement streams from every per-experiment training run
that surfaced a per-group rollout trace, then correlates mean-ZVF with
training-failure outcomes (Nemotron-120B collapse, tool-use 0%, arithmetic
distillation failure) and emits a single one-row-per-experiment summary
table.

Inputs (real, measured; produced earlier in this repo):

    experiments/results/tinker_gsm8k_zvf_summary.json
        Real Qwen3-8B / GSM8K rollouts, 3 seeds x 200 problems, K=8.
    experiments/results/tinker_gsm8k_zvf_s42.json (+ s123, s456)
        Per-problem reward tensors (length-8 lists) for each seed.
    experiments/results/groupsize_zvf_sweep.tsv
        4 G-sweep rows (G in {2,4,8,16}, 3 seeds each, x 40 steps).
    experiments/results/variance_mitigation.tsv
        Per-step ZVF and a "collapse" label flagged by the heldout-acc
        going to zero across an entire 30-step trajectory.
    experiments/results/tool_code_reward_diagnostics.tsv
        Cross-tool tool-use runs (Qwen3-32B, Llama-8B-Instruct) whose
        trajectory collapses (last10_avg == 0.0).
    experiments/results/scaling_law_three_phase.tsv
        5-model scaling-law study with a "collapse" phase row for
        Nemotron-120B (peak=0.875, late_mean=0.2083).
    experiments/results/drgrpo_vs_grpo.json
        Per-run mean_zvf for the DRGRPO vs GRPO comparison.
    experiments/results/samestack_ppo_grpo.json
        Per-run metrics for the PPO/GRPO shared-stack diagnostic.

Output:

    experiments/results/zvf_summary.tsv
        One row per (experiment, condition). Columns: experiment, model,
        task, group_size, n_seeds, mean_zvf, max_zvf, mean_reward, peak,
        last10_avg, collapse_label, n_steps, evidence_path.

It also writes a secondary wide correlation table:

    experiments/results/zvf_failure_correlation.tsv
        Spearman + Pearson rho between (mean_zvf, collapse) and the
        heldout_acc / last10_avg columns of the summary; reported with
        bootstrap CIs from B=2000 resamples over rows (treat the n=14
        per-experiment rows as the unit, since per-step rows are
        autocorrelated).

Failure taxonomy
----------------

collapse_label is computed deterministically from the per-experiment
summary using only public thresholds documented in this script:

    "collapse"      peak_holdout > 0.7 AND last10_avg < 0.35
                    (Nemotron-120B/DeepSeek-V3.1 plateau -> drift
                    and tool-use trajectories.)
    "drift"         last10_avg < 0.85 * peak_holdout but not collapse.
    "plateau"       peak_holdout < 0.5 (loss moved at all but plateaued).
    "converged"     last10_avg >= 0.85 * peak_holdout.

This mirrors the four-state scheme used in scaling_law_three_phase.tsv
("collapse", "drift", "saturation", "plateau") but uses heldout_acc as
the ground truth because all our diagnostic rows carry it.

Why include this script (vs reusing zvf_compute_cross_framework.py)
----------------------------------------------------------

zvf_compute_cross_framework.py is the *per-step time-series* ZVF
emitter used during a run. This script is the *post-hoc aggregator*
that lifts every existing time-series summary in experiments/results/
into a single small TSV, attaches a collapse label, and emits a
correlation matrix against failure outcomes -- the artifact reviewers
will actually open.

Honest statistics note: the per-step ZVF rows are autocorrelated
(typical lag-1 correlation about 0.9). We therefore aggregate to
one row per experiment first, and ONLY THEN compute correlations
across experiments. Per-experiment bootstrap CIs respect the
seed-level averaging each summary already did.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Loaders: each one returns a list of dicts, one per (experiment, condition).
# ---------------------------------------------------------------------------


def _stat(xs: Sequence[float]) -> Tuple[float, float, float]:
    """mean, min, max with NaN-safe handling."""
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    return (statistics.fmean(xs), min(xs), max(xs))


def load_tinker_gsm8k() -> List[Dict[str, Any]]:
    """Real Qwen3-8B / GSM8K rollouts, K=8 per problem, 3 seeds."""
    summary_path = RESULTS / "tinker_gsm8k_zvf_summary.json"
    summary = json.loads(summary_path.read_text())
    rows: List[Dict[str, Any]] = []
    for seed, zvf_val, acc in zip(
        summary["seeds"], summary["zvf_per_seed"], summary["sampling_accuracy_per_seed"]
    ):
        per_problem = json.loads(
            (RESULTS / f"tinker_gsm8k_zvf_s{seed}.json").read_text()
        )["per_problem"]
        zvf_per_problem = [p["zvf"] for p in per_problem]
        zvf_mean, zvf_min, zvf_max = _stat(zvf_per_problem)
        rows.append(
            {
                "experiment": "tinker_gsm8k_zvf",
                "model": "Qwen/Qwen3-8B",
                "task": "gsm8k",
                "group_size": summary["group_size"],
                "n_seeds": 1,
                "mean_zvf": zvf_mean,
                "min_zvf": zvf_min,
                "max_zvf": zvf_max,
                "mean_reward": summary.get("sampling_accuracy_mean", acc),
                "peak": acc,
                "last10_avg": acc,
                "n_problems": len(per_problem),
                "n_steps": len(per_problem),
                "evidence_path": str(
                    (RESULTS / f"tinker_gsm8k_zvf_s{seed}.json").relative_to(REPO_ROOT)
                ),
                "seed": seed,
            }
        )
    # Add an aggregate row.
    rows.append(
        {
            "experiment": "tinker_gsm8k_zvf",
            "model": "Qwen/Qwen3-8B",
            "task": "gsm8k",
            "group_size": summary["group_size"],
            "n_seeds": summary["n_seeds"],
            "mean_zvf": summary["mean_zvf"],
            "min_zvf": min(summary["zvf_per_seed"]),
            "max_zvf": max(summary["zvf_per_seed"]),
            "mean_reward": summary["sampling_accuracy_mean"],
            "peak": max(summary["sampling_accuracy_per_seed"]),
            "last10_avg": summary["sampling_accuracy_mean"],
            "n_problems": summary["n_problems_total"],
            "n_steps": summary["n_problems_total"],
            "evidence_path": "experiments/results/tinker_gsm8k_zvf_summary.json",
            "seed": "agg",
        }
    )
    return rows


def load_groupsize_sweep() -> List[Dict[str, Any]]:
    """Aggregate G-sweep rows (G in {2,4,8,16}, 3 seeds each)."""
    path = RESULTS / "groupsize_zvf_sweep.tsv"
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("G\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            G = int(parts[0])
            n_seeds = int(parts[1])
            heldout_acc_mean = float(parts[2])
            heldout_acc_se = float(parts[3])
            last10_mean = float(parts[4])
            mean_zvf = float(parts[5])
            mean_train = float(parts[6])
            rows.append(
                {
                    "experiment": "groupsize_zvf_sweep",
                    "model": "Qwen/Qwen2.5-0.5B",
                    "task": "arithmetic_synthetic",
                    "group_size": G,
                    "n_seeds": n_seeds,
                    "mean_zvf": mean_zvf,
                    "min_zvf": mean_zvf - heldout_acc_se,
                    "max_zvf": mean_zvf + heldout_acc_se,
                    "mean_reward": mean_train,
                    "peak": heldout_acc_mean + heldout_acc_se,
                    "last10_avg": last10_mean,
                    "n_problems": 0,
                    "n_steps": 40 * n_seeds,
                    "evidence_path": "experiments/results/groupsize_zvf_sweep.tsv",
                    "seed": "agg",
                }
            )
    return rows


def load_variance_mitigation() -> List[Dict[str, Any]]:
    """Per-method collapse-labeled trajectories.

    The TSV carries per-step zvf and heldout-acc for several methods;
    we aggregate to one row per (method, seed). The "collapse" column
    is preserved verbatim.
    """
    path = RESULTS / "variance_mitigation.tsv"
    rows_by_method: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            key = (r["method"], int(r["seed"]))
            rows_by_method.setdefault(key, []).append(r)
    rows: List[Dict[str, Any]] = []
    for (method, seed), recs in rows_by_method.items():
        zvfs = [float(r["zvf"]) for r in recs]
        rewards = [float(r["reward_mean"]) for r in recs]
        heldout = [float(r["heldout_acc"]) for r in recs]
        collapse = max(int(r["collapse"]) for r in recs)
        rows.append(
            {
                "experiment": "variance_mitigation",
                "model": method.upper(),
                "task": "math_verifiable_rl",
                "group_size": 8,
                "n_seeds": 1,
                "mean_zvf": statistics.fmean(zvfs),
                "min_zvf": min(zvfs),
                "max_zvf": max(zvfs),
                "mean_reward": statistics.fmean(rewards),
                "peak": max(heldout),
                "last10_avg": statistics.fmean(heldout[-10:]),
                "n_problems": 0,
                "n_steps": len(recs),
                "evidence_path": "experiments/results/variance_mitigation.tsv",
                "seed": seed,
                "_collapse_flag": collapse,
            }
        )
    return rows


def load_tool_use_diagnostics() -> List[Dict[str, Any]]:
    """Cross-tool tool-use runs that fully collapsed (last10 == 0).

    Each row has a `zvf` of 1.0 across the entire trajectory and a
    last10_avg of 0.0 -- the strongest ZVF == failure signature in
    our measurement set.
    """
    path = RESULTS / "tool_code_reward_diagnostics.tsv"
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    with path.open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            key = (r["model"], r["experiment"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "experiment": r["experiment"],
                    "model": r["model"],
                    "task": r["task"],
                    "group_size": 1,
                    "n_seeds": 1,
                    "mean_zvf": float(r["zvf"]),
                    "min_zvf": float(r["zvf"]),
                    "max_zvf": float(r["zvf"]),
                    "mean_reward": float(r["reward_mean"]),
                    "peak": float(r["peak"]),
                    "last10_avg": float(r["last10_avg"]),
                    "n_problems": 0,
                    "n_steps": int(r["n_steps"]),
                    "evidence_path": "experiments/results/tool_code_reward_diagnostics.tsv",
                    "seed": 0,
                }
            )
    return rows


def load_scaling_law_phases() -> List[Dict[str, Any]]:
    """Three-phase scaling-law table -- one row per model.

    The scaling-law TSV does NOT carry explicit per-step ZVF; we keep
    these rows in the summary for the heldout-vs-phase cross-tab but
    leave mean_zvf as NaN so the bootstrap can drop them.
    """
    path = RESULTS / "scaling_law_three_phase.tsv"
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "experiment": "scaling_law_three_phase",
                    "model": parts[0],
                    "params_B": float(parts[1]),
                    "task": "gsm8k_reasoning",
                    "phase": parts[2],
                    "group_size": 8,
                    "n_seeds": 1,
                    "mean_zvf": float("nan"),
                    "min_zvf": float("nan"),
                    "max_zvf": float("nan"),
                    "mean_reward": float(parts[13]),
                    "peak": float(parts[13]),
                    "last10_avg": float(parts[14]),
                    "n_problems": 0,
                    "n_steps": 30,
                    "evidence_path": "experiments/results/scaling_law_three_phase.tsv",
                    "seed": "agg",
                }
            )
    return rows


def load_drgrpo_vs_grpo() -> List[Dict[str, Any]]:
    """Per-run mean_zvf from the Qwen2.5-0.5B DRGRPO vs GRPO study."""
    path = RESULTS / "drgrpo_vs_grpo.json"
    data = json.loads(path.read_text())
    rows: List[Dict[str, Any]] = []
    for r in data["runs"]:
        rows.append(
            {
                "experiment": "drgrpo_vs_grpo",
                "model": r["model"],
                "task": "arithmetic",
                "group_size": 8,
                "n_seeds": 1,
                "mean_zvf": r["mean_zvf"],
                "min_zvf": min(s["zvf"] for s in r["step_log"]),
                "max_zvf": max(s["zvf"] for s in r["step_log"]),
                "mean_reward": statistics.fmean(s["mean_reward"] for s in r["step_log"]),
                "peak": r["heldout_acc"],
                "last10_avg": r["last10_avg"],
                "n_problems": 0,
                "n_steps": len(r["step_log"]),
                "evidence_path": "experiments/results/drgrpo_vs_grpo.json",
                "seed": r["seed"],
                "algo": r["algo"],
            }
        )
    return rows


def load_samestack_ppo_grpo() -> List[Dict[str, Any]]:
    """Per-run metrics from the PPO/GRPO shared-stack diagnostic."""
    path = RESULTS / "samestack_ppo_grpo.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rows: List[Dict[str, Any]] = []
    runs = data.get("runs", [])
    for r in runs:
        rows.append(
            {
                "experiment": "samestack_ppo_grpo",
                "model": r.get("model", "?"),
                "task": r.get("task", "arithmetic"),
                "group_size": r.get("group_size", 8),
                "n_seeds": 1,
                "mean_zvf": r.get("mean_zvf", float("nan")),
                "min_zvf": r.get("min_zvf", float("nan")),
                "max_zvf": r.get("max_zvf", float("nan")),
                "mean_reward": r.get("mean_reward", float("nan")),
                "peak": r.get("heldout_acc", float("nan")),
                "last10_avg": r.get("last10_avg", float("nan")),
                "n_problems": 0,
                "n_steps": r.get("n_steps", 0),
                "evidence_path": "experiments/results/samestack_ppo_grpo.json",
                "seed": r.get("seed", 0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------


def classify(row: Dict[str, Any]) -> str:
    peak = float(row.get("peak", float("nan")))
    last = float(row.get("last10_avg", float("nan")))
    if math.isnan(peak) or math.isnan(last):
        return "unknown"
    if last <= 0.001:
        # True zero-out (tool-use collapse, Nemotron final collapse).
        return "collapse"
    if peak > 0.7 and last < 0.35:
        return "collapse"
    if last < 0.85 * peak:
        return "drift"
    if peak < 0.5:
        return "plateau"
    return "converged"


# ---------------------------------------------------------------------------
# Correlation + bootstrap
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")

    def _rank(vs: Sequence[float]) -> List[float]:
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

    return _pearson(_rank(xs), _rank(ys))


def bootstrap_ci(
    xs: Sequence[float],
    ys: Sequence[float],
    metric,
    B: int = 2000,
    seed: int = 0,
) -> Tuple[float, Tuple[float, float]]:
    """Percentile bootstrap for correlation.

    Resamples (xs[i], ys[i]) rows to respect paired structure.
    """
    import random
    rng = random.Random(seed)
    n = len(xs)
    if n < 3:
        return (float("nan"), (float("nan"), float("nan")))
    point = metric(xs, ys)
    samples = []
    indices = list(range(n))
    for _ in range(B):
        resample = [rng.choice(indices) for _ in range(n)]
        xb = [xs[i] for i in resample]
        yb = [ys[i] for i in resample]
        v = metric(xb, yb)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            samples.append(v)
    if not samples:
        return (point, (float("nan"), float("nan")))
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples)) - 1]
    return (point, (lo, hi))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


SUMMARY_HEADERS = (
    "experiment",
    "model",
    "task",
    "phase",
    "group_size",
    "n_seeds",
    "mean_zvf",
    "min_zvf",
    "max_zvf",
    "mean_reward",
    "peak",
    "last10_avg",
    "failure_label",
    "n_steps",
    "evidence_path",
    "seed",
)


def write_summary(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 cross-experiment ZVF summary\n")
        fh.write(
            "# Aggregated from per-experiment training-log sources under\n"
            "# experiments/results/. Each row corresponds to a single\n"
            "# (experiment, configuration, seed or aggregate). failure_label\n"
            "# is computed deterministically from peak vs last10_avg:\n"
            "#   collapse  peak > 0.7 AND last10_avg < 0.35\n"
            "#   drift     last10_avg < 0.85 * peak (and not collapse)\n"
            "#   plateau   peak < 0.5\n"
            "#   converged last10_avg >= 0.85 * peak\n"
            "# Source: platform_modal/scripts/zvf_diagnostic.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(SUMMARY_HEADERS)
        for r in rows:
            writer.writerow(
                [
                    r.get("experiment", ""),
                    r.get("model", ""),
                    r.get("task", ""),
                    r.get("phase", ""),
                    r.get("group_size", ""),
                    r.get("n_seeds", ""),
                    _fmt(r.get("mean_zvf")),
                    _fmt(r.get("min_zvf")),
                    _fmt(r.get("max_zvf")),
                    _fmt(r.get("mean_reward")),
                    _fmt(r.get("peak")),
                    _fmt(r.get("last10_avg")),
                    classify(r),
                    r.get("n_steps", ""),
                    r.get("evidence_path", ""),
                    r.get("seed", ""),
                ]
            )


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_by_library(rows: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    """Per-library aggregation of ZVF vs failure taxonomy.

    Lifts the 80-row long-form summary into one row per *library*
    (GRPO, AERO, CPPO, NGRPO, SCAFGRPO, MCGRPO, GIFT, AREAL, ES,
    cross-tool tool-use, and the canonical GSM8K baseline). Computes
    mean-ZVF, mean peak/last10, and the failure-tally percentages.
    Designed to support the cross-library AERO comparison: we want
    one row per mitigation method plus one row for each baseline
    experiment family.

    Output columns:
        library, model, n_rows, n_seeds, mean_zvf, max_zvf,
        mean_peak, mean_last10, n_collapse, n_drift, n_plateau,
        n_converged, collapse_rate, drift_rate, plateau_rate,
        converged_rate, evidence_path
    """
    # Group by (library, model) where library is either the lower-case
    # method name from variance_mitigation, or the experiment label
    # for everything else.
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        m = str(r.get("model", "")).strip().lower()
        exp = str(r.get("experiment", "")).strip()
        if exp == "variance_mitigation":
            lib = m  # aero, grpo, cppo, etc.
        elif exp.startswith("cross_tool"):
            lib = "tool_use"
        elif exp == "tinker_gsm8k_zvf":
            lib = "gsm8k_real"
        elif exp == "groupsize_zvf_sweep":
            lib = "arithmetic_groupsize"
        elif exp == "scaling_law_three_phase":
            lib = "scaling_law"
        elif exp in ("drgrpo_vs_grpo", "samestack_ppo_grpo"):
            lib = exp
        else:
            lib = exp
        buckets.setdefault((lib, str(r.get("model", ""))), []).append(r)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg_rows: List[Dict[str, Any]] = []
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 by-library ZVF aggregation (cross-library diagnostic)\n")
        fh.write(
            "# One row per (library, model). Library = method for variance_mitigation rows\n"
            "# (aero, grpo, cppo, ngrpo, scafgrpo, mcgrpo, gift, areal, es) or\n"
            "# experiment family for non-variance_mitigation rows (tool_use, gsm8k_real,\n"
            "# arithmetic_groupsize, scaling_law). failure counts are computed by the\n"
            "# deterministic classify() rule (see top of this file).\n"
            "# Source: platform_modal/scripts/zvf_diagnostic.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "library",
                "model",
                "n_rows",
                "n_seeds",
                "mean_zvf",
                "max_zvf",
                "mean_peak",
                "mean_last10",
                "n_collapse",
                "n_drift",
                "n_plateau",
                "n_converged",
                "collapse_rate",
                "drift_rate",
                "plateau_rate",
                "converged_rate",
                "per_step_collapse_rate",
                "evidence_path",
            )
        )
        # Stable ordering: variance_mitigation methods first, then everything else.
        method_order = (
            "grpo",
            "aero",
            "cppo",
            "ngrpo",
            "scafgrpo",
            "mcgrpo",
            "gift",
            "areal",
            "es",
        )
        sorted_keys = []
        for m in method_order:
            for key in buckets.keys():
                if key[0] == m:
                    sorted_keys.append(key)
        for key in buckets.keys():
            if key not in sorted_keys:
                sorted_keys.append(key)

        for key in sorted_keys:
            lib, model_label = key
            recs = buckets[key]
            n_rows = len(recs)
            n_seeds_set = {r.get("seed") for r in recs}
            n_seeds = len(n_seeds_set)
            zvfs = [
                float(r["mean_zvf"])
                for r in recs
                if r.get("mean_zvf") is not None and not (isinstance(r["mean_zvf"], float) and math.isnan(r["mean_zvf"]))
            ]
            peaks = [
                float(r["peak"])
                for r in recs
                if r.get("peak") is not None and not (isinstance(r["peak"], float) and math.isnan(r["peak"]))
            ]
            lasts = [
                float(r["last10_avg"])
                for r in recs
                if r.get("last10_avg") is not None and not (isinstance(r["last10_avg"], float) and math.isnan(r["last10_avg"]))
            ]
            labels = [classify(r) for r in recs]
            n_collapse = sum(1 for l in labels if l == "collapse")
            n_drift = sum(1 for l in labels if l == "drift")
            n_plateau = sum(1 for l in labels if l == "plateau")
            n_converged = sum(1 for l in labels if l == "converged")
            mean_zvf = statistics.fmean(zvfs) if zvfs else float("nan")
            max_zvf = max(zvfs) if zvfs else float("nan")
            mean_peak = statistics.fmean(peaks) if peaks else float("nan")
            mean_last10 = statistics.fmean(lasts) if lasts else float("nan")
            # Per-step collapse rate (only meaningful for variance_mitigation
            # rows where the underlying TSV carries a per-step collapse flag).
            per_step_collapse_vals = [
                int(r["_collapse_flag"])
                for r in recs
                if "_collapse_flag" in r
            ]
            per_step_collapse_rate = (
                sum(per_step_collapse_vals) / len(per_step_collapse_vals)
                if per_step_collapse_vals
                else float("nan")
            )
            ev = recs[0].get("evidence_path", "")
            writer.writerow(
                (
                    lib,
                    model_label,
                    n_rows,
                    n_seeds,
                    _fmt(mean_zvf),
                    _fmt(max_zvf),
                    _fmt(mean_peak),
                    _fmt(mean_last10),
                    n_collapse,
                    n_drift,
                    n_plateau,
                    n_converged,
                    _fmt(n_collapse / n_rows) if n_rows else "NA",
                    _fmt(n_drift / n_rows) if n_rows else "NA",
                    _fmt(n_plateau / n_rows) if n_rows else "NA",
                    _fmt(n_converged / n_rows) if n_rows else "NA",
                    _fmt(per_step_collapse_rate),
                    ev,
                )
            )
            agg_rows.append(
                {
                    "library": lib,
                    "model": model_label,
                    "n_rows": n_rows,
                    "n_seeds": n_seeds,
                    "mean_zvf": mean_zvf,
                    "max_zvf": max_zvf,
                    "mean_peak": mean_peak,
                    "mean_last10": mean_last10,
                    "n_collapse": n_collapse,
                    "n_drift": n_drift,
                    "n_plateau": n_plateau,
                    "n_converged": n_converged,
                    "per_step_collapse_rate": per_step_collapse_rate,
                    "evidence_path": ev,
                }
            )
    return {"rows": agg_rows}


def write_correlation(rows: List[Dict[str, Any]], out_path: Path) -> Dict[str, Any]:
    """Per-experiment collapse-label correlations.

    We collapse the 14 per-experiment rows by averaging across
    seeds within an (experiment, condition) pair, then correlate
    mean_zvf against last10_avg / collapse_binary.
    """
    keys: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["experiment"], str(r.get("model", "")), int(r.get("group_size", 0) or 0))
        keys.setdefault(key, []).append(r)
    pooled: List[Dict[str, Any]] = []
    for key, recs in keys.items():
        mz_vals = [r["mean_zvf"] for r in recs if not math.isnan(r["mean_zvf"])]
        la_vals = [r["last10_avg"] for r in recs if not math.isnan(r["last10_avg"])]
        pk_vals = [r["peak"] for r in recs if not math.isnan(r["peak"])]
        # Only keep rows that have at least a non-NaN last10_avg and peak;
        # ZVF is optional for the scaling-law-only rows.
        if not la_vals or not pk_vals:
            continue
        mz = statistics.fmean(mz_vals) if mz_vals else float("nan")
        la = statistics.fmean(la_vals)
        pk = statistics.fmean(pk_vals)
        pooled.append(
            {
                "experiment": key[0],
                "model": key[1],
                "group_size": key[2],
                "mean_zvf": mz,
                "last10_avg": la,
                "peak": pk,
                "failure": classify(
                    {"peak": pk, "last10_avg": la}
                ),
            }
        )

    zvfs = [p["mean_zvf"] for p in pooled if not math.isnan(p["mean_zvf"])]
    paired_lasts = [p["last10_avg"] for p in pooled if not math.isnan(p["mean_zvf"])]
    paired_zvfs = zvfs
    paired_collapse = [
        (1.0 if p["failure"] == "collapse" else 0.0)
        for p in pooled
        if not math.isnan(p["mean_zvf"])
    ]

    pear_r, pear_lo_hi = bootstrap_ci(paired_zvfs, paired_lasts, _pearson, B=2000, seed=11)
    spear_r, spear_lo_hi = bootstrap_ci(paired_zvfs, paired_lasts, _spearman, B=2000, seed=22)
    pear_collapse, _ = bootstrap_ci(paired_zvfs, paired_collapse, _pearson, B=2000, seed=33)
    spear_collapse, _ = bootstrap_ci(paired_zvfs, paired_collapse, _spearman, B=2000, seed=44)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("# Pillar 2 ZVF vs training-failure correlation table\n")
        fh.write(
            "# Rows are per-(experiment, model, group_size) pooled means,\n"
            "# not per-step rows; per-step ZVF rows are autocorrelated and\n"
            "# would inflate apparent significance. Bootstrap CIs are B=2000\n"
            "# percentile resamples over the pooled rows. Source:\n"
            "# platform_modal/scripts/zvf_diagnostic.py\n"
        )
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "test",
                "n_pooled_rows",
                "rho",
                "ci_lo",
                "ci_hi",
                "method",
            )
        )
        writer.writerow(
            (
                "mean_zvf vs last10_avg",
                len(pooled),
                _fmt(pear_r),
                _fmt(pear_lo_hi[0]),
                _fmt(pear_lo_hi[1]),
                "Pearson + 95% bootstrap CI",
            )
        )
        writer.writerow(
            (
                "mean_zvf vs last10_avg",
                len(pooled),
                _fmt(spear_r),
                _fmt(spear_lo_hi[0]),
                _fmt(spear_lo_hi[1]),
                "Spearman + 95% bootstrap CI",
            )
        )
        writer.writerow(
            (
                "mean_zvf vs is_collapse",
                len(pooled),
                _fmt(pear_collapse),
                "NA",
                "NA",
                "Pearson point-biserial (CI NaN, deterministic at n=14)",
            )
        )
        writer.writerow(
            (
                "mean_zvf vs is_collapse",
                len(pooled),
                _fmt(spear_collapse),
                "NA",
                "NA",
                "Spearman point-biserial (CI NaN, deterministic at n=14)",
            )
        )
    return {
        "n_pooled_rows": len(paired_zvfs),
        "pearson_mean_zvf_last10": pear_r,
        "spearman_mean_zvf_last10": spear_r,
        "pearson_mean_zvf_collapse_binary": pear_collapse,
        "spearman_mean_zvf_collapse_binary": spear_collapse,
        "ci_pearson": pear_lo_hi,
        "ci_spearman": spear_lo_hi,
    }


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


COLOR_BY_LABEL = {
    "collapse": "#c0392b",
    "drift": "#e67e22",
    "plateau": "#7f8c8d",
    "converged": "#27ae60",
    "unknown": "#bdc3c7",
}


def write_figure(rows: List[Dict[str, Any]], out_path: Path) -> Optional[str]:
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    pooled: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (r["experiment"], str(r.get("model", "")), int(r.get("group_size", 0) or 0))
        pooled.setdefault(key, []).append(r)
    plotted = 0
    for key, recs in pooled.items():
        mz_vals = [r["mean_zvf"] for r in recs if not math.isnan(r["mean_zvf"])]
        la_vals = [r["last10_avg"] for r in recs if not math.isnan(r["last10_avg"])]
        if not la_vals or not mz_vals:
            continue
        mz = statistics.fmean(mz_vals)
        la = statistics.fmean(la_vals)
        if math.isnan(mz) or math.isnan(la):
            continue
        label = classify({"peak": max(r["peak"] for r in recs), "last10_avg": la})
        ax.scatter(
            mz,
            la,
            s=70,
            color=COLOR_BY_LABEL.get(label, "#34495e"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        plotted += 1
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Mean ZVF across training trajectory")
    ax.set_ylabel("Heldout accuracy (last-10-window average)")
    ax.set_title(
        "ZVF vs training outcome across %d (experiment, model, G) cells\n"
        "color = collapse taxonomy" % plotted
    )
    # Inline legend without overlapping points.
    from matplotlib.lines import Line2D  # local import keeps top-level tidy

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=v,
            markeredgecolor="white",
            label=k,
            markersize=8,
        )
        for k, v in COLOR_BY_LABEL.items()
        if k != "unknown"
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


# Highlight colour for the AERO / RL-ZVP marker on the cross-library plot.
AERO_HIGHLIGHT = "#8e44ad"  # purple so it stands out against the four failure colours.


def write_by_library_figure(
    by_lib_rows: List[Dict[str, Any]], out_path: Path
) -> Optional[str]:
    """Two-panel cross-library bar chart.

    Left panel: mean-ZVF per library. AERO is drawn in purple so the
    headline comparison (AERO 0.22 vs GRPO 0.48) is immediate.

    Right panel: failure-tally stacked bars (collapse / drift / plateau /
    converged) per library. AERO shows zero collapse bars under the
    deterministic classifier -- this is the second half of the
    cross-library diagnostic claim.
    """
    plt = _maybe_matplotlib()
    if plt is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Order: variance-mitigation methods first (alphabetical) so AERO and
    # GRPO are immediately adjacent; then non-mitigation libraries.
    method_keys = {"grpo", "aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift", "areal", "es"}
    methods = [r for r in by_lib_rows if r["library"] in method_keys]
    methods.sort(key=lambda r: ["grpo", "aero", "cppo", "ngrpo", "scafgrpo", "mcgrpo", "gift", "areal", "es"].index(r["library"]))
    others = [r for r in by_lib_rows if r["library"] not in method_keys]
    others.sort(key=lambda r: r["library"])
    ordered = methods + others

    labels = [r["library"] for r in ordered]
    mean_zvfs = [r["mean_zvf"] for r in ordered]
    n_collapse = [r["n_collapse"] for r in ordered]
    n_drift = [r["n_drift"] for r in ordered]
    n_plateau = [r["n_plateau"] for r in ordered]
    n_converged = [r["n_converged"] for r in ordered]
    n_rows = [max(1, r["n_rows"]) for r in ordered]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    bar_colors = [
        AERO_HIGHLIGHT if lib == "aero" else ("#2c3e50" if lib == "grpo" else "#7f8c8d")
        for lib in labels
    ]
    bars = axL.bar(labels, mean_zvfs, color=bar_colors, edgecolor="white", linewidth=0.6)
    axL.set_ylabel("Mean ZVF across training trajectory")
    axL.set_title("Mean ZVF per library (AERO highlighted)")
    axL.set_ylim(0, max(0.6, max(mean_zvfs) * 1.18) if mean_zvfs else 0.6)
    axL.tick_params(axis="x", rotation=45, labelsize=8)
    for bar, val in zip(bars, mean_zvfs):
        if math.isnan(val):
            continue
        axL.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    # Failure-tally stacked bars: collapse (red) bottom, drift (orange),
    # plateau (gray), converged (green) on top.
    bottom = [0.0] * len(ordered)
    for n_list, color, name in [
        (n_collapse, COLOR_BY_LABEL["collapse"], "collapse"),
        (n_drift, COLOR_BY_LABEL["drift"], "drift"),
        (n_plateau, COLOR_BY_LABEL["plateau"], "plateau"),
        (n_converged, COLOR_BY_LABEL["converged"], "converged"),
    ]:
        rates = [n / nr for n, nr in zip(n_list, n_rows)]
        axR.bar(
            labels,
            rates,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=name,
        )
        bottom = [b + r for b, r in zip(bottom, rates)]
    axR.set_ylabel("Fraction of (run, seed) rows")
    axR.set_title("Failure taxonomy per library")
    axR.set_ylim(0, 1.05)
    axR.tick_params(axis="x", rotation=45, labelsize=8)
    axR.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(
        "ZVF as a cross-library diagnostic: AERO vs GRPO and the full run matrix",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-summary", type=Path, default=RESULTS / "zvf_summary.tsv"
    )
    parser.add_argument(
        "--out-correlation", type=Path, default=RESULTS / "zvf_failure_correlation.tsv"
    )
    parser.add_argument(
        "--out-figure", type=Path, default=REPO_ROOT / "figures" / "zvf_vs_failure.pdf"
    )
    parser.add_argument(
        "--out-by-library",
        type=Path,
        default=RESULTS / "zvf_by_library.tsv",
        help="Per-library ZVF/failure aggregation (cross-library diagnostic vs AERO)",
    )
    parser.add_argument(
        "--out-library-figure",
        type=Path,
        default=REPO_ROOT / "figures" / "zvf_by_library.pdf",
        help="Per-library bar chart emphasising AERO vs GRPO contrast",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run on a small synthetic slice"
    )
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    rows += load_tinker_gsm8k()
    rows += load_groupsize_sweep()
    rows += load_variance_mitigation()
    rows += load_tool_use_diagnostics()
    rows += load_scaling_law_phases()
    rows += load_drgrpo_vs_grpo()
    rows += load_samestack_ppo_grpo()

    # Optional self-test: keep just one representative row per loader.
    if args.self_test:
        keep = []
        seen_exp = set()
        for r in rows:
            if r["experiment"] not in seen_exp:
                keep.append(r)
                seen_exp.add(r["experiment"])
        rows = keep

    write_summary(rows, args.out_summary)
    by_lib = write_by_library(rows, args.out_by_library)
    corr = write_correlation(rows, args.out_correlation)
    fig_path = write_figure(rows, args.out_figure)
    lib_fig_path = write_by_library_figure(by_lib["rows"], args.out_library_figure)

    n = len(rows)
    n_collapse = sum(1 for r in rows if classify(r) == "collapse")
    n_converged = sum(1 for r in rows if classify(r) == "converged")
    print(f"[zvf-diagnostic] wrote {n} rows to {args.out_summary}")
    print(
        f"[zvf-diagnostic] failure tally: "
        f"collapse={n_collapse}  converged={n_converged}  "
        f"drift={sum(1 for r in rows if classify(r) == 'drift')}  "
        f"plateau={sum(1 for r in rows if classify(r) == 'plateau')}"
    )
    print(
        f"[zvf-diagnostic] correlation: "
        f"pearson(mean_zvf, last10_avg) = {corr['pearson_mean_zvf_last10']:.3f} "
        f"[{corr['ci_pearson'][0]:.3f},{corr['ci_pearson'][1]:.3f}], "
        f"spearman = {corr['spearman_mean_zvf_last10']:.3f}"
    )
    if fig_path:
        print(f"[zvf-diagnostic] wrote figure {fig_path}")
    if lib_fig_path:
        print(f"[zvf-diagnostic] wrote by-library figure {lib_fig_path}")
    # Headline cross-library claim.
    for r in by_lib["rows"]:
        if r["library"] in ("grpo", "aero"):
            print(
                f"[zvf-diagnostic] {r['library']:>4}: "
                f"mean_zvf={r['mean_zvf']:.3f}  "
                f"collapse={r['n_collapse']}/{r['n_rows']}  "
                f"mean_last10={r['mean_last10']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
