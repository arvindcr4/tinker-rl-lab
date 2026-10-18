#!/usr/bin/env python3
"""Versioned exploratory ZVF diagnostics.

Static sampling accuracy, training reward, and held-out trajectory accuracy
are distinct estimands. Missing measurements remain unavailable. Conditions
are never treated as independent experimental units for inferential intervals
without a documented sampling design.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "platform_hybrid" / "experiments" / "results"
ANALYSIS_VERSION = "review27-v3"


# ---------------------------------------------------------------------------
# Loaders: each one returns a list of dicts, one per (experiment, condition).
# ---------------------------------------------------------------------------


def _stat(xs: Sequence[float]) -> Tuple[float, float, float]:
    """mean, min, max with NaN-safe handling."""
    xs = [float(x) for x in xs if _finite(x)]
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    return (statistics.fmean(xs), min(xs), max(xs))


def load_tinker_gsm8k() -> List[Dict[str, Any]]:
    """Real Qwen3-8B / GSM8K rollouts, K=8 per problem, 3 seeds."""
    summary_path = RESULTS / "tinker_gsm8k_zvf_summary.json"
    summary = json.loads(summary_path.read_text())
    rows: List[Dict[str, Any]] = []
    if len({len(summary[k]) for k in ("seeds", "zvf_per_seed", "sampling_accuracy_per_seed")}) != 1 or len(set(summary["seeds"])) != len(summary["seeds"]):
        raise ValueError("inconsistent or duplicate sampling seed identities")
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
                "mean_reward": acc,
                "sampling_accuracy": acc,
                "peak": None,
                "last10_avg": None,
                "n_problems": len(per_problem),
                "n_steps": None,
                "evidence_path": str(
                    (RESULTS / f"tinker_gsm8k_zvf_s{seed}.json").resolve()
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
            "peak": None,
            "last10_avg": None,
            "sampling_accuracy": summary["sampling_accuracy_mean"],
            "n_problems": summary["n_problems_total"],
            "n_steps": None,
            "evidence_path": str((RESULTS / "tinker_gsm8k_zvf_summary.json").resolve()),
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
                    "min_zvf": None,
                    "max_zvf": None,
                    "mean_reward": mean_train,
                    "peak": None,
                    "heldout_accuracy": heldout_acc_mean,
                    "heldout_accuracy_se": heldout_acc_se,
                    "last10_avg": last10_mean,
                    "n_problems": None,
                    "n_steps": None,
                    "evidence_path": str((RESULTS / "groupsize_zvf_sweep.tsv").resolve()),
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
        recs = sorted(recs, key=lambda r: int(r["step"]))
        if len({int(r["step"]) for r in recs}) != len(recs) or any(r["collapse"] not in ("0", "1") for r in recs):
            raise ValueError("duplicate step or invalid collapse flag")
        zvfs = [float(r["zvf"]) for r in recs]
        rewards = [float(r["reward_mean"]) for r in recs]
        heldout = [float(r["heldout_acc"]) for r in recs]
        collapse = max(int(r["collapse"]) for r in recs)
        rows.append(
            {
                "experiment": "variance_mitigation",
                "model": "unknown",
                "metric_definition": "heldout_accuracy",
                "task": "math_verifiable_rl",
                "group_size": 8,
                "n_seeds": 1,
                "mean_zvf": statistics.fmean(zvfs),
                "min_zvf": min(zvfs),
                "max_zvf": max(zvfs),
                "mean_reward": statistics.fmean(rewards),
                "peak": max(heldout),
                "last10_avg": statistics.fmean(heldout[-10:]),
                "n_problems": None,
                "n_steps": len(recs),
                "evidence_path": str((RESULTS / "variance_mitigation.tsv").resolve()),
                "seed": seed,
                "_collapse_flag": collapse,
                "collapse_steps": sum(int(r["collapse"]) for r in recs),
                "observed_steps": len(recs),
                "algorithm": method,
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
                    "group_size": None,
                    "n_seeds": None,
                    "mean_zvf": float(r["zvf"]),
                    "min_zvf": float(r["zvf"]),
                    "max_zvf": float(r["zvf"]),
                    "mean_reward": float(r["reward_mean"]),
                    "peak": float(r["peak"]),
                    "last10_avg": float(r["last10_avg"]),
                    "n_problems": None,
                    "n_steps": int(r["n_steps"]),
                    "evidence_path": str((RESULTS / "tool_code_reward_diagnostics.tsv").resolve()),
                    "seed": None,
                    "row_kind": "summary",
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
            if len(parts) < 15:
                continue
            rows.append(
                {
                    "experiment": "scaling_law_three_phase",
                    "model": parts[0],
                    "params_B": float(parts[1]),
                    "task": "gsm8k_reasoning",
                    "phase": parts[11],
                    "peak_segment_mean": float(parts[13]),
                    "late_segment_mean": float(parts[14]),
                    "group_size": None,
                    "n_seeds": None,
                    "mean_zvf": float("nan"),
                    "min_zvf": float("nan"),
                    "max_zvf": float("nan"),
                    "mean_reward": None,
                    "peak": None,
                    "last10_avg": None,
                    "n_problems": None,
                    "n_steps": None,
                    "evidence_path": str((RESULTS / "scaling_law_three_phase.tsv").resolve()),
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
                "peak": None,
                "heldout_accuracy": r["heldout_acc"],
                "last10_avg": r["last10_avg"],
                "n_problems": None,
                "n_steps": len(r["step_log"]),
                "evidence_path": str((RESULTS / "drgrpo_vs_grpo.json").resolve()),
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
                "peak": r.get("peak"),
                "heldout_accuracy": r.get("heldout_acc"),
                "algorithm": r.get("algorithm", r.get("algo", r.get("method", "unknown"))),
                "last10_avg": r.get("last10_avg", float("nan")),
                "n_problems": None,
                "n_steps": r.get("n_steps", 0),
                "evidence_path": str((RESULTS / "samestack_ppo_grpo.json").resolve()),
                "seed": r.get("seed", 0),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------


def classify(row: Dict[str, Any]) -> str:
    peak = row.get("peak")
    last = row.get("last10_avg")
    if not _finite(peak) or not _finite(last):
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


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def observation(row):
    """Explicit measurement and condition identity; never infer an algorithm."""
    r = dict(row)
    r.setdefault("algorithm", r.get("algo", "unknown"))
    r.setdefault("row_kind", "aggregate" if r.get("seed") == "agg" else "run")
    r.setdefault("metric_definition", "sampling_accuracy" if r.get("experiment") == "tinker_gsm8k_zvf" else "source_reported_training_reward")
    r.setdefault("condition_id", "|".join(str(r.get(k, "unknown")) for k in
                 ("experiment", "model", "task", "group_size", "algorithm", "metric_definition")))
    return r


def independent_rows(rows):
    """Exclude a condition aggregate whenever constituent runs are available."""
    rows = [observation(r) for r in rows]
    run_conditions = {r["condition_id"] for r in rows if r["row_kind"] == "run"}
    result = [r for r in rows if r["row_kind"] != "aggregate" or r["condition_id"] not in run_conditions]
    identities = [(r["condition_id"], r["row_kind"], str(r.get("seed"))) for r in result]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate condition/run identity")
    return result


def _fmt(v):
    if v is None or isinstance(v, float) and not math.isfinite(v):
        return "NA"
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def _write_records(path, records, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        fh.write("# review27-v3: descriptive; missing measurements are NA\n")
        writer = csv.DictWriter(fh, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({k: _fmt(record.get(k)) for k in fields})


def write_summary(rows, out_path):
    records = [dict(observation(r), failure_label=classify(r)) for r in rows]
    fields = list(SUMMARY_HEADERS) + ["algorithm", "condition_id", "row_kind", "metric_definition", "sampling_accuracy", "peak_segment_mean", "late_segment_mean", "heldout_accuracy", "heldout_accuracy_se", "collapse_steps", "observed_steps"]
    _write_records(out_path, records, fields)


def write_by_library(rows, out_path):
    buckets = {}
    for r in independent_rows(rows):
        buckets.setdefault(r["condition_id"], []).append(r)
    output = []
    for condition, recs in sorted(buckets.items()):
        first = recs[0]
        labels = [classify(r) for r in recs]
        classified = sum(label != "unknown" for label in labels)
        def avg(key):
            values = [r[key] for r in recs if _finite(r.get(key))]
            return statistics.fmean(values) if values else None
        counted = [r for r in recs if type(r.get("collapse_steps")) is int and type(r.get("observed_steps")) is int and 0 <= r["collapse_steps"] <= r["observed_steps"] and r["observed_steps"] > 0]
        steps = sum(r["observed_steps"] for r in counted)
        collapsed = sum(r["collapse_steps"] for r in counted)
        seeds = {str(r["seed"]) for r in recs if r["row_kind"] == "run" and r.get("seed") is not None}
        record = dict(condition_id=condition, library=first.get("algorithm", "unknown"), model=first.get("model"), algorithm=first["algorithm"], row_kind=first["row_kind"], n_rows=len(recs), n_seeds=len(seeds), n_classified=classified,
            mean_zvf=avg("mean_zvf"), mean_peak=avg("peak"), mean_last10=avg("last10_avg"), collapse_steps=collapsed, observed_steps=steps,
            per_step_collapse_rate=collapsed/steps if steps else None,
            run_ever_collapsed_rate=sum(r["collapse_steps"] > 0 for r in counted)/len(counted) if counted else None,
            evidence_path=";".join(sorted({r.get("evidence_path", "") for r in recs})))
        for label in ("collapse", "drift", "plateau", "converged"):
            record["n_"+label] = labels.count(label)
            record[label+"_rate"] = labels.count(label)/classified if classified else None
        output.append(record)
    fields = list(output[0]) if output else ["condition_id", "n_rows"]
    _write_records(out_path, output, fields)
    return {"rows": output}


def write_correlation(rows, out_path):
    """Complete-case descriptive associations, stratified by metric definition.

    Shared models, prompts and protocols can make distinct conditions dependent.
    No CI is claimed until independent-cluster metadata is supplied and validated.
    """
    rows = independent_rows(rows)
    strata = sorted({r["metric_definition"] for r in rows}) or ["unavailable"]
    output = []
    for stratum in strata:
        subset = [r for r in rows if r["metric_definition"] == stratum]
        for outcome in ("last10_avg", "is_collapse"):
            buckets = {}
            for r in subset:
                y = r.get("last10_avg") if outcome == "last10_avg" else (None if classify(r) == "unknown" else float(classify(r) == "collapse"))
                if _finite(r.get("mean_zvf")) and _finite(y):
                    buckets.setdefault(r["condition_id"], []).append((r["mean_zvf"], y))
            pairs = [(statistics.fmean(x for x,y in values), statistics.fmean(y for x,y in values)) for values in buckets.values()]
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            for name, metric in (("Pearson", _pearson), ("Spearman", _spearman)):
                output.append(dict(test="mean_zvf vs "+outcome, metric_definition=stratum, n_pooled_rows=len(pairs), n_complete_run_rows=sum(map(len,buckets.values())), rho=metric(xs, ys), ci_lo=None, ci_hi=None, method=name+" descriptive; independent sampling design unavailable"))
    _write_records(out_path, output, list(output[0]))
    # Compatibility fields are unavailable when multiple estimands exist;
    # callers must consume the explicit per-stratum records instead of mixing.
    def value(index):
        return output[index]["rho"] if len(strata) == 1 else float("nan")
    return dict(rows=output, n_pooled_rows=output[0]["n_pooled_rows"] if len(strata)==1 else None, pearson_mean_zvf_last10=value(0), spearman_mean_zvf_last10=value(1), pearson_mean_zvf_collapse_binary=value(2), spearman_mean_zvf_collapse_binary=value(3), ci_pearson=(float("nan"),)*2, ci_spearman=(float("nan"),)*2)


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
    pooled = {}
    for r in independent_rows(rows):
        if _finite(r.get("mean_zvf")) and _finite(r.get("last10_avg")):
            pooled.setdefault(r["condition_id"], []).append(r)
    plotted = 0
    for key, recs in pooled.items():
        mz = statistics.fmean(r["mean_zvf"] for r in recs)
        la = statistics.fmean(r["last10_avg"] for r in recs)
        peaks = [r["peak"] for r in recs if _finite(r.get("peak"))]
        label = classify({"peak": statistics.fmean(peaks) if peaks else None, "last10_avg": la})
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
    ax.set_ylabel("Source-reported last-10 outcome (heterogeneous metrics)")
    ax.set_title(
        "ZVF vs training outcome across %d condition cells\n"
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
    return str(out_path.resolve())


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

    labels = [r["condition_id"] for r in ordered]
    mean_zvfs = [r["mean_zvf"] if r["mean_zvf"] is not None else float("nan") for r in ordered]
    n_collapse = [r["n_collapse"] for r in ordered]
    n_drift = [r["n_drift"] for r in ordered]
    n_plateau = [r["n_plateau"] for r in ordered]
    n_converged = [r["n_converged"] for r in ordered]
    n_rows = [max(1, r["n_classified"]) for r in ordered]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    bar_colors = [
        AERO_HIGHLIGHT if lib == "aero" else ("#2c3e50" if lib == "grpo" else "#7f8c8d")
        for lib in labels
    ]
    bars = axL.bar(labels, mean_zvfs, color=bar_colors, edgecolor="white", linewidth=0.6)
    axL.set_ylabel("Mean ZVF across training trajectory")
    axL.set_title("Mean ZVF per condition")
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
        "Exploratory condition summaries; unknown outcomes excluded from taxonomy denominator",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    fig.savefig(out_path.with_suffix(".png"), format="png", dpi=140)
    plt.close(fig)
    return str(out_path.resolve())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-summary", type=Path, default=RESULTS / "zvf_summary_v3.tsv"
    )
    parser.add_argument(
        "--out-correlation", type=Path, default=RESULTS / "zvf_failure_correlation_v3.tsv"
    )
    parser.add_argument(
        "--out-figure", type=Path, default=REPO_ROOT / "figures" / "zvf_vs_failure_v3.pdf"
    )
    parser.add_argument(
        "--out-by-library",
        type=Path,
        default=RESULTS / "zvf_by_library_v3.tsv",
        help="Per-library ZVF/failure aggregation (cross-library diagnostic vs AERO)",
    )
    parser.add_argument(
        "--out-library-figure",
        type=Path,
        default=REPO_ROOT / "figures" / "zvf_by_library_v3.pdf",
        help="Per-library bar chart emphasising AERO vs GRPO contrast",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run on a small synthetic slice"
    )
    parser.add_argument("--input-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    globals()["RESULTS"] = args.input_dir
    if args.self_test:
        assert _pearson([1.,2.,3.],[2.,4.,6.]) > .999
        assert _pearson([1.,2.,3.],[6.,4.,2.]) < -.999
        print("synthetic numerical self-test passed; no inputs loaded or artifacts written")
        return 0


    rows: List[Dict[str, Any]] = []
    rows += load_tinker_gsm8k()
    rows += load_groupsize_sweep()
    rows += load_variance_mitigation()
    rows += load_tool_use_diagnostics()
    rows += load_scaling_law_phases()
    rows += load_drgrpo_vs_grpo()
    rows += load_samestack_ppo_grpo()

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
                f"mean_zvf={_fmt(r['mean_zvf'])}  "
                f"collapse={r['n_collapse']}/{r['n_rows']}  "
                f"mean_last10={_fmt(r['mean_last10'])}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
