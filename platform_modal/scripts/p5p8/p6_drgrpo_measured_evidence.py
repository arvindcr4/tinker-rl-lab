#!/usr/bin/env python3
"""P6 v2: preserve task-specific paired estimates without inventing pooled uncertainty.

Input intervals are reproduced as reported, not recalculated. Raw paired runs
would be needed for an uncertainty-propagating fixed-task aggregate. This script
never patches the registry and makes no directional length-bias proxy claim.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import REPO_ROOT, RESULTS_DIR, REGISTRY_DIR
from _review01_contracts import (
    OUTPUT_ROOT,
    evidence_metadata,
    require_inputs,
    write_artifact_metadata,
)

ROOT = REPO_ROOT
ENTRIES = REGISTRY_DIR / "entries"
SCHEMA = REGISTRY_DIR / "schema.json"
LBIAS = RESULTS_DIR / "length_bias_iter60_grpo_vs_drgrpo.tsv"
OUT = OUTPUT_ROOT / "p6"


def welch_pooled_tstat(diffs):
    """Compatibility API: equal-weight descriptive mean; aggregate CI unavailable.

    Task means alone cannot identify within-task sampling uncertainty. The legacy
    function name is retained only for callers; no Welch test is performed.
    """
    if not diffs or not all(math.isfinite(float(x)) for x in diffs):
        raise ValueError("Finite, nonempty task effects are required")
    return statistics.mean(diffs), None, None, False


def summarize(rows):
    per_metric = {}
    identities = set()
    for row in rows:
        if row["metric"] not in {"neg_frac", "pos_frac", "L_star", "R_max_fit", "n_pts"}:
            continue
        if row["kind"] not in {"elasticity", "curvature"}:
            continue
        identity = row["task"], row["kind"], row["metric"]
        if identity in identities:
            raise ValueError(f"Duplicate task estimate: {identity}")
        identities.add(identity)
        estimate = {
            key: float(row[key])
            for key in ("mean_grpo", "mean_drgrpo", "mean_diff", "ci_lo", "ci_hi")
        }
        if (
            not all(math.isfinite(x) for x in estimate.values())
            or estimate["ci_lo"] > estimate["ci_hi"]
        ):
            raise ValueError(f"Invalid task estimate: {identity}")
        n_pairs = int(row["n_pairs"])
        if n_pairs < 2:
            raise ValueError("Task-specific intervals require at least two paired observations")
        estimate.update(
            task=row["task"], n_pairs=n_pairs, ci_method="reported_source_interval_not_recomputed"
        )
        per_metric.setdefault(row["metric"], []).append(estimate)
    if not per_metric:
        raise ValueError("No task-specific paired results found")
    result = {}
    for metric, tasks in per_metric.items():
        point, low, high, significant = welch_pooled_tstat([r["mean_diff"] for r in tasks])
        result[metric] = {
            "per_task": tasks,
            "descriptive_task_mean": point,
            "task_weights": {r["task"]: 1 / len(tasks) for r in tasks},
            "aggregate_ci": None,
            "significant": significant,
            "verdict": "INCONCLUSIVE",
            "predicted_sign": None,
            "reason": "Raw paired runs and justified directional proxy hypotheses are required",
        }
    return result


def main():
    require_inputs(LBIAS)
    with LBIAS.open() as stream:
        results = summarize(list(csv.DictReader(stream, delimiter="\t")))
    metadata = evidence_metadata(
        "task_specific_source_intervals_and_descriptive_equal_weight_task_mean", [LBIAS]
    )
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {**metadata, "metrics": results, "registry_update": False}
    (OUT / "p6_drgrpo_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (OUT / "p6_drgrpo_measured.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "metric",
                "task",
                "n_pairs",
                "mean_grpo",
                "mean_drgrpo",
                "mean_diff",
                "ci_lo",
                "ci_hi",
                "ci_method",
            ]
        )
        for metric, estimate in results.items():
            for task in estimate["per_task"]:
                writer.writerow(
                    [metric]
                    + [
                        task[key]
                        for key in [
                            "task",
                            "n_pairs",
                            "mean_grpo",
                            "mean_drgrpo",
                            "mean_diff",
                            "ci_lo",
                            "ci_hi",
                            "ci_method",
                        ]
                    ]
                )
    write_artifact_metadata(OUT, metadata)
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
