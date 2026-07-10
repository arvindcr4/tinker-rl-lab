#!/usr/bin/env python3
"""Aggregate rollout-quality audits across independent evaluation seeds.

This is deliberately labeled an *evaluation-seed* aggregate.  It estimates
uncertainty from prompt selection and sampling for one frozen checkpoint; it
must not be presented as a substitute for independent training-seed runs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from common import RESULTS_DIR, utc_now, write_result


CORE_METRICS = (
    "pass_at_1",
    "zero_variance_prompt_rate",
    "mean_group_reward_variance",
    "active_advantage_fraction",
)
LENGTH_METRICS = (
    "mean_output_tokens",
    "correct_minus_incorrect_tokens",
    "reward_length_correlation",
    "length_predictive_auc",
)


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: list[float], *, seed: int, n_bootstrap: int) -> dict:
    rng = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        resample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(resample))
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "ci_low": _percentile(means, 0.025),
        "ci_high": _percentile(means, 0.975),
        "min": min(values),
        "max": max(values),
        "n_seeds": len(values),
        "values": values,
    }


def aggregate(audits: list[dict], *, seed: int, n_bootstrap: int) -> dict:
    if not audits:
        raise ValueError("no audits supplied")
    for index, audit in enumerate(audits):
        if audit.get("kind") != "rollout_quality_audit" or audit.get("status") != "complete":
            raise ValueError(f"input {index} is not a completed rollout quality audit")

    compatibility_keys = (
        "model", "checkpoint_identity", "split", "temperature", "top_p",
        "rollouts_per_prompt", "n_prompts", "max_tokens"
    )
    reference = audits[0]["source"]
    mismatches = []
    for index, audit in enumerate(audits[1:], start=1):
        source = audit["source"]
        differing = {
            key: {"reference": reference.get(key), "input": source.get(key)}
            for key in compatibility_keys
            if source.get(key) != reference.get(key)
        }
        if differing:
            mismatches.append({"input_index": index, "differences": differing})
    if mismatches:
        raise ValueError(f"incompatible audit configurations: {mismatches}")

    evaluation_seeds = [audit["source"].get("seed") for audit in audits]
    if any(value is None for value in evaluation_seeds):
        raise ValueError("every audit must record its evaluation seed")
    if len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError(f"duplicate evaluation seeds: {evaluation_seeds}")

    summaries = {}
    for index, metric in enumerate(CORE_METRICS):
        values = [float(audit["core_metrics"][metric]) for audit in audits]
        summaries[metric] = summarize(
            values, seed=seed + index, n_bootstrap=n_bootstrap
        )

    length_summaries = {}
    for index, metric in enumerate(LENGTH_METRICS):
        values = [
            audit["length_diagnostics"].get(metric)
            for audit in audits
            if audit["length_diagnostics"].get("available")
            and audit["length_diagnostics"].get(metric) is not None
        ]
        if len(values) == len(audits):
            length_summaries[metric] = summarize(
                [float(value) for value in values],
                seed=seed + 100 + index,
                n_bootstrap=n_bootstrap,
            )

    return {
        "kind": "rollout_quality_seed_aggregate",
        "status": "complete",
        "created_at": utc_now(),
        "configuration": {key: reference.get(key) for key in compatibility_keys},
        "evaluation_seeds": evaluation_seeds,
        "n_evaluation_seeds": len(evaluation_seeds),
        "core_metrics": summaries,
        "length_metrics": length_summaries,
        "methodology": {
            "bootstrap_replicates": n_bootstrap,
            "bootstrap_seed": seed,
            "resampling_unit": "evaluation seed",
            "scope_warning": (
                "These are evaluation-seed replicates of a frozen checkpoint, "
                "not independent training-seed replicates."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--allow-fewer-than-three",
        action="store_true",
        help="permit exploratory aggregation with fewer than three seeds",
    )
    args = parser.parse_args()
    if len(args.audits) < 3 and not args.allow_fewer_than_three:
        parser.error("provide at least three audits from distinct evaluation seeds")
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    audits = [json.loads(path.read_text()) for path in args.audits]
    payload = aggregate(audits, seed=args.seed, n_bootstrap=args.bootstrap)
    payload["inputs"] = [str(path) for path in args.audits]
    output_path = args.out or RESULTS_DIR / "quality_seed_aggregate.json"
    write_result(output_path, payload)
    print(
        f"[seed-aggregate] n={payload['n_evaluation_seeds']} "
        f"pass@1={payload['core_metrics']['pass_at_1']['mean']:.4f} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
