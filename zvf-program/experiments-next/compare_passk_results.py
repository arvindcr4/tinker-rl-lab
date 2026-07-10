#!/usr/bin/env python3
"""Paired base-vs-post-RL pass@k comparison with clustered intervals.

New pass@k results include ordered prompt fingerprints.  The comparison
refuses unverified pairing unless the caller explicitly opts into legacy mode.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from common import RESULTS_DIR, utc_now, write_result
from passk_eval import pass_at_k


COMPATIBILITY_KEYS = (
    "model",
    "split",
    "seed",
    "n_problems",
    "n_per_problem",
    "ks",
    "temperature",
    "top_p",
    "max_tokens",
    "max_prompt_tokens",
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


def compare(
    base: dict,
    post: dict,
    *,
    n_bootstrap: int,
    seed: int,
    allow_legacy_unverified_pairing: bool = False,
) -> dict:
    for label, result in (("base", base), ("post", post)):
        if result.get("kind") != "passk_eval" or result.get("status") != "complete":
            raise ValueError(f"{label} input is not a completed passk_eval result")
    if base.get("which") != "base" or post.get("which") != "postrl":
        raise ValueError("expected a base result and a postrl result, in that order")

    differences = {
        key: {"base": base.get(key), "post": post.get(key)}
        for key in COMPATIBILITY_KEYS
        if base.get(key) != post.get(key)
    }
    if differences:
        raise ValueError(f"base/post configurations differ: {differences}")

    base_counts = base.get("per_problem_c") or []
    post_counts = post.get("per_problem_c") or []
    if not base_counts or len(base_counts) != len(post_counts):
        raise ValueError("base/post per-problem count arrays are missing or misaligned")

    base_fingerprints = base.get("prompt_fingerprints")
    post_fingerprints = post.get("prompt_fingerprints")
    pairing_verified = bool(base_fingerprints and post_fingerprints)
    if pairing_verified and base_fingerprints != post_fingerprints:
        raise ValueError("base/post prompt fingerprints differ")
    if not pairing_verified and not allow_legacy_unverified_pairing:
        raise ValueError(
            "prompt fingerprints are missing; pass explicit legacy override only "
            "if deterministic prompt ordering was independently verified"
        )

    n = int(base["n_per_problem"])
    ks = [int(k) for k in base["ks"]]
    rng = random.Random(seed)
    metrics = {}
    for k in ks:
        base_values = [pass_at_k(n, count, k) for count in base_counts]
        post_values = [pass_at_k(n, count, k) for count in post_counts]
        paired_deltas = [
            post_value - base_value
            for base_value, post_value in zip(base_values, post_values)
        ]
        bootstrap_deltas = []
        for _ in range(n_bootstrap):
            indices = [rng.randrange(len(paired_deltas)) for _ in paired_deltas]
            bootstrap_deltas.append(
                sum(paired_deltas[index] for index in indices) / len(indices)
            )
        bootstrap_deltas.sort()
        metrics[str(k)] = {
            "base": sum(base_values) / len(base_values),
            "post_rl": sum(post_values) / len(post_values),
            "paired_delta": sum(paired_deltas) / len(paired_deltas),
            "paired_delta_ci_low": _percentile(bootstrap_deltas, 0.025),
            "paired_delta_ci_high": _percentile(bootstrap_deltas, 0.975),
            "bootstrap_replicates": n_bootstrap,
            "resampling_unit": "paired problem",
        }

    return {
        "kind": "paired_passk_comparison",
        "status": "complete",
        "created_at": utc_now(),
        "pairing_verified_by_prompt_fingerprint": pairing_verified,
        "configuration": {key: base.get(key) for key in COMPATIBILITY_KEYS},
        "metrics": metrics,
        "interpretation": {
            "pass_at_1_only_gain": "primarily distribution sharpening",
            "pass_at_max_k_gain": "evidence consistent with capability-frontier expansion",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--allow-legacy-unverified-pairing", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    base = json.loads(args.base.read_text())
    post = json.loads(args.post.read_text())
    payload = compare(
        base,
        post,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        allow_legacy_unverified_pairing=args.allow_legacy_unverified_pairing,
    )
    payload["base_path"] = str(args.base)
    payload["post_path"] = str(args.post)
    output_path = args.out or RESULTS_DIR / f"paired_{base.get('tag')}_vs_{post.get('tag')}.json"
    write_result(output_path, payload)
    deltas = {k: round(v["paired_delta"], 4) for k, v in payload["metrics"].items()}
    print(f"[paired-passk] deltas={deltas} -> {output_path}")


if __name__ == "__main__":
    main()
