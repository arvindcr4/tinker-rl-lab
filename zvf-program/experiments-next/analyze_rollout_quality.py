#!/usr/bin/env python3
"""Audit a rollout pool for GRPO signal quality and length confounding.

The unit of resampling is the prompt, not the rollout.  This preserves the
within-prompt dependence induced by group sampling and avoids artificially
narrow confidence intervals.

Metrics:
  * zero-variance prompt rate (all rewards in a group are identical)
  * all-correct / all-incorrect prompt rates
  * mean group reward variance and active-advantage fraction
  * prompt-clustered bootstrap confidence intervals
  * reward/length correlation, correct-vs-incorrect token gap, length AUC,
    and accuracy by equal-count length quartile when token counts are present
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Callable, Iterable

from common import RESULTS_DIR, utc_now, write_result


EPS = 1e-12


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x, mean_y = _mean(xs), _mean(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom = math.sqrt(
        sum(x * x for x in centered_x) * sum(y * y for y in centered_y)
    )
    if denom <= EPS:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denom


def _binary_auc(scores: list[float], labels: list[float]) -> float | None:
    """Mann-Whitney AUC with average ranks for tied lengths."""
    if len(scores) != len(labels) or not scores:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_positive = sum(1 for _, label in pairs if label > 0.5)
    n_negative = len(pairs) - n_positive
    if not n_positive or not n_negative:
        return None

    positive_rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(
            1 for _, label in pairs[index:end] if label > 0.5
        )
        index = end
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)


def validate_prompts(pool: dict) -> list[dict]:
    prompts = pool.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("pool has no prompt rows")
    expected_rollouts = pool.get("rollouts_per_prompt")
    for position, row in enumerate(prompts):
        rewards = row.get("rewards")
        if not isinstance(rewards, list) or len(rewards) < 2:
            raise ValueError(f"prompt {position} must contain at least two rewards")
        if expected_rollouts is not None and len(rewards) != expected_rollouts:
            raise ValueError(
                f"prompt {position} has {len(rewards)} rewards, expected "
                f"{expected_rollouts}"
            )
        if any(not isinstance(reward, (int, float)) for reward in rewards):
            raise ValueError(f"prompt {position} has a non-numeric reward")
        token_counts = row.get("token_counts")
        if token_counts is not None and len(token_counts) != len(rewards):
            raise ValueError(
                f"prompt {position} token_counts/rewards length mismatch"
            )
    return prompts


def core_metrics(prompts: list[dict]) -> dict[str, float | int]:
    prompt_means: list[float] = []
    group_variances: list[float] = []
    zero_variance = all_correct = all_incorrect = 0
    active_advantages = total_advantages = 0

    for row in prompts:
        rewards = [float(reward) for reward in row["rewards"]]
        group_mean = _mean(rewards)
        variance = _mean((reward - group_mean) ** 2 for reward in rewards)
        prompt_means.append(group_mean)
        group_variances.append(variance)
        is_zero = variance <= EPS
        zero_variance += int(is_zero)
        all_correct += int(all(reward > 0.5 for reward in rewards))
        all_incorrect += int(all(reward <= 0.5 for reward in rewards))
        total_advantages += len(rewards)
        if not is_zero:
            active_advantages += sum(
                1 for reward in rewards if abs(reward - group_mean) > EPS
            )

    n_prompts = len(prompts)
    return {
        "n_prompts": n_prompts,
        "n_rollouts": total_advantages,
        "pass_at_1": _mean(prompt_means),
        "zero_variance_prompt_rate": zero_variance / n_prompts,
        "informative_prompt_rate": 1.0 - zero_variance / n_prompts,
        "all_correct_prompt_rate": all_correct / n_prompts,
        "all_incorrect_prompt_rate": all_incorrect / n_prompts,
        "mean_group_reward_variance": _mean(group_variances),
        "active_advantage_fraction": active_advantages / total_advantages,
    }


def clustered_bootstrap(
    prompts: list[dict],
    metric: Callable[[list[dict]], float],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float | int]:
    point = metric(prompts)
    rng = random.Random(seed)
    samples = []
    for _ in range(n_bootstrap):
        resampled = [prompts[rng.randrange(len(prompts))] for _ in prompts]
        value = metric(resampled)
        if math.isfinite(value):
            samples.append(value)
    samples.sort()
    return {
        "point": point,
        "ci_low": _percentile(samples, 0.025),
        "ci_high": _percentile(samples, 0.975),
        "bootstrap_replicates": len(samples),
        "resampling_unit": "prompt",
    }


def length_diagnostics(prompts: list[dict]) -> dict:
    covered = [row for row in prompts if row.get("token_counts") is not None]
    if not covered:
        return {
            "available": False,
            "prompt_coverage": 0.0,
            "reason": "pool rows lack token_counts; collect with instrumented build_pool.py",
        }

    lengths: list[float] = []
    rewards: list[float] = []
    for row in covered:
        lengths.extend(float(value) for value in row["token_counts"])
        rewards.extend(float(value) for value in row["rewards"])

    correct_lengths = [length for length, reward in zip(lengths, rewards) if reward > 0.5]
    incorrect_lengths = [length for length, reward in zip(lengths, rewards) if reward <= 0.5]
    ranked = sorted(zip(lengths, rewards), key=lambda item: item[0])
    quartiles = []
    for quartile in range(4):
        bucket = [
            pair for index, pair in enumerate(ranked)
            if min(3, 4 * index // len(ranked)) == quartile
        ]
        if not bucket:
            continue
        quartiles.append({
            "quartile": quartile + 1,
            "n": len(bucket),
            "min_tokens": min(length for length, _ in bucket),
            "max_tokens": max(length for length, _ in bucket),
            "accuracy": _mean(reward for _, reward in bucket),
        })

    return {
        "available": True,
        "prompt_coverage": len(covered) / len(prompts),
        "n_rollouts_with_lengths": len(lengths),
        "mean_output_tokens": _mean(lengths),
        "mean_tokens_correct": _mean(correct_lengths),
        "mean_tokens_incorrect": _mean(incorrect_lengths),
        "correct_minus_incorrect_tokens": (
            _mean(correct_lengths) - _mean(incorrect_lengths)
            if correct_lengths and incorrect_lengths else None
        ),
        "reward_length_correlation": _pearson(lengths, rewards),
        "length_predictive_auc": _binary_auc(lengths, rewards),
        "length_stratified_macro_accuracy": _mean(
            row["accuracy"] for row in quartiles
        ),
        "accuracy_by_length_quartile": quartiles,
    }


def analyze(pool: dict, *, n_bootstrap: int, seed: int) -> dict:
    prompts = validate_prompts(pool)
    core = core_metrics(prompts)
    metric_names = (
        "pass_at_1",
        "zero_variance_prompt_rate",
        "mean_group_reward_variance",
        "active_advantage_fraction",
    )
    intervals = {
        name: clustered_bootstrap(
            prompts,
            lambda rows, metric_name=name: float(core_metrics(rows)[metric_name]),
            n_bootstrap=n_bootstrap,
            seed=seed + index,
        )
        for index, name in enumerate(metric_names)
    }
    return {
        "kind": "rollout_quality_audit",
        "status": "complete",
        "created_at": utc_now(),
        "source": {
            "tag": pool.get("tag"),
            "model": pool.get("model"),
            "split": pool.get("split"),
            "seed": pool.get("seed"),
            "temperature": pool.get("temperature"),
            "top_p": pool.get("top_p"),
            "rollouts_per_prompt": pool.get("rollouts_per_prompt"),
            "n_prompts": pool.get("n_prompts"),
            "max_tokens": pool.get("max_tokens"),
            "checkpoint_identity": (
                f"base:{pool.get('model')}"
                if "requested_sampler_path" in pool
                and pool.get("requested_sampler_path") is None
                else pool.get("requested_sampler_path") or pool.get("sampler_path")
            ),
        },
        "core_metrics": core,
        "prompt_clustered_bootstrap_95_ci": intervals,
        "length_diagnostics": length_diagnostics(prompts),
        "methodology": {
            "bootstrap_seed": seed,
            "bootstrap_replicates": n_bootstrap,
            "zero_variance_definition": "population variance <= 1e-12 within prompt",
            "length_auc_interpretation": (
                "0.5 means length does not rank correct above incorrect responses"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="analyze a started pool explicitly; output records the partial status",
    )
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    pool = json.loads(args.pool.read_text())
    if pool.get("status") != "complete" and not args.allow_partial:
        raise SystemExit(
            f"pool status is {pool.get('status')!r}; pass --allow-partial explicitly"
        )
    payload = analyze(pool, n_bootstrap=args.bootstrap, seed=args.seed)
    payload["source"]["path"] = str(args.pool)
    payload["source"]["input_status"] = pool.get("status")
    output_path = args.out or RESULTS_DIR / f"quality_{args.pool.stem}.json"
    write_result(output_path, payload)
    core = payload["core_metrics"]
    print(
        f"[quality] pass@1={core['pass_at_1']:.4f} "
        f"zero_variance={core['zero_variance_prompt_rate']:.4f} "
        f"active_advantage={core['active_advantage_fraction']:.4f} "
        f"-> {output_path}"
    )


if __name__ == "__main__":
    main()
