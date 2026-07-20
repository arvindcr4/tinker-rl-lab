"""Reference PAM/GSR/EGM metrics for GRPO, PPO, and SAO.

The implementation is framework-independent and intentionally operates on
ordinary iterables. Trainer integrations can convert detached tensors to lists
or reproduce the same formulas natively on device.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def ppo_gate(ratio: float, advantage: float, epsilon: float) -> int:
    """PPO's sign-dependent clipped-surrogate survival gate.

    Clip-boundary equality survives: the clipped and unclipped branches agree
    there. Zero advantage also returns one because absence of credit is captured
    by PAM, not mislabeled as transport clipping.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if advantage > 0 and ratio > 1.0 + epsilon:
        return 0
    if advantage < 0 and ratio < 1.0 - epsilon:
        return 0
    return 1


def sao_gate(ratio: float, epsilon_low: float, epsilon_high: float) -> int:
    """SAO direct double-sided importance-sampling gate (strict interval)."""
    if epsilon_low < 0 or epsilon_high < 0:
        raise ValueError("SAO epsilons must be non-negative")
    return int(1.0 - epsilon_low < ratio < 1.0 + epsilon_high)


def _finite(values: Iterable[float], name: str) -> list[float]:
    output = [float(value) for value in values]
    if any(not math.isfinite(value) for value in output):
        raise ValueError(f"{name} contains a non-finite value")
    return output


def signal_metrics(
    ratios: Iterable[float],
    advantages: Iterable[float],
    gates: Iterable[int | bool],
    action_mask: Iterable[int | bool] | None = None,
) -> dict[str, float | int | bool]:
    """Compute Potential Advantage Mass, Gradient Survival Ratio, and EGM."""
    ratio_values = _finite(ratios, "ratios")
    advantage_values = _finite(advantages, "advantages")
    gate_values = [int(value) for value in gates]
    if not (len(ratio_values) == len(advantage_values) == len(gate_values)):
        raise ValueError("ratios, advantages, and gates must have equal length")
    if any(value not in (0, 1) for value in gate_values):
        raise ValueError("gates must contain only zero or one")
    if action_mask is None:
        mask_values = [1] * len(ratio_values)
    else:
        mask_values = [int(value) for value in action_mask]
        if len(mask_values) != len(ratio_values):
            raise ValueError("action_mask must have the same length as ratios")
        if any(value not in (0, 1) for value in mask_values):
            raise ValueError("action_mask must contain only zero or one")

    coefficients = [
        (ratio * advantage) ** 2
        for ratio, advantage, is_action in zip(ratio_values, advantage_values, mask_values)
        if is_action
    ]
    surviving = [
        gate * (ratio * advantage) ** 2
        for ratio, advantage, gate, is_action in zip(
            ratio_values, advantage_values, gate_values, mask_values
        )
        if is_action
    ]
    n = len(coefficients)
    if n == 0:
        raise ValueError("at least one action token is required")
    potential_sum = math.fsum(coefficients)
    surviving_sum = math.fsum(surviving)
    pam = potential_sum / n
    gsr = surviving_sum / potential_sum if potential_sum > 0 else 0.0
    egm = surviving_sum / n
    return {
        "n_action_tokens": n,
        "pam": pam,
        "gsr": gsr,
        "egm": egm,
        "exact_zero_update": egm == 0.0,
    }


def root_metrics(
    token_records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | bool]]:
    """Aggregate token records by root trajectory, independent of chunking."""
    roots: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in token_records:
        root_id = record.get("root_trajectory_id")
        if root_id is None:
            raise ValueError("token record is missing root_trajectory_id")
        roots[str(root_id)].append(record)
    output: dict[str, dict[str, float | int | bool]] = {}
    for root_id, records in roots.items():
        output[root_id] = signal_metrics(
            (record["ratio"] for record in records),
            (record["advantage"] for record in records),
            (record["gate"] for record in records),
            (record.get("is_action_token", True) for record in records),
        )
    return output
