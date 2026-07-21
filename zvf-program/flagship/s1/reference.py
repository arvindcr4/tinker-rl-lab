"""Framework-neutral S1 reference for GRPO-family conformance."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Literal, Sequence

import torch

RTOL = 1e-6
ATOL = 1e-8

Arm = Literal["grpo", "dapo", "gspo", "drgrpo", "aero"]
Policy = Literal[
    "static_g8", "static_g16", "symmetric_zvf", "failure_only",
    "boundary_aware", "full_triage",
]
RewardStatus = Literal["observed", "noisy", "missing", "delayed"]

_ARMS = frozenset(("grpo", "dapo", "gspo", "drgrpo", "aero"))
_POLICIES = frozenset((
    "static_g8", "static_g16", "symmetric_zvf", "failure_only",
    "boundary_aware", "full_triage",
))


@dataclass(frozen=True)
class Trace:
    arm: str
    advantages: torch.Tensor
    ratios: torch.Tensor
    mask: torch.Tensor
    surrogate: torch.Tensor
    loss: torch.Tensor
    gradient: torch.Tensor
    selected_indices: tuple[int, ...]


@dataclass(frozen=True)
class PolicyDecision:
    arm: str
    action: Literal["keep", "escalate", "drop", "recheck"]
    group_size: int
    reason: str
    wilson_low: float
    wilson_high: float


def _tensor(value: torch.Tensor | Sequence[float]) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _validate(rewards: torch.Tensor, logps: torch.Tensor, old: torch.Tensor, mask: torch.Tensor) -> None:
    if rewards.ndim != 2 or logps.ndim != 2:
        raise ValueError("rewards and logps must both be rank two")
    if logps.shape != old.shape or logps.shape != mask.shape:
        raise ValueError("logps, old_logps, and mask must have identical shapes")
    if rewards.numel() != logps.shape[0]:
        raise ValueError("flattened rewards must match completion rows")
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("mask values must be in [0, 1]")
    if not torch.any(mask.sum(dim=1) > 0):
        raise ValueError("at least one completion must be active")


def _advantages(
    rewards: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    arm: Arm,
    aero_successes: Sequence[int] | None,
    aero_observations: Sequence[int] | None,
) -> torch.Tensor:
    result = torch.zeros_like(rewards)
    active = active_rows.reshape_as(rewards).bool()
    for group in range(rewards.shape[0]):
        values = rewards[group][active[group]]
        if values.numel() == 0:
            continue
        centered = values - values.mean()
        if arm == "drgrpo":
            result[group][active[group]] = centered
            continue
        std = values.std(correction=1) if values.numel() > 1 else torch.tensor(0.0, dtype=values.dtype)
        if std > ATOL:
            result[group][active[group]] = centered / std
            continue
        if arm == "aero":
            if aero_successes is None or aero_observations is None:
                raise ValueError("AERO requires successes and observations")
            if len(aero_successes) != rewards.shape[0] or len(aero_observations) != rewards.shape[0]:
                raise ValueError("AERO statistics must have one entry per group")
            successes, observations = int(aero_successes[group]), int(aero_observations[group])
            if observations <= 0 or not 0 <= successes <= observations:
                raise ValueError("invalid AERO posterior statistics")
            posterior = (successes + 1.0) / (observations + 2.0)
            scale = sqrt(posterior * (1.0 - posterior))
            result[group][active[group]] = (values - posterior) / scale
    return result.flatten()


def canonical_advantages(
    rewards: torch.Tensor,
    active_rows: torch.Tensor,
    *,
    arm: Arm,
    aero_successes: Sequence[int] | None = None,
    aero_observations: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return the frozen preprocessing used by every intended stack adapter."""
    if arm not in _ARMS:
        raise ValueError(f"unknown objective arm: {arm}")
    return _advantages(
        rewards,
        active_rows,
        arm=arm,
        aero_successes=aero_successes,
        aero_observations=aero_observations,
    )


def objective_trace(
    *,
    arm: Arm,
    rewards: torch.Tensor | Sequence[Sequence[float]],
    logps: torch.Tensor | Sequence[Sequence[float]],
    old_logps: torch.Tensor | Sequence[Sequence[float]],
    mask: torch.Tensor | Sequence[Sequence[float]],
    selected_indices: Sequence[int] | None = None,
    aero_successes: Sequence[int] | None = None,
    aero_observations: Sequence[int] | None = None,
) -> Trace:
    if arm not in _ARMS:
        raise ValueError(f"unknown objective arm: {arm}")
    rewards_t, logps_t = _tensor(rewards), _tensor(logps).detach().requires_grad_(True)
    old_t, mask_t = _tensor(old_logps), _tensor(mask)
    _validate(rewards_t, logps_t, old_t, mask_t)
    chosen = tuple(range(logps_t.shape[0])) if selected_indices is None else tuple(selected_indices)
    if any(i < 0 or i >= logps_t.shape[0] for i in chosen):
        raise ValueError("selected index out of range")
    selected = torch.zeros(logps_t.shape[0], dtype=torch.float64)
    selected[list(chosen)] = 1
    mask_t = mask_t * selected[:, None]
    active = mask_t.sum(dim=1) > 0
    if not torch.any(active):
        raise ValueError("selection must retain one active completion")
    advantages = canonical_advantages(
        rewards_t, active, arm=arm,
        aero_successes=aero_successes, aero_observations=aero_observations,
    )
    delta = logps_t - old_t
    if arm == "gspo":
        seq_delta = (delta * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp_min(1)
        ratios = seq_delta.exp()[:, None].expand_as(logps_t)
    else:
        ratios = delta.exp()
    high = 0.28 if arm == "dapo" else 0.20
    raw = ratios * advantages[:, None]
    clipped = ratios.clamp(0.80, 1.0 + high) * advantages[:, None]
    surrogate = -torch.minimum(raw, clipped)
    counts = mask_t.sum(dim=1)
    per_completion = (surrogate[active] * mask_t[active]).sum(dim=1) / counts[active]
    loss = per_completion.mean()
    gradient = torch.autograd.grad(loss, logps_t)[0].flatten()
    return Trace(arm, advantages.detach(), ratios.detach(), mask_t.detach(),
                 surrogate.detach(), loss.detach(), gradient.detach(), chosen)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("invalid Wilson counts")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def decide_policy(
    arm: Policy,
    rewards: Sequence[float],
    *,
    history_successes: int = 0,
    history_total: int = 0,
    retry_floor: float = 0.05,
    mastery_threshold: float = 0.95,
) -> PolicyDecision:
    if arm not in _POLICIES:
        raise ValueError(f"unknown policy: {arm}")
    values = [float(x) for x in rewards]
    if len(values) < 2 or any(x < 0 or x > 1 for x in values):
        raise ValueError("at least two rewards in [0,1] are required")
    successes = history_successes + sum(x >= 1 - ATOL for x in values)
    low, high = _wilson(successes, history_total + len(values))
    all_wrong = all(abs(x) <= ATOL for x in values)
    all_correct = all(abs(x - 1) <= ATOL for x in values)
    zero_variance = max(values) - min(values) <= ATOL
    if arm == "static_g8":
        action, size, reason = "keep", 8, "fixed"
    elif arm == "static_g16":
        action, size, reason = "keep", 16, "fixed"
    elif arm == "symmetric_zvf" and zero_variance:
        action, size, reason = "escalate", 16, "zero variance"
    elif arm == "failure_only" and all_wrong:
        action, size, reason = "escalate", 16, "all wrong"
    elif arm in {"boundary_aware", "full_triage"} and all_wrong:
        if arm == "boundary_aware" or high >= retry_floor:
            action, size, reason = "escalate", 16, "recoverable failure"
        else:
            action, size, reason = "drop", 0, "irrecoverable failure"
    elif arm in {"boundary_aware", "full_triage"} and all_correct:
        if arm == "boundary_aware" or low >= mastery_threshold:
            action, size, reason = "drop", 0, "mastered"
        else:
            action, size, reason = "keep", 8, "insufficient confidence"
    else:
        action, size, reason = "keep", 8, "informative"
    return PolicyDecision(arm, action, size, reason, low, high)


def decide_policy_observation(
    arm: Policy,
    rewards: Sequence[float | None],
    *,
    status: RewardStatus = "observed",
    history_successes: int = 0,
    history_total: int = 0,
) -> PolicyDecision:
    """Fail closed when a reward observation is unresolved or explicitly noisy."""
    if arm not in _POLICIES:
        raise ValueError(f"unknown policy: {arm}")
    if status not in {"observed", "noisy", "missing", "delayed"}:
        raise ValueError(f"unknown reward status: {status}")
    if status != "observed":
        return PolicyDecision(
            arm=arm,
            action="recheck",
            group_size=0,
            reason=f"{status} reward must resolve before objective evaluation",
            wilson_low=0.0,
            wilson_high=1.0,
        )
    if any(value is None for value in rewards):
        raise ValueError("observed rewards cannot contain missing values")
    return decide_policy(
        arm,
        [float(value) for value in rewards if value is not None],
        history_successes=history_successes,
        history_total=history_total,
    )


def assert_trace_close(actual: Trace, expected: Trace) -> None:
    if actual.arm != expected.arm or actual.selected_indices != expected.selected_indices:
        raise AssertionError("trace metadata differs")
    for field in ("advantages", "ratios", "mask", "surrogate", "loss", "gradient"):
        torch.testing.assert_close(getattr(actual, field), getattr(expected, field),
                                   rtol=RTOL, atol=ATOL, msg=field)
