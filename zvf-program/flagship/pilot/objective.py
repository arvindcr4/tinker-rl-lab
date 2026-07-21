from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


Condition = Literal["intended_full", "native_trl", "epsilon_only", "reduction_only"]
CONDITIONS: tuple[Condition, ...] = (
    "intended_full",
    "native_trl",
    "epsilon_only",
    "reduction_only",
)
EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28
TRL_REWARD_EPSILON = 1e-4
NUMERICAL_FLOOR = 1e-12


class ObjectiveContractError(RuntimeError):
    """An objective diagnostic is undefined or violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class ConditionTrace:
    condition: Condition
    loss: float
    advantages: tuple[float, ...]
    gradient: tuple[float, ...]
    active_rows: tuple[int, ...]
    active_tokens: int


@dataclass(frozen=True, slots=True)
class GradientDiagnostic:
    cosine: float
    relative_l2: float
    intended_gradient_norm: float
    native_gradient_norm: float


def _validate_inputs(
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    if rewards.shape != (8,):
        raise ObjectiveContractError("pilot objective requires one eight-row reward group")
    if logps.ndim != 2 or logps.shape[0] != 8:
        raise ObjectiveContractError("pilot log probabilities must have shape [8, tokens]")
    if old_logps.shape != logps.shape or completion_mask.shape != logps.shape:
        raise ObjectiveContractError("log-probability and completion-mask shapes must match")
    if active_rows.shape != (8,) or active_rows.dtype != torch.bool:
        raise ObjectiveContractError("active_rows must be an eight-element boolean tensor")
    if int(active_rows.sum()) not in {6, 8}:
        raise ObjectiveContractError("pilot groups must retain exactly six or eight rows")
    if not torch.isfinite(rewards).all() or not torch.isfinite(logps).all():
        raise ObjectiveContractError("objective inputs must be finite")
    if not torch.isfinite(old_logps).all() or not torch.isfinite(completion_mask).all():
        raise ObjectiveContractError("objective inputs must be finite")
    if torch.any(completion_mask < 0) or torch.any(completion_mask > 1):
        raise ObjectiveContractError("completion masks must be in [0, 1]")
    row_token_counts = completion_mask.sum(dim=1)
    if torch.any(row_token_counts[active_rows] <= 0):
        raise ObjectiveContractError("every active row must retain at least one token")
    if torch.any(row_token_counts[~active_rows] != 0):
        raise ObjectiveContractError("inactive rows must have an all-zero completion mask")


def _advantages(
    rewards: torch.Tensor,
    active_rows: torch.Tensor,
    condition: Condition,
) -> torch.Tensor:
    result = torch.zeros_like(rewards)
    if condition == "native_trl":
        centered = rewards - rewards.mean()
        std = rewards.std(correction=1)
        return centered / (std + TRL_REWARD_EPSILON)

    selected = rewards[active_rows]
    centered = selected - selected.mean()
    std = selected.std(correction=1)
    if std <= NUMERICAL_FLOOR:
        return result
    denominator = std + TRL_REWARD_EPSILON if condition == "epsilon_only" else std
    result[active_rows] = centered / denominator
    return result


def condition_loss(
    *,
    condition: Condition,
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    active_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the differentiable frozen DAPO loss and detached advantages."""
    if condition not in CONDITIONS:
        raise ObjectiveContractError(f"unknown pilot condition: {condition}")
    dtype = logps.dtype
    device = logps.device
    rewards = rewards.to(dtype=dtype, device=device)
    old = old_logps.to(dtype=dtype, device=device)
    mask = completion_mask.to(dtype=dtype, device=device)
    rows = active_rows.to(device=device)
    _validate_inputs(rewards, logps, old, mask, rows)

    advantages = _advantages(rewards, rows, condition).detach()
    ratios = (logps - old).exp()
    clipped = ratios.clamp(1.0 - EPSILON_LOW, 1.0 + EPSILON_HIGH)
    per_token = -torch.minimum(
        ratios * advantages[:, None],
        clipped * advantages[:, None],
    )

    if condition in {"native_trl", "reduction_only"}:
        loss = (per_token * mask).sum() / mask.sum()
    else:
        selected_loss = per_token[rows]
        selected_mask = mask[rows]
        per_completion = (selected_loss * selected_mask).sum(dim=1) / selected_mask.sum(dim=1)
        loss = per_completion.mean()
    return loss, advantages


def condition_trace(
    *,
    condition: Condition,
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    active_rows: torch.Tensor,
) -> ConditionTrace:
    rewards = rewards.to(dtype=torch.float64, device="cpu")
    current = logps.to(dtype=torch.float64, device="cpu").detach().requires_grad_(True)
    old = old_logps.to(dtype=torch.float64, device="cpu")
    mask = completion_mask.to(dtype=torch.float64, device="cpu")
    rows = active_rows.to(device="cpu")
    loss, advantages = condition_loss(
        condition=condition,
        rewards=rewards,
        logps=current,
        old_logps=old,
        completion_mask=mask,
        active_rows=rows,
    )

    gradient = torch.autograd.grad(loss, current)[0]
    if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
        raise ObjectiveContractError("objective produced a non-finite loss or gradient")
    return ConditionTrace(
        condition=condition,
        loss=float(loss.detach()),
        advantages=tuple(float(value) for value in advantages.detach()),
        gradient=tuple(float(value) for value in gradient.detach().flatten()),
        active_rows=tuple(int(index) for index in torch.where(rows)[0]),
        active_tokens=int(mask.sum()),
    )


def gradient_diagnostic(intended: ConditionTrace, native: ConditionTrace) -> GradientDiagnostic:
    if intended.condition != "intended_full" or native.condition != "native_trl":
        raise ObjectiveContractError("diagnostic requires intended_full and native_trl traces")
    intended_gradient = torch.tensor(intended.gradient, dtype=torch.float64)
    native_gradient = torch.tensor(native.gradient, dtype=torch.float64)
    intended_norm = float(torch.linalg.vector_norm(intended_gradient))
    native_norm = float(torch.linalg.vector_norm(native_gradient))
    if intended_norm <= NUMERICAL_FLOOR or native_norm <= NUMERICAL_FLOOR:
        raise ObjectiveContractError("gradient comparison is undefined for a zero-norm gradient")
    cosine = float(torch.dot(intended_gradient, native_gradient) / (intended_norm * native_norm))
    relative_l2 = float(
        torch.linalg.vector_norm(intended_gradient - native_gradient) / intended_norm
    )
    if not math.isfinite(cosine) or not math.isfinite(relative_l2):
        raise ObjectiveContractError("gradient diagnostic is non-finite")
    return GradientDiagnostic(
        cosine=cosine,
        relative_l2=relative_l2,
        intended_gradient_norm=intended_norm,
        native_gradient_norm=native_norm,
    )
