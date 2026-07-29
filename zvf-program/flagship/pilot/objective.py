from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from pilot.entropic_gating import compute_entropy_density, eliminate_noise_components
from pilot.spectral_attention import legendre_spectral_projection, spectral_pairwise_distance


Condition = Literal[
    "intended_full",
    "native_trl",
    "epsilon_only",
    "reduction_only",
    "spectral_legendre",
    "entropic_givens",
]
CONDITIONS: tuple[Condition, ...] = (
    "intended_full",
    "native_trl",
    "epsilon_only",
    "reduction_only",
    "spectral_legendre",
    "entropic_givens",
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
    relation: str
    cosine: float | None
    relative_l2: float | None
    intended_gradient_norm: float
    native_gradient_norm: float


def _validate_inputs(
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    active_rows: torch.Tensor,
) -> None:
    if rewards.ndim != 1:
        raise ObjectiveContractError("pilot objective requires a 1D reward tensor")
    g_size = rewards.shape[0]
    if g_size < 1:
        raise ObjectiveContractError("pilot objective group size must be positive")
    if logps.ndim != 2 or logps.shape[0] != g_size:
        raise ObjectiveContractError(f"pilot log probabilities must have shape [{g_size}, tokens]")
    if old_logps.shape != logps.shape or completion_mask.shape != logps.shape:
        raise ObjectiveContractError("log-probability and completion-mask shapes must match")
    if active_rows.shape != (g_size,) or active_rows.dtype != torch.bool:
        raise ObjectiveContractError(f"active_rows must be a {g_size}-element boolean tensor")
    if g_size == 8 and int(active_rows.sum()) not in {6, 8}:
        raise ObjectiveContractError("pilot groups must retain exactly six or eight rows")
    if int(active_rows.sum()) < 1:
        raise ObjectiveContractError("pilot groups must retain at least one active row")
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


def spectral_legendre_grpo(
    *,
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    active_rows: torch.Tensor,
    n_modes: int = 8,
    lambda_entropy: float = 0.1,
    eps_spec: float = 1e-4,
    hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO objective augmented with Legendre polynomial spectral trajectory advantage."""
    dtype = logps.dtype
    device = logps.device
    rewards = rewards.to(dtype=dtype, device=device)
    old = old_logps.to(dtype=dtype, device=device)
    mask = completion_mask.to(dtype=dtype, device=device)
    rows = active_rows.to(device=device)
    _validate_inputs(rewards, logps, old, mask, rows)

    g_size = rewards.shape[0]
    base_advantages = _advantages(rewards, rows, "intended_full")

    if hidden_states is not None:
        traj_features = hidden_states.to(dtype=dtype, device=device)
    else:
        tokens = logps.shape[1]
        pos = torch.linspace(-1.0, 1.0, steps=tokens, dtype=dtype, device=device).unsqueeze(0).expand(g_size, tokens)
        traj_features = torch.stack([logps, pos], dim=-1)

    spectral_coeffs = legendre_spectral_projection(traj_features, n_modes, mask=mask)

    active_indices = torch.where(rows)[0]
    num_active = len(active_indices)
    s_scores = torch.zeros(g_size, dtype=dtype, device=device)
    for i_idx in active_indices:
        i = int(i_idx)
        c_i = spectral_coeffs[i]
        d_sum = torch.tensor(0.0, dtype=dtype, device=device)
        for j_idx in active_indices:
            j = int(j_idx)
            if i != j:
                c_j = spectral_coeffs[j]
                d_sum = d_sum + spectral_pairwise_distance(c_i, c_j)
        s_scores[i] = d_sum / max(num_active - 1, 1)

    selected_s = s_scores[rows]
    std_s = selected_s.std(correction=1)
    spectral_advantages = torch.zeros_like(rewards)
    if std_s > NUMERICAL_FLOOR:
        centered_s = selected_s - selected_s.mean()
        spectral_advantages[rows] = centered_s / (std_s + eps_spec)

    combined_advantages = (base_advantages + lambda_entropy * spectral_advantages).detach()

    ratios = (logps - old).exp()
    clipped = ratios.clamp(1.0 - EPSILON_LOW, 1.0 + EPSILON_HIGH)
    per_token = -torch.minimum(
        ratios * combined_advantages[:, None],
        clipped * combined_advantages[:, None],
    )

    selected_loss = per_token[rows]
    selected_mask = mask[rows]
    per_completion = (selected_loss * selected_mask).sum(dim=1) / selected_mask.sum(dim=1)
    loss = per_completion.mean()
    return loss, combined_advantages


def entropic_givens_grpo(
    *,
    rewards: torch.Tensor,
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    active_rows: torch.Tensor,
    tau_entropy: float = 0.5,
    lambda_entropy: float = 0.1,
    eps_spec: float = 1e-4,
    n_modes: int = 8,
    n_noise_dims: int = 1,
    hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GRPO objective augmented with Quantum-Inspired Givens Entropic Gating trajectory advantage."""
    dtype = logps.dtype
    device = logps.device
    rewards = rewards.to(dtype=dtype, device=device)
    old = old_logps.to(dtype=dtype, device=device)
    mask = completion_mask.to(dtype=dtype, device=device)
    rows = active_rows.to(device=device)
    _validate_inputs(rewards, logps, old, mask, rows)

    g_size = rewards.shape[0]
    base_advantages = _advantages(rewards, rows, "intended_full")

    if hidden_states is not None:
        traj_features = hidden_states.to(dtype=dtype, device=device)
    else:
        tokens = logps.shape[1]
        pos = torch.linspace(-1.0, 1.0, steps=tokens, dtype=dtype, device=device).unsqueeze(0).expand(g_size, tokens)
        traj_features = torch.stack([logps, pos], dim=-1)

    gated_features = traj_features.clone()
    active_indices = torch.where(rows)[0]
    for i_idx in active_indices:
        i = int(i_idx)
        prob_i = F.softmax(logps[i], dim=-1)
        h_i = compute_entropy_density(prob_i, dim=-1)
        if float(h_i.detach()) < tau_entropy:
            row_feat_proj, _ = eliminate_noise_components(gated_features[i], n_noise_dims=n_noise_dims)
            gated_features[i] = row_feat_proj

    spectral_coeffs = legendre_spectral_projection(gated_features, n_modes, mask=mask)

    num_active = len(active_indices)
    s_scores = torch.zeros(g_size, dtype=dtype, device=device)
    for i_idx in active_indices:
        i = int(i_idx)
        c_i = spectral_coeffs[i]
        d_sum = torch.tensor(0.0, dtype=dtype, device=device)
        for j_idx in active_indices:
            j = int(j_idx)
            if i != j:
                c_j = spectral_coeffs[j]
                d_sum = d_sum + spectral_pairwise_distance(c_i, c_j)
        s_scores[i] = d_sum / max(num_active - 1, 1)

    selected_s = s_scores[rows]
    std_s = selected_s.std(correction=1)
    givens_advantages = torch.zeros_like(rewards)
    if std_s > NUMERICAL_FLOOR:
        centered_s = selected_s - selected_s.mean()
        givens_advantages[rows] = centered_s / (std_s + eps_spec)

    combined_advantages = (base_advantages + lambda_entropy * givens_advantages).detach()

    ratios = (logps - old).exp()
    clipped = ratios.clamp(1.0 - EPSILON_LOW, 1.0 + EPSILON_HIGH)
    per_token = -torch.minimum(
        ratios * combined_advantages[:, None],
        clipped * combined_advantages[:, None],
    )

    selected_loss = per_token[rows]
    selected_mask = mask[rows]
    per_completion = (selected_loss * selected_mask).sum(dim=1) / selected_mask.sum(dim=1)
    loss = per_completion.mean()
    return loss, combined_advantages


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
    
    if condition == "spectral_legendre":
        return spectral_legendre_grpo(
            rewards=rewards,
            logps=logps,
            old_logps=old_logps,
            completion_mask=completion_mask,
            active_rows=active_rows,
        )
    if condition == "entropic_givens":
        return entropic_givens_grpo(
            rewards=rewards,
            logps=logps,
            old_logps=old_logps,
            completion_mask=completion_mask,
            active_rows=active_rows,
        )

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
    if not math.isfinite(intended_norm) or not math.isfinite(native_norm):
        raise ObjectiveContractError("gradient norm is non-finite")
    if intended_norm == 0.0 or native_norm == 0.0:
        if intended_norm == 0.0 and native_norm == 0.0:
            relation = "joint_zero"
        elif intended_norm == 0.0:
            relation = "intended_zero"
        else:
            relation = "native_zero"
        return GradientDiagnostic(
            relation=relation,
            cosine=None,
            relative_l2=None,
            intended_gradient_norm=intended_norm,
            native_gradient_norm=native_norm,
        )
    cosine = float(torch.dot(intended_gradient, native_gradient) / (intended_norm * native_norm))
    relative_l2 = float(
        torch.linalg.vector_norm(intended_gradient - native_gradient) / intended_norm
    )
    if not math.isfinite(cosine) or not math.isfinite(relative_l2):
        raise ObjectiveContractError("gradient diagnostic is non-finite")
    return GradientDiagnostic(
        relation="nonzero",
        cosine=cosine,
        relative_l2=relative_l2,
        intended_gradient_norm=intended_norm,
        native_gradient_norm=native_norm,
    )
