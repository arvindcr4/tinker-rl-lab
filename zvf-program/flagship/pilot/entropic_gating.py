from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropicGatingError(RuntimeError):
    """Raised when Givens entropic gating operations violate invariants."""


def compute_entropy_density(prob: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Compute Shannon entropic density H(p) = - sum(p * log(p + eps))."""
    if not torch.isfinite(prob).all():
        raise EntropicGatingError("probability tensor contains non-finite values")
    if torch.any(prob < 0):
        raise EntropicGatingError("probabilities must be non-negative")
    prob_safe = prob.clamp(min=0.0)
    entropy = -torch.sum(prob_safe * torch.log(prob_safe + eps), dim=dim)
    return entropy


def compute_attention_entropy(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Compute Shannon entropic density from attention logits."""
    if not torch.isfinite(logits).all():
        raise EntropicGatingError("logits tensor contains non-finite values")
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
    prob = F.softmax(logits, dim=dim)
    return compute_entropy_density(prob, dim=dim)


def givens_rotation_angle(v_i: torch.Tensor, v_j: torch.Tensor) -> torch.Tensor:
    """Compute Givens rotation angle theta = atan2(v_j, v_i)."""
    return torch.atan2(v_j, v_i)


def apply_givens_rotation_pair(
    x: torch.Tensor,
    i: int,
    j: int,
    theta: torch.Tensor,
) -> torch.Tensor:
    """Apply planar Givens unitary rotation on coordinate axes i and j along last dimension.

    G(i, j, theta) maps v_i -> cos(theta)*v_i + sin(theta)*v_j
                      v_j -> -sin(theta)*v_i + cos(theta)*v_j
    Exact vector norm is preserved: ||G v||_2 = ||v||_2.
    """
    d = x.shape[-1]
    if i < 0 or i >= d or j < 0 or j >= d or i == j:
        raise EntropicGatingError(f"invalid Givens rotation indices i={i}, j={j} for dimension {d}")

    cos = torch.cos(theta)
    sin = torch.sin(theta)

    x_out = x.clone()
    xi = x[..., i]
    xj = x[..., j]

    x_out[..., i] = cos * xi + sin * xj
    x_out[..., j] = -sin * xi + cos * xj
    return x_out


def eliminate_noise_components(
    x: torch.Tensor,
    n_noise_dims: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute Givens unitary rotations to rotate low-entropy noise components to designated noise axes,
    followed by hard projection to zero.

    Returns (x_projected, x_rotated).
    """
    d = x.shape[-1]
    if n_noise_dims <= 0 or n_noise_dims >= d:
        raise EntropicGatingError(f"n_noise_dims must be in [1, d-1], got {n_noise_dims} for d={d}")

    x_rotated = x.clone()
    for k in range(n_noise_dims):
        j = d - 1 - k  # Noise coordinate to zero out
        i = j - n_noise_dims  # Target signal coordinate to accumulate into
        if i < 0:
            i = 0
        v_i = x_rotated[..., i]
        v_j = x_rotated[..., j]
        theta = givens_rotation_angle(v_i, v_j)
        x_rotated = apply_givens_rotation_pair(x_rotated, i, j, theta)

    x_projected = x_rotated.clone()
    x_projected[..., (d - n_noise_dims) :] = 0.0
    return x_projected, x_rotated


class GivensEntropyGate(nn.Module):
    """Quantum-inspired entropic gating module using Givens unitary rotations."""

    def __init__(self, tau_entropy: float = 0.5, n_noise_dims: int = 1) -> None:
        super().__init__()
        if tau_entropy < 0:
            raise EntropicGatingError("tau_entropy must be non-negative")
        if n_noise_dims <= 0:
            raise EntropicGatingError("n_noise_dims must be positive")
        self.tau_entropy = tau_entropy
        self.n_noise_dims = n_noise_dims

    def forward(
        self,
        x: torch.Tensor,
        attention_logits: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim < 2:
            raise EntropicGatingError(f"input tensor x must have at least 2 dimensions, got ndim={x.ndim}")

        if attention_logits is not None:
            h = compute_attention_entropy(attention_logits, mask=mask, dim=-1)
        else:
            prob = F.softmax(x, dim=-1)
            h = compute_entropy_density(prob, dim=-1)

        mean_entropy = float(h.mean())
        if mean_entropy < self.tau_entropy:
            x_proj, _ = eliminate_noise_components(x, n_noise_dims=self.n_noise_dims)
            return x_proj
        return x
