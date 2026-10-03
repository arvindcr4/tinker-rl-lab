"""Critic foundation: GAE and value-loss helpers for a future value track.

Pure functions only (Schulman et al. 2016; Schulman et al. 2017).  The full
VAPO-style value-model track additionally needs a value head, value
pretraining, and training-loop integration — none of which exists yet.  These
helpers are the tested math foundation so that work starts from pinned
correctness, not from scratch.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def compute_gae_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    gamma: float = 1.0,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Generalized Advantage Estimation (Schulman et al. 2016, Eq. 11-12).

    ``values`` must carry one bootstrap entry (length T+1 for T rewards);
    ``dones`` marks episode terminals (no bootstrapping past them).
    Returns ``(advantages, returns)`` with ``returns = advantages +
    values[:T]``.  Length mismatches raise instead of silently shifting.
    """
    t = len(rewards)
    if len(values) != t + 1:
        raise ValueError(f"values needs one bootstrap entry: got {len(values)} for {t} rewards")
    if len(dones) != t:
        raise ValueError(f"dones must pair 1:1 with rewards, got {len(dones)} for {t}")
    advantages = [0.0] * t
    last_gae = 0.0
    for step in reversed(range(t)):
        mask = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
        last_gae = delta + gamma * lam * mask * last_gae
        advantages[step] = last_gae
    returns = [a + v for a, v in zip(advantages, values[:t])]
    return advantages, returns


def value_loss_mse(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Half-MSE value loss (PPO convention, Schulman et al. 2017 §7).

    ``0.5 * mean((V - R)^2)``.  Shapes must match exactly.
    """
    if values.shape != returns.shape:
        raise ValueError(f"shape mismatch: {tuple(values.shape)} vs {tuple(returns.shape)}")
    return 0.5 * ((values - returns) ** 2).mean()
