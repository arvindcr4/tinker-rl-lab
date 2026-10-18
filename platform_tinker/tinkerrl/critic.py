"""Prompt-conditioned value baseline (critic) + GAE/value-loss helpers.

The Tinker closure only sees sampled-token logprobs, so no backbone value
head is reachable through this interface.  The critic instead learns
``E[reward | prompt]`` from (prompt, reward) pairs with a tiny CPU model
(hash-embedding + MLP, ~70K params), used as a per-prompt baseline:
advantages are ``R - V(x)`` normalized batch-wide (PPO-style).  GAE and the
value loss below (Schulman et al. 2016; Schulman et al. 2017) are the
trajectory-level counterparts, kept for future token-level use.
"""

from __future__ import annotations

from copy import deepcopy
import math
import statistics
from collections.abc import Mapping
from typing import Any, List, Sequence, Tuple

import torch


class PromptValueCritic(torch.nn.Module):
    """Tiny CPU value head: ``V(x)`` from prompt token ids.

    Mean-pooled hash embedding over prompt token ids plus a normalized
    prompt-length scalar, through a 2-layer Tanh MLP to one value.
    Deterministic under ``torch.manual_seed`` (the training loop seeds
    before constructing it).  Runs wherever its parameters live (CPU in
    the training loop); scratch tensors follow the embedding device so a
    moved module does not crash on a device mismatch.
    """

    def __init__(
        self,
        vocab_buckets: int = 4096,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        max_prompt_tokens: int = 1024,
    ) -> None:
        super().__init__()
        self.vocab_buckets = vocab_buckets
        self.max_prompt_tokens = max_prompt_tokens
        self.embedding = torch.nn.EmbeddingBag(vocab_buckets, embed_dim, mode="mean")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim + 1, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_ids: Sequence[Sequence[int]]) -> torch.Tensor:
        """Return one value per prompt as a ``(B,)`` tensor.

        An empty batch returns a length-0 tensor.  An empty prompt is a
        zero-length bag (dummy bucket 0) with a length feature of 0.
        """
        device = self.embedding.weight.device
        if len(batch_ids) == 0:
            return torch.zeros(0, device=device)
        flats: List[int] = []
        offsets = [0]
        lengths = []
        for ids in batch_ids:
            clipped = [i % self.vocab_buckets for i in list(ids)[: self.max_prompt_tokens]]
            flats.extend(clipped or [0])
            offsets.append(len(flats))
            lengths.append(len(clipped) / self.max_prompt_tokens)
        indices = torch.tensor(flats, dtype=torch.long, device=device)
        offsets_t = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        pooled = self.embedding(indices, offsets_t)
        length_feat = torch.tensor(lengths, dtype=torch.float32, device=device).unsqueeze(1)
        return self.mlp(torch.cat([pooled, length_feat], dim=1)).squeeze(1)


def require_finite_critic_state(*states: Any) -> None:
    """Reject nonfinite tensors/scalars without modifying a nested state tree."""
    pending = list(states)
    visited = set()
    while pending:
        value = pending.pop()
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError("critic state must be finite")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("critic state must be finite")
        elif isinstance(value, (Mapping, list, tuple)) and id(value) not in visited:
            visited.add(id(value))
            pending.extend(value.values() if isinstance(value, Mapping) else value)


def restore_critic_state(
    critic: PromptValueCritic, optimizer: torch.optim.Adam, payload: Mapping[str, Any]
) -> None:
    """Restore only state compatible with a fresh configured critic/Adam pair.

    Hash verification is the caller's responsibility. These checks bind shapes,
    parameter order and optimizer options, including values after dtype casting.
    Rejection can modify this local pair; no external work should precede it.
    """
    error = "critic restore state does not match the configured model and Adam contract"
    expected_groups = deepcopy(optimizer.state_dict()["param_groups"])
    try:
        saved_groups = payload["optimizer"]["param_groups"]
        if not isinstance(saved_groups, list) or len(saved_groups) != len(expected_groups):
            raise ValueError(error)
        for saved, expected in zip(saved_groups, expected_groups, strict=True):
            ids = saved["params"]
            if (
                not isinstance(ids, list)
                or any(type(value) is not int for value in ids)
                or ids != expected["params"]
            ):
                raise ValueError(error)
        # The configured critic is real-valued. PyTorch otherwise silently
        # discards imaginary components when casting complex checkpoint tensors.
        tensors = list(payload["model"].values())
        tensors.extend(
            value for state in payload["optimizer"]["state"].values() for value in state.values()
        )
        if any(isinstance(value, torch.Tensor) and value.is_complex() for value in tensors):
            raise ValueError(error)
        critic.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        # Loading can cast finite float64 checkpoint values to float32 infinity.
        require_finite_critic_state(critic.state_dict(), optimizer.state_dict())
        if optimizer.state_dict()["param_groups"] != expected_groups:
            raise ValueError(error)
        parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        if not set(optimizer.state).issubset(set(parameters)):
            raise ValueError(error)
        for parameter in parameters:
            state = optimizer.state.get(parameter, {})
            if not state:
                continue  # Adam legitimately has no moments before its first update.
            if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(error)
            step = state["step"]
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.dtype == torch.bool:
                raise ValueError(error)
            count = float(step.item())
            if count < 0 or not count.is_integer():
                raise ValueError(error)
            for key in ("exp_avg", "exp_avg_sq"):
                moment = state[key]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                    or moment.device != parameter.device
                ):
                    raise ValueError(error)
            if (state["exp_avg_sq"] < 0).any():
                raise ValueError(error)
    except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
        raise ValueError(error) from exc


def train_critic_step(
    critic: PromptValueCritic,
    optimizer: torch.optim.Optimizer,
    batch_ids: Sequence[Sequence[int]],
    targets: Sequence[float],
    steps: int = 1,
) -> float:
    """Fit the critic toward Monte-Carlo returns; return mean half-MSE loss.

    Empty batches are a no-op returning ``0.0`` (degenerate steps still
    reach here when every group is filtered). Nonfinite targets, loss or
    gradients raise before that critic optimizer update; earlier iterations
    are not rolled back. Model/optimizer state is checked before training and
    after each update; post-update detection raises without rollback.
    """
    if not batch_ids or steps <= 0:
        return 0.0
    if len(batch_ids) != len(targets):
        raise ValueError(
            "critic targets must pair 1:1 with prompts, "
            f"got {len(targets)} targets for {len(batch_ids)} prompts"
        )
    require_finite_critic_state(critic.state_dict(), optimizer.state_dict())
    targets_t = torch.tensor(list(targets), dtype=torch.float32)
    if not torch.isfinite(targets_t).all():
        raise ValueError("critic targets must be finite and representable as float32")
    total = 0.0
    for _ in range(steps):
        optimizer.zero_grad()
        pred = critic(batch_ids)
        loss = value_loss_mse(pred, targets_t.to(pred.device))
        if not torch.isfinite(loss):
            raise ValueError("critic loss must be finite; refusing optimizer update")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in critic.parameters()
        ):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("critic gradients must be finite; refusing optimizer update")
        optimizer.step()
        require_finite_critic_state(critic.state_dict(), optimizer.state_dict())
        total += loss.item()
    return total / steps


def explained_variance(predicted: Sequence[float], targets: Sequence[float]) -> float:
    """``1 - Var(target - pred) / Var(target)``; ``0.0`` when target variance is 0.

    Lengths must match.  A shorter prediction used to be zipped and then
    divided by ``len(targets)``, which understated the residual variance.
    Near-zero variance (below ``1e-12``) also returns ``0.0``: the ratio
    is float noise there, not signal.
    """
    if len(predicted) != len(targets):
        raise ValueError(
            "explained_variance length mismatch: "
            f"{len(predicted)} predictions vs {len(targets)} targets"
        )
    n = len(targets)
    if n == 0:
        return 0.0
    var_t = statistics.pvariance(targets)
    if var_t <= 1e-12:
        return 0.0
    residuals = [t - p for t, p in zip(targets, predicted, strict=True)]
    var_r = statistics.pvariance(residuals)
    return 1.0 - var_r / var_t


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
    returns = [a + v for a, v in zip(advantages, values[:t], strict=True)]
    return advantages, returns


def value_loss_mse(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Half-MSE value loss (PPO convention, Schulman et al. 2017 §7).

    ``0.5 * mean((V - R)^2)``.  Shapes must match exactly.
    """
    if values.shape != returns.shape:
        raise ValueError(f"shape mismatch: {tuple(values.shape)} vs {tuple(returns.shape)}")
    return 0.5 * ((values - returns) ** 2).mean()
