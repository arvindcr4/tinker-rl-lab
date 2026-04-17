"""
Group Saturation Diagnostic — tracks zero-variance fraction per GRPO step.

Key insight: When all G completions in a group receive the same reward,
the advantage vector is all-zeros and the gradient vanishes. This diagnostic
measures how often this happens and correlates it with learning stalls.

Metrics logged to W&B per step:
  - zero_variance_frac: fraction of groups where std(rewards) < epsilon
  - mean_group_std: average reward std across groups
  - effective_groups: number of groups contributing non-zero gradient
  - gradient_utilization: effective_groups / total_groups
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EPSILON = 1e-6


@dataclass
class GroupStats:
    rewards: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.rewards)

    @property
    def std(self) -> float:
        return statistics.pstdev(self.rewards) if len(self.rewards) > 1 else 0.0

    @property
    def is_saturated(self) -> bool:
        return self.std < EPSILON

    @property
    def spread(self) -> float:
        return max(self.rewards) - min(self.rewards) if self.rewards else 0.0


@dataclass
class StepDiagnostic:
    step: int
    groups: list[GroupStats] = field(default_factory=list)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_saturated(self) -> int:
        return sum(1 for g in self.groups if g.is_saturated)

    @property
    def zero_variance_frac(self) -> float:
        return self.n_saturated / self.n_groups if self.n_groups else 0.0

    @property
    def mean_group_std(self) -> float:
        stds = [g.std for g in self.groups]
        return statistics.mean(stds) if stds else 0.0

    @property
    def effective_groups(self) -> int:
        return self.n_groups - self.n_saturated

    @property
    def gradient_utilization(self) -> float:
        return self.effective_groups / self.n_groups if self.n_groups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "n_groups": self.n_groups,
            "n_saturated": self.n_saturated,
            "zero_variance_frac": self.zero_variance_frac,
            "mean_group_std": self.mean_group_std,
            "effective_groups": self.effective_groups,
            "gradient_utilization": self.gradient_utilization,
        }


class SaturationTracker:
    """Accumulates per-step group saturation metrics across a training run."""

    def __init__(self) -> None:
        self.history: list[StepDiagnostic] = []

    def record_step(self, step: int, group_rewards: list[list[float]]) -> StepDiagnostic:
        groups = [GroupStats(rewards=r) for r in group_rewards]
        diag = StepDiagnostic(step=step, groups=groups)
        self.history.append(diag)
        return diag

    def summary(self) -> dict[str, Any]:
        if not self.history:
            return {}
        zvf = [d.zero_variance_frac for d in self.history]
        gu = [d.gradient_utilization for d in self.history]
        return {
            "total_steps": len(self.history),
            "mean_zero_variance_frac": statistics.mean(zvf),
            "max_zero_variance_frac": max(zvf),
            "mean_gradient_utilization": statistics.mean(gu),
            "min_gradient_utilization": min(gu),
            "saturation_onset_step": next(
                (d.step for d in self.history if d.zero_variance_frac > 0.5), None
            ),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.summary(),
            "steps": [d.to_dict() for d in self.history],
        }
        path.write_text(json.dumps(data, indent=2))


def log_to_wandb(diag: StepDiagnostic) -> None:
    """Log saturation metrics to W&B if available."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(diag.to_dict(), step=diag.step)
    except ImportError:
        pass
