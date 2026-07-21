"""Executable action-reversal construction for the flagship theory gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal, Sequence

Action = Literal["retry", "recheck"]


@dataclass(frozen=True)
class Construction:
    group_size: int = 8
    additional_samples: int = 8
    clean_success_probability: float = 0.10
    success_value: float = 1.0
    sample_cost: float = 0.02
    probe_cost: float = 0.03
    repaired_signal_value: float = 0.50


@dataclass(frozen=True)
class RegimeUtilities:
    retry: float
    recheck: float

    @property
    def optimal_action(self) -> Action:
        return "retry" if self.retry > self.recheck else "recheck"

    @property
    def gap(self) -> float:
        return abs(self.retry - self.recheck)


def validate(construction: Construction) -> None:
    if construction.group_size < 2 or construction.additional_samples < 1:
        raise ValueError("group sizes must be positive and the observed group must contain two samples")
    if not 0 < construction.clean_success_probability < 1:
        raise ValueError("clean success probability must lie strictly between zero and one")
    nonnegative = (
        construction.success_value,
        construction.sample_cost,
        construction.probe_cost,
        construction.repaired_signal_value,
    )
    if any(value < 0 for value in nonnegative):
        raise ValueError("values and costs must be nonnegative")


def matched_primary_observation(construction: Construction) -> dict[str, object]:
    """The observable state supplied to an outcome-only controller."""
    validate(construction)
    rewards = [0] * construction.group_size
    return {
        "rewards": rewards,
        "p_hat": 0.0,
        "group_size": construction.group_size,
        "homogeneous_group_rate": 1.0,
        "all_failure": True,
        "runtime_error": None,
        "latency_bucket": "normal",
        "reward_code_version": "matched",
    }


def clean_hard_utilities(construction: Construction) -> RegimeUtilities:
    """Clean verifier: additional samples can reveal a rare correct completion."""
    validate(construction)
    success_probability = 1.0 - (
        1.0 - construction.clean_success_probability
    ) ** construction.additional_samples
    retry = (
        construction.success_value * success_probability
        - construction.sample_cost * construction.additional_samples
    )
    return RegimeUtilities(retry=retry, recheck=-construction.probe_cost)


def broken_verifier_utilities(construction: Construction) -> RegimeUtilities:
    """Broken verifier: retrying cannot produce observed signal; a probe can repair it."""
    validate(construction)
    retry = -construction.sample_cost * construction.additional_samples
    recheck = construction.repaired_signal_value - construction.probe_cost
    return RegimeUtilities(retry=retry, recheck=recheck)


def minimax_outcome_only_regret(construction: Construction) -> float:
    """Best worst-case regret for any randomized policy measurable only in the matched observation."""
    clean, broken = clean_hard_utilities(construction), broken_verifier_utilities(construction)
    if clean.optimal_action == broken.optimal_action:
        return 0.0
    return clean.gap * broken.gap / (clean.gap + broken.gap)


def reversal_margins(construction: Construction) -> dict[str, float]:
    """Strict margins defining the open parameter region with opposite optima."""
    clean, broken = clean_hard_utilities(construction), broken_verifier_utilities(construction)
    return {
        "clean_retry_over_recheck": clean.retry - clean.recheck,
        "broken_recheck_over_retry": broken.recheck - broken.retry,
    }


def action_reversal_holds(construction: Construction) -> bool:
    return all(margin > 0 for margin in reversal_margins(construction).values())


def minimax_retry_probability(construction: Construction) -> float:
    """Retry probability that equalizes regret across the two latent regimes."""
    clean, broken = clean_hard_utilities(construction), broken_verifier_utilities(construction)
    if clean.optimal_action == broken.optimal_action:
        return float(clean.optimal_action == "retry")
    return clean.gap / (clean.gap + broken.gap)


def probe_policy_regret_bound(construction: Construction, probe_error: float) -> float:
    """Charged worst-case regret for a calibration probe with error epsilon."""
    if not 0 <= probe_error <= 1:
        raise ValueError("probe error must lie in [0,1]")
    gaps = (clean_hard_utilities(construction).gap, broken_verifier_utilities(construction).gap)
    return construction.probe_cost + probe_error * max(gaps)


def evaluate(construction: Construction) -> dict[str, object]:
    clean = clean_hard_utilities(construction)
    broken = broken_verifier_utilities(construction)
    return {
        "construction": asdict(construction),
        "matched_primary_observation": matched_primary_observation(construction),
        "clean_hard": asdict(clean) | {"optimal_action": clean.optimal_action, "gap": clean.gap},
        "broken_verifier": asdict(broken) | {"optimal_action": broken.optimal_action, "gap": broken.gap},
        "minimax_retry_probability": minimax_retry_probability(construction),
        "minimax_outcome_only_regret": minimax_outcome_only_regret(construction),
        "reversal_margins": reversal_margins(construction),
        "action_reversal_holds": action_reversal_holds(construction),
        "perfect_probe_regret_bound": probe_policy_regret_bound(construction, 0.0),
        "ten_percent_probe_error_regret_bound": probe_policy_regret_bound(construction, 0.10),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(json.dumps(evaluate(Construction()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
