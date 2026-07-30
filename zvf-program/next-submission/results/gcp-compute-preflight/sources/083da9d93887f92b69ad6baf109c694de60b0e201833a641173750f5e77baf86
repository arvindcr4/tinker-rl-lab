#!/usr/bin/env python3
"""Pure contract logic for the preregistered G2-to-G8 sampler.

The remote TRL adapter is deliberately thin: all decisions, slot assembly, and
charged-token accounting are defined here so they can be tested without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


ARM_BASELINE = "grpo_g8"
ARM_EARLY_STOP = "contrast_early_stop_g2_to_g8"
ARMS = (ARM_BASELINE, ARM_EARLY_STOP)
GROUP_SIZE = 8
INITIAL_SIZE = 2
EXPANSION_SIZE = GROUP_SIZE - INITIAL_SIZE
GroupClass = Literal["all_wrong", "all_correct", "mixed"]


class SamplerContractError(ValueError):
    """The rollout data do not satisfy the frozen sampler contract."""


@dataclass(frozen=True)
class Rollout:
    """One genuinely generated completion and its verifier outcome."""

    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]
    reward: float

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise SamplerContractError("prompt token sequence may not be empty")
        if not self.completion_ids:
            raise SamplerContractError("completion token sequence may not be empty")
        if self.reward not in (0.0, 1.0):
            raise SamplerContractError("the frozen sampler requires binary verifier rewards")


@dataclass(frozen=True)
class FixedSlots:
    """Eight TRL slots plus truthful generation accounting for one prompt."""

    prompt_ids: tuple[tuple[int, ...], ...]
    completion_ids: tuple[tuple[int, ...], ...]
    rewards: tuple[float | None, ...]
    active: tuple[bool, ...]
    group_class: GroupClass
    charged_generated_tokens: int
    generated_rollouts: int
    update_applied: bool

    def __post_init__(self) -> None:
        widths = {
            len(self.prompt_ids),
            len(self.completion_ids),
            len(self.rewards),
            len(self.active),
        }
        if widths != {GROUP_SIZE}:
            raise SamplerContractError("fixed output must contain exactly eight aligned slots")
        if self.charged_generated_tokens <= 0 or self.generated_rollouts <= 0:
            raise SamplerContractError("generation accounting must be positive")
        if self.update_applied and not any(self.active):
            raise SamplerContractError("an update cannot be applied without active slots")
        for reward, active in zip(self.rewards, self.active, strict=True):
            if active and reward not in (0.0, 1.0):
                raise SamplerContractError("active slots require binary rewards")
            if not active and reward is not None:
                raise SamplerContractError("inactive slots must be unscorable")


def classify_binary_rewards(rewards: Sequence[float]) -> GroupClass:
    """Classify a non-empty group under the locked binary verifier."""

    if not rewards:
        raise SamplerContractError("cannot classify an empty reward group")
    if any(value not in (0.0, 1.0) for value in rewards):
        raise SamplerContractError("the frozen sampler requires binary verifier rewards")
    if all(value == 0.0 for value in rewards):
        return "all_wrong"
    if all(value == 1.0 for value in rewards):
        return "all_correct"
    return "mixed"


def should_expand(initial_rewards: Sequence[float]) -> bool:
    """Return the preregistered decision after exactly two completions."""

    if len(initial_rewards) != INITIAL_SIZE:
        raise SamplerContractError("the early-stop decision requires exactly two rewards")
    return classify_binary_rewards(initial_rewards) == "mixed"


def assemble_group(
    arm: str,
    initial: Sequence[Rollout],
    expansion: Sequence[Rollout],
    *,
    eos_token_id: int,
) -> FixedSlots:
    """Assemble truthful generated rollouts into TRL's fixed eight-slot shape.

    The baseline must supply eight real rollouts.  The intervention supplies two
    real rollouts for a homogeneous initial pair and eight for a mixed pair.
    Homogeneous intervention groups are represented by eight inactive EOS slots;
    their two real samples remain counted in ``charged_generated_tokens``.
    """

    if arm not in ARMS:
        raise SamplerContractError(f"unknown arm: {arm}")
    if len(initial) != INITIAL_SIZE:
        raise SamplerContractError("every group must start with exactly two rollouts")
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise SamplerContractError("eos_token_id must be a non-negative integer")

    prompt_ids = initial[0].prompt_ids
    if any(rollout.prompt_ids != prompt_ids for rollout in [*initial, *expansion]):
        raise SamplerContractError("all rollouts in a group must share one tokenized prompt")

    initial_class = classify_binary_rewards([rollout.reward for rollout in initial])
    if arm == ARM_BASELINE:
        if len(expansion) != EXPANSION_SIZE:
            raise SamplerContractError("the G8 baseline requires six rollouts after the first two")
        generated = [*initial, *expansion]
        group_class = classify_binary_rewards([rollout.reward for rollout in generated])
        active = (True,) * GROUP_SIZE
        rewards: tuple[float | None, ...] = tuple(rollout.reward for rollout in generated)
        completions = tuple(rollout.completion_ids for rollout in generated)
        prompts = tuple(prompt_ids for _ in generated)
        update_applied = group_class == "mixed"
        # Canonical GRPO naturally yields zero advantages on homogeneous G8
        # groups, but the generated tokens are still fully charged.
    elif initial_class == "mixed":
        if len(expansion) != EXPANSION_SIZE:
            raise SamplerContractError("a mixed G2 group must expand by exactly six rollouts")
        generated = [*initial, *expansion]
        group_class = classify_binary_rewards([rollout.reward for rollout in generated])
        active = (True,) * GROUP_SIZE
        rewards = tuple(rollout.reward for rollout in generated)
        completions = tuple(rollout.completion_ids for rollout in generated)
        prompts = tuple(prompt_ids for _ in generated)
        update_applied = True
    else:
        if expansion:
            raise SamplerContractError("a homogeneous G2 group may not be expanded")
        generated = list(initial)
        group_class = initial_class
        prompts = tuple(prompt_ids for _ in range(GROUP_SIZE))
        completions = tuple((eos_token_id,) for _ in range(GROUP_SIZE))
        rewards = (None,) * GROUP_SIZE
        active = (False,) * GROUP_SIZE
        update_applied = False

    return FixedSlots(
        prompt_ids=prompts,
        completion_ids=completions,
        rewards=rewards,
        active=active,
        group_class=group_class,
        charged_generated_tokens=sum(len(rollout.completion_ids) for rollout in generated),
        generated_rollouts=len(generated),
        update_applied=update_applied,
    )


def aggregate_group_telemetry(groups: Sequence[FixedSlots]) -> dict[str, float | int]:
    """Aggregate complete, finite sampler telemetry for one generation batch."""

    if not groups:
        raise SamplerContractError("cannot aggregate an empty generation batch")
    count = len(groups)
    class_counts = {
        label: sum(group.group_class == label for group in groups)
        for label in ("all_wrong", "all_correct", "mixed")
    }
    if sum(class_counts.values()) != count:
        raise SamplerContractError("group classes do not partition the batch")
    return {
        "rollout_groups": count,
        "generated_rollouts": sum(group.generated_rollouts for group in groups),
        "charged_generated_tokens": sum(group.charged_generated_tokens for group in groups),
        "updated_groups": sum(group.update_applied for group in groups),
        "all_wrong_fraction": class_counts["all_wrong"] / count,
        "all_correct_fraction": class_counts["all_correct"] / count,
        "mixed_fraction": class_counts["mixed"] / count,
    }
