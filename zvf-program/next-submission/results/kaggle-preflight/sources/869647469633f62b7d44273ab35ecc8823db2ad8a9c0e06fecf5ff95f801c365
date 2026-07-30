#!/usr/bin/env python3
"""TRL 1.8 adapter for the preregistered contrast-aware sampler.

This module intentionally imports neither torch nor TRL at import time.  The
remote runner supplies its pinned ``GRPOTrainer`` class to
``build_sampler_trainer_class``; local conformance tests use a small fake.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from contrast_sampler import (
    ARM_BASELINE,
    ARM_EARLY_STOP,
    EXPANSION_SIZE,
    GROUP_SIZE,
    Rollout,
    SamplerContractError,
    aggregate_group_telemetry,
    assemble_group,
    should_expand,
)


RewardParser = Callable[[str, str], float]


def prompt_key(prompt: Any) -> str:
    """Stable key used to bind each rollout to its verifier target."""

    return json.dumps(prompt, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tokens(value: Any) -> tuple[int, ...]:
    raw = value.tolist() if hasattr(value, "tolist") else list(value)
    tokens = tuple(int(item) for item in raw)
    if not tokens:
        raise SamplerContractError("generation returned an empty token sequence")
    return tokens


def _unique_prompt_groups(prompts: Sequence[Any], group_size: int) -> list[Any]:
    if group_size != GROUP_SIZE:
        raise SamplerContractError("the frozen adapter requires num_generations=8")
    if not prompts or len(prompts) % group_size:
        raise SamplerContractError("input rows do not form integral G8 prompt groups")
    unique: list[Any] = []
    for start in range(0, len(prompts), group_size):
        group = prompts[start : start + group_size]
        keys = {prompt_key(prompt) for prompt in group}
        if len(keys) != 1:
            raise SamplerContractError("input rows are not contiguous identical-prompt groups")
        unique.append(group[0])
    return unique


def make_rollout_func(
    *,
    arm: str,
    answer_by_prompt: dict[str, str],
    reward_parser: RewardParser,
) -> Callable[[list[Any], Any], dict[str, Any]]:
    """Create the custom rollout callable consumed by TRL ``GRPOTrainer``."""

    if arm not in (ARM_BASELINE, ARM_EARLY_STOP):
        raise SamplerContractError(f"unknown arm: {arm}")

    def rollout_func(prompts: list[Any], trainer: Any) -> dict[str, Any]:
        unique_prompts = _unique_prompt_groups(prompts, trainer.num_generations)

        def generate_round(group_indices: list[int]) -> list[Rollout]:
            if not group_indices:
                return []
            selected = [unique_prompts[index] for index in group_indices]
            prompt_ids, images, fields = trainer._tokenize_prompts(selected)
            completion_ids, _ = trainer._generate_single_turn(prompt_ids, images, fields)
            if len(completion_ids) != len(selected):
                raise SamplerContractError("generation cardinality differs from requested prompts")
            generated: list[Rollout] = []
            for group_index, prompt_tokens, completion_tokens in zip(
                group_indices, prompt_ids, completion_ids, strict=True
            ):
                completion_tuple = _tokens(completion_tokens)
                text = trainer.processing_class.decode(completion_tuple, skip_special_tokens=True)
                key = prompt_key(unique_prompts[group_index])
                try:
                    gold = answer_by_prompt[key]
                except KeyError as exc:
                    raise SamplerContractError("prompt is not bound to a verifier target") from exc
                reward = float(reward_parser(text, gold))
                generated.append(Rollout(_tokens(prompt_tokens), completion_tuple, reward))
            return generated

        group_indices = list(range(len(unique_prompts)))
        initial: list[list[Rollout]] = [[] for _ in unique_prompts]
        for _ in range(2):
            for group_index, generated in zip(
                group_indices, generate_round(group_indices), strict=True
            ):
                initial[group_index].append(generated)

        expansion: list[list[Rollout]] = [[] for _ in unique_prompts]
        if arm == ARM_BASELINE:
            eligible = group_indices
        else:
            eligible = [
                index
                for index, rows in enumerate(initial)
                if should_expand([row.reward for row in rows])
            ]
        for _ in range(EXPANSION_SIZE):
            for group_index, generated in zip(eligible, generate_round(eligible), strict=True):
                expansion[group_index].append(generated)

        eos_token_id = trainer._tokenizer.eos_token_id
        groups = [
            assemble_group(
                arm,
                initial[index],
                expansion[index],
                eos_token_id=eos_token_id,
            )
            for index in group_indices
        ]
        telemetry = aggregate_group_telemetry(groups)
        cumulative = getattr(
            trainer,
            "_next_sampler_cumulative",
            {
                "charged_generated_tokens": 0,
                "generated_rollouts": 0,
                "rollout_groups": 0,
                "updated_groups": 0,
                "all_wrong_groups": 0,
                "all_correct_groups": 0,
                "mixed_groups": 0,
            },
        )
        for key in (
            "charged_generated_tokens",
            "generated_rollouts",
            "rollout_groups",
            "updated_groups",
            "all_wrong_groups",
            "all_correct_groups",
            "mixed_groups",
        ):
            cumulative.setdefault(key, 0)
        for key in (
            "charged_generated_tokens",
            "generated_rollouts",
            "rollout_groups",
            "updated_groups",
        ):
            cumulative[key] += int(telemetry[key])
        group_count = int(telemetry["rollout_groups"])
        for label in ("all_wrong", "all_correct", "mixed"):
            cumulative[f"{label}_groups"] += round(
                float(telemetry[f"{label}_fraction"]) * group_count
            )
        trainer._next_sampler_cumulative = cumulative
        trainer._next_sampler_last = telemetry

        return {
            "prompt_ids": [tokens for group in groups for tokens in group.prompt_ids],
            "completion_ids": [tokens for group in groups for tokens in group.completion_ids],
            "logprobs": None,
            "env_mask": [
                [1] * len(tokens) if active else [0] * len(tokens)
                for group in groups
                for tokens, active in zip(group.completion_ids, group.active, strict=True)
            ],
            "sampler_reward": [reward for group in groups for reward in group.rewards],
            "sampler_active": [active for group in groups for active in group.active],
        }

    return rollout_func


def sampler_reward(
    completions: list[Any],
    sampler_reward: list[float | None],
    sampler_active: list[bool],
    **_: Any,
) -> list[float | None]:
    """Return only rewards for slots authorized to contribute an update."""

    if not (len(completions) == len(sampler_reward) == len(sampler_active)):
        raise SamplerContractError("reward metadata is not aligned with completions")
    values: list[float | None] = []
    for reward, active in zip(sampler_reward, sampler_active, strict=True):
        if active:
            if reward not in (0.0, 1.0):
                raise SamplerContractError("active sampler slot lacks a binary reward")
            values.append(float(reward))
        else:
            if reward is not None:
                raise SamplerContractError("inactive sampler slot carries a reward")
            values.append(None)
    return values


def build_sampler_trainer_class(base_class: type[Any]) -> type[Any]:
    """Attach truthful sampler telemetry after TRL prepares each train batch."""

    class SamplerTrainer(base_class):
        def _generate_and_score_completions(self, inputs: list[dict[str, Any]]) -> dict[str, Any]:
            output = super()._generate_and_score_completions(inputs)
            mode = "train" if self.model.training else "eval"
            if mode != "train":
                return output
            telemetry = self._next_sampler_last
            cumulative = self._next_sampler_cumulative
            for key, value in telemetry.items():
                self._metrics[mode][f"sampler/{key}"].append(float(value))
            for key, value in cumulative.items():
                self._metrics[mode][f"sampler/{key}_cumulative"].append(float(value))
            return output

    SamplerTrainer.__name__ = "ContrastAwareGRPOTrainer"
    return SamplerTrainer
