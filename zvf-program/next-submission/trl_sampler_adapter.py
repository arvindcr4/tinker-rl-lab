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
                eos = trainer._tokenizer.eos_token_id
                finish = (
                    "stop"
                    if eos in completion_tuple
                    else (
                        "length" if len(completion_tuple) >= trainer.max_completion_length else None
                    )
                )
                generated.append(Rollout(_tokens(prompt_tokens), completion_tuple, reward, finish))
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
                "update_eligible_groups": 0,
                "all_wrong_groups": 0,
                "all_correct_groups": 0,
                "mixed_groups": 0,
            },
        )
        for key in (
            "truncated_generated_rollouts",
            "unknown_finish_rollouts",
            "charged_generated_tokens",
            "generated_rollouts",
            "rollout_groups",
            "update_eligible_groups",
            "all_wrong_groups",
            "all_correct_groups",
            "mixed_groups",
        ):
            cumulative.setdefault(key, 0)
        for key in (
            "truncated_generated_rollouts",
            "unknown_finish_rollouts",
            "charged_generated_tokens",
            "generated_rollouts",
            "rollout_groups",
            "update_eligible_groups",
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
        def compute_loss(self, model, inputs, *args, **kwargs):
            # Count tokens actually passed to the loss, with nonzero advantages.
            mask, advantage = inputs.get("completion_mask"), inputs.get("advantages")
            if mask is not None and advantage is not None:
                active = mask * (advantage.reshape(-1, 1) != 0)
                if inputs.get("env_mask") is not None:
                    active = active * inputs["env_mask"]
                self._next_loss_tokens = getattr(self, "_next_loss_tokens", 0) + int(
                    active.sum().item()
                )
            return super().compute_loss(model, inputs, *args, **kwargs)

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


def make_optimizer_observer(trainer, callback_base, tensor_lib, *, stop_on_applied=False):
    """Observe tokens, gradients and parameter change; dependencies are injectable."""
    import math

    class ObserveOptimizerUpdate(callback_base):
        """Measured optimizer evidence; rollout eligibility alone never proves an update."""

        def __init__(self):
            self.before = None
            self.gradient_norm = None
            self.eligible_previous = 0
            trainer._next_update_observations = []

        def on_pre_optimizer_step(self, args_, state, control, **kwargs):
            params = [p for p in trainer.model.parameters() if p.requires_grad]
            self.before = [p.detach().clone() for p in params]
            grads = [p.grad for p in params if p.grad is not None]
            self.gradient_norm = (
                float(tensor_lib.sqrt(sum((g.detach().float() ** 2).sum() for g in grads)).item())
                if grads
                else None
            )
            return control

        def on_optimizer_step(self, args_, state, control, **kwargs):
            params = [p for p in trainer.model.parameters() if p.requires_grad]
            changed = self.before is not None and any(
                not tensor_lib.equal(a, p.detach())
                for a, p in zip(self.before, params, strict=True)
            )
            tokens = int(getattr(trainer, "_next_loss_tokens", 0))
            eligible = int(trainer._next_sampler_cumulative.get("update_eligible_groups", 0))
            finite_gradient = self.gradient_norm is not None and math.isfinite(self.gradient_norm)
            finite_parameters = all(bool(tensor_lib.isfinite(p.detach()).all()) for p in params)
            applied = bool(
                changed
                and finite_parameters
                and tokens > 0
                and finite_gradient
                and self.gradient_norm > 0
            )
            trainer._next_update_observations.append(
                {
                    "optimizer_step": int(state.global_step) + 1,
                    "active_loss_tokens": tokens,
                    "finite_gradient_norm": self.gradient_norm,
                    "parameter_changed": bool(changed),
                    "finite_parameters": finite_parameters,
                    "optimizer_step_observed": True,
                    "update_applied": applied,
                    "update_eligible_groups": eligible - self.eligible_previous,
                }
            )
            self.eligible_previous = eligible
            self.before = None
            trainer._next_loss_tokens = 0
            if applied and stop_on_applied:
                control.should_training_stop = True
            return control

    return ObserveOptimizerUpdate()
