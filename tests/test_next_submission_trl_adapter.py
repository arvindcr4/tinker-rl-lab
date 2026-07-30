from __future__ import annotations

from collections import deque
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "zvf-program/next-submission"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SAMPLER = load_module("contrast_sampler", HERE / "contrast_sampler.py")
ADAPTER = load_module("trl_sampler_adapter", HERE / "trl_sampler_adapter.py")


class FakeTokenizer:
    eos_token_id = 2

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return str(tokens[-1])


class FakeTrainer:
    num_generations = 8
    _tokenizer = FakeTokenizer()
    processing_class = _tokenizer

    def __init__(self, rounds):
        self.rounds = deque(rounds)
        self.call_sizes = []

    def _tokenize_prompts(self, prompts):
        return [[len(str(prompt)), 99] for prompt in prompts], None, {}

    def _generate_single_turn(self, prompt_ids, images, fields):
        assert images is None and fields == {}
        values = self.rounds.popleft()
        assert len(values) == len(prompt_ids)
        self.call_sizes.append(len(prompt_ids))
        return [[value] for value in values], None


def prompt(name: str):
    return [{"role": "user", "content": name}]


def parity_reward(text: str, gold: str) -> float:
    return float(int(text) % 2 == int(gold))


def test_intervention_generates_six_extra_only_for_mixed_initial_groups():
    one, two = prompt("one"), prompt("two")
    prompts = [one] * 8 + [two] * 8
    answers = {ADAPTER.prompt_key(one): "0", ADAPTER.prompt_key(two): "0"}
    # Group one: even/even -> all correct. Group two: even/odd -> mixed.
    trainer = FakeTrainer([[2, 2], [4, 3], *([[5]] * 6)])
    rollout = ADAPTER.make_rollout_func(
        arm=SAMPLER.ARM_EARLY_STOP,
        answer_by_prompt=answers,
        reward_parser=parity_reward,
    )
    output = rollout(prompts, trainer)

    assert trainer.call_sizes == [2, 2, 1, 1, 1, 1, 1, 1]
    assert len(output["completion_ids"]) == 16
    assert output["env_mask"][:8] == [[0]] * 8
    assert output["env_mask"][8:] == [[1]] * 8
    assert output["sampler_reward"][:8] == [None] * 8
    assert trainer._next_sampler_last == {
        "rollout_groups": 2,
        "generated_rollouts": 10,
        "charged_generated_tokens": 10,
        "updated_groups": 1,
        "all_wrong_fraction": 0.0,
        "all_correct_fraction": 0.5,
        "mixed_fraction": 0.5,
    }


def test_baseline_uses_same_roundwise_adapter_but_generates_all_eight():
    one, two = prompt("one"), prompt("two")
    prompts = [one] * 8 + [two] * 8
    answers = {ADAPTER.prompt_key(one): "0", ADAPTER.prompt_key(two): "0"}
    trainer = FakeTrainer([[2, 3]] * 8)
    rollout = ADAPTER.make_rollout_func(
        arm=SAMPLER.ARM_BASELINE,
        answer_by_prompt=answers,
        reward_parser=parity_reward,
    )
    output = rollout(prompts, trainer)

    assert trainer.call_sizes == [2] * 8
    assert output["env_mask"] == [[1]] * 16
    assert trainer._next_sampler_last["generated_rollouts"] == 16
    assert trainer._next_sampler_last["charged_generated_tokens"] == 16


def test_reward_bridge_rejects_misaligned_or_leaky_inactive_slots():
    with pytest.raises(SAMPLER.SamplerContractError, match="aligned"):
        ADAPTER.sampler_reward(["a"], [1.0, 0.0], [True])
    with pytest.raises(SAMPLER.SamplerContractError, match="inactive"):
        ADAPTER.sampler_reward(["a"], [1.0], [False])
