"""Scoring hooks must see the same token snapshot that training receives."""

import socket
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tinker")
from tinker import types as T

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _Future


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("rollout snapshots must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def test_real_sdk_array_mutation_cannot_change_scored_rollout_or_behavior_logprobs():
    sequences = [
        T.SampledSequence(
            stop_reason="stop",
            tokens_np=np.array([token]),
            logprobs_np=np.array([lp], dtype=np.float32),
        )
        for token, lp in [(1, -0.2), (2, -1.2)]
    ]
    sampler = SimpleNamespace(
        sample=lambda *_args, **_kwargs: _Future(T.SampleResponse(sequences=sequences))
    )
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: [0], decode=lambda tokens, **_kwargs: str(tokens[0])
    )
    seen = []

    def score(text, _example):
        seen.append(text)
        sequences[1].tokens_np[0] = 1
        sequences[1].logprobs_np[0] = -99.0
        return float(text == "1")

    metadata = {}
    _, tokens, rewards, _ = grpo._sample_scored_group(
        tokenizer,
        sampler,
        T,
        grpo.GRPOConfig(name="snapshot", group_size=2),
        SimpleNamespace(score=score),
        grpo.TrainingExample(prompt="q"),
        meta=metadata,
    )
    assert tokens == [[1], [2]]
    assert seen == ["1", "2"]
    assert rewards == [1.0, 0.0]
    assert metadata["sampler_logprobs"][1] == pytest.approx([-1.2])


def test_callback_cannot_rewrite_truncation_metadata():
    sequences = [
        SimpleNamespace(tokens=[1], logprobs=[-0.1], stop_reason="stop"),
        SimpleNamespace(tokens=[2], logprobs=[-0.2], stop_reason="length"),
    ]
    sampler = SimpleNamespace(
        sample=lambda *_args, **_kwargs: _Future(SimpleNamespace(sequences=sequences))
    )
    tokenizer = SimpleNamespace(
        encode=lambda *_args, **_kwargs: [0], decode=lambda tokens, **_kwargs: str(tokens[0])
    )
    metadata = {}
    seen = []

    def score(text, _example):
        seen.append(text)
        sequences[1].tokens[0] = 1
        sequences[1].stop_reason = "stop"
        metadata["truncated"] = [False, False]
        return float(text == "1")

    _, tokens, rewards, _ = grpo._sample_scored_group(
        tokenizer,
        sampler,
        T,
        grpo.GRPOConfig(name="snapshot", group_size=2),
        SimpleNamespace(score=score),
        grpo.TrainingExample(prompt="q"),
        meta=metadata,
    )
    assert tokens == [[1], [2]]
    assert seen == ["1", "2"]
    assert rewards == [1.0, 0.0]
    assert metadata["truncated"] == [False, True]


def test_mutating_decoder_does_not_change_training_tokens():
    sequences = [SimpleNamespace(tokens=[1], logprobs=None, stop_reason="stop")]
    sampler = SimpleNamespace(
        sample=lambda *_args, **_kwargs: _Future(SimpleNamespace(sequences=sequences))
    )

    def decode(tokens, **_kwargs):
        answer = str(tokens[0])
        tokens.clear()
        return answer

    tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [0], decode=decode)
    _, tokens, rewards, _ = grpo._sample_scored_group(
        tokenizer,
        sampler,
        T,
        grpo.GRPOConfig(name="decoder", group_size=1),
        SimpleNamespace(score=lambda text, _example: float(text == "1")),
        grpo.TrainingExample(prompt="q"),
    )
    assert tokens == [[1]]
    assert sequences[0].tokens == [1]
    assert rewards == [1.0]


@pytest.mark.parametrize("prompt_budget", [2, 8])
def test_tokenizer_owned_prompt_cannot_change_after_sampling(prompt_budget):
    shared = [10, 11, 12]
    sampled = []

    def sample(prompt, **_kwargs):
        sampled.append(list(prompt.to_ints()))
        return _Future(
            SimpleNamespace(
                sequences=[SimpleNamespace(tokens=[1], logprobs=None, stop_reason="stop")]
            )
        )

    def decode(tokens, **_kwargs):
        shared[:] = [99]
        return "1"

    prompt, responses, rewards, _ = grpo._sample_scored_group(
        SimpleNamespace(encode=lambda *_a, **_k: shared, decode=decode),
        SimpleNamespace(sample=sample),
        T,
        grpo.GRPOConfig(name="prompt-snapshot", group_size=1, max_prompt_tokens=prompt_budget),
        SimpleNamespace(score=lambda *_a: 1.0),
        grpo.TrainingExample(prompt="q"),
    )
    expected = [10, 11, 12][:prompt_budget]
    assert shared == [99]  # Prove the callback actually mutated the tokenizer's list.
    assert prompt == sampled[0] == expected
    datum = grpo._build_datum(prompt, responses[0])
    assert datum.model_input.to_ints() == expected
    assert rewards == [1.0]
