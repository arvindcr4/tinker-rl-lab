"""Reasoning delimiters in answer data must not corrupt the scored payload."""

import json

import pytest

from platform_tinker.tinkerrl.grpo import (
    PavlovNonXLAMReward,
    StrictToolCallReward,
    TrainingExample,
)


@pytest.mark.parametrize("prefix", ["", "<think>plan</think>\n"])
@pytest.mark.parametrize("value", ["a</think>b", "<think>literal</think>", "</think></think>"])
def test_tool_argument_reasoning_delimiters_are_literal_data(prefix, value):
    target = {"tool": "search", "arguments": {"query": value}}
    example = TrainingExample(prompt="q", target=target, metadata={"reward_kind": "tool_call"})
    response = prefix + json.dumps(target)
    assert PavlovNonXLAMReward().score(response, example) == 1.0


@pytest.mark.parametrize("prefix", ["", "<think>plan</think>\n"])
def test_patch_source_reasoning_delimiter_is_preserved(prefix):
    patch = 'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+"</think>"'
    example = TrainingExample(prompt="q", target=patch, metadata={"reward_kind": "patch"})
    assert PavlovNonXLAMReward().score(prefix + patch, example) == 1.0


def test_deep_model_json_is_a_failed_reward_not_a_training_exception():
    depth = 2000
    response = '{"tool":"search","arguments":{"query":' + "[" * depth + "0" + "]" * depth + "}}"
    example = TrainingExample(
        prompt="q",
        target={"tool": "search", "arguments": {"query": 0}},
        metadata={"reward_kind": "tool_call"},
    )
    assert StrictToolCallReward().score(response, example) == 0.0
    assert PavlovNonXLAMReward().score(response, example) == 0.0
