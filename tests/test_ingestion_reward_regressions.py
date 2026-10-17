"""CPU-only regressions for ingestion and false exact-match scores."""

import json
import subprocess

import pytest

from platform_tinker.tinkerrl.grpo import (
    PatchReward,
    StrictToolCallReward,
    TrainingExample,
    _swe_gym_prompt,
)


@pytest.mark.parametrize(
    ("expected", "predicted"),
    [(True, 1), (False, 0), ([True], [1]), ({"enabled": False}, {"enabled": 0})],
)
def test_tool_booleans_are_not_numbers_even_in_nested_values(expected, predicted):
    example = TrainingExample(prompt="q", target={"tool": "set", "arguments": {"v": expected}})
    response = json.dumps({"tool": "set", "arguments": {"v": predicted}})
    assert StrictToolCallReward().score(response, example) == pytest.approx(0.7)


@pytest.mark.parametrize("conflict_in_target", [False, True])
def test_boolean_numeric_argument_alias_conflicts_are_rejected(conflict_in_target):
    payload = {"tool": "set", "arguments": {"v": [True]}}
    conflicted = {**payload, "parameters": {"v": [1]}}
    example = TrainingExample(prompt="q", target=conflicted if conflict_in_target else payload)
    response = json.dumps(payload if conflict_in_target else conflicted)
    assert StrictToolCallReward().score(response, example) == 0.0


def test_tool_numeric_equivalence_and_nested_object_order_remain_supported():
    example = TrainingExample(
        prompt="q", target={"tool": "set", "arguments": {"v": {"a": [True, 1], "b": None}}}
    )
    response = json.dumps({"tool": "set", "arguments": {"v": {"b": None, "a": [True, 1.0]}}})
    assert StrictToolCallReward().score(response, example) == 1.0


@pytest.mark.parametrize("literal", ["1e400", "-1e400"])
def test_tool_overflowing_json_numbers_are_rejected(literal):
    example = TrainingExample(prompt="q", target={"tool": "set", "arguments": {"v": 1}})
    response = '{"tool":"set","arguments":{"v":[' + literal + "]}}"
    assert StrictToolCallReward().score(response, example) == 0.0


def test_tool_overflow_cannot_hide_conflicting_argument_aliases():
    example = TrainingExample(prompt="q", target={"tool": "set", "arguments": {"v": 1}})
    response = '{"tool":"set","arguments":{"v":1e400},"parameters":{"v":1e500}}'
    assert StrictToolCallReward().score(response, example) == 0.0


def test_tool_finite_exponent_numbers_remain_supported():
    example = TrainingExample(prompt="q", target={"tool": "set", "arguments": {"v": 1e300}})
    response = '{"tool":"set","arguments":{"v":1e300}}'
    assert StrictToolCallReward().score(response, example) == 1.0


def _swe_row(**values):
    return {"repo": "owner/repo", "base_commit": "a" * 40, "problem_statement": "fix", **values}


def test_swe_test_names_accept_lists_and_json_encoded_lists_equivalently():
    lists = {"FAIL_TO_PASS": ["test_beta", "test_alpha"], "PASS_TO_PASS": ["test_stable"]}
    encoded = {key: json.dumps(value) for key, value in lists.items()}
    prompt = _swe_gym_prompt(_swe_row(**lists))
    assert _swe_gym_prompt(_swe_row(**encoded)) == prompt
    assert "test_alpha\ntest_beta\ntest_stable" in prompt


@pytest.mark.parametrize(
    "invalid", ['["test_unclosed"', '"test_scalar"', {"test": True}, [1], [""]]
)
def test_swe_malformed_test_names_fail_before_prompt_construction(invalid):
    with pytest.raises(ValueError, match="FAIL_TO_PASS"):
        _swe_gym_prompt(_swe_row(FAIL_TO_PASS=invalid))


def test_swe_absent_or_null_test_names_remain_optional():
    assert _swe_gym_prompt(_swe_row()) == _swe_gym_prompt(
        _swe_row(FAIL_TO_PASS=None, PASS_TO_PASS=[])
    )


PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n x\n \n-y\n+z\n"


def test_removing_blank_diff_context_is_not_an_exact_patch():
    candidate = PATCH.replace("\n \n", "\n")
    valid = subprocess.run(
        ["git", "apply", "--numstat", "-"], input=PATCH, text=True, capture_output=True
    )
    invalid = subprocess.run(
        ["git", "apply", "--numstat", "-"], input=candidate, text=True, capture_output=True
    )
    assert valid.returncode == 0
    assert invalid.returncode != 0
    example = TrainingExample(prompt="q", target=PATCH)
    assert PatchReward().score(candidate, example) < 1.0


def test_trailing_patch_content_whitespace_is_not_an_exact_match():
    example = TrainingExample(prompt="q", target=PATCH)
    assert PatchReward().score(PATCH.replace("+z\n", "+z \n"), example) < 1.0


def test_patch_line_endings_and_boundary_blank_lines_remain_equivalent():
    example = TrainingExample(prompt="q", target=PATCH)
    assert PatchReward().score("\r\n" + PATCH.replace("\n", "\r\n") + "\r\n", example) == 1.0
