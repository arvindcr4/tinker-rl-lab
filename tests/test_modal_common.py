"""Pin the GSM8K reward/gold/prompt helpers shared by the hybrid Modal launchers."""

from pathlib import Path

import pytest
from tests._shared_fakes import load_module

mc = load_module(
    "_modal_common",
    Path(__file__).resolve().parents[1] / "platform_hybrid/experiments/modal/_modal_common.py",
)


@pytest.mark.parametrize(
    ("response", "gold", "expected"),
    [
        (r"so \boxed{1 000}", "1000", 0.0),  # v3 rejects ambiguous numeric literals
        (r"\boxed{abc}", "abc", 0.0),  # GSM8K v3 requires numeric answers
        ("first 3 then the answer is 18", "18", 1.0),  # trailing-number rescue
        (r"\boxed{17} but really 18", "18", 0.0),  # final box is authoritative
        ("no answer here", "18", 0.0),
    ],
)
def test_boxed_or_last_number_reward(response, gold, expected):
    assert mc.boxed_or_last_number_reward(response, gold) == expected


@pytest.mark.parametrize(
    ("text", "gold", "expected"),
    [
        ("so 3*6 = 18", "18", 1.0),  # compatibility alias uses shared v3 fallback
        (r"\boxed{1,000}", "1000", 1.0),
        (r"\boxed{abc}", "abc", 0.0),
    ],
)
def test_boxed_only_reward(text, gold, expected):
    assert mc.boxed_only_reward(text, gold) == expected


def test_gsm8k_gold():
    assert mc.gsm8k_gold("work...\n#### 1,234") == "1234"
    assert mc.gsm8k_gold("no marker") is None


def test_chatml_prompt():
    assert mc.chatml_prompt("S", "U") == (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n"
    )


def test_push_to_hub_private_skips_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert mc.push_to_hub_private(object(), object(), "x/y") is False


def test_unsupported_completion_representation_fails():
    with pytest.raises(TypeError):
        mc.boxed_only_reward(None, "18")
