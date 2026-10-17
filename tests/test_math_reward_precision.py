"""Exact numerical reward decisions must not depend on binary float rounding."""

from decimal import Inexact, ROUND_UP, localcontext
from fractions import Fraction

import pytest
from hypothesis import given, strategies as st

from platform_tinker.tinkerrl.grpo import (
    ExactMathReward,
    MathReward,
    TrainingExample,
    _numeric_equal,
)


@pytest.mark.parametrize(
    ("candidate", "answer", "expected"),
    [
        ("9007199254740993", "9007199254740992", False),
        ("9007199254740992.02", "9007199254740992", False),
        ("9007199254740992.005", "9007199254740992", True),
        ("42.01", "42", False),
        ("42.009999999999999999999999999999999999", "42", True),
        ("0.01", "1e-1000000", True),
        ("0.01", "-1e-1000000", False),
        ("1e1000000", "1", False),
        ("NaN", "NaN", False),
        ("Infinity", "Infinity", False),
    ],
)
def test_decimal_tolerance_boundaries(candidate, answer, expected):
    assert _numeric_equal(candidate, answer) is expected
    # A caller's decimal settings cannot change the reward.
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        assert _numeric_equal(candidate, answer) is expected


@given(
    left=st.integers(-(10**60), 10**60),
    right=st.integers(-(10**60), 10**60),
    scale=st.integers(0, 60),
)
def test_numeric_comparison_matches_exact_rational_oracle(left, right, scale):
    candidate, answer = f"{left}e-{scale}", f"{right}e-{scale}"
    expected = abs(Fraction(candidate) - Fraction(answer)) < Fraction(1, 100)
    assert _numeric_equal(candidate, answer) == expected


@pytest.mark.parametrize("reward", [ExactMathReward(), MathReward()])
def test_boxed_large_integer_is_not_rounded_to_the_target(reward):
    example = TrainingExample(prompt="q", target="9007199254740992")
    assert reward.score(r"\boxed{9007199254740993}", example) < 1.0
    assert reward.score(r"\boxed{9007199254740992}", example) == 1.0


@pytest.mark.parametrize(
    ("response", "target", "expected"),
    [
        ("1e3", "3", 0.0),
        ("1e3", "1000", 1.0),
        (".5", "5", 0.0),
        (".5", "0.5", 1.0),
        ("0", 0, 1.0),
        ("-0", 0, 1.0),
    ],
)
def test_final_number_is_not_a_fragment_of_an_exponent_or_decimal(response, target, expected):
    assert ExactMathReward().score(response, TrainingExample(prompt="q", target=target)) == expected
