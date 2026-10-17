"""Finite advantages even when binary64 moment intermediates overflow."""

import math
import random
import sys
from decimal import localcontext

import pytest

from platform_tinker.tinkerrl.grpo import normalize_advantages_global, normalize_rewards


@pytest.mark.parametrize("normalize", [normalize_rewards, normalize_advantages_global])
@pytest.mark.parametrize(
    ("rewards", "epsilon", "expected"),
    [
        ([1e200, -1e200], 1e-8, [1.0, -1.0]),
        ([1e308, 9e307], 1e-8, [1.0, -1.0]),
        ([sys.float_info.max, -sys.float_info.max], sys.float_info.max, [0.5, -0.5]),
        ([1e-200, -1e-200], 1e-300, [1.0, -1.0]),
        (
            [sys.float_info.max] + [-sys.float_info.max] * 3,
            1e-8,
            [math.sqrt(3)] + [-1 / math.sqrt(3)] * 3,
        ),
    ],
    ids=[
        "variance-overflow",
        "mean-overflow",
        "denominator-overflow",
        "variance-underflow",
        "centering-overflow",
    ],
)
def test_standardized_extremes_have_representable_answers(normalize, rewards, epsilon, expected):
    assert normalize(rewards, epsilon=epsilon) == pytest.approx(expected, rel=1e-14)


@pytest.mark.parametrize("normalize", [normalize_rewards, normalize_advantages_global])
def test_unbiased_mean_overflow_recovers_but_true_output_overflow_raises(normalize):
    assert normalize([1e308, 9e307], unbiased=True) == pytest.approx([5e306, -5e306])
    with pytest.raises(ValueError, match="finite floats"):
        normalize([sys.float_info.max] + [-sys.float_info.max] * 3, unbiased=True)
    # Exact cancellation must not discard the small reward in the mean.
    result = normalize([1e308, 1e308, 1e-300, -1e308, -1e308], unbiased=True)
    assert result[2] == pytest.approx(8e-301, rel=1e-14, abs=0)


def test_global_excluded_outlier_keeps_basis_and_detects_unrepresentable_output():
    assert normalize_advantages_global(
        [-1e-300, 1e-300, 1e8], epsilon=1e-300, exclude=[False, False, True]
    ) == pytest.approx([-0.5, 0.5, 5e307])
    with pytest.raises(ValueError, match="finite floats"):
        normalize_advantages_global([0.0, 1e308], exclude=[False, True])


def test_ordinary_binary64_arithmetic_is_unchanged():
    rng = random.Random(41)
    for _ in range(50):
        rewards = [rng.uniform(-10, 10) for _ in range(7)]
        for unbiased in (False, True):
            mean = math.fsum(rewards) / len(rewards)
            std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
            expected = [r - mean if unbiased else (r - mean) / std for r in rewards]
            assert normalize_rewards(rewards, unbiased=unbiased) == expected
            assert normalize_advantages_global(rewards, unbiased=unbiased) == expected
            basis = rewards[::2]
            mean = math.fsum(basis) / len(basis)
            std = (sum((r - mean) ** 2 for r in basis) / len(basis)) ** 0.5 + 1e-8
            expected = [r - mean if unbiased else (r - mean) / std for r in rewards]
            assert (
                normalize_advantages_global(
                    rewards,
                    unbiased=unbiased,
                    exclude=[False, True, False, True, False, True, False],
                )
                == expected
            )


def test_fallback_is_independent_of_callers_decimal_context():
    with localcontext() as context:
        context.prec = 3
        context.Emax = 9
        assert normalize_rewards([1e308, 9e307]) == pytest.approx([1, -1])
        assert context.prec == 3
        assert context.Emax == 9
