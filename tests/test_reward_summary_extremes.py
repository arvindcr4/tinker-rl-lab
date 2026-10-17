"""Finite reward reporting must survive intermediate binary64 overflow."""

import json
import math
from decimal import localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_caliber_fixes import PROMPT, _runtime, _seq
from utils.stats import compute_bootstrap_ci


@pytest.mark.parametrize("dynamic", [False, True])
def test_extreme_reward_run_keeps_trace_logs_and_checkpoint_finite(tmp_path, monkeypatch, dynamic):
    class Reward:
        def score(self, text, example):
            return 1e308 if text == "1" else 9e307

    heldout_index = 0

    def sample(_prompt, num_samples, _params):
        nonlocal heldout_index
        if num_samples == 1:
            heldout_index += 1
            return SimpleNamespace(sequences=[_seq([heldout_index, 7])])
        return SimpleNamespace(sequences=[_seq([1, 7]), _seq([2, 7])])

    config = _config(
        tmp_path, name="finite-mean", steps=2, dynamic_sampling=dynamic, evaluate_heldout=True
    )
    with _runtime(monkeypatch, sample_fn=sample) as holder:
        result = grpo._run_one_seed(
            config,
            grpo.InMemoryDataset(train=[_example()], test=[_example(), _example()]),
            Reward(),
            _Tokenizer(tokens=PROMPT, decode_first_only=True),
            logger=lambda _: None,
        )
    assert holder["optim_calls"] == 2
    assert result.reward_trace == pytest.approx([9.5e307, 9.5e307])
    assert result.avg_first5 == pytest.approx(9.5e307)
    assert result.avg_last10 == pytest.approx(9.5e307)
    assert result.heldout_reward == pytest.approx(9.5e307)
    for row in holder["logs"]:
        for key, value in row.items():
            if "reward" in key:
                assert math.isfinite(value), key
    checkpoint = json.loads(
        (tmp_path / "finite-mean_seed42.json").read_text(),
        parse_constant=lambda value: pytest.fail(value),
    )
    assert checkpoint
    assert (
        grpo._load_checkpoint(config, config.seed)["result"]["reward_trace"] == result.reward_trace
    )


@pytest.mark.parametrize("values", [[1e308, 9e307], [-1e308, -9e307], [1e308, -1e308]])
def test_bootstrap_finite_extremes(values):
    mean, lower, upper = compute_bootstrap_ci(values, n_bootstrap=40)
    assert all(math.isfinite(value) for value in (mean, lower, upper))
    assert min(values) <= lower <= upper <= max(values)
    assert mean == pytest.approx((values[0] / 2) + (values[1] / 2))


def test_bootstrap_interpolation_between_opposite_extremes():
    class FixedResamples:
        def __init__(self):
            self.index = 0

        def choice(self, scores, size, replace):
            value = scores[self.index]
            self.index += 1
            return np.full(size, value)

    mean, lower, upper = compute_bootstrap_ci(
        [-1e308, 1e308], n_bootstrap=2, confidence=0.5, rng=FixedResamples()
    )
    assert (mean, lower, upper) == pytest.approx((0, -5e307, 5e307))


def test_bootstrap_ordinary_arithmetic_and_rng_are_unchanged():
    values = np.array([0.1, 0.2, 0.7, 1.0])
    rng = np.random.default_rng(91)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(101)]
    alpha = (1 - 0.95) / 2
    expected = (
        np.mean(values),
        np.percentile(means, 100 * alpha),
        np.percentile(means, 100 * (1 - alpha)),
    )
    actual = compute_bootstrap_ci(values, n_bootstrap=101, rng=np.random.default_rng(91))
    assert actual == expected


def test_reward_mean_overflow_cancellation_is_independent_of_decimal_context():
    values = [1e308, 1e308, -1e308, -1e308, 1e-300]
    with localcontext() as context:
        context.prec = 2
        context.Emax = 9
        assert grpo._reward_mean(values) == 2e-301
    ordinary = [0.1, 0.2, 0.7]
    assert grpo._reward_mean(ordinary) == sum(ordinary) / len(ordinary)


def test_bootstrap_constant_max_float_does_not_overflow():
    maximum = np.finfo(float).max
    assert compute_bootstrap_ci([maximum, maximum], n_bootstrap=3) == (maximum,) * 3
