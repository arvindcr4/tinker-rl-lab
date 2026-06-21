"""Unit tests for zvf_triage.core with hand-computed expected values."""

import math

import numpy as np
import pytest

from zvf_triage import core


def test_all_correct_group_counts_toward_zvf():
    # One group, both rollouts correct -> variance 0 -> zero-variance -> ZVF=1.
    rewards = [1.0, 1.0]
    group_ids = [0, 0]
    assert core.zvf(rewards, group_ids) == 1.0
    assert core.group_uniformity(rewards, group_ids) == 0.0


def test_all_wrong_group_counts_toward_zvf():
    # Both rollouts wrong -> variance 0 -> counts toward ZVF.
    assert core.zvf([0.0, 0.0], [0, 0]) == 1.0


def test_mixed_group_does_not_count():
    # One correct, one wrong -> nonzero variance -> NOT zero-variance -> ZVF=0.
    assert core.zvf([1.0, 0.0], [0, 0]) == 0.0
    assert core.group_uniformity([1.0, 0.0], [0, 0]) == 1.0


def test_two_groups_one_uniform_one_mixed():
    # Group 0: [1,1] uniform (counts). Group 1: [1,0] mixed (doesn't).
    rewards = [1.0, 1.0, 1.0, 0.0]
    group_ids = [0, 0, 1, 1]
    # 1 of 2 groups zero-variance -> ZVF = 0.5
    assert core.zvf(rewards, group_ids) == 0.5
    assert core.group_uniformity(rewards, group_ids) == 0.5


def test_four_groups_fraction():
    # 3 uniform, 1 mixed -> ZVF = 3/4.
    rewards = [
        1, 1,  # g0 uniform
        0, 0,  # g1 uniform
        1, 1,  # g2 uniform
        1, 0,  # g3 mixed
    ]
    group_ids = [0, 0, 1, 1, 2, 2, 3, 3]
    assert core.zvf(rewards, group_ids) == pytest.approx(0.75)
    assert core.gu(rewards, group_ids) == pytest.approx(0.25)


def test_empty_batch_is_fully_collapsed():
    # Empty batch -> degenerate -> ZVF = 1.0 (matches compute_zvf_and_gu).
    assert core.zvf([], []) == 1.0
    assert core.group_uniformity([], []) == 0.0


def test_singleton_group_is_zero_variance():
    # K=1 group -> variance undefined -> treated as zero-variance (ZVF counts it).
    # Paper boundary condition: "K=1 ... ZVF is trivially 1".
    assert core.zvf([1.0], [0]) == 1.0


def test_eps_tolerance_near_zero_variance():
    # Tiny variance below eps still counts as zero-variance.
    rewards = [0.5, 0.5 + 1e-9]
    assert core.zvf(rewards, [0, 0], eps=1e-6) == 1.0
    # ...but a clearly nonzero variance does not.
    assert core.zvf([0.0, 1.0], [0, 0], eps=1e-6) == 0.0


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        core.zvf([1.0, 0.0], [0])


def test_numpy_and_list_inputs_agree():
    rewards = [1.0, 0.0, 1.0, 1.0]
    gids = [0, 0, 1, 1]
    assert core.zvf(rewards, gids) == core.zvf(np.array(rewards), np.array(gids))


def test_per_prompt_zvf():
    rewards = [1.0, 1.0, 1.0, 0.0]
    gids = ["a", "a", "b", "b"]
    pp = core.per_prompt_zvf(rewards, gids)
    assert pp == {"a": 1.0, "b": 0.0}


def test_rolling_zvf_trailing_window():
    series = [0.0, 1.0, 1.0, 0.0]
    roll = core.rolling_zvf(series, window=2)
    # t0: mean[0]=0; t1: mean[0,1]=0.5; t2: mean[1,1]=1.0; t3: mean[1,0]=0.5
    np.testing.assert_allclose(roll, [0.0, 0.5, 1.0, 0.5])


def test_rolling_zvf_window_one_is_identity():
    series = [0.2, 0.4, 0.6]
    np.testing.assert_allclose(core.rolling_zvf(series, window=1), series)


def test_rolling_variance():
    series = [1.0, 1.0, 1.0]
    np.testing.assert_allclose(core.rolling_variance(series, window=3), [0.0, 0.0, 0.0])


def test_peak_to_tail_drift_hand_computed():
    # peak = 1.0, tail mean over last 2 = (0.2+0.0)/2 = 0.1
    # PTD = (1.0 - 0.1) / 1.0 = 0.9
    series = [1.0, 0.8, 0.2, 0.0]
    assert core.peak_to_tail_drift(series, tail=2) == pytest.approx(0.9)


def test_peak_to_tail_drift_no_degradation():
    # Monotone-increasing, peak is the last value -> small/zero drift.
    series = [0.1, 0.3, 0.5, 1.0]
    # peak=1.0, tail mean last 2 = (0.5+1.0)/2 = 0.75, PTD = 0.25
    assert core.peak_to_tail_drift(series, tail=2) == pytest.approx(0.25)


def test_peak_to_tail_drift_zero_peak_returns_zero():
    assert core.peak_to_tail_drift([0.0, 0.0, 0.0]) == 0.0


def test_stability_index_constant_tail_is_zero():
    assert core.stability_index([1.0, 1.0, 1.0], tail=3) == 0.0


def test_stability_index_zero_mean_returns_zero():
    assert core.stability_index([0.0, 0.0], tail=2) == 0.0


def test_gu_equals_one_minus_zvf_property():
    rng = np.random.default_rng(0)
    rewards = rng.integers(0, 2, size=40).astype(float)
    gids = np.repeat(np.arange(10), 4)
    z = core.zvf(rewards, gids)
    g = core.group_uniformity(rewards, gids)
    assert math.isclose(z + g, 1.0)
