"""Boundary regressions for shared historical-analysis helpers."""

import itertools
import math
import random
from pathlib import Path

import numpy as np
import pytest

from tests._shared_fakes import load_module

ROOT = Path(__file__).resolve().parents[1]
ac = load_module("_analysis_boundaries", ROOT / "platform_modal/scripts/_analysis_common.py")
st = load_module("_stats_boundaries", ROOT / "platform_modal/scripts/_stats.py")


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_bootstraps_reject_invalid_resample_counts(count):
    calls = [
        lambda: st.bootstrap_ci_mean_pct([1, 2], count, 0.05, 0),
        lambda: st.bootstrap_ci_statmean([1, 2], count, 0),
        lambda: st.bootstrap_ci_rng([1, 2], count, 0.05, random.Random(0)),
        lambda: st.paired_boot_pct([1], [2], count, 0),
        lambda: st.paired_step_bootstrap([], len, count, 0),
        lambda: ac.paired_bootstrap([1], [2], count, random.Random(0)),
        lambda: ac.paired_bootstrap_delta([1], [2], count, 0),
        lambda: ac.permutation_null([1, 2], [2, 1], count, 0),
        lambda: ac.bootstrap_ci_xy(np.array([1]), np.array([2]), np.mean, count),
        lambda: ac.bootstrap_ci(np.array([0]), np.array([2]), np.random.default_rng(0), count),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="positive integer"):
            call()


@pytest.mark.parametrize(("left", "right"), [([1], []), ([], [1]), ([1], [1, 2])])
def test_paired_bootstraps_never_silently_drop_observations(left, right):
    calls = [
        lambda: st.paired_boot_pct(left, right, 20, 0),
        lambda: ac.paired_bootstrap(left, right, 20, random.Random(0)),
        lambda: ac.paired_bootstrap_delta(left, right, 20, 0),
        lambda: ac.bootstrap_ci_xy(np.array(left), np.array(right), np.mean, 20),
        lambda: ac.bootstrap_ci(np.array(left), np.array(right), np.random.default_rng(0), 20),
        lambda: ac.permutation_null(left, right, 20, 0),
    ]
    for call in calls:
        with pytest.raises(ValueError, match="equal lengths"):
            call()


def test_empty_and_singleton_contracts():
    assert math.isnan(st.bootstrap_ci_mean_pct([], 20, 0.05, 0)[0])
    assert st.bootstrap_ci_statmean([], 20, 0) == (0, 0, 0)
    # Historical singleton policy intentionally differs for this helper.
    assert st.bootstrap_ci_rng([5], 20, 0.05, random.Random(0)) == (0, 0, 0)
    assert st.bootstrap_ci_statmean([5], 20, 0) == (5, 5, 5)
    assert st.paired_boot_pct([5], [2], 20, 0) == (3, 3, 3, 1)
    with pytest.raises(ValueError, match="nonempty"):
        st.paired_boot_pct([], [], 20, 0)
    assert ac.paired_bootstrap([], [], 20, random.Random(0))["n_pairs"] == 0
    assert ac.paired_bootstrap_delta([], [], 20, 0)["n"] == 0
    assert ac.paired_bootstrap_delta([5], [5], 20, 0)["p"] == 1
    assert ac.paired_bootstrap_delta([5], [6], 20, 0)["delta"] == 1


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan"), float("inf")])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        st.bootstrap_ci_mean_pct([1, 2], 20, alpha, 0)
    with pytest.raises(ValueError, match="alpha"):
        st.bootstrap_ci_rng([1, 2], 20, alpha, random.Random(0))


def test_tiny_alpha_does_not_index_past_last_resample():
    mean, lo, hi, n = st.bootstrap_ci_mean_pct([1, 2], 20, 1e-300, 0)
    assert lo <= mean <= hi
    assert n == 2


@pytest.mark.parametrize(
    ("p", "n", "z"),
    [
        (-0.1, 2, 1.96),
        (1.1, 2, 1.96),
        (0.5, 0, 1.96),
        (0.5, -1, 1.96),
        (0.5, float("inf"), 1.96),
        (float("nan"), 2, 1.96),
        (0.5, 2, 0),
        (0.5, 2, float("nan")),
    ],
)
def test_invalid_wilson_parameters(p, n, z):
    for fn in (st.wilson_centre_half, st.wilson_p, st.wilson_p_factored):
        with pytest.raises(ValueError, match="must"):
            fn(p, n, z)


def test_wilson_count_boundaries_and_empty_policy():
    assert st.wilson_ci(0, 0) == (0, 0, 0)
    for k, n in [(1, 0), (-1, 3), (4, 3), (1, -1)]:
        with pytest.raises(ValueError, match="must"):
            st.wilson_ci(k, n)
    with pytest.raises(ValueError, match="positive"):
        st.wilson(0, 0, 1.96)
    for k in (0, 1, 3):
        p, lo, hi = st.wilson(k, 3, 1.96)
        assert 0 <= lo <= p <= hi <= 1


@pytest.mark.parametrize("fn", [ac.auc_rank, ac.auroc_argsort_ranks])
def test_auc_ties_match_pairwise_wins_and_are_order_invariant(fn):
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.1, 0.9, 0.9])
    for order in itertools.permutations(range(4)):
        ix = list(order)
        assert fn(labels[ix], scores[ix]) == 0.5
    assert fn(np.array([0, 1]), np.array([2.0, 2.0])) == 0.5
    assert fn(labels, np.ones(4)) == 0.5
    assert fn(np.array([0, 1, 1]), np.array([0.0, 0.0, 1.0])) == 0.75
    assert fn(np.array([0, 1]), np.array([1.0, 0.0])) == 0.0
    assert math.isnan(fn(np.array([]), np.array([])))
    assert math.isnan(fn(np.array([1]), np.array([3.0])))


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([0], [0, 1]),
        ([0, 2], [1, 2]),
        ([0, 1], [1, float("nan")]),
        ([0, 1], [1, float("inf")]),
        ([[0, 1]], [[1, 2]]),
    ],
)
def test_auc_rejects_invalid_observations(labels, scores):
    for fn in (ac.auc_rank, ac.auroc_argsort_ranks):
        with pytest.raises(ValueError, match="must"):
            fn(np.array(labels), np.array(scores))


def test_undefined_permutation_correlation_cannot_be_reported_significant():
    with pytest.raises(ValueError, match="finite observed statistic"):
        ac.permutation_null([1, 1, 1], [1, 2, 3], 20, 0)


def test_paired_delta_rejects_multidimensional_samples():
    with pytest.raises(ValueError, match="one-dimensional"):
        ac.paired_bootstrap_delta([[1, 2]], [[2, 3]], 20, 0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_delta_does_not_become_zero_pvalue(bad):
    with pytest.raises(ValueError, match="finite"):
        ac.paired_bootstrap_delta([1, 2], [bad, 3], 20, 0)
    with pytest.raises(ValueError, match="finite"):
        ac.paired_bootstrap([1, 2], [bad, 3], 20, random.Random(0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_custom_statistic_rejected(bad):
    def undefined(values, axis=None):
        return np.full(values.shape[0], bad) if axis is not None else bad

    with pytest.raises(ValueError, match="statistic must be finite"):
        ac.paired_bootstrap_delta([1, 2], [2, 3], 20, 0, statistic=undefined)


def test_step_bootstrap_keeps_each_selected_cluster_intact():
    rows = [{"step": 1, "value": 1}, {"step": 1, "value": 2}, {"step": 2, "value": 8}]

    def cluster_sum(sample):
        ones = sum(row["value"] == 1 for row in sample)
        twos = sum(row["value"] == 2 for row in sample)
        assert ones == twos
        return sum(row["value"] for row in sample)

    result = st.paired_step_bootstrap(rows, cluster_sum, 100, 42)
    assert set(result) == {6, 11, 16}
    assert result == st.paired_step_bootstrap(rows, cluster_sum, 100, 42)
    assert st.paired_step_bootstrap(rows, lambda _: None, 10, 0) == []
    assert st.paired_step_bootstrap(rows, lambda _: float("nan"), 10, 0) == []


def test_valid_bootstrap_intervals_and_pooled_effect_size():
    values = [1, 2, 4, 8]
    mean, lo, hi, n = st.bootstrap_ci_mean_pct(values, 100, 0.05, 42)
    assert mean == 3.75
    assert n == 4
    assert lo <= mean <= hi
    assert (mean, lo, hi, n) == st.bootstrap_ci_mean_pct(values, 100, 0.05, 42)
    for result in (
        st.bootstrap_ci_statmean(values, 100, 42),
        st.bootstrap_ci_rng(values, 100, 0.05, random.Random(42)),
    ):
        assert result[1] <= result[0] <= result[2]
    assert st.cohens_d_pstdev_pooled([1, 2, 3], [2, 3, 4]) == pytest.approx(-math.sqrt(1.5))
    assert math.isnan(st.cohens_d_pstdev_pooled([1], [2]))
    assert math.isnan(st.cohens_d_pstdev_pooled([1, 1], [2, 2]))
    assert st.log_beta(1, 1) == 0
    assert st.pow_root(4) == 2


def test_valid_permutation_null_has_probability_in_range():
    result = ac.permutation_null([1, 2, 3, 4], [4, 3, 2, 1], 40, 42)
    assert result["obs_rho"] == -1
    assert 0 < result["p_perm"] <= 1
    assert result["B"] == 40
    assert result["n"] == 4
    assert result["null_q025"] <= result["null_q500"] <= result["null_q975"]


def test_bootstrap_xy_preserves_pairing_and_failure_contract():
    x = np.arange(4)
    y = x + 3

    def diff(a, b):
        return float(np.mean(b - a))

    assert ac.bootstrap_ci_xy(x, y, diff, 100) == (3, 3, 3)

    def undefined(a, b):
        raise ValueError("undefined statistic")

    assert all(math.isnan(v) for v in ac.bootstrap_ci_xy(x, y, undefined, 20))
    assert all(
        math.isnan(v)
        for v in ac.bootstrap_ci(np.ones(4), np.arange(4), np.random.default_rng(0), 20)
    )


def test_ccf_lags_beyond_series_length_have_zero_overlap():
    x = np.arange(5.0)
    result = ac.ccf_at_lags(x, x, 6)
    assert result.shape == (13,)
    assert result[6] == pytest.approx(1)
    assert np.array_equal(result[:4], np.zeros(4))
    assert np.array_equal(result[-4:], np.zeros(4))
    assert np.array_equal(ac.ccf_at_lags([], [], 2), np.zeros(5))
    assert np.array_equal(ac.ccf_at_lags([1], [2], 2), np.zeros(5))


@pytest.mark.parametrize("lag", [-1, 0.5, True])
def test_ccf_rejects_invalid_lags(lag):
    with pytest.raises(ValueError, match="nonnegative integer"):
        ac.ccf_at_lags([1, 2], [2, 3], lag)


@pytest.mark.parametrize(
    ("x", "y"), [([1], [1, 2]), ([1, float("nan")], [1, 2]), ([[1, 2]], [[1, 2]])]
)
def test_ccf_rejects_invalid_paired_inputs(x, y):
    with pytest.raises(ValueError, match="must"):
        ac.ccf_at_lags(x, y, 2)
