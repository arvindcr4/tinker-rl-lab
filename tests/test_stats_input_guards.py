"""Regression tests for utils/stats.py input guards (empty input, n=1)."""

import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from utils import stats


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError, match="at least one score"):
        stats.compute_bootstrap_ci(np.array([]))


def test_bootstrap_ci_single_value_has_unestimable_uncertainty():
    mean, lo, hi = stats.compute_bootstrap_ci(np.array([0.75]))
    assert mean == pytest.approx(0.75)
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_welch_ttest_single_observation_raises():
    with pytest.raises(ValueError, match="at least two scores"):
        stats.welch_ttest(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at least two scores"):
        stats.welch_ttest(np.array([]), np.array([1.0, 2.0]))


def test_welch_ttest_constant_inputs_finite_effect():
    out = stats.welch_ttest(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(out["effect_size_cohens_d"])


def test_mann_whitney_u_empty_raises():
    with pytest.raises(ValueError, match="at least one score per group"):
        stats.mann_whitney_u(np.array([]), np.array([1.0, 2.0]))


@pytest.mark.parametrize(("n_a", "n_b"), [(1, 1), (2, 3), (8, 12)])
def test_mann_whitney_u_all_tied_is_not_significant(n_a, n_b):
    out = stats.mann_whitney_u(np.full(n_a, 0.5), np.full(n_b, 0.5))
    assert out["u_statistic"] == n_a * n_b / 2
    assert out["p_value"] == 1.0
    assert out["significant_at_005"] is False


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mann_whitney_u_rejects_nonfinite_scores(value):
    with pytest.raises(ValueError, match="finite scores"):
        stats.mann_whitney_u(np.array([value, value]), np.array([value, value]))


def test_generate_results_table_single_seed(tmp_path):
    df = stats.generate_results_table(
        {"algo": np.array([0.9])}, output_path=str(tmp_path / "test_stats_single.tex")
    )
    assert len(df) == 1
    assert df.iloc[0]["Seeds"] == 1


def test_plot_learning_curves_single_seed(tmp_path):
    out = str(tmp_path / "curves.pdf")
    stats.plot_learning_curves_with_ci(
        {"algo": {42: [{"reward/mean": 0.5}, {"reward/mean": 0.7}]}},
        metric_key="reward/mean",
        output_path=out,
    )
    assert os.path.exists(out)


def test_plot_learning_curves_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="No curves to plot"):
        stats.plot_learning_curves_with_ci(
            {"algo": {}},
            output_path=str(tmp_path / "curves.pdf"),
        )


def test_load_multi_seed_results_rejects_bad_dirs(tmp_path):
    exp = tmp_path / "exp"
    good = exp / "seed_1"
    bad = exp / "seed_notanint"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    with open(good / "m.jsonl", "w") as f:
        f.write(json.dumps({"reward/mean": 1.0}) + "\n\n")
    with open(bad / "m.jsonl", "w") as f:
        f.write(json.dumps({"reward/mean": 0.0}) + "\n")
    with pytest.raises(ValueError, match="Invalid seed directory"):
        stats.load_multi_seed_results(str(tmp_path), "exp")
