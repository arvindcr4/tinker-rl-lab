"""Regression tests for utils/stats.py input guards (empty input, n=1)."""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import stats


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        stats.compute_bootstrap_ci(np.array([]))


def test_bootstrap_ci_single_value_collapses():
    mean, lo, hi = stats.compute_bootstrap_ci(np.array([0.75]))
    assert mean == pytest.approx(0.75)
    assert lo == pytest.approx(0.75)
    assert hi == pytest.approx(0.75)


def test_bootstrap_ci_normal_case_unchanged():
    rng = np.random.default_rng(0)
    mean, lo, hi = stats.compute_bootstrap_ci(
        np.array([0.5, 0.6, 0.7, 0.8]), n_bootstrap=500, rng=rng
    )
    assert mean == pytest.approx(0.65)
    assert lo <= mean <= hi


def test_welch_ttest_single_observation_raises():
    with pytest.raises(ValueError):
        stats.welch_ttest(np.array([1.0]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        stats.welch_ttest(np.array([]), np.array([1.0, 2.0]))


def test_welch_ttest_constant_inputs_finite_effect():
    out = stats.welch_ttest(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    assert np.isfinite(out["effect_size_cohens_d"])


def test_mann_whitney_u_empty_raises():
    with pytest.raises(ValueError):
        stats.mann_whitney_u(np.array([]), np.array([1.0, 2.0]))


def test_generate_results_table_single_seed():
    df = stats.generate_results_table(
        {"algo": np.array([0.9])}, output_path="/tmp/test_stats_single.tex"
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
    with pytest.raises(ValueError):
        stats.plot_learning_curves_with_ci(
            {"algo": {}},
            output_path=str(tmp_path / "curves.pdf"),
        )


def test_load_multi_seed_results_skips_bad_dirs(tmp_path):
    exp = tmp_path / "exp"
    good = exp / "seed_1"
    bad = exp / "seed_notanint"
    good.mkdir(parents=True)
    bad.mkdir(parents=True)
    with open(good / "m.jsonl", "w") as f:
        f.write(json.dumps({"reward/mean": 1.0}) + "\n\n")
    with open(bad / "m.jsonl", "w") as f:
        f.write(json.dumps({"reward/mean": 0.0}) + "\n")
    out = stats.load_multi_seed_results(str(tmp_path), "exp")
    assert list(out.keys()) == [1]
    assert out[1] == [{"reward/mean": 1.0}]


def test_compute_iqm_empty_is_nan():
    assert np.isnan(stats.compute_iqm(np.array([])))
