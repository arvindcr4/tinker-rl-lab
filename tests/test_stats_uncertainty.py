"""One observation is descriptive data, not evidence of zero sampling error."""

import json

import numpy as np
import pandas as pd
import pytest

from utils import stats


@pytest.mark.parametrize("count", [0, -1, True, False, 1.5, "10", None])
@pytest.mark.parametrize("scores", [[0.5], [0.2, 0.8]])
def test_bootstrap_count_requires_a_positive_integer(count, scores):
    with pytest.raises(ValueError, match="positive integer"):
        stats.compute_bootstrap_ci(np.array(scores), n_bootstrap=count)


def test_single_observation_standard_error_is_not_zero():
    assert np.isnan(stats.standard_error(np.array([0.7])))
    errors = stats.standard_error(np.array([[0.2, 0.8]]), axis=0)
    assert errors.shape == (2,)
    assert np.isnan(errors).all()


@pytest.mark.parametrize("scores", [[], [np.nan], [np.inf], [0.2, -np.inf]])
def test_standard_error_rejects_empty_or_nonfinite_scores(scores):
    with pytest.raises(ValueError, match="standard_error requires"):
        stats.standard_error(np.array(scores))


def test_single_seed_table_labels_uncertainty_as_unavailable(tmp_path):
    table = stats.generate_results_table(
        {"single": np.array([0.7])}, output_path=str(tmp_path / "table.tex")
    )
    row = table.iloc[0]
    assert row["Seeds"] == 1
    assert "not estimable" in row["Accuracy (mean ± SE)"]
    assert "not estimable" in row["95% CI"]
    assert "± 0.000" not in (tmp_path / "table.tex").read_text()


def test_single_seed_plot_has_no_confidence_band(tmp_path, monkeypatch):
    figure, axes = stats.plt.subplots()
    monkeypatch.setattr(stats.plt, "subplots", lambda *args, **kwargs: (figure, axes))
    stats.plot_learning_curves_with_ci(
        {"single": {42: [{"reward/mean": 0.2}, {"reward/mean": 0.8}]}},
        output_path=str(tmp_path / "curve.pdf"),
    )
    assert len(axes.collections) == 0
    assert "not estimable" in axes.get_legend_handles_labels()[1][0]


def test_single_seed_cli_keeps_mean_and_labels_unavailable_uncertainty(tmp_path, capsys):
    source = tmp_path / "data" / "demo" / "seed_42"
    source.mkdir(parents=True)
    (source / "metrics.jsonl").write_text(json.dumps({"accuracy": 0.7}) + "\n")
    output = tmp_path / "out"
    stats.main(["--results-dir", str(tmp_path / "data"), "--output-dir", str(output)])
    row = pd.read_csv(output / "results_table.csv").iloc[0]
    assert row["mean"] == pytest.approx(0.7)
    assert row["uncertainty_status"] == "insufficient_seeds"
    assert np.isnan(row["standard_error"])
    assert np.isnan(row["ci_lower"])
    assert np.isnan(row["ci_upper"])
    assert "not estimable" in capsys.readouterr().out
    assert "insufficient" in (output / "results_table.tex").read_text()


@pytest.mark.parametrize("name", ["seed_bad", "seed_-1", "seed_+1", "seed_1_2", "seed_seed_1"])
@pytest.mark.parametrize("has_metrics", [True, False])
def test_invalid_seed_directories_fail_even_without_metrics(tmp_path, name, has_metrics):
    source = tmp_path / "demo" / name
    source.mkdir(parents=True)
    if has_metrics:
        (source / "metrics.jsonl").write_text('{"accuracy": 0.7}\n')
    with pytest.raises(ValueError, match="Invalid seed directory"):
        stats.load_multi_seed_results(str(tmp_path), "demo")
