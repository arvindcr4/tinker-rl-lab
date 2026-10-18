"""Regression counterexamples for seed aggregation and rank-trimmed IQM."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import trim_mean

from utils import stats


def write_seed(root, seed, rows):
    folder = root / "demo" / f"seed_{seed}"
    folder.mkdir(parents=True)
    (folder / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_cli(tmp_path, seeds, *extra):
    for seed, rows in enumerate(seeds):
        write_seed(tmp_path / "data", seed, rows)
    stats.main(
        [
            "--legacy-exploratory",
            "--results-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "out"),
            "--format",
            "csv",
            "--bootstrap-samples",
            "10",
            *extra,
        ]
    )
    return pd.read_csv(tmp_path / "out" / "results_table.csv").iloc[0]


@pytest.mark.parametrize("scores", [[0, 1, 1, 1, 2, 3, 4, 5], [0, 1, 10]])
def test_iqm_trims_ranks_including_ties_and_small_samples(scores):
    assert stats.compute_iqm(scores) == pytest.approx(trim_mean(scores, 0.25))


@pytest.mark.parametrize("tau", [-0.1, 0.5, 0.9, np.nan, np.inf])
def test_iqm_rejects_invalid_trim_fraction(tau):
    with pytest.raises(ValueError, match="tau"):
        stats.compute_iqm([1, 2, 3], tau=tau)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("function", [stats.compute_iqm, stats.compute_bootstrap_ci])
def test_estimators_reject_nonfinite_scores(function, value):
    with pytest.raises(ValueError, match="finite"):
        function([0.5, value])


def test_loader_rejects_empty_seed(tmp_path):
    write_seed(tmp_path, 1, [])
    with pytest.raises(ValueError, match="Empty.*seed"):
        stats.load_multi_seed_results(str(tmp_path), "demo")


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ([[{"accuracy": 0.8}], []], "Empty"),
        ([[{"accuracy": 0.8}], [{"reward/mean": 0.2}]], "metric"),
        ([[{"accuracy": 0.8, "step": 100}], [{"accuracy": 0.2, "step": 1}]], "step"),
        ([[{"accuracy": 0.8, "step": 1}], [{"accuracy": 0.2}]], "step"),
        ([[{"accuracy": 0.8}], [{"accuracy": 0.2}, {"accuracy": 0.3}]], "observation"),
        ([[{"accuracy": 0.8, "step": 1}], [{"accuracy": 0.2, "step": True}]], "step"),
        ([[{"accuracy": 0.8, "step": 1}], [{"accuracy": 0.2, "step": np.inf}]], "step"),
        (
            [
                [{"accuracy": 0.8, "step": 1}],
                [{"accuracy": 0.2, "step": 2}, {"accuracy": 0.3, "step": 1}],
            ],
            "step",
        ),
    ],
)
def test_cli_rejects_incomplete_or_incomparable_seeds(tmp_path, seeds, message, capsys):
    with pytest.raises(SystemExit) as exc:
        run_cli(tmp_path, seeds)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out" / "results_table.csv").exists()


def test_explicit_metric_selects_same_quantity_and_records_horizon(tmp_path):
    row = run_cli(
        tmp_path,
        [
            [{"accuracy": 0.8, "reward/mean": 0.1, "step": 10}],
            [{"accuracy": 0.2, "step": 1}, {"accuracy": 0.4, "step": 10}],
        ],
        "--metric",
        "accuracy",
    )
    assert row["mean"] == pytest.approx(0.6)
    assert row["metric"] == "accuracy"
    assert row["final_step"] == 10
    assert row["step_mode"] == "explicit"


def test_legacy_auto_metric_records_observation_horizon(tmp_path):
    row = run_cli(tmp_path, [[{"accuracy": 0.8}], [{"accuracy": 0.2}]])
    assert row["mean"] == 0.5
    assert row["metric"] == "accuracy"
    assert row["final_step"] == 1
    assert row["step_mode"] == "observation_index"


def test_explicit_missing_metric_does_not_fall_back(tmp_path, capsys):
    with pytest.raises(SystemExit):
        run_cli(tmp_path, [[{"accuracy": 0.8}], [{"reward/mean": 0.2}]], "--metric", "accuracy")
    assert "finite numeric metric" in capsys.readouterr().err


@pytest.mark.parametrize("step_key", ["global_step", "_step", "optimizer_step"])
def test_unhandled_step_metadata_is_not_observation_index(tmp_path, step_key, capsys):
    with pytest.raises(SystemExit) as failure:
        run_cli(tmp_path, [[{"accuracy": 0.8, step_key: 100}], [{"accuracy": 0.2, step_key: 1}]])
    assert failure.value.code == 2
    assert "unsupported step metadata" in capsys.readouterr().err


def test_auto_metric_uses_common_key(tmp_path):
    row = run_cli(tmp_path, [[{"accuracy": 0.8, "reward/mean": 0.1}], [{"accuracy": 0.2}]])
    assert row["metric"] == "accuracy"
    assert row["mean"] == pytest.approx(0.5)


def test_loader_rejects_seed_directory_without_metrics(tmp_path):
    write_seed(tmp_path, 1, [{"accuracy": 0.5}])
    (tmp_path / "demo" / "seed_2").mkdir()
    with pytest.raises(ValueError, match="Missing metric source"):
        stats.load_multi_seed_results(str(tmp_path), "demo")


def test_missing_common_metric_requests_explicit_selection(tmp_path, capsys):
    with pytest.raises(SystemExit) as failure:
        run_cli(tmp_path, [[{"accuracy": 0.8}], [{"reward/mean": 0.2}]])
    assert failure.value.code == 2
    assert "--metric is required" in capsys.readouterr().err
