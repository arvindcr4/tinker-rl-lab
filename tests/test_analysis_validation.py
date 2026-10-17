"""Counterexamples for analysis ingestion, effect sizes and plotted observations."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from utils import stats
from utils.verify_results import _parse_result_file, verify


@pytest.mark.parametrize(
    "paths", [("seed_42/a.jsonl", "seed_42/z.jsonl"), ("seed_042/a.jsonl", "seed_42/a.jsonl")]
)
def test_duplicate_seed_sources_rejected(tmp_path, paths):
    for path in paths:
        file = tmp_path / "demo" / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{"accuracy": 0.1}\n')
    with pytest.raises(ValueError, match="Ambiguous metric sources"):
        stats.load_multi_seed_results(str(tmp_path), "demo")


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ([0, 1], [2, 3], -2.82842712474619),
        ([0, 2], [2, 4, 6], -3 / np.sqrt(10 / 3)),
        ([1, 1], [0, 2], 0),
    ],
)
def test_sample_pooled_cohens_d(a, b, expected):
    assert stats.welch_ttest(a, b)["effect_size_cohens_d"] == pytest.approx(expected)
    assert stats.welch_ttest(b, a)["effect_size_cohens_d"] == pytest.approx(-expected)


@pytest.mark.parametrize(
    "token", ["0.344e2", "0.344junk", "nan", "inf", "1e309", "-0.1", "1.01", "0.3 0.4"]
)
def test_log_invalid_complete_values(tmp_path, token):
    path = tmp_path / "demo_s42.log"
    path.write_text(f"[grpo_cli] Seed 42 done.\navg_last10 : {token}\npeak_reward : 0.625\n")
    with pytest.raises(ValueError, match="numeric|finite|grid|steps|step"):
        verify(tmp_path, {"demo": {"seed": 42, "last10": 0.344, "peak": 0.625}}, 0.05, 0.1)


@pytest.mark.parametrize("token", ["3.44e-1", "+0.344", ".344", "0.3440"])
def test_log_valid_complete_values(tmp_path, token):
    path = tmp_path / "demo_s42.log"
    path.write_text(f"[grpo_cli] Seed 42 done.\navg_last10 : {token}\npeak_reward : 6.25e-1\n")
    assert _parse_result_file(path)[0]["last10_avg"] == pytest.approx(0.344)


def test_legacy_percent_exponents(tmp_path):
    path = tmp_path / "demo_s42.log"
    path.write_text("Last-10 avg accuracy: 3.44e1%\nPeak accuracy: 62.5%\n")
    assert _parse_result_file(path)[0]["last10_avg"] == pytest.approx(0.344)


@pytest.mark.parametrize(
    "second",
    [
        [{"accuracy": 0.8}],
        [{"reward/mean": float("nan")}],
        [{"reward/mean": True}],
        [{"step": 500, "reward/mean": 0.8}],
        [{"step": 100, "reward/mean": 0.8}, {"step": 100, "reward/mean": 0.9}],
        [{"step": 200, "reward/mean": 0.8}, {"step": 100, "reward/mean": 0.9}],
        [{"step": 100, "reward/mean": 0.8}, {"reward/mean": 0.9}],
    ],
)
def test_invalid_plot_data_rejected(tmp_path, second):
    with pytest.raises(ValueError, match="numeric|finite|grid|steps|step"):
        stats.plot_learning_curves_with_ci(
            {"a": {42: [{"step": 100, "reward/mean": 1}], 43: second}},
            output_path=str(tmp_path / "bad.pdf"),
        )
    assert not (tmp_path / "bad.pdf").exists()


@pytest.mark.parametrize("explicit", [True, False])
def test_plot_preserves_coordinates_and_labels(monkeypatch, explicit):
    axes = MagicMock()
    monkeypatch.setattr(stats.plt, "subplots", lambda *a, **kw: (MagicMock(), axes))
    for name in ["tight_layout", "savefig", "close"]:
        monkeypatch.setattr(stats.plt, name, MagicMock())
    records = [{"reward/mean": np.float32(1)}, {"reward/mean": np.float64(0.5)}]
    if explicit:
        for row, step in zip(records, [np.int64(100), np.int64(500)]):
            row["step"] = step
    stats.plot_learning_curves_with_ci({"a": {42: records}})
    assert axes.plot.call_args.args[0] == ([100, 500] if explicit else [1, 2])
    np.testing.assert_equal(axes.plot.call_args.args[1], [1, 0.5])
    assert axes.set_xlabel.call_args.args[0] == (
        "Training Step" if explicit else "Observation index"
    )


def test_unequal_legacy_lengths_rejected(tmp_path):
    with pytest.raises(ValueError, match="same step grid"):
        stats.plot_learning_curves_with_ci(
            {"a": {1: [{"reward/mean": 1}], 2: [{"reward/mean": 1}] * 2}}
        )


def test_loader_rejects_nonobject_record(tmp_path):
    folder = tmp_path / "demo" / "seed_1"
    folder.mkdir(parents=True)
    (folder / "m.jsonl").write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="must be an object"):
        stats.load_multi_seed_results(str(tmp_path), "demo")
