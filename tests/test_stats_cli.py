"""Exercise requested CLI exports and fail-closed unsupported/invalid input."""

import json

import pandas as pd
import pytest

from utils import stats


def seed_file(root, seed, score):
    p = root / "demo" / f"seed_{seed}"
    p.mkdir(parents=True)
    (p / "metrics.jsonl").write_text(json.dumps({"accuracy": score}) + "\n")


@pytest.mark.parametrize("output_format", ["csv", "latex", "both"])
def test_requested_formats_and_consistent_summary(tmp_path, output_format, capsys):
    data = tmp_path / "data"
    output = tmp_path / "out"
    seed_file(data, 42, 0.2)
    seed_file(data, 123, 0.6)
    stats.main(
        [
            "--results-dir",
            str(data),
            "--output-dir",
            str(output),
            "--format",
            output_format,
            "--bootstrap-samples",
            "30",
        ]
    )
    assert (output / "results_table.csv").exists() == (output_format in {"csv", "both"})
    assert (output / "results_table.tex").exists() == (output_format in {"latex", "both"})
    assert "Final score: 0.4000" in capsys.readouterr().out
    if output_format in {"csv", "both"}:
        row = pd.read_csv(output / "results_table.csv").iloc[0]
        assert row["mean"] == pytest.approx(0.4)
        assert row["standard_error"] == pytest.approx(0.2)
        assert row["seeds"] == 2
        assert row["bootstrap_samples"] == 30
        assert row["ci_lower"] <= row["mean"] <= row["ci_upper"]


def test_rliable_rejected_before_output(tmp_path, capsys):
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        stats.main(["--rliable", "--output-dir", str(output)])
    assert exc.value.code == 2
    assert "not supported by this CLI" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf"), "0.2"])
def test_invalid_final_metric_not_silently_zero(tmp_path, value, capsys):
    seed_file(tmp_path / "data", 42, value)
    with pytest.raises(SystemExit) as exc:
        stats.main(["--results-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out")])
    assert exc.value.code == 2
    assert "finite numeric metric" in capsys.readouterr().err
    assert not list((tmp_path / "out").glob("results_table.*"))


@pytest.mark.parametrize("extra", [[], ["--bootstrap-samples", "0"], ["--results-dir", "missing"]])
def test_invalid_or_empty_input_rejected(tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        stats.main(["--results-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"), *extra])
    assert exc.value.code == 2
