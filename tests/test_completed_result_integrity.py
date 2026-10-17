"""A cached result must agree with its validated seed and reward history."""

import json
import socket
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("completed-result tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def _completed(tmp_path, monkeypatch):
    config = _config(tmp_path)
    dataset = grpo.InMemoryDataset(train=[_example()])
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    path = grpo._checkpoint_path(config, config.seed)
    return config, dataset, result, path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 999),
        ("seed", True),
        ("avg_first5", 123.0),
        ("avg_last10", 123.0),
        ("peak_reward", 123.0),
        ("heldout_reward", "0.9"),
        ("resumed_from_step", -1),
        ("run_id", "   "),
    ],
)
def test_contradictory_completed_result_stops_before_tracking(tmp_path, monkeypatch, field, value):
    config, dataset, _, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload["result"][field] = value
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with patch.object(
        grpo, "_start_wandb", side_effect=AssertionError("tracking started")
    ) as tracking:
        with pytest.raises(RuntimeError, match="Completed receipt"):
            grpo._run_one_seed(
                config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
            )
    tracking.assert_not_called()
    assert path.read_bytes() == before


def test_valid_completed_result_remains_readonly_and_does_not_create_provider(
    tmp_path, monkeypatch
):
    config, dataset, original, path = _completed(tmp_path, monkeypatch)
    before = path.read_bytes()
    with _runtime(monkeypatch) as holder:
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result == original
    assert "service_kwargs" not in holder
    assert path.read_bytes() == before


def test_cached_top_level_run_id_must_match_the_result(tmp_path, monkeypatch):
    config, dataset, _, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload["run_id"] = "different-synthetic-run"
    path.write_text(json.dumps(payload))
    with patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="contradictory Tinker run IDs"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()


def test_historical_summary_roundoff_is_preserved_without_rewriting(tmp_path, monkeypatch):
    config, dataset, _, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload["result"]["avg_first5"] += 1e-14
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
    assert result.avg_first5 == payload["result"]["avg_first5"]
    assert path.read_bytes() == before


def test_bad_evaluation_evidence_stops_before_tracking_and_keeps_cache(tmp_path, monkeypatch):
    config, dataset, _, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    payload["result"]["heldout_evaluation"]["status"] = "completed"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()
    assert path.read_bytes() == before
