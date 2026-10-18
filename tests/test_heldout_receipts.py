"""Offline evaluation receipts retain denominator and failure evidence."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime, _Seq


@pytest.mark.parametrize(
    ("enabled", "total", "failed", "status"),
    [
        (False, 2, 0, "disabled"),
        (True, 0, 0, "empty"),
        (True, 1, 0, "completed"),
        (True, 2, 1, "partial_failure"),
        (True, 2, 2, "all_failed"),
    ],
)
def test_heldout_receipt_records_outcome_without_repeating_evaluation(
    tmp_path, monkeypatch, enabled, total, failed, status
):
    calls = []

    def sample(_prompt, n, _params):
        if n == 1:
            calls.append(n)
            if len(calls) <= failed:
                raise RuntimeError("synthetic evaluation failure")
            return SimpleNamespace(sequences=[_Seq([1])])
        return SimpleNamespace(sequences=[_Seq([1]), _Seq([2])])

    config = _config(tmp_path, evaluate_heldout=enabled)
    dataset = grpo.InMemoryDataset(train=[_example()], test=[_example()] * total)
    with _runtime(monkeypatch, sample_fn=sample) as holder:
        with patch.object(
            grpo, "heldout_reward_summary", wraps=grpo.heldout_reward_summary
        ) as summarize:
            result = grpo._run_one_seed(
                config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
            )
    receipt = result.heldout_evaluation
    assert receipt["schema_version"] == 1
    assert receipt["status"] == status
    assert receipt["total_n"] == (total if enabled else 0)
    assert receipt["failed_n"] == (failed if enabled else 0)
    assert len(calls) == (total if enabled else 0)
    assert summarize.call_count == int(enabled and total > 0)
    if enabled and total:
        summary = next(row for row in holder["run"].logs if "test/failed_n" in row)
        assert receipt["summary"] == summary
        assert result.heldout_reward == (total - failed) / total
    else:
        assert receipt["summary"] == {}
        assert result.heldout_reward is None
    path = grpo._checkpoint_path(config, config.seed)
    persisted = json.loads(path.read_text())["result"]["heldout_evaluation"]
    assert persisted == receipt
    before = path.read_bytes()
    with _runtime(monkeypatch, sample_fn=sample) as replay:
        repeated = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert repeated == result
    assert "service_kwargs" not in replay
    assert len(calls) == (total if enabled else 0)
    assert path.read_bytes() == before
    receipt["summary"]["synthetic_mutation"] = 1
    assert (
        "synthetic_mutation"
        not in json.loads(path.read_text())["result"]["heldout_evaluation"]["summary"]
    )


def test_legacy_completed_receipt_does_not_invent_evaluation_evidence(tmp_path, monkeypatch):
    config = _config(tmp_path)
    dataset = grpo.InMemoryDataset(train=[_example()])
    with _runtime(monkeypatch):
        grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    path = grpo._checkpoint_path(config, config.seed)
    payload = json.loads(path.read_text())
    payload["result"].pop("heldout_evaluation", None)
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with _runtime(monkeypatch) as holder:
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.heldout_evaluation == {}
    assert "service_kwargs" not in holder
    assert path.read_bytes() == before


def test_tracking_callback_cannot_mutate_local_evaluation_summary(tmp_path, monkeypatch):
    def sample(_prompt, n, _params):
        return SimpleNamespace(sequences=[_Seq([1])] * n)

    config = _config(tmp_path, evaluate_heldout=True)
    dataset = grpo.InMemoryDataset(train=[_example()], test=[_example()])
    with _runtime(monkeypatch, sample_fn=sample) as holder:
        original = holder["run"].log

        def mutating_log(payload):
            original(dict(payload))
            if "test/failed_n" in payload:
                payload["test/failed_n"] = 999.0
            return True

        holder["run"].log = mutating_log
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.heldout_evaluation["summary"]["test/failed_n"] == 0.0
