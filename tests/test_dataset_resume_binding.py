"""Resume must not mix a newly fingerprinted run with different inputs."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _Tokenizer
from tests.test_grpo_coverage import _runtime


def examples():
    return [
        grpo.TrainingExample(prompt="synthetic-a", target="1", metadata={"id": "a"}),
        grpo.TrainingExample(prompt="synthetic-b", target="1", metadata={"id": "b"}),
    ]


def pause_after_one(tmp_path, monkeypatch, rows):
    config = _config(tmp_path, steps=2, save_every=1)
    saved = {}
    write = grpo._write_checkpoint

    def capture(path, payload):
        if payload.get("status") == "started" and payload.get("step") == 1:
            saved.update(deepcopy(payload))
        write(path, payload)

    with _runtime(monkeypatch), patch.object(grpo, "_write_checkpoint", side_effect=capture):
        grpo._run_one_seed(
            config,
            grpo.InMemoryDataset(train=rows),
            grpo.ExactMathReward(),
            _Tokenizer(),
            logger=lambda _: None,
        )
    path = grpo._checkpoint_path(config, config.seed)
    write(path, saved)
    return config, path, saved


@pytest.mark.parametrize("change", ["prompt", "target", "metadata", "order", "non_json"])
def test_changed_training_snapshot_fails_before_tracking(tmp_path, monkeypatch, change):
    rows = examples()
    config, path, _ = pause_after_one(tmp_path, monkeypatch, rows)
    before = path.read_bytes()
    if change == "order":
        rows.reverse()
    elif change == "non_json":
        rows[0].target = object()
    else:
        setattr(rows[0], change, {"changed": True} if change == "metadata" else "changed")
    with _runtime(monkeypatch), patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="training input fingerprint"):
            grpo._run_one_seed(
                config,
                grpo.InMemoryDataset(train=rows),
                grpo.ExactMathReward(),
                _Tokenizer(),
                logger=lambda _: None,
            )
    tracking.assert_not_called()
    assert path.read_bytes() == before


def test_unchanged_resume_keeps_original_fingerprint_scope(tmp_path, monkeypatch):
    rows = examples()
    config, path, saved = pause_after_one(tmp_path, monkeypatch, rows)
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config,
            grpo.InMemoryDataset(train=rows),
            grpo.ExactMathReward(),
            _Tokenizer(),
            logger=lambda _: None,
        )
    assert result.dataset_source == saved["dataset_source"]
    assert result.dataset_source["status"] == "fingerprinted"
    assert result.dataset_source["fingerprinted_from_step"] == 0
    assert result.dataset_source["train_count"] == 2
    assert len(result.dataset_source["train_sha256"]) == 64
    assert "synthetic-a" not in json.dumps(result.dataset_source)
    assert json.loads(path.read_text())["result"]["dataset_source"] == result.dataset_source


def test_legacy_partial_resume_only_records_new_scope(tmp_path, monkeypatch):
    rows = examples()
    config, path, saved = pause_after_one(tmp_path, monkeypatch, rows)
    saved.pop("dataset_source", None)
    grpo._write_checkpoint(path, saved)
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config,
            grpo.InMemoryDataset(train=rows),
            grpo.ExactMathReward(),
            _Tokenizer(),
            logger=lambda _: None,
        )
    assert result.dataset_source["fingerprinted_from_step"] == 1


@pytest.mark.parametrize("target", [(1, 2), float("nan")])
def test_unverifiable_custom_values_are_explicitly_unrecorded(tmp_path, monkeypatch, target):
    config = _config(tmp_path, steps=0)
    data = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="synthetic", target=target)])
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.dataset_source["status"] == "unrecorded"
    assert result.dataset_source["train_sha256"] is None
    assert result.dataset_source["fingerprinted_from_step"] is None


def test_completed_legacy_receipt_does_not_invent_source_binding(tmp_path, monkeypatch):
    config = _config(tmp_path)
    data = grpo.InMemoryDataset(train=examples())
    with _runtime(monkeypatch):
        grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    path = grpo._checkpoint_path(config, config.seed)
    payload = json.loads(path.read_text())
    payload["result"].pop("dataset_source", None)
    grpo._write_checkpoint(path, payload)
    before = path.read_bytes()
    with _runtime(monkeypatch) as holder:
        result = grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.dataset_source == {}
    assert "service_kwargs" not in holder
    assert path.read_bytes() == before


def test_fingerprint_is_canonical_typed_and_rejects_cycles():
    first = grpo.TrainingExample(prompt="q", target={"b": 2, "a": 1})
    second = grpo.TrainingExample(prompt="q", target={"a": 1, "b": 2})
    assert grpo._training_input_fingerprint([first]) == grpo._training_input_fingerprint([second])
    fingerprints = {
        grpo._training_input_fingerprint([grpo.TrainingExample(prompt="q", target=value)])
        for value in (None, True, 1, 1.0, "1")
    }
    assert len(fingerprints) == 5
    assert None not in fingerprints
    cycle = []
    cycle.append(cycle)
    assert (
        grpo._training_input_fingerprint([grpo.TrainingExample(prompt="q", target=cycle)]) is None
    )
    assert cycle[0] is cycle


def test_malformed_source_receipts_fail_before_tracking(tmp_path, monkeypatch):
    rows = examples()
    config, path, saved = pause_after_one(tmp_path, monkeypatch, rows)
    for field, value in (
        ("schema_version", True),
        ("schema_version", 2),
        ("status", "unknown"),
        ("train_count", True),
        ("fingerprinted_from_step", -1),
        ("train_sha256", None),
    ):
        damaged = deepcopy(saved)
        damaged["dataset_source"][field] = value
        grpo._write_checkpoint(path, damaged)
        before = path.read_bytes()
        with patch.object(grpo, "_start_wandb") as tracking:
            with pytest.raises(RuntimeError, match="training input fingerprint"):
                grpo._run_one_seed(
                    config, grpo.InMemoryDataset(train=rows), grpo.ExactMathReward(), _Tokenizer()
                )
        tracking.assert_not_called()
        assert path.read_bytes() == before


def test_completed_bound_receipt_cannot_be_reused_with_other_inputs(tmp_path, monkeypatch):
    config = _config(tmp_path)
    rows = examples()
    with _runtime(monkeypatch):
        grpo._run_one_seed(
            config,
            grpo.InMemoryDataset(train=rows),
            grpo.ExactMathReward(),
            _Tokenizer(),
            logger=lambda _: None,
        )
    path = grpo._checkpoint_path(config, config.seed)
    before = path.read_bytes()
    rows[0].prompt = "changed synthetic input"
    with patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="training input fingerprint"):
            grpo._run_one_seed(
                config, grpo.InMemoryDataset(train=rows), grpo.ExactMathReward(), _Tokenizer()
            )
    tracking.assert_not_called()
    assert path.read_bytes() == before
