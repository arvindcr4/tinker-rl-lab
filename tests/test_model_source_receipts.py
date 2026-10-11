"""Tokenizer revision requests must not become managed-weight attestations."""

import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("revision", [None, "synthetic-revision"])
def test_model_source_records_request_scope_without_attesting_weights(
    tmp_path, monkeypatch, supplied, revision
):
    config = _config(tmp_path, name="source", save_every=1, model_revision=revision)
    dataset = grpo.InMemoryDataset(train=[_example()])
    receipts = []
    write = grpo._write_checkpoint

    def capture(path, payload):
        receipts.append(json.loads(json.dumps(payload)))
        write(path, payload)

    with _runtime(monkeypatch, install_transformers=True) as holder:
        with patch.object(grpo, "_write_checkpoint", side_effect=capture):
            result = grpo._run_one_seed(
                config,
                dataset,
                grpo.ExactMathReward(),
                _Tokenizer() if supplied else None,
                logger=lambda _: None,
            )
    source = result.model_source
    assert source["schema_version"] == 1
    assert source["recorded_from_step"] == 0
    assert source["configured_model_revision"] == revision
    assert source["managed_base_weights"] == {"status": "unverified", "commit_sha": None}
    assert source["tokenizer"] == {
        "mode": "caller_supplied" if supplied else "pretrained_request",
        "requested_model": None if supplied else config.model,
        "requested_revision": None if supplied else revision,
    }
    assert source["initialization"] == {"mode": "managed_base_model_name", "checkpoint_path": None}
    assert "revision" not in holder["lora_kwargs"]
    assert holder["lora_kwargs"]["base_model"] == config.model
    if revision is not None:
        assert holder["lora_kwargs"]["user_metadata"]["model_revision"] == revision
    periodic = next(receipt for receipt in receipts if receipt["status"] != "completed")
    assert periodic["model_source"] == source
    assert receipts[-1]["result"]["model_source"] == source
    path = grpo._checkpoint_path(config, config.seed)
    before = path.read_bytes()
    source["managed_base_weights"]["status"] = "caller-mutated"
    assert path.read_bytes() == before


def test_completed_legacy_receipt_stays_unrecorded_and_is_not_rewritten(tmp_path, monkeypatch):
    config = _config(tmp_path, name="legacy-source")
    dataset = grpo.InMemoryDataset(train=[_example()])
    with _runtime(monkeypatch):
        grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    path = grpo._checkpoint_path(config, config.seed)
    payload = json.loads(path.read_text())
    payload["result"].pop("model_source", None)
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with _runtime(monkeypatch) as holder:
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), object(), logger=lambda _: None
        )
    assert result.model_source == {}
    assert "lora_kwargs" not in holder
    assert path.read_bytes() == before


def test_resumed_source_receipt_states_where_new_observation_begins(tmp_path, monkeypatch):
    config = _config(tmp_path, name="resumed-source", steps=2, save_every=1)
    dataset = grpo.InMemoryDataset(train=[_example()])
    with _runtime(monkeypatch):
        grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    path = grpo._checkpoint_path(config, config.seed)
    payload = json.loads(path.read_text())
    payload.update(status="started", step=1, train_state_path="tinker://synthetic/state")
    path.write_text(json.dumps(payload))
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.model_source["recorded_from_step"] == 1
    assert result.model_source["initialization"] == {
        "mode": "optimizer_checkpoint",
        "checkpoint_path": "tinker://synthetic/state",
    }
    assert result.model_source["managed_base_weights"]["status"] == "unverified"
