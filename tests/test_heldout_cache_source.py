"""Completed held-out evidence must stay bound to its recorded input cohort."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("held-out cache regressions must stay offline")

    monkeypatch.setattr("socket.socket.connect", blocked)


def completed(tmp_path, monkeypatch, rows=None, enabled=True):
    config = _config(tmp_path, evaluate_heldout=enabled)
    if rows is None:
        rows = [_example(prompt="heldout-a"), _example(prompt="heldout-b")]
    data = grpo.InMemoryDataset(train=[_example()], test=rows)
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    return config, data, result, grpo._checkpoint_path(config, config.seed)


@pytest.mark.parametrize("change", ["prompt", "target", "metadata", "order", "count", "non_json"])
def test_changed_heldout_inputs_reject_cached_score_before_tracking(tmp_path, monkeypatch, change):
    config, data, _, path = completed(tmp_path, monkeypatch)
    before = path.read_bytes()
    if change == "order":
        data.test.reverse()
    elif change == "count":
        data.test.pop()
    elif change == "metadata":
        data.test[0].metadata["id"] = "changed"
    elif change == "non_json":
        data.test[0].target = object()
    else:
        setattr(data.test[0], change, "changed")
    with patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="held-out input fingerprint"):
            grpo._run_one_seed(config, data, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("empty", [False, True])
def test_unchanged_cohort_reuses_exact_receipt_without_provider(tmp_path, monkeypatch, empty):
    config, data, original, path = completed(tmp_path, monkeypatch, rows=[] if empty else None)
    before = path.read_bytes()
    source = original.heldout_evaluation["input_source"]
    assert source["status"] == "fingerprinted"
    assert source["count"] == len(data.test)
    assert source["sha256"] == grpo._example_input_fingerprint(data.test, domain="heldout")
    assert source["sha256"] != grpo._training_input_fingerprint(data.test)
    assert "heldout-a" not in json.dumps(source)
    with _runtime(monkeypatch) as holder:
        actual = grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert actual == original
    assert "service_kwargs" not in holder
    assert path.read_bytes() == before


def test_empty_evaluation_cache_cannot_hide_new_examples(tmp_path, monkeypatch):
    config, data, _, _ = completed(tmp_path, monkeypatch, rows=[])
    data.test.append(_example())
    with patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="held-out input fingerprint"):
            grpo._run_one_seed(config, data, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_or_disabled_cache_does_not_invent_test_input_evidence(
    tmp_path, monkeypatch, legacy
):
    config, data, _, path = completed(tmp_path, monkeypatch, enabled=legacy)
    payload = json.loads(path.read_text())
    payload["result"]["heldout_evaluation"].pop("input_source", None)
    grpo._write_checkpoint(path, payload)
    before = path.read_bytes()
    with (
        _runtime(monkeypatch),
        patch.object(
            grpo.InMemoryDataset, "test_examples", side_effect=AssertionError("test read")
        ),
    ):
        result = grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert "input_source" not in result.heldout_evaluation
    assert path.read_bytes() == before


def test_unverifiable_custom_examples_are_explicitly_unrecorded(tmp_path, monkeypatch):
    row = _example()
    row.metadata["custom"] = object()
    config, data, original, path = completed(tmp_path, monkeypatch, rows=[row])
    source = original.heldout_evaluation["input_source"]
    assert source["status"] == "unrecorded"
    assert source["sha256"] is None
    before = path.read_bytes()
    with _runtime(monkeypatch):
        result = grpo._run_one_seed(
            config, data, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.heldout_evaluation["input_source"] == source
    assert path.read_bytes() == before


def test_malformed_input_binding_rejected_without_rewriting_cache(tmp_path, monkeypatch):
    config, data, _, path = completed(tmp_path, monkeypatch)
    original = json.loads(path.read_text())
    for field, value in (
        ("schema_version", True),
        ("schema_version", 2),
        ("status", "unknown"),
        ("count", True),
        ("count", 3),
        ("sha256", None),
        ("sha256", "not-a-digest"),
        ("status", "unrecorded"),
    ):
        payload = deepcopy(original)
        payload["result"]["heldout_evaluation"]["input_source"][field] = value
        grpo._write_checkpoint(path, payload)
        before = path.read_bytes()
        with patch.object(grpo, "_start_wandb") as tracking:
            with pytest.raises(RuntimeError, match="held-out input fingerprint"):
                grpo._run_one_seed(config, data, grpo.ExactMathReward(), _Tokenizer())
        tracking.assert_not_called()
        assert path.read_bytes() == before


def test_training_fingerprint_remains_byte_compatible_with_public_prechange_code():
    rows = [grpo.TrainingExample(prompt="π", target={"b": 2, "a": 1}, metadata={"id": "a"})]
    # Computed independently with the public 39440af1 implementation before
    # extracting the shared, domain-separated input-hashing helper.
    assert grpo._training_input_fingerprint(rows) == (
        "d6a2b33d4619d85beac0977e593be02ae6949f388df74fbec1653c2848de5e3b"
    )
