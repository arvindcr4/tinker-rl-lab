"""A completed result must not disagree with its own export references."""

from copy import deepcopy
import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _Tokenizer
from tests.test_completed_result_integrity import _completed
from tests.test_grpo_coverage import _runtime


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("cached checkpoint checks must stay offline")

    monkeypatch.setattr("socket.socket.connect", blocked)


@pytest.mark.parametrize("change", ["sampler", "urls", "commits", "top_level"])
def test_contradictory_export_evidence_stops_before_tracking(tmp_path, monkeypatch, change):
    config, dataset, _, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    if change == "sampler":
        payload["result"]["sampler_path"] = "tinker://synthetic/different-model"
    elif change == "urls":
        payload["result"]["checkpoint_urls"] = ["https://huggingface.co/synthetic/different"]
        payload["checkpoint_urls"] = payload["result"]["checkpoint_urls"]
    elif change == "commits":
        payload["result"]["checkpoint_commit_shas"] = ["a" * 40]
        payload["checkpoint_commit_shas"] = payload["result"]["checkpoint_commit_shas"]
    else:
        payload["checkpoint_urls"] = ["https://huggingface.co/synthetic/different"]
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    with _runtime(monkeypatch), patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(RuntimeError, match="Completed receipt.*checkpoint"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("legacy_arrays", [False, True])
def test_consistent_exports_replay_without_rewriting_or_provider(
    tmp_path, monkeypatch, legacy_arrays
):
    config, dataset, original, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    if legacy_arrays:
        for key in ("checkpoint_urls", "checkpoint_commit_shas"):
            payload.pop(key, None)
            payload["result"].pop(key, None)
        grpo._write_checkpoint(path, payload)
    before = path.read_bytes()
    with _runtime(monkeypatch) as holder:
        result = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert result.sampler_path == original.sampler_path
    assert result.checkpoint_receipts == original.checkpoint_receipts
    assert "service_kwargs" not in holder
    assert path.read_bytes() == before
    if legacy_arrays:
        assert result.checkpoint_urls == []
        assert result.checkpoint_commit_shas == []
    else:
        assert result == original


def test_malformed_export_shapes_fail_before_tracking(tmp_path, monkeypatch):
    config, dataset, _, path = _completed(tmp_path, monkeypatch)
    original = json.loads(path.read_text())
    for kind in (
        "missing_sampler",
        "bad_sampler",
        "receipts_not_list",
        "bad_item",
        "bad_step",
        "bad_source",
        "duplicate_final",
        "bad_url_item",
    ):
        payload = deepcopy(original)
        result = payload["result"]
        if kind == "missing_sampler":
            result["sampler_path"] = None
        elif kind == "bad_sampler":
            result["sampler_path"] = ""
        elif kind == "receipts_not_list":
            result["checkpoint_receipts"] = {}
        elif kind == "bad_item":
            result["checkpoint_receipts"] = ["not a receipt"]
        elif kind == "bad_step":
            result["checkpoint_receipts"][-1]["step"] = True
        elif kind == "bad_source":
            result["checkpoint_receipts"][-1]["source_path"] = ""
        elif kind == "duplicate_final":
            result["checkpoint_receipts"].append(deepcopy(result["checkpoint_receipts"][-1]))
            result["checkpoint_urls"].append(result["checkpoint_urls"][-1])
            result["checkpoint_commit_shas"].append(result["checkpoint_commit_shas"][-1])
        else:
            result["checkpoint_urls"] = [False]
        for key in ("checkpoint_receipts", "checkpoint_urls", "checkpoint_commit_shas"):
            payload[key] = deepcopy(result[key])
        grpo._write_checkpoint(path, payload)
        before = path.read_bytes()
        with patch.object(grpo, "_start_wandb") as tracking:
            with pytest.raises(RuntimeError, match="Completed receipt.*checkpoint"):
                grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
        tracking.assert_not_called()
        assert path.read_bytes() == before
