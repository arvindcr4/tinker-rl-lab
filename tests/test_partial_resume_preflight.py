"""Reject malformed resume references before tracking or paid client creation."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _Tokenizer
from tests.test_grpo_coverage import _receipt


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("resume preflight must stay offline")

    monkeypatch.setattr("socket.socket.connect", blocked)


@pytest.mark.parametrize(
    "change", ["actor_control", "actor_option", "receipt_list", "receipt_item", "receipt_source"]
)
def test_partial_resume_references_fail_before_tracking(tmp_path, change):
    config = grpo.GRPOConfig(name="partial-references", steps=2, checkpoint_dir=str(tmp_path))
    receipt = {
        "config": grpo._config_fingerprint(config, config.seed),
        "status": "started",
        "step": 1,
        "train_state_path": "tinker://synthetic/state",
        "reward_trace": [0.5],
        "checkpoint_receipts": [_receipt(step=1)],
    }
    if change == "actor_control":
        receipt["train_state_path"] = "tinker://synthetic/\nstate"
    elif change == "actor_option":
        receipt["train_state_path"] = "--unexpected-option"
    elif change == "receipt_list":
        receipt["checkpoint_receipts"] = {}
    elif change == "receipt_item":
        receipt["checkpoint_receipts"] = [False]
    else:
        receipt["checkpoint_receipts"][0]["source_path"] = ""
    path = grpo._checkpoint_path(config, config.seed)
    grpo._write_checkpoint(path, deepcopy(receipt))
    before = path.read_bytes()
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with patch.object(
        grpo, "_start_wandb", side_effect=AssertionError("tracking started")
    ) as tracking:
        with pytest.raises(RuntimeError, match="checkpoint|receipt"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()
    assert path.read_bytes() == before
