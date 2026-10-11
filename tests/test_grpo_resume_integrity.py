"""Malformed local resume receipts must fail before tracking or paid clients."""

import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo


@pytest.mark.parametrize("step", [-1, 3, True, "1", 1.5, None])
def test_resume_step_is_not_coerced_or_outside_configured_budget(tmp_path, step):
    config = grpo.GRPOConfig(name="invalid-step", steps=2, checkpoint_dir=str(tmp_path))
    receipt = {
        "config": grpo._config_fingerprint(config, config.seed),
        "status": "started",
        "step": step,
        "train_state_path": "tinker://synthetic/state",
    }
    path = grpo._checkpoint_path(config, config.seed)
    grpo._write_checkpoint(path, receipt)
    before = path.read_bytes()
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with (
        patch.object(grpo, "_start_wandb") as tracking,
        pytest.raises(RuntimeError, match="checkpoint step must be an integer"),
    ):
        grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), tokenizer=None)
    tracking.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("state", [None, "", "   ", 42, False])
def test_nonzero_resume_cannot_restart_a_fresh_actor(tmp_path, state):
    config = grpo.GRPOConfig(name="missing-state", steps=2, checkpoint_dir=str(tmp_path))
    grpo._write_checkpoint(
        grpo._checkpoint_path(config, config.seed),
        {
            "config": grpo._config_fingerprint(config, config.seed),
            "status": "started",
            "step": 1,
            "train_state_path": state,
        },
    )
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with (
        patch.object(grpo, "_start_wandb") as tracking,
        pytest.raises(RuntimeError, match="no actor optimizer state"),
    ):
        grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), tokenizer=None)
    tracking.assert_not_called()


@pytest.mark.parametrize("payload", [None, [], 1, "receipt"])
def test_resume_receipt_requires_json_object(tmp_path, payload):
    config = grpo.GRPOConfig(name="invalid-shape", checkpoint_dir=str(tmp_path))
    path = grpo._checkpoint_path(config, config.seed)
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checkpoint must be a JSON object"):
        grpo._load_checkpoint(config, config.seed)
