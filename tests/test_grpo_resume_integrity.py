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


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_receipt_fails_before_tracking(tmp_path, number):
    config = grpo.GRPOConfig(name="nonfinite", checkpoint_dir=str(tmp_path))
    payload = json.dumps(
        {
            "config": grpo._config_fingerprint(config, config.seed),
            "status": "completed",
            "result": {"reward_trace": ["NUMBER"]},
        }
    )
    path = grpo._checkpoint_path(config, config.seed)
    path.write_text(payload.replace('"NUMBER"', number))
    before = path.read_bytes()
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(ValueError, match="checkpoint must be a JSON object"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), tokenizer=None)
    tracking.assert_not_called()
    assert path.read_bytes() == before


def test_duplicate_receipt_fields_are_rejected(tmp_path):
    config = grpo.GRPOConfig(name="duplicate", checkpoint_dir=str(tmp_path))
    path = grpo._checkpoint_path(config, config.seed)
    payload = json.dumps({"config": grpo._config_fingerprint(config, config.seed), "step": 0})
    path.write_text(payload[:-1] + ', "step": 1}')
    with pytest.raises(ValueError, match="checkpoint must be a JSON object"):
        grpo._load_checkpoint(config, config.seed)


def test_nonfinite_checkpoint_write_preserves_previous_receipt(tmp_path):
    path = tmp_path / "checkpoint.json"
    grpo._write_checkpoint(path, {"step": 0})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Out of range float values"):
        grpo._write_checkpoint(path, {"step": 1, "reward_trace": [float("inf")]})
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "trace", [None, {}, "0.5", [], [True], ["0.5"], [10**400], [0.1, 0.2, 0.3]]
)
def test_resume_history_requires_finite_numeric_trace_covering_committed_steps(tmp_path, trace):
    config = grpo.GRPOConfig(name="history", steps=2, checkpoint_dir=str(tmp_path))
    payload = {
        "config": grpo._config_fingerprint(config, config.seed),
        "status": "started",
        "step": 1,
        "train_state_path": "tinker://synthetic/state",
        "reward_trace": trace,
    }
    path = grpo._checkpoint_path(config, config.seed)
    grpo._write_checkpoint(path, payload)
    before = path.read_bytes()
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with patch.object(grpo, "_start_wandb", side_effect=AssertionError("tracking started")):
        with pytest.raises(RuntimeError, match="checkpoint reward history"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), tokenizer=None)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("zero_loss_steps", True),
        ("zero_loss_steps", -1),
        ("zero_loss_steps", 2),
        ("zero_reward_steps", "0"),
        ("zero_reward_steps", 1),
    ],
)
def test_resume_counters_cannot_be_coerced_or_contradict_history(tmp_path, field, value):
    config = grpo.GRPOConfig(name="counters", steps=2, checkpoint_dir=str(tmp_path))
    payload = {
        "config": grpo._config_fingerprint(config, config.seed),
        "status": "started",
        "step": 1,
        "train_state_path": "tinker://synthetic/state",
        "reward_trace": [0.5],
        field: value,
    }
    grpo._write_checkpoint(grpo._checkpoint_path(config, config.seed), payload)
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with patch.object(grpo, "_start_wandb", side_effect=AssertionError("tracking started")):
        with pytest.raises(RuntimeError, match="checkpoint reward history"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), tokenizer=None)


def test_completed_history_must_cover_the_full_configured_budget(tmp_path):
    config = grpo.GRPOConfig(name="short-completed", steps=2, checkpoint_dir=str(tmp_path))
    payload = {
        "config": grpo._config_fingerprint(config, config.seed),
        "status": "completed",
        "step": 1,
        "result": {"seed": 42, "run_id": "synthetic", "reward_trace": [0.5]},
    }
    grpo._write_checkpoint(grpo._checkpoint_path(config, config.seed), payload)
    dataset = grpo.InMemoryDataset(train=[grpo.TrainingExample(prompt="q", target="1")])
    with patch.object(grpo, "_start_wandb", side_effect=AssertionError("tracking started")):
        with pytest.raises(RuntimeError, match="checkpoint reward history"):
            grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), tokenizer=None)


def test_rewound_nested_final_uses_only_committed_prefix_counters():
    prior = {
        "status": "started",
        "result": {"reward_trace": [0.0, 0.5], "zero_loss_steps": 2, "zero_reward_steps": 1},
    }
    assert grpo._resume_reward_history(prior, resume_step=1, steps=2) == ([0.0], 0, 1)
    assert prior["result"]["reward_trace"] == [0.0, 0.5]


def test_legacy_missing_reward_counter_is_reconstructed_without_inventing_trace():
    prior = {"status": "started", "reward_trace": [0.0, 0.5]}
    assert grpo._resume_reward_history(prior, resume_step=2, steps=2) == ([0.0, 0.5], 0, 1)


@pytest.mark.parametrize("result", [[], "bad", 1])
def test_nested_result_shape_is_validated_before_resume(result):
    with pytest.raises(RuntimeError, match="checkpoint reward history"):
        grpo._resume_reward_history({"result": result}, resume_step=0, steps=1)
