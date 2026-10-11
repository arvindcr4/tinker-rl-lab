"""Native-driver contracts, using CPU mocks and no provider/network execution."""

import importlib
import subprocess
import sys
import types
from unittest.mock import Mock, patch

import pytest

from platform_modal.openrlhf.command import build_openrlhf_command
from platform_modal.openrlhf.config import OpenRLHFConfig
from platform_modal.openrlhf import trainer


def config(**kwargs):
    return OpenRLHFConfig(
        algorithm={"algorithm": "grpo", "sample_num": 6},
        remote_rm_url="http://127.0.0.1:8765/get_reward",
        **kwargs,
    )


def test_command_forwards_nondefault_runtime_settings(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = config(
        seed=17,
        epochs=3,
        train_batch_size=24,
        micro_batch_size=2,
        rollout_batch_size=4,
        max_samples=96,
        model={"model_name": "synthetic/model", "lora_rank": 7, "use_flash_attention": False},
        data={"max_prompt_length": 93, "max_response_length": 127},
        tensor_parallel_size=2,
        num_gpus=2,
    )
    cmd = build_openrlhf_command(cfg, "/data.jsonl", "/output")
    assert cmd[:3] == [sys.executable, "-m", "openrlhf.cli.train_ppo_ray"]
    for key, expected in {
        "seed": "17",
        "max_epochs": "3",
        "train_batch_size": "24",
        "micro_train_batch_size": "2",
        "rollout_batch_size": "4",
        "max_samples": "96",
        "n_samples_per_prompt": "6",
        "generate_max_len": "127",
        "prompt_max_len": "93",
        "lora_rank": "7",
        "vllm_tensor_parallel_size": "2",
        "actor_num_gpus_per_node": "2",
    }.items():
        assert cmd[cmd.index("--" + key) + 1] == expected
    assert "--flash_attn" not in cmd
    assert "--use_wandb" not in cmd


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"max_steps": 5}, "max_steps"),
        ({"remote_rm_url": "verifiable_gsm8k"}, "HTTP"),
        ({"algorithm": {"algorithm": "ppo"}}, "only algorithm=grpo"),
        ({"algorithm": {"algorithm": "grpo", "sample_num": 1}}, "sample_num"),
        ({"data": {"train_data": ["custom.json"]}}, "fixed GSM8K"),
    ],
)
def test_invalid_requests_fail_before_provider_or_filesystem_work(tmp_path, changes, match):
    values = config().model_dump()
    values.update(changes)
    output = tmp_path / "not-created"
    with patch.object(trainer, "_prepare_gsm8k_500_jsonl") as prepare:
        with pytest.raises(ValueError, match=match):
            trainer.run_openrlhf_training(OpenRLHFConfig(**values), str(output))
    prepare.assert_not_called()
    assert not output.exists()


def test_failure_cannot_create_success_receipt(tmp_path):
    with (
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl"),
        patch.object(
            trainer.subprocess, "run", side_effect=subprocess.CalledProcessError(9, "native")
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        trainer.run_openrlhf_training(config(), str(tmp_path))
    assert not (tmp_path / "result.json").exists()


def test_success_records_config_without_inventing_measurements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl") as prepare,
        patch.object(trainer.subprocess, "run") as native,
    ):
        result = trainer.run_openrlhf_training(config(seed=19), "relative")
    assert prepare.call_args.args[0] == str(tmp_path / "relative/gsm8k500.jsonl")
    assert native.call_args.kwargs["check"] is True
    assert result["seed"] == 19
    assert result["group_size"] == 6
    assert result["steps"] is None
    assert result["reward_trace"] == []
    assert result["metrics_status"] == "not_collected"
    assert "peak_reward" not in result


def test_import_does_not_start_emissions_tracker():
    fake = types.ModuleType("codecarbon")
    fake.EmissionsTracker = Mock()
    with patch.dict(sys.modules, {"codecarbon": fake}):
        importlib.reload(trainer)
    fake.EmissionsTracker.assert_not_called()


def test_existing_receipt_is_preserved_before_any_provider_work(tmp_path):
    receipt = tmp_path / "result.json"
    receipt.write_text("existing evidence")
    with patch.object(trainer, "_prepare_gsm8k_500_jsonl") as prepare:
        with pytest.raises(FileExistsError, match="fresh output"):
            trainer.run_openrlhf_training(config(), str(tmp_path))
    prepare.assert_not_called()
    assert receipt.read_text() == "existing evidence"
