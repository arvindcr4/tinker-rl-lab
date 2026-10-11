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
from tests._shared_fakes import write_native_input_fixture


@pytest.fixture
def native_runtime():
    with patch.object(trainer, "_validate_native_runtime", return_value="0.8.4"):
        yield


def config(**kwargs):
    kwargs.setdefault("train_batch_size", 8)
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
        ({"num_gpus": 2, "tensor_parallel_size": 1}, "single colocated"),
    ],
)
def test_invalid_requests_fail_before_provider_or_filesystem_work(tmp_path, changes, match):
    values = config().model_dump()
    values.update(changes)
    output = tmp_path / "not-created"
    with patch.object(
        trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture
    ) as prepare:
        with pytest.raises(ValueError, match=match):
            trainer.run_openrlhf_training(OpenRLHFConfig(**values), str(output))
    prepare.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("gpus", [1, 2, 4, 8])
def test_supported_topology_satisfies_upstream_colocation_contract(gpus):
    command = build_openrlhf_command(
        config(num_gpus=gpus, tensor_parallel_size=gpus), "/data", "/output"
    )

    def integer_option(name):
        return int(command[command.index("--" + name) + 1])

    assert "--colocate_all_models" in command
    assert integer_option("actor_num_gpus_per_node") == (
        integer_option("vllm_num_engines") * integer_option("vllm_tensor_parallel_size")
    )


def test_failure_cannot_create_success_receipt(tmp_path, native_runtime):
    with (
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture),
        patch.object(
            trainer.subprocess, "run", side_effect=subprocess.CalledProcessError(9, "native")
        ),
        pytest.raises(subprocess.CalledProcessError),
    ):
        trainer.run_openrlhf_training(config(), str(tmp_path / "run"))
    assert not (tmp_path / "run/result.json").exists()


def test_success_records_config_without_inventing_measurements(
    tmp_path, monkeypatch, native_runtime
):
    monkeypatch.chdir(tmp_path)
    with (
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(
            trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture
        ) as prepare,
        patch.object(trainer.subprocess, "run") as native,
    ):
        result = trainer.run_openrlhf_training(config(seed=19), "relative")
    assert prepare.call_args.args[0] == str(tmp_path / "relative/gsm8k500.jsonl")
    assert native.call_args.kwargs["check"] is True
    assert result["input_artifact"]["rows"] == 500
    assert result["input_artifact"]["source_revision"] is None
    assert len(result["input_artifact"]["sha256"]) == 64
    assert "--apply_chat_template" in native.call_args.args[0]
    assert result["seed"] == 19
    assert result["group_size"] == 6
    assert result["steps"] is None
    assert result["optimizer_steps"] is None
    assert result["budget_plan"]["planned_rollouts"] == 30
    assert result["budget_plan"]["planned_update_boundaries"] == 3600
    assert result["reward_trace"] == []
    assert result["metrics_status"] == "not_collected"
    assert "peak_reward" not in result


def test_native_receipt_uses_launch_snapshot_during_caller_mutation(tmp_path, native_runtime):
    cfg = config(seed=17)
    expected = cfg.to_dict()

    def mutate_caller(command, **kwargs):
        assert command[command.index("--seed") + 1] == "17"
        cfg.seed = 99
        cfg.optimizer.learning_rate = 0.5
        cfg.algorithm.sample_num = 12
        cfg.model.model_name = "changed/after-launch"
        cfg.data.train_data.append("changed-after-launch.jsonl")
        cfg.max_samples = 24
        cfg.log_to_wandb = True

    with (
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture),
        patch.object(trainer.subprocess, "run", side_effect=mutate_caller),
    ):
        result = trainer.run_openrlhf_training(cfg, str(tmp_path / "snapshot"))
    assert result["config"] == expected
    assert result["seed"] == 17
    assert result["learning_rate"] == expected["optimizer"]["learning_rate"]
    assert result["group_size"] == 6
    assert result["model"] == expected["model"]["model_name"]
    assert result["budget_plan"]["planned_rollouts"] == 30
    assert result.get("metrics_reason") != "native_wandb_selected_no_local_metrics"
    assert cfg.seed == 99
    assert cfg.data.train_data == ["changed-after-launch.jsonl"]


def test_import_does_not_start_emissions_tracker():
    fake = types.ModuleType("codecarbon")
    fake.EmissionsTracker = Mock()
    with patch.dict(sys.modules, {"codecarbon": fake}):
        importlib.reload(trainer)
    fake.EmissionsTracker.assert_not_called()


def test_existing_receipt_is_preserved_before_any_provider_work(tmp_path):
    receipt = tmp_path / "result.json"
    receipt.write_text("existing evidence")
    with patch.object(
        trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture
    ) as prepare:
        with pytest.raises(FileExistsError, match="fresh output"):
            trainer.run_openrlhf_training(config(), str(tmp_path))
    prepare.assert_not_called()
    assert receipt.read_text() == "existing evidence"


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"max_samples": 7}, "whole rollout"),
        ({"max_samples": 241}, "whole rollout"),
        ({"micro_batch_size": 3}, "distributed micro"),
        ({"tensor_parallel_size": 2}, "cannot exceed"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"run_name": "../old-run"}, "directory component"),
    ],
)
def test_rejects_unrepresentable_batch_and_logging_settings(changes, match):
    with pytest.raises(ValueError, match=match):
        build_openrlhf_command(config(**changes), "/data", "/output")


def test_environment_key_alone_does_not_switch_local_metrics_to_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-test-key")
    command = build_openrlhf_command(config(), "/data", "/output")
    assert "--use_wandb" not in command
    assert command[command.index("--use_tensorboard") + 1] == "/output/tensorboard"
    explicit = build_openrlhf_command(config(log_to_wandb=True), "/data", "/output")
    assert "--use_wandb" in explicit
    assert explicit[explicit.index("--use_wandb") + 1] == "environment"
    assert "synthetic-test-key" not in " ".join(explicit)
    assert "--use_tensorboard" not in explicit


def test_native_wandb_credential_stays_out_of_argv_and_receipt(
    tmp_path, monkeypatch, native_runtime
):
    key = "synthetic-only-wandb-credential"
    monkeypatch.setenv("WANDB_API_KEY", key)
    with (
        patch.dict(sys.modules, {"openrlhf": types.ModuleType("openrlhf")}),
        patch.object(trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture),
        patch.object(trainer.subprocess, "run") as native,
    ):
        output = tmp_path / "run"
        trainer.run_openrlhf_training(config(log_to_wandb=True), str(output))
    assert native.call_args.kwargs["env"]["WANDB_API_KEY"] == key
    assert key not in repr(native.call_args.args[0])
    assert key not in (output / "result.json").read_text()


def test_explicit_wandb_requires_credentials(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(ValueError, match="requires WANDB_API_KEY"):
        build_openrlhf_command(config(log_to_wandb=True), "/data", "/output")


@pytest.mark.parametrize("version", ["0.8.3", "0.8.4.dev1", "0.9.0"])
def test_unvalidated_versions_fail_before_dataset_work(tmp_path, version):
    with (
        patch.object(trainer.metadata, "version", return_value=version),
        patch.object(
            trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture
        ) as prepare,
        pytest.raises(RuntimeError, match="Unsupported OpenRLHF version"),
    ):
        trainer.run_openrlhf_training(config(), str(tmp_path / "run"))
    prepare.assert_not_called()
    assert not (tmp_path / "run").exists()


def test_supported_version_requires_local_logging_dependency():
    with (
        patch.object(trainer.metadata, "version", return_value="0.8.4"),
        patch.object(trainer.util, "find_spec", return_value=None),
        pytest.raises(RuntimeError, match="require tensorboard"),
    ):
        trainer._validate_native_runtime(local_metrics=True)


def test_directory_with_stale_events_cannot_be_reused(tmp_path):
    (tmp_path / "events.out.tfevents.old").write_text("preserve")
    with (
        patch.object(trainer, "_validate_native_runtime") as runtime,
        pytest.raises(FileExistsError, match="fresh output"),
    ):
        trainer.run_openrlhf_training(config(), str(tmp_path))
    runtime.assert_not_called()
    assert (tmp_path / "events.out.tfevents.old").read_text() == "preserve"


def test_legacy_overrides_are_rejected_before_native_work(tmp_path):
    for field in (
        "num_workers",
        "num_actors",
        "actor_num_gpus",
        "critic_num_gpus",
        "eval_interval",
        "save_interval",
    ):
        output = tmp_path / field
        with (
            patch.object(
                trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture
            ) as prepare,
            patch.object(trainer, "_validate_native_runtime") as runtime,
            patch.object(trainer.subprocess, "run") as launch,
            pytest.raises(ValueError, match=field),
        ):
            trainer.run_openrlhf_training(config(**{field: 99}), str(output))
        prepare.assert_not_called()
        runtime.assert_not_called()
        launch.assert_not_called()
        assert not output.exists()


def test_native_default_contract_disables_periodic_work_explicitly():
    command = build_openrlhf_command(config(), "/data", "/output")
    for name in ("save_steps", "eval_steps"):
        assert command[command.index("--" + name) + 1] == "-1"
    assert command[command.index("--save_path") + 1] == "/output/ckpt"


def test_nonfinite_native_settings_fail_before_any_external_work(tmp_path):
    fields = {
        "optimizer": ("learning_rate", "beta1", "beta2", "eps", "weight_decay"),
        "algorithm": ("gamma", "lam", "epsilon", "kl_coef", "temperature"),
    }
    for section, names in fields.items():
        for name in names:
            for value in (float("nan"), float("inf"), -float("inf"), True, "0.5", 10**400):
                cfg = config()
                setattr(getattr(cfg, section), name, value)
                output = tmp_path / "not-created"
                with (
                    patch.object(
                        trainer, "_prepare_gsm8k_500_jsonl", side_effect=write_native_input_fixture
                    ) as prepare,
                    patch.object(trainer, "_validate_native_runtime") as runtime,
                    patch.object(trainer.subprocess, "run") as launch,
                    pytest.raises(ValueError, match=name),
                ):
                    trainer.run_openrlhf_training(cfg, str(output))
                prepare.assert_not_called()
                runtime.assert_not_called()
                launch.assert_not_called()
                assert not output.exists()


def test_mutated_native_counts_and_dataset_limits_fail_before_launch():
    for field in (
        "epochs",
        "train_batch_size",
        "micro_batch_size",
        "num_gpus",
        "rollout_batch_size",
        "max_samples",
        "tensor_parallel_size",
        "seed",
    ):
        for value in (True, 2.0, "2", None):
            cfg = config()
            setattr(cfg, field, value)
            with pytest.raises(ValueError, match="integer"):
                build_openrlhf_command(cfg, "/data", "/output")
    cfg = config()
    cfg.max_samples = 504
    with pytest.raises(ValueError, match="500-row"):
        build_openrlhf_command(cfg, "/data", "/output")
    for section, field in (
        ("algorithm", "sample_num"),
        ("model", "lora_rank"),
        ("data", "max_prompt_length"),
        ("data", "max_response_length"),
    ):
        cfg = config()
        setattr(getattr(cfg, section), field, 2.0)
        with pytest.raises(ValueError, match="integer"):
            build_openrlhf_command(cfg, "/data", "/output")


def test_native_optimizer_boundaries_preserve_zero_modes():
    for field, value in (
        ("learning_rate", -0.1),
        ("weight_decay", -0.1),
        ("beta1", -0.1),
        ("beta1", 1.0),
        ("beta2", 1.0),
    ):
        cfg = config()
        setattr(cfg.optimizer, field, value)
        with pytest.raises(ValueError, match=field):
            build_openrlhf_command(cfg, "/data", "/output")
    cfg = config()
    cfg.optimizer.learning_rate = 0.0
    cfg.optimizer.weight_decay = 0.0
    cfg.optimizer.beta1 = 0.0
    cfg.algorithm.temperature = 0.0
    cfg.algorithm.epsilon = 0.0
    command = build_openrlhf_command(cfg, "/data", "/output")
    for name in ("actor_learning_rate", "temperature", "eps_clip", "l2"):
        assert float(command[command.index("--" + name) + 1]) == 0.0
