"""CPU-only launch contracts: never execute generated scripts or provider drivers."""

import contextlib
import io
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest

ROOT = Path(os.environ.get("LAUNCHER_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from platform_local.unified.canonical import CanonicalSpec
from platform_local.unified.launcher import UnifiedLauncher
from platform_local.unified.__main__ import main


def launcher(framework, algorithm="grpo"):
    item = UnifiedLauncher()
    item.framework = framework
    item.algorithm = algorithm
    item.model = "synthetic/model"
    item.spec = CanonicalSpec(model=item.model)
    return item


@pytest.mark.parametrize("algorithm", ["ppo", "reinforce", "dapo", "dpo", "idpo"])
def test_tinker_rejects_unimplemented_algorithms_before_launch(algorithm):
    with patch("subprocess.run") as run, pytest.raises(ValueError, match="only algorithm=grpo"):
        launcher("tinker", algorithm)._run_tinker()
    run.assert_not_called()


def test_tinker_canonical_settings_reach_real_cli_config():
    from platform_tinker.tinkerrl.grpo_cli import _parse_args, build_config

    item = launcher("tinker")
    with patch("subprocess.run") as run:
        result = item._run_tinker()
    command = run.call_args.args[0]
    actual = build_config(_parse_args(command[3:]))
    assert actual.model == item.model
    assert actual.group_size == item.spec.num_generations
    assert actual.lr == item.spec.learning_rate
    assert actual.lora_rank == item.spec.lora_rank
    assert actual.max_response_tokens == item.spec.max_completion_length
    assert actual.batch_size == item.spec.per_device_train_batch_size
    assert actual.steps == item.spec.training_steps
    assert actual.seed == item.spec.seed
    assert result.algorithm == "grpo"


def test_openrlhf_driver_receives_typed_requested_configuration():
    # Do not import the package initializer, which can import native/provider modules.
    package = types.ModuleType("platform_modal.openrlhf")
    package.__path__ = [str(ROOT / "platform_modal/openrlhf")]
    driver = types.ModuleType("platform_modal.openrlhf.trainer")
    captured = []
    driver.run_openrlhf_training = lambda config, output_dir: (
        captured.append(config) or {"steps": 7}
    )
    with patch.dict(sys.modules, {package.__name__: package, driver.__name__: driver}):
        item = launcher("openrlhf")
        result = item._run_openrlhf()
        config = captured[0]
        assert config.model_name == item.model
        assert config.learning_rate == item.spec.learning_rate
        assert config.algorithm.algorithm == "grpo"
        assert config.algorithm.kl_coef == item.spec.beta
        assert config.algorithm.sample_num == item.spec.num_generations
        assert config.model.lora_rank == item.spec.lora_rank
        assert config.data.max_response_length == item.spec.max_completion_length
        assert config.train_batch_size == item.spec.per_device_train_batch_size
        assert result.final_step == 7
        assert config.seed == item.spec.seed
        assert config.max_steps == item.spec.training_steps
        assert config.to_dict()["model"]["model_name"] == item.model
        assert result.model == item.model


def test_openrlhf_rejects_unsupported_algorithm_before_driver_import():
    with (
        patch.dict(sys.modules, {"platform_modal.openrlhf.trainer": None}),
        pytest.raises(ValueError, match="only algorithm=grpo"),
    ):
        launcher("openrlhf", "ppo")._run_openrlhf()


def invoke_trl(extra=(), generate=False):
    driver = types.ModuleType("platform_local.trl_integrations.trainer")
    driver.TRLTrainer = type("InertTrainer", (), {})
    captured = []
    driver.generate_trl_train_script = lambda config, path: captured.append(config.model_dump())
    argv = [
        "unified",
        "--framework",
        "trl",
        "--model",
        "synthetic/model",
        "--train-data",
        "synthetic.json",
        "--reward-callable",
        "task_rewards:reward",
        "--output-dir",
        "synthetic-fresh-output",
    ]
    argv += list(extra)
    if generate:
        argv += ["--generate-script", "synthetic-not-written.py"]
    with (
        patch.dict(sys.modules, {driver.__name__: driver}),
        patch.object(sys, "argv", argv),
        patch.object(UnifiedLauncher, "run", lambda self: self._run_trl()),
        patch.object(UnifiedLauncher, "_resolve_spec", return_value=CanonicalSpec()),
        patch("tempfile.mkdtemp", return_value="synthetic-not-created"),
        patch("subprocess.run") as run,
        contextlib.redirect_stdout(io.StringIO()),
    ):
        main()
    assert run.call_count == (0 if generate else 1)
    return captured[0]


@pytest.mark.parametrize("backend", ["local", "colab"])
def test_direct_trl_preserves_budget_and_cli_settings(backend):
    config = invoke_trl(
        [
            "--backend",
            backend,
            "--total-token-budget",
            "640",
            "--tokens-per-sample",
            "10",
            "--batch-size",
            "2",
            "--lr",
            "0.0002",
            "--lora-rank",
            "7",
            "--epochs",
            "3",
            "--peft-num-virtual-tokens",
            "12",
            "--deepspeed",
            "synthetic-ds.json",
        ]
    )
    assert config["max_steps"] == 4
    assert config["data"]["train_batch_size"] == 2
    assert config["optimizer"]["learning_rate"] == 0.0002
    assert config["model"]["lora_rank"] == 7
    assert config["model"]["peft_num_virtual_tokens"] == 12
    assert config["deepspeed"] == "synthetic-ds.json"
    assert config["epochs"] == 3
    assert config["reward_callable"] == "task_rewards:reward"
    assert config["output_dir"] == "synthetic-fresh-output"
    assert config["seed"] == CanonicalSpec().seed
    assert config["algorithm"]["num_generations"] == CanonicalSpec().num_generations
    assert config["algorithm"]["kl_coef"] == CanonicalSpec().beta
    assert config["data"]["max_response_length"] == CanonicalSpec().max_completion_length
    assert (
        config["data"]["gradient_accumulation_steps"] == CanonicalSpec().gradient_accumulation_steps
    )


def test_trl_generation_and_direct_execution_share_configuration():
    args = [
        "--total-token-budget",
        "640",
        "--tokens-per-sample",
        "10",
        "--batch-size",
        "2",
        "--lr",
        "0.0002",
        "--lora-rank",
        "7",
        "--epochs",
        "3",
    ]
    assert invoke_trl(args) == invoke_trl(args, generate=True)


@pytest.mark.parametrize(
    "extra",
    [
        ["--total-token-budget", "0"],
        ["--total-token-budget", "-1"],
        ["--total-token-budget", "10", "--tokens-per-sample", "0"],
        ["--total-token-budget", "10", "--batch-size", "0"],
        ["--total-token-budget", "10", "--backend", "modal"],
        ["--total-token-budget", "10", "--framework", "tinker"],
    ],
)
def test_invalid_or_unsupported_budget_rejected_before_launch(extra):
    with pytest.raises(SystemExit) as failure:
        invoke_trl(extra)
    assert failure.value.code == 2


def test_direct_trl_rejects_idpo_with_pinned_backend():
    with pytest.raises(SystemExit) as failure:
        invoke_trl(["--algorithm", "idpo"])
    assert failure.value.code == 2


@pytest.mark.parametrize("extra", [["--algorithm", "ppo"], ["--no-peft", "--load-in-4bit"]])
def test_direct_trl_rejects_unsupported_configuration(extra):
    with pytest.raises(SystemExit) as failure:
        invoke_trl(extra)
    assert failure.value.code == 2
