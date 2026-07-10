from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from pydantic import ValidationError

from platform_local.trl_integrations.config import TRLConfig, TRLModelConfig
from platform_local.trl_integrations.trainer import generate_trl_train_script
from platform_local.unified.peft_utils import (
    apply_bitfit,
    get_peft_config,
    get_trainable_state_dict,
    save_bitfit_checkpoint,
)


def test_peft_method_validation_does_not_require_optional_dependency():
    assert get_peft_config("bitfit") is None
    with pytest.raises(ValueError, match="Unknown PEFT method"):
        get_peft_config("not-a-method")


def test_bitfit_freezes_everything_except_biases(tmp_path):
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=True),
        torch.nn.LayerNorm(3),
        torch.nn.Linear(3, 2, bias=False),
    )

    apply_bitfit(model)

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {"0.bias", "1.bias"}
    assert set(get_trainable_state_dict(model)) == trainable_names

    checkpoint_path = save_bitfit_checkpoint(
        model,
        tmp_path / "bitfit.pt",
        base_model_name="test/model",
    )
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["format"] == "bitfit-v1"
    assert checkpoint["base_model_name"] == "test/model"
    assert set(checkpoint["state_dict"]) == trainable_names


def test_bitfit_rejects_models_without_bias_parameters():
    model = torch.nn.Linear(4, 2, bias=False)
    with pytest.raises(ValueError, match="no bias parameters"):
        apply_bitfit(model)


@pytest.mark.parametrize(
    "model_config",
    [
        {"peft_method": "unknown"},
        {"load_in_4bit": True, "load_in_8bit": True},
        {"load_in_4bit": True, "use_peft": False},
    ],
)
def test_model_config_rejects_invalid_peft_combinations(model_config):
    with pytest.raises(ValidationError):
        TRLModelConfig(**model_config)


def test_generated_grpo_script_is_valid_and_wires_quantized_prompt_tuning(tmp_path):
    output_path = tmp_path / "nested" / "train_grpo.py"
    config = TRLConfig(
        model={
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "peft_method": "prompt_tuning",
            "peft_num_virtual_tokens": 24,
            "load_in_4bit": True,
        },
        data={"train_data": ["train.json"]},
    )

    script = generate_trl_train_script(config, output_path)

    compile(script, str(output_path), "exec")
    assert output_path.read_text(encoding="utf-8") == script
    assert "PEFT_METHOD = 'prompt_tuning'" in script
    assert "LOAD_IN_4BIT = True" in script
    assert 'bnb_4bit_quant_type="nf4"' in script
    assert "bnb_4bit_compute_dtype=COMPUTE_DTYPE" in script
    assert "quantized=quantization_config is not None" in script
    assert "BOXED_MARKER = '\\\\boxed{'" in script


def test_generated_bitfit_script_avoids_full_model_checkpoints(tmp_path):
    script = generate_trl_train_script(
        TRLConfig(
            model={"peft_method": "bitfit"},
            data={"train_data": ["train.json"]},
        ),
        tmp_path / "bitfit.py",
    )

    assert 'save_strategy="no" if USE_PEFT and PEFT_METHOD == "bitfit" else "steps"' in script
    assert "save_bitfit_checkpoint(" in script


def test_generator_rejects_missing_data_and_incomplete_algorithms(tmp_path):
    with pytest.raises(ValidationError, match="mutually exclusive"):
        TRLConfig(bf16=True, fp16=True)

    with pytest.raises(ValueError, match="train_data"):
        generate_trl_train_script(TRLConfig(), tmp_path / "missing.py")

    dpo_config = TRLConfig(
        algorithm={"algorithm": "dpo"},
        data={"train_data": ["preferences.json"]},
    )
    with pytest.raises(NotImplementedError, match="GRPO only"):
        generate_trl_train_script(dpo_config, tmp_path / "dpo.py")


def test_cli_generates_script_instead_of_running_smoke_training(tmp_path):
    output_path = tmp_path / "train_grpo.py"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "platform_local.unified.launcher",
            "--framework",
            "trl",
            "--peft-method",
            "prefix_tuning",
            "--train-data",
            "train.json",
            "--generate-script",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    assert "PEFT_METHOD = 'prefix_tuning'" in output_path.read_text(encoding="utf-8")


def test_cli_rejects_quantized_full_fine_tuning(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "platform_local.unified.launcher",
            "--framework",
            "trl",
            "--no-peft",
            "--load-in-4bit",
            "--train-data",
            "train.json",
            "--generate-script",
            str(tmp_path / "invalid.py"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "quantized training cannot be combined with --no-peft" in completed.stderr
