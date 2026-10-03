"""
Tests to ensure Pydantic support across all configurations and models in the codebase.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pydantic
from pydantic import BaseModel, Field, ValidationError
import pytest

from platform_local.trl_integrations.config import (
    TRLAlgorithmConfig,
    TRLConfig,
    TRLDataConfig,
    TRLModelConfig,
    TROptimizerConfig,
)
from platform_tinker.atropos.tinker_atropos.config import (
    EnvConfig,
    OpenAIServerConfig,
    TinkerAtroposConfig,
    TinkerConfig,
)
from platform_tinker.atropos.tinker_atropos.types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    GenerateRequest,
)
from verl.config import (
    VERLAlgorithmConfig,
    VERLConfig,
    VERLDataConfig,
    VERLModelConfig,
    VERLOptimizerConfig,
)


def _load_module_directly(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pydantic_version_is_v2():
    """Ensure installed Pydantic is version 2.x and custom BaseModel works."""
    assert int(pydantic.__version__.split(".")[0]) == 2

    class SampleModel(BaseModel):
        val: int = Field(gt=0)

    assert SampleModel(val=10).val == 10
    with pytest.raises(ValidationError):
        SampleModel(val=-1)


def test_trl_configs_instantiate_and_validate():
    """Ensure TRL configurations validate fields and serialize properly."""
    model_cfg = TRLModelConfig()
    assert model_cfg.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert model_cfg.use_peft is True
    assert model_cfg.peft_method == "lora"

    # Mutual exclusivity of 4-bit and 8-bit
    with pytest.raises(ValidationError, match="mutually exclusive"):
        TRLModelConfig(load_in_4bit=True, load_in_8bit=True)

    # Quantization requires PEFT
    with pytest.raises(ValidationError, match="requires use_peft=True"):
        TRLModelConfig(load_in_4bit=True, use_peft=False)

    opt_cfg = TROptimizerConfig(learning_rate=2e-5)
    assert opt_cfg.learning_rate == 2e-5

    algo_cfg = TRLAlgorithmConfig(algorithm="grpo", epsilon_low=0.15, epsilon_high=0.25)
    assert algo_cfg.algorithm == "grpo"

    data_cfg = TRLDataConfig(train_data=["train.jsonl"])
    assert data_cfg.train_data == ["train.jsonl"]

    full_cfg = TRLConfig(
        model=model_cfg,
        optimizer=opt_cfg,
        algorithm=algo_cfg,
        data=data_cfg,
    )
    dumped = full_cfg.model_dump()
    assert dumped["model"]["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert dumped["optimizer"]["learning_rate"] == 2e-5
    assert dumped["algorithm"]["epsilon_low"] == 0.15
    assert dumped["data"]["train_data"] == ["train.jsonl"]


def test_verl_configs_instantiate_and_serialize():
    """Ensure VERL configurations instantiate, dump, and reject invalid types."""
    model_cfg = VERLModelConfig(model_name="Qwen/Qwen2.5-7B")
    assert model_cfg.model_name == "Qwen/Qwen2.5-7B"

    opt_cfg = VERLOptimizerConfig(learning_rate=5e-6)
    algo_cfg = VERLAlgorithmConfig(algorithm="grpo")
    data_cfg = VERLDataConfig(train_data=["path/to/data.jsonl"])

    cfg = VERLConfig(
        model=model_cfg,
        optimizer=opt_cfg,
        algorithm=algo_cfg,
        data=data_cfg,
        epochs=10,
    )
    dumped = cfg.model_dump()
    assert dumped["epochs"] == 10
    assert dumped["data"]["train_data"] == ["path/to/data.jsonl"]

    # Rejection of invalid types
    with pytest.raises(ValidationError):
        VERLConfig(epochs="not-an-integer")  # type: ignore[arg-type]


def test_openrlhf_configs_instantiate_and_serialize():
    """Ensure OpenRLHF configurations instantiate and validate."""
    root_dir = Path(__file__).resolve().parent.parent
    openrlhf_config_path = root_dir / "platform_modal" / "openrlhf" / "config.py"
    mod = _load_module_directly("openrlhf_config", openrlhf_config_path)

    openrlhf_config_cls = mod.OpenRLHFConfig
    openrlhf_model_config_cls = mod.OpenRLHFModelConfig

    cfg = openrlhf_config_cls(model=openrlhf_model_config_cls(model_name="test-model"))
    dumped = cfg.model_dump()
    assert dumped["model"]["model_name"] == "test-model"
    assert dumped["algorithm"]["diagnostic_metric"] == "zvf"

    with pytest.raises(ValidationError):
        openrlhf_config_cls(model="not-a-model-config")  # type: ignore[arg-type]


def test_atropos_configs_instantiate_and_validate():
    """Ensure Atropos configurations support Pydantic schema validation."""
    env_cfg = EnvConfig(group_size=8, batch_size=64)
    assert env_cfg.group_size == 8
    assert env_cfg.batch_size == 64

    openai_cfg = OpenAIServerConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8001/v1",
    )
    assert openai_cfg.model_name == "meta-llama/Llama-3.1-8B-Instruct"

    tinker_cfg = TinkerConfig(lora_rank=16, learning_rate=1e-4)
    atropos_cfg = TinkerAtroposConfig(
        env=env_cfg,
        openai=[openai_cfg],
        tinker=tinker_cfg,
    )

    assert atropos_cfg.group_size == 8
    assert atropos_cfg.batch_size == 64
    assert atropos_cfg.base_model == "meta-llama/Llama-3.1-8B-Instruct"
    assert atropos_cfg.inference_api_url == "http://localhost:8001"

    dumped = atropos_cfg.model_dump()
    assert dumped["env"]["group_size"] == 8
    assert dumped["tinker"]["lora_rank"] == 16


def test_atropos_types_request_response_schemas():
    """Ensure request and response schemas validate properly with Pydantic."""
    comp_req = CompletionRequest(prompt="Write a fibonacci function", max_tokens=150)
    assert comp_req.prompt == "Write a fibonacci function"
    assert comp_req.max_tokens == 150

    comp_resp = CompletionResponse(
        id="cmpl-123",
        choices=[{"text": "def fib(n): ...", "index": 0}],
        created=1234567890,
        model="test-model",
    )
    assert comp_resp.id == "cmpl-123"

    chat_req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.7,
    )
    assert len(chat_req.messages) == 1
    assert chat_req.messages[0].role == "user"

    chat_resp = ChatCompletionResponse(
        id="chatcmpl-123",
        choices=[{"message": {"role": "assistant", "content": "Hi"}, "index": 0}],
        created=1234567890,
        model="test-model",
    )
    assert chat_resp.id == "chatcmpl-123"

    gen_req = GenerateRequest(text="prompt text", return_logprob=True)
    assert gen_req.text == "prompt text"
    assert gen_req.return_logprob is True

    # Required fields missing
    with pytest.raises(ValidationError):
        ChatMessage()  # type: ignore[call-arg]


def test_all_subprojects_use_same_pydantic_and_aligned_lib_versions():
    """Ensure all subprojects and manifests specify the same pydantic constraint and aligned libraries."""
    root_dir = Path(__file__).resolve().parent.parent

    manifests = [
        root_dir / "pyproject.toml",
        root_dir / "requirements.txt",
        root_dir / "platform_tinker" / "atropos" / "pyproject.toml",
        root_dir / "platform_tinker" / "atropos" / "requirements_unsloth.txt",
        root_dir / "zvf-program" / "zvf-triage" / "pyproject.toml",
        root_dir / "platform_hybrid" / "experiments" / "implementations" / "requirements.txt",
        root_dir / "platform_hf_spaces" / "defense_live_demo" / "requirements.txt",
    ]

    expected_pydantic = "pydantic>=2.13.0,<3.0.0"

    for manifest in manifests:
        assert manifest.exists(), f"Manifest {manifest} does not exist"
        content = manifest.read_text(encoding="utf-8")
        assert expected_pydantic in content, (
            f"{manifest.relative_to(root_dir)} does not contain '{expected_pydantic}'"
        )

    # Check key ML framework version alignment across manifests that declare them
    core_checks = [
        ("torch", "torch==2.7.1"),
        ("transformers", "transformers==5.5.4"),
        ("datasets", "datasets==4.8.4"),
        ("peft", "peft==0.19.1"),
        ("accelerate", "accelerate==1.13.0"),
        ("trl", "trl==1.2.0"),
        ("wandb", "wandb==0.21.0"),
        ("numpy", "numpy==2.2.6"),
    ]

    for manifest in [
        root_dir / "pyproject.toml",
        root_dir / "requirements.txt",
        root_dir / "platform_tinker" / "atropos" / "requirements_unsloth.txt",
        root_dir / "zvf-program" / "zvf-triage" / "pyproject.toml",
        root_dir / "platform_hybrid" / "experiments" / "implementations" / "requirements.txt",
    ]:
        content = manifest.read_text(encoding="utf-8")
        for lib, expected_str in core_checks:
            if lib in content:
                assert expected_str in content, (
                    f"{manifest.relative_to(root_dir)} mentions '{lib}' but does not match expected '{expected_str}'"
                )
