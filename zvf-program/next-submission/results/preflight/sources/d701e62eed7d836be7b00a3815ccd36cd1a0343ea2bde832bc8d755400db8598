#!/usr/bin/env python3
"""Fail closed unless Colab provides the pinned preflight stack and adapter seam."""

from importlib import metadata
import inspect
import json

import torch
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, Qwen3Config


EXPECTED = {
    "trl": "1.8.0",
    "transformers": "5.13.1",
    "datasets": "4.8.5",
    "peft": "0.19.1",
    "torchao": "0.17.0",
    "wandb": "0.28.0",
}
REQUIRED_CONFIG_FIELDS = {
    "generation_batch_size",
    "num_generations",
    "max_completion_length",
    "importance_sampling_level",
    "scale_rewards",
    "loss_type",
}


versions = {name: metadata.version(name) for name in EXPECTED}
wrong = {
    name: {"expected": EXPECTED[name], "observed": value}
    for name, value in versions.items()
    if value != EXPECTED[name]
}
if wrong:
    raise RuntimeError("locked package mismatch: " + json.dumps(wrong, sort_keys=True))
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError("preflight requires CUDA with bfloat16 support")
missing = sorted(REQUIRED_CONFIG_FIELDS - set(inspect.signature(GRPOConfig).parameters))
if missing:
    raise RuntimeError(f"GRPOConfig lacks required fields: {missing}")
if "rollout_func" not in inspect.signature(GRPOTrainer).parameters:
    raise RuntimeError("GRPOTrainer lacks the custom rollout seam")
generate_source = inspect.getsource(GRPOTrainer._generate)
if "env_mask" not in generate_source or "extra_fields" not in generate_source:
    raise RuntimeError("GRPOTrainer lacks inactive-slot mask propagation")

tiny_model = AutoModelForCausalLM.from_config(
    Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
)
get_peft_model(
    tiny_model,
    LoraConfig(r=2, lora_alpha=4, target_modules="all-linear", task_type="CAUSAL_LM"),
)

print(
    "NEXT_PREFLIGHT_ENV_OK "
    + json.dumps(
        {
            "versions": versions,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
        },
        sort_keys=True,
    ),
    flush=True,
)
