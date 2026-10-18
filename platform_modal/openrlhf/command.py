"""Pure, CPU-testable command contract for the OpenRLHF v0.8.4 CLI."""

import math
import os
import sys
from urllib.parse import urlsplit

from .config import OpenRLHFConfig
from .budget import plan_native_budget


def _validate_reward_service_url(value: object) -> None:
    # OpenRLHF prints its argparse namespace. Never accept URL userinfo, and
    # reject characters that urllib may silently strip or reinterpret.
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("remote_rm_url must be an HTTP(S) URL without whitespace or controls")
    try:
        url = urlsplit(value)
        port = url.port
        hostname = url.hostname
    except ValueError:
        # Parser diagnostics may echo credential-bearing input.
        raise ValueError("remote_rm_url has an invalid host or port") from None
    if url.scheme not in {"http", "https"} or not hostname:
        raise ValueError("remote_rm_url must name an explicitly configured HTTP(S) reward service")
    if "@" in url.netloc:
        raise ValueError("remote_rm_url must not contain userinfo; native arguments are logged")
    if (port is not None and not 1 <= port <= 65535) or url.netloc.endswith(":"):
        raise ValueError("remote_rm_url has an invalid port")
    if url.fragment:
        raise ValueError("remote_rm_url must not contain a fragment")


def build_openrlhf_command(config: OpenRLHFConfig, data_path: str, output_dir: str) -> list[str]:
    if (
        not isinstance(config.algorithm.algorithm, str)
        or config.algorithm.algorithm.lower() != "grpo"
    ):
        raise ValueError("Native OpenRLHF driver supports only algorithm=grpo")
    plan_native_budget(config)
    numeric = {
        "learning_rate": config.optimizer.learning_rate,
        "beta1": config.optimizer.beta1,
        "beta2": config.optimizer.beta2,
        "eps": config.optimizer.eps,
        "weight_decay": config.optimizer.weight_decay,
        "gamma": config.algorithm.gamma,
        "lam": config.algorithm.lam,
        "epsilon": config.algorithm.epsilon,
        "kl_coef": config.algorithm.kl_coef,
        "temperature": config.algorithm.temperature,
        "gpu_memory_utilization": config.gpu_memory_utilization,
    }
    for name, value in numeric.items():
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite number for native training")
    for name in ("learning_rate", "weight_decay", "epsilon"):
        if numeric[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    if numeric["temperature"] <= 0:
        raise ValueError("temperature must be strictly positive for grouped GRPO rollouts")
    for name in ("gamma", "lam"):
        if not 0 <= numeric[name] <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if numeric["kl_coef"] < 0:
        raise ValueError("kl_coef must be nonnegative")
    for name in ("beta1", "beta2"):
        if not 0 <= numeric[name] < 1:
            raise ValueError(f"{name} must be in [0, 1)")
    if type(config.seed) is not int:
        raise ValueError("seed must be an integer")
    if type(config.model.lora_rank) is not int or config.model.lora_rank < 0:
        raise ValueError("lora_rank must be a nonnegative integer")
    # These historical knobs were never mapped by this native driver. Keep
    # loading old default configurations, but do not silently accept overrides.
    legacy_defaults = {
        "num_workers": 1,
        "num_actors": 1,
        "actor_num_gpus": 1,
        "critic_num_gpus": 1,
        "eval_interval": 5,
        "save_interval": 10,
    }
    unsupported = [
        name for name, default in legacy_defaults.items() if getattr(config, name) != default
    ]
    if unsupported:
        raise ValueError(
            "Native OpenRLHF does not support overrides for "
            + ", ".join(unsupported)
            + "; remove these legacy keys. Use num_gpus and tensor_parallel_size "
            "for the single colocated engine. Periodic evaluation and checkpoints "
            "are disabled; final model saving is separate."
        )
    _validate_reward_service_url(config.remote_rm_url)
    if config.env_class != "gsm8k" or any(
        (config.data.train_data, config.data.val_data, config.data.test_data)
    ):
        raise ValueError("This native driver supports only its fixed GSM8K-500 dataset")
    if config.model.load_in_8bit or config.optimizer.eps != 1e-8:
        raise ValueError("The v0.8.4 command does not support custom Adam eps or 8-bit loading")
    if config.inference_backend != "vllm":
        raise ValueError("This native driver requires inference_backend=vllm")
    if config.algorithm.sample_num < 2:
        raise ValueError("GRPO requires sample_num >= 2")
    positive = (
        config.epochs,
        config.train_batch_size,
        config.micro_batch_size,
        config.num_gpus,
        config.tensor_parallel_size,
        config.data.max_prompt_length,
        config.data.max_response_length,
    )
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError(
            "Training sizes, epochs, token caps and GPU counts must be positive integers"
        )
    if config.tensor_parallel_size > config.num_gpus:
        raise ValueError("colocated tensor_parallel_size cannot exceed num_gpus")
    if config.tensor_parallel_size != config.num_gpus:
        # v0.8.4 checks this only after waiting for a Ray placement group.
        # This wrapper uses one synchronous, colocated vLLM engine.
        raise ValueError(
            "The single colocated vLLM engine requires tensor_parallel_size == num_gpus"
        )
    if not 0 < config.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    run_name = config.run_name or "openrlhf-grpo"
    if run_name in {".", ".."} or any(char in run_name for char in ("/", "\\")):
        raise ValueError("run_name must be a single directory component for native logging")
    options = {
        "pretrain": config.model.model_path or config.model_name,
        "prompt_data": data_path,
        "input_key": "prompt",
        "label_key": "label",
        "advantage_estimator": "group_norm",
        "n_samples_per_prompt": config.algorithm.sample_num,
        "train_batch_size": config.train_batch_size,
        "micro_train_batch_size": config.micro_batch_size,
        "rollout_batch_size": config.rollout_batch_size,
        "micro_rollout_batch_size": config.micro_batch_size,
        "max_epochs": config.epochs,
        "num_episodes": 1,
        "prompt_max_len": config.data.max_prompt_length,
        "generate_max_len": config.data.max_response_length,
        "actor_learning_rate": config.learning_rate,
        "init_kl_coef": config.algorithm.kl_coef,
        "temperature": config.algorithm.temperature,
        "eps_clip": config.algorithm.epsilon,
        "gamma": config.algorithm.gamma,
        "lambd": config.algorithm.lam,
        "l2": config.optimizer.weight_decay,
        "kl_estimator": "k3",
        "save_steps": -1,
        "eval_steps": -1,
        "logging_steps": 1,
        "max_samples": config.max_samples,
        "save_path": os.path.join(output_dir, "ckpt"),
        "ckpt_path": os.path.join(output_dir, "ckpt"),
        "vllm_num_engines": 1,
        "vllm_tensor_parallel_size": config.tensor_parallel_size,
        "vllm_gpu_memory_utilization": config.gpu_memory_utilization,
        "actor_num_gpus_per_node": config.num_gpus,
        "ref_num_gpus_per_node": config.num_gpus,
        "remote_rm_url": config.remote_rm_url,
        "seed": config.seed,
        "lora_rank": config.model.lora_rank,
    }
    command = [sys.executable, "-m", "openrlhf.cli.train_ppo_ray"]
    for name, value in options.items():
        command.extend(["--" + name, str(value)])
    command.extend(["--adam_betas", str(config.optimizer.beta1), str(config.optimizer.beta2)])
    command.extend(["--use_kl_loss", "--colocate_all_models", "--bf16", "--apply_chat_template"])
    if config.model.use_flash_attention:
        command.append("--flash_attn")
    if config.model.load_in_4bit:
        command.append("--load_in_4bit")
    command.extend(["--wandb_run_name", run_name])
    if config.log_to_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("log_to_wandb requires WANDB_API_KEY")
        command.extend(
            [
                "--use_wandb",
                # v0.8.4 logs its entire argparse namespace to W&B and stdout.
                # The SDK reads WANDB_API_KEY from the inherited environment;
                # this truthy selector must never contain the credential.
                "environment",
                "--wandb_project",
                config.project_name,
            ]
        )
    else:
        command.extend(["--use_tensorboard", os.path.join(output_dir, "tensorboard")])
    return command
