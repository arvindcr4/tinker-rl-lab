"""Pure, CPU-testable command contract for the OpenRLHF v0.8.4 CLI."""

import os
import sys
from urllib.parse import urlsplit

from .config import OpenRLHFConfig
from .budget import plan_native_budget


def build_openrlhf_command(config: OpenRLHFConfig, data_path: str, output_dir: str) -> list[str]:
    if config.algorithm.algorithm.lower() != "grpo":
        raise ValueError("Native OpenRLHF driver supports only algorithm=grpo")
    plan_native_budget(config)
    url = urlsplit(config.remote_rm_url or "")
    if url.scheme not in {"http", "https"} or not url.netloc:
        raise ValueError("remote_rm_url must name an explicitly configured HTTP(S) reward service")
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
    if any(value < 1 for value in positive):
        raise ValueError("Training sizes, epochs, token caps and GPU counts must be positive")
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
        "ref_pretrain": config.model.model_path or config.model_name,
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
    command.extend(["--use_kl_loss", "--colocate_all_models", "--bf16"])
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
