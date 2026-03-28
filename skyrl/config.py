"""
SkyRL Configuration for tinker-rl-lab

Matches the SkyRL training configuration structure while supporting
vast.ai and Google Colab backends.
"""

import random
import string
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from pydantic import BaseModel, Field


def generate_run_suffix() -> str:
    """Generate a random 4-character suffix for unique wandb run names."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


class SkyRLPolicyConfig(BaseModel):
    """Policy model configuration"""

    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: Optional[str] = None  # Local path override
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = Field(
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
    use_gradient_checkpointing: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False


class SkyRLOptimizerConfig(BaseModel):
    """Optimizer configuration"""

    learning_rate: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01


class SkyRLAlgorithmConfig(BaseModel):
    """RL algorithm configuration"""

    algorithm: str = "grpo"  # grpo, ppo, reinforce
    advantage_estimator: str = "grpo"  # grpo,GAE,gae
    use_kl_loss: bool = True
    kl_coef: float = 0.01
    gamma: float = 1.0
    lam: float = 0.95
    epsilon: float = 0.2  # PPO clip


class SkyRLDataConfig(BaseModel):
    """Data configuration"""

    train_data: List[str] = Field(default_factory=list)
    val_data: List[str] = Field(default_factory=list)
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    data_seed: int = 42


class SkyRLGeneratorConfig(BaseModel):
    """Inference generator configuration"""

    n_samples_per_prompt: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    max_generate_length: int = 1024
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1


class SkyRLPlacementConfig(BaseModel):
    """Multi-GPU/multi-node placement configuration"""

    policy_num_gpus_per_node: int = 1
    ref_num_gpus_per_node: int = 1
    policy_num_nodes: int = 1
    ref_num_nodes: int = 1
    colocate_all: bool = False


class SkyRLBackendConfig(BaseModel):
    """Backend configuration for compute resources"""

    backend: str = "auto"  # auto, vastai, colab, tinker, local
    # vast.ai specific
    vastai_api_key: Optional[str] = None
    vastai_instance_type: Optional[str] = "a100-80gb"
    vastai_num_instances: int = 1
    # Colab specific
    use_colab: bool = False
    colab_gpu: str = "T4"  # T4, A100, L4
    # Tinker API
    tinker_api_key: Optional[str] = None
    tinker_endpoint: Optional[str] = None


class SkyRLLoggerConfig(BaseModel):
    """Logging configuration"""

    logger: str = "wandb"  # wandb, tensorboard, console
    project_name: str = "skyrl-tinker"
    run_name: Optional[str] = None
    run_group: Optional[str] = None
    wandb_api_key: Optional[str] = None


class SkyRLConfig(BaseModel):
    """Complete SkyRL training configuration"""

    # Core settings
    trainer: SkyRLAlgorithmConfig = Field(default_factory=SkyRLAlgorithmConfig)
    policy: SkyRLPolicyConfig = Field(default_factory=SkyRLPolicyConfig)
    optimizer: SkyRLOptimizerConfig = Field(default_factory=SkyRLOptimizerConfig)
    data: SkyRLDataConfig = Field(default_factory=SkyRLDataConfig)
    generator: SkyRLGeneratorConfig = Field(default_factory=SkyRLGeneratorConfig)
    placement: SkyRLPlacementConfig = Field(default_factory=SkyRLPlacementConfig)
    backend: SkyRLBackendConfig = Field(default_factory=SkyRLBackendConfig)
    logger: SkyRLLoggerConfig = Field(default_factory=SkyRLLoggerConfig)

    # Training settings
    epochs: int = 20
    train_batch_size: int = 1024
    policy_mini_batch_size: int = 256
    micro_train_batch_size_per_gpu: int = 40
    micro_forward_batch_size_per_gpu: int = 40
    update_epochs_per_batch: int = 1
    eval_batch_size: int = 1024
    eval_interval: int = 5
    eval_before_train: bool = True
    ckpt_interval: int = 10
    ckpt_path: str = "./checkpoints/"
    resume_mode: str = "null"  # null, full, finetune

    # Inference backend
    inference_backend: str = "vllm"  # vllm, sglang, tinker
    run_engines_locally: bool = True
    weight_sync_backend: str = "nccl"  # nccl, local
    async_engine: bool = True
    batched: bool = True
    num_inference_engines: int = 1

    # Environment
    environment_env_class: str = "gsm8k"

    # Convenience properties
    @property
    def model_name(self) -> str:
        return self.policy.model_name

    @property
    def learning_rate(self) -> float:
        return self.optimizer.learning_rate

    @property
    def lora_rank(self) -> int:
        return self.policy.lora_rank

    @property
    def wandb_project(self) -> str:
        return self.logger.project_name

    @property
    def wandb_group(self) -> Optional[str]:
        return self.logger.run_group

    @property
    def wandb_run_name(self) -> str:
        return self.logger.run_name or f"skyrl-run-{generate_run_suffix()}"

    def to_skyrl_dict(self) -> Dict[str, Any]:
        """Convert to SkyRL-style config dict for skyrl-train."""
        return {
            "data.train_data": self.data.train_data,
            "data.val_data": self.data.val_data,
            "trainer.algorithm.advantage_estimator": self.trainer.advantage_estimator,
            "trainer.policy.model.path": self.policy.model_name,
            "trainer.placement.colocate_all": str(self.placement.colocate_all).lower(),
            "trainer.strategy": "fsdp2",
            "trainer.placement.policy_num_gpus_per_node": str(self.placement.policy_num_gpus_per_node),
            "trainer.placement.ref_num_gpus_per_node": str(self.placement.ref_num_gpus_per_node),
            "trainer.placement.ref_num_nodes": str(self.placement.ref_num_nodes),
            "trainer.placement.policy_num_nodes": str(self.placement.policy_num_nodes),
            "generator.num_inference_engines": str(self.num_inference_engines),
            "generator.inference_engine_tensor_parallel_size": str(self.generator.tensor_parallel_size),
            "trainer.epochs": str(self.epochs),
            "trainer.eval_batch_size": str(self.eval_batch_size),
            "trainer.eval_before_train": str(self.eval_before_train).lower(),
            "trainer.eval_interval": str(self.eval_interval),
            "trainer.update_epochs_per_batch": str(self.update_epochs_per_batch),
            "trainer.train_batch_size": str(self.train_batch_size),
            "trainer.policy_mini_batch_size": str(self.policy_mini_batch_size),
            "trainer.micro_forward_batch_size_per_gpu": str(self.micro_forward_batch_size_per_gpu),
            "trainer.micro_train_batch_size_per_gpu": str(self.micro_train_batch_size_per_gpu),
            "trainer.ckpt_interval": str(self.ckpt_interval),
            "trainer.max_prompt_length": str(self.data.max_prompt_length),
            "generator.sampling_params.max_generate_length": str(self.generator.max_generate_length),
            "trainer.policy.optimizer_config.lr": str(self.optimizer.learning_rate),
            "trainer.algorithm.use_kl_loss": str(self.trainer.use_kl_loss).lower(),
            "generator.backend": self.inference_backend,
            "generator.run_engines_locally": str(self.run_engines_locally).lower(),
            "generator.weight_sync_backend": self.weight_sync_backend,
            "generator.async_engine": str(self.async_engine).lower(),
            "generator.batched": str(self.batched).lower(),
            "environment.env_class": self.environment_env_class,
            "generator.n_samples_per_prompt": str(self.generator.n_samples_per_prompt),
            "generator.gpu_memory_utilization": str(self.generator.gpu_memory_utilization),
            "trainer.logger": self.logger.logger,
            "trainer.project_name": self.logger.project_name,
            "trainer.run_name": self.wandb_run_name,
            "trainer.resume_mode": self.resume_mode,
            "trainer.ckpt_path": self.ckpt_path,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "SkyRLConfig":
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        return cls(**yaml_data)

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
