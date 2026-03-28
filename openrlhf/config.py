"""
OpenRLHF Configuration for tinker-rl-lab
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field


class OpenRLHFModelConfig(BaseModel):
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: Optional[str] = None
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class OpenRLHFOptimizerConfig(BaseModel):
    """Optimizer configuration"""
    learning_rate: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01


class OpenRLHFAlgorithmConfig(BaseModel):
    """RL algorithm configuration"""
    algorithm: str = "dapo"  # dapo, ppo, reinforce, dpo
    gamma: float = 1.0
    lam: float = 0.95
    epsilon: float = 0.2  # PPO clip
    kl_coef: float = 0.01
    temperature: float = 1.0
    sample_num: int = 16


class OpenRLHFDataConfig(BaseModel):
    """Data configuration"""
    train_data: List[str] = Field(default_factory=list)
    val_data: List[str] = Field(default_factory=list)
    max_prompt_length: int = 512
    max_response_length: int = 1024


class OpenRLHFConfig(BaseModel):
    """Complete OpenRLHF training configuration"""

    model: OpenRLHFModelConfig = Field(default_factory=OpenRLHFModelConfig)
    optimizer: OpenRLHFOptimizerConfig = Field(default_factory=OpenRLHFOptimizerConfig)
    algorithm: OpenRLHFAlgorithmConfig = Field(default_factory=OpenRLHFAlgorithmConfig)
    data: OpenRLHFDataConfig = Field(default_factory=OpenRLHFDataConfig)

    # Training settings
    epochs: int = 20
    train_batch_size: int = 1024
    micro_batch_size: int = 1

    # Ray cluster settings
    num_gpus: int = 1
    num_workers: int = 1
    num_actors: int = 1
    actor_num_gpus: int = 1
    critic_num_gpus: int = 1
    eval_interval: int = 5
    save_interval: int = 10

    # Logging
    project_name: str = "openrlhf-tinker"
    run_name: Optional[str] = None

    # vLLM settings
    inference_backend: str = "vllm"
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1

    # Environment
    env_class: str = "gsm8k"

    @property
    def model_name(self) -> str:
        return self.model.model_name

    @property
    def learning_rate(self) -> float:
        return self.optimizer.learning_rate

    @property
    def wandb_project(self) -> str:
        return self.project_name

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "OpenRLHFConfig":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        return cls(**yaml_data)

    def to_yaml(self, yaml_path: str | Path) -> None:
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
