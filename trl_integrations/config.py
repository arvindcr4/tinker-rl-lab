"""
TRL Configuration for tinker-rl-lab
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field


class TRLModelConfig(BaseModel):
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: Optional[str] = None
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_peft: bool = True
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0


class TROptimizerConfig(BaseModel):
    """Optimizer configuration"""
    learning_rate: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01


class TRLAlgorithmConfig(BaseModel):
    """RL algorithm configuration"""
    algorithm: str = "grpo"  # grpo, ppo, reinforce, dpo
    gamma: float = 1.0
    lam: float = 0.95
    epsilon: float = 0.2  # PPO clip
    kl_coef: float = 0.01
    max_grad_norm: float = 1.0


class TRLDataConfig(BaseModel):
    """Data configuration"""
    train_data: List[str] = Field(default_factory=list)
    val_data: List[str] = Field(default_factory=list)
    max_prompt_length: int = 512
    max_response_length: int = 1024
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 4


class TRLConfig(BaseModel):
    """Complete TRL training configuration"""

    model: TRLModelConfig = Field(default_factory=TRLModelConfig)
    optimizer: TROptimizerConfig = Field(default_factory=TROptimizerConfig)
    algorithm: TRLAlgorithmConfig = Field(default_factory=TRLAlgorithmConfig)
    data: TRLDataConfig = Field(default_factory=TRLDataConfig)

    # Training settings
    epochs: int = 20
    micro_batch_size: int = 1
    eval_interval: int = 5
    save_interval: int = 10
    max_steps: int = -1  # -1 for epochs

    # Hardware
    num_gpus: int = 1
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True

    # Logging
    project_name: str = "trl-tinker"
    run_name: Optional[str] = None
    report_to: str = "wandb"

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
    def from_yaml(cls, yaml_path: str | Path) -> "TRLConfig":
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
