"""
OpenRLHF Configuration for tinker-rl-lab
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator


class OpenRLHFModelConfig(BaseModel):
    """Model configuration"""
    model_config = ConfigDict(extra="forbid")
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: Optional[str] = None
    revision: Optional[str] = None  # immutable Hub commit, or explicit local model_path
    use_flash_attention: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    lora_rank: int = Field(default=0, ge=0)


class OpenRLHFOptimizerConfig(BaseModel):
    """Optimizer configuration"""
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01


class OpenRLHFAlgorithmConfig(BaseModel):
    """RL algorithm configuration"""
    model_config = ConfigDict(extra="forbid")
    algorithm: str = "grpo"  # this versioned driver supports GRPO only
    gamma: float = 1.0
    lam: float = 0.95
    epsilon: float = 0.2  # PPO clip
    kl_coef: float = 0.01
    temperature: float = 1.0
    sample_num: int = 16
    # LIMITATION: ZVF is fragile outside math/format-gated tasks.
    # ZVF=1 means every reward within each group agrees; it is not a task-gradient measurement.
    # KL/other losses can remain active at zero reward contrast.
    diagnostic_metric: str = "zvf"


class OpenRLHFDataConfig(BaseModel):
    """Data configuration"""
    model_config = ConfigDict(extra="forbid")
    train_data: List[str] = Field(default_factory=list)
    val_data: List[str] = Field(default_factory=list)
    # LIMITATION: held-out generalization not established for this dryrun/short path.
    test_data: List[str] = Field(default_factory=list)
    dataset_revision: Optional[str] = None
    max_prompt_length: int = 512
    max_response_length: int = 1024


class OpenRLHFConfig(BaseModel):
    """Complete OpenRLHF training configuration"""
    model_config = ConfigDict(extra="forbid")

    execution_contract: Literal["candidate-0.8.0", "native-0.8.4"] = "candidate-0.8.0"

    @model_validator(mode="before")
    @classmethod
    def contract_defaults(cls, values):
        # Defaults are versioned; an explicit user setting is never rewritten.
        if isinstance(values, dict) and values.get("execution_contract") == "native-0.8.4":
            values = dict(values)
            for name, default in {"epochs": 20, "train_batch_size": 128,
                                  "max_steps": None, "eval_interval": 5}.items():
                values.setdefault(name, default)
        return values

    model: OpenRLHFModelConfig = Field(default_factory=OpenRLHFModelConfig)
    optimizer: OpenRLHFOptimizerConfig = Field(default_factory=OpenRLHFOptimizerConfig)
    algorithm: OpenRLHFAlgorithmConfig = Field(default_factory=OpenRLHFAlgorithmConfig)
    data: OpenRLHFDataConfig = Field(default_factory=OpenRLHFDataConfig)

    # Training settings
    epochs: int = 1  # optimizer epochs per rollout
    train_batch_size: int = 128  # completions, not prompts
    micro_batch_size: int = 1
    # LIMITATION: default seed path is N=1 unless callers parameterize multi-seed.
    seed: int = 42
    # LIMITATION: short max_samples/steps yield early-training snapshots only.
    max_steps: Optional[int] = 30  # optimizer updates, not reward observations
    wall_timeout_s: int = 7200

    # Native mode requires whole prompt/global batches per rollout.
    # Explicit dataset budget; this is not an optimizer-step budget.
    max_samples: int = Field(default=240, gt=0, le=500)
    rollout_batch_size: int = Field(default=8, gt=0)
    remote_rm_url: Optional[str] = None

    # Ray cluster settings
    num_gpus: int = 1
    # Legacy compatibility fields: native mode accepts only these defaults.
    # They do not control native placement or periodic evaluation/checkpoints.
    num_workers: int = 1
    num_actors: int = 1
    actor_num_gpus: int = 1
    critic_num_gpus: int = 1
    eval_interval: int = -1  # held-out evaluation unsupported in this driver
    save_interval: int = 10

    # Logging
    project_name: str = "openrlhf-tinker"
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    # Explicit opt-in: native W&B suppresses TensorBoard in v0.8.4.
    log_to_wandb: bool = False

    # vLLM settings
    inference_backend: str = "vllm"
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    vllm_num_engines: int = 1

    # Environment
    env_class: str = "gsm8k"
    # LIMITATION: do not attribute Tinker managed-default gaps to the open algorithm alone.

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
