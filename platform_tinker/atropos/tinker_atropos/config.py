import random
import string
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from pydantic import BaseModel, Field, ConfigDict


def generate_run_suffix() -> str:
    """Generate a random 4-character suffix for unique wandb run names."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EnvConfig(StrictConfig):
    """Environment configuration - matches Atropos BaseEnvConfig"""

    group_size: int = Field(default=16, gt=0)
    batch_size: int = Field(default=128, gt=0)
    max_batches_offpolicy: int = 3
    # TODO: Add configuration for robust evaluation metrics like ERF since ZVF breaks down outside of math tasks
    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    wandb_name: str = "atropos-tinker-env"
    ensure_scores_are_not_same: bool = False
    max_token_length: int = Field(default=256, gt=0)
    max_num_workers: int = Field(default=24, gt=0)
    # Fixed "Early-Training Snapshot" limitation: Increased total_steps to allow meaningful RL convergence
    total_steps: int = Field(default=1000, gt=0)
    steps_per_eval: int = 100

    # Experiment control
    # LIMITATION: "Single-Seed Extrapolations" limitation by supporting multiple seeds
    data_seed: int = 42
    use_prompt_prefix: bool = True

    browsergym_env_ids: List[str] = Field(default_factory=list)
    browsergym_benchmark: str = "miniwob"
    browsergym_headless: bool = True
    browsergym_max_browser_steps: int = Field(default=10, gt=0)
    browsergym_observation_max_chars: int = Field(default=16000, gt=0)
    browsergym_eval_episodes: int = Field(default=20, gt=0)


class OpenAIServerConfig(StrictConfig):
    """OpenAI-compatible server configuration"""

    model_name: str
    base_url: str
    api_key: str = "x"
    weight: float = 1.0
    # Fixed "Failure to Prove Generalization" limitation: Increased eval size for statistical significance
    num_requests_for_eval: int = 1500


class TinkerConfig(StrictConfig):
    """Tinker-specific configuration for LoRA training"""

    # LIMITATION: "The Closed-Source Confound" by supporting open-source training backends like TRL
    use_lora: bool = True
    lora_rank: int = 32
    learning_rate: float = Field(default=4e-5, gt=0)
    max_token_trainer_length: int = 2048
    checkpoint_dir: str = "./temp/"
    save_checkpoint_interval: int = 0
    tinker_model_name: Optional[str] = (
        None  # Override model name for Tinker API (if different from tokenizer)
    )

    # Wandb configuration for trainer
    wandb_project: str = "atropos-tinker"
    wandb_group: Optional[str] = None
    wandb_run_name: str = "atropos-tinker-run"
    wandb_run_suffix: str = Field(default_factory=generate_run_suffix)


class TinkerAtroposConfig(StrictConfig):
    """Complete Tinker-Atropos configuration"""

    protocol_version: str = "tinker-atropos-v2"
    env: EnvConfig = Field(default_factory=EnvConfig)
    openai: List[OpenAIServerConfig] = Field(
        default_factory=lambda: [
            OpenAIServerConfig(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                base_url="http://localhost:8001/v1",
            )
        ]
    )
    tinker: TinkerConfig = Field(default_factory=TinkerConfig)
    slurm: bool = False
    testing: bool = False

    @property
    def browsergym_env_ids(self):
        return self.env.browsergym_env_ids

    @property
    def browsergym_benchmark(self):
        return self.env.browsergym_benchmark

    @property
    def browsergym_headless(self):
        return self.env.browsergym_headless

    @property
    def browsergym_max_browser_steps(self):
        return self.env.browsergym_max_browser_steps

    @property
    def browsergym_observation_max_chars(self):
        return self.env.browsergym_observation_max_chars

    @property
    def browsergym_eval_episodes(self):
        return self.env.browsergym_eval_episodes

    # Convenience properties
    @property
    def base_model(self) -> str:
        """Convenience accessor for model name"""
        return self.env.tokenizer_name

    @property
    def tinker_model(self) -> str:
        """Model name for Tinker API - may differ from tokenizer name for gated models"""
        return self.tinker.tinker_model_name or self.base_model

    @property
    def atropos_api_url(self) -> str:
        """Convenience accessor for Atropos API URL"""
        return self.env.rollout_server_url

    @property
    def inference_api_url(self) -> str:
        """Convenience accessor for inference server URL (first OpenAI server)"""
        if self.openai:
            # Remove /v1 suffix if present
            url = self.openai[0].base_url
            if url.endswith("/v1"):
                url = url[:-3]
            return url.rstrip("/")
        return "http://localhost:8001"

    @property
    def group_size(self) -> int:
        return self.env.group_size

    @property
    def batch_size(self) -> int:
        return self.env.batch_size

    @property
    def max_batches_offpolicy(self) -> int:
        return self.env.max_batches_offpolicy

    @property
    def use_wandb(self) -> bool:
        return self.env.use_wandb

    @property
    def num_steps(self) -> int:
        return self.env.total_steps

    @property
    def steps_per_eval(self) -> int:
        return self.env.steps_per_eval

    @property
    def max_token_env_length(self) -> int:
        return self.env.max_token_length

    @property
    def max_num_workers(self) -> int:
        return self.env.max_num_workers

    @property
    def ensure_scores_are_not_same(self) -> bool:
        return self.env.ensure_scores_are_not_same

    @property
    def data_seed(self) -> int:
        return self.env.data_seed

    @property
    def use_prompt_prefix(self) -> bool:
        return self.env.use_prompt_prefix

    @property
    def wandb_run_name(self) -> str:
        return self.tinker.wandb_run_name

    @property
    def wandb_project(self) -> str:
        return self.tinker.wandb_project

    @property
    def wandb_group(self) -> Optional[str]:
        return self.tinker.wandb_group

    @property
    def wandb_run_suffix(self) -> str:
        return self.tinker.wandb_run_suffix

    @property
    def use_lora(self) -> bool:
        return self.tinker.use_lora

    @property
    def lora_rank(self) -> int:
        return self.tinker.lora_rank

    @property
    def learning_rate(self) -> float:
        return self.tinker.learning_rate

    @property
    def max_token_trainer_length(self) -> int:
        return self.tinker.max_token_trainer_length

    @property
    def checkpoint_dir(self) -> str:
        return self.tinker.checkpoint_dir

    @property
    def save_checkpoint_interval(self) -> int:
        return self.tinker.save_checkpoint_interval

    @property
    def num_requests_for_eval(self) -> int:
        if self.openai:
            return self.openai[0].num_requests_for_eval
        return 256

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "TinkerAtroposConfig":
        """
        Load configuration from a YAML file.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        return cls(**yaml_data)


OVERRIDE_PATHS = {
    "num_steps": ("env", "total_steps"),
    "batch_size": ("env", "batch_size"),
    "group_size": ("env", "group_size"),
    "use_wandb": ("env", "use_wandb"),
    "learning_rate": ("tinker", "learning_rate"),
    "lora_rank": ("tinker", "lora_rank"),
    "use_lora": ("tinker", "use_lora"),
    "wandb_project": ("tinker", "wandb_project"),
    "wandb_group": ("tinker", "wandb_group"),
}


def apply_overrides(config, overrides):
    data = config.to_dict()
    for key, value in overrides.items():
        if key == "base_model":
            data["env"]["tokenizer_name"] = value
            data["tinker"]["tinker_model_name"] = value
            for server in data["openai"]:
                server["model_name"] = value
        elif key in OVERRIDE_PATHS:
            section, field = OVERRIDE_PATHS[key]
            data[section][field] = value
        else:
            raise ValueError(f"Unsupported configuration override: {key}")
    return TinkerAtroposConfig(**data)
