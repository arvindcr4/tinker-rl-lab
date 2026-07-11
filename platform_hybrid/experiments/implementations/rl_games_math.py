"""
rl_games (NVIDIA) PPO Math RL Implementation
=============================================
Port of Tinker Math RL to NVIDIA's rl_games library.

rl_games is designed for high-performance GPU training.
Used in Isaac Gym for robotics simulation.
"""

import os
import sys
import logging
import argparse
import tempfile
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

import wandb
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.seed import set_global_seed, get_seed_from_args


@dataclass
class AlgoConfig:
    name: str = "a2c_discrete"


@dataclass
class ModelConfig:
    name: str = "discrete_a2c"


@dataclass
class MLPConfig:
    units: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "elu"
    initializer: Dict[str, str] = field(default_factory=lambda: {"name": "default"})


@dataclass
class SpaceConfig:
    discrete: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkConfig:
    name: str = "actor_critic"
    separate: bool = False
    space: SpaceConfig = field(default_factory=SpaceConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)


@dataclass
class PPOConfig:
    name: str = "arithmetic_ppo"
    env_name: str = "arithmetic"
    score_to_win: float = 0.95
    normalize_input: bool = True
    normalize_value: bool = True
    num_actors: int = 16
    horizon_length: int = 128
    minibatch_size: int = 512
    mini_epochs: int = 4
    gamma: float = 0.99
    tau: float = 0.95
    e_clip: float = 0.2
    entropy_coef: float = 0.01
    critic_coef: float = 0.5
    learning_rate: float = 1e-4
    lr_schedule: str = "constant"
    grad_norm: float = 0.5
    max_epochs: int = 1000
    device: str = "cuda:0"
    device_name: str = "cuda:0"


@dataclass
class RLGamesParams:
    seed: int = 42
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    config: PPOConfig = field(default_factory=PPOConfig)


@dataclass
class RLGamesConfig:
    params: RLGamesParams = field(default_factory=RLGamesParams)


class ArithmeticEnv(gym.Env):
    """Arithmetic environment for rl_games."""

    def __init__(self, max_num: int = 99) -> None:
        super().__init__()
        self.max_num = max_num
        self.max_answer = max_num * 2

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_answer + 1)

        self.current_nums: Optional[np.ndarray] = None
        self.correct_answer: Optional[int] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_nums = self.np_random.integers(1, self.max_num + 1, size=2)
        self.correct_answer = int(self.current_nums.sum())
        obs = self.current_nums.astype(np.float32) / self.max_num
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 1.0 if action == self.correct_answer else 0.0
        obs = self.current_nums.astype(np.float32) / self.max_num
        return obs, reward, True, False, {"correct": reward == 1.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rl_games (NVIDIA) PPO Math RL")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wandb-project", type=str, default="tinker-rl-games", help="WandB project name")
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace repo ID for checkpointing")
    return parser.parse_known_args()[0]


def main() -> None:
    """
    Main function for rl_games training.

    Note: Full rl_games integration requires rl_games package.
    This shows the configuration pattern.
    """
    args = parse_args()
    
    # Prioritize argparse seed over utility seed
    seed = args.seed if args.seed is not None else get_seed_from_args()
    set_global_seed(seed)
    
    wandb.init(project=args.wandb_project, config=vars(args))

    logger.info("=" * 60)
    logger.info("rl_games (NVIDIA) PPO Math RL Configuration")
    logger.info("=" * 60)

    config = RLGamesConfig()
    config.params.seed = seed
    config.params.config.device = args.device
    config.params.config.device_name = args.device
    config.params.config.learning_rate = args.learning_rate

    logger.info("PPO Configuration:")
    ppo_config = config.params.config
    for key in ["learning_rate", "e_clip", "gamma", "tau", "mini_epochs"]:
        logger.info("  %s: %s", key, getattr(ppo_config, key))

    logger.info("Network Configuration:")
    net_config = config.params.network.mlp
    logger.info("  units: %s", net_config.units)
    logger.info("  activation: %s", net_config.activation)

    # Test environment
    logger.info("--- Testing Environment ---")
    env = ArithmeticEnv(max_num=99)
    obs, _ = env.reset()
    logger.info("Observation shape: %s", obs.shape)
    logger.info("Action space: %s", env.action_space)

    # Random baseline
    correct = 0
    for epoch in range(100):
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        if info["correct"]:
            correct += 1
        wandb.log({"reward": reward, "epoch": epoch})

    accuracy = correct / 100.0
    logger.info("Random baseline: %s%%", accuracy * 100)
    wandb.log({"accuracy": accuracy})

    def push_to_hub(repo_id: str, model_info: dict):
        logger.info(f"Pushing to HuggingFace Hub repo: {repo_id}")
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create repo (might exist): {e}")
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.json")
            with open(model_path, "w") as f:
                json.dump(model_info, f)
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="model.json",
                repo_id=repo_id,
                commit_message="Add rl_games model checkpoint"
            )

    if args.hf_repo:
        push_to_hub(args.hf_repo, {"accuracy": accuracy, "seed": seed, "config": vars(args)})

    wandb.finish()

    # Full rl_games training message
    logger.info("--- Full rl_games Training (requires rl_games) ---")
    logger.info("To integrate, use env_configurations.register('arithmetic', ...) and Runner().load(config)")


if __name__ == "__main__":
    main()
