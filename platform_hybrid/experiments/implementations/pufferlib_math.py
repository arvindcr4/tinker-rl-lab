"""
PufferLib Math RL Implementation
=================================
Port of Tinker Math RL to PufferLib for high-throughput training.

PufferLib features:
- VTrace for off-policy correction
- Priority sampling
- High throughput with async environments
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import simulation_result


import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, Callable

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


def push_to_hub(repo_id: str, filename: str, filepath: str):
    """Push model to HuggingFace Hub."""
    try:
        api = HfApi()
        try:
            api.create_repo(repo_id, exist_ok=True)
        except Exception:
            pass
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=repo_id,
        )
        logger.info(f"Successfully pushed {filename} to {repo_id}")
    except Exception as e:
        logger.error(f"Error pushing to hub: {e}")


@dataclass
class PufferTrainConfig:
    total_timesteps: int = 100_000
    learning_rate: float = 1e-4
    batch_size: int = 2048
    minibatch_size: int = 512
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    vtrace: bool = True
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0
    num_envs: int = 16
    num_steps: int = 128


@dataclass
class PufferEnvConfig:
    max_num: int = 99


@dataclass
class PufferLibConfig:
    train: PufferTrainConfig = field(default_factory=PufferTrainConfig)
    env: PufferEnvConfig = field(default_factory=PufferEnvConfig)


class ArithmeticEnv(gym.Env):
    """
    Arithmetic environment compatible with PufferLib.

    Observation: [num1, num2] normalized to [0, 1]
    Action: predicted answer (discrete)
    Reward: 1.0 if correct, 0.0 otherwise
    """

    def __init__(self, max_num: int = 99) -> None:
        super().__init__()
        self.max_num = max_num
        self.max_answer = max_num * 2

        # Normalized observations for neural network
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_answer + 1)

        self.current_nums: Optional[np.ndarray] = None
        self.correct_answer: Optional[int] = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_nums = self.np_random.integers(1, self.max_num + 1, size=2)
        self.correct_answer = int(self.current_nums.sum())

        # Normalize to [0, 1]
        obs = self.current_nums.astype(np.float32) / self.max_num
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Verifiable binary reward
        reward = 1.0 if action == self.correct_answer else 0.0

        # Normalize observation
        obs = self.current_nums.astype(np.float32) / self.max_num

        return obs, reward, True, False, {
            "correct": reward == 1.0,
            "predicted": action,
            "expected": self.correct_answer,
        }


def make_env_creator(config: PufferLibConfig) -> Callable[[], gym.Env]:
    """Create environment factory for PufferLib."""
    def create_env() -> gym.Env:
        return ArithmeticEnv(max_num=config.env.max_num)
    return create_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PufferLib Math RL")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--track", action="store_true", help="Track experiments with WandB")
    parser.add_argument("--wandb-project-name", type=str, default="pufferlib_math", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HuggingFace Hub")
    parser.add_argument("--hf-repo-id", type=str, default="arvindcr4/pufferlib-math", help="HuggingFace Hub Repo ID")
    parser.add_argument("--exp-name", type=str, default="pufferlib_math", help="Experiment name")
    return parser.parse_known_args()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Arithmetic environment simulation only")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    print(simulation_result("pufferlib_math", {}, simulation=args.simulation, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
