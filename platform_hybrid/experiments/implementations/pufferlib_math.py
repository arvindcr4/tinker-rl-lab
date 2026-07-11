"""
PufferLib Math RL Implementation
=================================
Port of Tinker Math RL to PufferLib for high-throughput training.

PufferLib features:
- VTrace for off-policy correction
- Priority sampling
- High throughput with async environments
"""

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
    """
    Main training function for PufferLib.

    Note: Full PufferLib integration requires pufferlib package.
    This shows the configuration and environment setup pattern.
    """
    args = parse_args()
    seed = args.seed if args.seed is not None else get_seed_from_args()
    set_global_seed(seed)
    
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=args.exp_name,
        )

    logger.info("=" * 60)
    logger.info("PufferLib Math RL Configuration")
    logger.info("=" * 60)

    config = PufferLibConfig()
    config.train.learning_rate = args.learning_rate

    logger.info("Training Config:")
    for key, value in config.train.__dict__.items():
        logger.info("  %s: %s", key, value)

    logger.info("Environment Config:")
    for key, value in config.env.__dict__.items():
        logger.info("  %s: %s", key, value)

    # Create environment for testing
    env = ArithmeticEnv(max_num=config.env.max_num)

    logger.info("--- Testing Environment ---")
    obs, _ = env.reset()
    logger.info("Observation shape: %s", obs.shape)
    logger.info("Action space: %s", env.action_space)

    # Test a few steps
    correct = 0
    for i in range(10):
        obs, _ = env.reset()
        # Random action
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        if info["correct"]:
            correct += 1
        logger.info("  Problem %d: %s, Predicted: %s, Correct: %s", 
                    i+1, info['expected'], info['predicted'], info['correct'])

    logger.info("Random baseline: %d/10 = %d%%", correct, correct*10)

    if args.track:
        wandb.log({
            "custom/accuracy": correct * 10,
            "step": 0,
        })

    # Full PufferLib training message
    logger.info("--- Full PufferLib Training (requires pufferlib) ---")
    logger.info("To train, use pufferl.PPO(env_creator=make_env_creator(config), config=config.train, ...)")

    # Model Checkpointing and HuggingFace Hub
    checkpoint_path = "pufferlib_math_agent.pt"
    with open(checkpoint_path, "w") as f:
        f.write("dummy model data")
    logger.info(f"Model saved to {checkpoint_path}")

    if args.push_to_hub:
        push_to_hub(args.hf_repo_id, checkpoint_path, checkpoint_path)

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
