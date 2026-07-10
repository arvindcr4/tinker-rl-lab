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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.seed import set_global_seed, get_seed_from_args


class ArithmeticEnv(gym.Env):
    """
    Arithmetic environment compatible with PufferLib.

    Observation: [num1, num2] normalized to [0, 1]
    Action: predicted answer (discrete)
    Reward: 1.0 if correct, 0.0 otherwise
    """

    def __init__(self, max_num: int = 99):
        super().__init__()
        self.max_num = max_num
        self.max_answer = max_num * 2

        # Normalized observations for neural network
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_answer + 1)

        self.current_nums = None
        self.correct_answer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_nums = self.np_random.integers(1, self.max_num + 1, size=2)
        self.correct_answer = int(self.current_nums.sum())

        # Normalize to [0, 1]
        obs = self.current_nums.astype(np.float32) / self.max_num
        return obs, {}

    def step(self, action):
        # Verifiable binary reward
        reward = 1.0 if action == self.correct_answer else 0.0

        # Normalize observation
        obs = self.current_nums.astype(np.float32) / self.max_num

        return obs, reward, True, False, {
            "correct": reward == 1.0,
            "predicted": action,
            "expected": self.correct_answer,
        }


# PufferLib configuration
PUFFERLIB_CONFIG = {
    "train": {
        # Core training parameters
        "total_timesteps": 100_000,
        "learning_rate": 1e-4,  # Matching Tinker
        "batch_size": 2048,
        "minibatch_size": 512,
        "update_epochs": 4,

        # PPO parameters
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,

        # VTrace (PufferLib specialty)
        "vtrace": True,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,

        # Environment
        "num_envs": 16,
        "num_steps": 128,
    },
    "env": {
        "max_num": 99,
    }
}


def make_env_creator(config):
    """Create environment factory for PufferLib."""
    def create_env():
        return ArithmeticEnv(max_num=config["env"]["max_num"])
    return create_env


def main():
    """
    Main training function for PufferLib.

    Note: Full PufferLib integration requires pufferlib package.
    This shows the configuration and environment setup pattern.
    """
    seed = get_seed_from_args()
    set_global_seed(seed)
    print("=" * 60)
    print("PufferLib Math RL Configuration")
    print("=" * 60)

    config = PUFFERLIB_CONFIG

    print("\nTraining Config:")
    for key, value in config["train"].items():
        print(f"  {key}: {value}")

    print("\nEnvironment Config:")
    for key, value in config["env"].items():
        print(f"  {key}: {value}")

    # Create environment for testing
    env = ArithmeticEnv(max_num=config["env"]["max_num"])

    print("\n--- Testing Environment ---")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Test a few steps
    correct = 0
    for i in range(10):
        obs, _ = env.reset()
        # Random action
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        if info["correct"]:
            correct += 1
        print(f"  Problem {i+1}: {info['expected']}, Predicted: {info['predicted']}, Correct: {info['correct']}")

    print(f"\nRandom baseline: {correct}/10 = {correct*10}%")

    # Full PufferLib training would look like:
    print("\n--- Full PufferLib Training (requires pufferlib) ---")
    print("""
    from pufferlib import pufferl

    # Load and customize config
    args = pufferl.load_config('default')
    args.update(PUFFERLIB_CONFIG['train'])

    # Create trainer
    trainer = pufferl.PPO(
        env_creator=make_env_creator(PUFFERLIB_CONFIG),
        policy=pufferl.MLP(hidden_sizes=[64, 64]),
        config=args,
    )

    # Train
    trainer.train()
    """)


if __name__ == "__main__":
    main()
