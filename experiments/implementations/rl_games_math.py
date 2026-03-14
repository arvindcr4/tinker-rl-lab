"""
rl_games (NVIDIA) PPO Math RL Implementation
=============================================
Port of Tinker Math RL to NVIDIA's rl_games library.

rl_games is designed for high-performance GPU training.
Used in Isaac Gym for robotics simulation.
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class ArithmeticEnv(gym.Env):
    """Arithmetic environment for rl_games."""

    def __init__(self, max_num: int = 99):
        super().__init__()
        self.max_num = max_num
        self.max_answer = max_num * 2

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_answer + 1)

        self.current_nums = None
        self.correct_answer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_nums = self.np_random.integers(1, self.max_num + 1, size=2)
        self.correct_answer = int(self.current_nums.sum())
        obs = self.current_nums.astype(np.float32) / self.max_num
        return obs, {}

    def step(self, action):
        reward = 1.0 if action == self.correct_answer else 0.0
        obs = self.current_nums.astype(np.float32) / self.max_num
        return obs, reward, True, False, {"correct": reward == 1.0}


# rl_games configuration (YAML-style dict)
RL_GAMES_CONFIG = {
    "params": {
        "seed": 42,
        "algo": {
            "name": "a2c_discrete"  # PPO for discrete actions
        },
        "model": {
            "name": "discrete_a2c"
        },
        "network": {
            "name": "actor_critic",
            "separate": False,
            "space": {
                "discrete": {}
            },
            "mlp": {
                "units": [64, 64],
                "activation": "elu",
                "initializer": {
                    "name": "default"
                }
            }
        },
        "config": {
            "name": "arithmetic_ppo",
            "env_name": "arithmetic",
            "score_to_win": 0.95,
            "normalize_input": True,
            "normalize_value": True,

            # Training params (matching Tinker)
            "num_actors": 16,
            "horizon_length": 128,
            "minibatch_size": 512,
            "mini_epochs": 4,

            # PPO params
            "gamma": 0.99,
            "tau": 0.95,  # GAE lambda
            "e_clip": 0.2,  # Clip range (matching Tinker)
            "entropy_coef": 0.01,
            "critic_coef": 0.5,

            # Learning rate (matching Tinker)
            "learning_rate": 1e-4,
            "lr_schedule": "constant",

            # Gradient clipping
            "grad_norm": 0.5,
            "max_epochs": 1000,

            # Device
            "device": "cuda:0",
            "device_name": "cuda:0",
        }
    }
}


def main():
    """
    Main function for rl_games training.

    Note: Full rl_games integration requires rl_games package.
    This shows the configuration pattern.
    """
    print("=" * 60)
    print("rl_games (NVIDIA) PPO Math RL Configuration")
    print("=" * 60)

    config = RL_GAMES_CONFIG

    print("\nPPO Configuration:")
    ppo_config = config["params"]["config"]
    for key in ["learning_rate", "e_clip", "gamma", "tau", "mini_epochs"]:
        print(f"  {key}: {ppo_config[key]}")

    print("\nNetwork Configuration:")
    net_config = config["params"]["network"]["mlp"]
    print(f"  units: {net_config['units']}")
    print(f"  activation: {net_config['activation']}")

    # Test environment
    print("\n--- Testing Environment ---")
    env = ArithmeticEnv(max_num=99)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Random baseline
    correct = 0
    for _ in range(100):
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        if info["correct"]:
            correct += 1

    print(f"\nRandom baseline: {correct}%")

    # Full rl_games training would look like:
    print("\n--- Full rl_games Training (requires rl_games) ---")
    print("""
    from rl_games.common import env_configurations
    from rl_games.torch_runner import Runner

    # Register environment
    env_configurations.register(
        'arithmetic',
        {'env_creator': lambda **kwargs: ArithmeticEnv(**kwargs)}
    )

    # Create runner
    runner = Runner()
    runner.load(RL_GAMES_CONFIG)

    # Train
    runner.run({
        'train': True,
        'play': False,
    })
    """)


if __name__ == "__main__":
    main()
