"""
Tianshou PPO Math RL Implementation
====================================
Port of Tinker Math RL to Tianshou.

Tianshou is a modular RL library with clean PyTorch implementation.
Features: vectorized environments, flexible policy networks.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import torch
from torch import nn

import gymnasium as gym
from gymnasium import spaces

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from utils.seed import set_global_seed, get_seed_from_args


class ArithmeticEnv(gym.Env):
    """Arithmetic environment for Tianshou."""

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


def make_env(max_num: int = 99):
    """Environment factory for Tianshou."""
    def _make():
        return ArithmeticEnv(max_num=max_num)
    return _make


def main():
    # Configuration (matching Tinker)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = get_seed_from_args()
    set_global_seed(seed)
    max_num = 99
    num_envs = 10
    hidden_sizes = [64, 64]
    lr = 1e-4  # Matching Tinker

    print("=" * 60)
    print("Tianshou PPO Math RL Training")
    print(f"Device: {device}")
    print("=" * 60)

    # Create environments
    train_envs = DummyVectorEnv([make_env(max_num) for _ in range(num_envs)])
    test_envs = DummyVectorEnv([make_env(max_num) for _ in range(num_envs)])

    # Get dimensions
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n

    # Create networks
    net = Net(
        state_shape=state_shape,
        hidden_sizes=hidden_sizes,
        device=device,
    )

    actor = Actor(
        preprocess_net=net,
        action_shape=action_shape,
        device=device,
    ).to(device)

    critic = Critic(
        preprocess_net=Net(
            state_shape=state_shape,
            hidden_sizes=hidden_sizes,
            device=device,
        ),
        device=device,
    ).to(device)

    # Optimizers
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr,
    )

    # PPO Policy (matching Tinker hyperparameters)
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=torch.distributions.Categorical,
        action_space=train_envs.action_space[0],
        # PPO specific
        eps_clip=0.2,  # Matching Tinker clip range
        vf_coef=0.5,
        ent_coef=0.01,
        gae_lambda=0.95,
        discount_factor=0.99,
        # Training
        max_grad_norm=0.5,
        deterministic_eval=True,
    )

    # Collectors
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(total_size=20000, buffer_num=num_envs),
    )
    test_collector = Collector(policy, test_envs)

    # Training
    print("\nStarting training...")

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.95  # Stop when accuracy ~95%

    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=100,
        step_per_epoch=1000,
        repeat_per_collect=10,
        episode_per_test=100,
        batch_size=64,
        step_per_collect=200,
        stop_fn=stop_fn,
        verbose=True,
        show_progress=True,
    ).run()

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best reward: {result.best_reward:.3f}")
    print(f"Best reward std: {result.best_reward_std:.3f}")
    print("=" * 60)

    # Save policy
    torch.save(policy.state_dict(), "tianshou_ppo_math.pt")
    print("Model saved to tianshou_ppo_math.pt")


if __name__ == "__main__":
    main()
