"""
CleanRL PPO Math RL Implementation
===================================
Port of Tinker Math RL to CleanRL-style single-file PPO.

CleanRL is research-friendly with transparent, single-file implementations.
This shows the core PPO loop with verifiable rewards.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


@dataclass
class Args:
    """Training arguments matching Tinker hyperparameters."""
    exp_name: str = "math_rl_cleanrl"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True

    # Environment
    max_num: int = 99
    total_timesteps: int = 100_000

    # PPO hyperparameters
    learning_rate: float = 1e-4  # Matching Tinker
    num_envs: int = 4
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01  # KL divergence for early stopping


class ArithmeticEnv(gym.Env):
    """Arithmetic environment with verifiable rewards."""

    def __init__(self, max_num: int = 99):
        super().__init__()
        self.max_num = max_num
        self.max_answer = max_num * 2

        self.observation_space = gym.spaces.Box(
            low=0, high=max_num, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.max_answer + 1)

        self.current_nums = None
        self.correct_answer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_nums = self.np_random.integers(1, self.max_num + 1, size=2)
        self.correct_answer = int(self.current_nums.sum())
        return self.current_nums.astype(np.float32), {}

    def step(self, action):
        reward = 1.0 if action == self.correct_answer else 0.0
        return self.current_nums.astype(np.float32), reward, True, False, {
            "correct": reward == 1.0
        }


def make_env(max_num, seed):
    def thunk():
        env = ArithmeticEnv(max_num=max_num)
        env.reset(seed=seed)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO Agent with separate actor and critic networks."""

    def __init__(self, envs):
        super().__init__()
        obs_shape = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n

        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        # Actor network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def main():
    args = Args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.max_num, args.seed + i) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Training loop
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

    episode_rewards = []
    episode_correct = []

    print("=" * 60)
    print("CleanRL PPO Math RL Training")
    print("Expected: accuracy ~50% -> ~100%")
    print("=" * 60)

    for update in range(1, num_updates + 1):
        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np).to(device)

            # Track metrics
            for i, info in enumerate(infos.get("final_info", []) if "final_info" in infos else []):
                if info and "correct" in info:
                    episode_correct.append(info["correct"])
                    episode_rewards.append(reward[i])

        # GAE computation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO update
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # KL divergence approximation
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping based on KL divergence
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Logging
        if update % 10 == 0 and episode_correct:
            accuracy = np.mean(episode_correct) * 100
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            sps = int(global_step / (time.time() - start_time))
            print(f"Step {global_step:>6} | Accuracy: {accuracy:5.1f}% | Reward: {avg_reward:.3f} | SPS: {sps}")
            episode_correct = []
            episode_rewards = []

    print("\nTraining complete!")
    torch.save(agent.state_dict(), "cleanrl_math_agent.pt")
    print("Model saved to cleanrl_math_agent.pt")


if __name__ == "__main__":
    main()
