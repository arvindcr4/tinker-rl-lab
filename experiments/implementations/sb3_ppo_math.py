"""
Stable Baselines3 PPO Math RL Implementation
=============================================
Port of Tinker Math RL to Stable Baselines3.

This uses a custom Gymnasium environment with verifiable rewards.
Note: SB3 is designed for RL agents, not LLMs. This implementation
shows the pattern for custom reward wrappers and PPO training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class ArithmeticEnv(gym.Env):
    """
    Arithmetic environment with verifiable rewards.

    Observation: [num1, num2] (the two numbers to add)
    Action: predicted answer (discrete, 0-199)
    Reward: 1.0 if correct, 0.0 otherwise
    """

    def __init__(self, max_num: int = 99):
        super().__init__()
        self.max_num = max_num
        self.max_answer = max_num * 2  # Maximum possible sum

        # Observation: two numbers to add
        self.observation_space = spaces.Box(
            low=0, high=max_num, shape=(2,), dtype=np.float32
        )

        # Action: predicted answer
        self.action_space = spaces.Discrete(self.max_answer + 1)

        self.current_nums = None
        self.correct_answer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate new problem
        self.current_nums = self.np_random.integers(1, self.max_num + 1, size=2)
        self.correct_answer = int(self.current_nums.sum())

        return self.current_nums.astype(np.float32), {}

    def step(self, action):
        # Verifiable binary reward (matching Tinker)
        reward = 1.0 if action == self.correct_answer else 0.0

        # Episode ends after one prediction
        terminated = True
        truncated = False

        info = {
            "correct": reward == 1.0,
            "predicted": action,
            "expected": self.correct_answer,
        }

        return self.current_nums.astype(np.float32), reward, terminated, truncated, info


class VerifiableRewardWrapper(gym.RewardWrapper):
    """
    Wrapper that applies verifiable reward function.

    This pattern from SB3 atari_wrappers shows how to modify rewards.
    """

    def __init__(self, env: gym.Env, format_penalty: float = -0.1):
        super().__init__(env)
        self.format_penalty = format_penalty

    def reward(self, reward):
        # In this simple env, reward is already binary
        # For LLM envs, add format checking here
        return reward


class MetricsCallback(BaseCallback):
    """Callback to track accuracy and reward metrics (like Tinker output)."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_correct = []

    def _on_step(self):
        # Check for episode completion
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    infos = self.locals.get("infos", [{}])
                    if i < len(infos) and "correct" in infos[i]:
                        self.episode_correct.append(infos[i]["correct"])

        return True

    def _on_rollout_end(self):
        if self.episode_correct:
            accuracy = np.mean(self.episode_correct)
            self.logger.record("custom/accuracy", accuracy)
            self.episode_correct = []


def main():
    print("Creating arithmetic environment...")

    # Create vectorized environment
    def make_env():
        env = ArithmeticEnv(max_num=99)
        env = VerifiableRewardWrapper(env)
        return env

    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel envs

    # Optional: normalize rewards (can help stability)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # PPO Configuration (matching Tinker hyperparameters where applicable)
    print("Initializing PPO...")
    model = PPO(
        "MlpPolicy",
        env,

        # Learning rate (matching Tinker)
        learning_rate=1e-4,

        # PPO-specific
        n_steps=2048,           # Steps per rollout
        batch_size=64,          # Minibatch size
        n_epochs=10,            # Epochs per update

        # GAE parameters
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE lambda

        # Clipping
        clip_range=0.2,         # PPO clip range

        # Loss coefficients
        ent_coef=0.01,          # Entropy coefficient
        vf_coef=0.5,            # Value function coefficient

        # Optimization
        max_grad_norm=0.5,      # Gradient clipping

        # KL divergence target (for early stopping)
        target_kl=0.01,

        # Logging
        verbose=1,
        tensorboard_log="./sb3_math_logs/",
    )

    # Train with metrics callback
    print("Starting PPO training...")
    print("=" * 50)
    print("Expected: accuracy increases from ~50% to ~100%")
    print("=" * 50)

    metrics_callback = MetricsCallback()

    model.learn(
        total_timesteps=100_000,
        callback=metrics_callback,
        progress_bar=True,
    )

    # Save model
    model.save("./sb3_math_ppo_final")
    print("Training complete! Model saved to ./sb3_math_ppo_final")

    # Evaluate
    print("\nEvaluating trained model...")
    test_env = ArithmeticEnv(max_num=99)
    correct = 0
    total = 100

    for _ in range(total):
        obs, _ = test_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, info = test_env.step(action)
        if info["correct"]:
            correct += 1

    print(f"Final accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()
