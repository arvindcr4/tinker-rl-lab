"""Side-effect-free arithmetic reward contract."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
        self.observation_space = spaces.Box(low=0, high=max_num, shape=(2,), dtype=np.float32)

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
