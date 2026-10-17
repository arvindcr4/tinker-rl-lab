import numpy as np
import pytest

pytest.importorskip("gymnasium")
from platform_hybrid.experiments.implementations.arithmetic_env import ArithmeticEnv


def test_arithmetic_rewards_termination_and_seed():
    env = ArithmeticEnv(max_num=99)
    observation, _ = env.reset(seed=42)
    assert observation.shape == (2,)
    repeated, _ = env.reset(seed=42)
    np.testing.assert_array_equal(observation, repeated)
    correct = int(observation.sum())
    _, reward, terminated, truncated, info = env.step(correct)
    assert reward == 1.0
    assert info["correct"] is True
    assert terminated is True
    assert truncated is False
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert info["correct"] is False
    assert terminated is True
    assert truncated is False
