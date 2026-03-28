"""
SkyRL Integration for tinker-rl-lab

A modular full-stack RL library for LLMs with support for:
- Training algorithms: GRPO, PPO, REINFORCE
- Environments: GSM8K, Math, HumanEval, Tool Use
- Backends: vast.ai, Google Colab, Tinker API
- Hardware: Multi-GPU, multi-node via Ray

Based on NovaSky-AI/SkyRL framework.
"""

__version__ = "0.1.0"

from .config import SkyRLConfig
from .trainer import SkyRLTrainer

__all__ = [
    "SkyRLConfig",
    "SkyRLTrainer",
]
