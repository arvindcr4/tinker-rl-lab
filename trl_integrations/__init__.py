"""
TRL (Transformer Reinforcement Learning) Integration for tinker-rl-lab

TRL: HuggingFace's full stack library for training transformer language models
with RL methods like SFT, GRPO, DPO, Reward Modeling, etc.

GitHub: https://github.com/huggingface/trl

Features:
- GRPO, PPO, DPO, RLWF training
- Easy integration with HuggingFace models
- Single GPU to multi-GPU support
- Works with transformers and peft
"""

__version__ = "0.16.0"

from .trainer import TRLTrainer
from .config import TRLConfig

__all__ = ["TRLTrainer", "TRLConfig"]
