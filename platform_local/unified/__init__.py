"""
Unified RL Training Launcher for tinker-rl-lab

Provides a single interface to run training with multiple frameworks:
- Tinker/SkyRL: Local Tinker API with SkyRL tx
- Tinker Atropos: Atropos environments with Tinker API
- verl: Volcano Engine Reinforcement Learning
- OpenRLHF: Ray + vLLM distributed RL
- TRL: HuggingFace Transformer RL
"""

from .launcher import UnifiedLauncher, TrainingResult

__all__ = ["UnifiedLauncher", "TrainingResult"]
