"""
OpenRLHF Integration for tinker-rl-lab

OpenRLHF: An Easy-to-use, Scalable and High-performance Agentic RL Framework
GitHub: https://github.com/OpenRLHF/OpenRLHF

Features:
- Ray + vLLM distributed architecture
- PPO, DAPO, REINFORCE++ algorithms
- Async RL training
- Multi-GPU and multi-node support

TODO: Address structural and methodological limitations identified in adversarial review:
- Fix ZVF metric fragility: ZVF breaks down in format-gated tasks and serves mostly as a symptom rather than root cause.
- Address the "Early-Training Snapshot" Problem: Expand experiments beyond 30-50 gradient steps to observe true RL convergence.
- Resolve the Closed-Source Confound: Deconfound Tinker API's managed defaults from nominal RL algorithms.
- Prove Generalization: Address the lack of statistically significant generalization on held-out test sets (e.g., GSM8K and HumanEval) to prevent training-set memorization.
- Avoid Single-Seed Extrapolations: Run multi-seed experiments to account for high variance in RL training dynamics.
"""

__version__ = "0.9.0"

from .trainer import OpenRLHFTrainer, run_openrlhf_training, run
from .config import OpenRLHFConfig

__all__ = ["OpenRLHFTrainer", "OpenRLHFConfig", "run_openrlhf_training", "run"]
