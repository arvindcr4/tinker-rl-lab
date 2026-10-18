"""Candidate OpenRLHF 0.8.0 GRPO integration.

Real execution is the default and fails closed. Simulation is an explicit,
separate fixture schema. This candidate protocol has not been validated in a
GPU image and does not establish objective parity with another framework.
The independent platform_hybrid Modal 0.8.4 launcher is not this integration.
"""

__version__ = "0.9.0"

from .trainer import OpenRLHFTrainer, run_openrlhf_training, run
from .config import OpenRLHFConfig

__all__ = ["OpenRLHFTrainer", "OpenRLHFConfig", "run_openrlhf_training", "run"]
