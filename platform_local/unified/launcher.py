import sys
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class TrainingResult:
    """Result of a training run."""
    framework: str
    model: str
    algorithm: str
    final_step: int
    reward_history: List[float]
    loss_history: List[float]
    total_time: float = 0.0


class UnifiedLauncher:
    """
    Unified launcher for all RL frameworks.

    Supports:
    - skyrl: SkyRL tx (Tinker API implementation)
    - tinker: Tinker Atropos (Atropos + Tinker API)
    - verl: Volcano Engine RL
    - openrlhf: OpenRLHF
    - trl: HuggingFace TRL
    """

    FRAMEWORKS = {
        "skyrl": "SkyRL tx (Local Tinker API)",
        "tinker": "Tinker Atropos",
        "verl": "Volcano Engine RL",
        "openrlhf": "OpenRLHF",
        "trl": "HuggingFace TRL",
    }

    ALGORITHMS = {
        "grpo": "Group Relative Policy Optimization",
        "ppo": "Proximal Policy Optimization",
        "reinforce": "REINFORCE",
        "dapo": "DAPO",
        "dpo": "Direct Preference Optimization",
    }

    def __init__(self):
        self.framework = None
        self.model = None
        self.algorithm = "grpo"
        self.epochs = 20
        self.config = None
        self.use_peft = True
        self.peft_method = "lora"

    def print_banner(self):
        """Print startup banner."""
        print("\n" + "=" * 60)
        print("  Unified RL Training Launcher")
        print("=" * 60)
        print(f"\nAvailable Frameworks:")
        for key, desc in self.FRAMEWORKS.items():
            marker = "→" if key == self.framework else " "
            print(f"  {marker} {key:12s} - {desc}")
        print(f"\nAvailable Algorithms:")
        for key, desc in self.ALGORITHMS.items():
            marker = "→" if key == self.algorithm else " "
            print(f"  {marker} {key:12s} - {desc}")
        print(f"\nSelected:")
        print(f"  Framework: {self.framework}")
        print(f"  Model: {self.model}")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Epochs: {self.epochs}")
        if self.framework == "trl":
            tuning = self.peft_method if self.use_peft else "full fine-tuning"
            print(f"  Tuning: {tuning}")
        print("=" * 60 + "\n")

    def run(self):
        """Run the unified launcher."""
        self.print_banner()

        start_time = time.time()
        result = None

        if self.framework == "skyrl":
            result = self._run_skyrl()
        elif self.framework == "tinker":
            result = self._run_tinker()
        elif self.framework == "verl":
            result = self._run_verl()
        elif self.framework == "openrlhf":
            result = self._run_openrlhf()
        elif self.framework == "trl":
            result = self._run_trl()
        else:
            print(f"Unknown framework: {self.framework}")
            print(f"Available: {', '.join(self.FRAMEWORKS.keys())}")
            sys.exit(1)

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("  Training Complete!")
        print("=" * 60)
        print(f"\n  Framework: {result.framework}")
        print(f"  Model: {result.model}")
        print(f"  Final Step: {result.final_step}")
        if result.reward_history:
            print(f"  Final Reward: {result.reward_history[-1]:.4f}")
            print(f"  Peak Reward: {max(result.reward_history):.4f}")
        print(f"  Total Time: {total_time:.1f}s")
        print("=" * 60 + "\n")

    def _run_skyrl(self) -> TrainingResult:
        """Run SkyRL tx training."""
        print("\n[SKYRL] Starting SkyRL tx training...")
        raise NotImplementedError("SkyRL tx training runner is not yet implemented.")

    def _run_tinker(self) -> TrainingResult:
        """Run Tinker Atropos training."""
        print("\n[TINKER] Starting Tinker Atropos training...")
        raise NotImplementedError("Tinker Atropos training runner is not yet implemented.")

    def _run_verl(self) -> TrainingResult:
        """Run verl training."""
        print("\n[VERL] Starting verl training...")

        try:
            from verl.config import VERLConfig, VERLModelConfig, VERLAlgorithmConfig
            from verl.trainer import VERLTrainer
            import asyncio

            config = VERLConfig(
                model=VERLModelConfig(model_name=self.model),
                algorithm=VERLAlgorithmConfig(algorithm=self.algorithm),
            )
            trainer = VERLTrainer(config)
            asyncio.run(trainer.run())
            
            return TrainingResult(
                framework="verl",
                model=self.model,
                algorithm=self.algorithm,
                final_step=self.epochs,
                reward_history=[],
                loss_history=[],
            )
        except ImportError:
            raise NotImplementedError("verl is not installed. Install with: pip install verl")

    def _run_openrlhf(self) -> TrainingResult:
        """Run OpenRLHF training."""
        print("\n[OPENRLHF] Starting OpenRLHF training...")
        raise NotImplementedError("OpenRLHF training runner is not yet implemented.")

    def _run_trl(self) -> TrainingResult:
        """Run TRL training."""
        print("\n[TRL] Starting HuggingFace TRL training...")
        raise NotImplementedError("TRL training runner is not yet implemented. Please use --generate-script to create a script instead.")
