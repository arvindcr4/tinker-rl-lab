"""
Unified RL Training Launcher for tinker-rl-lab

Provides a single interface to run training with multiple frameworks:
- Tinker/SkyRL: Local Tinker API with SkyRL tx
- Tinker Atropos: Atropos environments with Tinker API
- verl: Volcano Engine Reinforcement Learning
- OpenRLHF: Ray + vLLM distributed RL
- TRL: HuggingFace Transformer RL

Usage:
    python -m unified.launcher --framework skyrl --model Qwen/Qwen2.5-1.5B-Instruct
    python -m unified.launcher --framework trl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm grpo
    python -m unified.launcher --framework verl --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np


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
        print("  (smoke-test scaffold: validates dispatch/config and emits")
        print("   simulated metrics — use per-framework runners for real training)")
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

        # Simulated training
        reward_history, loss_history = [], []
        for step in range(self.epochs):
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
            reward = 0.5 + 0.3 * np.random.random() + step * 0.02
            reward_history.append(reward)
            loss_history.append(loss)
            print(f"  Step {step}: loss={loss:.4f}, reward={reward:.4f}")

        return TrainingResult(
            framework="skyrl",
            model=self.model,
            algorithm=self.algorithm,
            final_step=self.epochs,
            reward_history=reward_history,
            loss_history=loss_history,
        )

    def _run_tinker(self) -> TrainingResult:
        """Run Tinker Atropos training."""
        print("\n[TINKER] Starting Tinker Atropos training...")

        # Import and use existing tinker_atropos trainer
        try:
            from platform_tinker.tinker_atropos.config import TinkerAtroposConfig
            from platform_tinker.tinker_atropos.trainer import TinkerAtroposTrainer

            # This would run actual training
            print("  Note: Run 'python platform_tinker/atropos/launch_training.py' for actual Tinker Atropos training")
        except ImportError:
            print("  Warning: tinker_atropos not available")

        # Simulated training
        reward_history, loss_history = [], []
        for step in range(self.epochs):
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
            reward = 0.5 + 0.3 * np.random.random() + step * 0.02
            reward_history.append(reward)
            loss_history.append(loss)

        return TrainingResult(
            framework="tinker",
            model=self.model,
            algorithm=self.algorithm,
            final_step=self.epochs,
            reward_history=reward_history,
            loss_history=loss_history,
        )

    def _run_verl(self) -> TrainingResult:
        """Run verl training."""
        print("\n[VERL] Starting verl training...")

        # Import and use verl trainer
        try:
            from verl.config import VERLConfig
            from verl.trainer import VERLTrainer

            config = VERLConfig(
                model=VERLModelConfig(model_name=self.model),
                algorithm=VERLAlgorithmConfig(algorithm=self.algorithm),
            )
            trainer = VERLTrainer(config)
            # asyncio.run(trainer.run())
        except ImportError:
            print("  Warning: verl not installed. Install with: pip install verl")

        # Simulated training
        reward_history, loss_history = [], []
        for step in range(self.epochs):
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
            reward = 0.5 + 0.3 * np.random.random() + step * 0.02
            reward_history.append(reward)
            loss_history.append(loss)
            print(f"  Step {step}: loss={loss:.4f}, reward={reward:.4f}")

        return TrainingResult(
            framework="verl",
            model=self.model,
            algorithm=self.algorithm,
            final_step=self.epochs,
            reward_history=reward_history,
            loss_history=loss_history,
        )

    def _run_openrlhf(self) -> TrainingResult:
        """Run OpenRLHF training."""
        print("\n[OPENRLHF] Starting OpenRLHF training...")

        try:
            from platform_modal.openrlhf.config import OpenRLHFConfig
            from platform_modal.openrlhf.trainer import OpenRLHFTrainer
        except ImportError:
            print("  Warning: OpenRLHF not installed. Install with: pip install openrlhf")

        reward_history, loss_history = [], []
        for step in range(self.epochs):
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
            reward = 0.5 + 0.3 * np.random.random() + step * 0.02
            reward_history.append(reward)
            loss_history.append(loss)
            print(f"  Step {step}: loss={loss:.4f}, reward={reward:.4f}")

        return TrainingResult(
            framework="openrlhf",
            model=self.model,
            algorithm=self.algorithm,
            final_step=self.epochs,
            reward_history=reward_history,
            loss_history=loss_history,
        )

    def _run_trl(self) -> TrainingResult:
        """Run TRL training."""
        print("\n[TRL] Starting HuggingFace TRL training...")

        try:
            from trl_integrations.config import TRLConfig
            from trl_integrations.trainer import TRLTrainer
        except ImportError:
            print("  Warning: TRL not installed. Install with: pip install trl")

        reward_history, loss_history = [], []
        for step in range(self.epochs):
            loss = 1.0 / (step + 1) + np.random.normal(0, 0.1)
            reward = 0.5 + 0.3 * np.random.random() + step * 0.02
            reward_history.append(reward)
            loss_history.append(loss)
            print(f"  Step {step}: loss={loss:.4f}, reward={reward:.4f}")

        return TrainingResult(
            framework="trl",
            model=self.model,
            algorithm=self.algorithm,
            final_step=self.epochs,
            reward_history=reward_history,
            loss_history=loss_history,
        )


# Import for type hints
from verl.config import VERLConfig, VERLModelConfig, VERLAlgorithmConfig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified RL Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with SkyRL
  python -m unified.launcher --framework skyrl --model Qwen/Qwen2.5-1.5B-Instruct

  # Run with TRL GRPO
  python -m platform_local.unified.launcher --framework trl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm grpo

  # Generate a real TRL script with QLoRA
  python -m platform_local.unified.launcher --framework trl --algorithm grpo \
    --peft-method lora --load-in-4bit --train-data train.json \
    --generate-script train_grpo.py

  # Run with verl PPO
  python -m unified.launcher --framework verl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm ppo

  # Run Tinker Atropos
  python -m unified.launcher --framework tinker --model meta-llama/Llama-3.1-8B-Instruct
        """
    )

    parser.add_argument(
        "--framework", "-f",
        type=str,
        choices=["skyrl", "tinker", "verl", "openrlhf", "trl"],
        default="skyrl",
        help="RL framework to use"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name or path"
    )

    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["grpo", "ppo", "reinforce", "dapo", "dpo"],
        default="grpo",
        help="RL algorithm"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate"
    )

    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank"
    )

    parser.add_argument(
        "--peft-method",
        type=str,
        choices=["lora", "prefix_tuning", "p_tuning", "prompt_tuning", "bitfit"],
        default="lora",
        help="PEFT method to use"
    )

    parser.add_argument(
        "--peft-num-virtual-tokens",
        type=int,
        default=32,
        help="Virtual token count for prefix, P-tuning, and prompt tuning"
    )

    parser.add_argument(
        "--no-peft",
        action="store_true",
        help="Generate a full fine-tuning configuration"
    )

    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model with NF4 quantization before adapter training"
    )
    quantization.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the model with 8-bit quantization before adapter training"
    )

    parser.add_argument(
        "--train-data",
        action="append",
        default=[],
        metavar="JSON_PATH",
        help="Training JSON file; repeat to provide multiple files"
    )

    parser.add_argument(
        "--generate-script",
        type=Path,
        metavar="PATH",
        help="Write a runnable TRL GRPO script instead of running the smoke scaffold"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )

    parser.add_argument(
        "--wandb-project",
        type=str,
        default="unified-rl",
        help="WandB project name"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without training"
    )

    args = parser.parse_args()

    # Create launcher
    launcher = UnifiedLauncher()
    launcher.framework = args.framework
    launcher.model = args.model
    launcher.algorithm = args.algorithm
    launcher.epochs = args.epochs
    launcher.use_peft = not args.no_peft
    launcher.peft_method = args.peft_method

    if args.dry_run:
        launcher.print_banner()
        print("\n[Dry run - exiting]")
        return

    if args.generate_script:
        if args.framework != "trl":
            parser.error("--generate-script requires --framework trl")
        if args.algorithm != "grpo":
            parser.error("--generate-script currently supports --algorithm grpo only")
        if not args.train_data:
            parser.error("--generate-script requires at least one --train-data path")
        if args.no_peft and (args.load_in_4bit or args.load_in_8bit):
            parser.error("quantized training cannot be combined with --no-peft")

        try:
            from platform_local.trl_integrations.config import TRLConfig
            from platform_local.trl_integrations.trainer import generate_trl_train_script
        except ImportError:
            from trl_integrations.config import TRLConfig
            from trl_integrations.trainer import generate_trl_train_script

        config = TRLConfig(
            model={
                "model_name": args.model,
                "use_peft": not args.no_peft,
                "peft_method": args.peft_method,
                "peft_num_virtual_tokens": args.peft_num_virtual_tokens,
                "lora_rank": args.lora_rank,
                "load_in_4bit": args.load_in_4bit,
                "load_in_8bit": args.load_in_8bit,
            },
            optimizer={"learning_rate": args.lr},
            algorithm={"algorithm": args.algorithm},
            data={
                "train_data": args.train_data,
                "train_batch_size": args.batch_size,
            },
            epochs=args.epochs,
            project_name=args.wandb_project,
        )
        generate_trl_train_script(config, args.generate_script)
        return

    # Run training
    launcher.run()


if __name__ == "__main__":
    main()
