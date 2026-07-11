import argparse
from pathlib import Path

from platform_local.unified.launcher import UnifiedLauncher

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified RL Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with SkyRL
  python -m platform_local.unified --framework skyrl --model Qwen/Qwen2.5-1.5B-Instruct

  # Run with TRL GRPO
  python -m platform_local.unified --framework trl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm grpo

  # Generate a real TRL script with QLoRA
  python -m platform_local.unified --framework trl --algorithm grpo \\
    --peft-method lora --load-in-4bit --train-data train.json \\
    --generate-script train_grpo.py

  # Run with verl PPO
  python -m platform_local.unified --framework verl --model Qwen/Qwen2.5-1.5B-Instruct --algorithm ppo

  # Run Tinker Atropos
  python -m platform_local.unified --framework tinker --model meta-llama/Llama-3.1-8B-Instruct
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
        choices=["grpo", "ppo", "reinforce", "dapo", "dpo", "idpo"],
        default="grpo",
        help="RL algorithm"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Number of training epochs (overridden by --total-token-budget)"
    )

    parser.add_argument(
        "--total-token-budget",
        type=int,
        default=None,
        help="Target total sampled/training tokens (P1 causal ablation). Overrides epochs."
    )

    parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=2048,
        help="Expected average tokens per sample (used with --total-token-budget)"
    )

    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (e.g., for ZeRO-3 CPU offloading)"
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

    max_steps = -1
    if args.total_token_budget:
        # P1: SFT warm-up ablation: Match total sampled/training tokens
        total_samples = args.total_token_budget / args.tokens_per_sample
        # Assuming gradient_accumulation_steps=4 (default in config) for rough step match
        effective_batch_size = args.batch_size * 4
        max_steps = max(1, int(total_samples / effective_batch_size))
        print(f"[P1 Ablation] Matching token budget: {args.total_token_budget} tokens")
        print(f" -> Computed max_steps: {max_steps} (assuming batch size {args.batch_size}, 4 grad accum, ~{args.tokens_per_sample} tokens/sample)")

    if args.dry_run:
        launcher.print_banner()
        print("\n[Dry run - exiting]")
        return

    if args.generate_script:
        if args.framework != "trl":
            parser.error("--generate-script requires --framework trl")
        if args.algorithm not in ("grpo", "idpo"):
            parser.error("--generate-script currently supports --algorithm grpo or idpo")
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
            max_steps=max_steps,
            project_name=args.wandb_project,
            deepspeed=args.deepspeed,
        )
        generate_trl_train_script(config, args.generate_script)
        return

    # Run training
    launcher.run()


if __name__ == "__main__":
    main()
