"""GRPO training on GSM8K for small LLMs — AI Scientist template.

This experiment trains a small language model (0.5B-3B) using Group Relative
Policy Optimization (GRPO) on GSM8K math reasoning. It is designed to run
on a single GPU (T4/A10/A100) in under 2 hours.

Key variables an AI Scientist agent can modify:
- Model size and family (MODELS dict)
- Reward function (reward_fn)
- GRPO hyperparameters (group_size, learning_rate, etc.)
- Curriculum strategy (example selection/ordering)
- Prompt template (SYSTEM_PROMPT)
- Training schedule (num_steps, lr schedule)

Output: final_info.json with GSM8K training accuracy metrics.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ── Configuration ──────────────────────────────────────────────────────────

MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
}

DEFAULT_MODEL = "qwen-1.5b"
NUM_STEPS = 30
GROUP_SIZE = 4
LEARNING_RATE = 5e-6
BATCH_SIZE = 2
LORA_RANK = 16
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256
NUM_SEEDS = 3

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, "
    "then give your final numerical answer inside \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=list(MODELS.keys()))
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--group_size", type=int, default=GROUP_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK)
    parser.add_argument("--num_seeds", type=int, default=NUM_SEEDS)
    return parser.parse_args()


# ── Reward Function ────────────────────────────────────────────────────────

def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output."""
    # Try \boxed{answer} first
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip().replace(",", "")
    # Fall back to last number
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def reward_fn(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Binary exact-match reward for GSM8K.

    Called by GRPOTrainer with completions (model outputs) and any extra
    dataset columns as keyword args. The 'answer' column from our dataset
    is passed automatically.

    This is the core reward signal. Modifications here can dramatically
    change training dynamics. Ideas:
    - Partial credit for correct intermediate steps
    - Format bonus for using \\boxed{}
    - Length penalty for verbose responses
    - Curriculum-based reward scaling
    """
    rewards = []
    for completion, ans in zip(completions, answer):
        pred = extract_answer(completion)
        if pred is None:
            rewards.append(0.0)
            continue
        try:
            if abs(float(pred) - float(ans)) < 0.01:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except ValueError:
            rewards.append(1.0 if pred == ans else 0.0)
    return rewards


# ── Data Loading ───────────────────────────────────────────────────────────

def load_gsm8k():
    """Load and format GSM8K training examples."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not match:
            continue
        answer = match.group(1).replace(",", "").strip()
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{row['question']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        examples.append({"prompt": prompt, "answer": answer})
    return examples


# ── Training ───────────────────────────────────────────────────────────────

def train_single_seed(
    model_name: str,
    examples: list[dict],
    seed: int,
    args,
    out_dir: Path,
) -> dict:
    """Run one GRPO training seed and return metrics."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir = out_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    hf_model = MODELS[model_name]

    # Subsample training data for speed
    random.shuffle(examples)
    train_data = examples[:500]

    # Create dataset compatible with GRPOTrainer
    # Must have "prompt" column; extra columns (e.g. "answer") are passed
    # to reward_funcs as kwargs
    train_dataset = Dataset.from_list([
        {"prompt": ex["prompt"], "answer": ex["answer"]}
        for ex in train_data
    ])

    # LoRA configuration (passed to GRPOTrainer, not GRPOConfig)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Configure GRPO training
    config = GRPOConfig(
        output_dir=str(run_dir),
        num_train_epochs=1,
        max_steps=args.num_steps,
        per_device_train_batch_size=args.group_size,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=args.num_steps,
        remove_unused_columns=False,
        log_level="warning",
        report_to="none",
        seed=seed,
        # GRPO specific
        num_generations=args.group_size,
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
    )

    trainer = GRPOTrainer(
        model=hf_model,
        args=config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        peft_config=peft_config,
    )

    # Train and collect metrics
    start_time = time.time()
    train_result = trainer.train()
    duration = time.time() - start_time

    # Extract step-level rewards from training logs
    step_rewards = []
    for log_entry in trainer.state.log_history:
        if "reward" in log_entry:
            step_rewards.append(log_entry["reward"])
        elif "train/reward" in log_entry:
            step_rewards.append(log_entry["train/reward"])

    # Compute metrics
    last_10 = step_rewards[-10:] if len(step_rewards) >= 10 else step_rewards
    first_5 = step_rewards[:5] if len(step_rewards) >= 5 else step_rewards
    avg_last_10 = float(np.mean(last_10)) if last_10 else 0.0
    avg_first_5 = float(np.mean(first_5)) if first_5 else 0.0
    peak = float(max(step_rewards)) if step_rewards else 0.0

    metrics = {
        "last_10_accuracy": avg_last_10,
        "first_5_accuracy": avg_first_5,
        "peak_accuracy": peak,
        "training_loss": train_result.training_loss,
        "duration_seconds": duration,
        "num_steps_completed": len(step_rewards),
    }

    # Save per-seed results
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(run_dir / "reward_trace.npy", np.array(step_rewards))

    # Clean up GPU memory
    del trainer
    torch.cuda.empty_cache()

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== GRPO GSM8K Experiment ===")
    print(f"Model: {args.model} ({MODELS[args.model]})")
    print(f"Steps: {args.num_steps}, Group: {args.group_size}, LR: {args.lr}")
    print(f"LoRA rank: {args.lora_rank}, Seeds: {args.num_seeds}")
    print(f"Output: {out_dir}")

    examples = load_gsm8k()
    print(f"Loaded {len(examples)} GSM8K examples")

    all_metrics = []
    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx * 100
        print(f"\n--- Seed {seed} ({seed_idx + 1}/{args.num_seeds}) ---")
        metrics = train_single_seed(args.model, examples, seed, args, out_dir)
        all_metrics.append(metrics)
        print(f"  Last-10 acc: {metrics['last_10_accuracy']:.3f}")
        print(f"  Peak acc:    {metrics['peak_accuracy']:.3f}")
        print(f"  Duration:    {metrics['duration_seconds']:.0f}s")

    # Aggregate across seeds
    last_10_vals = [m["last_10_accuracy"] for m in all_metrics]
    peak_vals = [m["peak_accuracy"] for m in all_metrics]
    first_5_vals = [m["first_5_accuracy"] for m in all_metrics]
    loss_vals = [m["training_loss"] for m in all_metrics]
    duration_vals = [m["duration_seconds"] for m in all_metrics]

    # Write final_info.json in AI Scientist expected format
    final_info = {
        "gsm8k_training": {
            "means": {
                "last_10_accuracy_mean": float(np.mean(last_10_vals)),
                "peak_accuracy_mean": float(np.mean(peak_vals)),
                "first_5_accuracy_mean": float(np.mean(first_5_vals)),
                "training_loss_mean": float(np.mean(loss_vals)),
                "duration_seconds_mean": float(np.mean(duration_vals)),
            },
            "stderrs": {
                "last_10_accuracy_stderr": float(np.std(last_10_vals) / np.sqrt(len(last_10_vals))),
                "peak_accuracy_stderr": float(np.std(peak_vals) / np.sqrt(len(peak_vals))),
                "first_5_accuracy_stderr": float(np.std(first_5_vals) / np.sqrt(len(first_5_vals))),
                "training_loss_stderr": float(np.std(loss_vals) / np.sqrt(len(loss_vals))),
                "duration_seconds_stderr": float(np.std(duration_vals) / np.sqrt(len(duration_vals))),
            },
            "final_info_dict": {
                "last_10_accuracy": last_10_vals,
                "peak_accuracy": peak_vals,
                "first_5_accuracy": first_5_vals,
                "training_loss": loss_vals,
                "duration_seconds": duration_vals,
            },
        }
    }

    with open(out_dir / "final_info.json", "w") as f:
        json.dump(final_info, f, indent=2)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Last-10 accuracy: {np.mean(last_10_vals):.3f} +/- {np.std(last_10_vals):.3f}")
    print(f"Peak accuracy:    {np.mean(peak_vals):.3f} +/- {np.std(peak_vals):.3f}")
    print(f"Training loss:    {np.mean(loss_vals):.4f}")
    print(f"Results saved to: {out_dir / 'final_info.json'}")


if __name__ == "__main__":
    main()
