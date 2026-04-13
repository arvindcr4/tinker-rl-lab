"""
TRL GRPO GSM8K Word Problems
=============================
Port of Tinker Math RL (GSM8K) to TRL.

GSM8K is a dataset of grade school math word problems.
Unlike arithmetic, this requires:
- Chain-of-thought reasoning
- Answer extraction from text
- More complex verification
"""

import re
import sys
import os
import torch
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

# Add project root to path for utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.seed import set_global_seed, get_seed_from_args, log_experiment_metadata


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """
    Extract numeric answer from GSM8K-style response.

    GSM8K answers are typically formatted as:
    "#### 42" or just a number at the end.
    """
    # Look for #### pattern first
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if match:
        return float(match.group(1))

    # Fall back to last number in text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return float(numbers[-1])

    return None


def verify_gsm8k_answer(predicted: Optional[float], expected: float, tolerance: float = 0.01) -> bool:
    """Verify if predicted answer matches expected (with tolerance)."""
    if predicted is None:
        return False
    return abs(predicted - expected) < tolerance


def gsm8k_reward_function(
    completions: List[str],
    prompts: List[str],
    answers: List[float],
) -> List[float]:
    """
    Verifiable reward function for GSM8K.

    Reward structure:
    - reward=1.0: Correct final answer
    - reward=0.5: Has chain-of-thought but wrong answer
    - reward=0.0: No reasoning, wrong answer
    - reward=-0.1: Invalid format
    """
    rewards = []

    for completion, expected in zip(completions, answers):
        predicted = extract_gsm8k_answer(completion)

        # Check for chain-of-thought (has some reasoning)
        has_reasoning = len(completion.split()) > 10

        if predicted is None:
            # Invalid format
            rewards.append(-0.1)
        elif verify_gsm8k_answer(predicted, expected):
            # Correct answer
            rewards.append(1.0)
        elif has_reasoning:
            # Wrong answer but showed work
            rewards.append(0.5)
        else:
            # Wrong answer, no reasoning
            rewards.append(0.0)

    return rewards


def load_gsm8k_dataset(split: str = "train", num_samples: Optional[int] = None):
    """Load GSM8K dataset with proper formatting."""
    dataset = load_dataset("gsm8k", "main", split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    def format_example(example):
        # Extract numeric answer from solution
        answer_str = example["answer"].split("####")[-1].strip()
        answer = float(answer_str.replace(",", ""))

        return {
            "prompt": f"Solve this math problem step by step:\n\n{example['question']}\n\nShow your work and end with #### followed by the numeric answer.",
            "answer": answer,
        }

    return dataset.map(format_example)


def main():
    # Model configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    dataset = load_gsm8k_dataset(split="train", num_samples=500)
    print(f"Loaded {len(dataset)} problems")

    # GRPO Configuration for GSM8K
    grpo_config = GRPOConfig(
        output_dir="./grpo_gsm8k_output",

        # Batch settings
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,

        # GRPO-specific
        num_generations=4,
        beta=0.1,

        # Learning rate
        learning_rate=5e-5,

        # LoRA settings
        use_peft=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,

        # Generation - longer for chain-of-thought
        max_new_tokens=256,
        temperature=0.7,

        # Training
        num_train_epochs=2,
        logging_steps=1,
        save_steps=50,

        # Optimization
        max_grad_norm=1.0,
        warmup_ratio=0.1,
    )

    # Reward function wrapper
    def reward_fn(completions, prompts):
        batch_answers = [dataset[i]["answer"] for i in range(len(prompts))]
        return gsm8k_reward_function(completions, prompts, batch_answers)

    print("Initializing GRPOTrainer for GSM8K...")
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    print("Starting GSM8K GRPO training...")
    print("=" * 50)
    print("GSM8K is harder than arithmetic - expect slower convergence")
    print("=" * 50)

    trainer.train()

    # Save final model
    trainer.save_model("./grpo_gsm8k_final")
    print("Training complete! Model saved to ./grpo_gsm8k_final")


if __name__ == "__main__":
    main()
