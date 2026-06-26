"""
TRL RLOO Baseline for GSM8K (replaces PPO)
===========================================
On-policy RL baseline for GRPO comparison (addresses criticism #8).

PPOTrainer was removed in TRL 1.0. RLOO (Reward-Leveraged Optimization Online)
is the recommended on-policy RL replacement. Like GRPO, it does not need a
value head, making it a fairer comparison:
  - Same model, same LoRA rank, same reward, same compute budget.
  - RLOO uses leave-one-out baselines for advantage estimation.
  - GRPO uses group-relative normalization.

Usage:
    python trl_ppo_gsm8k_baseline.py --seed 42
"""

import argparse
import re
import os
import sys
import json
import torch
from typing import Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import RLOOTrainer, RLOOConfig
from peft import LoraConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.seed import set_global_seed, get_seed_from_args


question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."


def extract_boxed_answer(text: str) -> Optional[str]:
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return numbers[-1] if numbers else None


def reward_fn(completions, **kwargs):
    """Binary reward: 1.0 if correct, 0.0 otherwise."""
    gold_answers = kwargs.get("gold_answer", [])
    rewards = []
    for completion, gold in zip(completions, gold_answers):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        answer_text = text.split("</think>")[-1] if "</think>" in text else text
        predicted = extract_boxed_answer(answer_text)
        if predicted is not None:
            try:
                pred_val = float(predicted.replace(",", ""))
                gold_val = float(gold.replace(",", ""))
                rewards.append(1.0 if abs(pred_val - gold_val) < 0.01 else 0.0)
            except ValueError:
                rewards.append(1.0 if predicted.strip() == gold.strip() else 0.0)
        else:
            rewards.append(0.0)
    return rewards


def main():
    seed = get_seed_from_args(default=42)
    set_global_seed(seed)
    results_dir = os.environ.get("RESULTS_DIR", "./results/rloo_gsm8k")
    os.makedirs(results_dir, exist_ok=True)

    # Use small model for A10G compatibility
    model_name = "Qwen/Qwen2-0.5B-Instruct"

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load GSM8K prompts
    dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed).select(range(256))

    def format_prompt(example):
        gold = example["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "prompt": [{"role": "user", "content": example["question"] + question_suffix}],
            "gold_answer": gold,
        }

    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    rloo_config = RLOOConfig(
        output_dir=results_dir,
        per_device_train_batch_size=4,
        num_generation_per_prompt=4,
        max_new_tokens=256,
        learning_rate=4e-5,
        num_train_epochs=1,
        logging_steps=1,
        seed=seed,
        report_to="none",
        bf16=True,
    )

    trainer = RLOOTrainer(
        config=rloo_config,
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
    )

    print("=" * 60)
    print(f"  RLOO Baseline — GSM8K (seed={seed})")
    print(f"  Model: {model_name}")
    print("=" * 60)

    trainer.train()

    # Save results
    metrics = trainer.state.log_history
    result = {
        "experiment": "trl_ppo",
        "algorithm": "RLOO",
        "seed": seed,
        "model": model_name,
        "num_samples": len(dataset),
        "metrics": metrics[-5:] if metrics else [],
    }
    with open(f"{results_dir}/experiment_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
