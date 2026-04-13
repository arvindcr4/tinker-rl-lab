"""
TRL PPO Baseline for GSM8K
===========================
Matched PPO baseline for GRPO comparison (addresses criticism #8).

This trains the SAME model (Qwen3-8B base) on GSM8K using PPO with
the same reward function as GRPO (binary correctness), so the comparison
isolates the RL algorithm:
  - Same model, same LoRA rank, same reward, same compute budget.
  - PPO uses a learned value head for advantage estimation.
  - GRPO uses group-relative normalization (no value head).

If GRPO matches PPO: GRPO is a valid simplification (no critic needed).
If PPO dominates: GRPO's group normalization loses important signal.
If GRPO dominates: GRPO's simplicity is actually an advantage.

Usage:
    python trl_ppo_gsm8k_baseline.py --model Qwen/Qwen3-8B --lora-rank 32
"""

import argparse
import re
import torch
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig


question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."


def load_gsm8k_prompts(num_samples=None, seed=42):
    """Load GSM8K prompts for PPO rollouts."""
    dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    def format_prompt(example):
        gold = example["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "query": example["question"] + question_suffix,
            "gold_answer": gold,
        }

    return dataset.map(format_prompt, remove_columns=dataset.column_names)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    # Fallback: last number
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return numbers[-1] if numbers else None


def compute_rewards(responses: List[str], gold_answers: List[str]) -> List[float]:
    """Binary reward: 1.0 if correct, 0.0 otherwise (matching GRPO)."""
    rewards = []
    for response, gold in zip(responses, gold_answers):
        # Strip think tags if present
        answer_text = response.split("</think>")[-1] if "</think>" in response else response
        predicted = extract_boxed_answer(answer_text)

        if predicted is not None:
            # Normalize and compare
            try:
                pred_val = float(predicted.replace(",", ""))
                gold_val = float(gold.replace(",", ""))
                rewards.append(1.0 if abs(pred_val - gold_val) < 0.01 else 0.0)
            except ValueError:
                rewards.append(1.0 if predicted.strip() == gold.strip() else 0.0)
        else:
            rewards.append(0.0)

    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="PPO baseline for GSM8K (matched to GRPO setup)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model (must match GRPO experiment)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (must match GRPO experiment)")
    parser.add_argument("--learning-rate", type=float, default=4e-5,
                        help="Learning rate (must match GRPO experiment)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of training prompts (None = full dataset)")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO epochs per batch")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/ppo_gsm8k_baseline/",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config — match GRPO experiment exactly
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # PPO needs a value head on top of the model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        peft_config=peft_config,
    )

    # PPO config — match GRPO compute budget
    ppo_config = PPOConfig(
        output_dir=args.output_dir,

        # Match GRPO batch structure
        batch_size=128,       # GRPO batch_size
        mini_batch_size=16,   # GRPO group_size
        ppo_epochs=args.ppo_epochs,

        # Match GRPO learning rate
        learning_rate=args.learning_rate,

        # PPO-specific
        gamma=1.0,            # No discounting (episodic)
        lam=0.95,             # GAE lambda
        cliprange=0.2,        # PPO clip
        vf_coef=0.1,          # Value function coefficient
        max_grad_norm=1.0,

        # Generation
        temperature=1.0,
        top_k=0,
        top_p=1.0,

        # Logging
        log_with="wandb",
        seed=args.seed,
    )

    # Load prompts
    print("Loading GSM8K prompts...")
    dataset = load_gsm8k_prompts(num_samples=args.num_samples, seed=args.seed)
    print(f"Loaded {len(dataset)} prompts")

    print("Initializing PPOTrainer...")
    trainer = PPOTrainer(
        model=model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("=" * 60)
    print(f"  PPO Baseline — GSM8K")
    print(f"  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Batch: 128, Mini-batch: 16")
    print(f"  PPO epochs: {args.ppo_epochs}")
    print("=" * 60)

    # Training loop
    generation_kwargs = {
        "max_new_tokens": 512,
        "temperature": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    for step, batch in enumerate(trainer.dataloader):
        query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze()
                         for q in batch["query"]]

        # Generate responses
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Compute rewards (same binary reward as GRPO)
        rewards = compute_rewards(responses, batch["gold_answer"])
        reward_tensors = [torch.tensor(r) for r in rewards]

        # PPO step
        stats = trainer.step(query_tensors, response_tensors, reward_tensors)

        mean_reward = sum(rewards) / len(rewards)
        print(f"Step {step}: mean_reward={mean_reward:.4f}")

        if step >= 50:  # Match GRPO 50-step budget
            break

    trainer.save_pretrained(args.output_dir + "/final")
    print(f"PPO baseline saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
