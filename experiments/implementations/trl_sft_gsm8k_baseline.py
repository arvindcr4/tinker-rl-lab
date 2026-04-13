"""
TRL SFT Baseline for GSM8K
===========================
Matched baseline for GRPO comparison (addresses criticism #8).

This trains the SAME model (Qwen3-8B base) on GSM8K train solutions
using supervised fine-tuning with LoRA rank 32, so the comparison
with GRPO is apples-to-apples:
  - Same model, same LoRA rank, same dataset, same compute budget.
  - SFT trains on gold chain-of-thought solutions.
  - GRPO trains on model-generated solutions with binary reward.

If GRPO matches or exceeds SFT: GRPO adds value over supervised learning.
If SFT dominates: GRPO's gains are attributable to post-training in general.

Usage:
    python trl_sft_gsm8k_baseline.py --model Qwen/Qwen3-8B --lora-rank 32
    python trl_sft_gsm8k_baseline.py --model meta-llama/Llama-3.1-8B --lora-rank 32
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def load_gsm8k_sft_dataset(num_samples=None, seed=42):
    """
    Load GSM8K train set formatted for SFT.

    Each example becomes a (question, chain-of-thought + answer) pair,
    matching the format the GRPO environment uses.
    """
    dataset = load_dataset("gsm8k", "main", split="train").shuffle(seed=seed)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

    def format_for_sft(example):
        question = example["question"]
        # GSM8K answer format: "step-by-step reasoning #### numeric_answer"
        full_solution = example["answer"]
        numeric_answer = full_solution.split("####")[-1].strip().replace(",", "")

        # Format the solution to end with \boxed{} like GRPO evaluation expects
        reasoning = full_solution.split("####")[0].strip()
        formatted_solution = f"{reasoning}\n\\boxed{{{numeric_answer}}}"

        return {
            "text": (
                f"Question: {question}{question_suffix}\n\n"
                f"Answer: {formatted_solution}"
            ),
        }

    return dataset.map(format_for_sft, remove_columns=dataset.column_names)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baseline for GSM8K (matched to GRPO setup)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model (must match GRPO experiment)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (must match GRPO experiment)")
    parser.add_argument("--learning-rate", type=float, default=4e-5,
                        help="Learning rate (must match GRPO experiment)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of training examples (None = full dataset)")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max training steps (match GRPO step budget)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/sft_gsm8k_baseline/",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load GSM8K formatted for SFT
    print("Loading GSM8K dataset for SFT...")
    dataset = load_gsm8k_sft_dataset(num_samples=args.num_samples, seed=args.seed)
    print(f"Loaded {len(dataset)} examples")

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

    # SFT config — match GRPO compute budget
    sft_config = SFTConfig(
        output_dir=args.output_dir,

        # Match GRPO effective batch: 128 samples per step
        per_device_train_batch_size=8,
        gradient_accumulation_steps=16,  # 8 * 16 = 128

        # Match GRPO learning rate
        learning_rate=args.learning_rate,

        # Match GRPO step budget
        max_steps=args.max_steps,
        max_seq_length=2048,

        # Logging to match GRPO eval frequency
        logging_steps=1,
        save_steps=10,
        eval_steps=25,

        # Optimization
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.0,
        lr_scheduler_type="cosine",

        # Mixed precision
        bf16=True,
        fp16=False,

        # Misc
        seed=args.seed,
        report_to="wandb",
        run_name=f"sft-gsm8k-{args.model.split('/')[-1].lower()}",

        # Packing for efficiency
        packing=True,
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("=" * 60)
    print(f"  SFT Baseline — GSM8K")
    print(f"  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Steps: {args.max_steps}")
    print(f"  Effective batch: 128")
    print("=" * 60)

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    print(f"SFT baseline saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
