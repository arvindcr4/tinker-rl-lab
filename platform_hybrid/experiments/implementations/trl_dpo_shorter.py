"""
TRL DPO Preference Learning for Shorter Responses
==================================================
Port of Tinker Preference Shorter experiment to HuggingFace TRL.

Original Tinker Method:
- Generate group_size responses per prompt
- Pairwise comparison: shorter response wins
- Reward based on win_minus_loss score

This implementation uses DPOTrainer with length-penalized preferences.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List

import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import DPOTrainer, DPOConfig

from utils.seed import set_global_seed, log_experiment_metadata

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


@dataclass
class DPOScriptArguments:
    """Arguments for the DPO Shorter Responses Experiment."""
    model_name: str = field(default="Qwen/Qwen2-0.5B-Instruct", metadata={"help": "The model to train."})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    num_prompts: int = field(default=100, metadata={"help": "Number of prompts."})
    num_generations: int = field(default=4, metadata={"help": "Number of generations per prompt."})
    max_new_tokens: int = field(default=100, metadata={"help": "Max new tokens."})
    output_dir: str = field(default="./dpo_shorter_output", metadata={"help": "Output directory."})
    
    # DPO Config
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per device."})
    gradient_accumulation_steps: int = field(default=4, metadata={"help": "Gradient accumulation steps."})
    learning_rate: float = field(default=5e-7, metadata={"help": "Learning rate."})
    max_length: int = field(default=512, metadata={"help": "Max sequence length."})
    max_prompt_length: int = field(default=256, metadata={"help": "Max prompt length."})
    loss_type: str = field(default="sigmoid", metadata={"help": "DPO loss type."})
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    logging_steps: int = field(default=1, metadata={"help": "Logging steps."})
    save_steps: int = field(default=50, metadata={"help": "Save steps."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio."})

    # Tracking and Hub
    push_to_hub: bool = field(default=False, metadata={"help": "Push the model to HF Hub."})
    hub_model_id: str = field(default=None, metadata={"help": "The model ID to push to on the Hub."})
    wandb_project: str = field(default="tinker-dpo-shorter", metadata={"help": "Wandb project name."})
    wandb_run_name: str = field(default=None, metadata={"help": "Wandb run name."})


def create_preference_dataset(
    prompts: List[str],
    model,
    tokenizer,
    num_generations: int = 4,
    max_new_tokens: int = 128,
) -> Dataset:
    """
    Create pairwise preference dataset favoring shorter responses.

    For each prompt:
    1. Generate num_generations responses
    2. Create pairs where shorter response is chosen
    3. Filter pairs where both end properly (format check)
    """
    data = []
    model_to_use = model
    model_to_use.eval()

    for prompt in prompts:
        # Generate multiple responses
        inputs = tokenizer(prompt, return_tensors="pt").to(model_to_use.device)

        with torch.no_grad():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_generations,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )

        responses = [
            tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            for out in outputs
        ]

        # Check format (ends properly)
        valid_responses = [
            (r, len(r)) for r in responses
            if r.strip() and not r.endswith("...")  # Basic format check
        ]

        if len(valid_responses) < 2:
            continue

        # Sort by length (shortest first)
        valid_responses.sort(key=lambda x: x[1])

        # Create pairwise preferences (shorter is chosen)
        for i, (chosen, chosen_len) in enumerate(valid_responses[:-1]):
            for rejected, rejected_len in valid_responses[i + 1:]:
                if rejected_len > chosen_len * 1.1:  # At least 10% longer
                    data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

    return Dataset.from_list(data)


def load_prompts(num_prompts: int = 500) -> List[str]:
    """Load diverse prompts for preference training."""
    # Sample prompts (in practice, load from dataset like NoRobots)
    base_prompts = [
        "Explain what machine learning is.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "Describe the water cycle.",
        "What is the capital of France?",
        "Explain how computers work.",
        "What are the primary colors?",
        "How do airplanes fly?",
        "What is gravity?",
        "Explain the concept of democracy.",
    ]

    # Expand with variations
    prompts = []
    for i in range(num_prompts):
        prompt = base_prompts[i % len(base_prompts)]
        if i >= len(base_prompts):
            prompt = f"Briefly: {prompt}"
        prompts.append(prompt)

    return prompts


def main():
    parser = HfArgumentParser((DPOScriptArguments,))
    args, = parser.parse_args_into_dataclasses()
    
    # Seed management for reproducibility
    env_info = set_global_seed(args.seed)
    logger.info(f"Seed set to {args.seed} | Environment: {env_info}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )

    # Load reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )

    # Create preference dataset
    logger.info("Generating preference dataset (this may take a while)...")
    prompts = load_prompts(num_prompts=args.num_prompts)

    preference_dataset = create_preference_dataset(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
    )

    logger.info(f"Created {len(preference_dataset)} preference pairs")

    # DPO Configuration
    output_dir = os.environ.get("RESULTS_DIR", args.output_dir)
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=args.beta,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        loss_type=args.loss_type,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        report_to="wandb",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )

    logger.info("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
    )

    logger.info("Starting DPO training for shorter responses...")
    train_result = trainer.train(resume_from_checkpoint=True)
    
    # Log final metrics to wandb
    metrics = train_result.metrics
    wandb.log(metrics)
    
    final_output_dir = f"{output_dir}_final"
    trainer.save_model(final_output_dir)
    
    if args.push_to_hub:
        logger.info(f"Pushing to Hub: {args.hub_model_id}")
        trainer.push_to_hub()
        
    logger.info("Training complete!")
    wandb.finish()
    
    # Log experiment metadata
    log_experiment_metadata(
        experiment_name="trl_dpo_shorter",
        seed=args.seed,
        hyperparameters={
            "model_name": args.model_name,
            "learning_rate": args.learning_rate,
            "num_generations": args.num_generations,
            "beta": args.beta,
        },
        output_dir=final_output_dir,
    )


if __name__ == "__main__":
    main()
