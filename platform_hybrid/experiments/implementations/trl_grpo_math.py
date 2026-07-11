"""
TRL GRPO Math RL Implementation
================================
Port of Tinker Math RL (Arithmetic) experiment to HuggingFace TRL.

Original Tinker Results:
- Starting accuracy: 69.5%
- Final accuracy: 100%
- Steps to convergence: ~20

This implementation uses GRPOTrainer with verifiable binary rewards.
"""

import logging
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import torch
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, HfArgumentParser
from trl import GRPOTrainer, GRPOConfig

from utils.seed import set_global_seed, log_experiment_metadata

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)


@dataclass
class MathProblem:
    """Arithmetic problem with ground truth answer."""
    prompt: str
    answer: int


@dataclass
class ScriptArguments:
    """Arguments for the GRPO Math RL Experiment."""
    model_name: str = field(default="meta-llama/Llama-3.2-1B", metadata={"help": "The model to train."})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})
    num_train_problems: int = field(default=1000, metadata={"help": "Number of problems in the train dataset."})
    num_eval_problems: int = field(default=200, metadata={"help": "Number of problems in the eval dataset."})
    output_dir: str = field(default="./grpo_math_output", metadata={"help": "Output directory for the model."})
    
    # Batch settings
    per_device_train_batch_size: int = field(default=4, metadata={"help": "Batch size per device."})
    gradient_accumulation_steps: int = field(default=25, metadata={"help": "Gradient accumulation steps."})
    
    # GRPO-specific
    num_generations: int = field(default=4, metadata={"help": "Number of generations per prompt."})
    beta: float = field(default=0.1, metadata={"help": "KL penalty coefficient."})
    
    # Learning rate
    learning_rate: float = field(default=1e-4, metadata={"help": "Learning rate."})
    
    # LoRA settings
    use_peft: bool = field(default=True, metadata={"help": "Whether to use PEFT."})
    lora_r: int = field(default=32, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    
    # Generation settings
    max_new_tokens: int = field(default=5, metadata={"help": "Max new tokens to generate."})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    
    # Training
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    logging_steps: int = field(default=1, metadata={"help": "Logging steps."})
    save_steps: int = field(default=10, metadata={"help": "Save steps."})
    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy."})
    eval_steps: int = field(default=10, metadata={"help": "Evaluation steps."})
    save_strategy: str = field(default="steps", metadata={"help": "Save strategy."})
    load_best_model_at_end: bool = field(default=True, metadata={"help": "Load best model at end."})
    metric_for_best_model: str = field(default="eval_reward/mean", metadata={"help": "Metric for best model."})
    
    # Optimization
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    warmup_ratio: float = field(default=0.1, metadata={"help": "Warmup ratio."})
    
    # Wandb & Hub Settings
    wandb_project: Optional[str] = field(default="trl_grpo_math", metadata={"help": "Wandb project name."})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Wandb entity name."})
    push_to_hub: bool = field(default=False, metadata={"help": "Whether to push to HF Hub."})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "Hub model ID."})
    hub_token: Optional[str] = field(default=None, metadata={"help": "Hub token."})


def generate_arithmetic_dataset(num_problems: int = 1000, max_num: int = 99) -> Dataset:
    """Generate arithmetic addition problems."""
    problems = []
    for _ in range(num_problems):
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        prompt = f"What is {a} + {b}? Answer with just the number."
        answer = a + b
        problems.append({
            "prompt": prompt,
            "answer": str(answer),
        })

    return Dataset.from_list(problems)


def extract_answer(completion: str) -> Optional[int]:
    """Extract numeric answer from model completion."""
    # Look for numbers in the response
    numbers = re.findall(r'\b\d+\b', completion)
    if numbers:
        return int(numbers[-1])  # Take last number
    return None


def math_reward_function(completions: List[str], prompts: List[str], answers: List[str]) -> List[float]:
    """
    Verifiable binary reward function.

    Reward structure (matching Tinker):
    - reward=1.0: Correct answer
    - reward=0.0: Wrong answer, correct format
    - reward=-0.1: Wrong format (no number found)
    """
    rewards = []

    for completion, answer in zip(completions, answers):
        predicted = extract_answer(completion)
        expected = int(answer)

        if predicted is None:
            # Wrong format
            rewards.append(-0.1)
        elif predicted == expected:
            # Correct
            rewards.append(1.0)
        else:
            # Wrong answer
            rewards.append(0.0)

    return rewards


def main():
    parser = HfArgumentParser((ScriptArguments,))
    args, = parser.parse_args_into_dataclasses()

    # Setup wandb
    if args.wandb_project:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=asdict(args))

    # Seed management for reproducibility
    env_info = set_global_seed(args.seed)
    logger.info(f"Seed set to {args.seed} | Environment: {env_info}")

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )

    # Generate dataset
    logger.info("Generating arithmetic dataset...")
    dataset = generate_arithmetic_dataset(num_problems=args.num_train_problems)
    eval_dataset = generate_arithmetic_dataset(num_problems=args.num_eval_problems)

    # GRPO Configuration (matching Tinker hyperparameters)
    output_dir = os.environ.get("RESULTS_DIR", args.output_dir)
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        beta=args.beta,
        learning_rate=args.learning_rate,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        report_to="wandb" if args.wandb_project else "none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )

    # Create reward function wrapper
    def reward_fn(completions, prompts, **kwargs):
        batch_answers = kwargs.get("answer")
        if batch_answers and isinstance(batch_answers[0], list):
            batch_answers = [a[0] for a in batch_answers]
        return math_reward_function(completions, prompts, batch_answers)

    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting GRPO training...")
    logger.info("=" * 50)
    logger.info("Expected: reward=0.67 -> 1.0, accuracy=70% -> 100%")
    logger.info("=" * 50)

    trainer.train(resume_from_checkpoint=True)

    # Save final model
    final_output_dir = f"{output_dir}_final_seed{args.seed}"
    trainer.save_model(final_output_dir)
    logger.info(f"Training complete! Model saved to {final_output_dir}")

    if args.push_to_hub:
        logger.info("Pushing to Hugging Face Hub...")
        trainer.push_to_hub()

    # Log experiment metadata
    hyperparameters = {
        "model_name": args.model_name,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_r,
        "num_generations": args.num_generations,
        "beta": args.beta,
    }
    log_experiment_metadata(
        experiment_name="trl_grpo_math",
        seed=args.seed,
        hyperparameters=hyperparameters,
        output_dir=final_output_dir,
    )

    if wandb.run is not None:
        wandb.log({"training_completed": 1})
        wandb.finish()


if __name__ == "__main__":
    main()
