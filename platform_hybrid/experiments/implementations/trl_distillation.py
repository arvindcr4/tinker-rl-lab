"""
TRL Knowledge Distillation Implementation
==========================================
Port of Tinker Distillation experiments to TRL/transformers.

Two approaches:
1. Off-Policy: SFT on teacher outputs (OpenThoughts3 style)
2. On-Policy: KL divergence minimization to teacher
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir())))
from platform_hybrid.experiments.research_contracts import distillation_loss


import atexit
try:
    from codecarbon import EmissionsTracker
    _tracker = EmissionsTracker()
    _tracker.start()
    atexit.register(_tracker.stop)
except ImportError:
    pass


import sys
import os
import torch
import torch.nn.functional as F
import wandb
from typing import List
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# Add project root to path for utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.seed import set_global_seed, get_seed_from_args, log_experiment_metadata


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    teacher_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    student_model_name: str = "meta-llama/Llama-3.2-1B"
    temperature: float = 2.0
    alpha: float = 0.5
    max_length: int = 512


def generate_teacher_dataset(
    teacher_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
) -> Dataset:
    """Generate teacher outputs for off-policy distillation."""
    teacher_model.eval()
    data = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(teacher_model.device)

        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        data.append({
            "prompt": prompt,
            "completion": response,
            "text": f"{prompt}\n{response}",
        })

    return Dataset.from_list(data)


def train_off_policy_distillation(config: DistillationConfig, prompts: List[str]):
    """Off-policy distillation: SFT on teacher outputs."""
    print("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Generating teacher dataset...")
    teacher_dataset = generate_teacher_dataset(teacher, tokenizer, prompts)

    del teacher
    torch.cuda.empty_cache()

    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )

    sft_config = SFTConfig(
        output_dir=os.environ.get("RESULTS_DIR", "./distillation_off_policy"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        max_length=config.max_length,
        logging_steps=10,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=f"{config.student_model_name.split('/')[-1]}-distilled-off-policy",
    )

    trainer = SFTTrainer(
        model=student,
        train_dataset=teacher_dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    print("Starting off-policy distillation...")
    wandb.init(project="trl-distillation", name="off-policy")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("./distillation_off_policy_final")
    trainer.push_to_hub()
    wandb.finish()
    return trainer


class OnPolicyDistillationTrainer(Trainer):
    """Fixed-history teacher matching; compatibility name retained for imports."""

    def __init__(self, teacher_model, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.pop("labels", None)
        response_mask = inputs.pop("response_mask", None)
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        loss, task_loss, kl_loss = distillation_loss(student_outputs.logits, teacher_outputs.logits,
            labels, inputs.get("attention_mask"), response_mask, self.temperature, self.alpha)
        return (loss, student_outputs) if return_outputs else loss


def train_on_policy_distillation(config: DistillationConfig, train_dataset: Dataset):
    """On-policy distillation with KL minimization."""
    print("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )

    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
        device_map=None if "LOCAL_RANK" in os.environ else "auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=os.environ.get("RESULTS_DIR", "./distillation_fixed_history"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        fp16=True,
        remove_unused_columns=False,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=f"{config.student_model_name.split('/')[-1]}-distilled-fixed-history",
    )

    trainer = OnPolicyDistillationTrainer(
        teacher_model=teacher,
        temperature=config.temperature,
        alpha=config.alpha,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    teacher_tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Teacher/student tokenizer vocabularies differ")
    if "labels" not in train_dataset.column_names or "response_mask" not in train_dataset.column_names:
        raise ValueError("Fixed-history teacher matching requires explicit causal labels and response_mask")
    print("Starting fixed-history teacher matching (not on-policy distillation)...")
    wandb.init(project="trl-distillation", name="fixed-history")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("./distillation_fixed_history_final")
    trainer.push_to_hub()
    wandb.finish()
    return trainer


def main():
    raise ValueError("Supply an explicit labeled response dataset; default fixed prompts are not on-policy trajectories")
    config = DistillationConfig()

    prompts = [
        "Explain machine learning in simple terms.",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
    ] * 30

    print("=" * 60)
    print("Tinker Distillation -> TRL Implementation")
    print("=" * 60)

    print("\n--- Part 1: Off-Policy Distillation ---")
    train_off_policy_distillation(config, prompts)

    print("\n--- Part 2: On-Policy Distillation ---")
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_dict({"text": prompts})
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=config.max_length, padding="max_length"),
        batched=True
    )
    train_on_policy_distillation(config, tokenized)

    print("\nDistillation complete!")


if __name__ == "__main__":
    main()
