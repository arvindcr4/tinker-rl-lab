"""
TRL Knowledge Distillation Implementation
==========================================
Port of Tinker Distillation experiments to TRL/transformers.

Two approaches:
1. Off-Policy: SFT on teacher outputs (OpenThoughts3 style)
2. On-Policy: KL divergence minimization to teacher
"""

import torch
import torch.nn.functional as F
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
        device_map="auto",
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
        device_map="auto",
    )

    sft_config = SFTConfig(
        output_dir="./distillation_off_policy",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        max_seq_length=config.max_length,
        logging_steps=10,
    )

    trainer = SFTTrainer(
        model=student,
        train_dataset=teacher_dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    print("Starting off-policy distillation...")
    trainer.train()
    trainer.save_model("./distillation_off_policy_final")
    return trainer


class OnPolicyDistillationTrainer(Trainer):
    """On-policy distillation with KL divergence to teacher."""

    def __init__(self, teacher_model, temperature: float = 2.0, alpha: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Combined loss: task loss + KL divergence."""
        labels = inputs.pop("labels", None)

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        if labels is not None:
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            task_loss = torch.tensor(0.0, device=student_logits.device)

        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        kl_loss = kl_loss * (self.temperature ** 2)

        loss = self.alpha * task_loss + (1 - self.alpha) * kl_loss

        if return_outputs:
            return loss, student_outputs
        return loss


def train_on_policy_distillation(config: DistillationConfig, train_dataset: Dataset):
    """On-policy distillation with KL minimization."""
    print("Loading teacher model...")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir="./distillation_on_policy",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        fp16=True,
        remove_unused_columns=False,
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

    print("Starting on-policy distillation (KL minimization)...")
    trainer.train()
    trainer.save_model("./distillation_on_policy_final")
    return trainer


def main():
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
