"""
TRL Knowledge Distillation Implementation
==========================================
Port of Tinker Distillation experiments to TRL/transformers.

Two approaches:
1. Off-Policy: SFT on teacher outputs (OpenThoughts3 style)
2. Fixed-History: masked teacher matching on explicit labeled responses
3. On-Policy v1: fresh student continuations, response-token forward KL
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(
    0, str(next(p for p in _Path(__file__).resolve().parents if (p / "platform_hybrid").is_dir()))
)
from platform_hybrid.experiments.research_contracts import distillation_loss


import sys
import os
import torch
from typing import List
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Add project root to path for utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


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
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        data.append(
            {
                "prompt": prompt,
                "completion": response,
                "text": f"{prompt}\n{response}",
            }
        )

    return Dataset.from_list(data)


def train_off_policy_distillation(config: DistillationConfig, prompts: List[str]):
    """Off-policy distillation: SFT on teacher outputs."""
    import wandb
    from trl import SFTTrainer, SFTConfig

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


class FixedHistoryDistillationTrainer(Trainer):
    """Teacher matching on explicit, externally supplied response histories."""

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
        loss, task_loss, kl_loss = distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels,
            inputs.get("attention_mask"),
            response_mask,
            self.temperature,
            self.alpha,
        )
        return (loss, student_outputs) if return_outputs else loss


def train_fixed_history_distillation(config: DistillationConfig, train_dataset: Dataset):
    """Fixed-history distillation with explicit causal task targets."""
    import wandb

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

    trainer = FixedHistoryDistillationTrainer(
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
    if (
        "labels" not in train_dataset.column_names
        or "response_mask" not in train_dataset.column_names
    ):
        raise ValueError(
            "Fixed-history teacher matching requires explicit causal labels and response_mask"
        )
    print("Starting fixed-history teacher matching (not on-policy distillation)...")
    wandb.init(project="trl-distillation", name="fixed-history")
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model("./distillation_fixed_history_final")
    trainer.push_to_hub()
    wandb.finish()
    return trainer


def validate_tokenizer_compatibility(student_tokenizer, teacher_tokenizer, student, teacher):
    if student is teacher:
        raise ValueError("Teacher and student must be distinct models")
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Teacher/student tokenizer vocabularies differ")
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        if getattr(student_tokenizer, name) != getattr(teacher_tokenizer, name):
            raise ValueError("Teacher/student tokenizer special-token mappings differ")
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError("Teacher/student model vocabulary sizes differ")
    if max(student_tokenizer.get_vocab().values()) >= student.config.vocab_size:
        raise ValueError("Tokenizer token IDs exceed model vocabulary")

    # Equal token IDs alone do not establish equivalent segmentation/normalization.
    # V1 requires fast-tokenizer serialization, deliberately rejecting unverifiable
    # slow/custom tokenizers instead of silently assuming compatibility.
    if not getattr(student_tokenizer, "is_fast", False) or not getattr(
        teacher_tokenizer, "is_fast", False
    ):
        raise ValueError("On-policy v1 requires verifiable fast tokenizers")
    if student_tokenizer.backend_tokenizer.to_str() != teacher_tokenizer.backend_tokenizer.to_str():
        raise ValueError("Teacher/student tokenizer pipelines differ")


class OnPolicyDistillationTrainer(FixedHistoryDistillationTrainer):
    """Fresh student trajectories, pure forward KL; protocol student-trajectory-kl-v1.

    This is a local single-device semi-gradient objective. Samples are not cached
    across optimizer steps; teacher output tokens are never used as trajectories.
    """

    def __init__(
        self,
        *,
        student_tokenizer,
        teacher_tokenizer,
        max_new_tokens=256,
        eos_token_id,
        pad_token_id,
        alpha=0.0,
        **kwargs,
    ):
        validate_tokenizer_compatibility(
            student_tokenizer, teacher_tokenizer, kwargs["model"], kwargs["teacher_model"]
        )
        if alpha != 0.0:
            raise ValueError(
                "Student-trajectory KL requires alpha=0; sampled student tokens are not task targets"
            )
        super().__init__(alpha=alpha, **kwargs)
        if self.args.world_size != 1 or self.args.deepspeed or self.args.fsdp:
            raise ValueError("Student-trajectory v1 supports local single-device execution only")
        devices = {
            p.device for model in (self.model, self.teacher_model) for p in model.parameters()
        }
        if len(devices) != 1:
            raise ValueError("Teacher and student must share one device")
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        from platform_hybrid.experiments.student_distillation import PROTOCOL_VERSION

        self.protocol_version = PROTOCOL_VERSION
        self.model.config.distillation_protocol = PROTOCOL_VERSION
        self.last_trajectories = ()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from platform_hybrid.experiments.student_distillation import student_trajectory_loss

        if set(inputs) != {"input_ids", "attention_mask"}:
            raise ValueError(
                "On-policy input must contain prompt input_ids and attention_mask only"
            )
        loss, outputs, records = student_trajectory_loss(
            model,
            self.teacher_model,
            inputs["input_ids"],
            inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            behavior_version=int(self.state.global_step),
            temperature=self.temperature,
        )
        self.last_trajectories = records
        return (loss, outputs) if return_outputs else loss


def train_on_policy_distillation(
    config,
    train_dataset,
    *,
    student,
    teacher,
    student_tokenizer,
    teacher_tokenizer,
    training_args,
    max_new_tokens=256,
):
    """Opt-in v1 entry point with caller-supplied local models and prompt dataset.

    No model loading or publication is implicit. This changed protocol must use a
    new output directory; it does not establish results for historical runs.
    """
    if config.alpha != 0.0:
        raise ValueError("On-policy v1 is pure KL: explicitly set config.alpha=0")
    validate_tokenizer_compatibility(student_tokenizer, teacher_tokenizer, student, teacher)
    if set(train_dataset.column_names) != {"input_ids", "attention_mask"}:
        raise ValueError("On-policy dataset must contain tokenized prompts only")
    if training_args.remove_unused_columns:
        raise ValueError("Set remove_unused_columns=False to preserve explicit prompt masks")
    if training_args.push_to_hub or training_args.report_to:
        raise ValueError("Local v1 entry point requires push_to_hub=False and report_to=[]")
    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir):
        raise ValueError("Use a fresh output directory for student-trajectory-kl-v1")
    trainer = OnPolicyDistillationTrainer(
        model=student,
        teacher_model=teacher,
        args=training_args,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        train_dataset=train_dataset,
        temperature=config.temperature,
        alpha=config.alpha,
        eos_token_id=student_tokenizer.eos_token_id,
        pad_token_id=student_tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
    )
    trainer.train(resume_from_checkpoint=False)
    return trainer


def main():
    raise ValueError(
        "Supply an explicit labeled response dataset; default fixed prompts are not on-policy trajectories"
    )
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
        lambda x: tokenizer(
            x["text"], truncation=True, max_length=config.max_length, padding="max_length"
        ),
        batched=True,
    )
    train_fixed_history_distillation(config, tokenized)

    print("\nDistillation complete!")


if __name__ == "__main__":
    main()
