"""
TRL GRPO Training Script
Generated for config: Qwen/Qwen2.5-1.5B-Instruct
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer
from platform_local.unified.peft_utils import apply_peft_method, save_bitfit_checkpoint

# Configuration
MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
TRAIN_FILES = ['test.json']
PEFT_METHOD = 'lora'
USE_PEFT = True
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
BF16 = True
FP16 = False
BOXED_MARKER = '\\boxed{'
COMPUTE_DTYPE = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
quantization_config = None
if LOAD_IN_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    )
elif LOAD_IN_8BIT:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_kwargs = {
    "device_map": None if "LOCAL_RANK" in os.environ else "auto",
}
if BF16:
    model_kwargs["torch_dtype"] = torch.bfloat16
elif FP16:
    model_kwargs["torch_dtype"] = torch.float16
if quantization_config is not None:
    model_kwargs["quantization_config"] = quantization_config

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)

# PEFT Setup
if USE_PEFT:
    model = apply_peft_method(
        model,
        method=PEFT_METHOD,
        quantized=quantization_config is not None,
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=None,
        num_virtual_tokens=32,
        encoder_hidden_size=128,
    )

# Data
train_dataset = load_dataset("json", data_files=TRAIN_FILES, split="train")

# Process Reward Model (PRM) & Diagnostic tracking
import numpy as np
import wandb

def prm_reward_fn(completions, prompts=None, **kwargs):
    """
    Evaluates step-level fidelity using a Process Reward Model (PRM) approach
    and tracks Advantage Collapse Rate (ACR) across groups.
    """
    def completion_text(completion):
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
            return str(completion[-1].get("content", ""))
        return str(completion)

    rewards = []
    lengths = []
    for completion in completions:
        text = completion_text(completion).lower()
        lengths.append(len(text))
        step_reward = 0.0
        
        # PRM intermediate step fidelity (mock logic for step reasoning)
        steps_found = text.count("step") + text.count("first") + text.count("then")
        if steps_found > 0:
            step_reward += min(0.5, 0.1 * steps_found)
            
        # Final outcome ORM
        if BOXED_MARKER in text:
            step_reward += 0.5
            
        rewards.append(step_reward)
        
    # Advantage Diagnostics (ACR & Variance & Length Bias)
    if len(rewards) > 1:
        variance = np.var(rewards)
        acr = 1.0 if variance < 1e-4 else 0.0
        
        correct_lens = [l for l, r in zip(lengths, rewards) if r > 0.5]
        incorrect_lens = [l for l, r in zip(lengths, rewards) if r <= 0.5]
        mean_len_correct = np.mean(correct_lens) if correct_lens else 0.0
        mean_len_incorrect = np.mean(incorrect_lens) if incorrect_lens else 0.0
        
        len_reward_corr = 0.0
        if correct_lens and incorrect_lens:
            n1, n0, n = len(correct_lens), len(incorrect_lens), len(lengths)
            std_y = np.std(lengths)
            len_reward_corr = ((mean_len_correct - mean_len_incorrect) / (std_y + 1e-8)) * np.sqrt((n1 * n0) / (n * (n - 1)))
            
        if wandb.run is not None:
            wandb.log({
                "diagnostics/advantage_variance": variance,
                "diagnostics/advantage_collapse_rate": acr,
                "diagnostics/mean_prm_reward": np.mean(rewards),
                "diagnostics/mean_len_correct": mean_len_correct,
                "diagnostics/mean_len_incorrect": mean_len_incorrect,
                "diagnostics/length_reward_corr": len_reward_corr
            }, commit=False)
            
    return rewards

# GRPO Config
grpo_config = GRPOConfig(
    output_dir="./checkpoints",
    num_train_epochs=20,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-06,
    max_grad_norm=1.0,
    max_steps=-1,
    bf16=BF16,
    fp16=FP16,
    gradient_checkpointing=True,
    report_to='wandb',
    logging_steps=1,
    save_strategy="no" if USE_PEFT and PEFT_METHOD == "bitfit" else "steps",
    save_steps=10,
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    reward_funcs=[prm_reward_fn],
    processing_class=tokenizer,
)

trainer.train()
if USE_PEFT and PEFT_METHOD == "bitfit":
    save_bitfit_checkpoint(
        model,
        "./checkpoints/bitfit_adapter.pt",
        base_model_name=MODEL_NAME,
    )
print("Training complete!")
