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

import torch
from typing import List
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


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
    # Model configuration (matching Tinker)
    model_name = "Qwen/Qwen2-0.5B-Instruct"  # Small instruct model

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load reference model (frozen)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Create preference dataset
    print("Generating preference dataset (this may take a while)...")
    prompts = load_prompts(num_prompts=100)  # Reduced for demo

    preference_dataset = create_preference_dataset(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        num_generations=4,  # group_size in Tinker
        max_new_tokens=100,
    )

    print(f"Created {len(preference_dataset)} preference pairs")

    # DPO Configuration
    dpo_config = DPOConfig(
        output_dir="./dpo_shorter_output",
        beta=0.1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        max_length=512,
        max_prompt_length=256,
        loss_type="sigmoid",
        num_train_epochs=1,
        logging_steps=1,
        save_steps=50,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
    )

    print("Initializing DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=preference_dataset,
        tokenizer=tokenizer,
    )

    print("Starting DPO training for shorter responses...")
    trainer.train()
    trainer.save_model("./dpo_shorter_final")
    print("Training complete!")


if __name__ == "__main__":
    main()
