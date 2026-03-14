"""
TRL Chat SL (Supervised Fine-Tuning) Implementation
====================================================
Port of Tinker Chat SL experiment to TRL SFTTrainer.

Original Tinker experiment:
- dataset=no_robots (HuggingFace dataset)
- Standard supervised fine-tuning
- learning_rate=5e-4
- batch_size=32
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def load_no_robots_dataset(num_samples=None):
    """
    Load NoRobots dataset for chat SFT.

    NoRobots is a high-quality instruction-following dataset
    without synthetic/AI-generated responses.
    """
    dataset = load_dataset("HuggingFaceH4/no_robots", split="train")

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    def format_chat(example):
        # Format as chat template
        messages = example.get("messages", [])
        if messages:
            # Convert to text format
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    text += f"User: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            return {"text": text.strip()}
        return {"text": ""}

    return dataset.map(format_chat)


def main():
    # Model configuration (matching Tinker)
    model_name = "meta-llama/Llama-3.2-1B"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load NoRobots dataset (matching Tinker)
    print("Loading NoRobots dataset...")
    dataset = load_no_robots_dataset(num_samples=1000)  # Reduced for demo
    print(f"Loaded {len(dataset)} examples")

    # SFT Configuration (matching Tinker hyperparameters)
    sft_config = SFTConfig(
        output_dir="./chat_sft_output",

        # Batch settings (matching Tinker: batch_size=32)
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch = 32

        # Learning rate (matching Tinker: 5e-4)
        learning_rate=5e-4,

        # Training
        num_train_epochs=3,
        max_seq_length=512,

        # LoRA for efficiency
        use_peft=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,

        # Logging
        logging_steps=10,
        save_steps=100,

        # Optimization
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        weight_decay=0.01,

        # Mixed precision
        fp16=False,
        bf16=True,
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    print("Starting Chat SFT training...")
    print("=" * 50)
    print("Supervised fine-tuning on NoRobots dataset")
    print("=" * 50)

    trainer.train()

    # Save final model
    trainer.save_model("./chat_sft_final")
    print("Training complete! Model saved to ./chat_sft_final")


if __name__ == "__main__":
    main()
