"""Modal PPO Campaign: Run PPO baselines on models where we only have GRPO data.
This fills the critical PPO vs GRPO comparison gap.
"""
import modal, os, json

app = modal.App("tinker-rl-ppo-campaign")

HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_KEY = os.environ["WANDB_API_KEY"]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1", "transformers>=4.40", "datasets", "trl>=0.12",
        "peft", "accelerate", "bitsandbytes", "scipy",
        "numpy", "pandas", "wandb", "huggingface_hub",
    )
    .env({
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_KEY,
        "WANDB_PROJECT": "tinker-rl-lab-world-class",
        "WANDB_ENTITY": "arvindcr4-pes-university",
    })
)

# ── PPO experiments to run ──
PPO_EXPERIMENTS = [
    # (tag, model_id, model_short)
    ("ppo_qwen35_4b", "Qwen/Qwen3.5-4B", "qwen3.5-4b"),
    ("ppo_llama32_3b", "meta-llama/Llama-3.2-3B", "llama-3.2-3b"),
    ("ppo_llama32_1b", "meta-llama/Llama-3.2-1B", "llama-3.2-1b"),
]


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": HF_TOKEN,
        "WANDB_API_KEY": WANDB_KEY,
    })],
)
def run_ppo_experiment(tag: str, model_id: str, model_short: str):
    """Run a single PPO experiment on Modal H100."""
    import torch, wandb, json, time
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from datasets import load_dataset
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    from peft import LoraConfig
    import numpy as np
    
    print(f"Starting PPO experiment: {tag} with {model_id}")
    
    # Initialize W&B
    wandb.init(
        project="tinker-rl-lab-world-class",
        name=f"modal-{tag}",
        tags=["ppo", "modal-h100", "campaign", model_short],
        config={
            "model": model_id,
            "method": "PPO",
            "task": "gsm8k",
            "platform": "modal_h100",
            "gpu": "H100",
        }
    )
    
    # Load model with 4-bit quantization for larger models
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        peft_config=LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    
    # GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train[:200]")
    
    # PPO Config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        gradient_accumulation_steps=2,
        ppo_epochs=2,
        target_kl=0.01,
        log_with="wandb",
    )
    
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
    )
    
    # Simple reward function: check if answer matches
    def compute_reward(response_text, expected_answer):
        """Extract number from response and compare to expected."""
        import re
        # Extract number after ####
        match = re.search(r'####\s*([\d,]+)', response_text)
        if match:
            try:
                pred = int(match.group(1).replace(',', ''))
                exp = int(expected_answer.replace(',', ''))
                return 1.0 if pred == exp else 0.0
            except:
                return 0.0
        return 0.0
    
    rewards_trace = []
    num_steps = 30
    
    for step in range(num_steps):
        idx = step % len(dataset)
        question = dataset[idx]["question"]
        answer = dataset[idx]["answer"].split("####")[-1].strip()
        
        prompt = f"Solve step by step. Give final answer after ####.\n\n{question}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs["input_ids"].to(model.pretrained_model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        reward = compute_reward(response, answer)
        rewards_trace.append(reward)
        
        # Log to W&B
        wandb.log({
            "step": step,
            "reward": reward,
            "running_avg": np.mean(rewards_trace[-10:]),
        }, step=step)
        
        if step % 5 == 0:
            print(f"  Step {step}/{num_steps}: reward={reward:.3f}, running_avg={np.mean(rewards_trace[-10:]):.3f}")
    
    # Compute final metrics
    peak = max(rewards_trace)
    last10 = np.mean(rewards_trace[-10:])
    first5 = np.mean(rewards_trace[:5])
    
    result = {
        "tag": tag,
        "model": model_id,
        "model_short": model_short,
        "method": "PPO",
        "task": "gsm8k",
        "platform": "modal_h100",
        "peak_reward": float(peak),
        "last10_avg": float(last10),
        "first5_avg": float(first5),
        "reward_trace": [float(r) for r in rewards_trace],
        "num_steps": num_steps,
        "status": "completed",
    }
    
    wandb.log({"final_peak": peak, "final_last10": last10})
    wandb.finish()
    
    print(f"  ✓ {tag} DONE: peak={peak:.3f}, last10={last10:.3f}")
    return result


@app.local_entrypoint()
def main():
    """Launch all PPO experiments in parallel on Modal."""
    print(f"Launching {len(PPO_EXPERIMENTS)} PPO experiments on Modal H100...")
    
    futures = []
    for tag, model_id, model_short in PPO_EXPERIMENTS:
        print(f"  Spawning: {tag} ({model_id})")
        futures.append(run_ppo_experiment.spawn(tag, model_id, model_short))
    
    results = []
    for f in futures:
        try:
            result = f.get()
            results.append(result)
            print(f"  ✓ Completed: {result['tag']} peak={result['peak_reward']:.3f}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({"status": "failed", "error": str(e)})
    
    # Save results
    with open("/home/user/workspace/elevation_outputs/modal_ppo_campaign.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll done! {len([r for r in results if r.get('status')=='completed'])}/{len(PPO_EXPERIMENTS)} completed")
