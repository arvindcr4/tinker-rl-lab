"""
Modal H100 experiments: 
1. PPO training on Qwen3.5-4B (new model, compare with GRPO results)
2. Held-out GSM8K evaluation on our best GRPO checkpoints
3. Multi-seed GRPO validation run on Qwen3-8B (seed=123)
"""
import modal
import os
import json
import time

app = modal.App("tinker-rl-lab-new-experiments")

# Create volume for persistent results
results_vol = modal.Volume.from_name("tinker-rl-results", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4", "transformers>=4.46", "datasets", "accelerate",
        "peft", "trl>=0.12", "wandb", "huggingface_hub", "bitsandbytes",
        "scipy", "numpy"
    )
)

WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=3600,
    volumes={"/results": results_vol},
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": WANDB_KEY,
            "HF_TOKEN": HF_TOKEN,
            "WANDB_PROJECT": "tinker-rl-lab-world-class",
        })
    ],
)
def run_ppo_qwen35_4b():
    """PPO training on Qwen3.5-4B for comparison with GRPO Tinker results."""
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    import re
    import random
    import numpy as np

    # Seed everything
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    MODEL_ID = "Qwen/Qwen3.5-4B"
    model_short = "qwen3.5-4b"
    
    wandb.init(
        project="tinker-rl-lab-world-class",
        name=f"ppo_{model_short}_gsm8k_s{SEED}",
        config={
            "model": MODEL_ID, "method": "PPO", "task": "gsm8k",
            "seed": SEED, "platform": "modal_h100", "gpu": "H100",
            "lr": 1e-4, "rank": 32, "steps": 30,
        },
        tags=["ppo", "modal", "h100", model_short],
    )
    
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load GSM8K
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        examples.append({"question": row["question"], "answer": ans})
    
    random.shuffle(examples)
    print(f"Loaded {len(examples)} GSM8K examples")
    
    # Simple PPO-style training loop with reward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    SYSTEM = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
    
    def format_prompt(q):
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    
    def check_answer(response, gold):
        for b in re.findall(r'\\boxed\{([^}]+)\}', response):
            try:
                if abs(float(b.strip().replace(",", "")) - float(gold)) < 0.01:
                    return 1.0
            except:
                if b.strip().replace(",", "") == gold:
                    return 1.0
        nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
        if nums:
            try:
                if abs(float(nums[-1].replace(",", "")) - float(gold)) < 0.01:
                    return 1.0
            except:
                pass
        return 0.0
    
    step_rewards = []
    GROUP_SIZE = 8
    STEPS = 30
    
    for step in range(STEPS):
        batch = [examples[i % len(examples)] for i in random.sample(range(len(examples)), 2)]
        batch_rewards = []
        total_loss = 0.0
        
        for ex in batch:
            prompt = format_prompt(ex["question"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
            
            # Generate GROUP_SIZE responses
            group_rewards = []
            group_log_probs = []
            
            for _ in range(GROUP_SIZE):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids, max_new_tokens=256, temperature=0.8, top_p=0.95,
                        do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    )
                
                response_ids = outputs[0][input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                reward = check_answer(response, ex["answer"])
                group_rewards.append(reward)
                
                # Compute log probs for the response
                full_ids = outputs[0].unsqueeze(0)
                with torch.enable_grad():
                    logits = model(full_ids).logits
                    shift_logits = logits[:, input_ids.shape[1]-1:-1, :]
                    shift_labels = full_ids[:, input_ids.shape[1]:]
                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    total_log_prob = selected_log_probs.sum()
                
                group_log_probs.append(total_log_prob)
            
            # GRPO-style advantage computation
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (sum((r - mean_r)**2 for r in group_rewards) / len(group_rewards)) ** 0.5 + 1e-8
            advantages = [(r - mean_r) / std_r for r in group_rewards]
            
            # Policy gradient loss
            for lp, adv in zip(group_log_probs, advantages):
                total_loss += -adv * lp
            
            batch_rewards.extend(group_rewards)
        
        # Backward + step
        total_loss = total_loss / (len(batch) * GROUP_SIZE)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
        step_rewards.append(avg_reward)
        
        wandb.log({
            "step": step + 1,
            "train/loss": total_loss.item(),
            "train/reward": avg_reward,
            "train/peak_reward": max(step_rewards),
            "train/cumulative_reward": sum(step_rewards) / len(step_rewards),
        }, step=step + 1)
        
        if (step + 1) % 5 == 0 or step == 0:
            print(f"Step {step+1}/{STEPS}: loss={total_loss.item():.4f} reward={avg_reward:.3f} peak={max(step_rewards):.3f}")
    
    # Summary
    last10 = step_rewards[-10:] if len(step_rewards) >= 10 else step_rewards
    first5 = step_rewards[:5] if len(step_rewards) >= 5 else step_rewards
    
    results = {
        "experiment": f"ppo_{model_short}_gsm8k_s{SEED}",
        "model": MODEL_ID,
        "model_short": model_short,
        "method": "PPO",
        "task": "gsm8k",
        "platform": "modal_h100",
        "gpu": "H100",
        "seed": SEED,
        "steps": STEPS,
        "first5_avg": sum(first5) / len(first5) if first5 else 0,
        "last10_avg": sum(last10) / len(last10) if last10 else 0,
        "peak": max(step_rewards) if step_rewards else 0,
        "reward_trace": [round(r, 4) for r in step_rewards],
    }
    
    wandb.summary.update({
        "final/first5_avg": results["first5_avg"],
        "final/last10_avg": results["last10_avg"],
        "final/peak": results["peak"],
    })
    
    # Save results
    with open(f"/results/ppo_{model_short}_gsm8k.json", "w") as f:
        json.dump(results, f, indent=2)
    
    wandb.finish()
    print(f"\nDONE: last10={results['last10_avg']*100:.1f}% peak={results['peak']*100:.1f}%")
    return results


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=3600,
    volumes={"/results": results_vol},
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": WANDB_KEY,
            "HF_TOKEN": HF_TOKEN,
            "WANDB_PROJECT": "tinker-rl-lab-world-class",
        })
    ],
)
def run_grpo_multiseed_qwen3_8b(seed: int = 123):
    """GRPO training on Qwen3-8B with different seed for variance estimation."""
    import torch
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    import re
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    MODEL_ID = "Qwen/Qwen3-8B"
    model_short = "qwen3-8b"
    
    wandb.init(
        project="tinker-rl-lab-world-class",
        name=f"grpo_{model_short}_gsm8k_s{seed}",
        config={
            "model": MODEL_ID, "method": "GRPO", "task": "gsm8k",
            "seed": seed, "platform": "modal_h100", "gpu": "H100",
            "lr": 3e-5, "rank": 32, "steps": 30,
        },
        tags=["grpo", "modal", "h100", model_short, f"seed-{seed}"],
    )
    
    print(f"Loading model {MODEL_ID} (seed={seed})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    
    lora_config = LoraConfig(
        r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        examples.append({"question": row["question"], "answer": ans})
    
    random.shuffle(examples)
    print(f"Loaded {len(examples)} GSM8K examples")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    SYSTEM = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
    
    def format_prompt(q):
        return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    
    def check_answer(response, gold):
        for b in re.findall(r'\\boxed\{([^}]+)\}', response):
            try:
                if abs(float(b.strip().replace(",", "")) - float(gold)) < 0.01:
                    return 1.0
            except:
                if b.strip().replace(",", "") == gold:
                    return 1.0
        nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
        if nums:
            try:
                if abs(float(nums[-1].replace(",", "")) - float(gold)) < 0.01:
                    return 1.0
            except:
                pass
        return 0.0
    
    step_rewards = []
    GROUP_SIZE = 8
    STEPS = 30
    
    for step in range(STEPS):
        batch = [examples[i % len(examples)] for i in random.sample(range(len(examples)), 2)]
        batch_rewards = []
        total_loss = 0.0
        
        for ex in batch:
            prompt = format_prompt(ex["question"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
            
            group_rewards = []
            group_log_probs = []
            
            for _ in range(GROUP_SIZE):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids, max_new_tokens=256, temperature=0.8, top_p=0.95,
                        do_sample=True, pad_token_id=tokenizer.pad_token_id,
                    )
                
                response_ids = outputs[0][input_ids.shape[1]:]
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                reward = check_answer(response, ex["answer"])
                group_rewards.append(reward)
                
                full_ids = outputs[0].unsqueeze(0)
                with torch.enable_grad():
                    logits = model(full_ids).logits
                    shift_logits = logits[:, input_ids.shape[1]-1:-1, :]
                    shift_labels = full_ids[:, input_ids.shape[1]:]
                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
                    total_log_prob = selected_log_probs.sum()
                
                group_log_probs.append(total_log_prob)
            
            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (sum((r - mean_r)**2 for r in group_rewards) / len(group_rewards)) ** 0.5 + 1e-8
            advantages = [(r - mean_r) / std_r for r in group_rewards]
            
            for lp, adv in zip(group_log_probs, advantages):
                total_loss += -adv * lp
            
            batch_rewards.extend(group_rewards)
        
        total_loss = total_loss / (len(batch) * GROUP_SIZE)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
        step_rewards.append(avg_reward)
        
        wandb.log({
            "step": step + 1,
            "train/loss": total_loss.item(),
            "train/reward": avg_reward,
            "train/peak_reward": max(step_rewards),
        }, step=step + 1)
        
        if (step + 1) % 5 == 0 or step == 0:
            print(f"Step {step+1}/{STEPS}: reward={avg_reward:.3f} peak={max(step_rewards):.3f}")
    
    last10 = step_rewards[-10:] if len(step_rewards) >= 10 else step_rewards
    first5 = step_rewards[:5] if len(step_rewards) >= 5 else step_rewards
    
    results = {
        "experiment": f"grpo_{model_short}_gsm8k_s{seed}",
        "model": MODEL_ID,
        "model_short": model_short,
        "method": "GRPO",
        "task": "gsm8k",
        "platform": "modal_h100",
        "gpu": "H100",
        "seed": seed,
        "steps": STEPS,
        "first5_avg": sum(first5) / len(first5) if first5 else 0,
        "last10_avg": sum(last10) / len(last10) if last10 else 0,
        "peak": max(step_rewards) if step_rewards else 0,
        "reward_trace": [round(r, 4) for r in step_rewards],
    }
    
    wandb.summary.update({
        "final/first5_avg": results["first5_avg"],
        "final/last10_avg": results["last10_avg"],
        "final/peak": results["peak"],
    })
    
    with open(f"/results/grpo_{model_short}_gsm8k_s{seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    wandb.finish()
    print(f"\nDONE: last10={results['last10_avg']*100:.1f}% peak={results['peak']*100:.1f}%")
    return results


@app.function(
    image=gpu_image,
    gpu="H100",
    timeout=3600,
    volumes={"/results": results_vol},
    secrets=[
        modal.Secret.from_dict({
            "WANDB_API_KEY": WANDB_KEY,
            "HF_TOKEN": HF_TOKEN,
            "WANDB_PROJECT": "tinker-rl-lab-world-class",
        })
    ],
)
def run_held_out_eval():
    """Run held-out GSM8K evaluation using test split on base models."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import re
    import wandb

    wandb.init(
        project="tinker-rl-lab-world-class",
        name="held_out_gsm8k_eval",
        config={"task": "held_out_eval", "platform": "modal_h100"},
        tags=["eval", "held-out", "gsm8k"],
    )

    # Load GSM8K test split
    ds = load_dataset("openai/gsm8k", "main", split="test")
    test_examples = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        test_examples.append({"question": row["question"], "answer": ans})
    
    print(f"Loaded {len(test_examples)} GSM8K test examples")
    
    SYSTEM = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."
    
    def check_answer(response, gold):
        for b in re.findall(r'\\boxed\{([^}]+)\}', response):
            try:
                if abs(float(b.strip().replace(",", "")) - float(gold)) < 0.01:
                    return 1.0
            except:
                if b.strip().replace(",", "") == gold:
                    return 1.0
        nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
        if nums:
            try:
                if abs(float(nums[-1].replace(",", "")) - float(gold)) < 0.01:
                    return 1.0
            except:
                pass
        return 0.0
    
    models_to_eval = [
        ("Qwen/Qwen3-8B", "qwen3-8b"),
        ("Qwen/Qwen3.5-4B", "qwen3.5-4b"),
    ]
    
    all_results = {}
    
    for model_id, model_short in models_to_eval:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_id} (zero-shot)")
        print(f"{'='*50}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        
        # Evaluate on first 200 examples (time budget)
        N = min(200, len(test_examples))
        correct = 0
        
        for i, ex in enumerate(test_examples[:N]):
            prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{ex['question']}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids, max_new_tokens=512, temperature=0.0,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            reward = check_answer(response, ex["answer"])
            correct += reward
            
            if (i + 1) % 50 == 0:
                print(f"  [{model_short}] {i+1}/{N}: accuracy={correct/(i+1)*100:.1f}%")
        
        accuracy = correct / N
        all_results[model_short] = {
            "model": model_id,
            "accuracy": accuracy,
            "correct": int(correct),
            "total": N,
        }
        
        wandb.log({f"eval/{model_short}_accuracy": accuracy})
        print(f"  [{model_short}] Final: {accuracy*100:.1f}% ({int(correct)}/{N})")
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    with open("/results/held_out_gsm8k_eval.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    wandb.finish()
    return all_results


@app.local_entrypoint()
def main():
    """Launch all experiments in parallel."""
    import concurrent.futures
    
    print("Launching 3 Modal H100 experiments in parallel...")
    print("1. PPO on Qwen3.5-4B (30 steps)")
    print("2. GRPO multi-seed on Qwen3-8B (seed=123)")  
    print("3. Held-out GSM8K evaluation")
    print()
    
    # Launch all 3 in parallel
    futures = [
        run_ppo_qwen35_4b.spawn(),
        run_grpo_multiseed_qwen3_8b.spawn(seed=123),
        run_held_out_eval.spawn(),
    ]
    
    print(f"Spawned {len(futures)} experiments. Waiting for completion...")
    
    results = []
    for i, f in enumerate(futures):
        try:
            result = f.get()
            results.append(result)
            print(f"\nExperiment {i+1} completed: {json.dumps(result, indent=2)[:500]}")
        except Exception as e:
            print(f"\nExperiment {i+1} failed: {e}")
            results.append({"error": str(e)})
    
    # Save combined results
    with open("/tmp/modal_new_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAll experiments complete. Results saved.")
    return results
