#!/usr/bin/env python3
"""
BITTER LESSON CAMPAIGN: Launch maximum parallel Tinker experiments.
Every untested model × GSM8K, plus base-vs-instruct pairs, group-size sweep,
multi-seed replication, and cross-task experiments.
"""

import json, time, os, sys, traceback
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Credentials ──────────────────────────────────────────────────────────
API_KEY = os.environ["TINKER_API_KEY"]
WANDB_KEY = os.environ["WANDB_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

os.environ["WANDB_API_KEY"] = WANDB_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

from tinker import ServiceClient

# ── Experiment Definitions ───────────────────────────────────────────────
# Each experiment: (tag, model, task_prompt_fn, config_overrides)

EXPERIMENTS = []

# Default config
DEFAULT_CONFIG = {
    "num_steps": 30,
    "group_size": 8,
    "learning_rate": 1e-5,
    "seed": 42,
}

# GSM8K prompt template
GSM8K_SYSTEM = "You are a helpful math tutor. Solve the problem step by step and give the final answer after ####."
GSM8K_PROBLEMS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to make up the remaining amount?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Mark has a garden with flowers. He planted plants of three colors in it. Ten of them are yellow, and there are 80% more of those that are red. Blue flowers make up to 100% more than yellow. How many flowers does Mark have in his garden?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then expression the following items into the box: a 1.5 pounds textbook, a box of mass 0.3 pounds, and a bag of peanuts mass of 0.5 pounds. What is the weight of Ken's care package, in pounds?",
]

def make_gsm8k_prompts():
    return [{"role": "system", "content": GSM8K_SYSTEM},
            *[{"role": "user", "content": p} for p in GSM8K_PROBLEMS[:2]]]

# ═══════════════════════════════════════════════════════════════════════════
# WAVE 1: New Model Frontier (all untested models on GSM8K)
# ═══════════════════════════════════════════════════════════════════════════

WAVE1_MODELS = [
    # (tag, model_name, priority)
    ("qwen35-397b", "Qwen/Qwen3.5-397B-A17B", "CRITICAL"),
    ("gpt-oss-120b", "openai/gpt-oss-120b", "HIGH"),
    ("llama33-70b-inst", "meta-llama/Llama-3.3-70B-Instruct", "HIGH"),
    ("llama31-70b-base", "meta-llama/Llama-3.1-70B", "HIGH"),
    ("kimi-k25", "moonshotai/Kimi-K2.5", "HIGH"),
    ("kimi-k2-thinking", "moonshotai/Kimi-K2-Thinking", "HIGH"),
    ("qwen35-35b-moe", "Qwen/Qwen3.5-35B-A3B", "MEDIUM"),
    ("qwen3-8b-base", "Qwen/Qwen3-8B-Base", "HIGH"),
    ("deepseek-v31-base", "deepseek-ai/DeepSeek-V3.1-Base", "HIGH"),
    ("nemotron3-nano-30b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "MEDIUM"),
    ("llama32-1b", "meta-llama/Llama-3.2-1B", "MEDIUM"),
    ("llama32-3b", "meta-llama/Llama-3.2-3B", "MEDIUM"),
    ("llama31-8b-base", "meta-llama/Llama-3.1-8B", "HIGH"),
    ("qwen3-4b-inst-2507", "Qwen/Qwen3-4B-Instruct-2507", "MEDIUM"),
    ("qwen3-30b-inst-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", "MEDIUM"),
]

for tag, model, priority in WAVE1_MODELS:
    EXPERIMENTS.append({
        "tag": f"wave1_{tag}_gsm8k",
        "model": model,
        "wave": "1-frontier",
        "priority": priority,
        **DEFAULT_CONFIG,
    })

# ═══════════════════════════════════════════════════════════════════════════
# WAVE 2: Group-Size Sweep on Qwen3-8B (test 2-GRPO=DPO)
# ═══════════════════════════════════════════════════════════════════════════

for G in [2, 4, 16, 32]:
    EXPERIMENTS.append({
        "tag": f"wave2_qwen3-8b_G{G}_gsm8k",
        "model": "Qwen/Qwen3-8B",
        "wave": "2-groupsize",
        "priority": "HIGH",
        "num_steps": 30,
        "group_size": G,
        "learning_rate": 1e-5,
        "seed": 42,
    })

# ═══════════════════════════════════════════════════════════════════════════
# WAVE 3: Multi-Seed Replication (frontier models)
# ═══════════════════════════════════════════════════════════════════════════

MULTI_SEED_MODELS = [
    ("deepseek-v31", "deepseek-ai/DeepSeek-V3.1"),
    ("nemotron-120b", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"),
    ("qwen3-235b", "Qwen/Qwen3-235B-A22B-Instruct-2507"),
]

for tag, model in MULTI_SEED_MODELS:
    for seed in [123, 456, 789]:
        EXPERIMENTS.append({
            "tag": f"wave3_{tag}_seed{seed}_gsm8k",
            "model": model,
            "wave": "3-multiseed",
            "priority": "HIGH",
            "num_steps": 30,
            "group_size": 8,
            "learning_rate": 1e-5,
            "seed": seed,
        })

# ═══════════════════════════════════════════════════════════════════════════
# WAVE 4: LR Sweep on Qwen3-8B
# ═══════════════════════════════════════════════════════════════════════════

for lr in [5e-6, 3e-5, 1e-4, 3e-4]:
    EXPERIMENTS.append({
        "tag": f"wave4_qwen3-8b_lr{lr}_gsm8k",
        "model": "Qwen/Qwen3-8B",
        "wave": "4-lr-sweep",
        "priority": "MEDIUM",
        "num_steps": 30,
        "group_size": 8,
        "learning_rate": lr,
        "seed": 42,
    })

# ═══════════════════════════════════════════════════════════════════════════
# WAVE 5: Longer Training (100 steps)
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS.append({
    "tag": "wave5_qwen3-8b_100steps_gsm8k",
    "model": "Qwen/Qwen3-8B",
    "wave": "5-longrun",
    "priority": "HIGH",
    "num_steps": 100,
    "group_size": 8,
    "learning_rate": 1e-5,
    "seed": 42,
})

EXPERIMENTS.append({
    "tag": "wave5_llama31-8b-inst_100steps_gsm8k",
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "wave": "5-longrun",
    "priority": "HIGH",
    "num_steps": 100,
    "group_size": 8,
    "learning_rate": 1e-5,
    "seed": 42,
})


print(f"Total experiments planned: {len(EXPERIMENTS)}")
print()

# Group by wave
from collections import Counter
wave_counts = Counter(e["wave"] for e in EXPERIMENTS)
for w, c in sorted(wave_counts.items()):
    print(f"  {w}: {c} experiments")

print()
print("Experiment list:")
for i, e in enumerate(EXPERIMENTS):
    print(f"  {i+1:3d}. [{e['wave']:15s}] {e['tag']:45s} model={e['model']:50s} G={e.get('group_size',8)} LR={e.get('learning_rate',1e-5)} steps={e.get('num_steps',30)} seed={e.get('seed',42)}")


# ═══════════════════════════════════════════════════════════════════════════
# LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════

def run_single_experiment(exp):
    """Run a single Tinker experiment and return results."""
    tag = exp["tag"]
    model = exp["model"]
    
    try:
        client = ServiceClient(api_key=API_KEY)
        
        training = client.create_lora_training_client(
            model=model,
            train_method="grpo",
            train_config={
                "grpo_config": {
                    "num_generations": exp.get("group_size", 8),
                    "max_completion_tokens": 512,
                    "loss_type": "importance_sampling",
                },
                "optimizer_config": {"lr": exp.get("learning_rate", 1e-5)},
                "lora_config": {"rank": 32},
            },
        )
        
        # Training loop
        num_steps = exp.get("num_steps", 30)
        rewards = []
        
        for step in range(num_steps):
            # Sample 2 GSM8K problems per step
            problems = GSM8K_PROBLEMS[step % len(GSM8K_PROBLEMS): step % len(GSM8K_PROBLEMS) + 2]
            if len(problems) < 2:
                problems = GSM8K_PROBLEMS[:2]
            
            prompts = []
            for p in problems:
                prompts.append([
                    {"role": "system", "content": GSM8K_SYSTEM},
                    {"role": "user", "content": p},
                ])
            
            result = training.step(prompts=prompts)
            step_reward = result.mean_reward if hasattr(result, 'mean_reward') else 0
            rewards.append(float(step_reward))
            
            if step % 10 == 0:
                print(f"  [{tag}] Step {step}/{num_steps}: reward={step_reward:.4f}")
        
        # Compute metrics
        peak = max(rewards)
        last10 = sum(rewards[-10:]) / min(10, len(rewards))
        first5 = sum(rewards[:5]) / min(5, len(rewards))
        
        result_data = {
            "tag": tag,
            "model": model,
            "wave": exp["wave"],
            "status": "completed",
            "num_steps": num_steps,
            "group_size": exp.get("group_size", 8),
            "learning_rate": exp.get("learning_rate", 1e-5),
            "seed": exp.get("seed", 42),
            "peak_reward": peak,
            "last10_avg": last10,
            "first5_avg": first5,
            "reward_trace": rewards,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        print(f"  ✓ [{tag}] DONE: peak={peak:.3f}, last10={last10:.3f}")
        return result_data
        
    except Exception as e:
        print(f"  ✗ [{tag}] FAILED: {e}")
        traceback.print_exc()
        return {
            "tag": tag,
            "model": model,
            "wave": exp["wave"],
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


def launch_campaign(max_parallel=8):
    """Launch all experiments with controlled parallelism."""
    results = []
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_file = f"/home/user/workspace/tinker-rl-lab/experiments/tinker-runs/results/campaign_{timestamp}.json"
    
    print(f"\n{'='*70}")
    print(f"LAUNCHING BITTER LESSON CAMPAIGN — {len(EXPERIMENTS)} experiments")
    print(f"Max parallel: {max_parallel}")
    print(f"Results file: {results_file}")
    print(f"{'='*70}\n")
    
    # Sort by priority: CRITICAL first, then HIGH, then MEDIUM
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    sorted_exps = sorted(EXPERIMENTS, key=lambda e: priority_order.get(e.get("priority", "MEDIUM"), 2))
    
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_single_experiment, exp): exp for exp in sorted_exps}
        
        for i, future in enumerate(as_completed(futures)):
            exp = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save incrementally
                with open(results_file, "w") as f:
                    json.dump({
                        "campaign": "bitter_lesson",
                        "timestamp": timestamp,
                        "total_planned": len(EXPERIMENTS),
                        "completed": len(results),
                        "results": results,
                    }, f, indent=2)
                
                completed = len([r for r in results if r.get("status") == "completed"])
                failed = len([r for r in results if r.get("status") == "failed"])
                print(f"\n  Progress: {len(results)}/{len(EXPERIMENTS)} "
                      f"(✓{completed} ✗{failed})\n")
                
            except Exception as e:
                print(f"  Future error for {exp['tag']}: {e}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"CAMPAIGN COMPLETE")
    print(f"{'='*70}")
    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]
    print(f"Completed: {len(completed)}/{len(EXPERIMENTS)}")
    print(f"Failed: {len(failed)}/{len(EXPERIMENTS)}")
    
    if completed:
        print(f"\nTop results:")
        for r in sorted(completed, key=lambda x: x.get("last10_avg", 0), reverse=True)[:10]:
            print(f"  {r['tag']:45s} peak={r['peak_reward']:.3f} last10={r['last10_avg']:.3f}")
    
    print(f"\nResults saved to: {results_file}")
    return results


if __name__ == "__main__":
    if "--launch" in sys.argv:
        max_p = int(sys.argv[sys.argv.index("--launch") + 1]) if len(sys.argv) > sys.argv.index("--launch") + 1 else 8
        launch_campaign(max_parallel=max_p)
    else:
        print("\nDry run complete. Use --launch [N] to start (N = max parallel workers)")
