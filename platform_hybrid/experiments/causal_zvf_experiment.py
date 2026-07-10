#!/usr/bin/env python3
"""
Causal ZVF Experiment — Test whether GRPO learning signal depends on reward diversity.

This experiment directly tests the paper's central claim:
"GRPO learns only when sampled groups contain reward diversity."

Design: Hold everything fixed (model, backend, reward parser, seed, optimizer, LoRA config,
group size, evaluator); change only the prompt pool to create dead, mixed, or saturated
reward groups.

Primary endpoint: Mixed-prompt arm has substantially higher first-5-step GU and positive
reward slope; dead and saturated arms have high ZVF and little usable learning signal.

Secondary endpoint: Held-out GSM8K accuracy (but this is NOT the primary success criterion).

Usage:
    # Step 1: Bin prompts (do once)
    python causal_zvf_experiment.py --phase bin --model Qwen/Qwen3-8B --max-prompts 200

    # Step 2: Run three matched GRPO arms
    python causal_zvf_experiment.py --phase dead --model Qwen/Qwen3-8B --seed 42
    python causal_zvf_experiment.py --phase mixed --model Qwen/Qwen3-8B --seed 42
    python causal_zvf_experiment.py --phase saturated --model Qwen/Qwen3-8B --seed 42

    # Step 3: Evaluate on held-out GSM8K
    python causal_zvf_experiment.py --phase evaluate --run-ids <id1> <id2> <id3>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "model": "Qwen/Qwen3-8B",
    "tinker_model": "Qwen/Qwen3-8B",
    "group_size": 8,
    "total_steps": 30,
    "learning_rate": 1e-5,
    "lora_rank": 32,
    "seed": 42,
    "prompts_per_step": 8,
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.95,
    "use_wandb": True,
    "wandb_project": "tinker-causal-zvf",
    "bin_samples": 8,  # samples per prompt for binning
    "bin_threshold_easy": 0.75,  # >75% success = easy/saturated
    "bin_threshold_hard": 0.25,  # <25% success = hard/dead
    "heldout_n": 500,  # held-out evaluation size
    "heldout_seed": 0,
}

# Difficulty binning thresholds (refined for GSM8K)
# easy: p̂ > 0.75 → saturated (already solved)
# mid:  0.25 ≤ p̂ ≤ 0.75 → mixed (Goldilocks zone)
# hard: p̂ < 0.25 → dead (never solved)


# =============================================================================
# Dataclasses for logging
# =============================================================================

@dataclass
class StepMetrics:
    step: int
    mean_reward: float
    zvf: float  # Zero-Variance Fraction
    gu: float   # Gradient Utilization = 1 - ZVF
    n_effective_groups: int  # groups with nonzero gradient
    mean_group_reward_std: float
    mean_response_length: float
    n_prompts: int
    n_rollouts: int
    algorithm: str = "grpo"


@dataclass
class ExperimentResult:
    phase: str  # "dead", "mixed", "saturated"
    model: str
    seed: int
    config: dict
    run_id: str = ""
    tinker_model_id: str = ""
    steps_completed: int = 0
    metrics: list = field(default_factory=list)
    final_zvf: float = -1.0
    final_gu: float = -1.0
    reward_slope: float = 0.0  # slope of reward vs step (linear fit)
    peak_reward: float = 0.0
    last10_reward: float = 0.0
    heldout_accuracy: float = -1.0
    heldout_ci95: tuple = (-1.0, -1.0)
    elapsed_seconds: float = 0.0


# =============================================================================
# Utility functions
# =============================================================================

def reward_gsm8k(response: str, answer: str) -> float:
    """Standard GSM8K boxed-answer reward."""
    # Try boxed answer first
    m = re.search(r'####\s*(.+?)(?:\s|$)', response)
    if not m:
        # Fall back to last number
        m = re.search(r'(?:answer|Answer|result).*?(\-?\d[\d,]*\.?\d*)', response)
    if not m:
        return 0.0
    pred = m.group(1).replace(",", "").strip()
    target = answer.replace(",", "").strip()
    try:
        return 1.0 if abs(float(pred) - float(target)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if pred == target else 0.0


def compute_zvf_and_gu(rewards_per_prompt: list[list[float]]) -> tuple[float, float, int, float]:
    """
    Compute ZVF and GU from per-prompt reward lists.

    ZVF = fraction of prompts where all G completions have identical rewards
    GU = 1 - ZVF (fraction of prompts with usable gradient signal)
    """
    n_prompts = len(rewards_per_prompt)
    if n_prompts == 0:
        return 1.0, 0.0, 0, 0.0

    zero_var_count = 0
    total_groups = 0
    group_stds = []

    for rewards in rewards_per_prompt:
        if len(rewards) < 2:
            continue
        total_groups += 1
        std = np.std(rewards) if len(rewards) > 1 else 0.0
        group_stds.append(std)
        if std < 1e-8:  # All same reward
            zero_var_count += 1

    zvf = zero_var_count / max(total_groups, 1)
    gu = 1.0 - zvf
    n_effective = sum(1 for std in group_stds if std > 1e-8)
    mean_std = np.mean(group_stds) if group_stds else 0.0

    return zvf, gu, n_effective, mean_std


def compute_reward_slope(rewards_by_step: list[float]) -> float:
    """Compute linear slope of reward vs step number."""
    if len(rewards_by_step) < 2:
        return 0.0
    steps = np.arange(len(rewards_by_step))
    rewards = np.array(rewards_by_step)
    # Simple linear regression
    n = len(steps)
    mean_x = steps.mean()
    mean_y = rewards.mean()
    cov_xy = ((steps - mean_x) * (rewards - mean_y)).sum()
    var_x = ((steps - mean_x) ** 2).sum()
    if var_x < 1e-8:
        return 0.0
    slope = cov_xy / var_x
    return slope


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for accuracy."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


# =============================================================================
# Prompt binning
# =============================================================================

def load_gsm8k_prompts(tokenizer, max_examples: int = 200) -> list[dict]:
    """Load GSM8K train prompts."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for item in ds.select(range(min(len(ds), max_examples))):
        question = item["question"]
        answer = item["answer"].split("####")[-1].strip()
        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"Question: {question}\n\n"
            f"Show your work and put your final answer after ####."
        )
        examples.append({
            "prompt_text": prompt,
            "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False),
            "answer": answer,
        })
    return examples


def sample_prompt(sc, tokenizer, prompt_ids: list[int], answer: str,
                 num_samples: int = 8, max_tokens: int = 512,
                 temperature: float = 0.8, top_p: float = 0.95) -> tuple[list[float], list[int]]:
    """Sample G completions from a prompt and return rewards + response lengths."""
    import tinker.types as T
    prompt_mi = T.ModelInput.from_ints(prompt_ids)
    sp = T.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    responses = sc.sample(prompt_mi, num_samples=num_samples, sampling_params=sp).result()

    rewards = []
    lengths = []
    for resp in responses.sequences:
        text = tokenizer.decode(list(resp.tokens), skip_special_tokens=True)
        r = reward_gsm8k(text, answer)
        rewards.append(r)
        lengths.append(len(resp.tokens))

    return rewards, lengths


def bin_prompts_by_difficulty(
    sc, tokenizer, examples: list[dict],
    num_samples: int = 8,
    threshold_easy: float = 0.75,
    threshold_hard: float = 0.25,
    parallel: int = 16,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """
    Pre-sample each prompt to estimate base success rate, then bin into:
    - dead: p̂ < threshold_hard (cold-start, no learning signal)
    - mixed: threshold_hard ≤ p̂ ≤ threshold_easy (Goldilocks zone)
    - saturated: p̂ > threshold_easy (already solved, no learning signal)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    bins = {"dead": [], "mixed": [], "saturated": []}
    rates = {"dead": [], "mixed": [], "saturated": []}

    if verbose:
        print(f"Binning {len(examples)} prompts by difficulty ({num_samples} samples each)...")

    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(sample_prompt, sc, tokenizer, ex["prompt_ids"], ex["answer"],
                        num_samples): i
            for i, ex in enumerate(examples)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                rewards, lengths = future.result()
            except Exception as e:
                rewards = [0.0] * num_samples

            success_rate = sum(1 for r in rewards if r > 0.5) / len(rewards)
            examples[i]["base_success_rate"] = success_rate
            examples[i]["base_rewards"] = rewards

            if success_rate > threshold_easy:
                bin_name = "saturated"
            elif success_rate < threshold_hard:
                bin_name = "dead"
            else:
                bin_name = "mixed"

            bins[bin_name].append(examples[i])
            rates[bin_name].append(success_rate)
            done += 1

            if verbose and done % 25 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(examples)} binned "
                      f"(dead={len(bins['dead'])}, mixed={len(bins['mixed'])}, "
                      f"saturated={len(bins['saturated'])}) [{elapsed:.0f}s]")

    if verbose:
        elapsed = time.time() - t0
        print(f"\nBinning complete in {elapsed:.0f}s:")
        for bin_name in bins:
            mean_rate = sum(rates[bin_name]) / max(len(rates[bin_name]), 1)
            print(f"  {bin_name}: {len(bins[bin_name])} prompts, "
                  f"mean success rate = {mean_rate:.3f}")

    return bins


# =============================================================================
# GRPO Training Loop
# =============================================================================

def run_grpo_arm(
    phase: str,
    prompts: list[dict],
    sc,
    tc,
    tokenizer,
    config: dict,
    verbose: bool = True,
) -> ExperimentResult:
    """Run one GRPO arm (dead, mixed, or saturated)."""
    import torch
    import tinker.types as T

    group_size = config["group_size"]
    total_steps = config["total_steps"]
    lr = config["learning_rate"]
    max_tokens = config["max_tokens"]
    temperature = config["temperature"]
    top_p = config["top_p"]
    prompts_per_step = min(config.get("prompts_per_step", 8), len(prompts))

    result = ExperimentResult(
        phase=phase,
        model=config["model"],
        seed=config["seed"],
        config=config.copy(),
        run_id="",
        tinker_model_id=tc.model_id,
    )

    t0 = time.time()

    # Accumulate metrics
    step_rewards = []
    step_zvf = []
    step_gu = []

    _advantages: list[float] = []

    def grpo_loss_fn(data, logprobs_list):
        losses = []
        for i, logprobs in enumerate(logprobs_list):
            adv = _advantages[i]
            losses.append(-adv * logprobs.sum())
        loss = torch.stack(losses).mean()
        return loss, {"grpo_loss": loss.item()}

    for step in range(total_steps):
        # Sample batch of prompts
        batch = random.sample(prompts, min(prompts_per_step, len(prompts)))

        all_data: list = []
        all_advs: list[float] = []
        batch_rewards: list[float] = []
        batch_response_lengths: list[int] = []
        rewards_per_prompt: list[list[float]] = []

        for ex in batch:
            prompt_ids = ex["prompt_ids"]
            prompt_mi = T.ModelInput.from_ints(prompt_ids)
            sp = T.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            responses = sc.sample(prompt_mi, num_samples=group_size, sampling_params=sp).result()

            rewards = []
            resp_lengths = []
            for resp in responses.sequences:
                text = tokenizer.decode(list(resp.tokens), skip_special_tokens=True)
                r = reward_gsm8k(text, ex["answer"])
                rewards.append(r)
                resp_lengths.append(len(resp.tokens))

            rewards_per_prompt.append(rewards)
            batch_rewards.extend(rewards)
            batch_response_lengths.extend(resp_lengths)

            # GRPO advantages (group-normalized)
            mean_r = sum(rewards) / len(rewards)
            std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
            advs = [(r - mean_r) / std_r for r in rewards]

            for resp, adv in zip(responses.sequences, advs):
                resp_ids = list(resp.tokens)
                full_ids = prompt_ids + resp_ids
                target_ids = full_ids[1:] + [0]
                datum = T.Datum(
                    model_input=T.ModelInput.from_ints(full_ids),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(
                            data=target_ids, dtype="int64", shape=[len(target_ids)]
                        ),
                    },
                )
                all_data.append(datum)
                all_advs.append(adv)

        if not all_data:
            continue

        # Compute ZVF/GU for this step
        zvf, gu, n_effective, mean_std = compute_zvf_and_gu(rewards_per_prompt)
        step_zvf.append(zvf)
        step_gu.append(gu)

        avg_r = sum(batch_rewards) / len(batch_rewards)
        step_rewards.append(avg_r)

        _advantages = all_advs

        result_fwd = tc.forward_backward_custom(
            data=all_data,
            loss_fn=grpo_loss_fn,
            loss_type_input="logprobs",
        ).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()

        grpo_loss = result_fwd.metrics.get("grpo_loss", float("nan"))

        # Log step metrics
        metrics = StepMetrics(
            step=step,
            mean_reward=avg_r,
            zvf=zvf,
            gu=gu,
            n_effective_groups=n_effective,
            mean_group_reward_std=mean_std,
            mean_response_length=sum(batch_response_lengths) / max(len(batch_response_lengths), 1),
            n_prompts=len(batch),
            n_rollouts=len(batch_rewards),
            algorithm="grpo",
        )
        result.metrics.append(metrics)

        if verbose:
            print(
                f"[{phase:8s}] Step {step+1:2d}/{total_steps} | "
                f"loss={grpo_loss:.4f} | reward={avg_r:.3f} | "
                f"ZVF={zvf:.2f} | GU={gu:.2f} | n_eff={n_effective}"
            )

        # Update sampler
        sc = tc.save_weights_and_get_sampling_client()

    # Finalize result
    result.steps_completed = len(step_rewards)
    result.elapsed_seconds = time.time() - t0

    if step_rewards:
        result.peak_reward = max(step_rewards)
        result.last10_reward = sum(step_rewards[-10:]) / min(len(step_rewards[-10:]), 10)
        result.final_zvf = step_zvf[-1] if step_zvf else -1.0
        result.final_gu = step_gu[-1] if step_gu else -1.0
        result.reward_slope = compute_reward_slope(step_rewards)

    if verbose:
        print(f"\n[{phase}] Completed {result.steps_completed} steps in {result.elapsed_seconds:.0f}s")
        print(f"  Peak reward: {result.peak_reward:.3f}")
        print(f"  Last-10 reward: {result.last10_reward:.3f}")
        print(f"  Reward slope: {result.reward_slope:.4f}")
        print(f"  Final ZVF: {result.final_zvf:.3f}, GU: {result.final_gu:.3f}")

    return result


# =============================================================================
# Held-out Evaluation
# =============================================================================

def load_heldout_gsm8k(n: int = 500, seed: int = 0) -> list[tuple[str, str]]:
    """Load deterministic held-out slice of GSM8K test split."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for row in ds:
        m = re.search(r'####\s*([\-\d,\.]+)', row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        problems.append((row["question"], ans))

    rng = random.Random(seed)
    idx = list(range(len(problems)))
    rng.shuffle(idx)
    return [(problems[i][0], problems[i][1]) for i in idx[:n]]


def evaluate_checkpoint_on_heldout(
    checkpoint: str,
    model: str,
    problems: list[tuple[str, str]],
    tokenizer,
    max_tokens: int = 512,
) -> tuple[float, float, float]:
    """
    Evaluate a checkpoint on held-out GSM8K.
    Returns: (accuracy, ci95_low, ci95_high)
    """
    import tinker
    import tinker.types as T

    svc = tinker.ServiceClient(base_url=None)
    sc = svc.create_sampling_client(model_path=checkpoint)

    correct = 0
    for question, answer in problems:
        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"Question: {question}\n\n"
            f"Show your work and put your final answer after ####."
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_mi = T.ModelInput.from_ints(prompt_ids)
        sp = T.SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
        resp = sc.sample(prompt_mi, num_samples=1, sampling_params=sp).result()
        text = tokenizer.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
        r = reward_gsm8k(text, answer)
        correct += int(r > 0.5)

    n = len(problems)
    acc = correct / n if n > 0 else 0.0
    lo, hi = wilson_ci(correct, n)
    return acc, lo, hi


# =============================================================================
# Main experiment orchestration
# =============================================================================

def run_bin_phase(model: str, max_prompts: int, num_samples: int, output_path: str):
    """Phase 1: Bin prompts by difficulty."""
    from transformers import AutoTokenizer
    import tinker

    print(f"\n{'='*60}")
    print("PHASE: Bin prompts by difficulty")
    print(f"Model: {model}, Max prompts: {max_prompts}, Samples: {num_samples}")
    print(f"{'='*60}\n")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    examples = load_gsm8k_prompts(tok, max_examples=max_prompts)
    print(f"Loaded {len(examples)} GSM8K prompts")

    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model, rank=32)
    sc = tc.save_weights_and_get_sampling_client()
    print("Sampler ready")

    bins = bin_prompts_by_difficulty(
        sc, tok, examples,
        num_samples=num_samples,
        threshold_easy=DEFAULT_CONFIG["bin_threshold_easy"],
        threshold_hard=DEFAULT_CONFIG["bin_threshold_hard"],
        parallel=16,
        verbose=True,
    )

    # Save bin data
    result = {
        "model": model,
        "num_samples": num_samples,
        "total_prompts": len(examples),
        "thresholds": {
            "easy_threshold": DEFAULT_CONFIG["bin_threshold_easy"],
            "hard_threshold": DEFAULT_CONFIG["bin_threshold_hard"],
        },
        "bins": {
            bin_name: {
                "indices": [examples.index(ex) for ex in ex_list],
                "rates": [ex["base_success_rate"] for ex in ex_list],
                "count": len(ex_list),
            }
            for bin_name, ex_list in bins.items()
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"\nSaved bin data to: {output_path}")

    return result


def run_training_arm(
    phase: str,
    bin_data_path: str,
    config: dict,
    output_path: str,
):
    """Phase 2: Run GRPO on a specific difficulty bin."""
    from transformers import AutoTokenizer
    import tinker
    import wandb

    print(f"\n{'='*60}")
    print(f"PHASE: Training arm — {phase}")
    print(f"Model: {config['model']}, Seed: {config['seed']}, Steps: {config['total_steps']}")
    print(f"{'='*60}\n")

    # Load bin data
    bin_data = json.loads(Path(bin_data_path).read_text())
    bin_prompts = bin_data["bins"][phase]["indices"]
    bin_rates = bin_data["bins"][phase]["rates"]

    # Reload all prompts and filter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    all_prompts = load_gsm8k_prompts(tok, max_examples=bin_data["total_prompts"])
    prompts = [all_prompts[i] for i in bin_prompts if i < len(all_prompts)]

    for ex, rate in zip(prompts, bin_rates):
        ex["base_success_rate"] = rate

    print(f"Loaded {len(prompts)} prompts for '{phase}' bin")

    if len(prompts) < 10:
        print(f"ERROR: Too few prompts in '{phase}' bin ({len(prompts)}). Aborting.")
        return None

    # Init Tinker
    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(
        base_model=config["tinker_model"],
        rank=config["lora_rank"],
    )
    sc = tc.save_weights_and_get_sampling_client()

    # Init W&B
    if config.get("use_wandb"):
        wandb.init(
            project=config["wandb_project"],
            name=f"causal-zvf-{phase}-{config['seed']}",
            config={
                "phase": phase,
                "model": config["model"],
                "seed": config["seed"],
                "group_size": config["group_size"],
                "total_steps": config["total_steps"],
                "n_prompts": len(prompts),
                "mean_base_rate": sum(bin_rates) / max(len(bin_rates), 1),
            },
        )

    # Run training
    result = run_grpo_arm(phase, prompts, sc, tc, tok, config, verbose=True)

    # Save result
    result.run_id = tc.model_id
    Path(output_path).write_text(json.dumps(asdict(result), indent=2))
    print(f"\nSaved result to: {output_path}")

    if config.get("use_wandb"):
        # Log summary metrics
        wandb.log({
            "peak_reward": result.peak_reward,
            "last10_reward": result.last10_reward,
            "reward_slope": result.reward_slope,
            "final_zvf": result.final_zvf,
            "final_gu": result.final_gu,
            "steps_completed": result.steps_completed,
            "elapsed_seconds": result.elapsed_seconds,
        })
        wandb.finish()

    return result


def run_evaluate_phase(run_ids: list[str], model: str, heldout_n: int, output_path: str):
    """Phase 3: Evaluate checkpoints on held-out GSM8K."""
    from transformers import AutoTokenizer
    import tinker

    print(f"\n{'='*60}")
    print("PHASE: Held-out evaluation")
    print(f"Model: {model}, Held-out N: {heldout_n}")
    print(f"Run IDs: {run_ids}")
    print(f"{'='*60}\n")

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    problems = load_heldout_gsm8k(n=heldout_n, seed=DEFAULT_CONFIG["heldout_seed"])
    print(f"Loaded {len(problems)} held-out GSM8K problems")

    results = []
    for run_id in run_ids:
        checkpoint = f"tinker://{run_id}"
        print(f"\nEvaluating {checkpoint}...")
        acc, lo, hi = evaluate_checkpoint_on_heldout(checkpoint, model, problems, tok)
        print(f"  Accuracy: {acc:.3f} [95% CI: {lo:.3f}, {hi:.3f}]")
        results.append({
            "run_id": run_id,
            "accuracy": acc,
            "ci95_low": lo,
            "ci95_high": hi,
        })

    output = {
        "model": model,
        "heldout_n": heldout_n,
        "heldout_seed": DEFAULT_CONFIG["heldout_seed"],
        "results": results,
    }
    Path(output_path).write_text(json.dumps(output, indent=2))
    print(f"\nSaved evaluation to: {output_path}")

    return output


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal ZVF Experiment")
    subparsers = parser.add_subparsers(dest="phase", help="Experiment phase")

    # Bin phase
    bin_parser = subparsers.add_parser("bin", help="Bin prompts by difficulty")
    bin_parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    bin_parser.add_argument("--max-prompts", type=int, default=200)
    bin_parser.add_argument("--num-samples", type=int, default=8)
    bin_parser.add_argument("--output", default="./causal_zvf_bins.json")

    # Training phase
    train_parser = subparsers.add_parser("train", help="Run GRPO training arm")
    train_parser.add_argument("--phase", required=True, choices=["dead", "mixed", "saturated"])
    train_parser.add_argument("--bin-data", default="./causal_zvf_bins.json")
    train_parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--group-size", type=int, default=8)
    train_parser.add_argument("--steps", type=int, default=30)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--lora-rank", type=int, default=32)
    train_parser.add_argument("--output", default="./causal_zvf_result.json")

    # Evaluate phase
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on held-out GSM8K")
    eval_parser.add_argument("--run-ids", nargs="+", required=True)
    eval_parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    eval_parser.add_argument("--heldout-n", type=int, default=500)
    eval_parser.add_argument("--output", default="./causal_zvf_heldout.json")

    args = parser.parse_args()

    if args.phase == "bin":
        run_bin_phase(args.model, args.max_prompts, args.num_samples, args.output)

    elif args.phase == "train":
        config = DEFAULT_CONFIG.copy()
        config.update({
            "model": args.model,
            "tinker_model": args.model,
            "seed": args.seed,
            "group_size": args.group_size,
            "total_steps": args.steps,
            "learning_rate": args.lr,
            "lora_rank": args.lora_rank,
        })
        run_training_arm(args.phase, args.bin_data, config, args.output)

    elif args.phase == "evaluate":
        run_evaluate_phase(args.run_ids, args.model, args.heldout_n, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()