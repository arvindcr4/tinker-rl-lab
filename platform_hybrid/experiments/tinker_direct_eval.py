#!/usr/bin/env python3
"""
platform_hybrid/experiments/tinker_direct_eval.py

Direct Tinker API evaluation on GSM8K (no Atropos needed).
Uses Tinker sampling client for inference, local scoring for rewards.

This provides real model-generated data for ZVF analysis without
requiring the full Atropos environment server.

Usage:
    export TINKER_API_KEY=...
    python3 platform_hybrid/experiments/tinker_direct_eval.py --model Qwen/Qwen3-8B --n-problems 50

Output:
    platform_hybrid/experiments/results/tinker_direct_eval.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "experiments" / "results"
OUT_JSON = RESULTS_DIR / "tinker_direct_eval.json"


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{{}}."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    # Fallback: last number
    nums = re.findall(r"\b\d+\b", text)
    if nums:
        return nums[-1]
    return None


def score_answer(pred: Optional[str], gold: str) -> float:
    """Binary reward: 1 if match, 0 otherwise."""
    if pred is None:
        return 0.0
    # Normalize: remove commas, spaces
    pred_norm = pred.replace(",", "").replace(" ", "").strip()
    gold_norm = gold.replace(",", "").replace(" ", "").strip()
    return 1.0 if pred_norm == gold_norm else 0.0


async def evaluate_with_tinker(
    model_name: str,
    n_problems: int = 50,
    temperature: float = 1.0,
    max_tokens: int = 512,
    group_size: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate model on GSM8K using Tinker API."""
    
    # LIMITATION: "Closed-Source Confound" by benchmarking against open-source alternatives 
    # (e.g., TRL or vLLM) to ensure performance gaps aren't just due to Tinker's proprietary optimizations.
    try:
        import tinker
        from tinker.types import ModelInput, SamplingParams
    except ImportError:
        print("ERROR: tinker package not installed. Run: uv pip install tinker")
        sys.exit(1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K test split
    # LIMITATION: "The Early-Training Snapshot Problem" and "API cost constraints" by evaluating 
    # on the full test set instead of small subsets once budget allows.
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)
    problems = ds.select(range(min(n_problems, len(ds))))

    # Create Tinker client
    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=model_name)

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

    results: List[Dict[str, Any]] = []
    zvf_per_problem = []

    for idx, item in enumerate(problems):
        question = item["question"]
        gold = item["answer"].split("#")[-1].strip().replace(",", "")

        # Build prompt
        prompt_text = question + question_suffix
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        model_input = ModelInput.from_ints(prompt_tokens)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=[tokenizer.eos_token_id] if tokenizer.eos_token_id else [],
        )

        # Sample group_size rollouts (Tinker API may return coroutine)
        response = await sc.sample_async(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=group_size,
        )

        # Score each rollout
        rewards = []
        completions = []
        for seq in response.sequences:
            completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
            pred = extract_boxed_answer(completion_text)
            reward = score_answer(pred, gold)
            rewards.append(reward)
            completions.append(completion_text[:200])  # truncated

        # Compute ZVF for this problem
        # TODO: ZVF is fragile outside math domains and acts as a symptom rather than root cause.
        # Consider implementing ERF (Effective-Rollout Fraction) or monitoring policy entropy.
        group_var = np.var(rewards) if len(rewards) > 1 else 0.0
        zvf = 1.0 if group_var < 1e-6 else 0.0
        zvf_per_problem.append(zvf)

        results.append({
            "problem_id": idx,
            "question": question[:100],
            "gold_answer": gold,
            "rewards": rewards,
            "mean_reward": float(np.mean(rewards)),
            "zvf": zvf,
            "completions_truncated": completions,
        })

        if (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{n_problems} problems...")

    overall_acc = np.mean([r["mean_reward"] for r in results])
    overall_zvf = np.mean(zvf_per_problem)

    # LIMITATION: "Failure to Prove Generalization" by implementing statistical significance 
    # testing (e.g., p-values) against base model performance to rigorously prove reasoning uplift.

    summary = {
        "model": model_name,
        "n_problems": n_problems,
        "group_size": group_size,
        "temperature": temperature,
        "seed": seed,
        "overall_accuracy": float(overall_acc),
        "overall_zvf": float(overall_zvf),
        "per_problem": results,
    }

    return summary


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Direct Tinker API GSM8K evaluation")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="Model name")
    p.add_argument("--n-problems", type=int, default=50, help="Number of GSM8K problems")
    p.add_argument("--group-size", type=int, default=8, help="Rollout group size")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    # LIMITATION: "Single-Seed Extrapolations" by supporting multiple seeds and computing 
    # confidence intervals, as RL dynamics are notoriously high-variance and initialization-dependent.
    p.add_argument("--out", default=str(OUT_JSON))
    args = p.parse_args(argv)

    if "TINKER_API_KEY" not in os.environ:
        print("ERROR: TINKER_API_KEY not set")
        return 1

    print(f"Evaluating {args.model} on GSM8K ({args.n_problems} problems, G={args.group_size})")
    summary = asyncio.run(evaluate_with_tinker(
        model_name=args.model,
        n_problems=args.n_problems,
        group_size=args.group_size,
        temperature=args.temperature,
        seed=args.seed,
    ))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {args.out}")
    print(f"  Overall accuracy: {summary['overall_accuracy']:.3f}")
    print(f"  Overall ZVF: {summary['overall_zvf']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
