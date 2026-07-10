"""N8 — Per-prompt pass-rate spectrum (K rollouts/prompt) -> analytic ZVF(G).

Sampling-only Tinker experiment: K rollouts per GSM8K test prompt from the
untouched Qwen/Qwen3-8B base checkpoint, per-prompt pass rate p_i, then
analytic ZVF(G) = E_i[p_i^G + (1-p_i)^G] for G in {2,4,8,16,32,64} compared
against the empirical -0.230/decade slope. Doubles as N5's baseline-offset c
measurement for the 8B anchor. Budget: n_prompts*k <= 5000 completions (hard
guard). Launched 2026-07-04.

Usage:
    python3 n8_passrate_spectrum_20260704.py --smoke        # 2 prompts x 4
    python3 n8_passrate_spectrum_20260704.py                # 78 prompts x 64
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer

import wandb
import tinker
import tinker.types as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

EMPIRICAL_SLOPE = -0.230

def get_args():
    parser = argparse.ArgumentParser(description="N8 Passrate Spectrum")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--n-prompts", type=int, default=78)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out-dir", type=str, default="/home/claude/tinker-rl-lab/experiments/results/n8_passrate_spectrum")
    parser.add_argument("--wandb-project", type=str, default="tinker-new-research")
    parser.add_argument("--run-name", type=str, default="n8-passrate-spectrum-qwen3-8b-k64-20260704")
    parser.add_argument("--smoke", action="store_true")
    
    args = parser.parse_args()
    if args.smoke:
        args.n_prompts = 2
        args.k = 4
        args.chunk_size = 4
        args.run_name += "-smoke"
        
    return args

def extract_and_score(completion: str, gold: str) -> float:
    matches = re.findall(r'\\boxed\{([^}]+)\}', completion)
    cand_str = None
    if matches:
        cand_str = matches[-1]
    else:
        num_matches = re.findall(r'[-+]?\d[\d,]*\.?\d*', completion)
        if num_matches:
            cand_str = num_matches[-1]
            
    if cand_str is None:
        return 0.0
        
    cand_str = cand_str.strip().replace(',', '').replace('$', '').replace(' ', '')
    try:
        cand_val = float(cand_str)
        gold_val = float(gold)
        return 1.0 if abs(cand_val - gold_val) < 0.01 else 0.0
    except ValueError:
        return 1.0 if cand_str.lower() == gold.lower() else 0.0

async def process_prompt(
    prompt_id: int, 
    question: str, 
    gold: str, 
    args, 
    tokenizer, 
    sc, 
    sp: T.SamplingParams, 
    sem: asyncio.Semaphore, 
    file_lock: asyncio.Lock, 
    passrates_file: Path,
    state: dict
):
    prompt_text = question + " Provide a numerical answer without units, written inside \\boxed{}."
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    model_input = T.ModelInput.from_ints(prompt_tokens)
    
    k_requested = args.k
    chunks = []
    k_remaining = k_requested
    while k_remaining > 0:
        c = min(k_remaining, args.chunk_size)
        chunks.append(c)
        k_remaining -= c
        
    async with sem:
        rewards = []
        total_completion_tokens = 0
        sampling_errors = 0
        
        for c in chunks:
            success = False
            backoff_times = [2, 4, 8]
            for attempt in range(4):
                try:
                    resp = await sc.sample_async(prompt=model_input, num_samples=c, sampling_params=sp)
                    for seq in resp.sequences:
                        text = tokenizer.decode(list(seq.tokens), skip_special_tokens=True)
                        total_completion_tokens += len(seq.tokens)
                        score = extract_and_score(text, gold)
                        rewards.append(int(score))
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Error sampling prompt {prompt_id} (attempt {attempt+1}): {e}")
                    if attempt < len(backoff_times):
                        await asyncio.sleep(backoff_times[attempt])
                    else:
                        sampling_errors += 1
            if not success:
                logger.error(f"Failed to get {c} samples for prompt {prompt_id} after 4 attempts.")
                
        k_obtained = len(rewards)
        if k_obtained == 0:
            logger.error(f"Prompt {prompt_id} got 0 samples.")
            return

        n_correct = sum(rewards)
        p_hat = n_correct / k_obtained
        mean_completion_tokens = total_completion_tokens / k_obtained
        
        result_dict = {
            "prompt_id": prompt_id,
            "question_prefix": question[:120],
            "gold": gold,
            "k_requested": k_requested,
            "k_obtained": k_obtained,
            "n_correct": n_correct,
            "p_hat": p_hat,
            "rewards": rewards,
            "mean_completion_tokens": mean_completion_tokens,
            "sampling_errors": sampling_errors
        }
        
        async with file_lock:
            with open(passrates_file, "a") as f:
                f.write(json.dumps(result_dict) + "\n")
                
            state["prompts_done"] += 1
            state["completions_done"] += k_obtained
            state["sum_p"] += p_hat
            state["p_hats"].append(p_hat)
            running_mean_p = state["sum_p"] / state["prompts_done"]
            
            logger.info(f"[N8] prompt {prompt_id}/{args.n_prompts} done p_hat={p_hat:.2f} (total completions so far: {state['completions_done']})")
            wandb.log({
                "prompts_done": state["prompts_done"],
                "running_mean_p": running_mean_p,
                "completions_done": state["completions_done"]
            })

async def run_all(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    passrates_fname = "passrates_smoke.jsonl" if args.smoke else "passrates.jsonl"
    analysis_fname = "analysis_smoke.json" if args.smoke else "analysis.json"
    passrates_file = out_dir / passrates_fname
    
    done_ids = set()
    state = {
        "prompts_done": 0,
        "completions_done": 0,
        "sum_p": 0.0,
        "p_hats": []
    }
    
    if passrates_file.exists():
        with open(passrates_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                done_ids.add(data["prompt_id"])
                state["prompts_done"] += 1
                state["completions_done"] += data["k_obtained"]
                state["sum_p"] += data["p_hat"]
                state["p_hats"].append(data["p_hat"])
                
    logger.info(f"Resuming with {len(done_ids)} prompts already done.")
    
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(base_model=args.model)
    sp = T.SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
    
    sem = asyncio.Semaphore(args.concurrency)
    file_lock = asyncio.Lock()
    
    ds = ds.shuffle(seed=args.seed)
    if len(ds) > args.n_prompts:
        ds = ds.select(range(args.n_prompts))
        
    tasks = []
    for idx, item in enumerate(ds):
        if idx in done_ids:
            continue
            
        gold = item['answer'].split('####')[-1].strip().replace(',', '')
        question = item['question']
        tasks.append(process_prompt(idx, question, gold, args, tokenizer, sc, sp, sem, file_lock, passrates_file, state))
        
    if tasks:
        await asyncio.gather(*tasks)
        
    if len(state["p_hats"]) == 0:
        logger.warning("No prompts processed, skipping analysis.")
        return
        
    p = np.array(state["p_hats"])
    c = float(p.mean())
    
    Gs = [2, 4, 8, 16, 32, 64]
    zvf_pred = {}
    retention_pred = {}
    for G in Gs:
        zvf_val = float(np.mean(p**G + (1-p)**G))
        zvf_pred[G] = zvf_val
        retention_pred[G] = 1.0 - zvf_val
        
    logG = np.log10(Gs)
    zvf_vals_array = np.array([zvf_pred[G] for G in Gs])
    if len(Gs) > 1:
        slope, intercept = np.polyfit(logG, zvf_vals_array, 1)
    else:
        slope = 0.0
        
    slope_gap = slope - EMPIRICAL_SLOPE
    
    hist_counts, hist_edges = np.histogram(p, bins=16, range=(0,1))
    
    frac_all_fail = float(np.mean(p == 0))
    frac_all_pass = float(np.mean(p == 1))
    frac_mixed = float(np.mean((p > 0) & (p < 1)))
    
    analysis_data = {
        "model": args.model,
        "n_prompts_requested": args.n_prompts,
        "n_prompts_measured": int(len(p)),
        "total_completions": state["completions_done"],
        "k": args.k,
        "temperature": args.temperature,
        "c_baseline_pass_rate": c,
        "zvf_pred": zvf_pred,
        "retention_pred": retention_pred,
        "slope_per_decade": slope,
        "empirical_slope": EMPIRICAL_SLOPE,
        "slope_gap": slope_gap,
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist()
        },
        "frac_all_fail": frac_all_fail,
        "frac_all_pass": frac_all_pass,
        "frac_mixed": frac_mixed,
        "timestamp": time.time()
    }
    
    with open(out_dir / analysis_fname, "w") as f:
        json.dump(analysis_data, f, indent=2)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(p, bins=16, range=(0,1), color='skyblue', edgecolor='black')
    ax1.set_title("Histogram of p_hat")
    ax1.set_xlabel("p_hat")
    ax1.set_ylabel("Count")
    
    ax2.plot(Gs, zvf_vals_array, marker='o', label='Predicted ZVF(G)')
    
    g2_idx = Gs.index(2)
    zvf_g2 = zvf_vals_array[g2_idx]
    x_line = np.array(Gs)
    y_line = zvf_g2 + EMPIRICAL_SLOPE * (np.log10(x_line) - np.log10(2))
    
    ax2.plot(x_line, y_line, linestyle='--', color='red', label=f'empirical {EMPIRICAL_SLOPE}/decade')
    ax2.set_xscale('log')
    ax2.set_xticks(Gs)
    ax2.set_xticklabels([str(g) for g in Gs])
    
    ax2.set_title("Predicted ZVF(G) vs log10(G)")
    ax2.set_xlabel("G")
    ax2.set_ylabel("ZVF")
    ax2.legend()
    
    plt.tight_layout()
    fig_path = out_dir / "zvf_pred_vs_empirical.png"
    if args.smoke:
        fig_path = out_dir / "zvf_pred_vs_empirical_smoke.png"
    plt.savefig(fig_path)
    plt.close()
    
    summary = {
        "c_baseline_pass_rate": c,
        "slope_per_decade": slope,
        "slope_gap": slope_gap,
        "frac_all_fail": frac_all_fail,
        "frac_all_pass": frac_all_pass,
        "frac_mixed": frac_mixed,
    }
    for G in Gs:
        summary[f"zvf_pred_G{G}"] = zvf_pred[G]
        summary[f"retention_pred_G{G}"] = retention_pred[G]
        
    summary["zvf_plot"] = wandb.Image(str(fig_path))
    
    table = wandb.Table(columns=["G", "zvf_pred", "retention_pred"])
    for G in Gs:
        table.add_data(G, zvf_pred[G], retention_pred[G])
    summary["zvf_table"] = table
    
    wandb.log(summary)
    wandb.finish()


def main():
    args = get_args()
    
    total = args.n_prompts * args.k
    if total > 5000:
        sys.exit(f"Error: total completions {total} exceeds hard guard limit of 5000.")
        
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("Error: TINKER_API_KEY environment variable is not set.")
        
    wandb_config = vars(args).copy()
    wandb_config["total_completions_requested"] = total
    run = wandb.init(project=args.wandb_project, name=args.run_name, config=wandb_config)
    logger.info(f"[N8] wandb run path: {run.entity}/{run.project}/{run.id}")

    asyncio.run(run_all(args))

if __name__ == "__main__":
    main()
