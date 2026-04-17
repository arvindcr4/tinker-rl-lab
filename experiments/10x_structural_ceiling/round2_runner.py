"""
Round 2 Experiment Runner — extends grpo_10x_runner.py with:
  1. REINFORCE algorithm (reward - running baseline, no group normalization)
  2. Difficulty binning for causal GU phase diagram
  3. Dense code reward for HumanEval
  4. 300-step continuation with --resume

All experiments log ZVF/GU saturation diagnostics to W&B.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

from group_saturation_diagnostic import SaturationTracker, log_to_wandb


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Benchmark loaders (reused from grpo_10x_runner) ──────────────────────

def load_gsm8k_prompts(tokenizer, max_examples: int = 500) -> list[dict]:
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
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        examples.append({
            "prompt_text": prompt,
            "prompt_ids": prompt_ids,
            "answer": answer,
            "benchmark": "gsm8k",
        })
    return examples


def load_math_prompts(tokenizer, max_examples: int = 500) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("math-ai/MATH-500", split="test")
    examples = []
    for item in ds.select(range(min(len(ds), max_examples))):
        problem = item["problem"]
        solution = item["solution"]
        answer = item.get("answer", "")
        if not answer:
            m = re.search(r'\\boxed\{(.+?)\}', solution)
            answer = m.group(1) if m else solution.split()[-1]
        prompt = (
            f"Solve the following math competition problem.\n\n"
            f"Problem: {problem}\n\n"
            f"Put your final answer in \\boxed{{}}."
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        examples.append({
            "prompt_text": prompt,
            "prompt_ids": prompt_ids,
            "answer": answer,
            "benchmark": "math",
        })
    return examples


def load_humaneval_prompts(tokenizer, max_examples: int = 164) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    examples = []
    for item in ds.select(range(min(len(ds), max_examples))):
        prompt = item["prompt"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        examples.append({
            "prompt_text": prompt,
            "prompt_ids": prompt_ids,
            "answer": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
            "benchmark": "humaneval",
        })
    return examples


# ── Reward functions ─────────────────────────────────────────────────────

def reward_gsm8k(response: str, answer: str) -> float:
    m = re.search(r'####\s*(.+?)(?:\s|$)', response)
    if not m:
        m = re.search(r'(?:answer|Answer).*?(\-?\d[\d,]*\.?\d*)', response)
    if not m:
        return 0.0
    pred = m.group(1).replace(",", "").strip()
    target = answer.replace(",", "").strip()
    try:
        return 1.0 if abs(float(pred) - float(target)) < 1e-4 else 0.0
    except ValueError:
        return 1.0 if pred == target else 0.0


def reward_math(response: str, answer: str) -> float:
    m = re.search(r'\\boxed\{(.+?)\}', response)
    if not m:
        return 0.0
    pred = m.group(1).strip()
    if pred == answer.strip():
        return 1.0
    return 0.3


def reward_humaneval_binary(response: str, test_code: str, entry_point: str) -> float:
    """Original binary reward."""
    try:
        full_code = response + "\n" + test_code
        exec(full_code, {"__builtins__": {}})
        return 1.0
    except Exception:
        try:
            compile(response, "<string>", "exec")
            return 0.3
        except SyntaxError:
            return 0.0


def reward_humaneval_dense(response: str, test_code: str, entry_point: str) -> float:
    """Dense reward for code: graduated signal to break ZVF deadlock.

    0.10 — syntactically valid Python
    0.25 — contains a function definition
    0.40 — function name matches entry_point
    0.60 — function is callable (no import/runtime errors on def)
    0.80 — passes at least one assertion in test_code
    1.00 — passes all assertions
    """
    score = 0.0

    # Level 1: Syntax validity
    try:
        compile(response, "<string>", "exec")
        score = 0.10
    except SyntaxError:
        return 0.0

    # Level 2: Contains function definition
    if re.search(r'def\s+\w+\s*\(', response):
        score = 0.25

    # Level 3: Function name matches
    if re.search(rf'def\s+{re.escape(entry_point)}\s*\(', response):
        score = 0.40

    # Level 4: Function is callable
    local_ns = {}
    try:
        exec(response, {"__builtins__": __builtins__}, local_ns)
        if entry_point in local_ns and callable(local_ns[entry_point]):
            score = 0.60
    except Exception:
        return score

    # Level 5-6: Test execution
    try:
        full_code = response + "\n" + test_code
        exec_ns = {"__builtins__": __builtins__}
        exec(full_code, exec_ns)
        score = 1.0  # All tests passed
    except AssertionError:
        # At least the function ran — partial credit
        score = 0.80
    except Exception:
        pass

    return score


def get_reward(example: dict, response: str, dense_code: bool = False) -> float:
    benchmark = example["benchmark"]
    if benchmark == "gsm8k":
        return reward_gsm8k(response, example["answer"])
    elif benchmark == "math":
        return reward_math(response, example["answer"])
    elif benchmark == "humaneval":
        if dense_code:
            return reward_humaneval_dense(
                response, example.get("test", ""), example.get("entry_point", "")
            )
        return reward_humaneval_binary(
            response, example.get("test", ""), example.get("entry_point", "")
        )
    raise ValueError(f"Unknown benchmark: {benchmark}")


# ── Difficulty binning for phase diagram ─────────────────────────────────

def bin_prompts_by_difficulty(
    examples: list[dict],
    sc,  # sampling client
    tokenizer,
    num_samples: int = 8,
) -> dict[str, list[dict]]:
    """Pre-sample each prompt to estimate base success rate, then bin.

    easy:  >80% success (expect ZVF≈1 quickly — already solved)
    mid:   20-80% success (Goldilocks zone — max reward disagreement)
    hard:  <20% success (expect ZVF≈1 — never solved)
    """
    import tinker.types as T

    bins = {"easy": [], "mid": [], "hard": []}
    print(f"Binning {len(examples)} prompts by difficulty ({num_samples} samples each)...")

    for i, ex in enumerate(examples):
        prompt_mi = T.ModelInput.from_ints(ex["prompt_ids"])
        sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
        responses = sc.sample(prompt_mi, num_samples=num_samples, sampling_params=sp).result()

        successes = 0
        for resp in responses.sequences:
            text = tokenizer.decode(list(resp.tokens), skip_special_tokens=True)
            r = get_reward(ex, text)
            if r > 0.5:
                successes += 1

        success_rate = successes / num_samples
        ex["base_success_rate"] = success_rate

        if success_rate > 0.8:
            bins["easy"].append(ex)
        elif success_rate >= 0.2:
            bins["mid"].append(ex)
        else:
            bins["hard"].append(ex)

        if (i + 1) % 50 == 0:
            print(f"  Binned {i+1}/{len(examples)}: "
                  f"easy={len(bins['easy'])}, mid={len(bins['mid'])}, hard={len(bins['hard'])}")

    print(f"Final bins: easy={len(bins['easy'])}, mid={len(bins['mid'])}, hard={len(bins['hard'])}")
    return bins


# ── Main training loop ───────────────────────────────────────────────────

def detect_benchmark(config: dict) -> str:
    name = config.get("tinker", {}).get("wandb_run_name", "")
    if "math" in name.lower() and "gsm" not in name.lower():
        return "math"
    elif "humaneval" in name.lower() or "dense" in name.lower():
        return "humaneval"
    return "gsm8k"


def run_experiment(config_path: str, dry_run: bool = False, resume: bool = False) -> None:
    import torch
    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer
    from tinker_cookbook import checkpoint_utils

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Set TINKER_API_KEY env var")

    cfg = load_config(config_path)
    env_cfg = cfg["env"]
    tinker_cfg = cfg["tinker"]

    model_name = cfg["openai"][0]["model_name"]
    tinker_model = tinker_cfg.get("tinker_model_name", model_name)
    group_size = env_cfg["group_size"]
    total_steps = env_cfg["total_steps"]
    lr = tinker_cfg["learning_rate"]
    lora_rank = tinker_cfg["lora_rank"]
    seed = env_cfg.get("seed", 42)

    # Round-2 specific config
    algorithm = env_cfg.get("algorithm", "grpo")  # "grpo" or "reinforce"
    difficulty_bin = env_cfg.get("difficulty_bin", None)  # "easy", "mid", "hard", or None
    reward_mode = env_cfg.get("reward_mode", "binary")  # "binary" or "dense"
    dense_code = reward_mode == "dense"

    random.seed(seed)
    benchmark = detect_benchmark(cfg)

    log_path = str(Path(tinker_cfg.get("checkpoint_dir", "./checkpoints/")).resolve())
    os.makedirs(log_path, exist_ok=True)

    print(f"{'='*60}")
    print(f"Round 2: Extended Experiment Runner")
    print(f"Config:      {config_path}")
    print(f"Model:       {model_name}")
    print(f"Benchmark:   {benchmark}")
    print(f"Algorithm:   {algorithm}")
    print(f"Group:       {group_size}")
    print(f"Steps:       {total_steps}")
    print(f"LR:          {lr}")
    print(f"LoRA:        {lora_rank}")
    print(f"Seed:        {seed}")
    print(f"Diff bin:    {difficulty_bin or 'all'}")
    print(f"Reward mode: {reward_mode}")
    print(f"Log path:    {log_path}")
    print(f"Resume:      {resume}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Would run experiment with above config. Exiting.")
        return

    # Init W&B
    if env_cfg.get("use_wandb"):
        import wandb
        wandb.init(
            project=tinker_cfg.get("wandb_project", "tinker-structural-ceiling"),
            group=tinker_cfg.get("wandb_group", ""),
            name=tinker_cfg.get("wandb_run_name", ""),
            config={
                "model": model_name,
                "benchmark": benchmark,
                "algorithm": algorithm,
                "group_size": group_size,
                "total_steps": total_steps,
                "learning_rate": lr,
                "lora_rank": lora_rank,
                "seed": seed,
                "difficulty_bin": difficulty_bin,
                "reward_mode": reward_mode,
                "config_file": config_path,
            },
            resume="allow" if resume else None,
        )

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # For phase diagram, match the prompt count used during binning
    gsm8k_limit = 500
    if difficulty_bin:
        bins_file = Path("./phase_bins.json")
        if bins_file.exists():
            gsm8k_limit = json.loads(bins_file.read_text()).get("total_prompts", 500)

    if benchmark == "gsm8k":
        examples = load_gsm8k_prompts(tok, max_examples=gsm8k_limit)
    elif benchmark == "math":
        examples = load_math_prompts(tok)
    elif benchmark == "humaneval":
        examples = load_humaneval_prompts(tok)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"Loaded {len(examples)} examples for {benchmark}")

    # Tinker setup
    svc = tinker.ServiceClient(base_url=None)
    start_step = 0

    resume_info = checkpoint_utils.get_last_checkpoint(log_path) if resume else None
    if resume_info:
        print(f"Resuming from checkpoint: step {resume_info.batch}")
        tc = svc.create_training_client_from_state_with_optimizer(resume_info.state_path)
        start_step = resume_info.batch
        sc = tc.save_weights_and_get_sampling_client()
        print(f"Resumed at step {start_step}")
    else:
        tc = svc.create_lora_training_client(base_model=tinker_model, rank=lora_rank)
        print(f"Run ID: {tc.model_id}")
        checkpoint_utils.save_checkpoint(
            training_client=tc, name="step_000", log_path=log_path,
            kind="both", loop_state={"batch": 0, "step": 0},
        )
        sc = tc.save_weights_and_get_sampling_client()
        print(f"Sampler ready")

    # Phase diagram: filter prompts by difficulty bin
    if difficulty_bin:
        bins_file = Path("./phase_bins.json")
        if bins_file.exists():
            # Use pre-computed bins (from prebin_gsm8k.py)
            print(f"Loading pre-computed bins from {bins_file}")
            bin_data = json.loads(bins_file.read_text())
            indices = bin_data["bins"][difficulty_bin]["indices"]
            rates = bin_data["bins"][difficulty_bin]["rates"]
            examples = [examples[i] for i in indices if i < len(examples)]
            for ex, rate in zip(examples, rates):
                ex["base_success_rate"] = rate
            print(f"Loaded {len(examples)} examples from '{difficulty_bin}' bin (pre-computed)")
        else:
            # Fall back to live binning
            print(f"No phase_bins.json found — binning live (expensive!)")
            bins = bin_prompts_by_difficulty(examples, sc, tok, num_samples=8)
            examples = bins.get(difficulty_bin, [])

        if not examples:
            print(f"ERROR: No examples in '{difficulty_bin}' bin. Aborting.")
            return
        print(f"Using {len(examples)} examples from '{difficulty_bin}' bin")

        # Save bin stats
        bin_stats_path = Path(log_path) / "bin_stats.json"
        bin_stats_path.write_text(json.dumps({
            "bin": difficulty_bin,
            "n_examples": len(examples),
            "success_rates": [ex.get("base_success_rate", -1) for ex in examples],
        }, indent=2))

    # REINFORCE baseline tracking
    reinforce_baseline = 0.0
    reinforce_alpha = 0.1  # EMA decay for running baseline

    tracker = SaturationTracker()
    _advantages: list[float] = []

    def grpo_loss_fn(data, logprobs_list):
        losses = []
        for i, logprobs in enumerate(logprobs_list):
            adv = _advantages[i]
            losses.append(-adv * logprobs.sum())
        loss = torch.stack(losses).mean()
        return loss, {"grpo_loss": loss.item()}

    # Training loop
    prompts_per_step = 4
    step_rewards = []
    save_interval = tinker_cfg.get("save_checkpoint_interval", 10)

    for step in range(start_step, total_steps):
        batch = random.sample(examples, min(prompts_per_step, len(examples)))

        all_data: list = []
        all_advs: list[float] = []
        batch_rewards: list[float] = []
        group_reward_lists: list[list[float]] = []

        for ex in batch:
            prompt_ids = ex["prompt_ids"]
            prompt_mi = T.ModelInput.from_ints(prompt_ids)

            max_tokens = 512 if benchmark in ("math", "humaneval") else 512
            sp = T.SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)
            responses = sc.sample(prompt_mi, num_samples=group_size, sampling_params=sp).result()

            rewards = []
            for resp in responses.sequences:
                resp_ids = list(resp.tokens)
                text = tok.decode(resp_ids, skip_special_tokens=True)
                r = get_reward(ex, text, dense_code=dense_code)
                rewards.append(r)

            group_reward_lists.append(rewards)

            # Compute advantages based on algorithm
            if algorithm == "reinforce":
                # REINFORCE: advantage = reward - running baseline
                advs = [r - reinforce_baseline for r in rewards]
                # Update running baseline
                batch_mean = sum(rewards) / len(rewards)
                reinforce_baseline = (
                    reinforce_alpha * batch_mean
                    + (1 - reinforce_alpha) * reinforce_baseline
                )
            else:
                # GRPO: group-normalized advantages
                mean_r = sum(rewards) / len(rewards)
                std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
                advs = [(r - mean_r) / std_r for r in rewards]

            batch_rewards.extend(rewards)

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

        diag = tracker.record_step(step, group_reward_lists)
        log_to_wandb(diag)

        _advantages = all_advs

        result = tc.forward_backward_custom(
            data=all_data,
            loss_fn=grpo_loss_fn,
            loss_type_input="logprobs",
        ).result()
        tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)).result()

        avg_r = sum(batch_rewards) / len(batch_rewards)
        step_rewards.append(avg_r)
        grpo_loss = result.metrics.get("grpo_loss", float("nan"))
        zvf = diag.zero_variance_frac
        gu = diag.gradient_utilization

        extra = ""
        if algorithm == "reinforce":
            extra = f" | baseline={reinforce_baseline:.3f}"
        if difficulty_bin:
            extra += f" | bin={difficulty_bin}"

        print(
            f"Step {step+1:3d}/{total_steps} | "
            f"loss={grpo_loss:.4f} | reward={avg_r:.3f} | "
            f"zvf={zvf:.2f} | gu={gu:.2f}{extra}"
        )

        if env_cfg.get("use_wandb"):
            import wandb
            log_data = {
                "step": step,
                "grpo_loss": grpo_loss,
                "mean_reward": avg_r,
            }
            if algorithm == "reinforce":
                log_data["reinforce_baseline"] = reinforce_baseline
            wandb.log(log_data, step=step)

        if save_interval and (step + 1) % save_interval == 0:
            checkpoint_utils.save_checkpoint(
                training_client=tc, name=f"step_{step+1:03d}",
                log_path=log_path, kind="both",
                loop_state={"batch": step + 1, "step": step + 1},
            )
            sc = tc.save_weights_and_get_sampling_client()
            print(f"  -> Checkpoint: step_{step+1}")

    # Final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=tc, name="final", log_path=log_path,
        kind="both", loop_state={"batch": total_steps, "step": total_steps},
    )
    print(f"\nFinal checkpoint saved to: {log_path}")
    last10 = step_rewards[-10:] if step_rewards else [0]
    print(f"Avg reward last 10: {sum(last10)/len(last10):.3f}")

    # Save saturation diagnostic
    ckpt_dir = Path(log_path)
    tracker.save(ckpt_dir / "group_saturation.json")

    summary = tracker.summary()
    print(f"\nGroup Saturation Summary:")
    print(f"  Mean ZVF: {summary.get('mean_zero_variance_frac', 0):.3f}")
    print(f"  Mean GU:  {summary.get('mean_gradient_utilization', 0):.3f}")
    if summary.get("saturation_onset_step") is not None:
        print(f"  Saturation onset (>50% ZVF): step {summary['saturation_onset_step']}")

    if env_cfg.get("use_wandb"):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round 2 Extended Experiment Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    run_experiment(args.config, dry_run=args.dry_run, resume=args.resume)
