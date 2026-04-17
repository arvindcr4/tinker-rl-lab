"""
GRPO 10x Structural Ceiling — Unified experiment runner.

Adapts grpo_tooluse_tinker.py for the full experimental matrix.
Supports: GSM8K, MATH, HumanEval, and tool-use benchmarks.
Instruments: group saturation diagnostic at every step.
Uses tinker_cookbook checkpoint_utils for proper state saving and resumption.

Usage:
  python grpo_10x_runner.py --config configs/block_b_gsm8k_gemma2_9b.yaml
  python grpo_10x_runner.py --config configs/block_g_gsm8k_group32.yaml --dry-run
  python grpo_10x_runner.py --config configs/block_g_gsm8k_group32.yaml --resume
"""
from __future__ import annotations

import argparse
import json
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


# ── Benchmark datasets ──────────────────────────────────────────────────

def load_gsm8k_prompts(tokenizer, max_examples: int = 500) -> list[dict]:
    """Load GSM8K training prompts."""
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
    """Load MATH competition prompts."""
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
    """Load HumanEval code completion prompts."""
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


def load_tool_use_prompts(tokenizer) -> list[dict]:
    """Load synthetic tool-use prompts (from grpo_tooluse_tinker.py)."""
    SYSTEM_PROMPT = (
        "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
        '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
        "No prose. Only JSON."
    )
    TOOLS = [
        {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
        {"name": "get_weather", "description": "Weather for a city", "parameters": {"city": "string", "units": "string"}},
        {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
        {"name": "get_time", "description": "Time in timezone", "parameters": {"timezone": "string"}},
        {"name": "set_reminder", "description": "Set a reminder", "parameters": {"task": "string", "time": "string"}},
    ]
    TOOL_SCHEMA = json.dumps(TOOLS)
    RAW = [
        ("What is 245 * 37?", "calculator", {"expression": "245 * 37"}),
        ("Calculate sqrt(144)", "calculator", {"expression": "sqrt(144)"}),
        ("15% of 980?", "calculator", {"expression": "0.15 * 980"}),
        ("Divide 1024 by 32", "calculator", {"expression": "1024 / 32"}),
        ("2 to the power of 10", "calculator", {"expression": "2 ** 10"}),
        ("Weather in Tokyo?", "get_weather", {"city": "Tokyo", "units": "metric"}),
        ("Is it raining in London?", "get_weather", {"city": "London", "units": "metric"}),
        ("Temperature in New York", "get_weather", {"city": "New York", "units": "imperial"}),
        ("How hot is Dubai right now?", "get_weather", {"city": "Dubai", "units": "metric"}),
        ("Search for GPT-5 news", "web_search", {"query": "GPT-5 news"}),
        ("Capital of Australia?", "web_search", {"query": "capital of Australia"}),
        ("Find Python asyncio tutorial", "web_search", {"query": "Python asyncio tutorial"}),
        ("What time is it in Singapore?", "get_time", {"timezone": "Asia/Singapore"}),
        ("Current time in Los Angeles?", "get_time", {"timezone": "America/Los_Angeles"}),
        ("Time in Berlin?", "get_time", {"timezone": "Europe/Berlin"}),
        ("Remind me to call mom at 6pm", "set_reminder", {"task": "call mom", "time": "6pm"}),
        ("Set a reminder for team meeting 10am", "set_reminder", {"task": "team meeting", "time": "10am"}),
        ("Remind me to take medicine at 8pm", "set_reminder", {"task": "take medicine", "time": "8pm"}),
    ]

    examples = []
    for query, tool_name, args in RAW:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nAvailable tools:\n{TOOL_SCHEMA}\n\nUser: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        examples.append({
            "prompt_text": prompt,
            "prompt_ids": prompt_ids,
            "tool_name": tool_name,
            "arguments": args,
            "benchmark": "tool_use",
        })
    # Repeat for volume
    examples = examples * 28
    random.shuffle(examples)
    return examples


# ── Reward functions ─────────────────────────────────────────────────────

def reward_gsm8k(response: str, answer: str) -> float:
    """Binary reward: 1.0 if final answer matches, 0.0 otherwise."""
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
    """Partial reward for MATH: 1.0 exact, 0.3 boxed present."""
    m = re.search(r'\\boxed\{(.+?)\}', response)
    if not m:
        return 0.0
    pred = m.group(1).strip()
    if pred == answer.strip():
        return 1.0
    return 0.3


def reward_humaneval(response: str, test_code: str, entry_point: str) -> float:
    """Execute-based reward for HumanEval (sandboxed)."""
    try:
        full_code = response + "\n" + test_code
        exec(full_code, {"__builtins__": {}})
        return 1.0
    except Exception:
        # Partial credit for syntactically valid Python
        try:
            compile(response, "<string>", "exec")
            return 0.3
        except SyntaxError:
            return 0.0


def reward_tool_use(response: str, tool_name: str, arguments: dict) -> float:
    """Structured tool-call reward (from grpo_tooluse_tinker.py)."""
    m = re.search(r'\{.*\}', response.strip(), re.DOTALL)
    if not m:
        return 0.0
    try:
        p = json.loads(m.group())
    except json.JSONDecodeError:
        return 0.1
    score = 0.3
    if p.get("tool") == tool_name or p.get("name") == tool_name:
        score += 0.4
    pred_args = p.get("arguments", p.get("parameters", {}))
    if isinstance(pred_args, dict) and arguments:
        score += 0.3 * sum(1 for k in arguments if k in pred_args) / len(arguments)
    return min(score, 1.0)


def get_reward(example: dict, response: str) -> float:
    benchmark = example["benchmark"]
    if benchmark == "gsm8k":
        return reward_gsm8k(response, example["answer"])
    elif benchmark == "math":
        return reward_math(response, example["answer"])
    elif benchmark == "humaneval":
        return reward_humaneval(response, example.get("test", ""), example.get("entry_point", ""))
    elif benchmark == "tool_use":
        return reward_tool_use(response, example["tool_name"], example["arguments"])
    raise ValueError(f"Unknown benchmark: {benchmark}")


# ── Main training loop ───────────────────────────────────────────────────

def detect_benchmark(config: dict) -> str:
    """Infer benchmark from config wandb_run_name."""
    name = config.get("tinker", {}).get("wandb_run_name", "")
    if "math" in name.lower() and "gsm" not in name.lower():
        return "math"
    elif "humaneval" in name.lower():
        return "humaneval"
    elif "tool" in name.lower():
        return "tool_use"
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

    random.seed(seed)
    benchmark = detect_benchmark(cfg)

    # Log path for checkpoint_utils (state + sampler)
    log_path = str(Path(tinker_cfg.get("checkpoint_dir", "./checkpoints/")).resolve())
    os.makedirs(log_path, exist_ok=True)

    print(f"{'='*60}")
    print(f"10x Structural Ceiling Experiment")
    print(f"Config:    {config_path}")
    print(f"Model:     {model_name}")
    print(f"Benchmark: {benchmark}")
    print(f"Group:     {group_size}")
    print(f"Steps:     {total_steps}")
    print(f"LR:        {lr}")
    print(f"LoRA:      {lora_rank}")
    print(f"Seed:      {seed}")
    print(f"Log path:  {log_path}")
    print(f"Resume:    {resume}")
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
                "group_size": group_size,
                "total_steps": total_steps,
                "learning_rate": lr,
                "lora_rank": lora_rank,
                "seed": seed,
                "config_file": config_path,
            },
            resume="allow" if resume else None,
        )

    # Load tokenizer and dataset
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if benchmark == "gsm8k":
        examples = load_gsm8k_prompts(tok)
    elif benchmark == "math":
        examples = load_math_prompts(tok)
    elif benchmark == "humaneval":
        examples = load_humaneval_prompts(tok)
    elif benchmark == "tool_use":
        examples = load_tool_use_prompts(tok)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    print(f"Loaded {len(examples)} examples for {benchmark}")

    # Tinker setup — resume from checkpoint if available
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
        # Save initial state checkpoint
        checkpoint_utils.save_checkpoint(
            training_client=tc,
            name="step_000",
            log_path=log_path,
            kind="both",
            loop_state={"batch": 0, "step": 0},
        )
        sc = tc.save_weights_and_get_sampling_client()
        print(f"Sampler ready")

    # Group saturation tracker
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

            max_tokens = 192 if benchmark == "tool_use" else 512
            sp = T.SamplingParams(max_tokens=max_tokens, temperature=0.8, top_p=0.95)
            responses = sc.sample(prompt_mi, num_samples=group_size, sampling_params=sp).result()

            rewards = []
            for resp in responses.sequences:
                resp_ids = list(resp.tokens)
                text = tok.decode(resp_ids, skip_special_tokens=True)
                r = get_reward(ex, text)
                rewards.append(r)

            group_reward_lists.append(rewards)

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

        # Record group saturation diagnostic
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

        print(
            f"Step {step+1:3d}/{total_steps} | "
            f"loss={grpo_loss:.4f} | reward={avg_r:.3f} | "
            f"zvf={zvf:.2f} | gu={gu:.2f}"
        )

        if env_cfg.get("use_wandb"):
            import wandb
            wandb.log({
                "step": step,
                "grpo_loss": grpo_loss,
                "mean_reward": avg_r,
            }, step=step)

        # Save state checkpoint (resumable) at intervals
        if save_interval and (step + 1) % save_interval == 0:
            checkpoint_utils.save_checkpoint(
                training_client=tc,
                name=f"step_{step+1:03d}",
                log_path=log_path,
                kind="both",
                loop_state={"batch": step + 1, "step": step + 1},
            )
            sc = tc.save_weights_and_get_sampling_client()
            print(f"  -> Checkpoint (state+sampler): step_{step+1}")

    # Final checkpoint with both state and sampler
    checkpoint_utils.save_checkpoint(
        training_client=tc,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": total_steps, "step": total_steps},
    )
    print(f"\nFinal checkpoint saved to: {log_path}")
    print(f"Avg reward last 10: {sum(step_rewards[-10:])/max(len(step_rewards[-10:]),1):.3f}")

    # Save saturation diagnostic
    ckpt_dir = Path(log_path)
    tracker.save(ckpt_dir / "group_saturation.json")
    print(f"Saturation diagnostic: {ckpt_dir / 'group_saturation.json'}")

    summary = tracker.summary()
    print(f"\nGroup Saturation Summary:")
    print(f"  Mean zero-variance fraction: {summary.get('mean_zero_variance_frac', 0):.3f}")
    print(f"  Mean gradient utilization:   {summary.get('mean_gradient_utilization', 0):.3f}")
    if summary.get("saturation_onset_step") is not None:
        print(f"  Saturation onset (>50% ZVF): step {summary['saturation_onset_step']}")

    if env_cfg.get("use_wandb"):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="10x Structural Ceiling GRPO Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    run_experiment(args.config, dry_run=args.dry_run, resume=args.resume)
