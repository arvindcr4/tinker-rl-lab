#!/usr/bin/env python3
"""
SkyRL + Hosted Tinker Bridge

Runs SkyRL-style GRPO training against the hosted Tinker API.
Combines SkyRL's reward/environment framework with Tinker's GPU training backend.

Usage:
    python run_skyrl_tinker.py --config configs/tinker_hosted.yaml
    python run_skyrl_tinker.py --config configs/tinker_hosted.yaml --model Qwen/Qwen2.5-7B-Instruct
    python run_skyrl_tinker.py --config configs/tinker_hosted.yaml --env gsm8k
"""
import argparse
import importlib
import json
import os
import random
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")


def load_config(config_path: str, overrides: dict) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Apply CLI overrides
    if overrides.get("model"):
        cfg["model"]["name"] = overrides["model"]
        cfg["model"]["tokenizer"] = overrides["model"]
    if overrides.get("steps"):
        cfg["training"]["steps"] = overrides["steps"]
    if overrides.get("group_size"):
        cfg["training"]["group_size"] = overrides["group_size"]
    if overrides.get("lora_rank"):
        cfg["tinker"]["lora_rank"] = overrides["lora_rank"]
    if overrides.get("lr"):
        cfg["tinker"]["learning_rate"] = overrides["lr"]
    return cfg


def load_environment(env_name: str):
    """Load a SkyRL gym environment or built-in task."""
    envs = {
        "gsm8k": GSM8KEnv,
        "math": MathEnv,
        "tool_use": ToolUseEnv,
    }
    if env_name in envs:
        return envs[env_name]()

    # Try loading from skyrl-gym
    try:
        mod = importlib.import_module(f"skyrl_gym.envs.{env_name}")
        return mod.Environment()
    except ImportError:
        pass

    # Try loading from tinker-atropos
    try:
        mod = importlib.import_module(f"tinker_atropos.environments.{env_name}")
        return mod.Environment()
    except ImportError:
        pass

    raise ValueError(f"Unknown environment: {env_name}. Available: gsm8k, math, tool_use")


# ── Built-in environments ─────────────────────────────────────────────────

class GSM8KEnv:
    """GSM8K math word problems."""

    name = "gsm8k"

    def __init__(self):
        self.prompts = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="train")
            for item in ds:
                q = item["question"]
                # Extract numeric answer after ####
                a = item["answer"].split("####")[-1].strip()
                self.prompts.append({"question": q, "answer": a})
            self._loaded = True
            print(f"  GSM8K: {len(self.prompts)} problems loaded")
        except Exception as e:
            print(f"  Warning: could not load GSM8K dataset: {e}")
            print("  Install: pip install datasets")
            sys.exit(1)

    def make_prompt(self, item: dict) -> str:
        return (
            "<|im_start|>system\n"
            "Solve the math problem step by step. "
            "Put your final numeric answer after ####.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def reward(self, response: str, item: dict) -> float:
        import re
        # Extract answer after ####
        match = re.search(r'####\s*([\d,.\-]+)', response)
        if not match:
            # Try last number in response
            nums = re.findall(r'[\d,]+\.?\d*', response)
            if not nums:
                return 0.0
            pred = nums[-1].replace(",", "")
        else:
            pred = match.group(1).replace(",", "")
        try:
            return 1.0 if float(pred) == float(item["answer"].replace(",", "")) else 0.1
        except ValueError:
            return 0.0

    def sample_batch(self, n: int) -> list[dict]:
        self.load()
        return random.sample(self.prompts, min(n, len(self.prompts)))


class MathEnv:
    """MATH competition problems."""

    name = "math"

    def __init__(self):
        self.prompts = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            from datasets import load_dataset
            ds = load_dataset("lighteval/MATH", split="train")
            for item in ds:
                self.prompts.append({
                    "question": item["problem"],
                    "answer": item["solution"],
                })
            self._loaded = True
            print(f"  MATH: {len(self.prompts)} problems loaded")
        except Exception as e:
            print(f"  Warning: could not load MATH dataset: {e}")
            sys.exit(1)

    def make_prompt(self, item: dict) -> str:
        return (
            "<|im_start|>system\n"
            "Solve the problem. Show your work. "
            "Box your final answer: \\boxed{answer}\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{item['question']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def reward(self, response: str, item: dict) -> float:
        import re
        match = re.search(r'\\boxed\{([^}]+)\}', response)
        if not match:
            return 0.0
        pred = match.group(1).strip()
        gold_match = re.search(r'\\boxed\{([^}]+)\}', item["answer"])
        if gold_match and pred == gold_match.group(1).strip():
            return 1.0
        return 0.1

    def sample_batch(self, n: int) -> list[dict]:
        self.load()
        return random.sample(self.prompts, min(n, len(self.prompts)))


class ToolUseEnv:
    """Tool-calling (same as grpo_tooluse_tinker.py)."""

    name = "tool_use"

    TOOLS = [
        {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
        {"name": "get_weather", "description": "Weather for a city", "parameters": {"city": "string", "units": "string"}},
        {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
        {"name": "get_time", "description": "Time in timezone", "parameters": {"timezone": "string"}},
        {"name": "set_reminder", "description": "Set a reminder", "parameters": {"task": "string", "time": "string"}},
    ]

    RAW = [
        ("What is 245 * 37?", "calculator", {"expression": "245 * 37"}),
        ("Calculate sqrt(144)", "calculator", {"expression": "sqrt(144)"}),
        ("15% of 980?", "calculator", {"expression": "0.15 * 980"}),
        ("Weather in Tokyo?", "get_weather", {"city": "Tokyo", "units": "metric"}),
        ("Is it raining in London?", "get_weather", {"city": "London", "units": "metric"}),
        ("Search for GPT-5 news", "web_search", {"query": "GPT-5 news"}),
        ("Find Python asyncio tutorial", "web_search", {"query": "Python asyncio tutorial"}),
        ("What time is it in Singapore?", "get_time", {"timezone": "Asia/Singapore"}),
        ("Current time in Los Angeles?", "get_time", {"timezone": "America/Los_Angeles"}),
        ("Remind me to call mom at 6pm", "set_reminder", {"task": "call mom", "time": "6pm"}),
        ("Set a reminder for team meeting 10am", "set_reminder", {"task": "team meeting", "time": "10am"}),
    ]

    def __init__(self):
        self.tool_schema = json.dumps(self.TOOLS)
        self.prompts = [
            {"query": q, "tool": t, "args": a}
            for q, t, a in self.RAW
        ] * 28
        random.shuffle(self.prompts)

    def load(self):
        pass  # Already loaded

    def make_prompt(self, item: dict) -> str:
        system = (
            "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
            '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
            "No prose. Only JSON."
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\nAvailable tools:\n{self.tool_schema}\n\n"
            f"User: {item['query']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def reward(self, response: str, item: dict) -> float:
        import re
        m = re.search(r'\{.*\}', response.strip(), re.DOTALL)
        if not m:
            return 0.0
        try:
            p = json.loads(m.group())
        except json.JSONDecodeError:
            return 0.1
        score = 0.3
        if p.get("tool") == item["tool"] or p.get("name") == item["tool"]:
            score += 0.4
        pred_args = p.get("arguments", p.get("parameters", {}))
        if isinstance(pred_args, dict) and item["args"]:
            score += 0.3 * sum(1 for k in item["args"] if k in pred_args) / len(item["args"])
        return min(score, 1.0)

    def sample_batch(self, n: int) -> list[dict]:
        return random.sample(self.prompts, min(n, len(self.prompts)))


# ── W&B Logging ───────────────────────────────────────────────────────────

def init_wandb(cfg: dict, env_name: str) -> "wandb.sdk.wandb_run.Run | None":
    """Initialize W&B run from config."""
    import wandb

    log_cfg = cfg.get("logging", {})
    if log_cfg.get("backend") != "wandb":
        return None

    model_short = cfg["model"]["name"].split("/")[-1]
    run_name = log_cfg.get("run_name") or f"{model_short}-{env_name}-{datetime.now():%Y%m%d-%H%M%S}"

    run = wandb.init(
        project=log_cfg.get("project", "skyrl-tinker"),
        name=run_name,
        config={
            "model": cfg["model"]["name"],
            "environment": env_name,
            "algorithm": cfg["training"]["algorithm"],
            "lora_rank": cfg["tinker"]["lora_rank"],
            "learning_rate": cfg["tinker"]["learning_rate"],
            "group_size": cfg["training"]["group_size"],
            "prompts_per_step": cfg["training"]["prompts_per_step"],
            "steps": cfg["training"]["steps"],
            "sampling_max_tokens": cfg["training"]["sampling"]["max_tokens"],
            "sampling_temperature": cfg["training"]["sampling"]["temperature"],
            "sampling_top_p": cfg["training"]["sampling"]["top_p"],
            "optimizer_beta1": cfg["tinker"].get("optimizer", {}).get("beta1", 0.9),
            "optimizer_beta2": cfg["tinker"].get("optimizer", {}).get("beta2", 0.95),
        },
        tags=[model_short, env_name, "grpo", "tinker", "skyrl"],
    )
    print(f"  W&B run: {run.url}")
    return run


def log_step_wandb(
    run,
    step: int,
    loss: float,
    avg_reward: float,
    rewards: list[float],
    step_time: float,
    samples: list[dict] | None = None,
    log_samples: bool = False,
):
    """Log a training step to W&B."""
    if run is None:
        return

    import wandb

    metrics = {
        "train/step": step,
        "train/loss": loss,
        "train/reward_mean": avg_reward,
        "train/reward_min": min(rewards) if rewards else 0,
        "train/reward_max": max(rewards) if rewards else 0,
        "train/reward_std": (sum((r - avg_reward) ** 2 for r in rewards) / max(len(rewards), 1)) ** 0.5 if rewards else 0,
        "train/step_time_s": step_time,
        "train/samples_per_step": len(rewards),
    }
    run.log(metrics, step=step)

    # Log sample table every 5 steps
    if log_samples and samples and step % 5 == 0:
        table = wandb.Table(columns=["step", "prompt", "response", "reward"])
        for s in samples[:8]:
            table.add_data(step, s["prompt"][:200], s["response"][:500], s["reward"])
        run.log({"samples": table}, step=step)


def log_checkpoint_wandb(run, step: int, ckpt_path: str):
    """Log checkpoint event to W&B."""
    if run is None:
        return
    run.log({"checkpoint/step": step, "checkpoint/path": ckpt_path}, step=step)


# ── HuggingFace Hub Upload ────────────────────────────────────────────────

def upload_to_hub(
    cfg: dict,
    env_name: str,
    svc: "tinker.ServiceClient",
    final_ckpt_path: str,
    run_id: str,
    metrics: dict,
    wandb_url: str | None = None,
):
    """Download LoRA weights from Tinker and push to HuggingFace Hub."""
    hf_cfg = cfg.get("huggingface", {})
    if not hf_cfg.get("push_to_hub", False):
        print("\nHF upload disabled in config.")
        return None

    import shutil
    import tarfile
    import tempfile
    import urllib.request

    from huggingface_hub import HfApi

    model_name = cfg["model"]["name"]
    model_short = model_name.split("/")[-1]
    hub_org = hf_cfg.get("hub_org", "arvindcr4")
    repo_prefix = hf_cfg.get("repo_prefix", "skyrl-tinker")
    repo_id = f"{hub_org}/{repo_prefix}-{model_short}-{env_name}".lower()
    private = hf_cfg.get("private", False)

    print(f"\nUploading to HuggingFace Hub: {repo_id}")

    api = HfApi()

    # Create repo if needed
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # Download weights from Tinker via signed archive URL
    local_dir = Path(f"/tmp/skyrl-hf-upload/{repo_id.replace('/', '_')}")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading weights from Tinker: {final_ckpt_path}")
    rc = svc.create_rest_client()
    archive_resp = rc.get_checkpoint_archive_url_from_tinker_path(final_ckpt_path).result()
    archive_url = archive_resp.url

    # Download and extract
    archive_path = local_dir / "checkpoint.tar.gz"
    urllib.request.urlretrieve(archive_url, str(archive_path))
    print(f"  Downloaded archive: {archive_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Auto-detect archive format (Tinker may return .tar or .tar.gz)
    try:
        with tarfile.open(str(archive_path), "r:gz") as tar:
            tar.extractall(path=str(local_dir))
    except tarfile.ReadError:
        with tarfile.open(str(archive_path), "r:") as tar:
            tar.extractall(path=str(local_dir))
    archive_path.unlink()
    print(f"  Extracted weights to: {local_dir}")

    # Save training config
    with open(local_dir / "training_config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Save metrics
    with open(local_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate model card
    if hf_cfg.get("model_card", True):
        avg_reward = metrics.get("final_avg_reward", 0)
        total_steps = metrics.get("total_steps", 0)
        card_content = f"""---
library_name: peft
base_model: {model_name}
tags:
- skyrl
- tinker
- grpo
- rl
- lora
- {env_name}
license: apache-2.0
---

# {repo_id.split('/')[-1]}

LoRA fine-tuned with GRPO via [SkyRL](https://github.com/NovaSky-AI/SkyRL) + [Tinker](https://thinkingmachines.ai).

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `{model_name}` |
| Method | GRPO (Group Relative Policy Optimization) |
| Environment | `{env_name}` |
| LoRA rank | {cfg['tinker']['lora_rank']} |
| Learning rate | {cfg['tinker']['learning_rate']} |
| Steps | {total_steps} |
| Group size | {cfg['training']['group_size']} |
| Final avg reward | {avg_reward:.3f} |
| Tinker run ID | `{run_id}` |
"""
        if wandb_url:
            card_content += f"| W&B | [{wandb_url}]({wandb_url}) |\n"

        card_content += f"""
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{model_name}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
```

## Training Curve

Trained for {total_steps} steps with final average reward of {avg_reward:.3f}.

## Framework

- **RL Framework**: SkyRL
- **Training Backend**: Tinker (hosted)
- **Algorithm**: GRPO
"""
        with open(local_dir / "README.md", "w") as f:
            f.write(card_content)

    # Upload all files
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        commit_message=f"Upload GRPO-trained LoRA ({env_name}, {total_steps} steps, reward={avg_reward:.3f})",
    )

    repo_url = f"https://huggingface.co/{repo_id}"
    print(f"  Uploaded: {repo_url}")
    return repo_url


# ── Main training loop ────────────────────────────────────────────────────

def train(cfg: dict, env, wandb_run):
    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    model_name = cfg["model"]["name"]
    tcfg = cfg["tinker"]
    trcfg = cfg["training"]
    log_cfg = cfg.get("logging", {})
    log_every = log_cfg.get("log_every", 1)
    log_samples = log_cfg.get("log_samples", False)

    # Connect to Tinker
    print(f"Connecting to Tinker (model={model_name})...")
    base_url = tcfg.get("base_url")
    svc = tinker.ServiceClient(base_url=base_url)
    tc = svc.create_lora_training_client(base_model=model_name, rank=tcfg["lora_rank"])
    run_id = tc.model_id
    print(f"  Run ID: {run_id}")

    if wandb_run:
        wandb_run.config.update({"tinker_run_id": run_id})

    # Tokenizer
    print(f"Loading tokenizer: {cfg['model']['tokenizer']}...")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"], trust_remote_code=True)

    # Initial sampler
    print("Saving initial sampler weights...")
    w0 = tc.save_weights_for_sampler(name="step_0").result()
    sc = tc.create_sampling_client(model_path=w0.path)
    print(f"  Sampler ready: {w0.path}")

    # Training params
    steps = trcfg["steps"]
    group_size = trcfg["group_size"]
    prompts_per_step = trcfg["prompts_per_step"]
    save_every = trcfg["save_every"]
    sp_cfg = trcfg["sampling"]
    lr = tcfg["learning_rate"]
    opt_cfg = tcfg.get("optimizer", {})

    print(f"\nGRPO — {steps} steps, group={group_size}, env={env.name}, model={model_name}\n")

    step_rewards = []
    all_rewards_flat = []
    train_start = time.time()

    for step in range(steps):
        step_start = time.time()
        batch = env.sample_batch(prompts_per_step)
        all_data = []
        all_advs = []
        batch_rewards = []
        step_samples = []

        for item in batch:
            prompt_text = env.make_prompt(item)
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_mi = T.ModelInput.from_ints(prompt_ids)

            sp = T.SamplingParams(
                max_tokens=sp_cfg["max_tokens"],
                temperature=sp_cfg["temperature"],
                top_p=sp_cfg["top_p"],
            )
            responses = sc.sample(prompt_mi, num_samples=group_size, sampling_params=sp).result()

            rewards = []
            for resp in responses.sequences:
                text = tok.decode(list(resp.tokens), skip_special_tokens=True)
                r = env.reward(text, item)
                rewards.append(r)
                if log_samples and len(step_samples) < 8:
                    step_samples.append({
                        "prompt": prompt_text.split("user\n")[-1].split("<|im_end|>")[0][:200],
                        "response": text[:500],
                        "reward": r,
                    })

            # GRPO advantages
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

        # GRPO loss via forward_backward_custom
        import torch

        _step_advs = all_advs

        def grpo_loss_fn(data, logprobs_list):
            losses = []
            for i, logprobs in enumerate(logprobs_list):
                adv = _step_advs[i]
                losses.append(-adv * logprobs.sum())
            loss = torch.stack(losses).mean()
            return loss, {"grpo_loss": loss.item()}

        result = tc.forward_backward_custom(
            data=all_data,
            loss_fn=grpo_loss_fn,
            loss_type_input="logprobs",
        ).result()
        tc.optim_step(T.AdamParams(
            learning_rate=lr,
            beta1=opt_cfg.get("beta1", 0.9),
            beta2=opt_cfg.get("beta2", 0.95),
            eps=opt_cfg.get("eps", 1e-8),
        )).result()

        step_time = time.time() - step_start
        avg_r = sum(batch_rewards) / len(batch_rewards)
        step_rewards.append(avg_r)
        all_rewards_flat.extend(batch_rewards)
        loss_val = result.metrics.get("grpo_loss", float("nan"))
        print(f"Step {step + 1:3d}/{steps} | loss={loss_val:.4f} | reward={avg_r:.3f} | {step_time:.1f}s")

        # W&B logging
        if (step + 1) % log_every == 0:
            log_step_wandb(
                wandb_run, step + 1, loss_val, avg_r, batch_rewards, step_time,
                samples=step_samples, log_samples=log_samples,
            )

        # Checkpoint
        if (step + 1) % save_every == 0:
            tc.save_state(name=f"state_{step + 1}")
            ckpt = tc.save_weights_for_sampler(name=f"step_{step + 1}").result()
            sc = tc.create_sampling_client(model_path=ckpt.path)
            log_checkpoint_wandb(wandb_run, step + 1, ckpt.path)
            print(f"  -> Checkpoint saved: step_{step + 1}")

    # Final save
    total_time = time.time() - train_start
    tc.save_state(name="final")
    final_ckpt = tc.save_weights_for_sampler(name="final").result()

    final_avg = sum(step_rewards[-10:]) / max(len(step_rewards[-10:]), 1)
    overall_avg = sum(all_rewards_flat) / max(len(all_rewards_flat), 1)

    print(f"\nDone in {total_time:.0f}s. Final checkpoint: {final_ckpt.path}")
    print(f"Avg reward (last 10 steps): {final_avg:.3f}")
    print(f"Run ID: {run_id}")

    # Log final summary to W&B
    if wandb_run:
        wandb_run.summary.update({
            "final_avg_reward": final_avg,
            "overall_avg_reward": overall_avg,
            "total_steps": steps,
            "total_time_s": total_time,
            "tinker_run_id": run_id,
            "final_checkpoint": final_ckpt.path,
        })

    # Collect metrics for HF model card
    metrics = {
        "final_avg_reward": final_avg,
        "overall_avg_reward": overall_avg,
        "total_steps": steps,
        "total_time_s": total_time,
        "reward_history": step_rewards,
    }

    # Upload to HuggingFace Hub
    wandb_url = wandb_run.url if wandb_run else None
    hf_url = upload_to_hub(
        cfg, env.name, svc, final_ckpt.path, run_id, metrics, wandb_url,
    )

    # Log HF link back to W&B
    if wandb_run and hf_url:
        wandb_run.summary["huggingface_url"] = hf_url
        wandb_run.log({"huggingface_url": hf_url})

    if wandb_run:
        wandb_run.finish()
        print(f"W&B run: {wandb_url}")

    if hf_url:
        print(f"HF model: {hf_url}")


def main():
    parser = argparse.ArgumentParser(description="SkyRL + Tinker Training Bridge")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--env", default="tool_use", help="Environment: gsm8k, math, tool_use")
    parser.add_argument("--model", default=None, help="Override model name")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--group-size", type=int, default=None, help="Override GRPO group size")
    parser.add_argument("--lora-rank", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-hf", action="store_true", help="Disable HF upload")
    args = parser.parse_args()

    cfg = load_config(args.config, vars(args))

    # CLI overrides for logging/upload
    if args.no_wandb:
        cfg.setdefault("logging", {})["backend"] = "none"
    if args.no_hf:
        cfg.setdefault("huggingface", {})["push_to_hub"] = False

    print("=" * 60)
    print("SkyRL + Tinker Training Bridge")
    print("=" * 60)
    print(f"  Config:     {args.config}")
    print(f"  Model:      {cfg['model']['name']}")
    print(f"  Environment:{args.env}")
    print(f"  LoRA rank:  {cfg['tinker']['lora_rank']}")
    print(f"  LR:         {cfg['tinker']['learning_rate']}")
    print(f"  Steps:      {cfg['training']['steps']}")
    print(f"  Group size: {cfg['training']['group_size']}")
    print(f"  Tinker:     {'hosted (cloud)' if not cfg['tinker'].get('base_url') else cfg['tinker']['base_url']}")
    print(f"  W&B:        {cfg.get('logging', {}).get('backend', 'none')}")
    print(f"  HF upload:  {cfg.get('huggingface', {}).get('push_to_hub', False)}")
    print("=" * 60)

    # Init W&B
    wandb_run = init_wandb(cfg, args.env)

    env = load_environment(args.env)
    train(cfg, env, wandb_run)


if __name__ == "__main__":
    main()
