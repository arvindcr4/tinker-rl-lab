#!/usr/bin/env python3
"""Tiny live Tinker GRPO probe with ZVF/GU logging.

This is intentionally bounded: Qwen3.5-4B, LoRA rank 4, group size 2,
batch size 1, and 15 steps.  It is meant to create one fresh live Tinker
run that can be used as a prospective check of the early-ZVF diagnostic
without relaunching the full ablation campaign.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "tinker-runs" / "results"
SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
QUESTION_SUFFIX = " Provide the final numerical answer inside \\boxed{}."

_ADVS: list[float] = []


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def reward_fn(response: str, answer: str) -> float:
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for item in boxed:
        cleaned = item.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(cleaned) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            if cleaned == answer:
                return 1.0
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if nums:
        try:
            if abs(float(nums[-1].replace(",", "")) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            pass
    return 0.0


def loss_fn(data, logprobs_list):
    losses = [(-_ADVS[i] * logprobs_list[i].sum()) for i in range(len(logprobs_list))]
    loss = losses[0] if len(losses) == 1 else __import__("torch").stack(losses).mean()
    return loss, {"loss": loss.item()}


def load_gsm8k_examples(limit: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples: list[tuple[str, str]] = []
    for row in ds:
        match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not match:
            continue
        answer = match.group(1).replace(",", "").strip()
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{row['question']}{QUESTION_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        examples.append((prompt, answer))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def run(args: argparse.Namespace) -> dict:
    load_env_file(ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set and was not found in .env")

    import torch
    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tag = args.tag or f"live_zvf_probe_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    output_path = RESULTS_DIR / f"{tag}.json"

    result: dict = {
        "tag": tag,
        "status": "started",
        "model": args.model,
        "model_short": args.model.split("/")[-1].lower(),
        "task": "gsm8k",
        "seed": args.seed,
        "rank": args.rank,
        "group_size": args.group,
        "batch": args.batch,
        "lr": args.lr,
        "temperature": args.temperature,
        "steps": args.steps,
        "started_at": started_at,
        "output_path": str(output_path.relative_to(ROOT)),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    try:
        examples = load_gsm8k_examples(args.example_limit, args.seed)
        if not examples:
            raise RuntimeError("No GSM8K examples loaded")

        print(f"[{tag}] Connecting to Tinker model={args.model} rank={args.rank}", flush=True)
        svc = tinker.ServiceClient(base_url=None)
        tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
        result["run_id"] = tc.model_id
        output_path.write_text(json.dumps(result, indent=2) + "\n")

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        initial = tc.save_weights_for_sampler(name="s0").result()
        sc = tc.create_sampling_client(model_path=initial.path)
        result["initial_sampler"] = initial.path

        step_rewards: list[float] = []
        step_log: list[dict] = []
        rng = random.Random(args.seed)

        for step in range(args.steps):
            batch_examples = rng.sample(examples, min(args.batch, len(examples)))
            all_data = []
            all_advs: list[float] = []
            batch_rewards: list[float] = []
            zero_var_prompts = 0

            for prompt, answer in batch_examples:
                prompt_ids = tok.encode(prompt, add_special_tokens=False)
                if len(prompt_ids) > args.max_prompt_tokens:
                    prompt_ids = prompt_ids[: args.max_prompt_tokens]
                sampling = T.SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                sampled = sc.sample(
                    T.ModelInput.from_ints(prompt_ids),
                    num_samples=args.group,
                    sampling_params=sampling,
                ).result()
                rewards: list[float] = []
                for sequence in sampled.sequences:
                    text = tok.decode(list(sequence.tokens), skip_special_tokens=True)
                    rewards.append(reward_fn(text, answer))

                mean_reward = sum(rewards) / len(rewards)
                variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
                std_reward = math.sqrt(variance) + 1e-8
                if variance < 1e-10:
                    zero_var_prompts += 1
                batch_rewards.extend(rewards)

                for sequence, reward in zip(sampled.sequences, rewards):
                    response_ids = list(sequence.tokens)
                    full_ids = prompt_ids + response_ids
                    target_ids = full_ids[1:] + [0]
                    all_data.append(
                        T.Datum(
                            model_input=T.ModelInput.from_ints(full_ids),
                            loss_fn_inputs={
                                "target_tokens": T.TensorData(
                                    data=target_ids,
                                    dtype="int64",
                                    shape=[len(target_ids)],
                                )
                            },
                        )
                    )
                    all_advs.append((reward - mean_reward) / std_reward)

            _ADVS.clear()
            _ADVS.extend(all_advs)
            fwdbwd = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
            tc.optim_step(
                T.AdamParams(
                    learning_rate=args.lr,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )
            ).result()

            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            zvf = zero_var_prompts / max(len(batch_examples), 1)
            loss_value = None
            try:
                loss_value = float(fwdbwd.metrics.get("loss", 0.0))
            except Exception:
                pass
            step_rewards.append(avg_reward)
            step_log.append(
                {
                    "step": step + 1,
                    "reward": avg_reward,
                    "loss": loss_value,
                    "zvf": zvf,
                    "gu": 1.0 - zvf,
                }
            )

            print(
                f"[{tag}] step {step+1:02d}/{args.steps} "
                f"reward={avg_reward:.3f} zvf={zvf:.2f} loss={loss_value}",
                flush=True,
            )

            if (step + 1) % args.save_every == 0:
                weights = tc.save_weights_for_sampler(name=f"s{step+1}").result()
                sc = tc.create_sampling_client(model_path=weights.path)
                result["latest_sampler"] = weights.path

            result.update(
                {
                    "status": "running",
                    "reward_trace": step_rewards,
                    "step_log": step_log,
                    "last_updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            output_path.write_text(json.dumps(result, indent=2) + "\n")

        final_weights = tc.save_weights_for_sampler(name="final").result()
        first5 = step_rewards[:5]
        last10 = step_rewards[-10:]
        result.update(
            {
                "status": "completed",
                "checkpoint": final_weights.path,
                "reward_trace": step_rewards,
                "step_log": step_log,
                "first5_avg": sum(first5) / len(first5) if first5 else 0.0,
                "last10_avg": sum(last10) / len(last10) if last10 else 0.0,
                "peak_reward": max(step_rewards) if step_rewards else 0.0,
                "zero_reward_pct": (
                    100.0 * sum(1 for value in step_rewards if value == 0.0) / len(step_rewards)
                    if step_rewards
                    else 0.0
                ),
                "zero_loss_pct": (
                    100.0
                    * sum(
                        1
                        for item in step_log
                        if item["loss"] is not None and abs(item["loss"]) < 1e-9
                    )
                    / max(len(step_log), 1)
                ),
                "wall_clock_sec": time.time() - t0,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[{tag}] completed run_id={result.get('run_id')} output={output_path}", flush=True)
        return result

    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_clock_sec": time.time() - t0,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--group", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--example-limit", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
