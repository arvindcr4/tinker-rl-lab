"""Parameterized GRPO trainer for the research loop.

Reads a YAML config (see research_loop/best_recipe.yaml for the schema),
runs training on Tinker API, emits `METRIC name=value` lines on stdout,
and writes a JSON result file.

Knobs:
    model, seed, rank, steps, lr, group_size, batch, temperature, max_tokens
    adv_norm        : std | none | rank
    reward_shape    : binary | graded | partial
    curriculum      : random | easy_first | hard_first
    eval_subset     : int, how many GSM8K train examples to sample from

Usage:
    python research_loop/train.py --config path/to/variant.yaml \\
        --output-json path/to/result.json

Based on grpo_gsm8k_base.py. Do not edit that file — edit this one and
copy fixes back if needed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

assert os.environ.get("TINKER_API_KEY"), "Set TINKER_API_KEY in env"

import torch
import tinker
import tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)


def emit_metric(name: str, value) -> None:
    """Print a standard METRIC line. Parsed by coordinator.py."""
    print(f"METRIC {name}={value}", flush=True)


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    required = ["model", "seed", "rank", "steps", "lr", "group_size", "batch",
                "temperature", "adv_norm", "reward_shape", "curriculum"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config {path} missing required fields: {missing}")
    cfg.setdefault("max_tokens", 512)
    cfg.setdefault("eval_subset", 500)
    return cfg


def load_gsm8k(
    eval_subset: int,
    curriculum: str,
    seed: int,
    tok,
) -> list[tuple[str, str, int]]:
    """Return list of (prompt, answer, prompt_length). Curriculum applied.

    Uses the tokenizer's chat template so the same code works for Qwen and
    Llama families (ChatML vs Llama-3 templates differ).
    """
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    examples: list[tuple[str, str, int]] = []
    for row in ds:
        q = row["question"]
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m:
            continue
        answer = m.group(1).replace(",", "").strip()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        examples.append((prompt, answer, len(q)))

    # Take a random subset of `eval_subset` from the full dataset, then order
    # by curriculum. This gives us a manageable working set while still
    # covering the dataset distribution.
    rng.shuffle(examples)
    examples = examples[:eval_subset]

    if curriculum == "easy_first":
        examples.sort(key=lambda e: e[2])
    elif curriculum == "hard_first":
        examples.sort(key=lambda e: e[2], reverse=True)
    elif curriculum == "random":
        rng.shuffle(examples)
    else:
        raise ValueError(f"unknown curriculum: {curriculum}")

    return examples


def reward_binary(response: str, answer: str) -> float:
    """1.0 if \\boxed{answer} matches or last number matches, else 0.0."""
    response = response.strip()
    for b in re.findall(r"\\boxed\{([^}]+)\}", response):
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            if b_clean == answer:
                return 1.0
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if nums:
        last = nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            pass
    return 0.0


def reward_graded(response: str, answer: str) -> float:
    """Dense reward: 1.0 for boxed correct, 0.5 for correct unboxed, 0.25 for
    boxed-but-wrong, 0.1 for any \\boxed{} presence, 0 else. Breaks zero-advantage."""
    response = response.strip()
    has_boxed = bool(re.search(r"\\boxed\{[^}]+\}", response))
    binary = reward_binary(response, answer)
    if binary == 1.0 and has_boxed:
        return 1.0
    if binary == 1.0:
        return 0.5
    if has_boxed:
        return 0.25
    return 0.0


def reward_partial(response: str, answer: str) -> float:
    """Graded by digit overlap between final number and target answer."""
    binary = reward_binary(response, answer)
    if binary == 1.0:
        return 1.0
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if not nums:
        return 0.0
    last = nums[-1].replace(",", "")
    try:
        last_int = str(int(abs(float(last))))
        ans_int = str(int(abs(float(answer))))
        if not last_int or not ans_int:
            return 0.0
        # digit-level Jaccard similarity, capped at 0.5
        overlap = len(set(last_int) & set(ans_int))
        union = len(set(last_int) | set(ans_int))
        return 0.5 * (overlap / union) if union else 0.0
    except ValueError:
        return 0.0


REWARD_FNS = {
    "binary": reward_binary,
    "graded": reward_graded,
    "partial": reward_partial,
}


def advantages(rewards: list[float], method: str) -> list[float]:
    n = len(rewards)
    mr = sum(rewards) / n
    if method == "none":
        return [r - mr for r in rewards]
    if method == "std":
        var = sum((r - mr) ** 2 for r in rewards) / n
        sr = var ** 0.5 + 1e-8
        return [(r - mr) / sr for r in rewards]
    if method == "rank":
        # Rank-based: sort rewards, use centered rank positions.
        order = sorted(range(n), key=lambda i: rewards[i])
        ranks = [0.0] * n
        for rank_idx, orig_idx in enumerate(order):
            ranks[orig_idx] = rank_idx
        mean_rank = (n - 1) / 2.0
        return [(r - mean_rank) / max(mean_rank, 1.0) for r in ranks]
    raise ValueError(f"unknown adv_norm: {method}")


def run(cfg: dict) -> dict:
    t0 = time.time()
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model_name = cfg["model"]
    tag = cfg.get("name", f"{model_name.split('/')[-1]}_s{cfg['seed']}")
    print(f"[{tag}] config={json.dumps({k: v for k, v in cfg.items() if k != 'hypothesis'})}", flush=True)

    reward_fn = REWARD_FNS[cfg["reward_shape"]]

    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model_name, rank=cfg["rank"])
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    examples = load_gsm8k(cfg["eval_subset"], cfg["curriculum"], cfg["seed"], tok)
    print(f"[{tag}] loaded {len(examples)} GSM8K examples (curriculum={cfg['curriculum']})", flush=True)
    w0 = tc.save_weights_for_sampler(name="s0").result()
    sc = tc.create_sampling_client(model_path=w0.path)
    print(f"[{tag}] connected to tinker: run={tc.model_id}", flush=True)

    _adv: list[float] = []

    def loss_fn(data, lp):
        losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
        loss = torch.stack(losses).mean()
        return loss, {"loss": loss.item()}

    step_rewards: list[float] = []
    zero_loss_steps = 0
    zero_reward_steps = 0
    cursor = 0  # curriculum cursor (so easy_first / hard_first order is respected)

    for step in range(cfg["steps"]):
        if cfg["curriculum"] == "random":
            batch = random.sample(examples, cfg["batch"])
        else:
            batch = []
            for _ in range(cfg["batch"]):
                batch.append(examples[cursor % len(examples)])
                cursor += 1

        all_data, all_advs, batch_r = [], [], []

        for prompt_text, ans, _ in batch:
            pid = tok.encode(prompt_text, add_special_tokens=False)
            if len(pid) > 1024:
                pid = pid[:1024]
            sp = T.SamplingParams(
                max_tokens=cfg["max_tokens"],
                temperature=cfg["temperature"],
                top_p=0.95,
            )
            resp = sc.sample(
                T.ModelInput.from_ints(pid),
                num_samples=cfg["group_size"],
                sampling_params=sp,
            ).result()

            rews = [reward_fn(tok.decode(list(r.tokens), skip_special_tokens=True), ans)
                    for r in resp.sequences]
            advs = advantages(rews, cfg["adv_norm"])
            batch_r.extend(rews)

            for r, a in zip(resp.sequences, advs):
                rid = list(r.tokens)
                fid = pid + rid
                tid = fid[1:] + [0]
                all_data.append(T.Datum(
                    model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])
                    },
                ))
                all_advs.append(a)

        if not all_data:
            continue

        _adv = all_advs
        result = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
        tc.optim_step(T.AdamParams(
            learning_rate=cfg["lr"], beta1=0.9, beta2=0.95, eps=1e-8
        )).result()

        avg = sum(batch_r) / len(batch_r)
        step_rewards.append(avg)
        loss_val = result.metrics.get("loss", 0)
        if abs(loss_val) < 1e-6:
            zero_loss_steps += 1
        if avg == 0:
            zero_reward_steps += 1

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"[{tag}] {step + 1:3d}/{cfg['steps']} | "
                f"loss={loss_val:.4f} | reward={avg:.3f} | acc={avg*100:.1f}%",
                flush=True,
            )

    # ── Aggregate ────────────────────────────────────────────────────────
    n = max(len(step_rewards), 1)
    first5 = step_rewards[:5]
    last10 = step_rewards[-10:]
    metrics = {
        "first5_avg_accuracy": sum(first5) / len(first5) if first5 else 0.0,
        "last10_avg_accuracy": sum(last10) / len(last10) if last10 else 0.0,
        "peak_accuracy": max(step_rewards) if step_rewards else 0.0,
        "mean_accuracy": sum(step_rewards) / n,
        "zero_loss_pct": 100.0 * zero_loss_steps / n,
        "zero_reward_pct": 100.0 * zero_reward_steps / n,
        "n_steps_completed": len(step_rewards),
    }

    for k, v in metrics.items():
        emit_metric(k, f"{v:.4f}" if isinstance(v, float) else v)
    emit_metric("wall_clock_seconds", int(time.time() - t0))
    emit_metric("status", "completed")

    return {
        "tag": tag,
        "model_id": tc.model_id,
        "metrics": metrics,
        "reward_trace": [round(r, 4) for r in step_rewards],
        "wall_clock_seconds": int(time.time() - t0),
        "status": "completed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-json", required=False, type=Path)
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        result = run(cfg)
    except Exception as exc:
        emit_metric("status", "failed")
        emit_metric("error", type(exc).__name__)
        result = {
            "config_path": str(args.config),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        if args.output_json:
            args.output_json.write_text(json.dumps(result, indent=2))
        raise

    result["config_path"] = str(args.config)
    result["config"] = cfg
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
