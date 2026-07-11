#!/usr/bin/env python3
"""Shared utilities for zvf-program/experiments-next.

Conventions mirror platform_hybrid/experiments/tinker-runs/live_zvf_probe.py
(same ChatML prompt construction, same boxed-answer reward parser) so that
pool statistics are comparable with the existing audit runs.

Provenance rule (same as sweep/): nothing here fabricates a number. Scripts
write JSON with an explicit "status" field; absent data stays absent.
"""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
REPO_ROOT = HERE.parents[1]  # .../tinker-rl-lab

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
QUESTION_SUFFIX = " Provide the final numerical answer inside \\boxed{}."


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines (optionally 'export '-prefixed) into os.environ."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_repo_env() -> None:
    load_env_file(REPO_ROOT / ".env")


def reward_fn(response: str, answer: str) -> float:
    """Binary verifiable reward: 1.0 iff a \\boxed{} value (or trailing number)
    matches the reference answer. Identical to the audit runners' parser."""
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
    all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            pass
    return 0.0


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + question + QUESTION_SUFFIX + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_gsm8k(split: str, limit: int, seed: int) -> list[tuple[str, str]]:
    """Return [(prompt, answer)] from GSM8K. split: 'train' (pool) or 'test'
    (held-out pass@k). Shuffled with the given seed before truncation."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    examples: list[tuple[str, str]] = []
    for row in ds:
        match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not match:
            continue
        answer = match.group(1).replace(",", "").strip()
        examples.append((build_prompt(row["question"]), answer))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit] if limit else examples


def make_sampler(model: str, sampler_path: str | None, lora_rank: int = 4):
    """Return (sampling_client, tokenizer, sampler_path_used).

    If sampler_path is given (an existing tinker:// weights path from a prior
    run), sample from it directly. Otherwise create a fresh LoRA client on the
    base model and snapshot zero-initialized weights (== base model sampling),
    which is the same pattern live_zvf_probe.py uses for step 0.
    """
    import tinker
    from transformers import AutoTokenizer

    svc = tinker.ServiceClient(base_url=None)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if sampler_path:
        sc = svc.create_sampling_client(model_path=sampler_path)
        return sc, tok, sampler_path
    tc = svc.create_lora_training_client(base_model=model, rank=lora_rank)
    initial = tc.save_weights_for_sampler(name="pool0").result()
    sc = tc.create_sampling_client(model_path=initial.path)
    return sc, tok, initial.path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_result(path: Path, payload: dict) -> None:
    """Atomically replace a result/checkpoint JSON file.

    A killed process can leave a normal ``write_text`` target truncated.  All
    resumable runners share this helper, so checkpoint durability belongs here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_pool(path: Path) -> dict:
    pool = json.loads(path.read_text())
    if pool.get("status") != "complete":
        raise SystemExit(
            f"Pool {path} has status={pool.get('status')!r}; refusing to "
            "analyze an incomplete pool (provenance rule)."
        )
    return pool
