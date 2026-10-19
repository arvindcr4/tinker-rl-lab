#!/usr/bin/env python3
"""Held-out GSM8K pass@k by sampling on Tinker.

Protocol matches zvf-program/experiments-next/modal_passk.py except the
sampler. Prompt, suffix, grader, and pass@k estimator are imported from that
file when it loads. It imports modal at module scope, so a missing modal
package uses the verbatim copies below and records `git rev-parse HEAD`.

Sampling follows platform_hybrid/experiments/tinker-runs/live_zvf_probe.py:
ServiceClient(base_url=None), create_sampling_client(base_model= or
model_path=), SamplingParams, ModelInput.from_ints, and the Qwen tokenizer.
Requests are submitted together and gathered from their futures.
TINKER_API_KEY is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Protocol constants from modal_passk.evaluate / its local entrypoint.
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 512
SEED = 42
KS = (1, 8, 32)

# Verbatim from modal_passk.py (SYSTEM_PROMPT, GSM8K_SUFFIX, build_prompt,
# gsm8k_reward, pass_at_k). Used only when that module cannot be imported.
SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
GSM8K_SUFFIX = " Provide the final numerical answer inside \\boxed{}."


def build_prompt(question: str, suffix: str) -> str:
    return (
        "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n"
        "<|im_start|>user\n" + question + suffix + "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def gsm8k_reward(response: str, answer: str) -> float:
    """Byte-identical logic to the Tinker harness parser."""
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    # Parser v2 (2026-07-11): score only the LAST boxed value (the final
    # answer); accepting ANY boxed match let intermediate boxed steps produce
    # false positives.
    for item in boxed[-1:]:
        cleaned = item.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(cleaned) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            if cleaned == answer:
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


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    r = 1.0
    for i in range(k):
        r *= (n - c - i) / (n - i)
    return 1.0 - r


def repo_root() -> Path:
    start = Path(__file__).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "zvf-program").is_dir():
            return candidate
    raise RuntimeError("repo root not found from " + str(start))


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def bind_protocol(repo: Path) -> tuple[str, str | None, str | None]:
    """Import modal_passk's protocol, or keep the verbatim copies.

    Returns (source, commit, import_error). source is "imported" or "copied".
    The commit is HEAD at runtime, recorded for the copy path.
    """
    global SYSTEM_PROMPT, GSM8K_SUFFIX, build_prompt, gsm8k_reward, pass_at_k

    source_path = repo / "zvf-program" / "experiments-next" / "modal_passk.py"
    spec = importlib.util.spec_from_file_location("_modal_passk_protocol", source_path)
    if spec is None or spec.loader is None:
        commit = git_head(repo)
        return "copied", commit, f"cannot load {source_path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        commit = git_head(repo)
        return "copied", commit, f"{type(exc).__name__}: {exc}"
    missing = [
        name
        for name in ("SYSTEM_PROMPT", "GSM8K_SUFFIX", "build_prompt", "gsm8k_reward", "pass_at_k")
        if not hasattr(module, name)
    ]
    if missing:
        raise ImportError("modal_passk is missing " + ", ".join(missing))
    SYSTEM_PROMPT = module.SYSTEM_PROMPT
    GSM8K_SUFFIX = module.GSM8K_SUFFIX
    build_prompt = module.build_prompt
    gsm8k_reward = module.gsm8k_reward
    pass_at_k = module.pass_at_k
    return "imported", None, None


def load_env_file(path: Path) -> None:
    """Fill missing env vars. Never overrides an existing value. Does not print."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
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


def redact(text: str) -> str:
    secret = os.environ.get("TINKER_API_KEY")
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text


def load_gsm8k_items(n_problems: int, seed: int) -> list[tuple[str, str]]:
    """GSM8K test, seeded shuffle, then the first n_problems. Same as modal_passk."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    items: list[tuple[str, str]] = []
    for row in dataset:
        match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if match:
            items.append(
                (
                    build_prompt(row["question"], GSM8K_SUFFIX),
                    match.group(1).replace(",", "").strip(),
                )
            )
    rng = random.Random(seed)
    rng.shuffle(items)
    if n_problems > len(items):
        raise RuntimeError(f"n_problems={n_problems} exceeds GSM8K test size {len(items)}")
    return items[:n_problems]


def default_tag(model: str, sampler_path: str | None, n_problems: int, n_samples: int) -> str:
    name = model.split("/")[-1].lower()
    if sampler_path:
        tail = sampler_path.rstrip("/").split("/")[-1][:40]
        adapter = "_" + tail
    else:
        adapter = "_base"
    return f"{name}{adapter}_gsm8k_p{n_problems}_n{n_samples}_s{SEED}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K held-out pass@k via Tinker sampling")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--sampler-path",
        default=None,
        help="tinker:// sampler_weights path; omit to sample the base model",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--n-problems", type=int, default=200)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--out", required=True, help="output JSON path")
    args = parser.parse_args(argv)
    if args.n_problems < 1 or args.n_samples < 1:
        parser.error("n-problems and n-samples must be positive")
    args.sampler_path = args.sampler_path or None
    return args


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = repo_root()
    protocol_source, protocol_commit, import_error = bind_protocol(repo)
    # Raw ChatML from build_prompt. Qwen3's chat template would insert a think stub.
    probe = build_prompt("Q", GSM8K_SUFFIX)
    if "<think>" in probe or not probe.endswith("<|im_start|>assistant\n"):
        raise RuntimeError("build_prompt is not the raw ChatML prompt from modal_passk")

    load_env_file(repo / ".env")
    load_env_file(repo / "platform_hybrid" / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set and was not found in .env")

    ks = [k for k in KS if k <= args.n_samples]
    if not ks:
        raise RuntimeError(f"no k in {list(KS)} is <= n-samples={args.n_samples}")
    tag = args.tag or default_tag(args.model, args.sampler_path, args.n_problems, args.n_samples)
    note = f"[tinker-passk] protocol={protocol_source}"
    if protocol_source == "copied":
        note += f" source_commit={protocol_commit}"
        if import_error:
            note += f" ({import_error})"
    print(note, flush=True)
    print(
        f"[tinker-passk] {args.model} adapter={args.sampler_path} "
        f"problems={args.n_problems} samples={args.n_samples} ks={ks} tag={tag}",
        flush=True,
    )

    started = time.time()
    items = load_gsm8k_items(args.n_problems, SEED)
    print(f"[tinker-passk] loaded {len(items)} GSM8K test problems", flush=True)

    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # SDK reads TINKER_API_KEY from the environment. Do not pass or print it.
    service = tinker.ServiceClient(base_url=None)
    if args.sampler_path:
        sampling_client = service.create_sampling_client(model_path=args.sampler_path)
    else:
        sampling_client = service.create_sampling_client(base_model=args.model)
    print("[tinker-passk] submitting samples", flush=True)

    # One request per sample with its own seed: a seed shared across num_samples=n
    # makes Tinker return n identical sequences (verified 2026-10-19).
    pending = []
    for i, (prompt, answer) in enumerate(items):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        futures = [
            sampling_client.sample(
                T.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=T.SamplingParams(
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    seed=(SEED * 1_000_003 + i * args.n_samples + j) % (2**31 - 1),
                ),
            )
            for j in range(args.n_samples)
        ]
        pending.append((answer, futures))

    per_problem_c: list[int] = []
    truncated = 0
    token_total = 0
    sample_total = 0
    for done, (answer, futures) in enumerate(pending, start=1):
        sequences = [seq for fut in futures for seq in fut.result().sequences]
        if len(sequences) != args.n_samples:
            raise RuntimeError(f"expected {args.n_samples} samples, got {len(sequences)}")
        correct = 0
        for sequence in sequences:
            text = tokenizer.decode(list(sequence.tokens), skip_special_tokens=True)
            correct += int(gsm8k_reward(text, answer))
            sample_total += 1
            token_total += len(list(sequence.tokens))
            if sequence.stop_reason == "length":
                truncated += 1
        per_problem_c.append(correct)
        print(f"[tinker-passk] {done}/{len(pending)} c={correct}", flush=True)

    if not sample_total:
        raise RuntimeError("no samples returned")
    payload = {
        "kind": "passk_eval_tinker",
        "backend": "tinker",
        "model": args.model,
        "adapter": args.sampler_path,
        "dataset": "gsm8k",
        "grader": "tinker-identical boxed/number parser",
        "n_problems": len(items),
        "n_per_problem": args.n_samples,
        "ks": ks,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
        "per_problem_c": per_problem_c,
        "pass_at_k": {
            str(k): round(sum(pass_at_k(args.n_samples, c, k) for c in per_problem_c) / len(per_problem_c), 4)
            for k in ks
        },
        "wall_seconds": round(time.time() - started, 1),
        "tag": tag,
        "truncation_rate": round(truncated / sample_total, 6),
        "mean_tokens": round(token_total / sample_total, 4),
        "protocol_source": protocol_source,
        "protocol_source_commit": protocol_commit,
    }
    out = Path(args.out)
    write_json(out, payload)
    print(
        f"[tinker-passk] pass@k={payload['pass_at_k']} "
        f"truncation_rate={payload['truncation_rate']} "
        f"mean_tokens={payload['mean_tokens']} -> {out}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except Exception:
        sys.stderr.write(redact(traceback.format_exc()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
