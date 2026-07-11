#!/usr/bin/env python3
"""eval_passk_standalone.py — portable pass@k evaluator (no Modal deps).

Runs anywhere with a GPU + vllm (Lightning Studio, Colab, bare metal).
Datasets: gsm8k (tinker-identical parser), math500 (strict boxed match,
lower bound), mbpp (unit-test execution — sandboxed subprocess w/ timeout;
run only on an isolated VM). Supports Tinker-exported HF LoRA adapters via
--adapter (remaps all-linear target modules and unembed_tokens -> lm_head,
same logic as modal_passk.py).

  python eval_passk_standalone.py --dataset mbpp --problems 200 --n 32 \
      --out passk_lightning_qwen3-8b_base_mbpp.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SYSTEM_PROMPT_MATH = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
SYSTEM_PROMPT_CODE = (
    "You are an expert Python programmer. Write a correct, self-contained "
    "Python function for the task. Output ONLY a Python code block."
)
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def chatml(system: str, user: str) -> str:
    return (
        IM_START
        + "system\n"
        + system
        + IM_END
        + "\n"
        + IM_START
        + "user\n"
        + user
        + IM_END
        + "\n"
        + IM_START
        + "assistant\n"
    )


def gsm8k_reward(response: str, answer: str) -> float:
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
        last = nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            pass
    return 0.0


def norm_math(a: str) -> str:
    a = a.strip().replace("\\left", "").replace("\\right", "")
    for junk in ("\\!", "\\,", "\\;", " "):
        a = a.replace(junk, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    if a.startswith("{") and a.endswith("}"):
        a = a[1:-1]
    return a


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i, depth, out = idx + 7, 1, []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if not depth:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def math_reward(response: str, answer: str) -> float:
    got = last_boxed(response)
    return 1.0 if got is not None and norm_math(got) == norm_math(answer) else 0.0


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(response: str) -> str:
    blocks = CODE_BLOCK_RE.findall(response)
    return blocks[-1] if blocks else response


def _run_candidate(payload):
    code, tests = payload
    prog = code + "\n\n" + "\n".join(tests) + "\nprint('__PASS__')\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(prog)
            path = f.name
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        return 1 if "__PASS__" in p.stdout else 0
    except Exception:
        return 0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    r = 1.0
    for i in range(k):
        r *= (n - c - i) / (n - i)
    return 1.0 - r


def write_checkpoint(path: Path, payload: dict) -> None:
    """Atomically persist progress so SIGKILL cannot corrupt the checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def validate_resume(existing: dict, expected: dict, fingerprints: list[str]) -> None:
    """Refuse to combine partial results from different evaluation configs."""
    mismatches = {
        key: {"existing": existing.get(key), "requested": value}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise ValueError("incompatible partial result: " + json.dumps(mismatches, sort_keys=True))
    if existing.get("prompt_fingerprints") != fingerprints:
        raise ValueError("prompt fingerprints changed; refusing unsafe resume")
    counts = existing.get("per_problem_c")
    if not isinstance(counts, list) or len(counts) > len(fingerprints):
        raise ValueError("partial result has invalid per_problem_c progress")
    checkpoint_every = int(expected["checkpoint_every"])
    if len(counts) < len(fingerprints) and len(counts) % checkpoint_every:
        raise ValueError("partial result does not end on a checkpoint boundary")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["gsm8k", "math500", "mbpp"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", default="", help="HF LoRA adapter repo (Tinker export; remapped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="resume a compatible partial output file")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="persist after this many completed problems",
    )
    args = ap.parse_args()

    if args.problems <= 0 or args.n <= 0:
        ap.error("--problems and --n must be positive")
    if args.checkpoint_every <= 0:
        ap.error("--checkpoint-every must be positive")
    if args.n < max(args.ks):
        ap.error(f"--n {args.n} must be >= max k {max(args.ks)}")

    import random
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    import vllm

    t0 = time.time()
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for row in ds:
            m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            if m:
                items.append(
                    (
                        chatml(
                            SYSTEM_PROMPT_MATH,
                            row["question"] + " Provide the final numerical answer inside "
                            "\\boxed{}.",
                        ),
                        m.group(1).replace(",", "").strip(),
                    )
                )
        grade_mode = "gsm8k"
    elif args.dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [
            (
                chatml(
                    SYSTEM_PROMPT_MATH, row["problem"] + " Put your final answer inside \\boxed{}."
                ),
                row["answer"],
            )
            for row in ds
        ]
        grade_mode = "math"
    else:
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        items = []
        for row in ds:
            user = (
                row["prompt"] + "\nYour code must pass these tests:\n" + "\n".join(row["test_list"])
            )
            items.append((chatml(SYSTEM_PROMPT_CODE, user), row["test_list"]))
        grade_mode = "mbpp"

    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.problems] if args.problems else items

    prompt_fingerprints = [
        hashlib.sha256(
            (prompt + "\0" + json.dumps(answer, sort_keys=True)).encode("utf-8")
        ).hexdigest()[:16]
        for prompt, answer in items
    ]
    output_path = Path(args.out)
    expected = {
        "model": args.model,
        "adapter": args.adapter or None,
        "dataset": args.dataset,
        "n_problems": len(items),
        "n_per_problem": args.n,
        "ks": args.ks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "checkpoint_every": args.checkpoint_every,
    }
    previous = None
    if output_path.exists():
        previous = json.loads(output_path.read_text())
        if not args.resume:
            ap.error(f"{output_path} exists; pass --resume or choose another --out")
        if previous.get("status") == "complete":
            mismatches = {
                key: {"existing": previous.get(key), "requested": value}
                for key, value in expected.items()
                if key in previous and previous.get(key) != value
            }
            if mismatches:
                ap.error(
                    "completed output uses another config: "
                    + json.dumps(mismatches, sort_keys=True)
                )
            print(f"[resume] already complete -> {output_path}", flush=True)
            return
        try:
            validate_resume(previous, expected, prompt_fingerprints)
        except ValueError as exc:
            ap.error(str(exc))

    lora_kwargs = {}
    lora_request = None
    if args.adapter:
        import shutil
        from vllm.lora.request import LoRARequest
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file, save_file

        adapter_path = snapshot_download(args.adapter)
        if os.path.exists(os.path.join(adapter_path, "final", "adapter_model.safetensors")):
            adapter_path = os.path.join(adapter_path, "final")
        patched = os.path.expanduser("~/adapter_patched_" + args.adapter.split("/")[-1][:40])
        shutil.copytree(adapter_path, patched, dirs_exist_ok=True, symlinks=False)
        wpath = os.path.join(patched, "adapter_model.safetensors")
        weights = load_file(wpath)
        remapped = {k.replace("model.unembed_tokens", "lm_head"): v for k, v in weights.items()}
        modules = set()
        for k in remapped:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p in ("lora_A", "lora_B") and i > 0:
                    modules.add(parts[i - 1])
        save_file(remapped, wpath)
        cfg_path = os.path.join(patched, "adapter_config.json")
        cfg = json.loads(open(cfg_path).read())
        cfg["target_modules"] = sorted(modules)
        if not cfg.get("base_model_name_or_path"):
            cfg["base_model_name_or_path"] = args.model
        open(cfg_path, "w").write(json.dumps(cfg))
        print(f"[adapter] remapped modules: {sorted(modules)}", flush=True)
        lora_kwargs = {"enable_lora": True, "max_lora_rank": 64}
        lora_request = LoRARequest("postrl", 1, patched)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=max(2048, args.max_tokens + 640),
        gpu_memory_utilization=0.92,
        enforce_eager=True,
        **lora_kwargs,
    )
    ks = [k for k in args.ks if k <= args.n]
    result = previous or {
        "kind": "passk_eval_standalone",
        "status": "started",
        "backend": "standalone-vllm",
        "vllm_version": vllm.__version__,
        "grader": {
            "gsm8k": "tinker-identical boxed/number parser",
            "math": "strict normalized-boxed match (LOWER BOUND)",
            "mbpp": "sanitized 3-assert execution, 10s timeout",
        }[grade_mode],
        "prompt_fingerprints": prompt_fingerprints,
        "per_problem_c": [],
        "started_at_unix": time.time(),
    }
    result.update(expected)
    result["status"] = "started"
    result["resumed"] = previous is not None
    result["completed_problem_count"] = len(result["per_problem_c"])
    write_checkpoint(output_path, result)

    cs = result["per_problem_c"]
    run_started = time.time()
    for start in range(len(cs), len(items), args.checkpoint_every):
        end = min(start + args.checkpoint_every, len(items))
        chunk = items[start:end]
        # A per-chunk seed makes a resumed run identical to an uninterrupted
        # run with the same checkpoint size.
        sp = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed + start,
        )
        prompts = [prompt for prompt, _ in chunk]
        outs = (
            llm.generate(prompts, sp, lora_request=lora_request)
            if lora_request
            else llm.generate(prompts, sp)
        )

        chunk_counts: list[int] = []
        if grade_mode == "mbpp":
            jobs, spans = [], []
            for (_, tests), out in zip(chunk, outs):
                job_start = len(jobs)
                jobs.extend((extract_code(o.text), tests) for o in out.outputs)
                spans.append((job_start, len(jobs)))
            with mp.Pool(min(8, max(1, len(jobs)))) as pool:
                flags = pool.map(_run_candidate, jobs)
            chunk_counts = [sum(flags[a:b]) for a, b in spans]
        else:
            fn = gsm8k_reward if grade_mode == "gsm8k" else math_reward
            for (_, answer), out in zip(chunk, outs):
                chunk_counts.append(sum(int(fn(o.text, answer)) for o in out.outputs))

        cs.extend(chunk_counts)
        result["completed_problem_count"] = len(cs)
        result["last_checkpoint_unix"] = time.time()
        write_checkpoint(output_path, result)
        print(f"[checkpoint] {len(cs)}/{len(items)} -> {output_path}", flush=True)

    result.update(
        {
            "status": "complete",
            "ks": ks,
            "pass_at_k": {
                str(k): round(sum(pass_at_k(args.n, c, k) for c in cs) / len(cs), 4) for k in ks
            },
            "wall_seconds": round(time.time() - t0, 1),
            "resume_wall_seconds": round(time.time() - run_started, 1),
        }
    )
    write_checkpoint(output_path, result)
    print(f"pass@k = {result['pass_at_k']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
