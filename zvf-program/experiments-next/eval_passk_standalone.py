#!/usr/bin/env python3
"""eval_passk_standalone.py — portable pass@k evaluator (no Modal deps).

Runs anywhere with a GPU + vllm (Lightning Studio, Colab, bare metal).
Datasets: gsm8k (tinker-identical parser), math500 (strict boxed match,
lower bound), mbpp (unit-test execution — sandboxed subprocess w/ timeout;
run only on an isolated VM).

  python eval_passk_standalone.py --dataset mbpp --problems 200 --n 32 \
      --out passk_lightning_qwen3-8b_base_mbpp.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
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


def chatml(system: str, user: str) -> str:
    return ("<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + user + "<|im_end|>\n"
            "<|im_start|>assistant\n")


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
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=10)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["gsm8k", "math500", "mbpp"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
                items.append((chatml(SYSTEM_PROMPT_MATH, row["question"] +
                              " Provide the final numerical answer inside \\boxed{}."),
                              m.group(1).replace(",", "").strip()))
        grade_mode = "gsm8k"
    elif args.dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [(chatml(SYSTEM_PROMPT_MATH, row["problem"] +
                  " Put your final answer inside \\boxed{}."), row["answer"])
                 for row in ds]
        grade_mode = "math"
    else:  # mbpp: sanitized test split, standard 3-assert protocol
        ds = load_dataset("google-research-datasets/mbpp", "sanitized",
                          split="test")
        items = []
        for row in ds:
            user = (row["prompt"] + "\nYour code must pass these tests:\n" +
                    "\n".join(row["test_list"]))
            items.append((chatml(SYSTEM_PROMPT_CODE, user), row["test_list"]))
        grade_mode = "mbpp"

    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[:args.problems] if args.problems else items

    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              max_model_len=2048, gpu_memory_utilization=0.92,
              enforce_eager=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature,
                        top_p=args.top_p, max_tokens=args.max_tokens,
                        seed=args.seed)
    outs = llm.generate([p for p, _ in items], sp)

    cs = []
    if grade_mode == "mbpp":
        jobs, spans = [], []
        for (prompt, tests), out in zip(items, outs):
            start = len(jobs)
            for o in out.outputs:
                jobs.append((extract_code(o.text), tests))
            spans.append((start, len(jobs)))
        with mp.Pool(8) as pool:
            flags = pool.map(_run_candidate, jobs)
        cs = [sum(flags[a:b]) for a, b in spans]
    else:
        fn = gsm8k_reward if grade_mode == "gsm8k" else math_reward
        for (prompt, ans), out in zip(items, outs):
            cs.append(sum(int(fn(o.text, ans)) for o in out.outputs))

    ks = [k for k in args.ks if k <= args.n]
    result = {
        "kind": "passk_eval_standalone",
        "status": "complete",
        "backend": "standalone-vllm",
        "vllm_version": vllm.__version__,
        "model": args.model,
        "dataset": args.dataset,
        "grader": {"gsm8k": "tinker-identical boxed/number parser",
                   "math": "strict normalized-boxed match (LOWER BOUND)",
                   "mbpp": "sanitized 3-assert execution, 10s timeout"}[grade_mode],
        "n_problems": len(items),
        "n_per_problem": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "per_problem_c": cs,
        "pass_at_k": {str(k): round(sum(pass_at_k(args.n, c, k) for c in cs)
                                    / len(cs), 4) for k in ks},
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"pass@k = {result['pass_at_k']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
