#!/usr/bin/env python3
"""
Deterministic evaluation suite for reasoning benchmarks.

This script is for claim-support evaluation after training, especially when the
question is whether a checkpoint generalizes beyond GSM8K saturation.

Supported benchmarks
--------------------
- gsm8k
- gsm1k
- gsm_symbolic_main
- gsm_symbolic_p1
- gsm_symbolic_p2
- math
- olympiadbench

The script talks to any OpenAI-compatible chat endpoint, including the local
`serve.py` server in this repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import aiohttp
from datasets import concatenate_datasets, load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from tinker_atropos.config import TinkerAtroposConfig


GSM_QUESTION_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."
MATH_QUESTION_SUFFIX = " Provide your answer inside \\boxed{}."

GSM_PREFIX = [
    {
        "role": "user",
        "content": "How many r's are in strawberry?" + GSM_QUESTION_SUFFIX,
    },
    {
        "role": "assistant",
        "content": (
            "Let's spell the word out and number all the letters: "
            "1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. "
            "We have r's at positions 3, 8, and 9. \\boxed{3}"
        ),
    },
]

MATH_PREFIX = [
    {
        "role": "user",
        "content": (
            "What is the sum of all positive integers n such that n^2 divides 12!?"
            + MATH_QUESTION_SUFFIX
        ),
    },
    {
        "role": "assistant",
        "content": (
            "We need n^2 | 12!. First, 12! = 2^10 * 3^5 * 5^2 * 7 * 11. "
            "For n^2 | 12!, each prime in n can appear at most floor(exponent/2) times. "
            "So n divides 2^5 * 3^2 * 5 = 1440. The sum of divisors of 1440 = "
            "(1+2+4+8+16+32)(1+3+9)(1+5) = 63 * 13 * 6 = 4914. \\boxed{4914}"
        ),
    },
]

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass
class Example:
    benchmark: str
    prompt: str
    gold_answers: List[str]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic reasoning benchmark evaluation against an OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Optional Tinker-Atropos YAML config. Used to infer model/base_url/max_tokens/prefix defaults.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["gsm8k"],
        help=(
            "Benchmarks to run. Supported: gsm8k gsm1k gsm_symbolic_main gsm_symbolic_p1 "
            "gsm_symbolic_p2 math olympiadbench all"
        ),
    )
    parser.add_argument("--base-url", type=str, help="OpenAI-compatible base URL, e.g. http://localhost:8001/v1")
    parser.add_argument("--model", type=str, help="Model name passed to the endpoint")
    parser.add_argument("--api-key", type=str, default="x", help="API key for the endpoint")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override generation max_tokens. Falls back to config env.max_token_length or 512.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Evaluation temperature. Keep at 0.0 for deterministic scoring.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrent requests per batch.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap per benchmark for smoke tests.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Sampling seed used when max_examples truncates a benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/reasoning_eval",
        help="Directory for per-benchmark summaries and prediction jsonl files.",
    )
    parser.add_argument(
        "--use-prefix",
        dest="use_prefix",
        action="store_true",
        default=None,
        help="Force benchmark-specific worked-example prefix on.",
    )
    parser.add_argument(
        "--no-prefix",
        dest="use_prefix",
        action="store_false",
        help="Force worked-example prefix off.",
    )
    parser.add_argument(
        "--include-olympiad-multi-answer",
        action="store_true",
        help="Include OlympiadBench multi-answer items instead of the default single-answer subset.",
    )
    return parser.parse_args()


def _clean_gold_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _extract_gsm_answer(answer: str) -> str:
    if "####" in answer:
        answer = answer.split("####")[-1]
    return answer.strip().replace(",", "")


def _extract_math_boxed_answer(solution: str) -> str:
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return _clean_gold_answer(solution)

    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution)):
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            if depth == 0:
                return _clean_gold_answer(solution[start:i])
            depth -= 1
    return _clean_gold_answer(solution[start:])


def _gold_parse(answer: str):
    return parse(
        "\\boxed{" + _clean_gold_answer(answer) + "}",
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
    )


def _response_parse(response_text: str):
    return parse(
        response_text.split("</think>")[-1],
        extraction_mode="first_match",
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False,
                    malformed_operators=False,
                    basic_latex=True,
                    boxed="all",
                    units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
    )


def _is_correct(response_text: str, gold_answers: Iterable[str]) -> bool:
    pred = _response_parse(response_text)
    for answer in gold_answers:
        gold = _gold_parse(answer)
        if verify(pred, gold):
            return True
    return False


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return base_url


def _sample_examples(examples: List[Example], max_examples: Optional[int], seed: int) -> List[Example]:
    if max_examples is None or len(examples) <= max_examples:
        return examples
    rng = random.Random(seed)
    sampled = list(examples)
    rng.shuffle(sampled)
    return sampled[:max_examples]


def load_gsm8k_examples(max_examples: Optional[int], seed: int) -> List[Example]:
    dataset = load_dataset("gsm8k", "main", split="test")
    examples = [
        Example(
            benchmark="gsm8k",
            prompt=row["question"],
            gold_answers=[_extract_gsm_answer(row["answer"])],
            metadata={"source": "gsm8k", "id": idx},
        )
        for idx, row in enumerate(dataset)
    ]
    return _sample_examples(examples, max_examples, seed)


def load_gsm1k_examples(max_examples: Optional[int], seed: int) -> List[Example]:
    dataset = load_dataset("ScaleAI/gsm1k", split="test")
    examples = [
        Example(
            benchmark="gsm1k",
            prompt=row["question"],
            gold_answers=[_clean_gold_answer(row["answer"]).replace(",", "")],
            metadata={"source": "ScaleAI/gsm1k", "id": idx},
        )
        for idx, row in enumerate(dataset)
    ]
    return _sample_examples(examples, max_examples, seed)


def load_gsm_symbolic_examples(variant: str, max_examples: Optional[int], seed: int) -> List[Example]:
    dataset = load_dataset("apple/GSM-Symbolic", variant, split="test")
    examples = [
        Example(
            benchmark=f"gsm_symbolic_{variant}",
            prompt=row["question"],
            gold_answers=[_extract_gsm_answer(row["answer"])],
            metadata={
                "source": "apple/GSM-Symbolic",
                "variant": variant,
                "id": row["id"],
                "instance": row["instance"],
                "original_id": row["original_id"],
            },
        )
        for row in dataset
    ]
    return _sample_examples(examples, max_examples, seed)


def load_math_examples(max_examples: Optional[int], seed: int) -> List[Example]:
    train_splits = []
    for subject in MATH_SUBJECTS:
        train_splits.append(load_dataset("EleutherAI/hendrycks_math", subject, split="test"))
    dataset = concatenate_datasets(train_splits)
    dataset = dataset.shuffle(seed=seed)
    examples = [
        Example(
            benchmark="math",
            prompt=row["problem"],
            gold_answers=[_extract_math_boxed_answer(row["solution"])],
            metadata={"source": "EleutherAI/hendrycks_math", "subject": row.get("type"), "id": idx},
        )
        for idx, row in enumerate(dataset)
    ]
    return _sample_examples(examples, max_examples, seed)


def load_olympiadbench_examples(
    max_examples: Optional[int],
    seed: int,
    include_multi_answer: bool,
) -> List[Example]:
    dataset = load_dataset("math-ai/olympiadbench", split="test")
    filtered = []
    for row in dataset:
        if row["subject"] != "Math":
            continue
        if row["language"] != "English":
            continue
        if row["modality"] != "Text-only":
            continue
        if (not include_multi_answer) and row["is_multiple_answer"]:
            continue
        gold_answers = [_clean_gold_answer(ans) for ans in row["final_answer"] if ans]
        if not gold_answers:
            continue
        filtered.append(
            Example(
                benchmark="olympiadbench",
                prompt=row["question"],
                gold_answers=gold_answers,
                metadata={
                    "source": "math-ai/olympiadbench",
                    "id": row["id"],
                    "answer_type": row["answer_type"],
                    "difficulty": row["difficulty"],
                    "is_multiple_answer": row["is_multiple_answer"],
                },
            )
        )
    return _sample_examples(filtered, max_examples, seed)


def load_examples(
    benchmark: str,
    max_examples: Optional[int],
    seed: int,
    include_olympiad_multi_answer: bool,
) -> List[Example]:
    if benchmark == "gsm8k":
        return load_gsm8k_examples(max_examples, seed)
    if benchmark == "gsm1k":
        return load_gsm1k_examples(max_examples, seed)
    if benchmark == "gsm_symbolic_main":
        return load_gsm_symbolic_examples("main", max_examples, seed)
    if benchmark == "gsm_symbolic_p1":
        return load_gsm_symbolic_examples("p1", max_examples, seed)
    if benchmark == "gsm_symbolic_p2":
        return load_gsm_symbolic_examples("p2", max_examples, seed)
    if benchmark == "math":
        return load_math_examples(max_examples, seed)
    if benchmark == "olympiadbench":
        return load_olympiadbench_examples(max_examples, seed, include_olympiad_multi_answer)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def benchmark_prefix(benchmark: str, use_prefix: bool) -> List[Dict[str, str]]:
    if not use_prefix:
        return []
    if benchmark.startswith("gsm"):
        return list(GSM_PREFIX)
    return list(MATH_PREFIX)


def benchmark_suffix(benchmark: str) -> str:
    if benchmark.startswith("gsm"):
        return GSM_QUESTION_SUFFIX
    return MATH_QUESTION_SUFFIX


async def fetch_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    api_key: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    url = _normalize_base_url(base_url) + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(url, headers=headers, json=payload) as response:
        raw_text = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {raw_text[:500]}")
        data = json.loads(raw_text)
        return data["choices"][0]["message"]["content"]


async def evaluate_example(
    session: aiohttp.ClientSession,
    example: Example,
    base_url: str,
    model_name: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    use_prefix: bool,
) -> Dict[str, Any]:
    messages = [
        *benchmark_prefix(example.benchmark, use_prefix),
        {"role": "user", "content": example.prompt + benchmark_suffix(example.benchmark)},
    ]
    started = time.time()
    try:
        response_text = await fetch_completion(
            session=session,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        correct = _is_correct(response_text, example.gold_answers)
        error = None
    except Exception as exc:
        response_text = ""
        correct = False
        error = str(exc)

    return {
        "benchmark": example.benchmark,
        "correct": correct,
        "error": error,
        "latency_sec": round(time.time() - started, 4),
        "gold_answers": example.gold_answers,
        "response": response_text,
        "metadata": example.metadata,
        "question": example.prompt,
    }


async def evaluate_benchmark(
    benchmark: str,
    examples: List[Example],
    base_url: str,
    model_name: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    use_prefix: bool,
) -> List[Dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=600)
    connector = aiohttp.TCPConnector(limit=max(1, concurrency))

    print(f"\n=== Evaluating {benchmark}: {len(examples)} examples ===")
    results: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for start in range(0, len(examples), concurrency):
            batch = examples[start : start + concurrency]
            batch_results = await asyncio.gather(
                *[
                    evaluate_example(
                        session=session,
                        example=example,
                        base_url=base_url,
                        model_name=model_name,
                        api_key=api_key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        use_prefix=use_prefix,
                    )
                    for example in batch
                ]
            )
            results.extend(batch_results)
            done = len(results)
            correct = sum(1 for r in results if r["correct"])
            print(f"  {benchmark}: {done}/{len(examples)} done | accuracy={correct / done:.4f}")
    return results


def summarize_results(
    benchmark: str,
    results: List[Dict[str, Any]],
    model_name: str,
    base_url: str,
    max_tokens: int,
    temperature: float,
    use_prefix: bool,
) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r["error"])
    return {
        "benchmark": benchmark,
        "total_examples": total,
        "num_correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "num_request_errors": errors,
        "model_name": model_name,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "use_prefix": use_prefix,
        "timestamp": int(time.time()),
    }


def write_outputs(
    output_dir: Path,
    benchmark: str,
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{benchmark}_summary.json"
    preds_path = output_dir / f"{benchmark}_predictions.jsonl"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    with open(preds_path, "w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote summary: {summary_path}")
    print(f"  Wrote predictions: {preds_path}")


def expand_benchmarks(benchmarks: List[str]) -> List[str]:
    supported = [
        "gsm8k",
        "gsm1k",
        "gsm_symbolic_main",
        "gsm_symbolic_p1",
        "gsm_symbolic_p2",
        "math",
        "olympiadbench",
    ]
    if benchmarks == ["all"]:
        return supported
    invalid = [b for b in benchmarks if b not in supported]
    if invalid:
        raise ValueError(f"Unsupported benchmarks: {invalid}. Supported: {supported}")
    return benchmarks


def resolve_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    config = None
    if args.config:
        config = TinkerAtroposConfig.from_yaml(args.config)

    model_name = args.model or (config.openai[0].model_name if config else None)
    base_url = args.base_url or (config.openai[0].base_url if config else None)
    api_key = args.api_key or (config.openai[0].api_key if config else "x")
    max_tokens = args.max_tokens or (config.max_token_env_length if config else 512)

    if model_name is None or base_url is None:
        raise ValueError("Provide either --config or both --model and --base-url.")

    use_prefix = args.use_prefix
    if use_prefix is None:
        use_prefix = config.use_prompt_prefix if config else True

    return {
        "model_name": model_name,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "use_prefix": use_prefix,
    }


async def async_main() -> None:
    args = parse_args()
    runtime = resolve_runtime(args)
    benchmarks = expand_benchmarks(args.benchmarks)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Reasoning Evaluation Suite")
    print("=" * 80)
    print(f"Model:       {runtime['model_name']}")
    print(f"Endpoint:    {runtime['base_url']}")
    print(f"Benchmarks:  {', '.join(benchmarks)}")
    print(f"Max tokens:  {runtime['max_tokens']}")
    print(f"Temperature: {args.temperature}")
    print(f"Use prefix:  {runtime['use_prefix']}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Max examples:{args.max_examples if args.max_examples is not None else 'full'}")
    print("=" * 80)

    combined_summary = []
    for benchmark in benchmarks:
        examples = load_examples(
            benchmark=benchmark,
            max_examples=args.max_examples,
            seed=args.sample_seed,
            include_olympiad_multi_answer=args.include_olympiad_multi_answer,
        )
        results = await evaluate_benchmark(
            benchmark=benchmark,
            examples=examples,
            base_url=runtime["base_url"],
            model_name=runtime["model_name"],
            api_key=runtime["api_key"],
            max_tokens=runtime["max_tokens"],
            temperature=args.temperature,
            concurrency=args.concurrency,
            use_prefix=runtime["use_prefix"],
        )
        summary = summarize_results(
            benchmark=benchmark,
            results=results,
            model_name=runtime["model_name"],
            base_url=runtime["base_url"],
            max_tokens=runtime["max_tokens"],
            temperature=args.temperature,
            use_prefix=runtime["use_prefix"],
        )
        write_outputs(output_dir, benchmark, summary, results)
        combined_summary.append(summary)
        print(
            f"  {benchmark}: accuracy={summary['accuracy']:.4f} "
            f"({summary['num_correct']}/{summary['total_examples']})"
        )

    combined_path = output_dir / "combined_summary.json"
    with open(combined_path, "w") as f:
        json.dump(combined_summary, f, indent=2, sort_keys=True)
    print(f"\nWrote combined summary: {combined_path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
