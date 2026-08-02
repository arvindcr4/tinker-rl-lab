#!/usr/bin/env python3
"""
ArenaHard evaluation harness.

Generates model responses for the ArenaHard benchmark and saves them in the format
required by Arena-Hard-Auto for GPT-4 judging.

LIMITATION: limitations from adversarial review:
- Single-Seed Extrapolations: We currently only generate one completion (n=1) and default to temperature 0.0. We should support multiple seeds/runs (N>1) to estimate variance and avoid statistical vulnerability.
- Failure to Prove Generalization: This generation script does not compute statistical significance. The downstream judging pipeline must be updated to calculate p-values against base models to prove true reasoning emergence.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from datasets import load_dataset

from tinker_atropos.config import TinkerAtroposConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArenaHard generation harness.")
    parser.add_argument("--config", type=str, help="Optional Tinker-Atropos YAML config.")
    parser.add_argument("--base-url", type=str, help="OpenAI-compatible base URL")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--api-key", type=str, default="x", help="API key")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrency")
    parser.add_argument("--max-examples", type=int, default=None, help="Max examples to run")
    parser.add_argument("--output-dir", type=str, default="logs/arenahard", help="Output directory")
    return parser.parse_args()


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return base_url


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
    # TODO: Fix Single-Seed Extrapolations limitation.
    # Hardcoding n=1 prevents capturing variance. We should accept `n` as an argument
    # or run this harness multiple times to allow N>1 runs for statistical significance testing.
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


async def generate_example(
    session: aiohttp.ClientSession,
    question_id: str,
    turns: List[Dict[str, str]],
    base_url: str,
    model_name: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, Any]:
    started = time.time()
    try:
        response_text = await fetch_completion(
            session=session,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            messages=turns,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        error = None
    except Exception as exc:
        response_text = ""
        error = str(exc)

    return {
        "question_id": question_id,
        "model_id": model_name,
        "choices": [{"index": 0, "turns": [response_text]}],
        "error": error,
        "latency_sec": round(time.time() - started, 4),
    }


async def main_async():
    args = parse_args()
    
    config = None
    if args.config:
        config = TinkerAtroposConfig.from_yaml(args.config)

    model_name = args.model or (config.openai[0].model_name if config else None)
    base_url = args.base_url or (config.openai[0].base_url if config else None)
    api_key = args.api_key or (config.openai[0].api_key if config else "x")
    max_tokens = args.max_tokens or (config.max_token_env_length if config else 4096)
    
    if model_name is None or base_url is None:
        raise ValueError("Provide either --config or both --model and --base-url.")
        
    print(f"Loading ArenaHard dataset...")
    dataset = load_dataset("lm-sys/arena-hard-auto-v0.1", split="test")
    
    examples = []
    for row in dataset:
        examples.append({
            "question_id": row["question_id"],
            "turns": row["turns"],
        })
        
    if args.max_examples is not None:
        examples = examples[:args.max_examples]
        
    print(f"Loaded {len(examples)} examples.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name.replace('/', '_')}_predictions.jsonl"
    
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=600)
    connector = aiohttp.TCPConnector(limit=max(1, args.concurrency))
    
    results = []
    errors = 0
    
    print(f"Generating completions for model {model_name}...")
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for start in range(0, len(examples), args.concurrency):
            batch = examples[start:start+args.concurrency]
            tasks = [
                generate_example(
                    session=session,
                    question_id=ex["question_id"],
                    turns=ex["turns"],
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                    max_tokens=max_tokens,
                    temperature=args.temperature,
                )
                for ex in batch
            ]
            batch_results = await asyncio.gather(*tasks)
            for res in batch_results:
                if res["error"]:
                    errors += 1
            results.extend(batch_results)
            print(f"  Done: {len(results)}/{len(examples)}")
            
    with open(out_path, "w") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            
    print(f"Finished generation with {errors} errors. Outputs saved to {out_path}.")

if __name__ == "__main__":
    asyncio.run(main_async())
