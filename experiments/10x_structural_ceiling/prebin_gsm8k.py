"""Pre-bin GSM8K prompts by difficulty — FAST version.

Optimizations vs v1:
  - 200 prompts (not 500) — enough for statistical binning
  - 4 samples per prompt (not 8) — sufficient for easy/mid/hard
  - Concurrent requests — fire batches of 16 in parallel via futures
  - Total: 800 calls in ~16 parallel batches ≈ 10-15 min
"""
from __future__ import annotations
import json, os, random, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tinker
import tinker.types as T
from transformers import AutoTokenizer


def load_gsm8k_prompts(tokenizer, max_examples=200):
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
        })
    return examples


def reward_gsm8k(response: str, answer: str) -> float:
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


def sample_one_prompt(sc, tok, ex, idx, num_samples=4):
    """Sample one prompt and return (index, success_rate). Thread-safe."""
    prompt_mi = T.ModelInput.from_ints(ex["prompt_ids"])
    sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
    responses = sc.sample(prompt_mi, num_samples=num_samples, sampling_params=sp).result()
    successes = sum(
        1 for resp in responses.sequences
        if reward_gsm8k(tok.decode(list(resp.tokens), skip_special_tokens=True), ex["answer"]) > 0.5
    )
    return idx, successes / num_samples


def main():
    model_name = "Qwen/Qwen3-8B"
    num_samples = 4
    max_prompts = 200
    parallel = 16  # concurrent requests
    out_path = Path("./phase_bins.json")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    examples = load_gsm8k_prompts(tok, max_examples=max_prompts)
    print(f"Loaded {len(examples)} prompts")

    svc = tinker.ServiceClient(base_url=None)
    tc = svc.create_lora_training_client(base_model=model_name, rank=32)
    sc = tc.save_weights_and_get_sampling_client()
    print(f"Sampler ready — binning {len(examples)} prompts ({num_samples} samples each, {parallel} concurrent)")

    t0 = time.time()
    rates = [0.0] * len(examples)
    done = 0

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(sample_one_prompt, sc, tok, ex, i, num_samples): i
            for i, ex in enumerate(examples)
        }
        for future in as_completed(futures):
            idx, rate = future.result()
            rates[idx] = rate
            done += 1
            if done % 25 == 0:
                elapsed = time.time() - t0
                print(f"  {done}/{len(examples)} binned ({elapsed:.0f}s elapsed)")

    # Classify
    bins = {"easy": {"indices": [], "rates": []},
            "mid": {"indices": [], "rates": []},
            "hard": {"indices": [], "rates": []}}

    for i, rate in enumerate(rates):
        if rate > 0.75:
            b = "easy"
        elif rate >= 0.25:
            b = "mid"
        else:
            b = "hard"
        bins[b]["indices"].append(i)
        bins[b]["rates"].append(rate)

    for b in bins:
        bins[b]["count"] = len(bins[b]["indices"])

    result = {
        "model": model_name,
        "num_samples": num_samples,
        "total_prompts": len(examples),
        "bins": bins,
        "overall_mean_rate": sum(rates) / len(rates),
        "elapsed_seconds": time.time() - t0,
    }

    out_path.write_text(json.dumps(result, indent=2))
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — saved to {out_path}")
    print(f"  easy: {bins['easy']['count']} (>75% success)")
    print(f"  mid:  {bins['mid']['count']} (25-75% success)")
    print(f"  hard: {bins['hard']['count']} (<25% success)")
    print(f"  overall mean success rate: {result['overall_mean_rate']:.3f}")


if __name__ == "__main__":
    main()
