"""Held-out GSM8K pass@k on the Colab GPU with vLLM (base Qwen3-8B and PEFT LoRA adapters).

Protocol = zvf-program/experiments-next/modal_passk.py / er2b_heldout_eval/scripts/tinker_passk.py:
GSM8K test, random.Random(42).shuffle, first 200; 32 samples per problem, temperature 1.0,
top_p 1.0, max_tokens 512; raw ChatML build_prompt + GSM8K_SUFFIX; gsm8k_reward (parser v2, last
boxed else last number); unbiased pass_at_k. Like tinker_passk, every sample is its own request
with its own seed (SEED*1_000_003 + i*n + j); distinctness of the 32 samples is checked and recorded.

  python er2b_vllm_passk.py OUT_RAW_DIR name=ADAPTER_DIR_or_base [name=...] [--n-problems N --n-samples N]
"""

import argparse, json, os, random, re, sys, time

# ---- verbatim from modal_passk.py ----
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
# ---- end verbatim ----

MODEL = "Qwen/Qwen3-8B"
TEMPERATURE, TOP_P, MAX_TOKENS, SEED, KS = 1.0, 1.0, 512, 42, (1, 8, 32)


def load_items(n_problems):
    from datasets import load_dataset

    items = []
    for row in load_dataset("openai/gsm8k", "main", split="test"):
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if m:
            items.append((build_prompt(row["question"], GSM8K_SUFFIX), m.group(1).replace(",", "").strip()))
    random.Random(SEED).shuffle(items)
    return items[:n_problems]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    todo = []
    for spec in a.models:
        name, path = spec.split("=", 1)
        if os.path.exists(f"{a.out_dir}/{name}.json"):
            print("skip (done)", name, flush=True)
            continue
        todo.append((name, None if path == "base" else path))
    if not todo:
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm, torch

    items = load_items(a.n_problems)
    n = a.n_samples
    ks = [k for k in KS if k <= n]
    llm = LLM(model=MODEL, dtype="bfloat16", seed=SEED, max_model_len=2048, gpu_memory_utilization=0.88,
              enable_lora=True, max_lora_rank=8, max_loras=1, enable_prefix_caching=True)
    for li, (name, adapter) in enumerate(todo, start=1):
        t0 = time.time()
        lora = LoRARequest(name, li, adapter) if adapter else None
        prompts, params = [], []
        for i, (p, _) in enumerate(items):
            for j in range(n):
                prompts.append(p)
                params.append(SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS,
                                             seed=(SEED * 1_000_003 + i * n + j) % (2**31 - 1)))
        outs = llm.generate(prompts, params, lora_request=lora)
        per_c, per_unique, trunc, toks, boxed = [], [], 0, 0, 0
        for i, (_, ans) in enumerate(items):
            group = [o.outputs[0] for o in outs[i * n : (i + 1) * n]]
            per_c.append(sum(int(gsm8k_reward(g.text, ans)) for g in group))
            per_unique.append(len({g.text for g in group}))
            for g in group:
                trunc += int(g.finish_reason == "length")
                toks += len(g.token_ids)
                boxed += int("\\boxed" in g.text)
        tot = len(items) * n
        res = {
            "kind": "passk_eval_colab_vllm",
            "backend": "colab-vllm",
            "gpu": torch.cuda.get_device_name(0),
            "vllm_version": vllm.__version__,
            "model": MODEL,
            "adapter": adapter,
            "dataset": "gsm8k",
            "grader": "modal_passk.gsm8k_reward (parser v2: last boxed, else last number)",
            "n_problems": len(items),
            "n_per_problem": n,
            "ks": ks,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "seed": SEED,
            "sample_seeds": "SEED*1_000_003 + i*n + j, one request per sample",
            "max_tokens": MAX_TOKENS,
            "per_problem_c": per_c,
            "per_problem_unique_texts": per_unique,
            "min_unique_texts_per_problem": min(per_unique),
            "pass_at_k": {str(k): round(sum(pass_at_k(n, c, k) for c in per_c) / len(per_c), 4) for k in ks},
            "truncation_rate": round(trunc / tot, 6),
            "mean_tokens": round(toks / tot, 4),
            "boxed_rate": round(boxed / tot, 4),
            "wall_seconds": round(time.time() - t0, 1),
            "tag": name,
        }
        json.dump(res, open(f"{a.out_dir}/{name}.json", "w"), indent=1)
        print("EVAL", name, res["pass_at_k"], "trunc", res["truncation_rate"], "tok", res["mean_tokens"],
              "min_unique", res["min_unique_texts_per_problem"], "s", res["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
