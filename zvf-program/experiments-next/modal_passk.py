#!/usr/bin/env python3
"""modal_passk.py — pass@k evaluation harness on Modal GPUs (vLLM).

Replaces the billing-blocked Tinker path for OPEN-WEIGHT evaluations:
base models from HF, optionally with a LoRA adapter repo (the 21
arvindcr4/tinker-rl-* adapter repos), on GSM8K-test or MATH-500.

Stack notes (MIN-REPORT-RL item 3): prompt template and GSM8K reward parser
are byte-identical to the Tinker harness (common.py / live_zvf_probe.py).
The sampler differs (vLLM vs Tinker) BY DESIGN — running the same config on
both backends measures the sampler stack effect directly. MATH-500 grading
is strict normalized-boxed match: a LOWER BOUND on accuracy (equivalent
forms not credited); recorded as such in the output JSON.

Usage (from repo root or this directory):
  modal run modal_passk.py --dataset math500 --problems 8 --n 4      # smoke
  modal run modal_passk.py --dataset math500 --problems 500 --n 32   # full
  modal run modal_passk.py --dataset gsm8k --problems 200 --n 32     # x-check
  modal run modal_passk.py --dataset math500 --adapter arvindcr4/tinker-rl-w2_qwen3-8b_g8-qwen3-8b-s42

Results: written to results/passk_modal_*.json locally (backfilled to W&B
by wandb_backfill.py) and mirrored to the modal volume zvf-passk-results.
"""

from __future__ import annotations

import json
import math
import re
import time

import modal

APP_NAME = "zvf-passk"
app = modal.App(APP_NAME)
results_vol = modal.Volume.from_name("zvf-passk-results", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04",
                              add_python="3.12")
    .pip_install("vllm", "datasets", "huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1",
          "CUDA_HOME": "/usr/local/cuda"})
)

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
GSM8K_SUFFIX = " Provide the final numerical answer inside \\boxed{}."
MATH_SUFFIX = " Put your final answer inside \\boxed{}."


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


def norm_math(ans: str) -> str:
    """Normalize a MATH boxed answer for strict comparison (lower bound)."""
    a = ans.strip()
    a = a.replace("\\left", "").replace("\\right", "")
    a = a.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace(" ", "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac")
    a = a.rstrip(".")
    if a.startswith("{") and a.endswith("}"):
        a = a[1:-1]
    return a


def extract_last_boxed(text: str) -> str | None:
    """Balanced-brace extraction of the LAST \\boxed{...} in text."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out) if depth == 0 else None


def math_reward(response: str, answer: str) -> float:
    got = extract_last_boxed(response)
    if got is None:
        return 0.0
    return 1.0 if norm_math(got) == norm_math(answer) else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    r = 1.0
    for i in range(k):
        r *= (n - c - i) / (n - i)
    return 1.0 - r


@app.function(image=image, gpu="A100-40GB", timeout=10800,
              volumes={"/results": results_vol},
              secrets=[modal.Secret.from_name("hf-token")])
def evaluate(model: str, dataset: str, problems: int, n: int, ks: list[int],
             temperature: float, top_p: float, max_tokens: int, seed: int,
             adapter: str | None) -> dict:
    import random

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    import vllm

    t0 = time.time()
    if dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = []
        for row in ds:
            m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
            if m:
                items.append((build_prompt(row["question"], GSM8K_SUFFIX),
                              m.group(1).replace(",", "").strip()))
        reward = gsm8k_reward
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [(build_prompt(row["problem"], MATH_SUFFIX), row["answer"])
                 for row in ds]
        reward = math_reward
    else:
        raise ValueError(f"unknown dataset {dataset}")

    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:problems] if problems else items

    lora_kwargs = {}
    lora_request = None
    if adapter:
        import os
        import shutil

        from vllm.lora.request import LoRARequest
        from huggingface_hub import snapshot_download
        adapter_path = snapshot_download(adapter)
        # adapters pushed by parallel_push_hf keep weights under final/
        if os.path.exists(os.path.join(adapter_path, "final",
                                       "adapter_model.safetensors")):
            adapter_path = os.path.join(adapter_path, "final")
        # Tinker exports differ from PEFT conventions in two ways vLLM
        # rejects: target_modules="all-linear" (shorthand) and the LM head
        # trained under the name "model.unembed_tokens" (vLLM expects
        # "lm_head"). Patch a local copy: remap weight keys, then derive
        # target_modules from what the weights actually contain.
        from safetensors.torch import load_file, save_file
        patched = "/tmp/adapter_patched"
        shutil.copytree(adapter_path, patched, dirs_exist_ok=True,
                        symlinks=False)
        wpath = os.path.join(patched, "adapter_model.safetensors")
        weights = load_file(wpath)
        remapped = {}
        for k, v in weights.items():
            remapped[k.replace("model.unembed_tokens", "lm_head")] = v
        modules = set()
        for k in remapped:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p in ("lora_A", "lora_B") and i > 0:
                    modules.add(parts[i - 1])
        save_file(remapped, wpath)
        cfg_path = os.path.join(patched, "adapter_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["target_modules"] = sorted(modules)
        if not cfg.get("base_model_name_or_path"):
            cfg["base_model_name_or_path"] = model
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        print(f"[adapter] remapped modules: {sorted(modules)}", flush=True)
        lora_kwargs = {"enable_lora": True, "max_lora_rank": 64}
        lora_request = LoRARequest("postrl", 1, patched)

    llm = LLM(model=model, dtype="bfloat16", seed=seed,
              max_model_len=4608, gpu_memory_utilization=0.90,
              enforce_eager=True, **lora_kwargs)
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                        max_tokens=max_tokens, seed=seed)
    prompts = [p for p, _ in items]
    outs = llm.generate(prompts, sp, lora_request=lora_request) \
        if lora_request else llm.generate(prompts, sp)

    cs = []
    boxed_hits = 0
    total_outs = 0
    for (prompt, ans), out in zip(items, outs):
        c = sum(int(reward(o.text, ans)) for o in out.outputs)
        cs.append(c)
        for o in out.outputs:
            total_outs += 1
            boxed_hits += int("\\boxed" in o.text)

    result = {
        "kind": "passk_eval_modal",
        "status": "complete",
        "backend": "modal-vllm",
        "vllm_version": vllm.__version__,
        "gpu": "A100-40GB",
        "model": model,
        "adapter": adapter,
        "dataset": dataset,
        "grader": ("tinker-identical boxed/number parser" if dataset == "gsm8k"
                   else "strict normalized-boxed match (LOWER BOUND)"),
        "n_problems": len(items),
        "n_per_problem": n,
        "ks": ks,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_tokens": max_tokens,
        "per_problem_c": cs,
        "boxed_rate": round(boxed_hits / max(total_outs, 1), 4),
        "pass_at_k": {str(k): round(sum(pass_at_k(n, c, k) for c in cs)
                                    / len(cs), 4) for k in ks},
        "wall_seconds": round(time.time() - t0, 1),
    }
    tag = (f"{model.split('/')[-1].lower()}"
           f"{'_' + adapter.split('/')[-1][:40] if adapter else '_base'}"
           f"_{dataset}_p{len(items)}_n{n}_s{seed}")
    with open(f"/results/passk_modal_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    result["tag"] = tag
    return result


@app.local_entrypoint()
def main(dataset: str = "math500", model: str = "Qwen/Qwen3-8B",
         problems: int = 500, n: int = 32, temperature: float = 1.0,
         top_p: float = 1.0, max_tokens: int = 512, seed: int = 42,
         adapter: str = ""):
    ks = [k for k in (1, 8, 32) if k <= n]
    print(f"[modal-passk] {model}{' + ' + adapter if adapter else ''} on "
          f"{dataset}: {problems} problems x {n} samples, T={temperature}",
          flush=True)
    result = evaluate.remote(model, dataset, problems, n, ks, temperature,
                             top_p, max_tokens, seed, adapter or None)
    print(f"[modal-passk] pass@k = {result['pass_at_k']} "
          f"({result['wall_seconds']}s on GPU)", flush=True)
    from pathlib import Path
    out = (Path(__file__).parent / "results"
           / f"passk_modal_{result['tag']}.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[modal-passk] -> {out}", flush=True)
