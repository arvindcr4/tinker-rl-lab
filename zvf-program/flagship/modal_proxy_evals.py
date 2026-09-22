#!/usr/bin/env python3
"""Modal GPU proxy evals: Omni-MATH + LAB-Bench LitQA2 subsets on Qwen3-8B.

Own-benchmark numbers only. Never E-lane scores. `modal run` entrypoint.
"""
import ast
import json
import random
import re

import modal

app = modal.App("pavlov-proxy-evals")
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets==3.6.0",
    "transformers==4.55.0",
    "accelerate==1.10.0",
    "torch==2.7.1",
)

MODEL_ID = "Qwen/Qwen3-8B"
N_OMNI = 20
N_LIT = 50


def norm(s: str) -> str:
    s = re.sub(r"\\(boxed|text|mathrm)\{?", "", s.strip())
    return re.sub(r"\s+", "", s)


@app.function(image=image, gpu="A10G", timeout=3600)
def run_proxies() -> dict:
    import torch
    from datasets import load_dataset
    from transformers import pipeline

    gen = pipeline(
        "text-generation",
        model=MODEL_ID,
        device_map="auto",
        max_new_tokens=4096,
    )

    full = load_dataset("KbsdJames/Omni-MATH", split="test")
    omni = full.select(
        [i for i, r in enumerate(full) if float(r["difficulty"]) <= 6.0][:N_OMNI]
    )
    res = []
    for i, row in enumerate(omni):
        out = gen(
            "Solve this math problem. End your response with the final answer in "
            "\\boxed{}.\n\n" + row["problem"],
            return_full_text=False,
        )[0]["generated_text"]
        m = re.findall(r"\\boxed\{([^}]*)\}", out)
        if not m:
            m = re.findall(r"FINAL:\s*(.+)", out)
        pred = norm(m[-1]) if m else ""
        gold = norm(row["answer"])
        ok = bool(pred) and (pred == gold or pred in gold or gold in pred)
        res.append({"i": i, "correct": bool(ok)})

    lit = load_dataset("futurehouse/lab-bench", "LitQA2", split=f"train[:{N_LIT}]")
    random.seed(211)
    res2 = []
    for row in lit:
        raw = row["distractors"]
        distractors = raw if isinstance(raw, list) else ast.literal_eval(raw)
        opts = distractors + [row["ideal"]]
        random.shuffle(opts)
        letters = "ABCD"
        prompt = (
            row["question"]
            + "\n"
            + "\n".join(f"{L}. {o}" for L, o in zip(letters, opts))
            + "\nAnswer with only the letter."
        )
        out = gen(prompt, return_full_text=False)[0]["generated_text"]
        m = re.search(r"\b([A-D])\b", out)
        pred = opts[letters.index(m.group(1))] if m else ""
        res2.append({"id": row["id"], "correct": bool(pred == row["ideal"])})

    return {
        "proxy_omnimath": {
            "proxy": "Omni-MATH-test-subset-modal",
            "model": MODEL_ID,
            "subset": "difficulty<=6.0, first 20 in test order",
            "n": len(res),
            "accuracy": sum(r["correct"] for r in res) / len(res),
            "note": "Own-benchmark proxy number only; never an E14/FrontierMath score.",
            "results": res,
        },
        "proxy_litqa": {
            "proxy": "LAB-Bench-LitQA2-subset-modal",
            "model": MODEL_ID,
            "n": len(res2),
            "accuracy": sum(r["correct"] for r in res2) / max(1, len(res2)),
            "note": "Own-benchmark proxy number only; never an E8/LifeSciBench score.",
            "results": res2,
        },
    }


@app.local_entrypoint()
def main():
    out = run_proxies.remote()
    with open("outputs/MODAL_PROXY_RUN_2026-09-20.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v["accuracy"] for k, v in out.items()}, indent=1))
