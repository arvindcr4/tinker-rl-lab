"""Modal held-out GSM8K eval of Llama-3.1-8B-Instruct (fills the skipped row).

The paper's held-out table (tab:heldout_gsm8k) left the Llama-3.1-8B-Instruct
row blank ("skipped due to a gated-tokenizer authentication error"). With a HF
token (Modal secret) we can now load the gated meta-llama checkpoint and run the
SAME held-out protocol: greedy decoding (T=0), boxed-answer scoring, a fixed
GSM8K test slice, Wilson 95% CI.

NOTE: This evaluates the *base* Llama-3.1-8B-Instruct checkpoint (the post-GRPO
Tinker checkpoint referenced by the table is not retrievable, and Tinker does
not host Llama-3.1-8B), so the result is the pre-RL held-out accuracy for this
model, reported as such.

Usage:
  modal run experiments/modal/modal_llama_heldout_eval.py
"""
import modal
import json
import os
import time

app = modal.App("tinkerrl-llama-heldout")
results_vol = modal.Volume.from_name("tinkerrl-results", create_if_missing=True)
RESULTS_DIR = "/results"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.3.0",
        "transformers>=4.46.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "numpy>=1.26.0,<2.0.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.26.0",
    )
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_PROBLEMS = 500
SLICE_SEED = 0
MAX_NEW = 512


@app.function(image=image, gpu="A10G", timeout=5400,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              volumes={RESULTS_DIR: results_vol}, retries=1)
def eval_llama(n_problems: int = N_PROBLEMS, slice_seed: int = SLICE_SEED) -> dict:
    import re
    import math
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kw = {"token": os.environ.get("HF_TOKEN")}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(MODEL, **tok_kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, **tok_kw).to(device)
    model.eval()

    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=slice_seed)
    ds = ds.select(range(min(n_problems, len(ds))))

    def gold_of(ans):
        return ans.split("####")[-1].strip().replace(",", "")

    def extract(text):
        m = re.findall(r"\\boxed\{([^}]+)\}", text)
        if m:
            cand = m[-1]
        else:
            nums = re.findall(r"-?\d[\d,]*", text)
            cand = nums[-1] if nums else None
        if cand is None:
            return None
        return cand.replace(",", "").replace("$", "").strip()

    def build(q):
        msgs = [{"role": "user",
                 "content": q + "\nSolve step by step and put the final numerical answer in \\boxed{}."}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    items = list(ds)
    per_item = []
    correct = 0
    t0 = time.time()
    B = 16
    for i in range(0, len(items), B):
        chunk = items[i:i + B]
        prompts = [build(it["question"]) for it in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for it, out in zip(chunk, outs):
            gold = gold_of(it["answer"])
            pred = extract(out)
            ok = int(pred is not None and pred == gold)
            correct += ok
            per_item.append({"gold": gold, "pred": pred, "correct": ok})
        if (i + B) % 80 == 0:
            print(f"  {min(i + B, len(items))}/{len(items)} acc so far={correct/max(1,len(per_item)):.3f}")

    n = len(per_item)
    acc = correct / n
    # Wilson 95% CI
    z = 1.96
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    wilson = [round(center - half, 4), round(center + half, 4)]

    res = {"experiment": "llama_heldout_gsm8k", "model": MODEL, "checkpoint_state": "base/pre-RL",
           "n": n, "slice_seed": slice_seed, "decoding": "greedy(T=0)", "max_new_tokens": MAX_NEW,
           "accuracy": round(acc, 4), "wilson_ci95": wilson, "n_correct": correct,
           "elapsed_seconds": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "per_item": per_item}
    os.makedirs(f"{RESULTS_DIR}/llama_heldout", exist_ok=True)
    with open(f"{RESULTS_DIR}/llama_heldout/result.json", "w") as f:
        json.dump(res, f, indent=2)
    results_vol.commit()
    print(f"[Llama-3.1-8B-Instruct held-out GSM8K] acc={acc:.4f} CI95={wilson} (n={n})")
    return res


@app.local_entrypoint()
def main():
    res = eval_llama.remote(N_PROBLEMS, SLICE_SEED)
    summary = {k: v for k, v in res.items() if k != "per_item"}
    with open("experiments/results/llama_heldout_gsm8k.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\n=== Llama-3.1-8B-Instruct held-out GSM8K (measured) ===")
    print(json.dumps(summary, indent=2))
    print("Saved experiments/results/llama_heldout_gsm8k.json")
