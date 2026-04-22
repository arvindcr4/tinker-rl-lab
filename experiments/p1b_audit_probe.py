#!/usr/bin/env python3
"""P1-B: Base-policy probe for the audit table.

Pure sampling driver -- no training step. For each configured (model, task, G),
it samples G completions from the base policy on a fixed prompt subset, computes
per-prompt hit rate p_x under the matching reward parser, and reports:

    mean_p_x,  P(usable group) = mean_x [1 - (1-p_x)^G - p_x^G],
    empirical zvf = (#all-zero + #all-one) / #prompts,
    and a per-prompt CSV with p_x estimates.

Writes experiments/results/<tag>.json + <tag>.csv. Requires TINKER_API_KEY env.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results"

IMS = "<" + "|im_start|>"
IME = "<" + "|im_end|>"

GSM8K_SYSTEM = (
    "You are a math assistant. Solve the problem step by step, then give your "
    "final numerical answer inside \\boxed{}."
)
GSM8K_SUFFIX = " Provide the final numerical answer inside \\boxed{}."


def require_env():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(
            "TINKER_API_KEY not set. Export it in your shell before running."
        )


def reward_gsm8k(response, answer):
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
        try:
            if abs(float(nums[-1].replace(",", "")) - float(answer)) < 0.01:
                return 1.0
        except ValueError:
            pass
    return 0.0


def reward_arith(response, answer):
    nums = re.findall(r"[-+]?\d+", response)
    try:
        target = int(answer)
    except ValueError:
        return 0.0
    for n in nums:
        try:
            if int(n) == target:
                return 1.0
        except ValueError:
            continue
    return 0.0


def build_prompt(system, user):
    return (
        IMS + "system\n" + system + IME + "\n"
        + IMS + "user\n" + user + IME + "\n"
        + IMS + "assistant\n"
    )


def load_gsm8k(limit, seed):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        prompt = build_prompt(GSM8K_SYSTEM, row["question"] + GSM8K_SUFFIX)
        examples.append((prompt, ans))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def load_math500(limit, seed):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    examples = []
    for row in ds:
        ans = str(row.get("answer", "")).strip()
        if not ans:
            continue
        prompt = build_prompt(GSM8K_SYSTEM, row["problem"] + GSM8K_SUFFIX)
        examples.append((prompt, ans))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def load_arith(limit, seed):
    rng = random.Random(seed)
    examples = []
    for _ in range(limit):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        user = "What is " + str(a) + " + " + str(b) + "? Answer with just the number."
        prompt = build_prompt("You are a math assistant.", user)
        examples.append((prompt, str(a + b)))
    return examples


TASKS = {
    "gsm8k": (load_gsm8k, reward_gsm8k),
    "math500": (load_math500, reward_gsm8k),
    "arith": (load_arith, reward_arith),
}


def run(args):
    require_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    random.seed(args.seed)
    tag = args.tag or ("p1b_audit_" + args.task + "_"
                        + args.model.split("/")[-1] + "_G" + str(args.group)
                        + "_n" + str(args.prompts) + "_"
                        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    out_json = OUT_DIR / (tag + ".json")
    out_csv = OUT_DIR / (tag + ".csv")
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    result = {
        "tag": tag,
        "status": "started",
        "model": args.model,
        "task": args.task,
        "group_size": args.group,
        "prompts": args.prompts,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "started_at": started_at,
    }
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        loader, reward_fn = TASKS[args.task]
        examples = loader(args.prompts, args.seed)

        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        svc = tinker.ServiceClient(base_url=None)
        tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
        result["run_id"] = tc.model_id
        initial = tc.save_weights_for_sampler(name="s0").result()
        sc = tc.create_sampling_client(model_path=initial.path)

        per_prompt = []
        for i, (prompt, answer) in enumerate(examples):
            prompt_ids = tok.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) > args.max_prompt_tokens:
                prompt_ids = prompt_ids[: args.max_prompt_tokens]
            sampled = sc.sample(
                T.ModelInput.from_ints(prompt_ids),
                num_samples=args.group,
                sampling_params=T.SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ),
            ).result()
            rewards = []
            for seq in sampled.sequences:
                text = tok.decode(list(seq.tokens), skip_special_tokens=True)
                rewards.append(reward_fn(text, answer))
            p_x = sum(rewards) / len(rewards)
            zero_var = 1 if all(r == rewards[0] for r in rewards) else 0
            per_prompt.append({
                "idx": i,
                "p_x": p_x,
                "n_correct": sum(1 for r in rewards if r > 0.5),
                "n_samples": len(rewards),
                "zero_variance": zero_var,
            })
            if (i + 1) % 5 == 0:
                mean_so_far = sum(p["p_x"] for p in per_prompt) / len(per_prompt)
                print("[" + tag + "] " + str(i + 1) + "/" + str(len(examples))
                      + " mean_p_x so far = " + ("%.3f" % mean_so_far), flush=True)

        n = len(per_prompt)
        mean_p = sum(p["p_x"] for p in per_prompt) / n
        G = args.group
        usable = sum(1 - (1 - p["p_x"]) ** G - p["p_x"] ** G for p in per_prompt) / n
        zvf_emp = sum(p["zero_variance"] for p in per_prompt) / n

        result.update({
            "status": "completed",
            "mean_p_x": mean_p,
            "predicted_usable_group_rate": usable,
            "empirical_initial_zvf": zvf_emp,
            "n_prompts": n,
            "per_prompt": per_prompt,
            "wall_clock_sec": time.time() - t0,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        out_json.write_text(json.dumps(result, indent=2) + "\n")

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "p_x", "n_correct", "n_samples", "zero_variance"])
            for p in per_prompt:
                w.writerow([p["idx"], p["p_x"], p["n_correct"], p["n_samples"], p["zero_variance"]])

        print("[" + tag + "] done. mean_p_x=" + ("%.3f" % mean_p)
              + " predicted_usable=" + ("%.3f" % usable)
              + " empirical_zvf=" + ("%.3f" % zvf_emp))
        print("wrote " + str(out_json))
        print("wrote " + str(out_csv))
        return result

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["wall_clock_sec"] = time.time() - t0
        out_json.write_text(json.dumps(result, indent=2) + "\n")
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--group", type=int, default=16)
    p.add_argument("--prompts", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--tag", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
