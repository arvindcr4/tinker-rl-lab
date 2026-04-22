#!/usr/bin/env python3
"""P2: Length-bias post-hoc probe.

Samples G rollouts from a base model on GSM8K and records per-rollout
(token_length, reward). Produces a scatter + boxplot + split-histogram to
answer: does the binary reward_fn systematically favour longer completions?

Cost target: <$3, <5 min. Two models, 40 prompts, G=8 => 640 rollouts total.

Outputs:
  experiments/results/p2_length_bias_<model>.{json,csv}
  paper/figures/v2/p2_length_bias.png (combined figure)
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
FIG_DIR = ROOT / "paper" / "figures" / "v2"

IMS = "<" + "|im_start|>"
IME = "<" + "|im_end|>"
GSM8K_SYSTEM = (
    "You are a math assistant. Solve the problem step by step, then give "
    "your final numerical answer inside \\boxed{}."
)
GSM8K_SUFFIX = " Provide the final numerical answer inside \\boxed{}."


def require_env():
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY not set")


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


def build_prompt(user):
    return (IMS + "system\n" + GSM8K_SYSTEM + IME + "\n"
            + IMS + "user\n" + user + GSM8K_SUFFIX + IME + "\n"
            + IMS + "assistant\n")


def load_gsm8k(limit, seed):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    examples = []
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        examples.append((build_prompt(row["question"]), ans))
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:limit]


def run(args):
    require_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    random.seed(args.seed)
    tag = args.tag or ("p2_length_bias_" + args.model.split("/")[-1])
    out_json = OUT_DIR / (tag + ".json")
    out_csv = OUT_DIR / (tag + ".csv")
    t0 = time.time()

    result = {
        "tag": tag,
        "status": "started",
        "model": args.model,
        "group_size": args.group,
        "prompts": args.prompts,
        "seed": args.seed,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    try:
        examples = load_gsm8k(args.prompts, args.seed)
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        svc = tinker.ServiceClient(base_url=None)
        tc = svc.create_lora_training_client(base_model=args.model, rank=args.rank)
        initial = tc.save_weights_for_sampler(name="s0").result()
        sc = tc.create_sampling_client(model_path=initial.path)

        rollouts = []  # (prompt_idx, token_length, reward)
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
            for seq in sampled.sequences:
                tokens = list(seq.tokens)
                text = tok.decode(tokens, skip_special_tokens=True)
                r = reward_gsm8k(text, answer)
                rollouts.append({"prompt_idx": i, "length": len(tokens), "reward": r})
            if (i + 1) % 10 == 0:
                print("[" + tag + "] " + str(i + 1) + "/" + str(len(examples)), flush=True)

        result.update({
            "status": "completed",
            "n_rollouts": len(rollouts),
            "rollouts": rollouts,
            "wall_clock_sec": time.time() - t0,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
        out_json.write_text(json.dumps(result, indent=2) + "\n")

        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["prompt_idx", "length", "reward"])
            for r in rollouts:
                w.writerow([r["prompt_idx"], r["length"], r["reward"]])

        pos = [r["length"] for r in rollouts if r["reward"] > 0.5]
        neg = [r["length"] for r in rollouts if r["reward"] <= 0.5]
        def stats(xs):
            if not xs: return (0, 0.0, 0.0)
            return (len(xs), sum(xs) / len(xs), sorted(xs)[len(xs) // 2])
        ns, ms, meds = stats(pos)
        nf, mf, medf = stats(neg)
        print("[" + tag + "] rewarded: n=" + str(ns) + " mean_len=" + ("{:.1f}".format(ms)) + " median=" + str(meds))
        print("[" + tag + "] failed  : n=" + str(nf) + " mean_len=" + ("{:.1f}".format(mf)) + " median=" + str(medf))
        print("wrote " + str(out_json))
        print("wrote " + str(out_csv))

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
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--prompts", type=int, default=40)
    p.add_argument("--seed", type=int, default=20260422)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--tag", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
