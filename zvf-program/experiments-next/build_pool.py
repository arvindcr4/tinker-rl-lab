#!/usr/bin/env python3
"""build_pool.py — the ONE sampling pass that funds E-T1, E-T2, and E-T3a.

Samples R rollouts (default 32) for each of N prompts (default 512) from a
frozen model/checkpoint via the Tinker sampling client and stores the binary
reward vector per prompt. Every downstream theory-validation analysis
(analyze_t1_ci.py, analyze_t2_floor.py, analyze_t3_gstar.py) is pure offline
resampling of this pool — zero further API cost.

Cost estimate at defaults: 512 prompts x 32 rollouts x ~300 output tokens
~= 5M sampled tokens per model. Run --dry-run first (prints the plan and
estimate, contacts nothing).

Usage:
  python3 build_pool.py --model Qwen/Qwen3-8B --dry-run
  python3 build_pool.py --model Qwen/Qwen3-8B
  python3 build_pool.py --model Qwen/Qwen3-8B --sampler-path tinker://... \
      --tag qwen3-8b-step50   # pool from a mid-training checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from common import (
    RESULTS_DIR,
    load_gsm8k,
    load_repo_env,
    make_sampler,
    reward_fn,
    utc_now,
    write_result,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="base model id (HF-style)")
    ap.add_argument("--sampler-path", default=None,
                    help="existing tinker:// weights path (defaults to base model)")
    ap.add_argument("--tag", default=None, help="output tag (default: derived)")
    ap.add_argument("--prompts", type=int, default=512)
    ap.add_argument("--rollouts", type=int, default=32,
                    help="R rollouts per prompt; must be >= max G you will analyze")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and cost estimate; contact nothing")
    ap.add_argument("--resume", action="store_true",
                    help="continue a partial pool with identical config "
                         "(same seed => deterministic example order)")
    args = ap.parse_args()

    tag = args.tag or (
        f"{args.model.split('/')[-1].lower()}"
        f"_{args.split}_n{args.prompts}_r{args.rollouts}_s{args.seed}"
    )
    out_path = RESULTS_DIR / f"pool_{tag}.json"

    est_tokens = args.prompts * args.rollouts * 300
    print(f"[pool:{tag}] plan: {args.prompts} prompts x {args.rollouts} rollouts "
          f"({args.split} split), T={args.temperature}, top_p={args.top_p}, "
          f"~{est_tokens/1e6:.1f}M output tokens estimated", flush=True)
    print(f"[pool:{tag}] output -> {out_path}", flush=True)
    if args.dry_run:
        print(f"[pool:{tag}] DRY RUN — nothing sampled.", flush=True)
        return

    load_repo_env()
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set (and not found in repo .env); aborting.")

    import tinker.types as T

    examples = load_gsm8k(args.split, args.prompts, args.seed)
    if len(examples) < args.prompts:
        print(f"[pool:{tag}] WARNING: only {len(examples)} examples available",
              flush=True)

    banked: list[dict] = []
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("status") == "complete":
            sys.exit(f"[pool:{tag}] pool already complete; nothing to resume.")
        same = all(prev.get(k) == getattr(args, k.replace("-", "_"), None) or
                   prev.get(k) == v for k, v in
                   [("seed", args.seed), ("split", args.split),
                    ("rollouts_per_prompt", args.rollouts),
                    ("temperature", args.temperature)])
        if not same:
            sys.exit(f"[pool:{tag}] existing partial pool has different "
                     "config; refusing to resume across configs.")
        banked = prev.get("prompts", [])
        print(f"[pool:{tag}] resuming: {len(banked)} prompts already banked; "
              f"{len(examples) - len(banked)} to go "
              "(same seed => same deterministic example order)", flush=True)

    result: dict = {
        "kind": "zvf_pool",
        "status": "started",
        "tag": tag,
        "model": args.model,
        "split": args.split,
        "n_prompts": len(examples),
        "rollouts_per_prompt": args.rollouts,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "started_at": utc_now(),
        "prompts": banked,
    }
    write_result(out_path, result)

    sc, tok, sampler_used = make_sampler(args.model, args.sampler_path,
                                         args.lora_rank)
    result["sampler_path"] = sampler_used
    sampling = T.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    t0 = time.time()
    for i, (prompt, answer) in enumerate(examples):
        if i < len(banked):
            continue  # resumed: already sampled in a prior run
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > args.max_prompt_tokens:
            prompt_ids = prompt_ids[: args.max_prompt_tokens]
        sampled = sc.sample(
            T.ModelInput.from_ints(prompt_ids),
            num_samples=args.rollouts,
            sampling_params=sampling,
        ).result()
        rewards = [
            reward_fn(tok.decode(list(seq.tokens), skip_special_tokens=True),
                      answer)
            for seq in sampled.sequences
        ]
        p_hat = sum(rewards) / len(rewards)
        result["prompts"].append({
            "idx": i,
            "answer": answer,
            "rewards": rewards,
            "p_hat": p_hat,
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[pool:{tag}] {i+1}/{len(examples)} prompts "
                  f"({elapsed:.0f}s, mean p_hat="
                  f"{sum(p['p_hat'] for p in result['prompts'])/(i+1):.3f})",
                  flush=True)
            write_result(out_path, result)  # checkpoint partial progress

    result["status"] = "complete"
    result["finished_at"] = utc_now()
    result["wall_seconds"] = round(time.time() - t0, 1)
    write_result(out_path, result)
    print(f"[pool:{tag}] complete -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
