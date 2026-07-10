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
import statistics
import sys
import time
from datetime import datetime

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
    ap.add_argument("--max-retries", type=int, default=3,
                    help="retries per prompt after transient sampling failures")
    ap.add_argument("--retry-backoff-seconds", type=float, default=2.0,
                    help="initial exponential retry backoff")
    args = ap.parse_args()

    if args.prompts <= 0 or args.rollouts <= 1:
        ap.error("--prompts must be positive and --rollouts must be at least 2")
    if args.max_retries < 0 or args.retry_backoff_seconds < 0:
        ap.error("retry settings must be non-negative")

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

    if out_path.exists() and not args.resume:
        existing = json.loads(out_path.read_text())
        sys.exit(
            f"[pool:{tag}] output already exists with status="
            f"{existing.get('status')!r}; pass --resume or choose a new --tag"
        )
    if out_path.exists() and args.resume:
        existing = json.loads(out_path.read_text())
        if existing.get("status") == "complete":
            sys.exit(f"[pool:{tag}] pool already complete; nothing to resume.")

    load_repo_env()
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set (and not found in repo .env); aborting.")

    import tinker.types as T

    examples = load_gsm8k(args.split, args.prompts, args.seed)
    if len(examples) < args.prompts:
        print(f"[pool:{tag}] WARNING: only {len(examples)} examples available",
              flush=True)

    banked: list[dict] = []
    previous: dict | None = None
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("status") == "complete":
            sys.exit(f"[pool:{tag}] pool already complete; nothing to resume.")
        expected = {
            "model": args.model,
            "split": args.split,
            "n_prompts": len(examples),
            "rollouts_per_prompt": args.rollouts,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        mismatches = {
            key: {"existing": prev.get(key), "requested": value}
            for key, value in expected.items()
            if prev.get(key) != value
        }
        if mismatches:
            sys.exit(
                f"[pool:{tag}] existing partial pool has incompatible config: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        if args.sampler_path and prev.get("sampler_path") != args.sampler_path:
            sys.exit(
                f"[pool:{tag}] sampler path differs from the partial pool; "
                "refusing to mix checkpoints"
            )
        banked = prev.get("prompts", [])
        if [row.get("idx") for row in banked] != list(range(len(banked))):
            sys.exit(
                f"[pool:{tag}] partial pool is not a contiguous prompt prefix; "
                "refusing unsafe resume"
            )
        previous = prev
        print(f"[pool:{tag}] resuming: {len(banked)} prompts already banked; "
              f"{len(examples) - len(banked)} to go "
              "(same seed => same deterministic example order)", flush=True)

    session_id = utc_now()
    result: dict = previous or {}
    result.update({
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
        "requested_sampler_path": args.sampler_path,
        "prompts": banked,
    })
    result.setdefault("started_at", utc_now())
    result.setdefault("failure_events", [])
    result.setdefault("retry_count", 0)
    if previous is not None:
        result["resumed_at"] = utc_now()
    result["sampling_session_id"] = session_id
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
        sample_started = time.time()
        sampled = None
        for attempt in range(args.max_retries + 1):
            try:
                sampled = sc.sample(
                    T.ModelInput.from_ints(prompt_ids),
                    num_samples=args.rollouts,
                    sampling_params=sampling,
                ).result()
                break
            except Exception as exc:
                message = str(exc)
                api_key = os.environ.get("TINKER_API_KEY")
                if api_key:
                    message = message.replace(api_key, "<redacted>")
                event = {
                    "at": utc_now(),
                    "session_id": session_id,
                    "prompt_idx": i,
                    "attempt": attempt + 1,
                    "error_type": type(exc).__name__,
                    "message": message[:500],
                }
                result["failure_events"].append(event)
                result["last_error"] = event
                write_result(out_path, result)
                if attempt >= args.max_retries:
                    raise
                result["retry_count"] += 1
                delay = args.retry_backoff_seconds * (2 ** attempt)
                print(
                    f"[pool:{tag}] prompt {i} attempt {attempt + 1} failed "
                    f"({type(exc).__name__}); retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)

        if sampled is None or len(sampled.sequences) != args.rollouts:
            raise RuntimeError(
                f"prompt {i}: expected {args.rollouts} sequences, got "
                f"{0 if sampled is None else len(sampled.sequences)}"
            )
        token_counts = [len(seq.tokens) for seq in sampled.sequences]
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
            "token_counts": token_counts,
            "sample_seconds": round(time.time() - sample_started, 3),
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"[pool:{tag}] {i+1}/{len(examples)} prompts "
                  f"({elapsed:.0f}s, mean p_hat="
                  f"{sum(p['p_hat'] for p in result['prompts'])/len(result['prompts']):.3f})",
                  flush=True)
            write_result(out_path, result)  # checkpoint partial progress

    segment_wall_seconds = time.time() - t0
    prior_wall_seconds = float(result.get("wall_seconds") or 0.0)
    cumulative_wall_seconds = prior_wall_seconds + segment_wall_seconds
    total_output_tokens = sum(
        sum(row.get("token_counts", [])) for row in result["prompts"]
    )
    token_count_prompts = sum(
        1 for row in result["prompts"] if row.get("token_counts") is not None
    )
    total_samples = sum(len(row.get("rewards", [])) for row in result["prompts"])
    prompt_attempts = len(result["prompts"]) + len(result["failure_events"])
    failure_times = []
    for event in result["failure_events"]:
        if event.get("session_id") != session_id:
            continue
        try:
            failure_times.append(datetime.fromisoformat(event["at"]))
        except (KeyError, TypeError, ValueError):
            continue
    failure_times.sort()
    failure_intervals = [
        (right - left).total_seconds()
        for left, right in zip(failure_times, failure_times[1:])
    ]

    result["status"] = "complete"
    result["finished_at"] = utc_now()
    result["segment_wall_seconds"] = round(segment_wall_seconds, 1)
    result["wall_seconds"] = round(cumulative_wall_seconds, 1)
    result["last_error"] = None
    result["infrastructure_metrics"] = {
        "completed_prompts": len(result["prompts"]),
        "total_samples": total_samples,
        "total_output_tokens": total_output_tokens,
        "token_count_prompt_coverage": round(
            token_count_prompts / max(len(result["prompts"]), 1), 6
        ),
        "failure_count": len(result["failure_events"]),
        "retry_count": result["retry_count"],
        "goodput_percent": round(
            100.0 * len(result["prompts"]) / max(prompt_attempts, 1), 3
        ),
        "output_tokens_per_wall_second": round(
            total_output_tokens / max(cumulative_wall_seconds, 1e-9), 3
        ),
        "mean_prompt_sample_seconds": round(
            statistics.fmean(
                row["sample_seconds"]
                for row in result["prompts"]
                if "sample_seconds" in row
            ),
            3,
        ) if any("sample_seconds" in row for row in result["prompts"]) else None,
        "mtbf_seconds": round(statistics.fmean(failure_intervals), 3)
        if failure_intervals else None,
    }
    write_result(out_path, result)
    print(f"[pool:{tag}] complete -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
