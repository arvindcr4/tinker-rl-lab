#!/usr/bin/env python3
"""passk_eval.py — MIN-REPORT-RL item 8: held-out pass@k curves.

Samples n completions per held-out problem (GSM8K test split by default) at a
pinned temperature and reports the unbiased pass@k estimator (Chen et al.,
2021, Codex paper):

    pass@k = E_problems[ 1 - C(n - c, k) / C(n, k) ]

for k in {1, 8, 32} by default (n must be >= max k). Run it once on the BASE
model and once on each post-RL checkpoint under the IDENTICAL config; the
sweep protocol amendment (sweep/README.md) requires both.

Interpretation contract (position/min_report_rl.tex item 8): gains
concentrated at k=1 with a flat k=32 frontier indicate distribution
sharpening; k=32 gains indicate capability expansion. This script only
reports the numbers.

Usage:
  python3 passk_eval.py --model Qwen/Qwen3-8B --dry-run
  python3 passk_eval.py --model Qwen/Qwen3-8B --problems 200
  python3 passk_eval.py --model Qwen/Qwen3-8B \
      --sampler-path tinker://.../post_rl_weights --tag qwen3-8b-postrl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

from common import (
    RESULTS_DIR,
    load_gsm8k,
    load_repo_env,
    make_sampler,
    reward_fn,
    utc_now,
    write_result,
)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k)/C(n, k). Numerically stable form."""
    if n <= 0 or c < 0 or c > n or k <= 0 or k > n:
        raise ValueError(f"invalid pass@k inputs: n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def _percentile(sorted_values: list[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_pass_at_k(
    counts: list[int],
    *,
    n: int,
    ks: list[int],
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, dict]]:
    """Point estimates and problem-clustered bootstrap intervals."""
    if not counts:
        raise ValueError("cannot summarize pass@k without per-problem counts")
    if any(count < 0 or count > n for count in counts):
        raise ValueError("a per-problem correct count falls outside [0, n]")
    point = {
        str(k): sum(pass_at_k(n, count, k) for count in counts) / len(counts)
        for k in ks
    }
    rng = random.Random(seed)
    bootstrap = {str(k): [] for k in ks}
    for _ in range(n_bootstrap):
        resample = [counts[rng.randrange(len(counts))] for _ in counts]
        for k in ks:
            bootstrap[str(k)].append(
                sum(pass_at_k(n, count, k) for count in resample) / len(resample)
            )
    intervals = {}
    for k in ks:
        values = sorted(bootstrap[str(k)])
        intervals[str(k)] = {
            "point": point[str(k)],
            "ci_low": _percentile(values, 0.025),
            "ci_high": _percentile(values, 0.975),
            "bootstrap_replicates": n_bootstrap,
            "resampling_unit": "problem",
        }
    return point, intervals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--sampler-path", default=None,
                    help="tinker:// weights of a post-RL checkpoint "
                         "(default: base model)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--n", type=int, default=32,
                    help="completions per problem (>= max k)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-prompt-tokens", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=2_000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260711)
    ap.add_argument("--from-result", type=Path, default=None,
                    help="offline: add clustered CIs to an existing pass@k result")
    ap.add_argument("--out", type=Path, default=None,
                    help="output for --from-result (never overwrites the input)")
    ap.add_argument("--resume", action="store_true",
                    help="resume a compatible partial online evaluation")
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.bootstrap < 100:
        ap.error("--bootstrap must be at least 100")
    if args.problems <= 0 or args.n <= 0:
        ap.error("--problems and --n must be positive")
    if args.max_retries < 0 or args.retry_backoff_seconds < 0:
        ap.error("retry settings must be non-negative")
    if args.from_result:
        source = json.loads(args.from_result.read_text())
        if source.get("status") != "complete":
            ap.error("--from-result requires a completed result")
        counts = source.get("per_problem_c") or []
        n_per_problem = int(source["n_per_problem"])
        source_ks = [int(k) for k in source["ks"]]
        point, intervals = summarize_pass_at_k(
            counts,
            n=n_per_problem,
            ks=source_ks,
            n_bootstrap=args.bootstrap,
            seed=args.bootstrap_seed,
        )
        payload = {
            "kind": "passk_bootstrap_audit",
            "status": "complete",
            "created_at": utc_now(),
            "source_path": str(args.from_result),
            "source_tag": source.get("tag"),
            "model": source.get("model"),
            "which": source.get("which"),
            "split": source.get("split"),
            "seed": source.get("seed"),
            "temperature": source.get("temperature"),
            "top_p": source.get("top_p"),
            "max_tokens": source.get("max_tokens"),
            "max_prompt_tokens": source.get("max_prompt_tokens"),
            "prompt_fingerprints": source.get("prompt_fingerprints"),
            "n_problems": len(counts),
            "n_per_problem": n_per_problem,
            "pass_at_k": {key: round(value, 6) for key, value in point.items()},
            "pass_at_k_95_ci": intervals,
        }
        output_path = args.out or RESULTS_DIR / f"passk_audit_{args.from_result.stem}.json"
        if output_path.resolve() == args.from_result.resolve():
            ap.error("--out must not overwrite --from-result")
        write_result(output_path, payload)
        print(f"[passk-audit] pass@k={payload['pass_at_k']} -> {output_path}")
        return
    if not args.model:
        ap.error("--model is required unless --from-result is used")

    if args.n < max(args.ks):
        sys.exit(f"--n {args.n} must be >= max k {max(args.ks)}")

    which = "postrl" if args.sampler_path else "base"
    tag = args.tag or (
        f"{args.model.split('/')[-1].lower()}_{which}"
        f"_{args.split}_p{args.problems}_n{args.n}_s{args.seed}"
    )
    out_path = RESULTS_DIR / f"passk_{tag}.json"
    est_tokens = args.problems * args.n * 300
    print(f"[passk:{tag}] plan: {args.problems} problems x {args.n} samples, "
          f"T={args.temperature}, k={args.ks}, "
          f"~{est_tokens/1e6:.1f}M output tokens", flush=True)
    if args.dry_run:
        print(f"[passk:{tag}] DRY RUN — nothing sampled.", flush=True)
        return

    previous = None
    if out_path.exists():
        previous = json.loads(out_path.read_text())
        if previous.get("status") == "complete":
            sys.exit(f"[passk:{tag}] result already complete")
        if not args.resume:
            sys.exit(
                f"[passk:{tag}] partial result exists; pass --resume or use a new --tag"
            )

    load_repo_env()
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set (and not found in repo .env); aborting.")

    import tinker.types as T

    examples = load_gsm8k(args.split, args.problems, args.seed)
    prompt_fingerprints = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        for prompt, _ in examples
    ]
    if previous is not None:
        expected = {
            "model": args.model,
            "which": which,
            "split": args.split,
            "n_problems": len(examples),
            "n_per_problem": args.n,
            "ks": args.ks,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "seed": args.seed,
        }
        mismatches = {
            key: {"existing": previous.get(key), "requested": value}
            for key, value in expected.items()
            if previous.get(key) != value
        }
        if mismatches:
            sys.exit(
                f"[passk:{tag}] incompatible partial result: "
                f"{json.dumps(mismatches, sort_keys=True)}"
            )
        if previous.get("prompt_fingerprints") != prompt_fingerprints:
            sys.exit(f"[passk:{tag}] prompt fingerprints changed; refusing resume")
        if args.sampler_path and previous.get("sampler_path") != args.sampler_path:
            sys.exit(f"[passk:{tag}] sampler path changed; refusing resume")
        counts = previous.get("per_problem_c") or []
        if len(counts) > len(examples):
            sys.exit(f"[passk:{tag}] partial result has too many problem rows")
    else:
        counts = []

    result: dict = previous or {}
    result.update({
        "kind": "passk_eval",
        "status": "started",
        "tag": tag,
        "model": args.model,
        "which": which,
        "split": args.split,
        "n_problems": len(examples),
        "n_per_problem": args.n,
        "ks": args.ks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "seed": args.seed,
        "requested_sampler_path": args.sampler_path,
        "prompt_fingerprints": prompt_fingerprints,
        "per_problem_c": counts,
    })
    result.setdefault("started_at", utc_now())
    result.setdefault("failure_events", [])
    result.setdefault("retry_count", 0)
    if previous is not None:
        result["resumed_at"] = utc_now()
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
        if i < len(counts):
            continue
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > args.max_prompt_tokens:
            prompt_ids = prompt_ids[: args.max_prompt_tokens]
        sampled = None
        for attempt in range(args.max_retries + 1):
            try:
                sampled = sc.sample(
                    T.ModelInput.from_ints(prompt_ids),
                    num_samples=args.n,
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
                    "problem_idx": i,
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
                    f"[passk:{tag}] problem {i} attempt {attempt + 1} failed; "
                    f"retrying in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        if sampled is None or len(sampled.sequences) != args.n:
            raise RuntimeError(
                f"problem {i}: expected {args.n} sequences, got "
                f"{0 if sampled is None else len(sampled.sequences)}"
            )
        c = sum(
            int(reward_fn(tok.decode(list(seq.tokens),
                                     skip_special_tokens=True), answer))
            for seq in sampled.sequences
        )
        result["per_problem_c"].append(c)
        if (i + 1) % 25 == 0:
            print(f"[passk:{tag}] {i+1}/{len(examples)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            write_result(out_path, result)

    cs = result["per_problem_c"]
    point, intervals = summarize_pass_at_k(
        cs,
        n=args.n,
        ks=args.ks,
        n_bootstrap=args.bootstrap,
        seed=args.bootstrap_seed,
    )
    result["pass_at_k"] = {
        key: round(value, 4) for key, value in point.items()
    }
    result["pass_at_k_95_ci"] = intervals
    # MIN-REPORT-RL item-8 block, ready to paste into the paper appendix.
    result["min_report_rl_item8"] = {
        "held_out_pass_at_k": result["pass_at_k"],
        "held_out_pass_at_k_95_ci": result["pass_at_k_95_ci"],
        "estimate_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "completions_per_problem": args.n,
            "estimator": "unbiased (Chen et al. 2021)",
            "interval": "problem-clustered percentile bootstrap",
            "bootstrap_replicates": args.bootstrap,
        },
        "which_checkpoint": which,
    }
    result["status"] = "complete"
    result["finished_at"] = utc_now()
    result["wall_seconds"] = round(time.time() - t0, 1)
    result["failure_count"] = len(result["failure_events"])
    result["last_error"] = None
    write_result(out_path, result)
    print(f"[passk:{tag}] pass@k = {result['pass_at_k']} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
