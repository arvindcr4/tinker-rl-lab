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
import math
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


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k)/C(n, k). Numerically stable form."""
    if n - c < k:
        return 1.0
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)
    return 1.0 - result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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

    load_repo_env()
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set (and not found in repo .env); aborting.")

    import tinker.types as T

    examples = load_gsm8k(args.split, args.problems, args.seed)
    result: dict = {
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
        "seed": args.seed,
        "started_at": utc_now(),
        "per_problem_c": [],
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
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        if len(prompt_ids) > args.max_prompt_tokens:
            prompt_ids = prompt_ids[: args.max_prompt_tokens]
        sampled = sc.sample(
            T.ModelInput.from_ints(prompt_ids),
            num_samples=args.n,
            sampling_params=sampling,
        ).result()
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
    result["pass_at_k"] = {
        str(k): round(sum(pass_at_k(args.n, c, k) for c in cs) / len(cs), 4)
        for k in args.ks
    }
    # MIN-REPORT-RL item-8 block, ready to paste into the paper appendix.
    result["min_report_rl_item8"] = {
        "held_out_pass_at_k": result["pass_at_k"],
        "estimate_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "completions_per_problem": args.n,
            "estimator": "unbiased (Chen et al. 2021)",
        },
        "which_checkpoint": which,
    }
    result["status"] = "complete"
    result["finished_at"] = utc_now()
    result["wall_seconds"] = round(time.time() - t0, 1)
    write_result(out_path, result)
    print(f"[passk:{tag}] pass@k = {result['pass_at_k']} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
