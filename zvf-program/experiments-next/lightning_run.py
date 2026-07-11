#!/usr/bin/env python3
"""lightning_run.py — run eval_passk_standalone.py on a Lightning AI Studio.

Provisions (or reuses) a Studio named 'zvf-passk' in the playground
teamspace, switches it to a GPU machine, uploads the standalone evaluator,
installs vllm, runs the eval, downloads the result JSON into results/, and
stops the Studio (so GPU credits stop burning) unless --keep-alive.

  python lightning_run.py --dataset mbpp --problems 200 --n 32
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mbpp", choices=["gsm8k", "math500", "mbpp"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--problems", type=int, default=200)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument(
        "--machine", default="L40S", help="Lightning machine type (L40S / L4 / A100 ...)"
    )
    ap.add_argument("--studio", default="zvf-passk")
    ap.add_argument(
        "--adapters",
        default="",
        help="comma-separated HF adapter repos; runs one eval per "
        "adapter in the same studio session",
    )
    ap.add_argument("--keep-alive", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume partial remote arms and skip compatible local completions",
    )
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    if args.checkpoint_every <= 0:
        ap.error("--checkpoint-every must be positive")

    from lightning_sdk import Machine, Studio

    machine = getattr(Machine, args.machine, None)
    if machine is None:
        sys.exit(
            f"unknown machine {args.machine}; options: "
            f"{[m for m in dir(Machine) if not m.startswith('_')]}"
        )

    tag = (
        f"{args.model.split('/')[-1].lower()}_base_{args.dataset}"
        f"_p{args.problems}_n{args.n}_s{args.seed}"
    )
    remote_out = f"passk_lightning_{tag}.json"
    local_out = HERE / "results" / f"passk_lightning_{tag}.json"

    print(
        f"[lightning] studio={args.studio} machine={args.machine} "
        f"dataset={args.dataset} p={args.problems} n={args.n}",
        flush=True,
    )
    studio = Studio(name=args.studio, teamspace="playground", user="arvindcr4", create_ok=True)
    print(f"[lightning] status: {studio.status}", flush=True)
    t0 = time.time()
    try:
        studio.start(machine)
        print(f"[lightning] started on {args.machine} ({time.time() - t0:.0f}s)", flush=True)

        studio.upload_file(str(HERE / "eval_passk_standalone.py"), "eval_passk_standalone.py")
        print("[lightning] uploaded evaluator", flush=True)

        out = studio.run("python -m pip install -q uv 2>&1 | tail -1")
        print(f"[lightning] uv install: {out.strip()[-120:]}", flush=True)

        # uv-run sidesteps the fragile Studio conda env entirely (venv is
        # forbidden; upgrading numpy in-place breaks torch/sklearn ABI).
        arms = [("", remote_out, local_out)]
        if args.adapters:
            arms = []
            for ad in args.adapters.split(","):
                ad = ad.strip()
                short = ad.split("/")[-1][:40]
                r = f"passk_lightning_{args.model.split('/')[-1].lower()}_{short}_{args.dataset}_p{args.problems}_n{args.n}_s{args.seed}.json"
                arms.append((ad, r, HERE / "results" / r))
        for adapter, r_out, l_out in arms:
            expected = {
                "model": args.model,
                "adapter": adapter or None,
                "dataset": args.dataset,
                "n_problems": args.problems,
                "n_per_problem": args.n,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
                "checkpoint_every": args.checkpoint_every,
            }
            if l_out.exists():
                existing = json.loads(l_out.read_text())
                mismatches = [
                    key
                    for key, value in expected.items()
                    if key in existing and existing.get(key) != value
                ]
                if mismatches:
                    raise RuntimeError(
                        f"local output collision at {l_out}; mismatched {mismatches}"
                    )
                if existing.get("status") == "complete":
                    print(f"[lightning] already complete; skipping {l_out}", flush=True)
                    continue

            ad_flag = f"--adapter {shlex.quote(adapter)} " if adapter else ""
            resume_flag = "--resume " if args.resume else ""
            cmd = (
                f"HF_HUB_ENABLE_HF_TRANSFER=1 python -m uv run --isolated "
                f"--no-project --python 3.12 "
                f"--with vllm --with 'datasets>=3.0' --with hf_transfer "
                f"eval_passk_standalone.py "
                f"--dataset {shlex.quote(args.dataset)} "
                f"--model {shlex.quote(args.model)} {ad_flag}"
                f"--problems {args.problems} --n {args.n} --seed {args.seed} "
                f"--max-tokens {args.max_tokens} "
                f"--checkpoint-every {args.checkpoint_every} {resume_flag}"
                f"--out {shlex.quote(r_out)}; echo REMOTE_EXIT=$?"
            )
            print(f"[lightning] running: {cmd}", flush=True)
            out = studio.run(cmd)
            print(out[-1200:], flush=True)
            if "REMOTE_EXIT=0" not in out:
                print(f"[lightning] ARM FAILED ({adapter or 'base'}); continuing", flush=True)
                continue
            Path(l_out).parent.mkdir(exist_ok=True)
            studio.download_file(r_out, str(l_out))
            print(f"[lightning] -> {l_out}", flush=True)
    finally:
        if not args.keep_alive:
            try:
                studio.stop()
                print("[lightning] studio stopped", flush=True)
            except Exception as exc:
                print(f"[lightning] stop skipped ({exc})", flush=True)


if __name__ == "__main__":
    main()
