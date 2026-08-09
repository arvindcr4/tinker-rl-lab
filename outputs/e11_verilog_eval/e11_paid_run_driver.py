#!/usr/bin/env python3
"""Drive the authorized E11 pass@1 run: W&B -> Tinker sampling -> native harness.

Ordering is a contract requirement: the W&B run is initialized ONLINE before any
Tinker client is constructed, so no paid call can happen off-record.

Sampling semantics come from ``e11_model_run``: exactly one sample per problem,
no retries, no best-of. A failed extraction is recorded as a miss, because
re-rolling a weak answer turns pass@1 into pass@k.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import e11_model_run as E11  # noqa: E402

CHECKOUT = HERE / "nvlabs_verilog_eval_c498220d"
IVERILOG_BIN = HERE / "toolchain" / "iverilog-12" / "bin"
DATASETS = ("code-complete-iccad2023", "spec-to-rtl")
GNUBIN = "/opt/homebrew/opt/coreutils/libexec/gnubin"  # GNU seq; BSD seq breaks the Makefile


def harness_env() -> dict[str, str]:
    """PATH with iverilog-12, gmake and GNU seq ahead of the BSD tools."""
    env = os.environ.copy()
    env["PATH"] = f"{IVERILOG_BIN}:{GNUBIN}:/opt/homebrew/bin:" + env.get("PATH", "")
    return env


def load_prompts(dataset: str) -> list[tuple[str, str]]:
    d = CHECKOUT / f"dataset_{dataset}"
    out = []
    for p in sorted(d.glob("*_prompt.txt")):
        out.append((p.name[: -len("_prompt.txt")], p.read_text(encoding="utf-8")))
    return out


def configure_build(dataset: str, build: Path, env: dict[str, str]) -> dict[str, Any]:
    build.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CHECKOUT / "configure"),
        f"--with-task={dataset}",
        "--with-model=manual-rtl-coder",
        "--with-examples=0",
        "--with-samples=1",
        "--with-temperature=0",
        "--with-top-p=0.01",
    ]
    r = subprocess.run(cmd, cwd=build, env=env, capture_output=True, text=True, timeout=300)
    return {"exit_code": r.returncode, "stderr": r.stderr[-800:]}


def run_harness(
    build: Path, env: dict[str, str], problem_ids: list[str]
) -> dict[str, Any]:
    """VERBOSE=1 is mandatory: without it bash 3.2 fails on `&>>` and scores a silent 0%."""
    t = subprocess.run(
        ["gmake", "-j", "4", "sv-iv-test", "VERBOSE=1"],
        cwd=build, env=env, capture_output=True, text=True, timeout=7200,
    )
    a = subprocess.run(
        ["gmake", "sv-iv-analyze", "VERBOSE=1"],
        cwd=build, env=env, capture_output=True, text=True, timeout=1800,
    )
    # The Makefile runs `sv-iv-analyze ... | tee summary.txt`, so the recipe's exit
    # status is tee's and an analyzer crash is invisible. Treat a missing or empty
    # summary.csv as the real failure signal rather than trusting analyze_exit.
    summary = build / "summary.csv"
    text = summary.read_text(encoding="utf-8") if summary.is_file() else ""
    direct_results: dict[str, bool] = {}
    direct_logs: dict[str, dict[str, Any]] = {}
    for problem_id in problem_ids:
        log = build / problem_id / f"{problem_id}_sample01-sv-iv-test.log"
        log_text = log.read_text(encoding="utf-8") if log.is_file() else ""
        direct_results[problem_id] = bool(
            re.search(r"(?m)^Mismatches:\s+0\s+in\s+\d+\s+samples\s*$", log_text)
        )
        direct_logs[problem_id] = {
            "present": log.is_file(),
            "sha256": (
                hashlib.sha256(log.read_bytes()).hexdigest()
                if log.is_file()
                else None
            ),
            "verdict": "PASS" if direct_results[problem_id] else "FAIL",
            "tail": log_text[-800:],
        }
    return {
        "test_exit": t.returncode,
        "analyze_exit": a.returncode,
        "summary_csv": text,
        "summary_csv_present": summary.is_file(),
        "summary_csv_rows": len([l for l in text.splitlines() if l.strip()]),
        "direct_results": direct_results,
        "direct_logs": direct_logs,
        "analyze_exit_is_tee_masked": True,
        "analyze_tail": (a.stdout or "")[-1200:],
        "test_tail": (t.stderr or "")[-800:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test: sample only N prompts per dataset")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=HERE / "e11_pass_at_1_receipt.json")
    ap.add_argument("--sampler-path", help="completed Tinker sampler path to evaluate")
    ap.add_argument("--hf-repo", help="public HF repository for the sampler checkpoint")
    ap.add_argument("--hf-revision", help="immutable pre-created HF checkpoint branch")
    ap.add_argument("--hf-commit", help="immutable 40-hex HF checkpoint commit")
    args = ap.parse_args()

    if args.sampler_path:
        if not all((args.hf_repo, args.hf_revision, args.hf_commit)):
            ap.error("--sampler-path requires --hf-repo, --hf-revision, and --hf-commit")
        if len(args.hf_commit) != 40 or any(c not in "0123456789abcdef" for c in args.hf_commit):
            ap.error("--hf-commit must be an immutable 40-hex commit")

    E11.require_authorization()

    all_prompts = {ds: load_prompts(ds) for ds in DATASETS}
    if args.limit:
        all_prompts = {ds: ps[: args.limit] for ds, ps in all_prompts.items()}
    n = sum(len(v) for v in all_prompts.values())
    chars = sum(len(t) for v in all_prompts.values() for _, t in v)

    projection = E11.project_cost(chars, n, args.max_tokens)
    print(json.dumps({"projection": projection}, indent=2))
    if not projection["within_gate"]:
        print(f"ABORT: projection ${projection['projected_usd']} exceeds gate ${E11.PROJECTION_GATE_USD}")
        return 2

    # --- W&B FIRST, online, before any Tinker client exists ---------------
    os.environ["WANDB_MODE"] = "online"
    import wandb

    model_role = "trained" if args.sampler_path else "base"
    wandb_run = wandb.init(
        entity="arvindcr4-pes-university",
        project="tinker-rl-lab-pavlov",
        group="pavlov-e1-e14-eval-20260809",
        job_type="primary-evaluation",
        name=f"e11_verilog_eval_{model_role}_seed{args.seed}",
        tags=["e11", "verilog_eval", "pass@1", model_role],
        config={
            "suite_id": "verilog_eval",
            "suite_role": "primary_eval",
            "model_id": E11.MODEL_ID,
            "model_revision": E11.MODEL_REVISION,
            "sampler_path": args.sampler_path,
            "hf_repo": args.hf_repo,
            "hf_revision": args.hf_revision,
            "hf_commit": args.hf_commit,
            "samples_per_problem": 1,
            "max_response_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "problems": n,
            "verifier_type": "native iverilog-12 + verilator sv-iv-test",
            "reward_type": "binary pass/fail per problem",
            "is_model_score": True,
            "projected_usd": projection["projected_usd"],
        },
    )
    print(f"W&B run: {wandb_run.url}")

    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(E11.MODEL_ID, revision=E11.MODEL_REVISION)
    service = tinker.ServiceClient(
        user_metadata={"campaign": "pavlov-18usd", "stage": "primary-evaluation", "suite_id": "verilog_eval"}
    )
    sampler = (
        service.create_sampling_client(model_path=args.sampler_path)
        if args.sampler_path
        else service.create_sampling_client(base_model=E11.MODEL_ID)
    )

    def generate(prompt_text: str) -> tuple[str, int, int]:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True
        )
        ids = tok.encode(chat, add_special_tokens=False)
        res = sampler.sample(
            T.ModelInput.from_ints(ids),
            num_samples=1,
            sampling_params=T.SamplingParams(
                max_tokens=args.max_tokens, temperature=args.temperature, top_p=0.95, seed=args.seed
            ),
        ).result()
        seq = res.sequences[0]
        toks = list(seq.tokens)
        return tok.decode(toks, skip_special_tokens=True), len(ids), len(toks)

    env = harness_env()
    per_dataset: dict[str, Any] = {}
    merged: dict[str, bool] = {}
    total_usd = 0.0
    total_pt = total_rt = 0
    extraction_failures = 0

    for ds, prompts in all_prompts.items():
        build = Path(tempfile.mkdtemp(prefix=f"e11_paid_{ds}_"))
        cfg = configure_build(ds, build, env)
        if cfg["exit_code"] != 0:
            print(f"configure failed for {ds}: {cfg['stderr']}")
            return 3
        sampled = E11.sample_all(prompts, generate, build, max_tokens=args.max_tokens)
        harness = run_harness(build, env, [problem_id for problem_id, _ in prompts])
        passed = dict(harness["direct_results"])
        for prob, ok in passed.items():
            merged[f"verilog_eval/{ds}/{prob}"] = ok
        total_usd += sampled["actual_usd"]
        total_pt += sampled["actual_prompt_tokens"]
        total_rt += sampled["actual_resp_tokens"]
        extraction_failures += sampled["extraction_failures"]
        per_dataset[ds] = {
            "build_dir": str(build),
            "sampled": {k: v for k, v in sampled.items() if k != "records"},
            "harness": {k: v for k, v in harness.items() if k != "summary_csv"},
            "scored_problems": len(passed),
        }
        print(f"{ds}: sampled={sampled['problems']} scored={len(passed)} "
              f"extract_fail={sampled['extraction_failures']} usd={sampled['actual_usd']}")

    score = E11.score_pass_at_1(merged)
    receipt = {
        "schema_version": "e11-pass-at-1-receipt-v1",
        "suite_id": "verilog_eval",
        "status": "SCORED",
        "is_model_score": True,
        "evidence_class": "primary_evaluation/pass@1",
        "model": {
            "model_id": E11.MODEL_ID,
            "revision": E11.MODEL_REVISION,
            "source": "tinker_sampler_checkpoint" if args.sampler_path else "immutable_base_model_revision",
            "sampler_path": args.sampler_path,
            "hf_repo": args.hf_repo,
            "hf_revision": args.hf_revision,
            "hf_commit": args.hf_commit,
        },
        "sampling": {"samples_per_problem": 1, "retries": 0, "max_tokens": args.max_tokens,
                     "temperature": args.temperature, "top_p": 0.95, "seed": args.seed},
        "score": score["corrected"]["pass_at_1"],
        "pass_at_1": score,
        "extraction_failures": extraction_failures,
        "verifier": {"iverilog": "12.0", "verilator": "5.050",
                     "native_target": "sv-iv-test + sv-iv-analyze (VERBOSE=1)"},
        "cost": {"actual_usd": round(total_usd, 4), "prompt_tokens": total_pt,
                 "response_tokens": total_rt, "projection": projection},
        "wandb": {"run_id": wandb_run.id,
                  "url": wandb_run.url,
                  "project": "tinker-rl-lab-pavlov"},
        "outstanding_blockers": [
            "decontamination receipt ID does not exist; the split-manifest validator still fails on it"
        ],
        "per_dataset": per_dataset,
        "limit_applied": args.limit or None,
    }
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    wandb_run.log({
        "test/pass_at_1_raw": score["raw"]["pass_at_1"],
        "test/pass_at_1_corrected": score["corrected"]["pass_at_1"],
        "test/passes": score["raw"]["passes"],
        "test/extraction_failures": extraction_failures,
        "cost/actual_usd": round(total_usd, 4),
    })
    wandb_run.finish()

    print(json.dumps({"pass_at_1": score, "actual_usd": round(total_usd, 4),
                      "extraction_failures": extraction_failures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
