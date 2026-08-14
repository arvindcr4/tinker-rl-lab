"""Tracked pass@1 validation for the Pavlov API-Bank + SWE-Gym portfolio.

This is a source-validation score, not an E1--E14 benchmark score.  It exists
to compare the base model and a trained Tinker sampler on held-back rows from
the two training sources while the native E-suite harnesses run separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .grpo import (
    PAVLOV_NON_XLAM_DATASET_REVISION,
    PAVLOV_NON_XLAM_SOURCE_REVISIONS,
    PavlovNonXLAMReward,
    TrainingExample,
    _decode_response,
    make_pavlov_non_xlam_dataset,
)

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
WANDB_ENTITY = "arvindcr4-pes-university"
WANDB_PROJECT = "tinker-rl-lab-pavlov"
WANDB_GROUP = "pavlov-portfolio-eval-20260809"


def select_examples(
    examples: Sequence[TrainingExample], *, per_suite: int
) -> list[TrainingExample]:
    """Take the first deterministic ``per_suite`` examples from each suite."""
    if per_suite < 1:
        raise ValueError("per_suite must be positive")
    selected: list[TrainingExample] = []
    counts: dict[str, int] = {}
    for example in examples:
        suite = str(example.metadata.get("suite_id") or "")
        if not suite or counts.get(suite, 0) >= per_suite:
            continue
        selected.append(example)
        counts[suite] = counts.get(suite, 0) + 1
    expected = {"api_bank_rlvr_train", "swe_gym_train"}
    if set(counts) != expected or any(counts[suite] != per_suite for suite in expected):
        raise RuntimeError(f"validation selection is incomplete: {counts}")
    return selected


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sampler-path")
    source.add_argument("--base-model", action="store_true")
    parser.add_argument("--hf-repo")
    parser.add_argument("--hf-revision")
    parser.add_argument("--hf-commit")
    parser.add_argument("--per-suite", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1809)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not os.environ.get("TINKER_API_KEY", "").startswith("tml-"):
        raise SystemExit("TINKER_API_KEY is required")
    if args.sampler_path:
        if not all((args.hf_repo, args.hf_revision, args.hf_commit)):
            raise SystemExit("trained sampler evaluation requires HF repo, revision, and commit")
        if len(args.hf_commit) != 40 or any(c not in "0123456789abcdef" for c in args.hf_commit):
            raise SystemExit("--hf-commit must be an immutable 40-hex commit")

    dataset = make_pavlov_non_xlam_dataset(seed=809)
    examples = select_examples(list(dataset.test_examples()), per_suite=args.per_suite)

    os.environ["WANDB_MODE"] = "online"
    import wandb

    role = "trained" if args.sampler_path else "base"
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="portfolio-source-validation",
        name=f"pavlov_portfolio_{role}_seed{args.seed}",
        tags=["pavlov", "portfolio-validation", role, "api-bank", "swe-gym"],
        config={
            "evidence_class": "portfolio_source_validation",
            "is_e1_e14_score": False,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampler_path": args.sampler_path,
            "hf_repo": args.hf_repo,
            "hf_revision": args.hf_revision,
            "hf_commit": args.hf_commit,
            "dataset_revision": PAVLOV_NON_XLAM_DATASET_REVISION,
            "source_revisions": PAVLOV_NON_XLAM_SOURCE_REVISIONS,
            "samples_per_problem": 1,
            "retries": 0,
            "per_suite": args.per_suite,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    )
    try:
        import tinker
        import tinker.types as T
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        service = tinker.ServiceClient(
            user_metadata={"campaign": "pavlov-portfolio", "stage": "source-validation"}
        )
        sampler = (
            service.create_sampling_client(model_path=args.sampler_path)
            if args.sampler_path
            else service.create_sampling_client(base_model=MODEL_ID)
        )
        reward = PavlovNonXLAMReward()
        records: list[dict[str, Any]] = []
        for index, example in enumerate(examples):
            prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)[:2048]
            result = sampler.sample(
                T.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=T.SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=0.95,
                    seed=args.seed + index,
                ),
            ).result()
            sequence = result.sequences[0]
            response = _decode_response(tokenizer, sequence)
            score = float(reward.score(response, example))
            record = {
                "suite_id": example.metadata["suite_id"],
                "source_id": example.metadata["source_id"],
                "prompt_sha256": _sha256_text(example.prompt),
                "response_sha256": _sha256_text(response),
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(sequence.tokens),
                "score": score,
                "exact": score == 1.0,
            }
            records.append(record)
            run.log(
                {
                    "eval/index": index,
                    "eval/score": score,
                    f"eval/{record['suite_id']}/score": score,
                    "eval/prompt_tokens": record["prompt_tokens"],
                    "eval/response_tokens": record["response_tokens"],
                },
                step=index,
            )

        suites: dict[str, dict[str, Any]] = {}
        for suite in sorted({str(row["suite_id"]) for row in records}):
            rows = [row for row in records if row["suite_id"] == suite]
            suites[suite] = {
                "count": len(rows),
                "mean_reward": sum(float(row["score"]) for row in rows) / len(rows),
                "exact_matches": sum(bool(row["exact"]) for row in rows),
                "exact_match_rate": sum(bool(row["exact"]) for row in rows) / len(rows),
            }
        mean_reward = sum(float(row["score"]) for row in records) / len(records)
        exact_rate = sum(bool(row["exact"]) for row in records) / len(records)
        receipt = {
            "schema_version": "pavlov-portfolio-source-eval-v1",
            "status": "SCORED",
            "evidence_class": "portfolio_source_validation",
            "is_model_score": True,
            "is_e1_e14_score": False,
            "role": role,
            "model": {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "sampler_path": args.sampler_path,
                "hf_repo": args.hf_repo,
                "hf_revision": args.hf_revision,
                "hf_commit": args.hf_commit,
            },
            "dataset_revision": PAVLOV_NON_XLAM_DATASET_REVISION,
            "source_revisions": PAVLOV_NON_XLAM_SOURCE_REVISIONS,
            "sampling": {
                "samples_per_problem": 1,
                "retries": 0,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
            "count": len(records),
            "mean_reward": mean_reward,
            "exact_match_rate": exact_rate,
            "suites": suites,
            "records": records,
            "wandb": {"run_id": run.id, "url": run.url, "project": WANDB_PROJECT},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run.summary["eval/mean_reward"] = mean_reward
        run.summary["eval/exact_match_rate"] = exact_rate
        run.summary["eval/count"] = len(records)
        run.finish(exit_code=0)
        print(json.dumps({k: receipt[k] for k in ("status", "role", "count", "mean_reward", "exact_match_rate", "suites", "wandb")}, indent=2))
        return 0
    except Exception:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
