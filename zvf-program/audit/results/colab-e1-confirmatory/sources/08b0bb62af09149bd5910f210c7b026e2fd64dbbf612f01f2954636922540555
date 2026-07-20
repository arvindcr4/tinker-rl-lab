#!/usr/bin/env python3
"""Resume E1 from a completed HF training checkpoint and finish evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from typing import Any

from e1_grpo_confirmatory import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    TREATMENT_CHANGES,
    arm_runtime_config,
    arm_training_overrides,
    metric_series,
    load_remote_evaluation_progress,
    package_versions,
    parse_marked_integer,
    prompt_for,
    upload_evaluation_progress,
    validate_evaluation_progress,
)


DEFAULT_EVAL_BATCH_SIZE = 8


def recovered_rollout_count(history: list[dict[str, Any]], arm: str) -> int:
    """Recover the generated-rollout count from immutable trainer telemetry."""
    if arm not in TREATMENT_CHANGES:
        raise ValueError(f"unknown E1 arm: {arm}")
    cumulative_key = {
        "dapo": "dapo/rollouts_cumulative",
        "aero": "aero/rollouts_cumulative",
    }.get(arm)
    if cumulative_key is None:
        return 30 * 16
    values = metric_series(history, cumulative_key, f"train/{cumulative_key}")
    if not values:
        raise RuntimeError(
            f"checkpoint telemetry lacks required {cumulative_key} for {arm} recovery"
        )
    final = values[-1]
    if not float(final).is_integer():
        raise RuntimeError(f"invalid recovered rollout count for {arm}: {final}")
    if arm == "aero":
        # AERO generates three initial samples per prompt and spends at most
        # one additional rescue sample per prompt.  With four prompt groups
        # per step, the frozen 30-step run therefore generates 360--480 real
        # rollouts; inactive output slots are masks, not generated rollouts.
        if not 30 * 12 <= final <= 30 * 16:
            raise RuntimeError(f"invalid recovered rollout count for {arm}: {final}")
    elif final < 30 * 16:
        raise RuntimeError(f"invalid recovered rollout count for {arm}: {final}")
    return int(final)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(TREATMENT_CHANGES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--unit-fingerprint", required=True)
    parser.add_argument("--stack-fingerprint", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=30)
    parser.add_argument("--heldout-n", type=int, default=500)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--progress-save-every", type=int, default=16)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_progress(progress: Any, args: argparse.Namespace) -> dict[str, Any]:
    return validate_evaluation_progress(
        progress,
        unit_fingerprint=args.unit_fingerprint,
        checkpoint_step=args.checkpoint_step,
        heldout_n=args.heldout_n,
    )


def rewind_unhashed_suffix(progress: dict[str, Any]) -> int | None:
    """Discard an unverifiable recovery suffix so it can be replayed with hashes."""
    trace = progress["trace"]
    first_unhashed = next(
        (
            index
            for index, row in enumerate(trace)
            if not isinstance(row.get("completion_sha256"), str)
            or len(row["completion_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["completion_sha256"])
        ),
        None,
    )
    if first_unhashed is None:
        return None
    previous_next_index = progress["next_index"]
    discarded = trace[first_unhashed:]
    progress["correct"] -= sum(row.get("correct") is True for row in discarded)
    del trace[first_unhashed:]
    progress["next_index"] = first_unhashed
    progress["updated_at"] = utc_now()
    progress["rewind_receipt"] = {
        "reason": "missing_completion_sha256",
        "from_next_index": previous_next_index,
        "to_next_index": first_unhashed,
        "discarded_rows": previous_next_index - first_unhashed,
        "repaired_at": progress["updated_at"],
    }
    return first_unhashed


def load_remote_progress(api: Any, args: argparse.Namespace) -> dict[str, Any]:
    return load_remote_evaluation_progress(
        api,
        repo_id=args.hf_repo,
        unit_fingerprint=args.unit_fingerprint,
        checkpoint_step=args.checkpoint_step,
        heldout_n=args.heldout_n,
    )


def upload_progress(api: Any, args: argparse.Namespace, progress: dict[str, Any], path: Path) -> str:
    return upload_evaluation_progress(
        api,
        progress,
        repo_id=args.hf_repo,
        path=path,
    )


def copy_final_adapter(checkpoint: Path, final_dir: Path, tokenizer: Any) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source = checkpoint / name
        if not source.is_file():
            raise RuntimeError(f"checkpoint lacks final adapter file: {source}")
        shutil.copy2(source, final_dir / name)
    tokenizer.save_pretrained(final_dir)


def main() -> int:
    args = parse_args()
    missing = [key for key in ("HF_TOKEN", "WANDB_API_KEY") if not os.environ.get(key)]
    if missing:
        raise SystemExit("missing required secret environment: " + ", ".join(missing))
    if args.heldout_n != 500 or args.checkpoint_step != 30 or args.max_completion_length != 1024:
        raise SystemExit("confirmatory recovery requires checkpoint=30, heldout_n=500, max_completion_length=1024")

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import wandb

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E1 recovery requires CUDA with bfloat16 support")
    os.environ.update(
        {
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_GROUP": args.wandb_group,
            "WANDB_NAME": args.wandb_run_name,
            "WANDB_RUN_ID": args.unit_fingerprint[:8],
            "WANDB_RESUME": "allow",
            "WANDB_MODE": "online",
            "WANDB_SILENT": "true",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    api = HfApi(token=os.environ["HF_TOKEN"])
    root = Path("/content/e1-recovery")
    prefix = f"checkpoints/checkpoint-{args.checkpoint_step}"
    snapshot_download(
        repo_id=args.hf_repo,
        repo_type="model",
        local_dir=root,
        allow_patterns=[f"{prefix}/*", f"{prefix}/**"],
        token=os.environ["HF_TOKEN"],
    )
    checkpoint = root / prefix
    state_path = checkpoint / "trainer_state.json"
    if not state_path.is_file():
        raise RuntimeError(f"checkpoint trainer state is missing: {state_path}")
    trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
    if trainer_state.get("global_step") != args.checkpoint_step:
        raise RuntimeError("checkpoint global step does not equal 30")
    history = trainer_state.get("log_history") or []
    rewards = metric_series(history, "reward", "train/reward")
    zvf = metric_series(history, "frac_reward_zero_std", "train/frac_reward_zero_std")
    step_times = metric_series(history, "step_time", "train/step_time")
    if len(rewards) != 30 or len(zvf) != 30:
        raise RuntimeError(f"checkpoint telemetry is incomplete: rewards={len(rewards)} zvf={len(zvf)}")

    dataset = load_dataset(DATASET_ID, "main", revision=DATASET_REVISION, trust_remote_code=False)
    heldout = dataset["test"].select(range(args.heldout_n))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.generation_config.use_cache = True
    torch.cuda.empty_cache()
    print(
        "[eval-config] "
        + json.dumps(
            {
                "batch_size": args.eval_batch_size,
                "max_new_tokens": args.max_completion_length,
                "use_cache": model.config.use_cache,
                "gradient_checkpointing": model.is_gradient_checkpointing,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        id=args.unit_fingerprint[:8],
        resume="allow",
        name=args.wandb_run_name,
        job_type=args.arm,
        tags=["colab", "e1", "confirmatory", args.arm, "checkpoint-recovery"],
    )
    progress_path = root / "evaluation-progress.json"
    progress = load_remote_progress(api, args)
    rewound_to = rewind_unhashed_suffix(progress)
    if rewound_to is not None:
        repair_commit = upload_progress(api, args, progress, progress_path)
        print(
            f"[eval-rewind] missing completion hashes; rewound to {rewound_to} "
            f"commit={repair_commit}",
            flush=True,
        )
    print(f"[eval-resume] next_index={progress['next_index']}", flush=True)

    try:
        start = int(progress["next_index"])
        for batch_start in range(start, args.heldout_n, args.eval_batch_size):
            batch_end = min(batch_start + args.eval_batch_size, args.heldout_n)
            batch = heldout.select(range(batch_start, batch_end))
            prompts = [
                tokenizer.apply_chat_template(
                    prompt_for(question),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                for question in batch["question"]
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            batch_clock = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_completion_length,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                    cache_implementation="dynamic",
                )
            elapsed = time.monotonic() - batch_clock
            completion_ids = generated[:, encoded.input_ids.shape[1] :]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            for offset, (text, gold) in enumerate(zip(completions, batch["answer"], strict=True)):
                prediction = parse_marked_integer(text)
                target = parse_marked_integer(gold)
                correct = prediction is not None and prediction == target
                progress["correct"] += int(correct)
                progress["trace"].append(
                    {
                        "index": batch_start + offset,
                        "prediction": prediction,
                        "target": target,
                        "correct": correct,
                        "completion_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                )
            progress["next_index"] = batch_end
            progress["evaluation_seconds"] += elapsed
            progress["updated_at"] = utc_now()
            run.log(
                {
                    "eval/completed": batch_end,
                    "eval/running_accuracy": progress["correct"] / batch_end,
                    "eval/batch_seconds": elapsed,
                }
            )
            print(
                f"[eval] {batch_end}/{args.heldout_n} batch_seconds={elapsed:.1f} "
                f"running_accuracy={progress['correct'] / batch_end:.4f}",
                flush=True,
            )
            if batch_end % args.progress_save_every == 0 or batch_end == args.heldout_n:
                upload_progress(api, args, progress, progress_path)

        heldout_score = progress["correct"] / args.heldout_n
        first_five_reward = statistics.fmean(rewards[:5])
        first_five_zvf = statistics.fmean(zvf[:5])
        mean_zvf = statistics.fmean(zvf)
        training_seconds = sum(step_times) if step_times else 0.0
        rollouts = recovered_rollout_count(history, args.arm)
        audit_record = {
            "arm": args.arm,
            "seed": args.seed,
            "heldout_n": args.heldout_n,
            "heldout_score": heldout_score,
            "last10_reward": statistics.fmean(rewards[-10:]),
            "mean_zvf": mean_zvf,
            "mean_gu": 1.0 - mean_zvf,
            "collapse": first_five_zvf >= 0.80 and first_five_reward <= 0.05,
            "rollouts": rollouts,
            "wall_clock_seconds": training_seconds + progress["evaluation_seconds"],
            "stack_fingerprint": args.stack_fingerprint,
            "treatment_changes": TREATMENT_CHANGES[args.arm],
            "manifest_path": "run_manifest.json",
        }
        remote_checkpoint_steps = [5, 10, 15, 20, 25, 30]
        manifest = {
            "schema_version": "e1-colab-confirmatory-run-v1",
            "evidence_class": "confirmatory",
            "completed_at": utc_now(),
            "audit_record": audit_record,
            "run_config": {
                "mode": "confirmatory",
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "max_steps": 30,
                "heldout_n": 500,
                "max_completion_length": 1024,
                "eval_batch_size": args.eval_batch_size,
                "evaluation_progress_save_every": args.progress_save_every,
                "save_steps": 5,
                "unit_fingerprint": args.unit_fingerprint,
                "stack_fingerprint": args.stack_fingerprint,
                "treatment_overrides": arm_training_overrides(args.arm),
                "treatment_runtime": arm_runtime_config(args.arm),
                "treatment_changes": TREATMENT_CHANGES[args.arm],
                "evaluation_recovery": {
                    "source_checkpoint": prefix,
                    "batch_size": args.eval_batch_size,
                    "cache_implementation": "dynamic",
                    "progress_save_every": args.progress_save_every,
                },
            },
            "runtime_versions": package_versions(torch),
            "remote_checkpoint_steps": remote_checkpoint_steps,
            "telemetry": {
                "reward": rewards,
                "zvf": zvf,
                "gu": [1.0 - value for value in zvf],
                "first_five_reward": first_five_reward,
                "first_five_zvf": first_five_zvf,
                "trainer_log_history": history,
            },
            "heldout_trace": progress["trace"],
            "evaluation_progress_path": "evaluation/progress.json",
            "wandb": {"run_id": run.id, "run_url": run.url},
        }
        with tempfile.TemporaryDirectory(prefix="e1-recovery-final-") as tmp:
            staging = Path(tmp)
            copy_final_adapter(checkpoint, staging / "final", tokenizer)
            (staging / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            commit = api.upload_folder(
                repo_id=args.hf_repo,
                repo_type="model",
                folder_path=staging,
                path_in_repo=".",
                commit_message=f"Complete E1 confirmatory {args.arm} seed {args.seed}",
            )
        remote = {
            "hf_repo": args.hf_repo,
            "hf_commit": commit.oid,
            "hf_checkpoint_url": f"https://huggingface.co/{args.hf_repo}/tree/{commit.oid}",
            "hf_manifest_path": "run_manifest.json",
            "hf_final_adapter_path": "final/adapter_model.safetensors",
            "hf_checkpoint_steps": remote_checkpoint_steps,
            "wandb_run_id": run.id,
            "wandb_run_url": run.url,
        }
        run.summary.update({**audit_record, **remote})
        wandb.finish(exit_code=0)
        result = {
            "schema_version": "e1-colab-result-v1",
            "status": "completed",
            "evidence_class": "confirmatory",
            "audit_record": audit_record,
            "remote": remote,
            "run_config": manifest["run_config"],
            "runtime_versions": manifest["runtime_versions"],
            "resume": {"resumed_from_step": 30, "resumed_from_path": prefix},
        }
        print("E1_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException:
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
