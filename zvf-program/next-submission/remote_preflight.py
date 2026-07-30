#!/usr/bin/env python3
"""Run one tracked next-submission sampler preflight on a Colab A100."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

from contrast_sampler import ARM_BASELINE, ARM_EARLY_STOP
from trl_sampler_adapter import (
    build_sampler_trainer_class,
    make_rollout_func,
    prompt_key,
    sampler_reward,
)


MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GSM8K_ID = "openai/gsm8k"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MATH_TRAIN_ID = "DigitalLearningGmbH/MATH-lighteval"
MATH_TRAIN_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"
MATH_EVAL_ID = "HuggingFaceH4/MATH-500"
MATH_EVAL_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
ARMS = (ARM_BASELINE, ARM_EARLY_STOP)
TASKS = ("gsm8k", "math500")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--unit-fingerprint", required=True)
    parser.add_argument("--stack-fingerprint", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-group", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    return parser.parse_args(argv)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        tail = completion[-1]
        if isinstance(tail, dict):
            return str(tail.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def parse_marked_integer(text: str) -> int | None:
    if "####" not in text:
        return None
    match = re.search(r"-?\d[\d,]*", text.rsplit("####", 1)[1])
    if match is None:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def gsm8k_reward(response: str, answer: str) -> float:
    prediction = parse_marked_integer(response)
    target = parse_marked_integer(answer)
    return float(prediction is not None and target is not None and prediction == target)


def last_boxed(text: str) -> str | None:
    index = text.rfind("\\boxed{")
    if index == -1:
        return None
    cursor, depth, output = index + 7, 1, []
    while cursor < len(text) and depth:
        character = text[cursor]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if not depth:
                break
        output.append(character)
        cursor += 1
    return "".join(output) if depth == 0 else None


def normalize_math(answer: str) -> str:
    value = answer.strip().replace("\\left", "").replace("\\right", "")
    for junk in ("\\!", "\\,", "\\;", " "):
        value = value.replace(junk, "")
    value = value.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1]
    return value


def math500_reward(response: str, answer: str) -> float:
    prediction = last_boxed(response)
    return float(prediction is not None and normalize_math(prediction) == normalize_math(answer))


def prompt_for(task: str, question: str) -> list[dict[str, str]]:
    if task == "gsm8k":
        instruction = (
            "Solve this math problem. Show your reasoning, then end your response "
            "with exactly `#### <integer>`."
        )
    elif task == "math500":
        instruction = (
            "Solve this competition mathematics problem carefully. Show your reasoning, "
            "then put the final answer inside exactly one \\boxed{...}."
        )
    else:
        raise ValueError(f"unknown task: {task}")
    return [{"role": "user", "content": f"{instruction}\n\n{question}"}]


def package_versions(torch_module: Any) -> dict[str, str]:
    versions = {}
    for name in ("trl", "transformers", "datasets", "peft", "torchao", "wandb", "huggingface_hub"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    versions.update(
        {
            "torch": torch_module.__version__,
            "cuda": str(torch_module.version.cuda),
            "gpu": torch_module.cuda.get_device_name(0),
        }
    )
    return versions


def dataset_manifest_hash(rows: Any) -> str:
    digest = hashlib.sha256()
    for index in range(len(rows)):
        row = rows[index]
        payload = {"index": index, "prompt": row["prompt"], "answer": row["answer"]}
        digest.update(
            (
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            ).encode()
        )
    return digest.hexdigest()


def load_task_data(
    load_dataset: Any, task: str, heldout_n: int
) -> tuple[Any, Any, Callable[[str, str], float], dict[str, str]]:
    if task == "gsm8k":
        source = load_dataset(GSM8K_ID, "main", revision=GSM8K_REVISION, trust_remote_code=False)
        train = source["train"].map(
            lambda row: {"prompt": prompt_for(task, row["question"]), "answer": row["answer"]}
        )
        heldout = (
            source["test"]
            .select(range(heldout_n))
            .map(lambda row: {"prompt": prompt_for(task, row["question"]), "answer": row["answer"]})
        )
        bindings = {
            "training_dataset": GSM8K_ID,
            "training_revision": GSM8K_REVISION,
            "evaluation_dataset": GSM8K_ID,
            "evaluation_revision": GSM8K_REVISION,
        }
        return train, heldout, gsm8k_reward, bindings

    train_source = load_dataset(
        MATH_TRAIN_ID,
        revision=MATH_TRAIN_REVISION,
        split="train",
        trust_remote_code=False,
    )

    def standardize_math_train(row: dict[str, Any]) -> dict[str, Any]:
        answer = last_boxed(str(row["solution"]))
        if answer is None:
            raise RuntimeError("MATH training solution lacks a final boxed answer")
        return {"prompt": prompt_for(task, str(row["problem"])), "answer": answer}

    train = train_source.map(standardize_math_train)
    heldout_source = load_dataset(
        MATH_EVAL_ID,
        revision=MATH_EVAL_REVISION,
        split="test",
        trust_remote_code=False,
    ).select(range(heldout_n))
    heldout = heldout_source.map(
        lambda row: {"prompt": prompt_for(task, str(row["problem"])), "answer": str(row["answer"])}
    )
    bindings = {
        "training_dataset": MATH_TRAIN_ID,
        "training_revision": MATH_TRAIN_REVISION,
        "evaluation_dataset": MATH_EVAL_ID,
        "evaluation_revision": MATH_EVAL_REVISION,
    }
    return train, heldout, math500_reward, bindings


def answer_map(rows: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in rows:
        key = prompt_key(row["prompt"])
        answer = str(row["answer"])
        if key in values and values[key] != answer:
            raise RuntimeError("one training prompt maps to conflicting verifier targets")
        values[key] = answer
    return values


def heldout_accuracy(
    model: Any,
    tokenizer: Any,
    rows: Any,
    *,
    reward_parser: Callable[[str, str], float],
    batch_size: int,
    max_new_tokens: int,
    torch_module: Any,
) -> tuple[float, list[dict[str, Any]], float]:
    model.eval()
    trace: list[dict[str, Any]] = []
    correct = 0
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        batch = rows.select(range(start, min(start + batch_size, len(rows))))
        prompts = [
            tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for prompt in batch["prompt"]
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch_module.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                cache_implementation="dynamic",
            )
        completion_ids = generated[:, encoded.input_ids.shape[1] :]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for offset, (text, gold) in enumerate(zip(completions, batch["answer"], strict=True)):
            is_correct = reward_parser(text, gold) == 1.0
            correct += int(is_correct)
            trace.append(
                {
                    "index": start + offset,
                    "correct": is_correct,
                    "completion_sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    return correct / len(rows), trace, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed <= 0 or args.max_completion_length <= 0 or args.eval_batch_size <= 0:
        raise SystemExit("seed, completion length, and evaluation batch size must be positive")
    missing = [key for key in ("HF_TOKEN", "WANDB_API_KEY") if not os.environ.get(key)]
    if missing:
        raise SystemExit("missing required secret environment: " + ", ".join(missing))

    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    import wandb

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("preflight requires CUDA with bfloat16 support")

    os.environ.update(
        {
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_GROUP": args.wandb_group,
            "WANDB_NAME": args.wandb_run_name,
            "WANDB_RUN_ID": args.unit_fingerprint[:8],
            "WANDB_RESUME": "allow",
            "WANDB_JOB_TYPE": "next-submission-preflight",
            "WANDB_MODE": "online",
            "WANDB_SILENT": "true",
            "TOKENIZERS_PARALLELISM": "false",
            "TRL_EXPERIMENTAL_SILENCE": "1",
        }
    )
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=args.hf_repo, repo_type="model", private=True, exist_ok=True)
    started_at = utc_now()
    wall_started = time.monotonic()
    runtime_versions = package_versions(torch)
    train_dataset, heldout, reward_parser, dataset_bindings = load_task_data(
        load_dataset, args.task, 8
    )
    train_manifest_sha256 = dataset_manifest_hash(train_dataset)
    evaluation_manifest_sha256 = dataset_manifest_hash(heldout)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    treatment = {
        "arm": args.arm,
        "group_size": 8,
        "initial_group_size": 2,
        "expansion_group_size": 6,
        "expand_rule": "expand iff the first two binary verifier rewards differ",
        "homogeneous_rule": "no update and no replacement prompt",
    }
    objective = {
        "epsilon": 0.2,
        "epsilon_high": 0.2,
        "importance_sampling_level": "token",
        "scale_rewards": "group",
        "loss_type": "grpo",
        "beta": 0.0,
    }
    training_args = GRPOConfig(
        output_dir="/content/next-preflight-output",
        run_name=args.wandb_run_name,
        report_to=["wandb"],
        model_init_kwargs={
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        },
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        generation_batch_size=16,
        num_generations=8,
        max_steps=1,
        max_completion_length=args.max_completion_length,
        learning_rate=1e-6,
        lr_scheduler_type="linear",
        warmup_steps=0,
        optim="adamw_torch_fused",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        chat_template_kwargs={"enable_thinking": True},
        num_iterations=1,
        mask_truncated_completions=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        push_to_hub=False,
        seed=args.seed,
        data_seed=args.seed,
        shuffle_dataset=True,
        remove_unused_columns=False,
        disable_tqdm=False,
        **objective,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        bias="none",
    )
    trainer_class = build_sampler_trainer_class(GRPOTrainer)
    trainer = trainer_class(
        model=MODEL_ID,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=sampler_reward,
        processing_class=tokenizer,
        peft_config=peft_config,
        rollout_func=make_rollout_func(
            arm=args.arm,
            answer_by_prompt=answer_map(train_dataset),
            reward_parser=reward_parser,
        ),
    )

    try:
        trainer.train()
        cumulative = dict(trainer._next_sampler_cumulative)
        rollout_groups = int(cumulative["rollout_groups"])
        if rollout_groups <= 0:
            raise RuntimeError("sampler emitted no rollout groups")
        class_total = sum(
            int(cumulative[f"{label}_groups"]) for label in ("all_wrong", "all_correct", "mixed")
        )
        if class_total != rollout_groups:
            raise RuntimeError("sampler class counts do not partition rollout groups")

        trainer.model.gradient_checkpointing_disable()
        trainer.model.config.use_cache = True
        trainer.model.generation_config.use_cache = True
        torch.cuda.empty_cache()
        heldout_score, heldout_trace, evaluation_seconds = heldout_accuracy(
            trainer.model,
            tokenizer,
            heldout,
            reward_parser=reward_parser,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.max_completion_length,
            torch_module=torch,
        )
        wandb_run = wandb.run
        if wandb_run is None:
            raise RuntimeError("W&B integration did not create a run")

        sampler_telemetry = {
            "charged_generated_tokens": int(cumulative["charged_generated_tokens"]),
            "generated_rollouts": int(cumulative["generated_rollouts"]),
            "rollout_groups": rollout_groups,
            "updated_groups": int(cumulative["updated_groups"]),
            "all_wrong_fraction": cumulative["all_wrong_groups"] / rollout_groups,
            "all_correct_fraction": cumulative["all_correct_groups"] / rollout_groups,
            "mixed_fraction": cumulative["mixed_groups"] / rollout_groups,
        }
        run_config = {
            "task": args.task,
            "arm": args.arm,
            "seed": args.seed,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "max_steps": 1,
            "heldout_n": 8,
            "max_completion_length": args.max_completion_length,
            "unit_fingerprint": args.unit_fingerprint,
            "stack_fingerprint": args.stack_fingerprint,
            "source_commit": args.source_commit,
            "protocol_sha256": args.protocol_sha256,
            "objective_fingerprint": canonical_hash(objective),
            "treatment": treatment,
            "dataset_bindings": dataset_bindings,
            "train_manifest_sha256": train_manifest_sha256,
            "evaluation_manifest_sha256": evaluation_manifest_sha256,
        }
        audit_record = {
            "task_id": args.task,
            "arm_id": args.arm,
            "seed": args.seed,
            "status": "complete",
            "evidence_tier": "preflight_not_scientific_evidence",
            "heldout_correct": sum(row["correct"] for row in heldout_trace),
            "heldout_n": 8,
            "heldout_accuracy": heldout_score,
            "wall_clock_seconds": time.monotonic() - wall_started,
            "evaluation_seconds": evaluation_seconds,
            **sampler_telemetry,
        }
        manifest = {
            "schema_version": "aiml-next-preflight-run-v1",
            "status": "complete",
            "evidence_class": "preflight-not-evidence",
            "started_at": started_at,
            "completed_at": utc_now(),
            "audit_record": audit_record,
            "run_config": run_config,
            "runtime_versions": runtime_versions,
            "heldout_trace": heldout_trace,
            "wandb": {"run_id": wandb_run.id, "run_url": wandb_run.url},
            "source_files": {
                path.name: sha256_file(path)
                for path in (
                    Path("/content/remote_preflight.py"),
                    Path("/content/contrast_sampler.py"),
                    Path("/content/trl_sampler_adapter.py"),
                )
            },
        }

        with tempfile.TemporaryDirectory(prefix="next-preflight-final-") as temporary:
            staging = Path(temporary)
            final_dir = staging / "final"
            trainer.save_model(str(final_dir))
            tokenizer.save_pretrained(final_dir)
            (staging / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            commit = api.upload_folder(
                repo_id=args.hf_repo,
                repo_type="model",
                folder_path=staging,
                path_in_repo=".",
                commit_message=(
                    f"Next submission preflight {args.task} {args.arm} seed {args.seed}"
                ),
            )

        remote = {
            "hf_repo": args.hf_repo,
            "hf_commit": commit.oid,
            "hf_manifest_path": "run_manifest.json",
            "hf_final_adapter_path": "final/adapter_model.safetensors",
            "wandb_run_id": wandb_run.id,
            "wandb_run_url": wandb_run.url,
        }
        wandb_run.config.update(
            {"treatment": treatment, "run_config": run_config}, allow_val_change=True
        )
        wandb_run.summary.update({**audit_record, **remote})
        wandb.finish(exit_code=0)
        result = {
            "schema_version": "aiml-next-preflight-result-v1",
            "status": "completed",
            "evidence_class": "preflight-not-evidence",
            "audit_record": audit_record,
            "run_config": run_config,
            "runtime_versions": runtime_versions,
            "remote": remote,
        }
        print("NEXT_PREFLIGHT_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except BaseException:
        if wandb.run is not None:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
