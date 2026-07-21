from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .artifacts import (
    CHECKPOINT_STEPS,
    greatest_compatible_checkpoint,
    validate_corpus_manifest,
    validate_full_record,
    with_fingerprint,
)
from .checkpointing import (
    load_checkpoint_bundle,
    load_replay_batch,
    save_checkpoint_bundle,
)
from .evaluation import evaluate_model
from .flops import PROFILED_STEPS, TorchPhaseProfiler, TrainingFlopLedger
from .protocol import (
    REPO_ROOT,
    PilotProtocol,
    PilotUnit,
    build_screening_plan,
    sha256_file,
)
from .remote_core import (
    RemoteContractError,
    require_a100,
    seed_everything,
    source_manifest,
    verify_runtime_versions,
)
from .training import run_replay_step


CHECKPOINT_MANIFEST = re.compile(r"^checkpoints/step-(\d+)/checkpoint_manifest\.json$")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_files(root: Path, files: Mapping[str, str], *, label: str) -> None:
    for relative, digest in files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RemoteContractError(f"{label} file hash mismatch: {relative}")


def download_corpus(
    *,
    api: Any,
    protocol: PilotProtocol,
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    from huggingface_hub import snapshot_download

    repo = plan["identity"]["corpus_hf_repo"]
    info = api.repo_info(repo_id=repo, repo_type="dataset")
    revision = str(info.sha)
    root = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            token=os.environ["HF_TOKEN"],
        )
    )
    manifest = _json(root / "corpus_manifest.json")
    manifest = validate_corpus_manifest(
        manifest,
        protocol=protocol,
        regime=plan["unit"]["regime"],
        seed=plan["unit"]["seed"],
    )
    _verify_files(root, manifest["artifact_files"], label="corpus")
    return root, manifest, revision


def _heldout_source(protocol: PilotProtocol, regime: str) -> tuple[list[str], list[str], list[int]]:
    from datasets import load_dataset

    record = protocol.payload["regimes"][regime]
    if regime == "balanced_equal_length":
        rows = load_dataset(
            record["dataset"],
            "main",
            revision=record["dataset_revision"],
            trust_remote_code=False,
        )["test"].select(range(128))
        questions = [str(row["question"]) for row in rows]
        answers = [str(row["answer"]) for row in rows]
    else:
        rows = load_dataset(
            record["dataset"],
            revision=record["dataset_revision"],
            trust_remote_code=False,
        )["test"].select(range(128))
        questions = [str(row["problem"]) for row in rows]
        answers = [str(row["answer"]) for row in rows]
    return questions, answers, list(range(128))


def _model_stack(protocol: PilotProtocol, seed: int) -> tuple[Any, Any, Any, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    runtime = protocol.payload["runtime"]
    contract = runtime["execution_contract"]
    seed_everything(seed, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["model"]["id"], revision=runtime["model"]["revision"]
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        runtime["model"]["id"],
        revision=runtime["model"]["revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = contract["lora"]
    model = get_peft_model(
        base,
        LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=lora["target_modules"],
            bias=lora["bias"],
            task_type="CAUSAL_LM",
        ),
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RemoteContractError("LoRA model exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(contract["learning_rate"]),
        weight_decay=float(contract["weight_decay"]),
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(contract["warmup_steps"]),
        num_training_steps=int(runtime["steps"]),
    )
    return model, tokenizer, optimizer, scheduler


def _remote_checkpoint_manifests(
    *,
    api: Any,
    repo: str,
) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        if type(exc).__name__ in {"RepositoryNotFoundError", "EntryNotFoundError"}:
            return []
        raise
    manifests: list[dict[str, Any]] = []
    for name in files:
        if CHECKPOINT_MANIFEST.match(name):
            path = hf_hub_download(
                repo_id=repo,
                repo_type="model",
                filename=name,
                token=os.environ["HF_TOKEN"],
            )
            manifests.append(_json(Path(path)))
    return manifests


def _download_checkpoint(*, repo: str, step: int) -> Path:
    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            repo_id=repo,
            repo_type="model",
            allow_patterns=[f"checkpoints/step-{step}/**"],
            token=os.environ["HF_TOKEN"],
        )
    )
    return snapshot / "checkpoints" / f"step-{step}"


def _checkpoint_upload(
    *,
    api: Any,
    repo: str,
    root: Path,
    step: int,
) -> str:
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=root,
        path_in_repo=f"checkpoints/step-{step}",
        commit_message=f"Flagship pilot checkpoint step {step}",
    )
    return str(commit.oid)


def _evaluation(
    *,
    model: Any,
    tokenizer: Any,
    protocol: PilotProtocol,
    plan: Mapping[str, Any],
    questions: list[str],
    answers: list[str],
    source_indices: list[int],
    output_root: Path,
    step: int,
) -> dict[str, Any]:
    contract = protocol.payload["runtime"]["execution_contract"]
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    evidence_path = output_root / "evaluations" / f"step-{step:03d}.jsonl"
    try:
        summary = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            regime=plan["unit"]["regime"],
            questions=questions,
            answers=answers,
            source_indices=source_indices,
            output_path=evidence_path,
            max_prompt_length=int(contract["max_prompt_length"]),
            max_completion_length=int(contract["max_completion_length"]),
            batch_size=8,
        )
    finally:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.train()
        torch.cuda.empty_cache()
    return {
        "step": step,
        **summary,
        "evidence_path": f"evaluations/{evidence_path.name}",
    }


def _final_artifacts(
    *,
    api: Any,
    repo: str,
    model: Any,
    tokenizer: Any,
    output_root: Path,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
    corpus_commit: str,
    receipts: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    ledger_record: Mapping[str, Any],
    sources: Mapping[str, str],
) -> tuple[dict[str, Any], str, str]:
    staging = output_root / "final-staging"
    staging.mkdir(parents=True, exist_ok=True)
    adapter = staging / "final" / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(staging / "final" / "tokenizer")
    for evaluation in evaluations:
        source = output_root / evaluation["evidence_path"]
        target = staging / evaluation["evidence_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    manifest = {
        "schema_version": "flagship-pilot-run-manifest-v1",
        "plan": plan,
        "corpus_fingerprint": corpus["fingerprint"],
        "corpus_commit": corpus_commit,
        "gradient_receipts": receipts,
        "evaluations": evaluations,
        "token_flop_ledger": ledger_record,
        "source_manifest": dict(sorted(sources.items())),
    }
    manifest_path = staging / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifact_commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=staging,
        path_in_repo=".",
        commit_message="Complete flagship pilot unit artifacts",
    )
    return manifest, sha256_file(manifest_path), str(artifact_commit.oid)


def train_unit(
    *,
    protocol: PilotProtocol,
    condition: str,
    regime: str,
    seed: int,
    wandb_project: str,
    wandb_entity: str | None,
) -> dict[str, Any]:
    protocol.require_gpu_authorization()
    missing = [key for key in ("HF_TOKEN", "WANDB_API_KEY") if not os.environ.get(key)]
    if missing:
        raise RemoteContractError(f"required credentials are missing: {', '.join(missing)}")

    import wandb
    from huggingface_hub import HfApi

    versions = verify_runtime_versions(protocol)
    accelerator = require_a100(torch)
    unit = PilotUnit(condition=condition, regime=regime, seed=seed)
    plan = build_screening_plan(protocol, unit)
    identity = plan["identity"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=identity["hf_repo"], repo_type="model", private=True, exist_ok=True)
    corpus_root, corpus, corpus_commit = download_corpus(
        api=api,
        protocol=protocol,
        plan=plan,
    )
    model, tokenizer, optimizer, scheduler = _model_stack(protocol, seed)
    questions, answers, source_indices = _heldout_source(protocol, regime)
    sources = source_manifest(REPO_ROOT)
    output_root = Path("/content/flagship-pilot-output") / unit.unit_id
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = _remote_checkpoint_manifests(api=api, repo=identity["hf_repo"])
    selected, rejected = greatest_compatible_checkpoint(
        manifests,
        plan=plan,
        corpus=corpus,
    )
    receipts: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    ledger = TrainingFlopLedger()
    start_step = 0
    if selected is not None:
        checkpoint_root = _download_checkpoint(repo=identity["hf_repo"], step=selected["step"])
        state, ledger, _ = load_checkpoint_bundle(
            root=checkpoint_root,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            plan=plan,
            corpus=corpus,
        )
        start_step = int(state["step"])
        receipts = list(state["gradient_receipts"])
        evaluations = list(state["evaluations"])
        for evaluation in evaluations:
            source = checkpoint_root / evaluation["evidence_path"]
            target = output_root / evaluation["evidence_path"]
            if not source.is_file():
                raise RemoteContractError(
                    f"resumed evaluation evidence is missing: {evaluation['evidence_path']}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        id=plan["fingerprint"][:8],
        resume="allow",
        group=identity["wandb_group"],
        name=identity["wandb_run"],
        config={
            "plan": plan,
            "corpus_fingerprint": corpus["fingerprint"],
            "corpus_commit": corpus_commit,
            "runtime_versions": versions,
            "accelerator": accelerator,
            "rejected_checkpoint_manifests": list(rejected),
        },
    )
    started_at = time.monotonic()
    if not evaluations:
        initial = _evaluation(
            model=model,
            tokenizer=tokenizer,
            protocol=protocol,
            plan=plan,
            questions=questions,
            answers=answers,
            source_indices=source_indices,
            output_root=output_root,
            step=0,
        )
        evaluations.append(initial)
        run.log({"eval/accuracy": initial["accuracy"], "eval/generated_tokens": initial["generated_tokens"]}, step=0)

    try:
        for step in range(start_step + 1, 101):
            group_record = corpus["groups"][step - 1]
            group_path = corpus_root / group_record["artifact_path"]
            batch = load_replay_batch(
                group_path,
                expected_fingerprint=group_record["fingerprint"],
            )
            profiler = TorchPhaseProfiler(torch, enabled=step in PROFILED_STEPS)
            receipt = run_replay_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                batch=batch,
                condition=condition,
                step=step,
                max_grad_norm=float(
                    protocol.payload["runtime"]["execution_contract"]["max_grad_norm"]
                ),
                phase_context=profiler,
            )
            if profiler.enabled:
                profiler.require_training_coverage()
            padded_model_tokens = int(batch.prompt_ids.numel() + batch.completion_ids.numel())
            ledger.add_step(
                step=step,
                active_tokens=receipt.active_tokens,
                padded_tokens=padded_model_tokens,
                phase_flops=profiler.phase_flops if profiler.enabled else None,
            )
            receipts.append(receipt.as_record())
            run.log(
                {
                    "train/selected_loss": receipt.selected_loss,
                    "mechanism/gradient_cosine": receipt.gradient_cosine,
                    "mechanism/gradient_relative_l2": receipt.gradient_relative_l2,
                    "mechanism/active_tokens": receipt.active_tokens,
                },
                step=step,
            )
            if step in CHECKPOINT_STEPS:
                evaluation = _evaluation(
                    model=model,
                    tokenizer=tokenizer,
                    protocol=protocol,
                    plan=plan,
                    questions=questions,
                    answers=answers,
                    source_indices=source_indices,
                    output_root=output_root,
                    step=step,
                )
                evaluations.append(evaluation)
                run.log(
                    {
                        "eval/accuracy": evaluation["accuracy"],
                        "eval/generated_tokens": evaluation["generated_tokens"],
                    },
                    step=step,
                )
                checkpoint_root = output_root / "checkpoints" / f"step-{step}"
                save_checkpoint_bundle(
                    destination=checkpoint_root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    plan=plan,
                    corpus=corpus,
                    receipts=receipts,
                    evaluations=evaluations,
                    flop_ledger=ledger,
                    source_hashes=sources,
                    evaluation_files={
                        int(item["step"]): output_root / item["evidence_path"]
                        for item in evaluations
                    },
                )
                checkpoint_commit = _checkpoint_upload(
                    api=api,
                    repo=identity["hf_repo"],
                    root=checkpoint_root,
                    step=step,
                )
                run.log({"checkpoint/commit": checkpoint_commit}, step=step)

        ledger_record = {
            **ledger.final_record(),
            "charged_generated_tokens": corpus["charged_generated_tokens"],
            "replay_generation_flops": corpus["flop_ledger"]["replay_generation_flops"],
        }
        manifest, manifest_sha, artifact_commit = _final_artifacts(
            api=api,
            repo=identity["hf_repo"],
            model=model,
            tokenizer=tokenizer,
            output_root=output_root,
            plan=plan,
            corpus=corpus,
            corpus_commit=corpus_commit,
            receipts=receipts,
            evaluations=evaluations,
            ledger_record=ledger_record,
            sources=sources,
        )
        adapter_path = output_root / "final-staging/final/adapter/adapter_model.safetensors"
        full_record = with_fingerprint(
            {
                "schema_version": "flagship-pilot-unit-v1",
                "status": "completed",
                "unit_fingerprint": plan["fingerprint"],
                "corpus_fingerprint": corpus["fingerprint"],
                "condition": condition,
                "regime": regime,
                "seed": seed,
                "training_steps": 100,
                "gradient_receipt_count": len(receipts),
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "evaluations": evaluations,
                "token_flop_ledger": ledger_record,
                "wall_clock_seconds": time.monotonic() - started_at,
                "runtime_versions": versions,
                "accelerator": accelerator,
                "wandb": {
                    "state": "finished",
                    "run_id": run.id,
                    "run_url": run.url,
                    "entity": run.entity,
                    "project": run.project,
                },
                "hugging_face": {
                    "private": True,
                    "repo": identity["hf_repo"],
                    "commit": artifact_commit,
                    "checkpoint_steps": list(CHECKPOINT_STEPS),
                    "final_adapter_sha256": sha256_file(adapter_path),
                    "manifest_sha256": manifest_sha,
                },
                "manifest": manifest,
            }
        )
        validate_full_record(full_record, plan=plan, corpus=corpus)
        full_record_path = output_root / "full_record.json"
        full_record_path.write_text(json.dumps(full_record, indent=2, sort_keys=True) + "\n")
        final_record_commit = api.upload_file(
            repo_id=identity["hf_repo"],
            repo_type="model",
            path_or_fileobj=full_record_path,
            path_in_repo="full_record.json",
            commit_message="Publish validated flagship pilot full record",
        )
        result = {
            "status": "completed",
            "unit": unit.unit_id,
            "unit_fingerprint": plan["fingerprint"],
            "corpus_fingerprint": corpus["fingerprint"],
            "hf_repo": identity["hf_repo"],
            "hf_artifact_commit": artifact_commit,
            "hf_record_commit": str(final_record_commit.oid),
            "wandb_run_id": run.id,
            "wandb_run_url": run.url,
            "final_accuracy": evaluations[-1]["accuracy"],
            "full_record_fingerprint": full_record["fingerprint"],
        }
        run.summary.update(result)
        wandb.finish(exit_code=0)
        return result
    except BaseException:
        wandb.finish(exit_code=1)
        raise
