from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .artifacts import validate_checkpoint_manifest, with_fingerprint
from .flops import TrainingFlopLedger
from .protocol import sha256_file
from .training import ReplayBatch


class CheckpointContractError(RuntimeError):
    """A checkpoint bundle is incomplete, stale, or cannot be restored exactly."""


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_replay_batch(path: Path, *, expected_fingerprint: str) -> ReplayBatch:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "flagship-pilot-replay-group-v1":
        raise CheckpointContractError(f"wrong replay group schema: {path}")
    group = payload.get("group")
    if not isinstance(group, dict) or group.get("fingerprint") != expected_fingerprint:
        raise CheckpointContractError(f"replay group fingerprint mismatch: {path}")
    try:
        batch = ReplayBatch(
            group_fingerprint=expected_fingerprint,
            prompt_ids=payload["prompt_ids"],
            prompt_mask=payload["prompt_mask"],
            completion_ids=payload["completion_ids"],
            completion_mask=payload["completion_mask"],
            rewards=payload["rewards"],
            active_rows=payload["active_rows"],
            old_logps=payload["old_logps"],
        )
    except KeyError as exc:
        raise CheckpointContractError(f"replay group tensor is missing: {exc.args[0]}") from exc
    tensors = (
        batch.prompt_ids,
        batch.prompt_mask,
        batch.completion_ids,
        batch.completion_mask,
        batch.rewards,
        batch.active_rows,
        batch.old_logps,
    )
    if not all(torch.is_tensor(tensor) for tensor in tensors):
        raise CheckpointContractError("replay payload contains a non-tensor training field")
    if not torch.isfinite(batch.old_logps).all():
        raise CheckpointContractError("replay old log probabilities are non-finite")
    return batch


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise CheckpointContractError("RNG checkpoint fields are incomplete or unexpected")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_files(root: Path, *, evaluation_steps: Sequence[int]) -> dict[str, str]:
    required = (
        "adapter/adapter_model.safetensors",
        "adapter/adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pt",
        "training_state.json",
        "token_flop_ledger.json",
        "source_manifest.json",
    )
    files: dict[str, str] = {}
    for relative in required:
        path = root / relative
        if not path.is_file():
            raise CheckpointContractError(f"checkpoint file is missing: {relative}")
        files[relative] = sha256_file(path)
    for step in evaluation_steps:
        relative = f"evaluations/step-{step:03d}.jsonl"
        path = root / relative
        if not path.is_file():
            raise CheckpointContractError(f"checkpoint evaluation evidence is missing: {relative}")
        files[relative] = sha256_file(path)
    return files


def save_checkpoint_bundle(
    *,
    destination: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    flop_ledger: TrainingFlopLedger,
    source_hashes: Mapping[str, str],
    evaluation_files: Mapping[int, Path],
) -> dict[str, Any]:
    if len(receipts) != step:
        raise CheckpointContractError("receipt count does not match checkpoint step")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        adapter = staging / "adapter"
        model.save_pretrained(adapter, safe_serialization=True)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict() if scheduler is not None else {}, staging / "scheduler.pt")
        torch.save(capture_rng_state(), staging / "rng_state.pt")
        training_state = {
            "step": step,
            "replay_cursor": step,
            "gradient_receipts": list(receipts),
            "evaluations": list(evaluations),
        }
        atomic_json(staging / "training_state.json", training_state)
        ledger_record = flop_ledger.record(require_complete=(step == 100))
        ledger_record = {
            **ledger_record,
            "charged_generated_tokens": corpus["charged_generated_tokens"],
        }
        atomic_json(staging / "token_flop_ledger.json", ledger_record)
        atomic_json(staging / "source_manifest.json", dict(sorted(source_hashes.items())))
        evaluation_steps = tuple(int(item["step"]) for item in evaluations)
        if set(evaluation_steps) != set(evaluation_files):
            raise CheckpointContractError("evaluation summaries and files do not have the same steps")
        for evaluation_step, source in evaluation_files.items():
            if not source.is_file():
                raise CheckpointContractError(
                    f"evaluation evidence source is missing for step {evaluation_step}"
                )
            target = staging / "evaluations" / f"step-{evaluation_step:03d}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        files = _checkpoint_files(staging, evaluation_steps=evaluation_steps)
        unit = plan["unit"]
        manifest = with_fingerprint(
            {
                "schema_version": "flagship-pilot-checkpoint-v1",
                "step": step,
                "unit_fingerprint": plan["fingerprint"],
                "protocol_sha256": plan["protocol"]["sha256"],
                "corpus_fingerprint": corpus["fingerprint"],
                "condition": unit["condition"],
                "regime": unit["regime"],
                "seed": unit["seed"],
                "replay_cursor": step,
                "gradient_receipt_count": len(receipts),
                "evaluation_steps": list(evaluation_steps),
                "token_flop_ledger": ledger_record,
                "files": files,
            }
        )
        validate_checkpoint_manifest(manifest, plan=plan, corpus=corpus)
        atomic_json(staging / "checkpoint_manifest.json", manifest)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


AdapterLoader = Callable[[Any, Path], None]


def load_checkpoint_bundle(
    *,
    root: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    plan: Mapping[str, Any],
    corpus: Mapping[str, Any],
    adapter_loader: AdapterLoader | None = None,
) -> tuple[dict[str, Any], TrainingFlopLedger, dict[str, Any]]:
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise CheckpointContractError("checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_checkpoint_manifest(manifest, plan=plan, corpus=corpus)
    for relative, digest in manifest["files"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise CheckpointContractError(f"checkpoint file hash mismatch: {relative}")

    if adapter_loader is None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = load_file(root / "adapter/adapter_model.safetensors")
        result = set_peft_model_state_dict(model, state)
        missing_adapter_keys = tuple(
            key
            for key in getattr(result, "missing_keys", ())
            if "lora_" in key or "modules_to_save" in key
        )
        unexpected_keys = tuple(getattr(result, "unexpected_keys", ()))
        if missing_adapter_keys or unexpected_keys:
            raise CheckpointContractError(
                "adapter restore reported missing or unexpected keys: "
                f"missing_adapter={missing_adapter_keys} "
                f"unexpected={unexpected_keys}"
            )
    else:
        adapter_loader(model, root / "adapter")
    optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler_state = torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    rng_state = torch.load(root / "rng_state.pt", map_location="cpu", weights_only=False)
    restore_rng_state(rng_state)
    training_state = json.loads((root / "training_state.json").read_text(encoding="utf-8"))
    if training_state.get("step") != manifest["step"]:
        raise CheckpointContractError("training state step does not match checkpoint manifest")
    ledger_record = json.loads((root / "token_flop_ledger.json").read_text(encoding="utf-8"))
    ledger = TrainingFlopLedger.from_record(ledger_record)
    return training_state, ledger, manifest
