#!/usr/bin/env python3
"""Independently verify every frozen E1 unit against local, W&B, and HF state."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aggregate_audit
import run_colab_e1_confirmatory as shared


DEFAULT_INPUT = HERE / "results" / "full"
DEFAULT_OUTPUT = HERE / "results" / "campaign-verification.json"
AUDIT_FIELDS = (
    "arm",
    "seed",
    "heldout_n",
    "heldout_score",
    "last10_reward",
    "mean_zvf",
    "mean_gu",
    "collapse",
    "rollouts",
    "wall_clock_seconds",
    "stack_fingerprint",
    "treatment_changes",
)
CHECKPOINT_LEAVES = (
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def required_hf_files(checkpoint_steps: list[int]) -> set[str]:
    required = {
        "run_manifest.json",
        "final/adapter_config.json",
        "final/adapter_model.safetensors",
    }
    required.update(
        f"checkpoints/checkpoint-{step}/{leaf}"
        for step in checkpoint_steps
        for leaf in CHECKPOINT_LEAVES
    )
    return required


def validate_manifest_pair(
    record: dict[str, Any],
    manifest: dict[str, Any],
    *,
    checkpoint_steps: list[int],
) -> list[str]:
    errors: list[str] = []
    if record.get("evidence_class") != "confirmatory":
        errors.append("local evidence_class is not confirmatory")
    for field in ("fingerprint", "stack_fingerprint"):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"local {field} is not a SHA-256 value")
    audit_record = manifest.get("audit_record")
    if not isinstance(audit_record, dict):
        return ["manifest audit_record is missing"]
    for field in AUDIT_FIELDS:
        if audit_record.get(field) != record.get(field):
            errors.append(f"manifest audit_record disagrees on {field}")
    run_config = manifest.get("run_config")
    if not isinstance(run_config, dict):
        errors.append("manifest run_config is missing")
    else:
        if run_config.get("unit_fingerprint") != record.get("fingerprint"):
            errors.append("manifest unit fingerprint disagrees with local record")
        if run_config.get("stack_fingerprint") != record.get("stack_fingerprint"):
            errors.append("manifest stack fingerprint disagrees with local record")
    if manifest.get("remote_checkpoint_steps") != checkpoint_steps:
        errors.append("manifest checkpoint cadence disagrees with preregistration")
    trace = manifest.get("heldout_trace")
    heldout_n = record.get("heldout_n")
    if not isinstance(trace, list) or len(trace) != heldout_n:
        errors.append("manifest held-out trace length disagrees with local record")
    else:
        correct = sum(row.get("correct") is True for row in trace if isinstance(row, dict))
        score = record.get("heldout_score")
        if not isinstance(score, (int, float)) or not math.isclose(
            correct / heldout_n, score, rel_tol=0.0, abs_tol=1e-12
        ):
            errors.append("manifest held-out trace disagrees with local score")
    wandb_run_id = (manifest.get("wandb") or {}).get("run_id")
    if wandb_run_id != (record.get("remote") or {}).get("wandb_run_id"):
        errors.append("manifest W&B run ID disagrees with local provenance")
    return errors


def verify_unit(
    path: Path,
    record: dict[str, Any],
    *,
    checkpoint_steps: list[int],
    max_steps: int,
    credentials: dict[str, str],
) -> dict[str, Any]:
    arm = str(record["arm"])
    seed = int(record["seed"])
    unit = f"{arm}/seed-{seed}"
    remote = record.get("remote")
    if not isinstance(remote, dict):
        raise RuntimeError(f"{unit}: local record lacks remote provenance")
    repo_id = remote.get("hf_repo")
    commit = remote.get("hf_commit")
    if not isinstance(repo_id, str) or not isinstance(commit, str):
        raise RuntimeError(f"{unit}: invalid HF repository or commit")

    api = HfApi(token=credentials["HF_TOKEN"])
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=commit)
    if info.private is not True:
        raise RuntimeError(f"{unit}: HF repository is not private")
    if info.sha != commit:
        raise RuntimeError(f"{unit}: HF revision resolved to {info.sha}, expected {commit}")

    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model", revision=commit))
    missing = sorted(required_hf_files(checkpoint_steps) - files)
    if missing:
        raise RuntimeError(f"{unit}: HF commit lacks required files: {missing}")

    trainer_states: dict[str, dict[str, int]] = {}
    for step in checkpoint_steps:
        filename = f"checkpoints/checkpoint-{step}/trainer_state.json"
        state_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=filename,
            revision=commit,
            token=credentials["HF_TOKEN"],
        )
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        global_step = state.get("global_step")
        observed_max_steps = state.get("max_steps")
        if global_step != step or observed_max_steps != max_steps:
            raise RuntimeError(
                f"{unit}: {filename} has global_step={global_step}, "
                f"max_steps={observed_max_steps}"
            )
        trainer_states[str(step)] = {
            "global_step": global_step,
            "max_steps": observed_max_steps,
        }

    manifest_ref = record.get("manifest_path")
    if not isinstance(manifest_ref, str):
        raise RuntimeError(f"{unit}: local manifest_path is invalid")
    local_manifest_path = Path(manifest_ref)
    if not local_manifest_path.is_absolute():
        local_manifest_path = path.parent / local_manifest_path
    local_bytes = local_manifest_path.read_bytes()
    remote_manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename="run_manifest.json",
            revision=commit,
            token=credentials["HF_TOKEN"],
        )
    )
    remote_bytes = remote_manifest_path.read_bytes()
    if local_bytes != remote_bytes:
        raise RuntimeError(f"{unit}: local and exact-commit manifests differ")
    manifest = json.loads(local_bytes)
    shared.validate_remote_manifest(
        manifest,
        {"audit_record": manifest.get("audit_record"), "remote": remote},
        record["fingerprint"],
        checkpoint_steps,
        "confirmatory",
    )
    manifest_errors = validate_manifest_pair(
        record, manifest, checkpoint_steps=checkpoint_steps
    )
    if manifest_errors:
        raise RuntimeError(f"{unit}: " + "; ".join(manifest_errors))

    wandb_url = remote.get("wandb_run_url")
    if not isinstance(wandb_url, str):
        raise RuntimeError(f"{unit}: W&B URL is missing")
    wandb = shared.verify_wandb_run(credentials["WANDB_API_KEY"], wandb_url)
    if wandb.get("run_id") != remote.get("wandb_run_id"):
        raise RuntimeError(f"{unit}: W&B run ID disagrees with local provenance")
    if wandb.get("state") != "finished":
        raise RuntimeError(f"{unit}: W&B state is {wandb.get('state')!r}")

    trace = manifest["heldout_trace"]
    hashes = [row["completion_sha256"] for row in trace]
    return {
        "unit": unit,
        "record_path": str(path),
        "manifest_path": str(local_manifest_path),
        "manifest_sha256": sha256_bytes(local_bytes),
        "heldout_rows": len(trace),
        "unique_completion_hashes": len(set(hashes)),
        "heldout_correct": sum(row.get("correct") is True for row in trace),
        "heldout_score": record["heldout_score"],
        "stack_fingerprint": record["stack_fingerprint"],
        "unit_fingerprint": record["fingerprint"],
        "treatment_changes": record["treatment_changes"],
        "hf_repo": repo_id,
        "hf_commit": commit,
        "hf_private": True,
        "trainer_states": trainer_states,
        "wandb": wandb,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=HERE / "preregistration.json")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise SystemExit("workers must be positive")

    prereg = aggregate_audit.load_json(args.prereg)
    paths = sorted(args.input_dir.glob("*.json"))
    records = [(path, aggregate_audit.load_json(path)) for path in paths]
    indexed, local_errors = aggregate_audit.validate_records(prereg, records)
    missing = aggregate_audit.missing_units(prereg, indexed)
    required_units = sum(
        1
        for _arm in prereg["core_stratum"]["arms"]
        for _seed in prereg["core_stratum"]["seeds"]
    )
    if local_errors or missing or len(indexed) != required_units:
        report = {
            "status": "INCOMPLETE",
            "required_units": required_units,
            "locally_validated_units": len(indexed),
            "errors": local_errors,
            "missing_units": missing,
            "remotely_verified_units": 0,
        }
        atomic_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    credentials = shared.load_credentials()
    core = prereg["core_stratum"]
    checkpoint_steps = list(range(5, int(core["train_steps"]) + 1, 5))
    results: list[dict[str, Any]] = []
    remote_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                verify_unit,
                path,
                record,
                checkpoint_steps=checkpoint_steps,
                max_steps=int(core["train_steps"]),
                credentials=credentials,
            ): (record["arm"], record["seed"])
            for path, record in records
        }
        for future in as_completed(futures):
            arm, seed = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                remote_errors.append(f"{arm}/seed-{seed}: {exc}")
            else:
                results.append(result)
                print(f"[verified] {result['unit']}", flush=True)

    results.sort(key=lambda item: item["unit"])
    report = {
        "status": "COMPLETE" if not remote_errors and len(results) == required_units else "INCOMPLETE",
        "required_units": required_units,
        "locally_validated_units": len(indexed),
        "remotely_verified_units": len(results),
        "errors": sorted(remote_errors),
        "units": results,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "units"}, indent=2, sort_keys=True))
    return 0 if report["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
