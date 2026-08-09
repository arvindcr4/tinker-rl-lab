#!/usr/bin/env python3
"""Write a fail-closed receipt for an exact one-task BankerToolBench Harbor preflight.

This module deliberately does not launch Harbor, an agent, a verifier, W&B, or
Tinker.  It converts already-observed, pinned native-task metadata into a
machine-readable E4 receipt.  A caller must supply the observed Harbor state;
``READY`` is still insufficient to authorize an experiment until tracking,
Tinker, immutable HF weights, verifier credentials, and the USD cap are all
recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "e4-banker-toolbench-harbor-rerun-v1"
SUITE_ID = "banker_toolbench_eval"
DATASET_ID = "handshake-ai-research/bankertoolbench"
DATASET_REVISION = "2c63d3b6d429687d2c29550e820dd136baafee95"
OFFICIAL_REPO_REVISION = "ff6db552a44632643df20393065056e5f1f0092c"
PINNED_TASK_ID = "707cba99-59a7-47bd-bc4d-7f36212e99f3"
EXPECTED_TASK_COUNT = 100
HARBOR_EXECUTABLE = "/Users/arvind/.local/bin/harbor"
REQUIRED_CREDENTIALS = ("WANDB_API_KEY", "TINKER_API_KEY", "GEMINI_API_KEY")


class HarborRerunReceiptError(ValueError):
    """Raised for an unpinned or malformed BankerToolBench preflight input."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _task_id_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def _read_jsonl_tasks(path: Path) -> list[Mapping[str, Any]]:
    try:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise HarborRerunReceiptError(f"cannot read public task manifest {path}: {exc}") from exc
    if not records or any(not isinstance(record, Mapping) for record in records):
        raise HarborRerunReceiptError("public task manifest must contain JSON object records")
    return records


def _credential_presence(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    values = os.environ if environment is None else environment
    return {name: "PRESENT" if bool(values.get(name)) else "ABSENT" for name in REQUIRED_CREDENTIALS}


def _read_native_contract(task_dir: Path) -> dict[str, Any]:
    task_toml = task_dir / "task.toml"
    grader_toml = task_dir / "tests" / "grader.toml"
    compose = task_dir / "environment" / "docker-compose.yaml"
    if not all(path.is_file() for path in (task_toml, grader_toml, compose)):
        raise HarborRerunReceiptError("generated native task is missing task.toml, grader.toml, or compose file")
    task = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    grader = tomllib.loads(grader_toml.read_text(encoding="utf-8"))
    try:
        return {
            "task_dir": str(task_dir),
            "task_toml_sha256": _sha256_file(task_toml),
            "grader_toml_sha256": _sha256_file(grader_toml),
            "compose_sha256": _sha256_file(compose),
            "environment": task["environment"],
            "verifier": {
                "model": grader["model"],
                "mode": grader["mode"],
                "rubric_path": grader["rubric_path"],
                "mcp_servers": grader["mcp_servers"],
                "key_reference": task["verifier"]["env"]["LLM_API_KEY"],
            },
        }
    except (KeyError, TypeError) as exc:
        raise HarborRerunReceiptError(f"generated native task lacks exact Harbor contract fields: {exc}") from exc


def build_harbor_rerun_receipt(
    *,
    manifest_path: Path,
    task_dir: Path,
    harbor_version: str,
    native_start_state: str,
    credential_presence: Mapping[str, str],
    native_start_command: str,
    operational_event: str,
) -> dict[str, Any]:
    """Build the E4 receipt from exact local artifacts and observed state only."""

    records = _read_jsonl_tasks(manifest_path)
    task_ids = [record.get("task_id") for record in records]
    if len(records) != EXPECTED_TASK_COUNT or any(not isinstance(value, str) for value in task_ids):
        raise HarborRerunReceiptError("public manifest must contain exactly 100 deterministic task IDs")
    if len(set(task_ids)) != EXPECTED_TASK_COUNT or PINNED_TASK_ID not in task_ids:
        raise HarborRerunReceiptError("public manifest does not contain the exact selected E4 task")
    if harbor_version != "0.20.0":
        raise HarborRerunReceiptError("receipt requires observed Harbor 0.20.0")
    native = _read_native_contract(task_dir)

    missing = [name for name in REQUIRED_CREDENTIALS if credential_presence.get(name) != "PRESENT"]
    blockers = []
    if native_start_state != "READY":
        blockers.append(f"native Harbor environment is not ready: {native_start_state}")
    if "WANDB_API_KEY" in missing:
        blockers.append("online W&B project codex cannot be initialized: WANDB_API_KEY is absent")
    if "TINKER_API_KEY" in missing:
        blockers.append("Tinker sampler/evaluation access is absent: TINKER_API_KEY is absent")
    if "GEMINI_API_KEY" in missing:
        blockers.append("native Gandalf verifier cannot run: GEMINI_API_KEY is absent")
    blockers.append("no immutable Hugging Face checkpoint receipt exists for a weight-changing run")
    blockers.append("no projected paid lane cost is recorded at or below USD 1.00")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "terminal_status": "BLOCKED",
        "suite_id": SUITE_ID,
        "evidence_scope": "exact native Harbor preflight only; no task result or benchmark score is asserted",
        "source": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "official_repo_revision": OFFICIAL_REPO_REVISION,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
            "task_count": len(records),
            "unique_task_count": len(set(task_ids)),
            "selected_task_id": PINNED_TASK_ID,
            "selected_task_id_sha256": _task_id_hash(PINNED_TASK_ID),
            "selected_task_count": 1,
        },
        "harbor": {
            "executable": HARBOR_EXECUTABLE,
            "version": harbor_version,
            "plugins_discovery": "harbor plugins list: no installed Harbor plugins",
            "auth_discovery": "harbor auth status: not authenticated; legacy credentials require login",
            "commands": {
                "version": f"{HARBOR_EXECUTABLE} --version",
                "plugins": f"{HARBOR_EXECUTABLE} plugins list",
                "auth": f"{HARBOR_EXECUTABLE} auth status",
                "task_generation": (
                    "BTB_HF_REVISION="
                    f"{DATASET_REVISION} uv run python -m adapters.btb.run_adapter "
                    f"--data-dir {manifest_path.parent} --output-dir {task_dir.parent} "
                    f"--task-ids {PINNED_TASK_ID}"
                ),
                "native_start": native_start_command,
            },
            "native_start_command": native_start_command,
            "native_start_state": native_start_state,
            "operational_event": operational_event,
        },
        "native_environment": native,
        "gates": {
            "credentials": dict(credential_presence),
            "wandb": {"project": "codex", "state": "NOT_INITIALIZED_NO_EXPERIMENT"},
            "tinker": {"state": "NOT_CALLED", "paid_attempts": 0, "cost_usd": 0.0},
            "hf": {"state": "PINNED_PUBLIC_DATA_LOCAL_NO_IMMUTABLE_MODEL_CHECKPOINT"},
            "authorization": {"launch_authorized": False, "maximum_usd": 1.0},
        },
        "execution": {
            "executed_task_count": 0,
            "sample_count": 0,
            "score": None,
            "metrics": None,
            "result_claims": None,
            "paid_calls": 0,
        },
        "blockers": blockers,
    }


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--harbor-version", required=True)
    parser.add_argument("--native-start-state", required=True)
    parser.add_argument("--native-start-command", required=True)
    parser.add_argument("--operational-event", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_harbor_rerun_receipt(
            manifest_path=args.manifest,
            task_dir=args.task_dir,
            harbor_version=args.harbor_version,
            native_start_state=args.native_start_state,
            credential_presence=_credential_presence(),
            native_start_command=args.native_start_command,
            operational_event=args.operational_event,
        )
        _write_json(receipt, args.out)
    except HarborRerunReceiptError as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
