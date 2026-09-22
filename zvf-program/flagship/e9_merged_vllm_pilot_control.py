"""Single-flight launch and collection controls for the E9 merged-vLLM pilot.

This module intentionally keeps the new arm separate from legacy Tinker E9
coverage.  It reserves GPU plus CPU cost before a remote spawn, persists the
recoverable Modal FunctionCall ID, validates the returned receipt and payloads,
and reconciles only the observed estimate into the campaign spend ledger.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from e9_merged_vllm_arm import (
    ARM_ID,
    ARM_SCHEMA_V1,
    ARM_SCHEMA_V2,
    ARM_SCHEMA_V3,
    CPU_PREPARATION_FAILURE_CLAIM_BOUNDARY,
    CPU_PREPARATION_FAILURE_PHASE,
    CPU_PREPARATION_FAILURE_SCHEMA_V1,
    EXPECTED_ADAPTER_COMMIT,
    EXPECTED_BASE_COMMIT,
    EXPECTED_MERGE_RECEIPT_SHA256,
    EXPECTED_SHARD_COUNT,
    EXPECTED_WEIGHT_BYTES,
    _sanitize_failure_message,
    reserve_pilot_budget,
    validate_allocation_attestation_v2,
    validate_gpu_allocation_gate,
    validate_v3_generation_binding,
)


APP_NAME = "pavlov-e9-mle-bench-streaming"
DEFAULT_COMPETITION_ID = "hotel-id-2021-fgvc8"
SUPPORTED_COMPETITIONS = {
    "alaska2-image-steganalysis",
    "h-and-m-personalized-fashion-recommendations",
    "hotel-id-2021-fgvc8",
    "hubmap-kidney-segmentation",
    "imet-2020-fgvc7",
    "predict-volcanic-eruptions-ingv-oe",
    "vesuvius-challenge-ink-detection",
}
PROMPT_TEMPLATE_VERSION = "e9_hard_bounded_collections_v7"
OUTPUT_DIR = Path("outputs/e9_mle_bench/merged_vllm")
SPEND_LEDGER = Path("outputs/e1_e14_incremental_spend_ledger_2026-08-29.json")
DRIVER_STATE = Path(".codex-run/e9_pipeline_driver_state.json")
LEGACY_OUTPUT_DIR = Path("outputs/e9_mle_bench/modal_streaming")
GPU_RESERVATION_USD = round(1800 * 0.000694, 9)
CPU_RESERVATION_USD = round(7200 * 0.00007016, 9)
LAUNCH_SCHEMA_V1 = "e9-merged-vllm-spawn-v1"
LAUNCH_SCHEMA_V2 = "e9-merged-vllm-spawn-v2"
LAUNCH_SCHEMA_V3 = "e9-merged-vllm-spawn-v3"
LAUNCH_SCHEMA = LAUNCH_SCHEMA_V3
PAID_LAUNCH_ENABLED = False
COLLECTION_SCHEMA = "e9-merged-vllm-collect-v1"
RESERVATION_SCHEMA = "e9-merged-vllm-spend-reservations-v1"
SPAWN_INTENT_STATUS = "REMOTE_SPAWN_INTENT_DURABLY_PERSISTED"
SPAWN_CALL_ID_STATUS = "REMOTE_CALL_ID_DURABLY_RECORDED"
SPAWN_ORPHAN_STATUS = "REMOTE_SPAWN_ORPHAN_REQUIRES_PROVIDER_RECONCILIATION"
SPAWN_OUTCOME_UNKNOWN_STATUS = "SPAWN_OUTCOME_UNKNOWN"
SPAWN_CANONICAL_STATUS = "REMOTE_CALL_CANONICAL_LAUNCH_PERSISTED"


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _rehash_launch(launch: dict[str, Any]) -> dict[str, Any]:
    """Return a launch receipt with exactly one current self-hash."""

    launch.pop("launch_sha256", None)
    launch["launch_sha256"] = _canonical_sha256(launch)
    return launch


def _sanitized_spawn_error(exc: Exception) -> dict[str, Any]:
    """Keep actionable failure evidence without persisting secrets or paths."""

    raw = str(exc) or repr(exc)
    tail, redacted = _sanitize_failure_message(raw[-3000:])
    return {
        "spawn_error_type": type(exc).__name__,
        "spawn_error_message_tail": tail,
        "spawn_error_message_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "spawn_error_redaction_applied": redacted,
    }


def _safe_pending_launch_path(
    *, pending_dir: Path, competition_id: str, reservation_id: str, call_id: str | None = None
) -> Path:
    """Make a basename-only pending receipt path from validated local identifiers."""

    if competition_id not in SUPPORTED_COMPETITIONS:
        raise ValueError("pending launch competition is outside the prospective pilot set")
    if re.fullmatch(r"[0-9a-f]{32}", reservation_id) is None:
        raise ValueError("pending launch reservation id is invalid")
    suffix = reservation_id if call_id is None else call_id
    if call_id is not None and (
        not call_id.startswith("fc-")
        or Path(call_id).name != call_id
        or "/" in call_id
        or "\\" in call_id
    ):
        raise ValueError("pending launch Modal FunctionCall ID is invalid")
    return pending_dir / f"{competition_id}-{suffix}.json"


def _validate_v3_success_evidence(receipt: dict[str, Any], launch: dict[str, Any]) -> None:
    """Authenticate a v3 success before generic artifact validation runs."""

    required_launch = {
        "launch_nonce",
        "spend_reservation_id",
        "prompt_builder_sha256",
        "deployment_revision",
        "source_bundle_sha256",
        "allocation_attestation_public_key_base64",
        "allocation_gate",
        "generation_backend",
    }
    if launch.get("arm_id") != ARM_ID or any(field not in launch for field in required_launch):
        raise ValueError("v3 launch provenance is incomplete")
    prompt = launch.get("v3_prompt") or receipt.get("generation_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("v3 launch prompt is invalid")
    if launch.get("prompt_template_version") != PROMPT_TEMPLATE_VERSION:
        raise ValueError("v3 launch does not use the mandatory v7 prompt")
    if launch.get("generation_backend") != "merged_vllm":
        raise ValueError("v3 launch generation backend is invalid")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if "prompt_sha256" in launch and launch.get("prompt_sha256") != prompt_sha256:
        raise ValueError("v3 launch prompt self-binding is invalid")
    try:
        allocation_gate = validate_gpu_allocation_gate(launch["allocation_gate"])
        launch_attestation = launch.get("allocation_attestation")
        attestation_candidate = launch_attestation or receipt.get("allocation_attestation")
        if launch_attestation and (
            launch.get("allocation_attestation_sha256")
            != launch_attestation.get("attestation_sha256")
            or receipt.get("allocation_attestation") != launch_attestation
        ):
            raise ValueError("v3 allocation attestation does not match the launch")
        attestation = validate_allocation_attestation_v2(
            attestation_candidate,
            public_key_base64=launch["allocation_attestation_public_key_base64"],
            expected_competition_id=str(launch["competition_id"]),
            expected_reservation_id=str(launch["spend_reservation_id"]),
            expected_launch_nonce=str(launch["launch_nonce"]),
            expected_prompt_sha256=prompt_sha256,
            expected_prompt_builder_sha256=launch["prompt_builder_sha256"],
            expected_deployment_revision=launch["deployment_revision"],
            expected_source_bundle_sha256=launch["source_bundle_sha256"],
            generation_started_at=(
                receipt.get("generation", {}).get("generation_started_at")
                if isinstance(receipt.get("generation"), dict)
                else None
            ),
        )
        validate_v3_generation_binding(
            receipt.get("generation"),
            competition_id=str(launch["competition_id"]),
            prompt=prompt,
            allocation_attestation=attestation,
        )
    except Exception as exc:
        raise ValueError("v3 authenticated provenance validation failed") from exc
    if receipt.get("allocation_gate") != allocation_gate:
        raise ValueError("v3 receipt allocation gate does not match the launch")
    payload = attestation.get("payload")
    if not isinstance(payload, dict) or payload.get("reservation") != launch.get(
        "combined_gpu_cpu_reservation"
    ):
        raise ValueError("v3 attestation reservation does not match the launch")
    checkpoint = receipt.get("checkpoint_verification")
    attested_checkpoint = payload.get("checkpoint_verification") if isinstance(payload, dict) else None
    if (
        not isinstance(checkpoint, dict)
        or checkpoint != allocation_gate.get("checkpoint_verification")
        or checkpoint != attested_checkpoint
        or checkpoint.get("verified_shard_count") != EXPECTED_SHARD_COUNT
        or checkpoint.get("merge_receipt_sha256") != EXPECTED_MERGE_RECEIPT_SHA256
        or checkpoint.get("base_commit") != EXPECTED_BASE_COMMIT
        or checkpoint.get("adapter_commit") != EXPECTED_ADAPTER_COMMIT
        or checkpoint.get("verified_weight_bytes") != EXPECTED_WEIGHT_BYTES
    ):
        raise ValueError("v3 checkpoint evidence is not the pinned merged checkpoint")
    if receipt.get("reservation_id") != launch.get("spend_reservation_id") or receipt.get(
        "launch_nonce"
    ) != launch.get("launch_nonce"):
        raise ValueError("v3 receipt reservation or nonce binding is invalid")
    generation = receipt.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("v3 generation evidence is absent")
    load_receipt = generation.get("load_receipt")
    if not isinstance(load_receipt, dict) or load_receipt.get("checkpoint_verification") != checkpoint:
        raise ValueError("v3 GPU load evidence does not match the checkpoint")
    for label, evidence, paid_started_at in (
        ("load", load_receipt.get("wandb_receipt"), load_receipt.get("paid_load_started_at")),
        ("generation", generation.get("wandb_receipt"), generation.get("generation_started_at")),
    ):
        if not isinstance(evidence, dict) or not isinstance(paid_started_at, str):
            raise ValueError(f"v3 {label} W&B evidence is absent")
        initialized_at = evidence.get("initialized_at")
        if (
            evidence.get("mode") != "online"
            or evidence.get("server_confirmed") is not True
            or not isinstance(evidence.get("run_id"), str)
            or not evidence["run_id"]
            or not isinstance(evidence.get("url"), str)
            or not evidence["url"].startswith("https://wandb.ai/")
            or not isinstance(initialized_at, str)
        ):
            raise ValueError(f"v3 {label} W&B evidence is invalid")
        try:
            initialized = datetime.fromisoformat(initialized_at.replace("Z", "+00:00"))
            started = datetime.fromisoformat(paid_started_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"v3 {label} W&B timestamps are invalid") from exc
        if initialized.tzinfo is None or started.tzinfo is None or initialized >= started:
            raise ValueError(f"v3 {label} W&B does not precede paid work")


def _reservations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": RESERVATION_SCHEMA, "reservations": []}
    value = _load_json(path)
    if value.get("schema_version") != RESERVATION_SCHEMA or not isinstance(
        value.get("reservations"), list
    ):
        raise ValueError("unsupported or malformed merged-vLLM reservation ledger")
    return value


def _active_provider_tasks() -> int:
    """Read the Modal control plane and fail closed if it cannot be audited."""

    modal_cli = shutil.which("modal")
    if not modal_cli:
        raise RuntimeError("Modal CLI is unavailable for the single-flight provider gate")
    completed = subprocess.run(
        [modal_cli, "app", "list", "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("Modal app inventory failed before pilot launch")
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("Modal app inventory is malformed")
    active = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        description = row.get("Description", row.get("description"))
        state = str(row.get("State", row.get("state", ""))).lower()
        if description == APP_NAME and state not in {"stopped", "completed"}:
            active += int(row.get("Tasks", row.get("tasks", 0)))
    return active


def _validate_campaign_gate(
    *,
    competition_id: str,
    output_dir: Path = OUTPUT_DIR,
    spend_ledger_path: Path = SPEND_LEDGER,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed on stale campaign state, duplicate work, or insufficient budget."""

    if competition_id not in SUPPORTED_COMPETITIONS:
        raise RuntimeError("competition is outside the prospective merged-vLLM pilot set")
    provider_tasks = _active_provider_tasks()
    if provider_tasks:
        raise RuntimeError(f"{provider_tasks} active merged-vLLM provider task(s) remain")
    state = _load_json(DRIVER_STATE)
    remaining = state.get("remaining_unattempted_competitions")
    if not isinstance(remaining, list) or competition_id not in remaining:
        raise RuntimeError(f"{competition_id} is no longer an unattempted E9 competition")
    if state.get("terminal_native_competitions", 0) < 0:
        raise RuntimeError("E9 driver state is malformed")

    terminal_receipts = []
    for receipt_path in Path("outputs/e9_mle_bench").glob("**/receipt.json"):
        try:
            receipt = _load_json(receipt_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if receipt.get("competition_id") == competition_id:
            terminal_receipts.append(str(receipt_path))
    if terminal_receipts:
        raise RuntimeError(
            f"{competition_id} already has terminal receipt evidence: "
            + ", ".join(terminal_receipts)
        )

    pending_dir = output_dir / "_pending"
    unresolved = []
    if pending_dir.is_dir():
        unresolved_statuses = {
            SPAWN_INTENT_STATUS,
            SPAWN_CALL_ID_STATUS,
            SPAWN_ORPHAN_STATUS,
            SPAWN_OUTCOME_UNKNOWN_STATUS,
            "REMOTE_CALL_SPAWNED_UNCOLLECTED",
        }
        for launch_path in pending_dir.glob("*.json"):
            if launch_path.name == "spend_reservations.json":
                continue
            launch = _load_json(launch_path)
            if (
                launch.get("schema_version")
                in {LAUNCH_SCHEMA_V1, LAUNCH_SCHEMA_V2, LAUNCH_SCHEMA_V3}
                and launch.get("status") in unresolved_statuses
            ):
                unresolved.append(str(launch_path))
    if unresolved:
        raise RuntimeError("merged-vLLM call already unresolved: " + ", ".join(unresolved))

    reservation = reserve_pilot_budget(gpu_usd=GPU_RESERVATION_USD, cpu_usd=CPU_RESERVATION_USD)
    ledger = _load_json(spend_ledger_path)
    remaining_usd = ledger.get("remaining_authorized_incremental_spend_usd")
    if ledger.get("within_cap") is not True or not isinstance(remaining_usd, (int, float)):
        raise RuntimeError("campaign spend ledger is not launchable")
    reservations_path = pending_dir / "spend_reservations.json"
    reservations = _reservations(reservations_path)
    existing_reserved = round(
        sum(
            float(item.get("reserved_usd", 0.0))
            for item in reservations["reservations"]
            if item.get("reconciled_to_spend_ledger") is not True
        ),
        9,
    )
    projected = float(reservation["projected_combined_usd"])
    if float(remaining_usd) - existing_reserved < projected:
        raise RuntimeError("campaign ledger cannot cover the merged-vLLM pilot reservation")
    gate = {
        "driver_state": str(DRIVER_STATE),
        "driver_state_sha256": _file_sha256(DRIVER_STATE),
        "spend_ledger": str(spend_ledger_path),
        "spend_ledger_sha256": _file_sha256(spend_ledger_path),
        "remaining_authorized_incremental_spend_usd": remaining_usd,
        "existing_unreconciled_reservations_usd": existing_reserved,
        "projected_combined_usd": projected,
        "terminal_receipts_before_spawn": 0,
        "unresolved_launches_before_spawn": 0,
        "active_provider_tasks_before_spawn": provider_tasks,
    }
    return gate, reservation


def spawn_pilot(
    *,
    competition_id: str,
    output_dir: Path = OUTPUT_DIR,
    spend_ledger_path: Path = SPEND_LEDGER,
) -> Path:
    """Reserve and spawn exactly one recoverable prospective pilot."""

    if not PAID_LAUNCH_ENABLED:
        raise RuntimeError(
            "merged-vLLM paid launches are frozen until authenticated v3 provenance, "
            "strict v7 binding, and crash recovery pass adversarial review"
        )

    import modal

    pending_dir = output_dir / "_pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    lock_path = pending_dir / ".spawn.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        gate, arm_reservation = _validate_campaign_gate(
            competition_id=competition_id,
            output_dir=output_dir,
            spend_ledger_path=spend_ledger_path,
        )
        reservations_path = pending_dir / "spend_reservations.json"
        reservations = _reservations(reservations_path)
        reservation_id = uuid.uuid4().hex
        launch_nonce = uuid.uuid4().hex + uuid.uuid4().hex
        v3_contract = {
            "public_key_base64": os.environ.get("E9_ALLOCATION_ATTESTATION_PUBLIC_KEY_BASE64", ""),
            "prompt_builder_sha256": os.environ.get("E9_V3_PROMPT_BUILDER_SHA256", ""),
            "deployment_revision": os.environ.get("E9_V3_DEPLOYMENT_REVISION", ""),
            "source_bundle_sha256": os.environ.get("E9_V3_SOURCE_BUNDLE_SHA256", ""),
        }
        if any(not isinstance(value, str) or not value for value in v3_contract.values()):
            raise RuntimeError("authenticated v3 producer contract is not configured")
        reservation = {
            "reservation_id": reservation_id,
            "competition_id": competition_id,
            "created_at_epoch": time.time(),
            "status": SPAWN_INTENT_STATUS,
            "reserved_usd": arm_reservation["projected_combined_usd"],
            "arm_reservation": arm_reservation,
            "launch_nonce": launch_nonce,
            "reconciled_to_spend_ledger": False,
        }
        reservations["reservations"].append(reservation)
        _write_json_atomic(reservations_path, reservations)

        intent_path = _safe_pending_launch_path(
            pending_dir=pending_dir,
            competition_id=competition_id,
            reservation_id=reservation_id,
        )
        launch = {
            "schema_version": LAUNCH_SCHEMA,
            "status": SPAWN_INTENT_STATUS,
            "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
            "score": None,
            "is_full_suite_score": False,
            "legacy_tinker_coverage_increment": 0,
            "competition_id": competition_id,
            "arm_id": ARM_ID,
            "created_at_epoch": time.time(),
            "output_dir": str(output_dir),
            "spend_ledger": str(spend_ledger_path),
            "spend_reservation_id": reservation_id,
            "launch_nonce": launch_nonce,
            "combined_gpu_cpu_reservation": arm_reservation,
            "allocation_gate": None,
            "generation_backend": "merged_vllm",
            "allocation_attestation_public_key_base64": v3_contract["public_key_base64"],
            "prompt_builder_sha256": v3_contract["prompt_builder_sha256"],
            "deployment_revision": v3_contract["deployment_revision"],
            "source_bundle_sha256": v3_contract["source_bundle_sha256"],
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
            "campaign_gate": gate,
            "claim_boundary": (
                "A spawned call is execution-control evidence only. Collection and immutable "
                "native receipt validation are required; this arm never changes legacy E9 coverage."
            ),
        }
        _rehash_launch(launch)
        _write_json_atomic(intent_path, launch)
        reservation.update(
            {
                "launch_intent_receipt": str(intent_path),
                "launch_intent_sha256": _file_sha256(intent_path),
            }
        )
        _write_json_atomic(reservations_path, reservations)
        try:
            preflight = modal.Function.from_name(APP_NAME, "preflight_merged_vllm_allocation")
            allocation_gate = preflight.remote()
            validate_gpu_allocation_gate(allocation_gate)
            # Record the exact preflight evidence before dispatch.  Recovery must
            # never reconstruct a launched v2 receipt from a mutable provider gate.
            launch["allocation_gate"] = allocation_gate
            _rehash_launch(launch)
            _write_json_atomic(intent_path, launch)
            reservation["launch_intent_sha256"] = _file_sha256(intent_path)
            _write_json_atomic(reservations_path, reservations)
        except Exception as exc:
            error = _sanitized_spawn_error(exc)
            reservation.update(
                {
                    "status": "SPAWN_FAILED_RELEASED",
                    "reserved_usd": 0.0,
                    **error,
                }
            )
            _write_json_atomic(reservations_path, reservations)
            launch.update(
                {
                    "status": "SPAWN_FAILED_RELEASED",
                    "allocation_gate": allocation_gate if "allocation_gate" in locals() else None,
                    **error,
                }
            )
            _rehash_launch(launch)
            _write_json_atomic(intent_path, launch)
            raise
        try:
            deployed = modal.Function.from_name(APP_NAME, "run_one")
            call = deployed.spawn(
                competition_id,
                50.0,
                None,
                None,
                "merged_vllm",
                arm_reservation,
                allocation_gate,
                reservation_id,
                launch_nonce,
                v3_contract,
            )
        except Exception as exc:
            # A client-side exception from spawn is not proof the provider did
            # not accept the call.  Retain the reservation and block retries.
            error = _sanitized_spawn_error(exc)
            reservation.update({"status": SPAWN_OUTCOME_UNKNOWN_STATUS, **error})
            _write_json_atomic(reservations_path, reservations)
            launch.update({"status": SPAWN_OUTCOME_UNKNOWN_STATUS, **error})
            _rehash_launch(launch)
            _write_json_atomic(intent_path, launch)
            reservation["launch_intent_sha256"] = _file_sha256(intent_path)
            _write_json_atomic(reservations_path, reservations)
            raise

        call_id = str(call.object_id)
        # This write is deliberately ordered before the canonical launch receipt:
        # recovery can rebuild the latter if the process dies after remote spawn.
        reservation.update(
            {
                "status": SPAWN_CALL_ID_STATUS,
                "function_call_id": call_id,
            }
        )
        _write_json_atomic(reservations_path, reservations)
        launch.update(
            {
                "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                "function_call_id": call_id,
                "allocation_gate": allocation_gate,
            }
        )
        _rehash_launch(launch)
        launch_path = _safe_pending_launch_path(
            pending_dir=pending_dir,
            competition_id=competition_id,
            reservation_id=reservation_id,
            call_id=call_id,
        )
        _write_json_atomic(launch_path, launch)
        reservation.update(
            {
                "status": "REMOTE_CALL_RESERVED_UNRECONCILED",
                "launch_receipt": str(launch_path),
            }
        )
        intent = _load_json(intent_path)
        intent.update(
            {
                "status": SPAWN_CANONICAL_STATUS,
                "function_call_id": call_id,
                "canonical_launch_receipt": str(launch_path),
            }
        )
        _rehash_launch(intent)
        _write_json_atomic(intent_path, intent)
        reservation["launch_intent_sha256"] = _file_sha256(intent_path)
        _write_json_atomic(reservations_path, reservations)
    return launch_path


def recover_orphaned_spawns(*, output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Recover a recorded call ID or fail closed on an acknowledged-unknown spawn.

    A local process cannot atomically commit a Modal dispatch and a filesystem
    write.  Therefore an intent is persisted before dispatch, a returned call ID
    is persisted in the reservation before the canonical launch receipt, and an
    intent with no call ID remains an explicit blocking orphan rather than being
    released or re-spawned.
    """

    pending_dir = output_dir / "_pending"
    reservations_path = pending_dir / "spend_reservations.json"
    if not reservations_path.is_file():
        return []
    recovered: list[Path] = []
    lock_path = pending_dir / ".spawn.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        reservations = _reservations(reservations_path)
        dirty = False
        for reservation in reservations["reservations"]:
            if not isinstance(reservation, dict):
                raise ValueError("merged-vLLM reservation entry is malformed")
            if reservation.get("status") not in {
                SPAWN_INTENT_STATUS,
                SPAWN_CALL_ID_STATUS,
                SPAWN_ORPHAN_STATUS,
                SPAWN_OUTCOME_UNKNOWN_STATUS,
            }:
                continue
            intent_text = reservation.get("launch_intent_receipt")
            if not isinstance(intent_text, str):
                raise ValueError("spawn reservation is missing its durable intent receipt")
            intent_path = Path(intent_text)
            if not intent_path.is_file() or _file_sha256(intent_path) != reservation.get(
                "launch_intent_sha256"
            ):
                raise ValueError("spawn reservation intent receipt is missing or tampered")
            launch = _load_json(intent_path)
            unhashed = dict(launch)
            stored_hash = unhashed.pop("launch_sha256", None)
            if stored_hash != _canonical_sha256(unhashed):
                raise ValueError("spawn intent self-hash is invalid")
            if (
                launch.get("status") not in {
                    SPAWN_INTENT_STATUS,
                    SPAWN_CALL_ID_STATUS,
                    SPAWN_ORPHAN_STATUS,
                    SPAWN_CANONICAL_STATUS,
                }
                or launch.get("competition_id") != reservation.get("competition_id")
                or launch.get("spend_reservation_id") != reservation.get("reservation_id")
                or launch.get("launch_nonce") != reservation.get("launch_nonce")
            ):
                raise ValueError("spawn intent does not match its reservation")
            call_id = reservation.get("function_call_id")
            if not isinstance(call_id, str) or not call_id.startswith("fc-"):
                if reservation.get("status") not in {
                    SPAWN_ORPHAN_STATUS,
                    SPAWN_OUTCOME_UNKNOWN_STATUS,
                }:
                    reservation["status"] = SPAWN_ORPHAN_STATUS
                    reservation["orphaned_at_utc"] = time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    )
                    launch["status"] = SPAWN_ORPHAN_STATUS
                    _rehash_launch(launch)
                    _write_json_atomic(intent_path, launch)
                    reservation["launch_intent_sha256"] = _file_sha256(intent_path)
                    dirty = True
                continue
            launch_path = _safe_pending_launch_path(
                pending_dir=pending_dir,
                competition_id=str(launch["competition_id"]),
                reservation_id=str(reservation["reservation_id"]),
                call_id=call_id,
            )
            launch.update(
                {
                    "status": "REMOTE_CALL_SPAWNED_UNCOLLECTED",
                    "function_call_id": call_id,
                }
            )
            _rehash_launch(launch)
            if launch_path.is_file() and _load_json(launch_path) != launch:
                raise ValueError("recovered launch receipt differs from existing evidence")
            if not launch_path.is_file():
                _write_json_atomic(launch_path, launch)
            reservation.update(
                {
                    "status": "REMOTE_CALL_RESERVED_UNRECONCILED",
                    "launch_receipt": str(launch_path),
                }
            )
            intent = _load_json(intent_path)
            intent.update(
                {
                    "status": SPAWN_CANONICAL_STATUS,
                    "function_call_id": call_id,
                    "canonical_launch_receipt": str(launch_path),
                }
            )
            _rehash_launch(intent)
            _write_json_atomic(intent_path, intent)
            reservation["launch_intent_sha256"] = _file_sha256(intent_path)
            dirty = True
            recovered.append(launch_path)
        if dirty:
            _write_json_atomic(reservations_path, reservations)
    return recovered


def validate_returned_result(result: Any, launch: dict[str, Any]) -> dict[str, Any]:
    """Bind all returned artifacts to the prospective merged-vLLM launch contract."""

    if not isinstance(result, dict) or not isinstance(result.get("receipt"), dict):
        raise ValueError("remote merged-vLLM result has no receipt")
    receipt = result["receipt"]
    launch_schema = launch.get("schema_version")
    if launch_schema not in {LAUNCH_SCHEMA_V1, LAUNCH_SCHEMA_V2, LAUNCH_SCHEMA_V3}:
        raise ValueError("launch schema is not a supported merged-vLLM version")
    expected_arm_schema = {
        LAUNCH_SCHEMA_V1: ARM_SCHEMA_V1,
        LAUNCH_SCHEMA_V2: ARM_SCHEMA_V2,
        LAUNCH_SCHEMA_V3: ARM_SCHEMA_V3,
    }[launch_schema]
    if receipt.get("schema_version") != expected_arm_schema or receipt.get("arm_id") != ARM_ID:
        raise ValueError("remote receipt is not the prospective merged-vLLM arm")
    if launch_schema == LAUNCH_SCHEMA_V2:
        allocation_gate = validate_gpu_allocation_gate(launch.get("allocation_gate"))
        if receipt.get("allocation_gate") != allocation_gate:
            raise ValueError("remote receipt allocation gate does not match the launch")
    elif launch_schema == LAUNCH_SCHEMA_V3:
        _validate_v3_success_evidence(receipt, launch)
    elif "allocation_gate" in receipt:
        raise ValueError("v1 remote receipt must not claim a GPU allocation gate")
    competition_id = str(launch.get("competition_id") or "")
    if competition_id not in SUPPORTED_COMPETITIONS:
        raise ValueError("launch competition is outside the prospective pilot set")
    if receipt.get("competition_id") != competition_id:
        raise ValueError("remote receipt competition does not match the pilot")
    run_id = receipt.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id.startswith(f"{competition_id}-")
        or Path(run_id).name != run_id
        or "\\" in run_id
    ):
        raise ValueError("remote run_id is not a safe competition-scoped basename")
    stored_hash = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if stored_hash != _canonical_sha256(unhashed):
        raise ValueError("remote receipt self-hash is invalid")
    if (
        receipt.get("score") is not None
        or receipt.get("is_full_suite_score") is not False
        or receipt.get("legacy_tinker_coverage_increment") != 0
        or receipt.get("sample_reused") is not False
    ):
        raise ValueError("remote receipt violates the merged-vLLM scientific boundary")
    if receipt.get("combined_gpu_cpu_reservation") != launch.get("combined_gpu_cpu_reservation"):
        raise ValueError("remote receipt reservation does not match the launch")
    generation = receipt.get("generation")
    if not isinstance(generation, dict) or generation.get("generation_mode") != (
        "fresh_merged_vllm_generation"
    ):
        raise ValueError("remote receipt does not prove fresh merged-vLLM generation")
    if generation.get("prompt_template_version") != launch.get("prompt_template_version"):
        raise ValueError("remote prompt-template version does not match the launch")
    checkpoint = receipt.get("checkpoint_verification")
    if not isinstance(checkpoint, dict) or checkpoint.get("verified_shard_count") != 26:
        raise ValueError("remote receipt does not verify all merged checkpoint shards")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("remote receipt artifact hashes are absent")
    solution = result.get("solution_code")
    if (
        not isinstance(solution, str)
        or artifacts.get("solution_sha256")
        != hashlib.sha256((solution + "\n").encode("utf-8")).hexdigest()
    ):
        raise ValueError("returned solution does not match its receipt")
    grade = result.get("native_grade_json")
    receipt_grade = receipt.get("native_grade")
    if not isinstance(grade, dict) or receipt_grade != grade:
        raise ValueError("returned native grade does not match its receipt payload")
    if grade.get("competition_id") != competition_id:
        raise ValueError("returned native grade competition does not match the pilot")
    valid_submission = grade.get("valid_submission") is True
    grade_score = grade.get("score")
    expected_status = (
        "NATIVE_SINGLE_COMPETITION_GRADED"
        if grade_score is not None and valid_submission
        else "NATIVE_SINGLE_COMPETITION_INVALID"
    )
    expected_competition_score = grade_score if expected_status.endswith("GRADED") else None
    if receipt.get("status") != expected_status:
        raise ValueError("remote receipt status does not match the native grade")
    if receipt.get("competition_score") != expected_competition_score:
        raise ValueError("remote receipt competition score does not match the native grade")
    grade_hash = hashlib.sha256(json.dumps(grade, sort_keys=True).encode("utf-8")).hexdigest()
    if artifacts.get("native_grade_sha256") != grade_hash:
        raise ValueError("returned native grade does not match its receipt")
    submission = result.get("submission_csv")
    submission_hash = (
        hashlib.sha256(submission.encode("utf-8")).hexdigest()
        if isinstance(submission, str)
        else None
    )
    if artifacts.get("submission_sha256") != submission_hash:
        raise ValueError("returned submission does not match its receipt")
    budget = receipt.get("merged_vllm_budget")
    actual = budget.get("estimated_combined_usd") if isinstance(budget, dict) else None
    reserved = launch["combined_gpu_cpu_reservation"]["projected_combined_usd"]
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
        or actual < 0
        or actual > reserved
    ):
        raise ValueError("returned merged-vLLM cost is absent or exceeds reservation")
    if launch_schema == LAUNCH_SCHEMA_V3:
        if set(budget) != {
            "mode",
            "estimated_modal_gpu_usd",
            "estimated_modal_cpu_usd",
            "estimated_combined_usd",
            "reservation",
        }:
            raise ValueError("v3 returned merged-vLLM budget fields are invalid")
        gpu = budget.get("estimated_modal_gpu_usd")
        cpu = budget.get("estimated_modal_cpu_usd")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in (gpu, cpu)
        ) or actual != round(gpu + cpu, 9):
            raise ValueError("v3 returned GPU/CPU budget is inconsistent")
        if budget.get("reservation") != launch.get("combined_gpu_cpu_reservation"):
            raise ValueError("v3 returned budget reservation does not match the launch")
    return receipt


def validate_cpu_preparation_failure_result(result: Any, launch: dict[str, Any]) -> dict[str, Any]:
    """Validate a v3 CPU-only failure without creating model evidence."""

    if launch.get("schema_version") != LAUNCH_SCHEMA_V3:
        raise ValueError("CPU preparation failure requires a v3 launch")
    if not isinstance(result, dict) or set(result) != {"receipt"}:
        raise ValueError("CPU preparation failure result must contain only its receipt")
    receipt = result.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("CPU preparation failure receipt is absent")
    expected_receipt_fields = {
        "schema_version",
        "status",
        "scientific_status",
        "arm_id",
        "competition_id",
        "reservation_id",
        "launch_nonce",
        "recorded_at_utc",
        "phase",
        "prepare_elapsed_seconds",
        "failure",
        "combined_gpu_cpu_reservation",
        "gpu_generation_started",
        "sample_reused",
        "competition_score",
        "score",
        "is_full_suite_score",
        "legacy_tinker_coverage_increment",
        "merged_vllm_budget",
        "claim_boundary",
        "receipt_sha256",
    }
    if set(receipt) != expected_receipt_fields:
        raise ValueError("CPU preparation failure receipt fields are invalid")
    if (
        receipt.get("schema_version") != CPU_PREPARATION_FAILURE_SCHEMA_V1
        or receipt.get("arm_id") != ARM_ID
        or receipt.get("status") != "CPU_PREPARATION_FAILED_BEFORE_GPU_GENERATION"
        or receipt.get("scientific_status") != "NON_SCORE_EXECUTION_CONTROL_EVIDENCE"
    ):
        raise ValueError("CPU preparation failure schema or status is invalid")
    if (
        receipt.get("phase") != CPU_PREPARATION_FAILURE_PHASE
        or receipt.get("claim_boundary") != CPU_PREPARATION_FAILURE_CLAIM_BOUNDARY
    ):
        raise ValueError("CPU preparation failure metadata is invalid")
    recorded_at_utc = receipt.get("recorded_at_utc")
    if (
        not isinstance(recorded_at_utc, str)
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)",
            recorded_at_utc,
        )
        is None
    ):
        raise ValueError("CPU preparation failure timestamp is invalid")
    try:
        parsed_recorded_at = datetime.fromisoformat(recorded_at_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("CPU preparation failure timestamp is invalid") from exc
    if parsed_recorded_at.utcoffset() != timedelta(0):
        raise ValueError("CPU preparation failure timestamp is not UTC")
    prepare_elapsed_seconds = receipt.get("prepare_elapsed_seconds")
    if (
        isinstance(prepare_elapsed_seconds, bool)
        or not isinstance(prepare_elapsed_seconds, (int, float))
        or not math.isfinite(prepare_elapsed_seconds)
        or prepare_elapsed_seconds < 0
    ):
        raise ValueError("CPU preparation failure elapsed time is invalid")
    failure = receipt.get("failure")
    if not isinstance(failure, dict) or set(failure) != {
        "error_type",
        "error_message_tail",
        "error_message_sha256",
        "redaction_applied",
    }:
        raise ValueError("CPU preparation failure error evidence is invalid")
    error_message_tail = failure.get("error_message_tail")
    error_type = failure.get("error_type")
    error_message_sha256 = failure.get("error_message_sha256")
    if (
        not isinstance(error_message_tail, str)
        or not error_message_tail
        or len(error_message_tail) > 3000
        or not isinstance(failure.get("redaction_applied"), bool)
        or not isinstance(error_type, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type) is None
        or not isinstance(error_message_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", error_message_sha256) is None
    ):
        raise ValueError("CPU preparation failure error evidence is invalid")
    canonical_tail, would_redact = _sanitize_failure_message(error_message_tail)
    if would_redact or canonical_tail != error_message_tail:
        raise ValueError("CPU preparation failure error evidence is not sanitized")
    stored_hash = receipt.get("receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if stored_hash != _canonical_sha256(unhashed):
        raise ValueError("CPU preparation failure receipt self-hash is invalid")
    expected_bindings = {
        "competition_id": launch.get("competition_id"),
        "reservation_id": launch.get("spend_reservation_id"),
        "launch_nonce": launch.get("launch_nonce"),
    }
    for field, expected in expected_bindings.items():
        if receipt.get(field) != expected:
            raise ValueError(f"CPU preparation failure {field} binding is invalid")
    if receipt.get("combined_gpu_cpu_reservation") != launch.get("combined_gpu_cpu_reservation"):
        raise ValueError("CPU preparation failure reservation does not match the launch")
    if (
        receipt.get("gpu_generation_started") is not False
        or receipt.get("sample_reused") is not False
        or receipt.get("competition_score") is not None
        or receipt.get("score") is not None
        or receipt.get("is_full_suite_score") is not False
        or receipt.get("legacy_tinker_coverage_increment") != 0
    ):
        raise ValueError("CPU preparation failure violates the scientific boundary")
    budget = receipt.get("merged_vllm_budget")
    if not isinstance(budget, dict) or set(budget) != {
        "mode",
        "estimated_modal_gpu_usd",
        "estimated_modal_cpu_usd",
        "estimated_combined_usd",
        "reservation",
    }:
        raise ValueError("CPU preparation failure budget is absent")
    actual_gpu = budget.get("estimated_modal_gpu_usd")
    actual_cpu = budget.get("estimated_modal_cpu_usd")
    actual_total = budget.get("estimated_combined_usd")
    reserved_cpu = launch["combined_gpu_cpu_reservation"]["projected_modal_cpu_usd"]
    if (
        budget.get("mode") != "cpu_preparation_failure_before_gpu_generation"
        or isinstance(actual_gpu, bool)
        or not isinstance(actual_gpu, (int, float))
        or not math.isfinite(actual_gpu)
        or actual_gpu != 0.0
        or isinstance(actual_cpu, bool)
        or not isinstance(actual_cpu, (int, float))
        or not math.isfinite(actual_cpu)
        or actual_cpu < 0
        or actual_cpu > reserved_cpu
        or isinstance(actual_total, bool)
        or not isinstance(actual_total, (int, float))
        or not math.isfinite(actual_total)
        or actual_total != actual_cpu
    ):
        raise ValueError("CPU preparation failure cost is invalid")
    if budget.get("reservation") != launch.get("combined_gpu_cpu_reservation"):
        raise ValueError("CPU preparation failure budget reservation is invalid")
    forbidden = {"generation", "native_grade", "artifacts", "allocation_attestation"}
    if forbidden.intersection(receipt):
        raise ValueError("CPU preparation failure claims forbidden model evidence")
    return receipt


def _cpu_failure_attempt_id(launch: dict[str, Any], receipt: dict[str, Any]) -> str:
    competition_id = receipt["competition_id"]
    if (
        not isinstance(competition_id, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", competition_id) is None
    ):
        raise ValueError("CPU preparation failure competition_id is not a safe basename")
    return f"{competition_id}-cpu-prepare-{str(launch['launch_nonce'])[:12]}"


def _materialize_cpu_preparation_failure(result: dict[str, Any], launch: dict[str, Any]) -> Path:
    receipt = validate_cpu_preparation_failure_result(result, launch)
    destination = Path(launch["output_dir"]) / _cpu_failure_attempt_id(launch, receipt)
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = {
        "solution.py",
        "native_grade.json",
        "submission.csv",
        "collection_receipt.json",
        "launch_terminal_receipt.json",
    }
    present = sorted(name for name in unexpected if (destination / name).exists())
    if present:
        raise ValueError(
            "CPU preparation failure destination contains model evidence: " + ", ".join(present)
        )
    receipt_path = destination / "receipt.json"
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if receipt_path.exists() and receipt_path.read_bytes() != payload:
        raise ValueError("existing CPU preparation failure receipt differs")
    if not receipt_path.exists():
        _write_bytes_atomic(receipt_path, payload)
    return destination


def _materialize(result: dict[str, Any], output_dir: Path) -> Path:
    receipt = result["receipt"]
    destination = output_dir / receipt["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    payloads = {
        "receipt.json": (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "solution.py": (result["solution_code"] + "\n").encode("utf-8"),
        "native_grade.json": json.dumps(result["native_grade_json"], sort_keys=True).encode(
            "utf-8"
        ),
    }
    if result.get("submission_csv") is not None:
        payloads["submission.csv"] = result["submission_csv"].encode("utf-8")
    for name, payload in payloads.items():
        path = destination / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"existing materialized artifact differs: {path}")
        if not path.exists():
            _write_bytes_atomic(path, payload)
    return destination


def write_legacy_terminal_override(receipt_path: Path) -> Path:
    """Prevent the legacy driver from resampling a terminal separate-arm task."""

    receipt = _load_json(receipt_path)
    competition_id = str(receipt.get("competition_id") or "")
    run_id = str(receipt.get("run_id") or "")
    if competition_id not in SUPPORTED_COMPETITIONS or not run_id.startswith(f"{competition_id}-"):
        raise ValueError("merged-vLLM receipt does not bind to a prospective pilot")
    terminal_dir = LEGACY_OUTPUT_DIR / f"{competition_id}-merged-vllm-{run_id.rsplit('-', 1)[-1]}"
    terminal_path = terminal_dir / "launch_terminal_receipt.json"
    terminal = {
        "schema_version": "pavlov-e9-modal-launch-terminal-v1",
        "status": "MERGED_VLLM_SEPARATE_ARM_TERMINAL",
        "competition_id": competition_id,
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "relaunch_allowed": False,
        "model_sample_reused": False,
        "score": None,
        "is_full_suite_score": False,
        "separate_arm_id": receipt.get("arm_id"),
        "separate_arm_status": receipt.get("status"),
        "separate_arm_receipt": str(receipt_path),
        "separate_arm_receipt_sha256": _file_sha256(receipt_path),
        "reason": (
            "A fresh merged-vLLM sample completed and returned a terminal native receipt. "
            "The legacy Tinker driver must not resample this competition."
        ),
        "claim_boundary": (
            "Operational no-resample evidence only; not a model score, not legacy Tinker "
            "coverage, and not a 75-competition E9 suite score."
        ),
    }
    terminal["receipt_sha256"] = _canonical_sha256(terminal)
    if terminal_path.is_file():
        existing = _load_json(terminal_path)
        if existing != terminal:
            # recorded_at is observational metadata; preserve the original immutable receipt.
            existing_unhashed = dict(existing)
            stored_hash = existing_unhashed.pop("receipt_sha256", None)
            if stored_hash != _canonical_sha256(existing_unhashed):
                raise ValueError("existing launch-terminal override self-hash is invalid")
            return terminal_path
    _write_json_atomic(terminal_path, terminal)
    return terminal_path


def _reconciliation_identity(launch: dict[str, Any], receipt: dict[str, Any]) -> str:
    """Return a stable identity for the one permitted ledger reconciliation."""

    return _canonical_sha256(
        {
            "arm_id": ARM_ID,
            "reservation_id": launch["spend_reservation_id"],
            "run_id": receipt["run_id"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
    )


def _expected_pilot_ledger_entry(
    *, launch: dict[str, Any], receipt: dict[str, Any], destination: Path
) -> dict[str, Any]:
    actual = float(receipt["merged_vllm_budget"]["estimated_combined_usd"])
    return {
        "reconciliation_id": _reconciliation_identity(launch, receipt),
        "arm_id": ARM_ID,
        "competition_id": receipt["competition_id"],
        "run_id": receipt["run_id"],
        "estimated_actual_compute_usd": actual,
        "reservation_usd": launch["combined_gpu_cpu_reservation"]["projected_combined_usd"],
        "receipt": str(destination / "receipt.json"),
        "scientific_boundary": (
            "Separate fresh merged-vLLM arm; never added to legacy Tinker E9 coverage."
        ),
    }


def _expected_cpu_failure_ledger_entry(
    *, launch: dict[str, Any], receipt: dict[str, Any], destination: Path
) -> dict[str, Any]:
    actual = float(receipt["merged_vllm_budget"]["estimated_combined_usd"])
    return {
        "reconciliation_id": _canonical_sha256(
            {
                "arm_id": ARM_ID,
                "reservation_id": launch["spend_reservation_id"],
                "launch_nonce": launch["launch_nonce"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
        ),
        "arm_id": ARM_ID,
        "competition_id": receipt["competition_id"],
        "reservation_id": launch["spend_reservation_id"],
        "launch_nonce": launch["launch_nonce"],
        "failure_status": receipt["status"],
        "estimated_actual_compute_usd": actual,
        "reserved_cpu_usd": launch["combined_gpu_cpu_reservation"]["projected_modal_cpu_usd"],
        "reserved_combined_usd": launch["combined_gpu_cpu_reservation"]["projected_combined_usd"],
        "receipt": str(destination / "receipt.json"),
        "scientific_boundary": "CPU-only operational failure; no model sample or score.",
    }


def _load_reconciled_cpu_failure_spend_evidence(
    *, launch: dict[str, Any], receipt: dict[str, Any], destination: Path
) -> dict[str, Any]:
    ledger = _load_json(Path(launch["spend_ledger"]))
    reservations_path = Path(launch["output_dir"]) / "_pending/spend_reservations.json"
    reservations = _reservations(reservations_path)
    matches = [
        item
        for item in reservations["reservations"]
        if item.get("reservation_id") == launch["spend_reservation_id"]
    ]
    if len(matches) != 1:
        raise ValueError("launch spend reservation is not unique")
    reservation = matches[0]
    expected = _expected_cpu_failure_ledger_entry(
        launch=launch, receipt=receipt, destination=destination
    )
    if (
        reservation.get("status") != "CPU_PREPARATION_FAILURE_RECONCILED_TO_SPEND_LEDGER"
        or reservation.get("reconciled_to_spend_ledger") is not True
        or reservation.get("estimated_actual_compute_usd")
        != expected["estimated_actual_compute_usd"]
        or reservation.get("materialized_receipt") != expected["receipt"]
        or reservation.get("reconciliation_id") != expected["reconciliation_id"]
    ):
        raise ValueError("CPU preparation reservation is not reconciled")
    failures = ledger.get("modal_e9_merged_vllm_cpu_preparation_failures")
    if not isinstance(failures, list):
        raise ValueError("CPU preparation failure spend history is malformed")
    matching_entries = [
        item for item in failures if item.get("reconciliation_id") == expected["reconciliation_id"]
    ]
    if len(matching_entries) != 1 or matching_entries[0] != expected:
        raise ValueError("CPU preparation failure spend evidence is absent or ambiguous")
    try:
        authorized = float(ledger["authorized_incremental_spend_usd"])
        total = float(ledger["total_counted_incremental_spend_usd"])
        remaining = float(ledger["remaining_authorized_incremental_spend_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("merged-vLLM spend ledger totals are malformed") from exc
    if (
        total > authorized
        or ledger.get("within_cap") is not True
        or remaining != round(authorized - total, 12)
    ):
        raise ValueError("CPU preparation failure spend is not reconciled within cap")
    return ledger


def _reconcile_cpu_preparation_failure_spend(
    *, launch: dict[str, Any], receipt: dict[str, Any], destination: Path
) -> dict[str, Any]:
    ledger_path = Path(launch["spend_ledger"])
    ledger = _load_json(ledger_path)
    reservations_path = Path(launch["output_dir"]) / "_pending/spend_reservations.json"
    reservations = _reservations(reservations_path)
    matches = [
        item
        for item in reservations["reservations"]
        if item.get("reservation_id") == launch["spend_reservation_id"]
    ]
    if len(matches) != 1:
        raise ValueError("launch spend reservation is not unique")
    reservation = matches[0]
    entry = _expected_cpu_failure_ledger_entry(
        launch=launch, receipt=receipt, destination=destination
    )
    failures = ledger.get("modal_e9_merged_vllm_cpu_preparation_failures")
    if failures is None:
        failures = []
        ledger["modal_e9_merged_vllm_cpu_preparation_failures"] = failures
    if not isinstance(failures, list):
        raise ValueError("CPU preparation failure spend history is malformed")
    matches_in_ledger = [
        item for item in failures if item.get("reconciliation_id") == entry["reconciliation_id"]
    ]
    if len(matches_in_ledger) > 1:
        raise ValueError("CPU preparation failure reconciliation is not unique")
    if matches_in_ledger:
        if matches_in_ledger[0] != entry:
            raise ValueError("existing CPU preparation reconciliation differs")
    else:
        if reservation.get("reconciled_to_spend_ledger") is True:
            raise ValueError("reservation is reconciled but CPU failure ledger evidence is absent")
        previous_total = float(ledger["total_counted_incremental_spend_usd"])
        authorized = float(ledger["authorized_incremental_spend_usd"])
        new_total = round(previous_total + entry["estimated_actual_compute_usd"], 12)
        if new_total > authorized:
            raise ValueError("CPU preparation cost would exceed authorized campaign spend")
        ledger["total_counted_incremental_spend_usd"] = new_total
        ledger["remaining_authorized_incremental_spend_usd"] = round(authorized - new_total, 12)
        ledger["within_cap"] = new_total <= authorized
        ledger["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        failures.append(entry)
        _write_json_atomic(ledger_path, ledger)
    if reservation.get("reconciled_to_spend_ledger") is not True:
        reservation.update(
            {
                "status": "CPU_PREPARATION_FAILURE_RECONCILED_TO_SPEND_LEDGER",
                "reconciled_to_spend_ledger": True,
                "estimated_actual_compute_usd": entry["estimated_actual_compute_usd"],
                "materialized_receipt": entry["receipt"],
                "reconciliation_id": entry["reconciliation_id"],
                "reconciled_at_utc": ledger["recorded_at_utc"],
            }
        )
        _write_json_atomic(reservations_path, reservations)
    else:
        _load_reconciled_cpu_failure_spend_evidence(
            launch=launch, receipt=receipt, destination=destination
        )
    return ledger


def _load_reconciled_spend_evidence(
    *, launch: dict[str, Any], receipt: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Fail closed unless both ledgers prove this exact collection was reconciled."""

    ledger = _load_json(Path(launch["spend_ledger"]))
    reservations_path = Path(launch["output_dir"]) / "_pending/spend_reservations.json"
    reservations = _reservations(reservations_path)
    matches = [
        item
        for item in reservations["reservations"]
        if item.get("reservation_id") == launch["spend_reservation_id"]
    ]
    if len(matches) != 1:
        raise ValueError("launch spend reservation is not unique")
    reservation = matches[0]
    expected_entry = _expected_pilot_ledger_entry(
        launch=launch, receipt=receipt, destination=destination
    )
    actual = expected_entry["estimated_actual_compute_usd"]
    if (
        reservation.get("status") != "REMOTE_RESULT_RECONCILED_TO_SPEND_LEDGER"
        or reservation.get("reconciled_to_spend_ledger") is not True
        or reservation.get("estimated_actual_compute_usd") != actual
        or reservation.get("materialized_receipt") != expected_entry["receipt"]
    ):
        raise ValueError("spend reservation does not prove this result was reconciled")
    pilots = ledger.get("modal_e9_merged_vllm_pilots")
    if not isinstance(pilots, list):
        raise ValueError("merged-vLLM pilot spend history is malformed")
    matching_entries = [
        item
        for item in pilots
        if item.get("reconciliation_id") == expected_entry["reconciliation_id"]
    ]
    if matching_entries:
        if len(matching_entries) != 1 or matching_entries[0] != expected_entry:
            raise ValueError("spend ledger does not prove this result was reconciled")
    else:
        # Seven materialized pilot entries predate reconciliation_id.  Their
        # receipt hash is independently bound by the launch, collection, and
        # terminal override checks above, while the historical ledger stores
        # the immutable receipt path and the remaining identity fields.
        legacy_entry = dict(expected_entry)
        legacy_entry.pop("reconciliation_id")
        legacy_matches = [
            item for item in pilots if "reconciliation_id" not in item and item == legacy_entry
        ]
        if len(legacy_matches) != 1:
            raise ValueError("spend ledger does not prove this result was reconciled")
    try:
        authorized = float(ledger["authorized_incremental_spend_usd"])
        total = float(ledger["total_counted_incremental_spend_usd"])
        remaining = float(ledger["remaining_authorized_incremental_spend_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("merged-vLLM spend ledger totals are malformed") from exc
    if (
        total > authorized
        or ledger.get("within_cap") is not True
        or remaining != round(authorized - total, 12)
    ):
        raise ValueError("merged-vLLM spend ledger totals do not prove an in-cap reconciliation")
    return ledger


def _reconcile_spend(
    *, launch: dict[str, Any], receipt: dict[str, Any], destination: Path
) -> dict[str, Any]:
    ledger_path = Path(launch["spend_ledger"])
    ledger = _load_json(ledger_path)
    reservations_path = Path(launch["output_dir"]) / "_pending/spend_reservations.json"
    reservations = _reservations(reservations_path)
    matches = [
        item
        for item in reservations["reservations"]
        if item.get("reservation_id") == launch["spend_reservation_id"]
    ]
    if len(matches) != 1:
        raise ValueError("launch spend reservation is not unique")
    reservation = matches[0]
    pilot_entry = _expected_pilot_ledger_entry(
        launch=launch, receipt=receipt, destination=destination
    )
    actual = pilot_entry["estimated_actual_compute_usd"]
    pilots = ledger.get("modal_e9_merged_vllm_pilots")
    if pilots is None:
        pilots = []
        ledger["modal_e9_merged_vllm_pilots"] = pilots
    if not isinstance(pilots, list):
        raise ValueError("merged-vLLM pilot spend history is malformed")
    matching_entries = [
        item for item in pilots if item.get("reconciliation_id") == pilot_entry["reconciliation_id"]
    ]
    if len(matching_entries) > 1:
        raise ValueError("merged-vLLM pilot reconciliation identity is not unique")
    if matching_entries:
        if matching_entries[0] != pilot_entry:
            raise ValueError("existing spend reconciliation differs from receipt")
    elif any(item.get("run_id") == receipt["run_id"] for item in pilots):
        raise ValueError("existing pilot spend history conflicts with this run")
    else:
        if reservation.get("reconciled_to_spend_ledger") is True:
            raise ValueError("spend reservation is reconciled but ledger evidence is absent")
        previous_total = float(ledger["total_counted_incremental_spend_usd"])
        authorized = float(ledger["authorized_incremental_spend_usd"])
        new_total = round(previous_total + actual, 12)
        if new_total > authorized:
            raise ValueError("observed pilot estimate would exceed authorized campaign spend")
        ledger["total_counted_incremental_spend_usd"] = new_total
        ledger["remaining_authorized_incremental_spend_usd"] = round(authorized - new_total, 12)
        ledger["within_cap"] = new_total <= authorized
        ledger["recorded_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pilots.append(pilot_entry)
        _write_json_atomic(ledger_path, ledger)
    if reservation.get("reconciled_to_spend_ledger") is not True:
        reservation.update(
            {
                "status": "REMOTE_RESULT_RECONCILED_TO_SPEND_LEDGER",
                "reconciled_to_spend_ledger": True,
                "estimated_actual_compute_usd": actual,
                "materialized_receipt": str(destination / "receipt.json"),
                "reconciled_at_utc": ledger["recorded_at_utc"],
            }
        )
        _write_json_atomic(reservations_path, reservations)
    else:
        _load_reconciled_spend_evidence(launch=launch, receipt=receipt, destination=destination)
    return ledger


def _validate_materialized_cpu_preparation_failure(
    launch: dict[str, Any], destination: Path
) -> None:
    receipt_path = destination / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError("materialized CPU preparation failure receipt is absent")
    unexpected = {
        "solution.py",
        "native_grade.json",
        "submission.csv",
        "collection_receipt.json",
        "launch_terminal_receipt.json",
    }
    present = sorted(name for name in unexpected if (destination / name).exists())
    if present:
        raise ValueError(
            "materialized CPU preparation failure contains model evidence: " + ", ".join(present)
        )
    receipt = _load_json(receipt_path)
    validate_cpu_preparation_failure_result({"receipt": receipt}, launch)
    receipt_hash = _file_sha256(receipt_path)
    if (
        launch.get("status") != "CPU_PREPARATION_FAILURE_MATERIALIZED"
        or launch.get("scientific_status") != "NON_SCORE_EXECUTION_CONTROL_EVIDENCE"
        or launch.get("cpu_preparation_status") != receipt.get("status")
        or launch.get("competition_score") is not None
        or launch.get("score") is not None
        or launch.get("is_full_suite_score") is not False
        or launch.get("legacy_tinker_coverage_increment") != 0
        or launch.get("materialized_output_dir") != str(destination)
        or launch.get("remote_receipt_sha256") != receipt_hash
        or launch.get("counted_incremental_spend_usd")
        != receipt["merged_vllm_budget"]["estimated_combined_usd"]
        or "legacy_no_resample_override" in launch
    ):
        raise ValueError("materialized CPU preparation launch does not match its receipt")
    ledger = _load_reconciled_cpu_failure_spend_evidence(
        launch=launch, receipt=receipt, destination=destination
    )
    try:
        authorized = float(ledger["authorized_incremental_spend_usd"])
        snapshot_remaining = float(launch["remaining_authorized_incremental_spend_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("CPU preparation spend snapshot is malformed") from exc
    if snapshot_remaining < 0 or snapshot_remaining > authorized:
        raise ValueError("CPU preparation spend snapshot is outside the campaign cap")


def _validate_materialized_collection(launch: dict[str, Any], destination: Path) -> None:
    """Re-check every immutable artifact before trusting a prior collection."""

    receipt_path = destination / "receipt.json"
    solution_path = destination / "solution.py"
    grade_path = destination / "native_grade.json"
    collection_path = destination / "collection_receipt.json"
    for path in (receipt_path, solution_path, grade_path, collection_path):
        if not path.is_file():
            raise ValueError(f"materialized launch is missing required evidence: {path.name}")
    receipt = _load_json(receipt_path)
    solution_payload = solution_path.read_text(encoding="utf-8")
    if not solution_payload.endswith("\n"):
        raise ValueError("materialized solution is not newline-terminated")
    native_grade = _load_json(grade_path)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("materialized receipt artifact hashes are absent")
    submission_path = destination / "submission.csv"
    expected_submission_hash = artifacts.get("submission_sha256")
    if expected_submission_hash is None:
        if submission_path.exists():
            raise ValueError("materialized result has an unexpected submission")
        submission = None
    else:
        if not submission_path.is_file():
            raise ValueError("materialized result is missing its submission")
        submission = submission_path.read_text(encoding="utf-8")
    validate_returned_result(
        {
            "receipt": receipt,
            "solution_code": solution_payload[:-1],
            "native_grade_json": native_grade,
            "submission_csv": submission,
        },
        launch,
    )
    receipt_hash = _file_sha256(receipt_path)
    if launch.get("remote_receipt_sha256") != receipt_hash:
        raise ValueError("materialized launch receipt hash does not match its evidence")
    collection = _load_json(collection_path)
    if (
        collection.get("schema_version") != COLLECTION_SCHEMA
        or collection.get("status") != "REMOTE_RESULT_MATERIALIZED"
        or collection.get("function_call_id") != launch.get("function_call_id")
        or collection.get("competition_id") != receipt.get("competition_id")
        or collection.get("arm_id") != ARM_ID
        or collection.get("output_dir") != str(destination)
        or collection.get("remote_receipt_sha256") != receipt_hash
        or collection.get("competition_score") != receipt.get("competition_score")
        or collection.get("score") is not None
        or collection.get("is_full_suite_score") is not False
        or collection.get("legacy_tinker_coverage_increment") != 0
        or collection.get("counted_incremental_spend_usd")
        != receipt["merged_vllm_budget"]["estimated_combined_usd"]
    ):
        raise ValueError("materialized collection receipt does not match immutable evidence")
    terminal_path = Path(str(collection.get("legacy_no_resample_override") or ""))
    if not terminal_path.is_file():
        raise ValueError("materialized collection is missing its terminal override")
    terminal = _load_json(terminal_path)
    terminal_unhashed = dict(terminal)
    terminal_hash = terminal_unhashed.pop("receipt_sha256", None)
    if terminal_hash != _canonical_sha256(terminal_unhashed):
        raise ValueError("materialized terminal override self-hash is invalid")
    terminal_status = terminal.get("status")
    expected_terminal_statuses = {"MERGED_VLLM_SEPARATE_ARM_TERMINAL"}
    if receipt.get("status") == "NATIVE_SINGLE_COMPETITION_GRADED":
        expected_terminal_statuses.add("MERGED_VLLM_SEPARATE_ARM_TERMINAL_GRADED")
    elif receipt.get("status") == "NATIVE_SINGLE_COMPETITION_INVALID":
        expected_terminal_statuses.add("MERGED_VLLM_SEPARATE_ARM_TERMINAL_INVALID")
    if (
        terminal.get("schema_version") != "pavlov-e9-modal-launch-terminal-v1"
        or terminal_status not in expected_terminal_statuses
        or terminal.get("competition_id") != receipt.get("competition_id")
        or terminal.get("relaunch_allowed") is not False
        or terminal.get("model_sample_reused") is not False
        or terminal.get("score") is not None
        or terminal.get("is_full_suite_score") is not False
        or terminal.get("separate_arm_id") != ARM_ID
        or terminal.get("separate_arm_status") != receipt.get("status")
        or terminal.get("separate_arm_receipt") != str(receipt_path)
        or terminal.get("separate_arm_receipt_sha256") != receipt_hash
    ):
        raise ValueError("materialized terminal override does not match immutable evidence")
    if (
        launch.get("scientific_status") != receipt.get("status")
        or launch.get("competition_score") != receipt.get("competition_score")
        or launch.get("materialized_output_dir") != str(destination)
        or launch.get("remote_receipt_sha256") != receipt_hash
        or launch.get("collected_at_epoch") != collection.get("collected_at_epoch")
    ):
        raise ValueError("materialized launch does not match immutable collection evidence")
    ledger = _load_reconciled_spend_evidence(
        launch=launch, receipt=receipt, destination=destination
    )
    collection_remaining = collection.get("remaining_authorized_incremental_spend_usd")
    try:
        authorized = float(ledger["authorized_incremental_spend_usd"])
        snapshot_remaining = float(collection_remaining)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("materialized collection spend snapshot is malformed") from exc
    # This is an immutable collection-time snapshot.  Later independent pilots
    # legitimately change the mutable campaign ledger balance, so equality to
    # its current remaining amount would invalidate sound historical evidence.
    if snapshot_remaining < 0 or snapshot_remaining > authorized:
        raise ValueError("materialized collection spend snapshot is outside the campaign cap")


def collect_pilot(launch_path: Path, *, timeout_seconds: float = 5.0) -> Path:
    """Collect, validate, materialize, and reconcile a spawned pilot."""

    import modal

    launch = _load_json(launch_path)
    stored_hash = launch.get("launch_sha256")
    unhashed = dict(launch)
    unhashed.pop("launch_sha256", None)
    if launch.get("schema_version") not in {
        LAUNCH_SCHEMA_V1,
        LAUNCH_SCHEMA_V2,
        LAUNCH_SCHEMA_V3,
    } or stored_hash != _canonical_sha256(unhashed):
        raise ValueError("invalid merged-vLLM launch receipt")
    call_id = str(launch.get("function_call_id") or "")
    if not call_id.startswith("fc-"):
        raise ValueError("invalid Modal FunctionCall ID")
    if launch.get("status") == "REMOTE_RESULT_MATERIALIZED":
        destination = Path(launch["materialized_output_dir"])
        _validate_materialized_collection(launch, destination)
        return destination
    if launch.get("status") == "CPU_PREPARATION_FAILURE_MATERIALIZED":
        destination = Path(launch["materialized_output_dir"])
        _validate_materialized_cpu_preparation_failure(launch, destination)
        return destination
    call = modal.FunctionCall.from_id(call_id)
    result = call.get(timeout=timeout_seconds)
    result_receipt = result.get("receipt") if isinstance(result, dict) else None
    if (
        isinstance(result_receipt, dict)
        and result_receipt.get("schema_version") == CPU_PREPARATION_FAILURE_SCHEMA_V1
    ):
        receipt = validate_cpu_preparation_failure_result(result, launch)
        destination = _materialize_cpu_preparation_failure(result, launch)
        ledger = _reconcile_cpu_preparation_failure_spend(
            launch=launch, receipt=receipt, destination=destination
        )
        receipt_hash = _file_sha256(destination / "receipt.json")
        launch.update(
            {
                "status": "CPU_PREPARATION_FAILURE_MATERIALIZED",
                "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
                "cpu_preparation_status": receipt["status"],
                "competition_score": None,
                "score": None,
                "is_full_suite_score": False,
                "legacy_tinker_coverage_increment": 0,
                "materialized_output_dir": str(destination),
                "remote_receipt_sha256": receipt_hash,
                "counted_incremental_spend_usd": receipt["merged_vllm_budget"][
                    "estimated_combined_usd"
                ],
                "remaining_authorized_incremental_spend_usd": ledger[
                    "remaining_authorized_incremental_spend_usd"
                ],
                "collected_at_epoch": time.time(),
            }
        )
        launch.pop("launch_sha256", None)
        launch["launch_sha256"] = _canonical_sha256(launch)
        _write_json_atomic(launch_path, launch)
        return destination
    receipt = validate_returned_result(result, launch)
    destination = _materialize(result, Path(launch["output_dir"]))
    terminal_override = write_legacy_terminal_override(destination / "receipt.json")
    ledger = _reconcile_spend(launch=launch, receipt=receipt, destination=destination)
    collection = {
        "schema_version": COLLECTION_SCHEMA,
        "status": "REMOTE_RESULT_MATERIALIZED",
        "function_call_id": call_id,
        "competition_id": receipt["competition_id"],
        "arm_id": ARM_ID,
        "output_dir": str(destination),
        "remote_receipt_sha256": _file_sha256(destination / "receipt.json"),
        "legacy_no_resample_override": str(terminal_override),
        "competition_score": receipt.get("competition_score"),
        "score": None,
        "is_full_suite_score": False,
        "legacy_tinker_coverage_increment": 0,
        "counted_incremental_spend_usd": receipt["merged_vllm_budget"]["estimated_combined_usd"],
        "remaining_authorized_incremental_spend_usd": ledger[
            "remaining_authorized_incremental_spend_usd"
        ],
        "collected_at_epoch": time.time(),
    }
    _write_json_atomic(destination / "collection_receipt.json", collection)
    launch.update(
        {
            "status": "REMOTE_RESULT_MATERIALIZED",
            "scientific_status": receipt.get("status"),
            "competition_score": receipt.get("competition_score"),
            "materialized_output_dir": str(destination),
            "remote_receipt_sha256": collection["remote_receipt_sha256"],
            "collected_at_epoch": collection["collected_at_epoch"],
        }
    )
    launch.pop("launch_sha256", None)
    launch["launch_sha256"] = _canonical_sha256(launch)
    _write_json_atomic(launch_path, launch)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--competition-id", required=True)
    spawn = sub.add_parser("spawn")
    spawn.add_argument("--competition-id", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("launch_receipt", type=Path)
    collect.add_argument("--timeout-seconds", type=float, default=5.0)
    terminalize = sub.add_parser("terminalize")
    terminalize.add_argument("receipt", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "preflight":
        gate, reservation = _validate_campaign_gate(competition_id=args.competition_id)
        print(json.dumps({"gate": gate, "reservation": reservation}, indent=2))
    elif args.command == "spawn":
        path = spawn_pilot(competition_id=args.competition_id)
        print(json.dumps({"launch_receipt": str(path)}, indent=2))
    elif args.command == "collect":
        path = collect_pilot(args.launch_receipt, timeout_seconds=args.timeout_seconds)
        print(json.dumps({"output_dir": str(path)}, indent=2))
    else:
        path = write_legacy_terminal_override(args.receipt)
        print(json.dumps({"terminal_override": str(path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
