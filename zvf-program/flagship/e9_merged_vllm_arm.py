"""Fail-closed contracts for a *separate* E9 merged-vLLM experiment arm.

This module deliberately contains no Modal launch entrypoint.  It is the small
contract that a future one-task GPU/CPU harness must satisfy before it is
allowed to spend: verify the immutable merged checkpoint, establish an online
W&B receipt, reserve both phases under the pilot cap, generate fresh code, and
write a non-aggregate E9 receipt.  In particular, it cannot fill holes in the
legacy Tinker-sampled E9 inventory.
"""

from __future__ import annotations

import hashlib
import json
import base64
import re
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

try:
    from .e9_mle_bench_streaming import LaunchGateError, extract_python_code
except ImportError:
    from e9_mle_bench_streaming import LaunchGateError, extract_python_code


ARM_SCHEMA_V1 = "pavlov-e9-merged-vllm-arm-v1"
ARM_SCHEMA_V2 = "pavlov-e9-merged-vllm-arm-v2"
ARM_SCHEMA_V3 = "pavlov-e9-merged-vllm-arm-v3"
ARM_SCHEMA = ARM_SCHEMA_V2
ALLOCATION_GATE_SCHEMA = "pavlov-e9-merged-vllm-allocation-gate-v1"
ALLOCATION_ATTESTATION_SCHEMA_V2 = "pavlov-e9-merged-vllm-allocation-attestation-v2"
CPU_PREPARATION_FAILURE_SCHEMA_V1 = "pavlov-e9-merged-vllm-cpu-prepare-failure-v1"
CPU_PREPARATION_FAILURE_PHASE = "mlebench_prepare"
CPU_PREPARATION_FAILURE_CLAIM_BOUNDARY = (
    "CPU preparation failure before model generation is operational evidence only; "
    "it is not a model sample, native grade, suite score, or legacy coverage."
)
ALLOCATION_ATTESTATION_KEY_ID = "e9-allocation-2026-08"
ALLOCATION_ATTESTATION_DOMAIN = b"PAVLOV-E9-ALLOCATION-V2\0"
ALLOCATION_ATTESTATION_MAX_TTL_SECONDS = 300
ALLOCATION_ATTESTATION_MAX_FUTURE_SKEW_SECONDS = 30
PROMPT_TEMPLATE_VERSION_V7 = "e9_hard_bounded_collections_v7"
ARM_ID = "e9_merged_vllm_seed809_v1"
MERGE_SCHEMA = "pavlov-e1-exact-lora-merge-v1"
PILOT_CAP_USD = Decimal("2.00")
EXPECTED_SHARD_COUNT = 26
LEGACY_TINKER_COVERAGE_INCREMENT = 0
EXPECTED_BASE_COMMIT = "995ad96eacd98c81ed38be0c5b274b04031597b0"
EXPECTED_ADAPTER_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"
EXPECTED_MERGE_METHOD = "streaming_lora_delta_merge_v1"
EXPECTED_WEIGHT_BYTES = 71_903_776_776
EXPECTED_MERGE_RECEIPT_SHA256 = "b509fa300db280880a910201943bce7f4516daf75052122d8db18d189844fe46"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lower_hex(value: Any, name: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
    ):
        raise LaunchGateError(f"{name} must be {length} lowercase hexadecimal characters")
    return value


def _decode_base64(value: Any, name: str, *, expected_bytes: int) -> bytes:
    if not isinstance(value, str) or not value:
        raise LaunchGateError(f"{name} is missing")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise LaunchGateError(f"{name} is not valid base64") from exc
    if len(decoded) != expected_bytes:
        raise LaunchGateError(f"{name} must decode to exactly {expected_bytes} bytes")
    return decoded


def _ed25519_types() -> tuple[Any, Any, Any]:
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise LaunchGateError(
            "cryptography is required for authenticated allocation attestations"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _money(value: Any, name: str) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise LaunchGateError(f"{name} is not a decimal USD amount") from exc
    if amount < 0:
        raise LaunchGateError(f"{name} is negative")
    return amount


def _utc_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LaunchGateError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LaunchGateError(f"{name} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise LaunchGateError(f"{name} must include a timezone")
    return parsed


def validate_merge_receipt_and_shards(
    merge_receipt: Mapping[str, Any],
    *,
    merged_root: Path,
    expected_total_bytes: int = EXPECTED_WEIGHT_BYTES,
    expected_receipt_sha256: str = EXPECTED_MERGE_RECEIPT_SHA256,
) -> dict[str, Any]:
    """Verify the receipt's self-hash and every expected checkpoint shard.

    The full 26-shard check is intentionally performed before model load.  A
    mere existing directory, a subset of hashes, or an E1 backend receipt is
    not sufficient provenance for this distinct arm.
    """

    receipt = dict(merge_receipt)
    stored_hash = receipt.pop("receipt_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != _canonical_sha256(receipt):
        raise LaunchGateError("merged checkpoint receipt self-hash is invalid")
    if stored_hash != expected_receipt_sha256:
        raise LaunchGateError("merged checkpoint receipt does not match the pinned artifact")
    if receipt.get("schema_version") != MERGE_SCHEMA:
        raise LaunchGateError("merged checkpoint receipt schema drifted")
    if receipt.get("status") != "READY":
        raise LaunchGateError("merged checkpoint receipt is not READY")
    if receipt.get("all_adapter_tensors_consumed") is not True:
        raise LaunchGateError("merged checkpoint has unconsumed adapter tensors")
    expected_identity = {
        "base_commit": EXPECTED_BASE_COMMIT,
        "adapter_commit": EXPECTED_ADAPTER_COMMIT,
        "merge_method": EXPECTED_MERGE_METHOD,
        "weight_bytes": expected_total_bytes,
        "merged_path": str(merged_root),
    }
    for field, expected in expected_identity.items():
        if receipt.get(field) != expected:
            raise LaunchGateError(f"merged checkpoint {field} drifted")
    shard_hashes = receipt.get("weight_shard_sha256")
    if not isinstance(shard_hashes, Mapping) or len(shard_hashes) != EXPECTED_SHARD_COUNT:
        raise LaunchGateError("merged checkpoint must pin exactly 26 shard hashes")
    if receipt.get("weight_file_count") != EXPECTED_SHARD_COUNT:
        raise LaunchGateError("merged checkpoint shard count drifted")
    if not merged_root.is_dir():
        raise LaunchGateError("merged checkpoint directory is missing")
    verified: dict[str, str] = {}
    actual_total_bytes = 0
    for name, expected_hash in sorted(shard_hashes.items()):
        if not isinstance(name, str) or Path(name).name != name:
            raise LaunchGateError("merged checkpoint has an unsafe shard name")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise LaunchGateError("merged checkpoint has an invalid shard hash")
        shard = merged_root / name
        if not shard.is_file() or _file_sha256(shard) != expected_hash:
            raise LaunchGateError(f"merged checkpoint shard hash drifted: {name}")
        actual_total_bytes += shard.stat().st_size
        verified[name] = expected_hash
    if actual_total_bytes != expected_total_bytes:
        raise LaunchGateError("merged checkpoint shard byte total drifted")
    return {
        "merge_receipt_sha256": stored_hash,
        "merged_path": str(merged_root),
        "verified_shard_count": len(verified),
        "verified_weight_bytes": actual_total_bytes,
        "weight_shard_sha256": verified,
    }


def reserve_pilot_budget(*, gpu_usd: Any, cpu_usd: Any) -> dict[str, Any]:
    """Reserve combined GPU and native-grading CPU cost under a hard $2 cap."""

    gpu = _money(gpu_usd, "projected GPU cost")
    cpu = _money(cpu_usd, "projected CPU cost")
    total = gpu + cpu
    if total <= 0 or total > PILOT_CAP_USD:
        raise LaunchGateError(
            f"combined GPU+CPU pilot reservation must be in (0, {PILOT_CAP_USD}] USD"
        )
    return {
        "pilot_cap_usd": float(PILOT_CAP_USD),
        "projected_modal_gpu_usd": float(gpu),
        "projected_modal_cpu_usd": float(cpu),
        "projected_combined_usd": float(total),
        "reservation_status": "RESERVED_PRE_LAUNCH",
    }


def require_online_wandb_before_paid_load(
    wandb_receipt: Any, *, paid_phase_started_at: Any
) -> dict[str, str]:
    """Require online W&B initialization strictly before a paid phase starts."""

    if not isinstance(wandb_receipt, Mapping):
        raise LaunchGateError("structured online W&B receipt is required")
    if wandb_receipt.get("mode") != "online":
        raise LaunchGateError("W&B must be initialized in online mode")
    run_id = wandb_receipt.get("run_id")
    url = wandb_receipt.get("url")
    if not isinstance(run_id, str) or not run_id:
        raise LaunchGateError("online W&B receipt has no concrete run_id")
    if not isinstance(url, str) or not url.startswith("https://wandb.ai/"):
        raise LaunchGateError("online W&B receipt has no concrete run URL")
    initialized_at = _utc_timestamp(wandb_receipt.get("initialized_at"), "W&B initialized_at")
    paid_started_at = _utc_timestamp(paid_phase_started_at, "paid phase started_at")
    if initialized_at >= paid_started_at:
        raise LaunchGateError("W&B initialization did not strictly precede the paid phase")
    return {
        "mode": "online",
        "run_id": run_id,
        "url": url,
        "initialized_at": initialized_at.isoformat(),
    }


def _validate_allocation_checkpoint(checkpoint_verification: Any) -> dict[str, Any]:
    if not isinstance(checkpoint_verification, Mapping):
        raise LaunchGateError("GPU allocation gate checkpoint verification is absent")
    checkpoint = dict(checkpoint_verification)
    if checkpoint.get("verified_shard_count") != EXPECTED_SHARD_COUNT:
        raise LaunchGateError("GPU allocation gate does not verify all 26 checkpoint shards")
    if checkpoint.get("base_commit") != EXPECTED_BASE_COMMIT:
        raise LaunchGateError("GPU allocation gate base commit drifted")
    if checkpoint.get("adapter_commit") != EXPECTED_ADAPTER_COMMIT:
        raise LaunchGateError("GPU allocation gate adapter commit drifted")
    if not isinstance(checkpoint.get("merge_receipt_sha256"), str):
        raise LaunchGateError("GPU allocation gate merge receipt hash is absent")
    shard_hashes = checkpoint.get("weight_shard_sha256")
    if not isinstance(shard_hashes, Mapping) or len(shard_hashes) != EXPECTED_SHARD_COUNT:
        raise LaunchGateError("GPU allocation gate does not retain all 26 shard hashes")
    return checkpoint


def build_gpu_allocation_gate(
    *,
    wandb_receipt: Mapping[str, Any],
    checkpoint_verification: Mapping[str, Any],
    preflight_completed_at: Any,
) -> dict[str, Any]:
    """Create the CPU-side provenance gate required before GPU allocation."""

    _utc_timestamp(preflight_completed_at, "GPU allocation preflight_completed_at")
    require_online_wandb_before_paid_load(
        wandb_receipt, paid_phase_started_at=preflight_completed_at
    )
    if wandb_receipt.get("server_confirmed") is not True:
        raise LaunchGateError("GPU allocation gate requires server-confirmed online W&B")
    if (
        not isinstance(wandb_receipt.get("server_run_path"), str)
        or not wandb_receipt["server_run_path"]
    ):
        raise LaunchGateError("GPU allocation gate W&B server confirmation is absent")
    checkpoint = _validate_allocation_checkpoint(checkpoint_verification)
    gate: dict[str, Any] = {
        "schema_version": ALLOCATION_GATE_SCHEMA,
        "status": "PRE_ALLOCATION_PROVENANCE_VERIFIED",
        "arm_id": ARM_ID,
        "preflight_completed_at": preflight_completed_at,
        "wandb_receipt": dict(wandb_receipt),
        "checkpoint_verification": checkpoint,
    }
    gate["allocation_gate_sha256"] = _canonical_sha256(gate)
    return gate


def validate_gpu_allocation_gate(gate: Any) -> dict[str, Any]:
    """Validate a self-hashed CPU preflight gate without allocating a GPU."""

    if not isinstance(gate, Mapping):
        raise LaunchGateError("GPU allocation gate is absent")
    normalized = dict(gate)
    stored_hash = normalized.pop("allocation_gate_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != _canonical_sha256(normalized):
        raise LaunchGateError("GPU allocation gate self-hash is invalid")
    if (
        normalized.get("schema_version") != ALLOCATION_GATE_SCHEMA
        or normalized.get("status") != "PRE_ALLOCATION_PROVENANCE_VERIFIED"
        or normalized.get("arm_id") != ARM_ID
    ):
        raise LaunchGateError("GPU allocation gate schema or arm identity is invalid")
    preflight_completed_at = normalized.get("preflight_completed_at")
    _utc_timestamp(preflight_completed_at, "GPU allocation preflight_completed_at")
    require_online_wandb_before_paid_load(
        normalized.get("wandb_receipt"), paid_phase_started_at=preflight_completed_at
    )
    wandb_receipt = normalized["wandb_receipt"]
    if (
        wandb_receipt.get("server_confirmed") is not True
        or not isinstance(wandb_receipt.get("server_run_path"), str)
        or not wandb_receipt["server_run_path"]
    ):
        raise LaunchGateError("GPU allocation gate requires server-confirmed online W&B")
    _validate_allocation_checkpoint(normalized.get("checkpoint_verification"))
    return dict(gate)


def build_allocation_attestation_v2(
    *,
    competition_id: str,
    reservation_id: str,
    launch_nonce: str,
    prompt_sha256: str,
    prompt_builder_sha256: str,
    deployment_revision: str,
    source_bundle_sha256: str,
    wandb_receipt: Mapping[str, Any],
    checkpoint_verification: Mapping[str, Any],
    reservation: Mapping[str, Any],
    issued_at_utc: str,
    expires_at_utc: str,
    private_key_seed_base64: str,
    key_id: str = ALLOCATION_ATTESTATION_KEY_ID,
) -> dict[str, Any]:
    """Sign a short-lived, launch-specific CPU provenance attestation."""

    if not isinstance(competition_id, str) or not competition_id:
        raise LaunchGateError("allocation attestation competition_id is missing")
    _lower_hex(reservation_id, "allocation attestation reservation_id", length=32)
    _lower_hex(launch_nonce, "allocation attestation launch_nonce")
    _lower_hex(prompt_sha256, "allocation attestation prompt_sha256")
    _lower_hex(
        prompt_builder_sha256,
        "allocation attestation prompt_builder_sha256",
    )
    _lower_hex(source_bundle_sha256, "allocation attestation source_bundle_sha256")
    if not isinstance(deployment_revision, str) or not deployment_revision:
        raise LaunchGateError("allocation attestation deployment revision is missing")
    issued = _utc_timestamp(issued_at_utc, "allocation attestation issued_at_utc")
    expires = _utc_timestamp(expires_at_utc, "allocation attestation expires_at_utc")
    ttl_seconds = (expires - issued).total_seconds()
    if ttl_seconds <= 0 or ttl_seconds > ALLOCATION_ATTESTATION_MAX_TTL_SECONDS:
        raise LaunchGateError("allocation attestation TTL is outside the 300-second limit")
    require_online_wandb_before_paid_load(wandb_receipt, paid_phase_started_at=issued_at_utc)
    if (
        wandb_receipt.get("server_confirmed") is not True
        or not isinstance(wandb_receipt.get("server_run_path"), str)
        or not wandb_receipt["server_run_path"]
    ):
        raise LaunchGateError("allocation attestation requires server-confirmed online W&B")
    checkpoint = _validate_allocation_checkpoint(checkpoint_verification)
    if reservation.get("reservation_status") != "RESERVED_PRE_LAUNCH":
        raise LaunchGateError("allocation attestation reservation is absent")
    if _money(reservation.get("projected_combined_usd"), "combined reservation") > PILOT_CAP_USD:
        raise LaunchGateError("allocation attestation reservation exceeds the pilot cap")
    payload: dict[str, Any] = {
        "arm_id": ARM_ID,
        "competition_id": competition_id,
        "reservation_id": reservation_id,
        "launch_nonce": launch_nonce,
        "launch_schema_version": "e9-merged-vllm-spawn-v3",
        "issued_at_utc": issued_at_utc,
        "expires_at_utc": expires_at_utc,
        "prompt_contract": {
            "template_version": PROMPT_TEMPLATE_VERSION_V7,
            "prompt_sha256": prompt_sha256,
            "prompt_builder_sha256": prompt_builder_sha256,
        },
        "deployment": {
            "app_name": "pavlov-e9-mle-bench-streaming",
            "deployment_revision": deployment_revision,
            "source_bundle_sha256": source_bundle_sha256,
            "run_function": "run_one",
            "generator_function": "E9MergedVllmGenerator.generate_program",
        },
        "reservation": dict(reservation),
        "reservation_sha256": _canonical_sha256(reservation),
        "wandb_receipt": dict(wandb_receipt),
        "checkpoint_verification": checkpoint,
    }
    seed = _decode_base64(
        private_key_seed_base64,
        "allocation attestation private key seed",
        expected_bytes=32,
    )
    Ed25519PrivateKey, _, _ = _ed25519_types()
    signature = Ed25519PrivateKey.from_private_bytes(seed).sign(
        ALLOCATION_ATTESTATION_DOMAIN + _canonical_bytes(payload)
    )
    attestation: dict[str, Any] = {
        "schema_version": ALLOCATION_ATTESTATION_SCHEMA_V2,
        "payload": payload,
        "signature": {
            "algorithm": "Ed25519",
            "key_id": key_id,
            "signature_base64": base64.b64encode(signature).decode("ascii"),
        },
    }
    attestation["attestation_sha256"] = _canonical_sha256(attestation)
    return attestation


def validate_allocation_attestation_v2(
    attestation: Any,
    *,
    public_key_base64: str,
    expected_competition_id: str,
    expected_reservation_id: str,
    expected_launch_nonce: str,
    expected_prompt_sha256: str,
    expected_prompt_builder_sha256: str,
    expected_deployment_revision: str,
    expected_source_bundle_sha256: str,
    now_utc: str | None = None,
    generation_started_at: str | None = None,
    key_id: str = ALLOCATION_ATTESTATION_KEY_ID,
) -> dict[str, Any]:
    """Authenticate and bind a short-lived allocation attestation."""

    _lower_hex(
        expected_prompt_builder_sha256,
        "expected allocation prompt_builder_sha256",
    )
    _lower_hex(
        expected_source_bundle_sha256,
        "expected allocation source_bundle_sha256",
    )
    if not isinstance(expected_deployment_revision, str) or not expected_deployment_revision:
        raise LaunchGateError("expected allocation deployment revision is missing")
    if not isinstance(attestation, Mapping):
        raise LaunchGateError("allocation attestation is absent")
    normalized = dict(attestation)
    stored_hash = normalized.pop("attestation_sha256", None)
    if not isinstance(stored_hash, str) or stored_hash != _canonical_sha256(normalized):
        raise LaunchGateError("allocation attestation storage hash is invalid")
    if set(normalized) != {"schema_version", "payload", "signature"}:
        raise LaunchGateError("allocation attestation envelope fields are invalid")
    if normalized["schema_version"] != ALLOCATION_ATTESTATION_SCHEMA_V2:
        raise LaunchGateError("allocation attestation schema is invalid")
    payload = normalized.get("payload")
    signature = normalized.get("signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        raise LaunchGateError("allocation attestation payload or signature is absent")
    if signature.get("algorithm") != "Ed25519" or signature.get("key_id") != key_id:
        raise LaunchGateError("allocation attestation signing identity is invalid")
    signature_bytes = _decode_base64(
        signature.get("signature_base64"),
        "allocation attestation signature",
        expected_bytes=64,
    )
    public_key = _decode_base64(
        public_key_base64,
        "allocation attestation public key",
        expected_bytes=32,
    )
    _, Ed25519PublicKey, InvalidSignature = _ed25519_types()
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature_bytes,
            ALLOCATION_ATTESTATION_DOMAIN + _canonical_bytes(payload),
        )
    except InvalidSignature as exc:
        raise LaunchGateError("allocation attestation signature is invalid") from exc
    issued = _utc_timestamp(payload.get("issued_at_utc"), "allocation attestation issued_at_utc")
    expires = _utc_timestamp(payload.get("expires_at_utc"), "allocation attestation expires_at_utc")
    generation_started = (
        _utc_timestamp(generation_started_at, "allocation attestation generation_started_at")
        if generation_started_at is not None
        else None
    )
    now = (
        _utc_timestamp(now_utc, "allocation attestation now_utc")
        if now_utc is not None
        else datetime.now(timezone.utc)
    )
    if expires <= issued or (expires - issued).total_seconds() > (
        ALLOCATION_ATTESTATION_MAX_TTL_SECONDS
    ):
        raise LaunchGateError("allocation attestation TTL is invalid")
    if generation_started is not None:
        if generation_started < issued or generation_started >= expires:
            raise LaunchGateError("allocation attestation does not cover paid generation")
    elif issued > now + timedelta(seconds=ALLOCATION_ATTESTATION_MAX_FUTURE_SKEW_SECONDS):
        raise LaunchGateError("allocation attestation is not yet valid")
    elif now >= expires:
        raise LaunchGateError("allocation attestation is expired")
    expected_bindings = {
        "arm_id": ARM_ID,
        "competition_id": expected_competition_id,
        "reservation_id": expected_reservation_id,
        "launch_nonce": expected_launch_nonce,
        "launch_schema_version": "e9-merged-vllm-spawn-v3",
    }
    for field, expected in expected_bindings.items():
        if payload.get(field) != expected:
            raise LaunchGateError(f"allocation attestation {field} binding is invalid")
    prompt_contract = payload.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise LaunchGateError("allocation attestation prompt contract is absent")
    if (
        prompt_contract.get("template_version") != PROMPT_TEMPLATE_VERSION_V7
        or prompt_contract.get("prompt_sha256") != expected_prompt_sha256
        or prompt_contract.get("prompt_builder_sha256") != expected_prompt_builder_sha256
    ):
        raise LaunchGateError("allocation attestation v7 prompt binding is invalid")
    deployment = payload.get("deployment")
    if not isinstance(deployment, Mapping):
        raise LaunchGateError("allocation attestation deployment binding is absent")
    if (
        deployment.get("app_name") != "pavlov-e9-mle-bench-streaming"
        or deployment.get("deployment_revision") != expected_deployment_revision
        or deployment.get("source_bundle_sha256") != expected_source_bundle_sha256
        or deployment.get("run_function") != "run_one"
        or deployment.get("generator_function") != "E9MergedVllmGenerator.generate_program"
    ):
        raise LaunchGateError("allocation attestation deployment binding is invalid")
    reservation = payload.get("reservation")
    if not isinstance(reservation, Mapping) or payload.get("reservation_sha256") != (
        _canonical_sha256(reservation)
    ):
        raise LaunchGateError("allocation attestation reservation hash is invalid")
    if reservation.get("reservation_status") != "RESERVED_PRE_LAUNCH":
        raise LaunchGateError("allocation attestation reservation is invalid")
    wandb_receipt = payload.get("wandb_receipt")
    require_online_wandb_before_paid_load(
        wandb_receipt, paid_phase_started_at=payload.get("issued_at_utc")
    )
    if (
        wandb_receipt.get("server_confirmed") is not True
        or not isinstance(wandb_receipt.get("server_run_path"), str)
        or not wandb_receipt["server_run_path"]
    ):
        raise LaunchGateError("allocation attestation W&B binding is invalid")
    _validate_allocation_checkpoint(payload.get("checkpoint_verification"))
    return dict(attestation)


def validate_v3_generation_binding(
    generation: Any,
    *,
    competition_id: str,
    prompt: str,
    allocation_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Require returned generation bytes to bind to v7 and the signed launch."""

    if not isinstance(generation, Mapping):
        raise LaunchGateError("v3 generation receipt is absent")
    if generation.get("generation_mode") != "fresh_merged_vllm_generation":
        raise LaunchGateError("v3 requires fresh merged-vLLM generation")
    if generation.get("competition_id") != competition_id:
        raise LaunchGateError("v3 generation competition binding is invalid")
    if generation.get("prompt_template_version") != PROMPT_TEMPLATE_VERSION_V7:
        raise LaunchGateError("v3 generation does not use the mandatory v7 prompt")
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if generation.get("prompt_sha256") != prompt_sha256:
        raise LaunchGateError("v3 generation prompt hash is invalid")
    response_text = generation.get("response_text")
    if not isinstance(response_text, str) or generation.get("response_sha256") != (
        hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    ):
        raise LaunchGateError("v3 generation response hash is invalid")
    from e9_mle_bench_streaming import extract_python_code

    extracted_program = extract_python_code(response_text)
    program = generation.get("program")
    if not isinstance(program, str) or generation.get("program_sha256") != (
        hashlib.sha256((program + "\n").encode("utf-8")).hexdigest()
    ):
        raise LaunchGateError("v3 generation program hash is invalid")
    if program != extracted_program:
        raise LaunchGateError("v3 generation program does not match the sampled response")
    if generation.get("allocation_attestation_sha256") != allocation_attestation.get(
        "attestation_sha256"
    ):
        raise LaunchGateError("v3 generation allocation attestation binding is invalid")
    return dict(generation)


def _sanitize_failure_message(message: str) -> tuple[str, bool]:
    sanitized = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        message,
    )
    sanitized = re.sub(
        r"""(?i)(["']?(?:api[_-]?key|access[_-]?token|token|secret|password)["']?\s*[:=]\s*["']?)(\[REDACTED\]|[^"'\s,;}\]]+)""",
        lambda match: (
            match.group(0) if match.group(2) == "[REDACTED]" else f"{match.group(1)}[REDACTED]"
        ),
        sanitized,
    )
    sanitized = re.sub(
        r"""(?<![A-Za-z0-9])/(?:Users|home|root|tmp|private)(?:/[^\s,;:"'\]\)}]+)*""",
        "[REDACTED_PATH]",
        sanitized,
    )
    return sanitized, sanitized != message


def build_cpu_preparation_failure_receipt_v1(
    *,
    competition_id: str,
    reservation_id: str,
    launch_nonce: str,
    reservation: Mapping[str, Any],
    prepare_elapsed_seconds: Any,
    estimated_cpu_usd: Any,
    error_type: str,
    error_message: str,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Record bounded CPU preparation failure without claiming model evidence."""

    if not isinstance(competition_id, str) or not competition_id:
        raise LaunchGateError("CPU preparation failure competition_id is missing")
    _lower_hex(reservation_id, "CPU preparation failure reservation_id", length=32)
    _lower_hex(launch_nonce, "CPU preparation failure launch_nonce")
    _utc_timestamp(recorded_at_utc, "CPU preparation failure recorded_at_utc")
    try:
        elapsed = float(prepare_elapsed_seconds)
    except (TypeError, ValueError) as exc:
        raise LaunchGateError("CPU preparation elapsed time is invalid") from exc
    if elapsed < 0:
        raise LaunchGateError("CPU preparation elapsed time is negative")
    cpu_cost = _money(estimated_cpu_usd, "CPU preparation cost")
    reserved_cpu = _money(
        reservation.get("projected_modal_cpu_usd"), "reserved CPU preparation cost"
    )
    if cpu_cost > reserved_cpu:
        raise LaunchGateError("CPU preparation cost exceeds its reservation")
    if (
        not isinstance(error_type, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", error_type) is None
    ):
        raise LaunchGateError("CPU preparation error type is invalid")
    if not isinstance(error_message, str) or not error_message:
        raise LaunchGateError("CPU preparation error message is missing")
    sanitized_error, redaction_applied = _sanitize_failure_message(error_message)
    error_tail = sanitized_error[-3000:]
    receipt: dict[str, Any] = {
        "schema_version": CPU_PREPARATION_FAILURE_SCHEMA_V1,
        "status": "CPU_PREPARATION_FAILED_BEFORE_GPU_GENERATION",
        "scientific_status": "NON_SCORE_EXECUTION_CONTROL_EVIDENCE",
        "arm_id": ARM_ID,
        "competition_id": competition_id,
        "reservation_id": reservation_id,
        "launch_nonce": launch_nonce,
        "recorded_at_utc": recorded_at_utc,
        "phase": CPU_PREPARATION_FAILURE_PHASE,
        "prepare_elapsed_seconds": round(elapsed, 3),
        "failure": {
            "error_type": error_type,
            "error_message_tail": error_tail,
            "error_message_sha256": hashlib.sha256(error_message.encode("utf-8")).hexdigest(),
            "redaction_applied": redaction_applied,
        },
        "combined_gpu_cpu_reservation": dict(reservation),
        "gpu_generation_started": False,
        "sample_reused": False,
        "competition_score": None,
        "score": None,
        "is_full_suite_score": False,
        "legacy_tinker_coverage_increment": 0,
        "merged_vllm_budget": {
            "mode": "cpu_preparation_failure_before_gpu_generation",
            "estimated_modal_gpu_usd": 0.0,
            "estimated_modal_cpu_usd": float(cpu_cost),
            "estimated_combined_usd": float(cpu_cost),
            "reservation": dict(reservation),
        },
        "claim_boundary": CPU_PREPARATION_FAILURE_CLAIM_BOUNDARY,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


class E9MergedVllmProgramGenerator:
    """Fresh E9-program generator once caller has completed the paid-load gate.

    ``sample`` is intentionally a narrow callback around vLLM's generate call.
    It receives the exact E9 prompt and must return the newly sampled response;
    accepting a solution override would make this arm indistinguishable from a
    replay and is therefore prohibited.
    """

    def __init__(
        self,
        sample: Callable[[str], str],
        *,
        wandb_receipt: Any,
        gpu_load_started_at: Any,
    ) -> None:
        self._sample = sample
        self._wandb_receipt = require_online_wandb_before_paid_load(
            wandb_receipt, paid_phase_started_at=gpu_load_started_at
        )

    def generate_program(self, prompt: str, *, generation_started_at: Any) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LaunchGateError("E9 generation prompt is absent")
        require_online_wandb_before_paid_load(
            self._wandb_receipt, paid_phase_started_at=generation_started_at
        )
        response_text = self._sample(prompt)
        if not isinstance(response_text, str):
            raise LaunchGateError("merged-vLLM sampler returned a non-text response")
        program = extract_python_code(response_text)
        return {
            "generation_mode": "fresh_merged_vllm_generation",
            "response_text": response_text,
            "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "program": program,
            "program_sha256": hashlib.sha256((program + "\n").encode("utf-8")).hexdigest(),
            "wandb_receipt": self._wandb_receipt,
            "generation_started_at": generation_started_at,
        }


def build_task_receipt(
    *,
    competition_id: str,
    native_grade: Mapping[str, Any],
    generation: Mapping[str, Any],
    checkpoint_verification: Mapping[str, Any],
    reservation: Mapping[str, Any],
    allocation_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a per-task receipt that cannot be mistaken for legacy E9 evidence."""

    if not competition_id:
        raise LaunchGateError("competition_id is required")
    if native_grade.get("competition_id") != competition_id:
        raise LaunchGateError("native grade competition_id does not match the arm task")
    if generation.get("generation_mode") != "fresh_merged_vllm_generation":
        raise LaunchGateError("E9 merged-vLLM arm requires a fresh generation")
    require_online_wandb_before_paid_load(
        generation.get("wandb_receipt"),
        paid_phase_started_at=generation.get("generation_started_at"),
    )
    if checkpoint_verification.get("verified_shard_count") != EXPECTED_SHARD_COUNT:
        raise LaunchGateError("all 26 merged checkpoint shards must be verified")
    if reservation.get("reservation_status") != "RESERVED_PRE_LAUNCH":
        raise LaunchGateError("combined GPU+CPU reservation is absent")
    if _money(reservation.get("projected_combined_usd"), "combined reservation") > PILOT_CAP_USD:
        raise LaunchGateError("combined GPU+CPU reservation exceeds the pilot cap")
    validated_gate = validate_gpu_allocation_gate(allocation_gate)
    gate_checkpoint = validated_gate["checkpoint_verification"]
    if any(
        dict(checkpoint_verification).get(field) != value
        for field, value in gate_checkpoint.items()
    ):
        raise LaunchGateError(
            "task receipt checkpoint verification differs from GPU allocation gate"
        )
    score = native_grade.get("score")
    valid = score is not None and native_grade.get("valid_submission") is True
    receipt: dict[str, Any] = {
        "schema_version": ARM_SCHEMA_V2,
        "arm_id": ARM_ID,
        "competition_id": competition_id,
        "status": "NATIVE_SINGLE_COMPETITION_GRADED"
        if valid
        else "NATIVE_SINGLE_COMPETITION_INVALID",
        "competition_score": score if valid else None,
        "native_grade": dict(native_grade),
        "allocation_gate": validated_gate,
        "generation": dict(generation),
        "checkpoint_verification": dict(checkpoint_verification),
        "combined_gpu_cpu_reservation": dict(reservation),
        "is_full_suite_score": False,
        "score": None,
        "legacy_tinker_coverage_increment": LEGACY_TINKER_COVERAGE_INCREMENT,
        "claim_boundary": (
            "One fresh merged-vLLM E9 competition result in a separate arm; it is "
            "not a 75-competition suite score and is never unioned with legacy "
            "Tinker-sampled E9 coverage."
        ),
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def build_task_receipt_v3(
    *,
    competition_id: str,
    native_grade: Mapping[str, Any],
    generation: Mapping[str, Any],
    checkpoint_verification: Mapping[str, Any],
    reservation: Mapping[str, Any],
    allocation_gate: Mapping[str, Any],
    allocation_attestation: Mapping[str, Any],
    allocation_attestation_public_key_base64: str,
    reservation_id: str,
    launch_nonce: str,
    prompt: str,
    prompt_builder_sha256: str,
    deployment_revision: str,
    source_bundle_sha256: str,
) -> dict[str, Any]:
    """Build a fully bound v3 native result after an authenticated generation."""

    generation_started_at = generation.get("generation_started_at")
    attestation = validate_allocation_attestation_v2(
        allocation_attestation,
        public_key_base64=allocation_attestation_public_key_base64,
        expected_competition_id=competition_id,
        expected_reservation_id=reservation_id,
        expected_launch_nonce=launch_nonce,
        expected_prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        expected_prompt_builder_sha256=prompt_builder_sha256,
        expected_deployment_revision=deployment_revision,
        expected_source_bundle_sha256=source_bundle_sha256,
        generation_started_at=generation_started_at,
    )
    validate_v3_generation_binding(
        generation,
        competition_id=competition_id,
        prompt=prompt,
        allocation_attestation=attestation,
    )
    receipt = build_task_receipt(
        competition_id=competition_id,
        native_grade=native_grade,
        generation=generation,
        checkpoint_verification=checkpoint_verification,
        reservation=reservation,
        allocation_gate=allocation_gate,
    )
    receipt.update(
        {
            "schema_version": ARM_SCHEMA_V3,
            "reservation_id": reservation_id,
            "launch_nonce": launch_nonce,
            "allocation_attestation": dict(attestation),
        }
    )
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt
