#!/usr/bin/env python3
"""Offline metadata-first boundary checker for ``verilog_eval`` primary evaluation.

The checker validates immutable identifiers, task-id determinism, split integrity,
verifier/environment contracts, and mandatory run/checkpoint receipts.
It never contacts networked services, never reads credentials, and never enables
paid launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pavlov-verilog-eval-adapter-v2"
SUITE_ID = "verilog_eval"
SUITE_ROLE = "primary_eval"
_EXPECTED_CATEGORY = "code"
_EXPECTED_STATEFUL = False
_EXPECTED_ARTIFACT_OR_SIDE_EFFECT = True

_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_URL_RE = re.compile(r"^https://[^\s]+$")

_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "todo",
    "tbd",
    "pending",
    "placeholder",
    "unrecorded",
    "not_provided",
    "to_be_pinned",
    "to_be_pinned_before_paid_runs",
    "latest",
    "main",
    "head",
    "master",
    "tip",
}

_BANNED_MARKERS = {
    "glaive",
    "glaiveai/glaive-function-calling-v2",
    "simulated",
    "simulator",
    "xlam",
    "salesforce/xlam-function-calling-60k",
    "tool_use_tinker.py",
    "bfclv4_tool_use.py",
}

_ALLOWED_TINKER_COST_STATUS = frozenset({
    "authorized",
    "approved",
    "observed",
    "within_cap",
    "complete",
    "zero_cost",
})

_REQUIRED_WANDB_FIELDS = ("entity", "project", "group", "run_id", "run_url")
_REQUIRED_HF_STAGES = ("initial", "periodic", "final")

_CREDENTIAL_KEYS = (
    "credential_ref",
    "api_key",
    "api_token",
    "hf_token",
    "hf_api_token",
    "openai_api_key",
    "openrouter_api_key",
    "secret_access_key",
    "wandb_api_key",
    "oauth_token",
    "access_token",
)


class VerilogEvalBoundaryError(ValueError):
    """Raised for malformed/verbatim invalid verilog_eval boundary metadata."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if not isinstance(value, str):
        return True
    return value.strip().lower() in _PLACEHOLDERS


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or _is_placeholder(value):
        raise VerilogEvalBoundaryError(f"{field} must be non-placeholder text")
    return value.strip()


def _require_status_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerilogEvalBoundaryError(f"{field} must be explicit non-empty text")
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in (_PLACEHOLDERS - {"pending"}):
        raise VerilogEvalBoundaryError(f"{field} must be explicit non-empty text")
    return normalized


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise VerilogEvalBoundaryError(f"{field} must be explicit boolean")
    return value


def _require_hex40(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not _HEX40_RE.fullmatch(text):
        raise VerilogEvalBoundaryError(f"{field} must be immutable 40-char hex")
    return text


def _require_hex64(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not _HEX64_RE.fullmatch(text):
        raise VerilogEvalBoundaryError(f"{field} must be 64-char hex")
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not _SHA256_RE.fullmatch(text):
        raise VerilogEvalBoundaryError(f"{field} must be a sha256 digest")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _require_url(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not _URL_RE.fullmatch(text):
        raise VerilogEvalBoundaryError(f"{field} must be an https URL")
    return text


def _contains_banned_markers(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_banned_markers(item) for item in value.values() if item is not None)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str, bytearray)):
        return any(_contains_banned_markers(item) for item in value)
    return False


def _load_authority(contract_path: str | Path | None = None) -> Mapping[str, Any]:
    base = Path(__file__).resolve().parent
    path = Path(contract_path) if contract_path is not None else base / "pavlovs_domain_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerilogEvalBoundaryError(f"cannot read authority contract: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerilogEvalBoundaryError(f"authority contract is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise VerilogEvalBoundaryError("contract must be a JSON object")
    registry = payload.get("suite_registry")
    if not isinstance(registry, Mapping):
        raise VerilogEvalBoundaryError("contract missing suite_registry")
    suite = registry.get(SUITE_ID)
    if not isinstance(suite, Mapping):
        raise VerilogEvalBoundaryError("contract missing verilog_eval entry")
    return suite


def _validate_source(source_value: Any, authority: Mapping[str, Any]) -> str:
    source = _require_url(source_value, "source")
    authoritative = _require_url(
        authority.get("url"), "contract suite url"
    ).rstrip("/").lower()
    if source.rstrip("/").lower() != authoritative:
        raise VerilogEvalBoundaryError("source must match authoritative verilog_eval source")
    return source


def _validate_dataset(dataset: Any, authority: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(dataset, Mapping):
        raise VerilogEvalBoundaryError("dataset must be an object")

    revision = _require_hex40(dataset.get("revision"), "dataset.revision")
    license_text = _require_text(dataset.get("license"), "dataset.license")
    source = _require_url(dataset.get("source"), "dataset.source")
    expected_source = _require_url(authority.get("url"), "contract suite url").rstrip("/").lower()
    if source.rstrip("/").lower() != expected_source:
        raise VerilogEvalBoundaryError("dataset.source must match authoritative verilog_eval source")

    if _contains_banned_markers(dataset.get("name")):
        raise VerilogEvalBoundaryError("dataset references blocked xLAM/Glaive source")
    return revision, license_text, source


def _validate_category(value: Any) -> str:
    category = _require_text(value, "category")
    if category != _EXPECTED_CATEGORY:
        raise VerilogEvalBoundaryError(
            f"category must be {_EXPECTED_CATEGORY!r}, found {category!r}"
        )
    return category


def _validate_role(value: Any) -> str:
    role = _require_text(value, "role")
    if role != SUITE_ROLE:
        raise VerilogEvalBoundaryError(f"role must be {SUITE_ROLE!r}, found {role!r}")
    return role


def _validate_verifier(verifier: Any) -> dict[str, Any]:
    if not isinstance(verifier, Mapping):
        raise VerilogEvalBoundaryError("verifier must be an object")
    identity = _require_text(verifier.get("identity"), "verifier.identity")
    if _contains_banned_markers(identity):
        raise VerilogEvalBoundaryError("verifier references blocked benchmark substitution")

    verifier_hash = _require_hex64(
        verifier.get("hash") if verifier.get("hash") is not None else verifier.get("digest"),
        "verifier.hash",
    )
    return {"identity": identity, "hash": verifier_hash}


def _validate_environment(environment: Any) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        raise VerilogEvalBoundaryError("environment must be an object")

    container = _require_text(environment.get("container"), "environment.container")
    image = _require_text(environment.get("image"), "environment.image")
    container_digest = _require_sha256(
        environment.get("container_digest"), "environment.container_digest"
    )
    runtime_digest = _require_sha256(
        environment.get("runtime_digest"), "environment.runtime_digest"
    )

    if _contains_banned_markers(container) or _contains_banned_markers(image):
        raise VerilogEvalBoundaryError("environment references blocked substitution markers")
    return {
        "container": container,
        "image": image,
        "container_digest": container_digest,
        "runtime_digest": runtime_digest,
    }


def _validate_task_id_hashes(value: Any, field: str = "task_id_hashes") -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VerilogEvalBoundaryError(f"{field} must be a non-empty list")
    ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_hash = _require_hex64(item, f"{field}[{index}]")
        if item_hash in seen:
            raise VerilogEvalBoundaryError(f"{field} contains duplicate IDs")
        seen.add(item_hash)
        ids.append(item_hash)

    if not ids:
        raise VerilogEvalBoundaryError(f"{field} cannot be empty")
    if ids != sorted(ids):
        raise VerilogEvalBoundaryError(f"{field} must be deterministically sorted")
    return ids


def _validate_split(value: Any, task_ids: list[str]) -> tuple[list[str], str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalBoundaryError("split must be an object")

    split_ids = value.get("primary_eval")
    if split_ids is None:
        raise VerilogEvalBoundaryError("split.primary_eval is required")

    split_hash_value = _require_sha256(
        value.get("hash") if value.get("hash") is not None else value.get("sha256"),
        "split.primary_eval.hash",
    )

    split_task_ids = _validate_task_id_hashes(split_ids, "split.primary_eval")
    if split_task_ids != task_ids:
        raise VerilogEvalBoundaryError("split.primary_eval must match task_id_hashes")

    aggregate = _sha256("\n".join(task_ids))
    if split_hash_value != f"sha256:{aggregate}":
        raise VerilogEvalBoundaryError("split.primary_eval aggregate hash does not match tasks")
    return split_task_ids, aggregate, f"sha256:{aggregate}"


def _validate_split_hashes(value: Mapping[str, Any], aggregate: str) -> dict[str, str]:
    split_hashes = value.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise VerilogEvalBoundaryError("split_hashes must be an object")

    declared = split_hashes.get("primary_eval")
    if declared is None:
        raise VerilogEvalBoundaryError("split_hashes.primary_eval is required")
    observed = _require_sha256(declared, "split_hashes.primary_eval")
    expected = f"sha256:{aggregate}"
    if observed != expected:
        raise VerilogEvalBoundaryError("split_hashes.primary_eval does not match task aggregate")
    return {"primary_eval": observed}


def _validate_split_manifest_hash(
    manifest: Mapping[str, Any],
    task_aggregate: str,
) -> tuple[str, str]:
    payload = canonical_json({"primary_eval": task_aggregate})
    observed = f"sha256:{_sha256(payload)}"

    declared = _require_sha256(
        manifest.get("split_manifest_hash"),
        "split_manifest_hash",
    )
    if declared != observed:
        raise VerilogEvalBoundaryError("split_manifest_hash does not match observed split")

    receipt_ref = _require_sha256(
        manifest.get("split_manifest_receipt_ref"),
        "split_manifest_receipt_ref",
    )
    return observed, receipt_ref


def _validate_scope(manifest: Mapping[str, Any], *, task_aggregate: str) -> dict[str, bool]:
    if not isinstance(manifest.get("scope"), Mapping):
        raise VerilogEvalBoundaryError("scope must be an object")

    scope = manifest["scope"]
    assert isinstance(scope, Mapping)
    is_portfolio = _require_bool(scope.get("is_portfolio"), "scope.is_portfolio")
    if is_portfolio:
        raise VerilogEvalBoundaryError("scope.is_portfolio must be false for this boundary")

    is_held_out = _require_bool(scope.get("is_held_out"), "scope.is_held_out")
    held_out = _require_bool(manifest.get("held_out"), "held_out")
    if held_out != is_held_out:
        raise VerilogEvalBoundaryError("scope.is_held_out must match held_out")

    held_out_receipt = manifest.get("held_out_receipt_ref")
    if held_out:
        _require_sha256(held_out_receipt, "held_out_receipt_ref")
    elif held_out_receipt is not None and not _is_placeholder(held_out_receipt):
        raise VerilogEvalBoundaryError("held_out_receipt_ref is forbidden when held_out is false")

    return {
        "is_portfolio": is_portfolio,
        "is_held_out": is_held_out,
        "held_out": held_out,
        "task_aggregate": task_aggregate,
    }


def _validate_wandb(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise VerilogEvalBoundaryError("wandb_run_identity must be an object")
    if raw.get("online") is not True:
        raise VerilogEvalBoundaryError("wandb_run_identity.online must be true")

    normalized: dict[str, str] = {}
    for field in _REQUIRED_WANDB_FIELDS:
        normalized[field] = _require_text(raw.get(field), f"wandb_run_identity.{field}")
        if field == "run_url":
            _require_url(normalized[field], f"wandb_run_identity.{field}")
    if "wandb.ai" not in normalized["run_url"]:
        raise VerilogEvalBoundaryError("wandb_run_identity.run_url must be a W&B run URL")
    return normalized


def _validate_tinker(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise VerilogEvalBoundaryError("tinker_run_identity must be an object")
    run_id = _require_text(raw.get("run_id"), "tinker_run_identity.run_id")
    cost_status = _require_status_text(
        raw.get("cost_status"), "tinker_run_identity.cost_status"
    ).lower()
    if cost_status not in _ALLOWED_TINKER_COST_STATUS:
        raise VerilogEvalBoundaryError("tinker_run_identity.cost_status is invalid")
    return {"run_id": run_id, "cost_status": cost_status}


def _validate_hf_checkpoints(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise VerilogEvalBoundaryError("hf_checkpoints must be a list")
    if not raw:
        raise VerilogEvalBoundaryError("hf_checkpoints cannot be empty")

    observed_stages: set[str] = set()
    seen_payload: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []

    for index, checkpoint in enumerate(raw):
        if not isinstance(checkpoint, Mapping):
            raise VerilogEvalBoundaryError(
                f"hf_checkpoints[{index}] must be an object"
            )
        repo_url = _require_url(checkpoint.get("repo_url"), f"hf_checkpoints[{index}].repo_url")
        revision = _require_hex40(checkpoint.get("revision"), f"hf_checkpoints[{index}].revision")
        stage = _require_text(checkpoint.get("stage"), f"hf_checkpoints[{index}].stage").lower()
        if stage not in _REQUIRED_HF_STAGES:
            raise VerilogEvalBoundaryError("hf_checkpoints stage must be one of initial, periodic, final")
        if stage in observed_stages:
            raise VerilogEvalBoundaryError(
                f"hf_checkpoints contains duplicate stage {stage!r}"
            )
        observed_stages.add(stage)

        url = _require_url(checkpoint.get("url"), f"hf_checkpoints[{index}].url")
        if checkpoint.get("safe_public_artifact") is not True:
            raise VerilogEvalBoundaryError(
                f"hf_checkpoints[{index}].safe_public_artifact must be true"
            )
        visibility = _require_text(checkpoint.get("visibility"), f"hf_checkpoints[{index}].visibility").lower()
        if visibility not in {"public", "private"}:
            raise VerilogEvalBoundaryError(
                f"hf_checkpoints[{index}].visibility must be public or private"
            )
        artifact_key = (repo_url, revision)
        if artifact_key in seen_payload:
            raise VerilogEvalBoundaryError(
                f"hf_checkpoints[{index}] duplicates repo/revision pair"
            )
        seen_payload.add(artifact_key)

        normalized.append(
            {
                "repo_url": repo_url,
                "revision": revision,
                "url": url,
                "stage": stage,
                "safe_public_artifact": True,
                "visibility": visibility,
            }
        )

    if observed_stages != set(_REQUIRED_HF_STAGES):
        missing = ", ".join(sorted(set(_REQUIRED_HF_STAGES) - observed_stages))
        raise VerilogEvalBoundaryError(
            f"hf_checkpoints missing required stage(s): {missing}"
        )
    return normalized


def _validate_costs(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise VerilogEvalBoundaryError("costs must be an object")

    status = _require_status_text(raw.get("status"), "costs.status").lower()
    if status not in {"authorized", "observed", "complete", "within_cap", "zero_cost"}:
        raise VerilogEvalBoundaryError("costs.status is invalid")

    total_usd = raw.get("total_usd")
    if not isinstance(total_usd, (int, float)) or isinstance(total_usd, bool):
        raise VerilogEvalBoundaryError("costs.total_usd must be a number")
    if total_usd < 0:
        raise VerilogEvalBoundaryError("costs.total_usd cannot be negative")
    if raw.get("paid_work") is True:
        raise VerilogEvalBoundaryError("costs.paid_work is not allowed for offline boundary")
    return {"status": status, "total_usd": float(total_usd)}


def _validate_no_network_or_credentials(manifest: Mapping[str, Any], blockers: list[str]) -> None:
    if manifest.get("requires_network") is True:
        blockers.append("dataset download/network is disallowed for verifier boundary")
    if manifest.get("paid_launch_allowed") is True:
        blockers.append("paid_launch_allowed cannot be true in offline verifier boundary")

    for key in _CREDENTIAL_KEYS:
        value = manifest.get(key)
        if value is not None and not _is_placeholder(value):
            blockers.append(f"{key} is not allowed in offline verifier boundary")


def build_verilog_eval_boundary_record(
    manifest: Mapping[str, Any],
    *,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return deterministic boundary decision for a single verilog_eval receipt."""

    blockers: list[str] = []
    try:
        if not isinstance(manifest, Mapping):
            raise VerilogEvalBoundaryError("manifest must be a JSON object")

        authority = _load_authority(contract_path)
        if str(authority.get("role", "")).strip().lower() != SUITE_ROLE:
            blockers.append("contract role mismatch for verilog_eval")
        if authority.get("stateful") is not _EXPECTED_STATEFUL:
            blockers.append("contract stateful value changed from expected")
        if authority.get("artifact_or_side_effect") is not _EXPECTED_ARTIFACT_OR_SIDE_EFFECT:
            blockers.append(
                "contract artifact_or_side_effect value changed from expected"
            )

        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            blockers.append(f"suite_id must be {SUITE_ID!r}")

        source = _validate_source(manifest.get("source"), authority)
        category = _validate_category(manifest.get("category"))
        role = _validate_role(manifest.get("role"))
        dataset_revision, dataset_license, dataset_source = _validate_dataset(
            manifest.get("dataset"), authority
        )
        verifier = _validate_verifier(manifest.get("verifier"))
        environment = _validate_environment(manifest.get("environment"))

        task_ids = _validate_task_id_hashes(manifest.get("task_id_hashes"), "task_id_hashes")
        split_task_ids, split_aggregate, split_aggregate_digest = _validate_split(
            manifest.get("split"), task_ids
        )
        split_hashes = _validate_split_hashes(manifest, split_aggregate)
        split_manifest_hash, split_manifest_receipt_ref = _validate_split_manifest_hash(
            manifest,
            split_aggregate,
        )
        scope = _validate_scope(
            manifest,
            task_aggregate=split_aggregate,
        )
        held_out = scope["is_held_out"]

        wandb = _validate_wandb(
            manifest.get("wandb_run_identity")
            if isinstance(manifest.get("wandb_run_identity"), Mapping)
            else manifest.get("wandb")
        )
        tinker = _validate_tinker(
            manifest.get("tinker_run_identity")
            if isinstance(manifest.get("tinker_run_identity"), Mapping)
            else manifest.get("tinker")
        )
        hf_checkpoints = _validate_hf_checkpoints(manifest.get("hf_checkpoints"))
        costs = _validate_costs(manifest.get("costs"))

        evidence = {
            "dataset_source": dataset_source,
            "verifier_identity": verifier["identity"],
            "environment": environment,
            "split": split_task_ids,
            "split_hashes": split_hashes,
        }
        if _contains_banned_markers(evidence):
            blockers.append("manifest evidence references blocked substitution markers")

        _validate_no_network_or_credentials(manifest, blockers)

        blockers = sorted(set(blockers))

        task_id_digest = f"sha256:{_sha256(canonical_json(task_ids))}"
        split_payload = {
            "suite_id": SUITE_ID,
            "task_id_hashes": task_ids,
            "split_manifest_hash": split_manifest_hash,
            "verifier_hash": verifier["hash"],
            "environment": environment,
        }
        split_payload_digest = f"sha256:{_sha256(canonical_json(split_payload))}"

        # Keep an explicit adapter-oriented flag for downstream boundary coordinators.
        ready = not blockers
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": suite_id,
            "exact_suite": suite_id == SUITE_ID,
            "adapter_ready": ready,
            "eval_ready": ready,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "authoritative_source": source,
            "contract_authority": {
                "source": authority.get("url"),
                "role": authority.get("role"),
                "domains": list(authority.get("domains", [])),
                "stateful": authority.get("stateful"),
                "artifact_or_side_effect": authority.get("artifact_or_side_effect"),
            },
            "category": category,
            "role": role,
            "dataset_revision": dataset_revision,
            "dataset_license": dataset_license,
            "dataset_source": dataset_source,
            "verifier": verifier,
            "environment": environment,
            "split": {
                "primary_eval": {
                    "count": len(split_task_ids),
                    "task_id_hashes": split_task_ids,
                    "aggregate_sha256": split_aggregate,
                    "aggregate_sha256_digest": split_aggregate_digest,
                }
            },
            "split_hashes": split_hashes,
            "split_manifest_hash": split_manifest_hash,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": split_manifest_receipt_ref,
            "scope": scope,
            "held_out": held_out,
            "held_out_receipt_ref": manifest.get("held_out_receipt_ref"),
            "held_out_receipt_proven": bool(held_out),
            "wandb": wandb,
            "tinker": tinker,
            "hf_checkpoints": hf_checkpoints,
            "costs": costs,
            "task_id_digest": task_id_digest,
            "authority_stateful": bool(authority.get("stateful", False)),
            "authority_split_manifest_digest": split_payload_digest,
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }
    except VerilogEvalBoundaryError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "suite_id": manifest.get("suite_id", "<missing>"),
            "exact_suite": manifest.get("suite_id") == SUITE_ID,
            "adapter_ready": False,
            "eval_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "authoritative_source": None,
            "contract_authority": {},
            "category": manifest.get("category"),
            "role": manifest.get("role", "<missing>"),
            "dataset_revision": None,
            "dataset_license": None,
            "dataset_source": None,
            "verifier": None,
            "environment": None,
            "split": {"primary_eval": {}},
            "split_hashes": {},
            "split_manifest_hash": manifest.get("split_manifest_hash"),
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": manifest.get("split_manifest_receipt_ref"),
            "scope": manifest.get("scope") if isinstance(manifest.get("scope"), Mapping) else None,
            "held_out": bool(manifest.get("held_out", False)) if isinstance(manifest, Mapping) else False,
            "held_out_receipt_ref": manifest.get("held_out_receipt_ref") if isinstance(manifest, Mapping) else None,
            "held_out_receipt_proven": False,
            "wandb": None,
            "tinker": None,
            "hf_checkpoints": [],
            "costs": None,
            "task_id_digest": None,
            "authority_stateful": None,
            "authority_split_manifest_digest": None,
            "blockers": [str(exc)],
            "status": "BLOCKED",
        }


def validate_verilog_eval_boundary_record(record: Mapping[str, Any]) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record root must be a JSON object"]
    ready = bool(record.get("eval_ready", False) or record.get("adapter_ready", False))
    if record.get("status") == "READY" and not ready:
        return ["record status is READY but eval_ready is false"]
    return [] if record.get("status") == "READY" else list(record.get("blockers", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SystemExit("manifest must be a JSON object")

    report = build_verilog_eval_boundary_record(payload)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
