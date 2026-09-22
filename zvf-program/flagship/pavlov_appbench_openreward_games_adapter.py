"""Offline validator for the E12/E13 Pavlov evaluation boundary adapter.

The validator is deliberately strict and pure:
- requires exactly the E12 and E13 boundaries in the expected order and naming
- binds each boundary to an authoritative source and immutable revision
- validates deterministic task IDs and split hashes for each boundary
- requires native artifact/environment/verifier receipts for each boundary
- requires explicit stateful trajectory and disallows paid launch flags
- rejects benchmark substitution toward xLAM or related sources
- requires W&B, Tinker, and Hugging Face receipt fields
"""

from __future__ import annotations

import hashlib
import json
import re
import argparse
import os
import base64
import math
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pathlib import Path
from typing import Any, Mapping


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[0-9A-Za-z]{8}$")
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REPO = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

PRIMARY_EVAL = "primary_eval"
RECEIPT_PROVEN_HELDOUT = "receipt_proven_heldout"
APPBENCH_GRANT_SCHEMA = "appbench-provider-grant-v1"
APPBENCH_HANDOFF_SCHEMA = "appbench-provider-handoff-v1"
APPBENCH_RESULT_SCHEMA = "appbench-provider-result-v1"
APPBENCH_TRUST_ROOT_SCHEMA = "provider-lane-trust-root-v1"
APPBENCH_EXPECTED_HELDOUT_TASK_COUNT = 6

BANNED_SOURCE_MARKERS = {
    "xlam",
    "xlama",
    "related benchmark",
    "related-benchmark",
    "related_benchmark",
}

BOUNDARY_SPECS: dict[str, dict[str, Any]] = {
    "E12": {
        "expected_name": "appbench_eval",
        "expected_role": RECEIPT_PROVEN_HELDOUT,
        "required_source_marker": "appbench",
    },
    "E13": {
        "expected_name": "openreward_games_eval",
        "expected_role": PRIMARY_EVAL,
        "required_source_marker": "openreward",
    },
}


class PavlovAppbenchOpenrewardGamesAdapterError(ValueError):
    """Raised when a Pavlov adapter contract payload is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PavlovAppbenchOpenrewardGamesAdapterError(message)


def _require_mapping(payload: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), f"{label} must be a mapping")
    return payload


def _require_text(value: Any, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    _require(value.strip(), f"{label} cannot be empty")
    return value


def _require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    hex_value = _require_text(value, label)
    _require(
        pattern.fullmatch(hex_value) is not None,
        f"{label} must be a hex digest of length {40 if pattern is HEX40 else 64}",
    )
    _require(
        len(set(hex_value)) > 1,
        f"{label} must not be an all-identical placeholder digest",
    )
    return hex_value


def _require_path(value: Any, label: str) -> str:
    path = _require_text(value, label)
    _require("\n" not in path and "\r" not in path, f"{label} must be a valid path-like string")
    return path


def _validate_task_ids(value: Any, label: str) -> list[str]:
    _require(isinstance(value, list), f"{label}.task_ids must be a list")
    _require(len(value) > 0, f"{label}.task_ids cannot be empty")
    task_ids: list[str] = []
    for index, task_id in enumerate(value):
        task_ids.append(_require_hex(task_id, HEX64, f"{label}.task_ids[{index}]"))
    _require(len(task_ids) == len(set(task_ids)), f"{label}.task_ids must be unique")
    return task_ids


def _deterministic_split_hash(task_ids: list[str]) -> str:
    payload = json.dumps(sorted(task_ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_split_hash(value: Any, task_ids: list[str], label: str) -> str:
    split_hash = _require_hex(value, HEX64, f"{label}.split_hash")
    expected = _deterministic_split_hash(task_ids)
    _require(
        split_hash == expected,
        f"{label}.split_hash is not the deterministic hash of task_ids",
    )
    return split_hash


def _fold_source(text: str) -> str:
    """Lowercase and drop every non-alphanumeric character.

    Upstream identifiers are punctuated inconsistently: the E12 dataset is
    ``AfterQuery/App-Bench`` on the Hub and ``app-bench`` in the leaderboard URL,
    while the marker is spelled ``appbench``. Folding punctuation lets a receipt
    cite the real source URL verbatim. It also strengthens the substitution ban,
    since ``x-lam`` and ``related_benchmark`` fold onto their banned forms.
    """

    return "".join(ch for ch in text.lower() if ch.isalnum())


def _validate_source(value: Any, label: str, required_marker: str) -> str:
    source = _require_text(value, label)
    folded = _fold_source(source)
    for marker in BANNED_SOURCE_MARKERS:
        _require(
            _fold_source(marker) not in folded,
            f"{label} references blocked source marker: {marker}",
        )
    _require(
        _fold_source(required_marker) in folded,
        f"{label} must identify the authoritative source {required_marker!r}",
    )
    return source


def _validate_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    sha256 = _require_hex(payload.get("sha256"), HEX64, f"{label}.sha256")
    path = payload.get("path")
    if path is None:
        return {"sha256": sha256}
    path_text = _require_path(path, f"{label}.path")
    return {"sha256": sha256, "path": path_text}


def _validate_wandb_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    _require_text(payload.get("project"), f"{label}.project")
    _require_text(payload.get("entity"), f"{label}.entity")
    run_id = _require_text(payload.get("run_id"), f"{label}.run_id")
    _require(
        RUN_ID.fullmatch(run_id) is not None, f"{label}.run_id must be 8 alphanumeric characters"
    )
    run_url = _require_text(payload.get("run_url"), f"{label}.run_url")
    _require(run_url.startswith("https://wandb.ai/"), f"{label}.run_url must point to wandb.ai")
    return {
        "project": str(payload["project"]),
        "entity": str(payload["entity"]),
        "run_id": run_id,
        "run_url": run_url,
    }


def _validate_tinker_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    job_id = _require_text(payload.get("job_id"), f"{label}.job_id")
    _require(UUID.fullmatch(job_id) is not None, f"{label}.job_id must be a hyphenated UUID string")
    receipt_sha = _require_hex(payload.get("receipt_sha256"), HEX64, f"{label}.receipt_sha256")
    return {"job_id": job_id, "receipt_sha256": receipt_sha}


def _validate_hf_receipt(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    repo_id = _require_text(payload.get("repo_id"), f"{label}.repo_id")
    _require(REPO.fullmatch(repo_id) is not None, f"{label}.repo_id must be owner/repo")
    commit = _require_hex(payload.get("commit"), HEX40, f"{label}.commit")
    return {
        "repo_id": repo_id,
        "commit": commit,
        "receipt_sha256": _require_hex(
            payload.get("receipt_sha256"), HEX64, f"{label}.receipt_sha256"
        ),
    }


def _validate_environment_contract(value: Any, label: str) -> dict[str, str]:
    payload = _require_mapping(value, label)
    return {
        "container": _require_hex(
            payload.get("container"),
            HEX64,
            f"{label}.container",
        ),
        "decontamination": _require_hex(
            payload.get("decontamination"),
            HEX64,
            f"{label}.decontamination",
        ),
        "container_source": _require_text(
            payload.get("container_source"), f"{label}.container_source"
        ),
        "decontamination_source": _require_text(
            payload.get("decontamination_source"),
            f"{label}.decontamination_source",
        ),
    }


def _validate_verifier_contract(value: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(value, label)
    verifier = {
        "artifact_verifier_receipt": _validate_receipt(
            payload.get("artifact_verifier_receipt"),
            f"{label}.artifact_verifier_receipt",
        ),
        "state_verifier_receipt": _validate_receipt(
            payload.get("state_verifier_receipt"),
            f"{label}.state_verifier_receipt",
        ),
        "wandb": _validate_wandb_receipt(payload.get("wandb"), f"{label}.wandb"),
        "tinker": _validate_tinker_receipt(payload.get("tinker"), f"{label}.tinker"),
        "hugging_face": _validate_hf_receipt(payload.get("hugging_face"), f"{label}.hugging_face"),
    }

    artifact_source = payload.get("artifact")
    if artifact_source is not None:
        verifier["artifact"] = _require_text(artifact_source, f"{label}.artifact")
    return verifier


def _validate_native_contract(value: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(value, label)
    return {
        "artifact": {
            "sha256": _require_hex(
                payload.get("artifact_sha256"), HEX64, f"{label}.artifact_sha256"
            ),
            "source": _require_text(payload.get("artifact_source"), f"{label}.artifact_source"),
            "size_bytes": int(
                _require_non_negative_int(
                    payload.get("artifact_size_bytes"), f"{label}.artifact_size_bytes"
                )
            ),
        },
        "environment": _validate_environment_contract(
            payload.get("environment"),
            f"{label}.environment",
        ),
        "verifier": _validate_verifier_contract(payload.get("verifier"), f"{label}.verifier"),
    }


def _require_non_negative_int(value: Any, label: str) -> int:
    _require(isinstance(value, int), f"{label} must be an integer")
    _require(value >= 0, f"{label} must be non-negative")
    return value


def _validate_boundary(
    action: str, payload: Mapping[str, Any], spec: Mapping[str, str]
) -> dict[str, Any]:
    label = f"boundaries[{action}]"
    name = _require_text(payload.get("name"), f"{label}.name")
    _require(name == spec["expected_name"], f"{label}.name must be {spec['expected_name']!r}")
    source = _validate_source(
        payload.get("authoritative_source"),
        f"{label}.authoritative_source",
        spec["required_source_marker"],
    )
    revision = _require_hex(payload.get("revision"), HEX40, f"{label}.revision")
    role = _require_text(payload.get("evaluation_role"), f"{label}.evaluation_role")
    _require(
        role == spec["expected_role"], f"{label}.evaluation_role must be {spec['expected_role']!r}"
    )
    task_ids = _validate_task_ids(payload.get("task_ids"), label)
    split_hash = _validate_split_hash(payload.get("split_hash"), task_ids, label)
    license_receipt = _validate_receipt(payload.get("license"), f"{label}.license")
    native_contract = _validate_native_contract(
        payload.get("native_contract"), f"{label}.native_contract"
    )
    return {
        "name": name,
        "authoritative_source": source,
        "revision": revision,
        "evaluation_role": role,
        "task_ids": task_ids,
        "split_hash": split_hash,
        "license": license_receipt,
        "native_contract": native_contract,
    }


def _normalize_boundaries(boundaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    present = set(boundaries)
    required = set(BOUNDARY_SPECS)
    _require(present == required, f"boundaries must contain exactly {sorted(required)}")

    normalized: dict[str, Any] = {}
    seen_task_ids: set[str] = set()
    for action_id in sorted(required):
        raw = _require_mapping(boundaries[action_id], f"boundaries[{action_id}]")
        normalized[action_id] = _validate_boundary(action_id, raw, BOUNDARY_SPECS[action_id])

        action_task_ids = set(normalized[action_id]["task_ids"])
        _require(
            seen_task_ids.isdisjoint(action_task_ids),
            f"boundaries[{action_id}].task_ids must not overlap with other boundaries",
        )
        seen_task_ids.update(action_task_ids)

    return normalized


def validate_appbench_openreward_games_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a Pavlov contract containing the E12 and E13 offline boundaries."""

    contract = _require_mapping(payload, "contract")
    paid_launch_allowed = contract.get("paid_launch_allowed")
    _require(isinstance(paid_launch_allowed, bool), "paid_launch_allowed must be a boolean")
    _require(not paid_launch_allowed, "paid_launch_allowed must be False")
    _require(
        contract.get("stateful_trajectory") is True,
        "stateful_trajectory must be true",
    )

    boundaries = _require_mapping(contract.get("boundaries"), "boundaries")
    normalized_boundaries = _normalize_boundaries(boundaries)

    return {
        "paid_launch_allowed": False,
        "stateful_trajectory": True,
        "boundaries": normalized_boundaries,
    }


validate_appbench_contract = validate_appbench_openreward_games_contract
validate_pavlov_appbench_openreward_games_contract = validate_appbench_openreward_games_contract
validate_pavlov_openreward_games_adapter = validate_appbench_openreward_games_contract


def _canonical_hash(value: Mapping[str, Any], omit: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in value.items() if k != omit}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _exact_map(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PavlovAppbenchOpenrewardGamesAdapterError(f"{label} has missing or unknown fields")
    return value


def _load_appbench_trust_root(trust_root: Mapping[str, Any] | str | Path | None) -> dict[str, str]:
    try:
        raw = (
            json.loads(Path(trust_root).read_text())
            if isinstance(trust_root, (str, Path))
            else trust_root
        )
        required = {"schema_version", "lane", "suite_id", "provider", "key_id", "public_key_hex"}
        if (
            not isinstance(raw, Mapping)
            or set(raw) != required
            or not all(isinstance(raw[key], str) for key in required)
        ):
            raise ValueError("schema")
        root = {key: str(value) for key, value in raw.items()}
        if (
            (root["schema_version"], root["lane"], root["suite_id"], root["provider"])
            != (APPBENCH_TRUST_ROOT_SCHEMA, "E12", "appbench_eval", "AfterQuery")
            or not root["key_id"]
            or not HEX64.fullmatch(root["public_key_hex"])
        ):
            raise ValueError("identity")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(root["public_key_hex"]))
        root["document_sha256"] = _canonical_hash(raw, "__never_present__")
        root["key_fingerprint"] = hashlib.sha256(bytes.fromhex(root["public_key_hex"])).hexdigest()
        return root
    except Exception as exc:
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "explicit valid E12 provider trust root is required"
        ) from exc


def _verify_appbench_signature(
    payload: Mapping[str, Any], trust_root: Mapping[str, Any] | str | Path | None
) -> dict[str, str]:
    try:
        root = _load_appbench_trust_root(trust_root)
        if payload.get("signature_key_id") != root["key_id"]:
            raise ValueError("wrong lane key")
        signature = base64.b64decode(str(payload["signature"]), validate=True)
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(root["public_key_hex"])).verify(
            signature,
            json.dumps(
                {k: v for k, v in payload.items() if k != "signature"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
        )
        issued = datetime.fromisoformat(str(payload["issued_at"]).replace("Z", "+00:00"))
        expires = datetime.fromisoformat(str(payload["expires_at"]).replace("Z", "+00:00"))
        if (
            issued.tzinfo is None
            or expires.tzinfo is None
            or issued > datetime.now(timezone.utc)
            or expires <= datetime.now(timezone.utc)
        ):
            raise ValueError("invalid validity window")
        return root
    except Exception as exc:
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "grant trust-root signature or validity window is invalid"
        ) from exc


def validate_appbench_provider_grant(
    grant: Mapping[str, Any], *, trust_root: Mapping[str, Any] | str | Path | None = None
) -> dict[str, Any]:
    """Require the real licence, deployment, and two-human-grader protocol."""
    required = {
        "schema_version",
        "lane",
        "suite_id",
        "provider",
        "issued_at",
        "expires_at",
        "grant_id",
        "license",
        "deployment",
        "graders",
        "heldout",
        "runtime",
        "verifier",
        "receipt_url",
        "signature_key_id",
        "signature",
    }
    if not isinstance(grant, Mapping) or set(grant) != required:
        raise PavlovAppbenchOpenrewardGamesAdapterError("grant has missing or unknown fields")
    data = dict(grant)
    if (
        data["schema_version"] != APPBENCH_GRANT_SCHEMA
        or data["lane"] != "E12"
        or data["suite_id"] != "appbench_eval"
        or data["provider"] != "AfterQuery"
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError("grant lane, provider, or suite is invalid")
    root = _verify_appbench_signature(data, trust_root)
    license_data, deployment, graders, heldout = (
        data["license"],
        data["deployment"],
        data["graders"],
        data["heldout"],
    )
    license_data = _exact_map(license_data, {"approved", "receipt_id", "sha256"}, "license")
    if (
        license_data.get("approved") is not True
        or not license_data.get("receipt_id")
        or not HEX64.fullmatch(str(license_data.get("sha256", "")))
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError("explicit pinned licence grant is required")
    deployment = _exact_map(
        deployment, {"revision", "container_sha256", "reset_receipt_id"}, "deployment"
    )
    if (
        not HEX40.fullmatch(str(deployment.get("revision", "")))
        or not HEX64.fullmatch(str(deployment.get("container_sha256", "")))
        or not _require_text(deployment.get("reset_receipt_id"), "deployment.reset_receipt_id")
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "official immutable deployment/reset contract is required"
        )
    graders = _exact_map(graders, {"count", "re_adjudication", "protocol_sha256"}, "graders")
    if (
        isinstance(graders.get("count"), bool)
        or graders.get("count") != 2
        or graders.get("re_adjudication") is not True
        or not HEX64.fullmatch(str(graders.get("protocol_sha256", "")))
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "two-grader re-adjudication protocol is required"
        )
    heldout = _exact_map(
        heldout,
        {"contamination_checked", "manifest_sha256", "task_count", "task_artifact_sha256"},
        "heldout",
    )
    if (
        heldout.get("contamination_checked") is not True
        or not HEX64.fullmatch(str(heldout.get("manifest_sha256", "")))
        or isinstance(heldout.get("task_count"), bool)
        or not isinstance(heldout.get("task_count"), int)
        or heldout["task_count"] != APPBENCH_EXPECTED_HELDOUT_TASK_COUNT
        or not HEX64.fullmatch(str(heldout.get("task_artifact_sha256", "")))
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "held-out contamination receipt must bind the exact six-task AppBench suite"
        )
    if not isinstance(data["receipt_url"], str) or not data["receipt_url"].startswith("https://"):
        raise PavlovAppbenchOpenrewardGamesAdapterError("provider receipt_url must be HTTPS")
    runtime = _exact_map(data["runtime"], {"revision", "container_sha256", "endpoint"}, "runtime")
    verifier = _exact_map(data["verifier"], {"revision", "sha256", "approval_id"}, "verifier")
    if (
        not HEX40.fullmatch(str(runtime.get("revision", "")))
        or not HEX64.fullmatch(str(runtime.get("container_sha256", "")))
        or not str(runtime.get("endpoint", "")).startswith("https://")
        or not HEX40.fullmatch(str(verifier.get("revision", "")))
        or not HEX64.fullmatch(str(verifier.get("sha256", "")))
        or not verifier.get("approval_id")
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError("pinned runtime and verifier are required")
    data["grant_fingerprint"] = _canonical_hash(data, "signature")
    data["trust_root_sha256"] = root["document_sha256"]
    data["trust_root_key_fingerprint"] = root["key_fingerprint"]
    return data


def build_appbench_provider_handoff(
    grant: Mapping[str, Any], *, trust_root: Mapping[str, Any] | str | Path | None = None
) -> dict[str, Any]:
    data = validate_appbench_provider_grant(grant, trust_root=trust_root)
    return {
        "schema_version": APPBENCH_HANDOFF_SCHEMA,
        "status": "PROVIDER_EXECUTION_REQUIRED",
        "paid_launch_allowed": False,
        "suite_id": "appbench_eval",
        "grant_fingerprint": data["grant_fingerprint"],
        "trust_root_sha256": data["trust_root_sha256"],
        "trust_root_key_fingerprint": data["trust_root_key_fingerprint"],
        "deployment": data["deployment"],
        "graders": data["graders"],
        "score": None,
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_appbench_utc(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_require_text(value, label).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            f"{label} must be an ISO-8601 UTC timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise PavlovAppbenchOpenrewardGamesAdapterError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validate_result_receipts(payload: Any) -> dict[str, Any]:
    """Validate the immutable training and logging receipts in a signed result."""
    receipts = _exact_map(payload, {"wandb", "tinker", "hugging_face"}, "result.receipts")
    wandb = _exact_map(
        receipts["wandb"], {"project", "entity", "run_id", "run_url"}, "result.receipts.wandb"
    )
    tinker = _exact_map(receipts["tinker"], {"job_id", "receipt_sha256"}, "result.receipts.tinker")
    hugging_face = _exact_map(
        receipts["hugging_face"],
        {"repo_id", "commit", "receipt_sha256"},
        "result.receipts.hugging_face",
    )
    return {
        "wandb": _validate_wandb_receipt(wandb, "result.receipts.wandb"),
        "tinker": _validate_tinker_receipt(tinker, "result.receipts.tinker"),
        "hugging_face": _validate_hf_receipt(hugging_face, "result.receipts.hugging_face"),
    }


def collect_appbench_signed_result(
    grant: Mapping[str, Any],
    handoff: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    trust_root: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Accept the sole E12 completion path: an exact provider-signed result.

    Public CSVs, local harness outputs, and unsigned provider narratives cannot
    satisfy this intake: it derives the authoritative handoff from the signed
    grant, verifies a separately signed result against the same lane-scoped
    trust root, and binds every execution identity before returning a score.
    """
    verified = validate_appbench_provider_grant(grant, trust_root=trust_root)
    expected_handoff = build_appbench_provider_handoff(grant, trust_root=trust_root)
    if not isinstance(handoff, Mapping) or dict(handoff) != expected_handoff:
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "handoff is not the exact verified provider execution handoff"
        )

    required = {
        "schema_version",
        "suite_id",
        "provider",
        "issued_at",
        "expires_at",
        "signature_key_id",
        "signature",
        "grant_fingerprint",
        "handoff_fingerprint",
        "heldout_manifest_sha256",
        "task_count",
        "task_artifact_sha256",
        "deployment_revision",
        "deployment_container_sha256",
        "runtime_revision",
        "runtime_container_sha256",
        "verifier_revision",
        "verifier_sha256",
        "graders",
        "model_revision",
        "artifact_sha256",
        "wandb_before_tinker",
        "receipts",
        "metric",
        "score",
        "completed_at",
    }
    if not isinstance(result, Mapping) or set(result) != required:
        raise PavlovAppbenchOpenrewardGamesAdapterError("result has missing or unknown fields")
    payload = dict(result)
    root = _verify_appbench_signature(payload, trust_root)
    receipts = _validate_result_receipts(payload["receipts"])
    graders = _exact_map(
        payload["graders"], {"count", "re_adjudication", "protocol_sha256"}, "result.graders"
    )
    model_revision = _require_hex(payload["model_revision"], HEX40, "result.model_revision")
    artifact_sha256 = _require_hex(payload["artifact_sha256"], HEX64, "result.artifact_sha256")
    score = payload["score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError("result.score must be a finite number")
    if not 0.0 <= float(score) <= 1.0:
        raise PavlovAppbenchOpenrewardGamesAdapterError("result.score must be bounded to [0, 1]")

    expected_handoff_fingerprint = hashlib.sha256(
        _canonical_json(expected_handoff).encode()
    ).hexdigest()
    if (
        payload["schema_version"] != APPBENCH_RESULT_SCHEMA
        or payload["suite_id"] != "appbench_eval"
        or payload["provider"] != "AfterQuery"
        or payload["issued_at"] != verified["issued_at"]
        or payload["expires_at"] != verified["expires_at"]
        or root["document_sha256"] != verified["trust_root_sha256"]
        or root["key_fingerprint"] != verified["trust_root_key_fingerprint"]
        or payload["grant_fingerprint"] != verified["grant_fingerprint"]
        or payload["handoff_fingerprint"] != expected_handoff_fingerprint
        or payload["heldout_manifest_sha256"] != verified["heldout"]["manifest_sha256"]
        or isinstance(payload["task_count"], bool)
        or payload["task_count"] != APPBENCH_EXPECTED_HELDOUT_TASK_COUNT
        or payload["task_count"] != verified["heldout"]["task_count"]
        or payload["task_artifact_sha256"] != verified["heldout"]["task_artifact_sha256"]
        or payload["deployment_revision"] != verified["deployment"]["revision"]
        or payload["deployment_container_sha256"] != verified["deployment"]["container_sha256"]
        or payload["runtime_revision"] != verified["runtime"]["revision"]
        or payload["runtime_container_sha256"] != verified["runtime"]["container_sha256"]
        or payload["verifier_revision"] != verified["verifier"]["revision"]
        or payload["verifier_sha256"] != verified["verifier"]["sha256"]
        or graders != verified["graders"]
        or model_revision != receipts["hugging_face"]["commit"]
        or payload["wandb_before_tinker"] is not True
        or payload["metric"] != "appbench_score"
    ):
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "result bindings, two-grader protocol, or execution receipts are invalid"
        )

    completed = _parse_appbench_utc(payload["completed_at"], "result.completed_at")
    issued = _parse_appbench_utc(verified["issued_at"], "grant.issued_at")
    expires = _parse_appbench_utc(verified["expires_at"], "grant.expires_at")
    if completed < issued or completed > expires or completed > datetime.now(timezone.utc):
        raise PavlovAppbenchOpenrewardGamesAdapterError(
            "result.completed_at is outside the signed validity window"
        )

    return {
        "status": "COMPLETE",
        "suite_id": "appbench_eval",
        "score": float(score),
        "metric": payload["metric"],
        "task_count": payload["task_count"],
        "artifact_sha256": artifact_sha256,
        "grant_fingerprint": verified["grant_fingerprint"],
        "handoff_fingerprint": expected_handoff_fingerprint,
        "provider_signed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline E12 provider handoff and signed-result collector"
    )
    parser.add_argument("--mode", choices=("handoff", "collect"), default="handoff")
    parser.add_argument("--grant", required=True)
    parser.add_argument("--trust-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--handoff")
    parser.add_argument("--result")
    args = parser.parse_args(argv)
    try:
        grant = json.loads(Path(args.grant).read_text(encoding="utf-8"))
        if args.mode == "collect":
            if not args.handoff or not args.result:
                raise PavlovAppbenchOpenrewardGamesAdapterError(
                    "collect mode requires --handoff and --result"
                )
            payload = collect_appbench_signed_result(
                grant,
                json.loads(Path(args.handoff).read_text(encoding="utf-8")),
                json.loads(Path(args.result).read_text(encoding="utf-8")),
                trust_root=args.trust_root,
            )
        else:
            payload = build_appbench_provider_handoff(grant, trust_root=args.trust_root)
        target = Path(args.out)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        print(json.dumps(payload))
        return 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "score": None, "errors": [str(exc)]}))
        return 2


__all__ = [
    "PavlovAppbenchOpenrewardGamesAdapterError",
    "validate_appbench_openreward_games_contract",
    "validate_appbench_contract",
    "validate_pavlov_appbench_openreward_games_contract",
    "validate_pavlov_openreward_games_adapter",
    "validate_appbench_provider_grant",
    "build_appbench_provider_handoff",
    "collect_appbench_signed_result",
]

if __name__ == "__main__":
    raise SystemExit(main())
