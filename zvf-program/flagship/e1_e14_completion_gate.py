#!/usr/bin/env python3
"""Fail-closed readiness and provider-package validation for E2--E14.

This is deliberately an audit boundary.  It neither invokes an adapter nor
accepts a package as a benchmark result: a valid package only proves that the
declared immutable inputs are internally consistent and available locally.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_STATUS = REPO_ROOT / "outputs/E1_E14_Terminal_Status_2026-08-29.json"
DEFAULT_FOLLOWUP = REPO_ROOT / "outputs/E3_E14_EXTERNAL_FOLLOWUP_RECEIPT_2026-08-29.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/E2_E14_COMPLETION_GATE_2026-08-30.json"
SCHEMA_VERSION = "e1-e14-completion-gate-v1"
PACKAGE_SCHEMA_VERSION = "e1-e14-provider-package-v1"
PROVIDER_GRANT_SCHEMA_VERSION = "e1-e14-provider-grant-v2"
TRUST_ROOTS_SCHEMA_VERSION = "e1-e14-provider-trust-roots-v2"
HASH_LENGTH = 64
EVIDENCE_CLASSES = frozenset(
    {
        "COMPLETE_EXACT",
        "PARTIAL_EXACT",
        "PARTIAL_RECOVERY",
        "BLOCKED_EXTERNAL",
        "LOCAL_SETUP_READY_PROVIDER_INPUT_REQUIRED",
    }
)
SOURCE_STATUS_TO_CLASS = {
    "SCORED_EXACT_SUITE": "COMPLETE_EXACT",
    "PARTIAL_EXACT": "PARTIAL_EXACT",
    "PARTIAL_RECOVERY": "PARTIAL_RECOVERY",
    "BLOCKED_EXTERNAL": "BLOCKED_EXTERNAL",
    "LOCAL_SETUP_READY_PROVIDER_INPUT_REQUIRED": "LOCAL_SETUP_READY_PROVIDER_INPUT_REQUIRED",
}


# These are paths to local integration surfaces, not evidence that their
# upstream benchmark inputs are licensed, complete, or runnable.
LANE_SPECS: dict[str, dict[str, Any]] = {
    "E2": {
        "suite_id": "frontier_swe_eval",
        "expected_task_count": 17,
        "class": "PARTIAL_EXACT",
        "adapter": "zvf-program/flagship/pavlov_frontier_swe_eval_adapter.py",
        "runner": "zvf-program/flagship/frontier_swe_eval.py",
        "test": "zvf-program/flagship/test_pavlov_frontier_swe_eval_adapter.py",
        "required": [
            "immutable task bundle",
            "applicable license",
            "pinned native runtime",
            "native grader",
            "sufficient budget",
        ],
    },
    "E3": {
        "suite_id": "sdab_eval",
        "expected_task_count": 80,
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_sdab_eval_adapter.py",
        "runner": "zvf-program/flagship/eval_pavlov_sdab.py",
        "test": "zvf-program/flagship/test_pavlov_sdab_eval_adapter.py",
        "required": [
            "private immutable 80-task bundle",
            "provider license",
            "provider runtime/reset contract",
            "native grader",
            "sufficient budget",
        ],
    },
    "E4": {
        "suite_id": "banker_toolbench_eval",
        "expected_task_count": 100,
        "class": "PARTIAL_RECOVERY",
        "adapter": "zvf-program/flagship/pavlov_banker_toolbench_eval_adapter.py",
        "runner": "zvf-program/flagship/run_harbor_via_tinker_bridge.py",
        "test": "zvf-program/flagship/test_pavlov_banker_toolbench_eval_adapter.py",
        "required": [
            "immutable full task bundle",
            "applicable license",
            "pinned Harbor runtime",
            "native grader",
            "sufficient budget",
        ],
    },
    "E5": {
        "suite_id": "apex_agents_eval",
        "expected_task_count": 480,
        "class": "PARTIAL_EXACT",
        "adapter": "zvf-program/flagship/pavlov_apex_agents_eval_adapter.py",
        "runner": "zvf-program/flagship/eval_apex_agents.py",
        "test": "zvf-program/flagship/test_pavlov_apex_agents_eval_adapter.py",
        "required": [
            "immutable full task bundle",
            "applicable license",
            "pinned runtime",
            "native grader",
            "budget for full suite and judges",
        ],
    },
    "E6": {
        "suite_id": "webbench_eval",
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_webbench_eval_adapter.py",
        "runner": "zvf-program/flagship/webbench_eval.py",
        "test": "zvf-program/flagship/test_pavlov_webbench_eval_adapter.py",
        "required": [
            "official live environment",
            "provider license/write authorization",
            "reset and ground-truth contract",
            "native grader",
            "sufficient budget",
        ],
    },
    "E7": {
        "suite_id": "binaryaudit_eval",
        "expected_task_count": 46,
        "class": "PARTIAL_EXACT",
        "adapter": "zvf-program/flagship/pavlov_binaryaudit_eval_adapter.py",
        "runner": "zvf-program/flagship/modal_non_e11_readiness.py",
        "test": "zvf-program/flagship/test_pavlov_binaryaudit_eval_adapter.py",
        "required": [
            "immutable full task split/assets",
            "applicable license",
            "pinned amd64 reverse-engineering runtime",
            "native verifier",
            "sufficient budget",
        ],
    },
    "E8": {
        "suite_id": "lifescibench_eval",
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_lifescibench_eval_adapter.py",
        "runner": "zvf-program/flagship/pavlov_lifescibench_eval_adapter.py",
        "test": "zvf-program/flagship/test_pavlov_lifescibench_eval_adapter.py",
        "required": [
            "official immutable task package",
            "provider license",
            "pinned runtime",
            "native grader",
            "sufficient budget",
        ],
    },
    "E9": {
        "suite_id": "mle_bench_eval",
        "expected_task_count": 75,
        "class": "PARTIAL_EXACT",
        "adapter": "zvf-program/flagship/e9_mle_bench_streaming.py",
        "runner": "zvf-program/flagship/modal_e9_mle_bench_streaming.py",
        "test": "zvf-program/flagship/test_e9_mle_bench_streaming.py",
        "required": [
            "immutable competition/task registry",
            "accepted competition terms",
            "pinned benchmark runtime",
            "native competition grades",
            "budget for missing coverage",
        ],
    },
    "E10": {
        "suite_id": "agentharm_eval",
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py",
        "runner": "zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py",
        "test": "zvf-program/flagship/test_pavlov_agentharm_frontiermath_adapter.py",
        "required": [
            "private held-out tasks",
            "provider license",
            "pinned runtime",
            "official native grader",
            "sufficient budget",
        ],
    },
    "E12": {
        "suite_id": "appbench_eval",
        "expected_task_count": 6,
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_appbench_openreward_games_adapter.py",
        "runner": "zvf-program/flagship/pavlov_appbench_openreward_games_adapter.py",
        "test": "zvf-program/flagship/test_pavlov_appbench_openreward_games_adapter.py",
        "required": [
            "official deployment/task artifacts",
            "provider license",
            "deployment/runtime receipt",
            "native grading protocol",
            "sufficient budget",
        ],
    },
    "E13": {
        "suite_id": "openreward_games_eval",
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_appbench_openreward_games_adapter.py",
        "runner": "zvf-program/flagship/e13_openreward_games_local_runner.py",
        "test": "zvf-program/flagship/test_e13_openreward_games_local_runner.py",
        "required": [
            "provider-defined immutable held-out suite",
            "provider license",
            "provider runtime/deployment receipt",
            "native grading contract",
            "API/billing budget",
        ],
    },
    "E14": {
        "suite_id": "frontiermath_eval",
        "class": "BLOCKED_EXTERNAL",
        "adapter": "zvf-program/flagship/pavlov_agentharm_frontiermath_adapter.py",
        "runner": "zvf-program/flagship/e14_frontiermath_public_samples.py",
        "test": "zvf-program/flagship/test_e14_frontiermath_public_samples.py",
        "required": [
            "private hosted evaluation bundle",
            "provider license",
            "hosted runtime receipt",
            "native grader",
            "sufficient budget",
        ],
    },
}


class PackageValidationError(ValueError):
    """A provider package did not meet the immutable-input gate."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def classify_evidence(benchmark_status: Any) -> str:
    """Translate only recognized source evidence states; unknown states fail closed."""
    if not isinstance(benchmark_status, str) or benchmark_status not in SOURCE_STATUS_TO_CLASS:
        raise PackageValidationError(f"unsupported benchmark evidence status: {benchmark_status!r}")
    return SOURCE_STATUS_TO_CLASS[benchmark_status]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise PackageValidationError(f"value is not canonical JSON: {exc}") from exc


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageValidationError(f"{path}: root must be an object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise PackageValidationError(f"{context}: unknown fields {sorted(unknown)}")
    if missing:
        raise PackageValidationError(f"{context}: missing fields {sorted(missing)}")


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == HASH_LENGTH
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _require_hash(value: Any, context: str) -> str:
    if not _is_hash(value):
        raise PackageValidationError(f"{context}: must be a lowercase SHA-256")
    return value


def _require_string(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value.strip().lower() in {"placeholder", "todo", "tbd", "unknown", "n/a"}
    ):
        raise PackageValidationError(f"{context}: must be a non-placeholder string")
    return value


def _safe_relative_path(value: Any, context: str) -> PurePosixPath:
    raw = _require_string(value, context)
    path = PurePosixPath(raw)
    if path.is_absolute() or "\\" in raw or any(part in {"", ".", ".."} for part in path.parts):
        raise PackageValidationError(f"{context}: must be a safe relative POSIX path")
    return path


def _safe_file(root: Path, relative: Any, context: str) -> Path:
    rel = _safe_relative_path(relative, context)
    candidate = root.joinpath(*rel.parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # defensive even after PurePosixPath validation
        raise PackageValidationError(f"{context}: escapes package root") from exc
    current = root
    for part in rel.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise PackageValidationError(f"{context}: missing file {rel}") from exc
        if stat.S_ISLNK(mode):
            raise PackageValidationError(f"{context}: symlinks are not allowed ({rel})")
    if not candidate.is_file():
        raise PackageValidationError(f"{context}: must name a regular file")
    return candidate


def _verify_file(root: Path, item: dict[str, Any], context: str) -> dict[str, str]:
    _require_keys(item, {"path", "sha256"}, context)
    expected_hash = _require_hash(item["sha256"], f"{context}.sha256")
    path = _safe_file(root, item["path"], f"{context}.path")
    observed_hash = _sha256_file(path)
    if observed_hash != expected_hash:
        raise PackageValidationError(f"{context}: SHA-256 mismatch for {item['path']}")
    return {"path": str(item["path"]), "sha256": observed_hash}


def _utc_timestamp(value: Any, context: str) -> datetime:
    text = _require_string(value, context)
    if not text.endswith("Z"):
        raise PackageValidationError(f"{context}: must be a UTC ISO-8601 timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise PackageValidationError(f"{context}: invalid UTC ISO-8601 timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PackageValidationError(f"{context}: must use UTC")
    return parsed


def _load_trust_roots(path: Path, requested_lane: str) -> dict[str, Any]:
    """Load one fully validated, lane-scoped trust-root document.

    The document digest is intentionally over the complete canonical document,
    rather than only the selected public key.  This means provider metadata and
    grant allow-list changes are part of the signed grant identity as well.
    """
    if requested_lane not in LANE_SPECS:
        raise PackageValidationError(
            f"trust roots requested for unknown or complete lane: {requested_lane}"
        )
    roots = _load_json(path)
    _require_keys(roots, {"schema_version", "trust_roots"}, "trust_roots")
    if roots["schema_version"] != TRUST_ROOTS_SCHEMA_VERSION:
        raise PackageValidationError("trust_roots.schema_version is unsupported")
    raw_roots = roots["trust_roots"]
    if not isinstance(raw_roots, list) or not raw_roots:
        raise PackageValidationError("trust_roots.trust_roots must be a non-empty array")
    result: list[dict[str, Any]] = []
    root_keys: set[tuple[str, str]] = set()
    grant_ids: set[str] = set()
    for index, root in enumerate(raw_roots):
        context = f"trust_roots[{index}]"
        if not isinstance(root, dict):
            raise PackageValidationError(f"{context}: must be an object")
        _require_keys(
            root,
            {
                "lane",
                "suite_id",
                "provider",
                "key_id",
                "public_key_b64",
                "public_key_sha256",
                "accepted_grant_ids",
            },
            context,
        )
        lane = _require_string(root["lane"], f"{context}.lane")
        if lane != requested_lane:
            raise PackageValidationError(
                f"{context}.lane: trust-root document must be scoped to {requested_lane}"
            )
        suite_id = _require_string(root["suite_id"], f"{context}.suite_id")
        if suite_id != LANE_SPECS[requested_lane]["suite_id"]:
            raise PackageValidationError(f"{context}.suite_id: does not bind the lane")
        key_id = _require_string(root["key_id"], f"{context}.key_id")
        root_key = (lane, key_id)
        if root_key in root_keys:
            raise PackageValidationError(f"{context}: duplicate lane/key_id trust root")
        root_keys.add(root_key)
        try:
            public_key = base64.b64decode(
                _require_string(root["public_key_b64"], f"{context}.public_key_b64"), validate=True
            )
        except (ValueError, UnicodeEncodeError) as exc:
            raise PackageValidationError(f"{context}.public_key_b64: invalid base64") from exc
        if len(public_key) != 32:
            raise PackageValidationError(
                f"{context}.public_key_b64: Ed25519 public keys must be 32 bytes"
            )
        fingerprint = _require_hash(root["public_key_sha256"], f"{context}.public_key_sha256")
        if _sha256_bytes(public_key) != fingerprint:
            raise PackageValidationError(f"{context}.public_key_sha256: does not match public key")
        ids = root["accepted_grant_ids"]
        if not isinstance(ids, list) or not ids:
            raise PackageValidationError(f"{context}.accepted_grant_ids: must be a non-empty array")
        normalized_ids = [
            _require_string(item, f"{context}.accepted_grant_ids[{item_index}]")
            for item_index, item in enumerate(ids)
        ]
        if len(normalized_ids) != len(set(normalized_ids)):
            raise PackageValidationError(f"{context}.accepted_grant_ids: contains duplicates")
        overlap = grant_ids.intersection(normalized_ids)
        if overlap:
            raise PackageValidationError(
                f"{context}.accepted_grant_ids: grant IDs are not globally unique"
            )
        grant_ids.update(normalized_ids)
        result.append(
            {
                "lane": lane,
                "suite_id": suite_id,
                "provider": _require_string(root["provider"], f"{context}.provider"),
                "key_id": key_id,
                "public_key": public_key,
                "public_key_sha256": fingerprint,
                "accepted_grant_ids": normalized_ids,
            }
        )
    return {"roots": result, "sha256": _canonical_sha256(roots)}


def _verify_grant(
    value: Any,
    *,
    lane: str,
    suite_id: str,
    bindings: dict[str, str],
    trust_root_document: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PackageValidationError("provider_grant: must be an object")
    _require_keys(
        value,
        {
            "schema_version",
            "lane",
            "suite_id",
            "provider",
            "grant_id",
            "issued_at",
            "expires_at",
            "key_id",
            "public_key_sha256",
            "trust_root_sha256",
            "bindings",
            "signature_b64",
        },
        "provider_grant",
    )
    if value["schema_version"] != PROVIDER_GRANT_SCHEMA_VERSION:
        raise PackageValidationError("provider_grant.schema_version is unsupported")
    if value["lane"] != lane or value["suite_id"] != suite_id:
        raise PackageValidationError("provider_grant must bind the requested lane and suite")
    provider = _require_string(value["provider"], "provider_grant.provider")
    grant_id = _require_string(value["grant_id"], "provider_grant.grant_id")
    key_id = _require_string(value["key_id"], "provider_grant.key_id")
    fingerprint = _require_hash(value["public_key_sha256"], "provider_grant.public_key_sha256")
    trust_root_sha256 = _require_hash(
        value["trust_root_sha256"], "provider_grant.trust_root_sha256"
    )
    if trust_root_sha256 != trust_root_document["sha256"]:
        raise PackageValidationError(
            "provider_grant.trust_root_sha256: does not match the complete trust-root document"
        )
    issued_at = _utc_timestamp(value["issued_at"], "provider_grant.issued_at")
    expires_at = _utc_timestamp(value["expires_at"], "provider_grant.expires_at")
    now = datetime.now(timezone.utc)
    if issued_at > now or expires_at <= now or expires_at <= issued_at:
        raise PackageValidationError(
            "provider_grant: is not currently within its UTC validity interval"
        )
    if not isinstance(value["bindings"], dict):
        raise PackageValidationError("provider_grant.bindings: must be an object")
    _require_keys(value["bindings"], set(bindings), "provider_grant.bindings")
    for name, expected in bindings.items():
        observed = _require_hash(value["bindings"][name], f"provider_grant.bindings.{name}")
        if observed != expected:
            raise PackageValidationError(
                f"provider_grant.bindings.{name}: does not match package input"
            )
    candidates = [
        root
        for root in trust_root_document["roots"]
        if root["lane"] == lane
        and root["suite_id"] == suite_id
        and root["provider"] == provider
        and root["key_id"] == key_id
        and root["public_key_sha256"] == fingerprint
        and grant_id in root["accepted_grant_ids"]
    ]
    if len(candidates) != 1:
        raise PackageValidationError(
            "provider_grant: no exact pinned trust root grants this identity"
        )
    try:
        signature = base64.b64decode(
            _require_string(value["signature_b64"], "provider_grant.signature_b64"), validate=True
        )
    except (ValueError, UnicodeEncodeError) as exc:
        raise PackageValidationError("provider_grant.signature_b64: invalid base64") from exc
    if len(signature) != 64:
        raise PackageValidationError(
            "provider_grant.signature_b64: Ed25519 signatures must be 64 bytes"
        )
    signed = {name: entry for name, entry in value.items() if name != "signature_b64"}
    try:
        Ed25519PublicKey.from_public_bytes(candidates[0]["public_key"]).verify(
            signature, _canonical_json(signed)
        )
    except InvalidSignature as exc:
        raise PackageValidationError(
            "provider_grant.signature_b64: signature does not verify"
        ) from exc
    return {
        "provider": provider,
        "grant_id": grant_id,
        "issued_at": value["issued_at"],
        "expires_at": value["expires_at"],
        "key_id": key_id,
        "public_key_sha256": fingerprint,
        "trust_root_sha256": trust_root_sha256,
        "bindings": dict(value["bindings"]),
    }


def validate_package(
    lane: str, manifest_path: Path, package_root: Path, trust_roots_path: Path | None = None
) -> dict[str, Any]:
    """Validate a package without running it or changing any score/status."""
    if lane not in LANE_SPECS:
        raise PackageValidationError(f"unknown or complete lane: {lane}")
    if not package_root.is_dir() or package_root.is_symlink():
        raise PackageValidationError("package root must be a non-symlink directory")
    if trust_roots_path is None:
        raise PackageValidationError(
            "--trust-roots is required; provider packages fail closed without roots"
        )
    trust_root_document = _load_trust_roots(trust_roots_path, lane)
    manifest = _load_json(manifest_path)
    _require_keys(
        manifest,
        {
            "schema_version",
            "lane",
            "suite_id",
            "provider_grant",
            "assets",
            "license",
            "runtime",
            "native_grader",
            "tasks",
            "budget_usd",
            "deployment_receipt",
        },
        "manifest",
    )
    if manifest["schema_version"] != PACKAGE_SCHEMA_VERSION:
        raise PackageValidationError("manifest.schema_version is unsupported")
    if manifest["lane"] != lane:
        raise PackageValidationError("manifest.lane does not match --lane")
    if manifest["suite_id"] != LANE_SPECS[lane]["suite_id"]:
        raise PackageValidationError("manifest.suite_id does not match the lane")
    if not isinstance(manifest["assets"], list) or not manifest["assets"]:
        raise PackageValidationError("manifest.assets must be a non-empty array")
    assets = [
        _verify_file(package_root, item, f"assets[{index}]")
        if isinstance(item, dict)
        else (_ for _ in ()).throw(PackageValidationError(f"assets[{index}]: must be an object"))
        for index, item in enumerate(manifest["assets"])
    ]
    if len({item["path"] for item in assets}) != len(assets):
        raise PackageValidationError("manifest.assets contains duplicate paths")
    license_item = manifest["license"]
    if not isinstance(license_item, dict):
        raise PackageValidationError("license: must be an object")
    _require_keys(license_item, {"path", "sha256", "text_sha256"}, "license")
    license_record = _verify_file(
        package_root, {"path": license_item["path"], "sha256": license_item["sha256"]}, "license"
    )
    try:
        license_text = _safe_file(package_root, license_item["path"], "license.path").read_text(
            encoding="utf-8"
        )
    except UnicodeDecodeError as exc:
        raise PackageValidationError("license.path: license must be strict UTF-8 text") from exc
    if _sha256_bytes(license_text.encode("utf-8")) != _require_hash(
        license_item["text_sha256"], "license.text_sha256"
    ):
        raise PackageValidationError("license.text_sha256 does not match UTF-8 license text")
    runtime = _verify_file(
        package_root,
        manifest["runtime"] if isinstance(manifest["runtime"], dict) else {},
        "runtime",
    )
    grader = _verify_file(
        package_root,
        manifest["native_grader"] if isinstance(manifest["native_grader"], dict) else {},
        "native_grader",
    )
    if not isinstance(manifest["tasks"], list) or not manifest["tasks"]:
        raise PackageValidationError("manifest.tasks must be a non-empty array")
    task_ids: set[str] = set()
    split_ids: set[str] = set()
    tasks: list[dict[str, str]] = []
    for index, task in enumerate(manifest["tasks"]):
        if not isinstance(task, dict):
            raise PackageValidationError(f"tasks[{index}]: must be an object")
        _require_keys(task, {"task_id", "split_id"}, f"tasks[{index}]")
        task_id = _require_string(task["task_id"], f"tasks[{index}].task_id")
        split_id = _require_string(task["split_id"], f"tasks[{index}].split_id")
        if task_id in task_ids:
            raise PackageValidationError(f"duplicate task_id: {task_id}")
        task_ids.add(task_id)
        split_ids.add(split_id)
        tasks.append({"task_id": task_id, "split_id": split_id})
    if not split_ids:
        raise PackageValidationError("no split IDs supplied")
    expected_task_count = LANE_SPECS[lane].get("expected_task_count")
    if expected_task_count is not None and len(tasks) != expected_task_count:
        raise PackageValidationError(
            f"manifest.tasks must contain the exact {lane} full suite of "
            f"{expected_task_count} tasks; got {len(tasks)}"
        )
    budget = manifest["budget_usd"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (int, float))
        or not math.isfinite(float(budget))
        or float(budget) < 0
    ):
        raise PackageValidationError("budget_usd must be a finite non-boolean number >= 0")
    deployment: dict[str, str]
    if isinstance(manifest["deployment_receipt"], dict):
        deployment = _verify_file(
            package_root, manifest["deployment_receipt"], "deployment_receipt"
        )
    else:
        raise PackageValidationError(
            "deployment_receipt must be an object bound by the provider grant"
        )
    bindings = {
        "assets_sha256": _canonical_sha256(sorted(assets, key=lambda item: item["path"])),
        "license_sha256": license_record["sha256"],
        "license_text_sha256": license_item["text_sha256"],
        "runtime_sha256": runtime["sha256"],
        "native_grader_sha256": grader["sha256"],
        "deployment_receipt_sha256": deployment["sha256"],
        "task_split_sha256": _canonical_sha256(sorted(tasks, key=lambda item: item["task_id"])),
        "task_count_sha256": _canonical_sha256(
            {
                "expected_task_count": expected_task_count,
                "task_count": len(tasks),
            }
        ),
    }
    grant = _verify_grant(
        manifest["provider_grant"],
        lane=lane,
        suite_id=LANE_SPECS[lane]["suite_id"],
        bindings=bindings,
        trust_root_document=trust_root_document,
    )
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "lane": lane,
        "suite_id": LANE_SPECS[lane]["suite_id"],
        "provider_grant": grant,
        "trust_root_sha256": trust_root_document["sha256"],
        "assets": assets,
        "license": license_record | {"text_sha256": license_item["text_sha256"]},
        "runtime": runtime,
        "native_grader": grader,
        "task_count": len(tasks),
        "expected_task_count": expected_task_count,
        "split_ids": sorted(split_ids),
        "budget_usd": float(budget),
        "deployment_receipt": deployment,
        "package_bindings": bindings,
        "validation": "VALIDATED_INPUTS_ONLY_NOT_A_SCORE_OR_LAUNCH_AUTHORIZATION",
    }


def _verified_exact_source_lanes(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the two exact-score assertions made by the audit boundary."""
    raw_lanes = status.get("lanes")
    if not isinstance(raw_lanes, list):
        raise PackageValidationError("terminal status.lanes must be an array")
    verified: list[dict[str, Any]] = []
    for lane in ("E1", "E11"):
        entries = [
            entry for entry in raw_lanes if isinstance(entry, dict) and entry.get("lane") == lane
        ]
        if len(entries) != 1:
            raise PackageValidationError(
                f"terminal status must contain exactly one {lane} exact-completion record"
            )
        entry = entries[0]
        if entry.get("benchmark_status") != "SCORED_EXACT_SUITE":
            raise PackageValidationError(
                f"{lane}: exact-completion claim requires benchmark_status SCORED_EXACT_SUITE"
            )
        score = entry.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise PackageValidationError(
                f"{lane}: exact-completion claim requires a finite non-boolean score"
            )
        verified.append(
            {
                "lane": lane,
                "benchmark_status": "SCORED_EXACT_SUITE",
                "score": float(score),
            }
        )
    return verified


def audit(
    status_path: Path = DEFAULT_STATUS, followup_path: Path = DEFAULT_FOLLOWUP
) -> dict[str, Any]:
    status = _load_json(status_path)
    followup = _load_json(followup_path)
    raw_lanes = status.get("lanes")
    if not isinstance(raw_lanes, list):
        raise PackageValidationError("terminal status.lanes must be an array")
    source_lanes = {entry.get("lane"): entry for entry in raw_lanes if isinstance(entry, dict)}
    expected_lanes = set(LANE_SPECS)
    exact_sources = _verified_exact_source_lanes(status)
    records: list[dict[str, Any]] = []
    for lane, spec in LANE_SPECS.items():
        source = source_lanes.get(lane)
        if source is None:
            raise PackageValidationError(f"terminal status lacks {lane}")
        evidence_class = classify_evidence(source.get("benchmark_status"))
        if evidence_class not in EVIDENCE_CLASSES:
            raise AssertionError(f"invalid internal class {evidence_class}")
        if evidence_class != spec["class"]:
            raise PackageValidationError(
                f"{lane}: source evidence class {evidence_class} conflicts with pinned inventory {spec['class']}"
            )
        paths = {
            kind: path
            for kind, path in (
                ("adapter", spec["adapter"]),
                ("runner", spec["runner"]),
                ("test", spec["test"]),
            )
        }
        records.append(
            {
                "lane": lane,
                "suite_id": spec["suite_id"],
                "evidence_class": evidence_class,
                "score": None,
                "historical_benchmark_status": source.get("benchmark_status"),
                "historical_work_status": source.get("work_status"),
                "required_inputs": spec["required"],
                "local_surfaces": [
                    {"kind": kind, "path": path, "exists": (REPO_ROOT / path).is_file()}
                    for kind, path in paths.items()
                ],
                "blocker": source.get("blocker"),
                "claim_boundary": "No partial, recovery, public, harness, readiness, or package-validation evidence is converted into a suite score.",
            }
        )
    if {entry["lane"] for entry in records} != expected_lanes:
        raise AssertionError("lane inventory drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _utc_now(),
        "mode": "READ_ONLY_AUDIT_NO_NETWORK_NO_SPEND_NO_LAUNCH",
        "supported_evidence_classes": sorted(EVIDENCE_CLASSES),
        "included_lanes": list(LANE_SPECS),
        "excluded_complete_lanes": ["E11"],
        "provider_access_granted": followup.get("provider_response_check", {}).get(
            "provider_access_granted", False
        ),
        "verified_exact_source_lanes": exact_sources,
        "lanes": records,
        "claim_boundary": (
            "E11 is the only complete exact lane inside the E2--E14 gate; E1 is also "
            "complete exact but is outside this gate's scope. This receipt reports readiness "
            "constraints only and all included scores are null."
        ),
    }


def write_atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temp = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="write a read-only readiness receipt")
    audit_parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    audit_parser.add_argument("--followup", type=Path, default=DEFAULT_FOLLOWUP)
    audit_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    package_parser = commands.add_parser(
        "validate-package", help="validate a provider immutable-input package"
    )
    package_parser.add_argument("--lane", required=True, choices=sorted(LANE_SPECS))
    package_parser.add_argument("--manifest", required=True, type=Path)
    package_parser.add_argument("--package-root", required=True, type=Path)
    package_parser.add_argument("--trust-roots", required=True, type=Path)
    package_parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            receipt = audit(args.status, args.followup)
            write_atomic_json(args.output, receipt)
        else:
            receipt = validate_package(
                args.lane, args.manifest, args.package_root, args.trust_roots
            )
            if args.output:
                write_atomic_json(args.output, receipt)
            else:
                print(json.dumps(receipt, indent=2, sort_keys=True))
    except PackageValidationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
