"""Offline boundary and receipt validation for the SWE-bench Pro evaluation.

This module deliberately does not load the dataset, build Docker images, run a
verifier, contact W&B/Hugging Face, or invoke Tinker.  It validates a caller's
already-materialized manifest and completed result receipt against the pinned
public SWE-bench Pro identity.  A public ``primary_eval`` manifest is never
silently promoted to a held-out result; a held-out claim needs its own explicit
receipt proof.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pavlov-swe-bench-pro-boundary-v1"
BENCHMARK_ID = "swe_bench_pro_eval"
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
DATASET_SPLIT = "test"
PUBLIC_TASK_COUNT = 731
PRIVATE_TASK_COUNT = 276
HELDOUT_TASK_COUNT = 858
TOTAL_TASK_COUNT = PUBLIC_TASK_COUNT + PRIVATE_TASK_COUNT + HELDOUT_TASK_COUNT
DATASET_URL = "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro"
OFFICIAL_EVAL_REPO_ID = "scaleapi/SWE-bench_Pro-os"
OFFICIAL_EVAL_REPO_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
OFFICIAL_EVAL_REPO_URL = "https://github.com/scaleapi/SWE-bench_Pro-os"
PUBLIC_LEADERBOARD_URL = "https://scale.com/leaderboard/swe_bench_pro_public"
PRIVATE_LEADERBOARD_URL = "https://labs.scale.com/leaderboard/swe_bench_pro_private"
PUBLIC_SUBSET = "public"
PRIVATE_SUBSET = "private"
HELDOUT_SUBSET = "heldout"
EVALUATION_CODE_LICENSE = "MIT"
EVALUATION_CODE_LICENSE_URL = (
    "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/"
    f"{OFFICIAL_EVAL_REPO_REVISION}/LICENSE"
)
DATASET_CARD_URL = (
    "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/"
    f"raw/{DATASET_REVISION}/README.md"
)
RUNTIME = "docker"
CONTAINER_IMAGE_REGISTRY = "jefzda/sweap-images"
PRIMARY_EVAL = "primary_eval"
RECEIPT_PROVEN_HELDOUT = "receipt_proven_heldout"

_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DOCKER_TAG_RE = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_INSTANCE_ID_MIN_LENGTH = 65
_INSTANCE_ID_MAX_LENGTH = 120
_REQUIRED_SOURCE_KEYS = (
    "dataset_id",
    "dataset_revision",
    "dataset_url",
    "split",
    "official_eval_repo_id",
    "official_eval_repo_revision",
    "official_eval_repo_url",
    "evaluation_code_license",
    "evaluation_code_license_url",
    "dataset_card_url",
    "dataset_card_license_declared",
    "public_leaderboard_url",
    "private_leaderboard_url",
    "public_task_count",
    "private_task_count",
    "heldout_task_count",
    "total_task_count",
)


class BoundaryValidationError(ValueError):
    """Raised when an SWE-bench Pro boundary or result is inadmissible."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKeyError("duplicate JSON object key")
    return dict(pairs)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise BoundaryValidationError("value is not strict JSON") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, _DuplicateKeyError, ValueError) as exc:
        raise BoundaryValidationError(f"cannot parse JSON receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise BoundaryValidationError("JSON receipt root must be an object")
    return value


def _same_json(left: Any, right: Any, path: str = "$") -> None:
    if type(left) is not type(right):
        raise BoundaryValidationError(f"{path} changed type")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise BoundaryValidationError(f"{path} keys differ")
        for key in sorted(left):
            _same_json(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list):
        if len(left) != len(right):
            raise BoundaryValidationError(f"{path} length differs")
        for index, (actual, expected) in enumerate(zip(left, right)):
            _same_json(actual, expected, f"{path}[{index}]")
        return
    if left != right:
        raise BoundaryValidationError(f"{path} scalar differs")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BoundaryValidationError(f"{path} must be an object")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BoundaryValidationError(f"{path} must be a non-empty string")
    return value


def _bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise BoundaryValidationError(f"{path} must be boolean")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BoundaryValidationError(f"{path} must be a non-negative integer")
    return value


def _finite_number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoundaryValidationError(f"{path} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise BoundaryValidationError(f"{path} must be finite") from exc
    if not math.isfinite(number):
        raise BoundaryValidationError(f"{path} must be finite")
    if minimum is not None and number < minimum:
        raise BoundaryValidationError(f"{path} must be >= {minimum}")
    return number


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> str:
    """Return the hash input used for immutable manifest fragments."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise BoundaryValidationError("manifest fragment is not canonical JSON") from exc


def compute_task_ids_sha256(task_ids: Sequence[str]) -> str:
    """Hash sorted task IDs with one terminal newline and no hidden ordering."""
    ids = [_string(task_id, "task_ids") for task_id in task_ids]
    if not ids:
        raise BoundaryValidationError("task IDs must be non-empty")
    if len(ids) != len(set(ids)):
        raise BoundaryValidationError("task IDs must be unique")
    canonical_ids = sorted(ids)
    return _sha256_bytes(("\n".join(canonical_ids) + "\n").encode("utf-8"))


def compute_split_manifest_sha256(
    *,
    split_name: str,
    role: str,
    subset: str,
    coverage: str,
    task_ids: Sequence[str],
    tasks: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the source-bound split manifest without its self-referential hashes."""
    body = {
        "benchmark_id": BENCHMARK_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": split_name,
        "role": role,
        "subset": subset,
        "coverage": coverage,
        "task_ids": sorted(task_ids),
        "tasks": list(tasks),
    }
    return _sha256_bytes(canonical_json(body).encode("utf-8"))


def _expected_source(*, split: str = DATASET_SPLIT) -> dict[str, Any]:
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_url": DATASET_URL,
        "split": split,
        "official_eval_repo_id": OFFICIAL_EVAL_REPO_ID,
        "official_eval_repo_revision": OFFICIAL_EVAL_REPO_REVISION,
        "official_eval_repo_url": OFFICIAL_EVAL_REPO_URL,
        "evaluation_code_license": EVALUATION_CODE_LICENSE,
        "evaluation_code_license_url": EVALUATION_CODE_LICENSE_URL,
        "dataset_card_url": DATASET_CARD_URL,
        # The pinned dataset card exposes schema/size metadata but no global
        # SPDX license.  Per-task license receipts are therefore mandatory;
        # this boolean records the upstream fact instead of inventing a
        # dataset-wide license.
        "dataset_card_license_declared": False,
        "public_leaderboard_url": PUBLIC_LEADERBOARD_URL,
        "private_leaderboard_url": PRIVATE_LEADERBOARD_URL,
        "public_task_count": PUBLIC_TASK_COUNT,
        "private_task_count": PRIVATE_TASK_COUNT,
        "heldout_task_count": HELDOUT_TASK_COUNT,
        "total_task_count": TOTAL_TASK_COUNT,
    }


def canonical_source_identity(*, split: str = DATASET_SPLIT) -> dict[str, Any]:
    """Return a detached official source identity for a boundary manifest."""
    if split != DATASET_SPLIT:
        raise BoundaryValidationError("the pinned public source identity only exposes split=test")
    return copy.deepcopy(_expected_source(split=split))


def _validate_source(source: Mapping[str, Any]) -> dict[str, Any]:
    expected = _expected_source(split=DATASET_SPLIT)
    for key in _REQUIRED_SOURCE_KEYS:
        if key not in source:
            raise BoundaryValidationError(f"source.{key} is required")
    _same_json(dict(source), expected, "$.source")
    return dict(source)


def _validate_hash(value: Any, path: str) -> str:
    value = _string(value, path)
    if not _HEX64_RE.fullmatch(value):
        raise BoundaryValidationError(f"{path} must be a lowercase SHA-256 hex digest")
    return value


def _validate_instance_id(value: Any, path: str) -> str:
    value = _string(value, path)
    if not _INSTANCE_ID_RE.fullmatch(value) or not (
        _INSTANCE_ID_MIN_LENGTH <= len(value) <= _INSTANCE_ID_MAX_LENGTH
    ):
        raise BoundaryValidationError(f"{path} is not a deterministic task identifier")
    return value


def _optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _official_test_names(value: Any, path: str) -> list[str]:
    """Parse the upstream list-as-string fields without executing code."""
    raw = _string(value, path)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise BoundaryValidationError(f"{path} is not a literal test-name list") from exc
    if not isinstance(parsed, list) or not parsed:
        raise BoundaryValidationError(f"{path} must encode a non-empty list")
    names: list[str] = []
    for index, name in enumerate(parsed):
        names.append(_string(name, f"{path}[{index}]"))
    if len(names) != len(set(names)):
        raise BoundaryValidationError(f"{path} must not contain duplicate test names")
    return names


def _validate_task(task: Mapping[str, Any], index: int) -> dict[str, Any]:
    path = f"tasks[{index}]"
    required = (
        "instance_id",
        "repo",
        "base_commit",
        "dockerhub_tag",
        "container_digest",
        "license_receipt",
        "problem_statement",
        "repo_language",
        "issue_specificity",
        "issue_categories",
        "requirements",
        "interface",
        "before_repo_set_cmd",
        "selected_test_files_to_run",
        "fail_to_pass",
        "pass_to_pass",
        "artifact_contract",
        "verifier_contract",
    )
    for key in required:
        if key not in task:
            raise BoundaryValidationError(f"{path}.{key} is required")
    instance_id = _validate_instance_id(task["instance_id"], f"{path}.instance_id")
    repo = _string(task["repo"], f"{path}.repo")
    if not _REPO_ID_RE.fullmatch(repo):
        raise BoundaryValidationError(f"{path}.repo must be owner/name")
    base_commit = _string(task["base_commit"], f"{path}.base_commit")
    if not _HEX40_RE.fullmatch(base_commit):
        raise BoundaryValidationError(f"{path}.base_commit must be immutable")
    dockerhub_tag = _string(task["dockerhub_tag"], f"{path}.dockerhub_tag")
    if not _DOCKER_TAG_RE.fullmatch(dockerhub_tag):
        raise BoundaryValidationError(f"{path}.dockerhub_tag is not sanitized")
    container_digest = _string(task["container_digest"], f"{path}.container_digest")
    if not _SHA256_RE.fullmatch(container_digest):
        raise BoundaryValidationError(f"{path}.container_digest must be immutable")

    license_receipt = _mapping(task["license_receipt"], f"{path}.license_receipt")
    for key in ("spdx", "source_url", "source_revision", "receipt_sha256"):
        if key not in license_receipt:
            raise BoundaryValidationError(f"{path}.license_receipt.{key} is required")
    _string(license_receipt["spdx"], f"{path}.license_receipt.spdx")
    source_url = _string(license_receipt["source_url"], f"{path}.license_receipt.source_url")
    if not source_url.startswith("https://"):
        raise BoundaryValidationError(f"{path}.license_receipt.source_url must be HTTPS")
    source_revision = _string(
        license_receipt["source_revision"], f"{path}.license_receipt.source_revision"
    )
    if not _HEX40_RE.fullmatch(source_revision):
        raise BoundaryValidationError(f"{path}.license_receipt.source_revision must be immutable")
    _validate_hash(license_receipt["receipt_sha256"], f"{path}.license_receipt.receipt_sha256")

    for key in ("problem_statement", "repo_language", "issue_specificity", "issue_categories"):
        _string(task[key], f"{path}.{key}")
    _optional_text(task["requirements"], f"{path}.requirements")
    _optional_text(task["interface"], f"{path}.interface")
    _string(task["before_repo_set_cmd"], f"{path}.before_repo_set_cmd")
    for key in ("selected_test_files_to_run", "fail_to_pass", "pass_to_pass"):
        _official_test_names(task[key], f"{path}.{key}")

    artifact = _mapping(task["artifact_contract"], f"{path}.artifact_contract")
    if artifact.get("kind") != "repository_patch" or artifact.get("format") != "unified_diff":
        raise BoundaryValidationError(f"{path}.artifact_contract is not native patch output")
    if artifact.get("required") is not True:
        raise BoundaryValidationError(f"{path}.artifact_contract.required must be true")

    verifier = _mapping(task["verifier_contract"], f"{path}.verifier_contract")
    if verifier.get("kind") != "swe_bench_pro_official":
        raise BoundaryValidationError(f"{path}.verifier_contract is not official")
    if verifier.get("resolve_rule") != "fail_to_pass_and_pass_to_pass":
        raise BoundaryValidationError(f"{path}.verifier_contract has an invalid resolve rule")
    if verifier.get("requires_native_tests") is not True:
        raise BoundaryValidationError(f"{path}.verifier_contract must require native tests")
    return dict(task)


def _validate_environment(environment: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "runtime",
        "container_image_registry",
        "artifact_contract",
        "verifier_contract",
    )
    for key in required:
        if key not in environment:
            raise BoundaryValidationError(f"environment.{key} is required")
    if environment.get("runtime") != RUNTIME:
        raise BoundaryValidationError("environment.runtime must be Docker")
    if environment.get("container_image_registry") != CONTAINER_IMAGE_REGISTRY:
        raise BoundaryValidationError("environment image registry is not official")
    artifact = _mapping(environment["artifact_contract"], "environment.artifact_contract")
    if artifact.get("kind") != "repository_patch" or artifact.get("native_artifact") != "git_diff":
        raise BoundaryValidationError("environment artifact contract is not native")
    verifier = _mapping(environment["verifier_contract"], "environment.verifier_contract")
    if verifier.get("kind") != "swe_bench_pro_official":
        raise BoundaryValidationError("environment verifier is not official")
    if verifier.get("resolve_rule") != "fail_to_pass_and_pass_to_pass":
        raise BoundaryValidationError("environment verifier rule is not strict")
    return dict(environment)


def _validate_heldout_proof(
    value: Any,
    *,
    task_ids_hash: str,
    manifest_hash: str,
) -> dict[str, Any]:
    proof = _mapping(value, "heldout_proof")
    required = (
        "status",
        "subset",
        "source_identity",
        "access_receipt",
        "decontamination_receipt",
        "license_receipt",
        "task_ids_sha256",
        "split_manifest_sha256",
    )
    for key in required:
        if key not in proof:
            raise BoundaryValidationError(f"heldout_proof.{key} is required")
    proof_status = _string(proof.get("status"), "heldout_proof.status")
    if proof_status != "receipt_proven":
        raise BoundaryValidationError("heldout_proof.status must be receipt_proven")
    proof_subset = _string(proof.get("subset"), "heldout_proof.subset")
    if proof_subset not in {HELDOUT_SUBSET, PRIVATE_SUBSET}:
        raise BoundaryValidationError("heldout_proof.subset must be heldout or private")
    source = _mapping(proof["source_identity"], "heldout_proof.source_identity")
    for key in ("source_id", "source_url", "source_revision", "split"):
        if key not in source:
            raise BoundaryValidationError(f"heldout_proof.source_identity.{key} is required")
    _string(source["source_id"], "heldout_proof.source_identity.source_id")
    source_url = _string(source["source_url"], "heldout_proof.source_identity.source_url")
    if not source_url.startswith("https://"):
        raise BoundaryValidationError("heldout source URL must be HTTPS")
    source_revision = _string(
        source["source_revision"], "heldout_proof.source_identity.source_revision"
    )
    if not _HEX40_RE.fullmatch(source_revision):
        raise BoundaryValidationError("heldout source revision must be immutable")
    source_split = _string(source["split"], "heldout_proof.source_identity.split")
    if source_split != proof_subset:
        raise BoundaryValidationError(
            "heldout source split must match its explicit heldout/private subset"
        )
    if source["source_id"] == DATASET_ID or source_url == DATASET_URL:
        raise BoundaryValidationError("public dataset identity cannot prove heldout/private access")
    for key in ("access_receipt", "decontamination_receipt", "license_receipt"):
        receipt = _mapping(proof[key], f"heldout_proof.{key}")
        for receipt_key in ("source_revision", "receipt_sha256"):
            if receipt_key not in receipt:
                raise BoundaryValidationError(f"heldout_proof.{key}.{receipt_key} is required")
        receipt_revision = _string(receipt["source_revision"], f"heldout_proof.{key}.source_revision")
        if not _HEX40_RE.fullmatch(receipt_revision):
            raise BoundaryValidationError(f"heldout_proof.{key}.source_revision is not immutable")
        if receipt_revision != source_revision:
            raise BoundaryValidationError(
                f"heldout_proof.{key}.source_revision disagrees with source identity"
            )
        _validate_hash(receipt["receipt_sha256"], f"heldout_proof.{key}.receipt_sha256")
    if proof["task_ids_sha256"] != task_ids_hash:
        raise BoundaryValidationError("heldout proof task hash disagrees")
    if proof["split_manifest_sha256"] != manifest_hash:
        raise BoundaryValidationError("heldout proof manifest hash disagrees")
    normalized = dict(proof)
    normalized["status"] = proof_status
    normalized["subset"] = proof_subset
    return normalized


def validate_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a source-bound SWE-bench Pro task manifest offline."""
    if not isinstance(payload, Mapping):
        raise BoundaryValidationError("manifest must be an object")
    actual = _strict_json_copy(dict(payload))
    if actual.get("schema_version") != SCHEMA_VERSION:
        raise BoundaryValidationError("manifest schema version is unsupported")
    if actual.get("benchmark_id") != BENCHMARK_ID:
        raise BoundaryValidationError("manifest is not SWE-bench Pro")

    split = _mapping(actual.get("split"), "split")
    role = _string(split.get("role"), "split.role")
    if role not in {PRIMARY_EVAL, RECEIPT_PROVEN_HELDOUT}:
        raise BoundaryValidationError("split.role must be primary_eval or receipt_proven_heldout")
    split_name = _string(split.get("name"), "split.name")
    subset = _string(split.get("subset"), "split.subset")
    if role == PRIMARY_EVAL and split_name != DATASET_SPLIT:
        raise BoundaryValidationError("primary_eval must use the pinned public test split")
    if role == PRIMARY_EVAL and subset != PUBLIC_SUBSET:
        raise BoundaryValidationError("primary_eval must identify the public subset")
    if role == RECEIPT_PROVEN_HELDOUT and split_name != "heldout_receipt":
        raise BoundaryValidationError("held-out claims require an explicit receipt split")
    if role == RECEIPT_PROVEN_HELDOUT and subset not in {HELDOUT_SUBSET, PRIVATE_SUBSET}:
        raise BoundaryValidationError("held-out claims must identify heldout or private subset")
    coverage = _string(split.get("coverage"), "split.coverage")
    if coverage not in {"sampled_public", "full_public", "receipt_proven_heldout"}:
        raise BoundaryValidationError("split.coverage is unsupported")
    if role == RECEIPT_PROVEN_HELDOUT and coverage != "receipt_proven_heldout":
        raise BoundaryValidationError("held-out claims require receipt_proven_heldout coverage")
    if role == PRIMARY_EVAL and coverage == "receipt_proven_heldout":
        raise BoundaryValidationError("primary_eval cannot use held-out coverage")

    source = _validate_source(_mapping(actual.get("source"), "source"))
    tasks_value = actual.get("tasks")
    if not isinstance(tasks_value, list) or not tasks_value:
        raise BoundaryValidationError("tasks must be a non-empty list")
    tasks = [
        _validate_task(_mapping(task, f"tasks[{index}]"), index)
        for index, task in enumerate(tasks_value)
    ]
    task_ids = [_validate_instance_id(task["instance_id"], "task.instance_id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise BoundaryValidationError("task instance IDs must be unique")
    if task_ids != sorted(task_ids):
        raise BoundaryValidationError("tasks must be sorted by instance_id")
    declared_task_ids = split.get("task_ids")
    if not isinstance(declared_task_ids, list) or not declared_task_ids:
        raise BoundaryValidationError("split.task_ids must be a non-empty list")
    declared_task_ids = [
        _validate_instance_id(value, "split.task_ids") for value in declared_task_ids
    ]
    if declared_task_ids != sorted(declared_task_ids):
        raise BoundaryValidationError("split.task_ids must be sorted deterministically")
    if declared_task_ids != sorted(task_ids):
        raise BoundaryValidationError("split.task_ids must exactly match task rows")
    task_ids_hash = compute_task_ids_sha256(declared_task_ids)
    if split.get("task_ids_sha256") != task_ids_hash:
        raise BoundaryValidationError("split.task_ids_sha256 does not match task IDs")
    manifest_hash = compute_split_manifest_sha256(
        split_name=split_name,
        role=role,
        subset=subset,
        coverage=coverage,
        task_ids=declared_task_ids,
        tasks=tasks,
    )
    if split.get("split_manifest_sha256") != manifest_hash:
        raise BoundaryValidationError("split.split_manifest_sha256 does not match manifest")
    if coverage in {"sampled_public", "full_public"} and len(tasks) > PUBLIC_TASK_COUNT:
        raise BoundaryValidationError("public coverage cannot exceed the pinned 731 public tasks")
    if coverage == "full_public" and len(tasks) != PUBLIC_TASK_COUNT:
        raise BoundaryValidationError("full_public must contain the pinned 731 public tasks")
    if role == RECEIPT_PROVEN_HELDOUT:
        heldout_proof = _validate_heldout_proof(
            actual.get("heldout_proof"),
            task_ids_hash=task_ids_hash,
            manifest_hash=manifest_hash,
        )
        if heldout_proof["subset"] != subset:
            raise BoundaryValidationError("heldout proof subset disagrees with split")
        maximum_tasks = (
            HELDOUT_TASK_COUNT if subset == HELDOUT_SUBSET else PRIVATE_TASK_COUNT
        )
        if len(tasks) > maximum_tasks:
            raise BoundaryValidationError(
                f"{subset} receipt cannot exceed its pinned task count ({maximum_tasks})"
            )
    else:
        if actual.get("heldout_proof") is not None:
            raise BoundaryValidationError("primary_eval cannot carry a held-out proof")
        heldout_proof = None

    environment = _validate_environment(_mapping(actual.get("environment"), "environment"))
    normalized = copy.deepcopy(actual)
    normalized["source"] = source
    normalized["tasks"] = tasks
    normalized["task_ids_sha256"] = task_ids_hash
    normalized["split_manifest_sha256"] = manifest_hash
    normalized["environment"] = environment
    if heldout_proof is not None:
        normalized["heldout_proof"] = heldout_proof
    return normalized


def _extract_id(payload: Mapping[str, Any], *, kind: str) -> str:
    if kind == "wandb":
        values = [payload.get("wandb_run_id"), payload.get("wandb_id")]
        nested = payload.get("wandb")
    else:
        values = [payload.get("tinker_run_id"), payload.get("tinker_id")]
        nested = payload.get("tinker")
    if isinstance(nested, Mapping):
        values.extend((nested.get("run_id"), nested.get("id")))
    present = [value for value in values if value is not None]
    if not present:
        raise BoundaryValidationError(f"{kind} run ID is required")
    values = [_string(value, f"{kind} run ID") for value in present]
    if any(value != values[0] for value in values[1:]):
        raise BoundaryValidationError(f"{kind} run ID aliases disagree")
    return values[0]


def _validate_hf_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    required = ("repo_id", "revision", "commit_sha", "repo_url", "revision_url", "commit_url")
    for key in required:
        if key not in receipt:
            raise BoundaryValidationError(f"hf_receipt.{key} is required")
    repo_id = _string(receipt["repo_id"], "hf_receipt.repo_id")
    revision = _string(receipt["revision"], "hf_receipt.revision")
    commit_sha = _string(receipt["commit_sha"], "hf_receipt.commit_sha")
    if not _REPO_ID_RE.fullmatch(repo_id) or repo_id.startswith("xlam/"):
        raise BoundaryValidationError("hf_receipt.repo_id is not sanitized")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", revision):
        raise BoundaryValidationError("hf_receipt.revision is not sanitized")
    if not _HEX40_RE.fullmatch(commit_sha):
        raise BoundaryValidationError("hf_receipt.commit_sha is not immutable")
    repo_url = f"https://huggingface.co/{repo_id}"
    if receipt["repo_url"] != repo_url:
        raise BoundaryValidationError("hf_receipt.repo_url is not canonical")
    if receipt["revision_url"] != f"{repo_url}/tree/{revision}":
        raise BoundaryValidationError("hf_receipt.revision_url is not canonical")
    if receipt["commit_url"] != f"{repo_url}/commit/{commit_sha}":
        raise BoundaryValidationError("hf_receipt.commit_url is not canonical")
    return dict(receipt)


def _validate_task_results(
    value: Any, tasks: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(value, list) or len(value) != len(tasks):
        raise BoundaryValidationError("result.task_results must cover every manifest task")
    expected_by_id = {task["instance_id"]: task for task in tasks}
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    resolved_count = 0
    for index, raw in enumerate(value):
        item = _mapping(raw, f"result.task_results[{index}]")
        for key in ("instance_id", "status", "tests", "container_digest", "artifact_sha256"):
            if key not in item:
                raise BoundaryValidationError(f"result.task_results[{index}].{key} is required")
        instance_id = _validate_instance_id(
            item["instance_id"], f"result.task_results[{index}].instance_id"
        )
        if instance_id in seen or instance_id not in expected_by_id:
            raise BoundaryValidationError("result.task_results IDs must match manifest exactly")
        seen.add(instance_id)
        task = expected_by_id[instance_id]
        if item["container_digest"] != task["container_digest"]:
            raise BoundaryValidationError("task result container digest disagrees with manifest")
        artifact_sha = _string(
            item["artifact_sha256"], f"result.task_results[{index}].artifact_sha256"
        )
        if not _HEX64_RE.fullmatch(artifact_sha):
            raise BoundaryValidationError("task result artifact hash is not immutable")
        status = _string(item["status"], f"result.task_results[{index}].status")
        if status not in {"resolved", "unresolved"}:
            raise BoundaryValidationError("task result status is unsupported")
        tests = item["tests"]
        if not isinstance(tests, list) or not tests:
            raise BoundaryValidationError("task result tests must be non-empty")
        passed: set[str] = set()
        observed_names: set[str] = set()
        normalized_tests: list[dict[str, Any]] = []
        for test_index, raw_test in enumerate(tests):
            test = _mapping(raw_test, f"result.task_results[{index}].tests[{test_index}]")
            name = _string(test.get("name"), "test result name")
            if name in observed_names:
                raise BoundaryValidationError("task result tests must not contain duplicate names")
            observed_names.add(name)
            test_status = _string(test.get("status"), "test result status")
            if test_status not in {"PASSED", "FAILED", "ERROR", "SKIPPED"}:
                raise BoundaryValidationError("test result status is unsupported")
            if test_status == "PASSED":
                passed.add(name)
            normalized_tests.append(dict(test))
        required_tests = set(_official_test_names(task["fail_to_pass"], "fail_to_pass"))
        required_tests.update(_official_test_names(task["pass_to_pass"], "pass_to_pass"))
        if not required_tests <= observed_names:
            raise BoundaryValidationError(
                "task result must include evidence for every fail-to-pass/pass-to-pass test"
            )
        expected_resolved = required_tests <= passed
        if (status == "resolved") != expected_resolved:
            raise BoundaryValidationError("task result status disagrees with native test evidence")
        if status == "resolved":
            resolved_count += 1
        normalized_item = dict(item)
        normalized_item["tests"] = normalized_tests
        normalized.append(normalized_item)
    if [item["instance_id"] for item in normalized] != sorted(seen):
        raise BoundaryValidationError("result.task_results must be sorted by instance_id")
    return normalized, resolved_count


def _validate_result_counts(result: Mapping[str, Any], task_count: int) -> None:
    resolved = _nonnegative_int(result.get("resolved_count"), "result.resolved_count")
    if resolved > task_count:
        raise BoundaryValidationError("result.resolved_count exceeds task count")
    reported_count = _nonnegative_int(result.get("task_count"), "result.task_count")
    if reported_count != task_count:
        raise BoundaryValidationError("result.task_count disagrees with manifest")
    rate = _finite_number(result.get("resolve_rate"), "result.resolve_rate", minimum=0.0)
    expected = resolved / task_count
    if rate > 1.0 or not math.isclose(rate, expected, rel_tol=0.0, abs_tol=1e-12):
        raise BoundaryValidationError("result.resolve_rate disagrees with counts")


def validate_result_receipt(
    payload: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a completed SWE-bench Pro result against a validated manifest."""
    checked_manifest = validate_manifest(manifest)
    if not isinstance(payload, Mapping):
        raise BoundaryValidationError("result receipt must be an object")
    actual = _strict_json_copy(dict(payload))
    if actual.get("schema_version") != SCHEMA_VERSION:
        raise BoundaryValidationError("result schema version is unsupported")
    if actual.get("benchmark_id") != BENCHMARK_ID:
        raise BoundaryValidationError("result is not SWE-bench Pro")
    if actual.get("status") != "completed":
        raise BoundaryValidationError("result must be completed")
    result_role = actual.get("split_role")
    manifest_role = checked_manifest["split"]["role"]
    if result_role != manifest_role:
        raise BoundaryValidationError("result split role disagrees with manifest")
    split = checked_manifest["split"]
    for key in ("task_ids_sha256", "split_manifest_sha256"):
        if actual.get(key) != split[key]:
            raise BoundaryValidationError(f"result.{key} disagrees with manifest")
    task_ids = actual.get("evaluated_task_ids")
    expected_ids = split["task_ids"]
    if not isinstance(task_ids, list) or task_ids != expected_ids:
        raise BoundaryValidationError("result.evaluated_task_ids must exactly match manifest")
    task_results, observed_resolved_count = _validate_task_results(
        actual.get("task_results"), checked_manifest["tasks"]
    )
    _validate_result_counts(actual, len(expected_ids))
    if actual["resolved_count"] != observed_resolved_count:
        raise BoundaryValidationError("result.resolved_count disagrees with task receipts")
    wandb_run_id = _extract_id(actual, kind="wandb")
    tinker_run_id = _extract_id(actual, kind="tinker")
    hf = _validate_hf_receipt(_mapping(actual.get("hf_receipt"), "hf_receipt"))

    environment_receipt = _mapping(actual.get("environment_receipt"), "environment_receipt")
    if environment_receipt.get("runtime") != RUNTIME:
        raise BoundaryValidationError("environment_receipt.runtime must be Docker")
    digests = {task["container_digest"] for task in checked_manifest["tasks"]}
    if environment_receipt.get("container_digests") != sorted(digests):
        raise BoundaryValidationError("environment receipt digests disagree with manifest")
    artifact = _mapping(actual.get("artifact_receipt"), "artifact_receipt")
    if artifact.get("kind") != "repository_patch" or artifact.get("format") != "unified_diff":
        raise BoundaryValidationError("artifact receipt is not a native patch")
    artifact_sha = _string(artifact.get("sha256"), "artifact_receipt.sha256")
    if not _HEX64_RE.fullmatch(artifact_sha):
        raise BoundaryValidationError("artifact receipt hash is not immutable")
    verifier = _mapping(actual.get("verifier_receipt"), "verifier_receipt")
    if verifier.get("kind") != "swe_bench_pro_official":
        raise BoundaryValidationError("verifier receipt is not official")
    if (
        verifier.get("fail_to_pass_passed") is not True
        or verifier.get("pass_to_pass_passed") is not True
    ):
        raise BoundaryValidationError("native fail-to-pass and pass-to-pass receipts are required")
    if manifest_role == RECEIPT_PROVEN_HELDOUT:
        if checked_manifest.get("heldout_proof", {}).get("status") != "receipt_proven":
            raise BoundaryValidationError("held-out result lacks receipt proof")
    elif actual.get("heldout_claim"):
        raise BoundaryValidationError("primary_eval cannot be claimed as held-out")

    normalized = copy.deepcopy(actual)
    normalized["wandb_run_id"] = wandb_run_id
    normalized["tinker_run_id"] = tinker_run_id
    normalized["hf_receipt"] = hf
    normalized["task_results"] = task_results
    return normalized


def validate_evaluation_boundary(
    manifest: Mapping[str, Any], result: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate a manifest and, when supplied, its completed result receipt."""
    checked = validate_manifest(manifest)
    if result is not None:
        checked["result"] = validate_result_receipt(result, checked)
    return checked


def parse_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(_load_json(Path(path)))


def parse_result(path: str | Path, manifest: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    checked_manifest = (
        parse_manifest(manifest)
        if isinstance(manifest, (str, Path))
        else validate_manifest(manifest)
    )
    return validate_result_receipt(_load_json(Path(path)), checked_manifest)


# Compatibility aliases for small local gate scripts.
validate_swe_bench_pro_manifest = validate_manifest
validate_swe_bench_pro_result = validate_result_receipt
parse_boundary = parse_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = parse_manifest(args.manifest)
        result = parse_result(args.result, manifest) if args.result else None
    except BoundaryValidationError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1
    output = {
        "status": "accepted",
        "benchmark_id": BENCHMARK_ID,
        "split_role": manifest["split"]["role"],
        "task_count": len(manifest["tasks"]),
    }
    if result is not None:
        output["result_status"] = result["status"]
        output["resolve_rate"] = result["resolve_rate"]
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
