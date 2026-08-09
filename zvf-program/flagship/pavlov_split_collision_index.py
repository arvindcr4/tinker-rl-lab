#!/usr/bin/env python3
"""Build a deterministic, metadata-only Pavlov split collision index.

The portfolio registry proves that the expected 12 training and 14
``primary_eval`` suite manifests are present.  This module provides a global
owner map over their ordered task digests and over separately supplied
component manifests (including xLAM).  It never loads a dataset, contacts a
provider, launches a job, or copies prompts/targets into the index.

An index can be emitted in a blocked state so every collision and contract
violation is inspectable at once.  ``verify_collision_index`` is the strict
gate for callers that require a collision-free, receipt-backed index.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "pavlov-split-collision-index-v1"
TRAIN_ROLE = "train"
PRIMARY_EVAL_ROLE = "primary_eval"
COMPONENT_SCOPE = "component"
PORTFOLIO_SCOPE = "portfolio"
XLAM_SUITE_ID = "pavlov_xlam"
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
_TASK_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RAW_CONTENT_KEYS = {"prompt", "prompts", "target", "targets"}

# These are intentionally literal and sorted.  Contract cardinality fields
# and caller-provided counts are not a source of truth for the collision gate.
FROZEN_TRAIN_SUITE_IDS = (
    "agentdojo_train",
    "api_bank_rlvr_train",
    "bfcl_train",
    "browsergym_train",
    "crafter_train",
    "openr1_math_train",
    "openreward_train",
    "rtlcoder_train",
    "scienceworld_train",
    "swe_gym_train",
    "unix_ctf_train",
    "visual_app_train",
)
FROZEN_PRIMARY_EVAL_SUITE_IDS = (
    "agentharm_eval",
    "apex_agents_eval",
    "appbench_eval",
    "banker_toolbench_eval",
    "binaryaudit_eval",
    "frontier_swe_eval",
    "frontiermath_eval",
    "lifescibench_eval",
    "mle_bench_eval",
    "openreward_games_eval",
    "sdab_eval",
    "swe_bench_pro_eval",
    "verilog_eval",
    "webbench_eval",
)
EXPECTED_TRAIN_SUITE_IDS = FROZEN_TRAIN_SUITE_IDS
EXPECTED_PRIMARY_EVAL_SUITE_IDS = FROZEN_PRIMARY_EVAL_SUITE_IDS
FROZEN_PORTFOLIO_SUITE_IDS = frozenset(
    FROZEN_TRAIN_SUITE_IDS + FROZEN_PRIMARY_EVAL_SUITE_IDS
)
_EXPECTED_ROLE_BY_ID = {
    **{suite_id: TRAIN_ROLE for suite_id in FROZEN_TRAIN_SUITE_IDS},
    **{suite_id: PRIMARY_EVAL_ROLE for suite_id in FROZEN_PRIMARY_EVAL_SUITE_IDS},
}

_REQUIRED_PORTFOLIO_RECEIPTS = (
    "revision",
    "license",
    "container",
    "decontamination",
    "verifier",
    "split_manifest",
)
_REQUIRED_COMPONENT_RECEIPTS = (
    "revision",
    "license",
    "container",
    "decontamination",
    "split_manifest",
)
_RECEIPT_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "nil",
    "pending",
    "placeholder",
    "todo",
    "tbd",
    "unset",
    "unrecorded",
    "unknown",
    "not_available",
    "not-applicable",
    "n/a",
    "na",
    "not_provided",
    "not provided",
    "to_be_pinned_before_paid_runs",
    "to be pinned",
}
_MUTABLE_RECEIPT_MARKERS = {
    "latest",
    "main",
    "master",
    "head",
    "tip",
    "pending",
    "current",
    "branch",
}
_IMMUTABLE_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMMUTABLE_DIGEST_URI = re.compile(
    r"(?:^|[@/#:])sha256:[0-9a-f]{64}(?:$|[/?#])",
    re.IGNORECASE,
)


class SplitCollisionIndexError(ValueError):
    """Raised when a collision index cannot pass its strict gate."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _aggregate_hash(task_hashes: Sequence[str]) -> str:
    return _sha256("\n".join(task_hashes))


def _read_json(value: str | Path) -> Any:
    try:
        return json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitCollisionIndexError(f"cannot read JSON {value!s}: {exc}") from exc


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _assert_metadata_only(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_CONTENT_KEYS:
                raise SplitCollisionIndexError(
                    f"{path} contains raw {str(key).lower()} content; collision index is metadata-only"
                )
            _assert_metadata_only(child, f"{path}.{key}")
    elif _is_sequence(value):
        for index, child in enumerate(value):
            _assert_metadata_only(child, f"{path}[{index}]")


def _looks_placeholder(value: Any) -> bool:
    if value is None or not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    if normalized in _RECEIPT_PLACEHOLDERS:
        return True
    return normalized.startswith(
        (
            "todo:",
            "tbd:",
            "pending:",
            "placeholder:",
            "unrecorded:",
            "unset:",
            "to_be_pinned",
        )
    ) or any(
        token in normalized
        for token in ("placeholder", "unrecorded", "not provided", "pending", "todo", "unset")
    )


def _receipt_is_immutable(value: Any) -> bool:
    """Accept receipt references, while rejecting prose and mutable aliases."""

    if _looks_placeholder(value) or not isinstance(value, str):
        return False
    reference = value.strip()
    lowered = reference.lower()
    if any(character.isspace() for character in reference):
        return False
    parts = {part for part in re.split(r"[/:?#._@-]+", lowered) if part}
    if parts.intersection(_MUTABLE_RECEIPT_MARKERS):
        return False
    if _IMMUTABLE_SHA256.fullmatch(lowered) or _IMMUTABLE_DIGEST_URI.search(lowered):
        return True
    if lowered.startswith((
        "receipt://",
        "receipt:",
        "urn:receipt:",
        "hf://",
        "oci://",
        "git://",
        "file://",
        "https://",
        "http://",
    )):
        # A URI with a stable scheme and no mutable marker is a receipt
        # reference.  The immutable revision itself is checked separately.
        return len(reference.split(":", 1)[1].strip("/")) >= 2
    return False


def _extract_revision(manifest: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    values = [
        manifest[key]
        for key in ("revision", "dataset_revision", "immutable_revision", "revision_sha")
        if key in manifest
    ]
    errors: list[str] = []
    if not values:
        return None, ["missing immutable revision"]
    if any(not isinstance(value, str) for value in values) or len(set(values)) != 1:
        return None, ["conflicting revision inputs"]
    revision = values[0]
    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        return None, ["revision must be an immutable 40-character lower-case commit SHA"]
    return revision, errors


def _receipt_source(manifest: Mapping[str, Any]) -> dict[str, Any]:
    source: dict[str, Any] = {}
    for field in ("receipt_refs", "provenance_receipts", "receipts", "runtime_receipts"):
        value = manifest.get(field)
        if isinstance(value, Mapping):
            source.update(value)
    aliases = {
        "revision": ("revision_receipt_ref", "dataset_revision_receipt_ref", "immutable_revision_receipt_ref"),
        "license": ("license_receipt_ref", "license_ref"),
        "container": ("container_receipt_ref", "container_runtime_receipt_ref", "container_ref", "runtime_ref"),
        "decontamination": ("decontamination_receipt_ref", "decontamination_ref"),
        "verifier": ("verifier_receipt_ref", "verifier_ref", "verification_receipt_ref"),
        "split_manifest": ("split_manifest_receipt_ref", "task_hash_receipt_ref", "split_manifest_ref"),
    }
    for canonical, keys in aliases.items():
        for key in keys:
            if key in manifest and canonical not in source:
                source[canonical] = manifest[key]
    return source


def _extract_receipts(
    manifest: Mapping[str, Any],
    *,
    required: Sequence[str],
) -> tuple[dict[str, str], list[str]]:
    source = _receipt_source(manifest)
    receipts: dict[str, str] = {}
    errors: list[str] = []
    for key in required:
        value = source.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"missing {key} receipt")
            continue
        normalized = value.strip()
        if not _receipt_is_immutable(normalized):
            errors.append(f"placeholder or mutable {key} receipt")
            continue
        receipts[key] = normalized
    # A receipt supplied under an optional field must not smuggle a placeholder
    # through just because the field is not required for this scope.
    for key, value in source.items():
        if key in receipts or key in required:
            continue
        if key.endswith("receipt") or key.endswith("receipt_ref") or key in {"held_out", "runtime"}:
            if value is not None and not _receipt_is_immutable(value):
                errors.append(f"placeholder or mutable {key} receipt")
    return receipts, errors


def _scalar_roles(manifest: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    values = [
        str(manifest[key])
        for key in ("suite_role", "portfolio_split_role", "role", "split_role")
        if isinstance(manifest.get(key), str)
    ]
    unique = sorted(set(values))
    if not unique:
        return None, ["manifest must declare an explicit role"]
    if len(unique) != 1:
        return None, ["manifest declares conflicting role inputs"]
    return unique[0], []


def _task_hash_lists(
    manifest: Mapping[str, Any],
    *,
    scope: str,
    explicit_role: str | None,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Extract role-labelled ordered digest lists without inferring portfolio roles."""

    errors: list[str] = []
    value = manifest.get("task_hashes")
    raw_lists: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, hashes in value.items():
            key_text = str(key)
            if key_text in ("test", "eval", "primary_eval"):
                role = PRIMARY_EVAL_ROLE
            elif key_text == "train":
                role = TRAIN_ROLE
            else:
                role = key_text
            raw_lists.append((role, hashes))
    elif _is_sequence(value):
        # The public index stores role-labelled lists so it can be verified
        # without retaining the original input shape.
        if value and all(isinstance(item, Mapping) and "task_hashes" in item for item in value):
            for item in value:
                raw_lists.append((str(item.get("role", "<missing-role>")), item["task_hashes"]))
        else:
            raw_lists.append((explicit_role or "<missing-role>", value))
    else:
        aliases = {
            TRAIN_ROLE: ("train_task_hashes",),
            PRIMARY_EVAL_ROLE: ("primary_eval_task_hashes", "test_task_hashes", "eval_task_hashes"),
        }
        if explicit_role in aliases:
            for key in aliases[explicit_role]:
                if key in manifest:
                    raw_lists.append((explicit_role or "<missing-role>", manifest[key]))
                    break
        elif scope == COMPONENT_SCOPE:
            for role, keys in aliases.items():
                for key in keys:
                    if key in manifest:
                        raw_lists.append((role, manifest[key]))
                        break
    if not raw_lists:
        return [], ["task hashes are missing"]

    normalized: list[tuple[str, list[str]]] = []
    for role, raw_hashes in raw_lists:
        if not _is_sequence(raw_hashes) or not raw_hashes:
            errors.append(f"{role} task hashes must be a non-empty ordered list")
            continue
        hashes: list[str] = []
        seen: set[str] = set()
        for index, value in enumerate(raw_hashes):
            if not isinstance(value, str) or not _TASK_DIGEST.fullmatch(value):
                errors.append(f"{role} task hash at index {index} is not a SHA-256 digest")
                continue
            if value in seen:
                errors.append(f"duplicate task hash within {role} split")
            seen.add(value)
            hashes.append(value)
        if hashes:
            normalized.append((role, hashes))
            _check_declared_aggregate(manifest, role, hashes, errors)
    return normalized, errors


def _check_declared_aggregate(
    manifest: Mapping[str, Any],
    role: str,
    hashes: Sequence[str],
    errors: list[str],
) -> None:
    expected = _aggregate_hash(hashes)
    declared: Any = None
    aggregate_hashes = manifest.get("aggregate_hashes")
    if isinstance(aggregate_hashes, Mapping):
        declared = aggregate_hashes.get(role)
        if declared is None and role == PRIMARY_EVAL_ROLE:
            declared = aggregate_hashes.get("test")
    if declared is None:
        declared = manifest.get(f"{role}_aggregate_sha256")
    if declared is None:
        declared = manifest.get("test_aggregate_sha256") if role == PRIMARY_EVAL_ROLE else None
    if declared is None:
        declared = manifest.get("aggregate_sha256")
    if declared is not None and declared != expected:
        errors.append(f"{role} aggregate hash does not match ordered task hashes")
    counts = manifest.get("counts")
    if isinstance(counts, Mapping):
        count = counts.get(role)
        if count is None and role == PRIMARY_EVAL_ROLE:
            count = counts.get("test")
        if count is not None and count != len(hashes):
            errors.append(f"{role} count does not match ordered task hashes")


def _is_component_manifest(manifest: Mapping[str, Any], *, forced: bool = False) -> bool:
    if forced:
        return True
    suite_id = str(manifest.get("suite_id", manifest.get("id", ""))).lower()
    component = str(manifest.get("component", "")).lower()
    return (
        suite_id in {XLAM_SUITE_ID.lower(), "xlam"}
        or component == "xlam"
        or manifest.get("portfolio_suite") is False
        or "component_id" in manifest
    )


def _coerce_collection(value: Any, *, label: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[str]]:
    """Return ordinary records, discovered component records, and errors."""

    if value is None:
        return [], [], []
    if isinstance(value, (str, Path)):
        value = _read_json(value)
    if isinstance(value, Mapping):
        if label == PORTFOLIO_SCOPE and _is_sequence(value.get("suites")):
            records = [item for item in value["suites"] if isinstance(item, Mapping)]
            errors = ["suite collection contains a non-object manifest"] if any(
                not isinstance(item, Mapping) for item in value["suites"]
            ) else []
            discovered = []
            for key in ("xlam_component_preflight", "xlam_component", "xlam"):
                if isinstance(value.get(key), Mapping):
                    discovered.append(value[key])
            return records, discovered, errors
        for key in ("components", "component_manifests", "runtime_receipts"):
            if _is_sequence(value.get(key)):
                return [item for item in value[key] if isinstance(item, Mapping)], [], [
                    "component collection contains a non-object manifest"
                ] if any(not isinstance(item, Mapping) for item in value[key]) else []
        if any(key in value for key in ("suite_id", "id", "component", "component_id", "task_hashes")):
            return [value], [], []
        # ID-keyed mapping form.
        records: list[Mapping[str, Any]] = []
        errors: list[str] = []
        for key, item in value.items():
            if not isinstance(item, Mapping):
                errors.append(f"{label} entry {key!s} is not a JSON object")
                continue
            record = dict(item)
            record.setdefault("suite_id", str(key))
            records.append(record)
        return records, [], errors
    if _is_sequence(value):
        records: list[Mapping[str, Any]] = []
        discovered: list[Mapping[str, Any]] = []
        errors: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and label == PORTFOLIO_SCOPE and _is_sequence(item.get("suites")):
                nested_records, nested_components, nested_errors = _coerce_collection(item, label=label)
                records.extend(nested_records)
                discovered.extend(nested_components)
                errors.extend(nested_errors)
            elif _is_sequence(item):
                nested_records, nested_components, nested_errors = _coerce_collection(item, label=label)
                records.extend(nested_records)
                discovered.extend(nested_components)
                errors.extend(nested_errors)
            elif isinstance(item, Mapping):
                records.append(item)
            else:
                errors.append(f"{label} collection contains a non-object manifest")
        return records, discovered, errors
    raise SplitCollisionIndexError(f"{label} manifests must be a JSON object, list, or local JSON path")


def _record_id(manifest: Mapping[str, Any], *, component: bool) -> str:
    value = manifest.get("suite_id", manifest.get("id", manifest.get("component_id")))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "<missing-component-id>" if component else "<missing-suite-id>"


def _collision_kind(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if left["scope"] != right["scope"]:
        return "component_portfolio"
    if left["suite_id"] == right["suite_id"]:
        if left["role"] == right["role"]:
            return "within_split"
        return "within_suite_cross_role"
    if left["role"] == right["role"]:
        return "cross_suite_same_role"
    return "cross_suite_cross_role"


def _owner_sort_key(owner: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(owner.get("scope", "")),
        str(owner.get("suite_id", "")),
        str(owner.get("role", "")),
        int(owner.get("position", 0)),
    )


def _normalise_record(
    manifest: Mapping[str, Any],
    *,
    scope: str,
    source_index: int,
    forced_component: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    component = scope == COMPONENT_SCOPE or _is_component_manifest(manifest, forced=forced_component)
    suite_id = _record_id(manifest, component=component)
    errors: list[str] = []
    _assert_metadata_only(manifest)
    revision, revision_errors = _extract_revision(manifest)
    errors.extend(f"{suite_id}: {error}" for error in revision_errors)
    required = _REQUIRED_COMPONENT_RECEIPTS if component else _REQUIRED_PORTFOLIO_RECEIPTS
    receipts, receipt_errors = _extract_receipts(manifest, required=required)
    errors.extend(f"{suite_id}: {error}" for error in receipt_errors)
    explicit_role, role_errors = _scalar_roles(manifest)
    if component and role_errors and isinstance(manifest.get("task_hashes"), Mapping):
        # Component manifests such as xLAM legitimately label roles by split
        # keys rather than by one scalar suite role.
        role_errors = []
    if not component:
        errors.extend(f"{suite_id}: {error}" for error in role_errors)
        if explicit_role not in (TRAIN_ROLE, PRIMARY_EVAL_ROLE):
            if explicit_role is not None:
                errors.append(f"{suite_id}: role must be train or primary_eval")
        expected_role = _EXPECTED_ROLE_BY_ID.get(suite_id)
        if expected_role is None:
            errors.append(f"extra portfolio suite ID: {suite_id}")
        elif explicit_role != expected_role:
            errors.append(
                f"{suite_id}: role {explicit_role!r} disagrees with frozen role {expected_role!r}"
            )
    else:
        if explicit_role is not None and explicit_role not in {
            TRAIN_ROLE,
            PRIMARY_EVAL_ROLE,
            "component",
            "<missing-role>",
        }:
            errors.append(f"{suite_id}: component role {explicit_role!r} is not recognized")
    task_lists, task_errors = _task_hash_lists(
        manifest,
        scope=COMPONENT_SCOPE if component else PORTFOLIO_SCOPE,
        explicit_role=explicit_role,
    )
    errors.extend(f"{suite_id}: {error}" for error in task_errors)
    owner_entries: list[dict[str, Any]] = []
    for role, hashes in task_lists:
        for position, digest in enumerate(hashes):
            owner_entries.append(
                {
                    "scope": COMPONENT_SCOPE if component else PORTFOLIO_SCOPE,
                    "suite_id": suite_id,
                    "role": role,
                    "position": position,
                }
            )
    # Preserve only metadata needed to rebuild and verify the index.  No input
    # object is copied wholesale.
    normalized = {
        "scope": COMPONENT_SCOPE if component else PORTFOLIO_SCOPE,
        "suite_id": suite_id,
        "role": explicit_role if explicit_role is not None else "<missing-role>",
        "revision": revision,
        "receipt_refs": {key: receipts[key] for key in sorted(receipts)},
        "task_hashes": [
            {"role": role, "task_hashes": list(hashes), "aggregate_sha256": _aggregate_hash(hashes)}
            for role, hashes in sorted(task_lists)
        ],
        "source_index": source_index,
    }
    # source_index is useful only during local diagnostics; omit it from the
    # public metadata so permutation of input records remains deterministic.
    normalized.pop("source_index")
    return {"normalized": normalized, "owners": owner_entries}, errors


def _build_from_records(
    portfolio_items: Sequence[Mapping[str, Any]],
    component_items: Sequence[Mapping[str, Any]],
    initial_errors: Iterable[str] = (),
) -> dict[str, Any]:
    errors = list(initial_errors)
    normalized_records: list[dict[str, Any]] = []
    owner_map: dict[str, list[dict[str, Any]]] = {}
    portfolio_ids: list[str] = []
    component_ids: list[str] = []
    for index, manifest in enumerate(portfolio_items):
        record, record_errors = _normalise_record(
            manifest,
            scope=PORTFOLIO_SCOPE,
            source_index=index,
        )
        errors.extend(record_errors)
        normalized = record["normalized"]
        if normalized["scope"] == COMPONENT_SCOPE:
            component_ids.append(str(normalized["suite_id"]))
        else:
            portfolio_ids.append(str(normalized["suite_id"]))
        normalized_records.append(normalized)
        task_lists = normalized["task_hashes"]
        for task_list in task_lists:
            for position, digest in enumerate(task_list["task_hashes"]):
                owner = {
                    "scope": normalized["scope"],
                    "suite_id": normalized["suite_id"],
                    "role": task_list["role"],
                    "position": position,
                }
                owner_map.setdefault(digest, []).append(owner)
    for offset, manifest in enumerate(component_items):
        record, record_errors = _normalise_record(
            manifest,
            scope=COMPONENT_SCOPE,
            source_index=offset,
            forced_component=True,
        )
        errors.extend(record_errors)
        normalized = record["normalized"]
        component_ids.append(str(normalized["suite_id"]))
        normalized_records.append(normalized)
        for task_list in normalized["task_hashes"]:
            for position, digest in enumerate(task_list["task_hashes"]):
                owner = {
                    "scope": COMPONENT_SCOPE,
                    "suite_id": normalized["suite_id"],
                    "role": task_list["role"],
                    "position": position,
                }
                owner_map.setdefault(digest, []).append(owner)

    # Roster and role gates use the literal frozen IDs, never the caller's
    # count arguments or a mutable contract snapshot.
    portfolio_id_set = set(portfolio_ids)
    duplicate_portfolio_ids = sorted(
        suite_id for suite_id in portfolio_id_set if portfolio_ids.count(suite_id) > 1
    )
    if duplicate_portfolio_ids:
        errors.extend(f"duplicate portfolio suite ID: {suite_id}" for suite_id in duplicate_portfolio_ids)
    missing_ids = sorted(FROZEN_PORTFOLIO_SUITE_IDS - portfolio_id_set)
    extra_ids = sorted(portfolio_id_set - FROZEN_PORTFOLIO_SUITE_IDS)
    if missing_ids:
        errors.append("missing frozen portfolio suite IDs: " + ", ".join(missing_ids))
    if extra_ids:
        errors.append("extra portfolio suite IDs: " + ", ".join(extra_ids))
    train_actual = sorted(
        suite_id for suite_id in portfolio_id_set if _EXPECTED_ROLE_BY_ID.get(suite_id) == TRAIN_ROLE
    )
    eval_actual = sorted(
        suite_id for suite_id in portfolio_id_set if _EXPECTED_ROLE_BY_ID.get(suite_id) == PRIMARY_EVAL_ROLE
    )
    if len(train_actual) != 12:
        errors.append(f"frozen train roster requires 12 suites, got {len(train_actual)}")
    if len(eval_actual) != 14:
        errors.append(f"frozen primary_eval roster requires 14 suites, got {len(eval_actual)}")
    if XLAM_SUITE_ID in portfolio_id_set:
        errors.append("xLAM component must remain outside the 26-suite portfolio")

    collisions: list[dict[str, Any]] = []
    for digest, owners in owner_map.items():
        owners.sort(key=_owner_sort_key)
        if len(owners) < 2:
            continue
        for left, right in itertools.combinations(owners, 2):
            collisions.append(
                {
                    "task_hash": digest,
                    "kind": _collision_kind(left, right),
                    "left": dict(left),
                    "right": dict(right),
                }
            )
    collisions.sort(
        key=lambda item: (
            item["task_hash"],
            item["kind"],
            _owner_sort_key(item["left"]),
            _owner_sort_key(item["right"]),
        )
    )
    for collision in collisions:
        errors.append(
            f"{collision['kind']} collision for {collision['task_hash']}: "
            f"{collision['left']['suite_id']}:{collision['left']['role']} and "
            f"{collision['right']['suite_id']}:{collision['right']['role']}"
        )

    normalized_records.sort(
        key=lambda record: (
            record["scope"],
            record["suite_id"],
            record["role"],
            record["revision"] or "",
            _canonical_json(record["task_hashes"]),
        )
    )
    public_owner_map = {
        digest: sorted(owners, key=_owner_sort_key)
        for digest, owners in sorted(owner_map.items())
    }
    unique_errors = sorted(set(str(error) for error in errors))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if not unique_errors else "BLOCKED",
        "launch_authorized": False,
        "launches_any_job": False,
        "evidence_scope": "metadata-only global task-hash collision preflight; xLAM is outside the 26 suites",
        "portfolio_roster": {
            "train": list(FROZEN_TRAIN_SUITE_IDS),
            "primary_eval": list(FROZEN_PRIMARY_EVAL_SUITE_IDS),
            "suite_count": 26,
        },
        "portfolio_counts": {TRAIN_ROLE: len(train_actual), PRIMARY_EVAL_ROLE: len(eval_actual)},
        "component_suite_ids": sorted(set(component_ids)),
        "component_count": len(component_ids),
        "records": normalized_records,
        "owner_map": public_owner_map,
        "collision_count": len(collisions),
        "collision_kinds": sorted({str(item["kind"]) for item in collisions}),
        "collisions": collisions,
        "errors": unique_errors,
    }


def build_collision_index(
    portfolio_manifests: Any,
    component_manifests: Any = (),
) -> dict[str, Any]:
    """Build a deterministic owner-map index without external I/O."""

    portfolio_items, discovered_components, portfolio_errors = _coerce_collection(
        portfolio_manifests,
        label=PORTFOLIO_SCOPE,
    )
    component_items, nested_components, component_errors = _coerce_collection(
        component_manifests,
        label=COMPONENT_SCOPE,
    )
    component_items = discovered_components + nested_components + component_items
    return _build_from_records(
        portfolio_items,
        component_items,
        [*portfolio_errors, *component_errors],
    )


def validate_collision_index(index: Mapping[str, Any]) -> list[str]:
    """Return deterministic drift/errors for an existing collision index."""

    if not isinstance(index, Mapping):
        return ["collision index must be a JSON object"]
    try:
        _assert_metadata_only(index)
    except SplitCollisionIndexError as exc:
        return [str(exc)]
    if index.get("schema_version") != SCHEMA_VERSION:
        return ["schema_version is not the collision-index schema"]
    records = index.get("records")
    if not _is_sequence(records) or any(not isinstance(record, Mapping) for record in records):
        return ["collision index records must be an ordered list of objects"]
    portfolio = [record for record in records if record.get("scope") == PORTFOLIO_SCOPE]
    components = [record for record in records if record.get("scope") == COMPONENT_SCOPE]
    rebuilt = build_collision_index(portfolio, components)
    if _canonical_json(dict(index)) != _canonical_json(rebuilt):
        return ["collision index metadata drift or non-canonical fields"]
    return []


def verify_collision_index(index: Mapping[str, Any] | str | Path) -> bool:
    """Raise unless an existing index is canonical, complete, and collision-free."""

    if isinstance(index, (str, Path)):
        index = _read_json(index)
    errors = validate_collision_index(index)
    if errors:
        raise SplitCollisionIndexError("invalid split collision index: " + "; ".join(errors))
    if index.get("status") != "READY" or index.get("collision_count") != 0:
        raise SplitCollisionIndexError(
            "split collision index is blocked: "
            + "; ".join(index.get("errors", []) or ["task-hash collisions remain"])
        )
    return True


# Convenient aliases for callers using different portfolio terminology.
build_index = build_collision_index
index_splits = build_collision_index
generate_collision_index = build_collision_index
build_split_collision_index = build_collision_index
build_owner_map = build_collision_index
validate_index = validate_collision_index
verify_index = verify_collision_index
verify = verify_collision_index
CollisionIndexError = SplitCollisionIndexError


def _write_json(value: Mapping[str, Any], path: Path | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


def _load_cli_collection(paths: Sequence[str]) -> list[Any]:
    values: list[Any] = []
    for path in paths:
        loaded = _read_json(path)
        if _is_sequence(loaded):
            values.extend(loaded)
        else:
            values.append(loaded)
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate", help="build an offline collision index")
    generate.add_argument("--portfolio", action="append", default=[])
    generate.add_argument("--manifest", action="append", default=[])
    generate.add_argument("--component", action="append", default=[])
    generate.add_argument("--out", type=Path)
    verify = subparsers.add_parser("verify", help="verify an existing collision index")
    verify.add_argument("--index", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "verify":
            verify_collision_index(args.index)
            print(json.dumps({"verified": True, "index": str(args.index)}, sort_keys=True))
            return 0
        if args.command != "generate":
            parser.error("choose generate or verify")
        portfolio_inputs = args.portfolio or args.manifest
        if not portfolio_inputs:
            parser.error("generate requires --portfolio or --manifest")
        components: list[Any] = list(args.component)
        # Multiple local files are composed as a list; each file can itself be
        # a registry/list/object and is normalized without network access.
        portfolio = _load_cli_collection(portfolio_inputs)
        components = _load_cli_collection(components)
        index = build_collision_index(portfolio, components)
        _write_json(index, args.out)
        return 0 if index["status"] == "READY" else 2
    except (SplitCollisionIndexError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
