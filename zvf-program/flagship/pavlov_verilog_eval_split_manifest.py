#!/usr/bin/env python3
"""Offline exact T4 split/decontamination boundary for ``verilog_eval``.

This validator is intentionally metadata-first and fail-closed:

* no synthetic/Glaive/xLAM substitution
* no network, credentials, or paid work
* pinned immutable dataset/split artifacts
* deterministic task hash ordering and exact split manifest hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "pavlov-verilog-eval-split-manifest-v1"
SUITE_ID = "verilog_eval"
SUITE_ROLE = "primary_eval"
_EXPECTED_CATEGORY = "code"
EXPECTED_DATASET_SOURCE = "https://github.com/NVlabs/verilog-eval"
_EXPECTED_STATEFUL = False
EXPECTED_ARTIFACT_OR_SIDE_EFFECT = True

HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
URL_RE = re.compile(r"^https://[^\s]+$")

_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "pending",
    "placeholder",
    "todo",
    "tbd",
    "unset",
    "unknown",
    "to_be_pinned_before_paid_runs",
    "to_be_pinned",
    "latest",
    "main",
    "head",
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

_ALLOWED_DECONTAMINATION_STATUS = {
    "verified",
    "clean",
    "passed",
    "admissible",
    "complete",
    "completed",
}

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


class VerilogEvalSplitManifestError(ValueError):
    """Raised for malformed verilog_eval split manifest input."""


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
        raise VerilogEvalSplitManifestError(f"{field} must be non-placeholder text")
    return value.strip()


def _require_status(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerilogEvalSplitManifestError(f"{field} must be explicit status text")
    return value.strip()


def _require_url(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if not URL_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be https URL")
    return text


def _require_hex40(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX40_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be immutable 40-char hex")
    return text


def _require_hex64(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not HEX64_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be 64-char hex")
    return text


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field).lower()
    if not SHA256_RE.fullmatch(text):
        raise VerilogEvalSplitManifestError(f"{field} must be a sha256 digest")
    return text if text.startswith("sha256:") else f"sha256:{text}"


def _contains_banned_markers(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in _BANNED_MARKERS)
    if isinstance(value, Mapping):
        return any(
            _contains_banned_markers(item) for item in value.values() if item is not None
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str, bytearray)):
        return any(_contains_banned_markers(item) for item in value)
    return False


def _load_authority(contract_path: str | Path | None = None) -> Mapping[str, Any]:
    base = Path(__file__).resolve().parent
    path = Path(contract_path) if contract_path is not None else base / "pavlovs_domain_contract.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerilogEvalSplitManifestError(f"cannot read authority contract: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise VerilogEvalSplitManifestError(
            f"authority contract is not valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise VerilogEvalSplitManifestError("authority contract must be a JSON object")

    registry = payload.get("suite_registry")
    if not isinstance(registry, Mapping):
        raise VerilogEvalSplitManifestError("authority contract missing suite_registry")
    suite = registry.get(SUITE_ID)
    if not isinstance(suite, Mapping):
        raise VerilogEvalSplitManifestError("authority contract missing verilog_eval suite")
    return suite


def _validate_source(value: Any, authority: Mapping[str, Any]) -> str:
    source = _require_url(value, "source")
    authoritative_source = _require_url(authority.get("url"), "authority suite url")
    if source.rstrip("/").lower() != authoritative_source.rstrip("/").lower():
        raise VerilogEvalSplitManifestError(
            "source must match authoritative verilog_eval source"
        )
    return source


def _validate_dataset(value: Any, authority: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalSplitManifestError("dataset must be an object")
    revision = _require_hex40(value.get("revision"), "dataset.revision")
    license_text = _require_text(value.get("license"), "dataset.license")
    source = _require_url(value.get("source"), "dataset.source")
    authoritative_source = _require_url(authority.get("url"), "contract suite url")
    if source.rstrip("/").lower() != authoritative_source.rstrip("/").lower():
        raise VerilogEvalSplitManifestError(
            "dataset.source must match authoritative verilog_eval source"
        )

    if _contains_banned_markers(value.get("name")):
        raise VerilogEvalSplitManifestError("dataset references blocked source")
    return revision, license_text, source


def _validate_category(value: Any) -> str:
    category = _require_text(value, "category")
    if category != _EXPECTED_CATEGORY:
        raise VerilogEvalSplitManifestError(
            f"category must be {_EXPECTED_CATEGORY!r}, found {category!r}"
        )
    return category


def _validate_role(value: Any) -> str:
    role = _require_text(value, "role")
    if role != SUITE_ROLE:
        raise VerilogEvalSplitManifestError(
            f"role must be {SUITE_ROLE!r}, found {role!r}"
        )
    return role


def _validate_task_hashes(value: Any, field: str = "task_id_hashes") -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise VerilogEvalSplitManifestError(f"{field} must be a non-empty list")
    ids: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_hash = _require_hex64(item, f"{field}[{index}]")
        if item_hash in seen:
            raise VerilogEvalSplitManifestError(f"{field} contains duplicate IDs")
        seen.add(item_hash)
        ids.append(item_hash)

    if not ids:
        raise VerilogEvalSplitManifestError(f"{field} cannot be empty")
    if ids != sorted(ids):
        raise VerilogEvalSplitManifestError(f"{field} must be deterministically sorted")
    return ids


def _validate_split(value: Any, task_ids: list[str]) -> tuple[list[str], str, str]:
    if not isinstance(value, Mapping):
        raise VerilogEvalSplitManifestError("split must be an object")
    split_task_ids = _validate_task_hashes(
        value.get("primary_eval"), "split.primary_eval"
    )
    if split_task_ids != task_ids:
        raise VerilogEvalSplitManifestError("split.primary_eval must match task_id_hashes")

    declared = _require_sha256(
        value.get("hash") if value.get("hash") is not None else value.get("sha256"),
        "split.primary_eval.hash",
    )
    aggregate = _sha256("\n".join(split_task_ids))
    expected = f"sha256:{aggregate}"
    if declared != expected:
        raise VerilogEvalSplitManifestError(
            "split.primary_eval hash does not match aggregate of task IDs"
        )
    return split_task_ids, aggregate, expected


def _validate_split_hashes(value: Mapping[str, Any], aggregate: str) -> dict[str, str]:
    split_hashes = value.get("split_hashes")
    if not isinstance(split_hashes, Mapping):
        raise VerilogEvalSplitManifestError("split_hashes must be an object")
    declared = _require_sha256(
        split_hashes.get("primary_eval"), "split_hashes.primary_eval"
    )
    expected = f"sha256:{aggregate}"
    if declared != expected:
        raise VerilogEvalSplitManifestError(
            "split_hashes.primary_eval does not match task aggregate"
        )
    return {"primary_eval": declared}


def _validate_split_manifest_hash(
    manifest: Mapping[str, Any], aggregate: str
) -> tuple[str, str]:
    payload = canonical_json({"primary_eval": aggregate})
    observed = f"sha256:{_sha256(payload)}"
    declared = _require_sha256(manifest.get("split_manifest_hash"), "split_manifest_hash")
    if declared != observed:
        raise VerilogEvalSplitManifestError(
            "split_manifest_hash does not match observed split aggregate"
        )
    receipt_ref = _require_sha256(
        manifest.get("split_manifest_receipt_ref"), "split_manifest_receipt_ref"
    )
    return observed, receipt_ref


def _validate_decontamination(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        receipt_id = _require_hex40(value, "decontamination.receipt_id")
        return {
            "status": "verified",
            "receipt_id": receipt_id,
            "visibility": "private",
            "safe_public_artifact": True,
            "url": None,
        }

    if not isinstance(value, Mapping):
        raise VerilogEvalSplitManifestError("decontamination must be an object")
    status = _require_status(value.get("status"), "decontamination.status").lower()
    if status not in _ALLOWED_DECONTAMINATION_STATUS:
        raise VerilogEvalSplitManifestError("decontamination.status is invalid")
    receipt_id = _require_hex40(
        value.get("receipt_id"), "decontamination.receipt_id"
    )
    visibility = _require_text(value.get("visibility", "private"), "decontamination.visibility")
    if visibility.lower() == "public" and not bool(value.get("safe_public_artifact")):
        raise VerilogEvalSplitManifestError(
            "public decontamination evidence must set safe_public_artifact=True"
        )
    return {
        "status": status,
        "receipt_id": receipt_id,
        "visibility": visibility.lower(),
        "safe_public_artifact": bool(value.get("safe_public_artifact")),
        "url": value.get("url"),
    }


def _validate_no_network_or_credentials(manifest: Mapping[str, Any], blockers: list[str]) -> None:
    if manifest.get("requires_network") is True:
        blockers.append("dataset download/network is disallowed for split validation")
    for key in _CREDENTIAL_KEYS:
        value = manifest.get(key)
        if not _is_placeholder(value):
            blockers.append(f"{key} is not allowed in offline split validation")


def build_split_manifest_record(
    manifest: Mapping[str, Any],
    *,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a deterministic verilog_eval split decision payload."""

    try:
        if not isinstance(manifest, Mapping):
            raise VerilogEvalSplitManifestError("manifest must be a JSON object")

        authority = _load_authority(contract_path)
        if str(authority.get("role", "")).strip().lower() != SUITE_ROLE:
            raise VerilogEvalSplitManifestError("contract role mismatch for verilog_eval")
        if authority.get("stateful") is not _EXPECTED_STATEFUL:
            raise VerilogEvalSplitManifestError("contract stateful value changed")
        if authority.get("artifact_or_side_effect") is not EXPECTED_ARTIFACT_OR_SIDE_EFFECT:
            raise VerilogEvalSplitManifestError(
                "contract artifact_or_side_effect value changed"
            )

        suite_id = _require_text(manifest.get("suite_id"), "suite_id")
        if suite_id != SUITE_ID:
            raise VerilogEvalSplitManifestError(f"suite_id must be {SUITE_ID!r}")

        source = _validate_source(manifest.get("source"), authority)
        category = _validate_category(manifest.get("category"))
        role = _validate_role(manifest.get("role"))
        dataset_revision, dataset_license, dataset_source = _validate_dataset(
            manifest.get("dataset"), authority
        )
        task_ids = _validate_task_hashes(manifest.get("task_id_hashes"), "task_id_hashes")
        split_ids, aggregate, aggregate_digest = _validate_split(
            manifest.get("split"), task_ids
        )
        split_hashes = _validate_split_hashes(manifest, aggregate)
        split_manifest_hash, split_manifest_receipt_ref = _validate_split_manifest_hash(
            manifest, aggregate
        )
        decontamination = _validate_decontamination(manifest.get("decontamination"))

        evidence = {
            "source": source,
            "dataset": manifest.get("dataset"),
            "split_hashes": split_hashes,
            "verifier": manifest.get("verifier"),
            "environment": manifest.get("environment"),
            "split": manifest.get("split"),
        }
        if _contains_banned_markers(evidence):
            raise VerilogEvalSplitManifestError("manifest contains blocked substitution markers")

        if _contains_banned_markers(source):
            raise VerilogEvalSplitManifestError("source includes blocked tokens")

        scope = manifest.get("scope")
        if isinstance(scope, Mapping) and scope.get("e_suite_ids"):
            raise VerilogEvalSplitManifestError("manifest scope includes E-suite entries")

        blockers: list[str] = []
        _validate_no_network_or_credentials(manifest, blockers)

        blockers = sorted(set(blockers))

        split_manifest_payload = {
            "suite_id": SUITE_ID,
            "task_ids": split_ids,
            "split_hashes": split_hashes,
            "dataset_revision": dataset_revision,
        }
        split_manifest_payload_digest = f"sha256:{_sha256(canonical_json(split_manifest_payload))}"

        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "verilog_eval_split_manifest",
            "suite_id": suite_id,
            "exact_suite": suite_id == SUITE_ID,
            "split_ready": not blockers,
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
            "split": {
                "primary_eval": {
                    "count": len(split_ids),
                    "task_id_hashes": split_ids,
                    "aggregate_sha256": aggregate,
                    "aggregate_sha256_digest": aggregate_digest,
                }
            },
            "split_hashes": split_hashes,
            "split_manifest_hash": split_manifest_hash,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash"),
            "split_manifest_receipt_ref": split_manifest_receipt_ref,
            "decontamination": decontamination,
            "task_id_digest": _sha256(canonical_json(split_ids)),
            "split_manifest_payload_digest": split_manifest_payload_digest,
            "blockers": blockers,
            "status": "READY" if not blockers else "BLOCKED",
        }
    except VerilogEvalSplitManifestError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_type": "verilog_eval_split_manifest",
            "suite_id": manifest.get("suite_id", "<missing>")
            if isinstance(manifest, Mapping)
            else "<invalid>",
            "exact_suite": manifest.get("suite_id") == SUITE_ID
            if isinstance(manifest, Mapping)
            else False,
            "split_ready": False,
            "paid_launch_allowed": False,
            "launch": {"allowed": False, "reasons": ["launch is intentionally disabled"]},
            "authoritative_source": None,
            "contract_authority": {},
            "category": manifest.get("category", "<missing>")
            if isinstance(manifest, Mapping)
            else "<missing>",
            "role": manifest.get("role", "<missing>") if isinstance(manifest, Mapping) else "<missing>",
            "dataset_revision": None,
            "dataset_license": None,
            "dataset_source": None,
            "split": {"primary_eval": {}},
            "split_hashes": {},
            "split_manifest_hash": manifest.get("split_manifest_hash") if isinstance(manifest, Mapping) else None,
            "declared_split_manifest_hash": manifest.get("split_manifest_hash")
            if isinstance(manifest, Mapping)
            else None,
            "split_manifest_receipt_ref": manifest.get("split_manifest_receipt_ref")
            if isinstance(manifest, Mapping)
            else None,
            "decontamination": None,
            "task_id_digest": None,
            "split_manifest_payload_digest": None,
            "blockers": [str(exc)],
            "status": "BLOCKED",
        }


# ---------------------------------------------------------------------------
# Manifest construction from the pinned NVlabs checkout
# ---------------------------------------------------------------------------
#
# The prior E11 receipt refused to promote a local directory listing to an
# authoritative task manifest, and that refusal was correct.  What makes the
# manifest below authoritative is that the task list is read from
# ``dataset_<name>/problems.txt``, a file upstream committed at the pinned
# revision, and every entry is then cross-checked against the artifacts on disk
# in both directions.  A listing that upstream does not vouch for is rejected,
# and an artifact upstream does not list is rejected too.

DATASET_NAMES = ("code-complete-iccad2023", "spec-to-rtl")
PINNED_REVISION = "c498220d0a52248f8e3fdffe279075215bde2da6"
DATASET_LICENSE = "MIT"

#: Artifacts every problem must ship, per dataset.  ``code-complete-iccad2023``
#: additionally ships ``_ifc.txt``; ``spec-to-rtl`` ships none, by design.
_REQUIRED_ARTIFACTS = {
    "code-complete-iccad2023": ("_prompt.txt", "_ifc.txt", "_ref.sv", "_test.sv"),
    "spec-to-rtl": ("_prompt.txt", "_ref.sv", "_test.sv"),
}


def canonical_task_id(dataset_name: str, problem_id: str) -> str:
    """Return the stable cross-dataset task identity string.

    A bare ``Prob001_zero`` is ambiguous because it names one problem in each of
    the two task framings, and the two framings ship different prompts (and, for
    9 problems, different references and 7 different test benches).  Qualifying
    the ID with the dataset is what makes the 312 prompts individually
    addressable.
    """

    return f"{SUITE_ID}/{dataset_name}/{problem_id}"


def task_id_hash(dataset_name: str, problem_id: str) -> str:
    return _sha256(canonical_task_id(dataset_name, problem_id))


def read_authoritative_problem_ids(checkout: Path, dataset_name: str) -> list[str]:
    """Read upstream's own committed problem list for one dataset."""

    listing = Path(checkout) / f"dataset_{dataset_name}" / "problems.txt"
    try:
        text = listing.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerilogEvalSplitManifestError(
            f"cannot read upstream problem list {listing}: {exc}"
        ) from exc

    problems = [line.strip() for line in text.splitlines() if line.strip()]
    if not problems:
        raise VerilogEvalSplitManifestError(f"upstream problem list is empty: {listing}")
    if len(set(problems)) != len(problems):
        raise VerilogEvalSplitManifestError(f"upstream problem list has duplicates: {listing}")
    return problems


def build_task_table(checkout: Path, dataset_name: str) -> list[dict[str, Any]]:
    """Return per-task identity and content hashes, cross-checked both ways."""

    checkout = Path(checkout)
    dataset_dir = checkout / f"dataset_{dataset_name}"
    if not dataset_dir.is_dir():
        raise VerilogEvalSplitManifestError(f"missing dataset directory: {dataset_dir}")

    required = _REQUIRED_ARTIFACTS.get(dataset_name)
    if required is None:
        raise VerilogEvalSplitManifestError(f"unknown dataset: {dataset_name}")

    problems = read_authoritative_problem_ids(checkout, dataset_name)
    listed = set(problems)

    # Reverse check: every problem that has artifacts on disk must be listed.
    on_disk = {
        path.name[: -len("_ref.sv")]
        for path in dataset_dir.iterdir()
        if path.name.endswith("_ref.sv")
    }
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise VerilogEvalSplitManifestError(
            f"{dataset_name}: artifacts present but absent from problems.txt: {unlisted}"
        )
    missing_artifacts = sorted(listed - on_disk)
    if missing_artifacts:
        raise VerilogEvalSplitManifestError(
            f"{dataset_name}: problems.txt lists problems with no reference: {missing_artifacts}"
        )

    table: list[dict[str, Any]] = []
    for problem_id in problems:
        artifacts: dict[str, str] = {}
        for suffix in required:
            path = dataset_dir / f"{problem_id}{suffix}"
            try:
                artifacts[suffix.lstrip("_")] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                raise VerilogEvalSplitManifestError(
                    f"{dataset_name}/{problem_id}: missing required artifact {path.name}: {exc}"
                ) from exc

        table.append(
            {
                "canonical_task_id": canonical_task_id(dataset_name, problem_id),
                "task_id_hash": task_id_hash(dataset_name, problem_id),
                "dataset": dataset_name,
                "problem_id": problem_id,
                "artifact_sha256": artifacts,
                "content_digest": _sha256(canonical_json(artifacts)),
            }
        )
    return table


def build_manifest_from_checkout(
    checkout: str | Path,
    *,
    revision: str = PINNED_REVISION,
    dataset_license: str = DATASET_LICENSE,
    datasets: Sequence[str] = DATASET_NAMES,
    decontamination: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact verilog_eval split manifest from the pinned checkout.

    ``decontamination`` is deliberately not synthesised.  Decontamination needs
    an external receipt that does not exist locally, so omitting it makes
    :func:`build_split_manifest_record` return ``BLOCKED`` with that single
    reason, which is the honest outcome.
    """

    checkout = Path(checkout)
    tasks: list[dict[str, Any]] = []
    per_dataset: dict[str, Any] = {}
    for dataset_name in datasets:
        table = build_task_table(checkout, dataset_name)
        tasks.extend(table)
        per_dataset[dataset_name] = {
            "problem_count": len(table),
            "problems_source": f"dataset_{dataset_name}/problems.txt",
            "task_id_hashes": sorted(item["task_id_hash"] for item in table),
        }

    task_hashes = sorted(item["task_id_hash"] for item in tasks)
    if len(set(task_hashes)) != len(task_hashes):
        raise VerilogEvalSplitManifestError("canonical task IDs collided across datasets")

    aggregate = _sha256("\n".join(task_hashes))
    tasks_by_id = sorted(tasks, key=lambda item: item["canonical_task_id"])
    receipt_ref = f"sha256:{_sha256(canonical_json(tasks_by_id))}"

    manifest: dict[str, Any] = {
        "suite_id": SUITE_ID,
        "source": EXPECTED_DATASET_SOURCE,
        "category": _EXPECTED_CATEGORY,
        "role": SUITE_ROLE,
        "dataset": {
            "name": "NVlabs/verilog-eval",
            "revision": revision,
            "license": dataset_license,
            "source": EXPECTED_DATASET_SOURCE,
        },
        "task_id_hashes": task_hashes,
        "split": {"primary_eval": list(task_hashes), "hash": f"sha256:{aggregate}"},
        "split_hashes": {"primary_eval": f"sha256:{aggregate}"},
        "split_manifest_hash": f"sha256:{_sha256(canonical_json({'primary_eval': aggregate}))}",
        "split_manifest_receipt_ref": receipt_ref,
        "requires_network": False,
        "task_id_scheme": {
            "template": f"{SUITE_ID}/<dataset>/<problem_id>",
            "hash": "sha256 of the canonical task id, lowercase hex",
            "aggregate": "sha256 of the newline-joined sorted task id hashes",
            "authority": (
                "dataset_<name>/problems.txt as committed at the pinned revision, cross-checked "
                "against on-disk artifacts in both directions"
            ),
        },
        "tasks": tasks_by_id,
        "datasets": per_dataset,
    }
    if decontamination is not None:
        manifest["decontamination"] = dict(decontamination)
    return manifest


def build_split_manifest_receipt(
    checkout: str | Path,
    *,
    revision: str = PINNED_REVISION,
    dataset_license: str = DATASET_LICENSE,
    datasets: Sequence[str] = DATASET_NAMES,
    decontamination: Mapping[str, Any] | None = None,
    contract_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the split manifest plus its validation record, as one receipt."""

    manifest = build_manifest_from_checkout(
        checkout,
        revision=revision,
        dataset_license=dataset_license,
        datasets=datasets,
        decontamination=decontamination,
    )
    record = build_split_manifest_record(manifest, contract_path=contract_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "verilog_eval_split_manifest_receipt",
        "suite_id": SUITE_ID,
        "checkout": str(Path(checkout)),
        "dataset_revision": revision,
        "task_count": len(manifest["task_id_hashes"]),
        "manifest": manifest,
        "validation": record,
        "status": record["status"],
        "is_model_score": False,
        "score": None,
        "launch": {"paid_work_launched": False, "weight_changing_run_launched": False},
    }


def validate_split_manifest_record(record: Mapping[str, Any]) -> list[str]:
    if not isinstance(record, Mapping):
        return ["record root must be a JSON object"]
    ready = bool(record.get("split_ready"))
    if record.get("status") == "READY" and not ready:
        return ["record status is READY but split_ready is false"]
    return [] if record.get("status") == "READY" else list(record.get("blockers", []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="Validate an existing manifest JSON file.")
    group.add_argument(
        "--checkout",
        type=Path,
        help="Build the manifest from a pinned NVlabs/verilog-eval checkout, then validate it.",
    )
    parser.add_argument("--revision", default=PINNED_REVISION)
    parser.add_argument("--output", type=Path, help="Write the receipt JSON here.")
    args = parser.parse_args(argv)

    if args.checkout is not None:
        receipt = build_split_manifest_receipt(args.checkout, revision=args.revision)
        payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": receipt["status"],
                        "task_count": receipt["task_count"],
                        "split_manifest_hash": receipt["manifest"]["split_manifest_hash"],
                        "aggregate": receipt["manifest"]["split_hashes"]["primary_eval"],
                        "blockers": receipt["validation"]["blockers"],
                        "output": str(args.output),
                    },
                    indent=2,
                )
            )
        else:
            print(payload)
        return 0 if receipt["status"] == "READY" else 1

    payload_obj = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(payload_obj, Mapping):
        raise SystemExit("manifest must be a JSON object")
    report = build_split_manifest_record(payload_obj)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
