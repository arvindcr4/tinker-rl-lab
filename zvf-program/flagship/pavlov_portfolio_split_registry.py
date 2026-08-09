#!/usr/bin/env python3
"""Build and verify the metadata-only Pavlov suite split registry.

This module is deliberately a local, zero-cost gate.  It consumes already
materialised split manifests; it never loads a dataset, calls a model, or
launches a job.  A suite manifest contains ordered task digests and receipts,
not the prompts or targets from which those digests were made.

The contract currently names 12 training suites and 14 ``primary_eval``
suites.  The latter are intentionally *not* called held-out by this module:
only an explicit, non-placeholder held-out receipt can prove that stronger
property.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "pavlovs_domain_contract.json"
SCHEMA_VERSION = "pavlov-portfolio-split-registry-v1"
XLAM_COMPONENT = "xLAM"
TRAIN_ROLE = "train"
PRIMARY_EVAL_ROLE = "primary_eval"
ROLES = (TRAIN_ROLE, PRIMARY_EVAL_ROLE)
PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
TASK_DIGEST = re.compile(r"^[0-9a-f]{64}$")

# These values are common in prospective manifests.  Treating them as
# receipts would make a blocked preflight look complete, so they are rejected
# case-insensitively and with a few useful prefix checks.
_PLACEHOLDER_RECEIPTS = {
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
    "to_be_pinned_before_paid_runs",
    "not_provided",
    "not provided",
    "to be pinned",
    "latest",
    "master",
    "head",
    "tip",
}
_RAW_CONTENT_KEYS = {"prompt", "prompts", "target", "targets"}
_RECEIPT_KEYS = (
    "revision",
    "license",
    "container",
    "decontamination",
    "verifier",
    "split_manifest",
)


class PortfolioSplitRegistryError(ValueError):
    """Raised when split manifests cannot form a valid portfolio registry."""


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortfolioSplitRegistryError(f"cannot read JSON {path!s}: {exc}") from exc


def load_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load a local domain contract without performing any external I/O."""

    value = _read_json(path)
    if not isinstance(value, Mapping):
        raise PortfolioSplitRegistryError("domain contract must be a JSON object")
    return dict(value)


def _as_contract(contract: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if contract is None:
        return load_contract()
    if isinstance(contract, (str, Path)):
        return load_contract(contract)
    if not isinstance(contract, Mapping):
        raise PortfolioSplitRegistryError("contract must be a JSON object or local path")
    return contract


def _contract_suite_registry(contract: Mapping[str, Any]) -> Mapping[str, Any]:
    registry = contract.get("suite_registry", {})
    if isinstance(registry, Mapping):
        return registry
    if isinstance(registry, Sequence) and not isinstance(registry, (str, bytes, bytearray)):
        result: dict[str, Any] = {}
        for item in registry:
            if not isinstance(item, Mapping):
                continue
            suite_id = item.get("suite_id", item.get("id"))
            if suite_id is not None:
                result[str(suite_id)] = item
        return result
    raise PortfolioSplitRegistryError("domain contract suite_registry must be an object or list")


def expected_suite_ids(
    contract: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Return the exact suite IDs required for each portfolio role."""

    contract = _as_contract(contract)
    registry = _contract_suite_registry(contract)
    by_role: dict[str, list[str]] = {role: [] for role in ROLES}
    for suite_id, entry in registry.items():
        if not isinstance(entry, Mapping):
            continue
        role = entry.get("role", entry.get("split_role"))
        if role in by_role:
            by_role[role].append(str(suite_id))
    return {role: tuple(sorted(ids)) for role, ids in by_role.items()}


def _declared_domains(contract: Mapping[str, Any]) -> tuple[str, ...]:
    values = contract.get("domains", contract.get("declared_domains", ()))
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise PortfolioSplitRegistryError("domain contract domains must be a list")
    domains = tuple(sorted({str(value) for value in values}))
    if not domains:
        raise PortfolioSplitRegistryError("domain contract declares no domains")
    return domains


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def aggregate_task_hashes(task_hashes: Iterable[str]) -> str:
    """Hash an ordered sequence of task digests with deterministic framing."""

    return _sha256("\n".join(task_hashes))


def _looks_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    if normalized in _PLACEHOLDER_RECEIPTS:
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


def _assert_metadata_only(value: Any, path: str = "manifest") -> None:
    """Reject raw prompt/target fields before anything can be copied out."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _RAW_CONTENT_KEYS:
                raise PortfolioSplitRegistryError(
                    f"{path} contains raw {str(key).lower()} content; manifests are metadata-only"
                )
            _assert_metadata_only(child, f"{path}.{key}")
    elif _is_sequence(value):
        for index, child in enumerate(value):
            _assert_metadata_only(child, f"{path}[{index}]")


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _extract_revision(manifest: Mapping[str, Any]) -> str:
    values = [
        manifest[key]
        for key in ("revision", "dataset_revision", "immutable_revision", "revision_sha")
        if key in manifest
    ]
    if not values:
        raise PortfolioSplitRegistryError("suite manifest is missing an immutable revision")
    if any(not isinstance(value, str) for value in values) or len({str(value) for value in values}) != 1:
        raise PortfolioSplitRegistryError("suite manifest has conflicting revision inputs")
    revision = values[0]
    if not isinstance(revision, str) or not PINNED_REVISION.fullmatch(revision):
        raise PortfolioSplitRegistryError(
            "suite revision must be an immutable 40-character lower-case commit SHA"
        )
    return revision


def _receipt_source(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("receipt_refs", "provenance_receipts", "receipts"):
        value = manifest.get(field)
        if isinstance(value, Mapping):
            result.update(value)
    result.update({key: value for key, value in manifest.items() if key.endswith("_receipt_ref")})
    return result


def _extract_receipts(manifest: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    source = _receipt_source(manifest)
    aliases: dict[str, tuple[str, ...]] = {
        "revision": (
            "revision",
            "revision_receipt_ref",
            "revision_ref",
            "dataset_revision_receipt_ref",
            "immutable_revision_receipt_ref",
        ),
        "license": ("license", "license_ref", "license_receipt", "license_receipt_ref"),
        "container": (
            "container",
            "container_runtime",
            "runtime",
            "container_receipt",
            "container_receipt_ref",
            "container_runtime_receipt_ref",
            "container_ref",
            "runtime_ref",
            "runtime_receipt_ref",
        ),
        "decontamination": (
            "decontamination",
            "decontamination_receipt",
            "decontamination_receipt_ref",
            "decontamination_ref",
        ),
        "verifier": (
            "verifier",
            "verifier_receipt",
            "verifier_receipt_ref",
            "verification",
            "verification_receipt_ref",
            "verifier_ref",
        ),
        "split_manifest": (
            "split_manifest",
            "split_manifest_receipt",
            "split_manifest_receipt_ref",
            "split_manifest_ref",
            "task_hash",
            "task_hash_receipt_ref",
            "task_id_hash",
            "task_id_hash_receipt_ref",
            "split_manifest_hash",
            "task_ids_hash",
        ),
        "held_out": (
            "held_out",
            "held_out_receipt",
            "held_out_receipt_ref",
            "heldout_receipt_ref",
            "heldout_receipt",
        ),
    }
    receipts: dict[str, str] = {}
    errors: list[str] = []
    for name, keys in aliases.items():
        value = _first_value(source, keys)
        if value is None:
            # ``held_out`` is a boolean assertion at the top level, while a
            # receipt mapping may legitimately use the same short key.  Never
            # mistake ``held_out: true`` for a receipt reference.
            manifest_keys = keys if name != "held_out" else tuple(key for key in keys if key != "held_out")
            value = _first_value(manifest, manifest_keys)
        if value is None:
            if name in _RECEIPT_KEYS:
                errors.append(f"missing {name} receipt")
            continue
        if _looks_placeholder(value):
            errors.append(f"placeholder {name} receipt")
            continue
        if not isinstance(value, str):
            errors.append(f"{name} receipt must be a string reference")
            continue
        receipts[name] = value.strip()
    return receipts, errors


def _extract_role(manifest: Mapping[str, Any], expected_role: str) -> tuple[str, list[str]]:
    candidates: list[str] = []
    for key in ("suite_role", "portfolio_split_role", "role", "split_role"):
        value = manifest.get(key)
        if isinstance(value, str):
            candidates.append(value)
    split_roles = manifest.get("split_roles")
    if isinstance(split_roles, Mapping):
        values = {str(value) for value in split_roles.values() if value is not None}
        if len(values) > 1:
            return "", ["suite manifest declares multiple split roles"]
        candidates.extend(values)
    if not candidates:
        return expected_role, []
    candidate_set = set(candidates)
    if (
        expected_role == PRIMARY_EVAL_ROLE
        and candidate_set <= {PRIMARY_EVAL_ROLE, "held_out"}
        and "held_out" in candidate_set
    ):
        return "held_out", []
    if len(candidate_set) != 1:
        return "", ["suite manifest has conflicting role inputs"]
    role = candidates[0]
    if role == "held_out" and expected_role == PRIMARY_EVAL_ROLE:
        # A held-out assertion is still a primary-eval suite in the portfolio;
        # _normalise_suite will require its explicit proof receipt before it
        # canonicalizes the role.
        return role, []
    if role not in ROLES:
        return role, [f"suite role must be train or primary_eval, got {role!r}"]
    if role != expected_role:
        return role, [f"suite role {role!r} disagrees with contract role {expected_role!r}"]
    return role, []


def _extract_task_hashes(manifest: Mapping[str, Any], role: str) -> tuple[list[str], list[str]]:
    value = manifest.get("task_hashes")
    hashes: Any = None
    if isinstance(value, Mapping):
        for key in (role, "test" if role == PRIMARY_EVAL_ROLE else "train", "primary_eval" if role == PRIMARY_EVAL_ROLE else ""):
            if key and key in value:
                hashes = value[key]
                break
        if hashes is None and len(value) == 1:
            hashes = next(iter(value.values()))
    elif _is_sequence(value):
        hashes = value
    if hashes is None:
        aliases = (
            ("primary_eval_task_hashes", "test_task_hashes", "eval_task_hashes")
            if role == PRIMARY_EVAL_ROLE
            else ("train_task_hashes",)
        )
        hashes = _first_value(manifest, aliases[0])
    errors: list[str] = []
    if not _is_sequence(hashes):
        errors.append(f"{role} task hashes must be a non-empty ordered list")
        return [], errors
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(hashes):
        if not isinstance(value, str) or not TASK_DIGEST.fullmatch(value):
            errors.append(f"{role} task hash at index {index} is not a 64-character SHA-256 digest")
            continue
        if value in seen:
            errors.append(f"duplicate task hash within {role} suite")
        seen.add(value)
        result.append(value)
    if not result:
        errors.append(f"{role} task hashes must be a non-empty ordered list")
    return result, errors


def _extract_aggregate(manifest: Mapping[str, Any], role: str, hashes: Sequence[str]) -> tuple[str, list[str]]:
    value = manifest.get("aggregate_hashes")
    declared: Any = None
    if isinstance(value, Mapping):
        for key in (role, "test" if role == PRIMARY_EVAL_ROLE else "train"):
            if key in value:
                declared = value[key]
                break
    elif isinstance(value, str):
        declared = value
    if declared is None:
        declared = _first_value(
            manifest,
            ("aggregate_sha256", "task_hash_aggregate", "aggregate_task_hash", f"{role}_aggregate_sha256"),
        )
    computed = aggregate_task_hashes(hashes)
    if declared is None:
        return computed, []
    if not isinstance(declared, str) or not TASK_DIGEST.fullmatch(declared):
        return computed, [f"{role} aggregate hash is not a 64-character SHA-256 digest"]
    if declared != computed:
        return computed, [f"{role} aggregate hash does not match ordered task hashes"]
    return computed, []


def _extract_domains(manifest: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    value = _first_value(manifest, ("domain_tags", "domains"))
    if isinstance(value, str):
        value = [value]
    if not _is_sequence(value):
        return [], ["suite manifest is missing domain tags"]
    domains = sorted({str(item) for item in value})
    if not domains:
        return [], ["suite manifest is missing domain tags"]
    return domains, []


def _normalise_suite(
    manifest: Mapping[str, Any],
    *,
    expected_role: str,
    declared_domains: Sequence[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    _assert_metadata_only(manifest)
    suite_id = manifest.get("suite_id", manifest.get("id"))
    if not isinstance(suite_id, str) or not suite_id:
        return None, ["suite manifest is missing suite_id"]
    role, role_errors = _extract_role(manifest, expected_role)
    errors.extend(role_errors)
    domains, domain_errors = _extract_domains(manifest)
    errors.extend(domain_errors)
    unknown = sorted(set(domains) - set(declared_domains))
    if unknown:
        errors.append(f"{suite_id} has undeclared domains: {', '.join(unknown)}")
    try:
        revision = _extract_revision(manifest)
    except PortfolioSplitRegistryError as exc:
        revision = ""
        errors.append(str(exc))
    receipts, receipt_errors = _extract_receipts(manifest)
    errors.extend(f"{suite_id}: {error}" for error in receipt_errors)
    if role == "held_out":
        if "held_out" not in receipts:
            errors.append(f"{suite_id}: held_out role requires a non-placeholder held-out receipt")
        role = PRIMARY_EVAL_ROLE
    hashes, hash_errors = _extract_task_hashes(manifest, role or expected_role)
    errors.extend(f"{suite_id}: {error}" for error in hash_errors)
    aggregate, aggregate_errors = _extract_aggregate(manifest, role or expected_role, hashes)
    errors.extend(f"{suite_id}: {error}" for error in aggregate_errors)
    counts = manifest.get("counts")
    if isinstance(counts, Mapping):
        count_value = counts.get(role or expected_role)
        if count_value is None and (role or expected_role) == PRIMARY_EVAL_ROLE:
            count_value = counts.get("test")
        if count_value is not None and count_value != len(hashes):
            errors.append(f"{suite_id}: task count disagrees with ordered task hashes")
    elif "count" in manifest and manifest["count"] != len(hashes):
        errors.append(f"{suite_id}: task count disagrees with ordered task hashes")

    held_out_flag = bool(manifest.get("held_out", manifest.get("is_held_out", False)))
    held_out_proven = "held_out" in receipts
    if errors:
        return None, errors
    # Only this small, explicit normalized record is copied to the registry.
    normalized = {
        "suite_id": suite_id,
        "role": role,
        "suite_role": role,
        "split_role": role,
        "domains": domains,
        "domain_tags": domains,
        "revision": revision,
        "receipt_refs": {key: receipts[key] for key in _RECEIPT_KEYS},
        "task_hashes": list(hashes),
        "ordered_task_hashes": list(hashes),
        "task_hash_count": len(hashes),
        "aggregate_sha256": aggregate,
        "aggregate_hashes": {role: aggregate},
        "held_out": held_out_flag,
        "held_out_receipt_proven": held_out_proven,
    }
    if held_out_proven:
        normalized["receipt_refs"]["held_out"] = receipts["held_out"]
    return normalized, []


def _coerce_manifests(manifests: Any) -> tuple[list[Mapping[str, Any]], list[str]]:
    """Accept a list, a single manifest, or an ID-keyed manifest mapping."""

    if isinstance(manifests, Mapping):
        if _looks_like_manifest(manifests):
            return [manifests], []
        if "suites" in manifests and _is_sequence(manifests["suites"]):
            suites = manifests["suites"]
            return [item for item in suites if isinstance(item, Mapping)], [
                "suite manifest list contains a non-object item"
            ] if any(not isinstance(item, Mapping) for item in suites) else []
        result: list[Mapping[str, Any]] = []
        errors: list[str] = []
        for key, value in manifests.items():
            if not isinstance(value, Mapping):
                errors.append(f"suite {key!s} is not a manifest object")
                continue
            item = dict(value)
            item.setdefault("suite_id", str(key))
            result.append(item)
        return result, errors
    if _is_sequence(manifests):
        return [item for item in manifests if isinstance(item, Mapping)], [
            "suite manifest list contains a non-object item"
        ] if any(not isinstance(item, Mapping) for item in manifests) else []
    raise PortfolioSplitRegistryError("suite manifests must be a list or object")


def _looks_like_manifest(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("suite_id", "id", "suite_role", "task_hashes", "revision"))


def _company_requirements(contract: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    companies = contract.get("companies", contract.get("company_requirements", ()))
    result: list[tuple[str, list[str]]] = []
    if isinstance(companies, Mapping):
        iterable = [{"name": name, **(entry if isinstance(entry, Mapping) else {"domains": entry})} for name, entry in companies.items()]
    elif _is_sequence(companies):
        iterable = list(companies)
    else:
        return result
    for index, company in enumerate(iterable):
        if not isinstance(company, Mapping):
            continue
        name = str(company.get("name", company.get("company", f"company_{index}")))
        values = company.get("required_domains", company.get("domains", ()))
        if isinstance(values, str):
            values = [values]
        if _is_sequence(values):
            result.append((name, sorted({str(value) for value in values})))
    return sorted(result, key=lambda pair: pair[0])


def _contract_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve contract-status/budget disagreements conservatively.

    A budget block saying that paid jobs may launch is not sufficient evidence
    when the contract itself is still a draft.  The registry records the
    contradiction and stays blocked; it never turns this gate into launch
    authorization.
    """

    raw_status = contract.get("status")
    status = str(raw_status).strip() if raw_status is not None else None
    normalized_status = status.lower() if status else None
    ready_statuses = {
        "ready",
        "approved",
        "authorized",
        "frozen",
        "validated",
        "complete",
        "completed",
    }
    blockers: list[str] = []
    if normalized_status and normalized_status not in ready_statuses:
        blockers.append(f"contract status is not finalized: {status}")
    budget = contract.get("budget_gate", {})
    if not isinstance(budget, Mapping):
        budget = {}
    budget_status = budget.get("status")
    paid_jobs_may_launch = budget.get("paid_jobs_may_launch")
    if (
        paid_jobs_may_launch is True
        and normalized_status
        and normalized_status not in ready_statuses
    ):
        blockers.append("budget authorization cannot override a non-final contract status")
    if paid_jobs_may_launch is True:
        maximum_usd = budget.get("maximum_usd")
        if not isinstance(maximum_usd, (int, float)) or maximum_usd <= 0:
            blockers.append("paid-job budget authorization lacks a positive maximum_usd")
    return {
        "status": status,
        "budget_status": str(budget_status) if budget_status is not None else None,
        "paid_jobs_may_launch": bool(paid_jobs_may_launch is True),
        "blockers": sorted(set(blockers)),
    }


def _normalize_input(
    manifests: Any,
    *,
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, tuple[str, ...]], tuple[str, ...]]:
    expected = expected_suite_ids(contract)
    expected_by_id = {suite_id: role for role, ids in expected.items() for suite_id in ids}
    declared = _declared_domains(contract)
    raw, errors = _coerce_manifests(manifests)
    seen: dict[str, int] = defaultdict(int)
    normalized: list[dict[str, Any]] = []
    for item in raw:
        suite_id = item.get("suite_id", item.get("id"))
        if not isinstance(suite_id, str) or not suite_id:
            errors.append("suite manifest is missing suite_id")
            continue
        seen[suite_id] += 1
        if seen[suite_id] > 1:
            errors.append(f"duplicate suite ID: {suite_id}")
            continue
        if suite_id not in expected_by_id:
            continue
        item_copy = dict(item)
        try:
            value, item_errors = _normalise_suite(
                item_copy,
                expected_role=expected_by_id[suite_id],
                declared_domains=declared,
            )
        except PortfolioSplitRegistryError as exc:
            value, item_errors = None, [str(exc)]
        errors.extend(item_errors)
        if value is not None:
            normalized.append(value)
    expected_ids = set(expected_by_id)
    actual_ids = set(seen)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing:
        errors.append("missing suite IDs: " + ", ".join(missing))
    if extra:
        errors.append("extra suite IDs: " + ", ".join(extra))
    return normalized, errors, expected, declared


def _portfolio_errors(
    normalized: Sequence[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    expected: Mapping[str, Sequence[str]],
    declared_domains: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    by_role = {role: [item for item in normalized if item.get("role") == role] for role in ROLES}
    unions = {role: sorted({domain for item in by_role[role] for domain in item.get("domains", [])}) for role in ROLES}
    for role in ROLES:
        missing = sorted(set(declared_domains) - set(unions[role]))
        if missing:
            errors.append(f"{role} domain union missing: {', '.join(missing)}")

    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for item in normalized:
        for digest in item.get("task_hashes", []):
            owners[digest].append((str(item["suite_id"]), str(item["role"])))
    overlaps: list[dict[str, Any]] = []
    for digest, item_owners in owners.items():
        suite_ids = sorted({suite_id for suite_id, _ in item_owners})
        roles = sorted({role for _, role in item_owners})
        if len(suite_ids) > 1:
            kind = "cross_role" if len(roles) > 1 else "cross_suite"
            overlaps.append({"kind": kind, "task_hash": digest, "suite_ids": suite_ids, "roles": roles})
            errors.append(f"{kind} task-hash overlap for {digest}")
    overlaps.sort(key=lambda item: (item["kind"], item["task_hash"]))

    coverage: dict[str, Any] = {}
    for company, required in _company_requirements(contract):
        unknown = sorted(set(required) - set(declared_domains))
        if unknown:
            errors.append(f"company {company} requires undeclared domains: {', '.join(unknown)}")
        role_report: dict[str, Any] = {"required_domains": required}
        for role in ROLES:
            missing = sorted(set(required) - set(unions[role]))
            role_report[f"{role}_missing"] = missing
            role_report[f"{role}_covered"] = not missing
            if missing:
                errors.append(f"company {company} {role} required-domain gap: {', '.join(missing)}")
        coverage[company] = role_report
    metadata = {
        "domain_unions": unions,
        "overlaps": overlaps,
        "company_domain_coverage": coverage,
    }
    return errors, metadata


def validate_portfolio(
    manifests: Any,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
    xlam_preflight: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return all generic portfolio validation errors without raising."""

    contract = _as_contract(contract)
    try:
        normalized, errors, expected, declared = _normalize_input(manifests, contract=contract)
        role_errors, _ = _portfolio_errors(
            normalized,
            contract=contract,
            expected=expected,
            declared_domains=declared,
        )
        all_errors = errors + role_errors
        all_errors.extend(_contract_gate(contract).get("blockers", []))
        if xlam_preflight is not None:
            try:
                portfolio_hashes = {
                    digest
                    for item in normalized
                    for digest in item.get("task_hashes", [])
                }
                component = _xlam_preflight(xlam_preflight, portfolio_hashes=portfolio_hashes)
                all_errors.extend(f"xLAM: {blocker}" for blocker in component.get("blockers", []))
            except PortfolioSplitRegistryError as exc:
                all_errors.append(str(exc))
        return all_errors
    except PortfolioSplitRegistryError as exc:
        return [str(exc)]


def _xlam_preflight(
    value: Any,
    *,
    portfolio_hashes: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize an optional xLAM component without counting it as a suite."""

    if value is None:
        return {
            "component": XLAM_COMPONENT,
            "suite_id": "pavlov_xlam",
            "portfolio_suite": False,
            "evidence_scope": "component preflight only; not portfolio-wide evidence",
            "status": "BLOCKED",
            "launch_authorized": False,
            "launches_any_job": False,
            "blockers": ["xLAM component preflight missing"],
        }
    if isinstance(value, (str, Path)):
        value = _read_json(value)
    if not isinstance(value, Mapping):
        return {
            "component": XLAM_COMPONENT,
            "suite_id": "pavlov_xlam",
            "portfolio_suite": False,
            "evidence_scope": "component preflight only; not portfolio-wide evidence",
            "status": "BLOCKED",
            "launch_authorized": False,
            "launches_any_job": False,
            "blockers": ["xLAM component preflight must be a JSON object"],
        }
    _assert_metadata_only(value)
    if (
        value.get("component") == XLAM_COMPONENT
        and value.get("status") == "BLOCKED"
        and value.get("blockers") == ["xLAM component preflight missing"]
        and "revision" not in value
        and "task_hashes" not in value
    ):
        return {
            "component": XLAM_COMPONENT,
            "suite_id": str(value.get("suite_id", "pavlov_xlam")),
            "portfolio_suite": False,
            "evidence_scope": "component preflight only; not portfolio-wide evidence",
            "status": "BLOCKED",
            "launch_authorized": False,
            "launches_any_job": False,
            "blockers": ["xLAM component preflight missing"],
        }
    blockers: list[str] = []
    suite_id = str(value.get("suite_id", value.get("id", "pavlov_xlam")))
    if suite_id in EXPECTED_TRAIN_SUITE_IDS or suite_id in EXPECTED_PRIMARY_EVAL_SUITE_IDS:
        blockers.append("xLAM component suite ID collides with a portfolio suite")
    if value.get("status") not in (None, "READY") and "blockers" not in value:
        blockers.append("xLAM component preflight is not READY")
    try:
        revision = _extract_revision(value)
    except PortfolioSplitRegistryError as exc:
        revision = None
        blockers.append(str(exc))
    receipts, receipt_errors = _extract_receipts(value)
    blockers.extend(f"xLAM: {error}" for error in receipt_errors)
    role_hashes: dict[str, list[str]] = {}
    task_hashes = value.get("task_hashes")
    if isinstance(task_hashes, Mapping):
        for role_key, aliases in ((TRAIN_ROLE, ("train",)), (PRIMARY_EVAL_ROLE, ("primary_eval", "test"))):
            source = next((task_hashes[key] for key in aliases if key in task_hashes), None)
            if source is not None:
                role_hashes[role_key] = list(source) if _is_sequence(source) else []
    for role_key, aliases in ((TRAIN_ROLE, ("train_task_hashes",)), (PRIMARY_EVAL_ROLE, ("test_task_hashes", "primary_eval_task_hashes"))):
        if role_key not in role_hashes:
            source = _first_value(value, aliases)
            if source is not None and _is_sequence(source):
                role_hashes[role_key] = list(source)
    aggregate_hashes: dict[str, str] = {}
    for role, hashes in role_hashes.items():
        if not hashes or any(not isinstance(item, str) or not TASK_DIGEST.fullmatch(item) for item in hashes):
            blockers.append(f"xLAM {role} task hashes are not ordered SHA-256 digests")
            continue
        if len(set(hashes)) != len(hashes):
            blockers.append(f"xLAM duplicate task hash within {role}")
        aggregate, aggregate_errors = _extract_aggregate(value, role, hashes)
        aggregate_hashes[role] = aggregate
        blockers.extend(f"xLAM: {error}" for error in aggregate_errors)
    if not role_hashes:
        blockers.append("xLAM task hashes are missing")
    if set(role_hashes.get(TRAIN_ROLE, [])) & set(role_hashes.get(PRIMARY_EVAL_ROLE, [])):
        blockers.append("xLAM train/test contamination overlap")
    if portfolio_hashes and portfolio_hashes.intersection(
        digest for hashes in role_hashes.values() for digest in hashes
    ):
        blockers.append("xLAM cross-suite task-hash overlap with portfolio")
    output: dict[str, Any] = {
        "component": XLAM_COMPONENT,
        "suite_id": suite_id,
        "portfolio_suite": False,
        "evidence_scope": "component preflight only; not portfolio-wide evidence",
        "dataset_id": value.get("dataset_id"),
        "seed": value.get("seed"),
        "revision": revision,
        "receipt_refs": {key: receipts[key] for key in receipts if key in _RECEIPT_KEYS},
        "counts": {role: len(hashes) for role, hashes in sorted(role_hashes.items())},
        "task_hashes": {role: role_hashes[role] for role in sorted(role_hashes)},
        "aggregate_hashes": aggregate_hashes,
        "status": "READY" if not blockers else "BLOCKED",
        "launch_authorized": False,
        "launches_any_job": False,
        "blockers": sorted(set(blockers)),
    }
    return output


def _registry_digest(registry: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in registry.items() if key != "registry_sha256"}
    return _sha256(_stable_json(payload))


def build_registry(
    manifests: Any,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
    xlam_preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate 26 suite manifests into deterministic metadata-only output."""

    contract = _as_contract(contract)
    normalized, errors, expected, declared = _normalize_input(manifests, contract=contract)
    role_errors, metadata = _portfolio_errors(
        normalized,
        contract=contract,
        expected=expected,
        declared_domains=declared,
    )
    all_errors = errors + role_errors
    if all_errors:
        raise PortfolioSplitRegistryError("; ".join(dict.fromkeys(all_errors)))
    portfolio_hashes = {
        digest
        for item in normalized
        for digest in item.get("task_hashes", [])
    }
    component = _xlam_preflight(xlam_preflight, portfolio_hashes=portfolio_hashes)
    contract_gate = _contract_gate(contract)
    held_out = sorted(
        str(item["suite_id"])
        for item in normalized
        if item.get("role") == PRIMARY_EVAL_ROLE and item.get("held_out_receipt_proven")
    )
    suites = sorted(normalized, key=lambda item: (str(item["role"]), str(item["suite_id"])))
    blockers = list(component.get("blockers", [])) + list(contract_gate.get("blockers", []))
    status = "READY" if not blockers else "BLOCKED"
    registry = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "launch_authorized": False,
        "launches_any_job": False,
        "suite_counts": {TRAIN_ROLE: len(expected[TRAIN_ROLE]), PRIMARY_EVAL_ROLE: len(expected[PRIMARY_EVAL_ROLE])},
        "suite_ids": {role: list(expected[role]) for role in ROLES},
        "declared_domains": list(declared),
        "domain_unions": metadata["domain_unions"],
        "company_domain_coverage": metadata["company_domain_coverage"],
        "held_out_proven_suite_ids": held_out,
        "held_out_proven_count": len(held_out),
        "suites": suites,
        "cross_suite_overlap": {"overlap_count": len(metadata["overlaps"]), "overlaps": metadata["overlaps"]},
        "contract_gate": contract_gate,
        "xlam_component_preflight": component,
        "blockers": sorted(set(blockers)),
    }
    registry["registry_sha256"] = _registry_digest(registry)
    return registry


# Descriptive aliases make the small API convenient for callers and preserve
# compatibility with likely names used by local preflight scripts.
aggregate_manifests = build_registry
build_manifest_registry = build_registry
build_portfolio_registry = build_registry
validate_manifests = validate_portfolio


def validate_registry(
    registry: Mapping[str, Any] | str | Path,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
    expected_revisions: Mapping[str, str] | None = None,
) -> list[str]:
    """Return errors for an existing registry, including deterministic drift."""

    try:
        value = _read_json(registry) if isinstance(registry, (str, Path)) else registry
        if not isinstance(value, Mapping):
            return ["registry must be a JSON object"]
        _assert_metadata_only(value)
        suites = value.get("suites")
        if not _is_sequence(suites):
            return ["registry is missing suites"]
        if expected_revisions:
            for suite_id, revision in expected_revisions.items():
                for suite in suites:
                    if isinstance(suite, Mapping) and suite.get("suite_id") == suite_id and suite.get("revision") != revision:
                        return [f"revision drift for suite {suite_id}"]
        rebuilt = build_registry(
            suites,
            contract=contract,
            xlam_preflight=value.get("xlam_component_preflight"),
        )
        if _stable_json(dict(value)) != _stable_json(rebuilt):
            return ["registry metadata drift or non-canonical fields"]
        return []
    except PortfolioSplitRegistryError as exc:
        return [str(exc)]


def verify_registry(
    registry: Mapping[str, Any] | str | Path,
    *,
    contract: Mapping[str, Any] | str | Path | None = None,
    expected_revisions: Mapping[str, str] | None = None,
) -> bool:
    errors = validate_registry(registry, contract=contract, expected_revisions=expected_revisions)
    if errors:
        raise PortfolioSplitRegistryError("; ".join(errors))
    return True


verify_manifest_registry = verify_registry


def _manifest_paths(paths: Sequence[str], directory: str | None) -> list[Path]:
    result = [Path(path) for path in paths]
    if directory:
        result.extend(sorted(Path(directory).glob("*.json")))
    if not result:
        raise PortfolioSplitRegistryError("at least one --manifest or --manifest-dir is required")
    return result


def _load_manifest_files(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    loaded: list[Mapping[str, Any]] = []
    for path in paths:
        value = _read_json(path)
        if isinstance(value, Mapping) and _is_sequence(value.get("suites")):
            if any(not isinstance(item, Mapping) for item in value["suites"]):
                raise PortfolioSplitRegistryError(f"manifest file {path} contains a non-object suite")
            loaded.extend(value["suites"])
        elif isinstance(value, Mapping):
            loaded.append(value)
        elif _is_sequence(value):
            if any(not isinstance(item, Mapping) for item in value):
                raise PortfolioSplitRegistryError(f"manifest file {path} contains a non-object suite")
            loaded.extend(value)
        else:
            raise PortfolioSplitRegistryError(f"manifest file {path} is not a JSON object/list")
    return loaded


def _write_json(value: Any, path: str | None) -> None:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path:
        Path(path).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate", help="aggregate local suite manifests")
    generate.add_argument("--manifest", action="append", default=[])
    generate.add_argument("--manifest-dir")
    generate.add_argument("--xlam-preflight")
    generate.add_argument("--contract", default=str(CONTRACT_PATH))
    generate.add_argument("--out")
    verify = subparsers.add_parser("verify", help="verify an existing registry")
    verify.add_argument("--registry", required=True)
    verify.add_argument("--contract", default=str(CONTRACT_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in {"generate", "verify", "-h", "--help"}:
        args_list.insert(0, "generate")
    parser = _parser()
    args = parser.parse_args(args_list)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        contract = load_contract(args.contract)
        if args.command == "generate":
            paths = _manifest_paths(args.manifest, args.manifest_dir)
            manifests = _load_manifest_files(paths)
            xlam = _read_json(args.xlam_preflight) if args.xlam_preflight else None
            registry = build_registry(manifests, contract=contract, xlam_preflight=xlam)
            _write_json(registry, args.out)
            return 0
        registry = _read_json(args.registry)
        verify_registry(registry, contract=contract)
        return 0
    except PortfolioSplitRegistryError as exc:
        print(json.dumps({"status": "INVALID", "errors": [str(exc)]}, sort_keys=True), file=sys.stderr)
        return 2


try:
    _default_contract = load_contract()
    _expected = expected_suite_ids(_default_contract)
    EXPECTED_TRAIN_SUITE_IDS = _expected[TRAIN_ROLE]
    EXPECTED_PRIMARY_EVAL_SUITE_IDS = _expected[PRIMARY_EVAL_ROLE]
    DECLARED_DOMAINS = _declared_domains(_default_contract)
except PortfolioSplitRegistryError:  # pragma: no cover - useful for isolated imports
    EXPECTED_TRAIN_SUITE_IDS = ()
    EXPECTED_PRIMARY_EVAL_SUITE_IDS = ()
    DECLARED_DOMAINS = ()

PORTFOLIO_TRAIN_SUITE_IDS = EXPECTED_TRAIN_SUITE_IDS
PORTFOLIO_PRIMARY_EVAL_SUITE_IDS = EXPECTED_PRIMARY_EVAL_SUITE_IDS
DECLARED_DOMAIN_TAGS = DECLARED_DOMAINS
RegistryError = PortfolioSplitRegistryError
generate_registry = build_registry
make_registry = build_registry
verify = verify_registry


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
