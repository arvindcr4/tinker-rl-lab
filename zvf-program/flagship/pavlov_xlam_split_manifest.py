#!/usr/bin/env python3
"""Freeze and verify the non-content xLAM split used by the Pavlov task.

The existing :func:`platform_tinker.tinkerrl.grpo.make_xlam_dataset` function
owns parsing and shuffling the xLAM records.  This module deliberately does not
duplicate that logic.  It records only SHA-256 digests of the resulting
prompt/target pairs, together with the provenance fields needed to compose this
suite into the larger Pavlov portfolio.

No training, sampling, evaluation, or external write is performed here.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


DATASET_ID = "Salesforce/xlam-function-calling-60k"
DEFAULT_SEED = 809
DEFAULT_SUITE_ID = "pavlov_xlam"
DEFAULT_DOMAIN_TAGS = ("tool_use",)
TRAIN_ROLE = "train"
TEST_ROLE = "primary_eval"
SCHEMA_VERSION = "pavlov-xlam-split-manifest-v1"
HASH_ALGORITHM = "sha256"
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class SplitManifestError(ValueError):
    """Raised when a split manifest cannot be generated or verified."""


def _require_pinned_revision(revision: Any) -> str:
    """Return a lower-case full commit SHA, rejecting mutable references."""

    if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
        raise SplitManifestError(
            "dataset revision must be an immutable 40-character lower-case commit SHA"
        )
    return revision


def _json_default(value: Any) -> Any:
    """Make the small set of values used by test fakes JSON-canonicalizable."""

    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda item: repr(item))
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    raise TypeError(f"value of type {type(value).__name__} is not JSON-canonicalizable")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def task_hash(example: Any) -> str:
    """Hash one example without retaining its prompt, target, or metadata.

    The canonical identity uses the prompt/target fields consumed by
    ``make_xlam_dataset`` and, when retained by an adapter, a source task ID.
    Metadata can contain operational annotations and is never copied into the
    manifest or used as a source of raw content.
    """

    prompt, target = _example_fields(example)
    task_id = _example_task_id(example)
    identity = {"prompt": prompt, "target": target}
    if task_id is not None:
        # Dataset adapters that retain a source task ID get a stable ID-based
        # collision check as well as the content binding.  The ID itself is
        # never written to the manifest.
        identity["task_id"] = task_id
    try:
        canonical = _canonical_json(identity)
    except (TypeError, ValueError) as exc:
        raise SplitManifestError(
            f"example fields are not JSON-canonicalizable: {exc}"
        ) from exc
    return _sha256(canonical)


# Descriptive alias for callers that prefer the longer name.
hash_task = task_hash


def aggregate_hash(task_hashes: Iterable[str]) -> str:
    """Hash an ordered sequence of task hashes.

    Newline framing preserves order and makes the empty sequence deterministic.
    """

    return _sha256("\n".join(task_hashes))


def _aggregate_all(train_hashes: Sequence[str], test_hashes: Sequence[str]) -> str:
    framed = {
        "train": list(train_hashes),
        "test": list(test_hashes),
    }
    return _sha256(_canonical_json(framed))


def _example_fields(example: Any) -> tuple[Any, Any]:
    if isinstance(example, Mapping):
        if "prompt" not in example:
            raise SplitManifestError("dataset example is missing prompt")
        return example["prompt"], example.get("target")
    if not hasattr(example, "prompt"):
        raise SplitManifestError("dataset example is missing prompt")
    return getattr(example, "prompt"), getattr(example, "target", None)


def _example_task_id(example: Any) -> Any | None:
    if isinstance(example, Mapping):
        if "task_id" in example:
            return example["task_id"]
        if "id" in example:
            return example["id"]
        metadata = example.get("metadata")
    else:
        metadata = getattr(example, "metadata", None)
    if isinstance(metadata, Mapping):
        return metadata.get("task_id", metadata.get("id"))
    return None


def _split_examples(dataset: Any, split: str) -> list[Any]:
    """Adapt the real ``InMemoryDataset`` and small in-memory test fakes."""

    method_name = f"{split}_examples"
    method = getattr(dataset, method_name, None)
    if callable(method):
        return list(method())

    value = getattr(dataset, split, None)
    if value is not None and not callable(value):
        return list(value)

    if isinstance(dataset, Mapping) and split in dataset:
        return list(dataset[split])

    try:
        return list(dataset[split])
    except (KeyError, IndexError, TypeError):
        raise SplitManifestError(f"dataset does not expose a {split!r} split") from None


def _invoke_dataset_factory(
    factory: Callable[..., Any], *, seed: int, revision: str
) -> Any:
    """Call a fake or adapter factory without masking its own TypeErrors."""

    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "seed" in parameters:
        kwargs["seed"] = seed
    if accepts_kwargs or "revision" in parameters:
        kwargs["revision"] = revision
    return factory(**kwargs)


def make_xlam_dataset(seed: int = DEFAULT_SEED) -> Any:
    """Lazy compatibility wrapper for the existing dataset implementation.

    Keeping this symbol at module scope also gives tests a narrow seam for a
    small fake without importing ``datasets`` or downloading anything.
    """

    from platform_tinker.tinkerrl.grpo import make_xlam_dataset as factory

    return factory(seed=seed)


_DEFAULT_MAKE_XLAM_DATASET = make_xlam_dataset


def _dataset_at_revision(revision: str, seed: int) -> Any:
    """Run the existing loader while forcing its Hugging Face revision input."""

    # A test or embedding caller can replace the module-level seam with an
    # in-memory factory.  In that case do not import ``datasets`` at all.
    if make_xlam_dataset is not _DEFAULT_MAKE_XLAM_DATASET:
        return _invoke_dataset_factory(make_xlam_dataset, seed=seed, revision=revision)

    try:
        import datasets  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise SplitManifestError(
            "datasets is required when no in-memory dataset is supplied"
        ) from exc

    original_loader = datasets.load_dataset

    def pinned_loader(dataset_id: str, *args: Any, **kwargs: Any) -> Any:
        if dataset_id != DATASET_ID:
            raise SplitManifestError(
                f"xLAM loader requested unexpected dataset {dataset_id!r}"
            )
        requested = kwargs.get("revision")
        if requested is not None and requested != revision:
            raise SplitManifestError(
                "dataset loader revision disagrees with the manifest revision"
            )
        kwargs["revision"] = revision
        return original_loader(dataset_id, *args, **kwargs)

    # make_xlam_dataset imports datasets.load_dataset inside its body.  A
    # temporary attribute replacement lets us reuse that implementation without
    # changing its public signature or its parsing/split behavior.
    datasets.load_dataset = pinned_loader
    try:
        return make_xlam_dataset(seed=seed)
    finally:
        datasets.load_dataset = original_loader


def _dataset_for_revision(
    revision: str,
    *,
    seed: int,
    dataset: Any | None,
    dataset_factory: Callable[..., Any] | None,
) -> Any:
    if dataset is not None and dataset_factory is not None:
        raise SplitManifestError("supply dataset or dataset_factory, not both")
    if dataset is not None:
        return dataset
    if dataset_factory is not None:
        return _invoke_dataset_factory(dataset_factory, seed=seed, revision=revision)
    return _dataset_at_revision(revision, seed)


def _normalise_tags(tags: Iterable[str] | None) -> list[str]:
    values = DEFAULT_DOMAIN_TAGS if tags is None else tags
    normalised = sorted({str(tag).strip() for tag in values if str(tag).strip()})
    if not normalised:
        raise SplitManifestError("at least one domain tag is required")
    return normalised


def _normalise_split_roles(value: Mapping[str, str] | None) -> dict[str, str]:
    if value is None:
        return {"train": TRAIN_ROLE, "test": TEST_ROLE}
    result = {str(key): str(role) for key, role in value.items()}
    if result.get("train") != TRAIN_ROLE or result.get("test") != TEST_ROLE:
        raise SplitManifestError(
            "split_roles must assign train to 'train' and test to 'primary_eval'"
        )
    return {"test": result["test"], "train": result["train"]}


def _normalise_receipts(
    revision: str,
    receipt_refs: Mapping[str, Any] | None,
    *,
    split_manifest_digest: str | None = None,
) -> dict[str, str]:
    supplied = dict(receipt_refs or {})
    defaults = {
        "revision": f"hf://{DATASET_ID}@{revision}",
        "license": "UNRECORDED",
        "container": "UNRECORDED",
        "decontamination": "UNRECORDED",
        "split_manifest": "UNRECORDED",
    }
    aliases = {
        "revision_receipt_ref": "revision",
        "license_receipt_ref": "license",
        "container_receipt_ref": "container",
        "container_runtime_receipt_ref": "container",
        "runtime_receipt_ref": "container",
        "decontamination_receipt_ref": "decontamination",
        "split_manifest_receipt_ref": "split_manifest",
        "task_hash_receipt_ref": "split_manifest",
    }
    for alias, canonical in aliases.items():
        if alias in supplied and canonical in supplied and supplied[alias] != supplied[canonical]:
            raise SplitManifestError(
                f"receipt references {alias!r} and {canonical!r} disagree"
            )
        if alias in supplied and canonical not in supplied:
            supplied[canonical] = supplied[alias]
    for key, default in defaults.items():
        if key == "split_manifest" and key not in supplied and split_manifest_digest:
            supplied[key] = f"sha256:{split_manifest_digest}"
        value = supplied.get(key, default)
        if not isinstance(value, str) or not value.strip():
            raise SplitManifestError(f"receipt reference {key!r} must be non-empty")
        defaults[key] = value.strip()
    return defaults


def _hash_splits(dataset: Any) -> tuple[list[str], list[str]]:
    train = [task_hash(example) for example in _split_examples(dataset, "train")]
    test = [task_hash(example) for example in _split_examples(dataset, "test")]
    train_seen = set()
    duplicate_train: list[str] = []
    for item in train:
        if item in train_seen:
            duplicate_train.append(item)
        train_seen.add(item)
    test_seen = set()
    duplicate_test: list[str] = []
    for item in test:
        if item in test_seen:
            duplicate_test.append(item)
        test_seen.add(item)
    if duplicate_train or duplicate_test:
        split = "train" if duplicate_train else "test"
        raise SplitManifestError(f"duplicate task hash within {split} split")
    overlap = sorted(set(train).intersection(test))
    if overlap:
        raise SplitManifestError(
            "train/test contamination: overlapping task hashes " + ", ".join(overlap)
        )
    return train, test


def _manifest_hashes(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    task_hashes = manifest.get("task_hashes")
    if isinstance(task_hashes, Mapping):
        train = task_hashes.get("train")
        test = task_hashes.get("test")
    else:
        train = manifest.get("train_task_hashes")
        test = manifest.get("test_task_hashes")
    return {
        "train": list(train) if isinstance(train, list) else [],
        "test": list(test) if isinstance(test, list) else [],
    }


def _suite_entries(manifest: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    suite_id = str(manifest.get("suite_id", "<unknown-suite>"))
    hashes = _manifest_hashes(manifest)
    return [
        (suite_id, role, digest)
        for role in ("train", "test")
        for digest in hashes[role]
    ]


def cross_suite_task_hash_overlaps(
    manifest: Mapping[str, Any], other_manifests: Iterable[Mapping[str, Any]] = ()
) -> list[dict[str, str]]:
    """Return hash collisions between this suite and other portfolio suites."""

    current_suite = str(manifest.get("suite_id", "<unknown-suite>"))
    current = _suite_entries(manifest)
    others: list[tuple[str, str, str]] = []
    for other in other_manifests:
        if not isinstance(other, Mapping):
            raise SplitManifestError("cross-suite manifest must be a JSON object")
        other_suite = str(other.get("suite_id", "<unknown-suite>"))
        if other_suite == current_suite and other is not manifest:
            raise SplitManifestError(
                f"duplicate suite_id {current_suite!r} in portfolio manifests"
            )
        others.extend(_suite_entries(other))
    collisions: list[dict[str, str]] = []
    for suite_id, role, digest in current:
        for other_suite, other_role, other_digest in others:
            if digest == other_digest:
                collisions.append(
                    {
                        "suite_id": suite_id,
                        "split_role": role,
                        "other_suite_id": other_suite,
                        "other_split_role": other_role,
                        "task_hash": digest,
                    }
                )
    return sorted(
        collisions,
        key=lambda item: (
            item["task_hash"],
            item["other_suite_id"],
            item["other_split_role"],
        ),
    )


# Shorter alias used by a few portfolio callers.
check_cross_suite_overlap = cross_suite_task_hash_overlaps


def _portfolio_roles(manifest: Mapping[str, Any]) -> set[str]:
    """Return portfolio roles represented by one manifest.

    A portfolio may store one manifest per role (``suite_role``) or one
    dataset manifest containing both train and primary-evaluation splits (the
    xLAM shape).  Supporting both keeps this artifact composable without
    pretending that xLAM alone covers the portfolio.
    """

    explicit = manifest.get("suite_role", manifest.get("portfolio_split_role"))
    if isinstance(explicit, str):
        return {explicit}
    roles = manifest.get("split_roles")
    if isinstance(roles, Mapping):
        return {str(role) for role in roles.values()}
    return set()


def validate_portfolio_domain_coverage(
    manifests: Iterable[Mapping[str, Any]],
    *,
    declared_domains: Iterable[str],
    company_required_domains: Mapping[str, Iterable[str]] | None = None,
    expected_training_suite_count: int = 12,
    expected_primary_eval_suite_count: int = 14,
) -> list[str]:
    """Validate the full portfolio's role counts, domain unions, and receipts.

    This check is intentionally separate from :func:`build_manifest`: one xLAM
    suite cannot establish the 12/14 portfolio claim.  Callers must supply all
    suite manifests before treating the claim as satisfied.
    """

    entries = list(manifests)
    errors: list[str] = []
    declared = {str(domain).strip() for domain in declared_domains if str(domain).strip()}
    if len(declared) != 16:
        errors.append(f"declared portfolio domain set must contain 16 domains, got {len(declared)}")

    train_suites: set[str] = set()
    eval_suites: set[str] = set()
    train_domains: set[str] = set()
    eval_domains: set[str] = set()
    seen_hashes: dict[str, tuple[str, str]] = {}
    for manifest in entries:
        if not isinstance(manifest, Mapping):
            errors.append("portfolio manifest must be a JSON object")
            continue
        suite_id = str(manifest.get("suite_id", "<unknown-suite>"))
        domains = manifest.get("domain_tags", manifest.get("domains", []))
        if not isinstance(domains, list):
            errors.append(f"{suite_id}: domain_tags must be a list")
            domains = []
        domain_set = {str(domain) for domain in domains}
        roles = _portfolio_roles(manifest)
        if TRAIN_ROLE in roles:
            train_suites.add(suite_id)
            train_domains.update(domain_set)
        if TEST_ROLE in roles:
            eval_suites.add(suite_id)
            eval_domains.update(domain_set)
        receipts = manifest.get("receipt_refs")
        for receipt in (
            "revision",
            "license",
            "container",
            "decontamination",
            "split_manifest",
        ):
            value = receipts.get(receipt) if isinstance(receipts, Mapping) else None
            if not isinstance(value, str) or not value.strip() or value.strip() == "UNRECORDED":
                errors.append(f"{suite_id}: missing {receipt} receipt reference")
        for _, role, digest in _suite_entries(manifest):
            previous = seen_hashes.get(digest)
            if previous is not None and previous != (suite_id, role):
                errors.append(
                    "cross-suite task-hash overlap: "
                    f"{previous[0]}:{previous[1]} and {suite_id}:{role} ({digest})"
                )
            else:
                seen_hashes[digest] = (suite_id, role)

    if len(train_suites) != expected_training_suite_count:
        errors.append(
            f"expected {expected_training_suite_count} train suites, got {len(train_suites)}"
        )
    if len(eval_suites) != expected_primary_eval_suite_count:
        errors.append(
            "expected "
            f"{expected_primary_eval_suite_count} primary_eval suites, "
            f"got {len(eval_suites)}"
        )
    missing_train = sorted(declared - train_domains)
    missing_eval = sorted(declared - eval_domains)
    unknown_train = sorted(train_domains - declared)
    unknown_eval = sorted(eval_domains - declared)
    if unknown_train:
        errors.append("train domain union has undeclared domains: " + ", ".join(unknown_train))
    if unknown_eval:
        errors.append(
            "primary_eval domain union has undeclared domains: " + ", ".join(unknown_eval)
        )
    if missing_train:
        errors.append("train domain union missing: " + ", ".join(missing_train))
    if missing_eval:
        errors.append("primary_eval domain union missing: " + ", ".join(missing_eval))
    for company, required in (company_required_domains or {}).items():
        required_set = {str(domain) for domain in required}
        if not required_set:
            errors.append(f"{company}: required domain set is empty")
            continue
        if not required_set.issubset(train_domains):
            errors.append(f"{company}: required domains are not all covered by train union")
        if not required_set.issubset(eval_domains):
            errors.append(f"{company}: required domains are not all covered by primary_eval union")
    return errors


validate_domain_coverage = validate_portfolio_domain_coverage


def verify_portfolio_domain_coverage(
    manifests: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> bool:
    errors = validate_portfolio_domain_coverage(manifests, **kwargs)
    if errors:
        raise SplitManifestError("invalid Pavlov portfolio: " + "; ".join(errors))
    return True


def build_manifest(
    revision: str,
    *,
    dataset: Any | None = None,
    dataset_factory: Callable[..., Any] | None = None,
    seed: int = DEFAULT_SEED,
    suite_id: str = DEFAULT_SUITE_ID,
    domain_tags: Iterable[str] | None = None,
    split_roles: Mapping[str, str] | None = None,
    receipt_refs: Mapping[str, Any] | None = None,
    cross_suite_manifests: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a deterministic, content-free manifest for the xLAM split."""

    revision = _require_pinned_revision(revision)
    if seed != DEFAULT_SEED:
        raise SplitManifestError(f"xLAM split manifest is frozen to seed {DEFAULT_SEED}")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise SplitManifestError("suite_id must be a non-empty string")
    suite_id = suite_id.strip()
    source_dataset = _dataset_for_revision(
        revision,
        seed=seed,
        dataset=dataset,
        dataset_factory=dataset_factory,
    )
    source_revision = getattr(source_dataset, "revision", None)
    if source_revision is not None and source_revision != revision:
        raise SplitManifestError(
            f"revision drift: dataset has {source_revision}, expected {revision}"
        )
    train_hashes, test_hashes = _hash_splits(source_dataset)
    tags = _normalise_tags(domain_tags)
    roles = _normalise_split_roles(split_roles)
    aggregate = _aggregate_all(train_hashes, test_hashes)
    receipts = _normalise_receipts(
        revision,
        receipt_refs,
        split_manifest_digest=aggregate,
    )

    provisional: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "dataset_split",
        "suite_id": suite_id,
        "dataset_id": DATASET_ID,
        "dataset": {"id": DATASET_ID, "revision": revision},
        "revision": revision,
        "dataset_revision": revision,
        "immutable_revision": revision,
        "revision_is_immutable": True,
        "seed": seed,
        "domain_tags": tags,
        "split_roles": roles,
        # Keep a singular compatibility field for consumers that model one
        # manifest as one role while retaining the train/test role mapping.
        "split_role": roles,
        "counts": {"train": len(train_hashes), "test": len(test_hashes)},
        "train_count": len(train_hashes),
        "test_count": len(test_hashes),
        "task_hashes": {"train": train_hashes, "test": test_hashes},
        "train_task_hashes": train_hashes,
        "test_task_hashes": test_hashes,
        "splits": {
            "train": {
                "role": roles["train"],
                "count": len(train_hashes),
                "task_hashes": train_hashes,
                "aggregate_sha256": aggregate_hash(train_hashes),
            },
            "test": {
                "role": roles["test"],
                "count": len(test_hashes),
                "task_hashes": test_hashes,
                "aggregate_sha256": aggregate_hash(test_hashes),
            },
        },
        "aggregate_hashes": {
            "train": aggregate_hash(train_hashes),
            "test": aggregate_hash(test_hashes),
            "all": aggregate,
        },
        "train_aggregate_sha256": aggregate_hash(train_hashes),
        "test_aggregate_sha256": aggregate_hash(test_hashes),
        "aggregate_sha256": aggregate,
        "hashing": {
            "algorithm": HASH_ALGORITHM,
            "task_identity": "canonical JSON of task fields plus optional source ID",
            "ordered_task_hashes": True,
            "aggregate_framing": "newline-joined ordered task hashes; all uses canonical split map",
        },
        "receipt_refs": receipts,
        "provenance_receipts": {
            "revision_receipt_ref": receipts["revision"],
            "license_receipt_ref": receipts["license"],
            "container_receipt_ref": receipts["container"],
            "decontamination_receipt_ref": receipts["decontamination"],
            "split_manifest_receipt_ref": receipts["split_manifest"],
        },
        "revision_receipt_ref": receipts["revision"],
        "license_receipt_ref": receipts["license"],
        "container_receipt_ref": receipts["container"],
        "container_runtime_receipt_ref": receipts["container"],
        "decontamination_receipt_ref": receipts["decontamination"],
        "split_manifest_receipt_ref": receipts["split_manifest"],
        "task_hash_receipt_ref": receipts["split_manifest"],
        "train_test_overlap": [],
        "cross_suite_overlap": {
            "checked_suite_ids": [],
            "overlap_count": 0,
            "overlaps": [],
        },
        "cross_suite_task_hash_overlaps": [],
        "status": "BLOCKED",
        "launch_authorized": False,
        "evidence_scope": "seed-809 split artifact only; not portfolio-wide evidence",
        "portfolio_contract": {
            "training_suite_count": 12,
            "primary_eval_suite_count": 14,
            "expected_training_suite_count": 12,
            "expected_primary_eval_suite_count": 14,
            "declared_domain_count": 16,
            "role_union_coverage_claim": "requires_full_portfolio_verification",
            "role_union_requirement": {
                "train": {
                    "suite_count": 12,
                    "must_cover_all_declared_domains": True,
                },
                "primary_eval": {
                    "suite_count": 14,
                    "must_cover_all_declared_domains": True,
                },
            },
            "primary_eval_is_not_assumed_held_out": True,
            "all_suite_receipts_required": [
                "revision",
                "license",
                "container",
                "decontamination",
                "split_manifest",
            ],
            "suite_is_one_portfolio_member": True,
        },
        "launches_any_job": False,
    }
    others = list(cross_suite_manifests)
    collisions = cross_suite_task_hash_overlaps(provisional, others)
    if collisions:
        first = collisions[0]
        raise SplitManifestError(
            "cross-suite task-hash overlap: "
            f"{first['suite_id']}:{first['split_role']} overlaps "
            f"{first['other_suite_id']}:{first['other_split_role']} "
            f"({first['task_hash']})"
        )
    checked = sorted(
        {
            str(other.get("suite_id"))
            for other in others
            if isinstance(other, Mapping)
        }
    )
    provisional["cross_suite_overlap"] = {
        "checked_suite_ids": checked,
        "overlap_count": 0,
        "overlaps": [],
    }
    return provisional


# Common builder spelling for external portfolio code.
generate_manifest = build_manifest
make_manifest = build_manifest


def build_split_manifest(
    dataset: Any,
    revision: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Dataset-first spelling for portfolio adapters and test fakes."""

    return build_manifest(revision, dataset=dataset, **kwargs)


_FORBIDDEN_CONTENT_KEYS = {
    "prompt",
    "prompts",
    "target",
    "targets",
}


def _find_forbidden_keys(value: Any, path: str = "manifest") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in _FORBIDDEN_CONTENT_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_find_forbidden_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_revision: str | None = None,
    dataset: Any | None = None,
    cross_suite_manifests: Iterable[Mapping[str, Any]] = (),
) -> list[str]:
    """Return all detected contract violations without mutating the manifest."""

    errors: list[str] = []
    if not isinstance(manifest, Mapping):
        return ["manifest must be a JSON object"]
    forbidden = _find_forbidden_keys(manifest)
    if forbidden:
        errors.append("manifest contains raw task content at " + ", ".join(forbidden))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is not the xLAM split-manifest schema")
    if manifest.get("dataset_id") != DATASET_ID:
        errors.append("dataset_id does not match Salesforce/xlam-function-calling-60k")
    dataset_meta = manifest.get("dataset")
    if not isinstance(dataset_meta, Mapping):
        errors.append("dataset metadata must record id and revision")
    else:
        if dataset_meta.get("id") != DATASET_ID:
            errors.append("dataset.id does not match dataset_id")
        if dataset_meta.get("revision") != manifest.get("revision"):
            errors.append("dataset.revision does not match revision")
    revision = manifest.get("revision")
    try:
        pinned_revision = _require_pinned_revision(revision)
    except SplitManifestError as exc:
        errors.append(str(exc))
        pinned_revision = None
    if expected_revision is not None:
        try:
            expected = _require_pinned_revision(expected_revision)
        except SplitManifestError as exc:
            errors.append("expected revision: " + str(exc))
        else:
            if pinned_revision is not None and expected != pinned_revision:
                errors.append(
                    f"revision drift: manifest has {pinned_revision}, expected {expected}"
                )
    if manifest.get("immutable_revision") != revision:
        errors.append("immutable_revision does not match revision")
    if manifest.get("dataset_revision") != revision:
        errors.append("dataset_revision does not match revision")
    if manifest.get("revision_is_immutable") is not True:
        errors.append("revision_is_immutable must be true")
    if manifest.get("seed") != DEFAULT_SEED:
        errors.append(f"seed must be frozen to {DEFAULT_SEED}")
    if not isinstance(manifest.get("suite_id"), str) or not manifest.get("suite_id"):
        errors.append("suite_id must be a non-empty string")
    if manifest.get("status") != "BLOCKED":
        errors.append("split manifest status must remain BLOCKED")
    if (
        manifest.get("launch_authorized") is not False
        or manifest.get("launches_any_job") is not False
    ):
        errors.append("split manifest must not authorize or launch a job")
    if manifest.get("split_roles") != {"train": TRAIN_ROLE, "test": TEST_ROLE}:
        errors.append("split_roles must preserve train and primary_eval roles")
    if manifest.get("split_role") != manifest.get("split_roles"):
        errors.append("split_role does not match split_roles")
    portfolio_contract = manifest.get("portfolio_contract")
    if not isinstance(portfolio_contract, Mapping):
        errors.append("portfolio_contract must describe the 12 train / 14 primary_eval structure")
    else:
        if portfolio_contract.get("training_suite_count") != 12:
            errors.append("portfolio_contract.training_suite_count must be 12")
        if portfolio_contract.get("primary_eval_suite_count") != 14:
            errors.append("portfolio_contract.primary_eval_suite_count must be 14")
        if portfolio_contract.get("declared_domain_count") != 16:
            errors.append("portfolio_contract.declared_domain_count must be 16")
        role_requirement = portfolio_contract.get("role_union_requirement")
        if not isinstance(role_requirement, Mapping):
            errors.append("portfolio_contract.role_union_requirement is missing")
        else:
            for role, count in ((TRAIN_ROLE, 12), (TEST_ROLE, 14)):
                requirement = role_requirement.get(role)
                if not isinstance(requirement, Mapping):
                    errors.append(f"portfolio_contract.role_union_requirement.{role} is missing")
                    continue
                if requirement.get("suite_count") != count:
                    errors.append(
                        "portfolio_contract.role_union_requirement."
                        f"{role}.suite_count must be {count}"
                    )
                if requirement.get("must_cover_all_declared_domains") is not True:
                    errors.append(
                        f"portfolio_contract.role_union_requirement.{role} must cover all domains"
                    )
        if portfolio_contract.get("primary_eval_is_not_assumed_held_out") is not True:
            errors.append("primary_eval must not be relabeled as universally held-out")

    hashes = _manifest_hashes(manifest)
    for split in ("train", "test"):
        values = hashes[split]
        if not isinstance(
            manifest.get("task_hashes", {}).get(split)
            if isinstance(manifest.get("task_hashes"), Mapping)
            else manifest.get(f"{split}_task_hashes"),
            list,
        ):
            errors.append(f"{split} task hashes must be a list")
        bad = [item for item in values if not isinstance(item, str) or not _DIGEST.fullmatch(item)]
        if bad:
            errors.append(f"{split} contains a non-SHA-256 task hash")
        valid_values = [item for item in values if isinstance(item, str)]
        if len(valid_values) != len(set(valid_values)):
            errors.append(f"duplicate task hash within {split} split")
    overlap = sorted(
        set(item for item in hashes["train"] if isinstance(item, str)).intersection(
            item for item in hashes["test"] if isinstance(item, str)
        )
    )
    if overlap:
        errors.append("train/test overlap: " + ", ".join(overlap))
    if manifest.get("train_test_overlap") not in ([], None):
        errors.append("manifest records train/test overlap")

    counts = manifest.get("counts")
    expected_counts = {"train": len(hashes["train"]), "test": len(hashes["test"])}
    if counts != expected_counts:
        errors.append("counts do not match ordered task-hash lengths")
    if manifest.get("train_count") != expected_counts["train"]:
        errors.append("train_count does not match ordered task-hash length")
    if manifest.get("test_count") != expected_counts["test"]:
        errors.append("test_count does not match ordered task-hash length")

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        errors.append("splits must record train and test metadata")
    else:
        for split in ("train", "test"):
            split_record = splits.get(split)
            if not isinstance(split_record, Mapping):
                errors.append(f"splits.{split} must be an object")
                continue
            if split_record.get("task_hashes") != hashes[split]:
                errors.append(f"splits.{split}.task_hashes do not match ordered task hashes")
            if split_record.get("count") != expected_counts[split]:
                errors.append(f"splits.{split}.count does not match ordered task-hash length")
            expected_role = TRAIN_ROLE if split == "train" else TEST_ROLE
            if split_record.get("role") != expected_role:
                errors.append(f"splits.{split}.role is not {expected_role!r}")
            if split_record.get("aggregate_sha256") != aggregate_hash(
                [item for item in hashes[split] if isinstance(item, str)]
            ):
                errors.append(f"splits.{split}.aggregate_sha256 does not match task hashes")

    safe_hashes = {
        split: [item for item in hashes[split] if isinstance(item, str)]
        for split in ("train", "test")
    }
    expected_aggregates = {
        "train": aggregate_hash(safe_hashes["train"]),
        "test": aggregate_hash(safe_hashes["test"]),
        "all": _aggregate_all(safe_hashes["train"], safe_hashes["test"]),
    }
    if manifest.get("aggregate_hashes") != expected_aggregates:
        errors.append("aggregate_hashes do not match ordered task hashes")
    for key, field in (
        ("train", "train_aggregate_sha256"),
        ("test", "test_aggregate_sha256"),
        ("all", "aggregate_sha256"),
    ):
        if manifest.get(field) != expected_aggregates[key]:
            errors.append(f"{field} does not match ordered task hashes")

    receipts = manifest.get("receipt_refs")
    if not isinstance(receipts, Mapping):
        errors.append(
            "receipt_refs must record revision/license/container/decontamination references"
        )
    else:
        for key in (
            "revision",
            "license",
            "container",
            "decontamination",
            "split_manifest",
        ):
            if not isinstance(receipts.get(key), str) or not receipts.get(key, "").strip():
                errors.append(f"receipt_refs.{key} must be non-empty")
        for key in (
            "revision",
            "license",
            "container",
            "decontamination",
            "split_manifest",
        ):
            alias = f"{key}_receipt_ref"
            if key == "split_manifest" and manifest.get(alias) is None:
                alias = "task_hash_receipt_ref"
            if manifest.get(alias) != receipts.get(key):
                errors.append(f"{alias} does not match receipt_refs.{key}")
        if manifest.get("container_runtime_receipt_ref") != receipts.get("container"):
            errors.append("container_runtime_receipt_ref does not match receipt_refs.container")
        split_receipt = receipts.get("split_manifest")
        if isinstance(split_receipt, str) and split_receipt.startswith("sha256:"):
            if split_receipt.removeprefix("sha256:") != expected_aggregates["all"]:
                errors.append("receipt_refs.split_manifest does not match aggregate hash")

    if dataset is not None and pinned_revision is not None:
        try:
            actual_train, actual_test = _hash_splits(dataset)
        except SplitManifestError as exc:
            errors.append(str(exc))
        else:
            if actual_train != hashes["train"]:
                errors.append("train task hashes drift from supplied dataset")
            if actual_test != hashes["test"]:
                errors.append("test task hashes drift from supplied dataset")
            actual_revision = getattr(dataset, "revision", None)
            if actual_revision is not None and actual_revision != pinned_revision:
                errors.append(
                    f"revision drift: dataset has {actual_revision}, manifest has {pinned_revision}"
                )

    others = list(cross_suite_manifests)
    try:
        collisions = cross_suite_task_hash_overlaps(manifest, others)
    except SplitManifestError as exc:
        errors.append(str(exc))
    else:
        if collisions:
            errors.append(
                "cross-suite task-hash overlap: "
                + ", ".join(item["task_hash"] for item in collisions)
            )
        recorded = manifest.get("cross_suite_overlap")
        if isinstance(recorded, Mapping):
            recorded_count = recorded.get("overlap_count")
            if recorded_count not in (0, len(collisions)):
                errors.append("cross_suite_overlap.overlap_count is inconsistent")
    return errors


def verify_manifest(
    manifest: Mapping[str, Any] | str | Path,
    *,
    expected_revision: str | None = None,
    revision: str | None = None,
    current_revision: str | None = None,
    dataset: Any | None = None,
    cross_suite_manifests: Iterable[Mapping[str, Any]] = (),
) -> bool:
    """Raise on the first verification failure and return ``True`` on success."""

    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    supplied_revisions = [
        value
        for value in (expected_revision, revision, current_revision)
        if value is not None
    ]
    if supplied_revisions and any(
        value != supplied_revisions[0] for value in supplied_revisions[1:]
    ):
        raise SplitManifestError("revision verification arguments disagree")
    expected = supplied_revisions[0] if supplied_revisions else None
    errors = validate_manifest(
        manifest,
        expected_revision=expected,
        dataset=dataset,
        cross_suite_manifests=cross_suite_manifests,
    )
    if errors:
        raise SplitManifestError("invalid xLAM split manifest: " + "; ".join(errors))
    return True


def verify_split_manifest(
    manifest: Mapping[str, Any],
    dataset: Any | None = None,
    *,
    revision: str | None = None,
    current_revision: str | None = None,
    cross_suite_manifests: Iterable[Mapping[str, Any]] = (),
) -> bool:
    """Dataset-first verifier spelling used by suite-portfolio callers."""

    return verify_manifest(
        manifest,
        revision=revision,
        current_revision=current_revision,
        dataset=dataset,
        cross_suite_manifests=cross_suite_manifests,
    )


def verify_manifest_file(
    path: str | Path,
    *,
    expected_revision: str | None = None,
    revision: str | None = None,
    current_revision: str | None = None,
    dataset: Any | None = None,
    cross_suite_manifests: Iterable[Mapping[str, Any]] = (),
) -> bool:
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    return verify_manifest(
        manifest,
        expected_revision=expected_revision,
        revision=revision,
        current_revision=current_revision,
        dataset=dataset,
        cross_suite_manifests=cross_suite_manifests,
    )


def _read_manifests(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitManifestError(f"cannot read portfolio manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SplitManifestError(f"portfolio manifest {path} is not a JSON object")
        result.append(payload)
    return result


def _render(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("generate", "verify"))
    parser.add_argument("--manifest", type=Path, help="manifest to verify (or output with --out)")
    parser.add_argument(
        "--verify", dest="verify_path", type=Path, help="compatibility alias for verify"
    )
    parser.add_argument(
        "--revision",
        "--dataset-revision",
        "--revision-input",
        dest="revision",
        help="40-character immutable dataset commit SHA",
    )
    parser.add_argument("--out", type=Path, help="write generated manifest here")
    parser.add_argument(
        "--cross-suite-manifest",
        dest="cross_suite_paths",
        action="append",
        type=Path,
        default=[],
        help="portfolio manifest to check for task-hash overlap; repeatable",
    )
    parser.add_argument(
        "--dataset-check", action="store_true", help="reload and rehash the pinned dataset"
    )
    parser.add_argument("--suite-id", default=DEFAULT_SUITE_ID)
    parser.add_argument("--domain-tag", dest="domain_tags", action="append", default=None)
    parser.add_argument("--revision-receipt-ref")
    parser.add_argument("--license-receipt-ref")
    parser.add_argument("--container-receipt-ref")
    parser.add_argument("--decontamination-receipt-ref")
    parser.add_argument("--split-manifest-receipt-ref")
    args = parser.parse_args(argv)

    command = args.command
    if args.verify_path is not None:
        if command not in (None, "verify"):
            parser.error("--verify cannot be combined with generate")
        command = "verify"
        args.manifest = args.verify_path
    if command is None:
        command = "verify" if args.manifest is not None else "generate"

    try:
        portfolio = _read_manifests(args.cross_suite_paths)
        if command == "verify":
            if args.manifest is None:
                parser.error("verify requires --manifest or --verify")
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            dataset = None
            if args.dataset_check:
                revision = args.revision or manifest.get("revision")
                revision = _require_pinned_revision(revision)
                dataset = _dataset_for_revision(
                    revision,
                    seed=DEFAULT_SEED,
                    dataset=None,
                    dataset_factory=None,
                )
            verify_manifest(
                manifest,
                expected_revision=args.revision,
                dataset=dataset,
                cross_suite_manifests=portfolio,
            )
            print(json.dumps({"verified": True, "manifest": str(args.manifest)}, sort_keys=True))
            return 0

        if args.revision is None:
            parser.error("generate requires --revision")
        receipts = {
            key: value
            for key, value in {
                "revision": args.revision_receipt_ref,
                "license": args.license_receipt_ref,
                "container": args.container_receipt_ref,
                "decontamination": args.decontamination_receipt_ref,
                "split_manifest": args.split_manifest_receipt_ref,
            }.items()
            if value is not None
        }
        manifest = build_manifest(
            args.revision,
            suite_id=args.suite_id,
            domain_tags=args.domain_tags,
            receipt_refs=receipts,
            cross_suite_manifests=portfolio,
        )
        rendered = _render(manifest)
        if args.out is None:
            sys.stdout.write(rendered)
        else:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
            print(json.dumps({"generated": True, "manifest": str(args.out)}, sort_keys=True))
        return 0
    except (SplitManifestError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
