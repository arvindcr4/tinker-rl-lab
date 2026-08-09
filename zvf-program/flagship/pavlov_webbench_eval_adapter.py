#!/usr/bin/env python3
"""Offline WebBench evaluation-boundary adapter for Pavlov's primary suite.

The adapter defines the exact WebBench boundary and validates manifests before a
runner is allowed to use them.  It performs no WebBench download, browser work,
Tinker call, W&B call, Hugging Face call, or result fabrication.  A structurally
valid ``primary_eval`` manifest may remain pending receipts; it is never called
receipt-proven-heldout until all immutable provenance and result receipts pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = "pavlov-webbench-eval-boundary-v1"
WEBBENCH_SUITE_ID = "webbench_eval"
WEBBENCH_NAME = "Halluminate/WebBench"
WEBBENCH_SOURCE_URL = "https://github.com/Halluminate/WebBench"
WEBBENCH_ROLE = "primary_eval"
WEBBENCH_SPLIT = "evaluation"
WEBBENCH_DOMAINS = ("browser", "computer_use", "enterprise")
WEBBENCH_RECEIPT_FIELDS = (
    "revision",
    "license",
    "split",
    "task",
    "container",
    "decontamination",
)
RESULT_RECEIPT_FIELDS = ("wandb", "tinker", "hf")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_hex(value: Any) -> str:
    """Hash canonical JSON, used for deterministic task/split manifests."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def task_ids_hash(task_ids: Sequence[str]) -> str:
    return sha256_hex(list(task_ids))


def split_manifest_hash(split_manifest: Mapping[str, Any]) -> str:
    return sha256_hex(dict(split_manifest))


def receipt_digest(binding: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    return sha256_hex({"binding": dict(binding), "payload": dict(payload)})


def _strict_hex(value: Any, length: int) -> bool:
    if type(value) is not str:
        return False
    pattern = HEX40_RE if length == 40 else HEX64_RE if length == 64 else None
    return bool(pattern and pattern.fullmatch(value))


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value))


def _add(blockers: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    blockers.append({"code": code, "message": message, "details": details})


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _url(value: Any, *, host: str | None = None) -> tuple[bool, str]:
    if type(value) is not str or not value.strip():
        return False, "url_not_string"
    try:
        parsed = urlsplit(value)
        actual_host = parsed.hostname.lower() if parsed.hostname else ""
        port = parsed.port
    except ValueError:
        return False, "url_parse_error"
    if parsed.scheme != "https" or not actual_host or parsed.username or parsed.password or port is not None:
        return False, "url_not_https_or_hosted"
    if actual_host.endswith(".invalid") or actual_host in {"example.com", "example.org", "example.net"}:
        return False, "url_placeholder_host"
    if host is not None and actual_host != host:
        return False, "url_host_not_authoritative"
    return True, "ok"


def _authoritative_url(value: Any, host: str) -> tuple[bool, str]:
    """Require HTTPS plus the named host or one of its real subdomains."""

    valid, reason = _url(value)
    if not valid:
        return False, reason
    try:
        actual_host = urlsplit(value).hostname.lower() if urlsplit(value).hostname else ""
    except ValueError:
        return False, "url_parse_error"
    if actual_host != host and not actual_host.endswith("." + host):
        return False, "url_host_not_authoritative"
    return True, "ok"


def _receipt(value: Any, expected_binding: Mapping[str, Any]) -> tuple[bool, str]:
    """Validate a structured authenticated receipt; strings never pass."""

    if not isinstance(value, Mapping):
        return False, "receipt_not_object"
    allowed = {
        "receipt_id",
        "record_id",
        "identity",
        "digest",
        "sha256",
        "authenticated",
        "cryptographically_bound",
        "binding",
        "payload",
    }
    if any(key not in allowed for key in value):
        return False, "receipt_has_unrecognized_fields"
    ids = [key for key in ("receipt_id", "record_id", "identity") if key in value]
    digests = [key for key in ("digest", "sha256") if key in value]
    if len(ids) != 1 or not _strict_hex(value[ids[0]], 40):
        return False, "receipt_identity_not_lower_hex40"
    if len(digests) != 1 or not _strict_hex(value[digests[0]], 64):
        return False, "receipt_digest_not_lower_hex64"
    for key in ("authenticated", "cryptographically_bound"):
        if type(value.get(key)) is not bool:
            return False, f"receipt_{key}_not_boolean"
        if value[key] is not True:
            return False, f"receipt_{key}_false"
    if not isinstance(value.get("binding"), Mapping) or dict(value["binding"]) != dict(expected_binding):
        return False, "receipt_binding_mismatch"
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        return False, "receipt_payload_not_object"
    payload_ids = [payload[key] for key in ("identity", "record_id") if key in payload]
    payload_digests = [
        payload[key]
        for key in ("artifact_digest", "digest", "sha256", "content_sha256")
        if key in payload
    ]
    if not payload_ids or any(not _strict_hex(item, 40) for item in payload_ids):
        return False, "receipt_payload_identity_not_lower_hex40"
    if not payload_digests or any(not _strict_hex(item, 64) for item in payload_digests):
        return False, "receipt_payload_digest_not_lower_hex64"
    if len(set(payload_ids)) != 1 or len(set(payload_digests)) != 1:
        return False, "receipt_payload_identity_or_digest_conflict"
    for key, child in payload.items():
        key_text = str(key).lower()
        if key_text in {"status", "verified", "complete", "heldout", "result", "readiness"}:
            return False, "receipt_payload_self_attestation"
        if key_text.endswith("_id") and not _strict_hex(child, 40):
            return False, "receipt_payload_id_not_lower_hex40"
        if any(token in key_text for token in ("hash", "digest", "sha256")) and not _strict_hex(child, 64):
            return False, "receipt_payload_digest_not_lower_hex64"
        if key_text in {"url", "source_url", "repo_url", "run_url", "checkpoint_url"}:
            valid, reason = _url(child)
            if not valid:
                return False, reason
    try:
        expected_digest = receipt_digest(value["binding"], payload)
    except (TypeError, ValueError):
        return False, "receipt_payload_not_canonicalizable"
    if value[digests[0]] != expected_digest:
        return False, "receipt_digest_binding_mismatch"
    return True, "ok"


def _receipt_payload(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    payload = value.get("payload")
    return payload if isinstance(payload, Mapping) else None


def _validate_result_receipt_payload(
    value: Any,
    *,
    service: str,
    expected_result_digest: str,
) -> tuple[bool, str]:
    if not isinstance(value, Mapping):
        return False, "receipt_not_object"
    payload = _receipt_payload(value)
    if payload is None:
        return False, "receipt_payload_not_object"
    result_digest = payload.get("result_digest", payload.get("artifact_digest"))
    if result_digest != expected_result_digest:
        return False, "result_digest_not_bound"
    if service == "wandb":
        run_id = payload.get("run_id")
        if not _strict_hex(run_id, 40):
            return False, "wandb_run_id_not_lower_hex40"
        valid, reason = _authoritative_url(payload.get("run_url"), "wandb.ai")
        if not valid:
            return False, reason
    elif service == "tinker":
        if payload.get("provider") != "Tinker" or not _strict_hex(payload.get("run_id"), 40):
            return False, "tinker_run_identity_invalid"
        if not _finite(payload.get("cumulative_cost_usd")) or float(payload["cumulative_cost_usd"]) < 0 or float(payload["cumulative_cost_usd"]) > 16.50:
            return False, "tinker_cost_cap_invalid"
    elif service == "hf":
        revision = payload.get("revision")
        if not _strict_hex(revision, 40):
            return False, "hf_revision_not_lower_hex40"
        if type(payload.get("visibility")) is not str or payload.get("visibility") not in {"public", "private"}:
            return False, "hf_visibility_invalid"
        for key in ("repo_url", "checkpoint_url"):
            valid, reason = _authoritative_url(payload.get(key), "huggingface.co")
            if not valid:
                return False, reason
        if not isinstance(payload.get("checkpoint_url"), str):
            return False, "hf_checkpoint_url_not_bound_to_revision"
        try:
            checkpoint_path = urlsplit(payload["checkpoint_url"]).path
        except ValueError:
            return False, "hf_checkpoint_url_not_bound_to_revision"
        if revision not in checkpoint_path:
            return False, "hf_checkpoint_url_not_bound_to_revision"
        for key in ("contains_prompts", "contains_secrets", "contains_restricted_data"):
            if key in payload and type(payload[key]) is not bool:
                return False, f"hf_{key}_not_boolean"
        if payload.get("visibility") == "public":
            safety_keys = ("contains_prompts", "contains_secrets", "contains_restricted_data")
            if any(key not in payload for key in safety_keys):
                return False, "hf_public_safety_flags_missing"
            if any(payload[key] is True for key in safety_keys):
                return False, "hf_public_artifact_contains_restricted_data"
    return True, "ok"


def _validate_source(manifest: Mapping[str, Any], blockers: list[dict[str, Any]]) -> bool:
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        _add(blockers, "source_identity_missing", "WebBench requires an authoritative source identity object.")
        return False
    good = True
    expected = {
        "suite_id": WEBBENCH_SUITE_ID,
        "name": WEBBENCH_NAME,
        "url": WEBBENCH_SOURCE_URL,
    }
    for key, expected_value in expected.items():
        if source.get(key) != expected_value:
            _add(blockers, "source_identity_mismatch", "The manifest must identify the authoritative WebBench source exactly.", field=key, expected=expected_value, actual=source.get(key))
            good = False
    valid_url, reason = _url(source.get("url"), host="github.com")
    if not valid_url:
        _add(blockers, "source_url_invalid", "The authoritative source URL is not a valid official GitHub URL.", reason=reason)
        good = False
    revision = source.get("revision")
    if not _strict_hex(revision, 40):
        _add(blockers, "revision_not_pinned", "WebBench needs an immutable lower-case 40-hex revision.")
        good = False
    license_name = source.get("license")
    if type(license_name) is not str or not license_name.strip() or license_name.lower() in {"unknown", "pending", "tbd", "to_be_confirmed"}:
        _add(blockers, "license_not_pinned", "WebBench needs an explicit non-placeholder license identifier.")
        good = False
    for field, kind in (("revision_receipt", "revision"), ("license_receipt", "license")):
        valid, reason = _receipt(
            source.get(field),
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": kind},
        )
        if not valid:
            _add(blockers, "source_receipt_invalid", "Pinned source identity/license needs a bound receipt.", field=field, reason=reason)
            good = False
        elif isinstance(source.get(field), Mapping):
            payload = _receipt_payload(source[field])
            expected_value = revision if kind == "revision" else license_name
            payload_key = "revision" if kind == "revision" else "license"
            if payload is None or payload.get(payload_key) != expected_value:
                _add(blockers, "source_receipt_payload_mismatch", "The source receipt must bind the exact pinned revision or license.", field=field)
                good = False
    return good


def _validate_tasks_and_split(manifest: Mapping[str, Any], blockers: list[dict[str, Any]]) -> bool:
    task_ids = manifest.get("task_ids")
    good = True
    if not isinstance(task_ids, list) or not task_ids or any(type(item) is not str or not item.strip() for item in task_ids):
        _add(blockers, "task_ids_invalid", "WebBench task_ids must be a non-empty list of strings.")
        return False
    if len(task_ids) != len(set(task_ids)) or task_ids != sorted(task_ids):
        _add(blockers, "task_ids_not_deterministic", "Task IDs must be unique and lexicographically sorted.")
        good = False
    expected_task_hash = task_ids_hash(task_ids)
    if manifest.get("task_id_hash") != expected_task_hash:
        _add(blockers, "task_id_hash_mismatch", "task_id_hash must equal the canonical hash of deterministic task IDs.", expected=expected_task_hash, actual=manifest.get("task_id_hash"))
        good = False
    split_manifest = manifest.get("split_manifest")
    if not isinstance(split_manifest, Mapping):
        _add(blockers, "split_manifest_missing", "A deterministic split_manifest object is required.")
        return False
    if split_manifest.get("suite_id") != WEBBENCH_SUITE_ID or split_manifest.get("role") != WEBBENCH_ROLE or split_manifest.get("split") != WEBBENCH_SPLIT or split_manifest.get("task_id_hash") != expected_task_hash:
        _add(blockers, "split_manifest_boundary_mismatch", "The split manifest must bind WebBench, primary_eval, evaluation, and the task hash.")
        good = False
    try:
        expected_split_hash = split_manifest_hash(split_manifest)
    except (TypeError, ValueError):
        _add(blockers, "split_manifest_not_canonicalizable", "split_manifest must contain only canonical JSON values.")
        return False
    if manifest.get("split_manifest_hash") != expected_split_hash:
        _add(blockers, "split_manifest_hash_mismatch", "split_manifest_hash must equal the canonical split manifest hash.", expected=expected_split_hash, actual=manifest.get("split_manifest_hash"))
        good = False
    task_receipt = manifest.get("task_receipt")
    valid, reason = _receipt(
        task_receipt,
        {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "task"},
    )
    if not valid:
        _add(blockers, "task_receipt_invalid", "Deterministic task IDs need a bound task receipt.", reason=reason)
        good = False
    elif _receipt_payload(task_receipt).get("task_id_hash") != expected_task_hash:
        _add(blockers, "task_receipt_payload_mismatch", "The task receipt must bind the exact task_id_hash.")
        good = False
    split_receipt = manifest.get("split_receipt")
    valid, reason = _receipt(
        split_receipt,
        {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "split"},
    )
    if not valid:
        _add(blockers, "split_receipt_invalid", "The split manifest needs a bound split receipt.", reason=reason)
        good = False
    elif _receipt_payload(split_receipt).get("split_manifest_hash") != expected_split_hash:
        _add(blockers, "split_receipt_payload_mismatch", "The split receipt must bind the exact split_manifest_hash.")
        good = False
    return good


def _validate_environment(manifest: Mapping[str, Any], blockers: list[dict[str, Any]]) -> bool:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        _add(blockers, "environment_contract_missing", "WebBench needs a native environment/artifact/verifier contract.")
        return False
    good = True
    for field in ("container_digest", "runtime_digest"):
        if not _strict_hex(environment.get(field), 64):
            _add(blockers, "environment_digest_invalid", "Container and runtime identities must be lower-case 64-hex digests.", field=field)
            good = False
    native = environment.get("native_environment")
    if not isinstance(native, Mapping) or type(native.get("entrypoint")) is not str or not native.get("entrypoint") or type(native.get("state_model")) is not str or not native.get("state_model"):
        _add(blockers, "native_environment_contract_invalid", "A native entrypoint and state model are required.")
        good = False
    artifact = environment.get("artifact_contract")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("required_artifacts"), list) or not artifact.get("required_artifacts") or not isinstance(artifact.get("state_integrity_checks"), list) or not artifact.get("state_integrity_checks") or type(artifact.get("side_effect_policy")) is not str or not artifact.get("side_effect_policy"):
        _add(blockers, "artifact_contract_invalid", "Artifact/state integrity and side-effect policy must be structured, non-empty fields.")
        good = False
    verifier = environment.get("verifier_contract")
    if not isinstance(verifier, Mapping) or type(verifier.get("verifier_id")) is not str or not verifier.get("verifier_id") or not _strict_hex(verifier.get("verifier_revision"), 40) or not isinstance(verifier.get("checks"), list) or not verifier.get("checks") or type(verifier.get("native_state_inspection")) is not bool:
        _add(blockers, "verifier_contract_invalid", "The verifier must identify an immutable revision, checks, and native-state inspection behavior.")
        good = False
    container_receipt = environment.get("container_receipt")
    valid, reason = _receipt(
        container_receipt,
        {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "container"},
    )
    if not valid:
        _add(blockers, "container_receipt_invalid", "The runtime/container identity needs a bound receipt.", reason=reason)
        good = False
    elif (
        _receipt_payload(container_receipt).get("container_digest") != environment.get("container_digest")
        or _receipt_payload(container_receipt).get("runtime_digest") != environment.get("runtime_digest")
    ):
        _add(blockers, "container_receipt_payload_mismatch", "The container receipt must bind the exact container and runtime digests.")
        good = False
    verifier_receipt = environment.get("verifier_receipt")
    valid, reason = _receipt(
        verifier_receipt,
        {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "verifier"},
    )
    if not valid:
        _add(blockers, "verifier_receipt_invalid", "The native verifier needs a bound receipt.", reason=reason)
        good = False
    elif not isinstance(verifier, Mapping) or _receipt_payload(verifier_receipt).get("verifier_revision") != verifier.get("verifier_revision"):
        _add(blockers, "verifier_receipt_payload_mismatch", "The verifier receipt must bind the exact verifier revision.")
        good = False
    return good


def _validate_result_receipts(manifest: Mapping[str, Any], blockers: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Return (any_result_material, all_result_receipts_valid)."""

    material_keys = [key for key in ("result", "results", "metrics", "score", "scores") if key in manifest]
    receipts = manifest.get("result_receipts")
    if not material_keys and receipts is None:
        return False, False
    if not isinstance(receipts, Mapping):
        _add(blockers, "result_receipts_missing", "Any WebBench result material requires W&B, Tinker, and HF receipts.")
        return bool(material_keys), False
    good = True
    unexpected_receipts = [key for key in receipts if key not in RESULT_RECEIPT_FIELDS]
    if unexpected_receipts:
        _add(blockers, "result_receipt_fields_invalid", "Result receipts must use exactly the W&B, Tinker, and HF fields.", fields=sorted(unexpected_receipts, key=str))
        good = False
    for field in RESULT_RECEIPT_FIELDS:
        valid, reason = _receipt(
            receipts.get(field),
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": f"result:{field}"},
        )
        if not valid:
            _add(blockers, "result_receipt_invalid", "Every result bundle needs bound W&B/Tinker/HF receipts.", field=field, reason=reason)
            good = False
    result = manifest.get("result")
    if result is None:
        _add(blockers, "result_record_missing", "W&B/Tinker/HF result receipts cannot stand in for a missing result record.")
        return True, False
    if result is not None:
        if not isinstance(result, Mapping) or type(result.get("n")) is not int or result["n"] <= 0 or not _finite(result.get("metric")) or not 0.0 <= float(result["metric"]) <= 1.0:
            _add(blockers, "result_record_invalid", "A recorded WebBench result needs a positive n and finite metric in [0,1].")
            good = False
        if isinstance(result, Mapping) and result.get("result_id") is not None and not _strict_hex(result.get("result_id"), 40):
            _add(blockers, "result_identity_invalid", "Result identities must be lower-case 40-hex values.")
            good = False
        if isinstance(result, Mapping):
            try:
                expected_result_digest = sha256_hex(dict(result))
            except (TypeError, ValueError):
                _add(blockers, "result_record_not_canonicalizable", "The recorded result must contain only canonical JSON values.")
                return True, False
            for field in RESULT_RECEIPT_FIELDS:
                valid, reason = _validate_result_receipt_payload(
                    receipts.get(field),
                    service=field,
                    expected_result_digest=expected_result_digest,
                )
                if not valid:
                    _add(blockers, "result_receipt_payload_invalid", "Each service receipt must bind the exact recorded result.", field=field, reason=reason)
                    good = False
    return True, good


def _scan_substitution_text(value: Any) -> bool:
    if isinstance(value, str):
        text = value.lower()
        return bool(
            ("xlam" in text or "gsm8k" in text or "math-500" in text or "bfcl" in text or "browsergym" in text)
            and any(term in text for term in ("substitut", "proxy", "stand-in", "instead of", "related benchmark"))
        )
    if isinstance(value, Mapping):
        return any(_scan_substitution_text(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_scan_substitution_text(child) for child in value)
    return False


def validate_webbench_manifest(manifest: Any) -> dict[str, Any]:
    """Validate a WebBench boundary without running or fabricating evaluation."""

    blockers: list[dict[str, Any]] = []
    if not isinstance(manifest, Mapping):
        _add(blockers, "manifest_not_object", "The WebBench manifest must be a structured object.")
        return _report(False, False, blockers)
    if manifest.get("suite_id") != WEBBENCH_SUITE_ID:
        _add(blockers, "suite_id_invalid", "Only the declared webbench_eval suite is accepted.")
    if manifest.get("role") != WEBBENCH_ROLE:
        _add(blockers, "role_invalid", "WebBench is primary_eval; heldout is a receipt state, not a role.")
    domains = manifest.get("domains")
    if not isinstance(domains, list) or tuple(domains) != WEBBENCH_DOMAINS:
        _add(blockers, "domain_boundary_invalid", "WebBench must retain its browser, computer_use, and enterprise domain slice.")
    heldout_status = manifest.get("heldout_status")
    if type(heldout_status) is not str or heldout_status not in {"pending_receipts", "receipt_proven_heldout"}:
        _add(blockers, "heldout_status_invalid", "heldout_status must be pending_receipts or receipt_proven_heldout.")
    for key in ("substitutes_for", "proxy_for", "related_benchmark", "related_benchmarks", "xlam", "gsm8k"):
        if key in manifest and manifest.get(key):
            _add(blockers, "related_benchmark_substitution", "Related benchmarks and xLAM cannot substitute for WebBench.", field=key)
    if _scan_substitution_text(manifest):
        _add(blockers, "related_benchmark_substitution", "Related benchmark or xLAM substitution text is not an evaluation boundary.")
    source_ok = _validate_source(manifest, blockers)
    tasks_ok = _validate_tasks_and_split(manifest, blockers)
    environment_ok = _validate_environment(manifest, blockers)
    provenance = manifest.get("receipts")
    provenance_complete = True
    if not isinstance(provenance, Mapping):
        provenance_complete = False
    else:
        source = manifest.get("source")
        environment = manifest.get("environment")
        expected_provenance = {
            "revision": ("revision", source.get("revision") if isinstance(source, Mapping) else None),
            "license": ("license", source.get("license") if isinstance(source, Mapping) else None),
            "split": ("split_manifest_hash", manifest.get("split_manifest_hash")),
            "task": ("task_id_hash", manifest.get("task_id_hash")),
            "container": ("container_digest", environment.get("container_digest") if isinstance(environment, Mapping) else None),
            "decontamination": ("decontamination_hash", None),
        }
        for field in WEBBENCH_RECEIPT_FIELDS:
            valid, reason = _receipt(
                provenance.get(field),
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": field},
            )
            if not valid:
                provenance_complete = False
                _add(blockers, "provenance_receipt_invalid", "Primary_eval is not receipt-proven-heldout until every immutable field is bound.", field=field, reason=reason)
                continue
            payload = _receipt_payload(provenance[field])
            payload_key, expected_value = expected_provenance[field]
            if payload is None or payload_key not in payload or (expected_value is not None and payload.get(payload_key) != expected_value):
                provenance_complete = False
                _add(blockers, "provenance_receipt_payload_mismatch", "Each provenance receipt must bind the exact WebBench revision, license, task/split, runtime, or decontamination digest.", field=field)
    result_material, result_receipts_ok = _validate_result_receipts(manifest, blockers)
    receipts_complete = provenance_complete and result_receipts_ok
    if heldout_status == "receipt_proven_heldout" and not receipts_complete:
        _add(blockers, "heldout_status_without_receipts", "A receipt_proven_heldout status cannot be self-attested.")
    if heldout_status == "pending_receipts" and receipts_complete:
        _add(blockers, "heldout_status_stale", "Complete receipts require the explicit receipt_proven_heldout state.")
    boundary_valid = not blockers
    receipt_proven_heldout = boundary_valid and heldout_status == "receipt_proven_heldout" and receipts_complete
    if boundary_valid and not receipt_proven_heldout:
        status = "READY_PRIMARY_EVAL_PENDING_RECEIPTS"
    elif receipt_proven_heldout:
        status = "READY_RECEIPT_PROVEN_HELDOUT"
    else:
        status = "BLOCKED"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "boundary_valid": boundary_valid,
        "primary_eval": boundary_valid,
        "receipt_proven_heldout": receipt_proven_heldout,
        "heldout_claim_allowed": receipt_proven_heldout,
        "result_material_present": result_material,
        "result_receipts_complete": result_receipts_ok,
        "blocker_codes": [item["code"] for item in blockers],
        "blockers": blockers,
        "checks": {
            "source": source_ok,
            "tasks_and_split": tasks_ok,
            "environment": environment_ok,
            "provenance_receipts_complete": provenance_complete,
        },
        "zero_cost": True,
        "network_accessed": False,
        "paid_calls_executed": False,
    }


validate_manifest = validate_webbench_manifest


def _report(boundary_valid: bool, receipt_proven_heldout: bool, blockers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY_RECEIPT_PROVEN_HELDOUT" if receipt_proven_heldout else "BLOCKED",
        "boundary_valid": boundary_valid,
        "primary_eval": boundary_valid,
        "receipt_proven_heldout": receipt_proven_heldout,
        "heldout_claim_allowed": receipt_proven_heldout,
        "result_material_present": False,
        "result_receipts_complete": False,
        "blocker_codes": [item["code"] for item in blockers],
        "blockers": blockers,
        "checks": {
            "source": False,
            "tasks_and_split": False,
            "environment": False,
            "provenance_receipts_complete": False,
        },
        "zero_cost": True,
        "network_accessed": False,
        "paid_calls_executed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report = _report(False, False, [{"code": "manifest_input_error", "message": str(exc), "details": {}}])
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    report = validate_webbench_manifest(manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["boundary_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
