#!/usr/bin/env python3
"""Offline WebBench evaluation-boundary adapter for Pavlov's primary suite.

The adapter defines the exact WebBench boundary and validates manifests before a
runner is allowed to use them.  It performs no WebBench download, browser work,
Tinker call, W&B call, Hugging Face call, or result fabrication.  A structurally
valid ``primary_eval`` manifest may remain pending receipts; it is never called
receipt-proven-heldout until all immutable provenance and result receipts pass.

The module has two halves:

* ``validate_webbench_manifest`` — the fail-closed boundary check.  It never
  accepts a manifest whose provenance, environment, or result receipts are
  unbound.
* ``build_split_artifacts`` — a purely local derivation over the pinned public
  ``webbenchfinal.csv``.  It derives stable task identities, per-task digests,
  aggregate hashes, the evaluation split manifest, and a train/eval disjointness
  proof.  Nothing it emits is an authenticated receipt; the derivation records
  say so explicitly so a local artifact can never be mistaken for a receipt the
  boundary check requires.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
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

# ---------------------------------------------------------------------------
# Pinned public-dataset facts.  These describe the MIT-licensed task CSV only.
# They say nothing about the Halluminate live environment or native verifier,
# neither of which is public.
# ---------------------------------------------------------------------------
SPLIT_SCHEMA_VERSION = "pavlov-webbench-split-derivation-v1"
WEBBENCH_REVISION = "ea7a1628443321363989f354401f0653e0cba6f4"
WEBBENCH_DATASET_FILE = "webbenchfinal.csv"
WEBBENCH_DATASET_SHA256 = "fd5311a38bdb6f941e8f544150735656c114d76fbfb17193da973d5de0165217"
WEBBENCH_LICENSE_SHA256 = "96804aa272fe40cdfb8b5c8f4d1d94757bcfaf1bf5596fb829214843d2371e58"
WEBBENCH_LICENSE = "MIT"
WEBBENCH_PUBLIC_TASK_COUNT = 2647
WEBBENCH_CSV_COLUMNS = ("ID", "Starting URL", "Category", "Task")
WEBBENCH_CATEGORIES = ("CREATE", "DELETE", "FILE_MANIPULATION", "READ", "UPDATE")
WEBBENCH_TASK_ID_PREFIX = "webbench-task-"
# The public CSV carries sparse integer IDs over 0..2724, so a fixed-width
# zero-padded suffix is required for lexicographic order to equal numeric order.
WEBBENCH_TASK_ID_WIDTH = 4
WEBBENCH_TASK_UID_PREFIX = "webbench-uid-"
WEBBENCH_TASK_UID_LENGTH = 16
# Fields the native Halluminate verifier would have to consume; the public CSV
# supplies only the first four.  Recorded so the access request is concrete.
WEBBENCH_PUBLIC_TASK_FIELDS = ("csv_id", "starting_url", "category", "task")
WEBBENCH_VERIFIER_REQUIRED_FIELDS = (
    "success_criteria",
    "expected_final_state",
    "answer_key_or_rubric",
    "allowed_side_effects",
    "credential_scope",
    "reset_procedure",
)
# Categories whose tasks mutate live third-party sites.  Used only to size the
# authorization ask; the split itself never depends on it.
WEBBENCH_WRITE_CATEGORIES = ("CREATE", "DELETE", "FILE_MANIPULATION", "UPDATE")
# Keyword probes.  These are heuristics over task prose, reported as such; they
# are not a classification and nothing downstream branches on them.
WEBBENCH_KEYWORD_PROBES = {
    "credential_or_account": r"\blog ?in\b|\bsign ?in\b|\baccount\b|\bcredential",
    "payment_or_checkout": r"\bcheckout\b|\bpayment\b|\bcredit card\b|\bpurchase\b|\bbuy\b",
    "captcha": r"captcha",
}


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


# ---------------------------------------------------------------------------
# Local split derivation over the pinned public CSV.
#
# Hash definitions (all lower-case hex SHA-256 over canonical JSON unless the
# name says otherwise):
#
#   task_digest        sha256({category, csv_id, starting_url, task})
#   task_uid           "webbench-uid-" + task_digest[:16]  (order-independent)
#   task_id            "webbench-task-" + zero-padded csv_id (order-defining)
#   task_id_hash       sha256([task_id, ...]) in sorted task_id order
#   task_digest_hash   sha256([task_digest, ...]) in sorted task_id order
#   task_index_hash    sha256([full task record, ...]) in sorted task_id order
#   legacy_*           the webbench_eval.py runner's own scheme, recomputed here
#                      so the two independent implementations can be compared
# ---------------------------------------------------------------------------


class WebBenchDatasetError(ValueError):
    """Raised when the pinned public CSV does not match its declared identity."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def registrable_host(host: str) -> str:
    """Collapse a leading ``www.`` label; everything else is left untouched."""

    lowered = (host or "").strip().lower()
    return lowered[4:] if lowered.startswith("www.") else lowered


def format_task_id(csv_id: int) -> str:
    if not isinstance(csv_id, int) or isinstance(csv_id, bool) or csv_id < 0:
        raise WebBenchDatasetError(f"task ID must be a non-negative int, got {csv_id!r}")
    return f"{WEBBENCH_TASK_ID_PREFIX}{csv_id:0{WEBBENCH_TASK_ID_WIDTH}d}"


def task_digest(record: Mapping[str, Any]) -> str:
    """Content address of one task; independent of row order and of task_id."""

    return sha256_hex(
        {
            "category": record["category"],
            "csv_id": record["csv_id"],
            "starting_url": record["starting_url"],
            "task": record["task"],
        }
    )


def read_webbench_csv(path: Path, *, expected_sha256: str | None = WEBBENCH_DATASET_SHA256) -> list[dict[str, Any]]:
    """Read and structurally validate the pinned public CSV. No network."""

    path = Path(path)
    if not path.is_file():
        raise WebBenchDatasetError(f"WebBench dataset file is missing: {path}")
    actual = file_sha256(path)
    if expected_sha256 is not None and actual.lower() != expected_sha256.lower():
        raise WebBenchDatasetError(
            f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    rows: list[dict[str, Any]] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            if columns != WEBBENCH_CSV_COLUMNS:
                raise WebBenchDatasetError(
                    f"dataset columns must be {WEBBENCH_CSV_COLUMNS!r}, got {columns!r}"
                )
            for line, row in enumerate(reader, start=2):
                try:
                    csv_id = int(str(row["ID"]).strip())
                except (TypeError, ValueError) as exc:
                    raise WebBenchDatasetError(f"row {line} has a non-integer ID") from exc
                if csv_id < 0:
                    raise WebBenchDatasetError(f"row {line} has a negative ID")
                starting_url = str(row["Starting URL"] or "").strip()
                category = str(row["Category"] or "").strip().upper()
                task_text = str(row["Task"] or "").strip()
                parsed = urlsplit(starting_url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                    raise WebBenchDatasetError(f"row {line} has an invalid Starting URL")
                if category not in WEBBENCH_CATEGORIES:
                    raise WebBenchDatasetError(f"row {line} has unsupported category {category!r}")
                if not task_text:
                    raise WebBenchDatasetError(f"row {line} has an empty task")
                rows.append(
                    {
                        "csv_id": csv_id,
                        "starting_url": starting_url,
                        "category": category,
                        "task": task_text,
                        "_scheme": parsed.scheme,
                        "_host": parsed.hostname.lower(),
                    }
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise WebBenchDatasetError(f"could not read WebBench CSV {path}: {exc}") from exc
    if not rows:
        raise WebBenchDatasetError("WebBench CSV has no tasks")
    ids = [row["csv_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise WebBenchDatasetError("WebBench CSV contains duplicate task IDs")
    return rows


def derive_task_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attach stable identities and per-task digests, sorted by task_id."""

    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            "task_id": format_task_id(int(row["csv_id"])),
            "csv_id": int(row["csv_id"]),
            "category": row["category"],
            "starting_url": row["starting_url"],
            "scheme": row.get("_scheme") or urlsplit(row["starting_url"]).scheme,
            "host": row.get("_host") or (urlsplit(row["starting_url"]).hostname or "").lower(),
            "task": row["task"],
            "task_char_len": len(row["task"]),
        }
        record["registrable_domain"] = registrable_host(record["host"])
        record["task_digest"] = task_digest(record)
        record["task_uid"] = WEBBENCH_TASK_UID_PREFIX + record["task_digest"][:WEBBENCH_TASK_UID_LENGTH]
        records.append(record)
    records.sort(key=lambda item: item["task_id"])
    task_ids = [item["task_id"] for item in records]
    if len(task_ids) != len(set(task_ids)):
        raise WebBenchDatasetError("derived task IDs are not unique")
    uids = [item["task_uid"] for item in records]
    if len(uids) != len(set(uids)):
        raise WebBenchDatasetError("derived task UIDs collide; widen WEBBENCH_TASK_UID_LENGTH")
    return records


def aggregate_task_hashes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate hashes over the derived records, plus the runner's own scheme."""

    ordered = sorted(records, key=lambda item: item["task_id"])
    task_ids = [item["task_id"] for item in ordered]
    digests = [item["task_digest"] for item in ordered]
    index_rows = [
        {
            "task_id": item["task_id"],
            "task_uid": item["task_uid"],
            "csv_id": item["csv_id"],
            "category": item["category"],
            "starting_url": item["starting_url"],
            "task_digest": item["task_digest"],
        }
        for item in ordered
    ]
    legacy_rows = [
        {
            "id": item["csv_id"],
            "starting_url": item["starting_url"],
            "category": item["category"],
            "task": item["task"],
        }
        for item in sorted(ordered, key=lambda entry: entry["csv_id"])
    ]
    return {
        "task_count": len(ordered),
        "task_id_hash": task_ids_hash(task_ids),
        "task_digest_hash": sha256_hex(digests),
        "task_index_hash": sha256_hex(index_rows),
        "task_uid_hash": sha256_hex([item["task_uid"] for item in ordered]),
        "legacy_runner_task_id_hash": _sha256_text(
            "\n".join(str(item["csv_id"]) for item in sorted(ordered, key=lambda e: e["csv_id"]))
        ),
        "legacy_runner_manifest_hash": sha256_hex(legacy_rows),
    }


def build_split_manifest(
    task_ids: Sequence[str],
    *,
    split: str = WEBBENCH_SPLIT,
    role: str = WEBBENCH_ROLE,
) -> dict[str, Any]:
    """The exact split_manifest object ``_validate_tasks_and_split`` expects."""

    ordered = sorted(task_ids)
    if len(ordered) != len(set(ordered)):
        raise WebBenchDatasetError("split manifest task IDs must be unique")
    return {
        "suite_id": WEBBENCH_SUITE_ID,
        "role": role,
        "split": split,
        "task_id_hash": task_ids_hash(ordered),
    }


def load_training_task_manifest(path: Path) -> dict[str, Any]:
    """Load the eval-only training manifest; an explicit empty list is valid."""

    path = Path(path)
    if not path.is_file():
        raise WebBenchDatasetError(f"training task-ID manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WebBenchDatasetError(f"training task-ID manifest is malformed: {path}: {exc}") from exc
    if isinstance(value, list):
        raw_ids, declared = value, None
    elif isinstance(value, Mapping):
        raw_ids, declared = value.get("task_ids"), value.get("task_id_hash")
    else:
        raise WebBenchDatasetError("training task-ID manifest must be a list or object")
    if not isinstance(raw_ids, list):
        raise WebBenchDatasetError("training task-ID manifest must contain a task_ids list")
    normalized: list[str] = []
    for item in raw_ids:
        if isinstance(item, bool):
            raise WebBenchDatasetError("training task-ID manifest contains a boolean ID")
        if isinstance(item, int):
            normalized.append(format_task_id(item))
        elif isinstance(item, str) and item.strip():
            normalized.append(item.strip())
        else:
            raise WebBenchDatasetError(f"training task-ID manifest has an unusable entry: {item!r}")
    if len(normalized) != len(set(normalized)):
        raise WebBenchDatasetError("training task-ID manifest contains duplicate IDs")
    # The webbench_eval.py runner hashes newline-joined integer IDs.  Recompute
    # that form only when every entry really is an integer, so a declared hash
    # from the runner can be verified rather than assumed.
    all_int = all(isinstance(item, int) and not isinstance(item, bool) for item in raw_ids)
    legacy_hash = _sha256_text("\n".join(str(item) for item in sorted(raw_ids))) if all_int else None
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "task_ids": sorted(normalized),
        "declared_task_id_hash": declared,
        "legacy_runner_task_id_hash": legacy_hash,
    }


def prove_split_disjointness(
    eval_task_ids: Sequence[str],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    """Set-level train/eval disjointness proof over normalized string IDs."""

    eval_set = set(eval_task_ids)
    train_set = set(training["task_ids"])
    overlap = sorted(eval_set & train_set)
    errors: list[str] = []
    if len(eval_set) != len(eval_task_ids):
        errors.append("evaluation task IDs are not unique")
    if overlap:
        errors.append(f"train/eval task-ID overlap: {len(overlap)} IDs")
    declared = training.get("declared_task_id_hash")
    legacy = training.get("legacy_runner_task_id_hash")
    declared_hash_verified: bool | None = None
    if declared is not None:
        if legacy is None:
            errors.append("training manifest declares a task_id_hash that cannot be recomputed")
        else:
            declared_hash_verified = str(declared).strip().lower().removeprefix("sha256:") == legacy
            if not declared_hash_verified:
                errors.append("training manifest declared task_id_hash does not recompute")
    return {
        "evaluation_task_count": len(eval_set),
        "training_task_count": len(train_set),
        "overlap_count": len(overlap),
        "overlap_task_ids": overlap,
        "disjoint": not overlap,
        "training_manifest_path": training["path"],
        "training_manifest_sha256": training["sha256"],
        "training_declared_task_id_hash": declared,
        "training_declared_task_id_hash_verified": declared_hash_verified,
        "training_task_id_hash": task_ids_hash(sorted(train_set)),
        "evaluation_task_id_hash": task_ids_hash(sorted(eval_set)),
        "errors": errors,
        "valid": not errors,
    }


def characterize_task_set(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Factual profile of the public task set, derived only from the CSV."""

    categories = Counter(item["category"] for item in records)
    hosts = Counter(item["host"] for item in records)
    domains = Counter(item["registrable_domain"] for item in records)
    schemes = Counter(item["scheme"] for item in records)
    per_domain = sorted(domains.values())
    lengths = sorted(item["task_char_len"] for item in records)
    csv_ids = sorted(item["csv_id"] for item in records)
    id_span = list(range(csv_ids[0], csv_ids[-1] + 1))
    missing_ids = sorted(set(id_span) - set(csv_ids))
    category_domains = {
        category: len({item["registrable_domain"] for item in records if item["category"] == category})
        for category in sorted(categories)
    }
    write_class = sum(value for key, value in categories.items() if key in WEBBENCH_WRITE_CATEGORIES)
    keyword_counts = {
        name: sum(1 for item in records if re.search(pattern, item["task"], re.IGNORECASE))
        for name, pattern in sorted(WEBBENCH_KEYWORD_PROBES.items())
    }
    return {
        "task_count": len(records),
        "csv_id_min": csv_ids[0],
        "csv_id_max": csv_ids[-1],
        "csv_id_span": len(id_span),
        "csv_id_gap_count": len(missing_ids),
        "csv_id_gaps": missing_ids,
        "csv_ids_contiguous": not missing_ids,
        "categories": dict(sorted(categories.items())),
        "category_share": {
            key: round(value / len(records), 4) for key, value in sorted(categories.items())
        },
        "distinct_hosts": len(hosts),
        "distinct_registrable_domains": len(domains),
        "url_schemes": dict(sorted(schemes.items())),
        "http_only_urls": sorted({item["starting_url"] for item in records if item["scheme"] == "http"}),
        "tasks_per_domain_min": per_domain[0],
        "tasks_per_domain_median": statistics.median(per_domain),
        "tasks_per_domain_max": per_domain[-1],
        "top_domains": [
            {"domain": key, "task_count": value} for key, value in domains.most_common(25)
        ],
        "distinct_domains_per_category": category_domains,
        "task_char_len_min": lengths[0],
        "task_char_len_median": statistics.median(lengths),
        "task_char_len_max": lengths[-1],
        "distinct_task_texts": len({item["task"] for item in records}),
        "write_class_categories": list(WEBBENCH_WRITE_CATEGORIES),
        "write_class_task_count": write_class,
        "write_class_share": round(write_class / len(records), 4),
        "keyword_probe_counts": keyword_counts,
        "keyword_probe_note": "regex probes over task prose; heuristic, not a classification",
        "public_task_fields": list(WEBBENCH_PUBLIC_TASK_FIELDS),
        "verifier_fields_absent_from_public_csv": list(WEBBENCH_VERIFIER_REQUIRED_FIELDS),
    }


def build_split_artifacts(
    dataset_path: Path,
    *,
    training_manifest_path: Path,
    license_path: Path | None = None,
    expected_dataset_sha256: str | None = WEBBENCH_DATASET_SHA256,
) -> dict[str, Any]:
    """Derive every offline split artifact. Local only; never a receipt."""

    rows = read_webbench_csv(dataset_path, expected_sha256=expected_dataset_sha256)
    records = derive_task_records(rows)
    aggregates = aggregate_task_hashes(records)
    task_ids = [item["task_id"] for item in records]
    split_manifest = build_split_manifest(task_ids)
    training = load_training_task_manifest(training_manifest_path)
    disjointness = prove_split_disjointness(task_ids, training)
    source: dict[str, Any] = {
        "suite_id": WEBBENCH_SUITE_ID,
        "name": WEBBENCH_NAME,
        "url": WEBBENCH_SOURCE_URL,
        "revision": WEBBENCH_REVISION,
        "license": WEBBENCH_LICENSE,
        "dataset_file": WEBBENCH_DATASET_FILE,
        # Resolved so the same file re-derives identically from any cwd.
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": file_sha256(Path(dataset_path)),
    }
    if license_path is not None and Path(license_path).is_file():
        source["license_path"] = str(Path(license_path).resolve())
        source["license_sha256"] = file_sha256(Path(license_path))
        source["license_sha256_matches_pin"] = source["license_sha256"] == WEBBENCH_LICENSE_SHA256
    derivation = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "suite_id": WEBBENCH_SUITE_ID,
        "role": WEBBENCH_ROLE,
        "split": WEBBENCH_SPLIT,
        "source": source,
        "task_id_scheme": {
            "task_id": f"{WEBBENCH_TASK_ID_PREFIX}<csv_id zero-padded to {WEBBENCH_TASK_ID_WIDTH}>",
            "task_uid": f"{WEBBENCH_TASK_UID_PREFIX}<task_digest[:{WEBBENCH_TASK_UID_LENGTH}]>",
            "task_digest": "sha256(canonical_json({category, csv_id, starting_url, task}))",
            "ordering": "lexicographic task_id, which equals numeric csv_id order",
        },
        "aggregates": aggregates,
        "split_manifest": split_manifest,
        "split_manifest_hash": split_manifest_hash(split_manifest),
        "disjointness": disjointness,
        "characterization": characterize_task_set(records),
        # Explicitly not a receipt.  The boundary check rejects local
        # derivations, and it must keep rejecting them.
        "receipt_class": "local_derivation",
        "authenticated_receipt": None,
        "externally_bound": False,
        "network_accessed": False,
        "paid_calls_executed": False,
    }
    derivation["derivation_hash"] = sha256_hex(
        {
            "aggregates": aggregates,
            "split_manifest_hash": derivation["split_manifest_hash"],
            "source_revision": WEBBENCH_REVISION,
            "dataset_sha256": source["dataset_sha256"],
        }
    )
    return {"records": records, "derivation": derivation}


def write_split_artifacts(bundle: Mapping[str, Any], out_dir: Path) -> dict[str, str]:
    """Write the derivation bundle to disk; returns {artifact: path}."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = bundle["records"]
    derivation = bundle["derivation"]
    index_path = out_dir / "webbench_task_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                _canonical_json(
                    {
                        "task_id": record["task_id"],
                        "task_uid": record["task_uid"],
                        "csv_id": record["csv_id"],
                        "category": record["category"],
                        "starting_url": record["starting_url"],
                        "registrable_domain": record["registrable_domain"],
                        "task_digest": record["task_digest"],
                        "task_char_len": record["task_char_len"],
                    }
                )
                + "\n"
            )
    split_path = out_dir / "webbench_eval_split_manifest.json"
    split_payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "suite_id": WEBBENCH_SUITE_ID,
        "role": WEBBENCH_ROLE,
        "split": WEBBENCH_SPLIT,
        "source": derivation["source"],
        "task_count": derivation["aggregates"]["task_count"],
        "task_id_hash": derivation["aggregates"]["task_id_hash"],
        "task_digest_hash": derivation["aggregates"]["task_digest_hash"],
        "task_index_hash": derivation["aggregates"]["task_index_hash"],
        "split_manifest": derivation["split_manifest"],
        "split_manifest_hash": derivation["split_manifest_hash"],
        "task_ids": [record["task_id"] for record in records],
        "receipt_class": "local_derivation",
        "authenticated_receipt": None,
    }
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    derivation_path = out_dir / "webbench_split_derivation.json"
    derivation_path.write_text(json.dumps(derivation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    disjoint_path = out_dir / "webbench_disjointness_proof.json"
    disjoint_path.write_text(
        json.dumps(
            {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "suite_id": WEBBENCH_SUITE_ID,
                "evaluation_split_manifest_hash": derivation["split_manifest_hash"],
                **derivation["disjointness"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    characterization_path = out_dir / "webbench_task_characterization.json"
    characterization_path.write_text(
        json.dumps(
            {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "suite_id": WEBBENCH_SUITE_ID,
                "source": derivation["source"],
                **derivation["characterization"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    index_sha = file_sha256(index_path)
    (out_dir / "webbench_task_index.jsonl.sha256").write_text(
        f"{index_sha}  {index_path.name}\n", encoding="utf-8"
    )
    return {
        "task_index": str(index_path),
        "task_index_sha256": index_sha,
        "split_manifest": str(split_path),
        "derivation": str(derivation_path),
        "disjointness_proof": str(disjoint_path),
        "characterization": str(characterization_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="boundary manifest to validate")
    parser.add_argument(
        "--build-split",
        action="store_true",
        help="derive the offline task index, split manifest, and disjointness proof",
    )
    parser.add_argument("--dataset", type=Path, help="pinned webbenchfinal.csv")
    parser.add_argument("--training-task-manifest", type=Path, help="eval-only training task-ID manifest")
    parser.add_argument("--license", type=Path, help="pinned WEBBENCH_LICENSE file")
    parser.add_argument("--out-dir", type=Path, help="directory for derived split artifacts")
    parser.add_argument(
        "--allow-unpinned-dataset",
        action="store_true",
        help="skip the dataset SHA-256 pin (for fixtures and tests only)",
    )
    args = parser.parse_args(argv)
    if args.build_split:
        if args.dataset is None or args.training_task_manifest is None:
            parser.error("--build-split requires --dataset and --training-task-manifest")
        try:
            bundle = build_split_artifacts(
                args.dataset,
                training_manifest_path=args.training_task_manifest,
                license_path=args.license,
                expected_dataset_sha256=None if args.allow_unpinned_dataset else WEBBENCH_DATASET_SHA256,
            )
        except WebBenchDatasetError as exc:
            print(json.dumps({"status": "BLOCKED", "error": str(exc)}, indent=2, sort_keys=True))
            return 1
        written = write_split_artifacts(bundle, args.out_dir) if args.out_dir else {}
        print(
            json.dumps(
                {"derivation": bundle["derivation"], "written": written},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if bundle["derivation"]["disjointness"]["valid"] else 1
    if args.manifest is None:
        parser.error("--manifest is required unless --build-split is given")
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
