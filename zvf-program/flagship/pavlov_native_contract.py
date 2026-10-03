"""Shared offline helpers for the Pavlov native-contract metadata adapters.

Used by ``pavlov_agentdojo_train_adapter`` and
``pavlov_agentharm_frontiermath_adapter``.  Each adapter keeps its own
``_NATIVE_CONTRACT`` table and passes it to ``_native_contract_for_suite``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

_PLACEHOLDER_WORDS = {
    "",
    "none",
    "null",
    "nil",
    "na",
    "n/a",
    "undefined",
    "unknown",
    "todo",
    "tbd",
    "unset",
    "pending",
    "missing",
    "placeholder",
    "to_be_pinned_before_paid_runs",
    "to_be_pinned",
    "license-receipt",
}

_ZERO_40 = "0" * 40
_ZERO_64 = "0" * 64

_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON encoding used for hashing and signature checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Hash textual input with SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_WORDS
    return False


def _first_value(record: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _as_sequence(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return None


def _is_hex40(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().removeprefix("sha256:")
    if normalized in {_ZERO_40, _ZERO_64[:40]}:
        return False
    return bool(_HEX40_RE.fullmatch(value.strip()))


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower().removeprefix("sha256:")
    if normalized in {_ZERO_64, _ZERO_40}:
        return False
    return bool(_SHA256_RE.fullmatch(value.strip()))


def _is_immutable_revision(value: Any) -> bool:
    return _is_hex40(value) or _is_sha256(value)


def _expected_source_id(source_url: str) -> str:
    parsed = urlparse(source_url)
    parts = [segment for segment in parsed.path.split("/") if segment]
    if parsed.netloc == "github.com" and len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    if parsed.netloc and parts:
        return f"{parsed.netloc}/" + "/".join(parts)
    return parsed.netloc


def _load_contract_suite(contract: Mapping[str, Any], suite_id: str) -> Mapping[str, Any]:
    suites = contract.get("suite_registry", {})
    if not isinstance(suites, Mapping):
        raise ValueError("contract suite_registry must be an object")
    suite = suites.get(suite_id)
    if not isinstance(suite, Mapping):
        raise ValueError(f"contract is missing suite {suite_id}")
    return suite


def _native_contract_signature(spec: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(spec))


def _native_contract_for_suite(
    native_contract: Mapping[str, Mapping[str, Any]], suite_id: str
) -> Mapping[str, Any]:
    spec = native_contract[suite_id]
    return {
        "environment": {
            "name": spec["environment"]["name"],
            "mode": spec["environment"]["mode"],
            "artifact_required": bool(spec["environment"]["artifact_required"]),
            "contract_sha256": _native_contract_signature(spec["environment"]),
        },
        "verifier": {
            "name": spec["verifier"]["name"],
            "mode": spec["verifier"]["mode"],
            "verifier_sha256": _native_contract_signature(spec["verifier"]),
        },
        "artifact": {
            "mode": spec["artifact"]["mode"],
            "artifact_sha256": _native_contract_signature(spec["artifact"]),
        },
    }


def _expected_bundle_signature(bundle: Mapping[str, Any]) -> str:
    payload = {key: bundle[key] for key in bundle if key != "bundle_signature"}
    return sha256_text(canonical_json(payload))


def update_bundle_signature(bundle: dict[str, Any]) -> str:
    """Attach a deterministic bundle signature after local mutation."""

    signature = _expected_bundle_signature(bundle)
    bundle["bundle_signature"] = signature
    return signature
