#!/usr/bin/env python3
"""Zero-cost, fail-closed validation for the Pavlov professor protocol bundle.

The validator reads only the local protocol, contract, and budget files.  It never
contacts Tinker, Hugging Face, W&B, or any other network service.  Its JSON output
is intentionally useful while the bundle is blocked: blockers are machine-readable
and the checks retain the observed counts, unions, and missing receipt fields.

An accepted immutable receipt is a structured record, never a URL/status string:
it carries a lower-case 40-hex identity, lower-case 64-hex digest, exact boolean
authentication flags, a suite/field binding, and a payload whose canonical digest
matches the record.  These checks are local integrity gates, not external service
attestations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit


HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL_PATH = HERE / "PAVLOV_EXPERIMENT_PROTOCOL_2026-08-09.md"
DEFAULT_CONTRACT_PATH = HERE / "pavlovs_domain_contract.json"
DEFAULT_BUDGET_PATH = HERE / "pavlov_tinker_budget.json"

SCHEMA_VERSION = "pavlov-protocol-validation-v2"
EXPECTED_TRAIN_SUITES = 12
EXPECTED_PRIMARY_EVAL_SUITES = 14
EXPECTED_DOMAINS = 16
EXPECTED_DOMAIN_SET = frozenset(
    {
        "alignment",
        "browser",
        "chip_design",
        "code",
        "computer_use",
        "design",
        "enterprise",
        "finance",
        "games",
        "long_horizon",
        "math",
        "ml",
        "multi_domain",
        "science",
        "security",
        "tool_use",
    }
)
EXPECTED_BUDGET = {
    "maximum_usd": Decimal("18.00"),
    "operational_cap_usd": Decimal("16.50"),
    "safety_reserve_usd": Decimal("1.50"),
}
# A paid launch is intentionally narrower than a generic "ready" or
# "approved" label.  Those labels are human status text, not authorization.
AUTHORIZED_CONTRACT_STATUS = "authorized"
# Backwards-compatible export for callers that displayed the old report
# vocabulary; validation still accepts only this one exact status string.
AUTHORIZED_CONTRACT_STATUSES = frozenset({AUTHORIZED_CONTRACT_STATUS})
AUTHORIZED_BUDGET_STATUS = "AUTHORIZED_TINKER_ONLY"
REQUIRED_RECEIPT_FIELDS = (
    "revision",
    "license",
    "split",
    "task",
    "container",
    "decontamination",
)

HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_ID_KEYS = ("receipt_id", "record_id", "identity", "id")
RECEIPT_DIGEST_KEYS = ("digest", "sha256", "content_sha256", "hash")
RECEIPT_PAYLOAD_KEYS = ("payload", "content", "evidence", "value")
PAYLOAD_IDENTITY_KEYS = (
    "identity",
    "revision",
    "model_revision",
    "record_id",
    "receipt_id",
    "id",
    "run_id",
    "checkpoint_id",
    "task_id",
)
PAYLOAD_DIGEST_KEYS = (
    "digest",
    "sha256",
    "content_sha256",
    "artifact_digest",
    "hash",
    "license_digest",
    "split_manifest_hash",
    "task_id_hashes",
    "container_digest",
    "decontamination_hash",
)
URL_KEYS = (
    "url",
    "source_url",
    "receipt_url",
    "run_url",
    "repo_url",
    "checkpoint_url",
    "revision_url",
)
PAID_ONLY_BLOCKER_CODES = {
    "contract_status_missing",
    "contract_status_type_invalid",
    "contract_status_not_authorized",
    "paid_jobs_may_launch_type_invalid",
    "paid_jobs_not_authorized",
    "budget_gate_status_invalid",
    "contract_budget_status_invalid",
    "budget_provider_invalid",
    "budget_file_provider_invalid",
}

# Direct fields are deliberately strict: the contract's human-readable ``split``
# description is not a split-manifest receipt.  Short aliases are accepted only
# inside an explicit receipt container, which keeps the current contract blocked.
DIRECT_RECEIPT_ALIASES = {
    "revision": ("immutable_revision", "dataset_revision", "revision_receipt", "revision"),
    "license": ("license_receipt", "license"),
    "split": ("split_manifest_hash", "split_manifest_receipt"),
    "task": ("task_id_hashes", "task_id_hash", "task_receipt", "task"),
    "container": ("container_digest", "container_receipt", "container"),
    "decontamination": (
        "decontamination_receipt",
        "decontamination_hash",
        "decontamination",
    ),
}
NESTED_RECEIPT_ALIASES = {
    "revision": DIRECT_RECEIPT_ALIASES["revision"] + ("revision",),
    "license": DIRECT_RECEIPT_ALIASES["license"] + ("license",),
    "split": DIRECT_RECEIPT_ALIASES["split"] + ("split",),
    "task": DIRECT_RECEIPT_ALIASES["task"] + ("task", "tasks"),
    "container": DIRECT_RECEIPT_ALIASES["container"] + ("container",),
    "decontamination": DIRECT_RECEIPT_ALIASES["decontamination"]
    + ("decontamination",),
}
RECEIPT_CONTAINER_KEYS = (
    "immutable_receipts",
    "receipts",
    "receipt",
)
TOP_LEVEL_RECEIPT_MAP_KEYS = (
    "primary_eval_receipts",
    "primary_evaluation_receipts",
    "evaluation_receipts",
    "suite_receipts",
)
PLACEHOLDER_RE = re.compile(
    r"(?:^|[-_ ])(?:to[_ -]?be[_ -]?pinned|todo|tbd|pending|missing|unknown|unset|none|n/?a)(?:$|[-_ ])",
    re.IGNORECASE,
)

# These patterns identify a present-tense assertion, not the protocol's allowed
# conditional wording such as "designated held out only after receipts".
HOLDOUT_CLAIM_PATTERNS = (
    re.compile(
        r"\b(?:all\s+)?14\s+primary[_-]?eval(?:uation)?\s+"
        r"(?:suites?|tasks?|manifests?)\s+"
        r"(?:are|is)\s+(?:already\s+)?held[- ]out\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:all\s+)?14(?:[- ]suite)?\s+"
        r"(?:(?:primary[-_ ]?)?evaluation\s+)?"
        r"(?:suites?|tasks?|manifests?)\s+"
        r"(?:are|is)\s+(?:already\s+)?held[- ]out\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:all\s+)?14\s+held[- ]out\s+"
        r"(?:(?:primary[-_ ]?)?evaluation\s+)?"
        r"(?:suites?|tasks?|manifests?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b14[- ]suite\s+(?:(?:primary[-_ ]?)?evaluation\s+)?holdout\b"
        r"(?!\s+(?:receipt|receipts|manifest|manifests|status))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b14[- ]suite\s+(?:(?:primary[-_ ]?)?evaluation\s+)?"
        r"held[- ]out\b",
        re.IGNORECASE,
    ),
)
CONDITIONAL_HOLDOUT_RE = re.compile(
    r"\b(?:pending|only\s+after|until|not\s+(?:called|yet|already)|unless|"
    r"requires?|when|once|if|before|blocked|designated)\b",
    re.IGNORECASE,
)


def _add_blocker(
    blockers: list[dict[str, Any]],
    code: str,
    message: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    blockers.append(
        {
            "code": code,
            "message": message,
            "details": dict(details or {}),
        }
    )


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _decimal(value: Any) -> Decimal | None:
    # JSON budget amounts must be numbers, not strings that merely look like
    # numbers.  Decimal(str(...)) makes the equality and arithmetic checks exact
    # for the finite JSON int/float values we accept.
    if type(value) not in (int, float, Decimal):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return result if result.is_finite() else None


def _strict_hex(value: Any, length: int) -> bool:
    """Return true only for a lower-case hexadecimal identity or digest."""

    if type(value) is not str:
        return False
    pattern = HEX40_RE if length == 40 else HEX64_RE if length == 64 else None
    return bool(pattern and pattern.fullmatch(value))


def _canonical_json(value: Any) -> str:
    """Canonical JSON used for the local receipt binding check."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_receipt_digest(
    suite_id: str, field: str, payload: Mapping[str, Any]
) -> str:
    """Compute the digest required for one suite/field receipt record.

    The digest covers both the suite/field binding and the immutable payload.  It
    is a local integrity check; it does not contact or attest to an external
    service.  A caller still needs an authenticated receipt record and an
    immutable 40-hex receipt identity for the validator to accept it.
    """

    bound = {
        "binding": {"suite_id": suite_id, "field": field},
        "payload": payload,
    }
    return hashlib.sha256(_canonical_json(bound).encode("utf-8")).hexdigest()


def _validate_url(
    value: Any,
    *,
    key: str,
    suite: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate URL syntax and bind suite-source URLs to the registered source.

    No network request is made.  A URL is never evidence by itself; it is only
    checked for strict syntax/host consistency while the receipt digest binds it
    into the authenticated payload.
    """

    if type(value) is not str or not value.strip():
        return False, "url_not_string"
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False, "url_parse_error"
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False, "url_not_https_or_hosted"
    hostname = parsed.hostname.lower()
    if hostname in {"example.com", "example.org", "example.net"} or hostname.endswith(
        ".invalid"
    ):
        return False, "url_placeholder_host"
    registered = suite.get("url")
    if key in {"url", "source_url", "receipt_url"}:
        if type(registered) is not str:
            return False, "registered_suite_url_missing"
        try:
            expected = urlsplit(registered)
        except ValueError:
            return False, "registered_suite_url_parse_error"
        if expected.scheme != "https" or not expected.hostname:
            return False, "registered_suite_url_invalid"
        if hostname != expected.hostname.lower():
            return False, "url_host_not_bound_to_suite"
    elif key == "run_url" and not (
        hostname == "wandb.ai"
        or hostname.endswith(".wandb.ai")
    ):
        return False, "run_url_host_untrusted"
    elif key in {"repo_url", "checkpoint_url", "revision_url"} and not (
        hostname == "huggingface.co"
        or hostname.endswith(".huggingface.co")
    ):
        return False, "hf_url_host_untrusted"
    return True, "ok"


def _validate_receipt_payload(
    payload: Any,
    *,
    suite: Mapping[str, Any],
) -> tuple[bool, str]:
    if not isinstance(payload, Mapping):
        return False, "payload_not_object"

    identities = [payload[key] for key in PAYLOAD_IDENTITY_KEYS if key in payload]
    if not identities:
        return False, "payload_identity_missing"
    if any(not _strict_hex(value, 40) for value in identities):
        return False, "payload_identity_not_lower_hex40"
    if len(set(identities)) != 1:
        return False, "payload_identity_conflict"

    digests = [payload[key] for key in PAYLOAD_DIGEST_KEYS if key in payload]
    if not digests:
        return False, "payload_digest_missing"
    if any(not _strict_hex(value, 64) for value in digests):
        return False, "payload_digest_not_lower_hex64"
    if len(set(digests)) != 1:
        return False, "payload_digest_conflict"
    for status_key in ("status", "state", "result"):
        if status_key in payload:
            return False, f"payload_{status_key}_not_evidence"

    # Fields that advertise hashes/digests are never allowed to carry a status
    # string, URL, prefix (e.g. ``sha256:``), or upper-case hexadecimal value.
    for key, value in payload.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ("hash", "digest", "sha256")):
            if not _strict_hex(value, 64):
                return False, f"payload_{key}_not_lower_hex64"
        if lowered.endswith("_id") or lowered in {"run_id", "checkpoint_id"}:
            if not _strict_hex(value, 40):
                return False, f"payload_{key}_not_lower_hex40"
        if key in URL_KEYS:
            valid, reason = _validate_url(value, key=key, suite=suite)
            if not valid:
                return False, reason
    if "visibility" in payload and (
        type(payload["visibility"]) is not str
        or payload["visibility"] not in {"public", "private"}
    ):
        return False, "payload_visibility_invalid"
    for key in ("public_artifact_safe", "contains_secrets", "contains_restricted_data"):
        if key in payload and type(payload[key]) is not bool:
            return False, f"payload_{key}_not_boolean"
    if payload.get("public_artifact_safe") is False:
        return False, "payload_public_artifact_not_safe"
    if payload.get("contains_secrets") is True or payload.get("contains_restricted_data") is True:
        return False, "payload_restricted_public_artifact"
    return True, "ok"


def _validate_receipt_record(
    value: Any,
    *,
    suite_id: str,
    field: str,
    suite: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate one authenticated, cryptographically bound receipt record.

    A raw URL, run ID, digest string, status string, or truthy value is not a
    receipt.  The record must have typed authentication flags, a lower-hex
    identity, a lower-hex digest, a suite/field binding, and a payload whose
    canonical digest matches the record.
    """

    if not isinstance(value, Mapping):
        return False, "receipt_record_not_object"
    allowed_record_keys = {
        *RECEIPT_ID_KEYS,
        *RECEIPT_DIGEST_KEYS,
        "authenticated",
        "cryptographically_bound",
        "immutable",
        "verified",
        "binding",
        "bound_to",
        *RECEIPT_PAYLOAD_KEYS,
    }
    unknown_record_keys = sorted(
        (repr(key) for key in value if key not in allowed_record_keys),
        key=str,
    )
    if unknown_record_keys:
        return False, "receipt_unrecognized_field"

    identity_values = [value[key] for key in RECEIPT_ID_KEYS if key in value]
    if len(identity_values) != 1:
        return False, "receipt_identity_missing_or_ambiguous"
    if not _strict_hex(identity_values[0], 40):
        return False, "receipt_identity_not_lower_hex40"

    digest_values = [value[key] for key in RECEIPT_DIGEST_KEYS if key in value]
    if len(digest_values) != 1:
        return False, "receipt_digest_missing_or_ambiguous"
    if not _strict_hex(digest_values[0], 64):
        return False, "receipt_digest_not_lower_hex64"

    for key in ("authenticated", "cryptographically_bound"):
        if type(value.get(key)) is not bool:
            return False, f"receipt_{key}_not_boolean"
        if value[key] is not True:
            return False, f"receipt_{key}_false"
    for key in ("immutable", "verified"):
        if key in value and type(value[key]) is not bool:
            return False, f"receipt_{key}_not_boolean"
        if key in value and value[key] is not True:
            return False, f"receipt_{key}_false"
    for key in URL_KEYS:
        if key in value:
            valid_url, url_reason = _validate_url(value[key], key=key, suite=suite)
            if not valid_url:
                return False, url_reason
    for key in ("run_id", "checkpoint_id"):
        if key in value and not _strict_hex(value[key], 40):
            return False, f"receipt_{key}_not_lower_hex40"

    binding_values = [value[key] for key in ("binding", "bound_to") if key in value]
    if len(binding_values) != 1:
        return False, "receipt_binding_missing_or_ambiguous"
    binding = binding_values[0]
    if not isinstance(binding, Mapping):
        return False, "receipt_binding_not_object"
    if type(binding.get("suite_id")) is not str or binding.get("suite_id") != suite_id:
        return False, "receipt_suite_binding_mismatch"
    if type(binding.get("field")) is not str or binding.get("field") != field:
        return False, "receipt_field_binding_mismatch"

    payload_values = [value[key] for key in RECEIPT_PAYLOAD_KEYS if key in value]
    if len(payload_values) != 1:
        return False, "receipt_payload_missing_or_ambiguous"
    payload = payload_values[0]
    valid_payload, payload_reason = _validate_receipt_payload(payload, suite=suite)
    if not valid_payload:
        return False, payload_reason
    try:
        expected_digest = compute_receipt_digest(suite_id, field, payload)
    except (TypeError, ValueError):
        return False, "receipt_payload_not_canonicalizable"
    if digest_values[0] != expected_digest:
        return False, "receipt_digest_binding_mismatch"
    return True, "ok"


def _receipt_sources(
    contract: Mapping[str, Any], suite_id: str, suite: Mapping[str, Any]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Return direct and nested receipt mappings for one primary suite."""

    direct: list[Mapping[str, Any]] = [suite]
    nested: list[Mapping[str, Any]] = []
    for key in RECEIPT_CONTAINER_KEYS:
        value = _mapping(suite.get(key))
        if value is not None:
            nested.append(value)

    for key in TOP_LEVEL_RECEIPT_MAP_KEYS:
        receipt_map = _mapping(contract.get(key))
        if receipt_map is None:
            continue
        suite_receipts = _mapping(receipt_map.get(suite_id))
        if suite_receipts is not None:
            nested.append(suite_receipts)
            direct.append(suite_receipts)
    return direct, nested


def _find_receipt(
    direct: Sequence[Mapping[str, Any]],
    nested: Sequence[Mapping[str, Any]],
    field: str,
) -> tuple[Any, str | None]:
    for source in direct:
        for alias in DIRECT_RECEIPT_ALIASES[field]:
            if alias in source:
                return source[alias], alias
    for source in nested:
        for alias in NESTED_RECEIPT_ALIASES[field]:
            if alias in source:
                return source[alias], alias
    return None, None


def _check_primary_receipts(
    contract: Mapping[str, Any],
    suites: Mapping[str, Any],
    primary_ids: Sequence[str],
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    complete_suite_ids: list[str] = []
    for suite_id in primary_ids:
        suite = _mapping(suites.get(suite_id)) or {}
        direct, nested = _receipt_sources(contract, suite_id, suite)
        suite_missing: list[dict[str, str]] = []
        for field in REQUIRED_RECEIPT_FIELDS:
            value, alias = _find_receipt(direct, nested, field)
            valid, reason = _validate_receipt_record(
                value,
                suite_id=suite_id,
                field=field,
                suite=suite,
            )
            if not valid:
                suite_missing.append(
                    {
                        "field": field,
                        "reason": reason,
                        "alias": alias or "<missing>",
                    }
                )
        if suite_missing:
            missing.append({"suite_id": suite_id, "missing": suite_missing})
        else:
            complete_suite_ids.append(suite_id)

    if missing:
        _add_blocker(
            blockers,
            "primary_suite_receipts_incomplete",
            "Every primary evaluation suite needs immutable revision, license, split, task, container, and decontamination receipts.",
            {"missing": missing},
        )
    return {
        "required_fields": list(REQUIRED_RECEIPT_FIELDS),
        "primary_suite_count": len(primary_ids),
        "complete_suite_ids": sorted(complete_suite_ids),
        "missing": missing,
        "complete": not missing and len(complete_suite_ids) == len(primary_ids),
    }


def _find_unqualified_holdout_claims(protocol_text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    lines = protocol_text.splitlines()
    for index, raw_line in enumerate(lines):
        line = raw_line.replace("*", " ")
        if not any(pattern.search(line) for pattern in HOLDOUT_CLAIM_PATTERNS):
            continue
        context = " ".join(
            item.replace("*", " ")
            for item in lines[max(0, index - 1) : min(len(lines), index + 2)]
        )
        if CONDITIONAL_HOLDOUT_RE.search(context):
            continue
        claims.append({"line": index + 1, "text": raw_line.strip()})
    return claims


def _check_contract_structure(
    contract: Mapping[str, Any], blockers: list[dict[str, Any]]
) -> dict[str, Any]:
    suites_value = contract.get("suite_registry")
    suites = _mapping(suites_value) or {}
    if not isinstance(suites_value, Mapping):
        _add_blocker(
            blockers,
            "suite_registry_invalid",
            "suite_registry must be a JSON object keyed by suite ID.",
        )
    domains_value = contract.get("domains")
    domains = list(domains_value) if isinstance(domains_value, list) else []
    domain_types_valid = all(type(domain) is str and bool(domain.strip()) for domain in domains)
    domain_set = set(domains) if domain_types_valid else set()
    if (
        len(domains) != EXPECTED_DOMAINS
        or len(domain_set) != len(domains)
        or not domain_types_valid
        or domain_set != EXPECTED_DOMAIN_SET
    ):
        _add_blocker(
            blockers,
            "domain_declaration_invalid",
            "The contract must declare exactly 16 unique domains.",
            {
                "expected": EXPECTED_DOMAINS,
                "actual": len(domains),
                "unique": len(domain_set),
                "all_lowercase_strings": domain_types_valid,
                "missing": sorted(EXPECTED_DOMAIN_SET - domain_set),
                "extra": sorted(domain_set - EXPECTED_DOMAIN_SET),
            },
        )

    role_ids: dict[str, list[str]] = {"train": [], "primary_eval": []}
    for suite_id, suite_value in suites.items():
        suite = _mapping(suite_value)
        if suite is None:
            _add_blocker(
                blockers,
                "suite_record_invalid",
                "Every suite registry entry must be an object.",
                {"suite_id": str(suite_id)},
            )
            continue
        role = suite.get("role")
        if type(suite_id) is not str or not suite_id.strip():
            _add_blocker(
                blockers,
                "suite_id_invalid",
                "Every suite ID must be a non-empty string.",
                {"suite_id": repr(suite_id)},
            )
        if role in role_ids:
            role_ids[role].append(str(suite_id))

    expected_counts = {"train": EXPECTED_TRAIN_SUITES, "primary_eval": EXPECTED_PRIMARY_EVAL_SUITES}
    for role, expected in expected_counts.items():
        actual = len(role_ids[role])
        if actual != expected:
            _add_blocker(
                blockers,
                "suite_count_mismatch",
                f"The contract must contain exactly {expected} {role} suites.",
                {"role": role, "expected": expected, "actual": actual},
            )

    unions: dict[str, set[str]] = {"train": set(), "primary_eval": set()}
    malformed_suites: list[str] = []
    for role, ids in role_ids.items():
        for suite_id in ids:
            suite = _mapping(suites.get(suite_id)) or {}
            tags = suite.get("domains")
            if not isinstance(tags, list) or any(
                type(tag) is not str or not tag.strip() for tag in tags
            ):
                malformed_suites.append(suite_id)
                continue
            unions[role].update(tags)
            for boolean_field in ("stateful", "artifact_or_side_effect"):
                if type(suite.get(boolean_field)) is not bool:
                    _add_blocker(
                        blockers,
                        "suite_boolean_type_invalid",
                        "Train and primary_eval suite control fields must be exact booleans.",
                        {"suite_id": suite_id, "field": boolean_field, "value": suite.get(boolean_field)},
                    )
    if malformed_suites:
        _add_blocker(
            blockers,
            "suite_domain_tags_invalid",
            "Every train and primary_eval suite must provide a domain-tag list.",
            {"suite_ids": sorted(malformed_suites)},
        )

    union_checks: dict[str, Any] = {}
    for role in ("train", "primary_eval"):
        missing = sorted(domain_set - unions[role])
        extra = sorted(unions[role] - domain_set)
        union_checks[role] = {
            "count": len(role_ids[role]),
            "domain_union": sorted(unions[role]),
            "missing_domains": missing,
            "extra_domains": extra,
            "matches_declared_domains": not missing and not extra and len(domain_set) == EXPECTED_DOMAINS,
        }
        if missing or extra or len(domain_set) != EXPECTED_DOMAINS:
            _add_blocker(
                blockers,
                "domain_union_mismatch",
                f"The {role} suite union must equal the 16 declared domains.",
                {"role": role, "missing": missing, "extra": extra},
            )

    companies_value = contract.get("companies")
    companies = companies_value if isinstance(companies_value, list) else []
    if not isinstance(companies_value, list):
        _add_blocker(
            blockers,
            "companies_invalid",
            "companies must be a JSON list for per-company coverage accounting.",
        )
    elif not companies:
        _add_blocker(
            blockers,
            "companies_invalid",
            "companies must contain at least one company with required domains.",
        )
    missing_train: list[dict[str, Any]] = []
    missing_eval: list[dict[str, Any]] = []
    unknown_company_domains: list[dict[str, Any]] = []
    seen_company_names: set[str] = set()
    duplicate_company_names: list[str] = []
    for company_value in companies:
        company = _mapping(company_value)
        if company is None:
            _add_blocker(
                blockers,
                "company_record_invalid",
                "Every company entry must be an object.",
            )
            company = {}
        name_value = company.get("name")
        name = name_value if type(name_value) is str and name_value.strip() else "<unnamed>"
        if name == "<unnamed>":
            _add_blocker(
                blockers,
                "company_name_invalid",
                "Every company must have a non-empty string name.",
                {"value": name_value},
            )
        if name in seen_company_names:
            duplicate_company_names.append(name)
        seen_company_names.add(name)
        if "required_domains" in company:
            company_domains = company.get("required_domains")
            if "domains" in company and company.get("domains") != company_domains:
                _add_blocker(
                    blockers,
                    "company_required_domains_conflict",
                    "required_domains and legacy domains must agree exactly when both are present.",
                    {"company": name},
                )
        else:
            # The current contract uses ``domains``; the validator treats it as
            # the company's required-domain list and reports it as such.
            company_domains = company.get("domains")
        if (
            not isinstance(company_domains, list)
            or not company_domains
            or any(
                type(domain) is not str or not domain.strip()
                for domain in company_domains
            )
        ):
            _add_blocker(
                blockers,
                "company_domains_invalid",
                "Every company must provide its required domains as a list.",
                {"company": name},
            )
            company_domains = []
        required = set(company_domains)
        unknown = sorted(required - domain_set)
        if unknown:
            unknown_company_domains.append({"company": name, "required_domains": unknown})
        train_missing = sorted(required - unions["train"])
        eval_missing = sorted(required - unions["primary_eval"])
        if train_missing:
            missing_train.append({"company": name, "required_domains": train_missing})
        if eval_missing:
            missing_eval.append({"company": name, "required_domains": eval_missing})

    if duplicate_company_names:
        _add_blocker(
            blockers,
            "duplicate_company_names",
            "Company entries must be uniquely named for coverage accounting.",
            {"names": sorted(set(duplicate_company_names))},
        )
    if unknown_company_domains:
        _add_blocker(
            blockers,
            "company_domain_unknown",
            "Company domains must come from the declared 16-domain set.",
            {"companies": unknown_company_domains},
        )
    if missing_train:
        _add_blocker(
            blockers,
            "company_domain_missing_train",
            "Every company-required domain must be covered by the train-suite union.",
            {"companies": missing_train},
        )
    if missing_eval:
        _add_blocker(
            blockers,
            "company_domain_missing_primary_eval",
            "Every company-required domain must be covered independently by the primary_eval union.",
            {"companies": missing_eval},
        )

    gsm_candidates: list[dict[str, Any]] = []
    for suite_id, suite_value in suites.items():
        suite = _mapping(suite_value) or {}
        haystack = f"{suite_id} {suite.get('url', '')}".lower()
        if "gsm8k" in haystack:
            gsm_candidates.append({"suite_id": str(suite_id), "role": suite.get("role")})
    if not gsm_candidates:
        _add_blocker(
            blockers,
            "gsm8k_calibration_missing",
            "The bundle must declare a GSM8K calibration_only suite.",
        )
    for candidate in gsm_candidates:
        if candidate["role"] in {"train", "primary_eval"}:
            _add_blocker(
                blockers,
                "gsm8k_primary_role_forbidden",
                "GSM8K may not be a train or primary_eval suite.",
                candidate,
            )
    explicit_gsm = _mapping(suites.get("gsm8k_calibration"))
    if explicit_gsm is None or explicit_gsm.get("role") != "calibration_only":
        _add_blocker(
            blockers,
            "gsm8k_calibration_role_invalid",
            "suite_registry.gsm8k_calibration must have role calibration_only.",
        )

    model_candidates_value = contract.get("model_candidates")
    model_candidates = model_candidates_value if isinstance(model_candidates_value, list) else []
    if not isinstance(model_candidates_value, list) or not model_candidates:
        _add_blocker(
            blockers,
            "model_candidates_invalid",
            "At least one model candidate with a pinned immutable revision is required.",
        )
    model_revision_checks: list[dict[str, Any]] = []
    for index, model_value in enumerate(model_candidates):
        model = _mapping(model_value)
        if model is None:
            _add_blocker(
                blockers,
                "model_candidate_invalid",
                "Every model candidate must be an object.",
                {"index": index},
            )
            model_revision_checks.append({"index": index, "revision": None, "valid": False})
            continue
        revision = model.get("revision")
        valid_revision = _strict_hex(revision, 40)
        model_revision_checks.append(
            {"index": index, "model_id": model.get("model_id"), "revision": revision, "valid": valid_revision}
        )
        if not valid_revision:
            _add_blocker(
                blockers,
                "model_revision_invalid",
                "Every model candidate revision must be an immutable lower-case 40-hex identity.",
                {"index": index, "model_id": model.get("model_id"), "revision": revision},
            )

    return {
        "declared_domain_count": len(domains),
        "declared_domains": sorted(domain_set),
        "suite_counts": {role: len(ids) for role, ids in role_ids.items()},
        "suite_ids": {role: sorted(ids) for role, ids in role_ids.items()},
        "domain_unions": union_checks,
        "company_count": len(companies),
        "company_domain_coverage": {
            "missing_train": missing_train,
            "missing_primary_eval": missing_eval,
            "unknown_domains": unknown_company_domains,
            "complete": bool(companies) and not missing_train and not missing_eval and not unknown_company_domains,
        },
        "gsm8k": {
            "candidates": gsm_candidates,
            "role": explicit_gsm.get("role") if explicit_gsm else None,
            "calibration_only": bool(explicit_gsm and explicit_gsm.get("role") == "calibration_only"),
        },
        "model_revisions": model_revision_checks,
    }


def _check_budget_and_status(
    contract: Mapping[str, Any], budget: Mapping[str, Any], blockers: list[dict[str, Any]]
) -> dict[str, Any]:
    gate_value = contract.get("budget_gate")
    gate = _mapping(gate_value) or {}
    if not isinstance(gate_value, Mapping):
        _add_blocker(
            blockers,
            "budget_gate_invalid",
            "contract.budget_gate must be an object.",
        )

    raw_contract_status = contract.get("status")
    if raw_contract_status is None:
        _add_blocker(
            blockers,
            "contract_status_missing",
            "Contract status is required before any paid launch.",
        )
    elif type(raw_contract_status) is not str:
        _add_blocker(
            blockers,
            "contract_status_type_invalid",
            "Contract status must be an exact string; booleans and status-like values are not authorization.",
            {"status": raw_contract_status},
        )
    elif raw_contract_status != AUTHORIZED_CONTRACT_STATUS:
        _add_blocker(
            blockers,
            "contract_status_not_authorized",
            "Paid launch is forbidden while contract status is draft, pending, contradictory, or otherwise unapproved.",
            {"status": raw_contract_status, "authorized_status": AUTHORIZED_CONTRACT_STATUS},
        )
    status_authorized = raw_contract_status == AUTHORIZED_CONTRACT_STATUS

    paid_flag = gate.get("paid_jobs_may_launch")
    if type(paid_flag) is not bool:
        _add_blocker(
            blockers,
            "paid_jobs_may_launch_type_invalid",
            "paid_jobs_may_launch must be an exact boolean.",
            {"paid_jobs_may_launch": paid_flag},
        )
    elif paid_flag is not True:
        _add_blocker(
            blockers,
            "paid_jobs_not_authorized",
            "The contract budget gate must explicitly authorize Tinker before launch.",
            {"paid_jobs_may_launch": paid_flag},
        )
    if gate.get("provider") != "Tinker":
        _add_blocker(
            blockers,
            "budget_provider_invalid",
            "The authorized paid provider must be Tinker only.",
            {"provider": gate.get("provider")},
        )
    if budget.get("provider") != "Tinker":
        _add_blocker(
            blockers,
            "budget_file_provider_invalid",
            "The local budget file must authorize Tinker only.",
            {"provider": budget.get("provider")},
        )
    gate_status = gate.get("status")
    if gate_status != AUTHORIZED_BUDGET_STATUS:
        _add_blocker(
            blockers,
            "contract_budget_status_invalid",
            "The contract budget gate must be AUTHORIZED_TINKER_ONLY.",
            {"status": gate_status, "expected": AUTHORIZED_BUDGET_STATUS},
        )

    sources = {"contract": gate, "budget": budget}
    parsed: dict[str, dict[str, str | None]] = {}
    for source_name, source in sources.items():
        parsed[source_name] = {}
        for field in EXPECTED_BUDGET:
            value = _decimal(source.get(field))
            parsed[source_name][field] = str(value) if value is not None else None

    mismatches: list[dict[str, Any]] = []
    for source_name, source in sources.items():
        for field, expected in EXPECTED_BUDGET.items():
            actual = _decimal(source.get(field))
            if actual != expected:
                mismatches.append(
                    {
                        "source": source_name,
                        "field": field,
                        "expected": str(expected),
                        "actual": str(actual) if actual is not None else None,
                    }
                )
    if mismatches:
        _add_blocker(
            blockers,
            "budget_values_mismatch",
            "The hard, operational, and reserve budget values must be exactly $18.00, $16.50, and $1.50.",
            {"mismatches": mismatches},
        )

    arithmetic: list[dict[str, Any]] = []
    for source_name, source in sources.items():
        maximum = _decimal(source.get("maximum_usd"))
        operational = _decimal(source.get("operational_cap_usd"))
        reserve = _decimal(source.get("safety_reserve_usd"))
        valid = (
            maximum is not None
            and operational is not None
            and reserve is not None
            and maximum == operational + reserve
            and operational <= maximum
            and reserve >= Decimal("0")
        )
        arithmetic.append(
            {
                "source": source_name,
                "maximum_usd": str(maximum) if maximum is not None else None,
                "operational_cap_usd": str(operational) if operational is not None else None,
                "safety_reserve_usd": str(reserve) if reserve is not None else None,
                "valid": valid,
            }
        )
        if not valid:
            _add_blocker(
                blockers,
                "budget_arithmetic_invalid",
                "Maximum budget must equal operational cap plus safety reserve, with cap ordering valid.",
                arithmetic[-1],
            )

    contract_budget_mismatch = []
    for field in EXPECTED_BUDGET:
        contract_value = _decimal(gate.get(field))
        budget_value = _decimal(budget.get(field))
        if contract_value != budget_value:
            contract_budget_mismatch.append(
                {
                    "field": field,
                    "contract": str(contract_value) if contract_value is not None else None,
                    "budget": str(budget_value) if budget_value is not None else None,
                }
            )
    if contract_budget_mismatch:
        _add_blocker(
            blockers,
            "contract_budget_mismatch",
            "Contract budget_gate and the local budget file must agree exactly.",
            {"mismatches": contract_budget_mismatch},
        )

    return {
        "contract_status": raw_contract_status,
        "status_authorized": status_authorized,
        "contract_budget_gate_status": gate_status,
        "paid_jobs_may_launch": paid_flag,
        "sources": parsed,
        "arithmetic": arithmetic,
        "contract_budget_match": not contract_budget_mismatch,
    }


def validate_bundle(
    protocol_text: str,
    contract: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an already-loaded local bundle and return a JSON-safe report."""

    blockers: list[dict[str, Any]] = []
    if not isinstance(contract, Mapping):
        _add_blocker(blockers, "contract_invalid", "Contract JSON must be an object.")
        contract = {}
    if not isinstance(budget, Mapping):
        _add_blocker(blockers, "budget_invalid", "Budget JSON must be an object.")
        budget = {}
    if not isinstance(protocol_text, str):
        _add_blocker(blockers, "protocol_invalid", "Protocol must be UTF-8 text.")
        protocol_text = str(protocol_text)

    structure = _check_contract_structure(contract, blockers)
    budget_check = _check_budget_and_status(contract, budget, blockers)
    suites = _mapping(contract.get("suite_registry")) or {}
    primary_ids = structure["suite_ids"]["primary_eval"]
    receipt_check = _check_primary_receipts(contract, suites, primary_ids, blockers)
    protocol_claims = _find_unqualified_holdout_claims(protocol_text)
    if protocol_claims and not receipt_check["complete"]:
        _add_blocker(
            blockers,
            "protocol_holdout_claim_without_receipts",
            "A protocol may not claim all 14 suites are held out until every primary suite has immutable receipts.",
            {"claims": protocol_claims},
        )

    blocker_codes = [item["code"] for item in blockers]
    protocol_blockers = [
        item for item in blockers if item["code"] not in PAID_ONLY_BLOCKER_CODES
    ]
    protocol_ready = not protocol_blockers
    paid_launch_allowed = protocol_ready and not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        # ``protocol_ready`` describes the zero-cost structural/provenance
        # contract.  ``paid_launch_allowed`` additionally requires explicit,
        # internally consistent Tinker authorization.  They are intentionally
        # separate so a draft/contradictory contract cannot masquerade as a
        # ready paid launch.
        "status": "READY" if paid_launch_allowed else "BLOCKED",
        "protocol_ready": protocol_ready,
        "launch_allowed": paid_launch_allowed,
        "paid_launch_allowed": paid_launch_allowed,
        "zero_cost": True,
        "network_accessed": False,
        "paid_calls_executed": False,
        "protocol_blocker_codes": [item["code"] for item in protocol_blockers],
        "blocker_codes": blocker_codes,
        "blockers": blockers,
        "checks": {
            "contract_structure": structure,
            "budget_and_status": budget_check,
            "primary_receipts": receipt_check,
            "protocol_holdout_claims": {
                "unqualified_claims": protocol_claims,
                "receipts_complete": receipt_check["complete"],
                "claim_rule_passes": not protocol_claims or receipt_check["complete"],
            },
        },
    }


# Explicit alias for callers that prefer the protocol-specific name.
validate_protocol_bundle = validate_bundle


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_paths(
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    budget_path: Path = DEFAULT_BUDGET_PATH,
) -> dict[str, Any]:
    """Read local files only; input failures are ordinary fail-closed blockers."""

    try:
        protocol_text = protocol_path.read_text(encoding="utf-8")
        contract = _load_json(contract_path)
        budget = _load_json(budget_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "protocol_ready": False,
            "launch_allowed": False,
            "paid_launch_allowed": False,
            "zero_cost": True,
            "network_accessed": False,
            "paid_calls_executed": False,
            "blocker_codes": ["bundle_input_error"],
            "blockers": [
                {
                    "code": "bundle_input_error",
                    "message": "Could not read the local protocol/contract/budget bundle.",
                    "details": {"error": str(exc)},
                }
            ],
            "checks": {},
        }
    report = validate_bundle(protocol_text, contract, budget)
    report["inputs"] = {
        "protocol": str(protocol_path),
        "contract": str(contract_path),
        "budget": str(budget_path),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET_PATH)
    args = parser.parse_args(argv)
    report = validate_paths(args.protocol, args.contract, args.budget)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["launch_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
