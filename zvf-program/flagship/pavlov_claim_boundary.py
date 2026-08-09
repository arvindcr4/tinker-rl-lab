#!/usr/bin/env python3
"""Offline, fail-closed claim-boundary classification for the Pavlov campaign.

This module classifies *evidence records*, not prose assertions.  A claim can be
bounded to a prospective protocol, one admissible component observation, exact
12/14 portfolio coverage, or receipt-proven primary held-out results.  Strings,
status labels, and booleans that merely attest to completion never substitute for
receipts.  The classifier does not contact Tinker, W&B, Hugging Face, or any
other service.
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


SCHEMA_VERSION = "pavlov-claim-boundary-v1"

CLAIM_KINDS = frozenset(
    {"protocol_preflight", "component_evidence", "portfolio_evidence", "heldout_result"}
)

TRAIN_SUITE_IDS = (
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
PRIMARY_EVAL_SUITE_IDS = (
    "agentharm_eval",
    "apex_agents_eval",
    "appbench_eval",
    "banker_toolbench_eval",
    "binaryaudit_eval",
    "frontiermath_eval",
    "frontier_swe_eval",
    "lifescibench_eval",
    "mle_bench_eval",
    "openreward_games_eval",
    "swe_bench_pro_eval",
    "verilog_eval",
    "webbench_eval",
    "sdab_eval",
)
DOMAIN_IDS = (
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
)

EXPECTED_TRAIN = frozenset(TRAIN_SUITE_IDS)
EXPECTED_PRIMARY = frozenset(PRIMARY_EVAL_SUITE_IDS)
EXPECTED_DOMAINS = frozenset(DOMAIN_IDS)
REQUIRED_HOLDOUT_RECEIPTS = (
    "revision",
    "license",
    "split",
    "task",
    "container",
    "decontamination",
)

HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
URL_KEYS = ("url", "source_url", "repo_url", "run_url", "checkpoint_url")

CONTROL_BOOLEAN_KEYS = frozenset(
    {
        "verified",
        "complete",
        "admissible",
        "heldout",
        "held_out",
        "portfolio_wide",
        "company_ready",
        "production_ready",
        "promoted",
        "improved",
        "final_holdout_untouched",
        "company_readiness",
        "portfolio_ready",
    }
)
CONTROL_STRING_KEYS = frozenset(
    {
        "status",
        "evidence_status",
        "result_status",
        "readiness",
        "claim_status",
        "company_readiness",
        "company_status",
        "production_status",
    }
)

_TEXT_PROMOTION_PATTERNS = (
    (
        "company_readiness_claim",
        re.compile(
            r"\b(?:all\s+53|every\s+company|company(?:-wide)?\s+readiness|"
            r"company[_ -]?(?:wide[_ -]?)?readiness|"
            r"production\s+readiness|production[- ]ready|ready\s+for\s+deployment|"
            r"portfolio\s+(?:is\s+)?(?:ready|complete))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "xlam_promotion_forbidden",
        re.compile(
            r"\bxlam\b.*\b(?:portfolio|all\s+16|all\s+53|company|production|"
            r"general|universal|cross[- ]domain|improv(?:e|ement|ed))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "xlam_holdout_promotion_forbidden",
        re.compile(r"\bxlam\b.*\bheld[- ]?out\b", re.IGNORECASE),
    ),
    (
        "gsm8k_promotion_forbidden",
        re.compile(
            r"\bgsm8k\b.*\b(?:primary|portfolio|training|train|held[- ]?out|"
            r"company|production|general|useful|improv(?:e|ement|ed)|promot(?:e|ed))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "related_benchmark_substitution",
        re.compile(
            r"\b(?:substitut(?:e|ed|ion)|proxy|stand[- ]?in|instead\s+of|"
            r"representative\s+of)\b.*\b(?:benchmark|suite|gsm8k|math[- ]?500|"
            r"xlam|bfcl|swe[- ]?bench)\b",
            re.IGNORECASE,
        ),
    ),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def compute_receipt_digest(binding: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest required by a receipt record."""

    body = {"binding": dict(binding), "payload": dict(payload)}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def compute_result_digest(result: Mapping[str, Any]) -> str:
    """Hash the recorded result fields, excluding its nested receipt.

    The result receipt must carry this digest so a valid receipt cannot be
    replayed alongside a different metric or interval.
    """

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")
    body = {key: value for key, value in result.items() if key != "receipt"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _strict_hex(value: Any, length: int) -> bool:
    if type(value) is not str:
        return False
    pattern = HEX40_RE if length == 40 else HEX64_RE if length == 64 else None
    return bool(pattern and pattern.fullmatch(value))


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and not isinstance(value, bool) and math.isfinite(float(value))


def _add(blockers: list[dict[str, Any]], code: str, message: str, **details: Any) -> None:
    blockers.append({"code": code, "message": message, "details": details})


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _validate_url(value: Any) -> tuple[bool, str]:
    if type(value) is not str or not value.strip():
        return False, "url_not_string"
    try:
        parsed = urlsplit(value)
        host = parsed.hostname.lower() if parsed.hostname else ""
    except ValueError:
        return False, "url_parse_error"
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        return False, "url_not_https_or_hosted"
    if host in {"example.com", "example.org", "example.net"} or host.endswith(".invalid"):
        return False, "url_placeholder_host"
    return True, "ok"


def _find_self_attestation(value: Any, path: str = "claim") -> list[dict[str, Any]]:
    """Find control strings/booleans that cannot be evidence."""

    found: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in CONTROL_BOOLEAN_KEYS:
                found.append({"path": child_path, "kind": "boolean", "value": child})
            elif key_text in CONTROL_STRING_KEYS:
                found.append({"path": child_path, "kind": "status", "value": child})
            else:
                found.extend(_find_self_attestation(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_find_self_attestation(child, f"{path}[{index}]"))
    return found


def _validate_receipt(
    value: Any,
    *,
    expected_binding: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate a local authenticated and cryptographically bound receipt."""

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
    identity_keys = [key for key in ("receipt_id", "record_id", "identity") if key in value]
    digest_keys = [key for key in ("digest", "sha256") if key in value]
    if len(identity_keys) != 1 or not _strict_hex(value[identity_keys[0]], 40):
        return False, "receipt_identity_not_lower_hex40"
    if len(digest_keys) != 1 or not _strict_hex(value[digest_keys[0]], 64):
        return False, "receipt_digest_not_lower_hex64"
    for key in ("authenticated", "cryptographically_bound"):
        if type(value.get(key)) is not bool:
            return False, f"receipt_{key}_not_boolean"
        if value[key] is not True:
            return False, f"receipt_{key}_false"
    binding = value.get("binding")
    payload = value.get("payload")
    if not isinstance(binding, Mapping) or dict(binding) != dict(expected_binding):
        return False, "receipt_binding_mismatch"
    if not isinstance(payload, Mapping):
        return False, "receipt_payload_not_object"
    identities = [payload[key] for key in ("identity", "revision", "record_id") if key in payload]
    digests = [
        payload[key]
        for key in ("artifact_digest", "digest", "sha256", "content_sha256")
        if key in payload
    ]
    if not identities or any(not _strict_hex(item, 40) for item in identities):
        return False, "receipt_payload_identity_not_lower_hex40"
    if not digests or any(not _strict_hex(item, 64) for item in digests):
        return False, "receipt_payload_digest_not_lower_hex64"
    if len(set(identities)) != 1 or len(set(digests)) != 1:
        return False, "receipt_payload_identity_or_digest_conflict"
    for key, child in payload.items():
        key_text = str(key).lower()
        if key_text in CONTROL_BOOLEAN_KEYS or key_text in CONTROL_STRING_KEYS:
            return False, "receipt_payload_self_attestation"
        if key_text in URL_KEYS:
            valid, reason = _validate_url(child)
            if not valid:
                return False, reason
        if key_text.endswith("_id") and not _strict_hex(child, 40):
            return False, "receipt_payload_id_not_lower_hex40"
        if any(token in key_text for token in ("hash", "digest", "sha256")) and not _strict_hex(child, 64):
            return False, "receipt_payload_digest_not_lower_hex64"
    try:
        expected_digest = compute_receipt_digest(binding, payload)
    except (TypeError, ValueError):
        return False, "receipt_payload_not_canonicalizable"
    if value[digest_keys[0]] != expected_digest:
        return False, "receipt_digest_binding_mismatch"
    return True, "ok"


def _validate_exact_list(
    claim: Mapping[str, Any],
    key: str,
    expected: frozenset[str],
    blockers: list[dict[str, Any]],
) -> bool:
    value = claim.get(key)
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        _add(blockers, f"{key}_invalid", f"{key} must be a list of exact suite/domain IDs.")
        return False
    actual = set(value)
    if len(value) != len(actual) or actual != expected:
        _add(
            blockers,
            f"{key}_not_exact",
            f"{key} must contain the exact frozen set.",
            expected=sorted(expected),
            actual=sorted(actual),
        )
        return False
    return True


def _validate_coverage(claim: Mapping[str, Any], blockers: list[dict[str, Any]]) -> bool:
    good = True
    good &= _validate_exact_list(claim, "training_suite_ids", EXPECTED_TRAIN, blockers)
    good &= _validate_exact_list(claim, "primary_eval_suite_ids", EXPECTED_PRIMARY, blockers)
    unions = claim.get("domain_unions")
    if not isinstance(unions, Mapping):
        _add(blockers, "domain_unions_invalid", "domain_unions must contain train and primary_eval lists.")
        return False
    for role in ("train", "primary_eval"):
        role_value = unions.get(role)
        if (
            not isinstance(role_value, list)
            or any(type(item) is not str for item in role_value)
            or set(role_value) != EXPECTED_DOMAINS
            or len(role_value) != len(EXPECTED_DOMAINS)
        ):
            _add(
                blockers,
                "domain_union_not_exact",
                "Each role union must span exactly the 16 declared domains.",
                role=role,
                actual=role_value,
                expected=sorted(EXPECTED_DOMAINS),
            )
            good = False
    coverage_receipts = claim.get("coverage_receipts")
    if not isinstance(coverage_receipts, Mapping):
        _add(blockers, "coverage_receipts_missing", "Exact 12/14 evidence requires structured coverage receipts.")
        return False
    for suite_id in (*TRAIN_SUITE_IDS, *PRIMARY_EVAL_SUITE_IDS):
        role = "train" if suite_id in EXPECTED_TRAIN else "primary_eval"
        record = coverage_receipts.get(suite_id)
        valid, reason = _validate_receipt(
            record,
            expected_binding={"subject": suite_id, "role": role, "kind": "coverage"},
        )
        if not valid:
            _add(blockers, "coverage_receipt_invalid", "Every exact suite ID needs a bound coverage receipt.", suite_id=suite_id, reason=reason)
            good = False
    return bool(good)


def _classify_text(
    text: Any,
    blockers: list[dict[str, Any]],
    *,
    claim_kind: str | None = None,
) -> None:
    if text is None:
        return
    if type(text) is not str:
        _add(blockers, "claim_text_invalid", "claim_text must be a string and cannot authorize evidence by itself.")
        return
    for code, pattern in _TEXT_PROMOTION_PATTERNS:
        if pattern.search(text):
            _add(blockers, code, "Claim wording crosses a declared evidence boundary.", text=text)
    lowered = text.lower()
    broad_words = r"portfolio|all\s+16|all\s+53|company|production|general|universal|cross[- ]domain|improv(?:e|ement|ed)"
    if re.search(r"\bxlam\b", lowered) and re.search(broad_words, lowered):
        _add(blockers, "xlam_promotion_forbidden", "xLAM is bounded to the strict tool-use component.", text=text)
    if re.search(r"\bgsm8k\b", lowered) and re.search(
        r"primary|portfolio|training|train|held[- ]?out|company|production|general|useful|improv|promot",
        lowered,
    ):
        _add(blockers, "gsm8k_promotion_forbidden", "GSM8K remains calibration-only.", text=text)
    if re.search(r"substitut(?:e|ed|ion)|proxy|stand[- ]?in|instead\s+of", lowered) and re.search(
        r"benchmark|suite|gsm8k|math[- ]?500|xlam|bfcl|swe[- ]?bench", lowered
    ):
        _add(blockers, "related_benchmark_substitution", "A related benchmark cannot substitute for the declared suite.", text=text)
    if claim_kind != "heldout_result" and re.search(
        r"\b(?:all\s+14|14\s+(?:primary\s+)?(?:eval(?:uation)?\s+)?suites?)\b.*\bheld[- ]?out\b",
        text,
        re.IGNORECASE,
    ):
        # The kind-specific validator may clear this only after all provenance
        # and result receipts have been checked; prose alone never does.
        _add(blockers, "heldout_claim_requires_receipts", "A 14-suite held-out claim requires receipt-proven results.", text=text)


def _classify_protocol(claim: Mapping[str, Any], blockers: list[dict[str, Any]]) -> None:
    if type(claim.get("prospective")) is not bool:
        _add(blockers, "prospective_flag_not_boolean", "Protocol/preflight evidence needs an exact prospective boolean.")
    elif claim["prospective"] is not True:
        _add(blockers, "protocol_not_prospective", "Protocol/preflight claims must be explicitly prospective.")
    if not isinstance(claim.get("evidence"), Mapping):
        _add(blockers, "protocol_evidence_not_structured", "Protocol evidence must be a structured object, not a string, list, or boolean.")
    if re.search(r"\b(?:observed|measured|improved|won|result|ready|complete)\b", str(claim.get("claim_text", "")), re.IGNORECASE):
        _add(blockers, "protocol_claim_contains_result", "Prospective protocol text cannot assert a completed result.")


def _classify_component(claim: Mapping[str, Any], blockers: list[dict[str, Any]]) -> None:
    component = claim.get("component")
    if not isinstance(component, Mapping):
        _add(blockers, "component_missing", "Component evidence requires a structured component record.")
        return
    component_id = component.get("id")
    if type(component_id) is not str or component_id not in {"xlam_baseline_observation", "gsm8k_calibration"}:
        _add(blockers, "component_id_invalid", "Only the bounded xLAM observation and GSM8K calibration component are admissible.")
        return
    if component_id == "gsm8k_calibration" and component.get("role") != "calibration_only":
        _add(blockers, "gsm8k_not_calibration_only", "GSM8K may only be classified as calibration_only.")
    if component_id == "xlam_baseline_observation":
        if component.get("scope") != "strict_tool_use_component":
            _add(blockers, "xlam_scope_invalid", "xLAM evidence is limited to its strict tool-use component scope.")
        if component.get("seed") != 809 or component.get("n") != 100 or component.get("successes") != 7:
            _add(
                blockers,
                "xlam_observation_not_exact",
                "The only admissible observed xLAM component is the recorded seed-809 7/100 slice.",
            )
        if component.get("promote_to_portfolio") is not None:
            _add(blockers, "xlam_promotion_flag_present", "A self-attested xLAM promotion flag cannot authorize a portfolio claim.")
    receipt = claim.get("receipt")
    valid, reason = _validate_receipt(
        receipt,
        expected_binding={"subject": component_id, "role": "component", "kind": "observation"},
    )
    if not valid:
        _add(blockers, "component_receipt_invalid", "Component evidence requires a bound receipt, not a status string.", reason=reason)


def _classify_heldout(claim: Mapping[str, Any], blockers: list[dict[str, Any]]) -> None:
    if not _validate_coverage(claim, blockers):
        return
    holdout_receipts = claim.get("holdout_receipts")
    if not isinstance(holdout_receipts, Mapping):
        _add(blockers, "holdout_receipts_missing", "Held-out results require provenance receipts for every primary suite.")
        return
    for suite_id in PRIMARY_EVAL_SUITE_IDS:
        suite_receipts = holdout_receipts.get(suite_id)
        if not isinstance(suite_receipts, Mapping):
            _add(blockers, "holdout_suite_receipts_invalid", "Each primary suite needs all immutable holdout receipt fields.", suite_id=suite_id)
            continue
        for field in REQUIRED_HOLDOUT_RECEIPTS:
            valid, reason = _validate_receipt(
                suite_receipts.get(field),
                expected_binding={"subject": suite_id, "role": "primary_eval", "kind": f"holdout:{field}"},
            )
            if not valid:
                _add(blockers, "holdout_receipt_invalid", "A primary suite holdout field is not receipt-proven.", suite_id=suite_id, field=field, reason=reason)
    results = claim.get("results")
    if not isinstance(results, Mapping):
        _add(blockers, "heldout_results_missing", "Receipt-proven held-out claims require one result record per primary suite.")
        return
    for suite_id in PRIMARY_EVAL_SUITE_IDS:
        result = results.get(suite_id)
        if not isinstance(result, Mapping):
            _add(blockers, "heldout_result_invalid", "Every primary suite needs a structured result record.", suite_id=suite_id)
            continue
        if type(result.get("n")) is not int or result["n"] <= 0:
            _add(blockers, "heldout_sample_size_invalid", "Held-out result n must be a positive integer.", suite_id=suite_id)
        if not _finite_number(result.get("metric")) or not 0.0 <= float(result["metric"]) <= 1.0:
            _add(blockers, "heldout_metric_invalid", "Held-out metric must be a finite number in [0, 1].", suite_id=suite_id)
        interval = result.get("ci95")
        if not isinstance(interval, Mapping) or not all(_finite_number(interval.get(key)) for key in ("lower", "upper")):
            _add(blockers, "heldout_interval_invalid", "Each held-out result needs a finite 95% interval.", suite_id=suite_id)
        elif (
            not 0.0 <= float(interval["lower"]) <= 1.0
            or not 0.0 <= float(interval["upper"]) <= 1.0
            or float(interval["lower"]) > float(interval["upper"])
        ):
            _add(blockers, "heldout_interval_invalid", "A 95% interval must be ordered and lie within [0, 1].", suite_id=suite_id)
        valid, reason = _validate_receipt(
            result.get("receipt"),
            expected_binding={"subject": suite_id, "role": "primary_eval", "kind": "result"},
        )
        if not valid:
            _add(blockers, "heldout_result_receipt_invalid", "Each held-out result needs a bound result receipt.", suite_id=suite_id, reason=reason)
        else:
            try:
                expected_result_digest = compute_result_digest(result)
            except (TypeError, ValueError):
                _add(blockers, "heldout_result_not_canonicalizable", "A held-out result must be canonical JSON before its receipt can bind it.", suite_id=suite_id)
                continue
            payload = result["receipt"].get("payload") if isinstance(result.get("receipt"), Mapping) else None
            digest_values = []
            if isinstance(payload, Mapping):
                digest_values = [
                    payload[key]
                    for key in ("artifact_digest", "result_digest", "digest", "sha256", "content_sha256")
                    if key in payload
                ]
            if expected_result_digest not in digest_values:
                _add(blockers, "heldout_result_not_bound", "The result receipt must bind the exact metric, sample size, and uncertainty interval.", suite_id=suite_id)


def classify_claim(claim: Any) -> dict[str, Any]:
    """Classify one claim record and return a JSON-safe blocking report."""

    blockers: list[dict[str, Any]] = []
    if not isinstance(claim, Mapping):
        _add(blockers, "claim_not_object", "A claim must be a structured JSON object; prose alone is not evidence.")
        return _report(None, blockers)
    claim_kind = claim.get("claim_kind")
    if type(claim_kind) is not str or claim_kind not in CLAIM_KINDS:
        _add(blockers, "claim_kind_invalid", "claim_kind must be an explicit supported evidence class.", claim_kind=claim_kind)
        return _report(claim_kind, blockers)
    attestations = _find_self_attestation(claim)
    # ``prospective`` is a required protocol gate and is validated separately;
    # no other self-attested control is permitted to influence classification.
    attestations = [item for item in attestations if item["path"] != "claim.prospective"]
    if attestations:
        _add(blockers, "self_attested_control", "Strings/booleans asserting readiness, completion, verification, or promotion are not evidence.", fields=attestations)
    _classify_text(claim.get("claim_text"), blockers, claim_kind=claim_kind)
    if claim_kind == "protocol_preflight":
        _classify_protocol(claim, blockers)
    elif claim_kind == "component_evidence":
        _classify_component(claim, blockers)
    elif claim_kind == "portfolio_evidence":
        _validate_coverage(claim, blockers)
        if re.search(r"\b(?:improv|held[- ]?out|portfolio[- ]?wide)\b", str(claim.get("claim_text", "")), re.IGNORECASE):
            _add(blockers, "portfolio_result_requires_heldout_class", "Portfolio result wording requires the heldout_result class and receipts.")
    elif claim_kind == "heldout_result":
        _classify_heldout(claim, blockers)
    return _report(claim_kind, blockers)


def _report(claim_kind: Any, blockers: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ALLOWED" if allowed else "BLOCKED",
        "claim_allowed": allowed,
        "claim_kind": claim_kind,
        "claim_class": claim_kind if allowed else None,
        "scope": claim_kind if allowed else None,
        "blocker_codes": [item["code"] for item in blockers],
        "blockers": blockers,
        "zero_cost": True,
        "network_accessed": False,
        "paid_calls_executed": False,
    }


def classify_claim_text(text: str) -> dict[str, Any]:
    """Explicitly reject unstructured prose as evidence."""

    blockers: list[dict[str, Any]] = []
    _add(blockers, "unstructured_claim_text", "Prose must be accompanied by a typed claim record and admissible receipts.")
    _classify_text(text, blockers)
    return _report(None, blockers)


def validate_claim(claim: Any) -> dict[str, Any]:
    """Alias for callers that prefer a validation verb."""

    return classify_claim(claim)


classify_claim_record = classify_claim
classify_evidence = classify_claim


def _load_claim(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--claim", type=Path, help="Path to a JSON claim record")
    source.add_argument("--claim-json", help="Inline JSON claim record")
    args = parser.parse_args(argv)
    try:
        claim = _load_claim(args.claim) if args.claim else json.loads(args.claim_json)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        report = _report(None, [{"code": "claim_input_error", "message": str(exc), "details": {}}])
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    report = classify_claim(claim)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["claim_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
