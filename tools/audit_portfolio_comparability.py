#!/usr/bin/env python3
"""Read-only arithmetic and recorded-configuration audit of base/trained receipts.

The current v1 format does not bind targets or scorer implementation. Matching
recorded settings therefore remain UNKNOWN for overall comparability, never a
scientific readiness or decontamination certificate. No scores are regraded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import stat
from typing import Any

MAX_BYTES = 8 * 1024 * 1024
SCHEMA = "pavlov-portfolio-source-eval-v1"
POLICY_FIELDS = (
    ("dataset_revision",),
    ("source_revisions",),
    ("model", "model_id"),
    ("model", "model_revision"),
    ("model", "model_revision_scope"),
    *(
        ("sampling", name)
        for name in (
            "samples_per_problem",
            "retries",
            "temperature",
            "max_tokens",
            "top_p",
            "max_prompt_tokens",
            "prompt_truncation",
            "seed",
        )
    ),
)
ROW_FIELDS = (
    "suite_id",
    "source_id",
    "prompt_sha256",
    "prompt_token_ids_sha256",
    "prompt_tokens",
    "original_prompt_tokens",
    "prompt_truncated",
    "sampling_seed",
)


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number, not a coerced value")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{field} is not representable") from None
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("receipt contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ValueError("receipt contains a nonfinite JSON constant")


def read_receipt(path: Path) -> tuple[dict[str, Any], str]:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_NONBLOCK"):
        raise ValueError("receipt reads require nonsymlink regular-file support")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("receipt must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(data) > MAX_BYTES:
        raise ValueError("receipt exceeds the 8 MiB audit limit")
    result = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
        parse_float=lambda value: _number(float(value), "JSON float"),
    )
    if not isinstance(result, dict):
        raise ValueError("receipt must be a JSON object")
    return result, hashlib.sha256(data).hexdigest()


def _equal_number(actual: Any, expected: float, field: str) -> None:
    if not math.isclose(_number(actual, field), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} disagrees with recorded rows")


def _count(actual: Any, expected: int, field: str) -> None:
    if type(actual) is not int or actual != expected:
        raise ValueError(f"{field} disagrees with recorded rows")


def validate_arithmetic(receipt: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if receipt.get("schema_version") != SCHEMA or receipt.get("status") != "SCORED":
        raise ValueError(f"{role}: unsupported schema or unscored receipt")
    if receipt.get("role") != role:
        raise ValueError(f"{role}: receipt role does not match the requested input")
    if (
        receipt.get("evidence_class") != "portfolio_source_validation"
        or receipt.get("is_e1_e14_score") is not False
        or receipt.get("is_model_score") is not True
    ):
        raise ValueError(f"{role}: source-validation evidence scope is missing or contradictory")
    rows = receipt.get("records")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{role}: records must be a nonempty list of objects")
    scores = []
    grouped: dict[str, list[float]] = {}
    identities = set()
    prompts = set()
    for index, row in enumerate(rows):
        for key in ROW_FIELDS:
            _validate_recorded_value(row.get(key), f"{role}.records[{index}].{key}")
        identity = (row.get("suite_id"), row.get("source_id"))
        prompt = row.get("prompt_sha256")
        if None not in identity:
            if identity in identities:
                raise ValueError(f"{role}: duplicate source identity")
            identities.add(identity)
        if prompt is not None:
            if prompt in prompts:
                raise ValueError(f"{role}: duplicate prompt fingerprint")
            prompts.add(prompt)
        score = _number(row.get("score"), f"{role}.records[{index}].score")
        if not 0 <= score <= 1:
            raise ValueError(f"{role}.records[{index}].score must be in [0, 1]")
        if type(row.get("exact")) is not bool or row["exact"] != (score == 1.0):
            raise ValueError(f"{role}.records[{index}].exact disagrees with its score")
        suite = row.get("suite_id")
        if not isinstance(suite, str) or not suite:
            raise ValueError(f"{role}.records[{index}].suite_id is missing")
        scores.append(score)
        grouped.setdefault(suite, []).append(score)
    if set(grouped) != {"api_bank_rlvr_train", "swe_gym_train"}:
        raise ValueError(f"{role}: the two portfolio source suites are not both present")
    _count(receipt.get("count"), len(rows), f"{role}.count")
    _equal_number(
        receipt.get("mean_reward"),
        math.fsum(score / len(scores) for score in scores),
        f"{role}.mean_reward",
    )
    _equal_number(
        receipt.get("exact_match_rate"),
        sum(score == 1 for score in scores) / len(scores),
        f"{role}.exact_match_rate",
    )
    summaries = receipt.get("suites")
    if not isinstance(summaries, dict) or set(summaries) != set(grouped):
        raise ValueError(f"{role}.suites disagree with recorded rows")
    for suite, values in grouped.items():
        summary = summaries[suite]
        if not isinstance(summary, dict):
            raise ValueError(f"{role}.suites must contain objects")
        _count(summary.get("count"), len(values), f"{role}.suite.count")
        _count(
            summary.get("exact_matches"),
            sum(value == 1 for value in values),
            f"{role}.suite.exact_matches",
        )
        _equal_number(
            summary.get("mean_reward"),
            math.fsum(value / len(values) for value in values),
            f"{role}.suite.mean_reward",
        )
        _equal_number(
            summary.get("exact_match_rate"),
            sum(value == 1 for value in values) / len(values),
            f"{role}.suite.exact_match_rate",
        )
    return rows


def _lookup(value: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_recorded_value(value: Any, field: str) -> None:
    if value is None:
        return  # Missing historical fields are unknown, never invented.
    name = field.rsplit(".", 1)[-1]
    valid = True
    if name in {"prompt_sha256", "prompt_token_ids_sha256"}:
        valid = isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    elif name == "source_revisions":
        valid = (
            isinstance(value, dict)
            and bool(value)
            and all(
                isinstance(key, str) and key.strip() and isinstance(item, str) and item.strip()
                for key, item in value.items()
            )
        )
    elif name in {"samples_per_problem", "retries"}:
        valid = type(value) is int and value == (1 if name == "samples_per_problem" else 0)
    elif name == "model_revision_scope":
        valid = value == "tokenizer_request_only"
    elif name == "prompt_truncation":
        valid = value in {"prefix", "reject_over_budget"}
    elif name in {"max_tokens", "max_prompt_tokens"}:
        valid = type(value) is int and value > 0
    elif name in {"prompt_tokens", "original_prompt_tokens"}:
        valid = type(value) is int and value >= 0
    elif name in {"seed", "sampling_seed"}:
        valid = type(value) is int
    elif name == "prompt_truncated":
        valid = type(value) is bool
    elif name in {"temperature", "top_p"}:
        number = _number(value, field)
        valid = number >= 0 if name == "temperature" else 0 < number <= 1
    else:
        valid = isinstance(value, str) and bool(value.strip())
    if not valid:
        raise ValueError(f"{field} has invalid recorded metadata")


def _validate_recorded_relations(
    receipt: dict[str, Any], rows: list[dict[str, Any]], role: str
) -> None:
    seed = _lookup(receipt, ("sampling", "seed"))
    limit = _lookup(receipt, ("sampling", "max_prompt_tokens"))
    truncation_policy = _lookup(receipt, ("sampling", "prompt_truncation"))
    for index, row in enumerate(rows):
        sampled_seed = row.get("sampling_seed")
        if seed is not None and sampled_seed is not None and sampled_seed != seed + index:
            raise ValueError(
                f"{role}.records[{index}].sampling_seed contradicts the sampling policy"
            )
        original, sent, truncated = (
            row.get(key) for key in ("original_prompt_tokens", "prompt_tokens", "prompt_truncated")
        )
        if all(value is not None for value in (limit, original, sent, truncated)) and (
            sent != min(original, limit)
            or truncated != (original > limit)
            or (truncation_policy == "reject_over_budget" and (original > limit or truncated))
        ):
            raise ValueError(
                f"{role}.records[{index}] has contradictory prompt truncation metadata"
            )


def audit(base: dict[str, Any], trained: dict[str, Any]) -> dict[str, Any]:
    base_rows = validate_arithmetic(base, "base")
    trained_rows = validate_arithmetic(trained, "trained")
    mismatches: set[str] = set()
    missing: set[str] = set()

    def compare(left: Any, right: Any, field: str) -> None:
        _validate_recorded_value(left, "base." + field)
        _validate_recorded_value(right, "trained." + field)
        if left is None or right is None or left == "" or right == "":
            missing.add(field)
        elif type(left) is not type(right) or left != right:
            mismatches.add(field)

    for path in POLICY_FIELDS:
        compare(_lookup(base, path), _lookup(trained, path), ".".join(path))
    if len(base_rows) != len(trained_rows):
        mismatches.add("ordered_cohort_length")
    # Length mismatch is already reported above; collect shared-prefix diagnostics.
    for left, right in zip(base_rows, trained_rows, strict=False):
        for key in ROW_FIELDS:
            compare(left.get(key), right.get(key), f"ordered_records.{key}")
    _validate_recorded_relations(base, base_rows, "base")
    _validate_recorded_relations(trained, trained_rows, "trained")
    recorded = "incompatible" if mismatches else "unknown" if missing else "compatible"
    return {
        "schema_version": 1,
        "status": "incompatible" if mismatches else "unknown",
        "arithmetic": "consistent",
        "recorded_configuration": recorded,
        "mismatches": sorted(mismatches),
        "missing_recorded_fields": sorted(missing),
        "missing_comparability_evidence": [
            "target_content_binding",
            "scorer_implementation_binding",
        ],
        "scientific_comparison_established": False,
        "not_established": [
            "provider_weight_identity",
            "decontamination",
            "causal_effect",
            "representativeness",
            "statistical_significance",
        ],
        "scope": "Checks recorded arithmetic and ordered prompt/sampling metadata only. V1 receipts do not bind targets or scorer implementation; no new metric, score delta, regrading or scientific readiness claim is produced.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        base, base_hash = read_receipt(args.base)
        trained, trained_hash = read_receipt(args.trained)
        result = audit(base, trained)
        result["input_sha256"] = {"base": base_hash, "trained": trained_hash}
    except (OSError, ValueError, OverflowError, RecursionError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "invalid",
                    "error": str(exc),
                    "scientific_comparison_established": False,
                }
            )
        )
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 1 if result["status"] == "incompatible" else 3


if __name__ == "__main__":
    raise SystemExit(main())
