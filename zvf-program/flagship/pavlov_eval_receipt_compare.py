#!/usr/bin/env python3
"""Offline, fail-closed comparison of the frozen xLAM base and trained receipts.

The comparator never contacts W&B, Tinker, Hugging Face, or any other service.  It
only reads two local JSON receipts and emits a comparison receipt.  A comparison
is admissible only when the same examples, immutable evaluation provenance,
verifier, and component-only claim boundary are present in both receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_BASE_RECEIPT = Path("autoresearch/orchestrator-260809-0922/base_eval_100.json")
DEFAULT_SUITE_ID = "xlam_component"
DEFAULT_DOMAINS = ("tool_use",)
PAVLOV_PRIMARY_EVAL_PORTFOLIO_ID = "pavlov-primary-eval-14-suite-v1"
PAVLOV_PRIMARY_EVAL_SUITE_COUNT = 14
BASE_EXPECTED_EXAMPLES = 100
BASE_EXPECTED_SUCCESSES = 7
WILSON_Z_95 = 1.959963984540054

# This is a primary-evaluation manifest, not evidence that the xLAM component
# has run or that any entry is held out.  The exact manifest prevents a receipt
# from using the portfolio label while silently substituting unknown suites.
PAVLOV_PRIMARY_EVAL_SUITE_DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("swe_bench_pro_eval", ("code", "long_horizon")),
    ("frontier_swe_eval", ("code", "ml", "long_horizon")),
    ("sdab_eval", ("code", "ml", "long_horizon", "enterprise")),
    ("banker_toolbench_eval", ("finance", "enterprise", "tool_use", "long_horizon")),
    ("apex_agents_eval", ("multi_domain", "finance", "enterprise", "long_horizon", "tool_use")),
    ("webbench_eval", ("browser", "computer_use", "enterprise")),
    ("binaryaudit_eval", ("security", "code", "long_horizon")),
    ("lifescibench_eval", ("science", "long_horizon", "tool_use")),
    ("mle_bench_eval", ("ml", "code", "long_horizon")),
    ("agentharm_eval", ("alignment", "security", "tool_use")),
    ("verilog_eval", ("chip_design", "code")),
    ("appbench_eval", ("design", "computer_use", "code")),
    ("openreward_games_eval", ("games", "long_horizon", "tool_use")),
    ("frontiermath_eval", ("math",)),
)
PAVLOV_PRIMARY_EVAL_SUITE_IDS = tuple(
    suite_id for suite_id, _ in PAVLOV_PRIMARY_EVAL_SUITE_DOMAINS
)


def _primary_eval_manifest() -> list[dict[str, Any]]:
    return [
        {"suite_id": suite_id, "domains": list(domains), "role": "primary_eval"}
        for suite_id, domains in PAVLOV_PRIMARY_EVAL_SUITE_DOMAINS
    ]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$", re.IGNORECASE)
_REVISION_RE = re.compile(
    r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}|sha256:[0-9a-fA-F]{64})$",
    re.IGNORECASE,
)
_URL_PREFIXES = ("https://", "http://")
_PLACEHOLDER_MARKERS = (
    "placeholder",
    "to_be_pinned",
    "to-be-pinned",
    "tbd",
    "todo",
    "changeme",
    "replace_me",
    "replace-me",
)
_COMMON_PROVENANCE_KEYS = (
    "dataset_revision",
    "split_manifest_sha256",
    "task_id_sha256",
    "verifier_sha256",
    "base_model_revision",
    "tokenizer_revision",
    "decontamination_sha256",
    "decontamination_receipt",
    "container_digest",
    "runtime_digest",
    "license_id",
    "license_receipt",
)
_CLAIM_FORBIDDEN_PHRASES = (
    "all pavlov",
    "all domains",
    "full portfolio",
    "all-company",
    "all company",
    "covers all",
)


class ReceiptComparisonError(ValueError):
    """Raised by :func:`require_comparable` when the pair is inadmissible."""


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered or lowered in {"none", "null", "na", "n/a", "unknown", "unset"}:
        return True
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return True
    zero_candidate = lowered.removeprefix("sha256:")
    return len(zero_candidate) in {40, 64} and set(zero_candidate) == {"0"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _receipt_hash(receipt: Mapping[str, Any]) -> str:
    try:
        payload = _canonical_json(receipt)
    except (TypeError, ValueError):
        # Preserve a deterministic comparison receipt for malformed input
        # (for example, a decoder that admits NaN) while validation blocks it.
        payload = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
            default=str,
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def wilson_uncertainty(
    successes: int,
    trials: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return the exact deterministic 95% Wilson output required in receipts."""

    if confidence_level != 0.95:
        raise ValueError("the receipt contract fixes confidence_level to 0.95")
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid Wilson counts")
    p = successes / trials
    z = WILSON_Z_95
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return {
        "method": "wilson",
        "confidence_level": 0.95,
        "successes": successes,
        "trials": trials,
        "estimate": p,
        "wilson_low": max(0.0, center - half),
        "wilson_high": min(1.0, center + half),
    }


def _mapping(value: Any, path: str, errors: list[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return None
    return value


def _required_text(
    mapping: Mapping[str, Any], key: str, path: str, errors: list[str]
) -> str | None:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip() or _is_placeholder(value):
        errors.append(f"{path}.{key} must be a non-empty string")
        return None
    return value.strip()


def _required_hash(
    mapping: Mapping[str, Any], key: str, path: str, errors: list[str]
) -> str | None:
    value = _required_text(mapping, key, path, errors)
    if value is not None and not _SHA256_RE.fullmatch(value):
        errors.append(f"{path}.{key} must be a 64-hex SHA256")
    return value.lower() if value is not None and _SHA256_RE.fullmatch(value) else value


def _required_revision(
    mapping: Mapping[str, Any], key: str, path: str, errors: list[str]
) -> str | None:
    value = _required_text(mapping, key, path, errors)
    if value is not None and not _REVISION_RE.fullmatch(value):
        errors.append(f"{path}.{key} must be an immutable revision or SHA256 digest")
    return value.lower() if value is not None and _REVISION_RE.fullmatch(value) else value


def _required_digest(
    mapping: Mapping[str, Any], key: str, path: str, errors: list[str]
) -> str | None:
    value = _required_text(mapping, key, path, errors)
    if value is not None and not _DIGEST_RE.fullmatch(value):
        errors.append(f"{path}.{key} must be a sha256:<64-hex> digest")
    return value.lower() if value is not None and _DIGEST_RE.fullmatch(value) else value


def _extract_rows(
    receipt: Mapping[str, Any], label: str, errors: list[str]
) -> list[dict[str, Any]]:
    rows = receipt.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append(f"{label}.rows must be a non-empty list")
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"{label}.rows[{position}] must be an object")
            continue
        raw_id = row.get("example_id", row.get("task_id", row.get("id")))
        if raw_id is None and isinstance(row.get("index"), int) and not isinstance(
            row.get("index"), bool
        ):
            raw_id = f"index:{row['index']}"
        if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool) or not str(raw_id).strip():
            errors.append(f"{label}.rows[{position}] lacks a stable example_id")
            continue
        example_id = str(raw_id)
        if example_id in seen:
            errors.append(f"{label}.rows duplicates example_id {example_id!r}")
            continue
        seen.add(example_id)
        score = row.get("score")
        if not _is_finite_number(score) or not 0.0 <= float(score) <= 1.0:
            errors.append(f"{label}.rows[{position}].score must be finite in [0, 1]")
            continue
        prompt_hash = row.get("prompt_sha256")
        target_hash = row.get("target_sha256")
        for field, value in (("prompt_sha256", prompt_hash), ("target_sha256", target_hash)):
            if value is not None and (
                not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            ):
                errors.append(f"{label}.rows[{position}].{field} must be a 64-hex SHA256")
        normalized.append(
            {
                "example_id": example_id,
                "prompt_sha256": prompt_hash.lower() if isinstance(prompt_hash, str) else None,
                "target_sha256": target_hash.lower() if isinstance(target_hash, str) else None,
                "score": float(score),
                "success": float(score) == 1.0,
            }
        )
    return normalized


def _validate_uncertainty(
    receipt: Mapping[str, Any],
    label: str,
    successes: int,
    trials: int,
    errors: list[str],
) -> Mapping[str, Any] | None:
    actual = _mapping(receipt.get("uncertainty"), f"{label}.uncertainty", errors)
    if actual is None:
        return None
    expected = wilson_uncertainty(successes, trials)
    if dict(actual) != expected:
        errors.append(
            f"{label}.uncertainty is not the exact deterministic Wilson output "
            f"for {successes}/{trials}"
        )
    return actual


def _validate_claim_boundary(
    receipt: Mapping[str, Any], label: str, errors: list[str]
) -> dict[str, Any]:
    suite_id = receipt.get("suite_id")
    if suite_id != DEFAULT_SUITE_ID:
        errors.append(f"{label}.suite_id must be {DEFAULT_SUITE_ID!r}")
    domains = receipt.get("domains")
    if domains != list(DEFAULT_DOMAINS):
        errors.append(f"{label}.domains must be exactly {list(DEFAULT_DOMAINS)!r}")
    if receipt.get("suite_role") != "component":
        errors.append(f"{label}.suite_role must be 'component'")
    portfolio = _mapping(receipt.get("portfolio"), f"{label}.portfolio", errors)
    if portfolio is not None:
        if portfolio.get("portfolio_id") != PAVLOV_PRIMARY_EVAL_PORTFOLIO_ID:
            errors.append(f"{label}.portfolio.portfolio_id is not the primary-eval portfolio")
        if portfolio.get("portfolio_role") != "primary_eval":
            errors.append(f"{label}.portfolio.portfolio_role must be 'primary_eval'")
        if portfolio.get("suite_count") != PAVLOV_PRIMARY_EVAL_SUITE_COUNT:
            errors.append(f"{label}.portfolio.suite_count must be 14")
        if portfolio.get("suite_ids") != list(PAVLOV_PRIMARY_EVAL_SUITE_IDS):
            errors.append(f"{label}.portfolio.suite_ids must be the exact primary-eval manifest")
        if portfolio.get("suites") != _primary_eval_manifest():
            errors.append(f"{label}.portfolio.suites must be the exact primary-eval manifest")
        if portfolio.get("component_only") is not True:
            errors.append(f"{label}.portfolio.component_only must be true")
        if portfolio.get("component_suite_id") != DEFAULT_SUITE_ID:
            errors.append(f"{label}.portfolio.component_suite_id is not xLAM component")
        if portfolio.get("component_domains") != list(DEFAULT_DOMAINS):
            errors.append(f"{label}.portfolio.component_domains must be tool_use only")
        claim = " ".join(
            str(portfolio.get(key, ""))
            for key in ("coverage_claim", "claim_boundary")
        ).lower()
        if any(phrase in claim for phrase in _CLAIM_FORBIDDEN_PHRASES):
            errors.append(f"{label}.portfolio claim exceeds the xLAM component boundary")
    return {
        "suite_id": suite_id,
        "domains": domains,
        "portfolio_id": portfolio.get("portfolio_id") if portfolio else None,
        "component_only": portfolio.get("component_only") if portfolio else None,
    }


def _validate_service_provenance(
    receipt: Mapping[str, Any], label: str, source_kind: str, errors: list[str]
) -> dict[str, Any]:
    wandb = _mapping(receipt.get("wandb"), f"{label}.wandb", errors)
    tinker = _mapping(receipt.get("tinker"), f"{label}.tinker", errors)
    hf = _mapping(receipt.get("hf", receipt.get("huggingface")), f"{label}.hf", errors)
    identities: dict[str, Any] = {}
    if wandb is not None:
        for key in ("id", "url", "entity", "project", "group"):
            _required_text(wandb, key, f"{label}.wandb", errors)
        if isinstance(wandb.get("url"), str) and not wandb["url"].startswith(_URL_PREFIXES):
            errors.append(f"{label}.wandb.url must be http(s)")
        identities["wandb"] = dict(wandb)
    if tinker is not None:
        run_id = tinker.get("run_id", tinker.get("tinker_run_id"))
        model_id = tinker.get("model_id", tinker.get("model"))
        if not isinstance(run_id, str) or not run_id.strip():
            errors.append(f"{label}.tinker.run_id must be non-empty")
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"{label}.tinker.model_id must be non-empty")
        identities["tinker"] = dict(tinker)
    if hf is not None:
        repo_id = hf.get("repo_id", hf.get("repository"))
        revision = hf.get("revision", hf.get("commit"))
        if not isinstance(repo_id, str) or not repo_id.strip():
            errors.append(f"{label}.hf.repo_id must be non-empty")
        if not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision):
            errors.append(f"{label}.hf.revision must be immutable")
        identities["hf"] = dict(hf)
    if source_kind == "sampler_path":
        provenance = receipt.get("provenance")
        adapter_revision = provenance.get("adapter_revision") if isinstance(provenance, Mapping) else None
        if not isinstance(adapter_revision, str) or not _REVISION_RE.fullmatch(adapter_revision):
            errors.append(f"{label}.provenance.adapter_revision is required for sampler_path")
        elif hf is not None and isinstance(hf.get("revision", hf.get("commit")), str):
            if hf.get("revision", hf.get("commit")).lower() != adapter_revision.lower():
                errors.append(f"{label}.hf.revision must equal adapter_revision")
    return identities


def _validate_receipt(
    receipt: Mapping[str, Any], label: str, expected_source_kind: str
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return {"errors": [f"{label} must be a JSON object"]}
    source_kind = receipt.get("source_kind")
    if source_kind != expected_source_kind:
        errors.append(f"{label}.source_kind must be {expected_source_kind!r}")
    rows = _extract_rows(receipt, label, errors)
    successes = sum(row["success"] for row in rows)
    trials = len(rows)
    if receipt.get("examples") != trials:
        errors.append(f"{label}.examples must equal the number of receipt rows")
    if trials and not _is_finite_number(receipt.get("mean_strict_reward")):
        errors.append(f"{label}.mean_strict_reward must be finite")
    if trials:
        expected_mean = sum(row["score"] for row in rows) / trials
        if receipt.get("mean_strict_reward") != expected_mean:
            errors.append(f"{label}.mean_strict_reward does not match row scores exactly")
    expected_rate = successes / trials if trials else None
    if receipt.get("perfect_call_rate") != expected_rate:
        errors.append(f"{label}.perfect_call_rate does not match row successes exactly")
    uncertainty = _validate_uncertainty(receipt, label, successes, trials, errors) if trials else None
    provenance_obj = _mapping(receipt.get("provenance"), f"{label}.provenance", errors)
    provenance: dict[str, Any] = {}
    if provenance_obj is not None:
        for key in _COMMON_PROVENANCE_KEYS:
            if key.endswith("_digest"):
                value = _required_digest(provenance_obj, key, f"{label}.provenance", errors)
            elif key.endswith("_revision"):
                value = _required_revision(provenance_obj, key, f"{label}.provenance", errors)
            elif key.endswith("_sha256"):
                value = _required_hash(provenance_obj, key, f"{label}.provenance", errors)
            else:
                value = _required_text(provenance_obj, key, f"{label}.provenance", errors)
            provenance[key] = value
        adapter_revision = provenance_obj.get("adapter_revision")
        if adapter_revision is not None and (
            not isinstance(adapter_revision, str) or not _REVISION_RE.fullmatch(adapter_revision)
        ):
            errors.append(f"{label}.provenance.adapter_revision must be immutable")
        provenance["adapter_revision"] = adapter_revision.lower() if isinstance(adapter_revision, str) else adapter_revision
    dataset_id = receipt.get("dataset_id")
    dataset_split = receipt.get("dataset_split")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        errors.append(f"{label}.dataset_id must be non-empty")
    if not isinstance(dataset_split, str) or not dataset_split.strip():
        errors.append(f"{label}.dataset_split must be non-empty")
    claim = _validate_claim_boundary(receipt, label, errors)
    identities = _validate_service_provenance(receipt, label, source_kind, errors)
    return {
        "errors": errors,
        "rows": rows,
        "example_ids": [row["example_id"] for row in rows],
        "row_identity": [
            (row["example_id"], row["prompt_sha256"], row["target_sha256"])
            for row in rows
        ],
        "successes": successes,
        "trials": trials,
        "uncertainty": uncertainty,
        "provenance": provenance,
        "dataset_id": dataset_id,
        "dataset_split": dataset_split,
        "seed": receipt.get("seed"),
        "tokenizer_model": receipt.get("tokenizer_model"),
        "claim": claim,
        "identities": identities,
        "source_kind": source_kind,
    }


def _compare_provenance(
    base: Mapping[str, Any], trained: Mapping[str, Any], errors: list[str]
) -> None:
    base_provenance = base.get("provenance", {})
    trained_provenance = trained.get("provenance", {})
    for key in _COMMON_PROVENANCE_KEYS:
        base_value = base_provenance.get(key)
        trained_value = trained_provenance.get(key)
        if base_value != trained_value:
            errors.append(f"paired provenance mismatch: {key}")
    for key in ("dataset_id", "dataset_split", "seed", "tokenizer_model"):
        if base.get(key) != trained.get(key):
            errors.append(f"paired evaluation mismatch: {key}")
    base_wandb = base.get("identities", {}).get("wandb", {})
    trained_wandb = trained.get("identities", {}).get("wandb", {})
    for key in ("entity", "project", "group"):
        if base_wandb.get(key) != trained_wandb.get(key):
            errors.append(f"paired W&B provenance mismatch: {key}")
    if base_wandb.get("id") == trained_wandb.get("id"):
        errors.append("paired W&B runs must have distinct run IDs")
    base_tinker = base.get("identities", {}).get("tinker", {})
    trained_tinker = trained.get("identities", {}).get("tinker", {})
    if base_tinker.get("model_id", base_tinker.get("model")) != trained_tinker.get(
        "model_id", trained_tinker.get("model")
    ):
        errors.append("paired Tinker provenance mismatch: model_id")
    if base_tinker.get("run_id", base_tinker.get("tinker_run_id")) == trained_tinker.get(
        "run_id", trained_tinker.get("tinker_run_id")
    ):
        errors.append("paired Tinker runs must have distinct run IDs")
    base_hf = base.get("identities", {}).get("hf", {})
    trained_hf = trained.get("identities", {}).get("hf", {})
    trained_base_revision = trained_hf.get("base_model_revision")
    if trained_base_revision is not None and trained_base_revision != base_provenance.get("base_model_revision"):
        errors.append("trained HF base_model_revision does not match the frozen base")
    if base_hf.get("revision") != base_provenance.get("base_model_revision"):
        errors.append("base HF revision does not match base_model_revision")


def compare_receipts(
    base_receipt: Mapping[str, Any], trained_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a PASS/BLOCKED paired comparison receipt without external I/O."""

    base = _validate_receipt(base_receipt, "base", "base_model")
    trained = _validate_receipt(trained_receipt, "trained", "sampler_path")
    errors = [*base.get("errors", []), *trained.get("errors", [])]
    if base.get("trials") != BASE_EXPECTED_EXAMPLES or base.get("successes") != BASE_EXPECTED_SUCCESSES:
        errors.append("base receipt must be the frozen 7/100 evaluation")
    if base.get("example_ids") != trained.get("example_ids"):
        errors.append("paired example IDs/order differ")
    if base.get("row_identity") != trained.get("row_identity"):
        errors.append("paired prompt/target identities differ")
    _compare_provenance(base, trained, errors)
    base_adapter = base.get("provenance", {}).get("adapter_revision")
    if base_adapter not in (None, ""):
        errors.append("base receipt must not contain an adapter revision")
    trained_adapter = trained.get("provenance", {}).get("adapter_revision")
    if trained_adapter == base.get("provenance", {}).get("base_model_revision"):
        errors.append("trained adapter revision must identify a trained adapter, not the base model")
    base_successes = int(base.get("successes", 0))
    trained_successes = int(trained.get("successes", 0))
    trials = min(int(base.get("trials", 0)), int(trained.get("trials", 0)))
    paired = {
        "trials": trials,
        "base_successes": base_successes,
        "trained_successes": trained_successes,
        "trained_only_successes": sum(
            row["success"] and not base_row["success"]
            for row, base_row in zip(trained.get("rows", []), base.get("rows", []))
        ),
        "base_only_successes": sum(
            base_row["success"] and not row["success"]
            for row, base_row in zip(trained.get("rows", []), base.get("rows", []))
        ),
        "ties": sum(
            row["success"] == base_row["success"]
            for row, base_row in zip(trained.get("rows", []), base.get("rows", []))
        ),
        "delta_perfect_call_rate": (
            trained_successes / trials - base_successes / trials if trials else None
        ),
    }
    base_rate = (
        base_receipt.get("perfect_call_rate")
        if isinstance(base_receipt, Mapping)
        else None
    )
    trained_rate = (
        trained_receipt.get("perfect_call_rate")
        if isinstance(trained_receipt, Mapping)
        else None
    )
    result = {
        "schema_version": "pavlov-xlam-paired-comparison-v1",
        "status": "PASS" if not errors else "BLOCKED",
        "comparison_kind": "frozen-base-vs-trained-xlam-component",
        "base_receipt_sha256": _receipt_hash(base_receipt),
        "trained_receipt_sha256": _receipt_hash(trained_receipt),
        "base": {
            "examples": base.get("trials"),
            "successes": base.get("successes"),
            "perfect_call_rate": base_rate,
            "uncertainty": base.get("uncertainty"),
        },
        "trained": {
            "examples": trained.get("trials"),
            "successes": trained.get("successes"),
            "perfect_call_rate": trained_rate,
            "uncertainty": trained.get("uncertainty"),
        },
        "paired": paired,
        "claim_boundary": {
            "suite_id": DEFAULT_SUITE_ID,
            "domains": list(DEFAULT_DOMAINS),
            "portfolio_id": PAVLOV_PRIMARY_EVAL_PORTFOLIO_ID,
            "portfolio_role": "primary_eval",
            "component_only": True,
        },
        "errors": errors,
    }
    return result


def require_comparable(
    base_receipt: Mapping[str, Any], trained_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    result = compare_receipts(base_receipt, trained_receipt)
    if result["status"] != "PASS":
        raise ReceiptComparisonError("receipt comparison blocked: " + "; ".join(result["errors"]))
    return result


def load_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "base_receipt", type=Path, nargs="?", default=DEFAULT_BASE_RECEIPT
    )
    parser.add_argument("trained_receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        base = load_receipt(args.base_receipt)
        trained = load_receipt(args.trained_receipt)
        comparison = compare_receipts(base, trained)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        comparison = {
            "schema_version": "pavlov-xlam-paired-comparison-v1",
            "status": "BLOCKED",
            "comparison_kind": "frozen-base-vs-trained-xlam-component",
            "errors": [f"receipt load/parse failure: {exc}"],
        }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0 if comparison["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
