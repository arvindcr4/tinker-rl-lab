"""Offline, fail-closed analysis for paired Pavlov xLAM receipts.

This module consumes evaluator receipts only.  It never contacts W&B, Hugging
Face, Tinker, or the network; URLs in receipts are treated as provenance
strings.  A successful report is deliberately scoped to the xLAM component and
does not authorize held-out generalization, portfolio, or company claims.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
from statistics import fmean
from typing import Any

from flagship.pavlov_statistics import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    StatisticsInputError,
    exact_mcnemar_two_sided,
    mcnemar_discordant_counts,
    newcombe_paired_risk_difference_interval,
    paired_bootstrap_mean_difference,
    wilson_interval,
)


ANALYSIS_SCHEMA_VERSION = "pavlov-xlam-analysis-v1"
RECEIPT_SCHEMA_VERSION = "pavlov-xlam-eval-v1"
BASELINE_PERFECT_CALLS = 7
BASELINE_TRIALS = 100
BASELINE_RATE = BASELINE_PERFECT_CALLS / BASELINE_TRIALS

_MISSING = object()
_ALLOWED_CLAIM_SCOPES = {
    "xlam",
    "xlam_eval",
    "xlam_component",
    "xlam_component_only",
    "xlam_function_calling",
}
_SUCCESSFUL_TINKER_STATUSES = {"complete", "completed", "success", "succeeded"}
_INVARIANT_PROVENANCE_FIELDS = (
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "dataset_id",
    "dataset_revision",
    "split_manifest_sha256",
    "task_id_manifest_sha256",
    "verifier_revision",
    "container_digest",
    "decontamination_receipt",
)
_SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "max_prompt_tokens",
    "max_response_tokens",
    "num_samples",
    "sampling_seed",
)


class XlamReceiptValidationError(ValueError):
    """Raised by the strict receipt validator with actionable diagnostics."""

    def __init__(self, diagnostics: Iterable[str]):
        unique = tuple(dict.fromkeys(str(item) for item in diagnostics if str(item)))
        self.diagnostics = unique or ("receipt validation failed",)
        super().__init__("; ".join(self.diagnostics))


@dataclass(frozen=True)
class _Row:
    identity: tuple[int, str, str]
    score: float


@dataclass(frozen=True)
class _Receipt:
    role: str
    source_kind: str
    evaluated_path: str
    tokenizer_model: str
    seed: int
    rows: tuple[_Row, ...]
    mean_strict_reward: float
    perfect_call_rate: float
    perfect_calls: int
    provenance: Mapping[str, Any]
    sampling: Mapping[str, Any]
    tracking_ids: Mapping[str, str]


def _finite_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise XlamReceiptValidationError((f"{name} must be a finite real number",))
    result = float(value)
    if not math.isfinite(result):
        raise XlamReceiptValidationError((f"{name} must be finite",))
    return result


def _integer(name: str, value: Any, *, minimum: int = 0) -> int:
    result = _finite_real(name, value)
    if not result.is_integer() or int(result) < minimum:
        raise XlamReceiptValidationError((f"{name} must be an integer >= {minimum}",))
    return int(result)


def _string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise XlamReceiptValidationError((f"{name} must be a non-empty string",))
    return value


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise XlamReceiptValidationError((f"{name} must be a mapping",))
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _read_receipt(source: Mapping[str, Any] | Path | str) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if not isinstance(source, (Path, str)):
        raise XlamReceiptValidationError(("receipt must be a mapping or JSON path",))
    try:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise XlamReceiptValidationError((f"could not read receipt: {exc}",)) from exc
    return _mapping("receipt", payload)


def _containers(receipt: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    values: list[Mapping[str, Any]] = []
    for key in ("provenance", "tracking_receipts"):
        value = receipt.get(key)
        if isinstance(value, Mapping):
            values.append(value)
            nested_tracking = value.get("tracking_receipts")
            if isinstance(nested_tracking, Mapping):
                values.append(nested_tracking)
    values.append(receipt)
    return tuple(values)


def _first_value(
    receipt: Mapping[str, Any], aliases: Sequence[str], *, mapping_only: bool = False
) -> Any:
    for container in _containers(receipt):
        for alias in aliases:
            if alias in container:
                value = container[alias]
                if not mapping_only or isinstance(value, Mapping):
                    return value
    return _MISSING


def _artifact(receipt: Mapping[str, Any], name: str) -> Any:
    return _first_value(
        receipt,
        (name, f"{name}_receipt", f"{name}_run"),
        mapping_only=True,
    )


def _claim_diagnostics(receipt: Mapping[str, Any], role: str) -> list[str]:
    diagnostics: list[str] = []
    for key in ("claim_scope", "analysis_scope", "evaluation_scope", "scope"):
        value = receipt.get(key, _MISSING)
        if value is _MISSING:
            continue
        if not isinstance(value, str):
            diagnostics.append(f"{role}: {key} must be a string")
            continue
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"portfolio", "all_company", "company", "production"}:
            diagnostics.append(f"{role}: forbidden {key} claim scope {value!r}")
        elif key in {"claim_scope", "evaluation_scope", "scope"} and normalized in {
            "heldout",
            "held_out",
        }:
            diagnostics.append(f"{role}: ambiguous non-xlam held-out claim scope")
        elif key in {"claim_scope", "evaluation_scope", "scope"} and normalized not in {
            *_ALLOWED_CLAIM_SCOPES,
            "",
        }:
            diagnostics.append(f"{role}: unsupported claim scope {value!r}")

    for key in ("portfolio_claim", "company_claim", "all_company_claim"):
        if receipt.get(key) is True:
            diagnostics.append(f"{role}: {key} must not be true for xLAM analysis")

    for key in ("claim", "claims"):
        value = receipt.get(key, _MISSING)
        if value is _MISSING:
            continue
        text = json.dumps(value, sort_keys=True).lower()
        if "portfolio" in text or "all compan" in text or "production readiness" in text:
            diagnostics.append(f"{role}: forbidden portfolio/company claim text")
        elif "heldout" in text or "held-out" in text:
            diagnostics.append(f"{role}: held-out claim text is not admissible")
    return diagnostics


def _normalize_provenance(
    receipt: Mapping[str, Any], role: str, diagnostics: list[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    provenance: dict[str, Any] = {}
    for field in _INVARIANT_PROVENANCE_FIELDS:
        aliases = {
            "model_id": ("model_id", "base_model_id", "model"),
            "model_revision": (
                "model_revision",
                "base_model_revision",
                "checkpoint_base_revision",
            ),
            "tokenizer_revision": ("tokenizer_revision", "tokenizer_sha256"),
            "dataset_id": ("dataset_id", "dataset"),
            "dataset_revision": ("dataset_revision", "dataset_sha256"),
            "split_manifest_sha256": (
                "split_manifest_sha256",
                "split_manifest_hash",
            ),
            "task_id_manifest_sha256": (
                "task_id_manifest_sha256",
                "task_manifest_sha256",
            ),
            "verifier_revision": ("verifier_revision", "verifier_hash"),
            "container_digest": ("container_digest", "environment_digest"),
            "decontamination_receipt": (
                "decontamination_receipt",
                "decontamination_receipt_hash",
            ),
        }[field]
        value = _first_value(receipt, aliases)
        if (
            value is _MISSING
            or value is None
            or value == ""
            or (isinstance(value, (Mapping, Sequence)) and not value)
        ):
            diagnostics.append(f"{role}: missing provenance field {field}")
        elif isinstance(value, (str, int, float, bool, Mapping, Sequence)):
            provenance[field] = value
        else:
            diagnostics.append(f"{role}: invalid provenance field {field}")

    sampling = _first_value(receipt, ("sampling", "evaluation_sampling", "sampler"))
    if not isinstance(sampling, Mapping):
        diagnostics.append(f"{role}: missing sampling provenance receipt")
        sampling = {}
    normalized_sampling: dict[str, Any] = {}
    for field in _SAMPLING_FIELDS:
        value = sampling.get(field, _MISSING)
        if value is _MISSING:
            diagnostics.append(f"{role}: missing sampling field {field}")
            continue
        if field in {"max_prompt_tokens", "max_response_tokens", "num_samples"}:
            try:
                normalized_sampling[field] = _integer(f"{role}.sampling.{field}", value, minimum=1)
            except XlamReceiptValidationError as exc:
                diagnostics.extend(exc.diagnostics)
        elif field == "sampling_seed":
            if isinstance(value, bool) or not isinstance(value, (int, str)):
                diagnostics.append(f"{role}: sampling_seed must be an int or recorded string")
            elif isinstance(value, str) and not value.strip():
                diagnostics.append(f"{role}: sampling_seed must be non-empty")
            else:
                normalized_sampling[field] = value
        else:
            try:
                normalized_sampling[field] = _finite_real(f"{role}.sampling.{field}", value)
            except XlamReceiptValidationError as exc:
                diagnostics.extend(exc.diagnostics)

    tracking_ids: dict[str, str] = {}
    wandb = _artifact(receipt, "wandb")
    if not isinstance(wandb, Mapping):
        diagnostics.append(f"{role}: missing W&B receipt")
    else:
        for field in ("run_id", "url", "mode"):
            value = wandb.get(field, _MISSING)
            if value is _MISSING:
                diagnostics.append(f"{role}: W&B receipt missing {field}")
            elif not isinstance(value, str) or not value.strip():
                diagnostics.append(f"{role}: W&B receipt {field} must be non-empty")
            else:
                tracking_ids[f"wandb_{field}"] = value
        if isinstance(wandb.get("mode"), str) and wandb["mode"].lower() != "online":
            diagnostics.append(f"{role}: W&B receipt must record online mode")

    hf = _artifact(receipt, "hf")
    if not isinstance(hf, Mapping):
        diagnostics.append(f"{role}: missing Hugging Face receipt")
    else:
        for field_aliases, normalized_name in (
            (("repo", "repository", "url"), "repo"),
            (("commit", "revision", "sha"), "commit"),
            (("visibility",), "visibility"),
        ):
            value = next((hf.get(alias) for alias in field_aliases if alias in hf), _MISSING)
            if value is _MISSING:
                diagnostics.append(f"{role}: Hugging Face receipt missing {normalized_name}")
            elif not isinstance(value, str) or not value.strip():
                diagnostics.append(
                    f"{role}: Hugging Face receipt {normalized_name} must be non-empty"
                )
            else:
                tracking_ids[f"hf_{normalized_name}"] = value
        visibility = hf.get("visibility")
        if role == "trained" and isinstance(visibility, str) and visibility.lower() != "private":
            diagnostics.append("trained: Hugging Face checkpoint must be private")

    tinker = _artifact(receipt, "tinker")
    if not isinstance(tinker, Mapping):
        diagnostics.append(f"{role}: missing Tinker receipt")
    else:
        run_id = tinker.get("run_id", _MISSING)
        if not isinstance(run_id, str) or not run_id.strip():
            diagnostics.append(f"{role}: Tinker receipt missing run_id")
        else:
            tracking_ids["tinker_run_id"] = run_id
        status = tinker.get("status", _MISSING)
        if status is _MISSING:
            diagnostics.append(f"{role}: Tinker receipt missing status")
        elif not isinstance(status, str) or status.lower() not in _SUCCESSFUL_TINKER_STATUSES:
            diagnostics.append(f"{role}: Tinker receipt is not completed")

    return provenance, normalized_sampling, tracking_ids


def _normalize_receipt(source: Mapping[str, Any] | Path | str, role: str) -> _Receipt:
    receipt = _read_receipt(source)
    diagnostics: list[str] = _claim_diagnostics(receipt, role)

    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        diagnostics.append(f"{role}: schema_version must be {RECEIPT_SCHEMA_VERSION!r}")
    source_kind = receipt.get("source_kind")
    if role == "base" and source_kind != "base_model":
        diagnostics.append("base: source_kind must be 'base_model'")
    if role == "trained" and source_kind not in {"sampler_path", "trained_adapter", "adapter"}:
        diagnostics.append("trained: source_kind must identify a trained sampler/adapter")

    try:
        evaluated_path = _string(f"{role}.evaluated_path", receipt.get("evaluated_path"))
        tokenizer_model = _string(f"{role}.tokenizer_model", receipt.get("tokenizer_model"))
        seed = _integer(f"{role}.seed", receipt.get("seed"), minimum=0)
        examples = _integer(f"{role}.examples", receipt.get("examples"), minimum=1)
    except XlamReceiptValidationError as exc:
        diagnostics.extend(exc.diagnostics)
        evaluated_path = ""
        tokenizer_model = ""
        seed = 0
        examples = 0

    raw_rows = receipt.get("rows")
    if isinstance(raw_rows, (str, bytes, bytearray)) or not isinstance(raw_rows, Iterable):
        diagnostics.append(f"{role}: rows must be a non-empty iterable")
        raw_rows = ()
    rows: list[_Row] = []
    indices: set[int] = set()
    identities: set[tuple[int, str, str]] = set()
    content_identities: set[tuple[str, str]] = set()
    for position, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            diagnostics.append(f"{role}: row {position} must be a mapping")
            continue
        try:
            index = _integer(f"{role}.rows[{position}].index", raw_row.get("index"), minimum=0)
            prompt_hash = _string(
                f"{role}.rows[{position}].prompt_sha256", raw_row.get("prompt_sha256")
            )
            target_hash = _string(
                f"{role}.rows[{position}].target_sha256", raw_row.get("target_sha256")
            )
            score = _finite_real(f"{role}.rows[{position}].score", raw_row.get("score"))
        except XlamReceiptValidationError as exc:
            diagnostics.extend(exc.diagnostics)
            continue
        if not 0.0 <= score <= 1.0:
            diagnostics.append(f"{role}.rows[{position}].score must be in [0, 1]")
        if index in indices:
            diagnostics.append(f"{role}: duplicate row index at row {position}")
        identity = (index, prompt_hash, target_hash)
        if identity in identities:
            diagnostics.append(f"{role}: duplicate paired example identity at row {position}")
        content_identity = (prompt_hash, target_hash)
        if content_identity in content_identities:
            diagnostics.append(f"{role}: duplicate prompt/target identity at row {position}")
        indices.add(index)
        identities.add(identity)
        content_identities.add(content_identity)
        for optional in ("response_sha256",):
            if optional in raw_row and (
                not isinstance(raw_row[optional], str) or not raw_row[optional].strip()
            ):
                diagnostics.append(f"{role}.rows[{position}].{optional} must be non-empty")
        rows.append(_Row(identity=identity, score=score))

    if not rows:
        diagnostics.append(f"{role}: rows must be non-empty")
    if examples and examples != len(rows):
        diagnostics.append(f"{role}: examples does not match row count")

    mean_value = receipt.get("mean_strict_reward", _MISSING)
    rate_value = receipt.get("perfect_call_rate", _MISSING)
    try:
        mean_strict_reward = _finite_real(f"{role}.mean_strict_reward", mean_value)
        perfect_call_rate = _finite_real(f"{role}.perfect_call_rate", rate_value)
    except XlamReceiptValidationError as exc:
        diagnostics.extend(exc.diagnostics)
        mean_strict_reward = 0.0
        perfect_call_rate = 0.0
    perfect_calls = sum(row.score == 1.0 for row in rows)
    if rows:
        observed_mean = fmean(row.score for row in rows)
        observed_rate = perfect_calls / len(rows)
        if not _close(mean_strict_reward, observed_mean):
            diagnostics.append(f"{role}: mean_strict_reward does not match rows")
        if not _close(perfect_call_rate, observed_rate):
            diagnostics.append(f"{role}: perfect_call_rate does not match rows")
    if not 0.0 <= perfect_call_rate <= 1.0:
        diagnostics.append(f"{role}: perfect_call_rate must be in [0, 1]")

    provenance, sampling, tracking_ids = _normalize_provenance(receipt, role, diagnostics)
    if diagnostics:
        raise XlamReceiptValidationError(diagnostics)
    return _Receipt(
        role=role,
        source_kind=str(source_kind),
        evaluated_path=evaluated_path,
        tokenizer_model=tokenizer_model,
        seed=seed,
        rows=tuple(rows),
        mean_strict_reward=mean_strict_reward,
        perfect_call_rate=perfect_call_rate,
        perfect_calls=perfect_calls,
        provenance=provenance,
        sampling=sampling,
        tracking_ids=tracking_ids,
    )


def validate_xlam_receipt(
    source: Mapping[str, Any] | Path | str, *, role: str = "receipt"
) -> dict[str, Any]:
    """Validate one receipt and return a JSON-safe normalized summary.

    This strict helper raises :class:`XlamReceiptValidationError`; the public
    comparison function converts the same diagnostics into a blocked report.
    """

    normalized = _normalize_receipt(source, role)
    return {
        "role": normalized.role,
        "source_kind": normalized.source_kind,
        "evaluated_path": normalized.evaluated_path,
        "tokenizer_model": normalized.tokenizer_model,
        "seed": normalized.seed,
        "examples": len(normalized.rows),
        "mean_strict_reward": normalized.mean_strict_reward,
        "perfect_calls": normalized.perfect_calls,
        "perfect_call_rate": normalized.perfect_call_rate,
        "tracking_receipts_present": True,
    }


def _fingerprint(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _compare_receipt_provenance(base: _Receipt, trained: _Receipt) -> list[str]:
    diagnostics: list[str] = []
    if base.seed != trained.seed:
        diagnostics.append("revision drift: seed differs between base and trained receipts")
    if base.tokenizer_model != trained.tokenizer_model:
        diagnostics.append("revision drift: tokenizer_model differs")
    for field in _INVARIANT_PROVENANCE_FIELDS:
        if _fingerprint(base.provenance.get(field)) != _fingerprint(trained.provenance.get(field)):
            diagnostics.append(f"revision drift: provenance field {field} differs")
    for field in _SAMPLING_FIELDS:
        if _fingerprint(base.sampling.get(field)) != _fingerprint(trained.sampling.get(field)):
            diagnostics.append(f"revision drift: sampling field {field} differs")
    return diagnostics


def _blocked_report(diagnostics: Iterable[str]) -> dict[str, Any]:
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "blocked",
        "analysis_scope": "xlam_component_only",
        "claim_scope": "xlam_component_only",
        "portfolio_claim_permitted": False,
        "company_claim_permitted": False,
        "generalization_claim_permitted": False,
        "baseline_reference": {
            "perfect_calls": BASELINE_PERFECT_CALLS,
            "trials": BASELINE_TRIALS,
            "perfect_call_rate": BASELINE_RATE,
        },
        "diagnostics": list(dict.fromkeys(str(item) for item in diagnostics if str(item))),
        "comparison": None,
    }


def analyze_xlam_receipts(
    base_receipt: Mapping[str, Any] | Path | str,
    trained_receipt: Mapping[str, Any] | Path | str,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Analyze a paired xLAM comparison without making broader claims.

    Invalid, incomplete, unpaired, revision-drifted, or claim-overreaching
    inputs return a ``status='blocked'`` report with no partial statistics.
    """

    diagnostics: list[str] = []
    base: _Receipt | None = None
    trained: _Receipt | None = None
    for source, role in ((base_receipt, "base"), (trained_receipt, "trained")):
        try:
            normalized = _normalize_receipt(source, role)
        except XlamReceiptValidationError as exc:
            diagnostics.extend(exc.diagnostics)
        else:
            if role == "base":
                base = normalized
            else:
                trained = normalized
    if diagnostics or base is None or trained is None:
        return _blocked_report(diagnostics or ("receipt validation failed",))

    if len(base.rows) != BASELINE_TRIALS or len(trained.rows) != BASELINE_TRIALS:
        diagnostics.append("xLAM comparison requires exactly 100 rows in both receipts")
    if base.perfect_calls != BASELINE_PERFECT_CALLS:
        diagnostics.append("base receipt is not the required 7/100 reference")
    if base.source_kind != "base_model":
        diagnostics.append("base receipt is not a base-model evaluation")
    if trained.source_kind == "base_model":
        diagnostics.append("trained receipt must identify an adapter/sampler evaluation")

    base_keys = tuple(row.identity for row in base.rows)
    trained_keys = tuple(row.identity for row in trained.rows)
    base_key_set = set(base_keys)
    trained_key_set = set(trained_keys)
    if base_key_set != trained_key_set:
        missing = len(base_key_set - trained_key_set)
        extra = len(trained_key_set - base_key_set)
        diagnostics.append(
            f"unpaired rows: identity sets differ (missing={missing}, extra={extra})"
        )
    diagnostics.extend(_compare_receipt_provenance(base, trained))
    if diagnostics:
        return _blocked_report(diagnostics)

    trained_by_identity = {row.identity: row.score for row in trained.rows}
    base_scores = tuple(row.score for row in base.rows)
    trained_scores = tuple(trained_by_identity[row.identity] for row in base.rows)
    base_binary = tuple(int(score == 1.0) for score in base_scores)
    trained_binary = tuple(int(score == 1.0) for score in trained_scores)
    try:
        base_wilson = wilson_interval(base.perfect_calls, len(base.rows))
        trained_perfect_calls = sum(trained_binary)
        trained_wilson = wilson_interval(trained_perfect_calls, len(trained.rows))
        b, c = mcnemar_discordant_counts(base_binary, trained_binary)
        mcnemar_pvalue = exact_mcnemar_two_sided(base_binary, trained_binary)
        newcombe = newcombe_paired_risk_difference_interval(base_binary, trained_binary)
        bootstrap = paired_bootstrap_mean_difference(
            base_scores,
            trained_scores,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    except StatisticsInputError as exc:
        return _blocked_report((f"statistics input validation failed: {exc}",))

    trained_rate = trained_perfect_calls / len(trained.rows)
    risk_difference = (b - c) / len(base.rows)
    rate_difference = trained_rate - base.perfect_call_rate
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "status": "admissible_xlam_component",
        "analysis_scope": "xlam_component_only",
        "claim_scope": "xlam_component_only",
        "portfolio_claim_permitted": False,
        "company_claim_permitted": False,
        "generalization_claim_permitted": False,
        "baseline_reference": {
            "perfect_calls": BASELINE_PERFECT_CALLS,
            "trials": BASELINE_TRIALS,
            "perfect_call_rate": BASELINE_RATE,
        },
        "provenance_match": True,
        "comparison": {
            "base": {
                "evaluated_path": base.evaluated_path,
                "perfect_calls": base.perfect_calls,
                "trials": len(base.rows),
                "perfect_call_rate": base.perfect_call_rate,
                "mean_strict_reward": base.mean_strict_reward,
                "wilson_interval_95": list(base_wilson),
            },
            "trained": {
                "evaluated_path": trained.evaluated_path,
                "perfect_calls": trained_perfect_calls,
                "trials": len(trained.rows),
                "perfect_call_rate": trained_rate,
                "mean_strict_reward": trained.mean_strict_reward,
                "wilson_interval_95": list(trained_wilson),
            },
            "paired": {
                "base_fail_trained_success": b,
                "base_success_trained_fail": c,
                "risk_difference_trained_minus_base": risk_difference,
                "perfect_call_rate_difference": rate_difference,
                "exact_mcnemar_two_sided_p": mcnemar_pvalue,
                "newcombe_interval_95": list(newcombe),
                "mean_strict_reward_bootstrap": bootstrap.as_dict(),
            },
            "improvement_vs_7_of_100": {
                "point_estimate_exceeds_base": trained_perfect_calls > BASELINE_PERFECT_CALLS,
                "perfect_call_rate_delta": rate_difference,
                "mean_strict_reward_delta": bootstrap.estimate,
                "newcombe_lower_bound_above_zero": newcombe[0] > 0.0,
            },
        },
        "tracking_receipts": {
            "base": dict(base.tracking_ids),
            "trained": dict(trained.tracking_ids),
        },
        "diagnostics": [],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_receipt", type=Path)
    parser.add_argument("trained_receipt", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args(argv)
    report = analyze_xlam_receipts(
        args.base_receipt,
        args.trained_receipt,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "admissible_xlam_component" else 2


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "BASELINE_PERFECT_CALLS",
    "BASELINE_RATE",
    "BASELINE_TRIALS",
    "RECEIPT_SCHEMA_VERSION",
    "XlamReceiptValidationError",
    "analyze_xlam_receipts",
    "main",
    "validate_xlam_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
