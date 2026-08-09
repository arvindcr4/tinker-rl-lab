"""Offline validator for exact Pavlov T3 BrowserGym result receipts.

The validator consumes a JSON-compatible receipt; it never creates a browser,
contacts W&B/HF/Tinker, reads credentials, or infers a result from a URL.  A
receipt marked ``observed_result`` is accepted only when it carries all three
external evidence references, a native ``env.step`` success check, exact
state/action/artifact hashes, and reconciled cost metadata.  References are
checked syntactically only.  They are not network evidence until an external
owner separately verifies them.

``offline_result_fixture`` is deliberately marked schema-only.  It exercises
the same validator without pretending that its synthetic observations or
offline references are a BrowserGym run, a WebBench/E6 result, or portfolio
evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

try:  # Package import used by the flagship test suite.
    from . import pavlov_browsergym_adapter as t3
except ImportError:  # Direct execution from this directory remains offline-safe.
    import pavlov_browsergym_adapter as t3


SCHEMA_VERSION = "pavlov-browsergym-t3-result-receipt-v1"
SUITE_ID = t3.SUITE_ID
SUITE_ROLE = t3.SUITE_ROLE
BENCHMARK_NAME = t3.BENCHMARK_NAME
DATASET_ID = t3.DATASET_ID
PINNED_DATASET_REVISION = t3.PINNED_DATASET_REVISION
PINNED_ENVIRONMENT_REVISION = t3.PINNED_ENVIRONMENT_REVISION
E6_SUITE_ID = t3.E6_SUITE_ID

RECEIPT_KIND_OFFLINE = "offline_synthetic_receipt"
RECEIPT_KIND_OBSERVED = "observed_result"
ALLOWED_RECEIPT_KINDS = frozenset((RECEIPT_KIND_OFFLINE, RECEIPT_KIND_OBSERVED))
EVIDENCE_STATUS_OFFLINE = "OFFLINE_SYNTHETIC_RECEIPT"
EVIDENCE_STATUS_OBSERVED = "OBSERVED_T3_RESULT_RECEIPT"
EPISODE_STATUS_OBSERVED = "observed"
EPISODE_EVIDENCE_STATUS_OBSERVED = "OBSERVED_T3_EPISODE"
CLAIM_BOUNDARY = "T3_RESULT_RECEIPT_ONLY"
NATIVE_VERIFIER_NAME = "browsergym.native_success"
NATIVE_VERIFIER_REVISION = "browsergym-env-step-success-v1"
NATIVE_VERIFIER_SOURCE = "browsergym.env.step"
OFFLINE_NATIVE_SOURCE = "offline_fixture"
MAX_OPERATIONAL_CAP_USD = Decimal("16.50")

EVIDENCE_KINDS = ("wandb", "tinker", "hf")
WANDB_REQUIRED_METRICS = (
    "train/browser_success_rate",
    "train/browser_reward_mean",
    "train/browser_action_count_mean",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,127}$")
_SECRET_REFERENCE_RE = re.compile(
    r"(?:[?&](?:token|api[_-]?key|access[_-]?token|password|secret)=|"
    r"(?:token|api[_-]?key|access[_-]?token|password|secret)\s*[:=])",
    re.IGNORECASE,
)

REQUIRED_FIELDS = (
    "schema_version",
    "receipt_kind",
    "result_status",
    "suite_id",
    "suite_role",
    "benchmark",
    "dataset_id",
    "dataset_revision",
    "environment_revision",
    "stateful",
    "artifact_or_side_effect",
    "task_id",
    "episode_hash",
    "episode",
    "native_verifier",
    "evidence",
    "cost",
    "e6_suite_id",
    "e6_substitute",
    "portfolio_evidence",
    "main_track_claim_allowed",
    "claim_boundary",
    "evidence_status",
    "receipt_hash",
)


class ReceiptSchemaError(ValueError):
    """Raised by strict constructors/helpers for malformed receipt metadata."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ReceiptSchemaError(f"value is not canonical JSON: {exc}") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not _is_mapping(value):
        raise ReceiptSchemaError(f"{label} must be an object")
    copied = dict(value)
    try:
        t3.assert_secret_free(copied)
    except t3.SecretMaterialError as exc:
        raise ReceiptSchemaError(str(exc)) from exc
    return copied


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReceiptSchemaError(f"{label} must be a lowercase SHA-256 digest")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiptSchemaError(f"{label} must be a non-empty string")
    return value


def _safe_reference(value: Any, label: str, *, observed: bool) -> None:
    reference = _nonempty_string(value, label)
    if not observed:
        return
    if reference.startswith("offline://"):
        raise ReceiptSchemaError(f"{label} cannot be an offline placeholder")
    if "@" in reference or _SECRET_REFERENCE_RE.search(reference):
        raise ReceiptSchemaError(f"{label} contains credential-like URL material")


def _safe_run_id(value: Any, label: str, *, observed: bool) -> None:
    _safe_reference(value, label, observed=observed)
    if observed and (
        not isinstance(value, str) or _SAFE_RUN_ID_RE.fullmatch(value) is None
    ):
        raise ReceiptSchemaError(f"{label} is not a safe run identifier")


def _finite_decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ReceiptSchemaError(f"{label} must be a finite non-negative number")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ReceiptSchemaError(f"{label} must be a finite non-negative number") from exc
    if not number.is_finite() or number < 0:
        raise ReceiptSchemaError(f"{label} must be a finite non-negative number")
    return number


def _rehashed(value: Mapping[str, Any], field: str) -> tuple[str, str]:
    supplied = value.get(field)
    if not isinstance(supplied, str) or _SHA256_RE.fullmatch(supplied) is None:
        return "", f"{field} must be a lowercase SHA-256 digest"
    try:
        expected = sha256_json({key: item for key, item in value.items() if key != field})
    except ReceiptSchemaError as exc:
        return supplied, f"{field} payload cannot be hashed: {exc}"
    if supplied != expected:
        return supplied, f"{field} does not match the canonical receipt payload"
    return supplied, ""


@dataclass(frozen=True)
class ReceiptValidationResult:
    """Validation outcome; ``ok`` means structure only, never a scientific claim."""

    ok: bool
    errors: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    receipt_hash: str = ""
    episode_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "metrics": dict(self.metrics),
            "receipt_hash": self.receipt_hash,
            "episode_hash": self.episode_hash,
            "e6_substitute": False,
            "portfolio_evidence": False,
            "main_track_claim_allowed": False,
            "claim_boundary": CLAIM_BOUNDARY,
        }


def _validate_episode(
    episode: Mapping[str, Any],
    *,
    receipt_kind: str,
    errors: list[str],
) -> tuple[str, Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Validate the adapter trace and return task/terminal/action context."""

    try:
        episode_data = _mapping(episode, "episode")
    except ReceiptSchemaError as exc:
        errors.append(str(exc))
        return "", {}, [], []
    supplied_hash, hash_error = _rehashed(episode_data, "trace_hash")
    if hash_error:
        errors.append(f"episode: {hash_error}")

    if episode_data.get("schema_version") != t3.SCHEMA_VERSION:
        errors.append("episode schema is not the T3 adapter schema")
    if episode_data.get("suite_id") != SUITE_ID:
        errors.append("episode suite_id must be browsergym_train")
    if episode_data.get("suite_role") != SUITE_ROLE:
        errors.append("episode suite_role must be train")
    if episode_data.get("benchmark") != BENCHMARK_NAME:
        errors.append("episode benchmark must be browsergym")
    if episode_data.get("dataset_id") != DATASET_ID:
        errors.append("episode dataset_id is not BrowserGym MiniWoB")
    if episode_data.get("dataset_revision") != PINNED_DATASET_REVISION:
        errors.append("episode dataset_revision is not pinned")
    if episode_data.get("environment_revision") != PINNED_ENVIRONMENT_REVISION:
        errors.append("episode environment_revision is not pinned")
    if episode_data.get("stateful") is not True:
        errors.append("T3 episode must be stateful")
    if episode_data.get("artifact_or_side_effect") is not True:
        errors.append("T3 episode must carry artifact_or_side_effect evidence")
    if episode_data.get("e6_substitute") is not False:
        errors.append("BrowserGym episode cannot substitute for E6/WebBench")
    if episode_data.get("portfolio_evidence") is not False:
        errors.append("T3 episode cannot claim portfolio evidence")

    episode_status = episode_data.get("status")
    if receipt_kind == RECEIPT_KIND_OFFLINE:
        if episode_status not in t3.ALLOWED_TRACE_STATUSES:
            errors.append("offline receipt episode must remain fixture/preflight status")
    elif episode_status != EPISODE_STATUS_OBSERVED:
        errors.append("observed receipt episode must have status=observed")
    if receipt_kind == RECEIPT_KIND_OBSERVED:
        if episode_data.get("evidence_status") != EPISODE_EVIDENCE_STATUS_OBSERVED:
            errors.append("observed episode evidence_status is not marked observed")

    task_id = ""
    terminal_data: Mapping[str, Any] = {}
    observations_data: list[Mapping[str, Any]] = []
    actions_data: list[Mapping[str, Any]] = []

    # The adapter verifier is intentionally offline-only.  Validate the same
    # state/action/artifact core by normalizing only its non-evidence labels;
    # the original episode hash above still guards the supplied observed trace.
    adapter_candidate = copy.deepcopy(episode_data)
    adapter_candidate["status"] = "offline_fixture"
    adapter_candidate["evidence_status"] = t3.EVIDENCE_STATUS
    adapter_candidate["claim_boundary"] = t3.CLAIM_BOUNDARY
    adapter_candidate["e6_substitute"] = False
    adapter_candidate["portfolio_evidence"] = False
    try:
        adapter_candidate["trace_hash"] = sha256_json(
            {key: item for key, item in adapter_candidate.items() if key != "trace_hash"}
        )
    except ReceiptSchemaError as exc:
        errors.append(f"episode payload cannot be hashed: {exc}")
        return task_id, terminal_data, observations_data, actions_data
    adapter_result = t3.verify_episode(adapter_candidate)
    if not adapter_result.ok:
        errors.extend(f"episode: {error}" for error in adapter_result.errors)

    task = episode_data.get("task")
    task_data = _mapping(task, "episode.task") if _is_mapping(task) else {}
    task_id = task_data.get("task_id", "")
    if not isinstance(task_id, str) or not task_id:
        errors.append("episode.task.task_id is required")

    observations = episode_data.get("observations")
    actions = episode_data.get("actions")
    if not isinstance(observations, list):
        errors.append("episode.observations must be a list")
        observations_data: list[Mapping[str, Any]] = []
    else:
        observations_data = [item for item in observations if _is_mapping(item)]
    if not isinstance(actions, list):
        errors.append("episode.actions must be a list")
        actions_data: list[Mapping[str, Any]] = []
    else:
        actions_data = [item for item in actions if _is_mapping(item)]
    terminal = episode_data.get("terminal")
    terminal_data = _mapping(terminal, "episode.terminal") if _is_mapping(terminal) else {}
    if not _is_mapping(terminal):
        errors.append("episode.terminal must be an object")
    return task_id, terminal_data, observations_data, actions_data


def _validate_native_verifier(
    native: Any,
    *,
    receipt_kind: str,
    task_id: str,
    episode_hash: str,
    terminal: Mapping[str, Any],
    action_count: int,
    errors: list[str],
) -> bool:
    if not _is_mapping(native):
        errors.append("native_verifier must be an object")
        return False
    try:
        data = _mapping(native, "native_verifier")
    except ReceiptSchemaError as exc:
        errors.append(str(exc))
        return False
    required = (
        "name",
        "revision",
        "source",
        "checked",
        "success",
        "task_id",
        "episode_hash",
        "final_state_hash",
    )
    for key in required:
        if key not in data:
            errors.append(f"native_verifier missing {key}")
    if data.get("name") != NATIVE_VERIFIER_NAME:
        errors.append("native_verifier name is not the BrowserGym success verifier")
    if data.get("revision") != NATIVE_VERIFIER_REVISION:
        errors.append("native_verifier revision is not pinned")
    expected_source = (
        NATIVE_VERIFIER_SOURCE
        if receipt_kind == RECEIPT_KIND_OBSERVED
        else OFFLINE_NATIVE_SOURCE
    )
    if data.get("source") != expected_source:
        errors.append(f"native_verifier source must be {expected_source}")
    if not isinstance(data.get("checked"), bool):
        errors.append("native_verifier.checked must be boolean")
    if receipt_kind == RECEIPT_KIND_OBSERVED and data.get("checked") is not True:
        errors.append("observed result requires a checked native env.step verifier")
    if receipt_kind == RECEIPT_KIND_OFFLINE and data.get("checked") is not False:
        errors.append("offline fixture cannot claim a native BrowserGym check")
    if not isinstance(data.get("success"), bool):
        errors.append("native_verifier.success must be boolean")
    if data.get("task_id") != task_id:
        errors.append("native_verifier task_id does not match episode task")
    if data.get("episode_hash") != episode_hash:
        errors.append("native_verifier episode_hash does not match episode trace_hash")
    final_state_hash = terminal.get("final_state_hash")
    try:
        _digest(data.get("final_state_hash"), "native_verifier.final_state_hash")
    except ReceiptSchemaError as exc:
        errors.append(str(exc))
    if data.get("final_state_hash") != final_state_hash:
        errors.append("native_verifier final_state_hash does not match terminal")
    task_success = terminal.get("task_success")
    if not isinstance(task_success, bool):
        errors.append("episode terminal.task_success must be boolean")
    elif data.get("success") != task_success:
        errors.append("native_verifier success disagrees with terminal.task_success")
    if receipt_kind == RECEIPT_KIND_OBSERVED:
        if data.get("action_count") != action_count:
            errors.append("native_verifier action_count does not match episode actions")
    return bool(data.get("checked") is True and data.get("success") is True)


def _validate_evidence(
    evidence: Any,
    *,
    receipt_kind: str,
    errors: list[str],
) -> dict[str, bool]:
    if not _is_mapping(evidence):
        errors.append("evidence must be an object")
        return {name: False for name in EVIDENCE_KINDS}
    try:
        evidence_data = _mapping(evidence, "evidence")
    except ReceiptSchemaError as exc:
        errors.append(str(exc))
        return {name: False for name in EVIDENCE_KINDS}
    observed_flags: dict[str, bool] = {}
    for name in EVIDENCE_KINDS:
        item = evidence_data.get(name)
        if not _is_mapping(item):
            errors.append(f"evidence.{name} must be an object")
            observed_flags[name] = False
            continue
        try:
            data = _mapping(item, f"evidence.{name}")
        except ReceiptSchemaError as exc:
            errors.append(str(exc))
            observed_flags[name] = False
            continue
        if not isinstance(data.get("observed"), bool):
            errors.append(f"evidence.{name}.observed must be boolean")
            observed = False
        else:
            observed = data["observed"]
        observed_flags[name] = observed
        if receipt_kind == RECEIPT_KIND_OBSERVED and not observed:
            errors.append(f"observed result is missing {name} evidence")
        if receipt_kind == RECEIPT_KIND_OFFLINE and observed:
            errors.append(f"offline fixture cannot claim observed {name} evidence")

        if name == "wandb":
            fields = ("run_id", "url", "project", "metrics")
            for field_name in fields:
                if field_name not in data:
                    errors.append(f"evidence.wandb missing {field_name}")
            try:
                _safe_run_id(data.get("run_id"), "evidence.wandb.run_id", observed=observed)
                _safe_reference(data.get("url"), "evidence.wandb.url", observed=observed)
                _safe_reference(data.get("project"), "evidence.wandb.project", observed=observed)
                if observed and (
                    not isinstance(data.get("url"), str)
                    or not data["url"].startswith("https://")
                    or "wandb" not in data["url"].lower()
                ):
                    raise ReceiptSchemaError("evidence.wandb.url is not a W&B HTTPS reference")
            except ReceiptSchemaError as exc:
                errors.append(str(exc))
            metrics = data.get("metrics")
            if not _is_mapping(metrics):
                errors.append("evidence.wandb.metrics must be an object")
            elif observed:
                for metric in WANDB_REQUIRED_METRICS:
                    if metric not in metrics:
                        errors.append(f"evidence.wandb missing metric {metric}")
                    else:
                        try:
                            _finite_decimal(metrics[metric], f"evidence.wandb.{metric}")
                        except ReceiptSchemaError as exc:
                            errors.append(str(exc))
            continue

        if name == "tinker":
            fields = ("run_id", "config_hash", "sampler_checkpoint", "final_checkpoint")
            for field_name in fields:
                if field_name not in data:
                    errors.append(f"evidence.tinker missing {field_name}")
            try:
                _safe_run_id(data.get("run_id"), "evidence.tinker.run_id", observed=observed)
                _safe_reference(
                    data.get("sampler_checkpoint"),
                    "evidence.tinker.sampler_checkpoint",
                    observed=observed,
                )
                _safe_reference(
                    data.get("final_checkpoint"),
                    "evidence.tinker.final_checkpoint",
                    observed=observed,
                )
            except ReceiptSchemaError as exc:
                errors.append(str(exc))
            if observed:
                try:
                    _digest(data.get("config_hash"), "evidence.tinker.config_hash")
                except ReceiptSchemaError as exc:
                    errors.append(str(exc))
            continue

        fields = ("repo_id", "revision", "checkpoint", "exported")
        for field_name in fields:
            if field_name not in data:
                errors.append(f"evidence.hf missing {field_name}")
        try:
            _safe_reference(data.get("repo_id"), "evidence.hf.repo_id", observed=observed)
            _safe_reference(data.get("revision"), "evidence.hf.revision", observed=observed)
            _safe_reference(data.get("checkpoint"), "evidence.hf.checkpoint", observed=observed)
        except ReceiptSchemaError as exc:
            errors.append(str(exc))
        if not isinstance(data.get("exported"), bool):
            errors.append("evidence.hf.exported must be boolean")
        if observed:
            revision = data.get("revision")
            if not isinstance(revision, str) or _GIT_SHA_RE.fullmatch(revision) is None:
                errors.append("observed HF evidence requires a 40-hex revision")
            if data.get("exported") is not True:
                errors.append("observed HF evidence requires exported=true")
            try:
                _digest(data.get("checkpoint_hash"), "evidence.hf.checkpoint_hash")
            except ReceiptSchemaError as exc:
                errors.append(str(exc))
    return observed_flags


def _validate_cost(cost: Any, *, receipt_kind: str, errors: list[str]) -> bool:
    if not _is_mapping(cost):
        errors.append("cost must be an object")
        return False
    try:
        data = _mapping(cost, "cost")
    except ReceiptSchemaError as exc:
        errors.append(str(exc))
        return False
    if data.get("currency") != "USD":
        errors.append("cost currency must be USD")
    component_names = ("prompt_usd", "sampling_usd", "training_usd", "other_usd")
    values: dict[str, Decimal] = {}
    for name in (*component_names, "total_usd", "charged_usd", "cap_usd"):
        try:
            values[name] = _finite_decimal(data.get(name), f"cost.{name}")
        except ReceiptSchemaError as exc:
            errors.append(str(exc))
    expected_total = sum((values.get(name, Decimal("0")) for name in component_names), Decimal("0"))
    if "total_usd" in values and values["total_usd"] != expected_total:
        errors.append("cost.total_usd does not equal component sum")
    if "cap_usd" in values:
        if values["cap_usd"] <= Decimal("0"):
            errors.append("cost.cap_usd must be positive")
        if values["cap_usd"] > MAX_OPERATIONAL_CAP_USD:
            errors.append("cost.cap_usd exceeds the operational cap")
    within_cap = values.get("charged_usd", Decimal("-1")) <= values.get(
        "cap_usd", Decimal("-2")
    )
    if not within_cap:
        errors.append("cost charged_usd exceeds cap_usd")
    if data.get("within_cap") is not within_cap:
        errors.append("cost.within_cap disagrees with charged_usd and cap_usd")
    if not isinstance(data.get("charged"), bool):
        errors.append("cost.charged must be boolean")
    if receipt_kind == RECEIPT_KIND_OFFLINE and data.get("charged") is not False:
        errors.append("offline fixture cannot claim charged paid work")
    if receipt_kind == RECEIPT_KIND_OFFLINE and values.get("charged_usd", Decimal("1")) != 0:
        errors.append("offline fixture charged_usd must be zero")
    return bool(within_cap and not any(error.startswith("cost.") for error in errors))


def validate_receipt(receipt: Mapping[str, Any]) -> ReceiptValidationResult:
    """Validate a T3 receipt without treating validation as portfolio evidence."""

    errors: list[str] = []
    receipt_hash = ""
    episode_hash = ""
    try:
        t3.assert_secret_free(receipt)
    except t3.SecretMaterialError as exc:
        errors.append(str(exc))
    if not _is_mapping(receipt):
        return ReceiptValidationResult(False, ("receipt must be an object",), {})
    data = dict(receipt)
    for key in REQUIRED_FIELDS:
        if key not in data:
            errors.append(f"missing required receipt field: {key}")

    receipt_hash, hash_error = _rehashed(data, "receipt_hash")
    if hash_error:
        errors.append(hash_error)
    kind = data.get("receipt_kind")
    if kind not in ALLOWED_RECEIPT_KINDS:
        errors.append("receipt_kind must be offline_synthetic_receipt or observed_result")
        kind = RECEIPT_KIND_OFFLINE
    expected_status = (
        EVIDENCE_STATUS_OBSERVED
        if kind == RECEIPT_KIND_OBSERVED
        else EVIDENCE_STATUS_OFFLINE
    )
    if data.get("evidence_status") != expected_status:
        errors.append(f"evidence_status must be {expected_status}")
    expected_result_status = (
        "OBSERVED_T3_RESULT" if kind == RECEIPT_KIND_OBSERVED else "OFFLINE_SCHEMA_ONLY"
    )
    if data.get("result_status") != expected_result_status:
        errors.append(f"result_status must be {expected_result_status}")
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version is not the T3 result receipt schema")
    if data.get("suite_id") != SUITE_ID:
        errors.append("suite_id must be browsergym_train")
    if data.get("suite_role") != SUITE_ROLE:
        errors.append("suite_role must be train")
    if data.get("benchmark") != BENCHMARK_NAME:
        errors.append("benchmark must be browsergym")
    if data.get("dataset_id") != DATASET_ID:
        errors.append("dataset_id must be browsergym/miniwob")
    if data.get("dataset_revision") != PINNED_DATASET_REVISION:
        errors.append("receipt dataset_revision is not pinned")
    if data.get("environment_revision") != PINNED_ENVIRONMENT_REVISION:
        errors.append("receipt environment_revision is not pinned")
    if data.get("stateful") is not True:
        errors.append("T3 result must be stateful")
    if data.get("artifact_or_side_effect") is not True:
        errors.append("T3 result must include artifact_or_side_effect evidence")
    if data.get("e6_suite_id") != E6_SUITE_ID:
        errors.append("e6_suite_id must explicitly identify webbench_eval")
    if data.get("e6_substitute") is not False:
        errors.append("BrowserGym result cannot substitute for E6 webbench_eval")
    if data.get("portfolio_evidence") is not False:
        errors.append("one T3 result receipt cannot claim portfolio evidence")
    if data.get("main_track_claim_allowed") is not False:
        errors.append("receipt validator does not authorize main-track claims")
    if data.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("claim_boundary must remain T3-result-receipt-only")

    episode = data.get("episode")
    task_id, terminal, observations, actions = _validate_episode(
        episode if _is_mapping(episode) else {},
        receipt_kind=kind,
        errors=errors,
    )
    episode_hash = episode.get("trace_hash", "") if _is_mapping(episode) else ""
    if data.get("task_id") != task_id:
        errors.append("receipt task_id does not match episode task")
    if data.get("episode_hash") != episode_hash:
        errors.append("receipt episode_hash does not match episode trace_hash")

    native_checked_success = _validate_native_verifier(
        data.get("native_verifier"),
        receipt_kind=kind,
        task_id=task_id,
        episode_hash=episode_hash,
        terminal=terminal,
        action_count=len(actions),
        errors=errors,
    )
    observed_flags = _validate_evidence(
        data.get("evidence"), receipt_kind=kind, errors=errors
    )
    cost_within_cap = _validate_cost(data.get("cost"), receipt_kind=kind, errors=errors)
    metrics = {
        "receipt_kind": kind,
        "task_success": bool(terminal.get("task_success")),
        "native_verifier_checked_success": native_checked_success,
        "wandb_observed": observed_flags.get("wandb", False),
        "tinker_observed": observed_flags.get("tinker", False),
        "hf_observed": observed_flags.get("hf", False),
        "all_external_evidence_observed": all(observed_flags.values())
        if observed_flags
        else False,
        "cost_within_cap": cost_within_cap,
        "e6_substitute": False,
        "portfolio_evidence": False,
        "main_track_claim_allowed": False,
    }
    return ReceiptValidationResult(
        not errors,
        tuple(errors),
        metrics,
        receipt_hash,
        episode_hash,
    )


def _offline_evidence() -> dict[str, Any]:
    placeholder = "offline://not-observed"
    return {
        "wandb": {
            "observed": False,
            "run_id": placeholder,
            "url": placeholder,
            "project": "offline_fixture",
            "metrics": {},
        },
        "tinker": {
            "observed": False,
            "run_id": placeholder,
            "config_hash": "offline",
            "sampler_checkpoint": placeholder,
            "final_checkpoint": placeholder,
        },
        "hf": {
            "observed": False,
            "repo_id": placeholder,
            "revision": "offline",
            "checkpoint": placeholder,
            "exported": False,
        },
    }


def offline_result_fixture() -> dict[str, Any]:
    """Build a deterministic schema-only receipt; no live evidence is claimed."""

    episode = t3.offline_dry_run_fixture()
    task_id = episode["task"]["task_id"]
    terminal = episode["terminal"]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_kind": RECEIPT_KIND_OFFLINE,
        "result_status": "OFFLINE_SCHEMA_ONLY",
        "suite_id": SUITE_ID,
        "suite_role": SUITE_ROLE,
        "benchmark": BENCHMARK_NAME,
        "dataset_id": DATASET_ID,
        "dataset_revision": PINNED_DATASET_REVISION,
        "environment_revision": PINNED_ENVIRONMENT_REVISION,
        "stateful": True,
        "artifact_or_side_effect": True,
        "task_id": task_id,
        "episode_hash": episode["trace_hash"],
        "episode": episode,
        "native_verifier": {
            "name": NATIVE_VERIFIER_NAME,
            "revision": NATIVE_VERIFIER_REVISION,
            "source": OFFLINE_NATIVE_SOURCE,
            "checked": False,
            "success": bool(terminal["task_success"]),
            "task_id": task_id,
            "episode_hash": episode["trace_hash"],
            "final_state_hash": terminal["final_state_hash"],
        },
        "evidence": _offline_evidence(),
        "cost": {
            "currency": "USD",
            "prompt_usd": 0.0,
            "sampling_usd": 0.0,
            "training_usd": 0.0,
            "other_usd": 0.0,
            "total_usd": 0.0,
            "charged_usd": 0.0,
            "cap_usd": 16.50,
            "charged": False,
            "within_cap": True,
        },
        "e6_suite_id": E6_SUITE_ID,
        "e6_substitute": False,
        "portfolio_evidence": False,
        "main_track_claim_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": EVIDENCE_STATUS_OFFLINE,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _cli_payload() -> dict[str, Any]:
    fixture = offline_result_fixture()
    return {"receipt": fixture, "validation": validate_receipt(fixture).to_dict()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the offline receipt JSON")
    args = parser.parse_args(argv)
    payload = _cli_payload()
    if args.json:
        print(_canonical_json(payload))
    else:
        print(
            "T3 BrowserGym receipt schema: "
            f"{payload['validation']['ok']} "
            "(offline schema-only; E6/portfolio claims disabled)"
        )
    return 0 if payload["validation"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
