#!/usr/bin/env python3
"""Deterministic, offline state machine for the first paid xLAM smoke lifecycle."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import pavlov_xlam_smoke_config as smoke_config


SCHEMA_VERSION = "pavlov-tinker-run-monitor-v1"
REQUIRED_STEPS = tuple(smoke_config.EXPECTED_CHECKPOINT_STEPS)
REQUIRED_STAGES = tuple(smoke_config.REQUIRED_CHECKPOINT_STAGES)

STATE_SEQUENCE = (
    "wandb_sync",
    "tinker_identity",
    "train_step_0",
    "hf_initial_export",
    "train_step_5",
    "hf_periodic_export",
    "train_step_10",
    "hf_final_export",
    "terminal_receipt",
)

WANDB_URL_RE = re.compile(r"^https://wandb\.ai/.+/.+/runs/.+$")
HTTP_URL_RE = re.compile(r"^https://[^\s]+$")
SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
HF_REPO_RE = re.compile(r"^https://huggingface\.co/.+$")

WANDB_SUCCESS_STATES = {"finished", "completed"}
TINKER_SUCCESS_STATES = {"finished", "completed", "success", "succeeded"}
TINKER_RETRYABLE_STATES = {"pending", "failed_infrastructure", "failed"}
TINKER_NON_RETRYABLE_STATES = {"failed_validation", "failed_fatal"}
TINKER_TERMINAL_STATES = (
    TINKER_SUCCESS_STATES | TINKER_RETRYABLE_STATES | TINKER_NON_RETRYABLE_STATES
)
_TINER_PROGRESS_STATES = {"running", "queued", "starting", "active", "started"}

_PLACEHOLDER_WORDS = frozenset(
    {
        "",
        "missing",
        "none",
        "null",
        "unknown",
        "pending",
        "todo",
        "placeholder",
        "to_be_pinned_before_paid_runs",
    }
)


class TinkerRunMonitorError(ValueError):
    """Raised when the monitor input payload is blocked from launch."""


def _coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate and candidate.lower() not in _PLACEHOLDER_WORDS:
            return candidate
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.isdigit() and len(candidate) < 16:
            return int(candidate)
    return None


def _first_text(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in value:
            candidate = _coerce_text(value[key])
            if candidate is not None:
                return candidate
    return None


def _to_list_of_ints(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            return None
        steps: list[int] = []
        for token in tokens:
            step = _coerce_int(token)
            if step is None:
                return None
            if step < 0:
                return None
            steps.append(step)
        return sorted(set(steps))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        steps: list[int] = []
        for raw in value:
            step = _coerce_int(raw)
            if step is None or step < 0:
                return None
            steps.append(step)
        return sorted(set(steps))
    return None


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_payload(payload: Any, *, label: str, read_strings: bool = True) -> tuple[Any, list[str]]:
    if payload is None:
        return None, [f"{label} must be provided"]
    if read_strings and isinstance(payload, (str, Path)):
        try:
            return _load_json(payload), []
        except FileNotFoundError:
            return None, [f"{label} must point to an existing JSON file"]
        except (OSError, json.JSONDecodeError) as exc:
            return None, [f"{label} must be valid JSON: {type(exc).__name__}"]
    return payload, []


def _normalise_wandb_record(value: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record: dict[str, Any] = {}

    if not isinstance(value, Mapping):
        return record, ["wandb sync metadata must be a JSON object"]

    raw = value["run"] if isinstance(value.get("run"), Mapping) else value
    entity = _first_text(raw, "entity", "team", "user")
    project = _first_text(raw, "project", "entity_project", "wandb_project")
    run_id = _first_text(raw, "run_id", "id", "runId")
    run_url = _first_text(raw, "run_url", "url", "link")
    state = _first_text(raw, "state", "status")
    mode = _first_text(raw, "mode", "wandb_mode", "offline_mode")
    group = _first_text(raw, "group", "run_group")

    if not entity:
        errors.append("wandb entity is missing")
    if not project:
        errors.append("wandb project is missing")
    if not run_id:
        errors.append("wandb run_id is missing")
    if not run_url or not WANDB_URL_RE.fullmatch(run_url):
        errors.append("wandb run_url must be a valid W&B URL")
    if not state:
        errors.append("wandb state is missing")
    elif state.lower() not in WANDB_SUCCESS_STATES:
        errors.append("wandb state must be finished")
    if not mode:
        errors.append("wandb mode is missing")
    elif mode.lower() != "online":
        errors.append("wandb mode must be online")

    record.update(
        {
            "entity": entity,
            "project": project,
            "run_id": run_id,
            "run_url": run_url,
            "state": state,
            "mode": mode,
            "group": group,
            "created_at": _coerce_int(raw.get("created_at") or raw.get("created") ),
        }
    )
    return record, errors


def _normalise_checkpoints(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    checkpoints: list[dict[str, Any]] = []

    if isinstance(value, Mapping):
        if isinstance(value.get("checkpoints"), list):
            raw = value["checkpoints"]
        elif isinstance(value.get("entries"), list):
            raw = value["entries"]
        else:
            return checkpoints, ["checkpoint JSON must contain checkpoints list"]
    elif isinstance(value, list):
        raw = value
    else:
        return checkpoints, ["checkpoint JSON must be a list or object with checkpoints"]

    if not raw:
        return checkpoints, ["checkpoint JSON must contain initial, periodic, and final entries"]

    payload_run_id = _first_text(
        value,
        "tinker_run_id",
        "run_id",
        "tinkerRunId",
    ) if isinstance(value, Mapping) else None
    seen_stages: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f"checkpoint[{index}] must be a JSON object")
            continue
        stage = _first_text(item, "stage", "label")
        if stage is None:
            errors.append(f"checkpoint[{index}] stage is missing")
            continue
        stage = stage.lower()
        if stage not in REQUIRED_STAGES:
            errors.append(f"checkpoint[{index}] has invalid stage {stage!r}")
            continue
        if stage in seen_stages:
            errors.append(f"checkpoint stage {stage!r} appears more than once")
            continue

        step = _coerce_int(item.get("step"))
        if step is None:
            errors.append(f"checkpoint[{index}] step must be an integer")
            continue

        item_run_id = _first_text(item, "run_id", "tinker_run_id", "tinkerRunId")
        if item_run_id is None:
            item_run_id = payload_run_id

        if payload_run_id is not None and item_run_id is not None and item_run_id != payload_run_id:
            errors.append("checkpoint manifest run_id mismatch inside entries")

        checkpoints.append({
            "stage": stage,
            "step": step,
            "run_id": item_run_id,
        })
        seen_stages.add(stage)

    if not errors:
        observed = {entry["stage"] for entry in checkpoints}
        if observed != set(REQUIRED_STAGES):
            missing = sorted(set(REQUIRED_STAGES) - observed)
            errors.append("checkpoint JSON missing stages: " + ", ".join(missing))
        elif len(checkpoints) != len(REQUIRED_STAGES):
            errors.append("checkpoint JSON must contain exactly one initial, periodic, and final entry")

    return checkpoints, errors


def _normalise_hf_receipts(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    receipts: list[dict[str, Any]] = []

    if isinstance(value, Mapping):
        raw: list[Any] = []
        for item in value.values():
            if isinstance(item, list):
                raw.extend(item)
            else:
                raw.append(item)
    elif isinstance(value, list):
        raw = value
    else:
        return receipts, ["hf receipts must be a list or object with stages"]

    if not raw:
        return receipts, ["hf receipts must contain initial, periodic, and final entries"]

    seen_stages: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f"hf_receipts[{index}] must be a JSON object")
            continue
        stage = _first_text(item, "stage", "label")
        if stage is None:
            errors.append(f"hf_receipts[{index}] stage is missing")
            continue
        stage = stage.lower()
        if stage not in REQUIRED_STAGES:
            errors.append(f"hf_receipts[{index}] has invalid stage {stage!r}")
            continue
        if stage in seen_stages:
            errors.append(f"hf receipt for stage {stage!r} appears more than once")
            continue

        step = _coerce_int(item.get("step"))
        if step is None:
            errors.append(f"hf_receipts[{index}] step must be an integer")
            continue
        revision = _first_text(item, "revision", "commit", "sha")
        if revision is None or not SHA40_RE.fullmatch(revision):
            errors.append(f"hf_receipts[{index}] revision must be a 40-char hex commit")
            continue
        repo_url = _first_text(item, "repo_url", "repo", "repository")
        if repo_url is None or not HF_REPO_RE.fullmatch(repo_url):
            errors.append(f"hf_receipts[{index}] repo_url must be a Hugging Face URL")
            continue
        visibility = _first_text(item, "visibility")
        if visibility is None:
            errors.append(f"hf_receipts[{index}] visibility is missing")
            continue
        visibility = visibility.lower()
        if visibility not in {"public", "private"}:
            errors.append(f"hf_receipts[{index}] visibility must be public or private")
            continue
        safe_public = item.get("safe_public_artifact")
        if visibility == "public" and safe_public is not True:
            errors.append(f"hf_receipts[{index}] must set safe_public_artifact true for public visibility")
            continue

        receipts.append(
            {
                "stage": stage,
                "step": step,
                "revision": revision,
                "repo_url": repo_url,
                "visibility": visibility,
                "run_id": _first_text(item, "run_id", "tinker_run_id", "tinkerRunId"),
                "safe_public_artifact": safe_public,
            }
        )
        seen_stages.add(stage)

    if not errors:
        observed = {entry["stage"] for entry in receipts}
        if observed != set(REQUIRED_STAGES):
            missing = sorted(set(REQUIRED_STAGES) - observed)
            errors.append("hf receipts missing stages: " + ", ".join(missing))
        elif len(receipts) != len(REQUIRED_STAGES):
            errors.append("hf receipts must contain exactly one initial, periodic, and final entry")

    return receipts, errors


def _normalise_terminal_receipt(value: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record: dict[str, Any] = {}

    if not isinstance(value, Mapping):
        return record, ["terminal receipt must be a JSON object"]

    run_id = _first_text(value, "run_id", "tinker_run_id", "tinkerRunId")
    status = _first_text(value, "status", "state")
    exit_code = _coerce_int(value.get("exit_code"))
    steps = _to_list_of_ints(value.get("steps") or value.get("completed_steps"))

    if not run_id:
        errors.append("terminal receipt run_id is missing")
    if not status:
        errors.append("terminal receipt status is missing")
    elif status.lower() not in TINKER_SUCCESS_STATES:
        errors.append("terminal receipt status must be successful")
    if exit_code is None:
        errors.append("terminal receipt exit_code is missing")
    elif exit_code != 0:
        errors.append(f"terminal receipt exit_code must be 0, got {exit_code}")
    if steps is None:
        errors.append("terminal receipt must include completed steps")

    run_url = _first_text(value, "run_url", "url", "link")
    if run_url is not None and not HTTP_URL_RE.fullmatch(run_url):
        errors.append("terminal receipt run_url must be a valid URL")

    record.update(
        {
            "run_id": run_id,
            "status": status,
            "exit_code": exit_code,
            "run_url": run_url,
            "steps": steps or [],
        }
    )
    return record, errors


def _normalise_attempt(value: Mapping[str, Any], fallback_attempt: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    attempt_no = _coerce_int(value.get("attempt", value.get("attempt_no", value.get("index"))))
    if attempt_no is None:
        attempt_no = fallback_attempt

    status = _first_text(value, "status", "state")
    if not status:
        errors.append(f"attempt {attempt_no} status is missing")
    else:
        status = status.lower()
        if status not in TINKER_TERMINAL_STATES | _TINER_PROGRESS_STATES:
            errors.append(f"attempt {attempt_no} has invalid status {status!r}")

    run_id = _first_text(value, "run_id", "id", "runId", "tinker_run_id", "tinkerRunId")
    if not run_id:
        errors.append(f"attempt {attempt_no} run_id is missing")

    record = {
        "attempt": attempt_no,
        "status": status,
        "run_id": run_id,
        "parent_attempt": _coerce_int(value.get("parent_attempt")),
        "parent_run_id": _first_text(value, "parent_run_id", "previous_run_id", "previous_run"),
        "started_at": _coerce_int(value.get("started_at") or value.get("started") or value.get("timestamp") ),
        "steps": _to_list_of_ints(value.get("steps") or value.get("completed_steps") or value.get("step_history")),
    }

    run_url = _first_text(value, "run_url", "url", "link")
    if run_url is not None and not HTTP_URL_RE.fullmatch(run_url):
        errors.append(f"attempt {attempt_no} run_url must be a valid URL")
    if run_url is not None:
        record["run_url"] = run_url

    return record, errors


def _normalise_tinker_attempts(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if value is None:
        return [], ["tinker attempts must be provided"]

    if isinstance(value, Mapping):
        if "attempts" in value:
            raw = value["attempts"]
            if not isinstance(raw, list):
                return [], ["tinker attempts must be a list"]
        elif "run_id" in value:
            raw = [value]
        else:
            return [], ["tinker attempts must be a list or contain attempts/run_id"]
    elif isinstance(value, list):
        raw = value
    else:
        return [], ["tinker attempts must be a JSON object or list"]

    if not raw:
        return [], ["tinker attempts must include at least one attempt"]

    attempts: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f"tinker attempts[{index}] must be a JSON object")
            continue
        attempt, attempt_errors = _normalise_attempt(item, index + 1)
        attempts.append(attempt)
        errors.extend(attempt_errors)
    return attempts, errors


def _validate_attempt_lineage(attempts: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not attempts:
        return ["tinker attempts are missing"]

    seen_run_ids: set[str] = set()
    for index, attempt in enumerate(attempts):
        attempt_no = attempt.get("attempt")
        expected_attempt = index + 1
        if attempt_no != expected_attempt:
            errors.append("tinker attempt numbering must be contiguous from 1")
            break

    for index, attempt in enumerate(attempts):
        attempt_no = attempt.get("attempt", index + 1)
        run_id = attempt.get("run_id")
        status = attempt.get("status")
        if not run_id:
            continue
        if run_id in seen_run_ids:
            errors.append(f"attempt {attempt_no} run_id must be unique")
        else:
            seen_run_ids.add(run_id)

        if index == 0:
            if attempt.get("parent_attempt") is not None:
                errors.append("attempt 1 must not have parent_attempt")
            if attempt.get("parent_run_id") is not None:
                errors.append("attempt 1 must not have parent_run_id")
        else:
            previous = attempts[index - 1]
            previous_no = previous.get("attempt")
            if attempt.get("parent_attempt") not in {previous_no, None}:
                errors.append(
                    f"attempt {attempt_no} parent_attempt must be {previous_no}"
                )
            expected_parent_run = previous.get("run_id")
            if expected_parent_run is not None and attempt.get("parent_run_id") not in {expected_parent_run, None}:
                errors.append(
                    f"attempt {attempt_no} parent_run_id must match previous attempt run_id"
                )

        if status in _TINER_PROGRESS_STATES:
            errors.append(f"attempt {attempt_no} is not terminal")
        elif status not in TINKER_TERMINAL_STATES:
            errors.append(f"attempt {attempt_no} has invalid status")
            continue

        is_last = index == len(attempts) - 1
        if is_last:
            if status not in TINKER_SUCCESS_STATES:
                errors.append("final tinker attempt must be completed successfully")
        else:
            if status in TINKER_NON_RETRYABLE_STATES:
                errors.append(
                    f"attempt {attempt_no} cannot be retried after status {status}"
                )
                continue
            if status not in TINKER_RETRYABLE_STATES:
                errors.append(
                    f"attempt {attempt_no} must be retryable before attempt {attempt_no + 1}"
                )

    return errors


def _validate_timing(
    wandb_record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    if not attempts:
        return
    wandb_started = wandb_record.get("created_at")
    first_started = attempts[0].get("started_at")
    if wandb_started is None or first_started is None:
        return
    if first_started < wandb_started:
        errors.append("first tinker attempt started before W&B run was created")


def _cross_check_ids(
    *,
    final_run_id: str | None,
    checkpoints: Sequence[Mapping[str, Any]],
    hf_receipts: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any] | None,
    errors: list[str],
) -> None:
    if final_run_id is None:
        errors.append("tinker final run_id must be supplied")
        return

    for checkpoint in checkpoints:
        run_id = _coerce_text(checkpoint.get("run_id"))
        if run_id is not None and run_id != final_run_id:
            errors.append("checkpoint run_id does not match final tinker run_id")
            break

    for receipt in hf_receipts:
        run_id = _coerce_text(receipt.get("run_id"))
        if run_id is not None and run_id != final_run_id:
            errors.append("hf receipt run_id does not match final tinker run_id")
            break

    if terminal is not None:
        run_id = _coerce_text(terminal.get("run_id"))
        if run_id is not None and run_id != final_run_id:
            errors.append("terminal receipt run_id does not match final tinker run_id")


def _validate_required_steps(
    attempts: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    hf_receipts: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    expected_steps = dict(zip(REQUIRED_STAGES, REQUIRED_STEPS))
    terminal_steps = terminal.get("steps") or []
    if not terminal_steps and attempts:
        terminal_steps = attempts[-1].get("steps") or []

    if not terminal_steps:
        errors.append("terminal steps must be provided")
        return

    terminal_step_set = set(terminal_steps)
    for step in REQUIRED_STEPS:
        if step not in terminal_step_set:
            errors.append(f"terminal run must include step {step}")

    checkpoint_steps = {
        entry["stage"]: entry["step"] for entry in checkpoints if entry.get("stage") in expected_steps
    }
    hf_steps = {
        entry["stage"]: entry["step"] for entry in hf_receipts if entry.get("stage") in expected_steps
    }

    for stage, expected in expected_steps.items():
        observed_cp = checkpoint_steps.get(stage)
        observed_hf = hf_steps.get(stage)
        if observed_cp != expected:
            errors.append(f"checkpoint[{stage}] step mismatch: expected {expected}, got {observed_cp}")
        if observed_hf != expected:
            errors.append(f"hf_receipts[{stage}] step mismatch: expected {expected}, got {observed_hf}")
        if observed_cp is not None and observed_hf is not None and observed_cp != observed_hf:
            errors.append(f"checkpoint and hf step mismatch for stage {stage}")


def _build_lifecycle(
    *,
    wandb_record: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    hf_receipts: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    steps_seen = set()
    if terminal:
        steps_seen.update(terminal.get("steps", []))
    if not steps_seen and attempts:
        steps_seen.update(attempts[-1].get("steps") or [])

    hf_stages = {entry.get("stage") for entry in hf_receipts}

    state_ok = {
        "wandb_sync": bool(wandb_record.get("run_id") and wandb_record.get("state", "").lower() in WANDB_SUCCESS_STATES and wandb_record.get("mode", "").lower() == "online"),
        "tinker_identity": bool(attempts),
        "train_step_0": 0 in steps_seen,
        "hf_initial_export": "initial" in hf_stages,
        "train_step_5": 5 in steps_seen,
        "hf_periodic_export": "periodic" in hf_stages,
        "train_step_10": 10 in steps_seen,
        "hf_final_export": "final" in hf_stages,
        "terminal_receipt": bool(terminal.get("status") in TINKER_SUCCESS_STATES and terminal.get("exit_code") == 0),
    }

    return [
        {
            "state": state,
            "ok": state_ok.get(state, False),
        }
        for state in STATE_SEQUENCE
    ]


def validate_tinker_run_monitor(
    config: Mapping[str, Any],
    *,
    wandb_sync: Any,
    tinker_attempts: Any,
    checkpoint_json: Any,
    hf_receipts: Any,
    terminal_receipt: Any,
) -> list[str]:
    """Validate smoke lifecycle artifacts and return blockers in deterministic order."""

    blockers: list[str] = []

    if not isinstance(config, Mapping):
        return ["smoke config must be a JSON object"]

    config_errors = smoke_config.validate_smoke_config(config)
    if config_errors:
        blockers.extend([f"smoke config: {error}" for error in config_errors])

    wandb_payload, wandb_load_errors = _load_payload(
        wandb_sync, label="wandb sync metadata"
    )
    tinker_payload, tinker_load_errors = _load_payload(
        tinker_attempts, label="tinker attempts"
    )
    checkpoint_payload, checkpoint_load_errors = _load_payload(
        checkpoint_json, label="checkpoint JSON"
    )
    hf_payload, hf_load_errors = _load_payload(hf_receipts, label="hf receipts")
    terminal_payload, terminal_load_errors = _load_payload(
        terminal_receipt, label="terminal receipt"
    )

    blockers.extend(wandb_load_errors)
    blockers.extend(tinker_load_errors)
    blockers.extend(checkpoint_load_errors)
    blockers.extend(hf_load_errors)
    blockers.extend(terminal_load_errors)

    wandb_record, wandb_errors = _normalise_wandb_record(wandb_payload)
    blockers.extend(wandb_errors)

    attempts, attempt_errors = _normalise_tinker_attempts(tinker_payload)
    blockers.extend(attempt_errors)

    checkpoints, checkpoint_errors = _normalise_checkpoints(checkpoint_payload)
    blockers.extend(checkpoint_errors)

    hf_records, hf_errors = _normalise_hf_receipts(hf_payload)
    blockers.extend(hf_errors)

    terminal_record, terminal_errors = _normalise_terminal_receipt(terminal_payload)
    blockers.extend(terminal_errors)

    if not blockers:
        _validate_timing(wandb_record, attempts, blockers)
        blockers.extend(_validate_attempt_lineage(attempts))

        final_run_id = attempts[-1].get("run_id") if attempts else None
        _cross_check_ids(
            final_run_id=final_run_id,
            checkpoints=checkpoints,
            hf_receipts=hf_records,
            terminal=terminal_record,
            errors=blockers,
        )

        _validate_required_steps(
            attempts=attempts,
            terminal=terminal_record,
            checkpoints=checkpoints,
            hf_receipts=hf_records,
            errors=blockers,
        )

    return blockers


def generate_tinker_run_monitor(
    config: Mapping[str, Any],
    *,
    wandb_sync: Any,
    tinker_attempts: Any,
    checkpoint_json: Any,
    hf_receipts: Any,
    terminal_receipt: Any,
) -> dict[str, Any]:
    """Build an offline lifecycle report and fail-closed status."""

    blockers = validate_tinker_run_monitor(
        config,
        wandb_sync=wandb_sync,
        tinker_attempts=tinker_attempts,
        checkpoint_json=checkpoint_json,
        hf_receipts=hf_receipts,
        terminal_receipt=terminal_receipt,
    )

    wandb_payload, _ = _load_payload(wandb_sync, label="wandb sync metadata")
    tinker_payload, _ = _load_payload(tinker_attempts, label="tinker attempts")
    checkpoint_payload, _ = _load_payload(checkpoint_json, label="checkpoint JSON")
    hf_payload, _ = _load_payload(hf_receipts, label="hf receipts")
    terminal_payload, _ = _load_payload(terminal_receipt, label="terminal receipt")

    wandb_record, _ = _normalise_wandb_record(wandb_payload)
    attempts, _ = _normalise_tinker_attempts(tinker_payload)
    checkpoint_records, _ = _normalise_checkpoints(checkpoint_payload)
    hf_records, _ = _normalise_hf_receipts(hf_payload)
    terminal_record, _ = _normalise_terminal_receipt(terminal_payload)

    final_attempt = attempts[-1] if attempts else {}

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "READY" if not blockers else "BLOCKED",
        "component_only": config.get("component_only"),
        "primary_eval": config.get("primary_eval"),
        "heldout": config.get("heldout"),
        "portfolio_claim": config.get("portfolio_claim"),
        "tinker_run_id": final_attempt.get("run_id"),
        "wandb_run_id": wandb_record.get("run_id"),
        "lineage": {
            "attempt_count": len(attempts),
            "has_retry": len(attempts) > 1,
            "terminal_attempt": final_attempt.get("attempt"),
            "attempts": [
                {
                    "attempt": attempt.get("attempt"),
                    "run_id": attempt.get("run_id"),
                    "status": attempt.get("status"),
                    "parent_attempt": attempt.get("parent_attempt"),
                    "parent_run_id": attempt.get("parent_run_id"),
                }
                for attempt in attempts
            ],
        },
        "lifecycle": _build_lifecycle(
            wandb_record=wandb_record,
            attempts=attempts,
            hf_receipts=hf_records,
            terminal=terminal_record,
        ),
        "blockers": blockers,
        "errors": blockers,
        "checkpoint_records": checkpoint_records,
        "hf_records": hf_records,
        "terminal_receipt": terminal_record,
        "launchable": False,
        "allowed": False,
        "smoke_model": config.get("model"),
        "smoke_model_revision": config.get("model_revision"),
        "smoke_xlam_revision": config.get("xlam_revision"),
    }


def assert_tinker_run_monitor(
    config: Mapping[str, Any],
    *,
    wandb_sync: Any,
    tinker_attempts: Any,
    checkpoint_json: Any,
    hf_receipts: Any,
    terminal_receipt: Any,
) -> dict[str, Any]:
    blockers = validate_tinker_run_monitor(
        config,
        wandb_sync=wandb_sync,
        tinker_attempts=tinker_attempts,
        checkpoint_json=checkpoint_json,
        hf_receipts=hf_receipts,
        terminal_receipt=terminal_receipt,
    )
    if blockers:
        raise TinkerRunMonitorError("tinker run monitor is blocked: " + "; ".join(blockers))
    return generate_tinker_run_monitor(
        config,
        wandb_sync=wandb_sync,
        tinker_attempts=tinker_attempts,
        checkpoint_json=checkpoint_json,
        hf_receipts=hf_receipts,
        terminal_receipt=terminal_receipt,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--wandb")
    parser.add_argument("--tinker-attempts")
    parser.add_argument("--checkpoints")
    parser.add_argument("--hf-receipts")
    parser.add_argument("--terminal")
    args = parser.parse_args(argv)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    report = generate_tinker_run_monitor(
        config,
        wandb_sync=args.wandb,
        tinker_attempts=args.tinker_attempts,
        checkpoint_json=args.checkpoints,
        hf_receipts=args.hf_receipts,
        terminal_receipt=args.terminal,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
