#!/usr/bin/env python3
"""Deterministic, offline launch audit for the first paid xLAM smoke run."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import pavlov_xlam_smoke_config as smoke_config

SCHEMA_VERSION = "pavlov-xlam-launch-audit-v1"

REQUIRED_STAGES = smoke_config.REQUIRED_CHECKPOINT_STAGES
REQUIRED_VISIBILITY = smoke_config.EXPECTED_CHECKPOINT_VISIBILITY
REQUIRED_STEPS = smoke_config.EXPECTED_CHECKPOINT_STEPS
REQUIRED_CONFIG_KEYS = {
    "schema_version",
    "smoke_id",
    "component",
    "model",
    "model_revision",
    "xlam_revision",
    "seed",
    "steps",
    "group",
    "batch",
    "rank",
    "learning_rate",
    "temperature",
    "top_p",
    "max_prompt_tokens",
    "max_response_tokens",
    "save_every_steps",
    "run_order",
    "wandb",
    "sampler_checkpoints",
    "runtime_constraints",
    "budget",
    "component_only",
    "primary_eval",
    "heldout",
    "portfolio_claim",
    "config_signature",
}

WANDB_STATES = ("finished", "completed")
SHA40_PREFIX_RE = re.compile(r"^([0-9a-f]{40}|sha256:[0-9a-f]{40})$")
HTTP_URL_RE = re.compile(r"^https://[^\s]+$")
WANDB_URL_RE = re.compile(r"^https://wandb\.ai/.+/.+/runs/.+$")

_PLACEHOLDER_WORDS = frozenset(
    {
        "",
        "missing",
        "none",
        "null",
        "todo",
        "placeholder",
        "unknown",
        "pending",
        "to_be_pinned_before_paid_runs",
    }
)


class LaunchAuditError(ValueError):
    """Raised for malformed launch audit records."""


def _coerce_text(value: Any) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _is_placeholder(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_WORDS
    return False


def _first_text(record: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in record:
            value = _coerce_text(record.get(key))
            if value is not None:
                return value
    return None


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_payload(payload: Any) -> Any:
    if isinstance(payload, (str, Path)):
        return _load_json(payload)
    return payload


def _load_payload(
    payload: Any, *, label: str, read_strings: bool = False
) -> tuple[Any, list[str]]:
    if payload is None:
        return None, [f"{label} must be provided"]
    if read_strings and isinstance(payload, (str, Path)):
        try:
            return _read_payload(payload), []
        except FileNotFoundError:
            return None, [f"{label} must point to an existing JSON file"]
        except (OSError, json.JSONDecodeError) as exc:
            return None, [f"{label} must be valid JSON: {type(exc).__name__}"]
    return payload, []


def _is_non_placeholder_id(value: Any) -> bool:
    text = _coerce_text(value)
    if text is None or _is_placeholder(text):
        return False
    if text.lower() in {"0", "null", "none", "placeholder", "todo", "pending"}:
        return False
    return True


def _normalise_wandb_record(value: Any) -> tuple[dict[str, Any], list[str]]:
    record: dict[str, Any] = {}
    errors: list[str] = []

    if not isinstance(value, Mapping):
        return record, ["wandb sync metadata must be a JSON object"]
    raw = value
    if isinstance(raw.get("run"), Mapping):
        raw = raw["run"]

    entity = _first_text(raw, "entity", "team", "user")
    project = _first_text(raw, "project", "entity_project", "wandb_project")
    run_id = _first_text(raw, "run_id", "id", "runId", "w_id")
    run_url = _first_text(raw, "run_url", "url", "link")
    state = _first_text(raw, "state", "status")
    mode = _first_text(raw, "mode", "offline_mode")
    group = _first_text(raw, "group", "run_group")
    if entity is None or not _is_non_placeholder_id(entity):
        errors.append("wandb entity is missing")
    if project is None or not _is_non_placeholder_id(project):
        errors.append("wandb project is missing")
    if run_id is None or not _is_non_placeholder_id(run_id):
        errors.append("wandb run_id is missing")
    if run_url is None or not WANDB_URL_RE.fullmatch(run_url):
        errors.append("wandb run_url must be a valid W&B run URL")
        if run_url is not None and not _is_non_placeholder_id(run_url):
            errors.append("wandb run_url must not be a placeholder")
    if state is None:
        errors.append("wandb state is missing")
    elif state.lower() not in WANDB_STATES:
        errors.append("wandb state must be finished after run")

    if mode is None:
        errors.append("wandb mode must be online")
    elif mode.lower() != "online":
        errors.append("wandb mode must be online")

    record.update(
        {
            "entity": entity,
            "project": project,
            "group": group,
            "run_id": run_id,
            "run_url": run_url,
            "state": state,
            "mode": mode,
        }
    )
    return record, errors


def _normalise_tinker_record(value: Any) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record: dict[str, Any] = {}
    if isinstance(value, str):
        run_id = _coerce_text(value)
        if run_id is None:
            errors.append("tinker run_id must be a non-empty string")
        record["run_id"] = run_id
        return record, errors
    if not isinstance(value, Mapping):
        return record, ["tinker run identity must be an object or run-id string"]
    raw = value
    run_id = _first_text(
        raw, "run_id", "id", "runId", "tinker_run_id", "tinkerRunId"
    )
    if run_id is None:
        errors.append("tinker run_id is missing")
    status = _first_text(raw, "status", "state")
    if status is None:
        errors.append("tinker run status is missing")
    elif status.lower() not in {"finished", "completed", "complete"}:
        errors.append("tinker run status must be finished")
    record["run_id"] = run_id
    record["status"] = status
    return record, errors


def _normalise_checkpoints(value: Any) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    errors: list[str] = []
    checkpoints: list[dict[str, Any]] = []
    payload_run_id: str | None = None

    if isinstance(value, Mapping):
        payload_run_id = _first_text(value, "tinker_run_id", "run_id", "runId", "tinkerRunId")
        raw_checkpoints = value.get("checkpoints") if isinstance(value.get("checkpoints"), list) else value.get("entries")
        if raw_checkpoints is None:
            errors.append("checkpoint manifest must include a checkpoints list")
            return checkpoints, payload_run_id, errors
        if not isinstance(raw_checkpoints, list):
            errors.append("checkpoints must be a list of checkpoint entries")
            return checkpoints, payload_run_id, errors
        candidate = raw_checkpoints
    elif isinstance(value, list):
        candidate = value
        if not value:
            errors.append("checkpoint manifest must not be empty")
    else:
        return checkpoints, payload_run_id, ["checkpoint manifest must be a list or object with checkpoints"]

    seen_steps: set[int] = set()
    seen_stages: set[str] = set()
    for index, item in enumerate(candidate):
        if not isinstance(item, Mapping):
            errors.append(f"checkpoint[{index}] must be a JSON object")
            continue
        step = item.get("step")
        if not isinstance(step, int) or isinstance(step, bool):
            errors.append(f"checkpoint[{index}] step must be an integer")
            continue
        stage = _coerce_text(item.get("stage"))
        if stage is None:
            errors.append(f"checkpoint[{index}] stage is missing")
            continue
        stage = stage.strip().lower()
        if stage not in REQUIRED_STAGES:
            errors.append(f"checkpoint[{index}] has invalid stage {stage!r}")
            continue
        if stage in seen_stages:
            errors.append(f"checkpoint[{stage}] appears more than once")
            continue
        if step in seen_steps:
            errors.append(f"checkpoint step {step} appears more than once")
            continue
        item_run_id = _first_text(item, "run_id", "tinker_run_id", "tinkerRunId")
        if payload_run_id is not None and item_run_id is not None and item_run_id != payload_run_id:
            errors.append("checkpoint manifest run_id mismatch inside entries")
        if item_run_id is None:
            item_run_id = payload_run_id
        entry = {
            "step": step,
            "stage": stage,
            "run_id": item_run_id,
            "repo_url": _coerce_text(item.get("repo_url", item.get("repo"))),
            "revision": _coerce_text(item.get("revision")),
        }
        checkpoints.append(entry)
        seen_steps.add(step)
        seen_stages.add(stage)

    if not errors:
        by_stage = {entry["stage"]: entry["step"] for entry in checkpoints}
        expected = dict(zip(REQUIRED_STAGES, REQUIRED_STEPS))
        observed = {entry["stage"] for entry in checkpoints}
        if observed != set(REQUIRED_STAGES):
            missing = sorted(set(REQUIRED_STAGES) - observed)
            errors.append("checkpoint manifest missing stages: " + ", ".join(missing))
        elif len(checkpoints) != len(REQUIRED_STAGES):
            errors.append("checkpoint manifest must contain exactly one initial, periodic, and final entry")
        else:
            for stage, step in expected.items():
                if by_stage.get(stage) != step:
                    errors.append(
                        f"checkpoint[{stage}] step mismatch: expected {step}, got {by_stage.get(stage)}"
                    )
    return checkpoints, payload_run_id, errors


def _normalise_hf_receipts(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    receipts: list[dict[str, Any]] = []

    raw: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, list):
                raw.extend(item)
            else:
                raw.append(item)
    elif isinstance(value, list):
        raw = value
    else:
        return receipts, ["hf receipts must be a list or stage/object map"]

    if not raw:
        return receipts, ["hf receipts must contain initial, periodic, and final entries"]

    seen_stages: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            errors.append(f"hf_receipts[{index}] must be a JSON object")
            continue
        stage = _coerce_text(item.get("stage", item.get("label", item.get("checkpoint_type"))))
        if stage is None:
            errors.append(f"hf_receipts[{index}] stage is missing")
            continue
        stage = stage.lower()
        step = item.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            step = None
        visibility = _coerce_text(item.get("visibility"))
        revision = _coerce_text(item.get("revision", item.get("commit", item.get("sha"))))
        repo = _coerce_text(item.get("repo_url", item.get("repo", item.get("repository"))))
        safe_public = item.get("safe_public_artifact")
        run_id = _first_text(item, "run_id", "tinker_run_id", "tinkerRunId")

        if stage not in REQUIRED_STAGES:
            errors.append(f"hf_receipts[{index}] has invalid stage {stage!r}")
            continue
        if stage in seen_stages:
            errors.append(f"hf receipt for stage {stage!r} appears more than once")
            continue
        if step is None:
            errors.append(f"hf_receipts[{index}] step is missing")
            continue
        if visibility is None:
            errors.append(f"hf_receipts[{index}] visibility is missing")
            continue
        visibility = visibility.lower()
        if visibility not in REQUIRED_VISIBILITY:
            errors.append(f"hf_receipts[{index}] visibility must be public or private")
            continue
        if visibility == "public" and safe_public is not True:
            errors.append(f"hf_receipts[{index}] must set safe_public_artifact true for public visibility")
        if revision is None or not SHA40_PREFIX_RE.fullmatch(revision):
            errors.append(f"hf_receipts[{index}] revision must be a 40-char hex commit")
            continue
        if repo is None or not HTTP_URL_RE.fullmatch(repo):
            errors.append(f"hf_receipts[{index}] repo_url must be a valid URL")
            continue
        if not _is_non_placeholder_id(repo):
            errors.append(f"hf_receipts[{index}] repo_url must not be a placeholder")
            continue
        if not _is_non_placeholder_id(revision):
            errors.append(f"hf_receipts[{index}] revision must not be a placeholder")
            continue
        receipts.append(
            {
                "stage": stage,
                "step": step,
                "visibility": visibility,
                "revision": revision,
                "repo_url": repo,
                "safe_public_artifact": safe_public,
                "run_id": run_id,
            }
        )
        seen_stages.add(stage)

    if not errors:
        by_stage = {entry["stage"]: entry["step"] for entry in receipts}
        observed = set(by_stage)
        if observed != set(REQUIRED_STAGES):
            missing = sorted(set(REQUIRED_STAGES) - observed)
            errors.append("hf receipts missing stages: " + ", ".join(missing))
        elif len(receipts) != len(REQUIRED_STAGES):
            errors.append("hf receipts must contain exactly one initial, periodic, and final entry")
        else:
            expected = dict(zip(REQUIRED_STAGES, REQUIRED_STEPS))
            for stage, step in expected.items():
                if by_stage.get(stage) != step:
                    errors.append(
                        f"hf_receipts[{stage}] step mismatch: expected {step}, got {by_stage.get(stage)}"
                    )
    return receipts, errors


def _extract_tinker_run_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        return _first_text(value, "run_id", "id", "runId", "tinker_run_id", "tinkerRunId")
    return _coerce_text(value)


def _extract_wandb_run_id(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw = value["run"] if isinstance(value.get("run"), Mapping) else value
    return _first_text(raw, "run_id", "id", "runId", "w_id")


def _collect_run_ids(
    *,
    tinker: Mapping[str, Any] | None,
    checkpoints: Sequence[Mapping[str, Any]],
    hf_receipts: Sequence[Mapping[str, Any]],
) -> list[str]:
    ids: set[str] = set()
    if tinker is not None and _is_non_placeholder_id(tinker.get("run_id")):
        ids.add(str(tinker["run_id"]))
    for item in checkpoints:
        if _is_non_placeholder_id(item.get("run_id")):
            ids.add(str(item["run_id"]))
    for item in hf_receipts:
        if _is_non_placeholder_id(item.get("run_id")):
            ids.add(str(item["run_id"]))
    return sorted(ids)


def _assert_id_coverage(*, tinker_id: str | None, run_ids: list[str], errors: list[str]) -> None:
    if tinker_id is None:
        errors.append("tinker run_id must be supplied")
        return
    if not run_ids:
        return
    missing = [value for value in run_ids if value != tinker_id]
    if missing:
        errors.append("run_id mismatch across checkpoint/hf metadata and tinker run_id")


def validate_launch_audit(
    config: Mapping[str, Any],
    *,
    wandb_sync: Any,
    tinker_run: Any,
    checkpoint_json: Any,
    hf_receipts: Any,
) -> list[str]:
    """Validate the launch audit inputs and return deterministic blockers."""

    errors: list[str] = []

    if not isinstance(config, Mapping):
        return ["smoke config must be a JSON object"]
    config_errors = smoke_config.validate_smoke_config(config)
    if config_errors:
        errors.extend([f"smoke config: {error}" for error in config_errors])
    if set(config.keys()) != REQUIRED_CONFIG_KEYS:
        errors.append("smoke config must contain exactly the required top-level keys")

    wandb, wandb_read_errors = _load_payload(
        wandb_sync, label="wandb sync metadata", read_strings=True
    )
    errors.extend(wandb_read_errors)

    tinker_payload, tinker_read_errors = _load_payload(
        tinker_run, label="tinker run identity", read_strings=False
    )
    errors.extend(tinker_read_errors)

    checkpoint_payload, checkpoint_read_errors = _load_payload(
        checkpoint_json, label="checkpoint JSON", read_strings=True
    )
    errors.extend(checkpoint_read_errors)

    hf_payload, hf_read_errors = _load_payload(
        hf_receipts, label="hf receipts", read_strings=True
    )
    errors.extend(hf_read_errors)

    wandb_record, wandb_errors = _normalise_wandb_record(wandb)
    errors.extend(wandb_errors)

    tinker_record, tinker_errors = _normalise_tinker_record(tinker_payload)
    errors.extend(tinker_errors)

    checkpoints, checkpoint_run_id, checkpoint_errors = _normalise_checkpoints(
        checkpoint_payload
    )
    errors.extend(checkpoint_errors)

    hf_records, hf_errors = _normalise_hf_receipts(hf_payload)
    errors.extend(hf_errors)

    if not errors:
        expected_ids = _collect_run_ids(
            tinker=tinker_record if tinker_record else None,
            checkpoints=checkpoints,
            hf_receipts=hf_records,
        )
        tinker_id = tinker_record.get("run_id") if isinstance(tinker_record, Mapping) else None
        _assert_id_coverage(
            tinker_id=tinker_id,
            run_ids=expected_ids,
            errors=errors,
        )
        if checkpoint_run_id is not None and tinker_id is not None and checkpoint_run_id != tinker_id:
            errors.append("checkpoint manifest run_id must match tinker run_id")

        for stage in REQUIRED_STAGES:
            cp_step = next(
                (item["step"] for item in checkpoints if item["stage"] == stage), None
            )
            hf_step = next((item["step"] for item in hf_records if item["stage"] == stage), None)
            if cp_step is None or hf_step is None:
                continue
            if cp_step != hf_step:
                errors.append(f"stage {stage} step mismatch between checkpoint ({cp_step}) and hf receipt ({hf_step})")
        if any(item.get("model_revision") for item in checkpoints):
            for item in checkpoints:
                revision = _coerce_text(item.get("model_revision"))
                if revision is not None and revision != config["model_revision"]:
                    errors.append("checkpoint model_revision does not match config")
                    break
        for item in hf_records:
            if _coerce_text(item.get("model_revision")) not in (None, config["model_revision"]):
                errors.append("hf receipt model_revision does not match config")
                break

    return errors


def generate_launch_audit(
    config: Mapping[str, Any],
    *,
    wandb_sync: Any,
    tinker_run: Any,
    checkpoint_json: Any,
    hf_receipts: Any,
) -> dict[str, Any]:
    """Return a deterministic audit envelope with BLOCKED/READY status."""

    config_mapping: Mapping[str, Any] = config if isinstance(config, Mapping) else {}
    blockers = validate_launch_audit(
        config,
        wandb_sync=wandb_sync,
        tinker_run=tinker_run,
        checkpoint_json=checkpoint_json,
        hf_receipts=hf_receipts,
    )
    tinker_payload, _ = _load_payload(
        tinker_run, label="tinker run identity", read_strings=False
    )
    wandb_payload, _ = _load_payload(
        wandb_sync, label="wandb sync metadata", read_strings=True
    )
    status = "READY" if not blockers else "BLOCKED"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "component_only": config_mapping.get("component_only"),
        "primary_eval": config_mapping.get("primary_eval"),
        "heldout": config_mapping.get("heldout"),
        "portfolio_claim": config_mapping.get("portfolio_claim"),
        "launchable": False,
        "allowed": False,
        "component": config_mapping.get("component"),
        "model": config_mapping.get("model"),
        "model_revision": config_mapping.get("model_revision"),
        "xlam_revision": config_mapping.get("xlam_revision"),
        "tinker_run_id": _extract_tinker_run_id(tinker_payload),
        "wandb_run_id": _extract_wandb_run_id(wandb_payload),
        "blockers": blockers,
        "errors": blockers,
    }


def assert_launch_audit(
    config: Mapping[str, Any],
    *,
    wandb_sync: Any,
    tinker_run: Any,
    checkpoint_json: Any,
    hf_receipts: Any,
) -> dict[str, Any]:
    """Validate and raise when the launch audit is incomplete."""

    blockers = validate_launch_audit(
        config,
        wandb_sync=wandb_sync,
        tinker_run=tinker_run,
        checkpoint_json=checkpoint_json,
        hf_receipts=hf_receipts,
    )
    if blockers:
        raise LaunchAuditError("launch audit is blocked: " + "; ".join(blockers))
    return generate_launch_audit(
        config,
        wandb_sync=wandb_sync,
        tinker_run=tinker_run,
        checkpoint_json=checkpoint_json,
        hf_receipts=hf_receipts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--wandb")
    parser.add_argument("--tinker")
    parser.add_argument("--checkpoints")
    parser.add_argument("--hf-receipts")
    args = parser.parse_args(argv)

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    wandb_payload = (
        _load_payload(args.wandb, label="wandb sync metadata", read_strings=True)[0]
        if args.wandb
        else None
    )
    tinker_payload = (
        _load_payload(args.tinker, label="tinker run identity", read_strings=True)[0]
        if args.tinker
        else None
    )
    checkpoint_payload = (
        _load_payload(args.checkpoints, label="checkpoint JSON", read_strings=True)[0]
        if args.checkpoints
        else None
    )
    hf_payload = (
        _load_payload(args.hf_receipts, label="hf receipts", read_strings=True)[0]
        if args.hf_receipts
        else None
    )
    report = generate_launch_audit(
        config,
        wandb_sync=wandb_payload,
        tinker_run=tinker_payload,
        checkpoint_json=checkpoint_payload,
        hf_receipts=hf_payload,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
