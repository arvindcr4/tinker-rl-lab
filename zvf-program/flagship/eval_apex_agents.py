#!/usr/bin/env python3
"""Run the smallest exact APEX-Agents evaluation through Archipelago.

This runner is intentionally fail-closed.  It binds the Mercor benchmark and
the native Archipelago grader to immutable revisions, requires one disjoint
task, starts W&B online before constructing a Tinker client, and enforces the
E5-specific $0.50 Tinker ceiling inside the shared $16.50/$1.50 envelope.

The benchmark is evaluation-only.  It must never be replaced with xLAM,
GSM8K, WebArena, BrowserGym, or another related task.  If the gated dataset,
the native environment, W&B, Tinker, or the runtime cannot be acquired, a
machine-readable ``BLOCKED`` receipt is written and no Tinker call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SUITE_ID = "apex_agents_eval"
DATASET_ID = "mercor/apex-agents"
DATASET_REVISION = "92c86856cf1b11f9833a8a076b3a45a63afa3929"
DATASET_LICENSE = "cc-by-4.0"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_API_URL = "https://huggingface.co/api/datasets/mercor/apex-agents"
DATASET_RESOLVE_URL = "https://huggingface.co/datasets/mercor/apex-agents/resolve"
ARCHIPELAGO_REPOSITORY = "https://github.com/Mercor-Intelligence/archipelago"
ARCHIPELAGO_REVISION = "1c3dcd4694b313020cd626699c9c7cc1c0a2fc58"
ARCHIPELAGO_LICENSE = "Apache-2.0"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MODEL_API_URL = "https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B"
TINKER_BRIDGE_BASE_URL = (
    "https://arvindcr4--pavlov-tinker-openai-bridge-tinkeropenaibridge-web.modal.run"
)
TINKER_BRIDGE_API_BASE = f"{TINKER_BRIDGE_BASE_URL}/v1"
TINKER_BRIDGE_MODEL_ALIAS = "pavlov-qwen36-tinker"
TINKER_SAMPLER_PATH = (
    "tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/sampler_weights/seed809_final"
)
TINKER_HF_REPO = (
    "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
)
TINKER_HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"
DEFAULT_JUDGE_MODEL = "vertex_ai/gemini-2.5-flash"
WANDB_ENTITY = "arvindcr4-pes-university"
WANDB_PROJECT = "tinker-rl-lab-pavlov"
WANDB_GROUP = "pavlov-e5-apex-agents"
MAX_TINKER_SPEND_USD = Decimal("0.50")
OPERATIONAL_CAP_USD = Decimal("16.50")
SAFETY_RESERVE_USD = Decimal("1.50")
HARD_CAP_USD = Decimal("18.00")
PREFILL_USD_PER_MILLION = Decimal("0.54")
SAMPLE_USD_PER_MILLION = Decimal("1.335")
DEFAULT_LIMIT = 1
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_PROMPT_TOKENS = 8192
DEFAULT_MAX_RESPONSE_TOKENS = 512
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_NATIVE_VERIFIER_PATH = "grading"

# ---------------------------------------------------------------------------
# APEX-Agents dataset schema.
#
# Every field below is required by Mercor's OWN loader, not inferred by us:
# `archipelago/examples/hugging_face_task/main.py` at revision
# 1c3dcd4694b313020cd626699c9c7cc1c0a2fc58 reads
#   task["task_id"], task["world_id"], task["task_name"], task["domain"],
#   task["prompt"], task.get("task_input_files"), task.get("rubric", []),
#   rubric_entry["verifier_id"], rubric_entry["criteria"],
#   world["world_id"], world["world_name"]
# and resolves `world_files_zipped/<world_id>.zip` plus
# `task_files/<task_id>/**`.
#
# Validating against this contract BEFORE a run means a malformed or
# unexpectedly-reshaped dataset fails at ingestion with a named field, instead
# of failing halfway through a paid agent rollout.
# ---------------------------------------------------------------------------
APEX_TASK_REQUIRED_FIELDS = ("task_id", "world_id", "task_name", "domain", "prompt")
# Confirmed against the real revision: `task_input_files` is `str | None` -- a
# snapshot id like `snap_<32 hex>`, NOT a list of filenames. Only its
# truthiness is meaningful; the per-task asset directory is keyed by task_id.
# 141 tasks carry one, and that set matches the repo's `task_files/` dirs
# exactly. The revision also carries three fields the upstream example never
# reads: `expected_output`, `gold_response`, `gold_response_type`.
APEX_TASK_OPTIONAL_FIELDS = (
    "task_input_files",
    "rubric",
    "expected_output",
    "gold_response",
    "gold_response_type",
)
APEX_RUBRIC_REQUIRED_FIELDS = ("verifier_id", "criteria")
APEX_WORLD_REQUIRED_FIELDS = ("world_id", "world_name")
# `DEFAULT_TASK = "task_9ba58a6197114140877a1df1754d2993"` in the upstream
# example: the literal prefix `task_` plus a 32-character lowercase hex uuid4.
APEX_TASK_ID_RE = re.compile(r"^task_[0-9a-f]{32}$")
APEX_EXPECTED_TASK_COUNT = 480
APEX_EXPECTED_WORLD_COUNT = 33
APEX_WORLD_ZIP_TEMPLATE = "world_files_zipped/{world_id}.zip"
APEX_TASK_FILES_TEMPLATE = "task_files/{task_id}"

_HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_PLACEHOLDER_WORDS = frozenset(
    {"", "none", "null", "unknown", "unset", "pending", "placeholder", "todo"}
)


class PreflightError(ValueError):
    """A malformed launch input, never a benchmark result."""


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    details: Mapping[str, Any]
    required_receipt: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "details": dict(self.details),
        }
        if self.required_receipt is not None:
            record["required_external_receipt"] = dict(self.required_receipt)
        return record


@dataclass(frozen=True)
class Probe:
    status: int | None
    body: bytes
    error: str | None
    headers: Mapping[str, str]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in _PLACEHOLDER_WORDS


def _require_revision(name: str, value: str) -> str:
    if not isinstance(value, str) or not _HEX40_RE.fullmatch(value.strip()):
        raise PreflightError(f"{name} must be an immutable 40-hex commit")
    return value.strip().lower()


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or not _HEX64_RE.fullmatch(value.strip()):
        raise PreflightError(f"{name} must be a 64-hex SHA-256")
    return value.strip().lower()


def _decimal(name: str, value: str | Decimal) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PreflightError(f"{name} must be a finite non-negative decimal") from exc
    if not result.is_finite() or result < 0:
        raise PreflightError(f"{name} must be a finite non-negative decimal")
    return result


def max_tinker_cost(
    *,
    task_count: int,
    max_steps: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> Decimal:
    """Conservative uncached prefill + sampled-token ceiling."""

    if min(task_count, max_steps, max_prompt_tokens, max_response_tokens) <= 0:
        raise PreflightError("task_count, max_steps, and token limits must be positive")
    tokens = (
        Decimal(task_count * max_steps * max_prompt_tokens) * PREFILL_USD_PER_MILLION
        + Decimal(task_count * max_steps * max_response_tokens) * SAMPLE_USD_PER_MILLION
    ) / Decimal(1_000_000)
    return tokens


def _probe(
    url: str,
    *,
    token: str | None = None,
    method: str = "GET",
    max_bytes: int = 8_000_000,
    opener: Callable[..., Any] | None = None,
) -> Probe:
    """Make a bounded, non-mutating HTTP probe without leaking credentials."""

    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method=method)
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(request, timeout=30) as response:
            body = b"" if method == "HEAD" else response.read(max_bytes + 1)
            if len(body) > max_bytes:
                return Probe(
                    int(getattr(response, "status", 200)),
                    b"",
                    f"response exceeded {max_bytes} bytes",
                    dict(response.headers.items()),
                )
            return Probe(
                int(getattr(response, "status", 200)),
                body,
                None,
                dict(response.headers.items()),
            )
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(max_bytes)
        except Exception:
            body = b""
        return Probe(exc.code, body, str(exc.reason), dict(exc.headers.items()))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return Probe(None, b"", str(exc), {})


def _probe_record(probe: Probe, url: str) -> dict[str, Any]:
    error_code = None
    if probe.status in {401, 403}:
        error_code = "gated_or_unauthorized"
    elif probe.status is None:
        error_code = "network_or_runtime_error"
    return {
        "url": url,
        "http_status": probe.status,
        "error": probe.error,
        "error_code": error_code,
        "response_sha256": sha256_bytes(probe.body) if probe.body else None,
    }


def _json_body(probe: Probe, label: str) -> Mapping[str, Any] | list[Any] | None:
    if probe.status != 200 or not probe.body:
        return None
    try:
        value = json.loads(probe.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{label} returned malformed JSON: {exc}") from exc
    if not isinstance(value, (Mapping, list)):
        raise PreflightError(f"{label} must return a JSON object or array")
    return value


def _gating_facts(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Report *how* the repo is gated, not merely *that* it is.

    The HF Hub reports three states for ``gated``: ``false`` (open),
    ``"auto"`` (automatic approval -- any logged-in user is granted access the
    moment they accept the terms), and ``"manual"`` (the repo owner reviews
    each request).  The difference decides whether the blocker is "log in" or
    "wait for a human", so the receipt must carry it verbatim.
    """
    gated = metadata.get("gated")
    if gated is False or gated is None:
        mode, unblock = "open", "no access request needed"
    elif gated == "auto":
        mode = "auto"
        unblock = (
            "log in to Hugging Face and accept the dataset terms;"
            " access is granted automatically, with no human review"
        )
    elif gated == "manual":
        mode = "manual"
        unblock = "request access and wait for the repo owner to approve"
    else:
        mode = str(gated)
        unblock = "unrecognised gating mode; inspect the dataset page"
    return {
        "gated": gated,
        "gating_mode": mode,
        "gating_unblock": unblock,
        "private": metadata.get("private"),
        "disabled": metadata.get("disabled"),
        "file_count": len(metadata.get("siblings") or []),
    }


def _dataset_metadata_gate(
    *, token: str | None, opener: Callable[..., Any] | None = None
) -> tuple[Gate, Mapping[str, Any] | None]:
    probe = _probe(DATASET_API_URL, token=token, opener=opener)
    record = _probe_record(probe, DATASET_API_URL)
    metadata = _json_body(probe, "APEX dataset metadata")
    if not isinstance(metadata, Mapping):
        return Gate(
            "benchmark_metadata",
            "BLOCKED",
            {**record, "dataset_id": DATASET_ID},
            {
                "kind": "dataset_metadata",
                "action": "retain the official dataset API metadata response",
                "url": DATASET_API_URL,
                "expected_dataset_revision": DATASET_REVISION,
            },
        ), None
    observed_id = metadata.get("id")
    observed_sha = metadata.get("sha")
    license_value = (metadata.get("cardData") or {}).get("license")
    errors: list[str] = []
    if observed_id != DATASET_ID:
        errors.append(f"dataset id mismatch: {observed_id!r}")
    if observed_sha != DATASET_REVISION:
        errors.append(
            f"dataset revision mismatch: expected {DATASET_REVISION}, got {observed_sha!r}"
        )
    if license_value != DATASET_LICENSE:
        errors.append(
            f"dataset license mismatch: expected {DATASET_LICENSE}, got {license_value!r}"
        )
    if errors:
        return Gate(
            "benchmark_metadata",
            "BLOCKED",
            {
                **record,
                "dataset_id": observed_id,
                "dataset_revision": observed_sha,
                "license": license_value,
                "errors": errors,
            },
            {
                "kind": "immutable_benchmark_metadata",
                "action": "resolve the metadata mismatch against Mercor's official release",
                "url": DATASET_API_URL,
            },
        ), metadata
    return Gate(
        "benchmark_metadata",
        "PASS",
        {
            **record,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "license_url": DATASET_LICENSE_URL,
            **_gating_facts(metadata),
        },
    ), metadata


DATASET_ACCESS_FILES = {
    "tasks": "tasks_and_rubrics.json",
    "worlds": "world_descriptions.json",
    "eval": "eval.yaml",
}


def _local_dataset_gate(dataset_dir: Path) -> tuple[Gate, dict[str, Path] | None]:
    """Accept an already-downloaded copy, but only on proof of the revision.

    ``hf download`` writes ``.cache/huggingface/download/<file>.metadata`` whose
    first line is the resolved commit hash.  Requiring it means a hand-assembled
    directory, or a copy pulled from ``main``, cannot masquerade as the pinned
    revision.
    """
    missing: list[str] = []
    wrong_revision: list[dict[str, Any]] = []
    resolved: dict[str, Path] = {}
    receipts: dict[str, Any] = {}
    for label, filename in DATASET_ACCESS_FILES.items():
        path = dataset_dir / filename
        if not path.is_file():
            missing.append(filename)
            continue
        meta_path = dataset_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
        commit = None
        if meta_path.is_file():
            first_line = meta_path.read_text(encoding="utf-8").splitlines()
            commit = first_line[0].strip() if first_line else None
        if commit != DATASET_REVISION:
            wrong_revision.append({"file": filename, "recorded_commit": commit})
            continue
        resolved[label] = path
        receipts[filename] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_bytes(path.read_bytes()),
            "recorded_commit": commit,
        }
    if missing or wrong_revision:
        return Gate(
            "benchmark_access",
            "BLOCKED",
            {
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "source": "local_dataset_dir",
                "dataset_dir": str(dataset_dir),
                "missing": missing,
                "wrong_revision": wrong_revision,
                "message": (
                    "local dataset copy is incomplete or does not carry the"
                    " pinned commit in its Hugging Face cache metadata"
                ),
            },
            {
                "kind": "huggingface_dataset_access",
                "action": "re-download the exact revision with hf download",
                "url": f"https://huggingface.co/datasets/{DATASET_ID}",
                "commands": [
                    "hf auth login",
                    f"hf download {DATASET_ID} --repo-type dataset"
                    f" --revision {DATASET_REVISION}"
                    f" --local-dir {dataset_dir}"
                    " tasks_and_rubrics.json world_descriptions.json eval.yaml",
                ],
            },
        ), None
    return Gate(
        "benchmark_access",
        "PASS",
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "source": "local_dataset_dir",
            "dataset_dir": str(dataset_dir),
            "revision_proof": "hf cache metadata commit hash matches the pin",
            "files": receipts,
        },
    ), resolved


def _dataset_access_gate(
    *,
    token: str | None,
    cache_dir: Path,
    opener: Callable[..., Any] | None = None,
    dataset_dir: Path | None = None,
) -> tuple[Gate, dict[str, Path] | None]:
    if dataset_dir is not None:
        return _local_dataset_gate(dataset_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = dict(DATASET_ACCESS_FILES)
    downloaded: dict[str, Path] = {}
    probes: list[dict[str, Any]] = []
    for label, filename in files.items():
        url = f"{DATASET_RESOLVE_URL}/{DATASET_REVISION}/{filename}"
        probe = _probe(url, token=token, opener=opener)
        probes.append(_probe_record(probe, url))
        if probe.status != 200:
            message = "HF dataset content is gated; request access and retain a successful download receipt"
            return Gate(
                "benchmark_access",
                "BLOCKED",
                {
                    "dataset_id": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "file": filename,
                    "probes": probes,
                    "message": message,
                },
                {
                    "kind": "huggingface_dataset_access",
                    "action": "accept Mercor's dataset terms/request access, then download the exact revision",
                    "url": f"https://huggingface.co/datasets/{DATASET_ID}",
                    # `gated: auto` on this repo -> acceptance is granted
                    # immediately, so both steps are self-service.
                    "commands": [
                        "hf auth login",
                        f"open https://huggingface.co/datasets/{DATASET_ID}"
                        " and click 'Agree and access repository'",
                        f"hf download {DATASET_ID} --repo-type dataset"
                        f" --revision {DATASET_REVISION}"
                        " tasks_and_rubrics.json world_descriptions.json eval.yaml",
                    ],
                    "required": [
                        "HTTP 200 for tasks_and_rubrics.json, world_descriptions.json, and eval.yaml",
                        f"resolved commit {DATASET_REVISION}",
                        "SHA-256 for each downloaded file",
                    ],
                },
            ), None
        path = cache_dir / filename
        path.write_bytes(probe.body)
        downloaded[label] = path
    return Gate(
        "benchmark_access",
        "PASS",
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "files": {
                label: {"path": str(path), "sha256": sha256_bytes(path.read_bytes())}
                for label, path in downloaded.items()
            },
        },
    ), downloaded


def _load_json_file(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{label} is unavailable or malformed: {exc}") from exc


def validate_task_records(records: Any) -> list[str]:
    """Validate ``tasks_and_rubrics.json`` against the upstream loader contract.

    Returns a list of human-readable errors; an empty list means the file
    matches what Archipelago's own HuggingFace example expects.  This never
    raises on bad input -- callers turn the error list into a BLOCKED gate.
    """
    errors: list[str] = []
    if not isinstance(records, list):
        return ["tasks_and_rubrics.json must be a JSON array of task objects"]
    if not records:
        return ["tasks_and_rubrics.json contains zero task records"]

    seen_task_ids: set[str] = set()
    seen_verifier_ids: set[str] = set()
    for index, record in enumerate(records):
        where = f"tasks[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{where} is not a JSON object")
            continue
        for field in APEX_TASK_REQUIRED_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}.{field} must be a non-empty string")
        task_id = record.get("task_id")
        if isinstance(task_id, str):
            if not APEX_TASK_ID_RE.match(task_id):
                errors.append(
                    f"{where}.task_id {task_id!r} does not match {APEX_TASK_ID_RE.pattern}"
                )
            if task_id in seen_task_ids:
                errors.append(f"{where}.task_id {task_id!r} is duplicated")
            seen_task_ids.add(task_id)

        rubric = record.get("rubric")
        if rubric is None:
            errors.append(f"{where}.rubric is missing; the task cannot be graded")
            continue
        if not isinstance(rubric, list) or not rubric:
            errors.append(f"{where}.rubric must be a non-empty JSON array")
            continue
        for r_index, criterion in enumerate(rubric):
            r_where = f"{where}.rubric[{r_index}]"
            if not isinstance(criterion, Mapping):
                errors.append(f"{r_where} is not a JSON object")
                continue
            for field in APEX_RUBRIC_REQUIRED_FIELDS:
                value = criterion.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{r_where}.{field} must be a non-empty string")
            verifier_id = criterion.get("verifier_id")
            if isinstance(verifier_id, str):
                if verifier_id in seen_verifier_ids:
                    errors.append(f"{r_where}.verifier_id {verifier_id!r} is duplicated")
                seen_verifier_ids.add(verifier_id)
    return errors


def validate_world_records(records: Any) -> list[str]:
    """Validate ``world_descriptions.json`` against the upstream loader contract."""
    errors: list[str] = []
    if not isinstance(records, list):
        return ["world_descriptions.json must be a JSON array of world objects"]
    if not records:
        return ["world_descriptions.json contains zero world records"]
    seen: set[str] = set()
    for index, record in enumerate(records):
        where = f"worlds[{index}]"
        if not isinstance(record, Mapping):
            errors.append(f"{where} is not a JSON object")
            continue
        for field in APEX_WORLD_REQUIRED_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{where}.{field} must be a non-empty string")
        world_id = record.get("world_id")
        if isinstance(world_id, str):
            if world_id in seen:
                errors.append(f"{where}.world_id {world_id!r} is duplicated")
            seen.add(world_id)
    return errors


def validate_dataset_references(tasks: Any, worlds: Any) -> list[str]:
    """Every ``task.world_id`` must resolve in ``world_descriptions.json``."""
    if not isinstance(tasks, list) or not isinstance(worlds, list):
        return ["cannot cross-check references: one or both files are not JSON arrays"]
    known = {
        record.get("world_id")
        for record in worlds
        if isinstance(record, Mapping) and isinstance(record.get("world_id"), str)
    }
    errors: list[str] = []
    for index, record in enumerate(tasks):
        if not isinstance(record, Mapping):
            continue
        world_id = record.get("world_id")
        if isinstance(world_id, str) and world_id not in known:
            errors.append(
                f"tasks[{index}].world_id {world_id!r} has no entry in world_descriptions.json"
            )
    return errors


def dataset_ingestion_report(tasks: Any, worlds: Any) -> dict[str, Any]:
    """Full ingestion report for the exact dataset revision.

    ``errors`` empty means the dataset can be handed to Archipelago as-is.
    ``count_warnings`` are advisory: a changed task/world count is not
    automatically wrong, but it does mean the pinned revision is not the one
    the published numbers were produced against.
    """
    errors = validate_task_records(tasks)
    errors.extend(validate_world_records(worlds))
    if not errors:
        errors.extend(validate_dataset_references(tasks, worlds))

    task_count = len(tasks) if isinstance(tasks, list) else 0
    world_count = len(worlds) if isinstance(worlds, list) else 0
    warnings: list[str] = []
    if task_count != APEX_EXPECTED_TASK_COUNT:
        warnings.append(f"task count {task_count} != documented {APEX_EXPECTED_TASK_COUNT}")
    if world_count != APEX_EXPECTED_WORLD_COUNT:
        warnings.append(f"world count {world_count} != documented {APEX_EXPECTED_WORLD_COUNT}")

    task_ids = sorted(
        str(record["task_id"])
        for record in (tasks if isinstance(tasks, list) else [])
        if isinstance(record, Mapping) and isinstance(record.get("task_id"), str)
    )
    world_ids = sorted(
        str(record["world_id"])
        for record in (worlds if isinstance(worlds, list) else [])
        if isinstance(record, Mapping) and isinstance(record.get("world_id"), str)
    )
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "task_count": task_count,
        "world_count": world_count,
        "expected_task_count": APEX_EXPECTED_TASK_COUNT,
        "expected_world_count": APEX_EXPECTED_WORLD_COUNT,
        "task_id_sha256": sha256_json(task_ids),
        "world_id_sha256": sha256_json(world_ids),
        "errors": errors,
        "count_warnings": warnings,
        "valid": not errors,
    }


def required_task_assets(task: Mapping[str, Any]) -> list[str]:
    """Repo-relative dataset paths the runner must download for one task."""
    paths = [
        "tasks_and_rubrics.json",
        "world_descriptions.json",
        APEX_WORLD_ZIP_TEMPLATE.format(world_id=task.get("world_id", "")),
    ]
    if task.get("task_input_files"):
        paths.append(APEX_TASK_FILES_TEMPLATE.format(task_id=task.get("task_id", "")))
    return paths


def _pin_official_hf_downloads(source: str) -> str:
    """Bind the upstream example's runtime asset downloads to our dataset pin."""

    dataset_needle = 'HF_DATASET = "mercor/apex-agents"\n'
    dataset_replacement = dataset_needle + (
        "HF_DATASET_REVISION = os.environ['APEX_HF_DATASET_REVISION']\n"
    )
    world_needle = '        HF_DATASET, f"world_files_zipped/{world_id}.zip", repo_type="dataset"\n'
    world_replacement = (
        '        HF_DATASET, f"world_files_zipped/{world_id}.zip", '
        'repo_type="dataset", revision=HF_DATASET_REVISION\n'
    )
    task_needle = (
        '            HF_DATASET, repo_type="dataset", allow_patterns=[f"{task_prefix}/**"]\n'
    )
    task_replacement = (
        '            HF_DATASET, repo_type="dataset", revision=HF_DATASET_REVISION, '
        'allow_patterns=[f"{task_prefix}/**"]\n'
    )
    for label, needle in (
        ("dataset constant", dataset_needle),
        ("world asset download", world_needle),
        ("task asset download", task_needle),
    ):
        if source.count(needle) != 1:
            raise RuntimeError(f"official Archipelago {label} seam drifted")
    return (
        source.replace(dataset_needle, dataset_replacement)
        .replace(world_needle, world_replacement)
        .replace(task_needle, task_replacement)
    )


def _prefetch_exact_task_assets(*, args: argparse.Namespace, task_id: str) -> list[dict[str, Any]]:
    """Populate the offline runner cache from the immutable dataset revision."""

    if args.dataset_dir is None:
        raise RuntimeError("exact task asset prefetch requires --dataset-dir")
    tasks_path = args.dataset_dir / "tasks_and_rubrics.json"
    tasks = _load_json_file(tasks_path, "tasks_and_rubrics.json")
    task = next(
        (row for row in tasks if isinstance(row, Mapping) and row.get("task_id") == task_id),
        None,
    )
    if not isinstance(task, Mapping):
        raise RuntimeError(f"selected task is absent from the pinned dataset: {task_id}")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is unavailable for exact asset prefetch") from exc

    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    world_filename = APEX_WORLD_ZIP_TEMPLATE.format(world_id=task["world_id"])
    world_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            filename=world_filename,
            repo_type="dataset",
            revision=DATASET_REVISION,
            cache_dir=str(cache_dir),
        )
    )
    receipts.append(
        {
            "kind": "world_snapshot",
            "filename": world_filename,
            "revision": DATASET_REVISION,
            "sha256": sha256_bytes(world_path.read_bytes()),
            "bytes": world_path.stat().st_size,
        }
    )
    if task.get("task_input_files"):
        prefix = APEX_TASK_FILES_TEMPLATE.format(task_id=task_id)
        snapshot_path = Path(
            snapshot_download(
                repo_id=DATASET_ID,
                repo_type="dataset",
                revision=DATASET_REVISION,
                allow_patterns=[f"{prefix}/**"],
                cache_dir=str(cache_dir),
            )
        )
        files = sorted(path for path in (snapshot_path / prefix).rglob("*") if path.is_file())
        receipts.append(
            {
                "kind": "task_files",
                "prefix": prefix,
                "revision": DATASET_REVISION,
                "file_count": len(files),
                "tree_sha256": sha256_json(
                    [
                        {
                            "path": str(path.relative_to(snapshot_path)),
                            "sha256": sha256_bytes(path.read_bytes()),
                        }
                        for path in files
                    ]
                ),
            }
        )
    return receipts


def _dataset_schema_gate(downloaded: Mapping[str, Path] | None) -> Gate:
    if downloaded is None:
        return Gate(
            "dataset_schema",
            "BLOCKED",
            {
                "reason": "exact dataset content was not acquired",
                "validated": False,
                "contract_source": (
                    f"archipelago/examples/hugging_face_task/main.py @ {ARCHIPELAGO_REVISION}"
                ),
                "required_task_fields": list(APEX_TASK_REQUIRED_FIELDS),
                "required_rubric_fields": list(APEX_RUBRIC_REQUIRED_FIELDS),
                "required_world_fields": list(APEX_WORLD_REQUIRED_FIELDS),
            },
            {
                "kind": "huggingface_dataset_access",
                "action": "download the exact dataset revision, then re-run preflight",
                "url": f"https://huggingface.co/datasets/{DATASET_ID}",
            },
        )
    tasks = _load_json_file(downloaded["tasks"], "tasks_and_rubrics.json")
    worlds = _load_json_file(downloaded["worlds"], "world_descriptions.json")
    report = dataset_ingestion_report(tasks, worlds)
    if not report["valid"]:
        return Gate(
            "dataset_schema",
            "BLOCKED",
            report,
            {
                "kind": "dataset_schema_mismatch",
                "action": (
                    "the pinned revision does not match the Archipelago loader"
                    " contract; re-pin the dataset or update the field contract"
                ),
            },
        )
    return Gate("dataset_schema", "PASS", report)


def _task_ids_from_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_file():
        raise PreflightError(f"training task-ID manifest is missing: {path}")
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return sorted({value for value in values if value and not value.startswith("#")})


def _select_tasks(
    *,
    task_records: Sequence[Any],
    task_id: str | None,
    task_index: int | None,
    limit: int,
) -> list[Mapping[str, Any]]:
    if limit != 1:
        raise PreflightError("E5 smallest exact evaluation requires --limit 1")
    rows = [row for row in task_records if isinstance(row, Mapping)]
    if not rows:
        raise PreflightError("APEX task file contains no task records")
    if task_id and task_index is not None:
        raise PreflightError("use either --task-id or --task-index, not both")
    if task_id:
        selected = [row for row in rows if row.get("task_id") == task_id]
        if not selected:
            raise PreflightError(f"task id not found in exact dataset revision: {task_id}")
        return selected[:1]
    if task_index is None:
        task_index = 0
    if task_index < 0 or task_index >= len(rows):
        raise PreflightError(f"task index must be in [0, {len(rows) - 1}]")
    return [rows[task_index]]


def _task_split_gate(
    *,
    downloaded: Mapping[str, Path] | None,
    task_id: str | None,
    task_index: int | None,
    training_task_ids_path: Path | None,
    limit: int,
) -> tuple[Gate, list[Mapping[str, Any]]]:
    if downloaded is None:
        return Gate(
            "task_split",
            "BLOCKED",
            {
                "task_ids": None,
                "training_task_ids": _task_ids_from_file(training_task_ids_path),
                "disjoint": False,
                "reason": "exact dataset content was not acquired",
            },
            {
                "kind": "split_manifest",
                "action": "download the exact dataset revision and seal selected task IDs plus training task IDs",
                "required": ["split_manifest_sha256", "task_id_sha256", "disjoint=true"],
            },
        ), []
    tasks = _load_json_file(downloaded["tasks"], "tasks_and_rubrics.json")
    if not isinstance(tasks, list):
        raise PreflightError("tasks_and_rubrics.json must be a JSON array")
    selected = _select_tasks(
        task_records=tasks, task_id=task_id, task_index=task_index, limit=limit
    )
    selected_ids = [str(row.get("task_id", "")) for row in selected]
    if any(not _nonempty(value) for value in selected_ids):
        raise PreflightError("selected task has no non-empty task_id")
    training_ids = _task_ids_from_file(training_task_ids_path)
    overlap = sorted(set(selected_ids) & set(training_ids))
    split_manifest = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "suite_id": SUITE_ID,
        "evaluation_task_ids": selected_ids,
        "training_task_ids": training_ids,
        "training_scope": "none (eval-only)" if not training_ids else "external manifest",
    }
    task_hash = sha256_json(selected_ids)
    split_hash = sha256_json(split_manifest)
    if overlap:
        return Gate(
            "task_split",
            "BLOCKED",
            {
                "evaluation_task_ids": selected_ids,
                "training_task_ids": training_ids,
                "overlap": overlap,
                "split_manifest_sha256": split_hash,
                "task_id_sha256": task_hash,
                "disjoint": False,
            },
            {
                "kind": "disjoint_task_ids",
                "action": "replace the evaluation task manifest with IDs disjoint from training",
            },
        ), selected
    return Gate(
        "task_split",
        "PASS",
        {
            "evaluation_task_ids": selected_ids,
            "training_task_ids": training_ids,
            "training_scope": split_manifest["training_scope"],
            "split_manifest": split_manifest,
            "split_manifest_sha256": split_hash,
            "task_id_sha256": task_hash,
            "disjoint": True,
        },
    ), selected


def _native_verifier_gate(archipelago_dir: Path) -> Gate:
    grading_dir = archipelago_dir / DEFAULT_NATIVE_VERIFIER_PATH
    required_paths = (
        archipelago_dir / "README.md",
        archipelago_dir / "LICENSE",
        grading_dir / "runner" / "main.py",
        grading_dir / "pyproject.toml",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return Gate(
            "native_verifier",
            "BLOCKED",
            {
                "repository": ARCHIPELAGO_REPOSITORY,
                "expected_revision": ARCHIPELAGO_REVISION,
                "missing": missing,
            },
            {
                "kind": "archipelago_checkout",
                "action": "acquire the official Archipelago repository at the immutable revision",
                "repository": ARCHIPELAGO_REPOSITORY,
                "revision": ARCHIPELAGO_REVISION,
            },
        )
    try:
        observed = subprocess.check_output(
            ["git", "-C", str(archipelago_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(archipelago_dir), "ls-tree", "-r", "HEAD", "--", "grading"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return Gate(
            "native_verifier",
            "BLOCKED",
            {"repository": str(archipelago_dir), "error": str(exc)},
            {
                "kind": "archipelago_git_receipt",
                "action": "record a clean official checkout and verifier tree hash",
            },
        )
    tree_hash = sha256_text(tree)
    if observed.lower() != ARCHIPELAGO_REVISION:
        return Gate(
            "native_verifier",
            "BLOCKED",
            {
                "repository": ARCHIPELAGO_REPOSITORY,
                "observed_revision": observed,
                "expected_revision": ARCHIPELAGO_REVISION,
                "grading_tree_sha256": tree_hash,
            },
            {
                "kind": "archipelago_revision",
                "action": "checkout the exact official Archipelago commit",
                "repository": ARCHIPELAGO_REPOSITORY,
                "revision": ARCHIPELAGO_REVISION,
            },
        )
    return Gate(
        "native_verifier",
        "PASS",
        {
            "repository": ARCHIPELAGO_REPOSITORY,
            "revision": ARCHIPELAGO_REVISION,
            "license": ARCHIPELAGO_LICENSE,
            "verifier_path": DEFAULT_NATIVE_VERIFIER_PATH,
            "grading_tree_sha256": tree_hash,
        },
    )


def _runtime_gate(
    archipelago_dir: Path,
    worktree_root: Path,
    environment_backend: str = "modal",
) -> Gate:
    required_commands = (
        ("docker", "uv", "git") if environment_backend == "local-docker" else ("uv", "git")
    )
    missing_commands = [name for name in required_commands if shutil.which(name) is None]
    try:
        python313 = subprocess.check_output(
            ["uv", "python", "find", "3.13"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        python313 = None
        missing_commands.append(f"python3.13 ({exc})")
    required_environment_file = (
        archipelago_dir / "environment" / "docker-compose.yml"
        if environment_backend == "local-docker"
        else archipelago_dir / "environment" / "Dockerfile"
    )
    if not required_environment_file.is_file():
        missing_commands.append("official Archipelago environment")
    if environment_backend == "modal":
        try:
            import modal  # noqa: F401 - presence is the preflight condition
        except ImportError:
            missing_commands.append("modal Python client")
    if missing_commands:
        return Gate(
            "isolated_runtime",
            "BLOCKED",
            {
                "worktree_root": str(worktree_root),
                "required_environment": str(worktree_root / ".codex" / "e5" / "venv"),
                "environment_backend": environment_backend,
                "missing": missing_commands,
            },
            {
                "kind": "isolated_runtime_receipt",
                "action": "create the per-worktree Python 3.13 environment and verify the selected runtime",
                "environment_path": str(worktree_root / ".codex" / "e5" / "venv"),
            },
        )
    return Gate(
        "isolated_runtime",
        "PASS",
        {
            "worktree_root": str(worktree_root),
            "python313": python313,
            "environment_path": str(worktree_root / ".codex" / "e5" / "venv"),
            "archipelago_environment": str(archipelago_dir / "environment"),
            "environment_backend": environment_backend,
            "local_docker_required": environment_backend == "local-docker",
            "global_python_mutation": False,
        },
    )


def _model_gate(
    *,
    token: str | None,
    base_model: str,
    base_model_revision: str,
    sampler_path: str | None,
    hf_checkpoint_repo: str | None,
    hf_checkpoint_revision: str | None,
    opener: Callable[..., Any] | None = None,
) -> Gate:
    if base_model != MODEL_ID:
        return Gate(
            "model_identity",
            "BLOCKED",
            {"observed_model": base_model, "expected_model": MODEL_ID},
            {"kind": "model_identity", "action": f"use exactly {MODEL_ID}"},
        )
    try:
        revision = _require_revision("base_model_revision", base_model_revision)
    except PreflightError as exc:
        return Gate("model_identity", "BLOCKED", {"error": str(exc)}, {"kind": "model_revision"})
    if revision != MODEL_REVISION:
        return Gate(
            "model_identity",
            "BLOCKED",
            {"model": MODEL_ID, "observed_revision": revision, "expected_revision": MODEL_REVISION},
            {
                "kind": "model_revision",
                "action": "resolve the Qwen model metadata and use its immutable commit",
                "url": MODEL_API_URL,
            },
        )
    probe = _probe(MODEL_API_URL, token=token, opener=opener)
    metadata = _json_body(probe, "model metadata")
    observed_sha = metadata.get("sha") if isinstance(metadata, Mapping) else None
    if observed_sha != MODEL_REVISION:
        return Gate(
            "model_identity",
            "BLOCKED",
            {
                **_probe_record(probe, MODEL_API_URL),
                "model": MODEL_ID,
                "observed_revision": observed_sha,
                "expected_revision": MODEL_REVISION,
            },
            {
                "kind": "model_metadata",
                "action": "resolve the exact public model commit",
                "url": MODEL_API_URL,
            },
        )
    if sampler_path:
        if not (
            _nonempty(hf_checkpoint_repo)
            and hf_checkpoint_revision
            and _HEX40_RE.fullmatch(hf_checkpoint_revision)
        ):
            return Gate(
                "model_identity",
                "BLOCKED",
                {
                    "mode": "eval_only_sampler",
                    "sampler_path": sampler_path,
                    "hf_checkpoint_repo": hf_checkpoint_repo,
                    "hf_checkpoint_revision": hf_checkpoint_revision,
                },
                {
                    "kind": "hf_checkpoint_commit",
                    "action": "bind the evaluated sampler to a public Hugging Face repository commit",
                    "required": ["repo_url", "40-hex revision", "commit URL"],
                },
            )
        checkpoint_url = (
            f"https://huggingface.co/{hf_checkpoint_repo}/commit/{hf_checkpoint_revision}"
        )
        return Gate(
            "model_identity",
            "PASS",
            {
                "mode": "eval_only_sampler",
                "base_model": MODEL_ID,
                "base_model_revision": MODEL_REVISION,
                "sampler_path": sampler_path,
                "evaluated_hf_checkpoint": {
                    "repo": hf_checkpoint_repo,
                    "revision": hf_checkpoint_revision,
                    "url": checkpoint_url,
                },
            },
        )
    return Gate(
        "model_identity",
        "PASS",
        {
            "mode": "base_model",
            "base_model": MODEL_ID,
            "base_model_revision": MODEL_REVISION,
            "checkpoint_binding": "immutable base-model revision (no compatible completed sampler checkpoint supplied)",
        },
    )


def _budget_gate(
    *,
    task_count: int,
    max_steps: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
    maximum_tinker_spend_usd: Decimal,
) -> Gate:
    projected = max_tinker_cost(
        task_count=task_count,
        max_steps=max_steps,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    errors: list[str] = []
    if maximum_tinker_spend_usd != MAX_TINKER_SPEND_USD:
        errors.append(f"E5 maximum must be exactly ${MAX_TINKER_SPEND_USD}")
    if projected > maximum_tinker_spend_usd:
        errors.append(
            f"projected ceiling ${projected} exceeds E5 maximum ${maximum_tinker_spend_usd}"
        )
    if OPERATIONAL_CAP_USD + SAFETY_RESERVE_USD != HARD_CAP_USD:
        errors.append("shared operational cap/reserve no longer sum to $18.00")
    details = {
        "projected_maximum_usd": str(projected),
        "maximum_tinker_spend_usd": str(maximum_tinker_spend_usd),
        "operational_cap_usd": str(OPERATIONAL_CAP_USD),
        "safety_reserve_usd": str(SAFETY_RESERVE_USD),
        "hard_cap_usd": str(HARD_CAP_USD),
        "max_steps": max_steps,
        "max_prompt_tokens": max_prompt_tokens,
        "max_response_tokens": max_response_tokens,
    }
    return Gate(
        "budget",
        "BLOCKED" if errors else "PASS",
        {**details, "errors": errors},
        {
            "kind": "budget_authorization",
            "action": "retain the $0.50 E5 ceiling, $16.50 operational cap, and $1.50 reserve",
        }
        if errors
        else None,
    )


def _wandb_credential_source() -> str | None:
    """Where a W&B credential would come from, without reading its value.

    ``wandb.login()`` falls back to ``~/.netrc`` when ``WANDB_API_KEY`` is
    unset, so checking only the environment variable over-reports the blocker.
    Only the *presence* of a machine entry is inspected -- never the secret.
    """
    if os.environ.get("WANDB_API_KEY"):
        return "WANDB_API_KEY"
    netrc_path = Path(os.environ.get("NETRC") or (Path.home() / ".netrc"))
    if not netrc_path.is_file():
        return None
    try:
        import netrc as _netrc

        entry = _netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except Exception:  # pragma: no cover - malformed/unreadable netrc
        return None
    if entry and entry[2]:
        return "netrc:api.wandb.ai"
    return None


def _tracking_gate() -> Gate:
    mode = os.environ.get("WANDB_MODE", "").strip().lower()
    disabled = os.environ.get("WANDB_DISABLED", "").strip().lower()
    importable = True
    import_error = None
    try:
        __import__("wandb")
    except Exception as exc:  # pragma: no cover - exercised in the live receipt
        importable = False
        import_error = f"{type(exc).__name__}: {exc}"
    credential_source = _wandb_credential_source()
    errors: list[str] = []
    if mode and mode != "online":
        errors.append("WANDB_MODE must be online or unset")
    if disabled in {"1", "true", "yes", "on"}:
        errors.append("WANDB_DISABLED disables tracking")
    if not importable:
        errors.append("wandb dependency is unavailable in the isolated runtime")
    if credential_source is None:
        errors.append("no W&B credential in WANDB_API_KEY or ~/.netrc")
    return Gate(
        "wandb_online_before_tinker",
        "BLOCKED" if errors else "PENDING_RUNTIME_ONLINE_HANDSHAKE",
        {
            "required_mode": "online",
            "importable": importable,
            "import_error": import_error,
            "api_key_present": bool(os.environ.get("WANDB_API_KEY")),
            "credential_source": credential_source,
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "errors": errors,
            "tinker_calls_allowed_before_run": False,
        },
        {
            "kind": "wandb_online_run",
            "action": "install wandb in the per-worktree environment, supply a W&B credential (WANDB_API_KEY or ~/.netrc), and retain a verified online run_id/run_url initialized before Tinker",
            "required": ["mode=online", "run_id", "run_url", "config acknowledged"],
        }
        if errors
        else None,
    )


def _resolve_bridge_credential() -> tuple[str | None, str | None]:
    value = os.environ.get("TINKER_BRIDGE_API_KEY")
    if value:
        return value, "environment"
    security_cli = shutil.which("security")
    if security_cli is None:
        return None, None
    try:
        account = subprocess.run(
            ["id", "-un"], check=False, capture_output=True, text=True
        ).stdout.strip()
        if not account:
            return None, None
        key_result = subprocess.run(
            [
                security_cli,
                "find-generic-password",
                "-a",
                account,
                "-s",
                "pavlov-tinker-openai-bridge",
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None
    value = key_result.stdout.strip()
    return (value, "macos_keychain") if value else (None, None)


def _tinker_gate() -> Gate:
    bridge_key, credential_source = _resolve_bridge_credential()
    errors = []
    if not bridge_key:
        errors.append("Tinker bridge credential is unavailable")
    return Gate(
        "tinker_access",
        "BLOCKED" if errors else "PASS",
        {
            "transport": "authenticated_modal_openai_bridge",
            "bridge_credential_present": bool(bridge_key),
            "credential_source": credential_source,
            "local_tinker_client_required": False,
            "calls_made": False,
        },
        {
            "kind": "tinker_access",
            "action": "provide the authenticated Pavlov Tinker bridge credential",
            "required": ["bridge credential", "READY bridge health", "estimated cost <= $0.50"],
        }
        if errors
        else None,
    )


def _judge_key_gate(judge_model: str = DEFAULT_JUDGE_MODEL) -> Gate:
    if judge_model == f"openai/{TINKER_BRIDGE_MODEL_ALIAS}":
        return Gate(
            "native_grader_credentials",
            "PASS",
            {
                "judge_model": judge_model,
                "provider_key_present": "TINKER_BRIDGE_API_KEY_OR_KEYCHAIN",
                "same_model_judge": True,
            },
        )
    provider = judge_model.split("/", 1)[0].lower()
    if provider in {"vertex_ai", "gemini"}:
        keys = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    elif provider == "openai":
        keys = ("OPENAI_API_KEY",)
    elif provider == "anthropic":
        keys = ("ANTHROPIC_API_KEY",)
    else:
        keys = ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")
    present = [name for name in keys if os.environ.get(name)]
    if present:
        return Gate(
            "native_grader_credentials",
            "PASS",
            {"judge_model": judge_model, "provider_key_present": present[0]},
        )
    return Gate(
        "native_grader_credentials",
        "BLOCKED",
        {"judge_model": judge_model, "provider_key_present": None},
        {
            "kind": "archipelago_grading_credentials",
            "action": "provide one official Archipelago grading-provider key for the native rubric judge",
            "required": list(keys),
        },
    )


def _make_config(
    *,
    args: argparse.Namespace,
    model_gate: Gate,
    split_gate: Gate,
    budget_gate: Gate,
    dataset_gate: Gate,
    native_gate: Gate,
) -> dict[str, Any]:
    provenance = {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": DATASET_LICENSE,
        "dataset_license_url": DATASET_LICENSE_URL,
        "native_verifier_repository": ARCHIPELAGO_REPOSITORY,
        "native_verifier_revision": ARCHIPELAGO_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    for gate in (dataset_gate, native_gate, model_gate, split_gate, budget_gate):
        details = gate.details
        if gate.name == "task_split":
            provenance["split_manifest_sha256"] = details.get("split_manifest_sha256")
            provenance["task_id_sha256"] = details.get("task_id_sha256")
        if gate.name == "native_verifier":
            provenance["native_verifier_tree_sha256"] = details.get("grading_tree_sha256")
        if gate.name == "model_identity":
            provenance["evaluated_hf_checkpoint"] = details.get("evaluated_hf_checkpoint")
    return {
        "schema_version": "pavlov-e5-apex-agents-config-v1",
        "campaign": "pavlov-18usd",
        "stage": "primary-evaluation",
        "suite_id": SUITE_ID,
        "suite_role": "primary_eval",
        "benchmark": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "native_verifier": {
                "repository": ARCHIPELAGO_REPOSITORY,
                "revision": ARCHIPELAGO_REVISION,
                "path": DEFAULT_NATIVE_VERIFIER_PATH,
            },
        },
        "model": {
            "id": MODEL_ID,
            "base_revision": MODEL_REVISION,
            "mode": "sampler" if args.sampler_path else "base_model",
            "sampler_path": args.sampler_path,
            "hf_checkpoint_repo": args.hf_checkpoint_repo,
            "hf_checkpoint_revision": args.hf_checkpoint_revision,
        },
        "task_selection": split_gate.details,
        "limits": {
            "tasks": args.limit,
            "max_steps": args.max_steps,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_response_tokens": args.max_response_tokens,
            "timeout_seconds": args.timeout_seconds,
        },
        "budget": budget_gate.details,
        "wandb": {
            "entity": WANDB_ENTITY,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "mode": "online",
        },
        "judge": {"model": getattr(args, "judge_model", DEFAULT_JUDGE_MODEL)},
        "environment": {
            "backend": getattr(args, "environment_backend", "modal"),
            "local_docker_required": getattr(args, "environment_backend", "modal")
            == "local-docker",
        },
        "provenance": provenance,
        "exact_benchmark_only": True,
        "substitutions": [],
        "dataset_access_status": dataset_gate.status,
    }


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_receipt(
    *,
    args: argparse.Namespace,
    gates: Sequence[Gate],
    config: Mapping[str, Any],
    status: str,
    selected_tasks: Sequence[Mapping[str, Any]],
    launch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate_dicts = [gate.as_dict() for gate in gates]
    return {
        "schema_version": "pavlov-e5-apex-agents-receipt-v1",
        "created_at": _now(),
        "status": status,
        "suite_id": SUITE_ID,
        "benchmark": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "license": DATASET_LICENSE,
            "license_url": DATASET_LICENSE_URL,
            "native_verifier_repository": ARCHIPELAGO_REPOSITORY,
            "native_verifier_revision": ARCHIPELAGO_REVISION,
        },
        "model": config.get("model"),
        "task_selection": {
            "selected_task_ids": [str(row.get("task_id")) for row in selected_tasks],
            "count": len(selected_tasks),
            "disjoint": next(
                (gate.details.get("disjoint") for gate in gates if gate.name == "task_split"), False
            ),
        },
        "config_sha256": sha256_json(config),
        "config": config,
        "gates": gate_dicts,
        "required_external_receipts": [
            gate.required_receipt
            for gate in gates
            if gate.required_receipt is not None and gate.status != "PASS"
        ],
        "launch": launch or {"tinker_calls": 0, "score": None, "score_status": "not_run"},
        "no_score_claim": status != "COMPLETED",
        "source_urls": {
            "dataset_metadata": DATASET_API_URL,
            "dataset_card": f"https://huggingface.co/datasets/{DATASET_ID}",
            "archipelago": ARCHIPELAGO_REPOSITORY,
            "model_metadata": MODEL_API_URL,
        },
        "invocation": {
            "launch_requested": bool(args.launch),
            "worktree": str(Path.cwd()),
            "global_python_mutation": False,
        },
    }


def _all_launch_gates_pass(gates: Sequence[Gate]) -> bool:
    required = {
        "benchmark_metadata",
        "benchmark_access",
        "dataset_schema",
        "task_split",
        "native_verifier",
        "isolated_runtime",
        "model_identity",
        "budget",
        "wandb_online_before_tinker",
        "tinker_access",
        "native_grader_credentials",
    }
    allowed = {
        "wandb_online_before_tinker": {"PASS", "PENDING_RUNTIME_ONLINE_HANDSHAKE"},
    }
    return all(
        gate.status in allowed.get(gate.name, {"PASS"}) for gate in gates if gate.name in required
    )


def _wait_for_modal_environment(url: str, token: str, timeout_seconds: int = 600) -> None:
    """Wait for an authenticated Archipelago health response from Modal."""

    deadline = time.monotonic() + timeout_seconds
    request = urllib.request.Request(
        f"{url.rstrip('/')}/health",
        headers={"Authorization": f"Bearer {token}"},
    )
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status == 200 and response.read().decode("utf-8").strip() == "OK":
                    return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Modal Archipelago environment failed its health check: {last_error}")


def _start_modal_environment(
    archipelago_dir: Path,
    *,
    task_id: str,
    timeout_seconds: int,
) -> tuple[Any, str, str, Mapping[str, Any]]:
    """Build and start the pinned Archipelago Dockerfile as a Modal Sandbox."""

    try:
        import modal
    except ImportError as exc:
        raise RuntimeError("Modal client is unavailable in the isolated E5 runtime") from exc

    dockerfile = archipelago_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        raise RuntimeError(f"official Archipelago Dockerfile is absent: {dockerfile}")

    app = modal.App.lookup("pavlov-e5-apex-environment", create_if_missing=True)
    image = modal.Image.from_dockerfile(
        dockerfile,
        context_dir=archipelago_dir,
    )
    sandbox = None
    try:
        with modal.enable_output():
            sandbox = modal.Sandbox.create(
                app=app,
                name=f"e5-{task_id[-8:]}-{int(time.time())}",
                tags={
                    "campaign": "pavlov-18usd",
                    "suite": SUITE_ID,
                    "task_id": task_id,
                    "archipelago_revision": ARCHIPELAGO_REVISION,
                },
                image=image,
                timeout=max(timeout_seconds + 3600, 7200),
                cpu=8.0,
                memory=16384,
                encrypted_ports=[8080],
            )
            # Waiting for tunnel metadata also waits for the image/container startup.
            sandbox.tunnels(timeout=1800)
        credentials = sandbox.create_connect_token(
            user_metadata={"campaign": "pavlov-18usd", "task_id": task_id},
            port=8080,
        )
        url = credentials.url.rstrip("/")
        _wait_for_modal_environment(url, credentials.token)
        return (
            sandbox,
            url,
            credentials.token,
            {
                "backend": "modal-sandbox",
                "sandbox_id": sandbox.object_id,
                "app_name": "pavlov-e5-apex-environment",
                "port": 8080,
                "connect_token_present": True,
                "connect_token_recorded": False,
                "local_docker_used": False,
            },
        )
    except Exception:
        if sandbox is not None:
            sandbox.terminate(wait=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--training-task-ids", type=Path)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--archipelago-dir", type=Path, default=Path(".codex/e5/archipelago"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".codex/e5/dataset-cache"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help=(
            "consume an already-downloaded copy of the dataset instead of"
            " re-fetching it; the directory must carry Hugging Face cache"
            " metadata recording the pinned commit"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--base-model-revision", default=MODEL_REVISION)
    parser.add_argument("--sampler-path", default=TINKER_SAMPLER_PATH)
    parser.add_argument("--hf-checkpoint-repo", default=TINKER_HF_REPO)
    parser.add_argument("--hf-checkpoint-revision", default=TINKER_HF_COMMIT)
    parser.add_argument("--maximum-tinker-spend-usd", default=str(MAX_TINKER_SPEND_USD))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-response-tokens", type=int, default=DEFAULT_MAX_RESPONSE_TOKENS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--wandb-name", default="apex-agents-e5")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--environment-backend",
        choices=("modal", "local-docker"),
        default="modal",
        help="run the official environment in an authenticated Modal Sandbox (default) or local Docker",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="retained directory for the official Archipelago configs and artifacts",
    )
    return parser


def run_preflight(
    args: argparse.Namespace, *, opener: Callable[..., Any] | None = None
) -> dict[str, Any]:
    """Run all local/network gates and return the receipt (never launches)."""

    if args.dataset_revision != DATASET_REVISION:
        raise PreflightError(
            f"E5 is pinned to the authoritative dataset revision {DATASET_REVISION}; mutable/other revisions are rejected"
        )
    if args.limit != 1:
        raise PreflightError("E5 smallest exact evaluation requires --limit 1")
    if args.task_id and args.task_index is not None:
        raise PreflightError("use either --task-id or --task-index")
    maximum = _decimal("maximum_tinker_spend_usd", args.maximum_tinker_spend_usd)
    if args.max_steps <= 0 or args.max_prompt_tokens <= 0 or args.max_response_tokens <= 0:
        raise PreflightError("max steps and token limits must be positive")
    metadata_gate, _ = _dataset_metadata_gate(token=os.environ.get("HF_TOKEN"), opener=opener)
    access_gate, downloaded = _dataset_access_gate(
        token=os.environ.get("HF_TOKEN"),
        cache_dir=args.cache_dir,
        opener=opener,
        dataset_dir=getattr(args, "dataset_dir", None),
    )
    schema_gate = _dataset_schema_gate(downloaded)
    # A schema mismatch must not be papered over by task selection succeeding
    # on the one record that happens to be well-formed.
    if schema_gate.status != "PASS":
        downloaded = None
    split_gate, selected_tasks = _task_split_gate(
        downloaded=downloaded,
        task_id=args.task_id,
        task_index=args.task_index,
        training_task_ids_path=args.training_task_ids,
        limit=args.limit,
    )
    native_gate = _native_verifier_gate(args.archipelago_dir)
    runtime_gate = _runtime_gate(
        args.archipelago_dir,
        Path.cwd(),
        getattr(args, "environment_backend", "modal"),
    )
    model_gate = _model_gate(
        token=os.environ.get("HF_TOKEN"),
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        sampler_path=args.sampler_path,
        hf_checkpoint_repo=args.hf_checkpoint_repo,
        hf_checkpoint_revision=args.hf_checkpoint_revision,
        opener=opener,
    )
    budget_gate = _budget_gate(
        task_count=args.limit,
        max_steps=args.max_steps,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        maximum_tinker_spend_usd=maximum,
    )
    tracking_gate = _tracking_gate()
    tinker_gate = _tinker_gate()
    judge_gate = _judge_key_gate(getattr(args, "judge_model", DEFAULT_JUDGE_MODEL))
    gates = [
        metadata_gate,
        access_gate,
        schema_gate,
        split_gate,
        native_gate,
        runtime_gate,
        model_gate,
        budget_gate,
        tracking_gate,
        tinker_gate,
        judge_gate,
    ]
    config = _make_config(
        args=args,
        model_gate=model_gate,
        split_gate=split_gate,
        budget_gate=budget_gate,
        dataset_gate=access_gate,
        native_gate=native_gate,
    )
    ready = _all_launch_gates_pass(gates)
    status = "READY" if ready else "BLOCKED"
    if not ready:
        status = "BLOCKED"
    return _build_receipt(
        args=args,
        gates=gates,
        config=config,
        status=status,
        selected_tasks=selected_tasks,
    )


def _launch_official_archipelago(
    args: argparse.Namespace, receipt: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Launch one exact task through the pinned upstream example.

    This path is intentionally reached only after every gate passes.  The
    production launcher is a small bootstrap around the official example: it
    pins Hugging Face downloads to ``DATASET_REVISION`` and materializes a
    temporary config so the upstream environment/grader remains untouched.
    """

    selected_task_ids = receipt.get("task_selection", {}).get("selected_task_ids")
    if not selected_task_ids:
        raise PreflightError("cannot launch without one selected exact APEX task")
    task_id = str(selected_task_ids[0])

    bridge_key, _ = _resolve_bridge_credential()
    if not bridge_key:
        raise RuntimeError("Tinker bridge credential is unavailable")

    health_request = urllib.request.Request(
        f"{TINKER_BRIDGE_BASE_URL}/health",
        headers={"Authorization": f"Bearer {bridge_key}"},
    )
    try:
        with urllib.request.urlopen(health_request, timeout=300) as response:
            health = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Tinker bridge health check failed: {exc}") from exc
    if health.get("status") != "READY":
        raise RuntimeError("Tinker bridge is not READY")
    if health.get("model") != TINKER_BRIDGE_MODEL_ALIAS:
        raise RuntimeError("Tinker bridge model alias drifted")
    if health.get("hf_commit") != TINKER_HF_COMMIT:
        raise RuntimeError("Tinker bridge Hugging Face checkpoint drifted")
    bridge_wandb_url = health.get("wandb_url")
    if not isinstance(bridge_wandb_url, str) or not bridge_wandb_url.startswith(
        "https://wandb.ai/"
    ):
        raise RuntimeError("Tinker bridge has no online W&B receipt")

    budget = health.get("budget")
    if not isinstance(budget, Mapping):
        raise RuntimeError("Tinker bridge budget receipt is absent")
    maximum = _decimal("bridge.maximum_usd", str(budget.get("maximum_usd")))
    charged = _decimal("bridge.charged_usd", str(budget.get("charged_usd", 0)))
    reserved = _decimal("bridge.reserved_usd", str(budget.get("reserved_usd", 0)))
    projected = _decimal(
        "projected_maximum_usd",
        str(receipt.get("config", {}).get("budget", {}).get("projected_maximum_usd")),
    )
    remaining = maximum - charged - reserved
    if remaining < projected:
        raise RuntimeError(
            "Tinker bridge budget is insufficient for the sealed one-task cap: "
            f"remaining ${remaining:.6f}, required ${projected:.6f}"
        )

    archipelago_dir = args.archipelago_dir.resolve()
    source_example = archipelago_dir / "examples" / "hugging_face_task"
    source_main = source_example / "main.py"
    if not source_main.is_file():
        raise RuntimeError(f"official Archipelago example is absent: {source_main}")
    run_dir = (args.run_dir or (args.out.parent / "official_run")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    static_files = (
        "agent_config.json",
        "scoring_config.json",
        "mcp_config_all_oss_servers.json",
        "grading_settings.json",
        "eval_configs.json",
    )
    for name in static_files:
        shutil.copy2(source_example / name, run_dir / name)
    (run_dir / "orchestrator_config.json").write_text(
        json.dumps(
            {
                "model": f"openai/{TINKER_BRIDGE_MODEL_ALIAS}",
                "extra_args": {"max_tokens": args.max_response_tokens},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    agent_config_path = run_dir / "agent_config.json"
    agent_config = json.loads(agent_config_path.read_text(encoding="utf-8"))
    agent_config["agent_config_values"]["timeout"] = args.timeout_seconds
    agent_config["agent_config_values"]["max_steps"] = args.max_steps
    agent_config_path.write_text(
        json.dumps(agent_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    grading_path = run_dir / "grading_settings.json"
    grading_settings = json.loads(grading_path.read_text(encoding="utf-8"))
    grading_settings["llm_judge_model"] = args.judge_model
    grading_path.write_text(
        json.dumps(grading_settings, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # The upstream example uses one inherited environment for both trajectory
    # and grading.  Scope the Tinker proxy variables to the agent subprocess so
    # the independent native rubric judge retains its own provider credentials.
    adapted_main = source_main.read_text(encoding="utf-8")
    transport_needle = 'ENV_URL = os.environ.get("ENV_URL", "http://localhost:8080")\n'
    transport_replacement = transport_needle + (
        "APEX_ENV_AUTH_TOKEN = os.environ.get('APEX_ENV_AUTH_TOKEN')\n"
        "_APEX_HTTPX_GET = httpx.get\n"
        "_APEX_HTTPX_POST = httpx.post\n"
        "_APEX_HTTPX_STREAM = httpx.stream\n\n"
        "def _apex_env_auth_kwargs(kwargs):\n"
        "    if APEX_ENV_AUTH_TOKEN:\n"
        "        kwargs = dict(kwargs)\n"
        "        headers = dict(kwargs.get('headers') or {})\n"
        "        headers.setdefault('Authorization', f'Bearer {APEX_ENV_AUTH_TOKEN}')\n"
        "        kwargs['headers'] = headers\n"
        "    return kwargs\n\n"
        "def _apex_httpx_get(*args, **kwargs):\n"
        "    return _APEX_HTTPX_GET(*args, **_apex_env_auth_kwargs(kwargs))\n\n"
        "def _apex_httpx_post(*args, **kwargs):\n"
        "    return _APEX_HTTPX_POST(*args, **_apex_env_auth_kwargs(kwargs))\n\n"
        "def _apex_httpx_stream(*args, **kwargs):\n"
        "    return _APEX_HTTPX_STREAM(*args, **_apex_env_auth_kwargs(kwargs))\n\n"
        "httpx.get = _apex_httpx_get\n"
        "httpx.post = _apex_httpx_post\n"
        "httpx.stream = _apex_httpx_stream\n"
    )
    if adapted_main.count(transport_needle) != 1:
        raise RuntimeError("official Archipelago environment URL seam drifted")
    adapted_main = adapted_main.replace(transport_needle, transport_replacement)
    adapted_main = _pin_official_hf_downloads(adapted_main)
    environment_start_needle = "    start_environment()\n"
    environment_start_replacement = (
        "    if os.environ.get('APEX_SKIP_LOCAL_ENVIRONMENT') == '1':\n"
        "        log('Using authenticated Modal Sandbox environment...')\n"
        "        if not wait_for_health(ENV_URL):\n"
        "            log('ERROR: Modal Sandbox environment failed health check')\n"
        "            sys.exit(1)\n"
        "    else:\n"
        "        start_environment()\n"
    )
    if adapted_main.count(environment_start_needle) != 1:
        raise RuntimeError("official Archipelago environment start seam drifted")
    adapted_main = adapted_main.replace(
        environment_start_needle,
        environment_start_replacement,
    )
    needle = "result = subprocess.run(agent_cmd, cwd=AGENTS_DIR)"
    replacement = (
        "agent_env = os.environ.copy()\n"
        "    agent_env['LITELLM_PROXY_API_BASE'] = os.environ['APEX_TINKER_BRIDGE_BASE']\n"
        "    agent_env['LITELLM_PROXY_API_KEY'] = os.environ['APEX_TINKER_BRIDGE_KEY']\n"
        "    result = subprocess.run(agent_cmd, cwd=AGENTS_DIR, env=agent_env)"
    )
    if adapted_main.count(needle) != 1:
        raise RuntimeError("official Archipelago agent launch seam drifted")
    adapted_main = adapted_main.replace(needle, replacement)
    agent_auth_needle = "    # Add extra args if present\n"
    agent_auth_replacement = (
        "    if APEX_ENV_AUTH_TOKEN:\n"
        "        agent_cmd.extend(['--mcp-gateway-auth-token', APEX_ENV_AUTH_TOKEN])\n\n"
        + agent_auth_needle
    )
    if adapted_main.count(agent_auth_needle) != 1:
        raise RuntimeError("official Archipelago agent auth seam drifted")
    adapted_main = adapted_main.replace(agent_auth_needle, agent_auth_replacement)
    grading_needle = "result = subprocess.run(grading_cmd, cwd=GRADING_DIR)"
    grading_replacement = (
        "grading_env = os.environ.copy()\n"
        "        if os.environ.get('APEX_JUDGE_VIA_TINKER') == '1':\n"
        "            grading_env['LITELLM_PROXY_API_BASE'] = os.environ['APEX_TINKER_BRIDGE_BASE']\n"
        "            grading_env['LITELLM_PROXY_API_KEY'] = os.environ['APEX_TINKER_BRIDGE_KEY']\n"
        "        result = subprocess.run(grading_cmd, cwd=GRADING_DIR, env=grading_env)"
    )
    if adapted_main.count(grading_needle) != 1:
        raise RuntimeError("official Archipelago grading launch seam drifted")
    adapted_main = adapted_main.replace(grading_needle, grading_replacement)
    adapted_main_path = run_dir / "main.py"
    adapted_main_path.write_text(adapted_main, encoding="utf-8")

    asset_receipts = _prefetch_exact_task_assets(args=args, task_id=task_id)

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is unavailable in the isolated E5 runtime") from exc
    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="primary-evaluation",
        name=args.wandb_name,
        mode="online",
        config=dict(receipt.get("config", {})),
        reinit=True,
    )
    if wandb_run is None or not getattr(wandb_run, "id", None):
        raise RuntimeError("online W&B initialization failed before Tinker")

    log_path = run_dir / "official_run.log"
    commands = (
        (["uv", "sync", "--locked"], archipelago_dir / "agents"),
        (["uv", "sync", "--locked"], archipelago_dir / "grading"),
        (
            ["uv", "run", "python", str(adapted_main_path), task_id],
            archipelago_dir / "agents",
        ),
    )
    return_code = 0
    score = None
    grades_path = run_dir / "output" / task_id / "grades.json"
    modal_sandbox = None
    environment_receipt: dict[str, Any] = {
        "backend": "local-docker",
        "local_docker_used": True,
    }
    try:
        environment_url = "http://localhost:8080"
        environment_token = ""
        if getattr(args, "environment_backend", "modal") == "modal":
            (
                modal_sandbox,
                environment_url,
                environment_token,
                modal_details,
            ) = _start_modal_environment(
                archipelago_dir,
                task_id=task_id,
                timeout_seconds=args.timeout_seconds,
            )
            environment_receipt = dict(modal_details)

        env = os.environ.copy()
        env.update(
            {
                "EXAMPLE_DIR": str(run_dir),
                "ARCHIPELAGO_DIR": str(archipelago_dir),
                "ENVIRONMENT_DIR": str(archipelago_dir / "environment"),
                "AGENTS_DIR": str(archipelago_dir / "agents"),
                "GRADING_DIR": str(archipelago_dir / "grading"),
                "HF_HUB_CACHE": str(args.cache_dir.resolve()),
                "HF_HUB_OFFLINE": "1",
                "APEX_HF_DATASET_REVISION": DATASET_REVISION,
                "APEX_TINKER_BRIDGE_BASE": TINKER_BRIDGE_API_BASE,
                "APEX_TINKER_BRIDGE_KEY": bridge_key,
                "APEX_JUDGE_VIA_TINKER": (
                    "1" if args.judge_model == f"openai/{TINKER_BRIDGE_MODEL_ALIAS}" else "0"
                ),
                "ENV_URL": environment_url,
                "APEX_ENV_AUTH_TOKEN": environment_token,
                "APEX_SKIP_LOCAL_ENVIRONMENT": (
                    "1" if getattr(args, "environment_backend", "modal") == "modal" else "0"
                ),
            }
        )
        with log_path.open("w", encoding="utf-8") as log:
            for command, cwd in commands:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_seconds + 1800,
                    check=False,
                )
                return_code = completed.returncode
                if return_code != 0:
                    break
        grades = (
            json.loads(grades_path.read_text(encoding="utf-8")) if grades_path.is_file() else None
        )
        score = (
            grades.get("scoring_results", {}).get("final_score")
            if isinstance(grades, Mapping)
            else None
        )
        wandb_run.log(
            {
                "e5/return_code": return_code,
                "e5/score": score,
                "e5/environment_backend": environment_receipt["backend"],
            }
        )
    finally:
        if modal_sandbox is not None:
            try:
                environment_receipt["termination_exit_code"] = modal_sandbox.terminate(wait=True)
                environment_receipt["terminated"] = True
            except Exception as exc:  # preserve the primary evaluation failure
                environment_receipt["terminated"] = False
                environment_receipt["termination_error"] = type(exc).__name__
        wandb_run.finish()
    if return_code != 0 or score is None:
        raise RuntimeError(
            f"official Archipelago run did not produce a native score; see {log_path}"
        )
    return {
        "tinker_calls": "one_or_more",
        "score": score,
        "score_status": "native_archipelago_grader",
        "judge_model": args.judge_model,
        "same_model_judge": args.judge_model == f"openai/{TINKER_BRIDGE_MODEL_ALIAS}",
        "task_id": task_id,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "grades_path": str(grades_path),
        "environment": environment_receipt,
        "wandb": {"run_id": wandb_run.id, "run_url": wandb_run.url, "mode": "online"},
        "bridge": {
            "model": TINKER_BRIDGE_MODEL_ALIAS,
            "hf_commit": TINKER_HF_COMMIT,
            "wandb_url": bridge_wandb_url,
            "budget_remaining_before_usd": str(remaining),
        },
        "provenance": {
            "official_main_sha256": sha256_bytes(source_main.read_bytes()),
            "adapted_main_sha256": sha256_bytes(adapted_main_path.read_bytes()),
            "dataset_revision": DATASET_REVISION,
            "archipelago_revision": ARCHIPELAGO_REVISION,
            "prefetched_assets": asset_receipts,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_preflight(args)
    except PreflightError as exc:
        receipt = {
            "schema_version": "pavlov-e5-apex-agents-receipt-v1",
            "created_at": _now(),
            "status": "BLOCKED",
            "suite_id": SUITE_ID,
            "error": str(exc),
            "no_score_claim": True,
            "required_external_receipts": [
                {"kind": "corrected_preflight_inputs", "action": str(exc)}
            ],
            "launch": {"tinker_calls": 0, "score": None, "score_status": "not_run"},
        }
        _write_receipt(args.out, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    if args.launch and receipt["status"] != "READY":
        _write_receipt(args.out, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    if args.launch:
        try:
            launch = _launch_official_archipelago(args, receipt)
        except Exception as exc:
            receipt = dict(receipt)
            receipt["status"] = "BLOCKED"
            receipt["launch"] = {
                "tinker_calls": "unknown_after_launch_failure",
                "score": None,
                "score_status": "not_run",
                "error": str(exc),
            }
            receipt["no_score_claim"] = True
            _write_receipt(args.out, receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 2
        receipt = dict(receipt)
        receipt["status"] = "COMPLETED"
        receipt["launch"] = dict(launch)
    _write_receipt(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"READY", "COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
