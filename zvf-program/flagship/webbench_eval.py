#!/usr/bin/env python3
"""Fail-closed E6 runner for the authoritative Halluminate WebBench.

The public Halluminate repository supplies the task CSV and MIT license, but it
does not publish the browser environment, task credentials, or native ground-
truth verifier.  This runner therefore has two deliberately separate paths:

* ``preflight`` is local and zero-cost.  It hashes the exact benchmark source,
  verifies task IDs and split disjointness, checks the per-worktree environment,
  model/checkpoint and budget contracts, and emits a machine-readable receipt.
* ``launch`` is an explicit, guarded seam for the official environment and
  verifier.  It cannot run unless every gate is satisfied.  W&B is initialized
  online before an optional Tinker command is invoked, and no score is accepted
  unless the native verifier returns one verdict for every requested task with
  screenshot and DOM evidence.

No related browser benchmark is accepted as a substitute.  In particular,
BrowserGym, WebArena, MiniWoB, synthetic tasks, and human labels are not native
WebBench verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pavlov-e6-webbench-receipt-v1"
SUITE_ID = "webbench_eval"
BENCHMARK_NAME = "Halluminate/WebBench"
BENCHMARK_REPO_URL = "https://github.com/Halluminate/WebBench"
BENCHMARK_LICENSE = "MIT"
BENCHMARK_REVISION = "ea7a1628443321363989f354401f0653e0cba6f4"
BENCHMARK_REVISION_URL = f"{BENCHMARK_REPO_URL}/tree/{BENCHMARK_REVISION}"
DATASET_FILE = "webbenchfinal.csv"
DATASET_RAW_URL = (
    f"https://raw.githubusercontent.com/Halluminate/WebBench/{BENCHMARK_REVISION}/"
    f"{DATASET_FILE}"
)
DATASET_SHA256 = "fd5311a38bdb6f941e8f544150735656c114d76fbfb17193da973d5de0165217"
LICENSE_SHA256 = "96804aa272fe40cdfb8b5c8f4d1d94757bcfaf1bf5596fb829214843d2371e58"
PUBLIC_TASK_COUNT = 2647
PUBLIC_TASK_ID_HASH = "22afbdd3cc47e6dba1e3c57ddbe5f762b54be5d2af6ac76bbd206c19eb83b12e"
PUBLIC_MANIFEST_HASH = "66da44a04ec48fe356b3b0d1c420c40679faa1a7ac650728e254b625bb674a07"
LICENSE_URL = f"{BENCHMARK_REPO_URL}/blob/{BENCHMARK_REVISION}/LICENSE"
GROUND_TRUTH_ISSUE_URL = f"{BENCHMARK_REPO_URL}/issues/2"

# Tinker-only policy: every suite binds a model that the Tinker service actually
# serves.  The previous binding here was ``Qwen/Qwen3-VL-30B-A3B-Instruct``, which
# get_server_capabilities() does not list at all -- a run against it could never
# have launched, and it silently diverged from the contract's authorized
# candidates.  See flagship/tinker_model_registry.json for the served set.
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
# Immutable HF commit resolved from refs/heads/main during this lane's
# read-only preflight.  The runner never follows a mutable branch at launch.
BASE_MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
BASE_MODEL_URL = f"https://huggingface.co/{MODEL_ID}"
BASE_MODEL_COMMIT_URL = f"{BASE_MODEL_URL}/commit/{BASE_MODEL_REVISION}"

PAVLOV_WANDB_ENTITY = "arvindcr4-pes-university"
PAVLOV_WANDB_PROJECT = "tinker-rl-lab-pavlov"
PAVLOV_WANDB_GROUP = "pavlov-e6-webbench"

SUITE_TINKER_MAX_USD = 0.50
GLOBAL_MAX_USD = 18.00
GLOBAL_OPERATIONAL_CAP_USD = 16.50
GLOBAL_SAFETY_RESERVE_USD = 1.50
PREFILL_USD_PER_MILLION = 0.54
SAMPLE_USD_PER_MILLION = 1.335

ALLOWED_CATEGORIES = frozenset(
    {"READ", "CREATE", "UPDATE", "DELETE", "FILE_MANIPULATION"}
)
REQUIRED_COLUMNS = ("ID", "Starting URL", "Category", "Task")
HEX40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$", re.IGNORECASE)
PLACEHOLDERS = frozenset(
    {
        "",
        "missing",
        "none",
        "null",
        "unknown",
        "pending",
        "placeholder",
        "todo",
        "unset",
        "tbd",
        "to_be_pinned",
    }
)

EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_ERROR = 1


class WebBenchError(ValueError):
    """Malformed input or a failed E6 gate."""


class LaunchBlocked(WebBenchError):
    """Raised when launch would violate an E6 gate."""

    def __init__(self, message: str, receipt: Mapping[str, Any]):
        super().__init__(message)
        self.receipt = dict(receipt)


@dataclass(frozen=True)
class Task:
    task_id: int
    starting_url: str
    category: str
    task: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "starting_url": self.starting_url,
            "category": self.category,
            "task": self.task,
        }


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation for receipt hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() not in PLACEHOLDERS and bool(
        value.strip()
    )


def _immutable_revision(value: Any) -> bool:
    return isinstance(value, str) and bool(HEX40_RE.fullmatch(value.strip()))


def _sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(HEX64_RE.fullmatch(value.strip()) or SHA256_DIGEST_RE.fullmatch(value.strip()))


def _https_url(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("https://")


def _web_url(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(("http://", "https://"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id_hash(task_ids: Iterable[int]) -> str:
    """Hash a sorted, duplicate-free task-ID set."""

    ids = [int(value) for value in task_ids]
    if len(ids) != len(set(ids)):
        raise WebBenchError("task IDs must be unique")
    if any(value < 0 for value in ids):
        raise WebBenchError("task IDs must be non-negative")
    return sha256_text("\n".join(str(value) for value in sorted(ids)))


def task_manifest_hash(tasks: Iterable[Task]) -> str:
    """Hash task identity/content without retaining credentials or trajectories."""

    ordered = [task.as_dict() for task in sorted(tasks, key=lambda item: item.task_id)]
    return sha256_json(ordered)


def load_tasks(path: Path, *, expected_sha256: str | None = None) -> list[Task]:
    """Load and validate the exact Halluminate CSV, including multiline tasks."""

    path = Path(path)
    if not path.is_file():
        raise WebBenchError(f"WebBench dataset file is missing: {path}")
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None and actual_sha256.lower() != expected_sha256.lower():
        raise WebBenchError(
            f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            if fieldnames != REQUIRED_COLUMNS:
                raise WebBenchError(
                    f"dataset columns must be {REQUIRED_COLUMNS!r}, got {fieldnames!r}"
                )
            rows: list[Task] = []
            for line, row in enumerate(reader, start=2):
                try:
                    task_id = int(str(row["ID"]).strip())
                except (TypeError, ValueError) as exc:
                    raise WebBenchError(f"row {line} has a non-integer ID") from exc
                starting_url = str(row["Starting URL"] or "").strip()
                category = str(row["Category"] or "").strip().upper()
                task_text = str(row["Task"] or "").strip()
                if task_id < 0 or not _web_url(starting_url):
                    raise WebBenchError(f"row {line} has invalid ID or Starting URL")
                if category not in ALLOWED_CATEGORIES:
                    raise WebBenchError(f"row {line} has unsupported category {category!r}")
                if not task_text:
                    raise WebBenchError(f"row {line} has an empty task")
                rows.append(Task(task_id, starting_url, category, task_text))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise WebBenchError(f"could not read WebBench CSV {path}: {exc}") from exc
    if not rows:
        raise WebBenchError("WebBench CSV has no tasks")
    ids = [task.task_id for task in rows]
    if len(ids) != len(set(ids)):
        raise WebBenchError("WebBench CSV contains duplicate task IDs")
    return rows


def select_tasks(tasks: Sequence[Task], task_ids: Sequence[int] | None = None, *, limit: int = 1) -> list[Task]:
    """Select the smallest exact subset deterministically (default: one task)."""

    by_id = {task.task_id: task for task in tasks}
    if task_ids:
        requested = [int(value) for value in task_ids]
        if len(requested) != len(set(requested)):
            raise WebBenchError("requested task IDs must be unique")
        missing = sorted(set(requested) - set(by_id))
        if missing:
            raise WebBenchError(f"requested task IDs are absent from WebBench: {missing}")
        return [by_id[value] for value in sorted(requested)]
    if isinstance(limit, bool) or limit < 1:
        raise WebBenchError("limit must be a positive integer")
    return [by_id[value] for value in sorted(by_id)[:limit]]


def load_task_id_manifest(path: Path) -> tuple[list[int], str | None]:
    """Load an explicit training-ID manifest; an empty list is valid and explicit."""

    path = Path(path)
    if not path.is_file():
        raise WebBenchError(f"training task-ID manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebBenchError(f"training task-ID manifest is malformed: {path}: {exc}") from exc
    if isinstance(value, list):
        raw_ids = value
        declared_hash = None
    elif isinstance(value, Mapping):
        raw_ids = value.get("task_ids")
        declared_hash = value.get("task_id_hash")
    else:
        raise WebBenchError("training task-ID manifest must be a list or object")
    if not isinstance(raw_ids, list):
        raise WebBenchError("training task-ID manifest must contain a task_ids list")
    try:
        ids = [int(item) for item in raw_ids]
    except (TypeError, ValueError) as exc:
        raise WebBenchError("training task-ID manifest contains a non-integer ID") from exc
    computed = task_id_hash(ids)
    if declared_hash is not None:
        if not _sha256(declared_hash) or str(declared_hash).replace("sha256:", "").lower() != computed:
            raise WebBenchError(
                f"training task-ID manifest hash mismatch: expected {computed}, got {declared_hash}"
            )
    return ids, computed


def _hash_matches(value: Any, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower().removeprefix("sha256:") == expected.lower()


def validate_disjoint_ids(eval_ids: Sequence[int], train_ids: Sequence[int]) -> list[str]:
    errors: list[str] = []
    eval_set = set(int(value) for value in eval_ids)
    train_set = set(int(value) for value in train_ids)
    if len(eval_set) != len(eval_ids):
        errors.append("evaluation task IDs are not unique")
    if len(train_set) != len(train_ids):
        errors.append("training task IDs are not unique")
    overlap = sorted(eval_set & train_set)
    if overlap:
        errors.append(f"training/evaluation task-ID overlap: {overlap}")
    return errors


def validate_budget(*, projected_tinker_usd: float, global_budget: Mapping[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    try:
        projected = float(projected_tinker_usd)
    except (TypeError, ValueError):
        errors.append("projected Tinker spend must be finite")
    else:
        if projected < 0 or projected != projected or projected == float("inf"):
            errors.append("projected Tinker spend must be finite and non-negative")
        elif projected > SUITE_TINKER_MAX_USD + 1e-12:
            errors.append(
                f"projected Tinker spend ${projected:.6f} exceeds E6 maximum ${SUITE_TINKER_MAX_USD:.2f}"
            )
    record = global_budget or {
        "maximum_usd": GLOBAL_MAX_USD,
        "operational_cap_usd": GLOBAL_OPERATIONAL_CAP_USD,
        "safety_reserve_usd": GLOBAL_SAFETY_RESERVE_USD,
    }
    expected = {
        "maximum_usd": GLOBAL_MAX_USD,
        "operational_cap_usd": GLOBAL_OPERATIONAL_CAP_USD,
        "safety_reserve_usd": GLOBAL_SAFETY_RESERVE_USD,
    }
    for key, value in expected.items():
        try:
            actual = float(record.get(key))
        except (TypeError, ValueError):
            errors.append(f"global budget {key} is missing or invalid")
            continue
        if abs(actual - value) > 1e-12:
            errors.append(f"global budget {key} must remain {value:.2f}, got {actual}")
    if abs(expected["maximum_usd"] - expected["operational_cap_usd"] - expected["safety_reserve_usd"]) > 1e-12:
        errors.append("global budget cap/reserve arithmetic is inconsistent")
    return errors


def estimate_tinker_cost(task_count: int, max_prompt_tokens: int = 2048, max_response_tokens: int = 256) -> float:
    """Conservative uncached estimate used before an optional Tinker seam."""

    if task_count < 1 or max_prompt_tokens < 1 or max_response_tokens < 1:
        raise WebBenchError("token/task counts must be positive")
    return round(
        (
            task_count * max_prompt_tokens * PREFILL_USD_PER_MILLION
            + task_count * max_response_tokens * SAMPLE_USD_PER_MILLION
        )
        / 1_000_000,
        9,
    )


def _load_json(path: Path | None, label: str) -> Mapping[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise WebBenchError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebBenchError(f"{label} is malformed: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise WebBenchError(f"{label} must be a JSON object")
    return value


def validate_native_receipt(receipt: Mapping[str, Any] | None) -> list[str]:
    """Validate the external Halluminate environment/verifier access receipt."""

    if receipt is None:
        return ["official Halluminate environment/verifier access receipt is missing"]
    errors: list[str] = []
    if receipt.get("provider") not in {"Halluminate", "Halluminate/WebBench"}:
        errors.append("native receipt provider must be Halluminate")
    if receipt.get("benchmark") != BENCHMARK_NAME:
        errors.append(f"native receipt benchmark must be {BENCHMARK_NAME}")
    if receipt.get("approved") is not True:
        errors.append("native receipt approved must be true")
    if not _nonempty(receipt.get("access_receipt_id")):
        errors.append("native receipt access_receipt_id is required")
    env = receipt.get("environment")
    if not isinstance(env, Mapping):
        errors.append("native receipt environment object is required")
    else:
        for key in ("environment_id", "environment_revision", "container_image_digest", "browser_revision"):
            if not _nonempty(env.get(key)):
                errors.append(f"native receipt environment.{key} is required")
        if not _immutable_revision(env.get("environment_revision")):
            errors.append("native receipt environment.environment_revision must be a 40-hex revision")
        if not SHA256_DIGEST_RE.fullmatch(str(env.get("container_image_digest", ""))):
            errors.append("native receipt environment.container_image_digest must be sha256:<64 hex>")
        for key in ("screenshot_capture", "dom_capture", "task_reset"):
            if env.get(key) is not True:
                errors.append(f"native receipt environment.{key} must be true")
        if not _nonempty(env.get("credential_scope")):
            errors.append("native receipt environment.credential_scope is required")
    verifier = receipt.get("native_verifier")
    if not isinstance(verifier, Mapping):
        errors.append("native receipt native_verifier object is required")
    else:
        if verifier.get("available") is not True:
            errors.append("native_verifier.available must be true")
        if verifier.get("ground_truth_available") is not True:
            errors.append("native_verifier.ground_truth_available must be true")
        for key in ("verifier_id", "verifier_revision", "verifier_sha256", "command"):
            if verifier.get(key) in (None, "", []):
                errors.append(f"native_verifier.{key} is required")
        if not _immutable_revision(verifier.get("verifier_revision")):
            errors.append("native_verifier.verifier_revision must be a 40-hex revision")
        if not _sha256(verifier.get("verifier_sha256")):
            errors.append("native_verifier.verifier_sha256 must be a SHA-256 digest")
        command = verifier.get("command")
        if not isinstance(command, list) or not all(_nonempty(item) for item in command):
            errors.append("native_verifier.command must be a non-empty argv list")
        if not _https_url(verifier.get("receipt_url")):
            errors.append("native_verifier.receipt_url must be an HTTPS access URL")
    return errors


def validate_model_receipt(
    *,
    model_id: str,
    base_model_revision: str,
    sampler: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if model_id != MODEL_ID:
        errors.append(f"model_id must be exactly {MODEL_ID}")
    if not _immutable_revision(base_model_revision):
        errors.append("base_model_revision must be an immutable 40-hex HF commit")
    if sampler is None:
        return (
            {
                "mode": "base_model",
                "reason": "no completed compatible Tinker sampler checkpoint supplied",
                "model_id": model_id,
                "evaluated_hf_commit": base_model_revision,
                "evaluated_hf_commit_url": f"https://huggingface.co/{model_id}/commit/{base_model_revision}",
            },
            errors,
        )
    if sampler.get("completed") is not True or sampler.get("compatible") is not True:
        errors.append("sampler receipt must mark completed=true and compatible=true")
    if sampler.get("model_id") != MODEL_ID:
        errors.append("sampler receipt model_id must match the experiment model")
    if not _immutable_revision(sampler.get("hf_commit")):
        errors.append("sampler receipt hf_commit must be an immutable 40-hex commit")
    if not _nonempty(sampler.get("hf_repo")) or not str(sampler.get("hf_repo", "")).startswith("https://huggingface.co/"):
        errors.append("sampler receipt hf_repo must be an HTTPS Hugging Face URL")
    if not _nonempty(sampler.get("tinker_run_id")):
        errors.append("sampler receipt tinker_run_id is required")
    if sampler.get("public") is not True:
        errors.append("evaluated sampler HF commit must be public")
    selected = dict(sampler)
    selected["mode"] = "completed_compatible_tinker_sampler"
    selected["evaluated_hf_commit"] = sampler.get("hf_commit")
    selected["evaluated_hf_commit_url"] = f"{str(sampler.get('hf_repo')).rstrip('/')}/commit/{sampler.get('hf_commit')}"
    return selected, errors


def choose_strongest_sampler(
    candidates: Sequence[Mapping[str, Any]], *, model_id: str = MODEL_ID
) -> Mapping[str, Any] | None:
    """Choose the highest-step completed compatible public sampler deterministically."""

    compatible = [
        candidate
        for candidate in candidates
        if candidate.get("completed") is True
        and candidate.get("compatible") is True
        and candidate.get("model_id") == model_id
        and candidate.get("public") is True
        and _immutable_revision(candidate.get("hf_commit"))
    ]
    if not compatible:
        return None
    return max(
        compatible,
        key=lambda candidate: (
            int(candidate.get("step", 0)) if str(candidate.get("step", "0")).isdigit() else 0,
            str(candidate.get("hf_commit")),
        ),
    )


def validate_isolated_environment(worktree_root: Path, env_path: Path) -> list[str]:
    """Ensure benchmark dependencies live in this worktree, never globally."""

    errors: list[str] = []
    root = Path(worktree_root).resolve()
    path = Path(env_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        errors.append(f"benchmark environment must be inside worktree: {path}")
        return errors
    if path == root:
        errors.append("benchmark environment cannot be the worktree root")
        return errors
    if not (path / "pyvenv.cfg").is_file():
        errors.append(f"isolated benchmark environment is missing pyvenv.cfg: {path}")
    python = path / "bin" / "python"
    if not python.is_file():
        python = path / "Scripts" / "python.exe"
    if not python.is_file():
        errors.append(f"isolated benchmark environment has no interpreter: {path}")
    return errors


def validate_wandb_receipt(receipt: Mapping[str, Any] | None) -> list[str]:
    if receipt is None:
        return ["online W&B initialization receipt is missing"]
    errors: list[str] = []
    if receipt.get("online") is not True or receipt.get("mode") != "online":
        errors.append("W&B receipt must explicitly be online")
    if receipt.get("initialized_before_tinker") is not True:
        errors.append("W&B receipt must prove initialization before any Tinker call")
    if receipt.get("entity") != PAVLOV_WANDB_ENTITY:
        errors.append("W&B entity does not match the Pavlov namespace")
    if receipt.get("project") != PAVLOV_WANDB_PROJECT:
        errors.append("W&B project does not match the Pavlov namespace")
    if not _nonempty(receipt.get("group")) or not _nonempty(receipt.get("run_id")):
        errors.append("W&B group and run_id are required")
    if not _https_url(receipt.get("run_url")):
        errors.append("W&B run_url must be HTTPS")
    return errors


def build_gate(
    gate_id: str,
    passed: bool,
    *,
    evidence: Mapping[str, Any] | None = None,
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "status": "PASS" if passed else "BLOCKED",
        "evidence": dict(evidence or {}),
        "blockers": list(blockers),
    }


def _required_external_receipts() -> list[dict[str, Any]]:
    return [
        {
            "id": "halluminate_webbench_environment_access",
            "provider": "Halluminate",
            "required_fields": [
                "approved=true",
                "access_receipt_id",
                "environment.environment_id",
                "environment.environment_revision (immutable)",
                "environment.container_image_digest (sha256)",
                "environment.browser_revision",
                "environment.screenshot_capture=true",
                "environment.dom_capture=true",
                "environment.task_reset=true",
                "environment.credential_scope",
            ],
            "why": "The public repository contains task CSVs, not the official live browser environment, credentials, or reset contract.",
        },
        {
            "id": "halluminate_webbench_native_verifier",
            "provider": "Halluminate",
            "evidence_url": GROUND_TRUTH_ISSUE_URL,
            "required_fields": [
                "native_verifier.available=true",
                "native_verifier.ground_truth_available=true",
                "native_verifier.verifier_id",
                "native_verifier.verifier_revision (immutable)",
                "native_verifier.verifier_sha256",
                "native_verifier.command",
                "native_verifier.receipt_url",
            ],
            "why": "The official repository's open Ground Truth issue requests the missing ground truth/script; no native verifier is public.",
        },
        {
            "id": "webbench_task_authorization",
            "provider": "Halluminate",
            "required_fields": [
                "allowed_task_ids or explicit task-ID manifest",
                "credential_scope",
                "side_effect_policy",
                "terms_or_license_signoff",
            ],
            "why": "WRITE tasks can mutate live websites; a launch needs an explicit authorization/credential scope rather than inferred permission.",
        },
    ]


def build_receipt(
    *,
    worktree_root: Path,
    dataset_path: Path | None,
    task_ids: Sequence[int] | None,
    training_task_ids: Sequence[int] | None,
    training_task_id_hash: str | None,
    model_id: str = MODEL_ID,
    base_model_revision: str = BASE_MODEL_REVISION,
    sampler: Mapping[str, Any] | None = None,
    native_receipt: Mapping[str, Any] | None = None,
    wandb_receipt: Mapping[str, Any] | None = None,
    env_path: Path | None = None,
    projected_tinker_usd: float | None = None,
    global_budget: Mapping[str, Any] | None = None,
    agent_command: Sequence[str] | None = None,
    launch_requested: bool = False,
) -> dict[str, Any]:
    """Build a receipt without network calls, model loading, or provider writes."""

    worktree_root = Path(worktree_root).resolve()
    blockers: list[str] = []
    gates: list[dict[str, Any]] = []
    dataset_evidence: dict[str, Any] = {
        "source_url": BENCHMARK_REPO_URL,
        "revision": BENCHMARK_REVISION,
        "revision_url": BENCHMARK_REVISION_URL,
        "dataset_file": DATASET_FILE,
        "dataset_raw_url": DATASET_RAW_URL,
        "license": BENCHMARK_LICENSE,
        "license_url": LICENSE_URL,
        "license_sha256": LICENSE_SHA256,
        "expected_dataset_sha256": DATASET_SHA256,
        "expected_public_task_count": PUBLIC_TASK_COUNT,
        "expected_public_task_id_hash": PUBLIC_TASK_ID_HASH,
        "expected_public_manifest_hash": PUBLIC_MANIFEST_HASH,
    }
    tasks: list[Task] = []
    if dataset_path is None:
        dataset_blockers = [f"exact WebBench dataset file is not acquired locally: {DATASET_FILE}"]
        gates.append(build_gate("authoritative_dataset", False, evidence=dataset_evidence, blockers=dataset_blockers))
        blockers.extend(dataset_blockers)
    else:
        try:
            tasks = load_tasks(dataset_path, expected_sha256=DATASET_SHA256)
        except WebBenchError as exc:
            dataset_blockers = [str(exc)]
            gates.append(build_gate("authoritative_dataset", False, evidence=dataset_evidence, blockers=dataset_blockers))
            blockers.extend(dataset_blockers)
        else:
            dataset_evidence.update(
                {
                    "path": str(Path(dataset_path).resolve()),
                    "dataset_sha256": file_sha256(dataset_path),
                    "task_count": len(tasks),
                    "task_id_hash": task_id_hash([task.task_id for task in tasks]),
                    "manifest_hash": task_manifest_hash(tasks),
                }
            )
            dataset_ok = (
                len(tasks) == PUBLIC_TASK_COUNT
                and dataset_evidence["task_id_hash"] == PUBLIC_TASK_ID_HASH
                and dataset_evidence["manifest_hash"] == PUBLIC_MANIFEST_HASH
            )
            dataset_blockers = [] if dataset_ok else [
                "local WebBench task manifest does not match the pinned authoritative CSV"
            ]
            gates.append(build_gate("authoritative_dataset", dataset_ok, evidence=dataset_evidence, blockers=dataset_blockers))
            blockers.extend(dataset_blockers)

    selected: list[Task] = []
    if tasks:
        try:
            selected = select_tasks(tasks, task_ids, limit=1)
        except WebBenchError as exc:
            blockers.append(str(exc))
    selected_ids = [task.task_id for task in selected]
    split_evidence: dict[str, Any] = {
        "evaluation_task_ids": selected_ids,
        "evaluation_task_id_hash": task_id_hash(selected_ids) if selected else None,
        "training_task_ids": list(training_task_ids) if training_task_ids is not None else None,
        "training_task_id_hash": training_task_id_hash,
    }
    split_blockers: list[str] = []
    if training_task_ids is None or training_task_id_hash is None:
        split_blockers.append("explicit training task-ID manifest/hash is required for disjointness")
    else:
        if not _hash_matches(training_task_id_hash, task_id_hash(training_task_ids)):
            split_blockers.append("training task-ID hash does not match the supplied manifest")
        if selected:
            split_blockers.extend(validate_disjoint_ids(selected_ids, training_task_ids))
    split_ok = bool(selected) and not split_blockers
    gates.append(build_gate("disjoint_task_ids", split_ok, evidence=split_evidence, blockers=split_blockers))
    blockers.extend(split_blockers)

    env_path = env_path or worktree_root / ".venv-webbench"
    env_blockers = validate_isolated_environment(worktree_root, env_path)
    gates.append(
        build_gate(
            "isolated_benchmark_environment",
            not env_blockers,
            evidence={"path": str(Path(env_path).resolve()), "scope": "per-worktree"},
            blockers=env_blockers,
        )
    )
    blockers.extend(env_blockers)

    selected_model, model_blockers = validate_model_receipt(
        model_id=model_id, base_model_revision=base_model_revision, sampler=sampler
    )
    gates.append(build_gate("model_or_sampler_binding", not model_blockers, evidence=selected_model, blockers=model_blockers))
    blockers.extend(model_blockers)

    native_blockers = validate_native_receipt(native_receipt)
    gates.append(
        build_gate(
            "official_native_environment_and_verifier",
            not native_blockers,
            evidence=native_receipt or {"status": "not_acquired"},
            blockers=native_blockers,
        )
    )
    blockers.extend(native_blockers)

    projected = estimate_tinker_cost(len(selected)) if projected_tinker_usd is None and selected else (projected_tinker_usd or 0.0)
    budget_blockers = validate_budget(projected_tinker_usd=projected, global_budget=global_budget)
    gates.append(
        build_gate(
            "budget_cap",
            not budget_blockers,
            evidence={
                "suite_max_usd": SUITE_TINKER_MAX_USD,
                "projected_tinker_usd": projected,
                "global_max_usd": GLOBAL_MAX_USD,
                "global_operational_cap_usd": GLOBAL_OPERATIONAL_CAP_USD,
                "global_safety_reserve_usd": GLOBAL_SAFETY_RESERVE_USD,
            },
            blockers=budget_blockers,
        )
    )
    blockers.extend(budget_blockers)

    # In launch mode the runner is responsible for creating an online W&B run
    # after static gates pass.  A supplied receipt is still validated, but its
    # absence is not a blocker before that initialization seam runs.
    wandb_blockers = validate_wandb_receipt(wandb_receipt) if wandb_receipt is not None else []
    gates.append(
        build_gate(
            "wandb_online_before_tinker",
            not wandb_blockers,
            evidence=wandb_receipt
            or {
                "required": True,
                "initialized": False,
                "will_initialize_before_tinker": launch_requested,
            },
            blockers=wandb_blockers,
        )
    )
    blockers.extend(wandb_blockers)

    agent_blockers: list[str] = []
    if launch_requested and (not agent_command or not all(_nonempty(item) for item in agent_command)):
        agent_blockers.append("launch requires an agent command that emits screenshot/DOM trajectories")
    gates.append(build_gate("native_agent_command", not agent_blockers, evidence={"command": list(agent_command or [])}, blockers=agent_blockers))
    blockers.extend(agent_blockers)

    status = "READY" if not blockers else "BLOCKED"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "suite_id": SUITE_ID,
        "benchmark": {
            "name": BENCHMARK_NAME,
            "source_url": BENCHMARK_REPO_URL,
            "revision": BENCHMARK_REVISION,
            "license": BENCHMARK_LICENSE,
            "dataset_file": DATASET_FILE,
            "task_count": len(selected),
            "task_ids": selected_ids,
            "task_id_hash": task_id_hash(selected_ids) if selected else None,
            "manifest_hash": task_manifest_hash(selected) if selected else None,
            "full_public_task_id_hash": PUBLIC_TASK_ID_HASH,
            "full_public_manifest_hash": PUBLIC_MANIFEST_HASH,
        },
        "experiment": {
            "evaluation_mode": "eval_only",
            "model_id": model_id,
            "base_model_revision": base_model_revision,
            "model_binding": selected_model,
            "screenshots_required": True,
            "dom_required": True,
            "native_verifier_required": True,
            "substitution_policy": "related_browser_benchmarks_forbidden",
        },
        "split": split_evidence,
        "budget": {
            "suite_tinker_max_usd": SUITE_TINKER_MAX_USD,
            "projected_tinker_usd": projected,
            "global_max_usd": GLOBAL_MAX_USD,
            "global_operational_cap_usd": GLOBAL_OPERATIONAL_CAP_USD,
            "global_safety_reserve_usd": GLOBAL_SAFETY_RESERVE_USD,
        },
        "gates": gates,
        "blockers": blockers,
        "external_access_receipt_needed": _required_external_receipts() if blockers else [],
        "launch": {
            "requested": launch_requested,
            "attempted": False,
            "tinker_call_count": 0,
            "score": None,
            "result_rows": [],
        },
        "evidence_boundary": {
            "score_is_claimable": False,
            "reason": "No score is accepted without the official native verifier receipt and complete trajectory evidence." if blockers else "Awaiting launch receipt.",
            "human_annotation_is_not_native_verification": True,
            "ground_truth_issue_url": GROUND_TRUTH_ISSUE_URL,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def start_wandb_online(config: Mapping[str, Any], wandb_module: Any | None = None) -> dict[str, Any]:
    """Initialize W&B online; this is intentionally the first provider side effect."""

    if os.environ.get("WANDB_MODE", "").strip().lower() in {"offline", "disabled"}:
        raise WebBenchError("WANDB_MODE=offline/disabled is forbidden for E6 launch")
    if wandb_module is None:
        try:
            import wandb as wandb_module  # type: ignore[import-not-found]
        except ImportError as exc:
            raise WebBenchError("wandb is required in the isolated benchmark environment") from exc
    run = wandb_module.init(
        entity=PAVLOV_WANDB_ENTITY,
        project=PAVLOV_WANDB_PROJECT,
        group=PAVLOV_WANDB_GROUP,
        mode="online",
        config=dict(config),
    )
    if run is None or not _nonempty(getattr(run, "id", None)):
        raise WebBenchError("wandb.init(mode='online') did not return a run identity")
    run_url = getattr(run, "url", None)
    if not _https_url(run_url):
        raise WebBenchError("W&B online run did not expose an HTTPS run URL")
    return {
        "online": True,
        "mode": "online",
        "initialized_before_tinker": True,
        "entity": PAVLOV_WANDB_ENTITY,
        "project": PAVLOV_WANDB_PROJECT,
        "group": PAVLOV_WANDB_GROUP,
        "run_id": str(run.id),
        "run_url": str(run_url),
    }


def _run_command(command: Sequence[str], *, env: Mapping[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    if not command or not all(_nonempty(item) for item in command):
        raise WebBenchError("external command must be a non-empty argv list")
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        check=False,
        text=True,
        capture_output=True,
    )


def run_native_evaluation(
    *,
    tasks: Sequence[Task],
    agent_command: Sequence[str],
    verifier_command: Sequence[str],
    worktree_root: Path,
    output_dir: Path,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run an official agent/verifier command pair and require native evidence."""

    if not tasks:
        raise WebBenchError("native evaluation requires at least one task")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_file = output_dir / "tasks.jsonl"
    trajectory_file = output_dir / "trajectories.jsonl"
    verdict_file = output_dir / "verdicts.jsonl"
    task_file.write_text("\n".join(canonical_json(task.as_dict()) for task in tasks) + "\n", encoding="utf-8")
    child_env = os.environ.copy()
    child_env.update({str(key): str(value) for key, value in (env or {}).items()})
    child_env.update(
        {
            "WEBBENCH_TASK_FILE": str(task_file),
            "WEBBENCH_TRAJECTORY_FILE": str(trajectory_file),
            "WEBBENCH_REQUIRE_SCREENSHOTS": "1",
            "WEBBENCH_REQUIRE_DOM": "1",
            "WEBBENCH_SUITE_ID": SUITE_ID,
        }
    )
    agent = _run_command(agent_command, env=child_env, cwd=Path(worktree_root))
    if agent.returncode != 0:
        raise WebBenchError(f"native agent command failed ({agent.returncode}): {agent.stderr[-1000:]}")
    if not trajectory_file.is_file():
        raise WebBenchError("native agent did not emit trajectories.jsonl")
    child_env["WEBBENCH_VERDICT_FILE"] = str(verdict_file)
    verifier_env = dict(child_env)
    verifier_env["WEBBENCH_TRAJECTORY_FILE"] = str(trajectory_file)
    verifier = _run_command(verifier_command, env=verifier_env, cwd=Path(worktree_root))
    if verifier.returncode != 0:
        raise WebBenchError(f"native verifier command failed ({verifier.returncode}): {verifier.stderr[-1000:]}")
    if not verdict_file.is_file():
        raise WebBenchError("native verifier did not emit verdicts.jsonl")
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(verdict_file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise WebBenchError(f"verdict line {line_number} is not an object")
            rows.append(dict(value))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WebBenchError(f"native verdict output is malformed: {exc}") from exc
    by_id = {int(row["task_id"]): row for row in rows if "task_id" in row}
    expected_ids = {task.task_id for task in tasks}
    if set(by_id) != expected_ids:
        raise WebBenchError(f"native verifier task IDs do not match requested IDs: expected {sorted(expected_ids)}, got {sorted(by_id)}")
    for task_id, row in by_id.items():
        if not isinstance(row.get("success"), bool):
            raise WebBenchError(f"native verdict for task {task_id} lacks boolean success")
        if not _sha256(row.get("screenshot_sha256")) or not _sha256(row.get("dom_sha256")):
            raise WebBenchError(f"native verdict for task {task_id} lacks screenshot/DOM hashes")
        if row.get("verifier") != "Halluminate/WebBench/native":
            raise WebBenchError(f"task {task_id} was not scored by the native Halluminate verifier")
    return [by_id[task.task_id] for task in tasks]


def launch(
    *,
    receipt: Mapping[str, Any],
    tasks: Sequence[Task],
    agent_command: Sequence[str],
    verifier_command: Sequence[str],
    worktree_root: Path,
    output_dir: Path,
    wandb_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    tinker_callback: Callable[[], Any] | None = None,
    native_runner: Callable[..., list[dict[str, Any]]] = run_native_evaluation,
) -> dict[str, Any]:
    """Launch only after a READY receipt; W&B precedes any Tinker callback."""

    if receipt.get("status") != "READY":
        raise LaunchBlocked("E6 launch is blocked by preflight", receipt)
    if not tasks:
        raise WebBenchError("launch requires selected tasks")
    config = {
        "campaign": "pavlov-e6-webbench",
        "suite_id": SUITE_ID,
        "model_id": receipt["experiment"]["model_id"],
        "base_model_revision": receipt["experiment"]["base_model_revision"],
        "evaluated_hf_commit": receipt["experiment"]["model_binding"]["evaluated_hf_commit"],
        "dataset_revision": BENCHMARK_REVISION,
        "dataset_manifest_hash": receipt["benchmark"]["manifest_hash"],
        "task_id_hash": receipt["benchmark"]["task_id_hash"],
        "budget_cap_usd": SUITE_TINKER_MAX_USD,
    }
    wandb_record = wandb_factory(config) if wandb_factory else start_wandb_online(config)
    if not isinstance(wandb_record, Mapping):
        raise WebBenchError("W&B factory did not return a receipt")
    wandb_errors = validate_wandb_receipt(wandb_record)
    if wandb_errors:
        raise WebBenchError("invalid online W&B receipt: " + "; ".join(wandb_errors))
    # A Tinker callback is optional for this eval-only lane.  If one is
    # supplied, W&B has already been initialized and the callback is bounded by
    # the preflight projected cost.
    tinker_result = None
    tinker_call_count = 0
    if tinker_callback is not None:
        tinker_result = tinker_callback()
        tinker_call_count = 1
    rows = native_runner(
        tasks=tasks,
        agent_command=agent_command,
        verifier_command=verifier_command,
        worktree_root=worktree_root,
        output_dir=output_dir,
    )
    score = sum(1 for row in rows if row["success"]) / len(rows)
    updated = json.loads(json.dumps(receipt))
    updated["status"] = "COMPLETED"
    updated["wandb_run_identity"] = dict(wandb_record)
    updated["launch"] = {
        "requested": True,
        "attempted": True,
        "tinker_call_count": tinker_call_count,
        "tinker_result": tinker_result,
        "score": score,
        "result_rows": rows,
    }
    updated["evidence_boundary"] = {
        "score_is_claimable": True,
        "reason": "The official native verifier returned one evidence-bound verdict per requested task.",
        "human_annotation_is_not_native_verification": True,
        "ground_truth_issue_url": GROUND_TRUTH_ISSUE_URL,
    }
    return updated


def _parse_task_ids(values: Sequence[str] | None) -> list[int] | None:
    if not values:
        return None
    ids: list[int] = []
    for value in values:
        for item in value.split(","):
            if not item.strip():
                continue
            try:
                ids.append(int(item.strip()))
            except ValueError as exc:
                raise WebBenchError(f"invalid task ID {item!r}") from exc
    return ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="pinned webbenchfinal.csv checkout")
    parser.add_argument("--training-task-manifest", type=Path, help="JSON task-ID manifest used for disjointness")
    parser.add_argument("--task-id", action="append", help="exact task ID(s), repeatable or comma-separated")
    parser.add_argument("--env", type=Path, help="per-worktree virtualenv (default: .venv-webbench)")
    parser.add_argument("--native-receipt", type=Path, help="Halluminate environment/native-verifier receipt JSON")
    parser.add_argument("--wandb-receipt", type=Path, help="pre-initialized online W&B receipt JSON")
    parser.add_argument("--sampler-receipt", type=Path, help="completed compatible public HF sampler receipt JSON")
    parser.add_argument("--model-revision", default=BASE_MODEL_REVISION)
    parser.add_argument("--projected-tinker-usd", type=float, default=None)
    parser.add_argument("--agent-command", nargs="+", help="official environment agent command")
    parser.add_argument("--verifier-command", nargs="+", help="official native verifier command")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e6-webbench"))
    parser.add_argument("--receipt", type=Path, default=Path("zvf-program/flagship/webbench_eval_receipt.json"))
    parser.add_argument("--launch", action="store_true", help="launch only after every gate passes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    worktree_root = Path(__file__).resolve().parents[2]
    try:
        dataset = args.dataset
        training_ids: list[int] | None = None
        training_hash: str | None = None
        if args.training_task_manifest:
            training_ids, training_hash = load_task_id_manifest(args.training_task_manifest)
        native_receipt = _load_json(args.native_receipt, "native receipt")
        wandb_receipt = _load_json(args.wandb_receipt, "W&B receipt")
        sampler_receipt = _load_json(args.sampler_receipt, "sampler receipt")
        task_ids = _parse_task_ids(args.task_id)
        receipt = build_receipt(
            worktree_root=worktree_root,
            dataset_path=dataset,
            task_ids=task_ids,
            training_task_ids=training_ids,
            training_task_id_hash=training_hash,
            base_model_revision=args.model_revision,
            sampler=sampler_receipt,
            native_receipt=native_receipt,
            wandb_receipt=wandb_receipt,
            env_path=args.env,
            projected_tinker_usd=args.projected_tinker_usd,
            agent_command=args.agent_command,
            launch_requested=args.launch,
        )
        if args.launch:
            if receipt["status"] != "READY":
                _write_json(args.receipt, receipt)
                print(json.dumps(receipt, indent=2, sort_keys=True))
                return EXIT_BLOCKED
            tasks = load_tasks(dataset, expected_sha256=DATASET_SHA256) if dataset else []
            tasks = select_tasks(tasks, task_ids, limit=1)
            native = native_receipt or {}
            verifier = native.get("native_verifier", {})
            verifier_command = args.verifier_command or verifier.get("command")
            if not isinstance(verifier_command, list) or not verifier_command:
                raise WebBenchError("native verifier command is missing")
            if not args.agent_command:
                raise WebBenchError("--agent-command is required for launch")
            receipt = launch(
                receipt=receipt,
                tasks=tasks,
                agent_command=args.agent_command,
                verifier_command=verifier_command,
                worktree_root=worktree_root,
                output_dir=args.output_dir,
            )
        _write_json(args.receipt, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return EXIT_OK if receipt["status"] in {"READY", "COMPLETED"} else EXIT_BLOCKED
    except LaunchBlocked as exc:
        _write_json(args.receipt, exc.receipt)
        print(json.dumps(exc.receipt, indent=2, sort_keys=True))
        return EXIT_BLOCKED
    except WebBenchError as exc:
        print(f"E6 ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_CATEGORIES",
    "BASE_MODEL_REVISION",
    "BENCHMARK_REVISION",
    "DATASET_SHA256",
    "GLOBAL_OPERATIONAL_CAP_USD",
    "GLOBAL_SAFETY_RESERVE_USD",
    "MODEL_ID",
    "PUBLIC_MANIFEST_HASH",
    "PUBLIC_TASK_COUNT",
    "PUBLIC_TASK_ID_HASH",
    "SUITE_ID",
    "SUITE_TINKER_MAX_USD",
    "Task",
    "WebBenchError",
    "build_receipt",
    "canonical_json",
    "choose_strongest_sampler",
    "estimate_tinker_cost",
    "file_sha256",
    "launch",
    "load_task_id_manifest",
    "load_tasks",
    "main",
    "run_native_evaluation",
    "select_tasks",
    "sha256_json",
    "task_id_hash",
    "task_manifest_hash",
    "validate_budget",
    "validate_disjoint_ids",
    "validate_isolated_environment",
    "validate_model_receipt",
    "validate_native_receipt",
    "validate_wandb_receipt",
]
