#!/usr/bin/env python3
"""Fail-closed runner for the exact Pavlov E3 SDAB evaluation.

SDAB is not a prompt-only benchmark.  The official environment starts a live
software system, gives the agent operational tooling, and grades the resulting
state with a provider-owned verifier.  This runner therefore accepts only a
provider-issued, metadata-only runtime manifest and a native adapter.  Related
benchmarks, synthetic tasks, local containers, and guessed verifiers are
rejected before W&B or Tinker is touched.

The command is safe to run in an ordinary checkout::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=zvf-program \\
      python3 -m flagship.eval_pavlov_sdab \\
      --preflight --out autoresearch/orchestrator-260809-0922/sdab_e3_launch_receipt.json

When the exact provider bundle has been granted, the same entry point can run
the smallest one-task evaluation with ``--run``.  The native adapter protocol
is documented in :func:`load_native_runtime`; it deliberately carries no raw
task prompts or targets in the local receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Protocol, Sequence


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

SDAB_SUITE_ID = "sdab_eval"
SDAB_BENCHMARK_NAME = "Software Development Automation Benchmark"
SDAB_PROVIDER = "Emulated"
SDAB_OFFICIAL_URL = "https://emulated.so/sdab"
SDAB_TASK_COUNT = 80
SDAB_EVALUATION_SPLIT = "evaluation"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
# Read-only Hugging Face metadata observed on 2026-08-09.  A caller may pass a
# different immutable commit explicitly, but floating ``main``/``latest`` is
# never accepted.
DEFAULT_BASE_MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"

PREFILL_USD_PER_MILLION = Decimal("0.54")
SAMPLE_USD_PER_MILLION = Decimal("1.335")
SDAB_MAX_TINKER_USD = Decimal("0.50")
PAVLOV_HARD_CAP_USD = Decimal("18.00")
PAVLOV_OPERATIONAL_CAP_USD = Decimal("16.50")
PAVLOV_RESERVE_USD = Decimal("1.50")

DEFAULT_LIMIT = 1
DEFAULT_MAX_ACTIONS = 8
DEFAULT_MAX_PROMPT_TOKENS = 4096
DEFAULT_MAX_RESPONSE_TOKENS = 256

_HEX40_OR_64 = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-fA-F]{64}$", re.IGNORECASE)
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDERS = {
    "",
    "none",
    "null",
    "unknown",
    "unset",
    "pending",
    "placeholder",
    "todo",
    "tbd",
    "latest",
    "main",
    "master",
    "not_available",
    "not provided",
    "to_be_pinned",
    "to_be_pinned_before_paid_runs",
}


class SDABGateError(ValueError):
    """Raised when an exact SDAB contract gate is not satisfied."""


class NativeSDABRuntime(Protocol):
    """Provider adapter protocol required by :func:`load_native_runtime`.

    ``list_tasks`` must return metadata-only task objects containing a unique
    ``task_id``.  ``evaluate_task`` owns the official environment reset,
    tool/agent interaction, and long-horizon loop.  It receives the Tinker
    sampler but must never replace the provider environment with a synthetic
    simulator.  ``verify_result`` must call the provider's native state/grading
    verifier and return either a boolean or a mapping containing ``verified``.
    """

    def list_tasks(self, *, split: str, limit: int, seed: int) -> Sequence[Mapping[str, Any]]: ...

    def evaluate_task(
        self,
        task: Mapping[str, Any],
        sampler: Any,
        *,
        max_actions: int,
        seed: int,
    ) -> Mapping[str, Any]: ...

    def verify_result(self, result: Mapping[str, Any]) -> bool | Mapping[str, Any]: ...


@dataclass(frozen=True)
class RuntimeBundle:
    manifest: Mapping[str, Any]
    runtime: NativeSDABRuntime
    entrypoint: str


def canonical_json(value: Any) -> str:
    """Return deterministic JSON suitable for a receipt digest."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def task_ids_sha256(task_ids: Sequence[str]) -> str:
    """Hash the ordered, newline-framed task IDs in an E3 manifest."""

    return sha256_text("\n".join(task_ids))


def _placeholder(value: Any) -> bool:
    if value is None or not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    return normalized in _PLACEHOLDERS or any(
        marker in normalized for marker in ("placeholder", "to_be_pinned", "unrecorded")
    )


def _text(name: str, value: Any) -> str:
    if _placeholder(value):
        raise SDABGateError(f"{name} is missing or placeholder-like")
    return str(value).strip()


def _revision(name: str, value: Any) -> str:
    value = _text(name, value)
    if not _HEX40_OR_64.fullmatch(value):
        raise SDABGateError(f"{name} must be an immutable 40- or 64-hex revision")
    return value.lower()


def _sha256(name: str, value: Any) -> str:
    value = _text(name, value)
    if not _HEX64.fullmatch(value):
        raise SDABGateError(f"{name} must be 64 hexadecimal characters")
    return value.lower()


def _digest(name: str, value: Any) -> str:
    value = _text(name, value)
    if not _DIGEST.fullmatch(value):
        raise SDABGateError(f"{name} must match sha256:<64 hexadecimal characters>")
    return value.lower()


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise SDABGateError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SDABGateError(f"{name} must be a positive integer") from exc
    if result <= 0 or str(result) != str(value).strip() and not isinstance(value, int):
        raise SDABGateError(f"{name} must be a positive integer")
    return result


def _positive_decimal(name: str, value: Any) -> Decimal:
    if isinstance(value, bool):
        raise SDABGateError(f"{name} must be finite and positive")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise SDABGateError(f"{name} must be finite and positive") from exc
    if not result.is_finite() or result <= 0:
        raise SDABGateError(f"{name} must be finite and positive")
    return result


def maximum_sdab_eval_cost(
    task_count: int,
    max_actions: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> float:
    """Return the conservative uncached Tinker cost for an E3 slice.

    Every action is charged at its configured maximum.  This intentionally
    overestimates short episodes and cannot be used to bypass the $0.50 suite
    ceiling.
    """

    tasks = _positive_int("task_count", task_count)
    actions = _positive_int("max_actions", max_actions)
    prompts = _positive_int("max_prompt_tokens", max_prompt_tokens)
    responses = _positive_int("max_response_tokens", max_response_tokens)
    cost = Decimal(tasks * actions) * (
        Decimal(prompts) * PREFILL_USD_PER_MILLION
        + Decimal(responses) * SAMPLE_USD_PER_MILLION
    ) / Decimal(1_000_000)
    return float(cost)


def _validate_task_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    value = manifest.get("task_ids")
    if not isinstance(value, list) or not value:
        raise SDABGateError("runtime manifest task_ids must be a non-empty list")
    task_ids = tuple(_text(f"task_ids[{index}]", item) for index, item in enumerate(value))
    if len(set(task_ids)) != len(task_ids):
        raise SDABGateError("runtime manifest task_ids must be unique")
    expected = task_ids_sha256(task_ids)
    observed = _sha256("task_id_sha256", manifest.get("task_id_sha256"))
    if observed != expected:
        raise SDABGateError("task_id_sha256 does not match the ordered task_ids")
    return task_ids


def validate_native_manifest(
    manifest: Mapping[str, Any], *, required_tasks: int = DEFAULT_LIMIT
) -> dict[str, Any]:
    """Validate the exact provider-issued SDAB runtime manifest.

    The manifest is metadata-only.  Raw prompts, targets, trajectories, or
    synthetic task definitions are rejected.  This is the central boundary
    separating an exact SDAB run from a related/local benchmark.
    """

    if not isinstance(manifest, Mapping):
        raise SDABGateError("runtime manifest must be a JSON object")
    if manifest.get("suite_id") != SDAB_SUITE_ID:
        raise SDABGateError("runtime manifest suite_id must be sdab_eval")
    if manifest.get("benchmark_name") != SDAB_BENCHMARK_NAME:
        raise SDABGateError("runtime manifest benchmark_name is not the official SDAB benchmark")
    if manifest.get("provider") != SDAB_PROVIDER:
        raise SDABGateError("runtime manifest provider must be Emulated")
    if manifest.get("official_url") != SDAB_OFFICIAL_URL:
        raise SDABGateError("runtime manifest official_url must be https://emulated.so/sdab")
    if manifest.get("split") != SDAB_EVALUATION_SPLIT:
        raise SDABGateError("only the exact SDAB evaluation split is permitted")
    if manifest.get("native_verifier") is not True:
        raise SDABGateError("native_verifier must be the boolean true")
    if not isinstance(manifest.get("stateful"), bool) or manifest.get("stateful") is not True:
        raise SDABGateError("SDAB runtime must declare stateful=true")
    if not isinstance(manifest.get("artifact_or_side_effect"), bool) or manifest.get(
        "artifact_or_side_effect"
    ) is not True:
        raise SDABGateError("SDAB runtime must declare artifact_or_side_effect=true")

    benchmark_revision = _revision("benchmark_revision", manifest.get("benchmark_revision"))
    license_id = _text("license_id", manifest.get("license_id"))
    license_receipt = _text("license_receipt", manifest.get("license_receipt"))
    task_ids = _validate_task_ids(manifest)
    if len(task_ids) != manifest.get("task_count"):
        raise SDABGateError("task_count must equal the number of task_ids")
    if len(task_ids) != SDAB_TASK_COUNT:
        raise SDABGateError(f"exact SDAB manifest must contain {SDAB_TASK_COUNT} tasks")
    if required_tasks < 1 or required_tasks > len(task_ids):
        raise SDABGateError("requested task limit is outside the exact SDAB manifest")

    train_hash = _sha256("train_task_id_sha256", manifest.get("train_task_id_sha256"))
    eval_hash = _sha256("task_id_sha256", manifest.get("task_id_sha256"))
    if train_hash == eval_hash:
        raise SDABGateError("training and evaluation task-ID hashes must be disjoint")
    disjointness_receipt = _text(
        "disjointness_receipt", manifest.get("disjointness_receipt")
    )

    container_digest = _digest("container_digest", manifest.get("container_digest"))
    environment_digest = _digest("environment_digest", manifest.get("environment_digest"))
    verifier_sha256 = _sha256("verifier_sha256", manifest.get("verifier_sha256"))
    verifier_identity = _text("verifier_identity", manifest.get("verifier_identity"))
    if "native" not in verifier_identity.lower() and "emulated" not in verifier_identity.lower():
        raise SDABGateError("verifier_identity must identify the provider-native verifier")
    entrypoint = _text("adapter_entrypoint", manifest.get("adapter_entrypoint"))
    if not _ENTRYPOINT.fullmatch(entrypoint):
        raise SDABGateError("adapter_entrypoint must have module:factory form")

    # Do not allow raw benchmark content to cross the receipt boundary.
    forbidden_keys = {"prompt", "prompts", "target", "targets", "trajectory", "trajectories"}
    if any(key in manifest for key in forbidden_keys):
        raise SDABGateError("runtime manifest must be metadata-only; raw task content is forbidden")

    return {
        "suite_id": SDAB_SUITE_ID,
        "benchmark_name": SDAB_BENCHMARK_NAME,
        "provider": SDAB_PROVIDER,
        "official_url": SDAB_OFFICIAL_URL,
        "benchmark_revision": benchmark_revision,
        "license_id": license_id,
        "license_receipt": license_receipt,
        "split": SDAB_EVALUATION_SPLIT,
        "task_count": len(task_ids),
        "task_id_sha256": eval_hash,
        "train_task_id_sha256": train_hash,
        "disjointness_receipt": disjointness_receipt,
        "container_digest": container_digest,
        "environment_digest": environment_digest,
        "verifier_sha256": verifier_sha256,
        "verifier_identity": verifier_identity,
        "native_verifier": True,
        "stateful": True,
        "artifact_or_side_effect": True,
        "adapter_entrypoint": entrypoint,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SDABGateError(f"cannot read runtime manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SDABGateError("runtime manifest root must be a JSON object")
    return value


def _factory_call(factory: Any, manifest: Mapping[str, Any]) -> Any:
    if not callable(factory):
        raise SDABGateError("native adapter entrypoint is not callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    parameters = signature.parameters
    if "manifest" in parameters or any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return factory(manifest=manifest)
    if parameters:
        return factory(manifest)
    return factory()


def load_native_runtime(
    manifest_path: str | Path, *, required_tasks: int = DEFAULT_LIMIT
) -> RuntimeBundle:
    """Load and validate the provider-native adapter from a manifest.

    The adapter must expose ``list_tasks``, ``evaluate_task``, and
    ``verify_result``.  The latter is mandatory even if the adapter also
    returns a score: the score is inadmissible until native verification passes.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise SDABGateError(f"exact SDAB runtime manifest is missing: {path}")
    manifest = validate_native_manifest(_load_json(path), required_tasks=required_tasks)
    module_name, factory_name = manifest["adapter_entrypoint"].split(":", 1)
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
    except (ImportError, AttributeError) as exc:
        raise SDABGateError(
            f"cannot import provider-native SDAB adapter {manifest['adapter_entrypoint']}: {exc}"
        ) from exc
    runtime = _factory_call(factory, manifest)
    for method in ("list_tasks", "evaluate_task", "verify_result"):
        if not callable(getattr(runtime, method, None)):
            raise SDABGateError(f"native SDAB adapter must expose {method}()")
    return RuntimeBundle(manifest=manifest, runtime=runtime, entrypoint=manifest["adapter_entrypoint"])


def _environment_gate(worktree_root: str | Path, environment_root: str | Path) -> dict[str, Any]:
    root = Path(worktree_root).expanduser().resolve()
    environment = Path(environment_root).expanduser().resolve()
    blockers: list[str] = []
    try:
        environment.relative_to(root)
    except ValueError:
        blockers.append("benchmark environment must live inside the current worktree")
    if environment == root:
        blockers.append("benchmark dependencies must use a dedicated per-worktree environment")
    if not environment.is_dir():
        blockers.append(f"isolated environment is missing: {environment}")
    elif not (environment / "pyvenv.cfg").is_file():
        blockers.append(f"isolated environment is not a Python venv: {environment}")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "worktree_root": str(root),
        "environment_root": str(environment),
        "isolated": not blockers,
        "blockers": blockers,
    }


def _dependency_gate() -> dict[str, Any]:
    dependencies = {name: importlib.util.find_spec(name) is not None for name in ("wandb", "tinker")}
    blockers = [name for name, present in dependencies.items() if not present]
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "dependencies": dependencies,
        "blockers": [f"required package is unavailable in the isolated environment: {name}" for name in blockers],
    }


def _model_gate(
    *,
    base_model: str,
    base_model_revision: str,
    sampler_path: str | None,
    sampler_hf_commit: str | None,
    adapter_revision: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    evaluated_hf_commit: str | None = None
    source_kind = "base_model"
    if base_model != MODEL_ID:
        blockers.append(f"model must be the authorized candidate {MODEL_ID}")
    try:
        base_revision = _revision("base_model_revision", base_model_revision)
    except SDABGateError as exc:
        blockers.append(str(exc))
        base_revision = None
    if sampler_path:
        source_kind = "sampler_path"
        if not Path(sampler_path).expanduser().exists():
            blockers.append(f"sampler path is missing: {sampler_path}")
        try:
            adapter = _revision("adapter_revision", adapter_revision)
        except SDABGateError as exc:
            blockers.append(str(exc))
            adapter = None
        try:
            evaluated_hf_commit = _revision("sampler_hf_commit", sampler_hf_commit)
        except SDABGateError as exc:
            blockers.append(str(exc))
    else:
        if sampler_hf_commit is not None or adapter_revision is not None:
            blockers.append("sampler_hf_commit/adapter_revision require --sampler-path")
        adapter = None
        evaluated_hf_commit = base_revision
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "source_kind": source_kind,
        "model_id": base_model,
        "base_model_revision": base_revision,
        "adapter_revision": adapter,
        "evaluated_hf_commit": evaluated_hf_commit,
        "blockers": blockers,
    }


def _budget_gate(
    *,
    task_limit: int,
    max_actions: int,
    max_prompt_tokens: int,
    max_response_tokens: int,
    max_cost_usd: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        projected = maximum_sdab_eval_cost(
            task_limit, max_actions, max_prompt_tokens, max_response_tokens
        )
    except SDABGateError as exc:
        projected = None
        blockers.append(str(exc))
    try:
        cap = _positive_decimal("max_cost_usd", max_cost_usd)
    except SDABGateError as exc:
        cap = None
        blockers.append(str(exc))
    if cap is not None:
        if cap > SDAB_MAX_TINKER_USD:
            blockers.append("E3 max_cost_usd cannot exceed the hard $0.50 suite cap")
        if projected is not None and Decimal(str(projected)) > cap:
            blockers.append(f"projected maximum ${projected:.6f} exceeds cap ${cap:.6f}")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "suite_max_usd": float(SDAB_MAX_TINKER_USD),
        "requested_cap_usd": float(cap) if cap is not None else None,
        "projected_max_cost_usd": projected,
        "shared_hard_cap_usd": float(PAVLOV_HARD_CAP_USD),
        "shared_operational_cap_usd": float(PAVLOV_OPERATIONAL_CAP_USD),
        "shared_safety_reserve_usd": float(PAVLOV_RESERVE_USD),
        "blockers": blockers,
    }


def _runtime_gate(manifest_path: str | Path | None, *, task_limit: int) -> dict[str, Any]:
    if manifest_path is None:
        return {
            "status": "BLOCKED",
            "manifest": None,
            "blockers": [
                "exact SDAB runtime manifest is missing; the official task bundle/native verifier is not present"
            ],
        }
    try:
        bundle = load_native_runtime(manifest_path, required_tasks=task_limit)
    except SDABGateError as exc:
        return {"status": "BLOCKED", "manifest": str(manifest_path), "blockers": [str(exc)]}
    return {
        "status": "PASS",
        "manifest": str(Path(manifest_path).expanduser().resolve()),
        "benchmark_revision": bundle.manifest["benchmark_revision"],
        "task_count": bundle.manifest["task_count"],
        "task_id_sha256": bundle.manifest["task_id_sha256"],
        "train_task_id_sha256": bundle.manifest["train_task_id_sha256"],
        "container_digest": bundle.manifest["container_digest"],
        "environment_digest": bundle.manifest["environment_digest"],
        "verifier_sha256": bundle.manifest["verifier_sha256"],
        "verifier_identity": bundle.manifest["verifier_identity"],
        "adapter_entrypoint": bundle.entrypoint,
        "blockers": [],
    }


def required_external_access() -> list[dict[str, str]]:
    """Describe the precise provider access receipt needed to unblock E3."""

    return [
        {
            "item": "official_task_bundle",
            "needed": "Provider-issued SDAB evaluation bundle containing all 80 task IDs and the evaluation split",
            "must_include": "immutable benchmark revision, license identifier and access receipt",
        },
        {
            "item": "disjoint_task_ids",
            "needed": "Provider or campaign split receipt proving train/evaluation task-ID disjointness",
            "must_include": "train_task_id_sha256, task_id_sha256, and a signed/immutable disjointness receipt",
        },
        {
            "item": "native_environment",
            "needed": "Emulated enterprise/ML runtime adapter with live services and deterministic reset",
            "must_include": "container_digest, environment_digest, per-worktree venv, and adapter_entrypoint",
        },
        {
            "item": "native_verifier",
            "needed": "Provider-native SDAB behavioral/operational/engineering grader outside the agent workspace",
            "must_include": "verifier_identity, verifier_sha256, and a receipt proving state/artifact verification",
        },
        {
            "item": "model_binding",
            "needed": f"Immutable {MODEL_ID} base commit or completed compatible Tinker sampler",
            "must_include": "base_model_revision or adapter_revision plus evaluated_hf_commit",
        },
        {
            "item": "tracked_launch",
            "needed": "Online W&B run and Tinker access in the isolated environment",
            "must_include": "W&B identity initialized before the first Tinker call and a live budget ledger",
        },
    ]


def preflight(
    *,
    runtime_manifest: str | Path | None,
    worktree_root: str | Path = REPO_ROOT,
    environment_root: str | Path | None = None,
    task_limit: int = DEFAULT_LIMIT,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    max_response_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
    max_cost_usd: float = float(SDAB_MAX_TINKER_USD),
    base_model: str = MODEL_ID,
    base_model_revision: str = DEFAULT_BASE_MODEL_REVISION,
    sampler_path: str | None = None,
    sampler_hf_commit: str | None = None,
    adapter_revision: str | None = None,
    probe_dependencies: bool = True,
) -> dict[str, Any]:
    """Run all zero-side-effect gates and return a JSON-safe report."""

    if environment_root is None:
        environment_root = Path(worktree_root).resolve() / ".venv-sdab-e3"
    gates = [
        {
            "name": "exact_native_sdab_runtime",
            **_runtime_gate(runtime_manifest, task_limit=task_limit),
        },
        {
            "name": "isolated_worktree_environment",
            **_environment_gate(worktree_root, environment_root),
        },
        {
            "name": "authorized_model_revision",
            **_model_gate(
                base_model=base_model,
                base_model_revision=base_model_revision,
                sampler_path=sampler_path,
                sampler_hf_commit=sampler_hf_commit,
                adapter_revision=adapter_revision,
            ),
        },
        {
            "name": "tinker_cost_envelope",
            **_budget_gate(
                task_limit=task_limit,
                max_actions=max_actions,
                max_prompt_tokens=max_prompt_tokens,
                max_response_tokens=max_response_tokens,
                max_cost_usd=max_cost_usd,
            ),
        },
    ]
    if probe_dependencies:
        gates.append({"name": "isolated_dependencies", **_dependency_gate()})
    blockers = [
        f"{gate['name']}: {reason}"
        for gate in gates
        for reason in gate.get("blockers", [])
    ]
    return {
        "schema_version": "pavlov-sdab-e3-preflight-v1",
        "status": "READY" if not blockers else "BLOCKED",
        "can_launch": not blockers,
        "suite_id": SDAB_SUITE_ID,
        "benchmark": {
            "name": SDAB_BENCHMARK_NAME,
            "provider": SDAB_PROVIDER,
            "official_url": SDAB_OFFICIAL_URL,
            "task_count": SDAB_TASK_COUNT,
            "evaluation_split": SDAB_EVALUATION_SPLIT,
            "access_status": "private_access_required",
        },
        "requested_slice": {
            "task_limit": task_limit,
            "max_actions": max_actions,
            "max_prompt_tokens": max_prompt_tokens,
            "max_response_tokens": max_response_tokens,
        },
        "gates": gates,
        "blockers": blockers,
        "required_external_access": required_external_access(),
        "launches_any_job": False,
        "tinker_calls": 0,
        "wandb_initialized": False,
    }


def _receipt_from_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    model_gate = next(
        (gate for gate in report.get("gates", []) if gate.get("name") == "authorized_model_revision"),
        {},
    )
    runtime_gate = next(
        (gate for gate in report.get("gates", []) if gate.get("name") == "exact_native_sdab_runtime"),
        {},
    )
    return {
        "schema_version": "pavlov-sdab-e3-receipt-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED" if report.get("status") != "READY" else "READY",
        "suite_id": SDAB_SUITE_ID,
        "benchmark": {
            "name": SDAB_BENCHMARK_NAME,
            "provider": SDAB_PROVIDER,
            "official_url": SDAB_OFFICIAL_URL,
            "task_count": SDAB_TASK_COUNT,
            "split": SDAB_EVALUATION_SPLIT,
            "access_status": "private_access_required",
            "revision": runtime_gate.get("benchmark_revision"),
            "license_id": None,
        },
        "model": {
            "model_id": model_gate.get("model_id", MODEL_ID),
            "source_kind": model_gate.get("source_kind"),
            "base_model_revision": model_gate.get("base_model_revision"),
            "adapter_revision": model_gate.get("adapter_revision"),
            "evaluated_hf_commit": model_gate.get("evaluated_hf_commit"),
        },
        "runtime": {
            "manifest": runtime_gate.get("manifest"),
            "container_digest": runtime_gate.get("container_digest"),
            "environment_digest": runtime_gate.get("environment_digest"),
            "verifier_identity": runtime_gate.get("verifier_identity"),
            "verifier_sha256": runtime_gate.get("verifier_sha256"),
            "native_verifier": runtime_gate.get("status") == "PASS",
        },
        "budget": next(
            (gate for gate in report.get("gates", []) if gate.get("name") == "tinker_cost_envelope"),
            {},
        ),
        "gates": report.get("gates", []),
        "blockers": report.get("blockers", []),
        "required_external_access": report.get("required_external_access", []),
        "task_ids": {
            "evaluation_sha256": runtime_gate.get("task_id_sha256"),
            "training_sha256": runtime_gate.get("train_task_id_sha256"),
            "disjointness_receipt": None,
        },
        "wandb": {"online": False, "initialized_before_tinker": False, "run_id": None, "run_url": None},
        "tinker": {"calls": 0, "run_id": None},
        "hf": {
            "eval_only": True,
            "evaluated_commit_bound": model_gate.get("evaluated_hf_commit") is not None,
            "checkpoint_uploads_required": False,
            "checkpoint_uploads": [],
        },
        # Explicit nulls are intentional: no score is fabricated while exact
        # task data/native verification are unavailable.
        "result": None,
        "score": None,
        "rows": [],
        "launches_any_job": False,
        "evidence_boundary": "BLOCKED preflight; no SDAB score or scientific result",
        "receipt_sha256": None,
    }


def _write_receipt(receipt: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(receipt)
    payload["receipt_sha256"] = sha256_json({key: value for key, value in payload.items() if key != "receipt_sha256"})
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _start_wandb(wandb: ModuleType, config: Mapping[str, Any], args: argparse.Namespace) -> Any:
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        job_type="sdab-evaluation",
        mode="online",
        config=dict(config),
    )
    if run is None or str(getattr(run, "mode", "")).lower() != "online":
        raise SDABGateError("W&B online initialization did not return an online run")
    config_obj = getattr(run, "config", None)
    update = getattr(config_obj, "update", None)
    if callable(update):
        update(dict(config), allow_val_change=False)
    return run


def _finish_wandb(run: Any, *, exit_code: int) -> None:
    finish = getattr(run, "finish", None)
    if not callable(finish):
        raise RuntimeError("W&B run has no finish method")
    finish(exit_code=exit_code)


def _create_tinker_sampler(
    tinker: ModuleType,
    *,
    model_gate: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, Any]:
    service_factory = getattr(tinker, "ServiceClient", None)
    if not callable(service_factory):
        raise RuntimeError("tinker.ServiceClient is unavailable")
    metadata = {
        "campaign": "pavlov-18usd",
        "suite_id": SDAB_SUITE_ID,
        "stage": "primary-evaluation",
        "model_id": model_gate["model_id"],
        "base_model_revision": model_gate["base_model_revision"],
        "evaluated_hf_commit": model_gate["evaluated_hf_commit"],
    }
    service = service_factory(
        base_model_revision=model_gate["base_model_revision"],
        user_metadata=metadata,
    )
    create = getattr(service, "create_sampling_client", None)
    if not callable(create):
        raise RuntimeError("Tinker service does not expose create_sampling_client")
    if model_gate["source_kind"] == "sampler_path":
        candidates = {"model_path": args.sampler_path, "revision": model_gate["evaluated_hf_commit"]}
    else:
        candidates = {"base_model": model_gate["model_id"], "revision": model_gate["base_model_revision"]}
    try:
        signature = inspect.signature(create)
        if not any(parameter.kind is parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            candidates = {key: value for key, value in candidates.items() if key in signature.parameters}
    except (TypeError, ValueError):
        pass
    return service, create(**candidates)


def _native_verified(runtime: NativeSDABRuntime, result: Mapping[str, Any]) -> Mapping[str, Any]:
    verification = runtime.verify_result(result)
    if isinstance(verification, Mapping):
        verified = verification.get("verified")
        receipt = dict(verification)
    else:
        verified = verification
        receipt = {"verified": verification}
    if verified is not True:
        raise RuntimeError("provider-native SDAB verifier did not acknowledge the result")
    return receipt


def _result_row(result: Mapping[str, Any], verification: Mapping[str, Any]) -> dict[str, Any]:
    task_id = _text("result.task_id", result.get("task_id"))
    score = result.get("score", result.get("composite_score"))
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise RuntimeError("native result score must be a finite number")
    row = {
        "task_id": task_id,
        "task_id_sha256": sha256_text(task_id),
        "score": float(score),
        "behavioral_score": result.get("behavioral_score"),
        "operational_score": result.get("operational_score"),
        "engineering_score": result.get("engineering_score"),
        "state_hash": result.get("state_hash", result.get("artifact_hash")),
        "native_verifier_receipt_sha256": sha256_json(verification),
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "sample_tokens": int(result.get("sample_tokens", 0)),
        "tool_calls": int(result.get("tool_calls", 0)),
    }
    if row["prompt_tokens"] < 0 or row["sample_tokens"] < 0 or row["tool_calls"] < 0:
        raise RuntimeError("native result token/tool counts must be non-negative")
    return row


def run_evaluation(
    *,
    report: Mapping[str, Any],
    args: argparse.Namespace,
    out: str | Path,
) -> dict[str, Any]:
    """Run one exact SDAB slice after all preflight gates pass."""

    if report.get("status") != "READY":
        return _write_receipt(_receipt_from_preflight(report), out)
    bundle = load_native_runtime(args.runtime_manifest, required_tasks=args.limit)
    model_gate = next(gate for gate in report["gates"] if gate["name"] == "authorized_model_revision")
    budget_gate = next(gate for gate in report["gates"] if gate["name"] == "tinker_cost_envelope")
    config = {
        "schema_version": "pavlov-sdab-e3-config-v1",
        "campaign": "pavlov-18usd",
        "suite_id": SDAB_SUITE_ID,
        "suite_role": "primary_eval",
        "benchmark_revision": bundle.manifest["benchmark_revision"],
        "split": SDAB_EVALUATION_SPLIT,
        "task_id_sha256": bundle.manifest["task_id_sha256"],
        "train_task_id_sha256": bundle.manifest["train_task_id_sha256"],
        "verifier_sha256": bundle.manifest["verifier_sha256"],
        "container_digest": bundle.manifest["container_digest"],
        "environment_digest": bundle.manifest["environment_digest"],
        "model_id": model_gate["model_id"],
        "base_model_revision": model_gate["base_model_revision"],
        "evaluated_hf_commit": model_gate["evaluated_hf_commit"],
        "task_limit": args.limit,
        "max_actions": args.max_actions,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "projected_max_cost_usd": budget_gate["projected_max_cost_usd"],
        "max_cost_usd": budget_gate["requested_cap_usd"],
    }
    receipt = _receipt_from_preflight(report)
    receipt.update(
        {
            "status": "RUNNING",
            "benchmark": {**receipt["benchmark"], **bundle.manifest, "license_id": bundle.manifest["license_id"]},
            "runtime": {**receipt["runtime"], "native_verifier": True},
            "task_ids": {
                "evaluation_sha256": bundle.manifest["task_id_sha256"],
                "training_sha256": bundle.manifest["train_task_id_sha256"],
                "disjointness_receipt": bundle.manifest["disjointness_receipt"],
            },
        }
    )
    wandb_run: Any | None = None
    try:
        # The first Tinker call is deliberately below this line.
        import wandb

        wandb_run = _start_wandb(wandb, config, args)
        receipt["wandb"] = {
            "online": True,
            "initialized_before_tinker": True,
            "run_id": getattr(wandb_run, "id", None),
            "run_url": getattr(wandb_run, "url", None),
        }
        import tinker

        service, sampler = _create_tinker_sampler(tinker, model_gate=model_gate, args=args)
        tasks = list(bundle.runtime.list_tasks(split=SDAB_EVALUATION_SPLIT, limit=args.limit, seed=args.seed))
        if len(tasks) != args.limit:
            raise RuntimeError("native SDAB adapter returned a task count different from --limit")
        rows: list[dict[str, Any]] = []
        for task in tasks:
            if not isinstance(task, Mapping):
                raise RuntimeError("native SDAB adapter returned a non-metadata task")
            result = bundle.runtime.evaluate_task(
                task,
                sampler,
                max_actions=args.max_actions,
                seed=args.seed,
            )
            if not isinstance(result, Mapping):
                raise RuntimeError("native SDAB adapter returned a non-object result")
            verification = _native_verified(bundle.runtime, result)
            rows.append(_result_row(result, verification))
        actual_cost = sum(
            (
                Decimal(row["prompt_tokens"]) * PREFILL_USD_PER_MILLION
                + Decimal(row["sample_tokens"]) * SAMPLE_USD_PER_MILLION
            )
            for row in rows
        ) / Decimal(1_000_000)
        if actual_cost > Decimal(str(budget_gate["requested_cap_usd"])):
            raise RuntimeError("observed token cost exceeded the E3 cap")
        score = sum(row["score"] for row in rows) / len(rows)
        receipt.update(
            {
                "status": "COMPLETED",
                "result": {"task_count": len(rows), "mean_composite_score": score},
                "score": score,
                "rows": rows,
                "tinker": {
                    "calls": len(rows),
                    "run_id": getattr(service, "run_id", getattr(service, "id", None)),
                },
                "cost": {
                    "actual_usd": float(actual_cost),
                    "projected_max_usd": budget_gate["projected_max_cost_usd"],
                    "cap_usd": budget_gate["requested_cap_usd"],
                },
                "evidence_boundary": "Exact SDAB evaluation receipt; interpret only for this revision/split/environment",
                "launches_any_job": True,
            }
        )
        receipt = _write_receipt(receipt, out)
        run_log = getattr(wandb_run, "log", None)
        if callable(run_log):
            run_log({"eval/mean_composite_score": score, "eval/task_count": len(rows), "eval/cost_usd": float(actual_cost)})
        artifact_factory = getattr(wandb, "Artifact", None)
        log_artifact = getattr(wandb_run, "log_artifact", None)
        if callable(artifact_factory) and callable(log_artifact):
            artifact = artifact_factory(name=f"{args.wandb_name}-sdab-receipt", type="evaluation-receipt")
            artifact.add_file(str(Path(out).resolve()), name="receipt.json")
            log_artifact(artifact)
        _finish_wandb(wandb_run, exit_code=0)
        return receipt
    except Exception:
        if wandb_run is not None:
            try:
                _finish_wandb(wandb_run, exit_code=1)
            except Exception:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="run zero-side-effect gates (default)")
    mode.add_argument("--run", action="store_true", help="run one exact SDAB task after gates pass")
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--worktree-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--environment-root", type=Path)
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument("--base-model-revision", default=DEFAULT_BASE_MODEL_REVISION)
    parser.add_argument("--sampler-path")
    parser.add_argument("--sampler-hf-commit")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--seed", type=int, default=809)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-response-tokens", type=int, default=DEFAULT_MAX_RESPONSE_TOKENS)
    parser.add_argument("--max-cost-usd", type=float, default=float(SDAB_MAX_TINKER_USD))
    parser.add_argument("--wandb-entity", default="arvindcr4-pes-university")
    parser.add_argument("--wandb-project", default="tinker-rl-lab-pavlov")
    parser.add_argument("--wandb-group", default="pavlov-sdab-e3")
    parser.add_argument("--wandb-name", default="pavlov-sdab-e3-eval")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = preflight(
        runtime_manifest=args.runtime_manifest,
        worktree_root=args.worktree_root,
        environment_root=args.environment_root,
        task_limit=args.limit,
        max_actions=args.max_actions,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        max_cost_usd=args.max_cost_usd,
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        sampler_path=args.sampler_path,
        sampler_hf_commit=args.sampler_hf_commit,
        adapter_revision=args.adapter_revision,
    )
    if args.run and report["status"] == "READY":
        receipt = run_evaluation(report=report, args=args, out=args.out)
    else:
        receipt = _write_receipt(_receipt_from_preflight(report), args.out)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" and args.run else (0 if args.preflight or not args.run else 2)


if __name__ == "__main__":
    raise SystemExit(main())
