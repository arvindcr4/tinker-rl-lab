#!/usr/bin/env python3
"""Fail-closed FrontierSWE preflight and native verifier runner.

This module deliberately speaks only to the public ``Proximal-Labs/frontier-
swe`` checkout.  It never substitutes another coding benchmark and it never
turns a missing receipt into a score.  ``preflight`` is zero-cost; ``run``
invokes the task's checked-in ``tests/test.sh`` inside the task's published
Docker image only after every provenance, budget, and disjointness gate passes.

The model is recorded even when the run falls back to the immutable base-model
revision.  A Tinker caller must use :func:`run_tinker_guarded`: the online
W&B run is created and verified before the supplied Tinker factory is called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OFFICIAL_REPO_URL = "https://github.com/Proximal-Labs/frontier-swe"
OFFICIAL_REPO_API_URL = "https://api.github.com/repos/Proximal-Labs/frontier-swe"
# Pinned from the live public main ref on 2026-08-09.  A different checkout is
# not silently accepted: the receipt must name a new immutable revision first.
PINNED_BENCHMARK_REVISION = "422b9bb95deb8efe436becb0ed3c44be23611e10"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_API_URL = "https://huggingface.co/api/models/Qwen/Qwen3.6-35B-A3B"

SUITE_MAXIMUM_USD = Decimal("0.50")
SHARED_MAXIMUM_USD = Decimal("18.00")
SHARED_OPERATIONAL_CAP_USD = Decimal("16.50")
SHARED_RESERVE_USD = Decimal("1.50")

OFFICIAL_TASK_IDS = (
    "cranelift-codegen-opt",
    "dart-style-haskell",
    "dependent-type-checker",
    "ffmpeg-swscale-rewrite",
    "frogsgame-rl",
    "git-to-zig",
    "granite-mamba2-inference-optimization",
    "inference-system-optimization",
    "libexpat-to-x86asm",
    "lua-native-compiler",
    "modular-stack-wan21",
    "notebook-compression",
    "optimizer-design",
    "pcqm4mv2-autoresearch",
    "postgres-sqlite-wire-adapter",
    "pyright-type-checking-optimization",
    "revideo-perf-opt",
)
# The checked-in official fixtures have a deterministic smallest task by
# aggregate Git-tree bytes.  It is a one-task exact evaluation, not a proxy.
SMALLEST_TASK_ID = "revideo-perf-opt"
DEFAULT_CANDIDATE_MOUNTS = {SMALLEST_TASK_ID: "/app/revideo"}
HEX40 = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)


class GateError(ValueError):
    """A malformed or unsafe launch receipt."""


@dataclass(frozen=True)
class Blocker:
    code: str
    message: str
    required_receipt: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "required_receipt": self.required_receipt,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _json_hash(value: Any) -> str:
    return _sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git(repo: Path, *args: str, timeout: float = 30.0) -> str:
    result = _run(("git", "-C", str(repo), *args), timeout=timeout)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise GateError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _normalise_remote(value: str) -> str:
    value = value.strip()
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.split(":", 1)[1]
    return value.rstrip("/")


def _revision(repo: Path) -> str:
    value = _git(repo, "rev-parse", "HEAD")
    if not HEX40.fullmatch(value):
        raise GateError(f"benchmark HEAD is not an immutable Git revision: {value!r}")
    return value.lower()


def _repo_remote(repo: Path) -> str:
    return _normalise_remote(_git(repo, "config", "--get", "remote.origin.url"))


def _is_clean(repo: Path) -> bool:
    return not bool(_git(repo, "status", "--porcelain"))


def official_task_ids(repo: Path) -> tuple[str, ...]:
    """Read task IDs from the Git tree, so sparse checkouts remain auditable."""

    output = _git(repo, "ls-tree", "-d", "--name-only", "HEAD:tasks")
    return tuple(sorted(line for line in output.splitlines() if line.strip()))


def _task_tree_hash(repo: Path, task_id: str) -> str:
    output = _git(repo, "ls-tree", "-r", "--full-tree", "HEAD", "--", f"tasks/{task_id}")
    if not output:
        raise GateError(f"task {task_id!r} is absent from the pinned Git tree")
    return _sha256_text(output + "\n")


def _task_spec(task_dir: Path) -> dict[str, Any]:
    path = task_dir / "task.toml"
    if not path.is_file():
        raise GateError(f"task metadata is missing: {path}")
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        raise GateError(f"task metadata is not an object: {path}")
    return value


def _required_task_files(task_dir: Path) -> tuple[Path, ...]:
    return (
        task_dir / "task.toml",
        task_dir / "instruction.md",
        task_dir / "job.yaml",
        task_dir / "tests" / "test.sh",
        task_dir / "tests" / "compute_reward.py",
    )


def _verifier_hash(task_dir: Path) -> str:
    tests_dir = task_dir / "tests"
    if not tests_dir.is_dir():
        raise GateError(f"native verifier directory is missing: {tests_dir}")
    digest = hashlib.sha256()
    files = sorted(path for path in tests_dir.rglob("*") if path.is_file())
    if not files:
        raise GateError(f"native verifier directory is empty: {tests_dir}")
    for path in files:
        relative = path.relative_to(task_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _container_digest(image: str) -> str | None:
    if not shutil.which("docker"):
        return None
    result = _run(("docker", "manifest", "inspect", image), timeout=60.0)
    if result.returncode:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    manifests = payload.get("manifests")
    if isinstance(manifests, list):
        for item in manifests:
            platform = item.get("platform", {}) if isinstance(item, Mapping) else {}
            if (
                isinstance(platform, Mapping)
                and platform.get("os") == "linux"
                and platform.get("architecture") == "amd64"
                and isinstance(item.get("digest"), str)
                and DIGEST.fullmatch(item["digest"])
            ):
                return item["digest"].lower()
        return None
    config = payload.get("config")
    if isinstance(config, Mapping) and isinstance(config.get("digest"), str):
        value = config["digest"]
        return value.lower() if DIGEST.fullmatch(value) else None
    return None


def resolve_model_revision(
    model_id: str = MODEL_ID,
    *,
    opener: Callable[..., Any] = urlopen,
) -> str:
    """Resolve the public Hub model's current commit without downloading weights."""

    if model_id != MODEL_ID:
        raise GateError(f"experiment model is fixed to {MODEL_ID!r}")
    request = Request(MODEL_API_URL, headers={"Accept": "application/json"})
    try:
        with opener(request, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise GateError(f"immutable base-model revision could not be resolved: {exc}") from exc
    revision = payload.get("sha") if isinstance(payload, Mapping) else None
    if not isinstance(revision, str) or not HEX40.fullmatch(revision):
        raise GateError("Hub model metadata did not return an immutable commit SHA")
    return revision.lower()


def _license_receipt(
    repo: Path,
    supplied: str | None,
) -> tuple[dict[str, Any], Blocker | None]:
    candidates = [repo / name for name in ("LICENSE", "LICENSE.md", "COPYING", "COPYING.md")]
    for path in candidates:
        if path.is_file():
            return (
                {
                    "status": "present",
                    "path": path.name,
                    "sha256": _sha256_bytes(path.read_bytes()),
                    "source_url": f"{OFFICIAL_REPO_URL}/blob/{PINNED_BENCHMARK_REVISION}/{path.name}",
                },
                None,
            )
    if supplied:
        path = Path(supplied)
        if path.is_file():
            return (
                {
                    "status": "supplied",
                    "path": str(path),
                    "sha256": _sha256_bytes(path.read_bytes()),
                },
                None,
            )
        if HEX64.fullmatch(supplied):
            return ({"status": "supplied_digest", "sha256": supplied.lower()}, None)
    return (
        {
            "status": "missing",
            "source_url": f"{OFFICIAL_REPO_URL}/blob/{PINNED_BENCHMARK_REVISION}/LICENSE",
        },
        Blocker(
            "benchmark_license_missing",
            "the official benchmark checkout has no root license file",
            "a license receipt or explicit maintainer authorization binding "
            f"{OFFICIAL_REPO_URL}@{PINNED_BENCHMARK_REVISION}",
        ),
    )


def _read_prior_task_ids(paths: Iterable[Path]) -> set[str]:
    task_ids: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GateError(f"prior receipt is unreadable: {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise GateError(f"prior receipt is not an object: {path}")
        benchmark = payload.get("benchmark")
        values = benchmark.get("task_ids") if isinstance(benchmark, Mapping) else payload.get("task_ids")
        if isinstance(values, list):
            task_ids.update(str(value) for value in values)
    return task_ids


def _sampler_source(path: Path | None, model_revision: str) -> dict[str, Any]:
    if path is None:
        return {
            "kind": "base_model",
            "model_id": MODEL_ID,
            "model_revision": model_revision,
            "evaluated_hf_commit": model_revision,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"sampler receipt is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GateError("sampler receipt must be a JSON object")
    status = str(payload.get("status", "")).lower()
    model = payload.get("model") or payload.get("model_id")
    commit = payload.get("hf_commit") or payload.get("hf_commit_sha") or payload.get("commit_sha")
    if status not in {"completed", "complete", "success", "succeeded"}:
        raise GateError("sampler receipt is not a completed run")
    if model != MODEL_ID:
        raise GateError(f"sampler receipt model must be {MODEL_ID!r}")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise GateError("completed sampler receipt must bind an immutable HF commit SHA")
    return {
        "kind": "tinker_sampler_checkpoint",
        "model_id": MODEL_ID,
        "sampler_receipt": str(path),
        "evaluated_hf_commit": commit.lower(),
    }


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool):
        raise GateError(f"{label} must be a finite non-negative decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise GateError(f"{label} must be a finite non-negative decimal") from exc
    if not parsed.is_finite() or parsed < 0:
        raise GateError(f"{label} must be a finite non-negative decimal")
    return parsed


def validate_task_disjointness(task_ids: Sequence[str], prior_task_ids: Iterable[str]) -> None:
    overlap = sorted(set(task_ids).intersection(set(prior_task_ids)))
    if overlap:
        raise GateError("task IDs are not disjoint: " + ", ".join(overlap))


def _base_receipt(
    *,
    repo: Path,
    task_ids: Sequence[str],
    model_revision: str,
    source: Mapping[str, Any],
    license_receipt: Mapping[str, Any],
    task_hashes: Mapping[str, str],
    verifier_hashes: Mapping[str, str],
    container_digests: Mapping[str, str | None],
    blockers: Sequence[Blocker],
    mode: str,
    projected_tinker_usd: Decimal,
    prior_task_ids: Sequence[str],
) -> dict[str, Any]:
    task_hash = _sha256_text("\n".join(f"{key}:{task_hashes[key]}" for key in task_ids) + "\n")
    task_id_hash = _sha256_text("\n".join(task_ids) + "\n")
    return {
        "schema_version": "frontier-swe-eval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "BLOCKED" if blockers else "READY",
        "mode": mode,
        "benchmark": {
            "id": "frontier_swe_eval",
            "name": "FrontierSWE",
            "repository": OFFICIAL_REPO_URL,
            "checkout_path": str(repo),
            "revision": _revision(repo),
            "pinned_revision": PINNED_BENCHMARK_REVISION,
            "official_task_ids": list(OFFICIAL_TASK_IDS),
            "task_ids": list(task_ids),
            "task_id_sha256": task_id_hash,
            "task_manifest_sha256": task_hash,
            "task_hashes": dict(task_hashes),
            "split": {
                "kind": "task_id_disjoint",
                "prior_task_ids": list(prior_task_ids),
                "disjoint": not bool(set(task_ids).intersection(prior_task_ids)),
            },
            "license": dict(license_receipt),
            "verifier_sha256": dict(verifier_hashes),
            "container_digests": dict(container_digests),
        },
        "model": {
            "model_id": MODEL_ID,
            "base_model_revision": model_revision,
            "source": dict(source),
        },
        "budget": {
            "suite_maximum_usd": float(SUITE_MAXIMUM_USD),
            "projected_tinker_usd": float(projected_tinker_usd),
            "shared_hard_maximum_usd": float(SHARED_MAXIMUM_USD),
            "shared_operational_cap_usd": float(SHARED_OPERATIONAL_CAP_USD),
            "safety_reserve_usd": float(SHARED_RESERVE_USD),
            "reserve_preserved": True,
        },
        "blockers": [item.as_dict() for item in blockers],
        "score": None if blockers else "not_run",
    }


def build_preflight_receipt(
    *,
    repo: Path,
    task_ids: Sequence[str] = (SMALLEST_TASK_ID,),
    prior_receipts: Sequence[Path] = (),
    license_receipt: str | None = None,
    sampler_receipt: Path | None = None,
    projected_tinker_usd: Decimal | str | float = SUITE_MAXIMUM_USD,
    mode: str = "preflight",
    use_tinker: bool = False,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete receipt without launching a model, Tinker, or verifier."""

    repo = Path(repo).resolve()
    blockers: list[Blocker] = []
    projected = _decimal(projected_tinker_usd, "projected_tinker_usd")
    if projected > SUITE_MAXIMUM_USD:
        blockers.append(
            Blocker(
                "suite_budget_exceeded",
                f"projected Tinker usage ${projected} exceeds the ${SUITE_MAXIMUM_USD} suite cap",
                "an explicit projection at or below USD 0.50",
            )
        )
    if not repo.is_dir():
        raise GateError(f"benchmark checkout is missing: {repo}")

    try:
        observed_revision = _revision(repo)
        remote = _repo_remote(repo)
        clean = _is_clean(repo)
        observed_task_ids = official_task_ids(repo)
    except GateError as exc:
        raise
    if remote != OFFICIAL_REPO_URL:
        blockers.append(
            Blocker(
                "benchmark_remote_mismatch",
                f"checkout remote is {remote!r}, not the official repository",
                OFFICIAL_REPO_URL,
            )
        )
    if observed_revision != PINNED_BENCHMARK_REVISION:
        blockers.append(
            Blocker(
                "benchmark_revision_mismatch",
                f"expected pinned revision {PINNED_BENCHMARK_REVISION}, got {observed_revision}",
                f"checkout {OFFICIAL_REPO_URL}@{PINNED_BENCHMARK_REVISION}",
            )
        )
    if not clean:
        blockers.append(
            Blocker(
                "benchmark_checkout_dirty",
                "benchmark checkout has uncommitted changes",
                "a clean checkout at the pinned revision",
            )
        )
    if tuple(observed_task_ids) != tuple(sorted(OFFICIAL_TASK_IDS)):
        blockers.append(
            Blocker(
                "benchmark_task_inventory_mismatch",
                "checkout task inventory is not the exact official 17-task roster",
                "the complete official tasks/ tree at the pinned revision",
            )
        )
    task_ids = tuple(dict.fromkeys(str(value) for value in task_ids))
    if not task_ids:
        raise GateError("at least one exact FrontierSWE task ID is required")
    unknown = sorted(set(task_ids).difference(OFFICIAL_TASK_IDS))
    if unknown:
        blockers.append(
            Blocker(
                "unknown_task_id",
                "task IDs are not in the official FrontierSWE roster: " + ", ".join(unknown),
                "an exact task ID from the pinned official tasks/ tree",
            )
        )
    prior_task_ids = sorted(_read_prior_task_ids(prior_receipts))
    try:
        validate_task_disjointness(task_ids, prior_task_ids)
    except GateError as exc:
        blockers.append(Blocker("task_ids_not_disjoint", str(exc), "a disjoint task-ID manifest"))

    task_hashes: dict[str, str] = {}
    verifier_hashes: dict[str, str] = {}
    container_digests: dict[str, str | None] = {}
    for task_id in task_ids:
        task_dir = repo / "tasks" / task_id
        try:
            task_hashes[task_id] = _task_tree_hash(repo, task_id)
            spec = _task_spec(task_dir)
            missing = [str(path.relative_to(repo)) for path in _required_task_files(task_dir) if not path.is_file()]
            if missing:
                blockers.append(
                    Blocker(
                        "native_fixture_missing",
                        f"native fixture files are missing for {task_id}: {', '.join(missing)}",
                        "the official task fixture checkout including tests/test.sh and tests/compute_reward.py",
                    )
                )
            verifier_hashes[task_id] = _verifier_hash(task_dir)
            image = spec.get("environment", {}).get("docker_image")
            if not isinstance(image, str) or not image.strip():
                raise GateError(f"task {task_id} has no environment.docker_image")
            container_digests[task_id] = _container_digest(image)
            if not container_digests[task_id]:
                blockers.append(
                    Blocker(
                        "native_verifier_image_unresolved",
                        f"published verifier image digest could not be resolved for {task_id}: {image}",
                        f"the immutable digest for {image} (or GHCR access receipt)",
                    )
                )
        except GateError as exc:
            blockers.append(Blocker("native_verifier_unavailable", str(exc), "the official task verifier checkout"))

    license_info, license_blocker = _license_receipt(repo, license_receipt)
    if license_blocker is not None:
        blockers.append(license_blocker)
    try:
        model_revision = resolve_model_revision()
    except GateError as exc:
        model_revision = ""
        blockers.append(Blocker("base_model_revision_unresolved", str(exc), "an immutable HF commit for Qwen/Qwen3.6-35B-A3B"))
    try:
        source = _sampler_source(sampler_receipt, model_revision)
    except GateError as exc:
        source = {"kind": "unresolved"}
        blockers.append(Blocker("sampler_checkpoint_invalid", str(exc), "a completed compatible public HF sampler receipt"))
    if use_tinker:
        if not os.environ.get("TINKER_API_KEY"):
            blockers.append(Blocker("tinker_api_key_missing", "TINKER_API_KEY is not available", "a Tinker API access receipt"))
        if not os.environ.get("WANDB_API_KEY"):
            blockers.append(Blocker("wandb_api_key_missing", "WANDB_API_KEY is not available for online tracking", "an online W&B run receipt"))
    if mode == "run" and environment is not None and not environment.get("submission_dir"):
        blockers.append(Blocker("submission_missing", "no model-produced candidate submission was supplied", "the candidate workspace generated by the Qwen evaluation agent"))
    receipt = _base_receipt(
        repo=repo,
        task_ids=task_ids,
        model_revision=model_revision,
        source=source,
        license_receipt=license_info,
        task_hashes=task_hashes,
        verifier_hashes=verifier_hashes,
        container_digests=container_digests,
        blockers=blockers,
        mode=mode,
        projected_tinker_usd=projected,
        prior_task_ids=prior_task_ids,
    )
    receipt["environment"] = dict(environment or {})
    receipt["environment"]["native_verifier"] = "official task tests/test.sh + compute_reward.py"
    receipt["environment"]["network"] = "none"
    return receipt


def _assert_online_run(run: Any) -> None:
    mode = getattr(run, "mode", None)
    if mode is None:
        settings = getattr(run, "settings", None) or getattr(run, "_settings", None)
        mode = getattr(settings, "mode", None)
    mode = getattr(mode, "value", mode)
    if mode != "online" or not isinstance(getattr(run, "id", None), str) or not run.id.strip():
        raise RuntimeError("W&B run is not verifiably online with a non-empty run ID")


def run_tinker_guarded(
    *,
    wandb_module: Any,
    wandb_config: Mapping[str, Any],
    tinker_factory: Callable[[], Any],
    wandb_kwargs: Mapping[str, Any] | None = None,
) -> tuple[Any, Any]:
    """Create online W&B first, then construct the Tinker client.

    The injected functions make the ordering independently testable without a
    network call.  Any non-online or anonymous W&B run fails before Tinker.
    """

    kwargs = dict(wandb_kwargs or {})
    kwargs.update({"mode": "online", "config": dict(wandb_config)})
    run = wandb_module.init(**kwargs)
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    try:
        _assert_online_run(run)
        client = tinker_factory()
        return run, client
    except Exception:
        finish = getattr(run, "finish", None)
        if callable(finish):
            finish(exit_code=1)
        raise


def _native_verify(
    *,
    task_dir: Path,
    image: str,
    submission_dir: Path,
    candidate_mount: str,
    output_dir: Path,
    cpus: int,
    memory_mb: int,
) -> tuple[int, dict[str, Any] | None, list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        str(cpus),
        "--memory",
        f"{memory_mb}m",
        "--entrypoint",
        "/bin/bash",
        "-v",
        f"{task_dir / 'tests'}:/tests:ro",
        "-v",
        f"{submission_dir}:{candidate_mount}:rw",
        "-v",
        f"{output_dir}:/logs/verifier:rw",
        image,
        "/tests/test.sh",
    ]
    result = _run(command, timeout=86400.0)
    reward_path = output_dir / "reward.json"
    reward: dict[str, Any] | None = None
    if reward_path.is_file():
        try:
            value = json.loads(reward_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                reward = value
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            reward = None
    return result.returncode, reward, [result.stdout[-2000:], result.stderr[-2000:]]


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--benchmark-repo", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids")
    parser.add_argument("--smallest", action="store_true", help="select the one smallest official fixture")
    parser.add_argument("--prior-receipt", action="append", type=Path, default=[])
    parser.add_argument("--license-receipt")
    parser.add_argument("--sampler-receipt", type=Path)
    parser.add_argument("--projected-tinker-usd", default=str(SUITE_MAXIMUM_USD))
    parser.add_argument("--use-tinker", action="store_true")
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--candidate-mount")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.task_ids and args.smallest:
        raise SystemExit("--task-id and --smallest are mutually exclusive")
    task_ids = tuple(args.task_ids or ((SMALLEST_TASK_ID,) if args.smallest or not args.task_ids else ()))
    environment = {"submission_dir": str(args.submission_dir) if args.submission_dir else None}
    try:
        receipt = build_preflight_receipt(
            repo=args.benchmark_repo,
            task_ids=task_ids,
            prior_receipts=args.prior_receipt,
            license_receipt=args.license_receipt,
            sampler_receipt=args.sampler_receipt,
            projected_tinker_usd=args.projected_tinker_usd,
            mode=args.mode,
            use_tinker=args.use_tinker,
            environment=environment,
        )
    except GateError as exc:
        raise SystemExit(f"FrontierSWE preflight failed: {exc}") from exc
    if receipt["status"] == "BLOCKED":
        write_receipt(args.out, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 2
    if args.mode == "run":
        if args.submission_dir is None:
            receipt["status"] = "BLOCKED"
            receipt["blockers"].append(
                Blocker("submission_missing", "no model-produced candidate submission was supplied", "the Qwen candidate workspace").as_dict()
            )
            receipt["score"] = None
            write_receipt(args.out, receipt)
            return 2
        task_id = task_ids[0]
        task_dir = args.benchmark_repo.resolve() / "tasks" / task_id
        spec = _task_spec(task_dir)
        image = spec["environment"]["docker_image"]
        mount = args.candidate_mount or DEFAULT_CANDIDATE_MOUNTS.get(task_id)
        if not mount:
            raise SystemExit("--candidate-mount is required for this task")
        output_dir = args.out.with_suffix(args.out.suffix + ".verifier")
        cpus = int(spec["environment"].get("cpus", 1))
        memory_mb = int(spec["environment"].get("memory_mb", 4096))
        code, reward, logs = _native_verify(
            task_dir=task_dir,
            image=image,
            submission_dir=args.submission_dir.resolve(),
            candidate_mount=mount,
            output_dir=output_dir,
            cpus=cpus,
            memory_mb=memory_mb,
        )
        receipt["status"] = "COMPLETED" if code == 0 and reward is not None else "FAILED"
        receipt["verifier"] = {"returncode": code, "logs_tail": logs, "reward": reward}
        receipt["score"] = reward.get("score", reward.get("reward")) if reward else None
    write_receipt(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"READY", "COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
