"""Fail-closed runner for the exact public SWE-bench Pro evaluation.

This module deliberately keeps patch generation separate from patch execution.
The Scale AI evaluator is the authority for repository images, test commands,
and result parsing; this wrapper only pins the benchmark, prepares an isolated
runtime, validates provenance/budget gates, and invokes that evaluator when the
gates are complete.

The default invocation is a one-task preflight.  It never treats a gold patch,
an evaluator smoke, or a planned run as model evidence.  A model run requires
an explicit patch JSON artifact and an explicit data-license receipt because
the public Hugging Face card currently does not declare a data licence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SUITE_ID = "swe_bench_pro_eval"
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
DATASET_SPLIT = "test"
DATASET_URL = "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro"
DATASET_RAW_README = (
    "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro/raw/"
    f"{DATASET_REVISION}/README.md"
)
EVALUATOR_REPOSITORY = "scaleapi/SWE-bench_Pro-os"
EVALUATOR_URL = "https://github.com/scaleapi/SWE-bench_Pro-os"
EVALUATOR_REVISION = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
EVALUATOR_LICENSE = "MIT"
DOCKERHUB_NAMESPACE = "jefzda"
DOCKERHUB_REPOSITORY = "sweap-images"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TINKER_MAXIMUM_USD = 0.50
OPERATIONAL_CAP_USD = 16.50
SAFETY_RESERVE_USD = 1.50
WANDB_PROJECT = "tinker-rl-lab-pavlov"
DEFAULT_TASK_ID = (
    "instance_NodeBB__NodeBB-04998908ba6721d64eba79ae3b65a351dcfbc5b5-vnan"
)
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PreflightBlocked(RuntimeError):
    """Raised when a launch would violate an evidence or spend gate."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_request(url: str, *, timeout: float = 30.0) -> Any:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "tinker-rl-lab-e1/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _text_request(url: str, *, timeout: float = 30.0) -> str:
    request = urllib.request.Request(
        url,
        headers={"Accept": "text/plain", "User-Agent": "tinker-rl-lab-e1/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 60.0,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_runtime(root: Path | None = None) -> Path:
    return (root or _repo_root()) / ".e1-swe-bench-pro"


def _load_contract_model() -> tuple[str, str]:
    contract_path = Path(__file__).with_name("pavlovs_domain_contract.json")
    try:
        contract = json.loads(contract_path.read_text())
        for candidate in contract.get("model_candidates", []):
            if candidate.get("model_id") == MODEL_ID:
                revision = candidate.get("revision")
                if isinstance(revision, str) and HEX40.fullmatch(revision):
                    return MODEL_ID, revision
    except (OSError, json.JSONDecodeError):
        pass
    return MODEL_ID, MODEL_REVISION


def _normalise_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Dataset row to JSON-safe values without changing semantics."""

    result: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        result[str(key)] = value
    return result


def load_dataset_rows(
    *,
    cache_dir: Path | None = None,
    revision: str = DATASET_REVISION,
) -> list[dict[str, Any]]:
    """Load the pinned public split through ``datasets``.

    The import is intentionally lazy so unit tests and receipt inspection do
    not require benchmark dependencies in the global interpreter.
    """

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the host
        raise PreflightBlocked(
            "isolated benchmark environment is missing the `datasets` package"
        ) from exc
    kwargs: dict[str, Any] = {"split": DATASET_SPLIT, "revision": revision}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(cache_dir)
    try:
        dataset = load_dataset(DATASET_ID, **kwargs)
    except Exception as exc:  # pragma: no cover - network/cache dependent
        raise PreflightBlocked(
            f"could not acquire {DATASET_ID}@{revision}: {type(exc).__name__}"
        ) from exc
    return [_normalise_row(row) for row in dataset]


def task_id_hash(task_ids: Iterable[str]) -> str:
    ids = sorted(set(task_ids))
    return sha256_text("\n".join(ids) + ("\n" if ids else ""))


def split_manifest_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    manifest = []
    for row in sorted(rows, key=lambda item: str(item.get("instance_id", ""))):
        manifest.append(
            {
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "base_commit": row.get("base_commit"),
                "dockerhub_tag": row.get("dockerhub_tag"),
                "selected_test_files_to_run": row.get("selected_test_files_to_run"),
            }
        )
    return sha256_text(canonical_json(manifest))


def _parse_declared_license(readme: str) -> str | None:
    for line in readme.splitlines():
        match = re.match(r"^\s*license\s*:\s*([^#]+?)\s*$", line, re.IGNORECASE)
        if match:
            value = match.group(1).strip().strip("'\"")
            if value and value.lower() not in {"unknown", "null", "none"}:
                return value
    return None


def fetch_benchmark_metadata() -> dict[str, Any]:
    api_url = f"https://huggingface.co/api/datasets/{DATASET_ID}"
    try:
        metadata = _json_request(api_url)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise PreflightBlocked(f"authoritative dataset metadata unavailable: {exc}") from exc
    try:
        readme = _text_request(DATASET_RAW_README)
    except (OSError, urllib.error.URLError) as exc:
        readme = ""
    card = metadata.get("cardData") or {}
    dataset_info = card.get("dataset_info") or {}
    splits = dataset_info.get("splits") or []
    test_split = next((item for item in splits if item.get("name") == DATASET_SPLIT), {})
    return {
        "dataset_id": DATASET_ID,
        "revision": metadata.get("sha"),
        "last_modified": metadata.get("lastModified"),
        "private": bool(metadata.get("private")),
        "gated": bool(metadata.get("gated")),
        "disabled": bool(metadata.get("disabled")),
        "declared_license": _parse_declared_license(readme),
        "readme_sha256": sha256_text(readme) if readme else None,
        "split": DATASET_SPLIT,
        "row_count": test_split.get("num_examples"),
        "api_url": api_url,
        "readme_url": DATASET_RAW_README,
    }


def _decode_github_content(payload: Mapping[str, Any]) -> bytes:
    content = str(payload.get("content", "")).replace("\n", "")
    if not content:
        return b""
    return base64.b64decode(content)


def fetch_evaluator_metadata() -> dict[str, Any]:
    commit_url = (
        f"https://api.github.com/repos/{EVALUATOR_REPOSITORY}/commits/"
        f"{EVALUATOR_REVISION}"
    )
    license_url = (
        f"https://api.github.com/repos/{EVALUATOR_REPOSITORY}/contents/LICENSE?"
        f"ref={EVALUATOR_REVISION}"
    )
    try:
        commit = _json_request(commit_url)
        license_payload = _json_request(license_url)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise PreflightBlocked(f"official evaluator metadata unavailable: {exc}") from exc
    license_bytes = _decode_github_content(license_payload)
    return {
        "repository": EVALUATOR_REPOSITORY,
        "revision": commit.get("sha"),
        "license": EVALUATOR_LICENSE,
        "license_blob_sha": license_payload.get("sha"),
        "license_sha256": sha256_bytes(license_bytes) if license_bytes else None,
        "commit_url": commit_url,
        "license_url": license_url,
    }


def acquire_evaluator_checkout(runtime_dir: Path) -> Path:
    checkout = runtime_dir / "evaluator"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if (checkout / ".git").exists():
        current = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
        if current.returncode == 0 and current.stdout.strip() == EVALUATOR_REVISION:
            return checkout
        fetch = _run(["git", "fetch", "--depth", "1", "origin", EVALUATOR_REVISION], cwd=checkout)
        if fetch.returncode == 0:
            checkout_result = _run(["git", "checkout", "--detach", EVALUATOR_REVISION], cwd=checkout)
            if checkout_result.returncode == 0:
                return checkout
        raise PreflightBlocked(
            f"evaluator checkout exists but is not pinned to {EVALUATOR_REVISION}"
        )
    clone = _run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            EVALUATOR_URL + ".git",
            str(checkout),
        ],
        timeout=300,
    )
    if clone.returncode != 0:
        raise PreflightBlocked("could not clone the official SWE-bench Pro evaluator")
    fetch = _run(["git", "fetch", "--depth", "1", "origin", EVALUATOR_REVISION], cwd=checkout, timeout=300)
    checkout_result = _run(["git", "checkout", "--detach", EVALUATOR_REVISION], cwd=checkout)
    if fetch.returncode != 0 or checkout_result.returncode != 0:
        raise PreflightBlocked(
            f"official evaluator clone could not be pinned to {EVALUATOR_REVISION}"
        )
    return checkout


def inspect_local_evaluator(checkout: Path) -> dict[str, Any]:
    if not checkout.is_dir():
        return {"present": False, "revision": None, "license": None}
    current = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
    license_path = checkout / "LICENSE"
    license_sha = sha256_file(license_path) if license_path.exists() else None
    return {
        "present": True,
        "revision": current.stdout.strip() if current.returncode == 0 else None,
        "license": EVALUATOR_LICENSE if license_path.exists() else None,
        "license_sha256": license_sha,
        "has_runner": (checkout / "swe_bench_pro_eval.py").is_file(),
        "has_run_scripts": (checkout / "run_scripts").is_dir(),
        "has_instance_dockerfiles": (checkout / "dockerfiles" / "instance_dockerfile").is_dir(),
    }


def _docker_image_tag(row: Mapping[str, Any]) -> str:
    repo = str(row.get("repo", ""))
    instance_id = str(row.get("instance_id", ""))
    if "/" not in repo or not instance_id:
        raise ValueError("dataset row lacks repo/instance_id")
    repo_base, repo_name = repo.lower().split("/", 1)
    suffix = instance_id.removeprefix("instance_")
    if suffix.endswith("-vnan"):
        suffix = suffix[:-5]
    if "element-hq" in repo and "element-web" in repo:
        repo_name = "element-web" if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan" else "element"
    tag = f"{repo_base}.{repo_name}-{suffix}"
    return tag[:128]


def docker_image_uri(row: Mapping[str, Any]) -> str:
    return f"{DOCKERHUB_NAMESPACE}/{DOCKERHUB_REPOSITORY}:{_docker_image_tag(row)}"


def inspect_docker_image(row: Mapping[str, Any]) -> dict[str, Any]:
    uri = docker_image_uri(row)
    result = _run(["docker", "manifest", "inspect", "--verbose", uri], timeout=90)
    if result.returncode != 0:
        return {"uri": uri, "available": False, "digest": None, "error": "manifest unavailable"}
    try:
        payload = json.loads(result.stdout)
        descriptor = payload.get("Descriptor") or {}
        return {
            "uri": uri,
            "available": bool(descriptor.get("digest")),
            "digest": descriptor.get("digest"),
            "platform": descriptor.get("platform"),
        }
    except json.JSONDecodeError:
        return {"uri": uri, "available": False, "digest": None, "error": "invalid manifest response"}


def inspect_docker_host() -> dict[str, Any]:
    result = _run(["docker", "info", "--format", "{{json .}}"], timeout=30)
    if result.returncode != 0:
        return {"available": False, "error": "docker info failed"}
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"available": False, "error": "docker info returned invalid JSON"}
    memory_bytes = int(info.get("MemTotal", 0) or 0)
    cpus = int(info.get("NCPU", 0) or 0)
    return {
        "available": True,
        "architecture": info.get("Architecture"),
        "cpus": cpus,
        "memory_bytes": memory_bytes,
        "memory_gib": round(memory_bytes / (1024**3), 3) if memory_bytes else 0,
        "recommended_cpus": 8,
        "recommended_memory_gib": 16,
        "meets_recommended_resources": cpus >= 8 and memory_bytes >= 16 * 1024**3,
    }


def inspect_isolated_environment(runtime_dir: Path) -> dict[str, Any]:
    venv_dir = runtime_dir / "venv"
    python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    present = python_path.is_file()
    packages: dict[str, bool] = {}
    if present:
        probe = _run(
            [str(python_path), "-c", "import importlib.util, json; print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in ['datasets','pandas','tqdm','docker','huggingface_hub']}))"],
            timeout=30,
        )
        if probe.returncode == 0:
            try:
                packages = json.loads(probe.stdout.strip())
            except json.JSONDecodeError:
                packages = {}
    return {
        "runtime_dir": str(runtime_dir),
        "venv_dir": str(venv_dir),
        "python": str(python_path) if present else None,
        "present": present,
        "packages": packages,
        "all_required_packages": present and all(packages.get(name, False) for name in ("datasets", "pandas", "tqdm", "docker", "huggingface_hub")),
    }


def bootstrap_isolated_environment(runtime_dir: Path, evaluator: Path | None = None) -> Path:
    """Create and populate the per-worktree environment; never touch global Python."""

    venv_dir = runtime_dir / "venv"
    if not (venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")).exists():
        runtime_dir.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
    python_path = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    requirements = runtime_dir / "requirements.lock.txt"
    requirements.write_text(
        "datasets==4.4.1\n"
        "pandas==2.3.3\n"
        "tqdm==4.67.1\n"
        "docker==7.1.0\n"
        "huggingface_hub==1.1.2\n"
    )
    pip = _run([str(python_path), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)], timeout=900)
    if pip.returncode != 0:
        raise PreflightBlocked("isolated SWE-bench Pro dependency installation failed")
    return python_path


def _load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightBlocked(f"invalid JSON receipt: {path}") from exc


def validate_license_receipt(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "present": False,
            "approved": False,
            "status": "not_declared_by_authoritative_dataset_card",
        }
    receipt = _load_json_file(path)
    approved = (
        isinstance(receipt, Mapping)
        and receipt.get("approved") is True
        and receipt.get("dataset_id") == DATASET_ID
        and receipt.get("dataset_revision") == DATASET_REVISION
        and isinstance(receipt.get("license"), str)
        and receipt.get("license", "").strip().lower() not in {"", "unknown", "unspecified"}
        and isinstance(receipt.get("source_url"), str)
        and receipt.get("source_url", "").startswith("http")
    )
    return {
        "present": True,
        "approved": approved,
        "path": str(path),
        "dataset_id": receipt.get("dataset_id") if isinstance(receipt, Mapping) else None,
        "dataset_revision": receipt.get("dataset_revision") if isinstance(receipt, Mapping) else None,
        "license": receipt.get("license") if isinstance(receipt, Mapping) else None,
        "source_url": receipt.get("source_url") if isinstance(receipt, Mapping) else None,
    }


def select_checkpoint(path: Path | None) -> dict[str, Any]:
    """Select the strongest completed compatible immutable HF sampler receipt."""

    base_model, base_revision = _load_contract_model()
    fallback = {
        "source": "base_model",
        "model_id": base_model,
        "model_revision": base_revision,
        "adapter_revision": None,
        "hf_commit": None,
        "selection_reason": "no completed compatible Tinker sampler checkpoint supplied",
    }
    if path is None:
        return fallback
    payload = _load_json_file(path)
    candidates = payload if isinstance(payload, list) else payload.get("checkpoints", []) if isinstance(payload, Mapping) else []
    compatible: list[dict[str, Any]] = []
    ignored: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_model = candidate.get("model_id") or candidate.get("model")
        commit = candidate.get("hf_commit") or candidate.get("hf_commit_sha") or candidate.get("commit_sha")
        complete = candidate.get("status", "complete") in {"complete", "COMPLETE", "finished"}
        if candidate_model != base_model or not complete or not isinstance(commit, str) or not HEX40.fullmatch(commit):
            ignored.append(str(candidate.get("step", candidate.get("name", "unknown"))))
            continue
        item = dict(candidate)
        item["_step"] = int(candidate.get("step", 0) or 0)
        compatible.append(item)
    if not compatible:
        fallback["ignored_candidates"] = ignored
        fallback["selection_reason"] = "no completed compatible Tinker sampler checkpoint; immutable base revision recorded"
        return fallback
    selected = max(compatible, key=lambda item: item["_step"])
    return {
        "source": "tinker_sampler_checkpoint",
        "model_id": base_model,
        "model_revision": base_revision,
        "adapter_revision": selected.get("adapter_revision") or selected.get("revision"),
        "hf_commit": selected.get("hf_commit") or selected.get("hf_commit_sha") or selected.get("commit_sha"),
        "hf_repo": selected.get("hf_repo") or selected.get("repo_id"),
        "step": selected.get("step", 0),
        "selection_reason": "strongest completed compatible checkpoint by step",
    }


def estimate_tinker_cost(
    *,
    task_count: int,
    prompt_tokens: int = 0,
    sample_tokens: int = 0,
    train_tokens: int = 0,
    samples_per_task: int = 1,
) -> float:
    return round(
        (
            0.54 * prompt_tokens * task_count * samples_per_task
            + 1.335 * sample_tokens * task_count * samples_per_task
            + 1.177 * train_tokens
        )
        / 1_000_000,
        9,
    )


def validate_patch_artifact(path: Path | None, task_ids: Sequence[str]) -> dict[str, Any]:
    if path is None:
        return {"present": False, "valid": False, "path": None, "count": 0}
    payload = _load_json_file(path)
    if not isinstance(payload, list):
        return {"present": True, "valid": False, "path": str(path), "count": 0, "error": "patch JSON must be a list"}
    seen: list[str] = []
    valid = True
    error = None
    for item in payload:
        if not isinstance(item, Mapping) or not isinstance(item.get("instance_id"), str):
            valid = False
            error = "each patch entry needs instance_id and patch"
            break
        instance_id = item["instance_id"]
        patch = item.get("patch", item.get("model_patch"))
        if not isinstance(patch, str) or not patch.strip():
            valid = False
            error = f"empty patch for {instance_id}"
            break
        seen.append(instance_id)
    if len(seen) != len(set(seen)):
        valid = False
        error = "duplicate patch instance_id"
    if sorted(seen) != sorted(task_ids):
        valid = False
        error = "patch IDs must exactly equal the selected disjoint task IDs"
    return {"present": True, "valid": valid, "path": str(path), "count": len(payload), "task_id_hash": task_id_hash(seen), "error": error}


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def build_preflight(
    *,
    runtime_dir: Path,
    task_ids: Sequence[str] | None = None,
    training_task_ids: Sequence[str] = (),
    license_receipt_path: Path | None = None,
    checkpoint_receipt_path: Path | None = None,
    patch_path: Path | None = None,
    evaluator_dir: Path | None = None,
    acquire_evaluator: bool = False,
    bootstrap_environment: bool = False,
    require_tinker: bool = False,
    network: bool = True,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    root = _repo_root()
    selected_ids = list(task_ids or [DEFAULT_TASK_ID])
    if len(selected_ids) != len(set(selected_ids)):
        blockers.append("evaluation task IDs contain duplicates")
    if not selected_ids:
        blockers.append("at least one exact public task ID is required")
    if set(selected_ids) & set(training_task_ids):
        blockers.append("training and primary evaluation task IDs overlap")

    benchmark: dict[str, Any]
    if network:
        try:
            benchmark = fetch_benchmark_metadata()
        except PreflightBlocked as exc:
            benchmark = {"dataset_id": DATASET_ID, "revision": None, "error": str(exc)}
            blockers.append(str(exc))
    else:
        benchmark = {"dataset_id": DATASET_ID, "revision": DATASET_REVISION, "split": DATASET_SPLIT, "row_count": None, "declared_license": None, "offline": True}
    if benchmark.get("revision") != DATASET_REVISION:
        blockers.append(f"dataset revision is not the pinned {DATASET_REVISION}")
    if benchmark.get("private") or benchmark.get("gated") or benchmark.get("disabled"):
        blockers.append("authoritative public SWE-bench Pro split is private, gated, or disabled")

    dataset_rows: list[dict[str, Any]]
    if rows is not None:
        dataset_rows = [_normalise_row(row) for row in rows]
    else:
        try:
            dataset_rows = load_dataset_rows(cache_dir=runtime_dir / "hf-cache")
        except PreflightBlocked as exc:
            dataset_rows = []
            blockers.append(str(exc))
    row_by_id = {str(row.get("instance_id")): row for row in dataset_rows}
    missing_ids = sorted(set(selected_ids) - set(row_by_id))
    if missing_ids:
        blockers.append("selected task IDs are not present in the pinned public split: " + ", ".join(missing_ids))
    selected_rows = [row_by_id[task_id] for task_id in selected_ids if task_id in row_by_id]
    if len(selected_rows) != len(selected_ids):
        selected_rows = []
    selected_manifest_hash = split_manifest_hash(selected_rows) if selected_rows else None

    evaluator: dict[str, Any]
    if network:
        try:
            evaluator = fetch_evaluator_metadata()
        except PreflightBlocked as exc:
            evaluator = {"repository": EVALUATOR_REPOSITORY, "revision": None, "error": str(exc)}
            blockers.append(str(exc))
    else:
        evaluator = {"repository": EVALUATOR_REPOSITORY, "revision": EVALUATOR_REVISION, "license": EVALUATOR_LICENSE, "offline": True}
    if evaluator.get("revision") != EVALUATOR_REVISION:
        blockers.append(f"official evaluator is not pinned to {EVALUATOR_REVISION}")
    if evaluator.get("license") != EVALUATOR_LICENSE:
        blockers.append("official evaluator MIT license receipt is missing")

    checkout = evaluator_dir or (runtime_dir / "evaluator")
    if acquire_evaluator:
        try:
            checkout = acquire_evaluator_checkout(runtime_dir)
        except PreflightBlocked as exc:
            blockers.append(str(exc))
    local_evaluator = inspect_local_evaluator(checkout)
    if not local_evaluator.get("present"):
        blockers.append("native SWE-bench Pro evaluator checkout is absent; acquire the pinned official checkout")
    else:
        if local_evaluator.get("revision") != EVALUATOR_REVISION:
            blockers.append(f"native evaluator checkout is not at {EVALUATOR_REVISION}")
        for key, label in (("has_runner", "runner"), ("has_run_scripts", "run_scripts"), ("has_instance_dockerfiles", "instance Dockerfiles")):
            if not local_evaluator.get(key):
                blockers.append(f"native evaluator checkout is missing {label}")

    environment = inspect_isolated_environment(runtime_dir)
    if bootstrap_environment:
        try:
            bootstrap_isolated_environment(runtime_dir, checkout if local_evaluator.get("present") else None)
            environment = inspect_isolated_environment(runtime_dir)
        except PreflightBlocked as exc:
            blockers.append(str(exc))
    if not environment.get("present"):
        blockers.append("benchmark dependencies must run in the per-worktree isolated venv; bootstrap it before launch")
    elif not environment.get("all_required_packages"):
        blockers.append("per-worktree venv is missing one or more native evaluator dependencies")

    docker_host = inspect_docker_host()
    if not docker_host.get("available"):
        blockers.append("Docker daemon is unavailable for the native repository verifier")
    elif not docker_host.get("meets_recommended_resources"):
        blockers.append("Docker host is below the official evaluator recommendation of 8 CPUs and 16 GiB RAM")
    image_receipts = [inspect_docker_image(row) for row in selected_rows] if selected_rows else []
    if any(not item.get("available") for item in image_receipts):
        blockers.append("one or more selected native SWE-bench Pro Docker images cannot be acquired")

    license_receipt = validate_license_receipt(license_receipt_path)
    if benchmark.get("declared_license"):
        license_receipt["dataset_card_license"] = benchmark["declared_license"]
    if not license_receipt.get("approved"):
        blockers.append("authoritative dataset data license is not approved; provide an immutable license receipt")

    model_id, model_revision = _load_contract_model()
    model: dict[str, Any] = {"model_id": model_id, "revision": model_revision, "source": "huggingface_base"}
    if network:
        try:
            model_meta = _json_request(f"https://huggingface.co/api/models/{model_id}")
            model["hub_revision"] = model_meta.get("sha")
            model["hub_private"] = bool(model_meta.get("private"))
            model["hub_gated"] = bool(model_meta.get("gated"))
            if model_meta.get("sha") != model_revision:
                blockers.append(f"model Hub revision does not match immutable {model_revision}")
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            model["error"] = str(exc)
            blockers.append("authoritative model revision could not be verified")
    model.update(select_checkpoint(checkpoint_receipt_path))
    if model.get("source") == "tinker_sampler_checkpoint" and not HEX40.fullmatch(str(model.get("hf_commit"))):
        blockers.append("evaluated Tinker sampler checkpoint is missing an immutable HF commit")

    patch_receipt = validate_patch_artifact(patch_path, selected_ids)
    if patch_path is not None and not patch_receipt.get("valid"):
        blockers.append("model patch artifact is invalid: " + str(patch_receipt.get("error")))
    if patch_path is None:
        blockers.append("no model patch artifact supplied; preflight cannot launch an evaluation")

    wandb = {
        "project": WANDB_PROJECT,
        "mode": "online",
        "api_key_present": bool(os.environ.get("WANDB_API_KEY")),
        "initialized": False,
        "required_before_tinker_call": True,
        "tinker_call_requested": require_tinker,
    }
    if require_tinker and not wandb["api_key_present"]:
        blockers.append("WANDB_API_KEY is required for online W&B initialization before any Tinker call")

    projected_cost = estimate_tinker_cost(task_count=len(selected_ids)) if require_tinker else 0.0
    budget = {
        "suite_maximum_usd": TINKER_MAXIMUM_USD,
        "projected_tinker_usd": projected_cost,
        "shared_maximum_usd": 18.0,
        "operational_cap_usd": OPERATIONAL_CAP_USD,
        "safety_reserve_usd": SAFETY_RESERVE_USD,
        "within_suite_cap": projected_cost <= TINKER_MAXIMUM_USD,
        "shared_cap_preserved": OPERATIONAL_CAP_USD + SAFETY_RESERVE_USD == 18.0,
    }
    if projected_cost > TINKER_MAXIMUM_USD:
        blockers.append(f"projected Tinker spend exceeds the hard ${TINKER_MAXIMUM_USD:.2f} suite cap")

    external_receipts = []
    if not license_receipt.get("approved"):
        external_receipts.append({"kind": "dataset_license", "required_fields": ["dataset_id", "dataset_revision", "license", "source_url", "approved=true"]})
    if patch_path is None:
        external_receipts.append({"kind": "model_patch_artifact", "required_fields": ["instance_id", "patch", "model_revision", "generation_run_id"]})
    if require_tinker and not wandb["api_key_present"]:
        external_receipts.append({"kind": "wandb_online", "required_fields": ["online=true", "run_id", "run_url"]})

    return {
        "schema_version": "e1-swe-bench-pro-receipt-v1",
        "suite_id": SUITE_ID,
        "status": "READY" if not _unique(blockers) else "BLOCKED",
        "launchable": not _unique(blockers),
        "launches_any_job": False,
        "generated_at": utc_now(),
        "benchmark": {
            **benchmark,
            "authority": "ScaleAI public SWE-bench Pro",
            "dataset_url": DATASET_URL,
            "revision_expected": DATASET_REVISION,
            "selected_task_ids": selected_ids,
            "selected_task_id_hash": task_id_hash(selected_ids),
            "training_task_ids": sorted(set(training_task_ids)),
            "training_task_id_hash": task_id_hash(training_task_ids),
            "task_ids_disjoint": not bool(set(selected_ids) & set(training_task_ids)),
            "selected_split_manifest_hash": selected_manifest_hash,
            "selected_row_count": len(selected_rows),
            "total_row_count": len(dataset_rows),
        },
        "evaluator": {**evaluator, "authority": EVALUATOR_URL, "revision_expected": EVALUATOR_REVISION, "local": local_evaluator, "docker_images": image_receipts},
        "native_environment": {"isolated": environment, "docker": docker_host, "host_platform": platform.platform()},
        "license": license_receipt,
        "model": model,
        "patch_artifact": patch_receipt,
        "wandb": wandb,
        "budget": budget,
        "blockers": _unique(blockers),
        "external_access_receipt_needed": external_receipts,
        "score": None,
        "evidence_class": "preflight_or_blocked" if _unique(blockers) else "ready_to_launch",
    }


def start_wandb_online(config: Mapping[str, Any]) -> dict[str, Any]:
    """Initialize W&B online before a future Tinker call, never offline."""

    if not os.environ.get("WANDB_API_KEY"):
        raise PreflightBlocked("WANDB_API_KEY is missing; refusing a Tinker call")
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise PreflightBlocked("wandb is missing from the isolated environment") from exc
    run = wandb.init(mode="online", project=WANDB_PROJECT, config=dict(config), reinit="finish_previous")
    return {
        "initialized": True,
        "mode": "online",
        "project": WANDB_PROJECT,
        "run_id": getattr(run, "id", None),
        "run_url": getattr(run, "url", None),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _official_eval_command(
    *,
    python_path: Path,
    evaluator: Path,
    raw_samples: Path,
    patches: Path,
    output_dir: Path,
    docker_platform: str | None,
) -> list[str]:
    command = [
        str(python_path),
        str(evaluator / "swe_bench_pro_eval.py"),
        f"--raw_sample_path={raw_samples}",
        f"--patch_path={patches}",
        f"--output_dir={output_dir}",
        f"--scripts_dir={evaluator / 'run_scripts'}",
        "--num_workers=1",
        f"--dockerhub_username={DOCKERHUB_NAMESPACE}",
        "--use_local_docker",
    ]
    if docker_platform:
        command.append(f"--docker_platform={docker_platform}")
    return command


def run_evaluation(
    *,
    receipt: dict[str, Any],
    runtime_dir: Path,
    evaluator: Path,
    rows: Sequence[Mapping[str, Any]],
    patch_path: Path,
    docker_platform: str | None = None,
) -> dict[str, Any]:
    if not receipt.get("launchable"):
        raise PreflightBlocked("launch blocked: " + "; ".join(receipt.get("blockers", [])))
    environment = receipt["native_environment"]["isolated"]
    python_path = Path(str(environment["python"]))
    output_dir = runtime_dir / "evaluation"
    raw_samples = runtime_dir / "selected_samples.jsonl"
    _write_rows(raw_samples, rows)
    command = _official_eval_command(
        python_path=python_path,
        evaluator=evaluator,
        raw_samples=raw_samples,
        patches=patch_path,
        output_dir=output_dir,
        docker_platform=docker_platform or ("linux/amd64" if platform.machine().lower() in {"arm64", "aarch64"} else None),
    )
    result = _run(command, cwd=evaluator, timeout=24 * 60 * 60)
    receipt["launches_any_job"] = True
    receipt["launch"] = {"command": command, "returncode": result.returncode, "output_dir": str(output_dir)}
    if result.returncode != 0:
        receipt["status"] = "FAILED"
        receipt["evidence_class"] = "execution_failed"
        receipt["blockers"] = _unique([*receipt.get("blockers", []), "official native evaluator exited non-zero"])
        return receipt
    results_path = output_dir / "eval_results.json"
    if not results_path.is_file():
        receipt["status"] = "FAILED"
        receipt["evidence_class"] = "execution_failed"
        receipt["blockers"] = _unique([*receipt.get("blockers", []), "official evaluator produced no eval_results.json"])
        return receipt
    results = _load_json_file(results_path)
    if not isinstance(results, Mapping):
        raise PreflightBlocked("official evaluator result is not a mapping")
    values = [bool(results.get(task_id)) for task_id in receipt["benchmark"]["selected_task_ids"]]
    receipt["score"] = {"resolved": sum(values), "total": len(values), "accuracy": sum(values) / len(values) if values else None, "per_task": dict(results)}
    receipt["status"] = "COMPLETE"
    receipt["evidence_class"] = "native_model_evaluation"
    return receipt


def _parse_task_ids(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        path = Path(value)
        if path.is_file():
            payload = _load_json_file(path)
            if isinstance(payload, Mapping):
                payload = payload.get("task_ids", payload.get("instance_ids", []))
            if not isinstance(payload, list):
                raise PreflightBlocked(f"task ID file must contain a JSON list: {path}")
            result.extend(str(item) for item in payload)
        else:
            result.append(value)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--out", type=Path, default=None, help="machine-readable receipt path")
    parser.add_argument("--runtime-dir", type=Path, default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--training-task-id", action="append", default=[])
    parser.add_argument("--license-receipt", type=Path, default=None)
    parser.add_argument("--checkpoint-receipt", type=Path, default=None)
    parser.add_argument("--patches", type=Path, default=None)
    parser.add_argument("--evaluator-dir", type=Path, default=None)
    parser.add_argument("--acquire-evaluator", action="store_true")
    parser.add_argument("--bootstrap", action="store_true", help="create/install the isolated venv")
    parser.add_argument("--tinker", action="store_true", help="request a Tinker-backed generation lane")
    parser.add_argument("--docker-platform", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _repo_root()
    runtime_dir = args.runtime_dir or _default_runtime(root)
    out = args.out or (root / "outputs" / "e1_swe_bench_pro" / "receipt.json")
    task_ids = _parse_task_ids(args.task_id) if args.task_id else [DEFAULT_TASK_ID]
    training_ids = _parse_task_ids(args.training_task_id) if args.training_task_id else []
    receipt = build_preflight(
        runtime_dir=runtime_dir,
        task_ids=task_ids,
        training_task_ids=training_ids,
        license_receipt_path=args.license_receipt,
        checkpoint_receipt_path=args.checkpoint_receipt,
        patch_path=args.patches,
        evaluator_dir=args.evaluator_dir,
        acquire_evaluator=args.acquire_evaluator,
        bootstrap_environment=args.bootstrap,
        require_tinker=args.tinker,
    )
    if args.command == "run" and receipt.get("launchable"):
        rows = load_dataset_rows(cache_dir=runtime_dir / "hf-cache")
        evaluator = Path(receipt["evaluator"]["local"].get("revision") and (args.evaluator_dir or runtime_dir / "evaluator"))
        receipt = run_evaluation(
            receipt=receipt,
            runtime_dir=runtime_dir,
            evaluator=evaluator,
            rows=[row for row in rows if row.get("instance_id") in task_ids],
            patch_path=Path(args.patches),
            docker_platform=args.docker_platform,
        )
    elif args.command == "run":
        receipt["status"] = "BLOCKED"
    _write_json(out, receipt)
    print(json.dumps({"status": receipt.get("status"), "launchable": receipt.get("launchable"), "receipt": str(out), "blockers": receipt.get("blockers", [])}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
