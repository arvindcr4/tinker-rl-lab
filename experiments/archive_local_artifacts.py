#!/usr/bin/env python3
"""Archive W&B logs, Hugging Face cache snapshots, and Tinker checkpoints locally.

This script intentionally reads service tokens only from environment variables and
does not write them into the archive. Large artifacts are bounded by disk budgets;
skipped items are still recorded in manifests.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WANDB_PROJECTS = [
    "arvindcr4-pes-university/tinker-agentic-smoke",
    "arvindcr4-pes-university/tinker-rl-lab-world-class",
    "arvindcr4-pes-university/tinker-rl-scaling",
    "arvindcr4-pes-university/atropos-tinker",
]
SECRET_PATTERNS = [
    re.compile(rb"wandb_v1_[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"tml-[A-Za-z0-9_\-]{20,}"),
]
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".out",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def log(message: str) -> None:
    print(f"[archive] {message}", flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def run_cmd(args: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def disk_free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return int(usage.free)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def safe_name(value: str, limit: int = 140) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("._")
    if len(cleaned) <= limit:
        return cleaned or "item"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[: limit - 13]}_{digest}"


def hardlink_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def ignore_sensitive_cache_files(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lowered = name.lower()
        if lowered in {"token", "stored_tokens", ".netrc"}:
            ignored.add(name)
        elif lowered.endswith((".token", ".secret", ".key", ".lock")):
            ignored.add(name)
    return ignored


def copy_tree_hardlinked(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        return {"copied": False, "reason": "source_missing", "source": str(src)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    before_free = disk_free_bytes(dst.parent)
    start = time.time()
    shutil.copytree(
        src,
        dst,
        copy_function=hardlink_or_copy,
        dirs_exist_ok=True,
        ignore=ignore_sensitive_cache_files,
        symlinks=True,
    )
    return {
        "copied": True,
        "source": str(src),
        "destination": str(dst),
        "logical_size_bytes": dir_size_bytes(src),
        "archive_logical_size_bytes": dir_size_bytes(dst),
        "free_bytes_before": before_free,
        "free_bytes_after": disk_free_bytes(dst.parent),
        "elapsed_seconds": round(time.time() - start, 3),
        "mode": "hardlink_or_copy_with_symlinks",
    }


def archive_local_wandb(out_root: Path) -> dict[str, Any]:
    src = REPO_ROOT / "atropos" / "wandb"
    dst = out_root / "wandb_local" / "atropos_wandb"
    result = copy_tree_hardlinked(src, dst)
    if result.get("copied"):
        result["run_dirs"] = sorted(p.name for p in dst.glob("run-*") if p.is_dir())
    return result


def archive_wandb_cloud(out_root: Path, projects: list[str], max_runs: int, history_limit: int, max_file_mb: float) -> dict[str, Any]:
    if "WANDB_API_KEY" not in os.environ:
        return {"enabled": False, "reason": "WANDB_API_KEY_not_set"}
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on env
        return {"enabled": False, "reason": f"wandb_import_failed: {exc}"}

    os.environ.setdefault("WANDB_SILENT", "true")
    api = wandb.Api(timeout=90)
    max_file_bytes = int(max_file_mb * 1024 * 1024)
    cloud_root = out_root / "wandb_cloud"
    summary: dict[str, Any] = {"enabled": True, "projects": {}, "errors": []}

    for project_path in projects:
        project_key = safe_name(project_path)
        project_dir = cloud_root / project_key
        project_summary = {
            "project": project_path,
            "runs_seen": 0,
            "runs_archived": 0,
            "files_downloaded": 0,
            "files_skipped_large": 0,
            "history_rows": 0,
            "errors": [],
        }
        summary["projects"][project_path] = project_summary
        log(f"W&B cloud: scanning {project_path}")
        try:
            runs_iter = api.runs(project_path, per_page=100)
            for run in runs_iter:
                if max_runs and project_summary["runs_archived"] >= max_runs:
                    break
                project_summary["runs_seen"] += 1
                run_name = safe_name(f"{run.id}_{run.name or 'run'}")
                run_dir = project_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                run_meta = {
                    "id": run.id,
                    "name": run.name,
                    "path": list(getattr(run, "path", []) or []),
                    "url": getattr(run, "url", None),
                    "state": getattr(run, "state", None),
                    "created_at": getattr(run, "created_at", None),
                    "updated_at": getattr(run, "updated_at", None),
                    "group": getattr(run, "group", None),
                    "job_type": getattr(run, "job_type", None),
                    "tags": list(getattr(run, "tags", []) or []),
                    "sweep": str(getattr(run, "sweep", None)) if getattr(run, "sweep", None) else None,
                }
                write_json(run_dir / "metadata.json", run_meta)
                try:
                    write_json(run_dir / "summary.json", dict(run.summary))
                except Exception as exc:
                    project_summary["errors"].append({"run": run.id, "stage": "summary", "error": str(exc)})
                try:
                    write_json(run_dir / "config.json", dict(run.config))
                except Exception as exc:
                    project_summary["errors"].append({"run": run.id, "stage": "config", "error": str(exc)})

                history_path = run_dir / "history.jsonl"
                history_count = 0
                try:
                    if history_path.exists():
                        history_path.unlink()
                    for row in run.scan_history(page_size=1000):
                        append_jsonl(history_path, row)
                        history_count += 1
                        if history_limit and history_count >= history_limit:
                            break
                    project_summary["history_rows"] += history_count
                except Exception as exc:
                    project_summary["errors"].append({"run": run.id, "stage": "history", "error": str(exc)})
                write_json(run_dir / "history_manifest.json", {"rows": history_count, "limit": history_limit or None})

                file_manifest = []
                try:
                    for file_obj in run.files(per_page=1000):
                        file_info = {
                            "name": getattr(file_obj, "name", None),
                            "size_bytes": getattr(file_obj, "size", None),
                            "md5": getattr(file_obj, "md5", None),
                            "url": getattr(file_obj, "url", None),
                        }
                        size = file_info["size_bytes"]
                        if isinstance(size, int) and size > max_file_bytes:
                            file_info["downloaded"] = False
                            file_info["skip_reason"] = f"larger_than_{max_file_mb:g}MB"
                            project_summary["files_skipped_large"] += 1
                        else:
                            try:
                                file_obj.download(root=str(run_dir / "files"), replace=True)
                                file_info["downloaded"] = True
                                project_summary["files_downloaded"] += 1
                            except Exception as exc:
                                file_info["downloaded"] = False
                                file_info["error"] = str(exc)
                        file_manifest.append(file_info)
                except Exception as exc:
                    project_summary["errors"].append({"run": run.id, "stage": "files", "error": str(exc)})
                write_json(run_dir / "files_manifest.json", file_manifest)
                project_summary["runs_archived"] += 1
        except Exception as exc:
            project_summary["errors"].append({"project": project_path, "stage": "runs", "error": str(exc)})
            summary["errors"].append({"project": project_path, "error": str(exc)})

    return summary


def list_hf_cache_repos(cache_root: Path) -> list[dict[str, Any]]:
    hub = cache_root / "hub"
    repos = []
    if not hub.exists():
        return repos
    for item in sorted(hub.iterdir()):
        if not item.is_dir():
            continue
        kind = None
        repo_id = None
        if item.name.startswith("models--"):
            kind = "model"
            repo_id = item.name.removeprefix("models--").replace("--", "/")
        elif item.name.startswith("datasets--"):
            kind = "dataset"
            repo_id = item.name.removeprefix("datasets--").replace("--", "/")
        elif item.name.startswith("spaces--"):
            kind = "space"
            repo_id = item.name.removeprefix("spaces--").replace("--", "/")
        if not repo_id:
            continue
        snapshots_dir = item / "snapshots"
        snapshots = sorted(p.name for p in snapshots_dir.iterdir() if p.is_dir()) if snapshots_dir.exists() else []
        repos.append(
            {
                "kind": kind,
                "repo_id": repo_id,
                "cache_dir": str(item),
                "snapshots": snapshots,
                "logical_size_bytes": dir_size_bytes(item),
            }
        )
    return repos


def discover_hf_references() -> list[str]:
    references: set[str] = set()
    patterns = [
        r"https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+/[A-Za-z0-9][A-Za-z0-9_.-]+)(?![A-Za-z0-9_.-])",
    ]
    include_globs = ["*.json", "*.yaml", "*.yml", "*.md", "*.py", "*.tex"]
    roots = [REPO_ROOT / "experiments", REPO_ROOT / "reports", REPO_ROOT / "atropos", REPO_ROOT / "paper"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not any(fnmatch.fnmatch(path.name, glob) for glob in include_globs):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for pattern in patterns:
                for match in re.findall(pattern, text):
                    if any(prefix in match for prefix in ["http", "tinker:", "wandb.ai"]):
                        continue
                    if match.count("/") == 1 and not match.startswith(("./", "../")):
                        left, right = match.split("/", 1)
                        if len(left) >= 2 and len(right) >= 2 and not left.endswith((".py", ".json", ".md", ".yaml", ".yml")):
                            references.add(match)
    return sorted(references)


def archive_hf(out_root: Path, remote_metadata_limit: int) -> dict[str, Any]:
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    dst = out_root / "huggingface" / "cache_hardlinked"
    log(f"HF: hardlinking cache from {cache_root}")
    cache_copy = copy_tree_hardlinked(cache_root, dst)
    cache_manifest = {
        "cache_copy": cache_copy,
        "cached_repos": list_hf_cache_repos(cache_root),
        "repo_references_in_project": discover_hf_references(),
    }
    metadata: dict[str, Any] = {"enabled": True, "cache": cache_manifest, "remote_model_metadata": {}, "errors": []}

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token and remote_metadata_limit:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            for repo_id in cache_manifest["repo_references_in_project"][:remote_metadata_limit]:
                try:
                    info = api.model_info(repo_id, files_metadata=True)
                    siblings = []
                    for sibling in getattr(info, "siblings", []) or []:
                        siblings.append(
                            {
                                "rfilename": getattr(sibling, "rfilename", None),
                                "size_bytes": getattr(sibling, "size", None),
                            }
                        )
                    metadata["remote_model_metadata"][repo_id] = {
                        "id": getattr(info, "id", repo_id),
                        "sha": getattr(info, "sha", None),
                        "private": getattr(info, "private", None),
                        "gated": getattr(info, "gated", None),
                        "downloads": getattr(info, "downloads", None),
                        "likes": getattr(info, "likes", None),
                        "tags": list(getattr(info, "tags", []) or []),
                        "siblings": siblings,
                        "total_file_size_bytes": sum(s.get("size_bytes") or 0 for s in siblings),
                    }
                except Exception as exc:
                    metadata["errors"].append({"repo_id": repo_id, "stage": "model_info", "error": str(exc)})
        except Exception as exc:
            metadata["errors"].append({"stage": "hf_import_or_api", "error": str(exc)})
    else:
        metadata["remote_metadata_note"] = "remote metadata skipped because token missing or limit is 0"

    write_json(out_root / "huggingface" / "manifest.json", metadata)
    return metadata


def discover_tinker_references() -> list[str]:
    refs: set[str] = set()
    roots = [REPO_ROOT / "experiments", REPO_ROOT / "reports", REPO_ROOT / "atropos", REPO_ROOT / "paper"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".yaml", ".yml", ".md", ".py", ".tex"}:
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for match in re.findall(r"tinker://[A-Za-z0-9:./_\-]+", text):
                refs.add(match.rstrip(".,);]}'\""))
    return sorted(refs)


def unwrap_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ["checkpoints", "data", "items", "results"]:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
    return []


def value_contains_tinker(value: Any) -> str | None:
    if isinstance(value, str) and value.startswith("tinker://"):
        return value
    return None


def checkpoint_path(record: dict[str, Any]) -> str | None:
    for key in ["tinker_path", "path", "checkpoint_path", "uri", "url"]:
        candidate = value_contains_tinker(record.get(key))
        if candidate:
            return candidate
    for value in record.values():
        candidate = value_contains_tinker(value)
        if candidate:
            return candidate
    return None


def parse_size_bytes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        match = re.match(r"([0-9.]+)\s*([KMGT]?I?B)", stripped, re.IGNORECASE)
        if match:
            number = float(match.group(1))
            unit = match.group(2).upper()
            power = {"B": 0, "KB": 1, "KIB": 1, "MB": 2, "MIB": 2, "GB": 3, "GIB": 3, "TB": 4, "TIB": 4}.get(unit, 0)
            return int(number * (1024**power))
    return None


def checkpoint_size(record: dict[str, Any]) -> int | None:
    for key in ["size_bytes", "archive_size_bytes", "storage_size_bytes", "file_size_bytes", "bytes", "size"]:
        size = parse_size_bytes(record.get(key))
        if size is not None:
            return size
    return None


def base_run_id(path: str) -> str:
    match = re.match(r"tinker://([^/:]+)", path)
    return match.group(1) if match else path


def tinker_priority(path: str, referenced: set[str], referenced_run_ids: set[str], size: int | None) -> tuple[int, int, str]:
    is_final = path.endswith("/final")
    is_sampler = "/sampler_weights/" in path
    run_ref = base_run_id(path) in referenced_run_ids
    if path in referenced:
        group = 0
    elif run_ref and is_final and is_sampler:
        group = 1
    elif run_ref and is_sampler:
        group = 2
    elif is_final and is_sampler:
        group = 3
    elif is_sampler:
        group = 4
    else:
        group = 5
    return (group, size if size is not None else 1024**4, path)


def archive_tinker(out_root: Path, tinker_bin: Path, budget_gb: float, reserve_free_gb: float, max_downloads: int) -> dict[str, Any]:
    env = os.environ.copy()
    if "TINKER_API_KEY" not in env:
        return {"enabled": False, "reason": "TINKER_API_KEY_not_set"}
    if not tinker_bin.exists():
        return {"enabled": False, "reason": f"tinker_binary_missing: {tinker_bin}"}

    tinker_root = out_root / "tinker"
    tinker_root.mkdir(parents=True, exist_ok=True)
    list_path = tinker_root / "checkpoints_list.json"
    log("Tinker: listing checkpoints")
    result = run_cmd([str(tinker_bin), "-f", "json", "checkpoint", "list", "--limit", "0"], env=env, timeout=300)
    list_path.write_text(result.stdout)
    list_meta = {
        "returncode": result.returncode,
        "stderr_tail": result.stderr[-4000:],
        "stdout_bytes": len(result.stdout.encode("utf-8")),
    }
    write_json(tinker_root / "checkpoints_list_command.json", list_meta)
    if result.returncode != 0:
        return {"enabled": True, "listed": False, "list_command": list_meta}

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return {"enabled": True, "listed": False, "error": f"json_decode_failed: {exc}", "list_command": list_meta}

    records = unwrap_records(raw)
    referenced_paths = set(discover_tinker_references())
    referenced_run_ids = {base_run_id(path) for path in referenced_paths}
    candidates: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in records:
        path = checkpoint_path(record)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        size = checkpoint_size(record)
        candidates.append(
            {
                "path": path,
                "size_bytes": size,
                "record": record,
                "referenced_exact": path in referenced_paths,
                "referenced_run": base_run_id(path) in referenced_run_ids,
                "priority": list(tinker_priority(path, referenced_paths, referenced_run_ids, size)),
            }
        )

    budget_bytes = int(budget_gb * 1024**3)
    reserve_bytes = int(reserve_free_gb * 1024**3)
    free_now = disk_free_bytes(tinker_root)
    usable_budget = max(0, min(budget_bytes, free_now - reserve_bytes))
    selected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    planned_bytes = 0
    for item in sorted(candidates, key=lambda x: tuple(x["priority"])):
        if max_downloads and len(selected) >= max_downloads:
            item["skip_reason"] = f"max_downloads_{max_downloads}_reached"
            skipped.append(item)
            continue
        estimated = item["size_bytes"] if item["size_bytes"] is not None else 1024**3
        if planned_bytes + estimated <= usable_budget:
            item["estimated_download_bytes"] = estimated
            selected.append(item)
            planned_bytes += estimated
        else:
            item["skip_reason"] = "outside_budget"
            item["estimated_download_bytes"] = estimated
            skipped.append(item)

    plan = {
        "budget_gb": budget_gb,
        "reserve_free_gb": reserve_free_gb,
        "free_bytes_at_plan": free_now,
        "usable_budget_bytes": usable_budget,
        "planned_bytes": planned_bytes,
        "referenced_paths": sorted(referenced_paths),
        "candidate_count": len(candidates),
        "selected": selected,
        "skipped": skipped,
    }
    write_json(tinker_root / "download_plan.json", plan)

    downloads = []
    checkpoints_dir = tinker_root / "checkpoints"
    for idx, item in enumerate(selected, start=1):
        path = item["path"]
        free_before = disk_free_bytes(tinker_root)
        estimated = item["estimated_download_bytes"]
        if free_before - estimated < reserve_bytes:
            status = {
                "path": path,
                "downloaded": False,
                "reason": "free_space_reserve_would_be_violated",
                "free_bytes_before": free_before,
                "estimated_bytes": estimated,
            }
            downloads.append(status)
            continue
        out_dir = checkpoints_dir / safe_name(path)
        log(f"Tinker: downloading {idx}/{len(selected)} {path}")
        started = time.time()
        proc = run_cmd([str(tinker_bin), "checkpoint", "download", path, "--output", str(out_dir), "--force"], env=env, timeout=1800)
        status = {
            "path": path,
            "output_dir": str(out_dir),
            "downloaded": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "elapsed_seconds": round(time.time() - started, 3),
            "free_bytes_before": free_before,
            "free_bytes_after": disk_free_bytes(tinker_root),
            "logical_size_bytes": dir_size_bytes(out_dir),
        }
        downloads.append(status)
        write_json(tinker_root / "downloads_progress.json", downloads)

    result_summary = {
        "enabled": True,
        "listed": True,
        "record_count": len(records),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "downloaded_count": sum(1 for d in downloads if d.get("downloaded")),
        "downloaded_bytes_logical": sum(d.get("logical_size_bytes", 0) for d in downloads),
        "downloads": downloads,
        "plan_path": str(tinker_root / "download_plan.json"),
        "list_path": str(list_path),
    }
    write_json(tinker_root / "manifest.json", result_summary)
    return result_summary


def redact_secrets(root: Path) -> dict[str, Any]:
    exact_values = []
    for key in ["WANDB_API_KEY", "TINKER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        value = os.environ.get(key)
        if value:
            exact_values.append(value.encode("utf-8"))
    changed_files = []
    scanned_files = 0
    skipped_large = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > 10 * 1024 * 1024:
            skipped_large += 1
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and size > 1024 * 1024:
            skipped_large += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        scanned_files += 1
        redacted = data
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub(b"[REDACTED_SECRET]", redacted)
        for value in exact_values:
            redacted = redacted.replace(value, b"[REDACTED_SECRET]")
        if redacted != data:
            path.write_bytes(redacted)
            changed_files.append(str(path.relative_to(root)))
    return {
        "scanned_files": scanned_files,
        "skipped_large_or_binary_files": skipped_large,
        "redacted_file_count": len(changed_files),
        "redacted_files": changed_files,
    }


def write_readme(out_root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Local Artifact Archive",
        "",
        f"Created: {manifest['created_at']}",
        "",
        "Contents:",
        "- `wandb_local/`: hardlinked/copy snapshot of local W&B run directories.",
        "- `wandb_cloud/`: run metadata, config, summary, history JSONL, and W&B run files within the file-size cap.",
        "- `huggingface/cache_hardlinked/`: local Hugging Face cache snapshot with model blobs hardlinked where possible.",
        "- `tinker/`: checkpoint list, download plan, and downloaded checkpoints within the disk budget.",
        "",
        "Notes:",
        "- API tokens are read from environment variables only and redacted from small text files after archiving.",
        "- Large skipped files/checkpoints are recorded in the manifests rather than silently ignored.",
        "- The HF cache copy uses hardlinks, so it preserves local model blobs without consuming another full copy of the cache.",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/local_copies/latest")
    parser.add_argument("--wandb-project", action="append", default=[], help="W&B entity/project; can be passed multiple times.")
    parser.add_argument("--wandb-max-runs", type=int, default=0, help="0 means all runs in each project.")
    parser.add_argument("--wandb-history-limit", type=int, default=0, help="0 means full scan_history.")
    parser.add_argument("--wandb-max-file-mb", type=float, default=512.0)
    parser.add_argument("--hf-remote-metadata-limit", type=int, default=200)
    parser.add_argument("--tinker-bin", default=str(REPO_ROOT / ".venv-at" / "bin" / "tinker"))
    parser.add_argument("--tinker-budget-gb", type=float, default=8.0)
    parser.add_argument("--reserve-free-gb", type=float, default=6.0)
    parser.add_argument("--tinker-max-downloads", type=int, default=0, help="0 means bounded only by budget.")
    parser.add_argument("--skip-wandb-cloud", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-tinker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = (REPO_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    projects = args.wandb_project or DEFAULT_WANDB_PROJECTS
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_root": str(out_root),
        "free_bytes_start": disk_free_bytes(out_root),
        "settings": {
            "wandb_projects": projects,
            "wandb_max_runs": args.wandb_max_runs,
            "wandb_history_limit": args.wandb_history_limit,
            "wandb_max_file_mb": args.wandb_max_file_mb,
            "hf_remote_metadata_limit": args.hf_remote_metadata_limit,
            "tinker_budget_gb": args.tinker_budget_gb,
            "reserve_free_gb": args.reserve_free_gb,
            "tinker_max_downloads": args.tinker_max_downloads,
        },
    }

    log(f"Writing archive under {out_root}")
    manifest["wandb_local"] = archive_local_wandb(out_root)
    write_json(out_root / "MANIFEST.json", manifest)

    if args.skip_wandb_cloud:
        manifest["wandb_cloud"] = {"enabled": False, "reason": "skipped_by_flag"}
    else:
        manifest["wandb_cloud"] = archive_wandb_cloud(
            out_root,
            projects,
            max_runs=args.wandb_max_runs,
            history_limit=args.wandb_history_limit,
            max_file_mb=args.wandb_max_file_mb,
        )
    write_json(out_root / "MANIFEST.json", manifest)

    if args.skip_hf:
        manifest["huggingface"] = {"enabled": False, "reason": "skipped_by_flag"}
    else:
        manifest["huggingface"] = archive_hf(out_root, args.hf_remote_metadata_limit)
    write_json(out_root / "MANIFEST.json", manifest)

    if args.skip_tinker:
        manifest["tinker"] = {"enabled": False, "reason": "skipped_by_flag"}
    else:
        manifest["tinker"] = archive_tinker(
            out_root,
            Path(args.tinker_bin),
            budget_gb=args.tinker_budget_gb,
            reserve_free_gb=args.reserve_free_gb,
            max_downloads=args.tinker_max_downloads,
        )
    write_json(out_root / "MANIFEST.json", manifest)

    manifest["redaction"] = redact_secrets(out_root)
    manifest["free_bytes_end"] = disk_free_bytes(out_root)
    manifest["archive_logical_size_bytes"] = dir_size_bytes(out_root)
    write_json(out_root / "MANIFEST.json", manifest)
    write_readme(out_root, manifest)
    log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
