#!/usr/bin/env python3
"""Archive W&B logs, Hugging Face cache snapshots, and Tinker checkpoints locally.

This script intentionally reads service tokens only from environment variables and
does not write them into the archive. Large artifacts are bounded by disk budgets;
skipped items are still recorded in manifests.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import math
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


ARCHIVE_PROTOCOL = "owned_archive_v3_aggregate_budget"
CONTROL_RESERVE_BYTES = 64 * 1024


class BudgetExceeded(RuntimeError):
    pass


class ArchiveBudget:
    """Conservative cumulative bytes written, including temporary/replaced files.

    Deadlines are checked before and after each local I/O chunk. This does not
    promise interruption of a stalled filesystem syscall. Remote SDK/CLI paths
    are unavailable until they support enforceable byte and lifetime quotas.
    """

    def __init__(self, max_bytes=8 * 1024**3, seconds=1800, reserve_bytes=0):
        if (not isinstance(max_bytes, int) or max_bytes < 0
                or not math.isfinite(seconds) or seconds <= 0
                or reserve_bytes < 0 or reserve_bytes > max_bytes):
            raise ValueError("invalid archive budget")
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes
        self.written_bytes = 0
        self.deadline = time.monotonic() + seconds

    def check(self, size=0, *, control=False):
        if not control and time.monotonic() >= self.deadline:
            raise BudgetExceeded("aggregate_deadline_exceeded")
        limit = self.max_bytes if control else self.max_bytes - self.reserve_bytes
        if size < 0 or self.written_bytes + size > limit:
            raise BudgetExceeded("aggregate_byte_budget_exceeded")

    def write(self, stream, data, *, control=False):
        self.check(len(data), control=control)
        # Charge before I/O: a failed/partial write never replenishes the quota.
        self.written_bytes += len(data)
        stream.write(data)
        self.check(control=control)

    def receipt(self):
        return {"protocol": ARCHIVE_PROTOCOL, "max_bytes": self.max_bytes,
                "charged_write_bytes": self.written_bytes,
                "deadline_expired": time.monotonic() >= self.deadline,
                "deadline_scope": "cooperative_local_io; remote_operations_disabled",
                "accounting": "cumulative_writes_including_failed_and_replaced_files"}


def atomic_bytes(path, data, budget, *, control=False):
    budget.check(len(data), control=control)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".archive-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            budget.write(stream, data, control=control)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def log(message: str) -> None:
    print(f"[archive] {message}", flush=True)


def write_json(path: Path, obj: Any, budget: ArchiveBudget | None = None) -> None:
    data = (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode()
    atomic_bytes(path, data, budget or ArchiveBudget())


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


def hardlink_or_copy(src: str, dst: str, budget: ArchiveBudget | None = None) -> None:
    """Compatibility name: bounded owned copy, never a shared evidence inode."""
    budget = budget or ArchiveBudget()
    source, target = Path(src), Path(dst)
    if source.resolve() == target.resolve():
        raise ValueError("archive source and destination must differ")
    budget.check()
    fd_in = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    temporary = None
    try:
        with os.fdopen(fd_in, "rb") as reader:
            info = os.fstat(reader.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("archive source must be a regular file")
            budget.check(info.st_size)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix=".archive-", dir=target.parent)
            temporary = Path(name)
            with os.fdopen(fd, "wb") as writer:
                while True:
                    budget.check()
                    chunk = reader.read(64 * 1024)
                    if not chunk:
                        break
                    budget.write(writer, chunk)
            budget.check()
            os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ignore_sensitive_cache_files(_: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lowered = name.lower()
        if lowered in {"token", "stored_tokens", ".netrc"}:
            ignored.add(name)
        elif lowered.endswith((".token", ".secret", ".key", ".lock")):
            ignored.add(name)
    return ignored


def copy_tree_hardlinked(src: Path, dst: Path, budget: ArchiveBudget | None = None) -> dict[str, Any]:
    budget = budget or ArchiveBudget()
    if not src.exists():
        return {"copied": False, "reason": "source_missing", "source": str(src)}
    source, target = src.resolve(), dst.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("archive source and destination must not overlap")
    if dst.exists() or dst.is_symlink():
        raise ValueError("archive destination must be fresh")
    before = budget.written_bytes
    files = 0
    try:
        budget.check()
        dst.mkdir(parents=True)
        for directory, dirs, names in os.walk(src, followlinks=False):
            budget.check()
            ignored = ignore_sensitive_cache_files(directory, dirs + names)
            # Do not introduce links that could escape the owned archive tree.
            dirs[:] = [d for d in dirs if d not in ignored and not (Path(directory) / d).is_symlink()]
            for name in names:
                budget.check()
                entry = Path(directory) / name
                if name in ignored or entry.is_symlink():
                    continue
                hardlink_or_copy(str(entry), str(dst / entry.relative_to(src)), budget)
                files += 1
        return {"copied": True, "files": files, "written_bytes": budget.written_bytes - before,
                "mode": ARCHIVE_PROTOCOL, "symlinks": "skipped"}
    except (BudgetExceeded, OSError, ValueError) as exc:
        # Only this fresh destination is owned. Never remove source evidence.
        if dst.exists():
            shutil.rmtree(dst)
        return {"copied": False, "reason": str(exc), "partial_files_removed": True,
                "charged_write_bytes": budget.written_bytes - before}


def archive_local_wandb(out_root: Path, budget: ArchiveBudget | None = None) -> dict[str, Any]:
    src = REPO_ROOT / "atropos" / "wandb"
    dst = out_root / "wandb_local" / "atropos_wandb"
    result = copy_tree_hardlinked(src, dst, budget)
    if result.get("copied"):
        result["run_dirs"] = sorted(p.name for p in dst.glob("run-*") if p.is_dir())
    return result


def archive_wandb_cloud(out_root: Path, projects: list[str], max_runs: int, history_limit: int, max_file_mb: float, budget: ArchiveBudget | None = None) -> dict[str, Any]:
    # SDK metadata/history pagination and downloads lack a shared hard quota.
    # A timeout argument or provider size estimate cannot establish that quota.
    return {"enabled": False, "reason": "unsupported_bounded_remote_transfer",
            "protocol": ARCHIVE_PROTOCOL, "skipped": ["metadata", "history", "files"]}


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


def archive_hf(out_root: Path, remote_metadata_limit: int, budget: ArchiveBudget | None = None) -> dict[str, Any]:
    budget = budget or ArchiveBudget()
    cache_root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))).expanduser()
    result = copy_tree_hardlinked(cache_root, out_root / "huggingface" / "cache_hardlinked", budget)
    return {"enabled": True, "cache_copy": result,
            "remote_metadata_note": "unsupported_bounded_remote_transfer",
            "protocol": ARCHIVE_PROTOCOL}


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


def archive_tinker(out_root: Path, tinker_bin: Path, budget_gb: float, reserve_free_gb: float, max_downloads: int, budget: ArchiveBudget | None = None) -> dict[str, Any]:
    # CLI listing capture and downloader writes are not quota-controlled. Fail
    # closed before spawning, even when provider metadata advertises small files.
    return {"enabled": False, "listed": False, "downloaded_count": 0,
            "reason": "unsupported_bounded_remote_transfer", "protocol": ARCHIVE_PROTOCOL,
            "skipped": ["checkpoint_listing", "checkpoint_downloads"]}


def redact_secrets(root: Path, budget: ArchiveBudget | None = None) -> dict[str, Any]:
    budget = budget or ArchiveBudget()
    exact_values = []
    for key in ["WANDB_API_KEY", "TINKER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]:
        value = os.environ.get(key)
        if value:
            exact_values.append(value.encode("utf-8"))
    changed_files = []
    scanned_files = 0
    skipped_large = 0
    for path in root.rglob("*"):
        budget.check()
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
            # A legacy archive may share an inode with the original evidence.
            # Replacing the archive entry never mutates that shared inode.
            fd, name = tempfile.mkstemp(prefix=".redacted-", dir=path.parent)
            temporary = Path(name)
            try:
                with os.fdopen(fd, "wb") as stream:
                    budget.write(stream, redacted)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            changed_files.append(str(path.relative_to(root)))
    return {
        "scanned_files": scanned_files,
        "skipped_large_or_binary_files": skipped_large,
        "redacted_file_count": len(changed_files),
        "redacted_files": changed_files,
    }


def write_readme(out_root: Path, manifest: dict[str, Any], budget: ArchiveBudget | None = None) -> None:
    text = ("# Local Artifact Archive\n\nProtocol: " + ARCHIVE_PROTOCOL + "\n"
            "Owned local W&B/HF copies only. Symlinks are skipped.\n"
            "Remote W&B metadata/history/files, HF metadata and Tinker CLI operations\n"
            "are disabled: unsupported_bounded_remote_transfer.\n"
            "One cumulative byte quota includes temporary writes and redaction.\n"
            "Local deadlines are cooperative; stalled filesystem calls cannot be interrupted.\n"
            "This receipt does not certify unattended execution or secret-free binary data.\n")
    atomic_bytes(out_root / "README.md", text.encode(), budget or ArchiveBudget())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-budget-gb", type=float, default=8.0)
    parser.add_argument("--archive-timeout-seconds", type=float, default=1800.0)
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
    # Validate before creating or writing anything. Existing archives/evidence
    # are never reused: their bytes and symlinks are outside this invocation.
    reserve = args.reserve_free_gb * 1024**3
    if not math.isfinite(reserve) or reserve < 0:
        raise ValueError("invalid free-space reserve")
    cap = args.archive_budget_gb * 1024**3
    if not math.isfinite(cap) or cap < CONTROL_RESERVE_BYTES:
        raise ValueError("archive budget must cover the 64 KiB control reserve")
    budget = ArchiveBudget(int(cap), args.archive_timeout_seconds, CONTROL_RESERVE_BYTES)
    out_root = (REPO_ROOT / args.output_root).resolve()
    if out_root.exists():
        raise ValueError("archive output root must be fresh")
    out_root.mkdir(parents=True)
    manifest = {"protocol": ARCHIVE_PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat()}
    exit_code = 0
    try:
        # Disk reserve reduces the same aggregate cap, never merely an estimate
        # of one provider's selected files.
        available = disk_free_bytes(out_root) - int(reserve)
        if available < budget.max_bytes:
            raise BudgetExceeded("insufficient_free_space_for_aggregate_budget")
        manifest["wandb_local"] = archive_local_wandb(out_root, budget)
        manifest["wandb_cloud"] = ({"enabled": False, "reason": "skipped_by_flag"} if args.skip_wandb_cloud else
            archive_wandb_cloud(out_root, args.wandb_project or DEFAULT_WANDB_PROJECTS,
                               args.wandb_max_runs, args.wandb_history_limit, args.wandb_max_file_mb, budget))
        manifest["huggingface"] = ({"enabled": False, "reason": "skipped_by_flag"} if args.skip_hf else
            archive_hf(out_root, args.hf_remote_metadata_limit, budget))
        manifest["tinker"] = ({"enabled": False, "reason": "skipped_by_flag"} if args.skip_tinker else
            archive_tinker(out_root, Path(args.tinker_bin), args.tinker_budget_gb,
                           args.reserve_free_gb, args.tinker_max_downloads, budget))
        redaction = redact_secrets(out_root, budget)
        manifest["redaction"] = {k: v for k, v in redaction.items() if k != "redacted_files"}
        write_readme(out_root, manifest, budget)
        for key in ("wandb_local", "huggingface"):
            result = manifest.get(key, {})
            result = result.get("cache_copy", result)
            if result.get("copied") is False and result.get("reason") != "source_missing":
                exit_code = 2
    except (BudgetExceeded, OSError, ValueError) as exc:
        manifest["error"] = str(exc)[:1000]
        # A failed redaction must not leave unredacted copied payload behind.
        for entry in out_root.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        manifest["payload_removed_after_failure"] = True
        exit_code = 2
    manifest["status"] = "partial_or_failed" if exit_code else "local_complete_remote_skipped"
    manifest["budget"] = budget.receipt()
    # Fixed-size, reserved terminal receipt survives payload/deadline exhaustion.
    # The receipt itself is counted; padding makes its exact size explicit.
    manifest["budget"]["charged_write_bytes"] += CONTROL_RESERVE_BYTES
    manifest["budget"]["terminal_receipt_bytes"] = CONTROL_RESERVE_BYTES
    data = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    if len(data) > CONTROL_RESERVE_BYTES:
        raise ValueError("terminal receipt exceeds reserved capacity")
    atomic_bytes(out_root / "MANIFEST.json", data.ljust(CONTROL_RESERVE_BYTES, b" "), budget, control=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
