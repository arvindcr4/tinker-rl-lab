#!/usr/bin/env python3
"""Build a NIA-friendly searchable view of the local artifact archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
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
SECRET_PATTERNS = [
    re.compile(rb"wandb_v1_[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"tml-[A-Za-z0-9_\-]{20,}"),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def looks_textual(path: Path, size: int, max_text_bytes: int) -> bool:
    if size > max_text_bytes:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    return bool(sample.strip()) and sum(byte < 9 for byte in sample) == 0


def redact_bytes(data: bytes, exact_values: list[bytes]) -> bytes:
    redacted = data
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(b"[REDACTED_SECRET]", redacted)
    for value in exact_values:
        redacted = redacted.replace(value, b"[REDACTED_SECRET]")
    return redacted


def category_for(path: Path) -> str:
    parts = path.parts
    if "wandb_cloud" in parts or "wandb_local" in parts:
        return "wandb"
    if "huggingface" in parts:
        return "huggingface"
    if "tinker" in parts:
        return "tinker"
    return "archive"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="artifacts/local_copies/latest")
    parser.add_argument("--dest", default="artifacts/local_copies/nia_index_payload/tinker-rl-lab-artifacts")
    parser.add_argument("--max-text-mb", type=float, default=25.0)
    args = parser.parse_args()

    source = (REPO_ROOT / args.source).resolve() if not Path(args.source).is_absolute() else Path(args.source)
    dest = (REPO_ROOT / args.dest).resolve() if not Path(args.dest).is_absolute() else Path(args.dest)
    if not source.exists():
        raise SystemExit(f"source does not exist: {source}")

    if dest.exists():
        shutil.rmtree(dest)
    (dest / "searchable").mkdir(parents=True)

    max_text_bytes = int(args.max_text_mb * 1024 * 1024)
    exact_values = [
        value.encode("utf-8")
        for key in ["WANDB_API_KEY", "TINKER_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]
        if (value := os.environ.get(key))
    ]

    inventory_path = dest / "ALL_FILES_INVENTORY.jsonl"
    counts: dict[str, int] = {
        "files_seen": 0,
        "files_copied_for_search": 0,
        "files_inventory_only": 0,
        "bytes_seen": 0,
        "bytes_copied_for_search": 0,
    }
    by_category: dict[str, dict[str, int]] = {}

    with inventory_path.open("w", encoding="utf-8") as inventory:
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError as exc:
                record = {"path": str(path.relative_to(source)), "error": str(exc), "included_in_search": False}
                inventory.write(json.dumps(record, sort_keys=True) + "\n")
                continue

            rel = path.relative_to(source)
            size = stat.st_size
            category = category_for(rel)
            counts["files_seen"] += 1
            counts["bytes_seen"] += size
            by_category.setdefault(category, {"files": 0, "bytes": 0, "searchable_files": 0})
            by_category[category]["files"] += 1
            by_category[category]["bytes"] += size

            include = looks_textual(path, size, max_text_bytes)
            record = {
                "path": str(rel),
                "category": category,
                "size_bytes": size,
                "suffix": path.suffix.lower(),
                "included_in_search": include,
            }
            if include:
                target = dest / "searchable" / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                data = redact_bytes(path.read_bytes(), exact_values)
                target.write_bytes(data)
                record["searchable_path"] = str(target.relative_to(dest))
                record["sha256"] = hashlib.sha256(data).hexdigest()
                counts["files_copied_for_search"] += 1
                counts["bytes_copied_for_search"] += len(data)
                by_category[category]["searchable_files"] += 1
            else:
                record["inventory_only_reason"] = "binary_or_larger_than_text_limit"
                if size <= 64 * 1024 * 1024:
                    try:
                        record["sha256"] = sha256_file(path)
                    except OSError:
                        pass
                counts["files_inventory_only"] += 1
            inventory.write(json.dumps(record, sort_keys=True) + "\n")

    source_manifest = source / "MANIFEST.json"
    if source_manifest.exists():
        shutil.copy2(source_manifest, dest / "ARCHIVE_MANIFEST.json")
    tinker_plan = source / "tinker" / "download_plan.json"
    if tinker_plan.exists():
        shutil.copy2(tinker_plan, dest / "TINKER_DOWNLOAD_PLAN.json")
    hf_manifest = source / "huggingface" / "manifest.json"
    if hf_manifest.exists():
        shutil.copy2(hf_manifest, dest / "HF_MANIFEST.json")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "dest": str(dest),
        "max_text_bytes": max_text_bytes,
        "counts": counts,
        "by_category": by_category,
    }
    write_json(dest / "NIA_PAYLOAD_MANIFEST.json", summary)

    readme = f"""# Tinker RL Lab Artifact Index Payload

Created: {summary['created_at']}

Source archive: `{source}`

This folder is the NIA indexing payload for the local W&B, Hugging Face, and
Tinker archive. It includes searchable text logs, configs, manifests, histories,
and metadata under `searchable/`.

Binary model/checkpoint blobs are intentionally not duplicated as searchable text.
They are represented in `ALL_FILES_INVENTORY.jsonl` with path, category, size,
and hashes for small files where practical.

Counts:

- Files seen: {counts['files_seen']}
- Files copied for search: {counts['files_copied_for_search']}
- Inventory-only files: {counts['files_inventory_only']}
- Bytes represented: {counts['bytes_seen']}
- Bytes copied for search: {counts['bytes_copied_for_search']}
"""
    (dest / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
