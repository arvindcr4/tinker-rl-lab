"""Explicit adapter export with immutable source/destination receipts.

Importing this module never reads credentials or mutates a remote repository.
Legacy unverified results are refused rather than silently exported as v3.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from platform_hybrid.experiments.framework_contracts import (
    export_branch,
    export_identity,
    require_completed,
    content_hash,
)

REPO = "arvindcr4/zvf-h2h-qwen3.5-4b"
ROOT = Path(__file__).resolve().parent / "results"
TINKER = ".venv/bin/tinker"


def branch_of(tag):
    return export_branch(tag)


def file_hashes(info):
    hashes = {}
    for item in info.siblings or []:
        if item.rfilename in {"adapter_model.safetensors", "adapter_config.json"}:
            lfs = item.lfs
            digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            digest = digest or item.blob_id
            if not digest:
                raise ValueError("Export file hash unavailable")
            hashes[item.rfilename] = digest
    if set(hashes) != {"adapter_model.safetensors", "adapter_config.json"}:
        raise ValueError("Complete adapter weights/config missing")
    return hashes


def push(tag, record, *, api, repo=REPO, download=None):
    from huggingface_hub import hf_hub_download

    download = download or hf_hub_download
    require_completed(record)
    source = export_identity(record)
    # Include source spec + checkpoint; same display tag can have distinct attempts.
    branch = branch_of(tag + "-" + content_hash(source))
    try:
        info = api.model_info(repo, revision=branch, files_metadata=True)
        receipt = json.loads(
            Path(
                download(
                    repo, "export_receipt.json", revision=info.sha, token=os.environ.get("HF_TOKEN")
                )
            ).read_text()
        )
        if receipt.get("source") == source and receipt.get("file_hashes") == file_hashes(info):
            return {
                "branch": branch,
                "status": "verified",
                "revision": info.sha,
                "source": source,
                "file_hashes": receipt["file_hashes"],
            }
    except Exception:
        pass  # Missing or mismatched receipt is not an idempotent success.
    api.create_branch(repo, branch=branch, exist_ok=True)
    subprocess.run(
        [
            TINKER,
            "checkpoint",
            "push-hf",
            source["checkpoint"],
            "--repo",
            repo,
            "--revision",
            branch,
            "--no-model-card",
            "--commit-message",
            tag,
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        check=True,
    )
    info = api.model_info(repo, revision=branch, files_metadata=True)
    receipt = {"source": source, "file_hashes": file_hashes(info), "weights_commit": info.sha}
    commit = api.upload_file(
        path_or_fileobj=json.dumps(receipt, sort_keys=True).encode(),
        path_in_repo="export_receipt.json",
        repo_id=repo,
        revision=branch,
        commit_message="Record verified export source and file hashes",
    )
    verified = api.model_info(repo, revision=commit.oid, files_metadata=True)
    if verified.sha != commit.oid or file_hashes(verified) != receipt["file_hashes"]:
        raise ValueError("Destination hashes changed during export")
    return {**receipt, "branch": branch, "revision": commit.oid, "status": "verified"}


def load_cells(root=ROOT):
    cells = {}
    for path in sorted((root / "week_h2h").glob("*.json")) + sorted(root.glob("h2h2_*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "completed" or not record.get("tag"):
            continue
        tag = record["tag"]
        if tag in cells and cells[tag] != record:
            raise ValueError(f"Conflicting export records for {tag}")
        cells[tag] = record
    return cells


def main(max_workers=5):
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    cells = load_cells()
    # Validate all source identities before creating any remote objects.
    for record in cells.values():
        require_completed(record)
        export_identity(record)
    api.create_repo(REPO, private=True, exist_ok=True, repo_type="model")
    receipts, failures = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = {pool.submit(push, tag, record, api=api): tag for tag, record in cells.items()}
        for future in as_completed(pending):
            try:
                receipts.append(future.result())
            except Exception as exc:
                failures.append({"tag": pending[future], "error": str(exc)})
    # Index only verified immutable revisions; failed destinations are not linked.
    lines = [
        "# Verified adapter exports",
        "",
        "Methods are identified in source receipts; historical labels are not equivalence claims.",
        "",
    ]
    for r in receipts:
        lines.append(f"- [{r['branch']}](https://huggingface.co/{REPO}/tree/{r['revision']})")
    api.upload_file(
        path_or_fileobj="\n".join(lines).encode(),
        path_in_repo="README.md",
        repo_id=REPO,
        revision="main",
        commit_message="Index verified immutable exports",
    )
    print(json.dumps({"verified": receipts, "failed": failures}, indent=2))


if __name__ == "__main__":
    main()
