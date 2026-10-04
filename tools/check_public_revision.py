#!/usr/bin/env python3
"""Validate an explicitly selected, separately versioned public thesis export.

Never generates receipts or refreshes expected hashes. The frozen October 3
checker remains a separate gate. Excludes only build/ and __pycache__/ scratch
subdirectories; all other exported files must appear in the manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

if __package__:
    from ._strict import require
    from .check_public_release import ROOT, HEX256, safe_file, sha256, strict_json, verify_manifest
    from .check_public_quality_receipt import verify_receipt, SCOPE
else:
    from _strict import require
    from check_public_release import ROOT, HEX256, safe_file, sha256, strict_json, verify_manifest
    from check_public_quality_receipt import verify_receipt, SCOPE


def inventory(root, directory):
    base = root / directory
    return {
        path.relative_to(root).as_posix()
        for path in base.rglob("*")
        if not any(part in {"build", "__pycache__"} for part in path.relative_to(base).parts)
        and (path.is_file() or path.is_symlink())
    }


def verify_availability(root, availability_path, expected_paths):
    """Verify exact local source-export bindings, without assessing withheld data."""
    availability_file = safe_file(root, availability_path)
    data = strict_json(availability_file)
    require(isinstance(data, dict), "source availability must be an object")
    require(data.get("schema") == "public-source-availability-v1", "unknown availability schema")
    for field in ("scope", "title", "date"):
        require(isinstance(data.get(field), str) and data[field].strip(), f"missing {field}")
    for field in ("withheld_categories", "residual_limits"):
        value = data.get(field)
        require(
            isinstance(value, list)
            and value
            and all(isinstance(v, str) and v.strip() for v in value),
            f"{field} must explicitly describe export limitations",
        )
    entries = data.get("included_files")
    require(isinstance(entries, list) and entries, "included_files must be nonempty")
    seen = set()
    for entry in entries:
        require(
            isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"},
            "invalid availability file entry",
        )
        name = entry["path"]
        path = safe_file(availability_file.parent, name)
        require(name not in seen, f"duplicate availability path: {name}")
        seen.add(name)
        require(type(entry["bytes"]) is int and entry["bytes"] >= 0, "invalid availability bytes")
        require(
            isinstance(entry["sha256"], str) and HEX256.fullmatch(entry["sha256"]),
            "invalid availability SHA-256",
        )
        require(path.stat().st_size == entry["bytes"], f"availability byte count mismatch: {name}")
        require(sha256(path) == entry["sha256"], f"availability SHA-256 mismatch: {name}")
    require(seen == set(expected_paths), "source availability inventory mismatch")
    return len(seen)


def check_revision(root, revision, pdf):
    root = Path(root).resolve()
    # Check canonical revision and PDF paths before directory traversal.
    revision_path = PurePosixPath(revision)
    require(
        isinstance(revision, str)
        and revision_path.as_posix() == revision
        and not revision_path.is_absolute()
        and ".." not in revision_path.parts
        and "\\" not in revision
        and revision not in ("", "."),
        "unsafe revision path",
    )
    manifest_path = f"{revision}/PUBLICATION_MANIFEST.json"
    manifest = strict_json(safe_file(root, manifest_path))
    pdf_path = f"{revision}/{pdf}"
    pdf_file = safe_file(root, pdf_path)
    require(pdf_file.parent == root / revision / "thesis", "PDF must be in revision/thesis")
    receipt_path = f"{revision}/thesis/public_quality_receipt.json"
    availability_path = f"{revision}/thesis/public_source_availability.json"
    expected = inventory(root, revision) - {manifest_path}
    require({pdf_path, receipt_path, availability_path} <= expected, "missing publication metadata")
    result = verify_manifest(root, manifest_path, expected)
    reviewed = verify_receipt(root, receipt_path, pdf_path)
    documents = manifest.get("documents")
    require(isinstance(documents, dict), "manifest documents must be an object")
    require(
        type(documents.get("thesis_pages")) is int
        and documents["thesis_pages"] == reviewed["recorded_pages"],
        "manifest and review page counts disagree",
    )
    thesis = f"{revision}/thesis"
    thesis_files = inventory(root, thesis) - {receipt_path, availability_path}
    source_paths = {str(PurePosixPath(path).relative_to(thesis)) for path in thesis_files}
    result["source_files_verified"] = verify_availability(root, availability_path, source_paths)
    return {"status": "PASS", **result, "pdf": pdf_path, "scope": SCOPE}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--revision", required=True, help="Repository-relative public revision directory"
    )
    parser.add_argument("--pdf", required=True, help="Revision-relative final PDF, under thesis/")
    args = parser.parse_args(argv)
    try:
        result = check_revision(args.root, args.revision, args.pdf)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc), "scope": SCOPE}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
