#!/usr/bin/env python3
"""Package the October 4 reviewed public release; never rebuild or submit it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

if __package__:
    from ._strict import require
    from .check_public_revision import check_revision
    from .check_public_release import ROOT, safe_file, strict_json
else:
    from _strict import require
    from check_public_revision import check_revision
    from check_public_release import ROOT, safe_file, strict_json

REVISION = "reports/public_revision_2026-10-04"
THESIS = f"{REVISION}/thesis"
PUBLICATION = f"{REVISION}/PUBLICATION_MANIFEST.json"
PDF = f"{THESIS}/Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-04_PUBLIC.pdf"
RECEIPT = f"{THESIS}/public_quality_receipt.json"
AVAILABILITY = f"{THESIS}/public_source_availability.json"
PACKAGE = "PUBLIC_REVIEW_PACKAGE_MANIFEST.json"
REQUIRED = {PDF, RECEIPT, AVAILABILITY} | {
    f"{THESIS}/{name}"
    for name in (
        "README.md",
        "build_thesis.py",
        "compile_figures.py",
        "frontmatter.tex",
        "preamble.tex",
        "references.bib",
        "ch01_introduction.md",
        "ch02_literature.md",
        "ch03_requirements.md",
        "ch04_methodology.md",
        "ch05_implementation.md",
        "ch06_results_core.md",
        "ch07_results_infra.md",
        "ch08_results_fraud.md",
        "ch09_results_campaign.md",
        "ch10_synthesis_conclusions.md",
        "ch_back_run_registry.md",
        "ch_back_reproducibility.md",
        "ch_back_notation.md",
        "ch_back_campaign_detail.md",
        "ch_back_errata.md",
        "ch_back_c1_evidence.md",
        "ch_back_coverage_audit.md",
        "ch_back_provenance.md",
        "c1_case_ledger.json",
        "c1_reference_bindings.json",
        "public_p11_manifest_bindings.json",
        "provenance_audit.json",
    )
}
MAX_BYTES = 256 * 1024 * 1024
MAX_FILES = 2000
SCOPE = (
    "Public review package with exact file hashes and a recorded PDF review binding. "
    "Not institutional certification, signatures, submission, a privacy certification, "
    "or reproduction/authentication of omitted experiments. No files are uploaded."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def public_name(name):
    require(isinstance(name, str) and name, "empty package path")
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and "\\" not in name
        and "\x00" not in name,
        "unsafe package path",
    )
    require(name.startswith(REVISION + "/"), f"outside public package roots: {name}")
    require(
        not any(part in {"build", "__pycache__"} or part.startswith(".") for part in path.parts)
        or name == ".gitignore",
        "scratch or hidden file",
    )
    require(path.name != "thesis_master.tex", "generated master is not an editable source")
    return name


def selected_files(root):
    publication = strict_json(safe_file(root, PUBLICATION))
    check_revision(root, REVISION, PDF.removeprefix(REVISION + "/"))
    names = {public_name(entry["path"]) for entry in publication["files"]}
    require(REQUIRED <= names, f"missing required public review inputs: {sorted(REQUIRED - names)}")
    # Source availability is a release-authored disclosure, not a certification.
    require(
        isinstance(strict_json(safe_file(root, AVAILABILITY)), dict),
        "source availability must be a JSON object",
    )
    return names | {PUBLICATION}


def verify_package(path):
    """Verify bounded, canonical ZIP contents and all release/receipt bindings."""
    path = Path(path)
    require(path.stat().st_size <= MAX_BYTES, "package exceeds size limit")
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as temporary:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        require(len(names) == len(set(names)), "duplicate ZIP member")
        require(1 < len(names) <= MAX_FILES and PACKAGE in names, "invalid package inventory")
        require(
            sum(entry.file_size for entry in members) <= MAX_BYTES, "expanded package too large"
        )
        root = Path(temporary)
        for entry in members:
            require(entry.orig_filename == entry.filename, "noncanonical ZIP member")
            if entry.filename != PACKAGE:
                public_name(entry.filename)
            require(
                not entry.is_dir() and stat.S_IFMT(entry.external_attr >> 16) in (0, stat.S_IFREG),
                "nonregular ZIP member",
            )
            require(not entry.flag_bits & 1, "encrypted ZIP member")
            require(
                entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "unsupported ZIP compression",
            )
            dest = root / entry.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(entry))
        record = strict_json(root / PACKAGE)
        require(
            isinstance(record, dict) and record.get("schema") == "public-review-package-v1",
            "unknown package schema",
        )
        require(record.get("scope") == SCOPE, "missing package limits")
        expected = selected_files(root)
        require(set(names) == expected | {PACKAGE}, "package inventory differs from publication")
        entries = record.get("files")
        require(isinstance(entries, list), "package files must be a list")
        seen = set()
        for item in entries:
            require(
                isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
                "invalid package manifest entry",
            )
            name = item["path"]
            require(
                isinstance(name, str) and name in expected and name not in seen,
                "invalid or duplicate package entry",
            )
            seen.add(name)
            data = (root / name).read_bytes()
            require(
                type(item["bytes"]) is int
                and item["bytes"] == len(data)
                and item["sha256"] == digest(data),
                f"package hash mismatch: {name}",
            )
        require(seen == expected, "incomplete package manifest")
        require(
            record.get("publication_manifest_sha256") == digest((root / PUBLICATION).read_bytes()),
            "publication manifest binding mismatch",
        )
    return {
        "status": "VERIFIED",
        "files": len(expected),
        "sha256": digest(path.read_bytes()),
        "scope": SCOPE,
    }


def build_package(root, output):
    root, output = Path(root), Path(output)
    require(not os.path.lexists(output), "output already exists; choose a new archive")
    names = selected_files(root)
    require(len(names) < MAX_FILES, "too many package files")
    total = sum(safe_file(root, name).stat().st_size for name in names)
    require(total < MAX_BYTES - 1024 * 1024, "package inputs exceed size limit")
    contents = {name: safe_file(root, name).read_bytes() for name in sorted(names)}
    record = {
        "schema": "public-review-package-v1",
        "scope": SCOPE,
        "publication_manifest_sha256": digest(contents[PUBLICATION]),
        "files": [
            {"path": name, "bytes": len(data), "sha256": digest(data)}
            for name, data in contents.items()
        ],
    }
    contents[PACKAGE] = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    require(output.parent.is_dir(), "output parent must exist")
    with tempfile.TemporaryDirectory(prefix=".public-review-", dir=output.parent) as temporary:
        prepared = Path(temporary) / "review.zip"
        with zipfile.ZipFile(prepared, "x", compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(contents.items()):
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
        result = verify_package(prepared)
        for name in names:
            require(
                safe_file(root, name).read_bytes() == contents[name],
                f"input changed while packaging: {name}",
            )
        os.link(prepared, output)  # Atomic no-replace publication, never replace an old ZIP.
    return {**result, "status": "PACKAGED", "archive": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="New ZIP path; refuses replacement")
    mode.add_argument(
        "--verify",
        type=Path,
        help="Existing ZIP to verify without extraction to disk outside temporary storage",
    )
    args = parser.parse_args(argv)
    try:
        result = (
            verify_package(args.verify) if args.verify else build_package(args.root, args.output)
        )
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
