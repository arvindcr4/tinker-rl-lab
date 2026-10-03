#!/usr/bin/env python3
"""Prepare a deterministic, local-only Tectonic ZIP from explicit existing resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SUFFIXES = {".tex", ".sty", ".def", ".cfg"}
RESERVED = {"FILELIST", "SHA256SUM"}
MAX_FILES = 10_000
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+@-]*\Z")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def local_path(value):
    """Accept filesystem paths only, rejecting URI schemes and all symlink components."""
    raw = os.fspath(value)
    require(raw and not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw), "use a local path, not a URL")
    require(".." not in Path(raw).parts, "use a canonical local path without '..'")
    path = Path(os.path.abspath(raw))
    for component in [*reversed(path.parents), path]:
        require(not component.is_symlink(), f"symlink not allowed: {component}")
    return path


def safe_name(name):
    require(isinstance(name, str) and NAME.fullmatch(name), f"unsafe resource name: {name!r}")
    require(name not in RESERVED, f"reserved resource name: {name}")
    return name


def read_resource(path):
    path = local_path(path)
    require(stat.S_ISREG(path.stat().st_mode), f"not a regular resource file: {path}")
    require(path.stat().st_size <= MAX_FILE_BYTES, f"resource exceeds size limit: {path}")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    require(len(data) <= MAX_FILE_BYTES, f"resource exceeds size limit: {path}")
    return data


def validate_bundle(path):
    """Validate our flat ZIP inventory, CRCs and content hashes without extracting it."""
    path = local_path(path)
    require(stat.S_ISREG(path.stat().st_mode), f"not a regular bundle file: {path}")
    require(path.stat().st_size <= MAX_TOTAL_BYTES + 4 * 1024 * 1024, "bundle exceeds size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [entry.filename for entry in members]
            require(len(names) == len(set(names)), "duplicate ZIP member")
            require(2 < len(names) <= MAX_FILES + 2, "bundle resource count outside limits")
            require(RESERVED <= set(names), "bundle needs FILELIST and SHA256SUM")
            require(
                sum(entry.file_size for entry in members) <= MAX_TOTAL_BYTES,
                "bundle exceeds expanded size limit",
            )
            for entry in members:
                require(
                    entry.orig_filename == entry.filename, "unsafe noncanonical ZIP member name"
                )
                if entry.filename not in RESERVED:
                    safe_name(entry.filename)
                require(not entry.is_dir(), "directories not allowed in flat bundle")
                mode = entry.external_attr >> 16
                require(stat.S_IFMT(mode) in (0, stat.S_IFREG), "nonregular ZIP member")
                require(not entry.flag_bits & 1, "encrypted ZIP member")
                require(
                    entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "unsupported ZIP compression",
                )
                require(entry.file_size <= MAX_FILE_BYTES, "ZIP member exceeds size limit")
            filelist = archive.read("FILELIST")
            identity = sha256(filelist)
            require(
                archive.read("SHA256SUM") == (identity + "\n").encode("ascii"),
                "bundle identity mismatch",
            )
            listed = {}
            for line in filelist.decode("ascii").splitlines():
                digest, separator, name = line.partition("  ")
                require(separator and HEX.fullmatch(digest), "invalid FILELIST entry")
                safe_name(name)
                require(name not in listed, "duplicate FILELIST entry")
                listed[name] = digest
            require(set(names) == set(listed) | RESERVED, "bundle inventory mismatch")
            expected = "".join(
                f"{digest}  {name}\n" for name, digest in sorted(listed.items())
            ).encode("ascii")
            require(filelist == expected, "noncanonical FILELIST")
            for name, digest in listed.items():
                require(
                    sha256(archive.read(name)) == digest, f"bundle resource digest mismatch: {name}"
                )
    except (zipfile.BadZipFile, zlib.error, UnicodeError, RuntimeError, EOFError) as exc:
        raise ValueError(f"invalid local TeX bundle: {exc}") from exc
    return {"bundle_identity_sha256": identity, "resource_files": len(listed)}


def labelled(value):
    label, separator, content = value.partition("=")
    require(separator and NAME.fullmatch(label) and content, "expected LABEL=VALUE")
    return label, content


def prepare_bundle(cache_dir, resource_roots, destination, license_evidence, cache_origin=None):
    """Flatten explicit roots; only deliberate cache-to-overlay replacement is allowed."""
    cache = local_path(cache_dir)
    require(cache.is_dir(), f"missing cache directory: {cache}")
    roots = [(label, local_path(path)) for label, path in resource_roots]
    require(
        all(label != "cache" and NAME.fullmatch(label) for label, _ in roots),
        "invalid resource label",
    )
    require(len({path for _, path in roots}) == len(roots), "duplicate resource root")
    require(all(path.is_dir() for _, path in roots), "missing resource root")
    labels = {"cache", *(label for label, _ in roots)}
    require(
        set(license_evidence) == labels,
        "supply exactly one license evidence note for each label, including cache",
    )
    require(
        all(isinstance(value, str) and value.strip() for value in license_evidence.values()),
        "empty license evidence",
    )
    require(".." not in Path(destination).parts, "use a canonical output path without '..'")
    destination = Path(os.path.abspath(destination))
    require(not os.path.lexists(destination), f"output directory already exists: {destination}")
    require(destination.parent.is_dir(), "output parent must already exist")
    destination = destination.parent.resolve() / destination.name
    require(not destination.is_relative_to(ROOT), "output directory must be outside this checkout")
    require(
        all(not destination.is_relative_to(path) for path in [cache, *(p for _, p in roots)]),
        "output directory must be outside resource inputs",
    )
    entries = {}
    overlays = {}
    total = 0

    def record(path, label, root):
        nonlocal total
        name = safe_name(path.name)
        data = read_resource(path)
        total += len(data)
        require(total <= MAX_TOTAL_BYTES - 4 * 1024 * 1024, "selected resources exceed size limit")
        info = {
            "name": name,
            "source": str(path),
            "source_root": str(root),
            "label": label,
            "bytes": len(data),
            "sha256": sha256(data),
            "license_evidence": license_evidence[label],
        }
        return name, (info, data)

    for path in sorted(cache.iterdir()):
        name, item = record(path, "cache", cache)
        entries[name] = item
        require(len(entries) <= MAX_FILES, "too many cached resources")
    require(entries, "cache directory contains no resources")
    for label, root in roots:
        count = 0

        # os.walk does not follow links; inspect every encountered entry anyway,
        # including excluded file types, so no silently skipped symlink is accepted.
        def walk_error(error):
            raise error

        for directory, subdirs, filenames in os.walk(root, followlinks=False, onerror=walk_error):
            for name in sorted(subdirs + filenames):
                path = Path(directory) / name
                require(not path.is_symlink(), f"symlink not allowed: {path}")
                mode = path.stat().st_mode
                require(
                    stat.S_ISDIR(mode) or stat.S_ISREG(mode), f"nonregular resource path: {path}"
                )
                if stat.S_ISREG(mode) and path.suffix in RUNTIME_SUFFIXES:
                    name, item = record(path, label, root)
                    require(name not in overlays, f"duplicate overlay basename: {name}")
                    overlays[name] = item
                    count += 1
                    require(len(set(entries) | set(overlays)) <= MAX_FILES, "too many resources")
        require(count, f"resource root contains no selected runtime files: {root}")
    replacements = []
    for name, item in sorted(overlays.items()):
        if name in entries:
            replacements.append(
                {"name": name, "previous": entries[name][0], "replacement": item[0]}
            )
        entries[name] = item
    records = [item[0] for _, item in sorted(entries.items())]
    filelist = "".join(f"{entry['sha256']}  {entry['name']}\n" for entry in records).encode("ascii")
    identity = sha256(filelist)
    contents = {name: data for name, (_, data) in entries.items()}
    contents.update({"FILELIST": filelist, "SHA256SUM": (identity + "\n").encode("ascii")})
    with tempfile.TemporaryDirectory(
        prefix=".public-tex-bundle-", dir=destination.parent
    ) as scratch:
        prepared = Path(scratch)
        bundle = prepared / "public-tex-bundle.zip"
        # Stored entries avoid zlib-version variation. Metadata, order and bytes
        # are fixed; no timestamps or randomly selected identifiers enter the ZIP.
        with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(contents.items()):
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
        validate_bundle(bundle)
        # Detect ordinary concurrent input changes before publishing either file.
        for record_info in [*records, *(item["previous"] for item in replacements)]:
            require(
                sha256(read_resource(record_info["source"])) == record_info["sha256"],
                f"resource changed during preparation: {record_info['source']}",
            )
        provenance = {
            "schema": "public-local-tex-bundle-v1",
            "bundle": bundle.name,
            "zip_sha256": sha256(bundle.read_bytes()),
            "zip_bytes": bundle.stat().st_size,
            "bundle_identity_sha256": identity,
            "resource_files": len(records),
            "cache_directory": str(cache),
            "cache_origin": cache_origin,
            "resource_roots": [{"label": label, "path": str(path)} for label, path in roots],
            "runtime_suffixes": sorted(RUNTIME_SUFFIXES),
            "zip_compression": "stored",
            "license_evidence": license_evidence,
            "overlays": replacements,
            "files": records,
            "scope": "Local toolchain inputs only; not a frozen scientific manifest or license audit. "
            "License notes are caller-supplied evidence, not permission to redistribute. "
            "No resources were downloaded or installed; nothing is uploaded automatically.",
        }
        receipt = prepared / "PROVENANCE.json"
        receipt.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        destination.mkdir(exist_ok=False)
        # No-replace links; PROVENANCE.json is the last completion marker. Keep
        # an incomplete new directory for inspection if publication fails.
        for source in (bundle, receipt):
            os.link(source, destination / source.name)
    return {
        "status": "PREPARED",
        "output_directory": str(destination),
        "zip_sha256": provenance["zip_sha256"],
        "resource_files": len(records),
    }


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache-dir", required=True, help="Existing flat Tectonic bundle data cache")
    cli.add_argument(
        "--resource-root",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Existing recursive overlay root; selects .tex/.sty/.def/.cfg (repeatable)",
    )
    cli.add_argument(
        "--license-evidence",
        action="append",
        required=True,
        metavar="LABEL=NOTE",
        help="Caller-reviewed license evidence for cache and each resource label",
    )
    cli.add_argument("--cache-origin", help="Optional provenance text/URL; never fetched")
    cli.add_argument(
        "--output-dir", required=True, help="New external directory under an existing parent"
    )
    args = cli.parse_args(argv)
    try:
        roots = [labelled(value) for value in args.resource_root]
        notes = [labelled(value) for value in args.license_evidence]
        require(
            len({label for label, _ in notes}) == len(notes), "duplicate license evidence label"
        )
        result = prepare_bundle(
            args.cache_dir, roots, args.output_dir, dict(notes), args.cache_origin
        )
    except (OSError, ValueError, TypeError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
