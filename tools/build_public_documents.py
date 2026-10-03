#!/usr/bin/env python3
"""Rebuild frozen PUBLIC documents in isolation; never replace reviewed artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

# Support both `python tools/build_public_documents.py` and module/test imports.
if __package__:
    from . import check_public_release as release
    from . import prepare_public_tex_bundle as tex_bundle
else:
    import check_public_release as release
    import prepare_public_tex_bundle as tex_bundle


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {
    "thesis": {
        "builder": "build_thesis.py",
        "pdf": "Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-03_PUBLIC.pdf",
        "generated": {"thesis_master.tex"},
    },
    "addendum": {
        "builder": "build_public_addendum.py",
        "pdf": "C1_Research_Addendum_2026-10-03_PUBLIC.pdf",
        "generated": {"C1_Research_Addendum_2026-10-03_PUBLIC.tex"},
    },
}
INPUT_SUFFIXES = {".md", ".tex", ".bib", ".py", ".json", ".csv", ".png", ".jpg", ".jpeg"}


class BuildError(ValueError):
    """An expected build failure that should be actionable without a traceback."""


def require(condition, message):
    if not condition:
        raise BuildError(message)


def destination_path(root, destination):
    """Require a fresh leaf under an existing real parent, outside the checkout."""
    require(".." not in Path(destination).parts, "use a canonical output path without '..'")
    destination = Path(os.path.abspath(destination))
    require(not os.path.lexists(destination), f"output directory already exists: {destination}")
    require(
        destination.parent.is_dir(), "output parent must already exist; choose an existing parent"
    )
    # Resolve aliases before checking the checkout boundary. mkdir below is the
    # exclusive reservation; this early check is not the no-clobber guarantee.
    destination = destination.parent.resolve() / destination.name
    require(
        not destination.is_relative_to(root), "output directory must be outside the source checkout"
    )
    return destination


def verified_manifest(root):
    path = release.safe_file(root, release.MANIFEST)
    digest = release.sha256(path)
    release.verify_manifest(root, expected_paths=release.snapshot_paths(root))
    manifest = release.strict_json(path)
    require(
        release.sha256(path) == digest, "publication manifest changed during verification; retry"
    )
    return manifest, digest


def stage_inputs(root, stage, manifest, document):
    """Copy only declared input bytes, never hardlink them to frozen sources."""
    prefix = f"{release.REVISION}/{document}/"
    spec = DOCUMENTS[document]
    copied = []
    for entry in manifest["files"]:
        name = entry["path"]
        if not name.startswith(prefix):
            continue
        local = PurePosixPath(name[len(prefix) :])
        if any(part in {"build", "__pycache__"} or part.startswith(".") for part in local.parts):
            continue
        if local.as_posix() in spec["generated"]:
            continue
        # All root-level final PDFs are outputs. Only declared figure PDFs are
        # inputs; the reviewed figure binaries intentionally remain unchanged.
        if local.suffix.lower() == ".pdf":
            if len(local.parts) < 2 or local.parts[0] != "figures":
                continue
        elif local.suffix.lower() not in INPUT_SUFFIXES:
            continue
        source = release.safe_file(root, name)
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        require(
            target.stat().st_size == entry["bytes"] and release.sha256(target) == entry["sha256"],
            f"input changed while staging: {name}; retry from a stable checkout",
        )
        copied.append(name)
    directory = stage / release.REVISION / document
    require((directory / spec["builder"]).is_file(), f"manifest has no {document} builder")
    require(not (directory / spec["pdf"]).exists(), "final PDF unexpectedly present before build")
    require(not (directory / "build").exists(), "scratch output unexpectedly present before build")
    return directory, copied


def stage_bundle(stage, supplied):
    """Bind one explicit local ZIP to immutable build-stage bytes; never fetch it."""
    source = tex_bundle.local_path(supplied)
    require(source.is_file(), f"missing local Tectonic bundle: {source}")
    # Validate before copying, including bounded ZIP size, member types and hashes.
    details = tex_bundle.validate_bundle(source)
    digest = release.sha256(source)
    size = source.stat().st_size
    target = stage / "local-tex-bundle.zip"
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    require(
        target.stat().st_size == size and release.sha256(target) == digest,
        "local Tectonic bundle changed while staging; nothing published",
    )
    require(tex_bundle.validate_bundle(target) == details, "local bundle inventory changed")
    record = {
        "source_name": source.name,
        "staged_filename": target.name,
        "bytes": size,
        "sha256": digest,
        "option": "--bundle",
        **details,
    }
    return source, target, record


def verify_bundle_unchanged(source, staged, record):
    for path in (source, staged):
        path = tex_bundle.local_path(path)
        require(
            path.is_file()
            and path.stat().st_size == record["bytes"]
            and release.sha256(path) == record["sha256"],
            "local Tectonic bundle changed during build; nothing published",
        )


def build_environment(stage, tectonic_bundle=None):
    require(os.name == "posix", "this wrapper requires POSIX process groups (Linux or macOS)")
    native = {name: shutil.which(name) for name in ("pandoc", "tectonic")}
    missing = [name for name, path in native.items() if path is None]
    require(
        not missing, f"missing native tool(s): {', '.join(missing)}; add approved tools to PATH"
    )
    native = {name: os.path.abspath(path) for name, path in native.items()}
    native_path = os.pathsep.join(
        os.path.abspath(entry or os.curdir)
        for entry in os.environ.get("PATH", os.defpath).split(os.pathsep)
    )
    # Frozen builders have no offline switch. Interpose on their Tectonic call
    # without changing builder bytes or accepting new CLI commands from callers.
    shim_dir = stage / "native-tools"
    shim_dir.mkdir()
    # A Python shebang cannot safely embed a venv path containing spaces. A
    # fixed POSIX shell interpreter plus quoted absolute commands also prevents
    # a relative PATH entry from changing meaning after the builder chdir.
    for name, executable in native.items():
        shim = shim_dir / name
        flags = " --only-cached --untrusted" if name == "tectonic" else ""
        if name == "tectonic" and tectonic_bundle is not None:
            flags += " --bundle " + shlex.quote(str(tectonic_bundle))
        shim.write_text(
            f"#!/bin/sh\nPATH={shlex.quote(native_path)}\nexport PATH\n"
            f'exec {shlex.quote(executable)} "$@"{flags}\n',
            encoding="utf-8",
        )
        shim.chmod(0o700)
    env = os.environ.copy()
    # No inherited PATH fallback: execvp can skip an EACCES/noexec shim and
    # otherwise reach the unrestricted executable. These frozen builders need
    # only the two explicit native tools. Each shim restores the normalized
    # original PATH only for its already-selected native executable/helpers.
    env["PATH"] = str(shim_dir)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_builder(directory, document, env, timeout, log):
    command = [sys.executable, "-B", str(directory / DOCUMENTS[document]["builder"])]
    # File-backed logging avoids unbounded captured output in wrapper memory.
    with log.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=directory,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Terminate the builder AND its native-tool descendants before the
            # fresh staging directory is cleaned up.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise BuildError(
                f"{document} build interrupted or timed out after {timeout:g}s"
            ) from None
    if returncode:
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4000))
            tail = stream.read().decode("utf-8", errors="replace")
        raise BuildError(
            f"{document} builder failed (exit {returncode}); no documents published. "
            "Missing cached resources require a separately prepared cache.\n" + tail
        )


def plausible_pdf(stage, directory, document):
    path = directory / DOCUMENTS[document]["pdf"]
    require(path.exists(), f"{document} builder produced no fresh PDF; no documents published")
    release.safe_file(stage, path.relative_to(stage).as_posix())
    require(path.stat().st_size >= 32, f"{document} builder produced an empty or truncated PDF")
    with path.open("rb") as stream:
        header = stream.read(16)
        stream.seek(max(0, path.stat().st_size - 1024))
        tail = stream.read()
    require(
        re.match(rb"%PDF-(?:1\.[0-9]|2\.0)(?:\r|\n)", header) and tail.rstrip().endswith(b"%%EOF"),
        f"{document} output lacks a plausible PDF header/trailer",
    )
    return path


def publish(destination, outputs, receipt):
    """Reserve once; atomically link complete files with no-replace semantics.

    A failure after reservation leaves an incomplete NEW directory for inspection.
    Never delete it: another process may have written files there in the meantime.
    The receipt is the final completion marker, not a transactional directory rename.
    """
    with tempfile.TemporaryDirectory(prefix=".public-publish-", dir=destination.parent) as scratch:
        prepared = Path(scratch)
        for source in outputs:
            target = prepared / source.name
            shutil.copyfile(source, target)
            require(release.sha256(source) == release.sha256(target), "publication copy mismatch")
        receipt_path = prepared / "BUILD_RECEIPT.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            destination.mkdir(exist_ok=False)
        except FileExistsError:
            raise BuildError(
                f"output directory already exists: {destination}; nothing replaced"
            ) from None
        try:
            for source in [*(prepared / path.name for path in outputs), receipt_path]:
                # Both paths are on the destination filesystem. link never
                # replaces an existing file, unlike rename/replace/copy2.
                os.link(source, destination / source.name)
        except OSError as exc:
            raise BuildError(
                f"publication incomplete at {destination}: {exc}; inspect it and choose a NEW output "
                "directory to retry. No existing files were replaced or removed"
            ) from exc


def build_documents(root, destination, documents, timeout=1800, tectonic_bundle=None):
    root = Path(root).resolve()
    require(
        math.isfinite(timeout) and 0 < timeout <= 3600, "timeout must be > 0 and <= 3600 seconds"
    )
    require(documents and len(set(documents)) == len(documents), "choose distinct documents")
    require(all(name in DOCUMENTS for name in documents), "unknown document selection")
    destination = destination_path(root, destination)
    manifest, digest = verified_manifest(root)
    with tempfile.TemporaryDirectory(
        prefix=".public-documents-", dir=destination.parent
    ) as scratch:
        stage = Path(scratch)
        require(
            not stage.is_relative_to(root),
            "temporary directory must be outside the source checkout",
        )
        bundle_record = None
        if tectonic_bundle is not None:
            bundle_source, bundle_stage, bundle_record = stage_bundle(stage, tectonic_bundle)
            env = build_environment(stage, bundle_stage)
        else:
            env = build_environment(stage)
        outputs = []
        records = {}
        for document in documents:
            directory, copied = stage_inputs(root, stage, manifest, document)
            log = stage / f"{document}_build.log"
            run_builder(directory, document, env, timeout, log)
            pdf = plausible_pdf(stage, directory, document)
            records[document] = {
                "pdf": pdf.name,
                "bytes": pdf.stat().st_size,
                "sha256": release.sha256(pdf),
                "staged_inputs": copied,
            }
            outputs.extend([pdf, log])
        # Detect concurrent edits to the manifest or frozen inputs before release.
        _, final_digest = verified_manifest(root)
        require(
            final_digest == digest, "publication manifest changed during build; nothing published"
        )
        if bundle_record is not None:
            verify_bundle_unchanged(bundle_source, bundle_stage, bundle_record)
        receipt = {
            "schema": "public-document-build-v1",
            "status": "BUILT",
            "publication_manifest_sha256": digest,
            "documents": records,
            "tectonic_flags": ["--only-cached", "--untrusted"],
            "network_isolation": False,
            "scope": (
                "Fresh build from manifest-verified PUBLIC inputs and existing figure PDFs. "
                "PDF signature/trailer checked; not a layout review, byte-identical reproduction, "
                "scientific validation, privacy certification or replay of omitted executions."
            ),
        }
        if bundle_record is not None:
            receipt["tectonic_bundle"] = bundle_record
        publish(destination, outputs, receipt)
    return {"status": "BUILT", "output_directory": str(destination), "documents": records}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--tectonic-bundle",
        help="Optional existing local TeX ZIP from prepare_public_tex_bundle.py; no URLs",
    )
    cli.add_argument(
        "--root", type=Path, default=ROOT, help="Frozen source checkout (default: this checkout)"
    )
    cli.add_argument("--document", choices=[*DOCUMENTS, "all"], required=True)
    cli.add_argument(
        "--output-dir", type=Path, required=True, help="New directory outside the source checkout"
    )
    cli.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Seconds per builder, >0 to 3600 (default: 1800)",
    )
    args = cli.parse_args(argv)
    documents = list(DOCUMENTS) if args.document == "all" else [args.document]
    try:
        result = build_documents(
            args.root,
            args.output_dir,
            documents,
            args.timeout,
            tectonic_bundle=args.tectonic_bundle,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
