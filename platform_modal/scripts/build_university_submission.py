#!/usr/bin/env python3
"""Build or verify the current, non-anonymous Phase 2 thesis review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THESIS = Path("outputs/PES_Phase2_Third_Review_2026-09-24/thesis")
DEFAULT_OUTPUT = ROOT / "dist/TinkerRL_Phase2_Submission.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def submission_files(root: Path) -> list[Path]:
    # Explicit selections exclude old abstracts/decks, secrets, caches, raw
    # benchmark answer keys, model weights, and unrelated historical releases.
    import runpy

    checker = runpy.run_path(str(root / "tools/check_thesis_evidence.py"))
    builder = runpy.run_path(str(root / THESIS / "build_thesis.py"))
    files = {
        Path(name)
        for name in (
            "SUBMISSION.md",
            "BASELINES.md",
            "REPRODUCE.md",
            "LICENSE",
            "CITATION.cff",
            "PROJECT_HISTORY.md",
            "pyproject.toml",
            "uv.lock",
            "requirements.txt",
            "platform_hybrid/sem 4 work/PROVENANCE.md",
            "outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/followup_fixes_2026-10-02.md",
            "outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/submission_preparation.md",
            "outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/submission_validation.md",
            "outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/selected_evidence_check.json",
            "outputs/e1_e14_small_scale_2026-09-26/E3/code/paired_finalize.py",
            "tools/check_thesis_evidence.py",
            "platform_modal/scripts/build_university_submission.py",
            "submission/demo/run_demo.py",
            "submission/demo/demo.sh",
            "submission/demo/README.md",
            "submission/demo/DEFENSE_RUNBOOK.md",
            "submission/demo/fixtures/offline_demo.json",
            "submission/demo/tests/test_demo.py",
            "platform_hybrid/experiments/results/tinker_direct_eval.json",
            "tests/test_thesis_evidence.py",
            "tests/test_thesis_build.py",
            "tests/test_thesis_submission.py",
        )
    }
    files.update(Path(name) for name in checker["EVIDENCE_FILES"])
    files.update(THESIS / name for name, _ in builder["CHAPTERS"] + builder["APPENDICES"])
    files.update(
        THESIS / name
        for name in (
            "README.md",
            "build_thesis.py",
            "compile_figures.py",
            "preamble.tex",
            "frontmatter.tex",
            "references.bib",
            "thesis_master.tex",
            "Thesis_Report_ArvindCR.pdf",
            "figures/pes_logo.png",
            "figures/signature_arvind.png",
        )
    )
    for entries in builder["FIGURES"].values():
        for name, *_ in entries:
            files.update(THESIS / "figures" / (name + suffix) for suffix in (".tex", ".pdf"))
    return sorted(files)


def write_bundle(root: Path, files: list[Path], output: Path, revision: str) -> None:
    """Validate all inputs first; replace the archive only after complete success."""
    contents = {}
    for relative in files:
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            raise ValueError(f"unsafe package path: {relative}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"package path escapes root: {relative}")
        if relative.as_posix() in {"MANIFEST.json", "SHA256SUMS"}:
            raise ValueError(f"reserved package path: {relative}")
        data = path.read_bytes()  # Missing/unreadable inputs must not be skipped.
        if not data:
            raise ValueError(f"empty package input: {relative}")
        contents[relative.as_posix()] = data
    manifest = {
        "schema": "tinkerrl-phase2-submission-v1",
        "base_git_revision": revision,
        "source": "Current working-tree bytes; may include changes after base_git_revision.",
        "scope": "Non-anonymous thesis and selected evidence, not the full research repository.",
        "files": {
            name: {"sha256": digest(data), "bytes": len(data)}
            for name, data in sorted(contents.items())
        },
    }
    contents["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    contents["SHA256SUMS"] = "".join(
        f"{digest(data)}  {name}\n" for name, data in sorted(contents.items())
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        temporary = Path(tmp) / "submission.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
                archive.writestr(info, data)
        verify_bundle(temporary)
        os.replace(temporary, output)
    output.with_suffix(".zip.sha256").write_text(
        f"{digest(output.read_bytes())}  {output.name}\n", encoding="utf-8"
    )


def verify_bundle(path: Path) -> None:
    """Verify every archive member without extracting or executing its contents."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in names):
            raise ValueError("unsafe archive member")
        manifest = json.loads(archive.read("MANIFEST.json"))
        if manifest["schema"] != "tinkerrl-phase2-submission-v1":
            raise ValueError("unsupported submission manifest")
        if set(names) != set(manifest["files"]) | {"MANIFEST.json", "SHA256SUMS"}:
            raise ValueError("archive membership does not match manifest")
        for name, expected in manifest["files"].items():
            data = archive.read(name)
            if {"sha256": digest(data), "bytes": len(data)} != expected:
                raise ValueError(f"checksum or size mismatch: {name}")
        expected_sums = "".join(
            f"{digest(archive.read(name))}  {name}\n"
            for name in sorted(set(names) - {"SHA256SUMS"})
        )
        if archive.read("SHA256SUMS").decode() != expected_sums:
            raise ValueError("SHA256SUMS mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", type=Path, help="verify an existing ZIP without rebuilding")
    args = parser.parse_args()
    try:
        if args.verify:
            verify_bundle(args.verify)
            print(f"Package integrity: PASS ({args.verify})")
            return 0
        # No provider calls or training. Tectonic may fetch missing TeX packages.
        commands = [
            [
                sys.executable,
                "tools/check_thesis_evidence.py",
                "--json",
                "outputs/PES_Phase2_Third_Review_2026-09-24/thesis_review_2026-09-27/selected_evidence_check.json",
            ],
            [sys.executable, "submission/demo/run_demo.py", "--self-test"],
            [sys.executable, str(THESIS / "compile_figures.py"), "--force"],
            [sys.executable, str(THESIS / "build_thesis.py")],
        ]
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        write_bundle(ROOT, submission_files(ROOT), args.output.resolve(), revision)
        print(f"Prepared: {args.output.resolve()}")
        print("Human approval/signatures and portal checks remain; see SUBMISSION.md.")
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Submission build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
