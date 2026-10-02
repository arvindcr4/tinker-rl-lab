"""Submission packaging preserves inputs and fails closed on incomplete bundles."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submission_builder", ROOT / "platform_modal/scripts/build_university_submission.py"
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_deterministic_package_and_checksums(tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 example")
    (tmp_path / "demo.sh").write_text("#!/bin/sh\nexit 0\n")
    output = tmp_path / "submission.zip"
    files = [Path("report.pdf"), Path("demo.sh")]
    builder.write_bundle(tmp_path, files, output, "example-revision")
    first = output.read_bytes()
    builder.write_bundle(tmp_path, list(reversed(files)), output, "example-revision")
    assert output.read_bytes() == first
    builder.verify_bundle(output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"report.pdf", "demo.sh", "MANIFEST.json", "SHA256SUMS"}
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["base_git_revision"] == "example-revision"
        assert archive.getinfo("demo.sh").external_attr >> 16 == 0o100755


def test_missing_input_does_not_replace_existing_bundle(tmp_path):
    output = tmp_path / "submission.zip"
    output.write_bytes(b"previous valid submission")
    with pytest.raises(FileNotFoundError):
        builder.write_bundle(tmp_path, [Path("missing.pdf")], output, "revision")
    assert output.read_bytes() == b"previous valid submission"


def test_symlink_outside_root_is_rejected(tmp_path):
    (tmp_path / "secret").symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="unsafe"):
        builder.write_bundle(tmp_path, [Path("secret")], tmp_path / "out.zip", "revision")


def test_modified_member_fails_verification(tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"%PDF original")
    output = tmp_path / "original.zip"
    builder.write_bundle(tmp_path, [Path("report.pdf")], output, "revision")
    corrupted = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(corrupted, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"tampered" if name == "report.pdf" else source.read(name))
    with pytest.raises(ValueError, match="checksum"):
        builder.verify_bundle(corrupted)
