"""Temporary-only preservation regressions; no service credentials or artifacts."""

import hashlib
import os

import pytest
from platform_hybrid.experiments import archive_local_artifacts as archive


def test_owned_copy_and_redaction_preserve_source(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    value = b"synthetic-test-only-value"
    source.write_bytes(value)
    monkeypatch.setenv("HF_TOKEN", value.decode())
    output = tmp_path / "archive"
    output.mkdir()
    target = output / "copy.json"
    archive.hardlink_or_copy(str(source), str(target))
    before = hashlib.sha256(source.read_bytes()).digest()
    assert source.stat().st_ino != target.stat().st_ino
    archive.redact_secrets(output)
    assert hashlib.sha256(source.read_bytes()).digest() == before
    assert target.read_bytes() == b"[REDACTED_SECRET]"
    source.write_bytes(b"later source mutation")
    assert target.read_bytes() == b"[REDACTED_SECRET]"


def test_redacting_existing_hardlink_does_not_mutate_evidence(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_bytes(b"synthetic-test-only-value")
    monkeypatch.setenv("HF_TOKEN", source.read_text())
    output = tmp_path / "archive"
    output.mkdir()
    target = output / "copy.json"
    os.link(source, target)
    archive.redact_secrets(output)
    assert source.read_bytes() == b"synthetic-test-only-value"
    assert target.read_bytes() == b"[REDACTED_SECRET]"
    assert source.stat().st_ino != target.stat().st_ino


def test_overwriting_legacy_hardlink_still_preserves_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("original")
    target = tmp_path / "target.json"
    os.link(source, target)
    archive.hardlink_or_copy(str(source), str(target))
    target.write_text("changed archive")
    assert source.read_text() == "original"


@pytest.mark.parametrize("nested", [False, True])
def test_archive_refuses_overlapping_paths(tmp_path, nested):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="overlap"):
        archive.copy_tree_hardlinked(source, source / "archive" if nested else source)
