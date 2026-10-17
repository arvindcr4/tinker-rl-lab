"""Checkpoint staging must not follow stale links or damage prior receipts."""

import os
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo


def test_stale_predictable_temporary_symlink_is_never_followed(tmp_path):
    victim = tmp_path / "unrelated.txt"
    victim.write_text("preserve this synthetic file")
    path = tmp_path / "receipt.json"
    stale = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    stale.symlink_to(victim)
    grpo._write_checkpoint(path, {"step": 1})
    assert victim.read_text() == "preserve this synthetic file"
    assert stale.is_symlink()
    assert path.is_file()
    assert not path.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [victim.name, stale.name, path.name]
    )


@pytest.mark.parametrize("failure", ["replace", "fsync"])
def test_precommit_io_failure_preserves_receipt_and_cleans_owned_stage(tmp_path, failure):
    path = tmp_path / "receipt.json"
    grpo._write_checkpoint(path, {"step": 1})
    before = path.read_bytes()
    with patch.object(grpo.os, failure, side_effect=OSError("synthetic I/O failure")):
        with pytest.raises(OSError, match="synthetic I/O failure"):
            grpo._write_checkpoint(path, {"step": 2})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_serialization_failure_has_no_filesystem_side_effects(tmp_path):
    path = tmp_path / "new-directory" / "receipt.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        grpo._write_checkpoint(path, {"step": float("nan")})
    assert not path.parent.exists()
