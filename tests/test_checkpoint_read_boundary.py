"""Checkpoint resume never reads symlinks or blocks on special files."""

import json
import os

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config


@pytest.mark.parametrize("dangling", [False, True])
def test_checkpoint_symlink_is_rejected_without_touching_target(tmp_path, dangling):
    config = _config(tmp_path)
    path = grpo._checkpoint_path(config, config.seed)
    target = tmp_path / "synthetic-target.json"
    if not dangling:
        target.write_text(json.dumps({"config": grpo._config_fingerprint(config, config.seed)}))
    path.symlink_to(target)
    before = target.read_bytes() if target.exists() else None
    with pytest.raises(RuntimeError, match="JSON checkpoint must be"):
        grpo._load_checkpoint(config, config.seed)
    assert path.is_symlink()
    assert (target.read_bytes() if target.exists() else None) == before


def test_fifo_checkpoint_is_rejected_without_blocking(tmp_path, monkeypatch):
    config = _config(tmp_path)
    path = grpo._checkpoint_path(config, config.seed)
    os.mkfifo(path)
    opened = []
    real_open = os.open

    def checked_open(target, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK
        assert flags & os.O_NOFOLLOW
        descriptor = real_open(target, flags, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(grpo.os, "open", checked_open)
    with pytest.raises(RuntimeError, match="JSON checkpoint must be a regular file"):
        grpo._load_checkpoint(config, config.seed)
    assert len(opened) == 1
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(opened[0])


def test_directory_checkpoint_is_rejected(tmp_path):
    config = _config(tmp_path)
    path = grpo._checkpoint_path(config, config.seed)
    path.mkdir()
    with pytest.raises(RuntimeError, match="JSON checkpoint must be a regular file"):
        grpo._load_checkpoint(config, config.seed)


def test_normal_receipt_snapshot_remains_compatible(tmp_path):
    config = _config(tmp_path)
    payload = {"config": grpo._config_fingerprint(config, config.seed), "step": 0}
    grpo._write_checkpoint(grpo._checkpoint_path(config, config.seed), payload)
    assert grpo._load_checkpoint(config, config.seed)["step"] == 0


def test_non_utf8_receipt_fails_with_bounded_error(tmp_path):
    config = _config(tmp_path)
    path = grpo._checkpoint_path(config, config.seed)
    raw = b"synthetic-private-marker\xff"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="JSON checkpoint must be valid UTF-8") as failure:
        grpo._load_checkpoint(config, config.seed)
    assert "synthetic-private-marker" not in str(failure.value)
    assert path.read_bytes() == raw
