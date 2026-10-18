"""Immutable critic artifacts and receipt-bound, snapshot-verified restores."""

import hashlib
import io
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from platform_tinker.tinkerrl import grpo


@pytest.fixture
def saved(tmp_path):
    config = grpo.GRPOConfig(name="bound", checkpoint_dir=str(tmp_path), critic_hidden_dim=4)
    critic = grpo.PromptValueCritic(hidden_dim=4)
    optimizer = torch.optim.Adam(critic.parameters())
    digest = grpo._save_critic(config, config.seed, critic, optimizer, step=1)
    path = grpo._critic_checkpoint_path(config, config.seed, digest)
    return config, critic, optimizer, digest, path


def test_content_addressed_state_is_bound_to_bytes_step_and_config(saved):
    config, critic, optimizer, digest, path = saved
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    payload = grpo._load_critic_checkpoint(config, config.seed, 1, {"critic_state_sha256": digest})
    assert payload["step"] == 1
    assert payload["config_sha256"] == grpo._critic_config_hash(config, config.seed)
    for name, tensor in critic.state_dict().items():
        torch.testing.assert_close(payload["model"][name], tensor)
    assert grpo._save_critic(config, config.seed, critic, optimizer, step=1) == digest
    assert len(list(path.parent.glob("critic-*.pt"))) == 1
    assert not grpo._critic_checkpoint_path(config, config.seed).exists()


def test_corruption_is_rejected_before_deserialization_and_preserved(saved):
    config, _, _, digest, path = saved
    corrupt = path.read_bytes()[::-1]
    path.write_bytes(corrupt)
    with (
        patch.object(torch, "load") as deserialize,
        pytest.raises(RuntimeError, match="SHA-256 mismatch"),
    ):
        grpo._load_critic_checkpoint(config, config.seed, 1, {"critic_state_sha256": digest})
    deserialize.assert_not_called()
    assert path.read_bytes() == corrupt


def test_deserializer_uses_the_verified_snapshot_even_if_source_changes(saved):
    config, _, _, digest, path = saved
    data = path.read_bytes()
    real_load = torch.load

    def replace_source(stream, **kwargs):
        assert isinstance(stream, io.BytesIO)
        assert stream.getvalue() == data
        assert kwargs["weights_only"] is True
        path.write_bytes(b"changed after snapshot")
        return real_load(stream, **kwargs)

    with patch.object(torch, "load", side_effect=replace_source):
        payload = grpo._load_critic_checkpoint(
            config, config.seed, 1, {"critic_state_sha256": digest}
        )
    assert payload["step"] == 1
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        grpo._load_critic_checkpoint(config, config.seed, 1, {"critic_state_sha256": digest})


@pytest.mark.parametrize("digest", [None, "", "../outside", "/outside", "g" * 64, 42])
def test_invalid_binding_cannot_fall_back_to_legacy_or_read_a_path(tmp_path, digest):
    config = grpo.GRPOConfig(name="invalid", checkpoint_dir=str(tmp_path))
    with (
        patch.object(os, "open") as read,
        pytest.raises(RuntimeError, match="invalid critic SHA-256 binding"),
    ):
        grpo._load_critic_checkpoint(config, config.seed, 1, {"critic_state_sha256": digest})
    read.assert_not_called()


def test_matching_legacy_sidecar_remains_readable_without_migration(saved):
    config, _, _, _, bound_path = saved
    payload = torch.load(bound_path, map_location="cpu", weights_only=True)
    payload.pop("config_sha256")
    legacy = grpo._critic_checkpoint_path(config, config.seed)
    torch.save(payload, legacy)
    before = legacy.read_bytes()
    loaded = grpo._load_critic_checkpoint(config, config.seed, 1, {})
    assert loaded["step"] == 1
    assert legacy.read_bytes() == before


def test_different_seed_or_configuration_cannot_use_the_same_binding(saved):
    config, _, _, digest, _ = saved
    changed = replace(config, seed=config.seed + 1)
    with pytest.raises(RuntimeError, match="configuration binding does not match"):
        grpo._load_critic_checkpoint(changed, changed.seed, 1, {"critic_state_sha256": digest})


def test_existing_corrupted_content_address_is_never_overwritten(saved):
    config, critic, optimizer, _, path = saved
    path.write_bytes(b"corrupt immutable artifact")
    with pytest.raises(RuntimeError, match="artifact is inconsistent"):
        grpo._save_critic(config, config.seed, critic, optimizer, step=1)
    assert path.read_bytes() == b"corrupt immutable artifact"
    assert list(path.parent.glob(".critic-*")) == []


def test_failed_publication_preserves_existing_state_and_cleans_own_temporary(saved):
    config, critic, optimizer, _, path = saved
    before = path.read_bytes()
    with patch.object(os, "link", side_effect=OSError("simulated publish failure")):
        with pytest.raises(OSError, match="publish failure"):
            grpo._save_critic(config, config.seed, critic, optimizer, step=1)
    assert path.read_bytes() == before
    assert list(path.parent.glob(".critic-*")) == []


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_nonregular_artifacts_are_rejected_without_opening(saved, kind):
    config, _, _, _, source = saved
    digest = "f" * 64
    path = grpo._critic_checkpoint_path(config, config.seed, digest)
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "directory":
        path.mkdir()
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO test requires POSIX")
        os.mkfifo(path)
    with (
        patch.object(Path, "read_bytes") as read,
        pytest.raises(RuntimeError, match="regular file"),
    ):
        grpo._load_critic_checkpoint(config, config.seed, 1, {"critic_state_sha256": digest})
    read.assert_not_called()


def test_symlink_swap_at_open_cannot_redirect_legacy_restore(saved):
    config, _, _, _, source = saved
    legacy = grpo._critic_checkpoint_path(config, config.seed)
    legacy.write_bytes(source.read_bytes())
    real_open = os.open

    def swap_before_open(path, flags, *args, **kwargs):
        if Path(path) == legacy:
            legacy.unlink()
            legacy.symlink_to(source)
        return real_open(path, flags, *args, **kwargs)

    with (
        patch.object(os, "open", side_effect=swap_before_open),
        patch.object(torch, "load") as deserialize,
        pytest.raises(RuntimeError, match="regular file"),
    ):
        grpo._load_critic_checkpoint(config, config.seed, 1, {})
    deserialize.assert_not_called()


def test_open_descriptor_pins_snapshot_after_path_is_replaced(saved):
    config, _, _, digest, path = saved
    replacement = path.with_name("replacement-fixture.pt")
    replacement.write_bytes(b"not the committed checkpoint")
    real_open = os.open

    def swap_after_open(candidate, flags, *args, **kwargs):
        descriptor = real_open(candidate, flags, *args, **kwargs)
        if Path(candidate) == path:
            path.unlink()
            path.symlink_to(replacement)
        return descriptor

    with patch.object(os, "open", side_effect=swap_after_open):
        payload = grpo._load_critic_checkpoint(
            config, config.seed, 1, {"critic_state_sha256": digest}
        )
    assert payload["step"] == 1
    with pytest.raises(RuntimeError, match="regular file"):
        grpo._load_critic_checkpoint(config, config.seed, 1, {"critic_state_sha256": digest})


def test_checkpoint_save_without_optimizer_fails_before_publication(tmp_path):
    output = tmp_path / "not-created"
    config = grpo.GRPOConfig(name="no-optimizer", checkpoint_dir=str(output))
    critic = grpo.PromptValueCritic(hidden_dim=4)
    with pytest.raises(ValueError, match="requires optimizer state"):
        grpo._save_critic(config, config.seed, critic, step=1)
    assert not output.exists()


def test_secure_reader_fails_closed_without_nofollow_support(saved, monkeypatch):
    _, _, _, _, path = saved
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with patch.object(os, "open") as opening:
        with pytest.raises(RuntimeError, match="requires O_NOFOLLOW"):
            grpo._read_regular_critic_bytes(path)
    opening.assert_not_called()
