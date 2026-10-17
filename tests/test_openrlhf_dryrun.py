"""Synthetic OpenRLHF fixtures require explicit opt-in and separate identity."""

import asyncio
from importlib import metadata
from unittest.mock import Mock

import pytest

from platform_modal.openrlhf.config import OpenRLHFConfig
from platform_modal.openrlhf import trainer


@pytest.mark.parametrize("version", [None, "0.8.4"])
def test_real_default_never_silently_downgrades_to_synthetic(monkeypatch, version):
    probe = Mock(side_effect=metadata.PackageNotFoundError) if version is None else lambda name: version
    monkeypatch.setattr(metadata, "version", probe)
    with pytest.raises(RuntimeError, match="Real execution|Unsupported OpenRLHF"):
        asyncio.run(trainer.OpenRLHFTrainer(OpenRLHFConfig(epochs=2)).run())


def test_explicit_synthetic_fixtures_respect_seed_without_empirical_fields():
    def run(seed):
        return asyncio.run(trainer.OpenRLHFTrainer(OpenRLHFConfig(seed=seed, epochs=3), dry_run=True).run())

    first, repeated, other = run(42), run(42), run(123)
    assert first["mode"] == "dryrun"
    assert first["data_kind"] == "synthetic_fixture"
    assert first["fixture_rewards"] == repeated["fixture_rewards"]
    assert first["fixture_rewards"] != other["fixture_rewards"]
    assert "reward_trace" not in first
    assert "final_step" not in first


def test_synthetic_failure_does_not_return_partial_success(monkeypatch):
    monkeypatch.setattr(trainer, "run_simulation", Mock(side_effect=RuntimeError("injected synthetic failure")))
    with pytest.raises(RuntimeError, match="injected synthetic failure"):
        asyncio.run(trainer.OpenRLHFTrainer(OpenRLHFConfig(epochs=3), dry_run=True).run())
