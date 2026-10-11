"""Synthetic OpenRLHF traces must not masquerade as native training evidence."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from platform_modal.openrlhf.config import OpenRLHFConfig
from platform_modal.openrlhf.trainer import OpenRLHFTrainer


def test_real_install_does_not_silently_downgrade_to_synthetic(monkeypatch):
    monkeypatch.setitem(sys.modules, "openrlhf", SimpleNamespace(__version__="0.8.4"))
    trainer = OpenRLHFTrainer(OpenRLHFConfig(epochs=2))
    with pytest.raises(RuntimeError, match="run_openrlhf_training"):
        asyncio.run(trainer.run())
    assert trainer.reward_history == []
    assert trainer.loss_history == []


def test_synthetic_traces_respect_configured_seed(monkeypatch):
    monkeypatch.setitem(sys.modules, "openrlhf", None)

    def run(seed):
        return asyncio.run(OpenRLHFTrainer(OpenRLHFConfig(seed=seed, epochs=3)).run())

    first, repeated, other = run(42), run(42), run(123)
    assert first["mode"] == "dryrun"
    for field in ("reward_trace", "loss_trace"):
        assert first[field] == repeated[field]
        assert first[field] != other[field]


def test_synthetic_step_failure_does_not_return_partial_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "openrlhf", None)
    trainer = OpenRLHFTrainer(OpenRLHFConfig(epochs=3))

    async def step(index):
        if index == 1:
            raise RuntimeError("injected synthetic failure")
        return await OpenRLHFTrainer.train_step(trainer, index)

    monkeypatch.setattr(trainer, "train_step", step)
    with pytest.raises(RuntimeError, match="injected synthetic failure"):
        asyncio.run(trainer.run())
    assert trainer.current_step == 1
