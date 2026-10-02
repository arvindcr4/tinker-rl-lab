"""Fail-fast regression tests for the verl dryrun trainer and seed helpers.

Fast: exercises only the seeded dryrun path — no model downloads, no GPUs.
"""

import ast
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verl.config import VERLConfig
from verl.trainer import VERLTrainer


def _tiny_config(epochs: int = 4) -> VERLConfig:
    return VERLConfig(epochs=epochs)


def test_run_success_reports_no_failure():
    trainer = VERLTrainer(_tiny_config())
    result = asyncio.run(trainer.run())
    assert result["failed"] is False
    assert result["failed_steps"] == []
    assert result["final_step"] == 4
    assert len(result["reward_trace"]) == 4


def test_run_strict_reraises_by_default():
    trainer = VERLTrainer(_tiny_config(epochs=4))

    async def boom(step: int):
        if step == 2:
            raise ValueError("injected step failure")
        return await VERLTrainer.train_step(trainer, step)

    trainer.train_step = boom  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="injected step failure"):
        asyncio.run(trainer.run())
    assert trainer.failed_steps == [2]


def test_run_non_strict_flags_partial_trace():
    trainer = VERLTrainer(_tiny_config(epochs=4))

    async def boom(step: int):
        if step == 1:
            raise RuntimeError("injected step failure")
        return await VERLTrainer.train_step(trainer, step)

    trainer.train_step = boom  # type: ignore[method-assign]
    result = asyncio.run(trainer.run(strict=False))
    assert result["failed"] is True
    assert result["failed_steps"] == [1]
    # Partial trace is flagged, never presented as a full win.
    assert result["final_step"] == 1
    assert len(result["reward_trace"]) == 1


def test_dryrun_traces_are_seeded_and_repeatable():
    first = asyncio.run(VERLTrainer(_tiny_config()).run())
    second = asyncio.run(VERLTrainer(_tiny_config()).run())
    assert first["reward_trace"] == second["reward_trace"]
    assert first["loss_trace"] == second["loss_trace"]


def test_deterministic_mode_failure_reraises_with_context(monkeypatch):
    """seed.py must not swallow deterministic-mode failures (except-pass)."""
    import utils.seed as seed_mod

    real_torch = sys.modules.get("torch")

    class _Backends:
        cudnn = type("cudnn", (), {"deterministic": False, "benchmark": True})()

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def manual_seed(seed: int) -> None:
            pass

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            pass

        @staticmethod
        def device_count() -> int:
            return 0

    class _FakeTorch:
        __version__ = "fake"
        backends = _Backends()
        cuda = _Cuda()
        version = type("version", (), {"cuda": "fake"})()

        @staticmethod
        def manual_seed(seed: int) -> None:
            pass

        @staticmethod
        def use_deterministic_algorithms(flag: bool) -> None:
            raise RuntimeError("some op is nondeterministic")

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    try:
        with pytest.raises(RuntimeError, match="deterministic"):
            seed_mod.set_global_seed(42, deterministic_cudnn=True)
    finally:
        if real_torch is not None:
            monkeypatch.setitem(sys.modules, "torch", real_torch)
        else:
            monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_tinker_grpo_signature_is_typed_and_returns_dict():
    """Check annotations/return via AST — avoids importing torch/tinker."""
    path = os.path.join(os.path.dirname(__file__), "..", "utils", "tinker_grpo.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_grpo_training"
    )
    assert fn.returns is not None, "run_grpo_training must declare a return type"
    untyped = [a.arg for a in fn.args.args if a.annotation is None]
    assert not untyped, f"untyped params: {untyped}"
    returns_value = any(
        isinstance(node, ast.Return) and node.value is not None for node in ast.walk(fn)
    )
    assert returns_value, "run_grpo_training must return a result dict"
