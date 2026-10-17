"""Ctrl+C must release local ownership and finalize initialized tracking as failed."""

import json
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo, pavlov_portfolio_eval as evaluator
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime, _Seq
from tests import test_portfolio_eval_integrity

harness = test_portfolio_eval_integrity.harness


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("cancellation checks must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def run(config):
    return grpo._run_one_seed(
        config,
        grpo.InMemoryDataset(train=[_example()]),
        grpo.ExactMathReward(),
        _Tokenizer(),
        logger=lambda _: None,
    )


@pytest.mark.parametrize("cleanup_failure", [None, "mark", "finish"])
def test_setup_interrupt_finishes_tracking_and_preserves_interrupt(
    tmp_path, monkeypatch, cleanup_failure
):
    config = _config(tmp_path)
    interruption = KeyboardInterrupt("synthetic user cancellation")
    with _runtime(monkeypatch) as holder:
        if cleanup_failure == "mark":

            def failed_status(*_args):
                raise RuntimeError("synthetic status failure")

            monkeypatch.setattr(grpo, "_mark_wandb_failure", failed_status)
        elif cleanup_failure == "finish":
            holder["run"].finish.side_effect = RuntimeError("synthetic finish failure")
        with patch.object(grpo, "_preflight_hf", side_effect=interruption):
            with pytest.raises(KeyboardInterrupt) as error:
                run(config)
    assert error.value is interruption
    holder["run"].finish.assert_called_once_with(exit_code=1)
    assert holder["run"].summary["final_status"] == "failed"
    assert "service_kwargs" not in holder
    assert not grpo._checkpoint_path(config, config.seed).exists()
    with grpo._active_checkpoint_lock(grpo._checkpoint_path(config, config.seed)):
        pass
    if cleanup_failure:
        assert isinstance(error.value.__cause__, RuntimeError)


def test_interrupt_after_committed_step_keeps_resume_receipt(tmp_path, monkeypatch):
    config = _config(tmp_path, steps=2, save_every=1)
    interruption = KeyboardInterrupt("synthetic user cancellation")
    calls = []

    def sample(*_args):
        calls.append(1)
        if len(calls) == 2:
            raise interruption
        return SimpleNamespace(sequences=[_Seq([1]), _Seq([2])])

    with _runtime(monkeypatch, sample_fn=sample) as holder:
        with pytest.raises(KeyboardInterrupt) as error:
            run(config)
    assert error.value is interruption
    holder["run"].finish.assert_called_once_with(exit_code=1)
    payload = json.loads(grpo._checkpoint_path(config, config.seed).read_text())
    assert payload["status"] == "started"
    assert payload["step"] == 1
    assert "result" not in payload
    with _runtime(monkeypatch):
        recovered = run(config)
    assert recovered.resumed_from_step == 1
    assert len(recovered.reward_trace) == 2


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_evaluation_interrupt_is_not_a_scored_receipt(harness, monkeypatch, cleanup_failure):
    interruption = KeyboardInterrupt("synthetic evaluation cancellation")

    def cancelled(*_args):
        raise interruption

    monkeypatch.setattr(evaluator, "PavlovNonXLAMReward", lambda: SimpleNamespace(score=cancelled))
    if cleanup_failure:
        init = sys.modules["wandb"].init

        def init_failing(**kwargs):
            run = init(**kwargs)

            def finish(**kwargs):
                harness.finished.append(kwargs["exit_code"])
                raise RuntimeError("synthetic finish failure")

            run.finish = finish
            return run

        monkeypatch.setattr(sys.modules["wandb"], "init", init_failing)
    with pytest.raises(KeyboardInterrupt) as error:
        harness.invoke()
    assert error.value is interruption
    assert harness.finished == [1]
    assert not harness.output.exists()
    if cleanup_failure:
        assert isinstance(error.value.__cause__, RuntimeError)
