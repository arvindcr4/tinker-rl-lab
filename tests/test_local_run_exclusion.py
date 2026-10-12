"""Local duplicate-launch exclusion, cleanup, and committed-cache replay."""

from concurrent.futures import ThreadPoolExecutor
import threading
import json
import os
import select
import subprocess
import sys

import torch
from unittest.mock import patch
import pytest
from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime
from tests.test_completed_result_integrity import _completed


def test_duplicate_local_seed_stops_before_second_external_setup(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("offline launch test must not use network")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    config = _config(tmp_path, name="same-seed", steps=1)
    dataset = grpo.InMemoryDataset(train=[_example()])
    entered, release = threading.Event(), threading.Event()
    calls = []

    def hold_first(_config):
        calls.append(1)
        if len(calls) == 1:
            torch.rand(3)
            entered.set()
            assert release.wait(30), "first test run was not released"
        return "owner"

    def run():
        return grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )

    with (
        _runtime(monkeypatch),
        patch.object(grpo, "_preflight_hf", side_effect=hold_first),
        patch.object(grpo, "_start_wandb", wraps=grpo._start_wandb) as tracking,
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(run)
            try:
                assert entered.wait(30), "first run did not reach the gate"
                before_rng = torch.get_rng_state().clone()
                second = executor.submit(run)
                with pytest.raises(RuntimeError, match="already active|already running"):
                    second.result(timeout=5)
                assert torch.equal(torch.get_rng_state(), before_rng)
                assert tracking.call_count == 1
            finally:
                release.set()
                first.result(timeout=30)
    assert len(calls) == 1


def test_lock_release_preserves_body_error_and_other_paths_remain_independent(tmp_path):
    path = tmp_path / "seed1.json"
    failure = OSError("synthetic training failure")
    with pytest.raises(OSError, match="synthetic training failure") as raised:
        with (
            grpo._active_checkpoint_lock(path),
            grpo._active_checkpoint_lock(tmp_path / "seed2.json"),
        ):
            raise failure
    assert raised.value is failure
    with grpo._active_checkpoint_lock(path):
        with pytest.raises(grpo._CheckpointLockError, match="already active"):
            with grpo._active_checkpoint_lock(path):
                pytest.fail("duplicate lock acquired")
    assert path.with_name(path.name + ".lock").exists()


def test_process_exit_releases_kernel_lock_without_deleting_marker(tmp_path):
    path = tmp_path / "seed.json"
    lock_path = path.with_name(path.name + ".lock")
    program = "import fcntl,os,sys; fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600); fcntl.flock(fd,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.readline()"
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 10)[0], "lock holder did not start"
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(grpo._CheckpointLockError, match="already active"):
            with grpo._active_checkpoint_lock(path):
                pytest.fail("another process owns the lock")
    finally:
        process.terminate()
        process.communicate(timeout=10)
    assert lock_path.exists()
    with grpo._active_checkpoint_lock(path):
        pass


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_nonregular_lock_is_rejected_without_modifying_target(tmp_path, kind):
    path = tmp_path / "seed.json"
    lock_path = path.with_name(path.name + ".lock")
    target = tmp_path / "preserve.txt"
    target.write_text("synthetic preserved bytes")
    if kind == "symlink":
        lock_path.symlink_to(target)
    else:
        os.mkfifo(lock_path)
    with pytest.raises(grpo._CheckpointLockError, match="regular"):
        with grpo._active_checkpoint_lock(path):
            pytest.fail("nonregular lock accepted")
    assert target.read_text() == "synthetic preserved bytes"


def test_completed_replay_needs_no_lock_and_uses_one_committed_snapshot(tmp_path, monkeypatch):
    config, dataset, original, path = _completed(tmp_path, monkeypatch)
    before = path.read_bytes()
    with (
        _runtime(monkeypatch) as holder,
        patch.object(
            grpo,
            "_active_checkpoint_lock",
            side_effect=AssertionError("cached replay requested a write lock"),
        ),
        patch.object(grpo, "_load_checkpoint", wraps=grpo._load_checkpoint) as load,
    ):
        replay = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert replay == original
    assert load.call_count == 1
    assert "service_kwargs" not in holder
    assert path.read_bytes() == before


def test_completed_snapshot_does_not_become_unlocked_training_after_path_changes(
    tmp_path, monkeypatch
):
    config, dataset, original, path = _completed(tmp_path, monkeypatch)
    payload = json.loads(path.read_text())
    pending = {"config": payload["config"], "status": "started", "step": 0, "reward_trace": []}
    examples = list(dataset.train_examples())

    def replace_receipt(_self):
        grpo._write_checkpoint(path, pending)
        return examples

    monkeypatch.setattr(type(dataset), "train_examples", replace_receipt)
    with _runtime(monkeypatch) as holder:
        replay = grpo._run_one_seed(
            config, dataset, grpo.ExactMathReward(), _Tokenizer(), logger=lambda _: None
        )
    assert replay == original
    assert "service_kwargs" not in holder
    assert json.loads(path.read_text()) == pending


def test_busy_lock_gets_concise_cli_error(monkeypatch, capsys):
    from platform_tinker.tinkerrl import grpo_cli

    def busy(_argv):
        raise grpo._CheckpointLockError("Checkpoint is already active")

    monkeypatch.setattr(grpo_cli, "main", busy)
    assert grpo_cli.cli_entrypoint([]) == 1
    assert "already active" in capsys.readouterr().err
