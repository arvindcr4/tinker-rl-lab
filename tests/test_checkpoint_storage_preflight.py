"""Known local storage failures must precede avoidable remote work."""

from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


def run(config):
    return grpo._run_one_seed(
        config,
        grpo.InMemoryDataset(train=[_example()]),
        grpo.ExactMathReward(),
        _Tokenizer(),
        logger=lambda _: None,
    )


def test_occupied_checkpoint_directory_fails_before_tracking_or_provider(tmp_path, monkeypatch):
    occupied = tmp_path / "occupied"
    occupied.write_text("preserve synthetic existing file")
    config = _config(occupied)
    with _runtime(monkeypatch) as holder, patch.object(grpo, "_start_wandb") as tracking:
        with pytest.raises(FileExistsError):
            run(config)
    tracking.assert_not_called()
    assert "service_kwargs" not in holder
    assert occupied.read_text() == "preserve synthetic existing file"


@pytest.mark.parametrize("stage", ["open", "fsync"])
def test_storage_probe_failure_cleans_only_its_stage(tmp_path, monkeypatch, stage):
    sentinel = tmp_path / "unrelated.txt"
    sentinel.write_text("keep")
    config = _config(tmp_path)
    owner, method = (grpo.tempfile, "NamedTemporaryFile") if stage == "open" else (grpo.os, "fsync")
    with _runtime(monkeypatch) as holder, patch.object(grpo, "_start_wandb") as tracking:
        with patch.object(owner, method, side_effect=OSError("synthetic storage failure")):
            with pytest.raises(OSError, match="synthetic storage failure"):
                run(config)
    tracking.assert_not_called()
    assert "service_kwargs" not in holder
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(("steps", "save_every"), [(2, 1), (1, 10)])
def test_critic_save_failure_precedes_that_steps_export(tmp_path, monkeypatch, steps, save_every):
    config = _config(
        tmp_path, steps=steps, save_every=save_every, critic_enabled=True, critic_hidden_dim=8
    )
    with _runtime(monkeypatch):
        with patch.object(grpo, "_publish_checkpoint", wraps=grpo._publish_checkpoint) as publish:
            with patch.object(
                grpo, "_save_critic", side_effect=OSError("synthetic critic storage failure")
            ):
                with pytest.raises(OSError, match="synthetic critic storage failure"):
                    run(config)
    assert [call.kwargs["step"] for call in publish.call_args_list] == [0]


def test_completed_receipt_does_not_need_a_write_probe(tmp_path, monkeypatch):
    config = _config(tmp_path)
    with _runtime(monkeypatch):
        run(config)
    path = grpo._checkpoint_path(config, config.seed)
    before = path.read_bytes()
    with _runtime(monkeypatch):
        with patch.object(
            grpo.tempfile, "NamedTemporaryFile", side_effect=AssertionError("unexpected write")
        ):
            result = run(config)
    assert result.reward_trace == [0.5]
    assert path.read_bytes() == before
