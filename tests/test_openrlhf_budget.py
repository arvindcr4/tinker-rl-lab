"""Enumerate batching arithmetic without GPU execution or fabricated update evidence."""

from dataclasses import replace
from unittest.mock import patch

import pytest

from platform_modal.openrlhf.budget import plan_native_budget
from platform_modal.openrlhf.config import OpenRLHFConfig
from platform_modal.openrlhf import trainer
from platform_local.unified.canonical import CanonicalSpec
from platform_local.unified.launcher import UnifiedLauncher


def config(**changes):
    settings = {
        "algorithm": {"algorithm": "grpo", "sample_num": 8},
        "max_samples": 32,
        "rollout_batch_size": 8,
        "train_batch_size": 64,
        "micro_batch_size": 2,
        "epochs": 1,
        "num_gpus": 1,
        "remote_rm_url": "http://127.0.0.1:8765/get_reward",
    }
    settings.update(changes)
    return OpenRLHFConfig(**settings)


@pytest.mark.parametrize(
    ("world", "micro", "global_batch", "epochs"),
    [(1, 1, 64, 1), (1, 2, 16, 3), (2, 2, 32, 2), (4, 4, 64, 1)],
)
def test_plans_match_enumerated_actor_batches_without_cross_rollout_accumulation(
    world, micro, global_batch, epochs
):
    cfg = config(
        num_gpus=world, micro_batch_size=micro, train_batch_size=global_batch, epochs=epochs
    )
    plan = plan_native_budget(cfg)
    # Independently enumerate equal actor shards, inner epochs and accumulation
    # boundaries. This tests arithmetic only, not DeepSpeed or optimizer success.
    counts = []
    for rank in range(world):
        boundaries = 0
        pending = 0
        for _ in range(4):
            completions = list(range(64))
            actor_samples = completions[rank * (64 // world) : (rank + 1) * (64 // world)]
            for _ in range(epochs):
                for index in range(0, len(actor_samples), micro):
                    batch = actor_samples[index : index + micro]
                    assert len(batch) == micro
                    pending += len(batch) * world
                    if pending == global_batch:
                        boundaries += 1
                        pending = 0
                assert pending == 0
            assert pending == 0
        counts.append(boundaries)
    assert counts == [plan["planned_update_boundaries"]] * world
    assert plan["planned_rollouts"] == 4
    assert "measured" in plan["scope"]
    assert "steps" not in plan


@pytest.mark.parametrize("train_batch", [1024, 48, 128])
def test_rejects_accumulation_spanning_rollouts_before_provider_work(tmp_path, train_batch):
    with (
        patch.object(trainer, "_validate_native_runtime") as runtime,
        patch.object(trainer, "_prepare_gsm8k_500_jsonl") as prepare,
        pytest.raises(ValueError, match="whole global training batches"),
    ):
        trainer.run_openrlhf_training(config(train_batch_size=train_batch), str(tmp_path / "run"))
    runtime.assert_not_called()
    prepare.assert_not_called()
    assert not (tmp_path / "run").exists()


def test_rollouts_epochs_and_planned_update_boundaries_are_different_quantities():
    plan = plan_native_budget(config(train_batch_size=16, epochs=3))
    assert plan["planned_rollouts"] == 4
    assert plan["planned_update_boundaries_per_rollout"] == 12
    assert plan["planned_update_boundaries"] == 48


@pytest.mark.parametrize("steps", [0, 1, 30])
def test_optimizer_budget_is_never_silently_converted_to_dataset_budget(steps):
    with pytest.raises(ValueError, match="max_steps"):
        plan_native_budget(config(max_steps=steps))


def test_unified_actual_native_path_rejects_canonical_budget_without_backend_work():
    item = UnifiedLauncher()
    item.framework = "openrlhf"
    item.algorithm = "grpo"
    item.model = "synthetic/model"
    item.spec = replace(CanonicalSpec(), training_steps=30)
    with (
        patch.object(trainer, "_validate_native_runtime") as runtime,
        patch.object(trainer.subprocess, "run") as native,
        pytest.raises(ValueError, match="max_steps"),
    ):
        item._run_openrlhf()
    runtime.assert_not_called()
    native.assert_not_called()
