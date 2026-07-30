from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "zvf-program/next-submission/contrast_sampler.py"
SPEC = importlib.util.spec_from_file_location("next_submission_sampler", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SAMPLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLER
SPEC.loader.exec_module(SAMPLER)


def rollout(reward: float, length: int, prompt: tuple[int, ...] = (10, 11)):
    return SAMPLER.Rollout(prompt, tuple(range(100, 100 + length)), reward)


@pytest.mark.parametrize(
    ("rewards", "label"),
    [([0.0, 0.0], "all_wrong"), ([1.0, 1.0], "all_correct"), ([0.0, 1.0], "mixed")],
)
def test_binary_group_classification(rewards, label):
    assert SAMPLER.classify_binary_rewards(rewards) == label


def test_baseline_requires_and_charges_all_eight_rollouts():
    group = SAMPLER.assemble_group(
        SAMPLER.ARM_BASELINE,
        [rollout(0.0, 3), rollout(1.0, 4)],
        [rollout(float(index % 2), 5) for index in range(6)],
        eos_token_id=2,
    )
    assert group.generated_rollouts == 8
    assert group.charged_generated_tokens == 37
    assert group.active == (True,) * 8
    assert group.update_applied is True


@pytest.mark.parametrize(("reward", "label"), [(0.0, "all_wrong"), (1.0, "all_correct")])
def test_homogeneous_intervention_stops_at_two_and_emits_no_update(reward, label):
    group = SAMPLER.assemble_group(
        SAMPLER.ARM_EARLY_STOP,
        [rollout(reward, 3), rollout(reward, 7)],
        [],
        eos_token_id=2,
    )
    assert group.group_class == label
    assert group.generated_rollouts == 2
    assert group.charged_generated_tokens == 10
    assert group.active == (False,) * 8
    assert group.rewards == (None,) * 8
    assert group.completion_ids == ((2,),) * 8
    assert group.update_applied is False


def test_mixed_intervention_expands_to_eight():
    group = SAMPLER.assemble_group(
        SAMPLER.ARM_EARLY_STOP,
        [rollout(0.0, 2), rollout(1.0, 3)],
        [rollout(0.0, 4) for _ in range(6)],
        eos_token_id=2,
    )
    assert group.generated_rollouts == 8
    assert group.charged_generated_tokens == 29
    assert group.active == (True,) * 8
    assert group.update_applied is True


def test_homogeneous_intervention_cannot_expand_or_replace():
    with pytest.raises(SAMPLER.SamplerContractError, match="may not be expanded"):
        SAMPLER.assemble_group(
            SAMPLER.ARM_EARLY_STOP,
            [rollout(0.0, 2), rollout(0.0, 2)],
            [rollout(1.0, 2) for _ in range(6)],
            eos_token_id=2,
        )


def test_mixed_intervention_cannot_underfill_expansion():
    with pytest.raises(SAMPLER.SamplerContractError, match="exactly six"):
        SAMPLER.assemble_group(
            SAMPLER.ARM_EARLY_STOP,
            [rollout(0.0, 2), rollout(1.0, 2)],
            [rollout(1.0, 2)],
            eos_token_id=2,
        )


def test_telemetry_partitions_groups_and_counts_real_generation():
    groups = [
        SAMPLER.assemble_group(
            SAMPLER.ARM_EARLY_STOP,
            [rollout(0.0, 2), rollout(0.0, 2)],
            [],
            eos_token_id=2,
        ),
        SAMPLER.assemble_group(
            SAMPLER.ARM_EARLY_STOP,
            [rollout(1.0, 3), rollout(1.0, 3)],
            [],
            eos_token_id=2,
        ),
        SAMPLER.assemble_group(
            SAMPLER.ARM_EARLY_STOP,
            [rollout(0.0, 4), rollout(1.0, 4)],
            [rollout(0.0, 4) for _ in range(6)],
            eos_token_id=2,
        ),
    ]
    telemetry = SAMPLER.aggregate_group_telemetry(groups)
    assert telemetry == {
        "rollout_groups": 3,
        "generated_rollouts": 12,
        "charged_generated_tokens": 42,
        "updated_groups": 1,
        "all_wrong_fraction": pytest.approx(1 / 3),
        "all_correct_fraction": pytest.approx(1 / 3),
        "mixed_fraction": pytest.approx(1 / 3),
    }


@pytest.mark.parametrize("rewards", [[], [0.5], [0.0, 1.0, 0.0]])
def test_early_stop_decision_rejects_noncontract_inputs(rewards):
    with pytest.raises(SAMPLER.SamplerContractError):
        SAMPLER.should_expand(rewards)
