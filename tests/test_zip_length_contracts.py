"""Offline regressions for aligned scientific pairs and intentional prefixes."""

from copy import deepcopy

import pytest
import torch

from platform_local.trl_integrations.scientific import correctness_rewards, group_diagnostics
from platform_tinker.tinkerrl.critic import explained_variance
from platform_tinker.tinkerrl.grpo import _nll_mask_for_group, make_grpo_loss_fn
from tools.check_thesis_evidence import mcnemar
from tools.audit_portfolio_comparability import audit
from tests.test_portfolio_comparability_audit import _receipt
from utils.stats import contract_scores


@pytest.mark.parametrize(("trained", "base"), [([1], []), ([], [1]), ([1, 0], [1]), ([1], [1, 0])])
def test_mcnemar_rejects_unpaired_observations(trained, base):
    with pytest.raises(ValueError, match="zip.*argument"):
        mcnemar(trained, base)


@pytest.mark.parametrize(("rewards", "cuts"), [([1.0], []), ([], [False]), ([1.0, 0.0], [False])])
@pytest.mark.parametrize("mask_truncated", [False, True])
def test_nll_flags_require_every_response_even_when_masking_disabled(rewards, cuts, mask_truncated):
    with pytest.raises(ValueError, match="zip.*argument"):
        _nll_mask_for_group(rewards, cuts, 0.5, mask_truncated)


def test_aligned_nll_flags_preserve_mask_behavior():
    assert _nll_mask_for_group([1.0, 1.0, 0.0], [False, True, False], 0.5, True) == [
        True,
        False,
        False,
    ]
    assert _nll_mask_for_group([1.0, 1.0, 0.0], [False, True, False], 0.5, False) == [
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("mask", [None, [False, False]])
def test_absent_or_empty_auxiliary_population_keeps_policy_loss(mask):
    logprobs = [torch.tensor([-0.2], requires_grad=True), torch.tensor([-0.4], requires_grad=True)]
    loss, metrics = make_grpo_loss_fn([1.0, -1.0], nll_mask=mask, nll_coef=0.5)(None, logprobs)
    assert loss.item() == pytest.approx(-0.1)
    assert metrics["nll_loss"] == 0.0
    loss.backward()
    assert logprobs[0].grad.item() == pytest.approx(-0.5)
    assert logprobs[1].grad.item() == pytest.approx(0.5)


@pytest.mark.parametrize("mask", [[], [True], [True, False, True]])
def test_present_auxiliary_mask_cannot_drop_or_add_responses(mask):
    with pytest.raises(ValueError, match="nll_mask must pair 1:1"):
        make_grpo_loss_fn([1.0, -1.0], nll_mask=mask)(
            None, [torch.tensor([-0.2]), torch.tensor([-0.4])]
        )


@pytest.mark.parametrize(("left", "right"), [([], [1.0]), ([1.0], []), ([1.0], [1.0, 1.0])])
def test_explained_variance_checks_lengths_before_constant_target_shortcut(left, right):
    with pytest.raises(ValueError, match="length mismatch"):
        explained_variance(left, right)


def test_grading_and_prompt_groups_reject_missing_metadata():
    with pytest.raises(ValueError, match="One gold answer"):
        correctness_rewards(["1", "2"], ["1"])
    with pytest.raises(ValueError, match="Verified prompt IDs"):
        group_diagnostics([1.0, 0.0], ["prompt"], 2)


def test_unequal_cohort_is_reported_while_shared_prefix_is_audited():
    base, trained = _receipt("base"), _receipt("trained")
    extra = deepcopy(trained["records"][0])
    extra["source_id"] = "extra"
    extra["sampling_seed"] = 14
    trained["records"].append(extra)
    trained["count"] = 3
    trained["mean_reward"] = 2 / 3
    trained["exact_match_rate"] = 2 / 3
    trained["suites"][extra["suite_id"]]["count"] = 2
    trained["suites"][extra["suite_id"]]["exact_matches"] = 2
    trained["records"][0]["source_id"] = "different"
    before = deepcopy((base, trained))
    result = audit(base, trained)
    assert result["status"] == "incompatible"
    assert "ordered_cohort_length" in result["mismatches"]
    assert "ordered_records.source_id" in result["mismatches"]
    assert (base, trained) == before


@pytest.mark.parametrize("steps", [[0], [0, 1], [0, 1, 2]])
def test_adjacent_step_checks_accept_singletons_and_increasing_sequences(steps):
    scores, _ = contract_scores({42: [{"step": step, "accuracy": 0.5} for step in steps]})
    assert scores == [0.5]


@pytest.mark.parametrize("steps", [[0, 0], [1, 0], [0, 2, 1]])
def test_adjacent_step_checks_still_reject_invalid_neighbors(steps):
    with pytest.raises(ValueError, match="strictly increasing"):
        contract_scores({42: [{"step": step, "accuracy": 0.5} for step in steps]})
