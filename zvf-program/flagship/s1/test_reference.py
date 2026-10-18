from __future__ import annotations

import unittest

import torch

from .reference import (
    ATOL,
    assert_trace_close,
    decide_policy,
    decide_policy_observation,
    objective_trace,
)
import pytest


class ObjectiveReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rewards = [[0.0, 1.0], [0.0, 1.0]]
        self.logps = [[-0.1, -0.2], [-0.2, -0.4], [-0.3, -0.5], [-0.4, -0.6]]
        self.old = [[-0.2, -0.2], [-0.2, -0.3], [-0.3, -0.4], [-0.5, -0.6]]
        self.mask = [[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]]

    def trace(self, arm: str, **kwargs):
        return objective_trace(
            arm=arm,
            rewards=kwargs.pop("rewards", self.rewards),
            logps=self.logps,
            old_logps=self.old,
            mask=self.mask,
            **kwargs,
        )

    def test_complete_trace_is_float64_and_has_gradient(self) -> None:
        trace = self.trace("grpo")
        assert trace.loss.dtype == torch.float64
        assert trace.gradient.shape == torch.Size([8])
        assert torch.isfinite(trace.gradient).all()
        assert trace.selected_indices == (0, 1, 2, 3)

    def test_drgrpo_changes_only_advantage_scale(self) -> None:
        grpo = self.trace("grpo")
        drgrpo = self.trace("drgrpo")
        assert not torch.allclose(grpo.advantages, drgrpo.advantages, atol=ATOL)
        assert torch.equal(grpo.mask, drgrpo.mask)
        assert torch.equal(grpo.ratios, drgrpo.ratios)

    def test_dapo_upper_clip_activates_without_changing_advantages(self) -> None:
        logps = [[-0.2, -0.2], [0.2, -0.4], [-0.3, -0.5], [-0.4, -0.6]]
        grpo = objective_trace(
            arm="grpo", rewards=self.rewards, logps=logps, old_logps=self.old, mask=self.mask
        )
        dapo = objective_trace(
            arm="dapo", rewards=self.rewards, logps=logps, old_logps=self.old, mask=self.mask
        )
        assert torch.equal(grpo.advantages, dapo.advantages)
        assert not torch.equal(grpo.surrogate, dapo.surrogate)

    def test_gspo_uses_one_ratio_per_completion(self) -> None:
        trace = self.trace("gspo")
        assert torch.equal(trace.ratios[:, 0], trace.ratios[:, 1])
        grpo = self.trace("grpo")
        assert not torch.equal(trace.ratios, grpo.ratios)

    def test_aero_degenerate_group_uses_frozen_posterior_formula(self) -> None:
        trace = self.trace(
            "aero",
            rewards=[[0.0, 0.0], [1.0, 1.0]],
            aero_successes=[0, 4],
            aero_observations=[4, 4],
        )
        expected_p0 = 1.0 / 6.0
        expected = (0.0 - expected_p0) / (expected_p0 * (1.0 - expected_p0)) ** 0.5
        self.assertAlmostEqual(trace.advantages[0].item(), expected, places=12)
        self.assertAlmostEqual(trace.advantages[1].item(), expected, places=12)

    def test_zero_mask_completion_has_zero_gradient_contribution(self) -> None:
        masked = objective_trace(
            arm="grpo",
            rewards=self.rewards,
            logps=self.logps,
            old_logps=self.old,
            mask=[[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        )
        assert torch.equal(masked.gradient[2:4], torch.zeros(2, dtype=torch.float64))

    def test_invalid_arm_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown objective arm"):
            self.trace("not-an-arm")

    def test_trace_comparator_checks_gradients(self) -> None:
        trace = self.trace("grpo")
        assert_trace_close(trace, trace)
        changed = self.trace("drgrpo")
        with pytest.raises(AssertionError):
            assert_trace_close(trace, changed)

    def test_autograd_matches_central_finite_difference(self) -> None:
        trace = self.trace("grpo")
        epsilon = 1e-6
        for flat_index in (0, 2, 7):
            row, column = divmod(flat_index, 2)
            plus = [values[:] for values in self.logps]
            minus = [values[:] for values in self.logps]
            plus[row][column] += epsilon
            minus[row][column] -= epsilon
            plus_loss = objective_trace(
                arm="grpo", rewards=self.rewards, logps=plus, old_logps=self.old, mask=self.mask
            ).loss
            minus_loss = objective_trace(
                arm="grpo", rewards=self.rewards, logps=minus, old_logps=self.old, mask=self.mask
            ).loss
            finite_difference = (plus_loss - minus_loss) / (2 * epsilon)
            self.assertAlmostEqual(
                trace.gradient[flat_index].item(),
                finite_difference.item(),
                places=7,
            )

    def test_completion_permutation_is_equivariant(self) -> None:
        baseline = self.trace("grpo")
        permutation = (1, 0, 3, 2)
        permuted = objective_trace(
            arm="grpo",
            rewards=[[1.0, 0.0], [1.0, 0.0]],
            logps=[self.logps[index] for index in permutation],
            old_logps=[self.old[index] for index in permutation],
            mask=[self.mask[index] for index in permutation],
        )
        inverse = torch.argsort(torch.tensor(permutation))
        self.assertAlmostEqual(permuted.loss.item(), baseline.loss.item(), places=12)
        assert torch.allclose(permuted.advantages[inverse], baseline.advantages)
        assert torch.allclose(
            permuted.gradient.reshape(4, 2)[inverse], baseline.gradient.reshape(4, 2)
        )

    def test_reward_translation_is_invariant_for_every_arm(self) -> None:
        translated = [[10.0, 11.0], [-4.0, -3.0]]
        for arm in ("grpo", "dapo", "gspo", "drgrpo", "aero"):
            with self.subTest(arm=arm):
                baseline = self.trace(arm)
                changed = self.trace(arm, rewards=translated)
                assert torch.allclose(baseline.advantages, changed.advantages)
                assert torch.allclose(baseline.gradient, changed.gradient)

    def test_positive_reward_scaling_has_prespecified_effect(self) -> None:
        scaled = [[0.0, 3.0], [0.0, 3.0]]
        for arm in ("grpo", "dapo", "gspo", "aero"):
            with self.subTest(arm=arm):
                baseline = self.trace(arm)
                changed = self.trace(arm, rewards=scaled)
                assert torch.allclose(baseline.advantages, changed.advantages)
                assert torch.allclose(baseline.gradient, changed.gradient)
        drgrpo = self.trace("drgrpo")
        scaled_drgrpo = self.trace("drgrpo", rewards=scaled)
        assert torch.allclose(scaled_drgrpo.advantages, 3.0 * drgrpo.advantages)
        assert torch.allclose(scaled_drgrpo.gradient, 3.0 * drgrpo.gradient)


class PolicyReferenceTests(unittest.TestCase):
    def test_fixed_arms(self) -> None:
        assert decide_policy("static_g8", [0, 1]).group_size == 8
        assert decide_policy("static_g16", [0, 1]).group_size == 16

    def test_symmetric_zvf_cannot_distinguish_sign(self) -> None:
        assert decide_policy("symmetric_zvf", [0, 0]).action == "escalate"
        assert decide_policy("symmetric_zvf", [1, 1]).action == "escalate"

    def test_failure_only_does_not_drop_all_correct(self) -> None:
        assert decide_policy("failure_only", [0, 0]).action == "escalate"
        assert decide_policy("failure_only", [1, 1]).action == "keep"

    def test_invalid_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown policy"):
            decide_policy("not-a-policy", [0, 1])

    def test_sign_aware_distinguishes_all_zero_and_all_one(self) -> None:
        assert decide_policy("boundary_aware", [0, 0]).action == "escalate"
        assert decide_policy("boundary_aware", [1, 1]).action == "drop"

    def test_wilson_arm_requires_evidence_before_drop(self) -> None:
        weak = decide_policy("full_triage", [1, 1])
        strong = decide_policy(
            "full_triage",
            [1, 1],
            history_successes=198,
            history_total=198,
        )
        assert weak.action == "keep"
        assert strong.action == "drop"

    def test_informative_group_is_kept_by_adaptive_arms(self) -> None:
        for arm in ("symmetric_zvf", "failure_only", "boundary_aware", "full_triage"):
            assert decide_policy(arm, [0, 1]).action == "keep"

    def test_unresolved_reward_states_fail_closed_to_recheck(self) -> None:
        for status in ("noisy", "missing", "delayed"):
            with self.subTest(status=status):
                decision = decide_policy_observation(
                    "full_triage",
                    [None, None],
                    status=status,
                )
                assert decision.action == "recheck"
                assert decision.group_size == 0

    def test_observed_status_preserves_frozen_policy_action(self) -> None:
        direct = decide_policy("boundary_aware", [0, 0])
        observed = decide_policy_observation("boundary_aware", [0, 0], status="observed")
        assert observed == direct

    def test_observed_missing_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="observed rewards cannot contain missing"):
            decide_policy_observation("full_triage", [0.0, None], status="observed")


if __name__ == "__main__":
    unittest.main()
