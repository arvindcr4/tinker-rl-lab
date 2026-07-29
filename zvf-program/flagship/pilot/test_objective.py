from __future__ import annotations

import math
import unittest

import torch

from pilot.objective import (
    ObjectiveContractError,
    condition_loss,
    condition_trace,
    gradient_diagnostic,
)


def fixture(
    *,
    lengths: list[int],
    rewards: list[float],
    active: list[bool],
) -> dict[str, torch.Tensor]:
    width = max(lengths)
    mask = torch.zeros((8, width), dtype=torch.float64)
    for index, (length, is_active) in enumerate(zip(lengths, active, strict=True)):
        if is_active:
            mask[index, :length] = 1.0
    old = torch.zeros((8, width), dtype=torch.float64)
    offsets = torch.linspace(-0.08, 0.08, steps=8, dtype=torch.float64)[:, None]
    positions = torch.linspace(0.2, 1.0, steps=width, dtype=torch.float64)[None, :]
    logps = old + offsets * positions
    return {
        "rewards": torch.tensor(rewards, dtype=torch.float64),
        "logps": logps,
        "old_logps": old,
        "completion_mask": mask,
        "active_rows": torch.tensor(active, dtype=torch.bool),
    }


class ObjectiveTests(unittest.TestCase):
    def test_balanced_equal_length_is_directionally_equivalent(self) -> None:
        data = fixture(
            lengths=[6] * 8,
            rewards=[0, 0, 0, 0, 1, 1, 1, 1],
            active=[True] * 8,
        )
        intended = condition_trace(condition="intended_full", **data)
        native = condition_trace(condition="native_trl", **data)
        diagnostic = gradient_diagnostic(intended, native)
        self.assertEqual(diagnostic.relation, "nonzero")
        self.assertGreaterEqual(diagnostic.cosine, 0.999999)
        self.assertLess(diagnostic.relative_l2, 0.001)

    def test_filtered_variable_length_changes_native_gradient_direction(self) -> None:
        data = fixture(
            lengths=[2, 3, 4, 8, 16, 24, 32, 40],
            rewards=[1, 0, 1, 0, 0, 1, 1, 0],
            active=[True, True, True, False, True, True, False, True],
        )
        intended = condition_trace(condition="intended_full", **data)
        native = condition_trace(condition="native_trl", **data)
        diagnostic = gradient_diagnostic(intended, native)
        self.assertLess(diagnostic.cosine, 0.99)
        self.assertGreater(diagnostic.relative_l2, 0.05)

    def test_reduction_ablation_changes_direction_but_epsilon_does_not(self) -> None:
        data = fixture(
            lengths=[2, 3, 4, 8, 16, 24, 32, 40],
            rewards=[1, 0, 1, 0, 0, 1, 1, 0],
            active=[True, True, True, False, True, True, False, True],
        )
        intended = condition_trace(condition="intended_full", **data)
        reduction = condition_trace(condition="reduction_only", **data)
        epsilon = condition_trace(condition="epsilon_only", **data)
        reduction_gradient = torch.tensor(reduction.gradient)
        intended_gradient = torch.tensor(intended.gradient)
        epsilon_gradient = torch.tensor(epsilon.gradient)
        reduction_cosine = torch.nn.functional.cosine_similarity(
            reduction_gradient, intended_gradient, dim=0
        )
        epsilon_cosine = torch.nn.functional.cosine_similarity(
            epsilon_gradient, intended_gradient, dim=0
        )
        self.assertLess(float(reduction_cosine), 0.99)
        self.assertGreater(float(epsilon_cosine), 0.999999)

    def test_all_equal_selected_rewards_produce_explicit_joint_zero_diagnostic(self) -> None:
        data = fixture(
            lengths=[4] * 8,
            rewards=[0] * 8,
            active=[True] * 8,
        )
        intended = condition_trace(condition="intended_full", **data)
        native = condition_trace(condition="native_trl", **data)
        diagnostic = gradient_diagnostic(intended, native)
        self.assertEqual(diagnostic.relation, "joint_zero")
        self.assertIsNone(diagnostic.cosine)
        self.assertIsNone(diagnostic.relative_l2)
        self.assertEqual(diagnostic.intended_gradient_norm, 0.0)
        self.assertEqual(diagnostic.native_gradient_norm, 0.0)

    def test_inactive_rows_must_have_zero_masks(self) -> None:
        data = fixture(
            lengths=[4] * 8,
            rewards=[0, 1, 0, 1, 0, 1, 0, 1],
            active=[True, True, True, False, True, True, False, True],
        )
        data["completion_mask"][3, 0] = 1.0
        with self.assertRaisesRegex(ObjectiveContractError, "inactive rows"):
            condition_trace(condition="intended_full", **data)

    def test_training_loss_preserves_autograd_to_supplied_logps(self) -> None:
        data = fixture(
            lengths=[2, 3, 4, 8, 16, 24, 32, 40],
            rewards=[1, 0, 1, 0, 0, 1, 1, 0],
            active=[True, True, True, False, True, True, False, True],
        )
        logps = data["logps"].detach().requires_grad_(True)
        loss, advantages = condition_loss(
            condition="intended_full",
            rewards=data["rewards"],
            logps=logps,
            old_logps=data["old_logps"],
            completion_mask=data["completion_mask"],
            active_rows=data["active_rows"],
        )
        gradient = torch.autograd.grad(loss, logps)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(advantages.is_leaf)
        self.assertFalse(advantages.requires_grad)

    def test_spectral_legendre_grpo_eliminates_zvf_starvation_under_zero_reward_variance(self) -> None:
        # All rewards equal 1.0 (zero reward variance / ZVF starvation)
        data = fixture(
            lengths=[2, 3, 4, 8, 16, 24, 32, 40],
            rewards=[1.0] * 8,
            active=[True] * 8,
        )
        logps = data["logps"].detach().requires_grad_(True)
        # Under standard GRPO / intended_full, advantages collapse to 0
        intended_loss, intended_adv = condition_loss(condition="intended_full", **data)
        self.assertTrue((intended_adv == 0).all())

        # Under spectral_legendre_grpo, trajectory dispersion restores non-zero advantages and gradient flow
        spectral_loss, spectral_adv = condition_loss(
            condition="spectral_legendre",
            rewards=data["rewards"],
            logps=logps,
            old_logps=data["old_logps"],
            completion_mask=data["completion_mask"],
            active_rows=data["active_rows"],
        )
        gradient = torch.autograd.grad(spectral_loss, logps)[0]
        self.assertTrue(torch.isfinite(spectral_loss))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)
        self.assertGreater(float(spectral_adv.std()), 0.0)

    def test_entropic_givens_grpo_eliminates_zvf_starvation(self) -> None:
        data = fixture(
            lengths=[2, 3, 4, 8, 16, 24, 32, 40],
            rewards=[0.0] * 8,
            active=[True] * 8,
        )
        logps = data["logps"].detach().requires_grad_(True)
        givens_loss, givens_adv = condition_loss(
            condition="entropic_givens",
            rewards=data["rewards"],
            logps=logps,
            old_logps=data["old_logps"],
            completion_mask=data["completion_mask"],
            active_rows=data["active_rows"],
        )
        gradient = torch.autograd.grad(givens_loss, logps)[0]
        self.assertTrue(torch.isfinite(givens_loss))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(torch.linalg.vector_norm(gradient)), 0.0)
        self.assertGreater(float(givens_adv.std()), 0.0)

    def test_spectral_and_givens_condition_trace(self) -> None:
        data = fixture(
            lengths=[4, 8, 12, 16, 20, 24, 28, 32],
            rewards=[1, 0, 1, 0, 1, 0, 1, 0],
            active=[True] * 8,
        )
        trace_spec = condition_trace(condition="spectral_legendre", **data)
        trace_givens = condition_trace(condition="entropic_givens", **data)
        self.assertEqual(trace_spec.condition, "spectral_legendre")
        self.assertEqual(trace_givens.condition, "entropic_givens")
        self.assertTrue(math.isfinite(trace_spec.loss))
        self.assertTrue(math.isfinite(trace_givens.loss))
        self.assertEqual(len(trace_spec.gradient), data["logps"].numel())
        self.assertEqual(len(trace_givens.gradient), data["logps"].numel())


if __name__ == "__main__":
    unittest.main()

