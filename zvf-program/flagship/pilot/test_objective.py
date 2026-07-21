from __future__ import annotations

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

    def test_all_equal_selected_rewards_produce_zero_norm_and_fail_closed_diagnostic(self) -> None:
        data = fixture(
            lengths=[4] * 8,
            rewards=[0] * 8,
            active=[True] * 8,
        )
        intended = condition_trace(condition="intended_full", **data)
        native = condition_trace(condition="native_trl", **data)
        with self.assertRaisesRegex(ObjectiveContractError, "zero-norm"):
            gradient_diagnostic(intended, native)

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


if __name__ == "__main__":
    unittest.main()
