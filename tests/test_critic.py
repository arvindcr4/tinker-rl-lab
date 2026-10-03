"""Tests for the critic foundation (GAE + value loss)."""

import math
import unittest

import torch

from platform_tinker.tinkerrl.critic import compute_gae_advantages, value_loss_mse


class TestComputeGaeAdvantages(unittest.TestCase):
    def test_hand_computed_trace(self):
        advs, rets = compute_gae_advantages(
            rewards=[1.0, 0.0, 1.0],
            values=[0.5, 0.5, 0.5, 0.0],
            dones=[False, False, True],
            gamma=1.0,
            lam=1.0,
        )
        for actual, want in zip(advs, [1.5, 0.5, 0.5]):
            self.assertTrue(math.isclose(actual, want, rel_tol=1e-9))
        for actual, want in zip(rets, [2.0, 1.0, 1.0]):
            self.assertTrue(math.isclose(actual, want, rel_tol=1e-9))

    def test_lambda_zero_is_td_residual(self):
        advs, _ = compute_gae_advantages(
            rewards=[1.0, 0.0, 1.0],
            values=[0.5, 0.5, 0.5, 0.0],
            dones=[False, False, True],
            gamma=1.0,
            lam=0.0,
        )
        for actual, want in zip(advs, [1.0, 0.0, 0.5]):
            self.assertTrue(math.isclose(actual, want, rel_tol=1e-9))

    def test_done_blocks_bootstrapping(self):
        advs, _ = compute_gae_advantages(
            rewards=[0.0, 0.0],
            values=[0.0, 100.0, 100.0],
            dones=[True, False],
            gamma=1.0,
            lam=1.0,
        )
        self.assertTrue(math.isclose(advs[0], 0.0, abs_tol=1e-9))

    def test_length_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            compute_gae_advantages([1.0], [0.5], [False])
        with self.assertRaises(ValueError):
            compute_gae_advantages([1.0], [0.5, 0.0], [False, True])


class TestValueLossMse(unittest.TestCase):
    def test_half_mse(self):
        loss = value_loss_mse(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 1.5]))
        self.assertTrue(math.isclose(loss.item(), 0.125, rel_tol=1e-6))

    def test_perfect_values_are_zero(self):
        loss = value_loss_mse(torch.tensor([1.0]), torch.tensor([1.0]))
        self.assertEqual(loss.item(), 0.0)

    def test_shape_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            value_loss_mse(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
