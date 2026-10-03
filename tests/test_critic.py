"""Tests for the critic (value head + GAE + value loss)."""

import math
import unittest

import torch

from platform_tinker.tinkerrl.critic import (
    PromptValueCritic,
    compute_gae_advantages,
    explained_variance,
    train_critic_step,
    value_loss_mse,
)


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


class TestPromptValueCritic(unittest.TestCase):
    def test_forward_shape(self):
        critic = PromptValueCritic()
        out = critic([[1, 2, 3], [4]])
        self.assertEqual(tuple(out.shape), (2,))

    def test_seeded_init_is_deterministic(self):
        torch.manual_seed(0)
        first = PromptValueCritic()([[1, 2, 3]])
        torch.manual_seed(0)
        second = PromptValueCritic()([[1, 2, 3]])
        self.assertTrue(torch.equal(first, second))

    def test_empty_prompt_does_not_crash(self):
        critic = PromptValueCritic()
        out = critic([[]])
        self.assertEqual(tuple(out.shape), (1,))
        self.assertTrue(torch.isfinite(out).all())

    def test_empty_batch_is_length_zero(self):
        critic = PromptValueCritic()
        out = critic([])
        self.assertEqual(tuple(out.shape), (0,))

    def test_large_ids_hash_into_buckets(self):
        critic = PromptValueCritic(vocab_buckets=64)
        out = critic([[10**9, 10**9 + 1]])
        self.assertEqual(tuple(out.shape), (1,))
        self.assertTrue(torch.isfinite(out).all())


class TestTrainCriticStep(unittest.TestCase):
    def test_fit_reduces_loss(self):
        torch.manual_seed(0)
        critic = PromptValueCritic()
        opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
        batch = [[1, 2], [3], [4, 5, 6]]
        before = value_loss_mse(critic(batch), torch.tensor([1.0, 0.0, 1.0])).item()
        train_critic_step(critic, opt, batch, [1.0, 0.0, 1.0], steps=50)
        after = value_loss_mse(critic(batch), torch.tensor([1.0, 0.0, 1.0])).item()
        self.assertLess(after, before)

    def test_empty_batch_is_noop(self):
        critic = PromptValueCritic()
        opt = torch.optim.Adam(critic.parameters())
        self.assertEqual(train_critic_step(critic, opt, [], [], steps=4), 0.0)

    def test_target_count_mismatch_fails_closed(self):
        critic = PromptValueCritic()
        opt = torch.optim.Adam(critic.parameters())
        with self.assertRaises(ValueError):
            train_critic_step(critic, opt, [[1], [2]], [1.0], steps=1)

    def test_save_load_round_trip(self):
        import tempfile
        from pathlib import Path

        torch.manual_seed(0)
        critic = PromptValueCritic()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "critic.pt"
            torch.save(critic.state_dict(), path)
            restored = PromptValueCritic()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        batch = [[1, 2], [3]]
        self.assertTrue(torch.equal(critic(batch), restored(batch)))


class TestExplainedVariance(unittest.TestCase):
    def test_perfect_is_one(self):
        self.assertTrue(math.isclose(explained_variance([1.0, 2.0], [1.0, 2.0]), 1.0))

    def test_constant_target_is_zero(self):
        self.assertEqual(explained_variance([0.5, 0.5], [1.0, 1.0]), 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(explained_variance([], []), 0.0)

    def test_length_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            explained_variance([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            explained_variance([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
