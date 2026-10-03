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

    def test_moved_module_runs_on_its_own_device(self):
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            # No native EmbeddingBag on MPS (torch 2.7), and the CPU
            # fallback flag is read before pytest imports torch, so a
            # moved module cannot run there. CUDA still covers this path.
            self.skipTest("EmbeddingBag is not implemented on MPS")
        if device is None:
            self.skipTest("no non-CPU torch device available")
        torch.manual_seed(0)
        critic = PromptValueCritic().to(device)
        opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
        batch = [[1, 2], [3]]
        out = critic(batch)
        self.assertEqual(out.device.type, device.type)
        loss = train_critic_step(critic, opt, batch, [1.0, 0.0], steps=2)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(critic(batch).device.type, device.type)


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


class TestCriticConvergence(unittest.TestCase):
    def test_learns_prompt_conditional_mean(self):
        means = [-1.0, -0.7, -0.4, -0.1, 0.1, 0.4, 0.7, 1.0]
        noise_sd = 0.3

        def batch(rng, n):
            protos = torch.randint(0, len(means), (n,), generator=rng).tolist()
            ids, rewards = [], []
            for proto in protos:
                length = 8 + (proto * 3) % 9
                ids.append((torch.randint(0, 20, (length,), generator=rng) + proto * 100).tolist())
                rewards.append(means[proto] + torch.randn((), generator=rng).item() * noise_sd)
            return ids, rewards

        torch.manual_seed(7)
        rng = torch.Generator().manual_seed(8)
        critic = PromptValueCritic()
        opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
        _, calibration = batch(torch.Generator().manual_seed(0), 4000)
        mean = sum(calibration) / len(calibration)
        total_var = sum((r - mean) ** 2 for r in calibration) / len(calibration)
        ceiling = 1.0 - noise_sd**2 / total_var
        for _ in range(150):
            batch_ids, rewards = batch(rng, 32)
            train_critic_step(critic, opt, batch_ids, rewards)
        with torch.no_grad():
            test_ids, test_rewards = batch(torch.Generator().manual_seed(1), 2000)
            ev = explained_variance(critic(test_ids).tolist(), test_rewards)
        self.assertGreater(ev, 0.9 * ceiling)

    def test_repeated_fit_is_deterministic(self):
        torch.manual_seed(11)
        first = PromptValueCritic()
        opt = torch.optim.Adam(first.parameters(), lr=1e-2)
        batch = [[10, 11], [210], [320, 321, 322]]
        train_critic_step(first, opt, batch, [1.0, 0.0, 0.5], steps=5)
        torch.manual_seed(11)
        second = PromptValueCritic()
        opt = torch.optim.Adam(second.parameters(), lr=1e-2)
        train_critic_step(second, opt, batch, [1.0, 0.0, 0.5], steps=5)
        self.assertTrue(torch.equal(first(batch), second(batch)))


class TestExplainedVariance(unittest.TestCase):
    def test_perfect_is_one(self):
        self.assertTrue(math.isclose(explained_variance([1.0, 2.0], [1.0, 2.0]), 1.0))

    def test_constant_target_is_zero(self):
        self.assertEqual(explained_variance([0.5, 0.5], [1.0, 1.0]), 0.0)

    def test_near_zero_variance_is_zero(self):
        self.assertEqual(explained_variance([1.0, 1.0 + 1e-9], [1.0, 1.0 + 1e-9]), 0.0)

    def test_small_but_real_variance_still_computes(self):
        ev = explained_variance([0.0, 1e-3], [0.0, 1e-3])
        self.assertTrue(math.isclose(ev, 1.0))

    def test_empty_is_zero(self):
        self.assertEqual(explained_variance([], []), 0.0)

    def test_length_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            explained_variance([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            explained_variance([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
