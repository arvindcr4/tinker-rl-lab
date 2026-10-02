"""Tests for the REAL GRPO loss in ``platform_tinker.tinkerrl.grpo``.

Exercises :func:`normalize_rewards` and :func:`make_grpo_loss_fn` imported
from the production module (not local redefinitions) with small synthetic
inputs. End-to-end :func:`run_grpo` is skip-marked: it requires a live
W&B run, Hugging Face auth, and the Tinker service / model download.
"""

import math
import unittest

import torch

from platform_tinker.tinkerrl.grpo import make_grpo_loss_fn, normalize_rewards


class TestNormalizeRewards(unittest.TestCase):
    def test_basic(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        advs = normalize_rewards(rewards)
        mean_adv = sum(advs) / len(advs)
        self.assertTrue(math.isclose(mean_adv, 0.0, abs_tol=1e-7))
        std_adv = (sum((a - mean_adv) ** 2 for a in advs) / len(advs)) ** 0.5
        self.assertTrue(math.isclose(std_adv, 1.0, rel_tol=1e-5))
        self.assertTrue(advs[0] < advs[1] < advs[2] < advs[3] < advs[4])
        self.assertEqual(advs[2], 0.0)

    def test_identical(self):
        advs = normalize_rewards([1.0, 1.0, 1.0, 1.0])
        for a in advs:
            self.assertTrue(math.isclose(a, 0.0, abs_tol=1e-7))

    def test_empty(self):
        self.assertEqual(normalize_rewards([]), [])

    def test_single_element(self):
        advs = normalize_rewards([5.0])
        self.assertTrue(math.isclose(advs[0], 0.0, abs_tol=1e-7))

    def test_epsilon_guards_divide_by_zero(self):
        advs = normalize_rewards([1.0, 1.0 + 1e-9], epsilon=1e-8)
        self.assertFalse(math.isnan(advs[0]))
        self.assertFalse(math.isinf(advs[0]))


class TestMakeGrpoLossFn(unittest.TestCase):
    def test_positive_advantage(self):
        loss_fn = make_grpo_loss_fn([2.0])
        logprobs = torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)
        loss, metrics = loss_fn(None, [logprobs])
        expected_loss = -(2.0) * (-0.8)
        self.assertTrue(math.isclose(loss.item(), expected_loss, rel_tol=1e-5))
        self.assertEqual(metrics["grpo_loss"], loss.item())

    def test_negative_advantage(self):
        loss_fn = make_grpo_loss_fn([-1.0])
        logprobs = torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)
        loss, _ = loss_fn(None, [logprobs])
        expected_loss = -(-1.0) * (-0.8)
        self.assertTrue(math.isclose(loss.item(), expected_loss, rel_tol=1e-5))

    def test_gradients(self):
        loss_fn = make_grpo_loss_fn([2.0, -3.0])
        logprobs1 = torch.tensor([-0.5, -0.5], requires_grad=True)
        logprobs2 = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
        loss, _ = loss_fn(None, [logprobs1, logprobs2])
        loss.backward()
        self.assertTrue(torch.allclose(logprobs1.grad, torch.tensor([-1.0, -1.0])))
        self.assertTrue(torch.allclose(logprobs2.grad, torch.tensor([1.5, 1.5, 1.5])))

    def test_zero_advantage(self):
        loss_fn = make_grpo_loss_fn([0.0])
        logprobs = torch.tensor([-0.5, -0.2], requires_grad=True)
        loss, _ = loss_fn(None, [logprobs])
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.allclose(logprobs.grad, torch.tensor([0.0, 0.0])))

    def test_batch_mean(self):
        loss_fn = make_grpo_loss_fn([1.0, -1.0, 0.0])
        logprobs_list = [torch.tensor([-1.0]), torch.tensor([-2.0]), torch.tensor([-3.0])]
        loss, metrics = loss_fn(None, logprobs_list)
        expected = (1.0 - 2.0 + 0.0) / 3.0
        self.assertTrue(math.isclose(loss.item(), expected, rel_tol=1e-5))
        self.assertEqual(metrics["grpo_loss"], loss.item())

    def test_empty_group(self):
        loss_fn = make_grpo_loss_fn([])
        loss, metrics = loss_fn(None, [])
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["grpo_loss"], 0.0)

    @unittest.skip(
        "run_grpo needs live W&B + Hugging Face auth + Tinker service/model "
        "download; not runnable as a fast unit test"
    )
    def test_run_grpo_end_to_end(self):
        pass


if __name__ == "__main__":
    unittest.main()
