"""Tests for the REAL GRPO loss in ``platform_tinker.tinkerrl.grpo``.

Exercises :func:`normalize_rewards` and :func:`make_grpo_loss_fn` imported
from the production module (not local redefinitions) with small synthetic
inputs. End-to-end :func:`run_grpo` is skip-marked: it requires a live
W&B run, Hugging Face auth, and the Tinker service / model download.
"""

import math
import unittest

import torch

from platform_tinker.tinkerrl.grpo import (
    make_grpo_loss_fn,
    make_gspo_loss_fn,
    normalize_advantages_global,
    normalize_rewards,
)


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

    def test_unbiased_returns_centered_unscaled(self):
        advs = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0], unbiased=True)
        self.assertEqual(advs, [-2.0, -1.0, 0.0, 1.0, 2.0])
        mean_adv = sum(advs) / len(advs)
        self.assertTrue(math.isclose(mean_adv, 0.0, abs_tol=1e-9))

    def test_unbiased_near_uniform_does_not_explode(self):
        advs = normalize_rewards([1.0, 1.0 + 1e-9], unbiased=True)
        for a in advs:
            self.assertLess(abs(a), 1e-6)

    def test_biased_mode_unchanged_regression_pin(self):
        advs = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = [-1.41421356, -0.70710678, 0.0, 0.70710678, 1.41421356]
        self.assertEqual(len(advs), len(expected))
        for actual, want in zip(advs, expected):
            self.assertTrue(math.isclose(actual, want, rel_tol=1e-5))


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

    def test_nll_aux_disabled_by_default(self):
        loss_fn = make_grpo_loss_fn([2.0])
        logprobs = torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)
        loss, metrics = loss_fn(None, [logprobs])
        self.assertTrue(math.isclose(loss.item(), 1.6, rel_tol=1e-5))
        self.assertEqual(metrics["grpo_loss"], loss.item())
        self.assertEqual(metrics["nll_loss"], 0.0)

    def test_nll_aux_correct_only(self):
        loss_fn = make_grpo_loss_fn([0.0, 0.0], nll_mask=[True, False], nll_coef=1.0)
        lp0 = torch.tensor([-0.5, -0.5])
        lp1 = torch.tensor([-9.0])
        loss, metrics = loss_fn(None, [lp0, lp1])
        self.assertTrue(math.isclose(metrics["nll_loss"], 0.5, rel_tol=1e-5))
        self.assertTrue(math.isclose(loss.item(), 0.5, rel_tol=1e-5))

    def test_nll_aux_empty_set_is_zero(self):
        loss_fn = make_grpo_loss_fn([2.0], nll_mask=[False], nll_coef=1.0)
        logprobs = torch.tensor([-0.5, -0.2, -0.1])
        loss, metrics = loss_fn(None, [logprobs])
        self.assertEqual(metrics["nll_loss"], 0.0)
        self.assertTrue(math.isclose(loss.item(), 1.6, rel_tol=1e-5))

    def test_nll_aux_token_mean_not_sum(self):
        loss_fn = make_grpo_loss_fn([0.0, 0.0], nll_mask=[True, True], nll_coef=1.0)
        lp_short = torch.tensor([-0.5, -0.5])
        lp_long = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        loss, metrics = loss_fn(None, [lp_short, lp_long])
        self.assertTrue(math.isclose(metrics["nll_loss"], 0.75, rel_tol=1e-5))
        self.assertTrue(math.isclose(loss.item(), 0.75, rel_tol=1e-5))

    def test_nll_aux_gradients(self):
        loss_fn = make_grpo_loss_fn([0.0, 0.0], nll_mask=[True, False], nll_coef=0.5)
        lp0 = torch.tensor([-0.5, -0.5], requires_grad=True)
        lp1 = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
        loss, _ = loss_fn(None, [lp0, lp1])
        loss.backward()
        self.assertTrue(torch.allclose(lp0.grad, torch.tensor([-0.25, -0.25])))
        self.assertTrue(torch.allclose(lp1.grad, torch.tensor([0.0, 0.0, 0.0])))

    def test_nll_mask_mismatch_fails_closed(self):
        loss_fn = make_grpo_loss_fn([1.0], nll_mask=[True, False], nll_coef=1.0)
        with self.assertRaises(ValueError):
            loss_fn(None, [torch.tensor([-0.5])])

    @unittest.skip(
        "run_grpo needs live W&B + Hugging Face auth + Tinker service/model "
        "download; not runnable as a fast unit test"
    )
    def test_run_grpo_end_to_end(self):
        pass


class TestMakeGspoLossFn(unittest.TestCase):
    def test_first_epoch_ratio_one_with_gradients(self):
        loss_fn = make_gspo_loss_fn([2.0])
        logprobs = torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)
        loss, metrics = loss_fn(None, [logprobs])
        self.assertTrue(math.isclose(loss.item(), -2.0, rel_tol=1e-5))
        self.assertEqual(metrics["gspo_loss"], loss.item())
        self.assertEqual(metrics["gspo_clip_frac"], 0.0)
        loss.backward()
        for g in logprobs.grad:
            self.assertTrue(math.isclose(g.item(), -2.0 / 3.0, rel_tol=1e-5))

    def test_clipping_positive_advantage(self):
        loss_fn = make_gspo_loss_fn([2.0], old_logprobs=[torch.tensor([-1.0, -1.0])])
        logprobs = torch.tensor([-0.99, -0.99])
        loss, metrics = loss_fn(None, [logprobs])
        # s = e^0.01 ≈ 1.01005 clips to 1.0004; min picks the clipped term.
        self.assertTrue(math.isclose(loss.item(), -2.0008, rel_tol=1e-4))
        self.assertEqual(metrics["gspo_clip_frac"], 1.0)

    def test_clipping_negative_advantage_picks_unclipped(self):
        loss_fn = make_gspo_loss_fn([-2.0], old_logprobs=[torch.tensor([-1.0, -1.0])])
        logprobs = torch.tensor([-0.99, -0.99])
        loss, metrics = loss_fn(None, [logprobs])
        self.assertTrue(math.isclose(loss.item(), 2.0201003, rel_tol=1e-4))
        self.assertEqual(metrics["gspo_clip_frac"], 1.0)

    def test_small_drift_unclipped(self):
        loss_fn = make_gspo_loss_fn([2.0], old_logprobs=[torch.tensor([-1.0, -1.0])])
        logprobs = torch.tensor([-1.0 + 1e-5, -1.0 + 1e-5])
        loss, metrics = loss_fn(None, [logprobs])
        self.assertTrue(math.isclose(loss.item(), -2.0 * math.exp(1e-5), rel_tol=1e-5))
        self.assertEqual(metrics["gspo_clip_frac"], 0.0)

    def test_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            make_gspo_loss_fn([1.0, 2.0])(None, [torch.tensor([-0.5])])
        with self.assertRaises(ValueError):
            make_gspo_loss_fn([1.0], old_logprobs=[torch.tensor([-0.5])] * 2)(
                None, [torch.tensor([-0.5])]
            )
        with self.assertRaises(ValueError):
            make_gspo_loss_fn([1.0], old_logprobs=[torch.tensor([-0.5, -0.5])])(
                None, [torch.tensor([-0.5])]
            )
        with self.assertRaises(ValueError):
            make_gspo_loss_fn([1.0])(None, [torch.tensor([])])

    def test_stash_captures_detached_first_call_only(self):
        stash = []
        loss_fn = make_gspo_loss_fn([1.0], stash=stash)
        lp = torch.tensor([-0.5, -0.2], requires_grad=True)
        loss_fn(None, [lp])
        loss_fn(None, [torch.tensor([-9.0, -9.0])])
        self.assertEqual(len(stash), 1)
        self.assertFalse(stash[0].requires_grad)
        self.assertTrue(torch.allclose(stash[0], torch.tensor([-0.5, -0.2])))

    def test_empty_group(self):
        loss_fn = make_gspo_loss_fn([])
        loss, metrics = loss_fn(None, [])
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["gspo_loss"], 0.0)
        self.assertEqual(metrics["gspo_clip_frac"], 0.0)


class TestNormalizeAdvantagesGlobal(unittest.TestCase):
    def test_batch_mean_zero_std_one(self):
        advs = normalize_advantages_global([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        mean_adv = sum(advs) / len(advs)
        self.assertTrue(math.isclose(mean_adv, 0.0, abs_tol=1e-7))
        std_adv = (sum((a - mean_adv) ** 2 for a in advs) / len(advs)) ** 0.5
        self.assertTrue(math.isclose(std_adv, 1.0, rel_tol=1e-5))

    def test_differs_from_per_group(self):
        batch = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
        per_group = normalize_rewards(batch[:3]) + normalize_rewards(batch[3:])
        batch_global = normalize_advantages_global(batch)
        self.assertFalse(
            all(math.isclose(a, b, rel_tol=1e-5) for a, b in zip(per_group, batch_global))
        )

    def test_unbiased_is_batch_centered(self):
        advs = normalize_advantages_global([1.0, 2.0, 3.0, 4.0], unbiased=True)
        self.assertEqual(advs, [-1.5, -0.5, 0.5, 1.5])

    def test_unbiased_near_uniform_does_not_explode(self):
        advs = normalize_advantages_global([1.0, 1.0 + 1e-9], unbiased=True)
        for a in advs:
            self.assertLess(abs(a), 1e-6)

    def test_exclude_leaves_stats_but_keeps_alignment(self):
        advs = normalize_advantages_global([0.0, 1.0, 1.0], exclude=[False, False, True])
        self.assertEqual(len(advs), 3)
        self.assertTrue(math.isclose(advs[0], -1.0, rel_tol=1e-5))
        self.assertTrue(math.isclose(advs[1], 1.0, rel_tol=1e-5))
        self.assertTrue(math.isclose(advs[2], 1.0, rel_tol=1e-5))

    def test_all_excluded_falls_back_to_full_vector(self):
        advs = normalize_advantages_global([0.0, 2.0], exclude=[True, True])
        self.assertEqual(len(advs), 2)
        self.assertTrue(math.isclose(advs[0], -1.0, rel_tol=1e-5))
        self.assertTrue(math.isclose(advs[1], 1.0, rel_tol=1e-5))

    def test_exclude_mismatch_fails_closed(self):
        with self.assertRaises(ValueError):
            normalize_advantages_global([1.0, 2.0], exclude=[False])

    def test_empty(self):
        self.assertEqual(normalize_advantages_global([]), [])


if __name__ == "__main__":
    unittest.main()
