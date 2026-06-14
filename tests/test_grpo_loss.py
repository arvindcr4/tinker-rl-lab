import unittest
import math
import torch

def normalize_rewards(rewards, epsilon=1e-8):
    n = len(rewards)
    if n == 0:
        return []
    mean_r = sum(rewards) / n
    std_r = (sum((r - mean_r) ** 2 for r in rewards) / n) ** 0.5 + epsilon
    return [(r - mean_r) / std_r for r in rewards]

def compute_grpo_loss(logprobs_list, advantages):
    losses = []
    for i, logprobs in enumerate(logprobs_list):
        adv = advantages[i]
        losses.append(-adv * logprobs.sum())
    if not losses:
        return torch.tensor(0.0), {"grpo_loss": 0.0}
    loss = torch.stack(losses).mean()
    return loss, {"grpo_loss": loss.item()}

class TestGRPOLoss(unittest.TestCase):

    def test_normalize_rewards_basic(self):
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        advs = normalize_rewards(rewards)
        mean_adv = sum(advs) / len(advs)
        self.assertTrue(math.isclose(mean_adv, 0.0, abs_tol=1e-7))
        std_adv = (sum((a - mean_adv) ** 2 for a in advs) / len(advs)) ** 0.5
        self.assertTrue(math.isclose(std_adv, 1.0, rel_tol=1e-5))
        self.assertTrue(advs[0] < advs[1] < advs[2] < advs[3] < advs[4])
        self.assertEqual(advs[2], 0.0)

    def test_normalize_rewards_identical(self):
        rewards = [1.0, 1.0, 1.0, 1.0]
        advs = normalize_rewards(rewards)
        for a in advs:
            self.assertTrue(math.isclose(a, 0.0, abs_tol=1e-7))

    def test_normalize_rewards_empty(self):
        self.assertEqual(normalize_rewards([]), [])

    def test_normalize_rewards_single_element(self):
        rewards = [5.0]
        advs = normalize_rewards(rewards)
        self.assertTrue(math.isclose(advs[0], 0.0, abs_tol=1e-7))

    def test_normalize_rewards_epsilon(self):
        rewards = [1.0, 1.0 + 1e-9]
        advs = normalize_rewards(rewards, epsilon=1e-8)
        self.assertFalse(math.isnan(advs[0]))
        self.assertFalse(math.isinf(advs[0]))

    def test_compute_grpo_loss_positive_advantage(self):
        logprobs = torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)
        advantages = [2.0]
        loss, metrics = compute_grpo_loss([logprobs], advantages)
        expected_loss = -(2.0) * (-0.8)
        self.assertTrue(math.isclose(loss.item(), expected_loss, rel_tol=1e-5))
        self.assertEqual(metrics["grpo_loss"], loss.item())

    def test_compute_grpo_loss_negative_advantage(self):
        logprobs = torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)
        advantages = [-1.0]
        loss, metrics = compute_grpo_loss([logprobs], advantages)
        expected_loss = -(-1.0) * (-0.8)
        self.assertTrue(math.isclose(loss.item(), expected_loss, rel_tol=1e-5))

    def test_compute_grpo_loss_gradients(self):
        logprobs1 = torch.tensor([-0.5, -0.5], requires_grad=True)
        logprobs2 = torch.tensor([-1.0, -1.0, -1.0], requires_grad=True)
        advantages = [2.0, -3.0]
        loss, _ = compute_grpo_loss([logprobs1, logprobs2], advantages)
        loss.backward()
        self.assertTrue(torch.allclose(logprobs1.grad, torch.tensor([-1.0, -1.0])))
        self.assertTrue(torch.allclose(logprobs2.grad, torch.tensor([1.5, 1.5, 1.5])))

    def test_compute_grpo_loss_zero_advantage(self):
        logprobs = torch.tensor([-0.5, -0.2], requires_grad=True)
        advantages = [0.0]
        loss, _ = compute_grpo_loss([logprobs], advantages)
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.allclose(logprobs.grad, torch.tensor([0.0, 0.0])))

    def test_compute_grpo_loss_batch(self):
        logprobs_list = [
            torch.tensor([-1.0]),
            torch.tensor([-2.0]),
            torch.tensor([-3.0])
        ]
        advantages = [1.0, -1.0, 0.0]
        loss, metrics = compute_grpo_loss(logprobs_list, advantages)
        expected = (1.0 - 2.0 + 0.0) / 3.0
        self.assertTrue(math.isclose(loss.item(), expected, rel_tol=1e-5))
        self.assertEqual(metrics["grpo_loss"], loss.item())

if __name__ == "__main__":
    unittest.main()
