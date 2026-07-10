"""Tests for the consolidated ``tinkerrl.grpo`` module."""

import unittest
from unittest.mock import MagicMock

from platform_tinker.tinkerrl.grpo import (
    GRPOConfig,
    GRPORunResult,
    InMemoryDataset,
    MathReward,
    ToolCallReward,
    TrainingExample,
    make_grpo_loss_fn,
    make_synthetic_math_dataset,
    make_synthetic_tool_use_dataset,
    normalize_rewards,
)


class TestNormalizeRewards(unittest.TestCase):
    def test_basic(self):
        advs = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sum(advs) / len(advs)
        self.assertAlmostEqual(mean, 0.0, places=7)
        std = (sum((a - mean) ** 2 for a in advs) / len(advs)) ** 0.5
        self.assertAlmostEqual(std, 1.0, places=5)

    def test_identical(self):
        advs = normalize_rewards([3.0, 3.0, 3.0])
        for a in advs:
            self.assertAlmostEqual(a, 0.0, places=7)

    def test_empty(self):
        self.assertEqual(normalize_rewards([]), [])

    def test_single(self):
        advs = normalize_rewards([42.0])
        self.assertAlmostEqual(advs[0], 0.0, places=7)

    def test_monotonic(self):
        advs = normalize_rewards([1.0, 2.0, 3.0, 4.0, 5.0])
        for i in range(len(advs) - 1):
            self.assertLess(advs[i], advs[i + 1])


class TestMakeGrpoLossFn(unittest.TestCase):
    def test_positive_advantage(self):
        import torch

        loss_fn = make_grpo_loss_fn([2.0])
        logprobs = [torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)]
        loss, metrics = loss_fn(None, logprobs)
        expected = -(2.0) * (-0.8)
        self.assertAlmostEqual(loss.item(), expected, places=5)
        self.assertEqual(metrics["grpo_loss"], loss.item())

    def test_negative_advantage(self):
        import torch

        loss_fn = make_grpo_loss_fn([-1.0])
        logprobs = [torch.tensor([-0.5, -0.2, -0.1], requires_grad=True)]
        loss, _ = loss_fn(None, logprobs)
        expected = -(-1.0) * (-0.8)
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_zero_advantage(self):
        import torch

        loss_fn = make_grpo_loss_fn([0.0])
        logprobs = [torch.tensor([-0.5, -0.2], requires_grad=True)]
        loss, _ = loss_fn(None, logprobs)
        self.assertEqual(loss.item(), 0.0)

    def test_batch(self):
        import torch

        loss_fn = make_grpo_loss_fn([1.0, -1.0, 0.0])
        logprobs = [
            torch.tensor([-1.0]),
            torch.tensor([-2.0]),
            torch.tensor([-3.0]),
        ]
        loss, _ = loss_fn(None, logprobs)
        expected = (1.0 - 2.0 + 0.0) / 3.0
        self.assertAlmostEqual(loss.item(), expected, places=5)

    def test_empty(self):
        import torch

        loss_fn = make_grpo_loss_fn([])
        loss, metrics = loss_fn(None, [])
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["grpo_loss"], 0.0)


class TestGRPOConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GRPOConfig(name="test")
        self.assertEqual(cfg.model, "Qwen/Qwen3-8B")
        self.assertEqual(cfg.lora_rank, 32)
        self.assertEqual(cfg.steps, 200)
        self.assertEqual(cfg.group_size, 8)
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.lr, 3e-5)
        self.assertEqual(cfg.temperature, 0.8)
        self.assertEqual(cfg.top_p, 0.95)
        self.assertEqual(cfg.max_prompt_tokens, 1024)
        self.assertEqual(cfg.max_response_tokens, 512)
        self.assertIsNone(cfg.save_every)
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.num_seeds, 1)
        self.assertFalse(cfg.evaluate_heldout)

    def test_effective_save_every_explicit(self):
        cfg = GRPOConfig(name="t", save_every=10)
        self.assertEqual(cfg.effective_save_every(), 10)

    def test_effective_save_every_computed(self):
        cfg = GRPOConfig(name="t", steps=200)
        self.assertEqual(cfg.effective_save_every(), 50)

    def test_effective_save_every_minimum(self):
        cfg = GRPOConfig(name="t", steps=10)
        self.assertEqual(cfg.effective_save_every(), 10)


class TestTrainingExample(unittest.TestCase):
    def test_defaults(self):
        ex = TrainingExample(prompt="hello")
        self.assertEqual(ex.prompt, "hello")
        self.assertIsNone(ex.target)
        self.assertEqual(ex.metadata, {})

    def test_with_target(self):
        ex = TrainingExample(prompt="q", target={"tool": "calc"})
        self.assertEqual(ex.target, {"tool": "calc"})


class TestInMemoryDataset(unittest.TestCase):
    def test_train_and_test(self):
        train = [TrainingExample(prompt="a"), TrainingExample(prompt="b")]
        test = [TrainingExample(prompt="c")]
        ds = InMemoryDataset(train=train, test=test)
        self.assertEqual(ds.train_examples(), train)
        self.assertEqual(ds.test_examples(), test)

    def test_empty_test(self):
        ds = InMemoryDataset(train=[TrainingExample(prompt="x")])
        self.assertEqual(ds.test_examples(), ())


class TestSyntheticToolUseDataset(unittest.TestCase):
    def test_non_empty(self):
        ds = make_synthetic_tool_use_dataset()
        self.assertGreater(len(ds.train_examples()), 0)

    def test_target_structure(self):
        ds = make_synthetic_tool_use_dataset()
        ex = ds.train_examples()[0]
        self.assertIn("tool", ex.target)
        self.assertIn("arguments", ex.target)

    def test_prompt_format(self):
        ds = make_synthetic_tool_use_dataset()
        ex = ds.train_examples()[0]
        self.assertIn("<|im_start|>", ex.prompt)
        self.assertIn("Available tools:", ex.prompt)


class TestSyntheticMathDataset(unittest.TestCase):
    def test_non_empty(self):
        ds = make_synthetic_math_dataset()
        self.assertGreater(len(ds.train_examples()), 0)

    def test_target_is_string(self):
        ds = make_synthetic_math_dataset()
        ex = ds.train_examples()[0]
        self.assertIsInstance(ex.target, str)

    def test_prompt_format(self):
        ds = make_synthetic_math_dataset()
        ex = ds.train_examples()[0]
        self.assertIn("<|im_start|>", ex.prompt)
        self.assertIn("\\boxed{}", ex.prompt)


class TestToolCallReward(unittest.TestCase):
    def _ex(self, tool, arguments):
        return TrainingExample(prompt="q", target={"tool": tool, "arguments": arguments})

    def test_perfect_json(self):
        r = ToolCallReward()
        resp = '{"tool": "calculator", "arguments": {"expression": "1+1"}}'
        score = r.score(resp, self._ex("calculator", {"expression": "1+1"}))
        self.assertAlmostEqual(score, 1.0)

    def test_correct_tool_wrong_args(self):
        r = ToolCallReward()
        resp = '{"tool": "calculator", "arguments": {"wrong": "x"}}'
        score = r.score(resp, self._ex("calculator", {"expression": "1+1"}))
        self.assertAlmostEqual(score, 0.7)

    def test_valid_json_wrong_tool(self):
        r = ToolCallReward()
        resp = '{"tool": "other_tool", "arguments": {}}'
        score = r.score(resp, self._ex("calculator", {"expression": "1+1"}))
        self.assertAlmostEqual(score, 0.3)

    def test_no_json(self):
        r = ToolCallReward()
        self.assertAlmostEqual(r.score("no json here", self._ex("calc", {})), 0.0)

    def test_invalid_json(self):
        r = ToolCallReward()
        # "{bad json" has no closing brace, so regex finds no JSON object => 0.0
        self.assertAlmostEqual(r.score("{bad json", self._ex("calc", {})), 0.0)

    def test_malformed_json_with_brace(self):
        r = ToolCallReward()
        # "{bad json}" has braces but json.loads fails => 0.1
        self.assertAlmostEqual(r.score("{bad json}", self._ex("calc", {})), 0.1)


class TestMathReward(unittest.TestCase):
    def _ex(self, answer):
        return TrainingExample(prompt="q", target=answer)

    def test_boxed_exact(self):
        r = MathReward()
        score = r.score("the answer is \\boxed{42}", self._ex("42"))
        self.assertAlmostEqual(score, 1.0)

    def test_boxed_float_match(self):
        r = MathReward()
        score = r.score("\\boxed{3.14}", self._ex("3.14"))
        self.assertAlmostEqual(score, 1.0)

    def test_boxed_wrong(self):
        r = MathReward()
        score = r.score("\\boxed{99}", self._ex("42"))
        self.assertAlmostEqual(score, 0.3)

    def test_standalone_number_match(self):
        r = MathReward()
        # Standalone number match (not last-number fallback) => 0.5
        score = r.score("I think the answer is 42", self._ex("42"))
        self.assertAlmostEqual(score, 0.5)

    def test_last_number_fallback(self):
        r = MathReward()
        # "42" appears standalone at end; the \b match fires first => 0.5
        score = r.score("the answer is 42", self._ex("42"))
        self.assertAlmostEqual(score, 0.5)

    def test_no_match(self):
        r = MathReward()
        score = r.score("I don't know", self._ex("42"))
        self.assertAlmostEqual(score, 0.0)

    def test_partial_math_chars(self):
        r = MathReward()
        score = r.score("1 + 2 = ?", self._ex("42"))
        self.assertAlmostEqual(score, 0.1)


class TestGRPORunResult(unittest.TestCase):
    def test_defaults(self):
        r = GRPORunResult(seed=0)
        self.assertEqual(r.seed, 0)
        self.assertIsNone(r.run_id)
        self.assertIsNone(r.sampler_path)
        self.assertEqual(r.reward_trace, [])
        self.assertEqual(r.avg_first5, 0.0)
        self.assertEqual(r.avg_last10, 0.0)
        self.assertEqual(r.peak_reward, 0.0)
        self.assertEqual(r.zero_loss_steps, 0)
        self.assertEqual(r.zero_reward_steps, 0)
        self.assertIsNone(r.heldout_reward)


if __name__ == "__main__":
    unittest.main()
