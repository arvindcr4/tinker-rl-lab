from __future__ import annotations

import unittest

import torch

from pilot.objective import condition_trace
from s1.reference import objective_trace
from s1.trl_adapter import trl_trace


class ExactStackDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lengths = [2, 3, 4, 8, 16, 24, 32, 40]
        cls.rewards = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.float64)
        cls.active = torch.tensor(
            [True, True, True, False, True, True, False, True], dtype=torch.bool
        )
        width = max(cls.lengths)
        cls.raw_mask = torch.zeros((8, width), dtype=torch.float64)
        for index, length in enumerate(cls.lengths):
            cls.raw_mask[index, :length] = 1.0
        cls.selected_mask = cls.raw_mask * cls.active[:, None]
        cls.old_logps = torch.zeros((8, width), dtype=torch.float64)
        offsets = torch.linspace(-0.08, 0.08, steps=8, dtype=torch.float64)[:, None]
        positions = torch.linspace(0.2, 1.0, steps=width, dtype=torch.float64)[None, :]
        cls.logps = cls.old_logps + offsets * positions
        cls.selected_indices = tuple(int(index) for index in torch.where(cls.active)[0])

    def test_intended_condition_matches_canonical_s1_dapo(self) -> None:
        expected = objective_trace(
            arm="dapo",
            rewards=self.rewards[None, :],
            logps=self.logps,
            old_logps=self.old_logps,
            mask=self.raw_mask,
            selected_indices=self.selected_indices,
        )
        actual = condition_trace(
            condition="intended_full",
            rewards=self.rewards,
            logps=self.logps,
            old_logps=self.old_logps,
            completion_mask=self.selected_mask,
            active_rows=self.active,
        )
        torch.testing.assert_close(
            torch.tensor(actual.advantages, dtype=torch.float64),
            expected.advantages,
            rtol=1e-6,
            atol=1e-8,
        )
        torch.testing.assert_close(
            torch.tensor(actual.gradient, dtype=torch.float64),
            expected.gradient,
            rtol=1e-6,
            atol=1e-8,
        )
        self.assertAlmostEqual(actual.loss, float(expected.loss), places=8)

    def test_native_condition_matches_pinned_trl_dapo(self) -> None:
        expected, config, provenance = trl_trace(
            arm="dapo",
            rewards=self.rewards[None, :],
            logps=self.logps,
            old_logps=self.old_logps,
            mask=self.raw_mask,
            selected_indices=self.selected_indices,
        )
        self.assertEqual(config.loss_type, "dapo")
        self.assertEqual(provenance.trl_version, "1.2.0")
        actual = condition_trace(
            condition="native_trl",
            rewards=self.rewards,
            logps=self.logps,
            old_logps=self.old_logps,
            completion_mask=self.selected_mask,
            active_rows=self.active,
        )
        torch.testing.assert_close(
            torch.tensor(actual.advantages, dtype=torch.float64),
            expected.advantages,
            rtol=1e-6,
            atol=1e-8,
        )
        torch.testing.assert_close(
            torch.tensor(actual.gradient, dtype=torch.float64),
            expected.gradient,
            rtol=1e-6,
            atol=1e-8,
        )
        self.assertAlmostEqual(actual.loss, float(expected.loss), places=8)


if __name__ == "__main__":
    unittest.main()
