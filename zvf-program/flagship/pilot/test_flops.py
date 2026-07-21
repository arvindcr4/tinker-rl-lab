from __future__ import annotations

import unittest

import torch

from pilot.flops import (
    PROFILED_STEPS,
    FlopAccountingError,
    TorchPhaseProfiler,
    TrainingFlopLedger,
)


class FlopAccountingTests(unittest.TestCase):
    def test_cpu_profiler_reports_positive_matrix_multiplication_flops(self) -> None:
        profiler = TorchPhaseProfiler(torch, enabled=True)
        left = torch.ones((8, 16))
        right = torch.ones((16, 4))
        with profiler("policy_forward"):
            output = left @ right
        self.assertEqual(output.shape, (8, 4))
        self.assertGreater(profiler.phase_flops["policy_forward"], 0)

    def test_disabled_profiler_is_a_noop(self) -> None:
        profiler = TorchPhaseProfiler(torch, enabled=False)
        with profiler("policy_forward"):
            _ = torch.ones(2) + 1
        self.assertEqual(profiler.phase_flops, {})

    def test_ledger_requires_exact_profiled_steps_and_extrapolates(self) -> None:
        ledger = TrainingFlopLedger()
        for step in range(1, 101):
            phase_flops = None
            if step in PROFILED_STEPS:
                phase_flops = {
                    "policy_forward": 100.0,
                    "optimizer_backward": 200.0,
                    "diagnostic_backward": 300.0,
                }
            ledger.add_step(
                step=step,
                active_tokens=64,
                padded_tokens=80,
                phase_flops=phase_flops,
            )
        record = ledger.final_record()
        self.assertEqual(record["profiled_steps"], list(PROFILED_STEPS))
        self.assertAlmostEqual(record["extrapolation_scale"], 100 / len(PROFILED_STEPS))
        self.assertAlmostEqual(record["policy_forward_flops"], 100.0 * 100)
        self.assertAlmostEqual(record["optimizer_backward_flops"], 200.0 * 100)
        self.assertAlmostEqual(record["diagnostic_backward_flops"], 300.0 * 100)
        restored = TrainingFlopLedger.from_record(record)
        self.assertEqual(restored.profiled_steps, list(PROFILED_STEPS))
        self.assertEqual(restored.total_padded_tokens, ledger.total_padded_tokens)

    def test_ledger_fails_closed_on_missing_phase_or_step(self) -> None:
        ledger = TrainingFlopLedger()
        with self.assertRaisesRegex(FlopAccountingError, "missing phases"):
            ledger.add_step(
                step=1,
                active_tokens=64,
                padded_tokens=80,
                phase_flops={"policy_forward": 1.0},
            )
        valid = TrainingFlopLedger()
        valid.add_step(
            step=1,
            active_tokens=64,
            padded_tokens=80,
            phase_flops={
                "policy_forward": 1.0,
                "optimizer_backward": 1.0,
                "diagnostic_backward": 1.0,
            },
        )
        with self.assertRaisesRegex(FlopAccountingError, "profiled steps mismatch"):
            valid.final_record()
        partial = valid.record(require_complete=False)
        self.assertEqual(partial["profiled_steps"], [1])


if __name__ == "__main__":
    unittest.main()
