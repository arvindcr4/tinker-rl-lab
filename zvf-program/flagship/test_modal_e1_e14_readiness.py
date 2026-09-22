from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .modal_e1_e14 import (
    LANES,
    NON_E11_ACTIONS,
    _e11_configure,
    _readiness_class,
    _receipt_launch_allowed,
    _receipt_ready,
)


class NonE11ReadinessTests(unittest.TestCase):
    def test_every_non_e11_lane_has_an_action(self) -> None:
        self.assertEqual(set(NON_E11_ACTIONS), set(LANES) - {"E11"})
        for action in NON_E11_ACTIONS.values():
            self.assertTrue(action["next_action"])
            self.assertIn("owner", action)

    def test_launch_flags_are_normalized_across_receipt_schemas(self) -> None:
        self.assertTrue(_receipt_launch_allowed({"launch_allowed": True}))
        self.assertTrue(_receipt_launch_allowed({"paid_launch_allowed": True}))
        self.assertTrue(_receipt_launch_allowed({"launch": {"allowed": True}}))
        self.assertTrue(
            _receipt_launch_allowed({"gates": {"authorization": {"launch_authorized": True}}})
        )
        self.assertFalse(_receipt_launch_allowed({"launch": {"allowed": False}}))

    def test_readiness_is_not_hard_coded_to_a_lane(self) -> None:
        self.assertTrue(_receipt_ready({"status": "READY"}))
        self.assertTrue(_receipt_ready({"status": "SCORED"}))
        self.assertEqual(
            _readiness_class(
                adapter_passed=True,
                source_has_model_score=False,
                launch_ready=True,
            ),
            "READY_FOR_FULL_MODAL_EVAL",
        )
        self.assertEqual(
            _readiness_class(
                adapter_passed=True,
                source_has_model_score=True,
                launch_ready=False,
            ),
            "RECORDED_MODEL_RESULT",
        )

    def test_e4_uses_the_recovery_score_receipt(self) -> None:
        self.assertEqual(
            LANES["E1"]["receipt"],
            "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro/seed1818/receipt.json",
        )
        self.assertEqual(
            LANES["E4"]["receipt"],
            "outputs/modal_e1_e14/2026-08-16/e4_recovery_pass16_receipt.json",
        )

    def test_e11_configure_can_find_pinned_iverilog(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "flagship.modal_e1_e14.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            _e11_configure("code-complete-iccad2023", Path(directory))

        env = run.call_args.kwargs["env"]
        self.assertTrue(env["PATH"].startswith("/opt/iverilog/bin:"))


if __name__ == "__main__":
    unittest.main()
