from __future__ import annotations

import unittest

from .modal_e1_e14 import (
    LANES,
    NON_E11_ACTIONS,
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
            _receipt_launch_allowed(
                {"gates": {"authorization": {"launch_authorized": True}}}
            )
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


if __name__ == "__main__":
    unittest.main()
