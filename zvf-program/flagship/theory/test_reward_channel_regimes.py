from __future__ import annotations

import unittest

from .reward_channel_regimes import E1_SOURCE_SHA256, regime_receipt, verify_e1_source


class RewardChannelRegimeTests(unittest.TestCase):
    def test_frozen_e1_source_hash_is_verified(self) -> None:
        path = verify_e1_source()
        self.assertTrue(path.name == "e1_grpo_confirmatory.py")
        self.assertEqual(E1_SOURCE_SHA256, "986811e3e78fe86ffcbede4a98599ada167ff1975b3341eef391eb2b2e7fe8c6")

    def test_primary_reward_observations_are_identical(self) -> None:
        receipt = regime_receipt()
        clean = receipt["clean_hard"]
        broken = receipt["silent_marker_mismatch"]
        assert isinstance(clean, dict) and isinstance(broken, dict)
        self.assertEqual(clean["primary_rewards"], broken["primary_rewards"])
        self.assertEqual(clean["primary_rewards"], [0.0] * 8)
        self.assertEqual(clean["telemetry"], broken["telemetry"])

    def test_same_path_known_correct_calibration_resolves_marker_mismatch(self) -> None:
        receipt = regime_receipt()
        clean = receipt["clean_hard"]
        broken = receipt["silent_marker_mismatch"]
        assert isinstance(clean, dict) and isinstance(broken, dict)
        self.assertEqual(clean["calibration_reward"], 1.0)
        self.assertEqual(broken["calibration_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
