import math
import unittest

from platform_hybrid.experiments.signal_starvation.metrics import (
    ppo_gate,
    root_metrics,
    sao_gate,
    signal_metrics,
)


class GateTests(unittest.TestCase):
    def test_ppo_sign_and_boundary_semantics(self):
        self.assertEqual(ppo_gate(1.21, 1.0, 0.2), 0)
        self.assertEqual(ppo_gate(1.20, 1.0, 0.2), 1)
        self.assertEqual(ppo_gate(0.79, -1.0, 0.2), 0)
        self.assertEqual(ppo_gate(0.80, -1.0, 0.2), 1)
        self.assertEqual(ppo_gate(1.50, -1.0, 0.2), 1)
        self.assertEqual(ppo_gate(0.50, 1.0, 0.2), 1)
        self.assertEqual(ppo_gate(4.00, 0.0, 0.2), 1)

    def test_sao_interval_is_strict_and_sign_independent(self):
        self.assertEqual(sao_gate(0.80, 0.2, 0.3), 0)
        self.assertEqual(sao_gate(1.30, 0.2, 0.3), 0)
        self.assertEqual(sao_gate(0.81, 0.2, 0.3), 1)
        self.assertEqual(sao_gate(1.29, 0.2, 0.3), 1)


class MetricTests(unittest.TestCase):
    def test_factorization_and_exact_zero(self):
        result = signal_metrics([1.0, 2.0], [1.0, -1.0], [1, 0])
        self.assertAlmostEqual(result["pam"], 2.5)
        self.assertAlmostEqual(result["gsr"], 0.2)
        self.assertAlmostEqual(result["egm"], result["pam"] * result["gsr"])
        self.assertFalse(result["exact_zero_update"])
        zero = signal_metrics([1.0, 1.0], [0.0, 0.0], [1, 1])
        self.assertEqual(zero["pam"], 0.0)
        self.assertEqual(zero["gsr"], 0.0)
        self.assertTrue(zero["exact_zero_update"])

    def test_observation_tokens_are_excluded(self):
        result = signal_metrics([1, 100], [2, 100], [1, 1], [1, 0])
        self.assertEqual(result["n_action_tokens"], 1)
        self.assertEqual(result["pam"], 4.0)

    def test_root_aggregation_is_chunk_invariant(self):
        tokens = [
            {"root_trajectory_id": "r", "chunk_id": "a", "ratio": 1, "advantage": 2, "gate": 1},
            {"root_trajectory_id": "r", "chunk_id": "a", "ratio": 2, "advantage": 1, "gate": 0},
            {"root_trajectory_id": "r", "chunk_id": "b", "ratio": 1, "advantage": -1, "gate": 1},
        ]
        original = root_metrics(tokens)["r"]
        for index, token in enumerate(tokens):
            token["chunk_id"] = f"repartition-{index}"
        repartitioned = root_metrics(tokens)["r"]
        self.assertEqual(original, repartitioned)
        self.assertTrue(math.isclose(original["egm"], original["pam"] * original["gsr"]))


if __name__ == "__main__":
    unittest.main()
