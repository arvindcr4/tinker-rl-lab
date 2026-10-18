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
        assert ppo_gate(1.21, 1.0, 0.2) == 0
        assert ppo_gate(1.2, 1.0, 0.2) == 1
        assert ppo_gate(0.79, -1.0, 0.2) == 0
        assert ppo_gate(0.8, -1.0, 0.2) == 1
        assert ppo_gate(1.5, -1.0, 0.2) == 1
        assert ppo_gate(0.5, 1.0, 0.2) == 1
        assert ppo_gate(4.0, 0.0, 0.2) == 1

    def test_sao_interval_is_strict_and_sign_independent(self):
        assert sao_gate(0.8, 0.2, 0.3) == 0
        assert sao_gate(1.3, 0.2, 0.3) == 0
        assert sao_gate(0.81, 0.2, 0.3) == 1
        assert sao_gate(1.29, 0.2, 0.3) == 1


class MetricTests(unittest.TestCase):
    def test_factorization_and_exact_zero(self):
        result = signal_metrics([1.0, 2.0], [1.0, -1.0], [1, 0])
        self.assertAlmostEqual(result["pam"], 2.5)
        self.assertAlmostEqual(result["gsr"], 0.2)
        self.assertAlmostEqual(result["egm"], result["pam"] * result["gsr"])
        assert not result["zero_policy_coefficient_mass"]
        zero = signal_metrics([1.0, 1.0], [0.0, 0.0], [1, 1])
        assert zero["pam"] == 0.0
        assert zero["gsr"] == 0.0
        assert zero["zero_policy_coefficient_mass"]

    def test_observation_tokens_are_excluded(self):
        result = signal_metrics([1, 100], [2, 100], [1, 1], [1, 0])
        assert result["n_action_tokens"] == 1
        assert result["pam"] == 4.0

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
        assert original == repartitioned
        assert math.isclose(original["egm"], original["pam"] * original["gsr"])


if __name__ == "__main__":
    unittest.main()
