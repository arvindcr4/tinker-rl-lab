from __future__ import annotations

import unittest
from itertools import product

from .action_reversal import (
    Construction,
    action_reversal_holds,
    broken_verifier_utilities,
    clean_hard_utilities,
    matched_primary_observation,
    minimax_outcome_only_regret,
    minimax_retry_probability,
    probe_policy_regret_bound,
    reversal_margins,
)


class ActionReversalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.construction = Construction()

    def test_primary_observation_is_the_same_all_failure_state(self) -> None:
        observation = matched_primary_observation(self.construction)
        self.assertEqual(observation["p_hat"], 0.0)
        self.assertEqual(observation["homogeneous_group_rate"], 1.0)
        self.assertEqual(observation["rewards"], [0] * self.construction.group_size)

    def test_optimal_action_reverses_across_latent_regimes(self) -> None:
        self.assertEqual(clean_hard_utilities(self.construction).optimal_action, "retry")
        self.assertEqual(broken_verifier_utilities(self.construction).optimal_action, "recheck")
        self.assertTrue(action_reversal_holds(self.construction))
        self.assertTrue(all(value > 0 for value in reversal_margins(self.construction).values()))

    def test_every_outcome_only_policy_has_positive_minimax_regret(self) -> None:
        regret = minimax_outcome_only_regret(self.construction)
        retry_probability = minimax_retry_probability(self.construction)
        self.assertGreater(regret, 0.0)
        self.assertGreater(retry_probability, 0.0)
        self.assertLess(retry_probability, 1.0)

    def test_perfect_probe_resolves_ambiguity_but_still_charges_probe_cost(self) -> None:
        self.assertEqual(
            probe_policy_regret_bound(self.construction, 0.0),
            self.construction.probe_cost,
        )

    def test_imperfect_probe_regret_scales_with_error(self) -> None:
        low = probe_policy_regret_bound(self.construction, 0.05)
        high = probe_policy_regret_bound(self.construction, 0.10)
        excess_low = low - self.construction.probe_cost
        excess_high = high - self.construction.probe_cost
        self.assertAlmostEqual(excess_high, 2.0 * excess_low, places=12)

    def test_action_reversal_survives_a_bounded_parameter_neighborhood(self) -> None:
        for q, sample_cost, probe_cost, repaired_value in product(
            (0.08, 0.10, 0.12),
            (0.016, 0.020, 0.024),
            (0.024, 0.030, 0.036),
            (0.40, 0.50, 0.60),
        ):
            with self.subTest(q=q, sample_cost=sample_cost, probe_cost=probe_cost):
                construction = Construction(
                    clean_success_probability=q,
                    sample_cost=sample_cost,
                    probe_cost=probe_cost,
                    repaired_signal_value=repaired_value,
                )
                self.assertTrue(action_reversal_holds(construction))


if __name__ == "__main__":
    unittest.main()
