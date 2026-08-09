from __future__ import annotations

import math
import unittest

from flagship.pavlov_statistics import (
    StatisticsInputError,
    domain_no_regression_gate,
    domain_safety_gate,
    equal_domain_macro_aggregate,
    equal_domain_macro_score,
    exact_mcnemar_two_sided,
    mcnemar_discordant_counts,
    newcombe_paired_risk_difference_interval,
    paired_bootstrap_mean_difference,
    portfolio_domain_gates,
    wilson_interval,
)


class WilsonIntervalTests(unittest.TestCase):
    def test_boundary_and_observed_rate_golden_values(self) -> None:
        self.assertEqual(wilson_interval(0, 100), (0.0, 0.03699349820698565))
        self.assertAlmostEqual(wilson_interval(7, 100)[0], 0.03431926106727267)
        self.assertAlmostEqual(wilson_interval(7, 100)[1], 0.13749514739073496)
        self.assertEqual(wilson_interval(100, 100), (0.9630065017930143, 1.0))

    def test_rejects_invalid_counts_and_confidence(self) -> None:
        for successes, trials in ((0, 0), (8, 7), (-1, 10)):
            with self.subTest(successes=successes, trials=trials):
                with self.assertRaises(StatisticsInputError):
                    wilson_interval(successes, trials)
        for confidence in (0.0, 1.0, math.nan, math.inf):
            with self.subTest(confidence=confidence):
                with self.assertRaises(StatisticsInputError):
                    wilson_interval(1, 10, confidence=confidence)


class McNemarTests(unittest.TestCase):
    def test_no_discordant_pairs_return_neutral_p_value(self) -> None:
        base = [0, 1, 0, 1]
        candidate = [0, 1, 0, 1]
        self.assertEqual(mcnemar_discordant_counts(base, candidate), (0, 0))
        self.assertEqual(exact_mcnemar_two_sided(base, candidate), 1.0)

    def test_exact_two_sided_golden_values(self) -> None:
        base = [0] * 3 + [1]
        candidate = [1] * 3 + [0]
        self.assertEqual(mcnemar_discordant_counts(base, candidate), (3, 1))
        self.assertAlmostEqual(exact_mcnemar_two_sided(base, candidate), 0.625)

        all_improvements = [0] * 100
        all_adapter_successes = [1] * 100
        self.assertAlmostEqual(
            exact_mcnemar_two_sided(all_improvements, all_adapter_successes),
            1.5777218104420236e-30,
        )

    def test_rejects_empty_mismatched_or_nonbinary_pairs(self) -> None:
        cases = (([], []), ([0], []), ([0], [2]), ([0.5], [1]))
        for base, candidate in cases:
            with self.subTest(base=base, candidate=candidate):
                with self.assertRaises(StatisticsInputError):
                    exact_mcnemar_two_sided(base, candidate)


class NewcombeIntervalTests(unittest.TestCase):
    def test_paired_orientation_and_boundary_are_deterministic(self) -> None:
        base = [0] * 100
        candidate = [1] * 7 + [0] * 93
        interval = newcombe_paired_risk_difference_interval(base, candidate)
        self.assertAlmostEqual(interval[0], 0.018603170911274163)
        self.assertAlmostEqual(interval[1], 0.13749514739073496)
        self.assertTrue(interval[0] <= 0.07 <= interval[1])

        reverse = newcombe_paired_risk_difference_interval(candidate, base)
        self.assertAlmostEqual(reverse[0], -interval[1])
        self.assertAlmostEqual(reverse[1], -interval[0])

    def test_all_successes_have_a_finite_bounded_interval(self) -> None:
        interval = newcombe_paired_risk_difference_interval([1] * 100, [1] * 100)
        self.assertTrue(all(math.isfinite(bound) for bound in interval))
        self.assertTrue(-1.0 <= interval[0] <= 0.0 <= interval[1] <= 1.0)

    def test_rejects_unpaired_inputs(self) -> None:
        with self.assertRaises(StatisticsInputError):
            newcombe_paired_risk_difference_interval([0, 1], [1])


class PairedBootstrapTests(unittest.TestCase):
    def test_seed_and_resample_count_make_the_interval_reproducible(self) -> None:
        base = [0.0, 0.1, 0.2, 0.3]
        candidate = [0.1, 0.2, 0.4, 0.1]
        first = paired_bootstrap_mean_difference(base, candidate, resamples=257, seed=123)
        second = paired_bootstrap_mean_difference(base, candidate, resamples=257, seed=123)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.estimate, 0.05)
        self.assertAlmostEqual(first.lower, -0.125)
        self.assertAlmostEqual(first.upper, 0.175)
        self.assertEqual(first.seed, 123)
        self.assertEqual(first.resamples, 257)
        self.assertEqual(first.sample_size, 4)
        self.assertEqual(first.as_dict()["method"], "paired_percentile_bootstrap")

    def test_rejects_nonfinite_or_unpaired_inputs(self) -> None:
        with self.assertRaises(StatisticsInputError):
            paired_bootstrap_mean_difference([0.0, math.nan], [0.1, 0.2])
        with self.assertRaises(StatisticsInputError):
            paired_bootstrap_mean_difference([0.0], [0.1, 0.2])
        with self.assertRaises(StatisticsInputError):
            paired_bootstrap_mean_difference([0.0], [0.1], resamples=0)


class MacroAggregationTests(unittest.TestCase):
    def test_equal_weight_per_suite_then_per_domain(self) -> None:
        scores = {"s1": 0.2, "s2": 0.6, "s3": 0.8}
        tags = {"s1": ["a"], "s2": ["a", "b"], "s3": ["b"]}
        aggregate = equal_domain_macro_aggregate(scores, tags)
        self.assertEqual(aggregate["domains"], ["a", "b"])
        self.assertEqual(aggregate["suite_counts"], {"a": 2, "b": 2})
        self.assertAlmostEqual(aggregate["domain_means"]["a"], 0.4)
        self.assertAlmostEqual(aggregate["domain_means"]["b"], 0.7)
        self.assertAlmostEqual(aggregate["macro_score"], 0.55)
        self.assertAlmostEqual(equal_domain_macro_score(scores, tags), 0.55)

    def test_rejects_key_domain_and_finiteness_mismatches(self) -> None:
        with self.assertRaises(StatisticsInputError):
            equal_domain_macro_aggregate({"s1": 0.2}, {})
        with self.assertRaises(StatisticsInputError):
            equal_domain_macro_aggregate({"s1": 0.2}, {"s1": ["a"]}, domains=["b"])
        with self.assertRaises(StatisticsInputError):
            equal_domain_macro_aggregate({"s1": math.inf}, {"s1": ["a"]})


class DomainGateTests(unittest.TestCase):
    def test_no_regression_requires_point_and_strict_lower_bound(self) -> None:
        passing = domain_no_regression_gate({"a": -0.049}, {"a": (-0.049, 0.01)})
        self.assertTrue(passing)
        self.assertEqual(passing.failures, ())

        boundary = domain_no_regression_gate({"a": -0.05}, {"a": (-0.05, 0.01)})
        self.assertFalse(boundary)
        self.assertEqual(boundary.failures, ("a",))

    def test_safety_allows_the_threshold_but_not_an_exceedance(self) -> None:
        passing = domain_safety_gate({"a": 0.02}, {"a": (-0.01, 0.02)})
        self.assertTrue(passing)

        point_failure = domain_safety_gate({"a": 0.021}, {"a": (-0.01, 0.02)})
        interval_failure = domain_safety_gate({"a": 0.01}, {"a": (-0.01, 0.021)})
        self.assertFalse(point_failure)
        self.assertFalse(interval_failure)

    def test_combined_gate_is_conjunctive_and_validates_domains(self) -> None:
        result = portfolio_domain_gates(
            {"a": 0.01},
            {"a": (0.0, 0.02)},
            {"a": 0.01},
            {"a": (0.0, 0.02)},
        )
        self.assertTrue(result["passed"])
        with self.assertRaises(StatisticsInputError):
            portfolio_domain_gates(
                {"a": 0.01},
                {"a": (0.0, 0.02)},
                {"b": 0.01},
                {"b": (0.0, 0.02)},
            )


if __name__ == "__main__":
    unittest.main()
