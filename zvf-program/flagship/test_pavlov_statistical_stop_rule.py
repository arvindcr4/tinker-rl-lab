from __future__ import annotations

import math
import unittest

from flagship.pavlov_statistical_stop_rule import (
    COMPONENT_ONLY_CLAIM,
    StopRuleInputError,
    budget_aware_stop,
    build_paired_arm_metrics,
    evaluate_domain_gates,
    rank_arms,
    select_winner,
    strictly_beats_base,
    successive_halving,
    validate_selection_vs_final_split,
)


def _gates(*, delta: float = 0.0, delta_interval: tuple[float, float] = (0.0, 0.01), safety: float = 0.0, safety_interval: tuple[float, float] = (0.0, 0.01)) -> dict[str, object]:
    return {
        "deltas": {"tool_use": delta},
        "delta_intervals": {"tool_use": delta_interval},
        "safety_increases": {"tool_use": safety},
        "safety_intervals": {"tool_use": safety_interval},
    }


def _arm(
    arm_id: str,
    *,
    count: int,
    mean: float,
    cost: float = 1.0,
    learning_rate: float = 1e-5,
    gates: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "arm_id": arm_id,
        "learning_rate": learning_rate,
        "estimated_cost_usd": cost,
        "split_id": "selection",
        "split_role": "selection",
        "perfect_call_count": count,
        "trials": 100,
        "mean_strict_reward": mean,
        "domain_gates": _gates() if gates is None else gates,
        "claim_scope": COMPONENT_ONLY_CLAIM,
    }


BASE = {
    "perfect_call_count": 7,
    "trials": 100,
    "mean_strict_reward": 0.07,
    "perfect_call_rate": 0.07,
}


class PairedMetricTests(unittest.TestCase):
    def test_computes_paired_statistics_and_records_reproducibility(self) -> None:
        base = [1.0] * 7 + [0.0] * 93
        candidate = base[:]
        candidate[0] = 0.0
        candidate[7] = 1.0
        candidate[8] = 1.0
        candidate[9] = 1.0
        metrics = build_paired_arm_metrics(
            "lr1e-5",
            base,
            candidate,
            learning_rate=1e-5,
            estimated_cost_usd=0.54,
            domain_gates=_gates(),
            bootstrap_resamples=31,
            bootstrap_seed=123,
        )
        self.assertEqual(metrics.perfect_call_count, 9)
        self.assertAlmostEqual(metrics.mean_strict_reward, 0.09)
        self.assertEqual(metrics.mcnemar_counts, (3, 1))
        self.assertAlmostEqual(metrics.exact_mcnemar_pvalue, 0.625)
        self.assertEqual(metrics.paired_bootstrap["seed"], 123)
        self.assertEqual(metrics.paired_bootstrap["resamples"], 31)
        self.assertTrue(metrics.domain_gates_passed)
        self.assertEqual(metrics.as_dict()["claim_scope"], COMPONENT_ONLY_CLAIM)

    def test_empty_mismatched_and_nonfinite_pairs_fail_closed(self) -> None:
        cases = (([], []), ([0.0], []), ([0.0, 1.0], [0.0]), ([math.nan], [0.0]), ([0.0], [math.inf]))
        for base, candidate in cases:
            with self.subTest(base=base, candidate=candidate):
                with self.assertRaises(StopRuleInputError):
                    build_paired_arm_metrics(
                        "arm",
                        base,
                        candidate,
                        learning_rate=1e-5,
                        estimated_cost_usd=1.0,
                        domain_gates=_gates(),
                        bootstrap_resamples=5,
                    )

    def test_selection_metadata_and_claim_boundary_are_strict(self) -> None:
        with self.assertRaises(StopRuleInputError):
            build_paired_arm_metrics(
                "arm",
                [0.0],
                [1.0],
                learning_rate=1e-5,
                estimated_cost_usd=1.0,
                split_id="final",
                final_split_id="final",
                domain_gates=_gates(),
                bootstrap_resamples=5,
            )
        with self.assertRaises(StopRuleInputError):
            build_paired_arm_metrics(
                "arm",
                [0.0],
                [1.0],
                learning_rate=1e-5,
                estimated_cost_usd=1.0,
                claim_scope="portfolio",
                domain_gates=_gates(),
                bootstrap_resamples=5,
            )


class RankingAndGateTests(unittest.TestCase):
    def test_exact_protocol_tie_breaks_without_rounding(self) -> None:
        arms = [
            _arm("z", count=8, mean=0.08, cost=1.0, learning_rate=2e-5),
            _arm("a", count=8, mean=0.0800000000001, cost=9.0, learning_rate=4e-5),
            _arm("b", count=8, mean=0.0800000000001, cost=1.0, learning_rate=4e-5),
            _arm("c", count=8, mean=0.0800000000001, cost=1.0, learning_rate=1e-5),
            _arm("d", count=8, mean=0.0800000000001, cost=1.0, learning_rate=1e-5),
        ]
        ordered = rank_arms(arms)
        self.assertEqual([item["arm_id"] for item in ordered], ["c", "d", "b", "a", "z"])

    def test_missing_or_failing_domain_gate_prevents_selection(self) -> None:
        missing = _arm("missing", count=9, mean=0.09)
        del missing["domain_gates"]
        failed = _arm("failed", count=9, mean=0.09, gates=_gates(delta_interval=(-0.06, 0.01)))
        report = select_winner([missing, failed], BASE, next_cost_usd=2.0)
        self.assertEqual(report["status"], "stopped_no_eligible_arm")
        self.assertIsNone(report["winner"])
        self.assertFalse(report["portfolio_claim_permitted"])
        reasons = {item["arm_id"]: item["reasons"] for item in report["eligibility"]}
        self.assertTrue(any("domain_gates_missing" in item for item in reasons["missing"]))
        self.assertTrue(any("domain_gates_failed" in item for item in reasons["failed"]))

    def test_gate_result_requires_both_no_regression_and_safety(self) -> None:
        result = evaluate_domain_gates(
            {"tool_use": 0.0},
            {"tool_use": (0.0, 0.01)},
            {"tool_use": 0.0},
            {"tool_use": (0.0, 0.01)},
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["no_regression"]["passed"])
        self.assertTrue(result["safety"]["passed"])

    def test_base_comparison_uses_count_then_exact_mean(self) -> None:
        self.assertTrue(strictly_beats_base(_arm("count", count=8, mean=0.0), BASE))
        self.assertTrue(strictly_beats_base(_arm("mean", count=7, mean=0.0700000000001), BASE))
        self.assertFalse(strictly_beats_base(_arm("tie", count=7, mean=0.07), BASE))


class BudgetAndSplitTests(unittest.TestCase):
    def test_budget_stops_at_operational_cap_and_preserves_reserve(self) -> None:
        allowed = budget_aware_stop(15.0, 1.5)
        self.assertFalse(allowed["stop"])
        self.assertEqual(allowed["projected_total_usd"], 16.5)
        stopped = budget_aware_stop(15.0, 1.5000000001)
        self.assertTrue(stopped["stop"])
        self.assertEqual(stopped["reason"], "operational_cap_exceeded")
        self.assertEqual(stopped["reserve_usd"], 1.5)

    def test_invalid_budget_and_split_overlap_fail_closed(self) -> None:
        with self.assertRaises(StopRuleInputError):
            budget_aware_stop(0.0, 1.0, operational_cap_usd=17.0, hard_max_usd=18.0, reserve_usd=1.5)
        with self.assertRaises(StopRuleInputError):
            validate_selection_vs_final_split("selection", "selection")
        with self.assertRaises(StopRuleInputError):
            validate_selection_vs_final_split(
                "selection", "final", selection_example_ids=[1, 2], final_example_ids=[2, 3]
            )
        with self.assertRaises(StopRuleInputError):
            validate_selection_vs_final_split("selection", "final", final_used_for_selection=True)


class SuccessiveHalvingTests(unittest.TestCase):
    def test_selects_only_eligible_winner_and_requires_final_evaluation(self) -> None:
        first = [_arm("lr2e-5", count=9, mean=0.09, cost=0.5), _arm("lr1e-5", count=8, mean=0.08, cost=0.5)]
        second = [_arm("lr2e-5", count=10, mean=0.10, cost=0.75), _arm("lr1e-5", count=8, mean=0.08, cost=0.75)]
        report = successive_halving([first, second], BASE, keep_counts=[1, 1])
        self.assertEqual(report["status"], "winner_selected")
        self.assertEqual(report["winner"]["arm_id"], "lr2e-5")
        self.assertTrue(report["final_evaluation_required"])
        self.assertEqual(report["claim_scope"], COMPONENT_ONLY_CLAIM)
        self.assertFalse(report["company_claim_permitted"])
        self.assertEqual(report["rounds"][0]["retained_arm_ids"], ["lr2e-5"])

    def test_successive_halving_stops_before_an_over_cap_round(self) -> None:
        arms = [_arm("a", count=9, mean=0.09, cost=2.0), _arm("b", count=8, mean=0.08, cost=2.0)]
        report = successive_halving(
            [arms, arms],
            BASE,
            projected_round_costs=[16.0, 1.0],
        )
        self.assertEqual(report["status"], "stopped_budget")
        self.assertEqual(report["reason"], "operational_cap_exceeded")
        self.assertIsNone(report["winner"])

    def test_no_candidate_beating_base_stops_without_claim(self) -> None:
        report = successive_halving([[_arm("tie", count=7, mean=0.07)]], BASE)
        self.assertEqual(report["status"], "stopped_no_eligible_arm")
        self.assertTrue(report["stop"])
        self.assertFalse(report["portfolio_claim_permitted"])


if __name__ == "__main__":
    unittest.main()
