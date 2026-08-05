from __future__ import annotations

import unittest
from unittest import mock

from platform_hybrid.paper.scripts import reviewer_36320_corpus_check as corpus


class Reviewer36320CorpusCheckTests(unittest.TestCase):
    def test_canonical_sets_are_complete(self) -> None:
        self.assertEqual(set(corpus.ACTIVE_ROSTER), {f"P{i}" for i in range(1, 13)})
        self.assertEqual(set(corpus.HISTORICAL_ROOTS), {"R01", "R02", "R06", "R07", "U01", "P08_fraud"})

    def test_checked_corpus_passes(self) -> None:
        self.assertEqual(corpus.check(), [])

    def test_residual_claims_are_rejected(self) -> None:
        text = (
            "ZVF can tell a practitioner what action to take next. "
            "The clean held-out Qwen3-8B result is 22.5."
        )
        errors = corpus.descendant_claim_errors(text)
        self.assertTrue(any("tell a practitioner" in error for error in errors))
        self.assertTrue(any("clean held-out" in error for error in errors))
        self.assertTrue(any("Qwen PPO" in error for error in errors))

    def test_primary_inferential_five_seed_claim_is_rejected(self) -> None:
        errors = corpus.descendant_claim_errors(
            "Primary inferential claims include the held-out 5-seed Qwen3-8B control."
        )
        self.assertTrue(any("primary inferential claim" in error for error in errors))

    def test_long_tail_gradient_controller_and_local_82_claims_are_rejected(self) -> None:
        text = (
            "The gradient vanishes. CCC is the recommended controller. "
            "Adopt ADAPTIVE_PP as the default controller. "
            "The 82.0 to 83.3 held-out gain might widen."
        )
        errors = corpus.descendant_claim_errors(text)
        self.assertTrue(any("gradient vanishes" in error for error in errors))
        self.assertTrue(corpus.p7_operational_errors(text))
        self.assertTrue(any("adoption prescription" in error for error in errors))
        self.assertTrue(any("unsafe local 82.0-to-83.3" in error for error in errors))

    def test_safe_forward_and_reversed_82_comparisons_are_accepted(self) -> None:
        forward = (
            "The fixed-baseline 82.0% versus 83.3% result is a one-sample "
            "seed-level test, not a paired-seed or reviewed-record test."
        )
        reversed_order = (
            "The 83.3% versus fixed-baseline 82.0% result is a one-sample "
            "seed-level test, not paired-seed and not reviewed-record evidence."
        )
        self.assertFalse(any("unsafe local 82.0-to-83.3" in error for error in corpus.descendant_claim_errors(forward)))
        self.assertFalse(any("unsafe local 82.0-to-83.3" in error for error in corpus.descendant_claim_errors(reversed_order)))

    def test_actual_p8_future_evidence_table_is_locally_scoped(self) -> None:
        text = corpus.inclusion_closure(corpus.ACTIVE_ROSTER["P8"])
        self.assertIn("fixed-baseline $82.0\\%$", text)
        self.assertIn("not paired-seed, not reviewed-record", text)
        self.assertFalse(any("unsafe local 82.0-to-83.3" in error for error in corpus.descendant_claim_errors(text)))

    def test_qwen_derived_comparison_statistic_is_rejected_but_quarantine_is_accepted(self) -> None:
        unsafe = "PPO vs GRPO on Qwen3-8B (Welch t-test) & 0.7605 & 0.7605"
        safe = "PPO vs GRPO on Qwen3-8B: provenance conflict; not estimable & ---"
        self.assertTrue(any("derived Qwen PPO comparison statistic" in error for error in corpus.descendant_claim_errors(unsafe)))
        self.assertFalse(any("derived Qwen PPO comparison statistic" in error for error in corpus.descendant_claim_errors(safe)))

    def test_reversed_qwen_comparison_and_ppo_trace_statistics_are_rejected(self) -> None:
        reversed_order = "Qwen3-8B: PPO vs GRPO (Mann-Whitney U) & +0.01 & 0.709"
        trace = "Late-10 reward (ppo_qwen3-8b) & Mann-Whitney U & +0.08 & 0.782"
        self.assertTrue(corpus.qwen_ppo_derived_errors(reversed_order))
        self.assertTrue(corpus.qwen_ppo_derived_errors(trace))
        self.assertEqual(
            corpus.qwen_ppo_derived_errors(
                "Qwen3-8B PPO trace: provenance conflict; not estimable & ---"
            ),
            [],
        )

    def test_qwen_quarantine_cannot_mask_raw_or_derived_numbers(self) -> None:
        self.assertTrue(
            corpus.qwen_ppo_derived_errors(
                "PPO & Qwen3-8B & quarantined; provenance conflict & 75.0% & 22.5%"
            )
        )
        self.assertTrue(
            corpus.qwen_ppo_derived_errors(
                "Qwen3-8B PPO: quarantined; provenance conflict\nWelch t-test & 0.709 \\\\"
            )
        )
        self.assertTrue(
            corpus.qwen_ppo_derived_errors(
                "Qwen3-8B PPO vs GRPO: provenance conflict; not estimable & 0.709"
            )
        )

    def test_p7_operational_imperatives_are_rejected_but_candidates_are_accepted(self) -> None:
        self.assertTrue(corpus.p7_operational_errors("Use the Bayesian controller as the default policy."))
        self.assertTrue(corpus.p7_operational_errors("We adopt C3 as the operating setting for deployment."))
        self.assertTrue(corpus.p7_operational_errors("We recommend C3 for deployment."))
        self.assertTrue(corpus.p7_operational_errors("C3 is recommended for deployment."))
        self.assertTrue(corpus.p7_operational_errors("C3 should be adopted for deployment."))
        self.assertTrue(corpus.p7_operational_errors("Pick C2 for the operating point."))
        self.assertTrue(corpus.p7_operational_errors("Default to C1 when telemetry is absent."))
        self.assertEqual(
            corpus.p7_operational_errors(
                "C3 is a pre-registered candidate requiring matched-budget held-out evaluation."
            ),
            [],
        )
        self.assertEqual(corpus.p7_operational_errors("Do not deploy C3 without held-out evaluation."), [])
        self.assertEqual(corpus.p7_operational_errors("C3 is not recommended for deployment."), [])
        self.assertEqual(corpus.p7_operational_errors("C3 should not be adopted for deployment."), [])

    def test_full_p7_claim_path_accepts_negated_recommendations(self) -> None:
        self.assertEqual(corpus.live_root_claim_errors("P7", "C3 is not a recommended controller."), [])
        self.assertEqual(corpus.live_root_claim_errors("P7", "We do not recommend C3 as default controller."), [])

    def test_p10_total_update_shorthand_is_rejected_and_centered_scope_is_accepted(self) -> None:
        self.assertTrue(corpus.p10_centered_contribution_errors("T2 bounds rollouts-to-nonzero-gradient."))
        self.assertTrue(corpus.p10_centered_contribution_errors("T2 permits a non-zero policy update."))
        self.assertTrue(corpus.p10_centered_contribution_errors(r"T2 permits an \emph{nonzero gradient}."))
        self.assertTrue(corpus.p10_centered_contribution_errors(r"T2 permits a non\mbox{-}zero policy update."))
        self.assertTrue(corpus.p10_centered_contribution_errors(r"T2 permits a non~zero policy update."))
        self.assertTrue(corpus.p10_centered_contribution_errors(r"T2 permits a non\hbox{-}zero policy update."))
        self.assertTrue(corpus.p10_centered_contribution_errors(r"T2 permits a non\allowbreak zero policy update."))
        self.assertTrue(corpus.p10_centered_contribution_errors(r"T2 permits a non{}zero policy update."))
        self.assertEqual(
            corpus.p10_centered_contribution_errors(
                "Under the stated assumptions, T2 bounds rollouts to a potentially nonzero "
                "centered reward-contrast contribution, not a total update."
            ),
            [],
        )

    def test_82_qualifiers_cannot_be_borrowed_from_a_neighboring_comparison(self) -> None:
        text = (
            "The 82.0% to 83.3% result is an asserted gain. "
            "A second 82.0% to 83.3% result is a one-sample seed-level test "
            "against a fixed baseline, not paired-seed and not reviewed-record evidence."
        )
        errors = corpus.descendant_claim_errors(text)
        self.assertTrue(any("unsafe local 82.0-to-83.3" in error for error in errors))

    def test_82_qualifiers_in_a_separate_sentence_cannot_mask_a_comparison(self) -> None:
        text = (
            "The 82.0% to 83.3% result is an asserted gain. "
            "A one-sample seed-level test against a fixed baseline is not paired-seed and not reviewed-record evidence."
        )
        errors = corpus.descendant_claim_errors(text)
        self.assertTrue(any("unsafe local 82.0-to-83.3" in error for error in errors))

    def test_82_qualifiers_cannot_borrow_across_a_semicolon_clause(self) -> None:
        text = (
            "The 82.0% to 83.3% result is an asserted gain; a one-sample seed-level "
            "test against a fixed baseline is not paired-seed and not reviewed-record evidence."
        )
        errors = corpus.descendant_claim_errors(text)
        self.assertTrue(any("unsafe local 82.0-to-83.3" in error for error in errors))

    def test_total_gradient_claim_in_a_live_root_is_rejected(self) -> None:
        original = corpus.inclusion_closure

        def closure_with_p2_violation(rel: str, seen: set[str] | None = None) -> str:
            if rel == corpus.ACTIVE_ROSTER["P2"]:
                return "An all-equal group contributes no gradient."
            return original(rel, seen)

        with mock.patch.object(corpus, "inclusion_closure", side_effect=closure_with_p2_violation):
            errors = corpus.check()
        self.assertTrue(any(error.startswith("P2:") and "contributes no gradient" in error for error in errors))

    def test_deployment_prescription_in_a_live_root_is_rejected(self) -> None:
        original = corpus.inclusion_closure

        def closure_with_p7_violation(rel: str, seen: set[str] | None = None) -> str:
            if rel == corpus.ACTIVE_ROSTER["P7"]:
                return "The joint trigger is the recommended deployment."
            return original(rel, seen)

        with mock.patch.object(corpus, "inclusion_closure", side_effect=closure_with_p7_violation):
            errors = corpus.check()
        self.assertTrue(any(error.startswith("P7:") and "recommended deployment" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
