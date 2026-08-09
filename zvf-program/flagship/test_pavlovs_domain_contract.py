from __future__ import annotations

import copy
import unittest

from flagship.pavlovs_domain_contract import load_contract, validate_contract


class PavlovsDomainContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()

    def test_frozen_snapshot_has_full_company_and_domain_coverage(self) -> None:
        self.assertEqual(validate_contract(self.contract), [])
        self.assertEqual(len(self.contract["companies"]), 53)
        self.assertEqual(len(self.contract["domains"]), 16)

    def test_gsm8k_cannot_be_promoted_to_primary(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["suite_registry"]["gsm8k_calibration"]["role"] = "primary_eval"
        errors = validate_contract(candidate)
        self.assertTrue(any("GSM8K must be calibration_only" in e for e in errors))

    def test_gsm8k_cannot_be_used_as_training_coverage(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["suite_registry"]["math_control_train"]["role"] = "train"
        errors = validate_contract(candidate)
        self.assertIn(
            "GSM8K may not be a training or primary evaluation suite", errors
        )

    def test_every_company_requires_a_known_domain(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["companies"][0]["domains"] = ["imaginary_domain"]
        errors = validate_contract(candidate)
        self.assertTrue(any("unknown domains" in e for e in errors))

    def test_every_domain_requires_train_and_primary_eval_coverage(self) -> None:
        candidate = copy.deepcopy(self.contract)
        for suite in candidate["suite_registry"].values():
            suite["domains"] = [d for d in suite["domains"] if d != "chip_design"]
        errors = validate_contract(candidate)
        self.assertIn("chip_design: no training suite", errors)
        self.assertIn("chip_design: no primary evaluation suite", errors)

    def test_paid_jobs_require_an_explicit_cap(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["budget_gate"]["paid_jobs_may_launch"] = True
        candidate["budget_gate"]["maximum_usd"] = None
        errors = validate_contract(candidate)
        self.assertIn("paid jobs require a positive explicit maximum_usd", errors)

    def test_training_mixture_cannot_collapse_to_prompt_only_math(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["sampling_contract"]["minimum_primary_domains_per_training_batch"] = 1
        candidate["sampling_contract"]["minimum_stateful_fraction"] = 0.0
        candidate["sampling_contract"]["minimum_artifact_or_side_effect_fraction"] = 0.0
        errors = validate_contract(candidate)
        self.assertIn("every training batch must cover at least six domains", errors)
        self.assertIn(
            "at least 60% of the training mixture must be stateful", errors
        )
        self.assertIn(
            "at least 50% of the training mixture must inspect artifacts or side effects",
            errors,
        )

    def test_models_must_remain_multimodal_tool_capable_agents(self) -> None:
        candidate = copy.deepcopy(self.contract)
        candidate["model_candidates"][0]["required_capabilities"].remove(
            "computer_use"
        )
        errors = validate_contract(candidate)
        self.assertTrue(any("missing capabilities" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
