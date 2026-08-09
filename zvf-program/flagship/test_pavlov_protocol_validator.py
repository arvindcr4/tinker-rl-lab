from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from flagship.pavlov_protocol_validator import (
    DEFAULT_BUDGET_PATH,
    DEFAULT_CONTRACT_PATH,
    DEFAULT_PROTOCOL_PATH,
    REQUIRED_RECEIPT_FIELDS,
    compute_receipt_digest,
    validate_bundle,
)


class PavlovProtocolValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = DEFAULT_PROTOCOL_PATH.read_text(encoding="utf-8")
        self.contract = json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
        self.budget = json.loads(DEFAULT_BUDGET_PATH.read_text(encoding="utf-8"))

    def _authorized_bundle(self) -> tuple[str, dict, dict]:
        contract = copy.deepcopy(self.contract)
        contract["status"] = "authorized"
        for index, model in enumerate(contract["model_candidates"], start=1):
            model["revision"] = f"{index:040x}"
        for suite_id, suite in contract["suite_registry"].items():
            if suite.get("role") == "primary_eval":
                suite["immutable_receipts"] = {}
                for index, field in enumerate(REQUIRED_RECEIPT_FIELDS, start=1):
                    payload = {
                        "identity": f"{index:040x}"[-40:],
                        "artifact_digest": f"{index:064x}"[-64:],
                    }
                    suite["immutable_receipts"][field] = {
                        "receipt_id": f"{(index + 100):040x}"[-40:],
                        "digest": compute_receipt_digest(suite_id, field, payload),
                        "authenticated": True,
                        "cryptographically_bound": True,
                        "binding": {"suite_id": suite_id, "field": field},
                        "payload": payload,
                    }
        return self.protocol, contract, copy.deepcopy(self.budget)

    def test_current_bundle_is_blocked_but_reports_structural_counts(self) -> None:
        report = validate_bundle(self.protocol, self.contract, self.budget)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["launch_allowed"])
        self.assertFalse(report["protocol_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertTrue(report["zero_cost"])
        self.assertFalse(report["network_accessed"])
        self.assertNotIn("contract_status_not_authorized", report["blocker_codes"])
        self.assertTrue(report["checks"]["budget_and_status"]["status_authorized"])
        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])
        structure = report["checks"]["contract_structure"]
        self.assertEqual(structure["suite_counts"], {"primary_eval": 14, "train": 12})
        self.assertEqual(structure["declared_domain_count"], 16)
        self.assertTrue(structure["company_domain_coverage"]["complete"])
        self.assertTrue(report["checks"]["protocol_holdout_claims"]["claim_rule_passes"])

    def test_complete_authorized_fixture_can_pass_without_paid_work(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        report = validate_bundle(protocol, contract, budget)

        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["protocol_ready"])
        self.assertTrue(report["launch_allowed"])
        self.assertTrue(report["paid_launch_allowed"])
        self.assertEqual(report["blockers"], [])
        self.assertTrue(report["checks"]["primary_receipts"]["complete"])

    def test_receipt_fields_are_required_for_every_primary_suite(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        del contract["suite_registry"]["swe_bench_pro_eval"]["immutable_receipts"]["task"]

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])
        missing = report["checks"]["primary_receipts"]["missing"]
        entry = next(item for item in missing if item["suite_id"] == "swe_bench_pro_eval")
        self.assertEqual(
            next(item for item in entry["missing"] if item["field"] == "task")["field"],
            "task",
        )

    def test_exact_suite_counts_and_independent_company_coverage_fail_closed(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        contract["suite_registry"]["swe_bench_pro_eval"]["role"] = "train"
        contract["companies"][0]["domains"] = ["alignment"]
        for suite in contract["suite_registry"].values():
            if suite.get("role") == "primary_eval":
                suite["domains"] = [domain for domain in suite["domains"] if domain != "alignment"]

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("suite_count_mismatch", report["blocker_codes"])
        self.assertIn("domain_union_mismatch", report["blocker_codes"])
        self.assertIn("company_domain_missing_primary_eval", report["blocker_codes"])

    def test_domain_union_and_gsm8k_rules_fail_closed(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        contract["suite_registry"]["gsm8k_calibration"]["role"] = "primary_eval"

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("suite_count_mismatch", report["blocker_codes"])
        self.assertIn("gsm8k_calibration_role_invalid", report["blocker_codes"])
        self.assertIn("gsm8k_primary_role_forbidden", report["blocker_codes"])

    def test_budget_values_and_arithmetic_are_exact(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        budget["safety_reserve_usd"] = 1.4

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("budget_values_mismatch", report["blocker_codes"])
        self.assertIn("budget_arithmetic_invalid", report["blocker_codes"])
        self.assertIn("contract_budget_mismatch", report["blocker_codes"])

    def test_unqualified_14_held_out_claim_requires_all_receipts(self) -> None:
        _, contract, budget = self._authorized_bundle()
        protocol = "All 14 primary evaluation suites are held-out.\n"
        for suite in contract["suite_registry"].values():
            if suite.get("role") == "primary_eval":
                suite.pop("immutable_receipts", None)

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])
        self.assertIn("protocol_holdout_claim_without_receipts", report["blocker_codes"])
        self.assertEqual(
            report["checks"]["protocol_holdout_claims"]["unqualified_claims"][0]["line"],
            1,
        )

    def test_qualified_protocol_wording_does_not_count_as_claim(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        protocol = "All 14 primary evaluation suites have holdout status pending receipts.\n"

        report = validate_bundle(protocol, contract, budget)

        self.assertEqual(report["checks"]["protocol_holdout_claims"]["unqualified_claims"], [])
        self.assertEqual(report["status"], "READY")

    def test_draft_contract_can_be_protocol_ready_but_never_paid_launchable(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        contract["status"] = "draft-awaiting-budget-cap"

        report = validate_bundle(protocol, contract, budget)

        self.assertTrue(report["protocol_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertFalse(report["launch_allowed"])
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("contract_status_not_authorized", report["blocker_codes"])

    def test_receipts_are_not_clearable_by_urls_run_ids_or_status_text(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        suite = contract["suite_registry"]["swe_bench_pro_eval"]
        suite["immutable_receipts"]["revision"] = {
            "receipt_id": "0" * 40,
            "digest": "0" * 64,
            "authenticated": True,
            "cryptographically_bound": True,
            "status": "complete",
            "run_id": "fake-run-id",
        }

        report = validate_bundle(protocol, contract, budget)

        self.assertFalse(report["protocol_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])

    def test_receipt_identity_digest_and_boolean_types_are_strict(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        record = contract["suite_registry"]["swe_bench_pro_eval"]["immutable_receipts"]["revision"]
        record["receipt_id"] = "A" * 40
        record["authenticated"] = "true"

        report = validate_bundle(protocol, contract, budget)

        missing = next(
            item
            for item in report["checks"]["primary_receipts"]["missing"]
            if item["suite_id"] == "swe_bench_pro_eval"
        )
        reasons = {item["field"]: item["reason"] for item in missing["missing"]}
        self.assertEqual(reasons["revision"], "receipt_identity_not_lower_hex40")
        self.assertFalse(report["paid_launch_allowed"])

    def test_receipt_digest_must_bind_suite_field_and_payload(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        record = contract["suite_registry"]["swe_bench_pro_eval"]["immutable_receipts"]["revision"]
        record["payload"]["identity"] = "f" * 40

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])
        self.assertFalse(report["paid_launch_allowed"])

    def test_status_text_inside_a_bound_payload_is_not_receipt_evidence(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        record = contract["suite_registry"]["swe_bench_pro_eval"]["immutable_receipts"]["revision"]
        record["payload"]["status"] = "verified"

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])
        self.assertFalse(report["paid_launch_allowed"])

    def test_malformed_receipt_url_is_a_blocker_not_a_validator_crash(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        record = contract["suite_registry"]["swe_bench_pro_eval"]["immutable_receipts"]["revision"]
        record["payload"]["source_url"] = "https://[malformed"

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("primary_suite_receipts_incomplete", report["blocker_codes"])
        self.assertFalse(report["paid_launch_allowed"])

    def test_budget_and_gate_types_are_exact(self) -> None:
        protocol, contract, budget = self._authorized_bundle()
        contract["budget_gate"]["paid_jobs_may_launch"] = "yes"
        budget["maximum_usd"] = "18.00"

        report = validate_bundle(protocol, contract, budget)

        self.assertIn("paid_jobs_may_launch_type_invalid", report["blocker_codes"])
        self.assertIn("budget_values_mismatch", report["blocker_codes"])
        self.assertFalse(report["paid_launch_allowed"])

    def test_cli_emits_json_and_nonzero_for_expected_current_blockers(self) -> None:
        script = Path(__file__).with_name("pavlov_protocol_validator.py")
        process = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent.parent.parent,
            check=False,
            capture_output=True,
            text=True,
        )

        report = json.loads(process.stdout)
        self.assertEqual(process.returncode, 1)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertNotIn("contract_status_not_authorized", report["blocker_codes"])
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
