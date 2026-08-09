from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_claim_boundary import (
    DOMAIN_IDS,
    PRIMARY_EVAL_SUITE_IDS,
    REQUIRED_HOLDOUT_RECEIPTS,
    TRAIN_SUITE_IDS,
    compute_receipt_digest,
    compute_result_digest,
    classify_claim,
    classify_claim_text,
)


class PavlovClaimBoundaryTests(unittest.TestCase):
    def _receipt(self, binding: dict[str, str], number: int) -> dict[str, object]:
        payload = {
            "identity": f"{number:040x}"[-40:],
            "artifact_digest": f"{number:064x}"[-64:],
        }
        return {
            "receipt_id": f"{number + 1000:040x}"[-40:],
            "digest": compute_receipt_digest(binding, payload),
            "authenticated": True,
            "cryptographically_bound": True,
            "binding": binding,
            "payload": payload,
        }

    def _coverage_claim(self) -> dict[str, object]:
        coverage: dict[str, object] = {}
        number = 1
        for suite_id in (*TRAIN_SUITE_IDS, *PRIMARY_EVAL_SUITE_IDS):
            role = "train" if suite_id in TRAIN_SUITE_IDS else "primary_eval"
            coverage[suite_id] = self._receipt(
                {"subject": suite_id, "role": role, "kind": "coverage"}, number
            )
            number += 1
        return {
            "claim_kind": "portfolio_evidence",
            "claim_text": "Exact suite coverage is a prospective portfolio contract; no result is claimed.",
            "training_suite_ids": list(TRAIN_SUITE_IDS),
            "primary_eval_suite_ids": list(PRIMARY_EVAL_SUITE_IDS),
            "domain_unions": {"train": list(DOMAIN_IDS), "primary_eval": list(DOMAIN_IDS)},
            "coverage_receipts": coverage,
        }

    def _heldout_claim(self) -> dict[str, object]:
        claim = self._coverage_claim()
        claim["claim_kind"] = "heldout_result"
        claim["claim_text"] = "Receipt-proven held-out primary results are reported for the exact suites."
        holdout_receipts: dict[str, dict[str, object]] = {}
        results: dict[str, dict[str, object]] = {}
        number = 100
        for suite_id in PRIMARY_EVAL_SUITE_IDS:
            suite_receipts: dict[str, object] = {}
            for field in REQUIRED_HOLDOUT_RECEIPTS:
                suite_receipts[field] = self._receipt(
                    {"subject": suite_id, "role": "primary_eval", "kind": f"holdout:{field}"},
                    number,
                )
                number += 1
            holdout_receipts[suite_id] = suite_receipts
            result = {
                "n": 100,
                "metric": 0.50,
                "ci95": {"lower": 0.40, "upper": 0.60},
            }
            result_receipt = self._receipt(
                {"subject": suite_id, "role": "primary_eval", "kind": "result"}, number
            )
            result_receipt["payload"]["artifact_digest"] = compute_result_digest(result)
            result_receipt["digest"] = compute_receipt_digest(result_receipt["binding"], result_receipt["payload"])
            result["receipt"] = result_receipt
            results[suite_id] = result
            number += 1
        claim["holdout_receipts"] = holdout_receipts
        claim["results"] = results
        return claim

    def test_prose_alone_is_not_evidence(self) -> None:
        report = classify_claim_text("The portfolio is ready and all 14 suites are held out.")

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("unstructured_claim_text", report["blocker_codes"])
        self.assertIn("company_readiness_claim", report["blocker_codes"])

    def test_prospective_protocol_record_is_distinct_from_result_evidence(self) -> None:
        report = classify_claim(
            {
                "claim_kind": "protocol_preflight",
                "prospective": True,
                "claim_text": "The protocol defines a six-domain batch gate before paid work.",
                "evidence": {"source": "local contract", "purpose": "preflight"},
            }
        )

        self.assertEqual(report["status"], "ALLOWED")
        self.assertEqual(report["scope"], "protocol_preflight")

    def test_protocol_result_text_is_blocked_even_with_prospective_flag(self) -> None:
        report = classify_claim(
            {
                "claim_kind": "protocol_preflight",
                "prospective": True,
                "claim_text": "The observed adapter improved the portfolio result.",
                "evidence": {"source": "local protocol"},
            }
        )

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("protocol_claim_contains_result", report["blocker_codes"])

    def test_protocol_evidence_must_be_an_object(self) -> None:
        report = classify_claim(
            {
                "claim_kind": "protocol_preflight",
                "prospective": True,
                "claim_text": "A prospective protocol boundary only.",
                "evidence": [],
            }
        )

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("protocol_evidence_not_structured", report["blocker_codes"])

    def test_xlam_component_observation_can_pass_only_at_its_bounded_scope(self) -> None:
        report = classify_claim(
            {
                "claim_kind": "component_evidence",
                "claim_text": "Observed xLAM strict tool-use baseline only.",
                "component": {
                    "id": "xlam_baseline_observation",
                    "scope": "strict_tool_use_component",
                    "seed": 809,
                    "n": 100,
                    "successes": 7,
                },
                "receipt": self._receipt(
                    {"subject": "xlam_baseline_observation", "role": "component", "kind": "observation"},
                    500,
                ),
            }
        )

        self.assertEqual(report["status"], "ALLOWED")
        self.assertEqual(report["scope"], "component_evidence")

    def test_xlam_portfolio_promotion_is_blocked(self) -> None:
        claim = self._coverage_claim()
        claim["claim_kind"] = "component_evidence"
        claim["claim_text"] = "The xLAM result proves portfolio improvement across all 16 domains."
        claim["component"] = {
            "id": "xlam_baseline_observation",
            "scope": "strict_tool_use_component",
            "seed": 809,
            "n": 100,
            "successes": 7,
        }
        claim["receipt"] = self._receipt(
            {"subject": "xlam_baseline_observation", "role": "component", "kind": "observation"},
            501,
        )

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("xlam_promotion_forbidden", report["blocker_codes"])

    def test_gsm8k_calibration_is_not_a_portfolio_claim(self) -> None:
        report = classify_claim(
            {
                "claim_kind": "component_evidence",
                "claim_text": "GSM8K is a calibration-only diagnostic.",
                "component": {"id": "gsm8k_calibration", "role": "calibration_only"},
                "receipt": self._receipt(
                    {"subject": "gsm8k_calibration", "role": "component", "kind": "observation"},
                    502,
                ),
            }
        )

        self.assertEqual(report["status"], "ALLOWED")

    def test_gsm8k_promotion_is_blocked(self) -> None:
        report = classify_claim(
            {
                "claim_kind": "component_evidence",
                "claim_text": "GSM8K improves primary portfolio training.",
                "component": {"id": "gsm8k_calibration", "role": "calibration_only"},
                "receipt": self._receipt(
                    {"subject": "gsm8k_calibration", "role": "component", "kind": "observation"},
                    503,
                ),
            }
        )

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("gsm8k_promotion_forbidden", report["blocker_codes"])

    def test_self_attested_strings_and_booleans_cannot_clear_claim(self) -> None:
        claim = self._coverage_claim()
        claim["verified"] = True
        claim["status"] = "complete"
        claim["coverage_receipts"] = {suite_id: "verified" for suite_id in (*TRAIN_SUITE_IDS, *PRIMARY_EVAL_SUITE_IDS)}

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("self_attested_control", report["blocker_codes"])
        self.assertIn("coverage_receipt_invalid", report["blocker_codes"])

    def test_exact_portfolio_coverage_passes_without_becoming_a_result_claim(self) -> None:
        report = classify_claim(self._coverage_claim())

        self.assertEqual(report["status"], "ALLOWED")
        self.assertEqual(report["scope"], "portfolio_evidence")

    def test_related_benchmark_substitution_is_blocked(self) -> None:
        claim = self._coverage_claim()
        claim["claim_text"] = "GSM8K is a proxy instead of the 14 primary suites."

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("related_benchmark_substitution", report["blocker_codes"])

    def test_exact_suite_and_domain_sets_are_required(self) -> None:
        claim = self._coverage_claim()
        claim["primary_eval_suite_ids"] = list(PRIMARY_EVAL_SUITE_IDS[:-1]) + ["math500_eval"]
        claim["domain_unions"] = {"train": list(DOMAIN_IDS[:-1]), "primary_eval": list(DOMAIN_IDS)}

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("primary_eval_suite_ids_not_exact", report["blocker_codes"])
        self.assertIn("domain_union_not_exact", report["blocker_codes"])

    def test_receipt_proven_heldout_results_pass(self) -> None:
        report = classify_claim(self._heldout_claim())

        self.assertEqual(report["status"], "ALLOWED")
        self.assertEqual(report["scope"], "heldout_result")

    def test_missing_primary_holdout_receipt_blocks_all_14_claim(self) -> None:
        claim = self._heldout_claim()
        del claim["holdout_receipts"][PRIMARY_EVAL_SUITE_IDS[0]]["split"]

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("holdout_receipt_invalid", report["blocker_codes"])

    def test_fabricated_heldout_boolean_and_result_string_block(self) -> None:
        claim = self._heldout_claim()
        claim["final_holdout_untouched"] = True
        claim["results"][PRIMARY_EVAL_SUITE_IDS[0]] = "complete"

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("self_attested_control", report["blocker_codes"])
        self.assertIn("heldout_result_invalid", report["blocker_codes"])

    def test_tampered_result_receipt_digest_blocks(self) -> None:
        claim = self._heldout_claim()
        claim["results"][PRIMARY_EVAL_SUITE_IDS[0]]["receipt"]["digest"] = "f" * 64

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("heldout_result_receipt_invalid", report["blocker_codes"])

    def test_result_receipt_cannot_be_replayed_for_a_changed_metric(self) -> None:
        claim = self._heldout_claim()
        result = claim["results"][PRIMARY_EVAL_SUITE_IDS[0]]
        result["metric"] = 0.99

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("heldout_result_not_bound", report["blocker_codes"])

    def test_unhashable_claim_kind_and_out_of_range_interval_fail_closed(self) -> None:
        report = classify_claim({"claim_kind": [], "prospective": True})
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("claim_kind_invalid", report["blocker_codes"])

        claim = self._heldout_claim()
        claim["results"][PRIMARY_EVAL_SUITE_IDS[0]]["ci95"] = {"lower": -0.1, "upper": 1.1}
        report = classify_claim(claim)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("heldout_interval_invalid", report["blocker_codes"])

    def test_company_readiness_is_not_implied_by_domain_coverage(self) -> None:
        claim = self._coverage_claim()
        claim["claim_text"] = "All 53 companies are production-ready from this portfolio coverage."

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("company_readiness_claim", report["blocker_codes"])

    def test_fabricated_company_readiness_field_is_not_evidence(self) -> None:
        claim = self._coverage_claim()
        claim["company_readiness"] = "ready"

        report = classify_claim(claim)

        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("self_attested_control", report["blocker_codes"])

    def test_cli_emits_json_and_nonzero_for_unstructured_claim(self) -> None:
        script = Path(__file__).with_name("pavlov_claim_boundary.py")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump("All 14 suites are held out", handle)
            handle.flush()
            process = subprocess.run(
                [sys.executable, str(script), "--claim", handle.name],
                check=False,
                capture_output=True,
                text=True,
            )

        report = json.loads(process.stdout)
        self.assertEqual(process.returncode, 1)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
