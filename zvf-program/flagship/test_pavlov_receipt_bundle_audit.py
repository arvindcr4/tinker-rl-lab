from __future__ import annotations

import copy
import unittest

from flagship.build_pavlov_receipt_bundle import build_bundle
from flagship.build_pavlovs_campaign_manifest import build_manifest
from flagship.pavlov_receipt_bundle_audit import (
    audit_bundle,
    audit_campaign,
    audit_cross_bindings,
    audit_live_receipt,
    audit_receipt_set,
    sha256_json,
)
from flagship.pavlovs_domain_contract import load_contract
from flagship.test_pavlov_live_run_receipt import valid_receipt


class PavlovReceiptBundleAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()

    def test_placeholder_bundle_is_blocked_but_hash_and_exact_coverage_are_audited(self) -> None:
        bundle = build_bundle(self.contract)
        errors = audit_bundle(bundle, self.contract)
        self.assertTrue(errors)
        self.assertTrue(any("dataset_or_source_revision" in error for error in errors))
        self.assertFalse(any("bundle_hash" in error for error in errors))
        self.assertFalse(any("exact 12 training IDs" in error for error in errors))
        self.assertFalse(any("exact 14 primary_eval IDs" in error for error in errors))

    def test_bundle_hash_entry_hash_and_frozen_ids_are_mutation_sensitive(self) -> None:
        bundle = build_bundle(self.contract)

        tampered = copy.deepcopy(bundle)
        tampered["bundle_hash"] = "0" * 64
        self.assertTrue(any("bundle_hash" in error for error in audit_bundle(tampered, self.contract)))

        tampered = copy.deepcopy(bundle)
        tampered["suites"][0]["entry_hash"] = "0" * 64
        self.assertTrue(any("entry_hash" in error for error in audit_bundle(tampered, self.contract)))

        tampered = copy.deepcopy(bundle)
        tampered["structural_held_out_suite_ids"] = ["agentharm_eval"]
        self.assertTrue(any("frozen six" in error for error in audit_bundle(tampered, self.contract)))
        tampered = copy.deepcopy(bundle)
        tampered["primary_eval_not_designated_held_out_suite_ids"] = []
        self.assertTrue(any("frozen eight" in error for error in audit_bundle(tampered, self.contract)))

    def test_strict_booleans_caps_and_hf_policy_mutations_fail_closed(self) -> None:
        bundle = build_bundle(self.contract)
        tampered = copy.deepcopy(bundle)
        tampered["budget_guard"]["paid_jobs_may_launch"] = "false"
        self.assertTrue(any("typed boolean" in error for error in audit_bundle(tampered, self.contract)))
        tampered = copy.deepcopy(bundle)
        tampered["budget_guard"]["maximum_usd"] = "17.99"
        self.assertTrue(any("maximum_usd" in error for error in audit_bundle(tampered, self.contract)))
        tampered = copy.deepcopy(bundle)
        tampered["hf_policy"]["safe_public_artifact_rule"] = False
        self.assertTrue(any("hf_policy" in error for error in audit_bundle(tampered, self.contract)))

    def test_campaign_audit_requires_manifest_hash_and_separates_claims(self) -> None:
        campaign = build_manifest(copy.deepcopy(self.contract))
        self.assertTrue(any("manifest_hash" in error for error in audit_campaign(campaign, self.contract)))
        campaign["manifest_hash"] = sha256_json(campaign)
        self.assertEqual(audit_campaign(campaign, self.contract), [])

        tampered = copy.deepcopy(campaign)
        tampered["manifest_hash"] = sha256_json({**tampered, "scientific_evidence_status": "observed"})
        tampered["scientific_evidence_status"] = "observed"
        self.assertTrue(any("scientific_evidence_status" in error or "evidence" in error for error in audit_campaign(tampered, self.contract)))

    def test_live_receipt_audit_is_independent_and_hash_bound(self) -> None:
        receipt = valid_receipt()
        self.assertEqual(audit_live_receipt(receipt), [])
        tampered = copy.deepcopy(receipt)
        tampered["receipt_hash"] = "0" * 64
        self.assertTrue(any("receipt_hash" in error for error in audit_live_receipt(tampered)))
        tampered = copy.deepcopy(receipt)
        tampered["sampler_checkpoints"][0]["url"] = "https://evil.example/checkpoint"
        tampered["receipt_hash"] = sha256_json({key: value for key, value in tampered.items() if key != "receipt_hash"})
        self.assertTrue(any("sampler_checkpoints" in error for error in audit_live_receipt(tampered)))

    def test_cross_binding_digest_requires_every_artifact_and_is_mutation_sensitive(self) -> None:
        bundle = {"bundle_hash": "a" * 64, "campaign_hash": "b" * 64}
        campaign = {"manifest_hash": "b" * 64, "receipt_bundle_hash": "a" * 64}
        live = {
            "run": {"run_id": "run-1"},
            "receipt_hash": "c" * 64,
            "bundle_hash": "a" * 64,
            "campaign_hash": "b" * 64,
        }
        live_hashes = {"run-1": "c" * 64}
        binding_payload = {
            "bundle_hash": "a" * 64,
            "campaign_hash": "b" * 64,
            "live_receipt_hashes": live_hashes,
        }
        bindings = {
            **binding_payload,
            "cross_binding_hash": sha256_json(binding_payload),
        }
        self.assertEqual(audit_cross_bindings(bundle, campaign, [live], bindings), [])
        tampered = copy.deepcopy(bindings)
        tampered["cross_binding_hash"] = "0" * 64
        self.assertTrue(any("cross_binding_hash" in error for error in audit_cross_bindings(bundle, campaign, [live], tampered)))

    def test_full_audit_is_blocked_without_explicit_cross_binding(self) -> None:
        bundle = build_bundle(self.contract)
        campaign = build_manifest(copy.deepcopy(self.contract))
        report = audit_receipt_set(bundle, campaign, [valid_receipt()], {})
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["launches_any_job"])
        self.assertTrue(any("bindings" in error or "manifest_hash" in error for error in report["blockers"]))


if __name__ == "__main__":
    unittest.main()
