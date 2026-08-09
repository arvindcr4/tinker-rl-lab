from __future__ import annotations

import copy
import unittest

from flagship.build_pavlov_receipt_bundle import (
    EXPECTED_PRIMARY_EVAL_SUITE_COUNT,
    EXPECTED_TRAINING_SUITE_COUNT,
    build_bundle,
    canonical_json,
    sha256_json,
    validate_bundle,
)
from flagship.pavlovs_domain_contract import load_contract


class PavlovReceiptBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()

    def test_preview_has_exact_contract_suite_coverage_and_domain_unions(self) -> None:
        bundle = build_bundle(self.contract)
        expected_train = sorted(
            suite_id
            for suite_id, suite in self.contract["suite_registry"].items()
            if suite["role"] == "train"
        )
        expected_eval = sorted(
            suite_id
            for suite_id, suite in self.contract["suite_registry"].items()
            if suite["role"] == "primary_eval"
        )
        self.assertEqual(len(expected_train), EXPECTED_TRAINING_SUITE_COUNT)
        self.assertEqual(len(expected_eval), EXPECTED_PRIMARY_EVAL_SUITE_COUNT)
        self.assertEqual(bundle["training_suite_ids"], expected_train)
        self.assertEqual(bundle["primary_eval_suite_ids"], expected_eval)
        self.assertEqual(
            [entry["suite_id"] for entry in bundle["suites"]],
            sorted(expected_train + expected_eval),
        )
        self.assertEqual(
            {
                domain
                for suite in self.contract["suite_registry"].values()
                if suite["role"] == "train"
                for domain in suite["domains"]
            },
            set(self.contract["domains"]),
        )
        self.assertEqual(
            {
                domain
                for suite in self.contract["suite_registry"].values()
                if suite["role"] == "primary_eval"
                for domain in suite["domains"]
            },
            set(self.contract["domains"]),
        )
        self.assertEqual(len(bundle["structural_held_out_suite_ids"]), 6)
        self.assertEqual(bundle["held_out_receipt_proven_suite_ids"], [])
        self.assertEqual(
            len(bundle["primary_eval_not_designated_held_out_suite_ids"]),
            8,
        )
        self.assertEqual(len(bundle["company_domain_coverage"]), 53)
        for coverage in bundle["company_domain_coverage"].values():
            self.assertTrue(all(coverage["training"].values()))
            self.assertTrue(all(coverage["primary_eval"].values()))
        self.assertNotEqual(
            set(bundle["structural_held_out_suite_ids"]),
            set(bundle["primary_eval_suite_ids"]),
        )

    def test_hashes_are_deterministic_and_canonical(self) -> None:
        first = build_bundle(self.contract)
        second = build_bundle(copy.deepcopy(self.contract))
        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(first["bundle_hash"], second["bundle_hash"])
        for entry in first["suites"]:
            payload = {
                key: value
                for key, value in entry.items()
                if key not in {"entry_hash", "blockers", "launchable", "admissible"}
            }
            self.assertEqual(entry["entry_hash"], sha256_json(payload))
        errors = validate_bundle(first, self.contract)
        self.assertFalse(any("budget status conflict" in error for error in errors))
        self.assertTrue(any("missing or invalid" in error for error in errors))

    def test_duplicate_and_missing_suite_entries_are_rejected(self) -> None:
        bundle = build_bundle(self.contract)
        duplicate = copy.deepcopy(bundle)
        duplicate["suites"].append(copy.deepcopy(duplicate["suites"][0]))
        errors = validate_bundle(duplicate, self.contract)
        self.assertTrue(any("duplicate suite entries" in error for error in errors))
        self.assertTrue(any("expected 26 suite entries" in error for error in errors))

        missing = copy.deepcopy(bundle)
        missing["suites"] = missing["suites"][1:]
        errors = validate_bundle(missing, self.contract)
        self.assertTrue(any("missing suite entries" in error for error in errors))
        self.assertTrue(any("expected 26 suite entries" in error for error in errors))

    def test_placeholders_are_blocked_and_never_launch(self) -> None:
        bundle = build_bundle(self.contract)
        self.assertEqual(bundle["status"], "BLOCKED")
        self.assertFalse(bundle["launchable"])
        self.assertFalse(bundle["launches_any_job"])
        self.assertFalse(bundle["allocation_allowed"])
        self.assertTrue(all(not entry["launchable"] for entry in bundle["suites"]))
        errors = validate_bundle(bundle, self.contract)
        self.assertTrue(errors)
        self.assertTrue(any("dataset_or_source_revision" in error for error in errors))
        self.assertTrue(any("hf_checkpoints" in error for error in errors))

    def test_xlam_and_gsm8k_scopes_are_separate(self) -> None:
        bundle = build_bundle(self.contract)
        self.assertEqual(bundle["xlam_component"]["claim_scope"], "component_only")
        self.assertFalse(bundle["xlam_component"]["admissible"])
        self.assertFalse(bundle["xlam_component"]["launchable"])
        self.assertEqual(bundle["gsm8k"]["role"], "calibration_only")
        self.assertFalse(bundle["gsm8k"]["primary_claim_allowed"])
        self.assertNotIn("gsm8k_calibration", bundle["training_suite_ids"])
        self.assertNotIn("gsm8k_calibration", bundle["primary_eval_suite_ids"])

    def test_validation_rejects_weak_identity_only_receipts(self) -> None:
        bundle = build_bundle(self.contract)
        entry = bundle["suites"][0]
        entry["wandb_run_identity"] = {"status": "complete", "online": True, "project": "p"}
        entry["hf_checkpoints"] = [{"repo": "org/model", "revision": "main"}]
        errors = validate_bundle(bundle, self.contract)
        self.assertTrue(any("wandb_run_identity" in error for error in errors))
        self.assertTrue(any("hf_checkpoints" in error for error in errors))

    def test_complete_local_receipts_validate_without_authorizing_a_launch(self) -> None:
        contract = copy.deepcopy(self.contract)
        contract["status"] = "authorized"
        contract["budget_gate"]["status"] = "AUTHORIZED_TINKER_ONLY"
        checkpoints = [
            {
                "stage": stage,
                "repo": "https://huggingface.co/org/pavlov",
                "revision": revision,
                "url": f"https://huggingface.co/org/pavlov/commit/{revision}",
                "visibility": visibility,
                "safe_public_artifact": True,
            }
            for stage, revision, visibility in zip(
                ("initial", "periodic", "final"),
                ("1" * 40, "2" * 40, "3" * 40),
                ("public", "private", "public"),
            )
        ]
        overrides = {
            suite_id: {
                "dataset_or_source_revision": "4" * 40,
                "license_or_approval": "MIT-approved",
                "split_task_id_hash": "5" * 64,
                "container_runtime_digest": "sha256:" + "6" * 64,
                "verifier_hash": "7" * 64,
                "model_revision": "8" * 40,
                "decontamination_status": {
                    "status": "verified",
                    "receipt_id": "9" * 40,
                },
                "budget_receipt": {
                    "authorization_id": "a" * 40,
                    "maximum_usd": 18.0,
                },
                "wandb_run_identity": {
                    "entity": "entity",
                    "project": "project",
                    "group": "group",
                    "run_id": "run-id",
                    "run_url": "https://wandb.ai/entity/project/runs/run-id",
                    "online": True,
                },
                "tinker_run_identity": {"run_id": "tinker-run", "cost_status": "authorized"},
                "cost_status": "authorized",
                "hf_checkpoints": checkpoints,
                "evidence_status": "observed",
            }
            for suite_id in (
                sorted(
                    suite_id
                    for suite_id, suite in contract["suite_registry"].items()
                    if suite["role"] in {"train", "primary_eval"}
                )
            )
        }
        bundle = build_bundle(contract, overrides)
        self.assertEqual(bundle["status"], "READY")
        self.assertFalse(bundle["launchable"])
        self.assertFalse(bundle["launches_any_job"])
        self.assertEqual(validate_bundle(bundle, contract), [])


if __name__ == "__main__":
    unittest.main()
