#!/usr/bin/env python3
"""Adversarial offline tests for T7 agentdojo_train boundary adapter."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from flagship import pavlov_agentdojo_train_adapter as adapter


class PavlovAgentdojoTrainAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = adapter.generate_boundary_bundle()

    def _make_filled_receipt(self, suite_id: str, *, include_public_first: bool = True) -> dict[str, Any]:
        task_id_hashes = ["a" * 64, "b" * 64, "c" * 64]
        checkpoint_visibility = "public" if include_public_first else "private"
        base = {
            "dataset_id": f"agentdojo-{suite_id}-tasks",
            "dataset_or_source_revision": "f" * 40,
            "license_or_approval": {
                "status": "approved",
                "license_id": "spdx:Apache-2.0",
                "approval_id": f"license-{suite_id}",
            },
            "task_id_hashes": task_id_hashes,
            "split_task_id_hash": adapter.aggregate_task_id_hashes(task_id_hashes),
            "split_manifest_hash": "1" * 64,
            "container_runtime_digest": "2" * 64,
            "verifier_hash": "3" * 64,
            "model_id": adapter.MODEL_ID,
            "model_revision": adapter.MODEL_REVISION,
            "decontamination_status": {
                "status": "verified",
                "receipt_id": f"decont-{suite_id}",
            },
            "budget_receipt": {
                "status": "authorized",
                "authorized": True,
                "maximum_usd": 4,
                "authorization_id": f"budget-{suite_id}",
            },
            "wandb_run_identity": {
                "online": True,
                "entity": "zvf-org",
                "project": "pavlov",
                "group": "offline",
                "run_id": f"wandb-{suite_id}",
                "run_url": f"https://wandb.ai/zvf-org/pavlov/runs/wandb-{suite_id}",
                "mode": "online",
                "status": "finished",
            },
            "tinker_run_identity": {
                "run_id": f"tinker-{suite_id}",
                "cost_status": "authorized",
            },
            "cost_status": "authorized",
            "hf_checkpoints": [
                {
                    "stage": "initial",
                    "run_id": f"tinker-{suite_id}",
                    "revision": "4" * 40,
                    "repo_url": "https://huggingface.co/org/pavlov",
                    "url": "https://huggingface.co/org/pavlov/initial",
                    "visibility": checkpoint_visibility,
                    "safe_public_artifact": checkpoint_visibility == "public",
                },
                {
                    "stage": "periodic",
                    "run_id": f"tinker-{suite_id}",
                    "revision": "5" * 40,
                    "repo_url": "https://huggingface.co/org/pavlov",
                    "url": "https://huggingface.co/org/pavlov/periodic",
                    "visibility": "private",
                    "safe_public_artifact": False,
                },
                {
                    "stage": "final",
                    "run_id": f"tinker-{suite_id}",
                    "revision": "6" * 40,
                    "repo_url": "https://huggingface.co/org/pavlov",
                    "url": "https://huggingface.co/org/pavlov/final",
                    "visibility": "public",
                    "safe_public_artifact": True,
                },
            ],
            "evidence_status": "observed",
        }
        return base

    def _with_complete_receipts(self) -> dict[str, Any]:
        filled = adapter.generate_boundary_bundle()
        for boundary in filled["boundaries"]:
            suite_id = boundary["suite_id"]
            boundary["receipts"] = self._make_filled_receipt(suite_id)
        adapter.update_bundle_signature(filled)
        return filled

    def _with_resigned_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        adapter.update_bundle_signature(bundle)
        return bundle

    def test_generate_boundary_bundle(self) -> None:
        self.assertEqual(self.bundle["schema_version"], adapter.SCHEMA_VERSION)
        self.assertEqual(self.bundle["adapter_id"], adapter.ADAPTER_ID)
        self.assertEqual(self.bundle["suite_count"], 1)
        self.assertEqual(len(self.bundle["boundaries"]), 1)
        self.assertEqual(self.bundle["boundaries"][0]["suite_id"], "agentdojo_train")

    def test_bundle_signature_is_stable(self) -> None:
        self.assertEqual(self.bundle["bundle_signature"], adapter.generate_boundary_bundle()["bundle_signature"])

    def test_validate_complete_boundary_bundle(self) -> None:
        bundle = self._with_complete_receipts()
        errors = adapter.validate_adapter_bundle(bundle)
        self.assertEqual(errors, [])
        result = adapter.evaluate_bundle(bundle)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["errors"], [])
        self.assertIn("agentdojo_train", result["suite_readiness"])

    def test_validate_rejects_schema_drift(self) -> None:
        bad = copy.deepcopy(self._with_complete_receipts())
        bad["schema_version"] = "bad"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertIn("adapter bundle schema_version mismatch", errors)

    def test_validate_rejects_unsupported_suite(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["suite_id"] = "other_suite"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("unsupported suite_id" in error for error in errors))

    def test_validate_requires_train_claim_flags(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["suite_role"] = "primary_eval"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("suite_role must be train" in error for error in errors))
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["primary_eval"] = True
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("primary_eval must be false" in error for error in errors))
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["heldout"] = True
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("heldout must be false" in error for error in errors))

    def test_validate_rejects_mutable_hashes_and_zeroes(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["dataset_or_source_revision"] = "main"
        bad["boundaries"][0]["receipts"]["hf_checkpoints"][0]["revision"] = "0" * 40
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("dataset_or_source_revision must be immutable" in error for error in errors),
            errors,
        )
        self.assertTrue(any("hf_checkpoints[0].revision must be a 40-char revision" in error for error in errors), errors)

    def test_validate_rejects_invalid_task_hash_and_split_hash(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["task_id_hashes"] = ["0" * 64, "1" * 64]
        bad["boundaries"][0]["receipts"]["split_task_id_hash"] = "0" * 64
        bad["boundaries"][0]["receipts"]["split_manifest_hash"] = "0" * 64
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("task_id_hashes[0] must be a SHA-256 digest" in error for error in errors),
            errors,
        )
        self.assertTrue(any("split_task_id_hash must be a SHA-256 digest" in error for error in errors), errors)
        self.assertTrue(any("split_manifest_hash must be a SHA-256 digest" in error for error in errors), errors)

    def test_validate_rejects_duplicate_task_hashes(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["task_id_hashes"] = ["f" * 64, "f" * 64]
        bad["boundaries"][0]["receipts"]["split_task_id_hash"] = adapter.aggregate_task_id_hashes(
            ["f" * 64, "f" * 64]
        )
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("task_id_hashes must be unique" in error for error in errors), errors)

    def test_validate_rejects_disallowed_dataset_substitution(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["dataset_id"] = "UKGovernmentBEIS/inspect_evals/agentharm_eval"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("substitute primary-eval prompts" in error for error in errors),
            errors,
        )

    def test_validate_rejects_authoritative_source_mismatch(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["authoritative_source"]["source_url"] = "https://example.com/other"
        bad["boundaries"][0]["authoritative_source"]["source_id"] = "other/project"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("authoritative_source.source_url must equal contract url" in error for error in errors),
            errors,
        )

    def test_validate_rejects_native_contract_drift(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["native_contract"]["environment"]["name"] = "wrong"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("native_contract.environment.name must be exact" in error for error in errors),
            errors,
        )

    def test_validate_rejects_wandb_offline(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["wandb_run_identity"]["online"] = False
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("wandb_run_identity.online must be true" in error for error in errors), errors)

    def test_validate_rejects_missing_hf_required_stages(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["hf_checkpoints"] = bad["boundaries"][0]["receipts"]["hf_checkpoints"][0:2]
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("missing required stage final" in error for error in errors), errors)

    def test_validate_rejects_public_checkpoint_without_safety(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["hf_checkpoints"][0]["safe_public_artifact"] = False
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("safe_public_artifact must be true for public visibility" in error for error in errors),
            errors,
        )

    def test_validate_rejects_tinker_checkpoint_run_mismatch(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["hf_checkpoints"][0]["run_id"] = "other-run"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("hf checkpoint run_id values must match tinker_run_identity.run_id" in error for error in errors),
            errors,
        )

    def test_validate_rejects_public_checkpoint_without_required_proof(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["dataset_id"] = "salesforce/xlam-function-calling-60k"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(any("substitute primary-eval prompts" in error for error in errors), errors)

    def test_disjointness_from_all_primary_eval_markers(self) -> None:
        bad = self._with_complete_receipts()
        bad["boundaries"][0]["receipts"]["dataset_id"] = "frontiermath_eval"
        self._with_resigned_bundle(bad)
        errors = adapter.validate_adapter_bundle(bad)
        self.assertTrue(
            any("substitute primary-eval prompts" in error for error in errors),
            errors,
        )

    def test_evaluate_bundle_marks_blocked_with_boundary_errors(self) -> None:
        blocked = self._with_complete_receipts()
        blocked["boundaries"][0]["suite_role"] = "primary_eval"
        result = adapter.evaluate_bundle(blocked)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("suite_role must be train", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
