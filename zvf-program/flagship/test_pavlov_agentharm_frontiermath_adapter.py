#!/usr/bin/env python3
"""Adversarial offline tests for E10 and E14 exact-boundary adapter."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from flagship import pavlov_agentharm_frontiermath_adapter as adapter


class PavlovAgentharmFrontiermathAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = adapter.generate_boundary_bundle()

    def _make_filled_receipt(self, suite_id: str, *, include_heldout_receipt: bool = True) -> dict:
        task_id_hashes = [
            "a" * 64,
            "b" * 64,
            "c" * 64,
        ]
        base = {
            "dataset_id": f"official-{suite_id}",
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
                    "visibility": "public",
                    "safe_public_artifact": True,
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
            "heldout_receipt": None,
        }
        if include_heldout_receipt:
            base["heldout_receipt"] = {
                "status": "verified",
                "receipt_id": f"heldout-{suite_id}",
            }
        return base

    def _with_complete_receipts(self, include_heldout_receipt: bool = True) -> dict:
        filled = adapter.generate_boundary_bundle()
        for boundary in filled["boundaries"]:
            suite_id = boundary["suite_id"]
            boundary["receipts"] = self._make_filled_receipt(
                suite_id,
                include_heldout_receipt=include_heldout_receipt,
            )
        adapter.update_bundle_signature(filled)
        return filled

    def _with_resigned_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        adapter.update_bundle_signature(bundle)
        return bundle

    def test_generate_boundary_bundle(self) -> None:
        bundle = self.bundle
        self.assertEqual(bundle["schema_version"], adapter.SCHEMA_VERSION)
        self.assertEqual(bundle["adapter_id"], adapter.ADAPTER_ID)
        self.assertEqual(bundle["suite_count"], 2)
        self.assertEqual(len(bundle["boundaries"]), 2)
        self.assertEqual(
            sorted(item["suite_id"] for item in bundle["boundaries"]),
            list(adapter.SUPPORTED_SUITE_IDS),
        )

    def test_bundle_signature_is_stable(self) -> None:
        bundle = self.bundle
        rerun = adapter.generate_boundary_bundle()
        self.assertEqual(
            bundle["bundle_signature"],
            rerun["bundle_signature"],
        )

    def test_validate_complete_boundary_bundle(self) -> None:
        bundle = self._with_complete_receipts()
        errors = adapter.validate_adapter_bundle(bundle)
        self.assertEqual(errors, [])
        result = adapter.evaluate_bundle(bundle)
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["errors"], [])

    def test_generate_bundle_boundaries_match_contract(self) -> None:
        for boundary in self.bundle["boundaries"]:
            self.assertEqual(boundary["suite_role"], "primary_eval")
            self.assertFalse(boundary["component_only"])
            self.assertEqual(boundary["schema_version"], adapter.SCHEMA_VERSION)
            self.assertEqual(boundary["adapter_id"], adapter.ADAPTER_ID)
            self.assertTrue(boundary["heldout"])

    def test_validate_rejects_schema_drift(self) -> None:
        broken = copy.deepcopy(self.bundle)
        broken["schema_version"] = "bad"
        errors = adapter.validate_adapter_bundle(broken)
        self.assertIn("adapter bundle schema_version mismatch", errors)

    def test_validate_rejects_unsupported_suite(self) -> None:
        broken = copy.deepcopy(self._with_complete_receipts())
        broken["boundaries"][0]["suite_id"] = "other_eval"
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(any("unsupported suite_id" in error for error in errors))

    def test_validate_rejects_zeroed_hard_immutable_hashes(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["receipts"]["dataset_or_source_revision"] = "0" * 40
        broken["boundaries"][0]["receipts"]["hf_checkpoints"][0]["revision"] = "0" * 40
        broken["boundaries"][0]["receipts"]["container_runtime_digest"] = "0" * 64
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("dataset_or_source_revision must be immutable" in error for error in errors),
            errors,
        )
        self.assertTrue(any("hf_checkpoints[0].revision" in error for error in errors), errors)
        self.assertTrue(any("container_runtime_digest must be a SHA-256 digest" in error for error in errors), errors)

    def test_validate_rejects_zeroed_task_and_manifest_hashes(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["receipts"]["task_id_hashes"] = ["0" * 64, "1" * 64]
        broken["boundaries"][0]["receipts"]["split_task_id_hash"] = "0" * 64
        broken["boundaries"][0]["receipts"]["split_manifest_hash"] = "0" * 64
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("task_id_hashes[0] must be a SHA-256 digest" in error for error in errors),
            errors,
        )
        self.assertTrue(any("split_task_id_hash must be a SHA-256 digest" in error for error in errors), errors)
        self.assertTrue(any("split_manifest_hash must be a SHA-256 digest" in error for error in errors), errors)

    def test_validate_rejects_duplicate_task_hashes(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["receipts"]["task_id_hashes"] = ["f" * 64, "f" * 64]
        broken["boundaries"][0]["receipts"]["split_task_id_hash"] = adapter.aggregate_task_id_hashes(
            ["f" * 64, "f" * 64]
        )
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(any("task_id_hashes must be unique" in error for error in errors), errors)

    def test_validate_rejects_authoritative_source_mismatch(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["authoritative_source"]["source_url"] = "https://github.com/UKGovernmentBEIS/inspect_evals"
        broken["boundaries"][0]["authoritative_source"]["source_id"] = "uk/inspect-evals"
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("authoritative_source.source_url must equal contract url" in error for error in errors),
            errors,
        )

    def test_validate_rejects_dataset_substitution_markers(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["receipts"]["dataset_id"] = "openr1_math"
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("substitute xLAM" in error for error in errors),
            errors,
        )

    def test_validate_split_hashes_are_deterministic(self) -> None:
        task_hashes = ["1" * 64, "2" * 64]
        expected = adapter.aggregate_task_id_hashes(task_hashes)
        actual = adapter.aggregate_task_id_hashes(task_hashes)
        self.assertEqual(expected, actual)
        different = adapter.aggregate_task_id_hashes(task_hashes[::-1])
        self.assertNotEqual(expected, different)

    def test_validate_rejects_split_task_id_mismatch(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["receipts"]["split_task_id_hash"] = "a" * 64
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("does not match task_id_hashes" in error for error in errors),
            errors,
        )

    def test_validate_requires_wandb_online(self) -> None:
        broken = self._with_complete_receipts()
        boundary = broken["boundaries"][0]
        boundary["receipts"]["wandb_run_identity"]["online"] = False
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(any("online must be true" in error for error in errors))

    def test_validate_requires_hf_checkpoint_stages(self) -> None:
        broken = self._with_complete_receipts()
        boundary = broken["boundaries"][0]
        boundary["receipts"]["hf_checkpoints"] = boundary["receipts"]["hf_checkpoints"][0:2]
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(any("missing required stage final" in error for error in errors), errors)

    def test_validate_rejects_public_checkpoint_without_safety(self) -> None:
        broken = self._with_complete_receipts()
        boundary = broken["boundaries"][0]
        boundary["receipts"]["hf_checkpoints"][0]["safe_public_artifact"] = False
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("safe_public_artifact must be true" in error for error in errors),
            errors,
        )

    def test_validate_rejects_tinker_and_checkpoint_id_mismatch(self) -> None:
        broken = self._with_complete_receipts()
        boundary = broken["boundaries"][1]
        boundary["receipts"]["hf_checkpoints"][0]["run_id"] = "other-run"
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("run_id values must match tinker_run_identity.run_id" in error for error in errors),
            errors,
        )

    def test_validate_requires_heldout_receipt_when_structural_heldout(self) -> None:
        broken = self._with_complete_receipts(include_heldout_receipt=False)
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(any("requires heldout_receipt" in error for error in errors), errors)

    def test_evaluate_bundle_marks_heldout_claim_blocked(self) -> None:
        blocked = self._with_complete_receipts(include_heldout_receipt=False)
        result = adapter.evaluate_bundle(blocked)
        self.assertEqual(result["status"], "BLOCKED")
        e10 = result["suite_readiness"]["agentharm_eval"]
        self.assertTrue(e10["structural_heldout"])
        self.assertFalse(e10["heldout_receipt_proven"])
        self.assertFalse(e10["primary_eval_claim_allowed"])

    def test_validate_rejects_unknown_benchmark_marker_in_source_reuse(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][1]["receipts"]["dataset_id"] = "frontiermath_math_500"
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(any("substitute xLAM" in error for error in errors), errors)

    def test_validate_rejects_duplicate_hf_checkpoint_stages(self) -> None:
        broken = self._with_complete_receipts()
        broken["boundaries"][0]["receipts"]["hf_checkpoints"][1]["stage"] = "initial"
        self._with_resigned_bundle(broken)
        errors = adapter.validate_adapter_bundle(broken)
        self.assertTrue(
            any("duplicate entries for stage initial" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
