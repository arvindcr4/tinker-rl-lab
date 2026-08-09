#!/usr/bin/env python3
"""Adversarial offline tests for E10 and E14 exact-boundary adapter."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
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


class AgentharmSplitBindingTests(unittest.TestCase):
    """E10 held-out gate: task-ID hashes, split manifest, verifier identity, fail-closed."""

    PUBLIC_BEHAVIORS = {
        "harmful": ["1-1", "1-2"],
        "benign": ["1-1", "2-1"],
        "chat": ["c-1"],
    }
    PRIVATE_BEHAVIORS = {
        "harmful": ["p1-1"],
        "benign": ["p1-1"],
        "chat": ["pc-1"],
    }

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "AgentHarm"
        self._write_split("test_public", self.PUBLIC_BEHAVIORS)
        self.package_root = Path(self._tmp.name) / "agentharm_pkg"
        for relative in adapter.AGENTHARM_VERIFIER_SOURCES:
            path = self.package_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# stub for {relative}\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def _write_split(self, split: str, behaviors: dict[str, list[str]]) -> None:
        for dataset_name, ids in behaviors.items():
            relative = adapter.AGENTHARM_SPLIT_FILES[split][dataset_name]
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"behaviors": [{"id": bid, "prompt": f"prompt {bid}"} for bid in ids]}
            path.write_text(json.dumps(payload), encoding="utf-8")

    def _run(self, **overrides: Any) -> dict[str, Any]:
        run: dict[str, Any] = {
            "label": "agentharm_heldout",
            "is_model_score": True,
            "raw_score": 0.37,
            "model_id": adapter.MODEL_ID,
            "model_revision": adapter.MODEL_REVISION,
            "policy_grader_artifact": {
                "status": "approved",
                "receipt_id": "aisi-policy-grader-001",
            },
        }
        run.update(overrides)
        return run

    # -- split file mapping mirrors the loader ------------------------------

    def test_split_files_match_inspect_evals_loader_mapping(self) -> None:
        self.assertEqual(
            adapter.AGENTHARM_SPLIT_FILES["test_private"],
            {
                "harmful": "benchmark/harmful_behaviors_test_private.json",
                "benign": "benchmark/benign_behaviors_test_private.json",
                "chat": "benchmark/chat_private_test.json",
            },
        )
        self.assertEqual(
            adapter.AGENTHARM_HELDOUT_FILES,
            (
                "benchmark/harmful_behaviors_test_private.json",
                "benchmark/benign_behaviors_test_private.json",
                "benchmark/chat_private_test.json",
            ),
        )

    # -- task ID hashes ------------------------------------------------------

    def test_task_id_hash_is_deterministic_and_domain_separated(self) -> None:
        first = adapter.agentharm_task_id_hash(
            "1-1", dataset_name="harmful", split="test_public"
        )
        again = adapter.agentharm_task_id_hash(
            "1-1", dataset_name="harmful", split="test_public"
        )
        self.assertEqual(first, again)
        self.assertRegex(first, r"^[0-9a-f]{64}$")

        other_split = adapter.agentharm_task_id_hash(
            "1-1", dataset_name="harmful", split="test_private"
        )
        other_dataset = adapter.agentharm_task_id_hash(
            "1-1", dataset_name="benign", split="test_public"
        )
        other_revision = adapter.agentharm_task_id_hash(
            "1-1",
            dataset_name="harmful",
            split="test_public",
            dataset_revision="9" * 40,
        )
        self.assertEqual(4, len({first, other_split, other_dataset, other_revision}))

    def test_task_id_hash_rejects_mutable_revision(self) -> None:
        with self.assertRaises(ValueError):
            adapter.agentharm_task_id_hash(
                "1-1", dataset_name="harmful", split="test_public", dataset_revision="main"
            )

    def test_task_id_hash_rejects_unknown_split_and_dataset(self) -> None:
        with self.assertRaises(ValueError):
            adapter.agentharm_task_id_hash("1-1", dataset_name="harmful", split="test_holdout")
        with self.assertRaises(ValueError):
            adapter.agentharm_task_id_hash("1-1", dataset_name="gsm8k", split="test_public")

    # -- split manifest ------------------------------------------------------

    def test_public_split_manifest_is_complete_and_bound(self) -> None:
        manifest = adapter.build_agentharm_split_manifest(self.root, "test_public")
        self.assertTrue(manifest["complete"])
        self.assertEqual([], manifest["missing_files"])
        self.assertFalse(manifest["heldout"])
        self.assertEqual(5, manifest["task_count"])
        self.assertEqual(
            adapter.aggregate_task_id_hashes(manifest["task_id_hashes"]),
            manifest["split_task_id_hash"],
        )
        self.assertRegex(manifest["split_manifest_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(adapter.AGENTHARM_DATASET_REVISION, manifest["dataset_revision"])

    def test_split_manifest_hash_is_path_independent(self) -> None:
        first = adapter.build_agentharm_split_manifest(self.root, "test_public")
        moved = Path(self._tmp.name) / "copy" / "AgentHarm"
        shutil.copytree(self.root, moved)
        second = adapter.build_agentharm_split_manifest(moved, "test_public")
        self.assertEqual(first["split_manifest_hash"], second["split_manifest_hash"])

    def test_split_manifest_hash_changes_when_content_changes(self) -> None:
        before = adapter.build_agentharm_split_manifest(self.root, "test_public")
        self._write_split("test_public", {**self.PUBLIC_BEHAVIORS, "chat": ["c-1", "c-2"]})
        after = adapter.build_agentharm_split_manifest(self.root, "test_public")
        self.assertNotEqual(before["split_manifest_hash"], after["split_manifest_hash"])
        self.assertNotEqual(before["split_task_id_hash"], after["split_task_id_hash"])

    def test_private_split_manifest_reports_all_three_missing_files(self) -> None:
        manifest = adapter.build_agentharm_split_manifest(self.root, "test_private")
        self.assertFalse(manifest["complete"])
        self.assertEqual(list(adapter.AGENTHARM_HELDOUT_FILES), manifest["missing_files"])
        self.assertEqual(0, manifest["task_count"])
        self.assertIsNone(manifest["split_task_id_hash"])
        self.assertTrue(manifest["heldout"])
        self.assertEqual('split="test_private"', manifest["loader_split_flag"])

    # -- verifier identity ---------------------------------------------------

    def test_verifier_identity_is_complete_and_hashed(self) -> None:
        identity = adapter.agentharm_verifier_identity(self.package_root)
        self.assertTrue(identity["complete"])
        self.assertEqual([], identity["missing_sources"])
        self.assertRegex(identity["verifier_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(adapter.AGENTHARM_HARNESS_REVISION, identity["harness_revision"])

    def test_verifier_hash_changes_when_a_grading_source_changes(self) -> None:
        before = adapter.agentharm_verifier_identity(self.package_root)
        (self.package_root / "benchmark/harmful_grading_functions.py").write_text(
            "# tampered\n", encoding="utf-8"
        )
        after = adapter.agentharm_verifier_identity(self.package_root)
        self.assertNotEqual(before["verifier_hash"], after["verifier_hash"])

    def test_verifier_identity_reports_missing_sources(self) -> None:
        (self.package_root / "scorer.py").unlink()
        identity = adapter.agentharm_verifier_identity(self.package_root)
        self.assertFalse(identity["complete"])
        self.assertIn("scorer.py", identity["missing_sources"])

    # -- held-out availability ----------------------------------------------

    def test_heldout_check_is_blocked_without_private_files(self) -> None:
        availability = adapter.check_heldout_split_available(self.root)
        self.assertFalse(availability["available"])
        self.assertEqual(list(adapter.AGENTHARM_HELDOUT_FILES), availability["missing_files"])

    def test_heldout_check_requires_all_three_files(self) -> None:
        self._write_split("test_private", {"harmful": ["p1-1"]})
        availability = adapter.check_heldout_split_available(self.root)
        self.assertFalse(availability["available"])
        self.assertEqual(2, len(availability["missing_files"]))
        self.assertEqual(1, len(availability["present_files"]))

    def test_heldout_check_passes_once_all_three_present(self) -> None:
        self._write_split("test_private", self.PRIVATE_BEHAVIORS)
        availability = adapter.check_heldout_split_available(self.root)
        self.assertTrue(availability["available"])
        self.assertEqual([], availability["missing_files"])

    # -- fail-closed emitter -------------------------------------------------

    def test_emitter_blocks_score_when_private_files_absent(self) -> None:
        receipt = adapter.emit_agentharm_score(
            split_manifest=adapter.build_agentharm_split_manifest(self.root, "test_private"),
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(),
        )
        self.assertIsNone(receipt["score"])
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertFalse(receipt["is_model_score"])
        self.assertTrue(
            any("held-out split files are absent" in b for b in receipt["blockers"]),
            receipt["blockers"],
        )

    def test_emitter_never_promotes_a_public_split_run(self) -> None:
        receipt = adapter.emit_agentharm_score(
            split_manifest=adapter.build_agentharm_split_manifest(self.root, "test_public"),
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(raw_score=0.99),
        )
        self.assertIsNone(receipt["score"])
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertTrue(
            any("is not the held-out split" in b for b in receipt["blockers"]),
            receipt["blockers"],
        )

    def test_emitter_never_promotes_harness_validation(self) -> None:
        self._write_split("test_private", self.PRIVATE_BEHAVIORS)
        receipt = adapter.emit_agentharm_score(
            split_manifest=adapter.build_agentharm_split_manifest(self.root, "test_private"),
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(
                label="harness_validation", is_model_score=False, raw_score=1.0
            ),
        )
        self.assertIsNone(receipt["score"])
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertTrue(
            any("harness validation may never be promoted" in b for b in receipt["blockers"]),
            receipt["blockers"],
        )

    def test_emitter_blocks_without_approved_policy_grader_artifact(self) -> None:
        self._write_split("test_private", self.PRIVATE_BEHAVIORS)
        receipt = adapter.emit_agentharm_score(
            split_manifest=adapter.build_agentharm_split_manifest(self.root, "test_private"),
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(policy_grader_artifact=None),
        )
        self.assertIsNone(receipt["score"])
        self.assertTrue(
            any("policy-grader artifact receipt is missing" in b for b in receipt["blockers"]),
            receipt["blockers"],
        )

    def test_emitter_blocks_when_verifier_sources_are_missing(self) -> None:
        self._write_split("test_private", self.PRIVATE_BEHAVIORS)
        (self.package_root / "benchmark/harmful_grading_functions.py").unlink()
        receipt = adapter.emit_agentharm_score(
            split_manifest=adapter.build_agentharm_split_manifest(self.root, "test_private"),
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(),
        )
        self.assertIsNone(receipt["score"])
        self.assertTrue(
            any("verifier identity is incomplete" in b for b in receipt["blockers"]),
            receipt["blockers"],
        )

    def test_emitter_blocks_on_tampered_split_task_id_hash(self) -> None:
        self._write_split("test_private", self.PRIVATE_BEHAVIORS)
        manifest = adapter.build_agentharm_split_manifest(self.root, "test_private")
        manifest["split_task_id_hash"] = "e" * 64
        receipt = adapter.emit_agentharm_score(
            split_manifest=manifest,
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(),
        )
        self.assertIsNone(receipt["score"])
        self.assertTrue(
            any("does not match task_id_hashes" in b for b in receipt["blockers"]),
            receipt["blockers"],
        )

    def test_emitter_raises_when_asked_to_fail_loudly(self) -> None:
        with self.assertRaises(adapter.HeldoutSplitUnavailable):
            adapter.emit_agentharm_score(
                split_manifest=adapter.build_agentharm_split_manifest(
                    self.root, "test_private"
                ),
                verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
                heldout_availability=adapter.check_heldout_split_available(self.root),
                run=self._run(),
                raise_on_block=True,
            )

    def test_emitter_releases_score_only_when_every_gate_passes(self) -> None:
        self._write_split("test_private", self.PRIVATE_BEHAVIORS)
        receipt = adapter.emit_agentharm_score(
            split_manifest=adapter.build_agentharm_split_manifest(self.root, "test_private"),
            verifier_identity=adapter.agentharm_verifier_identity(self.package_root),
            heldout_availability=adapter.check_heldout_split_available(self.root),
            run=self._run(),
        )
        self.assertEqual([], receipt["blockers"])
        self.assertEqual("COMPLETE", receipt["status"])
        self.assertTrue(receipt["is_model_score"])
        self.assertEqual(0.37, receipt["score"])
        self.assertEqual("test_private", receipt["split"])
        self.assertEqual(3, receipt["task_count"])


if __name__ == "__main__":
    unittest.main()
