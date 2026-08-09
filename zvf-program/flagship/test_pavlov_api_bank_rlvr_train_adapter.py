from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_api_bank_rlvr_train_adapter import (
    DATASET_ID,
    EXCLUDED_DATASET_ID,
    EXCLUDED_SUITE_ID,
    ROLE,
    SOURCE_URL,
    SUITE_ID,
    ApiBankRLVRTrainBoundaryError,
    build_api_bank_rlvr_train_boundary,
    exclusion_manifest_hash,
    main,
    split_manifest_hash,
    task_id_hash,
    validate_boundary,
    verify_boundary,
)


REVISION = "a" * 40
ENVIRONMENT_REVISION = "b" * 40
VERIFIER_REVISION = "c" * 40


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def environment_contract() -> dict[str, object]:
    return {
        "environment_id": "api-bank-finance-sandbox-native-v1",
        "environment_revision": ENVIRONMENT_REVISION,
        "native": True,
        "stateful": True,
        "sandboxed": True,
        "network_access": False,
        "reset_protocol": "fresh-account-state-and-deterministic-seed",
        "api_surface": "finance-api-sandbox-native",
        "finance_api_sandbox": True,
    }


def artifact_contract() -> dict[str, object]:
    return {
        "required": True,
        "artifact_types": ["api_call_trace", "state_transition_receipt"],
        "artifact_receipt_ref": "receipt://artifact/" + digest("artifact"),
        "side_effect_receipt_required": True,
    }


def verifier_contract() -> dict[str, object]:
    return {
        "verifier_id": "api-bank-native-finance-verifier",
        "verifier_revision": VERIFIER_REVISION,
        "native": True,
        "deterministic": True,
        "finance_api_sandbox": True,
        "checks": ["api_schema", "state_transition", "artifact", "final_answer"],
        "verifier_receipt_ref": "receipt://verifier/" + digest("verifier"),
    }


def budget_gate() -> dict[str, object]:
    return {
        "status": "AUTHORIZED_TINKER_ONLY",
        "provider": "Tinker",
        "paid_jobs_may_launch": True,
        "maximum_usd": 18.0,
        "operational_cap_usd": 16.5,
        "safety_reserve_usd": 1.5,
        "authorized_at": "2026-08-09",
    }


def result_receipts() -> dict[str, object]:
    return {
        "wandb": {
            "run_id": "wandb-run-t11-001",
            "entity": "offline-fixture",
            "project": "pavlov-t11",
            "run_url": "https://wandb.example/runs/t11-001",
            "summary_sha256": digest("wandb-summary"),
            "history_sha256": digest("wandb-history"),
            "config_sha256": digest("wandb-config"),
        },
        "tinker": {
            "job_id": "tinker-job-t11-001",
            "model_revision": "d" * 40,
            "adapter_revision": "e" * 40,
            "sampling_receipt_sha256": digest("tinker-sampling"),
            "result_receipt_sha256": digest("tinker-result"),
        },
        "hf": {
            "model_id": "offline/api-bank-model",
            "model_revision": "f" * 40,
            "dataset_id": DATASET_ID,
            "dataset_revision": REVISION,
            "artifact_url": "https://huggingface.example/artifacts/t11-001",
            "artifact_sha256": digest("hf-artifact"),
            "result_receipt_sha256": digest("hf-result"),
        },
    }


def exclusion_hashes() -> list[str]:
    return sorted(
        (
            digest("banker/heldout/001"),
            digest("banker/heldout/002"),
        )
    )


def build_valid(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "revision": REVISION,
        "license_id": "Apache-2.0",
        "license_receipt_ref": "sha256:" + digest("license"),
        "tasks": [
            {"task_id": "api-bank/train/002"},
            {"task_id": "api-bank/train/001"},
            {"task_id": "api-bank/train/003"},
        ],
        "heldout_task_id_hashes": exclusion_hashes(),
        "heldout_exclusion_receipt_ref": "receipt://banker-exclusion/" + digest("exclusion"),
        "environment_contract": environment_contract(),
        "artifact_contract": artifact_contract(),
        "verifier_contract": verifier_contract(),
        "budget_gate": budget_gate(),
        "result_receipts": result_receipts(),
    }
    arguments.update(overrides)
    return build_api_bank_rlvr_train_boundary(**arguments)


class PavlovApiBankRLVRTrainAdapterTests(unittest.TestCase):
    def test_authoritative_train_identity_hashes_and_disjointness(self) -> None:
        boundary = build_valid()
        self.assertEqual(boundary["status"], "READY")
        self.assertEqual(boundary["suite_id"], SUITE_ID)
        self.assertEqual(boundary["role"], ROLE)
        self.assertEqual(boundary["source_identity"]["dataset_id"], DATASET_ID)
        self.assertEqual(boundary["source_identity"]["url"], SOURCE_URL)
        expected_hashes = [
            task_id_hash("api-bank/train/001"),
            task_id_hash("api-bank/train/002"),
            task_id_hash("api-bank/train/003"),
        ]
        self.assertEqual(boundary["task_id_hashes"], expected_hashes)
        self.assertEqual(boundary["split_manifest_sha256"], split_manifest_hash(expected_hashes))
        self.assertEqual(boundary["exclusion_boundary"]["suite_id"], EXCLUDED_SUITE_ID)
        self.assertEqual(boundary["exclusion_boundary"]["dataset_id"], EXCLUDED_DATASET_ID)
        self.assertEqual(boundary["exclusion_boundary"]["split_manifest_sha256"], exclusion_manifest_hash(exclusion_hashes()))
        self.assertTrue(boundary["exclusion_boundary"]["receipt_proven"])
        self.assertFalse(boundary["launch_authorized"])
        self.assertFalse(boundary["launches_any_job"])
        self.assertTrue(verify_boundary(boundary))
        rendered = json.dumps(boundary, sort_keys=True).lower()
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"target"', rendered)
        self.assertIsNone(boundary["result_claims"])
        self.assertIsNone(boundary["training_claims"])

    def test_task_and_exclusion_order_are_deterministic(self) -> None:
        first = build_valid(tasks=["z", "a", "m"], heldout_task_id_hashes=list(reversed(exclusion_hashes())))
        second = build_valid(tasks=["m", "z", "a"], heldout_task_id_hashes=exclusion_hashes())
        self.assertEqual(first, second)
        changed = copy.deepcopy(first)
        changed["task_id_hashes"][0] = digest("mutated")
        self.assertTrue(any("task_id_aggregate" in error for error in validate_boundary(changed)))
        self.assertTrue(any("split_manifest" in error for error in validate_boundary(changed)))
        changed = copy.deepcopy(first)
        changed["exclusion_boundary"]["task_id_hashes"][0] = digest("mutated-exclusion")
        self.assertTrue(any("excluded" in error for error in validate_boundary(changed)))

    def test_train_overlap_with_e4_heldout_hashes_is_hard_failure(self) -> None:
        overlapping = [task_id_hash("api-bank/train/001"), exclusion_hashes()[0]]
        boundary = build_valid(tasks=["api-bank/train/001"], heldout_task_id_hashes=overlapping)
        self.assertEqual(boundary["status"], "BLOCKED")
        self.assertTrue(any("overlap BankerToolBench" in error for error in boundary["errors"]))
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "overlap"):
            verify_boundary(boundary)

    def test_missing_or_unproven_exclusion_blocks(self) -> None:
        missing = build_valid(heldout_task_id_hashes=None, heldout_exclusion_receipt_ref=None)
        self.assertEqual(missing["status"], "BLOCKED")
        self.assertTrue(any("disjointness" in error for error in missing["errors"]))
        bad_receipt = build_valid(heldout_exclusion_receipt_ref="UNRECORDED")
        self.assertEqual(bad_receipt["status"], "BLOCKED")
        self.assertFalse(bad_receipt["exclusion_boundary"]["receipt_proven"])
        self.assertTrue(any("exclusion receipt_ref" in error for error in bad_receipt["errors"]))
        wrong_manifest = build_valid(
            exclusion_manifest={
                "suite_id": "other_eval",
                "dataset_id": EXCLUDED_DATASET_ID,
                "task_id_hashes": exclusion_hashes(),
                "receipt_ref": "receipt://exclusion/" + digest("exclusion"),
            }
        )
        self.assertEqual(wrong_manifest["status"], "BLOCKED")

    def test_e4_xlam_and_related_substitutes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "authoritative API-Bank"):
            build_valid(
                source_identity={
                    "kind": "huggingface_dataset",
                    "dataset_id": EXCLUDED_DATASET_ID,
                    "url": "https://huggingface.co/datasets/handshake-ai-research/bankertoolbench",
                }
            )
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "xLAM"):
            build_valid(
                source_identity={
                    "kind": "huggingface_dataset",
                    "dataset_id": "Salesforce/xlam-function-calling-60k",
                    "url": SOURCE_URL,
                }
            )
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "related benchmarks"):
            build_valid(related_benchmark="BankerToolBench")
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "role must be train"):
            build_valid(role="primary_eval")

    def test_revision_license_and_role_are_pinned(self) -> None:
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "immutable 40-character"):
            build_valid(revision="main")
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "license_id"):
            build_valid(license_id="UNRECORDED")
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "license_receipt_ref"):
            build_valid(license_receipt_ref="latest")
        duplicate = build_valid(tasks=["same", "same"])
        self.assertEqual(duplicate["status"], "BLOCKED")
        self.assertTrue(any("duplicate" in error for error in duplicate["errors"]))
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "metadata-only"):
            build_valid(tasks=[{"prompt": "raw finance task"}])

    def test_finance_native_sandbox_contract_mutations_fail_closed(self) -> None:
        bad_environment = environment_contract()
        bad_environment["network_access"] = True
        bad_environment["sandboxed"] = False
        boundary = build_valid(environment_contract=bad_environment)
        self.assertEqual(boundary["status"], "BLOCKED")
        errors = "\n".join(boundary["errors"])
        self.assertIn("network_access=false", errors)
        self.assertIn("sandboxed=true", errors)
        bad_verifier = verifier_contract()
        bad_verifier["deterministic"] = False
        bad_verifier["finance_api_sandbox"] = False
        bad_verifier["verifier_revision"] = "latest"
        boundary = build_valid(verifier_contract=bad_verifier)
        errors = "\n".join(boundary["errors"])
        self.assertIn("deterministic=true", errors)
        self.assertIn("finance_api_sandbox=true", errors)
        self.assertIn("verifier_revision", errors)
        bad_artifact = artifact_contract()
        bad_artifact["side_effect_receipt_required"] = False
        bad_artifact["artifact_receipt_ref"] = "receipt://artifact/descriptive"
        boundary = build_valid(artifact_contract=bad_artifact)
        self.assertTrue(any("artifact" in error for error in boundary["errors"]))

    def test_budget_gate_is_required_and_never_launches(self) -> None:
        missing = build_valid(budget_gate=None)
        self.assertEqual(missing["status"], "BLOCKED")
        self.assertFalse(missing["launch_authorized"])
        bad_provider = budget_gate()
        bad_provider["provider"] = "OpenAI"
        boundary = build_valid(budget_gate=bad_provider)
        self.assertTrue(any("provider must be Tinker" in error for error in boundary["errors"]))
        too_large = budget_gate()
        too_large["maximum_usd"] = 18.01
        boundary = build_valid(budget_gate=too_large)
        self.assertTrue(any("18.0 USD cap" in error for error in boundary["errors"]))
        bad_reserve = budget_gate()
        bad_reserve["safety_reserve_usd"] = -1
        boundary = build_valid(budget_gate=bad_reserve)
        self.assertTrue(any("safety_reserve" in error for error in boundary["errors"]))
        launch_override = budget_gate()
        launch_override["launch_authorized"] = True
        boundary = build_valid(budget_gate=launch_override)
        self.assertTrue(any("cannot authorize" in error for error in boundary["errors"]))

    def test_result_receipts_are_required_without_fabricating_claims(self) -> None:
        boundary = build_valid(result_receipts=None)
        self.assertEqual(boundary["status"], "BLOCKED")
        self.assertEqual(boundary["result_receipts"]["status"], "UNRECORDED")
        self.assertFalse(boundary["result_receipts"]["recorded"])
        self.assertIsNone(boundary["result_claims"])
        invalid = result_receipts()
        invalid["hf"]["dataset_revision"] = "f" * 40
        invalid["tinker"]["adapter_revision"] = "mutable"
        boundary = build_valid(result_receipts=invalid)
        errors = "\n".join(boundary["errors"])
        self.assertIn("dataset_revision must match", errors)
        self.assertIn("tinker.adapter_revision", errors)
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "unrecorded|dataset_revision"):
            verify_boundary(boundary)

    def test_metadata_mutations_and_raw_content_are_detected(self) -> None:
        boundary = build_valid()
        mutated = copy.deepcopy(boundary)
        mutated["source_identity"]["dataset_id"] = EXCLUDED_DATASET_ID
        self.assertTrue(any("authoritative API-Bank" in error for error in validate_boundary(mutated)))
        mutated = copy.deepcopy(boundary)
        mutated["budget_gate"]["launch_authorized"] = True
        self.assertTrue(any("budget_gate metadata" in error for error in validate_boundary(mutated)))
        mutated = copy.deepcopy(boundary)
        mutated["exclusion_boundary"]["receipt_ref"] = "descriptive receipt"
        self.assertTrue(any("exclusion receipt_ref" in error for error in validate_boundary(mutated)))
        mutated = copy.deepcopy(boundary)
        mutated["disjoint_from"]["suite_id"] = "other_eval"
        self.assertTrue(any("disjoint_from" in error for error in validate_boundary(mutated)))
        mutated = copy.deepcopy(boundary)
        mutated["split_semantics"]["held_out"] = True
        self.assertTrue(any("train-only" in error for error in validate_boundary(mutated)))
        with self.assertRaisesRegex(ApiBankRLVRTrainBoundaryError, "metadata-only"):
            build_valid(environment_contract={**environment_contract(), "target": "raw"})

    def test_alias_inputs_and_exclusion_manifest_are_supported(self) -> None:
        hashes = exclusion_hashes()
        manifest = {
            "suite_id": EXCLUDED_SUITE_ID,
            "dataset_id": EXCLUDED_DATASET_ID,
            "task_id_hashes": hashes,
            "receipt_ref": "receipt://banker-exclusion/" + digest("exclusion"),
        }
        boundary = build_valid(
            heldout_task_id_hashes=None,
            exclusion_manifest=manifest,
        )
        self.assertEqual(boundary["status"], "READY")
        alias = build_valid(
            heldout_task_id_hashes=None,
            excluded_task_id_hashes=hashes,
            exclusion_receipt_ref=manifest["receipt_ref"],
        )
        self.assertEqual(alias["status"], "READY")

    def test_local_cli_generate_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "environment": root / "environment.json",
                "artifact": root / "artifact.json",
                "verifier": root / "verifier.json",
                "budget": root / "budget.json",
                "results": root / "results.json",
                "boundary": root / "boundary.json",
            }
            for name, value in (
                ("environment", environment_contract()),
                ("artifact", artifact_contract()),
                ("verifier", verifier_contract()),
                ("budget", budget_gate()),
                ("results", result_receipts()),
            ):
                paths[name].write_text(json.dumps(value), encoding="utf-8")
            args = [
                "generate",
                "--revision",
                REVISION,
                "--license-id",
                "Apache-2.0",
                "--license-receipt-ref",
                "sha256:" + digest("license"),
                "--task-id",
                "api-bank/train/001",
                "--task-id",
                "api-bank/train/002",
                "--heldout-task-id-hash",
                exclusion_hashes()[0],
                "--heldout-task-id-hash",
                exclusion_hashes()[1],
                "--heldout-exclusion-receipt-ref",
                "receipt://banker-exclusion/" + digest("exclusion"),
                "--environment-contract",
                str(paths["environment"]),
                "--artifact-contract",
                str(paths["artifact"]),
                "--verifier-contract",
                str(paths["verifier"]),
                "--budget-gate",
                str(paths["budget"]),
                "--result-receipts",
                str(paths["results"]),
                "--out",
                str(paths["boundary"]),
            ]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(["verify", "--boundary", str(paths["boundary"])]), 0)


if __name__ == "__main__":
    unittest.main()
