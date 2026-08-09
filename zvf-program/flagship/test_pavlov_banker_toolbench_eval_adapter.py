from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_banker_toolbench_eval_adapter import (
    DATASET_ID,
    ROLE,
    SOURCE_URL,
    SUITE_ID,
    BankerToolBenchBoundaryError,
    build_banker_toolbench_eval_boundary,
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
        "environment_id": "banker-toolbench-native-v1",
        "environment_revision": ENVIRONMENT_REVISION,
        "native": True,
        "stateful": True,
        "reset_protocol": "fresh-episode-and-deterministic-seed",
        "tool_api": "authoritative-banker-toolbench-native-api",
    }


def artifact_contract() -> dict[str, object]:
    return {
        "required": True,
        "artifact_types": ["state_transition_receipt", "banking_action_trace"],
        "artifact_receipt_ref": "receipt://artifact/" + digest("artifact"),
    }


def verifier_contract() -> dict[str, object]:
    return {
        "verifier_id": "banker-toolbench-native-verifier",
        "verifier_revision": VERIFIER_REVISION,
        "deterministic": True,
        "checks": ["tool_call_validity", "state_transition", "artifact", "final_answer"],
        "verifier_receipt_ref": "receipt://verifier/" + digest("verifier"),
    }


def result_receipts() -> dict[str, object]:
    return {
        "wandb": {
            "run_id": "wandb-run-e4-001",
            "entity": "offline-fixture",
            "project": "pavlov-e4",
            "run_url": "https://wandb.example/runs/e4-001",
            "summary_sha256": digest("wandb-summary"),
            "history_sha256": digest("wandb-history"),
            "config_sha256": digest("wandb-config"),
        },
        "tinker": {
            "job_id": "tinker-job-e4-001",
            "model_revision": "d" * 40,
            "adapter_revision": "e" * 40,
            "sampling_receipt_sha256": digest("tinker-sampling"),
            "result_receipt_sha256": digest("tinker-result"),
        },
        "hf": {
            "model_id": "offline/model",
            "model_revision": "f" * 40,
            "dataset_id": DATASET_ID,
            "dataset_revision": REVISION,
            "artifact_url": "https://huggingface.example/artifacts/e4",
            "artifact_sha256": digest("hf-artifact"),
            "result_receipt_sha256": digest("hf-result"),
        },
    }


def build_valid(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "revision": REVISION,
        "license_id": "CC-BY-4.0",
        "license_receipt_ref": "sha256:" + digest("license"),
        "tasks": [
            {"task_id": "banker/task/002"},
            {"task_id": "banker/task/001"},
            {"task_id": "banker/task/003"},
        ],
        "environment_contract": environment_contract(),
        "artifact_contract": artifact_contract(),
        "verifier_contract": verifier_contract(),
        "result_receipts": result_receipts(),
    }
    arguments.update(overrides)
    return build_banker_toolbench_eval_boundary(**arguments)


class PavlovBankerToolBenchEvalAdapterTests(unittest.TestCase):
    def test_authoritative_identity_and_ready_boundary_are_metadata_only(self) -> None:
        boundary = build_valid()
        self.assertEqual(boundary["status"], "READY")
        self.assertEqual(boundary["suite_id"], SUITE_ID)
        self.assertEqual(boundary["role"], ROLE)
        self.assertEqual(boundary["source_identity"]["dataset_id"], DATASET_ID)
        self.assertEqual(boundary["source_identity"]["url"], SOURCE_URL)
        self.assertEqual(boundary["task_count"], 3)
        expected_hashes = [
            task_id_hash("banker/task/001"),
            task_id_hash("banker/task/002"),
            task_id_hash("banker/task/003"),
        ]
        self.assertEqual(boundary["task_id_hashes"], expected_hashes)
        self.assertEqual(boundary["split_manifest_sha256"], split_manifest_hash(expected_hashes))
        self.assertTrue(verify_boundary(boundary))
        rendered = json.dumps(boundary, sort_keys=True)
        self.assertNotIn('"prompt"', rendered)
        self.assertNotIn('"target"', rendered)
        self.assertIsNone(boundary["result_claims"])

    def test_task_order_is_canonical_and_hash_mutations_fail(self) -> None:
        first = build_valid(tasks=["z", "a", "m"])
        second = build_valid(tasks=["m", "z", "a"])
        self.assertEqual(first, second)
        changed = copy.deepcopy(first)
        changed["task_id_hashes"][0] = digest("mutated")
        self.assertTrue(any("task_id_aggregate" in error for error in validate_boundary(changed)))
        self.assertTrue(any("split_manifest" in error for error in validate_boundary(changed)))

    def test_missing_results_remain_blocked_without_fabricating_metrics(self) -> None:
        boundary = build_valid(result_receipts=None)
        self.assertEqual(boundary["status"], "BLOCKED")
        self.assertEqual(boundary["result_receipts"]["status"], "UNRECORDED")
        self.assertFalse(boundary["result_receipts"]["recorded"])
        self.assertIsNone(boundary["result_claims"])
        self.assertTrue(any("result receipts are unrecorded" in error for error in boundary["errors"]))
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "unrecorded"):
            verify_boundary(boundary)

    def test_primary_eval_is_not_implicitly_held_out(self) -> None:
        boundary = build_valid()
        self.assertEqual(boundary["held_out"]["receipt_proven"], False)
        self.assertIsNone(boundary["held_out"]["receipt_ref"])
        claimed = build_valid(held_out_claimed=True)
        self.assertEqual(claimed["status"], "BLOCKED")
        self.assertTrue(any("held-out claim requires" in error for error in claimed["errors"]))
        proven = build_valid(held_out_receipt_ref="receipt://heldout/" + digest("heldout"))
        self.assertTrue(proven["held_out"]["receipt_proven"])
        self.assertEqual(proven["role"], ROLE)
        invalid = build_valid(held_out_receipt_ref="UNRECORDED")
        self.assertFalse(invalid["held_out"]["receipt_proven"])
        self.assertTrue(any("held-out receipt reference" in error for error in invalid["errors"]))

    def test_substitutes_and_related_benchmarks_are_rejected(self) -> None:
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "authoritative BankerToolBench"):
            build_valid(source_identity={"kind": "huggingface_dataset", "dataset_id": "Salesforce/xlam-function-calling-60k", "url": SOURCE_URL})
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "related benchmarks"):
            build_valid(source_identity={"kind": "huggingface_dataset", "dataset_id": DATASET_ID, "url": SOURCE_URL, "related_benchmark": "BFCL"})
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "primary_eval"):
            build_valid(role="held_out")

    def test_revision_license_and_task_id_gates_fail_closed(self) -> None:
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "immutable 40-character"):
            build_valid(revision="main")
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "license_id"):
            build_valid(license_id="UNRECORDED")
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "license_receipt_ref"):
            build_valid(license_receipt_ref="latest")
        duplicate = build_valid(tasks=["same", "same"])
        self.assertEqual(duplicate["status"], "BLOCKED")
        self.assertTrue(any("duplicate authoritative task ID" in error for error in duplicate["errors"]))
        with self.assertRaisesRegex(BankerToolBenchBoundaryError, "metadata-only"):
            build_valid(tasks=[{"prompt": "raw"}])

    def test_native_environment_artifact_and_verifier_contracts_are_required(self) -> None:
        bad_environment = environment_contract()
        bad_environment["native"] = False
        bad_environment["environment_revision"] = "latest"
        boundary = build_valid(environment_contract=bad_environment)
        self.assertEqual(boundary["status"], "BLOCKED")
        self.assertTrue(any("native=true" in error for error in boundary["errors"]))
        self.assertTrue(any("environment_revision" in error for error in boundary["errors"]))

        bad_artifact = artifact_contract()
        bad_artifact["artifact_receipt_ref"] = "UNRECORDED"
        boundary = build_valid(artifact_contract=bad_artifact)
        self.assertTrue(any("artifact_receipt_ref" in error for error in boundary["errors"]))

        bad_verifier = verifier_contract()
        bad_verifier["deterministic"] = False
        boundary = build_valid(verifier_contract=bad_verifier)
        self.assertTrue(any("deterministic=true" in error for error in boundary["errors"]))

    def test_result_receipt_fields_are_required_and_hash_checked(self) -> None:
        receipts = result_receipts()
        receipts["wandb"].pop("history_sha256")
        receipts["tinker"]["model_revision"] = "latest"
        receipts["hf"]["artifact_sha256"] = "not-a-hash"
        boundary = build_valid(result_receipts=receipts)
        self.assertEqual(boundary["status"], "BLOCKED")
        errors = "\n".join(boundary["errors"])
        self.assertIn("wandb result receipt missing history_sha256", errors)
        self.assertIn("tinker result receipt missing model_revision", errors)
        self.assertIn("hf.artifact_sha256 must be a SHA-256 digest", errors)

    def test_metadata_mutation_is_detected(self) -> None:
        boundary = build_valid()
        mutated = copy.deepcopy(boundary)
        mutated["source_identity"]["dataset_id"] = "Salesforce/xlam-function-calling-60k"
        self.assertTrue(any("authoritative BankerToolBench" in error for error in validate_boundary(mutated)))
        mutated = copy.deepcopy(boundary)
        mutated["environment_contract"]["environment_revision"] = "main"
        self.assertTrue(any("environment_revision" in error for error in validate_boundary(mutated)))

    def test_local_cli_generate_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment_path = root / "environment.json"
            artifact_path = root / "artifact.json"
            verifier_path = root / "verifier.json"
            results_path = root / "results.json"
            boundary_path = root / "boundary.json"
            for path, value in (
                (environment_path, environment_contract()),
                (artifact_path, artifact_contract()),
                (verifier_path, verifier_contract()),
                (results_path, result_receipts()),
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                main(
                    [
                        "generate",
                        "--revision",
                        REVISION,
                        "--license-id",
                        "CC-BY-4.0",
                        "--license-receipt-ref",
                        "sha256:" + digest("license"),
                        "--task-id",
                        "banker/task/001",
                        "--task-id",
                        "banker/task/002",
                        "--environment-contract",
                        str(environment_path),
                        "--artifact-contract",
                        str(artifact_path),
                        "--verifier-contract",
                        str(verifier_path),
                        "--result-receipts",
                        str(results_path),
                        "--out",
                        str(boundary_path),
                    ]
                ),
                0,
            )
            self.assertEqual(main(["verify", "--boundary", str(boundary_path)]), 0)


if __name__ == "__main__":
    unittest.main()
