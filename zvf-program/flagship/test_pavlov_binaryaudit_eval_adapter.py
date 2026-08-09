from __future__ import annotations

import copy
import unittest

from flagship.pavlov_binaryaudit_eval_adapter import (
    AUTHORITATIVE_SOURCE_ID,
    AUTHORITATIVE_SOURCE_URL,
    BENCHMARK_ID,
    BinaryAuditBoundaryError,
    PRIMARY_EVAL,
    RECEIPT_PROVEN_HELDOUT,
    RESULT_SCHEMA_VERSION,
    SUITE_ID,
    deterministic_task_id,
    split_manifest_sha256,
    task_id_manifest_sha256,
    validate_binaryaudit_boundary,
    validate_binaryaudit_result_receipt,
)


SOURCE_REVISION = "a" * 40
LICENSE_HASH = "b" * 64
CONTAINER_DIGEST = "sha256:" + "c" * 64
ENVIRONMENT_HASH = "d" * 64
ARTIFACT_HASH = "e" * 64
VERIFIER_REVISION = "f" * 40


def _boundary() -> dict[str, object]:
    split = {
        "train": ["task-001", "task-002"],
        "primary_eval": ["task-101", "task-102"],
        "receipt_proven_heldout": ["task-201", "task-202"],
    }
    task_ids = sorted(item for values in split.values() for item in values)
    return {
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "source_identity": {
            "id": AUTHORITATIVE_SOURCE_ID,
            "url": AUTHORITATIVE_SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license_spdx": "Apache-2.0",
            "license_text_sha256": LICENSE_HASH,
            "license_url": AUTHORITATIVE_SOURCE_URL + "/blob/" + SOURCE_REVISION + "/LICENSE",
        },
        "task_ids": task_ids,
        "task_id_manifest_sha256": task_id_manifest_sha256(task_ids),
        "split_manifest": split,
        "split_manifest_sha256": split_manifest_sha256(split),
        "native_environment": {
            "mode": "native",
            "container_digest": CONTAINER_DIGEST,
            "environment_manifest_sha256": ENVIRONMENT_HASH,
            "artifact_contract": {
                "required": True,
                "manifest_sha256": ARTIFACT_HASH,
                "types": ["binary", "stdout", "stderr"],
            },
            "verifier_contract": {
                "name": "binaryaudit-native-verifier",
                "revision": VERIFIER_REVISION,
                "receipt_schema": RESULT_SCHEMA_VERSION,
                "checks": ["exit_code", "artifact_hash", "state_digest"],
            },
        },
    }


def _receipt(role: str = PRIMARY_EVAL) -> dict[str, object]:
    boundary = _boundary()
    task_ids = boundary["split_manifest"][role]  # type: ignore[index]
    receipt: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "evaluation_role": role,
        "status": "completed",
        "source_identity": copy.deepcopy(boundary["source_identity"]),
        "task_ids": list(task_ids),
        "task_id_manifest_sha256": task_id_manifest_sha256(task_ids),
        "split_manifest_sha256": boundary["split_manifest_sha256"],
        "native_environment_receipt": {
            "container_digest": CONTAINER_DIGEST,
            "environment_manifest_sha256": ENVIRONMENT_HASH,
        },
        "artifact_receipt": {
            "manifest_sha256": ARTIFACT_HASH,
            "paths": ["artifacts/task-101/report.json"],
        },
        "verifier_receipt": {
            "name": "binaryaudit-native-verifier",
            "revision": VERIFIER_REVISION,
            "receipt_schema": RESULT_SCHEMA_VERSION,
        },
        "wandb": {
            "run_id": "wandb-binaryaudit-1",
            "url": "https://wandb.example/run/binaryaudit-1",
            "mode": "online",
            "config_sha256": "1" * 64,
            "metrics_receipt_sha256": "2" * 64,
            "metrics_logged": True,
        },
        "tinker": {
            "run_id": "tinker-binaryaudit-1",
            "status": "completed",
            "config_sha256": "3" * 64,
            "cost_usd": 0.25,
        },
        "hf": {
            "repo": "private/binaryaudit-adapter",
            "revision": "4" * 40,
            "visibility": "private",
            "artifact_manifest_sha256": ARTIFACT_HASH,
        },
    }
    if role == PRIMARY_EVAL:
        receipt["metrics"] = {"task_success_rate": 0.5, "artifact_integrity_rate": 1.0}
    else:
        receipt["heldout_proof"] = {
            "selection_task_id_manifest_sha256": task_id_manifest_sha256(
                boundary["split_manifest"]["train"]  # type: ignore[index]
            ),
            "heldout_task_id_manifest_sha256": task_id_manifest_sha256(task_ids),
            "disjoint": True,
            "selection_locked": True,
            "not_used_for_selection": True,
        }
    return receipt


class HashAndBoundaryTests(unittest.TestCase):
    def test_hashes_and_task_ids_are_deterministic(self) -> None:
        boundary = _boundary()
        self.assertEqual(boundary["task_id_manifest_sha256"], task_id_manifest_sha256(boundary["task_ids"]))
        self.assertEqual(boundary["split_manifest_sha256"], split_manifest_sha256(boundary["split_manifest"]))
        first = deterministic_task_id("raw-1", SOURCE_REVISION)
        second = deterministic_task_id("raw-1", SOURCE_REVISION)
        self.assertEqual(first, second)
        self.assertNotEqual(first, deterministic_task_id("raw-2", SOURCE_REVISION))

    def test_valid_boundary_pins_source_revision_license_and_native_contract(self) -> None:
        normalized = validate_binaryaudit_boundary(_boundary())
        self.assertEqual(normalized["source_identity"]["id"], AUTHORITATIVE_SOURCE_ID)
        self.assertEqual(normalized["source_identity"]["revision"], SOURCE_REVISION)
        self.assertEqual(normalized["native_environment"]["mode"], "native")
        self.assertTrue(normalized["native_environment"]["artifact_contract"]["required"])
        self.assertTrue(normalized["substitutes_rejected"])

    def test_mutable_revision_license_and_hash_drift_fail_closed(self) -> None:
        for mutation, expected in (
            (lambda item: item["source_identity"].update(revision="main"), "mutable tag/branch"),
            (lambda item: item["source_identity"].update(license_text_sha256="not-a-hash"), "license_text_sha256"),
            (lambda item: item.update(task_id_manifest_sha256="0" * 64), "does not match task_ids"),
            (lambda item: item.update(split_manifest_sha256="0" * 64), "does not match split_manifest"),
        ):
            with self.subTest(expected=expected):
                item = _boundary()
                mutation(item)
                with self.assertRaises(BinaryAuditBoundaryError) as raised:
                    validate_binaryaudit_boundary(item)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

    def test_related_benchmarks_and_xlam_are_not_substitutes(self) -> None:
        for mutation in (
            lambda item: item.update(benchmark_id="xlam"),
            lambda item: item.update(related_benchmark="BFCL"),
            lambda item: item.update(substitutes=["AgentHarm"]),
        ):
            item = _boundary()
            mutation(item)
            with self.assertRaises(BinaryAuditBoundaryError):
                validate_binaryaudit_boundary(item)

    def test_split_ids_must_be_sorted_unique_and_disjoint(self) -> None:
        item = _boundary()
        item["split_manifest"]["primary_eval"] = ["task-102", "task-101"]  # type: ignore[index]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_boundary(item)
        self.assertTrue(any("lexically sorted" in message for message in raised.exception.diagnostics))

        item = _boundary()
        item["split_manifest"]["primary_eval"] = ["task-001", "task-101"]  # type: ignore[index]
        item["task_ids"] = sorted(item["task_ids"])  # type: ignore[arg-type]
        item["task_id_manifest_sha256"] = task_id_manifest_sha256(item["task_ids"])  # type: ignore[arg-type]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_boundary(item)
        self.assertTrue(any("overlapping IDs" in message for message in raised.exception.diagnostics))


class ReceiptBoundaryTests(unittest.TestCase):
    def test_primary_eval_receipt_is_evidence_bearing_only_after_full_receipts(self) -> None:
        normalized = validate_binaryaudit_result_receipt(_boundary(), _receipt())
        self.assertEqual(normalized["status"], "admissible_primary_eval")
        self.assertTrue(normalized["primary_eval"])
        self.assertTrue(normalized["scientific_evidence"])
        self.assertEqual(normalized["metrics"]["task_success_rate"], 0.5)
        self.assertFalse(normalized["portfolio_claim_permitted"])
        self.assertEqual(normalized["tracking_receipts"]["wandb"]["mode"], "online")

    def test_receipt_proven_heldout_is_not_primary_eval(self) -> None:
        normalized = validate_binaryaudit_result_receipt(
            _boundary(), _receipt(RECEIPT_PROVEN_HELDOUT)
        )
        self.assertEqual(normalized["status"], "receipt_proven_heldout")
        self.assertFalse(normalized["primary_eval"])
        self.assertTrue(normalized["receipt_proven_heldout"])
        self.assertFalse(normalized["scientific_evidence"])
        self.assertTrue(normalized["primary_eval_required"])

    def test_heldout_label_without_proof_is_rejected(self) -> None:
        receipt = _receipt(RECEIPT_PROVEN_HELDOUT)
        del receipt["heldout_proof"]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_result_receipt(_boundary(), receipt)
        self.assertTrue(any("requires heldout_proof" in message for message in raised.exception.diagnostics))

    def test_related_receipt_and_wrong_split_are_rejected(self) -> None:
        for mutation, expected in (
            (lambda item: item.update(benchmark_id="xlam"), "BinaryAudit"),
            (lambda item: item.update(task_ids=["task-201", "task-202"]), "primary_eval split"),
            (lambda item: item.update(split_manifest_sha256="0" * 64), "differs from boundary"),
        ):
            with self.subTest(expected=expected):
                receipt = _receipt()
                mutation(receipt)
                if "task_ids" in receipt:
                    receipt["task_id_manifest_sha256"] = task_id_manifest_sha256(receipt["task_ids"])
                with self.assertRaises(BinaryAuditBoundaryError) as raised:
                    validate_binaryaudit_result_receipt(_boundary(), receipt)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

    def test_missing_or_unsafe_tracking_receipts_fail_closed(self) -> None:
        for key, mutation, expected in (
            ("wandb", lambda item: item["wandb"].update(mode="offline"), "W&B receipt mode"),
            ("tinker", lambda item: item["tinker"].update(status="failed"), "Tinker receipt status"),
            ("hf", lambda item: item["hf"].update(visibility="public"), "visibility"),
        ):
            with self.subTest(key=key):
                receipt = _receipt()
                mutation(receipt)
                with self.assertRaises(BinaryAuditBoundaryError) as raised:
                    validate_binaryaudit_result_receipt(_boundary(), receipt)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

        receipt = _receipt()
        del receipt["hf"]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_result_receipt(_boundary(), receipt)
        self.assertTrue(any("missing Hugging Face" in message for message in raised.exception.diagnostics))

    def test_native_artifact_and_verifier_drift_is_rejected(self) -> None:
        receipt = _receipt()
        receipt["native_environment_receipt"]["container_digest"] = "sha256:" + "9" * 64  # type: ignore[index]
        receipt["artifact_receipt"]["manifest_sha256"] = "8" * 64  # type: ignore[index]
        receipt["verifier_receipt"]["revision"] = "7" * 40  # type: ignore[index]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_result_receipt(_boundary(), receipt)
        diagnostics = " ".join(raised.exception.diagnostics)
        self.assertIn("container_digest differs", diagnostics)
        self.assertIn("artifact receipt manifest differs", diagnostics)
        self.assertIn("verifier receipt revision differs", diagnostics)


if __name__ == "__main__":
    unittest.main()
