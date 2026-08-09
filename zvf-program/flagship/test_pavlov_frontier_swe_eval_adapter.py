from __future__ import annotations

import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flagship import pavlov_frontier_swe_eval_adapter as adapter


TASK_IDS = ["frontier-swe-task-000", "frontier-swe-task-001", "frontier-swe-task-002"]


def _hex(char: str, length: int) -> str:
    return char * length


def _license(identifier: str = "MIT") -> dict[str, object]:
    return {
        "status": "verified",
        "identifier": identifier,
        "source_url": adapter.OFFICIAL_METADATA_URL,
        "content_sha256": _hex("1", 64),
        "observed_at": "2026-08-09T00:00:00Z",
    }


def _boundary(
    *,
    license_identifier: str | None = "MIT",
    license_receipt: dict[str, object] | None = None,
    evaluation_role: str = adapter.PRIMARY_EVAL,
    heldout_proof: dict[str, object] | None = None,
) -> dict[str, object]:
    return adapter.build_boundary(
        task_ids=TASK_IDS,
        split_id="frontier-swe-evaluation",
        environment=adapter.native_environment_contract(
            environment_digest=f"sha256:{_hex('a', 64)}",
            runner_entrypoint="frontier_swe.native:run_task",
        ),
        artifact=adapter.native_artifact_contract(
            manifest_sha256=_hex("b", 64),
            paths=["task-result.json", "environment-state/"],
        ),
        verifier=adapter.native_verifier_contract(
            verifier_sha256=_hex("c", 64),
            entrypoint="frontier_swe.native:verify_task",
        ),
        license_identifier=license_identifier,
        license_receipt=_license() if license_receipt is None and license_identifier else license_receipt,
        source_revision=adapter.PINNED_SOURCE_REVISION,
        evaluation_role=evaluation_role,
        heldout_proof=heldout_proof,
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "task_id": TASK_IDS[0],
            "status": "passed",
            "success": True,
            "state_integrity": True,
            "artifact_integrity": True,
            "verifier_status": "pass",
        },
        {
            "task_id": TASK_IDS[1],
            "status": "failed",
            "success": False,
            "state_integrity": False,
            "artifact_integrity": False,
            "verifier_status": "fail",
        },
        {
            "task_id": TASK_IDS[2],
            "status": "passed",
            "success": True,
            "state_integrity": True,
            "artifact_integrity": True,
            "verifier_status": "pass",
        },
    ]


def _receipt(boundary: dict[str, object]) -> dict[str, object]:
    artifact_sha256 = boundary["artifact"]["manifest_sha256"]
    adapter_revision = _hex("e", 40)
    tinker = {
        "run_id": "tinker-frontier-swe-e2",
        "model_id": "Qwen/Qwen3.6-35B-A3B",
        "base_model_revision": _hex("d", 40),
        "adapter_revision": adapter_revision,
        "service_client_status": "verified",
    }
    return adapter.build_result_receipt(
        boundary,
        task_rows=_rows(),
        wandb={
            "run_id": "wandb-frontier-swe-e2",
            "url": "https://wandb.ai/entity/project/runs/frontier-e2",
            "entity": "entity",
            "project": "pavlov",
            "group": "frontier-swe-e2",
            "mode": "online",
            "config_sha256": _hex("f", 64),
            "artifact_name": "frontier-swe-e2-receipt",
            "artifact_sha256": artifact_sha256,
            "artifact_acknowledged": True,
        },
        tinker=tinker,
        huggingface={
            "repo_id": "org/frontier-swe-e2-adapter",
            "revision": adapter_revision,
            "artifact_sha256": _hex("9", 64),
            "artifact_path": "adapter/adapter_model.safetensors",
            "upload_status": "acknowledged",
        },
    )


class FrontierSWEAdapterTests(unittest.TestCase):
    def test_authoritative_identity_and_unverified_license_are_explicit(self) -> None:
        self.assertEqual(adapter.AUTHORITATIVE_REPOSITORY, "Proximal-Labs/frontier-swe")
        self.assertEqual(adapter.AUTHORITATIVE_SOURCE_URL, "https://github.com/Proximal-Labs/frontier-swe")
        self.assertEqual(len(adapter.PINNED_SOURCE_REVISION), 40)
        self.assertIsNone(adapter.OFFICIAL_LICENSE_IDENTIFIER)
        self.assertEqual(adapter.OFFICIAL_LICENSE_STATUS, "not_declared_by_official_metadata")

        blocked = _boundary(license_identifier=None, license_receipt=None)
        errors = adapter.validate_boundary(blocked)
        self.assertTrue(any("license_identifier" in error for error in errors))

    def test_task_ids_and_split_hashes_are_deterministic_and_order_sensitive(self) -> None:
        first = adapter.deterministic_split_manifest(TASK_IDS, "frontier-swe-evaluation")
        second = adapter.deterministic_split_manifest(list(TASK_IDS), "frontier-swe-evaluation")
        reordered = adapter.deterministic_split_manifest(list(reversed(TASK_IDS)), "frontier-swe-evaluation")
        self.assertEqual(first, second)
        self.assertNotEqual(first["task_ids_sha256"], reordered["task_ids_sha256"])
        self.assertNotEqual(first["split_manifest_sha256"], reordered["split_manifest_sha256"])

    def test_valid_primary_boundary_passes(self) -> None:
        boundary = _boundary()
        self.assertEqual(adapter.validate_boundary(boundary), [])
        self.assertEqual(boundary["suite_id"], adapter.SUITE_ID)
        self.assertEqual(boundary["domains"], ["code", "ml", "long_horizon"])
        self.assertEqual(boundary["claim"]["evaluation_role"], adapter.PRIMARY_EVAL)
        self.assertEqual(boundary["claim"]["heldout_status"], "not_proven")

    def test_invalid_source_revision_or_license_blocks(self) -> None:
        for mutation in ("revision", "license_identifier", "license_receipt"):
            with self.subTest(mutation=mutation):
                boundary = _boundary()
                if mutation == "revision":
                    boundary["source"]["revision"] = "0" * 40
                elif mutation == "license_identifier":
                    boundary["source"]["license_identifier"] = "placeholder"
                else:
                    boundary["source"]["license_receipt"]["content_sha256"] = "bad"
                errors = adapter.validate_boundary(boundary)
                self.assertTrue(errors)

    def test_native_environment_artifact_and_verifier_contracts_are_required(self) -> None:
        boundary = _boundary()
        boundary["environment"]["stateful"] = False
        boundary["artifact"]["required"] = False
        boundary["verifier"]["checks_artifacts"] = False
        errors = adapter.validate_boundary(boundary)
        self.assertTrue(any("environment.stateful" in error for error in errors))
        self.assertTrue(any("artifact.required" in error for error in errors))
        self.assertTrue(any("verifier.checks_artifacts" in error for error in errors))

    def test_xlam_and_related_benchmarks_are_rejected_as_substitutes(self) -> None:
        for field, value in (
            ("suite_id", "xlam_component"),
            ("benchmark_id", "swebench"),
            ("dataset_id", "Salesforce/xlam-function-calling-60k"),
            ("source", {"repository": "openai/swe-bench"}),
        ):
            with self.subTest(field=field):
                boundary = _boundary()
                if field == "source":
                    boundary["source"].update(value)
                else:
                    boundary[field] = value
                errors = adapter.validate_boundary(boundary)
                self.assertTrue(any("rejected" in error or "authoritative" in error for error in errors))

    def test_primary_eval_cannot_be_labeled_receipt_proven_heldout_without_proof(self) -> None:
        boundary = _boundary()
        boundary["claim"]["evaluation_role"] = adapter.RECEIPT_PROVEN_HELDOUT
        boundary["claim"]["heldout_status"] = "receipt_proven"
        errors = adapter.validate_boundary(boundary)
        self.assertTrue(any("heldout_proof" in error for error in errors))

    def test_receipt_proven_heldout_requires_matching_exclusion_receipt(self) -> None:
        manifest = adapter.deterministic_split_manifest(TASK_IDS, "frontier-swe-evaluation")
        proof = {
            "status": "verified",
            "task_ids_sha256": manifest["task_ids_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
            "training_exclusion_sha256": _hex("6", 64),
            "proof_artifact_sha256": _hex("7", 64),
            "decontamination_receipt": "frontier-swe-exclusion-receipt-v1",
        }
        boundary = _boundary(
            evaluation_role=adapter.RECEIPT_PROVEN_HELDOUT,
            heldout_proof=proof,
        )
        boundary["claim"]["claim_text"] = "FrontierSWE receipt-proven heldout evaluation."
        self.assertEqual(adapter.validate_boundary(boundary), [])
        boundary["claim"]["heldout_proof"]["split_manifest_sha256"] = _hex("8", 64)
        self.assertTrue(adapter.validate_boundary(boundary))

    def test_valid_result_receipt_passes_with_all_service_provenance(self) -> None:
        boundary = _boundary()
        receipt = _receipt(boundary)
        self.assertEqual(adapter.validate_result_receipt(receipt), [])
        self.assertEqual(receipt["wandb"]["mode"], "online")
        self.assertEqual(receipt["claim"]["evaluation_role"], adapter.PRIMARY_EVAL)

    def test_missing_wandb_tinker_or_huggingface_receipt_blocks(self) -> None:
        for field in ("wandb", "tinker", "huggingface"):
            with self.subTest(field=field):
                receipt = _receipt(_boundary())
                del receipt[field]
                errors = adapter.validate_result_receipt(receipt)
                self.assertTrue(any(field in error for error in errors))

    def test_wandb_must_be_online_and_acknowledge_the_native_artifact(self) -> None:
        receipt = _receipt(_boundary())
        receipt["wandb"]["mode"] = "offline"
        receipt["wandb"]["artifact_acknowledged"] = False
        errors = adapter.validate_result_receipt(receipt)
        self.assertIn("wandb.mode must be online", errors)
        self.assertIn("wandb.artifact_acknowledged must be true", errors)

    def test_tinker_and_huggingface_revisions_must_bind_to_the_receipt(self) -> None:
        receipt = _receipt(_boundary())
        receipt["huggingface"]["revision"] = _hex("8", 40)
        receipt["tinker"]["adapter_revision"] = _hex("9", 40)
        errors = adapter.validate_result_receipt(receipt)
        self.assertIn("huggingface.revision must equal tinker.adapter_revision", errors)

    def test_result_rows_must_match_task_ids_and_exact_metrics(self) -> None:
        receipt = _receipt(_boundary())
        receipt["results"]["task_rows"][0]["task_id"] = "different-task"
        receipt["results"]["task_success_rate"] = 0.0
        errors = adapter.validate_result_receipt(receipt)
        self.assertTrue(any("task_id" in error for error in errors))
        self.assertIn("results.task_success_rate does not match task rows exactly", errors)

    def test_raw_result_payloads_are_not_admissible(self) -> None:
        receipt = _receipt(_boundary())
        receipt["results"]["task_rows"][0]["response_text"] = "raw response"
        errors = adapter.validate_result_receipt(receipt)
        self.assertTrue(any("raw result field response_text" in error for error in errors))

    def test_raw_result_payloads_are_rejected_inside_service_receipts(self) -> None:
        receipt = _receipt(_boundary())
        receipt["wandb"]["raw_response"] = "raw response"
        receipt["tinker"]["trajectory"] = [{"tool": "secret"}]
        errors = adapter.validate_result_receipt(receipt)
        self.assertTrue(any("receipt.wandb.raw_response" in error for error in errors))
        self.assertTrue(any("receipt.tinker.trajectory" in error for error in errors))

    def test_receipt_and_boundary_must_be_byte_equivalent_json_contracts(self) -> None:
        boundary = _boundary()
        receipt = _receipt(boundary)
        other = copy.deepcopy(boundary)
        other["split"]["split_id"] = "different"
        self.assertTrue(adapter.validate_result_receipt(receipt, other))

    def test_cli_is_local_and_writes_validation_receipt(self) -> None:
        boundary = _boundary()
        receipt = _receipt(boundary)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            receipt_path = root / "receipt.json"
            output_path = root / "validation.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            self.assertEqual(
                adapter.main([str(receipt_path), "--out", str(output_path)]), 0
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(output["status"], "PASS")

    def test_adapter_never_opens_network_sockets(self) -> None:
        receipt = _receipt(_boundary())
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            self.assertEqual(adapter.validate_result_receipt(receipt), [])

    def test_malformed_nested_boundary_fails_closed_without_exception(self) -> None:
        receipt = _receipt(_boundary())
        receipt["boundary"]["source"] = "not-an-object"
        receipt["boundary"]["split"] = "not-an-object"
        errors = adapter.validate_result_receipt(receipt)
        self.assertTrue(errors)
        self.assertTrue(any("must be an object" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
