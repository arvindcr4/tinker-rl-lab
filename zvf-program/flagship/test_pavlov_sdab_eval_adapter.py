"""Offline tests for the SDAB primary-evaluation boundary adapter."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from .pavlov_sdab_eval_adapter import (
    BENCHMARK_ID,
    BENCHMARK_NAME,
    CANONICAL_URL,
    OFFICIAL_TASK_COUNT,
    REQUIRED_HELDOUT_RECEIPTS,
    ROLE,
    SUITE_ID,
    SdabBoundaryError,
    build_result_receipt,
    build_sdab_boundary,
    canonical_json,
    main,
    sha256_digest,
    validate_result_receipt,
    validate_sdab_boundary,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _boundary_spec(task_ids: tuple[str, ...] = ("task-001", "task-002")) -> dict:
    ids = list(task_ids)
    return {
        "suite_id": SUITE_ID,
        "role": ROLE,
        "source_identity": {
            "provider": "Emulated, Inc.",
            "benchmark_id": BENCHMARK_ID,
            "benchmark_name": BENCHMARK_NAME,
            "canonical_url": CANONICAL_URL,
        },
        "source_revision": "sdab-evaluation-freeze-2026-04",
        "source_revision_digest": _digest("a"),
        "container_digest": _digest("b"),
        "license": {"spdx_id": "LicenseRef-SDAB", "name": "SDAB evaluation terms"},
        "license_receipt": {
            "receipt_id": "license-receipt",
            "digest": _digest("c"),
            "reference": "local://sdab/license/receipt-1",
        },
        "split": "evaluation",
        "tasks": [
            {"task_id": task_id, "category": "infrastructure_debugging"}
            for task_id in ids
        ],
        "task_ids": ids,
        "split_manifest": {
            "name": "evaluation",
            "revision": "sdab-evaluation-split-v1",
            "digest": _digest("d"),
            "task_ids": ids,
        },
    }


def _environment_receipt() -> dict:
    return {
        "environment_id": "sdab-env-001",
        "seed": "seed-001",
        "workspace_digest": _digest("e"),
        "state_before_digest": _digest("f"),
        "state_after_digest": _digest("0"),
        "container_digest": _digest("b"),
        "environment_digest": _digest("1"),
        "native": True,
    }


def _artifact_receipt() -> dict:
    return {
        "artifact_id": "artifact-001",
        "artifact_type": "patch-and-state",
        "artifact_digest": _digest("2"),
        "state_digest": _digest("3"),
        "native": True,
    }


def _verifier_receipt() -> dict:
    return {
        "verifier_id": "sdab-native-verifier",
        "verifier_revision": "sdab-verifier-v1",
        "verifier_digest": _digest("4"),
        "behavioral_tests_digest": _digest("5"),
        "rubric_digest": _digest("6"),
        "state_validation_digest": _digest("7"),
        "hidden_tests_digest": _digest("8"),
        "native": True,
        "hidden_tests_outside_policy_workspace": True,
    }


def _heldout_receipts() -> dict[str, dict[str, str]]:
    return {
        name: {
            "receipt_id": f"heldout-{name}-receipt",
            "digest": _digest(character),
            "reference": f"local://sdab/heldout/{name}",
        }
        for name, character in zip(REQUIRED_HELDOUT_RECEIPTS, "90abc")
    }


def _result(
    backend: str = "wandb",
    *,
    status: str = "completed",
    heldout: bool = False,
    **updates,
) -> dict:
    metrics = {"task_success": "0.500000", "state_integrity": "1.000000"}
    result = {
        "receipt_id": "result-receipt-001",
        "status": status,
        "started_at": "2026-08-09T00:00:00Z",
        "metrics": metrics,
        "metrics_digest": sha256_digest(metrics),
        "model_revision": "model-revision-001",
        "adapter_revision": "adapter-revision-001",
        "environment_receipt": _environment_receipt(),
        "artifact_receipt": _artifact_receipt(),
        "verifier_receipt": _verifier_receipt(),
        "heldout_claim_requested": heldout,
    }
    if status in {"completed", "failed", "rejected"}:
        result["completed_at"] = "2026-08-09T00:10:00Z"
    if heldout:
        result["heldout_receipts"] = _heldout_receipts()
    if backend == "wandb":
        result.update(
            {
                "wandb_entity": "pavlov",
                "wandb_project": "sdab-eval",
                "wandb_run_id": "wandb-run-001",
                "wandb_run_url": "https://wandb.example/pavlov/sdab-eval/runs/001",
                "wandb_summary_digest": _digest("9"),
            }
        )
    elif backend == "tinker":
        result.update(
            {
                "tinker_run_id": "tinker-run-001",
                "tinker_job_id": "tinker-job-001",
                # The backend receipt may alias the canonical result receipt;
                # aliases must agree exactly to avoid double identity.
                "tinker_receipt_id": "result-receipt-001",
                "tinker_provider": "Tinker",
            }
        )
    elif backend == "hf":
        result.update(
            {
                "hf_run_id": "hf-run-001",
                "hf_repo_id": "pavlov/sdab-adapter",
                "hf_revision": "hf-revision-001",
                "hf_receipt_id": "result-receipt-001",
            }
        )
    else:
        raise AssertionError(f"unknown fixture backend: {backend}")
    result.update(updates)
    return result


class SdabBoundaryTests(unittest.TestCase):
    def test_boundary_is_authoritative_primary_eval_and_not_heldout_evidence(self) -> None:
        boundary = build_sdab_boundary(_boundary_spec())
        self.assertEqual(boundary["suite_id"], SUITE_ID)
        self.assertEqual(boundary["role"], ROLE)
        self.assertTrue(boundary["primary_eval"])
        self.assertEqual(boundary["source_identity"]["benchmark_id"], BENCHMARK_ID)
        self.assertEqual(boundary["source_identity"]["canonical_url"], CANONICAL_URL)
        self.assertEqual(boundary["source_identity"]["provider"], "Emulated, Inc.")
        self.assertEqual(boundary["official_task_count"], OFFICIAL_TASK_COUNT)
        self.assertEqual(boundary["task_count"], 2)
        self.assertEqual(boundary["dataset_revision"], boundary["source_revision"])
        self.assertEqual(boundary["container_digest"], _digest("b"))
        self.assertFalse(boundary["receipt_proven_heldout"])
        self.assertFalse(boundary["heldout_claim_requested"])
        self.assertEqual(boundary["heldout_missing_receipts"], list(REQUIRED_HELDOUT_RECEIPTS))

    def test_task_and_split_hashes_are_order_independent_and_reproducible(self) -> None:
        first = build_sdab_boundary(_boundary_spec(("task-002", "task-001")))
        second = build_sdab_boundary(_boundary_spec(("task-001", "task-002")))
        self.assertEqual(first["task_ids"], ["task-001", "task-002"])
        self.assertEqual(first["task_id_hash"], second["task_id_hash"])
        self.assertEqual(first["task_id_hashes"], second["task_id_hashes"])
        self.assertEqual(first["split_manifest_hash"], second["split_manifest_hash"])
        self.assertEqual(first["split_manifest"], second["split_manifest"])
        self.assertEqual(first["task_id_hash"], sha256_digest(["task-001", "task-002"]))

    def test_canonical_json_keeps_decimal_text_without_float_conversion(self) -> None:
        self.assertEqual(canonical_json({"amount": "0.10"}), '{"amount":"0.10"}')
        self.assertEqual(sha256_digest(["task-001", "task-002"]), sha256_digest(["task-001", "task-002"]))

    def test_duplicate_task_ids_and_case_collisions_fail_closed(self) -> None:
        duplicate = _boundary_spec(("task-001", "task-001"))
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(duplicate)
        case_collision = _boundary_spec(("Task-001", "task-001"))
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(case_collision)

    def test_task_and_split_manifest_must_reconcile(self) -> None:
        mismatched_tasks = _boundary_spec()
        mismatched_tasks["task_ids"] = ["task-003", "task-004"]
        with self.assertRaisesRegex(SdabBoundaryError, "same tasks"):
            build_sdab_boundary(mismatched_tasks)
        mismatched_manifest = _boundary_spec()
        mismatched_manifest["split_manifest"]["task_ids"] = ["task-003", "task-004"]
        with self.assertRaisesRegex(SdabBoundaryError, "split_manifest.task_ids"):
            build_sdab_boundary(mismatched_manifest)
        bad_hash = _boundary_spec()
        bad_hash["task_id_hash"] = _digest("f")
        with self.assertRaisesRegex(SdabBoundaryError, "task_id_hash"):
            build_sdab_boundary(bad_hash)

    def test_revision_license_container_and_manifest_are_pinned(self) -> None:
        for field, value in (
            ("source_revision", "latest"),
            ("source_revision_digest", "not-a-digest"),
            ("container_digest", "unknown"),
        ):
            invalid = _boundary_spec()
            invalid[field] = value
            with self.assertRaises(SdabBoundaryError, msg=field):
                build_sdab_boundary(invalid)
        no_license_receipt = _boundary_spec()
        no_license_receipt.pop("license_receipt")
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(no_license_receipt)
        placeholder_license = _boundary_spec()
        placeholder_license["license"] = {"name": "latest"}
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(placeholder_license)
        ambiguous_license_receipt = _boundary_spec()
        ambiguous_license_receipt["license_receipt"]["uri"] = "local://a-different-license"
        with self.assertRaisesRegex(SdabBoundaryError, "aliases disagree"):
            build_sdab_boundary(ambiguous_license_receipt)
        no_manifest_revision = _boundary_spec()
        no_manifest_revision["split_manifest"].pop("revision")
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(no_manifest_revision)

    def test_optional_boundary_container_aliases_must_agree(self) -> None:
        boundary = _boundary_spec()
        boundary.pop("container_digest")
        built = build_sdab_boundary(boundary)
        self.assertIsNone(built["container_digest"])
        boundary["container_image_digest"] = _digest("b")
        built = build_sdab_boundary(boundary)
        self.assertEqual(built["container_digest"], _digest("b"))
        boundary["container_digest"] = _digest("c")
        with self.assertRaisesRegex(SdabBoundaryError, "aliases disagree"):
            build_sdab_boundary(boundary)

    def test_authoritative_identity_rejects_related_benchmarks_and_xlam(self) -> None:
        for bad_id in ("xLAM", "swe_bench_pro_eval", "frontier_swe_eval"):
            invalid = _boundary_spec()
            invalid["source_identity"]["benchmark_id"] = bad_id
            with self.assertRaises(SdabBoundaryError):
                build_sdab_boundary(invalid)
        invalid = _boundary_spec()
        invalid["substitute_for"] = "xlam"
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(invalid)
        invalid = _boundary_spec()
        invalid["source_identity"]["canonical_url"] = "https://example.invalid/sdab"
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(invalid)

    def test_native_contract_cannot_be_weakened(self) -> None:
        boundary = build_sdab_boundary(_boundary_spec())
        self.assertTrue(boundary["native_environment_contract"]["stateful"])
        self.assertTrue(boundary["artifact_contract"]["native"])
        self.assertTrue(boundary["verifier_contract"]["hidden_tests_outside_policy_workspace"])
        invalid = _boundary_spec()
        invalid["native_environment_contract"] = {
            "execution_mode": "mock",
            "stateful": True,
            "artifact_or_side_effect": True,
            "deterministic_seed_required": True,
            "policy_visible_surfaces": [
                "workspace",
                "running_infrastructure",
                "operational_tooling",
                "traffic_generator",
            ],
            "grading_harness": "outside_policy_visible_workspace",
        }
        with self.assertRaises(SdabBoundaryError):
            build_sdab_boundary(invalid)

    def test_validate_boundary_is_non_throwing_preflight(self) -> None:
        self.assertEqual(validate_sdab_boundary(_boundary_spec()), [])
        errors = validate_sdab_boundary({"suite_id": "xlam"})
        self.assertEqual(len(errors), 1)
        self.assertIn("sdab_eval", errors[0])


class SdabResultReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary = build_sdab_boundary(_boundary_spec())

    def test_primary_eval_receipt_does_not_claim_heldout_without_proof(self) -> None:
        receipt = build_result_receipt(self.boundary, _result(), "wandb")
        self.assertTrue(receipt["primary_eval"])
        self.assertFalse(receipt["receipt_proven_heldout"])
        self.assertFalse(receipt["heldout_claim_requested"])
        self.assertEqual(receipt["heldout_missing_receipts"], list(REQUIRED_HELDOUT_RECEIPTS))

    def test_completed_heldout_claim_requires_every_distinct_immutable_receipt(self) -> None:
        receipt = build_result_receipt(self.boundary, _result(heldout=True), "wandb")
        self.assertTrue(receipt["receipt_proven_heldout"])
        self.assertEqual(receipt["heldout_missing_receipts"], [])
        self.assertEqual(set(receipt["heldout_receipts"]), set(REQUIRED_HELDOUT_RECEIPTS))
        duplicate = _result(heldout=True)
        duplicate["heldout_receipts"]["license"]["receipt_id"] = duplicate["heldout_receipts"]["split"]["receipt_id"]
        with self.assertRaisesRegex(SdabBoundaryError, "distinct"):
            build_result_receipt(self.boundary, duplicate, "wandb")

    def test_claim_with_missing_receipt_fails_closed(self) -> None:
        result = _result(heldout=True)
        result["heldout_receipts"].pop("decontamination")
        with self.assertRaisesRegex(SdabBoundaryError, "decontamination"):
            build_result_receipt(self.boundary, result, "wandb")

    def test_pending_or_failed_receipts_never_become_heldout_evidence(self) -> None:
        pending = _result(status="pending", heldout=False)
        pending["heldout_receipts"] = _heldout_receipts()
        receipt = build_result_receipt(self.boundary, pending, "wandb")
        self.assertFalse(receipt["receipt_proven_heldout"])
        self.assertIn("completed_result", receipt["heldout_missing_receipts"])
        failed = _result(status="failed", heldout=False)
        failed["heldout_receipts"] = _heldout_receipts()
        receipt = build_result_receipt(self.boundary, failed, "wandb")
        self.assertFalse(receipt["receipt_proven_heldout"])
        self.assertIn("completed_result", receipt["heldout_missing_receipts"])

    def test_all_supported_backends_require_their_own_receipt_fields(self) -> None:
        for backend in ("wandb", "tinker", "hf"):
            with self.subTest(backend=backend):
                receipt = build_result_receipt(self.boundary, _result(backend), backend)
                self.assertEqual(receipt["backend"], backend)
                self.assertTrue(receipt["backend_fields"])
        self.assertEqual(
            build_result_receipt(self.boundary, _result("tinker"), "tinker")["backend_fields"]["provider"],
            "Tinker",
        )

    def test_backend_can_be_inferred_and_backend_receipt_id_can_be_distinct(self) -> None:
        result = _result("tinker")
        result["backend"] = "tinker"
        result["tinker_receipt_id"] = "billing-receipt-001"
        receipt = build_result_receipt(self.boundary, result)
        self.assertEqual(receipt["backend"], "tinker")
        self.assertEqual(receipt["backend_fields"]["receipt_id"], "billing-receipt-001")
        result["backend"] = "hf"
        with self.assertRaisesRegex(SdabBoundaryError, "does not match"):
            build_result_receipt(self.boundary, result, "tinker")

    def test_tinker_provider_cannot_be_mislabelled_as_source_provider(self) -> None:
        result = _result("tinker")
        result["tinker_provider"] = "Emulated, Inc."
        with self.assertRaisesRegex(SdabBoundaryError, "Tinker"):
            build_result_receipt(self.boundary, result, "tinker")

    def test_missing_or_conflicting_backend_fields_fail_closed(self) -> None:
        result = _result("wandb")
        result.pop("wandb_summary_digest")
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "wandb")
        result = _result("wandb")
        result["wandb_run_id"] = "latest"
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "wandb")
        result = _result("hf")
        result["hf_revision"] = "latest"
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "hf")
        result = _result("tinker")
        result["run_id"] = "different-run"
        with self.assertRaisesRegex(SdabBoundaryError, "aliases disagree"):
            build_result_receipt(self.boundary, result, "tinker")

    def test_environment_artifact_and_verifier_are_native_and_digest_pinned(self) -> None:
        checks = (
            ("environment_receipt", "native", False),
            ("artifact_receipt", "native", False),
            ("verifier_receipt", "native", False),
        )
        for section, key, value in checks:
            result = _result()
            result[section][key] = value
            with self.subTest(section=section):
                with self.assertRaises(SdabBoundaryError):
                    build_result_receipt(self.boundary, result, "wandb")
        result = _result()
        result["environment_receipt"].pop("container_digest")
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "wandb")
        result = _result()
        result["verifier_receipt"]["hidden_tests_outside_policy_workspace"] = False
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "wandb")
        result = _result()
        result["artifact_receipt"]["state_digest"] = "not-a-digest"
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "wandb")

    def test_result_identity_and_hashes_must_match_boundary(self) -> None:
        for field, value in (
            ("suite_id", "xlam"),
            ("evaluation_suite", "swe_bench_pro_eval"),
            ("benchmark_id", "xlam"),
            ("source_revision_digest", _digest("f")),
            ("dataset_revision", "another-freeze"),
            ("task_id_hash", _digest("f")),
            ("split_manifest_hash", _digest("f")),
        ):
            result = _result()
            result[field] = value
            with self.subTest(field=field):
                with self.assertRaises(SdabBoundaryError):
                    build_result_receipt(self.boundary, result, "wandb")
        result = _result()
        result["source_identity"] = copy.deepcopy(self.boundary["source_identity"])
        result["source_identity"]["canonical_url"] = "https://example.invalid/sdab"
        with self.assertRaises(SdabBoundaryError):
            build_result_receipt(self.boundary, result, "wandb")

    def test_metrics_digest_and_completed_metrics_are_not_fabricated(self) -> None:
        result = _result()
        result["metrics_digest"] = _digest("f")
        with self.assertRaisesRegex(SdabBoundaryError, "metrics_digest"):
            build_result_receipt(self.boundary, result, "wandb")
        result = _result()
        result["metrics"] = {}
        result["metrics_digest"] = sha256_digest({})
        with self.assertRaisesRegex(SdabBoundaryError, "must include metrics"):
            build_result_receipt(self.boundary, result, "wandb")

    def test_result_container_must_match_pinned_boundary_and_environment(self) -> None:
        result = _result()
        result["environment_receipt"]["container_digest"] = _digest("c")
        with self.assertRaisesRegex(SdabBoundaryError, "container_digest"):
            build_result_receipt(self.boundary, result, "wandb")
        result = _result()
        result["container_digest"] = _digest("c")
        with self.assertRaisesRegex(SdabBoundaryError, "container_digest"):
            build_result_receipt(self.boundary, result, "wandb")

    def test_result_receipt_digest_is_canonical_and_supplied_digest_is_checked(self) -> None:
        first = build_result_receipt(self.boundary, _result(), "wandb")
        second = build_result_receipt(self.boundary, _result(), "wandb")
        self.assertEqual(first["receipt_digest"], second["receipt_digest"])
        tampered = _result()
        tampered["receipt_digest"] = _digest("f")
        with self.assertRaisesRegex(SdabBoundaryError, "receipt_digest"):
            build_result_receipt(self.boundary, tampered, "wandb")

    def test_validate_result_receipt_returns_errors_without_throwing(self) -> None:
        self.assertEqual(validate_result_receipt(self.boundary, _result(), "wandb"), [])
        errors = validate_result_receipt(self.boundary, {"status": "completed"}, "wandb")
        self.assertEqual(len(errors), 1)
        self.assertIn("receipt_id", errors[0])


class SdabCliTests(unittest.TestCase):
    def test_cli_is_offline_and_emits_canonical_boundary_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.json"
            path.write_text(json.dumps(_boundary_spec()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main([str(path)])
            self.assertEqual(exit_code, 0)
            parsed = json.loads(output.getvalue())
            self.assertEqual(parsed["boundary"]["suite_id"], SUITE_ID)
            self.assertTrue(parsed["boundary"]["primary_eval"])

    def test_cli_fails_closed_for_missing_or_malformed_boundary(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main(["/tmp/does-not-exist-sdab-boundary.json"])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
