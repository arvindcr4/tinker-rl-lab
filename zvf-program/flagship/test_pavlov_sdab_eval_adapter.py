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
    SYNTHETIC_MARKER,
    SdabBoundaryError,
    SdabBundleError,
    build_ingest_receipt,
    build_result_receipt,
    build_runtime_manifest,
    build_sdab_boundary,
    build_split_manifest,
    canonical_json,
    ingest_task_bundle,
    main,
    newline_task_id_sha256,
    prove_split_disjointness,
    sha256_digest,
    validate_result_receipt,
    validate_sdab_boundary,
    validate_task_bundle,
)
from .eval_pavlov_sdab import task_ids_sha256, validate_native_manifest


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


# ---------------------------------------------------------------------------
# Bundle-ingestion fixtures
#
# ``synthetic_bundle`` is the deliverable fixture: every task ID carries the
# SYNTHETIC-NOT-SDAB marker and the bundle sets ``synthetic: true``.  It is not
# SDAB data, it is never written into a boundary, and it can never yield a
# score.  ``_shape_only_bundle`` is a test double used solely to prove the
# authoritative code path compiles a boundary and a runtime manifest; it lives
# in test code and is never written to disk as data.
# ---------------------------------------------------------------------------

SYNTHETIC_NOTICE = (
    "SYNTHETIC FIXTURE - GENERATED LOCALLY BY THE E3 TEST SUITE - "
    "THIS IS NOT SDAB DATA AND CAN NEVER PRODUCE A SCORE"
)
_CATEGORY_CYCLE = (
    "infrastructure_debugging",
    "migrations_and_upgrades",
    "ci_cd_and_deployment",
    "observability_and_incident_response",
    "distributed_systems",
)


def _bundle_license() -> dict:
    return {
        "license": {"spdx_id": "LicenseRef-SDAB-Evaluation", "name": "SDAB evaluation terms"},
        "license_receipt": {
            "receipt_id": "sdab-license-receipt-1",
            "digest": _digest("c"),
            "reference": "local://sdab/license/receipt-1",
        },
    }


def synthetic_bundle(task_count: int = OFFICIAL_TASK_COUNT) -> dict:
    """Return the clearly-marked synthetic 80-task fixture."""

    return {
        "synthetic": True,
        "provenance": SYNTHETIC_NOTICE,
        "benchmark_id": BENCHMARK_ID,
        "benchmark_name": BENCHMARK_NAME,
        "canonical_url": CANONICAL_URL,
        "provider": "Emulated, Inc.",
        "split": "evaluation",
        "revision": f"{SYNTHETIC_MARKER}-revision-0000",
        **_bundle_license(),
        "tasks": [
            {
                "task_id": f"{SYNTHETIC_MARKER}-{index:04d}",
                "category": _CATEGORY_CYCLE[index % len(_CATEGORY_CYCLE)],
                "prompt": "SYNTHETIC PLACEHOLDER - NOT AN SDAB TASK",
                "targets": ["SYNTHETIC PLACEHOLDER"],
            }
            for index in range(1, task_count + 1)
        ],
    }


def synthetic_train_task_ids(count: int = 20) -> list[str]:
    return [f"{SYNTHETIC_MARKER}-TRAIN-{index:04d}" for index in range(1, count + 1)]


def _shape_only_bundle(task_count: int = OFFICIAL_TASK_COUNT) -> dict:
    """Shape-only double for the authoritative path.  Not SDAB data."""

    return {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_name": BENCHMARK_NAME,
        "canonical_url": CANONICAL_URL,
        "provider": "Emulated, Inc.",
        "split": "evaluation",
        # The runner requires a 40- or 64-hex immutable revision.
        "revision": "a" * 40,
        **_bundle_license(),
        "tasks": [
            {
                "task_id": f"eval-{index:04d}",
                "category": _CATEGORY_CYCLE[index % len(_CATEGORY_CYCLE)],
            }
            for index in range(1, task_count + 1)
        ],
    }


def _shape_only_train_ids(count: int = 20) -> list[str]:
    return [f"train-{index:04d}" for index in range(1, count + 1)]


class SdabBundleSchemaTests(unittest.TestCase):
    def test_synthetic_fixture_is_unmistakably_marked(self) -> None:
        bundle = synthetic_bundle()
        self.assertTrue(bundle["synthetic"])
        self.assertIn("NOT SDAB DATA", bundle["provenance"])
        self.assertEqual(len(bundle["tasks"]), OFFICIAL_TASK_COUNT)
        for task in bundle["tasks"]:
            self.assertIn(SYNTHETIC_MARKER, task["task_id"])

    def test_bundle_requires_exactly_the_official_task_count(self) -> None:
        short = synthetic_bundle(task_count=OFFICIAL_TASK_COUNT - 1)
        with self.assertRaises(SdabBundleError) as ctx:
            validate_task_bundle(short, allow_synthetic=True)
        self.assertIn(str(OFFICIAL_TASK_COUNT), str(ctx.exception))

    def test_synthetic_bundle_is_refused_unless_explicitly_allowed(self) -> None:
        with self.assertRaises(SdabBundleError):
            validate_task_bundle(synthetic_bundle())

    def test_duplicate_and_casefold_colliding_task_ids_fail_closed(self) -> None:
        bundle = synthetic_bundle()
        bundle["tasks"][1]["task_id"] = bundle["tasks"][0]["task_id"]
        with self.assertRaises(SdabBundleError):
            validate_task_bundle(bundle, allow_synthetic=True)
        bundle = synthetic_bundle()
        bundle["tasks"][1]["task_id"] = bundle["tasks"][0]["task_id"].lower()
        with self.assertRaises(SdabBundleError):
            validate_task_bundle(bundle, allow_synthetic=True)

    def test_unknown_category_and_substitute_benchmarks_fail_closed(self) -> None:
        bundle = synthetic_bundle()
        bundle["tasks"][0]["category"] = "prompt_only_qa"
        with self.assertRaises(SdabBundleError):
            validate_task_bundle(bundle, allow_synthetic=True)
        bundle = synthetic_bundle()
        bundle["related_benchmark"] = "xlam"
        with self.assertRaises(SdabBoundaryError):
            validate_task_bundle(bundle, allow_synthetic=True)

    def test_raw_task_content_is_stripped_from_the_report(self) -> None:
        report = validate_task_bundle(synthetic_bundle(), allow_synthetic=True)
        self.assertEqual(report["raw_content_keys_stripped"], ["prompt", "targets"])
        for record in report["tasks"]:
            self.assertNotIn("prompt", record)
            self.assertNotIn("targets", record)
        self.assertNotIn("prompt", canonical_json(report["tasks"]))

    def test_task_id_hashing_matches_the_runner_hash_scheme(self) -> None:
        report = validate_task_bundle(synthetic_bundle(), allow_synthetic=True)
        self.assertEqual(
            report["task_id_sha256"], task_ids_sha256(report["task_ids"])
        )
        self.assertEqual(
            report["task_id_sha256"], newline_task_id_sha256(report["task_ids"])
        )
        self.assertEqual(report["task_id_digest"], sha256_digest(report["task_ids"]))
        self.assertEqual(report["task_count"], OFFICIAL_TASK_COUNT)

    def test_split_manifest_is_deterministic_and_order_independent(self) -> None:
        bundle = synthetic_bundle()
        first = build_split_manifest(validate_task_bundle(bundle, allow_synthetic=True))
        shuffled = synthetic_bundle()
        shuffled["tasks"] = list(reversed(shuffled["tasks"]))
        second = build_split_manifest(validate_task_bundle(shuffled, allow_synthetic=True))
        self.assertEqual(first, second)
        self.assertEqual(first["name"], "evaluation")
        self.assertEqual(len(first["task_ids"]), OFFICIAL_TASK_COUNT)


class SdabDisjointnessTests(unittest.TestCase):
    def test_disjoint_split_produces_a_proof(self) -> None:
        proof = prove_split_disjointness(["eval-1", "eval-2"], ["train-1"])
        self.assertTrue(proof["disjoint"])
        self.assertEqual(proof["intersection_count"], 0)
        self.assertNotEqual(proof["eval_task_id_sha256"], proof["train_task_id_sha256"])
        self.assertTrue(proof["proof_digest"].startswith("sha256:"))

    def test_overlapping_split_fails_closed(self) -> None:
        with self.assertRaises(SdabBundleError):
            prove_split_disjointness(["eval-1", "shared"], ["shared"])

    def test_casefold_overlap_is_still_contamination(self) -> None:
        with self.assertRaises(SdabBundleError):
            prove_split_disjointness(["Eval-1"], ["eval-1"])

    def test_missing_training_split_fails_closed_in_strict_mode(self) -> None:
        with self.assertRaises(SdabBundleError):
            prove_split_disjointness(["eval-1"], [])
        lenient = prove_split_disjointness(["eval-1"], [], strict=False)
        self.assertFalse(lenient["train_split_supplied"])
        self.assertIsNone(lenient["train_task_id_sha256"])


class SdabBundleIngestTests(unittest.TestCase):
    def test_synthetic_fixture_is_rejected_by_the_authoritative_path(self) -> None:
        with self.assertRaises(SdabBundleError):
            ingest_task_bundle(
                synthetic_bundle(),
                train_task_ids=synthetic_train_task_ids(),
                mode="authoritative",
            )

    def test_harness_validation_ingest_never_produces_a_score(self) -> None:
        report = ingest_task_bundle(
            synthetic_bundle(),
            train_task_ids=synthetic_train_task_ids(),
            mode="harness_validation",
        )
        self.assertIsNone(report["score"])
        self.assertFalse(report["is_model_score"])
        self.assertTrue(report["synthetic"])
        self.assertFalse(report["authoritative"])
        self.assertIsNone(report["boundary"])
        self.assertEqual(report["evidence_kind"], "harness_validation")
        # The fixture is provably refused by build_sdab_boundary.
        self.assertTrue(report["boundary_rejection"])
        self.assertIn("synthetic", report["boundary_rejection"][0])
        # Plumbing still ran end to end.
        self.assertEqual(report["bundle"]["task_count"], OFFICIAL_TASK_COUNT)
        self.assertEqual(len(report["split_manifest"]["task_ids"]), OFFICIAL_TASK_COUNT)
        self.assertTrue(report["disjointness_proof"]["disjoint"])

    def test_harness_validation_refuses_an_authentic_shaped_bundle(self) -> None:
        with self.assertRaises(SdabBundleError):
            ingest_task_bundle(
                _shape_only_bundle(),
                train_task_ids=_shape_only_train_ids(),
                mode="harness_validation",
            )

    def test_authoritative_ingest_builds_a_boundary_that_reconciles(self) -> None:
        report = ingest_task_bundle(
            _shape_only_bundle(),
            train_task_ids=_shape_only_train_ids(),
            mode="authoritative",
            source_revision_digest=_digest("a"),
            container_digest=_digest("b"),
        )
        boundary = report["boundary"]
        self.assertIsNotNone(boundary)
        self.assertEqual(boundary["task_count"], OFFICIAL_TASK_COUNT)
        self.assertEqual(boundary["official_task_count"], OFFICIAL_TASK_COUNT)
        self.assertEqual(boundary["split"], "evaluation")
        self.assertEqual(boundary["task_id_hash"], report["bundle"]["task_id_digest"])
        self.assertEqual(boundary["split_manifest_hash"], report["split_manifest_hash"])
        self.assertFalse(boundary["receipt_proven_heldout"])
        self.assertEqual(
            sorted(boundary["heldout_missing_receipts"]), sorted(REQUIRED_HELDOUT_RECEIPTS)
        )
        self.assertIsNone(report["score"])

    def test_ingest_is_deterministic(self) -> None:
        kwargs = {
            "train_task_ids": _shape_only_train_ids(),
            "mode": "authoritative",
            "source_revision_digest": _digest("a"),
            "container_digest": _digest("b"),
        }
        first = ingest_task_bundle(_shape_only_bundle(), **kwargs)
        second = ingest_task_bundle(_shape_only_bundle(), **kwargs)
        self.assertEqual(first["ingest_digest"], second["ingest_digest"])


class SdabRuntimeManifestTests(unittest.TestCase):
    def _authoritative_ingest(self) -> dict:
        return ingest_task_bundle(
            _shape_only_bundle(),
            train_task_ids=_shape_only_train_ids(),
            mode="authoritative",
            source_revision_digest=_digest("a"),
            container_digest=_digest("b"),
        )

    def _manifest(self, ingest: dict) -> dict:
        return build_runtime_manifest(
            ingest,
            container_digest=_digest("b"),
            environment_digest=_digest("1"),
            verifier_sha256="f" * 64,
            verifier_identity="emulated-native-sdab-verifier",
            adapter_entrypoint="emulated_sdab.runtime:create_runtime",
            disjointness_receipt="provider://sdab/split-receipt-1",
        )

    def test_runtime_manifest_is_accepted_by_the_runner_gate(self) -> None:
        manifest = self._manifest(self._authoritative_ingest())
        validated = validate_native_manifest(manifest, required_tasks=1)
        self.assertEqual(validated["task_count"], OFFICIAL_TASK_COUNT)
        self.assertEqual(validated["split"], "evaluation")
        self.assertTrue(validated["native_verifier"])
        self.assertNotEqual(
            validated["task_id_sha256"], validated["train_task_id_sha256"]
        )

    def test_runtime_manifest_is_metadata_only(self) -> None:
        manifest = self._manifest(self._authoritative_ingest())
        serialized = canonical_json(manifest)
        for key in ("prompt", "prompts", "target", "targets", "trajectory"):
            self.assertNotIn(key, manifest)
        self.assertNotIn("PLACEHOLDER", serialized)

    def test_runtime_manifest_refuses_a_synthetic_ingest(self) -> None:
        synthetic = ingest_task_bundle(
            synthetic_bundle(),
            train_task_ids=synthetic_train_task_ids(),
            mode="harness_validation",
        )
        with self.assertRaises(SdabBundleError):
            self._manifest(synthetic)


class SdabIngestReceiptTests(unittest.TestCase):
    def test_ingest_receipt_never_carries_a_score(self) -> None:
        report = ingest_task_bundle(
            synthetic_bundle(),
            train_task_ids=synthetic_train_task_ids(),
            mode="harness_validation",
        )
        receipt = build_ingest_receipt(
            report, status="BLOCKED", blockers=["synthetic fixture only"]
        )
        self.assertIsNone(receipt["score"])
        self.assertFalse(receipt["is_model_score"])
        self.assertEqual(receipt["status"], "BLOCKED")
        self.assertTrue(receipt["synthetic"])
        self.assertTrue(receipt["receipt_digest"].startswith("sha256:"))

    def test_cli_bundle_ingest_writes_a_blocked_synthetic_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "synthetic_bundle.json"
            train_path = Path(directory) / "synthetic_train_ids.json"
            out_path = Path(directory) / "ingest_receipt.json"
            bundle_path.write_text(json.dumps(synthetic_bundle()), encoding="utf-8")
            train_path.write_text(json.dumps(synthetic_train_task_ids()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--bundle",
                        str(bundle_path),
                        "--train-task-ids",
                        str(train_path),
                        "--mode",
                        "harness_validation",
                        "--out",
                        str(out_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            parsed = json.loads(output.getvalue())
            self.assertIsNone(parsed["ingest"]["score"])
            self.assertIsNone(parsed["ingest"]["boundary"])
            receipt = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "BLOCKED")
            self.assertIsNone(receipt["score"])
            self.assertFalse(receipt["is_model_score"])

    def test_cli_requires_a_boundary_or_a_bundle(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])
        self.assertEqual(exit_code, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
