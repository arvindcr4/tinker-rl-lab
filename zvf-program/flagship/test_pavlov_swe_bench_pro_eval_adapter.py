"""Thorough offline tests for the SWE-bench Pro evaluation boundary."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flagship.pavlov_swe_bench_pro_eval_adapter import (
    BENCHMARK_ID,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_CARD_URL,
    EVALUATION_CODE_LICENSE_URL,
    HELDOUT_SUBSET,
    HELDOUT_TASK_COUNT,
    PRIVATE_SUBSET,
    PRIVATE_TASK_COUNT,
    BoundaryValidationError,
    OFFICIAL_EVAL_REPO_REVISION,
    PRIMARY_EVAL,
    PUBLIC_TASK_COUNT,
    RECEIPT_PROVEN_HELDOUT,
    SCHEMA_VERSION,
    TOTAL_TASK_COUNT,
    canonical_source_identity,
    compute_split_manifest_sha256,
    compute_task_ids_sha256,
    main,
    parse_manifest,
    validate_evaluation_boundary,
    validate_manifest,
    validate_result_receipt,
)


def _task(index: int) -> dict[str, Any]:
    instance_id = f"fixture_repository__fixture_issue_{index:03d}_" + "x" * 40
    base_commit = f"{index + 1:040x}"
    source_revision = f"{index + 101:040x}"
    digest = f"sha256:{index + 201:064x}"
    return {
        "instance_id": instance_id,
        "repo": "fixture/repository",
        "base_commit": base_commit,
        "dockerhub_tag": f"fixture-{index:03d}",
        "container_digest": digest,
        "license_receipt": {
            "spdx": "GPL-3.0-or-later",
            "source_url": "https://github.com/fixture/repository/blob/main/LICENSE",
            "source_revision": source_revision,
            "receipt_sha256": "f" * 64,
        },
        "problem_statement": "Fix the fixture defect without changing its public contract.",
        "repo_language": "Python",
        "issue_specificity": "specific",
        "issue_categories": "bug_fix",
        "requirements": None,
        "interface": None,
        "before_repo_set_cmd": "git checkout --detach BASE_COMMIT",
        "selected_test_files_to_run": "['tests/test_fixture.py']",
        "fail_to_pass": "['tests/test_fixture.py::test_bug']",
        "pass_to_pass": "['tests/test_fixture.py::test_compatibility']",
        "artifact_contract": {
            "kind": "repository_patch",
            "format": "unified_diff",
            "required": True,
        },
        "verifier_contract": {
            "kind": "swe_bench_pro_official",
            "resolve_rule": "fail_to_pass_and_pass_to_pass",
            "requires_native_tests": True,
        },
    }


def _manifest(
    *,
    role: str = PRIMARY_EVAL,
    coverage: str = "sampled_public",
    task_count: int = 2,
    receipt_subset: str = HELDOUT_SUBSET,
) -> dict[str, Any]:
    # These rows/receipts are synthetic validation fixtures only; they are not
    # benchmark tasks or claims about inaccessible private/held-out data.
    tasks = [_task(index) for index in range(task_count)]
    task_ids = sorted(task["instance_id"] for task in tasks)
    split_name = "test" if role == PRIMARY_EVAL else "heldout_receipt"
    subset = "public" if role == PRIMARY_EVAL else receipt_subset
    split_hash = compute_split_manifest_sha256(
        split_name=split_name,
        role=role,
        subset=subset,
        coverage=coverage,
        task_ids=task_ids,
        tasks=tasks,
    )
    split = {
        "role": role,
        "name": split_name,
        "subset": subset,
        "coverage": coverage,
        "task_ids": task_ids,
        "task_ids_sha256": compute_task_ids_sha256(task_ids),
        "split_manifest_sha256": split_hash,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "source": canonical_source_identity(),
        "split": split,
        "tasks": tasks,
        "environment": {
            "runtime": "docker",
            "container_image_registry": "jefzda/sweap-images",
            "artifact_contract": {
                "kind": "repository_patch",
                "native_artifact": "git_diff",
            },
            "verifier_contract": {
                "kind": "swe_bench_pro_official",
                "resolve_rule": "fail_to_pass_and_pass_to_pass",
            },
        },
    }
    if role == RECEIPT_PROVEN_HELDOUT:
        manifest["heldout_proof"] = {
            "status": "receipt_proven",
            "subset": receipt_subset,
            "source_identity": {
                "source_id": "heldout-source-receipt-001",
                "source_url": (
                    "https://labs.scale.com/leaderboard/swe_bench_pro_private"
                    if receipt_subset == PRIVATE_SUBSET
                    else "https://scale.com/leaderboard/swe_bench_pro_public"
                ),
                "source_revision": "a" * 40,
                "split": receipt_subset,
            },
            "access_receipt": {
                "source_revision": "a" * 40,
                "receipt_sha256": "b" * 64,
            },
            "decontamination_receipt": {
                "source_revision": "a" * 40,
                "receipt_sha256": "c" * 64,
            },
            "license_receipt": {
                "source_revision": "a" * 40,
                "receipt_sha256": "d" * 64,
            },
            "task_ids_sha256": split["task_ids_sha256"],
            "split_manifest_sha256": split_hash,
        }
    return manifest


def _hf_receipt() -> dict[str, str]:
    repo_id = "arvindcr4/swe-pro-eval-receipt"
    revision = "eval-run-001"
    commit_sha = "e" * 40
    repo_url = f"https://huggingface.co/{repo_id}"
    return {
        "repo_id": repo_id,
        "revision": revision,
        "commit_sha": commit_sha,
        "repo_url": repo_url,
        "revision_url": f"{repo_url}/tree/{revision}",
        "commit_url": f"{repo_url}/commit/{commit_sha}",
    }


def _result(manifest: dict[str, Any]) -> dict[str, Any]:
    task_ids = manifest["split"]["task_ids"]
    digests = sorted(task["container_digest"] for task in manifest["tasks"])
    resolved = 1
    task_results = []
    for index, task in enumerate(manifest["tasks"]):
        fail_name = "tests/test_fixture.py::test_bug"
        pass_name = "tests/test_fixture.py::test_compatibility"
        task_results.append(
            {
                "instance_id": task["instance_id"],
                "status": "resolved" if index == 0 else "unresolved",
                "tests": [
                    {"name": fail_name, "status": "PASSED"},
                    {
                        "name": pass_name,
                        "status": "PASSED" if index == 0 else "FAILED",
                    },
                ],
                "container_digest": task["container_digest"],
                "artifact_sha256": hashlib.sha256(
                    f"fixture patch {index}".encode("utf-8")
                ).hexdigest(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "status": "completed",
        "split_role": manifest["split"]["role"],
        "task_ids_sha256": manifest["split"]["task_ids_sha256"],
        "split_manifest_sha256": manifest["split"]["split_manifest_sha256"],
        "evaluated_task_ids": task_ids,
        "task_results": task_results,
        "resolved_count": resolved,
        "task_count": len(task_ids),
        "resolve_rate": resolved / len(task_ids),
        "wandb_run_id": "wandb-swe-pro-001",
        "wandb": {"run_id": "wandb-swe-pro-001"},
        "tinker_run_id": "tinker-swe-pro-001:eval:0",
        "tinker": {"run_id": "tinker-swe-pro-001:eval:0"},
        "hf_receipt": _hf_receipt(),
        "environment_receipt": {
            "runtime": "docker",
            "container_digests": digests,
        },
        "artifact_receipt": {
            "kind": "repository_patch",
            "format": "unified_diff",
            "sha256": hashlib.sha256(b"fixture patch").hexdigest(),
        },
        "verifier_receipt": {
            "kind": "swe_bench_pro_official",
            "fail_to_pass_passed": True,
            "pass_to_pass_passed": True,
        },
    }


class SweBenchProEvalAdapterTests(unittest.TestCase):
    def assert_rejected(self, value: MappingLike, message: str = "") -> None:
        with self.assertRaises(BoundaryValidationError, msg=message):
            validate_manifest(value)

    def test_pinned_source_identity_and_public_primary_boundary(self) -> None:
        manifest = _manifest()
        accepted = validate_manifest(manifest)
        self.assertEqual(accepted["source"]["dataset_id"], DATASET_ID)
        self.assertEqual(accepted["source"]["dataset_revision"], DATASET_REVISION)
        self.assertEqual(accepted["source"]["split"], "test")
        self.assertEqual(
            accepted["source"]["official_eval_repo_revision"], OFFICIAL_EVAL_REPO_REVISION
        )
        self.assertEqual(accepted["source"]["evaluation_code_license"], "MIT")
        self.assertEqual(accepted["source"]["evaluation_code_license_url"], EVALUATION_CODE_LICENSE_URL)
        self.assertEqual(accepted["source"]["dataset_card_url"], DATASET_CARD_URL)
        self.assertFalse(accepted["source"]["dataset_card_license_declared"])
        self.assertEqual(accepted["source"]["public_task_count"], PUBLIC_TASK_COUNT)
        self.assertEqual(accepted["source"]["private_task_count"], PRIVATE_TASK_COUNT)
        self.assertEqual(accepted["source"]["heldout_task_count"], HELDOUT_TASK_COUNT)
        self.assertEqual(accepted["source"]["total_task_count"], TOTAL_TASK_COUNT)
        self.assertEqual(accepted["split"]["role"], PRIMARY_EVAL)
        self.assertEqual(accepted["split"]["coverage"], "sampled_public")
        self.assertNotIn("heldout_proof", accepted)

    def test_task_and_split_hashes_are_order_independent_but_exact(self) -> None:
        ids = ["z-task", "a-task"]
        self.assertEqual(compute_task_ids_sha256(ids), compute_task_ids_sha256(reversed(ids)))
        with self.assertRaises(BoundaryValidationError):
            compute_task_ids_sha256(["a-task", "a-task"])
        with self.assertRaises(BoundaryValidationError):
            compute_task_ids_sha256([])

        manifest = _manifest()
        accepted = validate_manifest(manifest)
        manifest["split"]["task_ids_sha256"] = "0" * 64
        self.assert_rejected(manifest)
        self.assertEqual(len(accepted["tasks"]), 2)

        manifest = _manifest()
        manifest["tasks"].reverse()
        self.assert_rejected(manifest)

    def test_malformed_json_types_are_rejected_as_boundary_errors(self) -> None:
        mutations = (
            ("split", "role", []),
            ("split", "subset", {"public": True}),
            ("split", "coverage", ["sampled_public"]),
        )
        for section, key, value in mutations:
            manifest = _manifest()
            manifest[section][key] = value
            self.assert_rejected(manifest, f"{section}.{key}")

        manifest = _manifest()
        manifest["tasks"][0]["requirements"] = {"pip": []}
        self.assert_rejected(manifest)

        heldout = _manifest(role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout")
        heldout["heldout_proof"]["subset"] = []
        self.assert_rejected(heldout)

    def test_related_benchmarks_and_xlam_are_not_substitutes(self) -> None:
        for benchmark_id in ("swe_bench", "swe_bench_verified", "xlam_component_only"):
            manifest = _manifest()
            manifest["benchmark_id"] = benchmark_id
            self.assert_rejected(manifest, benchmark_id)

        manifest = _manifest()
        manifest["source"]["dataset_id"] = "Salesforce/xlam-function-calling-60k"
        self.assert_rejected(manifest)

    def test_revision_license_and_official_source_aliases_are_immutable(self) -> None:
        for key, value in (
            ("dataset_revision", "1" * 40),
            ("official_eval_repo_revision", "2" * 40),
            ("evaluation_code_license", "Apache-2.0"),
            ("evaluation_code_license_url", "https://github.com/scaleapi/SWE-bench_Pro-os"),
            ("dataset_card_url", "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro"),
            ("dataset_card_license_declared", True),
            ("public_task_count", PUBLIC_TASK_COUNT + 1),
            ("official_eval_repo_url", "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro"),
        ):
            manifest = _manifest()
            manifest["source"][key] = value
            self.assert_rejected(manifest, key)

    def test_native_environment_artifact_and_verifier_contracts_are_required(self) -> None:
        manifest = _manifest()
        manifest["environment"]["runtime"] = "local"
        self.assert_rejected(manifest)

        manifest = _manifest()
        manifest["environment"]["container_image_registry"] = "untrusted/images"
        self.assert_rejected(manifest)

        manifest = _manifest()
        manifest["environment"]["artifact_contract"]["native_artifact"] = "text"
        self.assert_rejected(manifest)

        manifest = _manifest()
        manifest["environment"]["verifier_contract"]["resolve_rule"] = "exact_match"
        self.assert_rejected(manifest)

    def test_each_task_requires_immutable_repo_container_license_and_test_receipts(self) -> None:
        for key, value in (
            ("base_commit", "not-a-commit"),
            ("container_digest", "sha256:not-a-digest"),
            ("dockerhub_tag", "tag with spaces"),
        ):
            manifest = _manifest()
            manifest["tasks"][0][key] = value
            self.assert_rejected(manifest, key)

        manifest = _manifest()
        del manifest["tasks"][0]["license_receipt"]
        self.assert_rejected(manifest)

        for receipt_hash in ("not-a-hash", "f" * 63):
            manifest = _manifest()
            manifest["tasks"][0]["license_receipt"]["receipt_sha256"] = receipt_hash
            self.assert_rejected(manifest, receipt_hash)

        manifest = _manifest()
        manifest["tasks"][0]["fail_to_pass"] = "[]"
        self.assert_rejected(manifest)

        manifest = _manifest()
        manifest["tasks"][0]["fail_to_pass"] = "__import__('os').system('false')"
        self.assert_rejected(manifest)

        manifest = _manifest()
        manifest["tasks"][0]["artifact_contract"]["required"] = False
        self.assert_rejected(manifest)

        manifest = _manifest()
        manifest["tasks"][0]["verifier_contract"]["requires_native_tests"] = False
        self.assert_rejected(manifest)

    def test_full_public_count_is_pinned_but_sampled_receipts_are_explicit(self) -> None:
        manifest = _manifest(coverage="full_public")
        self.assert_rejected(manifest)
        manifest = _manifest(coverage="sampled_public")
        self.assertEqual(len(validate_manifest(manifest)["tasks"]), 2)

    def test_heldout_requires_an_explicit_proof_and_is_not_public_primary(self) -> None:
        heldout = _manifest(role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout")
        accepted = validate_manifest(heldout)
        self.assertEqual(accepted["split"]["role"], RECEIPT_PROVEN_HELDOUT)
        self.assertEqual(accepted["heldout_proof"]["status"], "receipt_proven")

        primary = _manifest()
        primary["split"]["role"] = RECEIPT_PROVEN_HELDOUT
        self.assert_rejected(primary)

        heldout = _manifest(role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout")
        del heldout["heldout_proof"]
        self.assert_rejected(heldout)

        heldout = _manifest(role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout")
        heldout["heldout_proof"]["task_ids_sha256"] = "0" * 64
        self.assert_rejected(heldout)

        private = _manifest(
            role=RECEIPT_PROVEN_HELDOUT,
            coverage="receipt_proven_heldout",
            receipt_subset=PRIVATE_SUBSET,
        )
        self.assertEqual(validate_manifest(private)["split"]["subset"], PRIVATE_SUBSET)

        mismatched = _manifest(role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout")
        mismatched["heldout_proof"]["source_identity"]["split"] = PRIVATE_SUBSET
        self.assert_rejected(mismatched)

        mismatched = _manifest(role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout")
        mismatched["heldout_proof"]["access_receipt"]["source_revision"] = "c" * 40
        self.assert_rejected(mismatched)

        public_identity = _manifest(
            role=RECEIPT_PROVEN_HELDOUT, coverage="receipt_proven_heldout"
        )
        public_identity["heldout_proof"]["source_identity"] = {
            "source_id": DATASET_ID,
            "source_url": "https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro",
            "source_revision": DATASET_REVISION,
            "split": HELDOUT_SUBSET,
        }
        self.assert_rejected(public_identity)

    def test_completed_result_receipt_requires_all_three_tracking_systems(self) -> None:
        manifest = _manifest()
        result = _result(manifest)
        accepted = validate_result_receipt(result, manifest)
        self.assertEqual(accepted["wandb_run_id"], "wandb-swe-pro-001")
        self.assertEqual(accepted["tinker_run_id"], "tinker-swe-pro-001:eval:0")
        self.assertEqual(accepted["hf_receipt"]["commit_sha"], "e" * 40)

        for key in ("wandb_run_id", "tinker_run_id", "hf_receipt"):
            broken = copy.deepcopy(result)
            del broken[key]
            if key == "wandb_run_id":
                del broken["wandb"]
            if key == "tinker_run_id":
                del broken["tinker"]
            with self.assertRaises(BoundaryValidationError, msg=key):
                validate_result_receipt(broken, manifest)

        broken = copy.deepcopy(result)
        broken["wandb"]["run_id"] = "different"
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(broken, manifest)

    def test_result_is_bound_to_task_hashes_counts_native_receipts_and_hf_revision(self) -> None:
        manifest = _manifest()
        result = _result(manifest)
        result["task_ids_sha256"] = "0" * 64
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["evaluated_task_ids"] = result["evaluated_task_ids"][:-1]
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["resolve_rate"] = 0.9
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["hf_receipt"]["commit_sha"] = "not-immutable"
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["environment_receipt"]["container_digests"] = []
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["verifier_receipt"]["pass_to_pass_passed"] = False
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["task_results"][0]["container_digest"] = manifest["tasks"][1]["container_digest"]
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["task_results"][0]["tests"] = result["task_results"][0]["tests"][:1]
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["task_results"][0]["tests"].append(
            {"name": "tests/test_fixture.py::test_bug", "status": "PASSED"}
        )
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

        result = _result(manifest)
        result["task_results"].reverse()
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

    def test_primary_result_cannot_make_a_heldout_claim(self) -> None:
        manifest = _manifest()
        result = _result(manifest)
        result["heldout_claim"] = True
        with self.assertRaises(BoundaryValidationError):
            validate_result_receipt(result, manifest)

    def test_boundary_wrapper_and_file_parser_are_offline(self) -> None:
        manifest = _manifest()
        result = _result(manifest)
        accepted = validate_evaluation_boundary(manifest, result)
        self.assertEqual(accepted["result"]["status"], "completed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(parse_manifest(path)["benchmark_id"], BENCHMARK_ID)

            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"benchmark_id":"a","benchmark_id":"b"}')
            with self.assertRaises(BoundaryValidationError):
                parse_manifest(duplicate)

            trailing = Path(directory) / "trailing.json"
            trailing.write_text('{"benchmark_id":"a"}{"benchmark_id":"b"}')
            with self.assertRaises(BoundaryValidationError):
                parse_manifest(trailing)

    def test_cli_reports_only_safe_summary(self) -> None:
        manifest = _manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(path)]), 0)
            self.assertIn('"status": "accepted"', output.getvalue())
            self.assertNotIn("problem_statement", output.getvalue())
            self.assertNotIn("access_receipt", output.getvalue())


# Keep the helper annotation local so the test file has no dependency on a
# runtime protocol or external typing package.
MappingLike = dict[str, Any]


if __name__ == "__main__":
    unittest.main()
