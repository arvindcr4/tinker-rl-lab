"""Offline tests for the exact E8 Life-Sci-Bench evaluation boundary."""

from __future__ import annotations

import copy
import unittest

try:
    from flagship import pavlov_lifescibench_eval_adapter as e8
except ModuleNotFoundError:  # Direct execution from the flagship directory.
    import pavlov_lifescibench_eval_adapter as e8


def _synthetic_ready_boundary() -> dict:
    value = copy.deepcopy(e8.build_offline_e8_boundary())
    revision = "a" * 40
    license_id = "life-sci-bench-license-test"
    task_id = "e8-synthetic-science-task-0001"
    task = {
        "task_id": task_id,
        "task_id_hash": e8._task_hash(task_id, revision),
        "family": "synthetic_family_for_schema_test",
        "domain": "science",
        "split": "evaluation",
        "artifact_expected": True,
    }
    value["dataset"].update(
        {"revision": revision, "license_id": license_id, "license_status": "approved"}
    )
    value["native_environment"]["revision"] = revision
    value["native_verifier"]["revision"] = revision
    value["task_manifest"] = [task]
    value["eval_split_manifest_hash"] = e8.task_manifest_hash([task])
    value["train_split_manifest_hash"] = "b" * 64
    return value


def _synthetic_receipt() -> dict:
    boundary = _synthetic_ready_boundary()
    revision = boundary["dataset"]["revision"]
    task = boundary["task_manifest"][0]
    row = {
        "task_id": task["task_id"],
        "task_id_hash": task["task_id_hash"],
        "family": task["family"],
        "domain": task["domain"],
        "observation_hash": "1" * 64,
        "action_hash": "2" * 64,
        "state_hash": "3" * 64,
        "artifact_digest": "4" * 64,
        "task_success": True,
    }
    proof = {
        "train_split_manifest_hash": boundary["train_split_manifest_hash"],
        "eval_split_manifest_hash": boundary["eval_split_manifest_hash"],
        "disjoint_task_ids": True,
        "disjoint_family_ids": True,
        "unseen_families": [task["family"]],
    }
    proof["proof_hash"] = e8.sha256_json(proof)
    value = {
        "schema_version": e8.RECEIPT_SCHEMA_VERSION,
        "receipt_status": e8.RECEIPT_STATUS_OBSERVED,
        "source": copy.deepcopy(boundary["source"]),
        "dataset": copy.deepcopy(boundary["dataset"]),
        "role": e8.ROLE,
        "split": e8.SPLIT,
        "dataset_revision": revision,
        "license_id": boundary["dataset"]["license_id"],
        "task_manifest": copy.deepcopy(boundary["task_manifest"]),
        "eval_split_manifest_hash": boundary["eval_split_manifest_hash"],
        "heldout_proof": proof,
        "native_verifier": {
            "name": e8.NATIVE_VERIFIER_NAME,
            "environment_name": e8.NATIVE_ENVIRONMENT_NAME,
            "environment_revision": revision,
            "observation_schema": e8.NATIVE_OBSERVATION_SCHEMA,
            "verifier_revision": revision,
            "checked": True,
            "stateful": True,
            "artifact_or_side_effect": True,
            "artifact_required": True,
            "episode_rows": [row],
        },
        "wandb": {
            "observed": True,
            "run_id": "e8_run_20260809",
            "url": "https://wandb.ai/example/e8/runs/e8_run_20260809",
            "project": "tinker-rl-lab-pavlov",
            "config_hash": "5" * 64,
            "sample_manifest_hash": "6" * 64,
            "metrics": {
                "eval/lifescibench_success_rate": 1.0,
                "eval/lifescibench_reward_mean": 1.0,
                "eval/lifescibench_action_count_mean": 1.0,
            },
        },
        "tinker": {
            "observed": True,
            "run_id": "e8_tinker_20260809",
            "initial_sampler": "tinker://e8/initial",
            "periodic_samplers": ["tinker://e8/step-0001"],
            "final_sampler": "tinker://e8/final",
            "checkpoint_receipt": "checkpoints/e8.json",
        },
        "hf": {
            "observed": True,
            "repository": "example/e8-lifescibench",
            "revision": "c" * 40,
            "checkpoint_manifest": "artifact-manifest/e8.json",
            "c0_receipt": "receipts/e8-c0.json",
            "exported": True,
        },
        "cost": {
            "currency": "USD",
            "charged_usd": 0.0,
            "cap_usd": 1.16,
            "within_cap": True,
        },
        "substitute_suite_id": None,
        "e6_substitute": False,
        "xlam_substitute": False,
        "portfolio_evidence": False,
        "claim_boundary": e8.CLAIM_BOUNDARY,
    }
    value["receipt_hash"] = e8.sha256_json(value)
    return value


class LifeSciBenchEvalAdapterTests(unittest.TestCase):
    def test_offline_boundary_is_authoritative_but_blocked(self) -> None:
        value = e8.build_offline_e8_boundary()
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertEqual(value["source"]["source_id"], e8.SOURCE_ID)
        self.assertEqual(value["source"]["url"], e8.SOURCE_URL)
        self.assertEqual(value["role"], "primary_eval")
        self.assertEqual(value["split"], "evaluation")
        self.assertFalse(value["claims"]["receipt_proven_heldout"])
        self.assertFalse(result.metrics["paid_launch_authorized"])
        self.assertTrue(any("immutable" in error or "license" in error for error in result.errors))

    def test_synthetic_pinned_boundary_is_schema_valid_not_result_evidence(self) -> None:
        value = _synthetic_ready_boundary()
        result = e8.validate_e8_boundary(value)
        self.assertTrue(result.ok, result.errors)
        self.assertTrue(result.metrics["primary_eval"])
        self.assertFalse(result.metrics["receipt_proven_heldout"])
        self.assertFalse(result.metrics["portfolio_evidence"])
        self.assertFalse(result.metrics["paid_launch_authorized"])

    def test_source_identity_and_role_cannot_drift(self) -> None:
        value = _synthetic_ready_boundary()
        value["source"]["url"] = "https://example.invalid/lifescibench"
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("authoritative E8 identity" in error for error in result.errors))

        value = _synthetic_ready_boundary()
        value["role"] = "train"
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("primary_eval" in error for error in result.errors))

    def test_dataset_license_identity_is_pinned(self) -> None:
        value = _synthetic_ready_boundary()
        value["dataset"]["license_id"] = e8.UNPINNED
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("license_id must be pinned" in error for error in result.errors))

        value = _synthetic_receipt()
        value["dataset"]["revision"] = "d" * 40
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=_synthetic_ready_boundary())
        self.assertFalse(result.ok)
        self.assertTrue(any("dataset.revision" in error for error in result.errors))

    def test_task_ids_and_split_hashes_are_deterministic(self) -> None:
        value = _synthetic_ready_boundary()
        task = value["task_manifest"][0]
        task["task_id_hash"] = "0" * 64
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("task hash" in error for error in result.errors))

        value = _synthetic_ready_boundary()
        value["eval_split_manifest_hash"] = "0" * 64
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("eval_split_manifest_hash" in error for error in result.errors))

    def test_native_environment_and_artifact_verifier_contract_is_required(self) -> None:
        value = _synthetic_ready_boundary()
        value["native_environment"]["artifact_or_side_effect"] = False
        value["native_verifier"]["artifact_required"] = False
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("artifact" in error for error in result.errors))

    def test_receipt_proven_heldout_is_distinct_from_protocol_primary_eval(self) -> None:
        boundary = _synthetic_ready_boundary()
        boundary_result = e8.validate_e8_boundary(boundary)
        self.assertTrue(boundary_result.ok, boundary_result.errors)
        self.assertFalse(boundary_result.metrics["receipt_proven_heldout"])

        observed = _synthetic_receipt()
        result = e8.validate_e8_receipt(observed, boundary=boundary)
        self.assertTrue(result.ok, result.errors)
        self.assertTrue(result.metrics["primary_eval"])
        self.assertTrue(result.metrics["receipt_proven_heldout"])
        self.assertFalse(result.metrics["portfolio_evidence"])
        self.assertFalse(result.metrics["paid_launch_authorized"])

    def test_receipt_task_state_action_artifact_hashes_fail_closed(self) -> None:
        boundary = _synthetic_ready_boundary()
        value = _synthetic_receipt()
        value["native_verifier"]["episode_rows"][0]["state_hash"] = "0" * 64
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("state_hash" in error for error in result.errors))

        value = _synthetic_receipt()
        value["task_manifest"][0]["task_id"] = "tampered-task"
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("task_id_hash" in error or "task hash" in error for error in result.errors))

    def test_native_and_external_receipt_evidence_is_required(self) -> None:
        boundary = _synthetic_ready_boundary()
        value = _synthetic_receipt()
        value["native_verifier"]["checked"] = False
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("checked" in error for error in result.errors))

        value = _synthetic_receipt()
        value["hf"]["exported"] = False
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("hf.exported" in error for error in result.errors))

    def test_xlam_and_related_benchmarks_cannot_substitute(self) -> None:
        boundary = _synthetic_ready_boundary()
        value = _synthetic_receipt()
        value["substitute_suite_id"] = "xlam"
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("substitution" in error for error in result.errors))

        value = _synthetic_receipt()
        value["e6_substitute"] = True
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("e6_substitute" in error for error in result.errors))

        value = _synthetic_receipt()
        value["substitute_suite_id"] = "unlisted-related-benchmark"
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        self.assertFalse(result.ok)
        self.assertTrue(any("any non-null substitute_suite_id" in error for error in result.errors))

    def test_secret_like_fields_are_rejected_without_redaction(self) -> None:
        value = _synthetic_ready_boundary()
        value["dataset"]["api_key"] = "must-not-appear"
        result = e8.validate_e8_boundary(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("secret-like field" in error for error in result.errors))

    def test_receipt_hash_and_proof_hash_are_deterministic(self) -> None:
        value = _synthetic_receipt()
        self.assertEqual(
            value["receipt_hash"],
            e8.sha256_json({key: item for key, item in value.items() if key != "receipt_hash"}),
        )
        self.assertEqual(
            value["heldout_proof"]["proof_hash"],
            e8.sha256_json(
                {key: item for key, item in value["heldout_proof"].items() if key != "proof_hash"}
            ),
        )

    def test_no_live_runtime_or_network_imports(self) -> None:
        import ast
        from pathlib import Path

        path = Path(e8.__file__)
        tree = ast.parse(path.read_text(), filename=str(path))
        forbidden = {"requests", "httpx", "browsergym", "playwright", "wandb", "tinker", "os", "subprocess"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(forbidden.isdisjoint(imported), imported & forbidden)
        self.assertNotIn("os.environ", path.read_text())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
