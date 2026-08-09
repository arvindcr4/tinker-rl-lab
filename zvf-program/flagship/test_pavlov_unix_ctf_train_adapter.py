from __future__ import annotations

import copy
import unittest

from flagship.pavlov_unix_ctf_train_adapter import (
    AUTHORITATIVE_SOURCE_ID,
    AUTHORITATIVE_SOURCE_URL,
    BINARYAUDIT_SUITE_ID,
    BOUNDARY_SCHEMA_VERSION,
    DEFAULT_MAXIMUM_USD,
    DEFAULT_OPERATIONAL_CAP_USD,
    DEFAULT_PROVIDER,
    DEFAULT_RESERVE_USD,
    PRIMARY_EVAL_REGISTRY_ID,
    RESULT_SCHEMA_VERSION,
    SUITE_ID,
    TRAINING_ONLY_CLAIM,
    UnixCtfTrainBoundaryError,
    canonical_sha256,
    task_id_manifest_sha256,
    validate_unix_ctf_training_boundary,
    validate_unix_ctf_training_receipt,
)


SOURCE_REVISION = "a" * 40
LICENSE_HASH = "b" * 64
GENERATOR_REVISION = "c" * 40
CONTAINER_DIGEST = "sha256:" + "d" * 64
ENVIRONMENT_HASH = "e" * 64
SANDBOX_POLICY_HASH = "f" * 64
SANDBOX_LIMITS_HASH = "1" * 64
ARTIFACT_HASH = "2" * 64
VERIFIER_REVISION = "3" * 40


def _boundary() -> dict[str, object]:
    train = ["unix-train-001", "unix-train-002", "unix-train-003"]
    binary = ["binaryaudit-101", "binaryaudit-102"]
    primary = ["primary-001", "primary-002", "primary-003"]
    parameters = {"difficulty": "mixed", "shell": "/bin/bash", "task_family": "security"}
    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "role": "train",
        "source_identity": {
            "id": AUTHORITATIVE_SOURCE_ID,
            "url": AUTHORITATIVE_SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license_spdx": "Apache-2.0",
            "license_text_sha256": LICENSE_HASH,
        },
        "procedural_generation": {
            "generator_id": "unix-ctf-procedural-v1",
            "generator_revision": GENERATOR_REVISION,
            "seed": 809,
            "seed_sha256": canonical_sha256(809),
            "task_ids": train,
            "task_id_manifest_sha256": task_id_manifest_sha256(train),
            "parameters": parameters,
            "parameters_sha256": canonical_sha256(parameters),
        },
        "disjointness": {
            "binaryaudit_eval": {
                "suite_id": BINARYAUDIT_SUITE_ID,
                "task_ids": binary,
                "task_id_manifest_sha256": task_id_manifest_sha256(binary),
            },
            "primary_evals": {
                "registry_id": PRIMARY_EVAL_REGISTRY_ID,
                "task_ids": primary,
                "task_id_manifest_sha256": task_id_manifest_sha256(primary),
            },
        },
        "native_environment": {
            "mode": "native",
            "container_digest": CONTAINER_DIGEST,
            "environment_manifest_sha256": ENVIRONMENT_HASH,
            "shell_sandbox": {
                "enabled": True,
                "network": "disabled",
                "filesystem": "isolated",
                "policy_sha256": SANDBOX_POLICY_HASH,
                "limits_sha256": SANDBOX_LIMITS_HASH,
                "command_allowlist": ["bash", "sh", "coreutils"],
            },
            "artifact_contract": {
                "required": True,
                "manifest_sha256": ARTIFACT_HASH,
                "types": ["stdout", "stderr", "patch"],
            },
            "verifier_contract": {
                "name": "unix-ctf-native-verifier",
                "revision": VERIFIER_REVISION,
                "receipt_schema": RESULT_SCHEMA_VERSION,
                "checks": ["exit_code", "filesystem_state", "artifact_hash"],
            },
        },
        "budget_gate": {
            "provider": DEFAULT_PROVIDER,
            "authorized": True,
            "maximum_usd": DEFAULT_MAXIMUM_USD,
            "operational_cap_usd": DEFAULT_OPERATIONAL_CAP_USD,
            "reserve_usd": DEFAULT_RESERVE_USD,
        },
    }


def _receipt() -> dict[str, object]:
    boundary = _boundary()
    receipt: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "role": "train",
        "status": "completed",
        "claim_scope": TRAINING_ONLY_CLAIM,
        "source_identity": copy.deepcopy(boundary["source_identity"]),
        "procedural_generation": copy.deepcopy(boundary["procedural_generation"]),
        "disjointness_receipt": {
            "binaryaudit_eval_task_id_manifest_sha256": boundary["disjointness"]["binaryaudit_eval"]["task_id_manifest_sha256"],  # type: ignore[index]
            "primary_eval_task_id_manifest_sha256": boundary["disjointness"]["primary_evals"]["task_id_manifest_sha256"],  # type: ignore[index]
            "verified": True,
            "overlap_count": 0,
        },
        "native_environment_receipt": {
            "container_digest": CONTAINER_DIGEST,
            "environment_manifest_sha256": ENVIRONMENT_HASH,
            "shell_sandbox": copy.deepcopy(boundary["native_environment"]["shell_sandbox"]),  # type: ignore[index]
        },
        "artifact_receipt": {
            "manifest_sha256": ARTIFACT_HASH,
            "paths": ["artifacts/unix-train-001/patch.diff"],
        },
        "verifier_receipt": {
            "name": "unix-ctf-native-verifier",
            "revision": VERIFIER_REVISION,
            "receipt_schema": RESULT_SCHEMA_VERSION,
        },
        "wandb": {
            "run_id": "wandb-unix-ctf-1",
            "url": "https://wandb.example/run/unix-ctf-1",
            "mode": "online",
            "config_sha256": "4" * 64,
            "metrics_receipt_sha256": "5" * 64,
            "metrics_logged": True,
        },
        "tinker": {
            "run_id": "tinker-unix-ctf-1",
            "status": "completed",
            "config_sha256": "6" * 64,
            "cost_usd": 1.25,
        },
        "hf": {
            "repo": "private/unix-ctf-adapter",
            "revision": "7" * 40,
            "visibility": "private",
            "artifact_manifest_sha256": ARTIFACT_HASH,
        },
        "budget_receipt": {
            "provider": DEFAULT_PROVIDER,
            "authorized": True,
            "spent_usd": 0.20,
            "projected_next_cost_usd": 1.25,
            "within_operational_cap": True,
        },
        "metrics": {"episode_success_rate": 0.0},
    }
    return receipt


class BoundaryTests(unittest.TestCase):
    def test_valid_boundary_pins_procedure_native_sandbox_and_budget(self) -> None:
        result = validate_unix_ctf_training_boundary(_boundary())
        self.assertEqual(result["source_identity"]["id"], AUTHORITATIVE_SOURCE_ID)
        self.assertEqual(result["procedural_generation"]["seed"], 809)
        self.assertEqual(result["native_environment"]["shell_sandbox"]["network"], "disabled")
        self.assertEqual(result["budget_gate"]["operational_cap_usd"], 16.5)
        self.assertFalse(result["primary_eval_claim_permitted"])

    def test_overlapping_training_ids_are_rejected_against_both_registries(self) -> None:
        item = _boundary()
        item["procedural_generation"]["task_ids"] = ["binaryaudit-101", "unix-train-002", "unix-train-003"]  # type: ignore[index]
        item["procedural_generation"]["task_id_manifest_sha256"] = task_id_manifest_sha256(item["procedural_generation"]["task_ids"])  # type: ignore[index]
        with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
            validate_unix_ctf_training_boundary(item)
        self.assertTrue(any("BinaryAudit" in message for message in raised.exception.diagnostics))

        item = _boundary()
        item["procedural_generation"]["task_ids"] = ["primary-001", "unix-train-002", "unix-train-003"]  # type: ignore[index]
        item["procedural_generation"]["task_id_manifest_sha256"] = task_id_manifest_sha256(item["procedural_generation"]["task_ids"])  # type: ignore[index]
        with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
            validate_unix_ctf_training_boundary(item)
        self.assertTrue(any("primary-evaluation" in message for message in raised.exception.diagnostics))

    def test_revision_seed_hash_sandbox_and_budget_mutations_fail_closed(self) -> None:
        mutations = (
            (lambda item: item["source_identity"].update(revision="main"), "mutable tag/branch"),
            (lambda item: item["procedural_generation"].update(seed=810), "procedural seed"),
            (lambda item: item["native_environment"]["shell_sandbox"].update(network="enabled"), "network must be disabled"),
            (lambda item: item["budget_gate"].update(operational_cap_usd=17.0), "authorized $18/$16.50/$1.50"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                item = _boundary()
                mutate(item)
                with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
                    validate_unix_ctf_training_boundary(item)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

    def test_missing_native_receipts_and_disjointness_metadata_block(self) -> None:
        item = _boundary()
        del item["native_environment"]["shell_sandbox"]  # type: ignore[index]
        with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
            validate_unix_ctf_training_boundary(item)
        self.assertTrue(any("shell_sandbox is required" in message for message in raised.exception.diagnostics))

        item = _boundary()
        del item["disjointness"]["primary_evals"]  # type: ignore[index]
        with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
            validate_unix_ctf_training_boundary(item)
        self.assertTrue(any("primary_evals is required" in message for message in raised.exception.diagnostics))


class ReceiptTests(unittest.TestCase):
    def test_valid_training_receipt_is_not_primary_evidence(self) -> None:
        result = validate_unix_ctf_training_receipt(_boundary(), _receipt())
        self.assertEqual(result["status"], "admissible_training_receipt")
        self.assertTrue(result["training_only"])
        self.assertFalse(result["primary_eval"])
        self.assertFalse(result["primary_eval_claim_permitted"])
        self.assertEqual(result["tracking_receipts"]["wandb"]["mode"], "online")

    def test_receipt_cannot_claim_primary_or_holdout(self) -> None:
        for mutation in (
            lambda item: item.update(evaluation_role="primary_eval"),
            lambda item: item.update(primary_eval=True),
            lambda item: item.update(claim_scope="primary_eval"),
        ):
            receipt = _receipt()
            mutation(receipt)
            with self.assertRaises(UnixCtfTrainBoundaryError):
                validate_unix_ctf_training_receipt(_boundary(), receipt)

    def test_missing_or_unsafe_tracking_and_budget_fail_closed(self) -> None:
        mutations = (
            (lambda item: item["wandb"].update(mode="offline"), "W&B receipt mode"),
            (lambda item: item["hf"].update(visibility="public"), "visibility"),
            (lambda item: item["budget_receipt"].update(projected_next_cost_usd=20.0), "operational cap"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                receipt = _receipt()
                mutate(receipt)
                with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
                    validate_unix_ctf_training_receipt(_boundary(), receipt)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

        receipt = _receipt()
        del receipt["tinker"]
        with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
            validate_unix_ctf_training_receipt(_boundary(), receipt)
        self.assertTrue(any("missing Tinker" in message for message in raised.exception.diagnostics))

    def test_provenance_environment_verifier_and_task_hash_drift_rejected(self) -> None:
        mutations = (
            (lambda item: item["procedural_generation"].update(generator_revision="main"), "differs from boundary"),
            (lambda item: item["disjointness_receipt"].update(verified=False), "disjointness must be verified"),
            (lambda item: item["native_environment_receipt"]["shell_sandbox"].update(network="enabled"), "shell_sandbox.network differs"),
            (lambda item: item["verifier_receipt"].update(revision="8" * 40), "verifier receipt revision differs"),
        )
        for mutate, expected in mutations:
            with self.subTest(expected=expected):
                receipt = _receipt()
                mutate(receipt)
                with self.assertRaises(UnixCtfTrainBoundaryError) as raised:
                    validate_unix_ctf_training_receipt(_boundary(), receipt)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

    def test_nonfinite_metric_is_not_an_admissible_receipt(self) -> None:
        receipt = _receipt()
        receipt["metrics"]["episode_success_rate"] = float("nan")  # type: ignore[index]
        with self.assertRaises(UnixCtfTrainBoundaryError):
            validate_unix_ctf_training_receipt(_boundary(), receipt)


if __name__ == "__main__":
    unittest.main()
