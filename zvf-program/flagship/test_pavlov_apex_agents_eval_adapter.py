from __future__ import annotations

import copy
import unittest

from flagship.pavlov_apex_agents_eval_adapter import (
    EXPECTED_ARCHIPELAGO_URL,
    EXPECTED_BENCHMARK_NAME,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_LICENSE,
    EXPECTED_DATASET_URL,
    EXPECTED_JOB_CATEGORIES,
    EXPECTED_PAPER_URL,
    EXPECTED_PAPER_REVISION,
    EXPECTED_SOURCE_URL,
    EXPECTED_TASK_COUNT,
    EXPECTED_WORLD_COUNT,
    SCHEMA_VERSION,
    SUITE_ID,
    _access_receipt_payload,
    _artifact_contract_hash,
    _artifact_receipt_hash,
    _dataset_revision_receipt_hash,
    _environment_receipt_hash,
    _hf_receipt_hash,
    _license_approval_hash,
    _metadata_receipt_payload,
    _source_identity_hash,
    _split_hash,
    _split_receipt_hash,
    _task_manifest_hash,
    _task_id_hash,
    _tinker_receipt_hash,
    _verifier_receipt_hash,
    _wandb_receipt_hash,
    build_boundary,
    is_valid_boundary,
    sha256_json,
    validate_boundary,
)
from flagship.pavlovs_domain_contract import load_contract


def valid_metadata() -> dict:
    verifier_rule = (
        "primary reward must inspect environment state or native artifacts "
        "whenever task correctness depends on them"
    )
    revisions = ["1" * 40, "2" * 40, "3" * 40]
    checkpoints = [
        {
            "stage": stage,
            "repo_url": "https://huggingface.co/example/apex-agents-e5",
            "revision": revision,
            "url": f"https://huggingface.co/example/apex-agents-e5/commit/{revision}",
            "visibility": "public" if stage != "periodic" else "private",
            "safe_public_artifact": stage != "periodic",
            "data_license_safe": stage != "periodic",
            "quota_safe": stage != "periodic",
            "private_artifact_safe": stage == "periodic",
        }
        for stage, revision in zip(("initial", "periodic", "final"), revisions)
    ]
    for checkpoint in checkpoints:
        checkpoint["receipt_hash"] = _hf_receipt_hash(checkpoint)
    source = {
        "source_id": "mercor-apex-agents",
        "publisher": "Mercor",
        "url": EXPECTED_SOURCE_URL,
        "authoritative": True,
    }
    source["identity_hash"] = _source_identity_hash(source)
    dataset = {
        "dataset_id": EXPECTED_DATASET_ID,
        "revision": "4" * 40,
        "revision_source_url": EXPECTED_DATASET_URL,
        "license": {
            "license_id": EXPECTED_DATASET_LICENSE,
            "approved": True,
            "source_url": EXPECTED_DATASET_URL,
        },
    }
    dataset["revision_receipt_hash"] = _dataset_revision_receipt_hash(dataset)
    dataset["license"]["approval_hash"] = _license_approval_hash(dataset["license"])
    task_ids = ["apex-task-001", "apex-task-002", "apex-task-003"]
    split = {
        "name": "sealed selection slice",
        "disjoint_from_training": True,
    }
    task_id_hash = _task_id_hash(task_ids)
    split_manifest_hash = _split_hash(split["name"], task_ids)
    split["receipt_hash"] = _split_receipt_hash(split, task_id_hash, split_manifest_hash)
    environment = {
        "native": True,
        "runtime": "archipelago-native-environment-v1",
        "container_digest": "0" * 64,
        "environment_digest": "1" * 64,
        "source_url": EXPECTED_ARCHIPELAGO_URL,
    }
    environment["receipt_hash"] = _environment_receipt_hash(environment)
    artifacts = {
        "required": True,
        "artifact_types": ["environment_snapshot", "trajectory", "grading_result", "artifact_metadata"],
        "source_url": EXPECTED_ARCHIPELAGO_URL,
    }
    artifacts["contract_hash"] = _artifact_contract_hash(artifacts)
    artifacts["receipt_hash"] = _artifact_receipt_hash(artifacts)
    verifier = {
        "rule": verifier_rule,
        "verifier_id": "archipelago-grading",
        "revision": "6" * 40,
        "verifier_hash": "7" * 64,
        "source_url": EXPECTED_ARCHIPELAGO_URL,
    }
    verifier["receipt_hash"] = _verifier_receipt_hash(verifier)
    wandb = {
        "online": True,
        "entity": "entity",
        "project": "pavlov",
        "group": "apex-agents-e5",
        "run_id": "apex-run-1",
        "run_url": "https://wandb.ai/entity/pavlov/runs/apex-run-1",
        "state": "finished",
        "success": True,
    }
    wandb["receipt_hash"] = _wandb_receipt_hash(wandb)
    tinker = {
        "provider": "Tinker",
        "run_id": "tinker-apex-1",
        "cost_status": "observed",
    }
    tinker["receipt_hash"] = _tinker_receipt_hash(tinker)
    access = {
        "contact_acceptance_required": True,
        "contact_acceptance_confirmed": True,
        "dataset_access_confirmed": True,
        "read_only_snapshot": True,
        "web_search_enabled": False,
        "network_used": False,
        "paid_calls_made": False,
    }
    access["access_receipt_hash"] = sha256_json(_access_receipt_payload(access))
    upstream = {
        "verification_status": "verified",
        "official_sources": {
            "publisher_url": EXPECTED_SOURCE_URL,
            "dataset_url": EXPECTED_DATASET_URL,
            "paper_url": EXPECTED_PAPER_URL,
            "archipelago_url": EXPECTED_ARCHIPELAGO_URL,
            "source_receipt_hashes": {
                "publisher_url": "a" * 64,
                "dataset_url": "b" * 64,
                "paper_url": "c" * 64,
                "archipelago_url": "d" * 64,
            },
        },
        "benchmark_name": EXPECTED_BENCHMARK_NAME,
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_revision": dataset["revision"],
        "paper_revision": EXPECTED_PAPER_REVISION,
        "task_count": EXPECTED_TASK_COUNT,
        "world_count": EXPECTED_WORLD_COUNT,
        "job_categories": EXPECTED_JOB_CATEGORIES,
        "license": EXPECTED_DATASET_LICENSE,
        "intended_use": "evaluation_only",
        "training_permitted": False,
        "crawling_permitted": False,
        "access_constraints": access,
    }
    upstream["metadata_receipt_hash"] = sha256_json(_metadata_receipt_payload(upstream))
    # This is deterministic offline fixture data, never a claimed benchmark
    # result, and is not used by production code.
    return {
        "authoritative_source": source,
        "dataset": dataset,
        "upstream_metadata": upstream,
        "task_ids": task_ids,
        "task_id_hash": task_id_hash,
        "split_manifest_hash": split_manifest_hash,
        "task_manifest_hash": _task_manifest_hash(dataset["revision"], split["name"], task_ids),
        "split": split,
        "native_environment": environment,
        "artifact_contract": artifacts,
        "verifier_contract": verifier,
        "result_receipts": {
            "wandb": wandb,
            "tinker": tinker,
            "hf_checkpoints": checkpoints,
        },
        "receipt_proven_heldout": False,
        "heldout_claim_allowed": False,
        "related_benchmarks": [],
        "xlam_substitute": False,
        "evidence_status": "prospective",
        "scientific_evidence_status": "not_established",
    }


class PavlovApexAgentsEvalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()

    def test_placeholder_boundary_is_exact_primary_eval_and_blocked(self) -> None:
        boundary = build_boundary(contract=self.contract)
        self.assertEqual(boundary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(boundary["suite_id"], SUITE_ID)
        self.assertEqual(boundary["role"], "primary_eval")
        self.assertTrue(boundary["primary_eval"])
        self.assertEqual(boundary["status"], "BLOCKED")
        self.assertFalse(boundary["launchable"])
        self.assertTrue(boundary["blockers"])
        self.assertTrue(isinstance(boundary["boundary_hash"], str))
        self.assertTrue(any("authoritative_source" in error for error in boundary["blockers"]))
        self.assertIsNone(boundary["dataset"]["dataset_id"])
        self.assertEqual(boundary["artifact_contract"]["artifact_types"], [])
        self.assertTrue(any("upstream_metadata" in error for error in boundary["blockers"]))

    def test_builder_never_invents_task_or_split_receipts(self) -> None:
        metadata = valid_metadata()
        for name in ("task_id_hash", "split_manifest_hash", "task_manifest_hash"):
            metadata.pop(name)
        metadata["split"].pop("receipt_hash")
        boundary = build_boundary(metadata, self.contract)
        self.assertEqual(boundary["status"], "BLOCKED")
        self.assertIsNone(boundary["task_id_hash"])
        self.assertIsNone(boundary["split_manifest_hash"])
        self.assertIsNone(boundary["task_manifest_hash"])
        self.assertTrue(any("task_id_hash" in error for error in boundary["blockers"]))
        self.assertTrue(any("split.receipt_hash" in error for error in boundary["blockers"]))

    def test_complete_local_metadata_builds_only_a_valid_boundary(self) -> None:
        boundary = build_boundary(valid_metadata(), self.contract)
        self.assertEqual(validate_boundary(boundary, self.contract), [])
        self.assertTrue(is_valid_boundary(boundary, self.contract))
        self.assertEqual(boundary["status"], "READY")
        self.assertTrue(boundary["provenance_ready"])
        self.assertFalse(boundary["launchable"])
        self.assertFalse(boundary["receipt_proven_heldout"])
        self.assertFalse(boundary["heldout_claim_allowed"])
        self.assertEqual(boundary["task_id_hash"], sha256_json(sorted(boundary["task_ids"])))

    def test_task_and_split_hashes_are_deterministic_and_mutation_sensitive(self) -> None:
        boundary = build_boundary(valid_metadata(), self.contract)
        mutated = copy.deepcopy(boundary)
        mutated["task_ids"][0] = "apex-task-000"
        self.assertTrue(any("task_id_hash" in error for error in validate_boundary(mutated, self.contract)))

        mutated = copy.deepcopy(boundary)
        mutated["split_manifest_hash"] = "0" * 64
        self.assertTrue(any("split_manifest_hash" in error for error in validate_boundary(mutated, self.contract)))

        mutated = copy.deepcopy(boundary)
        mutated["task_ids"] = sorted(mutated["task_ids"], reverse=True)
        self.assertTrue(any("task_ids" in error for error in validate_boundary(mutated, self.contract)))

    def test_source_revision_license_environment_artifact_and_verifier_are_pinned(self) -> None:
        for field, value, needle in (
            ("authoritative_source", {"url": "https://evil.example/apex", "authoritative": True}, "authoritative_source"),
            ("dataset", {"revision": "main"}, "dataset"),
            ("native_environment", {"native": True, "runtime": "runtime", "container_digest": "latest"}, "native_environment"),
            ("artifact_contract", {"required": True, "artifact_types": ["x"]}, "artifact_contract"),
            ("verifier_contract", {"rule": "wrong", "revision": "main"}, "verifier_contract"),
        ):
            boundary = build_boundary(valid_metadata(), self.contract)
            boundary[field] = value
            errors = validate_boundary(boundary, self.contract)
            self.assertTrue(any(needle in error for error in errors), field)

    def test_authoritative_upstream_metadata_and_access_constraints_are_exact(self) -> None:
        for mutate, needle in (
            (lambda value: value["official_sources"].__setitem__("dataset_url", "https://huggingface.co/datasets/neyralabs/apex-agents"), "official_sources.dataset_url"),
            (lambda value: value.__setitem__("task_count", 479), "upstream_metadata.task_count"),
            (lambda value: value.__setitem__("paper_revision", "v2"), "upstream_metadata.paper_revision"),
            (lambda value: value.__setitem__("dataset_revision", "5" * 40), "upstream_metadata.dataset_revision"),
            (lambda value: value.__setitem__("training_permitted", True), "upstream_metadata.training_permitted"),
            (lambda value: value["access_constraints"].__setitem__("contact_acceptance_confirmed", False), "access_constraints.contact_acceptance_confirmed"),
            (lambda value: value["access_constraints"].__setitem__("web_search_enabled", True), "access_constraints.web_search_enabled"),
            (lambda value: value["access_constraints"].__setitem__("network_used", True), "access_constraints.network_used"),
            (lambda value: value["access_constraints"].__setitem__("access_receipt_hash", "0" * 64), "access_constraints.access_receipt_hash"),
        ):
            boundary = build_boundary(valid_metadata(), self.contract)
            mutate(boundary["upstream_metadata"])
            errors = validate_boundary(boundary, self.contract)
            self.assertTrue(any(needle in error for error in errors), needle)

    def test_native_contract_receipts_are_bound_and_artifact_defaults_cannot_pass(self) -> None:
        for field, mutate, needle in (
            ("native_environment", lambda value: value.__setitem__("runtime", "different"), "native_environment"),
            ("artifact_contract", lambda value: value.__setitem__("artifact_types", []), "artifact_contract"),
            ("verifier_contract", lambda value: value.__setitem__("verifier_hash", "0" * 64), "verifier_contract"),
        ):
            boundary = build_boundary(valid_metadata(), self.contract)
            mutate(boundary[field])
            errors = validate_boundary(boundary, self.contract)
            self.assertTrue(any(needle in error for error in errors), field)

    def test_result_receipts_require_native_wandb_tinker_and_hf_proof(self) -> None:
        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["wandb"]["run_url"] = "https://evil.example/apex-run-1"
        self.assertTrue(any("W&B" in error for error in validate_boundary(boundary, self.contract)))

        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["tinker"]["provider"] = "other"
        self.assertTrue(any("Tinker" in error for error in validate_boundary(boundary, self.contract)))

        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["hf_checkpoints"][0]["quota_safe"] = False
        self.assertTrue(any("HF checkpoint" in error for error in validate_boundary(boundary, self.contract)))

        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["wandb"]["receipt_hash"] = "0" * 64
        self.assertTrue(any("W&B" in error for error in validate_boundary(boundary, self.contract)))

        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["hf_checkpoints"][0]["repo_url"] = EXPECTED_DATASET_URL
        self.assertTrue(any("HF checkpoint" in error for error in validate_boundary(boundary, self.contract)))

        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["hf_checkpoints"][0]["repo_url"] = "https://huggingface.co/datasets/example"
        boundary["result_receipts"]["hf_checkpoints"][0]["url"] = boundary["result_receipts"]["hf_checkpoints"][0]["repo_url"] + "/commit/" + boundary["result_receipts"]["hf_checkpoints"][0]["revision"]
        self.assertTrue(any("HF checkpoint" in error for error in validate_boundary(boundary, self.contract)))

        boundary = build_boundary(valid_metadata(), self.contract)
        boundary["result_receipts"]["wandb"]["online"] = 1
        self.assertTrue(any("W&B" in error for error in validate_boundary(boundary, self.contract)))

    def test_primary_eval_and_receipt_proven_heldout_semantics_are_separate(self) -> None:
        boundary = build_boundary(valid_metadata(), self.contract)
        self.assertTrue(boundary["primary_eval"])
        self.assertFalse(boundary["receipt_proven_heldout"])
        self.assertFalse(boundary["heldout_claim_allowed"])

        tampered = copy.deepcopy(boundary)
        tampered["heldout_claim_allowed"] = True
        self.assertTrue(any("heldout_claim_allowed" in error for error in validate_boundary(tampered, self.contract)))

        tampered = copy.deepcopy(boundary)
        tampered["related_benchmarks"] = ["agentharm_eval"]
        self.assertTrue(any("related_benchmarks" in error for error in validate_boundary(tampered, self.contract)))

        tampered = copy.deepcopy(boundary)
        tampered["xlam_substitute"] = True
        self.assertTrue(any("xlam_substitute" in error for error in validate_boundary(tampered, self.contract)))

        tampered = copy.deepcopy(boundary)
        tampered["source_split_description"] = "held-out worlds"
        self.assertTrue(any("source_split_description" in error for error in validate_boundary(tampered, self.contract)))

        tampered = copy.deepcopy(boundary)
        tampered["receipt_proven_heldout"] = True
        tampered["heldout_claim_allowed"] = True
        tampered["heldout_receipt"] = {"receipt_hash": "a" * 64}
        self.assertTrue(any("heldout_receipt" in error for error in validate_boundary(tampered, self.contract)))

    def test_boundary_hash_is_mandatory_and_detects_mutation(self) -> None:
        boundary = build_boundary(valid_metadata(), self.contract)
        tampered = copy.deepcopy(boundary)
        tampered["boundary_hash"] = "0" * 64
        self.assertTrue(any("boundary_hash" in error for error in validate_boundary(tampered, self.contract)))

        tampered = copy.deepcopy(boundary)
        del tampered["boundary_hash"]
        self.assertTrue(any("boundary_hash" in error for error in validate_boundary(tampered, self.contract)))


if __name__ == "__main__":
    unittest.main()
