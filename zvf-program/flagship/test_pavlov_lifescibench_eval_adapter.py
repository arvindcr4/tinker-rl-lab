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


def _neutral_plumbing_payload() -> dict:
    """Run the builder chain with non-synthetic identifiers.

    This exists to prove the local plumbing is schema-complete: the ONLY thing
    separating it from ``build_synthetic_fixture`` is the marker.  It is never
    written to disk and carries no measurement.
    """

    revision = "e" * 40
    eval_manifest = e8.build_split_manifest(
        [
            {
                "task_id": "PLUMBING-CHECK-eval-0001",
                "family": "plumbing-family-eval-a",
                "domain": "science",
                "workflow": "evidence_handling",
                "bio_domain": "genomics",
            },
            {
                "task_id": "PLUMBING-CHECK-eval-0002",
                "family": "plumbing-family-eval-b",
                "domain": "tool_use",
                "workflow": "analysis",
                "bio_domain": "assays_screening",
            },
        ],
        dataset_revision=revision,
        split=e8.SPLIT,
    )
    train_manifest = e8.build_split_manifest(
        [
            {
                "task_id": "PLUMBING-CHECK-train-0001",
                "family": "plumbing-family-train-a",
                "domain": "science",
                "workflow": "scientific_reasoning",
                "bio_domain": "molecular_cell_biology",
            }
        ],
        dataset_revision=revision,
        split="train",
    )
    proof = e8.build_heldout_proof(
        train_manifest=train_manifest, eval_manifest=eval_manifest
    )
    boundary = e8.build_pinned_boundary(
        dataset_revision=revision,
        license_id="plumbing-check-license",
        environment_revision="f" * 40,
        verifier_revision="a" * 40,
        eval_manifest=eval_manifest,
        train_split_manifest_hash=train_manifest["manifest_hash"],
    )
    receipt = e8.build_e8_result_receipt(
        boundary=boundary,
        heldout_proof=proof,
        episode_rows=[
            {
                "task_id": row["task_id"],
                "task_id_hash": row["task_id_hash"],
                "family": row["family"],
                "domain": row["domain"],
                "observation_hash": e8.sha256_json(["obs", row["task_id"]]),
                "action_hash": e8.sha256_json(["act", row["task_id"]]),
                "state_hash": e8.sha256_json(["state", row["task_id"]]),
                "artifact_digest": e8.sha256_json(["artifact", row["task_id"]]),
                "task_success": False,
            }
            for row in eval_manifest["rows"]
        ],
        wandb_evidence={
            "observed": True,
            "run_id": "plumbing_check_run",
            "url": "https://wandb.ai/plumbing/e8/runs/plumbing_check_run",
            "project": "plumbing-check",
            "config_hash": e8.sha256_json(["config"]),
            "sample_manifest_hash": e8.sha256_json(["samples"]),
            "metrics": {name: 0.0 for name in e8.WANDB_REQUIRED_METRICS},
        },
        tinker_evidence={
            "observed": True,
            "run_id": "plumbing_check_tinker",
            "initial_sampler": "tinker://plumbing/initial",
            "periodic_samplers": ["tinker://plumbing/step-0001"],
            "final_sampler": "tinker://plumbing/final",
            "checkpoint_receipt": "plumbing/checkpoint.json",
        },
        hf_evidence={
            "observed": True,
            "repository": "plumbing/e8-check",
            "revision": "b" * 40,
            "checkpoint_manifest": "plumbing/manifest.json",
            "c0_receipt": "plumbing/c0.json",
            "exported": True,
        },
        cost={"currency": "USD", "charged_usd": 0.0, "cap_usd": 1.0, "within_cap": True},
    )
    return {
        "eval_manifest": eval_manifest,
        "train_manifest": train_manifest,
        "heldout_proof": proof,
        "boundary": boundary,
        "receipt": receipt,
    }


class LifeSciBenchLocalBuilderTests(unittest.TestCase):
    """Cover everything the campaign can do locally, up to the access boundary."""

    def test_published_taxonomy_matches_primary_source(self) -> None:
        # openai.com/index/introducing-life-sci-bench/ and the linked preprint.
        self.assertEqual(e8.PUBLISHED_TASK_COUNT, 750)
        self.assertEqual(len(e8.PUBLISHED_WORKFLOWS), 7)
        self.assertEqual(len(e8.PUBLISHED_BIO_DOMAINS), 7)
        self.assertEqual(len(set(e8.PUBLISHED_WORKFLOWS)), 7)
        self.assertEqual(len(set(e8.PUBLISHED_BIO_DOMAINS)), 7)
        self.assertEqual(e8.PUBLISHED_RUBRIC_CRITERIA_COUNT, 19020)
        self.assertEqual(e8.PUBLISHED_ARTIFACT_COUNT, 1062)
        self.assertEqual(e8.PUBLISHED_PASS_THRESHOLD, "0.70")
        # The campaign capability tags are a separate axis from the bio domains.
        self.assertTrue(set(e8.ALLOWED_DOMAINS).isdisjoint(e8.PUBLISHED_BIO_DOMAINS))

    def test_task_id_hash_is_deterministic_and_revision_bound(self) -> None:
        first = e8.task_id_hash("task-a", "a" * 40)
        self.assertEqual(first, e8.task_id_hash("task-a", "a" * 40))
        self.assertNotEqual(first, e8.task_id_hash("task-a", "b" * 40))
        self.assertNotEqual(first, e8.task_id_hash("task-b", "a" * 40))
        with self.assertRaises(e8.LifeSciBenchSchemaError):
            e8.task_id_hash("", "a" * 40)
        with self.assertRaises(e8.LifeSciBenchSchemaError):
            e8.task_id_hash("task-a", "")

    def test_build_task_row_rejects_unbound_taxonomy(self) -> None:
        revision = "a" * 40
        row = e8.build_task_row(
            "task-a",
            dataset_revision=revision,
            family="fam-a",
            domain="science",
            workflow="translation",
            bio_domain="genomics",
        )
        self.assertEqual(row["task_id_hash"], e8.task_id_hash("task-a", revision))
        self.assertTrue(row["artifact_expected"])
        self.assertEqual(row["split"], e8.SPLIT)

        for kwargs in (
            {"domain": "not-a-campaign-domain"},
            {"domain": "science", "workflow": "not-a-workflow"},
            {"domain": "science", "bio_domain": "not-a-bio-domain"},
            {"domain": "science", "artifact_expected": False},
        ):
            with self.assertRaises(e8.LifeSciBenchSchemaError):
                e8.build_task_row(
                    "task-a", dataset_revision=revision, family="fam-a", **kwargs
                )

    def test_build_split_manifest_hashes_rows_and_rejects_duplicates(self) -> None:
        revision = "a" * 40
        specs = [
            {"task_id": "t1", "family": "f1", "domain": "science"},
            {"task_id": "t2", "family": "f2", "domain": "tool_use"},
        ]
        manifest = e8.build_split_manifest(
            specs, dataset_revision=revision, split=e8.SPLIT
        )
        self.assertEqual(manifest["task_count"], 2)
        self.assertEqual(
            manifest["manifest_hash"], e8.task_manifest_hash(manifest["rows"])
        )
        # Deterministic across calls.
        self.assertEqual(
            manifest["manifest_hash"],
            e8.build_split_manifest(
                specs, dataset_revision=revision, split=e8.SPLIT
            )["manifest_hash"],
        )
        # Order is part of the identity.
        self.assertNotEqual(
            manifest["manifest_hash"],
            e8.build_split_manifest(
                list(reversed(specs)), dataset_revision=revision, split=e8.SPLIT
            )["manifest_hash"],
        )
        with self.assertRaises(e8.LifeSciBenchSchemaError):
            e8.build_split_manifest(
                [specs[0], dict(specs[0])], dataset_revision=revision, split=e8.SPLIT
            )
        with self.assertRaises(e8.LifeSciBenchSchemaError):
            e8.build_split_manifest([], dataset_revision=revision, split=e8.SPLIT)

    def test_build_heldout_proof_fails_closed_on_task_or_family_overlap(self) -> None:
        revision = "a" * 40
        evaluation = e8.build_split_manifest(
            [{"task_id": "e1", "family": "fam-eval", "domain": "science"}],
            dataset_revision=revision,
            split=e8.SPLIT,
        )
        # Shared task_id.
        with self.assertRaises(e8.LifeSciBenchSchemaError) as ctx:
            e8.build_heldout_proof(
                train_manifest=e8.build_split_manifest(
                    [{"task_id": "e1", "family": "fam-train", "domain": "science"}],
                    dataset_revision=revision,
                    split="train",
                ),
                eval_manifest=evaluation,
            )
        self.assertIn("task_ids overlap", str(ctx.exception))

        # Distinct task_id but a shared family: a reworded near-duplicate.
        with self.assertRaises(e8.LifeSciBenchSchemaError) as ctx:
            e8.build_heldout_proof(
                train_manifest=e8.build_split_manifest(
                    [{"task_id": "t1", "family": "fam-eval", "domain": "science"}],
                    dataset_revision=revision,
                    split="train",
                ),
                eval_manifest=evaluation,
            )
        self.assertIn("families overlap", str(ctx.exception))

    def test_build_heldout_proof_is_canonical_and_disjoint(self) -> None:
        payload = _neutral_plumbing_payload()
        proof = payload["heldout_proof"]
        self.assertTrue(proof["disjoint_task_ids"])
        self.assertTrue(proof["disjoint_family_ids"])
        self.assertEqual(
            proof["proof_hash"],
            e8.sha256_json(
                {k: v for k, v in proof.items() if k != "proof_hash"}
            ),
        )
        self.assertNotEqual(
            proof["train_split_manifest_hash"], proof["eval_split_manifest_hash"]
        )
        self.assertEqual(
            sorted(proof["unseen_families"]),
            ["plumbing-family-eval-a", "plumbing-family-eval-b"],
        )

    def test_builder_chain_is_schema_complete_without_the_synthetic_marker(self) -> None:
        payload = _neutral_plumbing_payload()
        boundary_result = e8.validate_e8_boundary(payload["boundary"])
        self.assertTrue(boundary_result.ok, boundary_result.errors)
        self.assertTrue(boundary_result.metrics["task_manifest_valid"])
        self.assertTrue(boundary_result.metrics["immutable_revision_valid"])
        self.assertFalse(boundary_result.metrics["receipt_proven_heldout"])

        receipt_result = e8.validate_e8_receipt(
            payload["receipt"], boundary=payload["boundary"]
        )
        self.assertTrue(receipt_result.ok, receipt_result.errors)
        self.assertTrue(receipt_result.metrics["receipt_proven_heldout"])
        # Plumbing completeness never implies portfolio evidence or paid launch.
        self.assertFalse(receipt_result.metrics["portfolio_evidence"])
        self.assertFalse(receipt_result.metrics["paid_launch_authorized"])

    def test_receipt_emitter_seals_a_canonical_hash(self) -> None:
        receipt = _neutral_plumbing_payload()["receipt"]
        self.assertEqual(
            receipt["receipt_hash"],
            e8.sha256_json({k: v for k, v in receipt.items() if k != "receipt_hash"}),
        )
        self.assertEqual(receipt["claim_boundary"], e8.CLAIM_BOUNDARY)
        self.assertIsNone(receipt["substitute_suite_id"])
        self.assertFalse(receipt["portfolio_evidence"])

    def test_synthetic_fixture_is_rejected_as_the_only_failure(self) -> None:
        fixture = e8.build_synthetic_fixture()
        boundary_result = e8.validate_e8_boundary(fixture["boundary"])
        self.assertFalse(boundary_result.ok)
        # Exactly one error: everything else in the payload validated cleanly,
        # which is what proves the local plumbing is complete.
        self.assertEqual(boundary_result.errors, (e8.SYNTHETIC_REJECTION_ERROR,))

        receipt_result = e8.validate_e8_receipt(
            fixture["receipt"], boundary=fixture["boundary"]
        )
        self.assertFalse(receipt_result.ok)
        self.assertIn(e8.SYNTHETIC_REJECTION_ERROR, receipt_result.errors)
        self.assertFalse(receipt_result.metrics["schema_valid"])

    def test_synthetic_fixture_carries_no_score_and_is_unmistakable(self) -> None:
        fixture = e8.build_synthetic_fixture()
        self.assertIsNone(fixture["score"])
        self.assertEqual(fixture["marker"], e8.SYNTHETIC_MARKER)
        self.assertTrue(e8.contains_synthetic_marker(fixture))

        # Every task identifier and family is marked.
        for manifest_key in ("eval_manifest", "train_manifest"):
            for row in fixture[manifest_key]["rows"]:
                self.assertTrue(row["task_id"].startswith(e8.SYNTHETIC_TASK_ID_PREFIX))
                self.assertIn(e8.SYNTHETIC_MARKER, row["family"])

        # No row claims success and no metric carries a value.
        rows = fixture["receipt"]["native_verifier"]["episode_rows"]
        self.assertTrue(rows)
        self.assertTrue(all(row["task_success"] is False for row in rows))
        metrics = fixture["receipt"]["wandb"]["metrics"]
        self.assertEqual(set(metrics), set(e8.WANDB_REQUIRED_METRICS))
        self.assertTrue(all(value == 0.0 for value in metrics.values()))

    def test_synthetic_marker_cannot_be_smuggled_into_a_real_payload(self) -> None:
        boundary = _synthetic_ready_boundary()
        boundary["dataset"]["license_id"] = f"{e8.SYNTHETIC_MARKER}-license"
        result = e8.validate_e8_boundary(boundary)
        self.assertFalse(result.ok)
        self.assertIn(e8.SYNTHETIC_REJECTION_ERROR, result.errors)

    def test_offline_boundary_still_cannot_be_pinned_without_access(self) -> None:
        # The six access-gated blockers must still be reported by the shipped
        # metadata-only boundary, even now that the builders exist.
        result = e8.validate_e8_boundary(e8.build_offline_e8_boundary())
        self.assertFalse(result.ok)
        joined = " | ".join(result.errors)
        for expected in (
            "dataset.revision",
            "license",
            "native_environment.revision",
            "native_verifier.revision",
            "task_manifest",
            "train_split_manifest_hash",
        ):
            self.assertIn(expected, joined)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
