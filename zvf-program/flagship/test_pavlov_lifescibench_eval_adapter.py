"""Offline tests for the exact E8 Life-Sci-Bench evaluation boundary."""

from __future__ import annotations

import copy
import io
import json
import base64
import hashlib
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import pytest

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
    value["train_split_manifest_hash"] = e8.task_manifest_hash(
        [{"task_id": "train-fixture", "family": "train-fixture-family"}]
    )
    value["provider_package_contract"] = {
        "package_sha256": "1" * 64,
        "license_sha256": "2" * 64,
        "grant_sha256": "3" * 64,
        "container_digest": "4" * 64,
        "runtime_digest": "5" * 64,
        "verifier_revision": revision,
        "deployment_attestation_sha256": "6" * 64,
    }
    return value


_E8_TEST_PRIVATE_KEY_B64 = "vhS+4MmwVhFK1IYloOm5YT9HzA+g1h9PhlAmgnueiWk="


def _e8_test_trust_root() -> dict:
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(_E8_TEST_PRIVATE_KEY_B64))
    public_key = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return {
        "schema_version": "pavlov-provider-trust-root-v1",
        "provider_id": "openai-lifescibench",
        "lane": "e8_lifescibench",
        "suite_id": e8.SUITE_ID,
        "key_id": "e8-test-root-1",
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
        "public_key_b64": base64.b64encode(public_key).decode("ascii"),
        "signature_algorithm": "ed25519",
        "issued_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    }


def _provider_package(boundary: dict, trust_root: dict) -> dict:
    contract = boundary["provider_package_contract"]
    package = {
        "schema_version": e8.PACKAGE_INTAKE_SCHEMA_VERSION,
        "suite_id": e8.SUITE_ID,
        "package_revision": "b" * 40,
        "package_sha256": contract["package_sha256"],
        "license_id": boundary["dataset"]["license_id"],
        "license_sha256": contract["license_sha256"],
        "grant_sha256": contract["grant_sha256"],
        "boundary_sha256": e8.sha256_json(boundary),
        "container_digest": contract["container_digest"],
        "runtime_digest": contract["runtime_digest"],
        "verifier_revision": contract["verifier_revision"],
        "deployment_attestation": {
            "provider_id": "openai-lifescibench",
            "deployment_id": "provider-deployment-001",
            "issued_at": "2026-08-30T00:00:00+00:00",
            "attestation_sha256": contract["deployment_attestation_sha256"],
        },
    }
    binding = {
        "suite_id": e8.SUITE_ID,
        "lane": "e8_lifescibench",
        "package_sha256": package["package_sha256"],
        "grant_sha256": package["grant_sha256"],
        "boundary_sha256": package["boundary_sha256"],
        "container_digest": package["container_digest"],
        "runtime_digest": package["runtime_digest"],
        "verifier_revision": package["verifier_revision"],
    }
    grant = {
        "lane": "e8_lifescibench",
        "suite_id": e8.SUITE_ID,
        "key_id": trust_root["key_id"],
        "key_fingerprint": trust_root["key_fingerprint"],
        "trust_root_sha256": e8.sha256_json(trust_root),
        "signature_algorithm": "ed25519",
        "issued_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        "binding": binding,
    }
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(_E8_TEST_PRIVATE_KEY_B64))
    grant["detached_signature"] = base64.b64encode(
        key.sign(
            json.dumps(grant, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
    ).decode("ascii")
    package["provider_grant"] = grant
    return package


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
        "reward": 1.0,
        "action_count": 1,
    }
    train_rows = [{"task_id": "train-fixture", "family": "train-fixture-family"}]
    train = {"rows": train_rows, "manifest_hash": e8.task_manifest_hash(train_rows)}
    evaluation = {
        "rows": boundary["task_manifest"],
        "manifest_hash": boundary["eval_split_manifest_hash"],
    }
    boundary["train_split_manifest_hash"] = train["manifest_hash"]
    proof = e8.build_heldout_proof(train_manifest=train, eval_manifest=evaluation)
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
        assert not result.ok
        assert value["source"]["source_id"] == e8.SOURCE_ID
        assert value["source"]["url"] == e8.SOURCE_URL
        assert value["role"] == "primary_eval"
        assert value["split"] == "evaluation"
        assert not value["claims"]["receipt_proven_heldout"]
        assert not result.metrics["paid_launch_authorized"]
        assert any("immutable" in error or "license" in error for error in result.errors)

    def test_synthetic_pinned_boundary_is_schema_valid_not_result_evidence(self) -> None:
        value = _synthetic_ready_boundary()
        result = e8.validate_e8_boundary(value)
        assert result.ok, result.errors
        assert result.metrics["primary_eval"]
        assert not result.metrics["receipt_proven_heldout"]
        assert not result.metrics["portfolio_evidence"]
        assert not result.metrics["paid_launch_authorized"]

    def test_source_identity_and_role_cannot_drift(self) -> None:
        value = _synthetic_ready_boundary()
        value["source"]["url"] = "https://example.invalid/lifescibench"
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("authoritative E8 identity" in error for error in result.errors)

        value = _synthetic_ready_boundary()
        value["role"] = "train"
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("primary_eval" in error for error in result.errors)

    def test_dataset_license_identity_is_pinned(self) -> None:
        value = _synthetic_ready_boundary()
        value["dataset"]["license_id"] = e8.UNPINNED
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("license_id must be pinned" in error for error in result.errors)

        value = _synthetic_receipt()
        value["dataset"]["revision"] = "d" * 40
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=_synthetic_ready_boundary())
        assert not result.ok
        assert any("dataset.revision" in error for error in result.errors)

    def test_task_ids_and_split_hashes_are_deterministic(self) -> None:
        value = _synthetic_ready_boundary()
        task = value["task_manifest"][0]
        task["task_id_hash"] = "0" * 64
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("task hash" in error for error in result.errors)

        value = _synthetic_ready_boundary()
        value["eval_split_manifest_hash"] = "0" * 64
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("eval_split_manifest_hash" in error for error in result.errors)

    def test_native_environment_and_artifact_verifier_contract_is_required(self) -> None:
        value = _synthetic_ready_boundary()
        value["native_environment"]["artifact_or_side_effect"] = False
        value["native_verifier"]["artifact_required"] = False
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("artifact" in error for error in result.errors)

    def test_receipt_proven_heldout_is_distinct_from_protocol_primary_eval(self) -> None:
        boundary = _synthetic_ready_boundary()
        boundary_result = e8.validate_e8_boundary(boundary)
        assert boundary_result.ok, boundary_result.errors
        assert not boundary_result.metrics["receipt_proven_heldout"]

        observed = _synthetic_receipt()
        result = e8.validate_e8_receipt(observed, boundary=boundary)
        assert result.ok, result.errors
        assert result.metrics["primary_eval"]
        assert not result.metrics["receipt_proven_heldout"]
        assert result.metrics["manifest_disjointness_valid"]
        assert not result.metrics["portfolio_evidence"]
        assert not result.metrics["paid_launch_authorized"]

    def test_receipt_task_state_action_artifact_hashes_fail_closed(self) -> None:
        boundary = _synthetic_ready_boundary()
        value = _synthetic_receipt()
        value["native_verifier"]["episode_rows"][0]["state_hash"] = "0" * 64
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("state_hash" in error for error in result.errors)

        value = _synthetic_receipt()
        value["task_manifest"][0]["task_id"] = "tampered-task"
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("task_id_hash" in error or "task hash" in error for error in result.errors)

    def test_native_and_external_receipt_evidence_is_required(self) -> None:
        boundary = _synthetic_ready_boundary()
        value = _synthetic_receipt()
        value["native_verifier"]["checked"] = False
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("checked" in error for error in result.errors)

        value = _synthetic_receipt()
        value["hf"]["exported"] = False
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("hf.exported" in error for error in result.errors)

    def test_xlam_and_related_benchmarks_cannot_substitute(self) -> None:
        boundary = _synthetic_ready_boundary()
        value = _synthetic_receipt()
        value["substitute_suite_id"] = "xlam"
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("substitution" in error for error in result.errors)

        value = _synthetic_receipt()
        value["e6_substitute"] = True
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("e6_substitute" in error for error in result.errors)

        value = _synthetic_receipt()
        value["substitute_suite_id"] = "unlisted-related-benchmark"
        value["receipt_hash"] = e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        result = e8.validate_e8_receipt(value, boundary=boundary)
        assert not result.ok
        assert any("any non-null substitute_suite_id" in error for error in result.errors)

    def test_secret_like_fields_are_rejected_without_redaction(self) -> None:
        value = _synthetic_ready_boundary()
        value["dataset"]["api_key"] = "must-not-appear"
        result = e8.validate_e8_boundary(value)
        assert not result.ok
        assert any("secret-like field" in error for error in result.errors)

    def test_receipt_hash_and_proof_hash_are_deterministic(self) -> None:
        value = _synthetic_receipt()
        assert value["receipt_hash"] == e8.sha256_json(
            {key: item for key, item in value.items() if key != "receipt_hash"}
        )
        assert value["heldout_proof"]["proof_hash"] == e8.sha256_json(
            {key: item for key, item in value["heldout_proof"].items() if key != "proof_hash"}
        )

    def test_import_architecture_excludes_network_clients(self) -> None:
        import ast
        from pathlib import Path

        path = Path(e8.__file__)
        tree = ast.parse(path.read_text(), filename=str(path))
        forbidden = {
            "socket",
            "urllib",
            "http",
            "ftplib",
            "requests",
            "httpx",
            "browsergym",
            "playwright",
            "wandb",
            "tinker",
            "os",
            "subprocess",
        }
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert forbidden.isdisjoint(imported), imported & forbidden
        assert "os.environ" not in path.read_text()


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
    proof = e8.build_heldout_proof(train_manifest=train_manifest, eval_manifest=eval_manifest)
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
                "reward": 0.0,
                "action_count": 0,
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
        assert e8.PUBLISHED_TASK_COUNT == 750
        assert len(e8.PUBLISHED_WORKFLOWS) == 7
        assert len(e8.PUBLISHED_BIO_DOMAINS) == 7
        assert len(set(e8.PUBLISHED_WORKFLOWS)) == 7
        assert len(set(e8.PUBLISHED_BIO_DOMAINS)) == 7
        assert e8.PUBLISHED_RUBRIC_CRITERIA_COUNT == 19020
        assert e8.PUBLISHED_ARTIFACT_COUNT == 1062
        assert e8.PUBLISHED_PASS_THRESHOLD == "0.70"
        # The campaign capability tags are a separate axis from the bio domains.
        assert set(e8.ALLOWED_DOMAINS).isdisjoint(e8.PUBLISHED_BIO_DOMAINS)

    def test_task_id_hash_is_deterministic_and_revision_bound(self) -> None:
        first = e8.task_id_hash("task-a", "a" * 40)
        assert first == e8.task_id_hash("task-a", "a" * 40)
        assert first != e8.task_id_hash("task-a", "b" * 40)
        assert first != e8.task_id_hash("task-b", "a" * 40)
        with pytest.raises(e8.LifeSciBenchSchemaError):
            e8.task_id_hash("", "a" * 40)
        with pytest.raises(e8.LifeSciBenchSchemaError):
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
        assert row["task_id_hash"] == e8.task_id_hash("task-a", revision)
        assert row["artifact_expected"]
        assert row["split"] == e8.SPLIT

        for kwargs in (
            {"domain": "not-a-campaign-domain"},
            {"domain": "science", "workflow": "not-a-workflow"},
            {"domain": "science", "bio_domain": "not-a-bio-domain"},
            {"domain": "science", "artifact_expected": False},
        ):
            with pytest.raises(e8.LifeSciBenchSchemaError):
                e8.build_task_row("task-a", dataset_revision=revision, family="fam-a", **kwargs)

    def test_build_split_manifest_hashes_rows_and_rejects_duplicates(self) -> None:
        revision = "a" * 40
        specs = [
            {"task_id": "t1", "family": "f1", "domain": "science"},
            {"task_id": "t2", "family": "f2", "domain": "tool_use"},
        ]
        manifest = e8.build_split_manifest(specs, dataset_revision=revision, split=e8.SPLIT)
        assert manifest["task_count"] == 2
        assert manifest["manifest_hash"] == e8.task_manifest_hash(manifest["rows"])
        # Deterministic across calls.
        assert (
            manifest["manifest_hash"]
            == e8.build_split_manifest(specs, dataset_revision=revision, split=e8.SPLIT)[
                "manifest_hash"
            ]
        )
        # Order is part of the identity.
        assert (
            manifest["manifest_hash"]
            != e8.build_split_manifest(
                list(reversed(specs)), dataset_revision=revision, split=e8.SPLIT
            )["manifest_hash"]
        )
        with pytest.raises(e8.LifeSciBenchSchemaError):
            e8.build_split_manifest(
                [specs[0], dict(specs[0])], dataset_revision=revision, split=e8.SPLIT
            )
        with pytest.raises(e8.LifeSciBenchSchemaError):
            e8.build_split_manifest([], dataset_revision=revision, split=e8.SPLIT)

    def test_build_heldout_proof_fails_closed_on_task_or_family_overlap(self) -> None:
        revision = "a" * 40
        evaluation = e8.build_split_manifest(
            [{"task_id": "e1", "family": "fam-eval", "domain": "science"}],
            dataset_revision=revision,
            split=e8.SPLIT,
        )
        # Shared task_id.
        with pytest.raises(e8.LifeSciBenchSchemaError) as ctx:
            e8.build_heldout_proof(
                train_manifest=e8.build_split_manifest(
                    [{"task_id": "e1", "family": "fam-train", "domain": "science"}],
                    dataset_revision=revision,
                    split="train",
                ),
                eval_manifest=evaluation,
            )
        assert "task_ids overlap" in str(ctx.value)

        # Distinct task_id but a shared family: a reworded near-duplicate.
        with pytest.raises(e8.LifeSciBenchSchemaError) as ctx:
            e8.build_heldout_proof(
                train_manifest=e8.build_split_manifest(
                    [{"task_id": "t1", "family": "fam-eval", "domain": "science"}],
                    dataset_revision=revision,
                    split="train",
                ),
                eval_manifest=evaluation,
            )
        assert "families overlap" in str(ctx.value)

    def test_build_heldout_proof_is_canonical_and_disjoint(self) -> None:
        payload = _neutral_plumbing_payload()
        proof = payload["heldout_proof"]
        assert proof["disjoint_task_ids"]
        assert proof["disjoint_family_ids"]
        assert proof["proof_hash"] == e8.sha256_json(
            {k: v for k, v in proof.items() if k != "proof_hash"}
        )
        assert proof["train_split_manifest_hash"] != proof["eval_split_manifest_hash"]
        assert sorted(proof["unseen_families"]) == [
            "plumbing-family-eval-a",
            "plumbing-family-eval-b",
        ]

    def test_builder_chain_is_schema_complete_without_the_synthetic_marker(self) -> None:
        payload = _neutral_plumbing_payload()
        boundary_result = e8.validate_e8_boundary(payload["boundary"])
        assert boundary_result.ok, boundary_result.errors
        assert boundary_result.metrics["task_manifest_valid"]
        assert boundary_result.metrics["immutable_revision_valid"]
        assert not boundary_result.metrics["receipt_proven_heldout"]

        receipt_result = e8.validate_e8_receipt(payload["receipt"], boundary=payload["boundary"])
        assert receipt_result.ok, receipt_result.errors
        assert not receipt_result.metrics["receipt_proven_heldout"]
        assert receipt_result.metrics["manifest_disjointness_valid"]
        # Plumbing completeness never implies portfolio evidence or paid launch.
        assert not receipt_result.metrics["portfolio_evidence"]
        assert not receipt_result.metrics["paid_launch_authorized"]

    def test_receipt_emitter_seals_a_canonical_hash(self) -> None:
        receipt = _neutral_plumbing_payload()["receipt"]
        assert receipt["receipt_hash"] == e8.sha256_json(
            {k: v for k, v in receipt.items() if k != "receipt_hash"}
        )
        assert receipt["claim_boundary"] == e8.CLAIM_BOUNDARY
        assert receipt["substitute_suite_id"] is None
        assert not receipt["portfolio_evidence"]

    def test_synthetic_fixture_is_rejected_as_the_only_failure(self) -> None:
        fixture = e8.build_synthetic_fixture()
        boundary_result = e8.validate_e8_boundary(fixture["boundary"])
        assert not boundary_result.ok
        # Exactly one error: everything else in the payload validated cleanly,
        # which is what proves the local plumbing is complete.
        assert boundary_result.errors == (e8.SYNTHETIC_REJECTION_ERROR,)

        receipt_result = e8.validate_e8_receipt(fixture["receipt"], boundary=fixture["boundary"])
        assert not receipt_result.ok
        assert e8.SYNTHETIC_REJECTION_ERROR in receipt_result.errors
        assert not receipt_result.metrics["schema_valid"]

    def test_synthetic_fixture_carries_no_score_and_is_unmistakable(self) -> None:
        fixture = e8.build_synthetic_fixture()
        assert fixture["score"] is None
        assert fixture["marker"] == e8.SYNTHETIC_MARKER
        assert e8.contains_synthetic_marker(fixture)

        # Every task identifier and family is marked.
        for manifest_key in ("eval_manifest", "train_manifest"):
            for row in fixture[manifest_key]["rows"]:
                assert row["task_id"].startswith(e8.SYNTHETIC_TASK_ID_PREFIX)
                assert e8.SYNTHETIC_MARKER in row["family"]

        # No row claims success and no metric carries a value.
        rows = fixture["receipt"]["native_verifier"]["episode_rows"]
        assert rows
        assert all(row["task_success"] is False for row in rows)
        metrics = fixture["receipt"]["wandb"]["metrics"]
        assert set(metrics) == set(e8.WANDB_REQUIRED_METRICS)
        assert all(value == 0.0 for value in metrics.values())

    def test_synthetic_marker_cannot_be_smuggled_into_a_real_payload(self) -> None:
        boundary = _synthetic_ready_boundary()
        boundary["dataset"]["license_id"] = f"{e8.SYNTHETIC_MARKER}-license"
        result = e8.validate_e8_boundary(boundary)
        assert not result.ok
        assert e8.SYNTHETIC_REJECTION_ERROR in result.errors

    def test_offline_boundary_still_cannot_be_pinned_without_access(self) -> None:
        # The six access-gated blockers must still be reported by the shipped
        # metadata-only boundary, even now that the builders exist.
        result = e8.validate_e8_boundary(e8.build_offline_e8_boundary())
        assert not result.ok
        joined = " | ".join(result.errors)
        for expected in (
            "dataset.revision",
            "license",
            "native_environment.revision",
            "native_verifier.revision",
            "task_manifest",
            "train_split_manifest_hash",
        ):
            assert expected in joined

    def test_provider_package_intake_and_hosted_request_are_scoreless_and_pinned(self) -> None:
        boundary = _synthetic_ready_boundary()
        trust_root = _e8_test_trust_root()
        package = _provider_package(boundary, trust_root)
        missing_root = e8.validate_provider_package_intake(package, boundary=boundary)
        assert not missing_root.ok
        result = e8.validate_provider_package_intake(
            package, boundary=boundary, trust_root=trust_root
        )
        assert result.ok, result.errors
        request = e8.build_hosted_execution_request(
            package, boundary=boundary, trust_root=trust_root
        )
        assert request["mode"] == "hosted_submit"
        assert request["score"] is None
        assert not request["execution_performed"]
        assert not request["paid_calls_executed"]

        package["grant_sha256"] = "f" * 64
        rejected = e8.validate_provider_package_intake(
            package, boundary=boundary, trust_root=trust_root
        )
        assert not rejected.ok
        assert any("grant_sha256" in error for error in rejected.errors)

    def test_package_intake_cli_never_submits_or_fabricates_score(self) -> None:
        boundary = _synthetic_ready_boundary()
        trust_root = _e8_test_trust_root()
        package = _provider_package(boundary, trust_root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path = root / "boundary.json"
            package_path = root / "package.json"
            trust_root_path = root / "trust-root.json"
            boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
            package_path.write_text(json.dumps(package), encoding="utf-8")
            trust_root_path.write_text(json.dumps(trust_root), encoding="utf-8")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = e8.main(
                    [
                        "--boundary",
                        str(boundary_path),
                        "--intake-package",
                        str(package_path),
                        "--trust-roots",
                        str(trust_root_path),
                        "--hosted-submit",
                    ]
                )
            assert code == 0
            output = json.loads(stdout.getvalue())
            assert output["intake"]["ok"]
            assert output["score"] is None
            assert not output["execution_request"]["execution_performed"]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
