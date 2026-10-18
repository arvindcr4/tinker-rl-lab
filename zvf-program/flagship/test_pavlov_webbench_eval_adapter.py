from __future__ import annotations

import json
import base64
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from flagship.pavlov_webbench_eval_adapter import (
    WEBBENCH_CATEGORIES,
    WEBBENCH_DATASET_SHA256,
    WEBBENCH_DOMAINS,
    WEBBENCH_PUBLIC_TASK_COUNT,
    WEBBENCH_RECEIPT_FIELDS,
    WEBBENCH_ROLE,
    WEBBENCH_SOURCE_URL,
    WEBBENCH_SUITE_ID,
    WebBenchDatasetError,
    aggregate_task_hashes,
    build_split_artifacts,
    build_split_manifest,
    characterize_task_set,
    derive_task_records,
    format_task_id,
    load_training_task_manifest,
    prove_split_disjointness,
    read_webbench_csv,
    receipt_digest,
    registrable_host,
    sha256_hex,
    split_manifest_hash,
    task_digest,
    task_ids_hash,
    validate_webbench_manifest as _validate_webbench_manifest,
    write_split_artifacts,
)
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_DATASET = REPO_ROOT / "outputs" / "e6_webbench" / "webbenchfinal.csv"
REAL_TRAINING_MANIFEST = (
    REPO_ROOT / "outputs" / "e6_webbench" / "eval_only_training_task_manifest.json"
)

# Recomputed by webbench_eval.py's independent hashing scheme over the same
# pinned CSV.  If either implementation drifts, these stop matching.
RUNNER_TASK_ID_HASH = "22afbdd3cc47e6dba1e3c57ddbe5f762b54be5d2af6ac76bbd206c19eb83b12e"
RUNNER_MANIFEST_HASH = "66da44a04ec48fe356b3b0d1c420c40679faa1a7ac650728e254b625bb674a07"

CSV_HEADER = "ID,Starting URL,Category,Task\n"
_E6_TEST_PRIVATE_KEY_B64 = "xPRVwba6NQ4VhWLwCBr6el1kJhCHohse0elI09LXf5w="


def _e6_test_trust_root() -> dict[str, str]:
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(_E6_TEST_PRIVATE_KEY_B64))
    public_key = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return {
        "schema_version": "pavlov-provider-trust-root-v1",
        "provider_id": "halluminate-webbench",
        "lane": "e6_webbench",
        "suite_id": WEBBENCH_SUITE_ID,
        "key_id": "e6-test-root-1",
        "key_fingerprint": hashlib.sha256(public_key).hexdigest(),
        "public_key_b64": base64.b64encode(public_key).decode("ascii"),
        "signature_algorithm": "ed25519",
        "issued_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    }


_E6_TEST_TRUST_ROOT = _e6_test_trust_root()


def validate_webbench_manifest(manifest: object) -> dict[str, object]:
    """Test-only wrapper: production callers must supply --trust-roots/root input."""
    from unittest.mock import patch
    from flagship import pavlov_webbench_eval_adapter as module

    with (
        patch.object(
            module,
            "TRUSTED_PROVIDER_FINGERPRINTS",
            frozenset({_E6_TEST_TRUST_ROOT["key_fingerprint"]}),
        ),
        patch.object(
            module,
            "TRUSTED_RECEIPT_KEYS",
            {_E6_TEST_TRUST_ROOT["key_id"]: _E6_TEST_TRUST_ROOT["public_key_b64"]},
        ),
    ):
        return _validate_webbench_manifest(manifest, trust_root=_E6_TEST_TRUST_ROOT)


def _sign_receipt(record):
    key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(_E6_TEST_PRIVATE_KEY_B64))
    payload = {k: v for k, v in record.items() if k != "signature_b64"}
    record["signature_b64"] = base64.b64encode(
        key.sign(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        )
    ).decode()
    return record


def _csv(rows: list[tuple[int, str, str, str]], header: str = CSV_HEADER) -> str:
    lines = [header.rstrip("\n")]
    for csv_id, url, category, task in rows:
        escaped = task.replace('"', '""')
        lines.append(f'{csv_id},{url},{category},"{escaped}"')
    return "\n".join(lines) + "\n"


FIXTURE_ROWS = [
    (7, "https://www.example-shop.test", "READ", "List the price.\nOnly use the site."),
    (3, "https://docs.example-wiki.test", "CREATE", "Create a page.\nOnly use the site."),
    (1200, "http://example-jobs.test", "FILE_MANIPULATION", "Upload a resume.\nOnly use the site."),
]


class PavlovWebBenchBoundaryTests(unittest.TestCase):
    def _receipt(
        self, binding: dict[str, str], number: int, payload_extra: dict[str, object] | None = None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "identity": f"{number:040x}"[-40:],
            "artifact_digest": f"{number:064x}"[-64:],
        }
        if payload_extra:
            payload.update(payload_extra)
        record = {
            "schema_version": "pavlov-webbench-authenticated-receipt-v2",
            "key_id": _E6_TEST_TRUST_ROOT["key_id"],
            "receipt_id": f"{number + 1000:040x}"[-40:],
            "digest": receipt_digest(binding, payload),
            "authenticated": True,
            "cryptographically_bound": True,
            "binding": binding,
            "payload": payload,
        }
        return _sign_receipt(record)

    def _manifest(self) -> dict[str, object]:
        task_ids = ["webbench-task-0001", "webbench-task-0002", "webbench-task-0003"]
        split = {
            "suite_id": WEBBENCH_SUITE_ID,
            "role": WEBBENCH_ROLE,
            "split": "evaluation",
            "task_id_hash": task_ids_hash(task_ids),
            "manifest_version": "webbench-eval-v1",
        }
        environment = {
            "container_digest": "1" * 64,
            "runtime_digest": "2" * 64,
            "native_environment": {
                "entrypoint": "webbench_native_environment",
                "state_model": "browser_state_and_artifacts",
            },
            "artifact_contract": {
                "required_artifacts": ["browser_trace", "task_artifact"],
                "state_integrity_checks": ["native_state_hash", "artifact_manifest"],
                "side_effect_policy": "record_and_verify_declared_side_effects",
            },
            "verifier_contract": {
                "verifier_id": "webbench-native-verifier",
                "verifier_revision": "3" * 40,
                "checks": ["native_state", "artifact_integrity", "task_success"],
                "native_state_inspection": True,
            },
        }
        manifest: dict[str, object] = {
            "schema_version": "webbench-boundary-fixture-v1",
            "suite_id": WEBBENCH_SUITE_ID,
            "role": WEBBENCH_ROLE,
            "domains": list(WEBBENCH_DOMAINS),
            "heldout_status": "pending_receipts",
            "source": {
                "suite_id": WEBBENCH_SUITE_ID,
                "name": "Halluminate/WebBench",
                "url": WEBBENCH_SOURCE_URL,
                "revision": "a" * 40,
                "license": "Apache-2.0",
                "revision_receipt": self._receipt(
                    {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "revision"},
                    1,
                    {"revision": "a" * 40},
                ),
                "license_receipt": self._receipt(
                    {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "license"},
                    2,
                    {"license": "Apache-2.0"},
                ),
            },
            "task_ids": task_ids,
            "task_id_hash": task_ids_hash(task_ids),
            "split_manifest": split,
            "split_manifest_hash": split_manifest_hash(split),
            "task_receipt": self._receipt(
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "task"},
                3,
                {"task_id_hash": task_ids_hash(task_ids)},
            ),
            "split_receipt": self._receipt(
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "split"},
                4,
                {"split_manifest_hash": split_manifest_hash(split)},
            ),
            "environment": environment,
        }
        environment["container_receipt"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "container"},
            5,
            {
                "container_digest": environment["container_digest"],
                "runtime_digest": environment["runtime_digest"],
            },
        )
        environment["verifier_receipt"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "verifier"},
            6,
            {"verifier_revision": environment["verifier_contract"]["verifier_revision"]},
        )
        write_scope = {
            "operations": ["CREATE", "DELETE", "FILE_MANIPULATION", "READ", "UPDATE"],
            "allowed_domains": ["example-webbench-provider.test"],
        }
        provider_grant = {
            "receipt_id": "9" * 40,
            "historical": False,
            "unlaunchable": False,
            "issued_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "provider_identity": {
                "provider_id": "halluminate-webbench",
                "key_id": "8" * 40,
                "url": "https://github.com/Halluminate/WebBench",
            },
            "binding": {
                "suite_id": WEBBENCH_SUITE_ID,
                "lane": "e6_webbench",
                "revision": manifest["source"]["revision"],
                "container_digest": environment["container_digest"],
                "runtime_digest": environment["runtime_digest"],
                "verifier_revision": environment["verifier_contract"]["verifier_revision"],
                "reset_contract_digest": "7" * 64,
                "write_scope_digest": sha256_hex(write_scope),
            },
            "write_scope": write_scope,
            "signature_algorithm": "ed25519",
            "key_id": _E6_TEST_TRUST_ROOT["key_id"],
            "key_fingerprint": _E6_TEST_TRUST_ROOT["key_fingerprint"],
            "trust_root_sha256": sha256_hex(_E6_TEST_TRUST_ROOT),
        }
        signing_key = Ed25519PrivateKey.from_private_bytes(
            base64.b64decode(_E6_TEST_PRIVATE_KEY_B64)
        )
        signed = {
            key: value for key, value in provider_grant.items() if key != "detached_signature"
        }
        provider_grant["detached_signature"] = base64.b64encode(
            signing_key.sign(
                json.dumps(
                    signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            )
        ).decode("ascii")
        manifest["provider_access_receipt"] = provider_grant
        return manifest

    def _with_heldout_receipts(self, manifest: dict[str, object]) -> dict[str, object]:
        result = {"n": 100, "metric": 0.55, "result_id": "b" * 40}
        result_digest = sha256_hex(result)
        receipts: dict[str, object] = {}
        provenance_payloads = {
            "revision": {"revision": manifest["source"]["revision"]},
            "license": {"license": manifest["source"]["license"]},
            "split": {"split_manifest_hash": manifest["split_manifest_hash"]},
            "task": {"task_id_hash": manifest["task_id_hash"]},
            "container": {"container_digest": manifest["environment"]["container_digest"]},
            "decontamination": {"decontamination_hash": "e" * 64},
        }
        for offset, field in enumerate(WEBBENCH_RECEIPT_FIELDS, start=10):
            receipts[field] = self._receipt(
                {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": field},
                offset,
                provenance_payloads[field],
            )
        result_receipts: dict[str, object] = {}
        common = {"artifact_digest": result_digest, "result_digest": result_digest}
        result_receipts["wandb"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "result:wandb"},
            20,
            {**common, "run_id": "c" * 40, "run_url": "https://wandb.ai/example/webbench/runs/abc"},
        )
        result_receipts["tinker"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "result:tinker"},
            21,
            {**common, "provider": "Tinker", "run_id": "d" * 40, "cumulative_cost_usd": 1.25},
        )
        result_receipts["hf"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "result:hf"},
            22,
            {
                **common,
                "revision": f"{22:040x}"[-40:],
                "visibility": "private",
                "repo_url": "https://huggingface.co/example/webbench",
                "checkpoint_url": "https://huggingface.co/example/webbench/commit/"
                + f"{22:040x}"[-40:],
            },
        )
        # The repeated URL expression above is intentionally corrected to a
        # valid single URL before hashing; it also makes accidental URL edits in
        # a test fixture visible rather than silently accepted.
        hf_payload = result_receipts["hf"]["payload"]
        hf_payload["checkpoint_url"] = (
            "https://huggingface.co/example/webbench/commit/" + f"{22:040x}"[-40:]
        )
        result_receipts["hf"]["digest"] = receipt_digest(
            result_receipts["hf"]["binding"], hf_payload
        )
        manifest["receipts"] = receipts
        manifest["result"] = result
        manifest["result_receipts"] = result_receipts
        manifest["heldout_status"] = "receipt_proven_heldout"
        return manifest

    def test_primary_eval_boundary_is_valid_but_not_heldout_without_receipts(self) -> None:
        report = validate_webbench_manifest(self._manifest())

        assert report["boundary_valid"]
        assert report["primary_eval"]
        assert not report["receipt_proven_heldout"]
        assert report["status"] == "READY_PRIMARY_EVAL_PENDING_RECEIPTS"

    def test_live_grant_without_external_trust_root_fails_closed(self) -> None:
        report = _validate_webbench_manifest(self._manifest())
        assert not report["boundary_valid"]
        assert "provider_access_receipt_invalid" in report["blocker_codes"]

    def test_complete_receipts_prove_heldout_results(self) -> None:
        report = validate_webbench_manifest(self._with_heldout_receipts(self._manifest()))

        assert report["boundary_valid"]
        assert report["receipt_proven_heldout"]
        assert report["heldout_claim_allowed"]
        assert report["status"] == "READY_RECEIPT_PROVEN_HELDOUT"

    def test_heldout_role_is_not_a_substitute_for_primary_eval(self) -> None:
        manifest = self._manifest()
        manifest["role"] = "heldout"

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "role_invalid" in report["blocker_codes"]

    def test_self_attested_heldout_status_without_receipts_blocks(self) -> None:
        manifest = self._manifest()
        manifest["heldout_status"] = "receipt_proven_heldout"

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "heldout_status_without_receipts" in report["blocker_codes"]

    def test_provider_grant_binds_the_immutable_runtime_and_write_scope(self) -> None:
        manifest = self._manifest()
        manifest["provider_access_receipt"]["binding"]["runtime_digest"] = "f" * 64
        report = validate_webbench_manifest(manifest)
        assert not report["boundary_valid"]
        assert "provider_access_receipt_invalid" in report["blocker_codes"]

        manifest = self._manifest()
        manifest["provider_access_receipt"]["write_scope"]["operations"] = []
        report = validate_webbench_manifest(manifest)
        assert not report["boundary_valid"]
        assert "provider_access_receipt_invalid" in report["blocker_codes"]

    def test_only_explicit_historical_unlaunchable_receipts_are_compatible(self) -> None:
        manifest = self._manifest()
        manifest["provider_access_receipt"] = {
            "historical": True,
            "unlaunchable": True,
            "receipt_id": "old-provider-ticket",
            "issued_at": "2025-01-01T00:00:00+00:00",
            "provider_identity": "Halluminate archive",
            "note": "Historical access request; not launch authority.",
        }
        report = validate_webbench_manifest(manifest)
        assert report["boundary_valid"], report["blockers"]

        manifest["provider_access_receipt"]["unlaunchable"] = False
        report = validate_webbench_manifest(manifest)
        assert not report["boundary_valid"]
        assert "provider_access_receipt_invalid" in report["blocker_codes"]

    def test_authoritative_source_identity_rejects_xlam_and_related_benchmarks(self) -> None:
        manifest = self._manifest()
        manifest["source"]["url"] = "https://huggingface.co/example/xlam"
        manifest["related_benchmarks"] = ["xLAM"]

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "source_identity_mismatch" in report["blocker_codes"]
        assert "related_benchmark_substitution" in report["blocker_codes"]

    def test_fake_revision_license_and_receipt_strings_block(self) -> None:
        manifest = self._manifest()
        manifest["source"]["revision"] = "main"
        manifest["source"]["license"] = "pending"
        manifest["source"]["revision_receipt"] = "verified"

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "revision_not_pinned" in report["blocker_codes"]
        assert "license_not_pinned" in report["blocker_codes"]
        assert "source_receipt_invalid" in report["blocker_codes"]

    def test_task_and_split_hashes_must_be_deterministic_and_bound(self) -> None:
        manifest = self._manifest()
        manifest["task_ids"] = list(reversed(manifest["task_ids"]))
        manifest["task_id_hash"] = "f" * 64

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "task_ids_not_deterministic" in report["blocker_codes"]
        assert "task_id_hash_mismatch" in report["blocker_codes"]

    def test_task_receipt_cannot_attest_to_a_different_task_hash(self) -> None:
        manifest = self._manifest()
        receipt = manifest["task_receipt"]
        receipt["payload"]["task_id_hash"] = "f" * 64
        receipt["digest"] = receipt_digest(receipt["binding"], receipt["payload"])
        _sign_receipt(receipt)

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "task_receipt_payload_mismatch" in report["blocker_codes"]

    def test_native_artifact_and_verifier_contracts_are_required(self) -> None:
        manifest = self._manifest()
        del manifest["environment"]["artifact_contract"]
        manifest["environment"]["verifier_contract"] = "passed"

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "artifact_contract_invalid" in report["blocker_codes"]
        assert "verifier_contract_invalid" in report["blocker_codes"]

    def test_result_material_without_all_wandb_tinker_hf_receipts_blocks(self) -> None:
        manifest = self._manifest()
        manifest["result"] = {"n": 100, "metric": 0.5, "result_id": "b" * 40}

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "result_receipts_missing" in report["blocker_codes"]

    def test_fake_run_id_url_and_hf_visibility_cannot_clear_result(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["result_receipts"]["wandb"]["payload"]["run_id"] = "fake-run"
        manifest["result_receipts"]["wandb"]["digest"] = receipt_digest(
            manifest["result_receipts"]["wandb"]["binding"],
            manifest["result_receipts"]["wandb"]["payload"],
        )
        manifest["result_receipts"]["hf"]["payload"]["visibility"] = "verified"
        manifest["result_receipts"]["hf"]["digest"] = receipt_digest(
            manifest["result_receipts"]["hf"]["binding"],
            manifest["result_receipts"]["hf"]["payload"],
        )

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "result_receipt_payload_invalid" in report["blocker_codes"]

    def test_tampered_result_digest_blocks_even_with_receipt_shapes(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["result_receipts"]["tinker"]["payload"]["result_digest"] = "f" * 64
        manifest["result_receipts"]["tinker"]["digest"] = receipt_digest(
            manifest["result_receipts"]["tinker"]["binding"],
            manifest["result_receipts"]["tinker"]["payload"],
        )

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "result_receipt_payload_invalid" in report["blocker_codes"]

    def test_authoritative_service_urls_reject_credentials_and_query_only_revision(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        wandb = manifest["result_receipts"]["wandb"]
        wandb["payload"]["run_url"] = "https://user:secret@sub.wandb.ai/example/webbench/runs/abc"
        wandb["digest"] = receipt_digest(wandb["binding"], wandb["payload"])
        hf = manifest["result_receipts"]["hf"]
        revision = hf["payload"]["revision"]
        hf["payload"]["checkpoint_url"] = (
            "https://huggingface.co/example/webbench/commit/latest?revision=" + revision
        )
        hf["digest"] = receipt_digest(hf["binding"], hf["payload"])

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "result_receipt_payload_invalid" in report["blocker_codes"]

    def test_public_hf_artifact_requires_explicit_clean_safety_flags(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        hf = manifest["result_receipts"]["hf"]
        hf["payload"]["visibility"] = "public"
        hf["digest"] = receipt_digest(hf["binding"], hf["payload"])

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "result_receipt_payload_invalid" in report["blocker_codes"]

    def test_noncanonical_split_and_typed_boundary_fields_fail_closed(self) -> None:
        manifest = self._manifest()
        manifest["split_manifest"]["noncanonical"] = {"not-json"}
        manifest["domains"] = tuple(WEBBENCH_DOMAINS)
        manifest["heldout_status"] = []

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "split_manifest_not_canonicalizable" in report["blocker_codes"]
        assert "domain_boundary_invalid" in report["blocker_codes"]
        assert "heldout_status_invalid" in report["blocker_codes"]

    def test_pending_status_with_complete_receipts_is_not_silently_promoted(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["heldout_status"] = "pending_receipts"

        report = validate_webbench_manifest(manifest)

        assert not report["boundary_valid"]
        assert "heldout_status_stale" in report["blocker_codes"]

    def test_cli_is_offline_and_returns_json(self) -> None:
        script = Path(__file__).with_name("pavlov_webbench_eval_adapter.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            trust_root_path = root / "trust-root.json"
            manifest_path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            trust_root_path.write_text(json.dumps(_E6_TEST_TRUST_ROOT), encoding="utf-8")
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manifest",
                    str(manifest_path),
                    "--trust-roots",
                    str(trust_root_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        report = json.loads(process.stdout)
        assert process.returncode == 1
        assert report["status"] == "BLOCKED"
        assert not report["receipt_proven_heldout"]
        assert "Traceback" not in process.stderr


class WebBenchSplitDerivationTests(unittest.TestCase):
    """Offline derivation of task identities, hashes, and the split manifest."""

    def _write_fixture(
        self, directory: Path, rows: list[tuple[int, str, str, str]] | None = None
    ) -> Path:
        path = directory / "webbenchfinal.csv"
        path.write_text(_csv(rows if rows is not None else FIXTURE_ROWS), encoding="utf-8")
        return path

    def _records(
        self, directory: Path, rows: list[tuple[int, str, str, str]] | None = None
    ) -> list[dict]:
        return derive_task_records(
            read_webbench_csv(self._write_fixture(directory, rows), expected_sha256=None)
        )

    def test_task_ids_are_zero_padded_so_lexicographic_equals_numeric_order(self) -> None:
        ids = [format_task_id(value) for value in (0, 9, 10, 999, 1000, 2724)]

        assert ids[0] == "webbench-task-0000"
        assert ids[-1] == "webbench-task-2724"
        assert sorted(ids) == ids
        with pytest.raises(WebBenchDatasetError):
            format_task_id(-1)
        with pytest.raises(WebBenchDatasetError):
            format_task_id(True)

    def test_registrable_host_only_strips_leading_www(self) -> None:
        assert registrable_host("www.indeed.com") == "indeed.com"
        assert registrable_host("open.spotify.com") == "open.spotify.com"
        assert registrable_host("WWW.Example.COM") == "example.com"

    def test_task_digest_is_content_addressed_and_row_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            forward = self._records(Path(tmp))
            reversed_rows = list(reversed(FIXTURE_ROWS))
            reverse = self._records(Path(tmp), reversed_rows)

        assert [item["task_id"] for item in forward] == [item["task_id"] for item in reverse]
        assert [item["task_digest"] for item in forward] == [
            item["task_digest"] for item in reverse
        ]
        assert [item["task_uid"] for item in forward] == [item["task_uid"] for item in reverse]
        # The digest binds content, not the derived label.
        mutated = dict(forward[0])
        mutated["task"] = forward[0]["task"] + " extra"
        assert task_digest(mutated) != forward[0]["task_digest"]

    def test_derived_records_are_sorted_unique_and_carry_stable_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))

        task_ids = [item["task_id"] for item in records]
        assert task_ids == sorted(task_ids)
        assert len(task_ids) == len(set(task_ids))
        assert [item["csv_id"] for item in records] == [3, 7, 1200]
        assert records[0]["registrable_domain"] == "docs.example-wiki.test"
        assert records[2]["scheme"] == "http"
        assert all(item["task_uid"].startswith("webbench-uid-") for item in records)

    def test_csv_reader_fails_closed_on_malformed_input(self) -> None:
        cases = {
            "bad_columns": ("id,url,category,task\n1,https://a.test,READ,x\n", "columns"),
            "bad_category": (_csv([(1, "https://a.test", "BROWSE", "do it")]), "category"),
            "empty_task": (_csv([(1, "https://a.test", "READ", "")]), "empty task"),
            "bad_url": (_csv([(1, "ftp://a.test", "READ", "do it")]), "Starting URL"),
            "noninteger_id": (
                'ID,Starting URL,Category,Task\nx,https://a.test,READ,"do it"\n',
                "non-integer",
            ),
            "duplicate_id": (
                _csv([(1, "https://a.test", "READ", "one"), (1, "https://b.test", "READ", "two")]),
                "duplicate",
            ),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, (body, fragment) in cases.items():
                path = Path(tmp) / f"{name}.csv"
                path.write_text(body, encoding="utf-8")
                with self.subTest(case=name):
                    with pytest.raises(WebBenchDatasetError) as ctx:
                        read_webbench_csv(path, expected_sha256=None)
                    assert fragment in str(ctx.value)

    def test_csv_reader_enforces_the_dataset_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_fixture(Path(tmp))
            with pytest.raises(WebBenchDatasetError) as ctx:
                read_webbench_csv(path, expected_sha256="0" * 64)
            assert "SHA-256 mismatch" in str(ctx.value)
            missing = Path(tmp) / "absent.csv"
            with pytest.raises(WebBenchDatasetError):
                read_webbench_csv(missing, expected_sha256=None)

    def test_aggregate_hashes_are_deterministic_and_order_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
            shuffled = list(reversed(records))

        first = aggregate_task_hashes(records)
        second = aggregate_task_hashes(shuffled)
        assert first == second
        assert first["task_count"] == 3
        assert first["task_id_hash"] == task_ids_hash([item["task_id"] for item in records])
        assert first["task_digest_hash"] == sha256_hex([item["task_digest"] for item in records])
        # A single content change must move the content hashes but not the ID hash.
        mutated = [dict(item) for item in records]
        mutated[0]["task"] = mutated[0]["task"] + "!"
        mutated[0]["task_digest"] = task_digest(mutated[0])
        third = aggregate_task_hashes(mutated)
        assert third["task_id_hash"] == first["task_id_hash"]
        assert third["task_digest_hash"] != first["task_digest_hash"]
        assert third["legacy_runner_manifest_hash"] != first["legacy_runner_manifest_hash"]

    def test_built_split_manifest_satisfies_the_boundary_task_and_split_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
        task_ids = [item["task_id"] for item in records]
        split = build_split_manifest(task_ids)

        assert split["suite_id"] == WEBBENCH_SUITE_ID
        assert split["role"] == WEBBENCH_ROLE
        assert split["split"] == "evaluation"
        assert split["task_id_hash"] == task_ids_hash(task_ids)

        manifest = PavlovWebBenchBoundaryTests()._manifest()
        manifest["task_ids"] = task_ids
        manifest["task_id_hash"] = task_ids_hash(task_ids)
        manifest["split_manifest"] = split
        manifest["split_manifest_hash"] = split_manifest_hash(split)
        manifest["task_receipt"]["payload"]["task_id_hash"] = task_ids_hash(task_ids)
        manifest["task_receipt"]["digest"] = receipt_digest(
            manifest["task_receipt"]["binding"], manifest["task_receipt"]["payload"]
        )
        manifest["split_receipt"]["payload"]["split_manifest_hash"] = split_manifest_hash(split)
        manifest["split_receipt"]["digest"] = receipt_digest(
            manifest["split_receipt"]["binding"], manifest["split_receipt"]["payload"]
        )
        _sign_receipt(manifest["task_receipt"])
        _sign_receipt(manifest["split_receipt"])

        report = validate_webbench_manifest(manifest)
        assert report["checks"]["tasks_and_split"], report["blocker_codes"]
        assert "task_id_hash_mismatch" not in report["blocker_codes"]
        assert "split_manifest_hash_mismatch" not in report["blocker_codes"]

        with pytest.raises(WebBenchDatasetError):
            build_split_manifest(task_ids + task_ids[:1])

    def test_training_manifest_accepts_int_string_and_object_forms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            as_list = root / "list.json"
            as_list.write_text(json.dumps([7, 3]), encoding="utf-8")
            assert load_training_task_manifest(as_list)["task_ids"] == [
                "webbench-task-0003",
                "webbench-task-0007",
            ]

            as_strings = root / "strings.json"
            as_strings.write_text(
                json.dumps({"task_ids": ["webbench-task-0007"]}), encoding="utf-8"
            )
            loaded = load_training_task_manifest(as_strings)
            assert loaded["task_ids"] == ["webbench-task-0007"]
            assert loaded["legacy_runner_task_id_hash"] is None

            duplicated = root / "dupes.json"
            duplicated.write_text(json.dumps([5, 5]), encoding="utf-8")
            with pytest.raises(WebBenchDatasetError):
                load_training_task_manifest(duplicated)

            malformed = root / "malformed.json"
            malformed.write_text("{not json", encoding="utf-8")
            with pytest.raises(WebBenchDatasetError):
                load_training_task_manifest(malformed)

            with pytest.raises(WebBenchDatasetError):
                load_training_task_manifest(root / "absent.json")

    def test_disjointness_proof_detects_overlap_and_bad_declared_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_ids = ["webbench-task-0003", "webbench-task-0007"]

            empty = root / "empty.json"
            # sha256 of the empty string: the runner's hash of an empty ID list.
            empty.write_text(
                json.dumps(
                    {
                        "task_ids": [],
                        "task_id_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    }
                ),
                encoding="utf-8",
            )
            clean = prove_split_disjointness(eval_ids, load_training_task_manifest(empty))
            assert clean["disjoint"]
            assert clean["valid"]
            assert clean["training_declared_task_id_hash_verified"]
            assert clean["overlap_count"] == 0

            overlapping = root / "overlap.json"
            overlapping.write_text(json.dumps([3]), encoding="utf-8")
            dirty = prove_split_disjointness(eval_ids, load_training_task_manifest(overlapping))
            assert not dirty["disjoint"]
            assert not dirty["valid"]
            assert dirty["overlap_task_ids"] == ["webbench-task-0003"]

            wrong_hash = root / "wrong.json"
            wrong_hash.write_text(
                json.dumps({"task_ids": [], "task_id_hash": "f" * 64}), encoding="utf-8"
            )
            bad = prove_split_disjointness(eval_ids, load_training_task_manifest(wrong_hash))
            assert not bad["valid"]
            assert not bad["training_declared_task_id_hash_verified"]

    def test_characterization_reports_id_gaps_and_domain_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
        profile = characterize_task_set(records)

        assert profile["task_count"] == 3
        assert profile["csv_id_min"] == 3
        assert profile["csv_id_max"] == 1200
        assert not profile["csv_ids_contiguous"]
        assert profile["csv_id_gap_count"] == 1195
        assert profile["distinct_registrable_domains"] == 3
        assert profile["url_schemes"] == {"http": 1, "https": 2}
        assert profile["http_only_urls"] == ["http://example-jobs.test"]
        assert set(profile["categories"]) - set(WEBBENCH_CATEGORIES) == set()
        # CREATE + FILE_MANIPULATION mutate live sites; READ does not.
        assert profile["write_class_task_count"] == 2
        assert profile["keyword_probe_counts"]["captcha"] == 0

    def test_bundle_is_labelled_a_local_derivation_not_a_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._write_fixture(root)
            training = root / "training.json"
            training.write_text(json.dumps({"task_ids": []}), encoding="utf-8")
            bundle = build_split_artifacts(
                dataset, training_manifest_path=training, expected_dataset_sha256=None
            )
            written = write_split_artifacts(bundle, root / "out")

            derivation = bundle["derivation"]
            assert derivation["receipt_class"] == "local_derivation"
            assert derivation["authenticated_receipt"] is None
            assert not derivation["externally_bound"]
            assert not derivation["network_accessed"]
            assert derivation["disjointness"]["valid"]
            assert len(Path(written["task_index"]).read_text(encoding="utf-8").splitlines()) == 3
            split_payload = json.loads(Path(written["split_manifest"]).read_text(encoding="utf-8"))
            assert split_payload["task_count"] == 3
            assert split_payload["authenticated_receipt"] is None

            # Re-running is byte-identical: the derivation is deterministic.
            again = build_split_artifacts(
                dataset, training_manifest_path=training, expected_dataset_sha256=None
            )
            assert again["derivation"]["derivation_hash"] == derivation["derivation_hash"]

    def test_derived_split_alone_never_unblocks_the_environment_boundary(self) -> None:
        """A local split manifest must not make WebBench look runnable."""

        with tempfile.TemporaryDirectory() as tmp:
            records = self._records(Path(tmp))
        task_ids = [item["task_id"] for item in records]
        split = build_split_manifest(task_ids)
        manifest = {
            "suite_id": WEBBENCH_SUITE_ID,
            "role": WEBBENCH_ROLE,
            "domains": list(WEBBENCH_DOMAINS),
            "heldout_status": "pending_receipts",
            "source": {
                "suite_id": WEBBENCH_SUITE_ID,
                "name": "Halluminate/WebBench",
                "url": WEBBENCH_SOURCE_URL,
                "revision": "a" * 40,
                "license": "MIT",
            },
            "task_ids": task_ids,
            "task_id_hash": task_ids_hash(task_ids),
            "split_manifest": split,
            "split_manifest_hash": split_manifest_hash(split),
        }

        report = validate_webbench_manifest(manifest)
        assert report["status"] == "BLOCKED"
        assert "environment_contract_missing" in report["blocker_codes"]
        assert "task_receipt_invalid" in report["blocker_codes"]
        assert "split_receipt_invalid" in report["blocker_codes"]
        assert not report["receipt_proven_heldout"]


@unittest.skipUnless(REAL_DATASET.is_file(), "pinned WebBench CSV is not on disk")
class WebBenchPinnedDatasetTests(unittest.TestCase):
    """Assertions against the real, pinned, MIT-licensed public CSV."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.records = derive_task_records(
            read_webbench_csv(REAL_DATASET, expected_sha256=WEBBENCH_DATASET_SHA256)
        )
        cls.aggregates = aggregate_task_hashes(cls.records)

    def test_public_task_count_and_identity_scheme(self) -> None:
        assert len(self.records) == WEBBENCH_PUBLIC_TASK_COUNT
        task_ids = [item["task_id"] for item in self.records]
        assert task_ids == sorted(task_ids)
        assert len(set(task_ids)) == WEBBENCH_PUBLIC_TASK_COUNT
        assert len({item["task_uid"] for item in self.records}) == WEBBENCH_PUBLIC_TASK_COUNT
        assert len({item["task_digest"] for item in self.records}) == WEBBENCH_PUBLIC_TASK_COUNT

    def test_aggregates_agree_with_the_independent_runner_implementation(self) -> None:
        assert self.aggregates["legacy_runner_task_id_hash"] == RUNNER_TASK_ID_HASH
        assert self.aggregates["legacy_runner_manifest_hash"] == RUNNER_MANIFEST_HASH

    def test_public_ids_are_sparse_which_is_why_ids_must_be_sorted_explicitly(self) -> None:
        profile = characterize_task_set(self.records)
        assert not profile["csv_ids_contiguous"]
        assert profile["csv_id_min"] == 0
        assert profile["csv_id_max"] == 2724
        assert profile["csv_id_gap_count"] == 78
        assert sum(profile["categories"].values()) == WEBBENCH_PUBLIC_TASK_COUNT
        assert profile["categories"]["READ"] == 1637
        assert profile["write_class_task_count"] == 1010
        assert profile["distinct_registrable_domains"] == 448
        assert profile["keyword_probe_counts"]["credential_or_account"] == 536

    @unittest.skipUnless(REAL_TRAINING_MANIFEST.is_file(), "training manifest is not on disk")
    def test_evaluation_split_is_disjoint_from_the_training_manifest(self) -> None:
        training = load_training_task_manifest(REAL_TRAINING_MANIFEST)
        proof = prove_split_disjointness([item["task_id"] for item in self.records], training)

        assert proof["disjoint"], proof["overlap_task_ids"][:10]
        assert proof["valid"], proof["errors"]
        assert proof["evaluation_task_count"] == WEBBENCH_PUBLIC_TASK_COUNT
        assert proof["training_declared_task_id_hash_verified"]

    def test_cli_build_split_is_offline_and_writes_artifacts(self) -> None:
        script = Path(__file__).with_name("pavlov_webbench_eval_adapter.py")
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--build-split",
                    "--dataset",
                    str(REAL_DATASET),
                    "--training-task-manifest",
                    str(REAL_TRAINING_MANIFEST),
                    "--out-dir",
                    tmp,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            assert process.returncode == 0, process.stderr
            assert "Traceback" not in process.stderr
            payload = json.loads(process.stdout)
            assert payload["derivation"]["aggregates"]["task_count"] == WEBBENCH_PUBLIC_TASK_COUNT
            assert not payload["derivation"]["network_accessed"]
            index_lines = (
                Path(payload["written"]["task_index"]).read_text(encoding="utf-8").splitlines()
            )
            assert len(index_lines) == WEBBENCH_PUBLIC_TASK_COUNT


if __name__ == "__main__":
    unittest.main()
