from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_webbench_eval_adapter import (
    WEBBENCH_DOMAINS,
    WEBBENCH_RECEIPT_FIELDS,
    WEBBENCH_ROLE,
    WEBBENCH_SOURCE_URL,
    WEBBENCH_SUITE_ID,
    receipt_digest,
    sha256_hex,
    split_manifest_hash,
    task_ids_hash,
    validate_webbench_manifest,
)


class PavlovWebBenchBoundaryTests(unittest.TestCase):
    def _receipt(self, binding: dict[str, str], number: int, payload_extra: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "identity": f"{number:040x}"[-40:],
            "artifact_digest": f"{number:064x}"[-64:],
        }
        if payload_extra:
            payload.update(payload_extra)
        return {
            "receipt_id": f"{number + 1000:040x}"[-40:],
            "digest": receipt_digest(binding, payload),
            "authenticated": True,
            "cryptographically_bound": True,
            "binding": binding,
            "payload": payload,
        }

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
            {"container_digest": environment["container_digest"], "runtime_digest": environment["runtime_digest"]},
        )
        environment["verifier_receipt"] = self._receipt(
            {"subject": WEBBENCH_SUITE_ID, "role": WEBBENCH_ROLE, "kind": "verifier"},
            6,
            {"verifier_revision": environment["verifier_contract"]["verifier_revision"]},
        )
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
                "checkpoint_url": "https://huggingface.co/example/webbench/commit/" + f"{22:040x}"[-40:],
            },
        )
        # The repeated URL expression above is intentionally corrected to a
        # valid single URL before hashing; it also makes accidental URL edits in
        # a test fixture visible rather than silently accepted.
        hf_payload = result_receipts["hf"]["payload"]
        hf_payload["checkpoint_url"] = "https://huggingface.co/example/webbench/commit/" + f"{22:040x}"[-40:]
        result_receipts["hf"]["digest"] = receipt_digest(result_receipts["hf"]["binding"], hf_payload)
        manifest["receipts"] = receipts
        manifest["result"] = result
        manifest["result_receipts"] = result_receipts
        manifest["heldout_status"] = "receipt_proven_heldout"
        return manifest

    def test_primary_eval_boundary_is_valid_but_not_heldout_without_receipts(self) -> None:
        report = validate_webbench_manifest(self._manifest())

        self.assertTrue(report["boundary_valid"])
        self.assertTrue(report["primary_eval"])
        self.assertFalse(report["receipt_proven_heldout"])
        self.assertEqual(report["status"], "READY_PRIMARY_EVAL_PENDING_RECEIPTS")

    def test_complete_receipts_prove_heldout_results(self) -> None:
        report = validate_webbench_manifest(self._with_heldout_receipts(self._manifest()))

        self.assertTrue(report["boundary_valid"])
        self.assertTrue(report["receipt_proven_heldout"])
        self.assertTrue(report["heldout_claim_allowed"])
        self.assertEqual(report["status"], "READY_RECEIPT_PROVEN_HELDOUT")

    def test_heldout_role_is_not_a_substitute_for_primary_eval(self) -> None:
        manifest = self._manifest()
        manifest["role"] = "heldout"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("role_invalid", report["blocker_codes"])

    def test_self_attested_heldout_status_without_receipts_blocks(self) -> None:
        manifest = self._manifest()
        manifest["heldout_status"] = "receipt_proven_heldout"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("heldout_status_without_receipts", report["blocker_codes"])

    def test_authoritative_source_identity_rejects_xlam_and_related_benchmarks(self) -> None:
        manifest = self._manifest()
        manifest["source"]["url"] = "https://huggingface.co/example/xlam"
        manifest["related_benchmarks"] = ["xLAM"]

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("source_identity_mismatch", report["blocker_codes"])
        self.assertIn("related_benchmark_substitution", report["blocker_codes"])

    def test_fake_revision_license_and_receipt_strings_block(self) -> None:
        manifest = self._manifest()
        manifest["source"]["revision"] = "main"
        manifest["source"]["license"] = "pending"
        manifest["source"]["revision_receipt"] = "verified"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("revision_not_pinned", report["blocker_codes"])
        self.assertIn("license_not_pinned", report["blocker_codes"])
        self.assertIn("source_receipt_invalid", report["blocker_codes"])

    def test_task_and_split_hashes_must_be_deterministic_and_bound(self) -> None:
        manifest = self._manifest()
        manifest["task_ids"] = list(reversed(manifest["task_ids"]))
        manifest["task_id_hash"] = "f" * 64

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("task_ids_not_deterministic", report["blocker_codes"])
        self.assertIn("task_id_hash_mismatch", report["blocker_codes"])

    def test_task_receipt_cannot_attest_to_a_different_task_hash(self) -> None:
        manifest = self._manifest()
        receipt = manifest["task_receipt"]
        receipt["payload"]["task_id_hash"] = "f" * 64
        receipt["digest"] = receipt_digest(receipt["binding"], receipt["payload"])

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("task_receipt_payload_mismatch", report["blocker_codes"])

    def test_native_artifact_and_verifier_contracts_are_required(self) -> None:
        manifest = self._manifest()
        del manifest["environment"]["artifact_contract"]
        manifest["environment"]["verifier_contract"] = "passed"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("artifact_contract_invalid", report["blocker_codes"])
        self.assertIn("verifier_contract_invalid", report["blocker_codes"])

    def test_result_material_without_all_wandb_tinker_hf_receipts_blocks(self) -> None:
        manifest = self._manifest()
        manifest["result"] = {"n": 100, "metric": 0.5, "result_id": "b" * 40}

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipts_missing", report["blocker_codes"])

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

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_tampered_result_digest_blocks_even_with_receipt_shapes(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["result_receipts"]["tinker"]["payload"]["result_digest"] = "f" * 64
        manifest["result_receipts"]["tinker"]["digest"] = receipt_digest(
            manifest["result_receipts"]["tinker"]["binding"],
            manifest["result_receipts"]["tinker"]["payload"],
        )

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_authoritative_service_urls_reject_credentials_and_query_only_revision(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        wandb = manifest["result_receipts"]["wandb"]
        wandb["payload"]["run_url"] = "https://user:secret@sub.wandb.ai/example/webbench/runs/abc"
        wandb["digest"] = receipt_digest(wandb["binding"], wandb["payload"])
        hf = manifest["result_receipts"]["hf"]
        revision = hf["payload"]["revision"]
        hf["payload"]["checkpoint_url"] = "https://huggingface.co/example/webbench/commit/latest?revision=" + revision
        hf["digest"] = receipt_digest(hf["binding"], hf["payload"])

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_public_hf_artifact_requires_explicit_clean_safety_flags(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        hf = manifest["result_receipts"]["hf"]
        hf["payload"]["visibility"] = "public"
        hf["digest"] = receipt_digest(hf["binding"], hf["payload"])

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("result_receipt_payload_invalid", report["blocker_codes"])

    def test_noncanonical_split_and_typed_boundary_fields_fail_closed(self) -> None:
        manifest = self._manifest()
        manifest["split_manifest"]["noncanonical"] = {"not-json"}
        manifest["domains"] = tuple(WEBBENCH_DOMAINS)
        manifest["heldout_status"] = []

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("split_manifest_not_canonicalizable", report["blocker_codes"])
        self.assertIn("domain_boundary_invalid", report["blocker_codes"])
        self.assertIn("heldout_status_invalid", report["blocker_codes"])

    def test_pending_status_with_complete_receipts_is_not_silently_promoted(self) -> None:
        manifest = self._with_heldout_receipts(self._manifest())
        manifest["heldout_status"] = "pending_receipts"

        report = validate_webbench_manifest(manifest)

        self.assertFalse(report["boundary_valid"])
        self.assertIn("heldout_status_stale", report["blocker_codes"])

    def test_cli_is_offline_and_returns_json(self) -> None:
        script = Path(__file__).with_name("pavlov_webbench_eval_adapter.py")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as handle:
            json.dump(self._manifest(), handle)
            handle.flush()
            process = subprocess.run(
                [sys.executable, str(script), "--manifest", handle.name],
                check=False,
                capture_output=True,
                text=True,
            )

        report = json.loads(process.stdout)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(report["status"], "READY_PRIMARY_EVAL_PENDING_RECEIPTS")
        self.assertFalse(report["receipt_proven_heldout"])
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
