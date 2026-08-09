from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from flagship.pavlov_binaryaudit_eval_adapter import (
    AUTHORITATIVE_SOURCE_ID,
    AUTHORITATIVE_SOURCE_URL,
    BENCHMARK_ID,
    BinaryAuditBoundaryError,
    NOT_SCORED_SPLIT,
    PRIMARY_EVAL,
    RECEIPT_PROVEN_HELDOUT,
    RESULT_SCHEMA_VERSION,
    SPLIT_KEYS,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SUITE_ID,
    build_binaryaudit_split_manifest,
    deterministic_task_id,
    enumerate_binaryaudit_tasks,
    hash_task_directory,
    split_manifest_sha256,
    task_id_manifest_sha256,
    validate_binaryaudit_boundary,
    validate_binaryaudit_result_receipt,
)


REPO_TASKS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "e7_binaryaudit"
    / "BinaryAudit"
    / "tasks"
)
PINNED_REVISION = "cbd86c7cd8519f01ae6b7ad7db7fdb653ea54f23"


SOURCE_REVISION = "a" * 40
LICENSE_HASH = "b" * 64
CONTAINER_DIGEST = "sha256:" + "c" * 64
ENVIRONMENT_HASH = "d" * 64
ARTIFACT_HASH = "e" * 64
VERIFIER_REVISION = "f" * 40


def _boundary() -> dict[str, object]:
    split = {
        "train": ["task-001", "task-002"],
        "primary_eval": ["task-101", "task-102"],
        "receipt_proven_heldout": ["task-201", "task-202"],
    }
    task_ids = sorted(item for values in split.values() for item in values)
    return {
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "source_identity": {
            "id": AUTHORITATIVE_SOURCE_ID,
            "url": AUTHORITATIVE_SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license_spdx": "Apache-2.0",
            "license_text_sha256": LICENSE_HASH,
            "license_url": AUTHORITATIVE_SOURCE_URL + "/blob/" + SOURCE_REVISION + "/LICENSE",
        },
        "task_ids": task_ids,
        "task_id_manifest_sha256": task_id_manifest_sha256(task_ids),
        "split_manifest": split,
        "split_manifest_sha256": split_manifest_sha256(split),
        "native_environment": {
            "mode": "native",
            "container_digest": CONTAINER_DIGEST,
            "environment_manifest_sha256": ENVIRONMENT_HASH,
            "artifact_contract": {
                "required": True,
                "manifest_sha256": ARTIFACT_HASH,
                "types": ["binary", "stdout", "stderr"],
            },
            "verifier_contract": {
                "name": "binaryaudit-native-verifier",
                "revision": VERIFIER_REVISION,
                "receipt_schema": RESULT_SCHEMA_VERSION,
                "checks": ["exit_code", "artifact_hash", "state_digest"],
            },
        },
    }


def _receipt(role: str = PRIMARY_EVAL) -> dict[str, object]:
    boundary = _boundary()
    task_ids = boundary["split_manifest"][role]  # type: ignore[index]
    receipt: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "benchmark_id": BENCHMARK_ID,
        "evaluation_role": role,
        "status": "completed",
        "source_identity": copy.deepcopy(boundary["source_identity"]),
        "task_ids": list(task_ids),
        "task_id_manifest_sha256": task_id_manifest_sha256(task_ids),
        "split_manifest_sha256": boundary["split_manifest_sha256"],
        "native_environment_receipt": {
            "container_digest": CONTAINER_DIGEST,
            "environment_manifest_sha256": ENVIRONMENT_HASH,
        },
        "artifact_receipt": {
            "manifest_sha256": ARTIFACT_HASH,
            "paths": ["artifacts/task-101/report.json"],
        },
        "verifier_receipt": {
            "name": "binaryaudit-native-verifier",
            "revision": VERIFIER_REVISION,
            "receipt_schema": RESULT_SCHEMA_VERSION,
        },
        "wandb": {
            "run_id": "wandb-binaryaudit-1",
            "url": "https://wandb.example/run/binaryaudit-1",
            "mode": "online",
            "config_sha256": "1" * 64,
            "metrics_receipt_sha256": "2" * 64,
            "metrics_logged": True,
        },
        "tinker": {
            "run_id": "tinker-binaryaudit-1",
            "status": "completed",
            "config_sha256": "3" * 64,
            "cost_usd": 0.25,
        },
        "hf": {
            "repo": "private/binaryaudit-adapter",
            "revision": "4" * 40,
            "visibility": "private",
            "artifact_manifest_sha256": ARTIFACT_HASH,
        },
    }
    if role == PRIMARY_EVAL:
        receipt["metrics"] = {"task_success_rate": 0.5, "artifact_integrity_rate": 1.0}
    else:
        receipt["heldout_proof"] = {
            "selection_task_id_manifest_sha256": task_id_manifest_sha256(
                boundary["split_manifest"]["train"]  # type: ignore[index]
            ),
            "heldout_task_id_manifest_sha256": task_id_manifest_sha256(task_ids),
            "disjoint": True,
            "selection_locked": True,
            "not_used_for_selection": True,
        }
    return receipt


class HashAndBoundaryTests(unittest.TestCase):
    def test_hashes_and_task_ids_are_deterministic(self) -> None:
        boundary = _boundary()
        self.assertEqual(boundary["task_id_manifest_sha256"], task_id_manifest_sha256(boundary["task_ids"]))
        self.assertEqual(boundary["split_manifest_sha256"], split_manifest_sha256(boundary["split_manifest"]))
        first = deterministic_task_id("raw-1", SOURCE_REVISION)
        second = deterministic_task_id("raw-1", SOURCE_REVISION)
        self.assertEqual(first, second)
        self.assertNotEqual(first, deterministic_task_id("raw-2", SOURCE_REVISION))

    def test_valid_boundary_pins_source_revision_license_and_native_contract(self) -> None:
        normalized = validate_binaryaudit_boundary(_boundary())
        self.assertEqual(normalized["source_identity"]["id"], AUTHORITATIVE_SOURCE_ID)
        self.assertEqual(normalized["source_identity"]["revision"], SOURCE_REVISION)
        self.assertEqual(normalized["native_environment"]["mode"], "native")
        self.assertTrue(normalized["native_environment"]["artifact_contract"]["required"])
        self.assertTrue(normalized["substitutes_rejected"])

    def test_mutable_revision_license_and_hash_drift_fail_closed(self) -> None:
        for mutation, expected in (
            (lambda item: item["source_identity"].update(revision="main"), "mutable tag/branch"),
            (lambda item: item["source_identity"].update(license_text_sha256="not-a-hash"), "license_text_sha256"),
            (lambda item: item.update(task_id_manifest_sha256="0" * 64), "does not match task_ids"),
            (lambda item: item.update(split_manifest_sha256="0" * 64), "does not match split_manifest"),
        ):
            with self.subTest(expected=expected):
                item = _boundary()
                mutation(item)
                with self.assertRaises(BinaryAuditBoundaryError) as raised:
                    validate_binaryaudit_boundary(item)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

    def test_related_benchmarks_and_xlam_are_not_substitutes(self) -> None:
        for mutation in (
            lambda item: item.update(benchmark_id="xlam"),
            lambda item: item.update(related_benchmark="BFCL"),
            lambda item: item.update(substitutes=["AgentHarm"]),
        ):
            item = _boundary()
            mutation(item)
            with self.assertRaises(BinaryAuditBoundaryError):
                validate_binaryaudit_boundary(item)

    def test_split_ids_must_be_sorted_unique_and_disjoint(self) -> None:
        item = _boundary()
        item["split_manifest"]["primary_eval"] = ["task-102", "task-101"]  # type: ignore[index]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_boundary(item)
        self.assertTrue(any("lexically sorted" in message for message in raised.exception.diagnostics))

        item = _boundary()
        item["split_manifest"]["primary_eval"] = ["task-001", "task-101"]  # type: ignore[index]
        item["task_ids"] = sorted(item["task_ids"])  # type: ignore[arg-type]
        item["task_id_manifest_sha256"] = task_id_manifest_sha256(item["task_ids"])  # type: ignore[arg-type]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_boundary(item)
        self.assertTrue(any("overlapping IDs" in message for message in raised.exception.diagnostics))


class ReceiptBoundaryTests(unittest.TestCase):
    def test_primary_eval_receipt_is_evidence_bearing_only_after_full_receipts(self) -> None:
        normalized = validate_binaryaudit_result_receipt(_boundary(), _receipt())
        self.assertEqual(normalized["status"], "admissible_primary_eval")
        self.assertTrue(normalized["primary_eval"])
        self.assertTrue(normalized["scientific_evidence"])
        self.assertEqual(normalized["metrics"]["task_success_rate"], 0.5)
        self.assertFalse(normalized["portfolio_claim_permitted"])
        self.assertEqual(normalized["tracking_receipts"]["wandb"]["mode"], "online")

    def test_receipt_proven_heldout_is_not_primary_eval(self) -> None:
        normalized = validate_binaryaudit_result_receipt(
            _boundary(), _receipt(RECEIPT_PROVEN_HELDOUT)
        )
        self.assertEqual(normalized["status"], "receipt_proven_heldout")
        self.assertFalse(normalized["primary_eval"])
        self.assertTrue(normalized["receipt_proven_heldout"])
        self.assertFalse(normalized["scientific_evidence"])
        self.assertTrue(normalized["primary_eval_required"])

    def test_heldout_label_without_proof_is_rejected(self) -> None:
        receipt = _receipt(RECEIPT_PROVEN_HELDOUT)
        del receipt["heldout_proof"]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_result_receipt(_boundary(), receipt)
        self.assertTrue(any("requires heldout_proof" in message for message in raised.exception.diagnostics))

    def test_related_receipt_and_wrong_split_are_rejected(self) -> None:
        for mutation, expected in (
            (lambda item: item.update(benchmark_id="xlam"), "BinaryAudit"),
            (lambda item: item.update(task_ids=["task-201", "task-202"]), "primary_eval split"),
            (lambda item: item.update(split_manifest_sha256="0" * 64), "differs from boundary"),
        ):
            with self.subTest(expected=expected):
                receipt = _receipt()
                mutation(receipt)
                if "task_ids" in receipt:
                    receipt["task_id_manifest_sha256"] = task_id_manifest_sha256(receipt["task_ids"])
                with self.assertRaises(BinaryAuditBoundaryError) as raised:
                    validate_binaryaudit_result_receipt(_boundary(), receipt)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

    def test_missing_or_unsafe_tracking_receipts_fail_closed(self) -> None:
        for key, mutation, expected in (
            ("wandb", lambda item: item["wandb"].update(mode="offline"), "W&B receipt mode"),
            ("tinker", lambda item: item["tinker"].update(status="failed"), "Tinker receipt status"),
            ("hf", lambda item: item["hf"].update(visibility="public"), "visibility"),
        ):
            with self.subTest(key=key):
                receipt = _receipt()
                mutation(receipt)
                with self.assertRaises(BinaryAuditBoundaryError) as raised:
                    validate_binaryaudit_result_receipt(_boundary(), receipt)
                self.assertTrue(any(expected in message for message in raised.exception.diagnostics))

        receipt = _receipt()
        del receipt["hf"]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_result_receipt(_boundary(), receipt)
        self.assertTrue(any("missing Hugging Face" in message for message in raised.exception.diagnostics))

    def test_native_artifact_and_verifier_drift_is_rejected(self) -> None:
        receipt = _receipt()
        receipt["native_environment_receipt"]["container_digest"] = "sha256:" + "9" * 64  # type: ignore[index]
        receipt["artifact_receipt"]["manifest_sha256"] = "8" * 64  # type: ignore[index]
        receipt["verifier_receipt"]["revision"] = "7" * 40  # type: ignore[index]
        with self.assertRaises(BinaryAuditBoundaryError) as raised:
            validate_binaryaudit_result_receipt(_boundary(), receipt)
        diagnostics = " ".join(raised.exception.diagnostics)
        self.assertIn("container_digest differs", diagnostics)
        self.assertIn("artifact receipt manifest differs", diagnostics)
        self.assertIn("verifier receipt revision differs", diagnostics)


def _write_task(root: Path, name: str, *, marker: str | None = None, pytest_verifier: bool = False,
                script: str = "echo 1 > /logs/verifier/reward.txt\n") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "instruction.md").write_text(f"# {name}\n", encoding="utf-8")
    (task / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
    (task / "environment" / "Dockerfile").write_text("FROM binaryaudit-base:latest\n", encoding="utf-8")
    (task / "tests" / "test.sh").write_text(script, encoding="utf-8")
    if pytest_verifier:
        (task / "tests" / "test_outputs.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    if marker:
        (task / marker).write_text("upstream status\n", encoding="utf-8")
    return task


class BinaryAuditSplitManifestTests(unittest.TestCase):
    def _synthetic_root(self, stack: tempfile.TemporaryDirectory) -> Path:
        root = Path(stack.name) / "tasks"
        root.mkdir()
        _write_task(root, "lighttpd-backdoor-detect",
                    script='EXPECTED_FUNC_START="0x1"\nEXPECTED_FUNC_END="0x2"\n')
        _write_task(root, "lighttpd-backdoor-detect-negative", script='if [ "$A" = "NO" ]; then :; fi\n')
        _write_task(root, "dnsmasq-backdoor-detect", script='EXPECTED_FUNC_START="0x1"\n')
        _write_task(root, "sozu-timebomb-multiple-binaries-detect", pytest_verifier=True)
        _write_task(root, "caddy-backdoor-simple-detect")
        _write_task(root, "caddy-backdoor-detect", marker="STATUS_FAILING.md")
        _write_task(root, "pingora-backdoor-detect", marker="STATUS_NOT_FINISHED.md")
        _write_task(root, "ghidra-decompile-vanilla")
        _write_task(root, "radare2-decompile")
        return root

    def test_hash_task_directory_is_deterministic_and_content_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            root.mkdir()
            task = _write_task(root, "dnsmasq-backdoor-detect")
            first = hash_task_directory(task)
            self.assertEqual(first, hash_task_directory(task))
            self.assertEqual(first["file_count"], 4)
            (task / "instruction.md").write_text("# changed\n", encoding="utf-8")
            self.assertNotEqual(first["content_sha256"], hash_task_directory(task)["content_sha256"])

    def test_hash_task_directory_rejects_missing_and_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BinaryAuditBoundaryError):
                hash_task_directory(Path(tmp) / "absent")
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(BinaryAuditBoundaryError):
                hash_task_directory(empty)

    def test_enumeration_classifies_targets_categories_and_verifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stack = type("S", (), {"name": tmp})()
            root = self._synthetic_root(stack)  # type: ignore[arg-type]
            records = {r["raw_task_id"]: r for r in enumerate_binaryaudit_tasks(root, PINNED_REVISION)}
            self.assertEqual(len(records), 9)

            self.assertEqual(records["dnsmasq-backdoor-detect"]["target"], "dnsmasq")
            self.assertEqual(records["dnsmasq-backdoor-detect"]["target_language"], "C")
            self.assertEqual(records["dnsmasq-backdoor-detect"]["category"], "backdoor_detect")
            self.assertEqual(records["dnsmasq-backdoor-detect"]["verifier_kind"], "bash_address_range")
            self.assertEqual(records["dnsmasq-backdoor-detect"]["split"], PRIMARY_EVAL)

            self.assertEqual(records["lighttpd-backdoor-detect-negative"]["category"], "negative_control")
            self.assertEqual(records["lighttpd-backdoor-detect-negative"]["verifier_kind"], "bash_exact_no")

            timebomb = records["sozu-timebomb-multiple-binaries-detect"]
            self.assertEqual(timebomb["category"], "timebomb")
            self.assertEqual(timebomb["verifier_kind"], "pytest_result_json")
            self.assertEqual(timebomb["split"], RECEIPT_PROVEN_HELDOUT)

            self.assertEqual(records["ghidra-decompile-vanilla"]["category"], "tool_operation")
            self.assertEqual(records["radare2-decompile"]["split"], NOT_SCORED_SPLIT)

            for quarantined, marker in (
                ("caddy-backdoor-detect", "STATUS_FAILING.md"),
                ("pingora-backdoor-detect", "STATUS_NOT_FINISHED.md"),
            ):
                self.assertEqual(records[quarantined]["split"], NOT_SCORED_SPLIT)
                self.assertEqual(records[quarantined]["split_rule"], "R1_upstream_quarantine_marker")
                self.assertEqual(records[quarantined]["upstream_status_markers"], [marker])
                self.assertFalse(records[quarantined]["scored"])

            # Quarantine outranks the language rule: caddy is Go but is not held out.
            self.assertEqual(records["caddy-backdoor-simple-detect"]["split"], RECEIPT_PROVEN_HELDOUT)

    def test_task_ids_are_revision_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            root.mkdir()
            _write_task(root, "dnsmasq-backdoor-detect")
            other = "b" * 40
            here = enumerate_binaryaudit_tasks(root, PINNED_REVISION)[0]["task_id"]
            there = enumerate_binaryaudit_tasks(root, other)[0]["task_id"]
            self.assertNotEqual(here, there)
            self.assertEqual(here, deterministic_task_id("dnsmasq-backdoor-detect", PINNED_REVISION))

    def test_enumeration_rejects_unknown_target_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tasks"
            root.mkdir()
            _write_task(root, "nginx-backdoor-detect")
            with self.assertRaises(BinaryAuditBoundaryError) as raised:
                enumerate_binaryaudit_tasks(root, PINNED_REVISION)
            self.assertIn("unknown target family", " ".join(raised.exception.diagnostics))

    def test_enumeration_rejects_missing_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(BinaryAuditBoundaryError):
                enumerate_binaryaudit_tasks(Path(tmp) / "absent", PINNED_REVISION)
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(BinaryAuditBoundaryError):
                enumerate_binaryaudit_tasks(empty, PINNED_REVISION)

    def test_split_manifest_is_disjoint_covering_and_self_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stack = type("S", (), {"name": tmp})()
            root = self._synthetic_root(stack)  # type: ignore[arg-type]
            manifest = build_binaryaudit_split_manifest(root, PINNED_REVISION)

            self.assertEqual(manifest["schema_version"], SPLIT_MANIFEST_SCHEMA_VERSION)
            self.assertEqual(manifest["manifest_provenance"], "lane_constructed_not_upstream_official")

            proof = manifest["disjointness_proof"]
            self.assertTrue(proof["pairwise_disjoint"])
            self.assertTrue(proof["union_equals_task_ids"])
            self.assertTrue(proof["size_sum_equals_total"])
            self.assertEqual(proof["total_tasks"], 9)
            self.assertEqual(proof["union_size"], 9)
            self.assertEqual(sum(proof["split_sizes"].values()), 9)
            self.assertEqual(set(proof["pairwise_intersection_sizes"].values()), {0})

            self.assertEqual(
                manifest["task_id_manifest_sha256"], task_id_manifest_sha256(manifest["task_ids"])
            )
            self.assertEqual(
                manifest["split_manifest_sha256"], split_manifest_sha256(manifest["split_manifest"])
            )
            for key in SPLIT_KEYS:
                ids = manifest["split_manifest"][key]
                self.assertEqual(ids, sorted(ids))
                self.assertEqual(len(ids), len(set(ids)))

    def test_generated_split_manifest_is_accepted_by_the_boundary_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stack = type("S", (), {"name": tmp})()
            root = self._synthetic_root(stack)  # type: ignore[arg-type]
            manifest = build_binaryaudit_split_manifest(root, PINNED_REVISION)

            boundary = _boundary()
            boundary["source_identity"]["revision"] = PINNED_REVISION  # type: ignore[index]
            boundary["task_ids"] = manifest["task_ids"]
            boundary["task_id_manifest_sha256"] = manifest["task_id_manifest_sha256"]
            boundary["split_manifest"] = manifest["split_manifest"]
            boundary["split_manifest_sha256"] = manifest["split_manifest_sha256"]

            validated = validate_binaryaudit_boundary(boundary)
            self.assertEqual(validated["split_manifest"], manifest["split_manifest"])
            self.assertEqual(validated["task_ids"], manifest["task_ids"])


@unittest.skipUnless(REPO_TASKS_ROOT.is_dir(), "BinaryAudit checkout not present")
class BinaryAuditPinnedCheckoutTests(unittest.TestCase):
    """Pins the real 46-task manifest so an upstream change is loud, not silent."""

    def test_pinned_revision_yields_forty_six_disjoint_tasks(self) -> None:
        manifest = build_binaryaudit_split_manifest(REPO_TASKS_ROOT, PINNED_REVISION)
        proof = manifest["disjointness_proof"]
        self.assertEqual(proof["total_tasks"], 46)
        self.assertEqual(
            proof["split_sizes"],
            {"train": 8, PRIMARY_EVAL: 28, RECEIPT_PROVEN_HELDOUT: 10},
        )
        self.assertTrue(proof["pairwise_disjoint"])
        self.assertTrue(proof["union_equals_task_ids"])
        self.assertEqual(
            manifest["task_id_manifest_sha256"],
            "c6c708d518303148efaf22c17537918e8659014c1121ecd49da35d099383fbca",
        )
        self.assertEqual(
            manifest["split_manifest_sha256"],
            "1420f980bf89e7778af8c7c2327c0d23d742f7f28c4cc294bc9889077207b1a2",
        )
        self.assertEqual(
            manifest["task_content_manifest_sha256"],
            "81f0d585d461b34dbda6e439a9f847edd7d665e7339d89c1f595cb0520d09157",
        )

    def test_quarantined_tasks_are_never_scored(self) -> None:
        manifest = build_binaryaudit_split_manifest(REPO_TASKS_ROOT, PINNED_REVISION)
        quarantined = {
            record["raw_task_id"]
            for record in manifest["tasks"]
            if record["upstream_status_markers"]
        }
        self.assertEqual(quarantined, {"caddy-backdoor-detect", "pingora-backdoor-detect"})
        for record in manifest["tasks"]:
            if record["upstream_status_markers"]:
                self.assertFalse(record["scored"])
                self.assertEqual(record["split"], NOT_SCORED_SPLIT)

    def test_written_manifest_matches_a_fresh_build(self) -> None:
        path = REPO_TASKS_ROOT.parents[1] / "split_manifest.json"
        if not path.is_file():
            self.skipTest("split_manifest.json not generated yet")
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        fresh = build_binaryaudit_split_manifest(REPO_TASKS_ROOT, PINNED_REVISION)
        for key in ("task_id_manifest_sha256", "split_manifest_sha256", "task_content_manifest_sha256"):
            self.assertEqual(on_disk[key], fresh[key])
        self.assertEqual(on_disk["split_manifest"], fresh["split_manifest"])


if __name__ == "__main__":
    unittest.main()
