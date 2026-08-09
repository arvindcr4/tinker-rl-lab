#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from . import pavlov_verilog_eval_split_manifest as validator  # noqa: F401
else:
    try:
        from . import pavlov_verilog_eval_split_manifest as validator
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        validator = import_module("pavlov_verilog_eval_split_manifest")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_ids() -> list[str]:
    return sorted(hashlib.sha256(f"verilog-eval-task-{index}".encode("utf-8")).hexdigest() for index in range(3))


def _aggregate(task_hashes: list[str]) -> str:
    return _sha256("\n".join(task_hashes))


def _split_manifest_hash(aggregate: str) -> str:
    payload = _canonical_json({"primary_eval": aggregate})
    return "sha256:" + _sha256(payload)


def _valid_manifest() -> dict[str, Any]:
    task_hashes = _task_ids()
    aggregate = _aggregate(task_hashes)
    return {
        "suite_id": "verilog_eval",
        "source": "https://github.com/NVlabs/verilog-eval",
        "category": "code",
        "role": "primary_eval",
        "dataset": {
            "revision": "a" * 40,
            "license": "apache-2.0",
            "source": "https://github.com/NVlabs/verilog-eval",
        },
        "task_id_hashes": task_hashes,
        "split": {
            "primary_eval": list(task_hashes),
            "hash": "sha256:" + aggregate,
        },
        "split_hashes": {
            "primary_eval": "sha256:" + aggregate,
        },
        "split_manifest_hash": _split_manifest_hash(aggregate),
        "split_manifest_receipt_ref": "sha256:" + ("a" * 64),
        "decontamination": {
            "status": "verified",
            "receipt_id": "b" * 40,
            "visibility": "private",
            "safe_public_artifact": True,
        },
    }


class VerilogEvalSplitManifestBoundaryTests(unittest.TestCase):
    def test_valid_manifest_builds_ready_and_disables_launch(self) -> None:
        report = validator.build_split_manifest_record(_valid_manifest())
        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["split_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertFalse(report["launch"]["allowed"])
        self.assertIn("launch is intentionally disabled", report["launch"]["reasons"])
        self.assertEqual(report["split"]["primary_eval"]["count"], 3)
        self.assertEqual(len(report["split"]["primary_eval"]["task_id_hashes"]), 3)

    def test_wrong_suite_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "openreward_games_eval"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertIn("suite_id must be 'verilog_eval'", " ".join(blocked["blockers"]))

    def test_authoritative_source_is_required(self) -> None:
        manifest = _valid_manifest()
        manifest["source"] = "https://github.com/example/fake"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(
            any("source must match authoritative verilog_eval source" in item for item in blocked["blockers"])
        )

    def test_rejects_xlam_and_glaive_substitution_markers(self) -> None:
        manifest = _valid_manifest()
        manifest["dataset"]["source"] = "https://github.com/Salesforce/xlam-function-calling-60k"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("blocked source" in item or "authoritative" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["source"] = "https://github.com/glaiveai/glaive-function-calling-v2"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("source must match authoritative verilog_eval source" in item for item in blocked["blockers"]))

    def test_task_and_split_hashes_must_be_deterministic(self) -> None:
        manifest = _valid_manifest()
        manifest["task_id_hashes"][0], manifest["task_id_hashes"][1] = manifest["task_id_hashes"][1], manifest["task_id_hashes"][0]
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("must be deterministically sorted" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["split"]["hash"] = "sha256:" + ("d" * 64)
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("hash does not match aggregate" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["split_manifest_hash"] = "sha256:" + ("e" * 64)
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(
            any(
                "split_manifest_hash does not match observed split aggregate" in item
                for item in blocked["blockers"]
            )
        )

    def test_decontamination_status_is_constrained(self) -> None:
        manifest = _valid_manifest()
        manifest["decontamination"]["status"] = "pending"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("decontamination.status is invalid" in item for item in blocked["blockers"]))

    def test_scope_with_e_suite_reference_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["scope"] = {"e_suite_ids": ["swe_bench_pro_eval"]}
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("E-suite" in item or "E-suite entries" in item for item in blocked["blockers"]))

    def test_no_network_or_credentials(self) -> None:
        manifest = _valid_manifest()
        manifest["requires_network"] = True
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("network is disallowed" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["api_key"] = "secret-token"
        blocked = validator.build_split_manifest_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("api_key is not allowed" in item for item in blocked["blockers"]))

    def test_dataset_and_receipt_digests_are_deterministic(self) -> None:
        first = validator.build_split_manifest_record(_valid_manifest())
        second = validator.build_split_manifest_record(copy.deepcopy(_valid_manifest()))
        self.assertEqual(first["status"], "READY")
        self.assertEqual(second["status"], "READY")
        self.assertEqual(first["task_id_digest"], second["task_id_digest"])
        self.assertEqual(
            first["split_manifest_payload_digest"],
            second["split_manifest_payload_digest"],
        )

    def test_validate_record_helper_matches_status(self) -> None:
        ready = validator.build_split_manifest_record(_valid_manifest())
        self.assertEqual(validator.validate_split_manifest_record(ready), [])

        blocked = validator.build_split_manifest_record({"suite_id": "x"})
        self.assertNotEqual(validator.validate_split_manifest_record(blocked), [])

    def test_main_reports_status(self) -> None:
        temporary = Path(__file__).resolve().parent / "_tmp_verilog_eval_split_manifest.json"
        try:
            manifest = _valid_manifest()
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validator.main(["--manifest", str(temporary)]), 0)

            manifest["suite_id"] = "not-verilog"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validator.main(["--manifest", str(temporary)]), 1)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    unittest.main()
