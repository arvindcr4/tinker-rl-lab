from __future__ import annotations

import json
import unittest
from importlib import import_module
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import pavlov_bfcl_adapter as adapter  # noqa: F401
else:
    try:
        from . import pavlov_bfcl_adapter as adapter
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        adapter = import_module("pavlov_bfcl_adapter")


def _valid_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "description": "tool name"},
            "arguments": {"type": "object", "description": "argument payload"},
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    }


def _valid_manifest() -> dict:
    return {
        "suite_id": "bfcl_train",
        "category": "tool_use",
        "dataset": {
            "revision": "a" * 40,
            "license": "cc-by-4.0",
            "source": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        },
        "verifier": {
            "identity": "platform_tinker.tinkerrl.grpo.StrictToolCallReward",
            "category": "tool_use",
            "function_call_schema": _valid_schema(),
        },
        "function_call_schema": _valid_schema(),
        "train_records": [
            {"id": "0" * 64, "hash": "1" * 64},
            {"id": "2" * 64, "hash": "3" * 64},
        ],
        "artifact_receipts": [
            {"kind": "wandb", "identity": "sha256:" + "5" * 64},
            {"kind": "checkpoint", "identity": "sha256:" + "6" * 64},
        ],
    }


class BFCLAdapterBoundaryTests(unittest.TestCase):
    def test_valid_payload_marks_adapter_ready_and_keeps_paid_launch_false(self) -> None:
        report = adapter.build_boundary_record(_valid_manifest())
        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["adapter_ready"])
        self.assertFalse(report["paid_launch_allowed"])
        self.assertFalse(report["launch"]["allowed"])
        self.assertIn("launch is intentionally disabled", report["launch"]["reasons"])
        self.assertEqual(report["artifact_receipt_count"], 2)
        self.assertEqual(report["train_count"], 2)

    def test_non_native_suite_id_blocks_adapter_ready(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "openreward_train"
        report = adapter.build_boundary_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["adapter_ready"])
        self.assertIn("suite_id must be 'bfcl_train'", " ".join(report["blockers"]))

    def test_rejects_disallowed_glaive_dataset(self) -> None:
        manifest = _valid_manifest()
        manifest["dataset"]["source"] = "glaiveai/glaive-function-calling-v2"
        report = adapter.build_boundary_record(manifest)
        self.assertFalse(report["adapter_ready"])
        self.assertIn("dataset source references glaive evidence", report["blockers"])

    def test_rejects_simulated_scaffolds_as_evidence(self) -> None:
        manifest = _valid_manifest()
        manifest["notes"] = "This run reused SimulatedBFCLv4 and bfclv4_tool_use.py"
        report = adapter.build_boundary_record(manifest)
        self.assertFalse(report["adapter_ready"])
        self.assertIn(
            "evidence source references synthetic simulator or Glaive artifacts",
            report["blockers"],
        )

    def test_strict_schema_rejects_missing_tool_or_arguments(self) -> None:
        manifest = _valid_manifest()
        manifest["verifier"]["function_call_schema"] = {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
            },
            "required": ["tool"],
            "additionalProperties": False,
        }
        report = adapter.build_boundary_record(manifest)
        self.assertFalse(report["adapter_ready"])
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn(
            "function-call schema missing property 'arguments'",
            report["blockers"],
        )

    def test_rejects_non_deterministic_train_records(self) -> None:
        manifest = _valid_manifest()
        manifest["train_records"] = [
            {"id": "2" * 64, "hash": "3" * 64},
            {"id": "0" * 64, "hash": "1" * 64},
        ]
        report = adapter.build_boundary_record(manifest)
        self.assertFalse(report["adapter_ready"])
        self.assertIn(
            "train_records must be sorted deterministically by id/hash",
            report["blockers"],
        )

    def test_rejects_empty_or_bad_train_records(self) -> None:
        manifest = _valid_manifest()
        manifest["train_records"] = []
        report = adapter.build_boundary_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("train_records cannot be empty", report["blockers"])

        manifest = _valid_manifest()
        manifest["train_records"] = [{"id": "x" * 64, "hash": "1" * 64}]
        report = adapter.build_boundary_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn(
            "train_records[0].id must be a 64-character hash",
            report["blockers"],
        )

    def test_missing_artifact_receipts_block(self) -> None:
        manifest = _valid_manifest()
        manifest["artifact_receipts"] = []
        report = adapter.build_boundary_record(manifest)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("artifact_receipts must be" in item for item in report["blockers"]))

    def test_no_network_or_credentials_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["requires_network"] = True
        report = adapter.build_boundary_record(manifest)
        self.assertFalse(report["adapter_ready"])
        self.assertIn("dataset download/network is disallowed", ", ".join(report["blockers"]))

        manifest = _valid_manifest()
        manifest["credential_ref"] = "secret-token"
        report = adapter.build_boundary_record(manifest)
        self.assertFalse(report["adapter_ready"])
        self.assertIn(
            "credential reference is not allowed in offline boundary",
            report["blockers"],
        )

    def test_launch_flag_is_deterministic_hash(self) -> None:
        first = adapter.build_boundary_record(_valid_manifest())
        second = adapter.build_boundary_record(_valid_manifest())
        self.assertEqual(first["train_records_digest"], second["train_records_digest"])
        self.assertEqual(
            first["function_call_schema_sha256"],
            second["function_call_schema_sha256"],
        )

    def test_main_reads_manifest_and_emits_status(self) -> None:
        manifest = _valid_manifest()
        temporary = Path(__file__).resolve().parent / "_tmp_bfcl_manifest.json"
        try:
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(adapter.main(["--manifest", str(temporary)]), 0)
            manifest["suite_id"] = "wrong_suite"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(adapter.main(["--manifest", str(temporary)]), 1)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    raise SystemExit(unittest.main())
