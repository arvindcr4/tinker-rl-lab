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
    from . import pavlov_verilog_eval_adapter as validator  # noqa: F401
else:
    try:
        from . import pavlov_verilog_eval_adapter as validator
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        validator = import_module("pavlov_verilog_eval_adapter")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _task_ids() -> list[str]:
    return sorted(
        hashlib.sha256(f"verilog-eval-task-{index}".encode("utf-8")).hexdigest()
        for index in (0, 1, 2)
    )


def _split_aggregate(task_hashes: list[str]) -> str:
    return hashlib.sha256("\n".join(task_hashes).encode("utf-8")).hexdigest()


def _split_manifest_hash(aggregate: str) -> str:
    payload = _canonical_json({"primary_eval": aggregate}).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _valid_manifest() -> dict[str, Any]:
    task_hashes = _task_ids()
    aggregate = _split_aggregate(task_hashes)

    return {
        "suite_id": "verilog_eval",
        "source": "https://github.com/NVlabs/verilog-eval",
        "category": "code",
        "role": "primary_eval",
        "task_id_hashes": task_hashes,
        "split": {
            "primary_eval": list(task_hashes),
            "hash": f"sha256:{aggregate}",
        },
        "split_hashes": {
            "primary_eval": f"sha256:{aggregate}",
        },
        "split_manifest_hash": _split_manifest_hash(aggregate),
        "split_manifest_receipt_ref": "sha256:" + ("a" * 64),
        "dataset": {
            "revision": "a" * 40,
            "license": "apache-2.0",
            "source": "https://github.com/NVlabs/verilog-eval",
        },
        "verifier": {
            "identity": "platform_tinker.tinkerrl.grpo.StrictVerilogReward",
            "hash": "a" * 64,
        },
        "environment": {
            "container": "verilog-container",
            "image": "verilog-runner-image",
            "container_digest": "sha256:" + ("a" * 64),
            "runtime_digest": "sha256:" + ("b" * 64),
        },
        "held_out": False,
        "held_out_receipt_ref": None,
        "scope": {
            "is_portfolio": False,
            "is_held_out": False,
        },
        "wandb_run_identity": {
            "online": True,
            "entity": "pavlov",
            "project": "verilog-eval",
            "group": "suite",
            "run_id": "run-id-abc123",
            "run_url": "https://wandb.ai/pavlov/verilog-eval/runs/run-id-abc123",
        },
        "tinker_run_identity": {
            "run_id": "tinker-run-001",
            "cost_status": "authorized",
        },
        "hf_checkpoints": [
            {
                "repo_url": "https://huggingface.co/example-org/verilog-runner",
                "revision": "a" * 40,
                "url": "https://huggingface.co/example-org/verilog-runner/commit/a",
                "stage": "initial",
                "safe_public_artifact": True,
                "visibility": "public",
            },
            {
                "repo_url": "https://huggingface.co/example-org/verilog-runner",
                "revision": "b" * 40,
                "url": "https://huggingface.co/example-org/verilog-runner/commit/b",
                "stage": "periodic",
                "safe_public_artifact": True,
                "visibility": "private",
            },
            {
                "repo_url": "https://huggingface.co/example-org/verilog-runner",
                "revision": "c" * 40,
                "url": "https://huggingface.co/example-org/verilog-runner/commit/c",
                "stage": "final",
                "safe_public_artifact": True,
                "visibility": "public",
            },
        ],
        "costs": {
            "status": "authorized",
            "total_usd": 12.5,
        },
    }


class VerilogEvalBoundaryTests(unittest.TestCase):
    def test_valid_manifest_builds_ready_and_disables_launch(self) -> None:
        report = validator.build_verilog_eval_boundary_record(_valid_manifest())
        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["eval_ready"])
        self.assertTrue(report.get("adapter_ready", report["eval_ready"]))
        self.assertFalse(report["paid_launch_allowed"])
        self.assertFalse(report["launch"]["allowed"])
        self.assertIn("launch is intentionally disabled", report["launch"]["reasons"])
        self.assertFalse(report["held_out"])
        self.assertFalse(report["held_out_receipt_proven"])
        self.assertEqual(report["split"]["primary_eval"]["count"], 3)
        self.assertEqual(len(report["hf_checkpoints"]), 3)

    def test_authoritative_source_and_suite_must_match(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "swe_bench_pro_eval"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("suite_id must be 'verilog_eval'" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["source"] = "https://github.com/other/repo"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("source must match authoritative" in item for item in blocked["blockers"]))

    def test_pinned_dataset_revision_and_license_required(self) -> None:
        manifest = _valid_manifest()
        manifest["dataset"]["revision"] = "main"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("dataset.revision" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["dataset"]["license"] = "to_be_pinned_before_paid_runs"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("dataset.license" in item or "dataset license" in item for item in blocked["blockers"]))

    def test_rejects_xlam_and_glaive_markers(self) -> None:
        manifest = _valid_manifest()
        manifest["dataset"]["source"] = "https://github.com/Salesforce/xlam-function-calling-60k"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("must match authoritative" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["verifier"]["identity"] = "glaive-function-calling-v2"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("verifier references" in item for item in blocked["blockers"]))

    def test_task_hash_and_split_hashes_are_deterministic(self) -> None:
        manifest = _valid_manifest()
        manifest["task_id_hashes"] = sorted(manifest["task_id_hashes"], reverse=True)
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("deterministically sorted" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["split"]["hash"] = "sha256:" + ("d" * 64)
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("aggregate hash does not match" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["split_manifest_hash"] = "sha256:" + ("d" * 64)
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("split_manifest_hash does not match observed split" in item for item in blocked["blockers"]))

    def test_held_out_requires_receipt(self) -> None:
        manifest = _valid_manifest()
        manifest["held_out"] = True
        manifest["scope"]["is_held_out"] = True
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("held_out_receipt_ref" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["held_out"] = False
        manifest["held_out_receipt_ref"] = "sha256:" + ("d" * 64)
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("held_out_receipt_ref is forbidden" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["held_out"] = True
        manifest["held_out_receipt_ref"] = "sha256:" + ("d" * 64)
        manifest["scope"]["is_held_out"] = True
        report = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(report["status"], "READY")
        self.assertTrue(report["held_out"])
        self.assertTrue(report["held_out_receipt_proven"])

    def test_scope_must_be_explicit_non_portfolio(self) -> None:
        manifest = _valid_manifest()
        manifest["scope"] = {"is_portfolio": True, "is_held_out": False}
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("scope.is_portfolio must be false" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        del manifest["scope"]
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("scope must be an object" in item for item in blocked["blockers"]))

    def test_wandb_tinker_and_hf_receipts_required_and_hard(self) -> None:
        manifest = _valid_manifest()
        manifest["wandb_run_identity"]["online"] = False
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("online must be true" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["tinker_run_identity"]["cost_status"] = "pending"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("cost_status is invalid" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["hf_checkpoints"] = manifest["hf_checkpoints"][:2]
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("missing required stage" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["hf_checkpoints"][0]["safe_public_artifact"] = False
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("safe_public_artifact must be true" in item for item in blocked["blockers"]))

    def test_costs_require_non_negative_number(self) -> None:
        manifest = _valid_manifest()
        manifest["costs"]["status"] = "pending"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("costs.status is invalid" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["costs"]["total_usd"] = -1
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("cannot be negative" in item for item in blocked["blockers"]))

    def test_offline_gate_blocks_network_and_credentials(self) -> None:
        manifest = _valid_manifest()
        manifest["requires_network"] = True
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("network is disallowed" in item for item in blocked["blockers"]))

        manifest = _valid_manifest()
        manifest["api_key"] = "sk-test"
        blocked = validator.build_verilog_eval_boundary_record(manifest)
        self.assertEqual(blocked["status"], "BLOCKED")
        self.assertTrue(any("api_key is not allowed" in item for item in blocked["blockers"]))

    def test_split_and_output_digests_are_repeatable(self) -> None:
        report_a = validator.build_verilog_eval_boundary_record(_valid_manifest())
        report_b = validator.build_verilog_eval_boundary_record(copy.deepcopy(_valid_manifest()))
        self.assertEqual(report_a["status"], "READY")
        self.assertEqual(report_b["status"], "READY")
        self.assertEqual(report_a["task_id_digest"], report_b["task_id_digest"])
        self.assertEqual(
            report_a["authority_split_manifest_digest"],
            report_b["authority_split_manifest_digest"],
        )

    def test_validator_api_matches_status(self) -> None:
        report = validator.build_verilog_eval_boundary_record(_valid_manifest())
        self.assertEqual(validator.validate_verilog_eval_boundary_record(report), [])

        blocked = validator.build_verilog_eval_boundary_record({"suite_id": "x"})
        self.assertNotEqual(validator.validate_verilog_eval_boundary_record(blocked), [])

    def test_main_reports_status(self) -> None:
        valid = _valid_manifest()
        temporary = Path(__file__).resolve().parent / "_tmp_verilog_eval_manifest.json"
        try:
            temporary.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(validator.main(["--manifest", str(temporary)]), 0)

            valid["suite_id"] = "not-verilog"
            temporary.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(validator.main(["--manifest", str(temporary)]), 1)
        finally:
            if temporary.exists():
                temporary.unlink()


if __name__ == "__main__":
    unittest.main()
