#!/usr/bin/env python3
"""Adversarial tests for the xLAM launch audit."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from flagship import pavlov_xlam_launch_audit as launch_audit
from flagship import pavlov_xlam_smoke_config as smoke


class PavlovXlamLaunchAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = smoke.generate_smoke_config()

    def _valid_payloads(self) -> tuple[dict, dict, dict, list[dict]]:
        wandb = {
            "entity": "tesla-lab",
            "project": "pavlov",
            "group": "xlam-smoke",
            "run_id": "wandb-run-42",
            "run_url": "https://wandb.ai/tesla-lab/pavlov/runs/wandb-run-42",
            "state": "finished",
            "mode": "online",
        }
        checkpoints = [
            {"stage": "initial", "step": 0, "run_id": "tinker-run-42"},
            {"stage": "periodic", "step": 5, "run_id": "tinker-run-42"},
            {"stage": "final", "step": 10, "run_id": "tinker-run-42"},
        ]
        hf_receipts = [
            {
                "stage": "initial",
                "step": 0,
                "revision": "1" * 40,
                "repo_url": "https://huggingface.co/org/pavlov_xlam_initial",
                "visibility": "public",
                "safe_public_artifact": True,
                "run_id": "tinker-run-42",
            },
            {
                "stage": "periodic",
                "step": 5,
                "revision": "2" * 40,
                "repo_url": "https://huggingface.co/org/pavlov_xlam_periodic",
                "visibility": "private",
                "run_id": "tinker-run-42",
            },
            {
                "stage": "final",
                "step": 10,
                "revision": "3" * 40,
                "repo_url": "https://huggingface.co/org/pavlov_xlam_final",
                "visibility": "public",
                "safe_public_artifact": True,
                "run_id": "tinker-run-42",
            },
        ]
        return wandb, {"run_id": "tinker-run-42", "status": "finished"}, checkpoints, hf_receipts

    def test_generate_launch_audit_ready_when_complete(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        report = launch_audit.generate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertEqual(errors, [])
        self.assertEqual(report["schema_version"], launch_audit.SCHEMA_VERSION)
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["component_only"], True)
        self.assertFalse(report["launchable"])
        self.assertFalse(report["allowed"])
        self.assertEqual(report["tinker_run_id"], "tinker-run-42")
        self.assertEqual(report["wandb_run_id"], "wandb-run-42")
        self.assertEqual(report["blockers"], [])

    def test_smoke_config_drifts_blocked(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        bad = copy.deepcopy(self.config)
        bad["model"] = "gpt-4"
        errors = launch_audit.validate_launch_audit(
            bad,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertEqual(len(errors), 2)
        self.assertIn("smoke config: model must be Qwen/Qwen3.6-35B-A3B", errors)
        self.assertIn("smoke config: config_signature is invalid or missing", errors)

    def test_wandb_state_or_mode_missing_blocks(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        bad = copy.deepcopy(wandb)
        bad["state"] = "running"
        bad["mode"] = "offline"
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=bad,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertTrue(any("wandb state must be finished after run" in error for error in errors))
        self.assertTrue(any("wandb mode must be online" in error for error in errors))

    def test_checkpoint_step_and_stage_cross_checks(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        checkpoints[1]["step"] = 6
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertTrue(any("checkpoint[periodic] step mismatch" in error for error in errors))

    def test_missing_hf_stage_blocks(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf[0:2],
        )
        self.assertTrue(any("hf receipts missing stages: final" in error for error in errors))
        report = launch_audit.generate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf[0:2],
        )
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["launchable"])
        self.assertFalse(report["allowed"])

    def test_public_receipts_must_be_safe(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        hf[0]["safe_public_artifact"] = False
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertTrue(
            any("must set safe_public_artifact true for public visibility" in error for error in errors)
        )

    def test_tinker_run_id_is_cross_checked_across_artifacts(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        checkpoints[0]["run_id"] = "other-run"
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run={"run_id": "tinker-run-42", "status": "finished"},
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertIn("run_id mismatch across checkpoint/hf metadata and tinker run_id", errors)

    def test_placeholder_hf_commits_are_rejected(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        hf[1]["revision"] = "pending"
        errors = launch_audit.validate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run=tinker,
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertTrue(
            any("hf_receipts[1] revision must be a 40-char hex commit" in error for error in errors)
        )

    def test_assert_launch_audit_blocks_when_artifacts_missing(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        with self.assertRaises(launch_audit.LaunchAuditError) as exc:
            launch_audit.assert_launch_audit(
                self.config,
                wandb_sync=wandb,
                tinker_run=tinker,
                checkpoint_json={"checkpoints": checkpoints},
                hf_receipts=hf[0:2],
            )
        self.assertIn("launch audit is blocked", str(exc.exception))
        self.assertIn("hf receipts missing stages", str(exc.exception))

    def test_missing_artifact_file_is_blocking(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            checkpoint_payload = {"checkpoints": checkpoints}
            report = launch_audit.generate_launch_audit(
                self.config,
                wandb_sync=(root / "missing-wandb.json"),
                tinker_run=tinker,
                checkpoint_json=checkpoint_payload,
                hf_receipts=hf,
            )
            self.assertEqual(report["status"], "BLOCKED")
            self.assertTrue(any("wandb sync metadata must point to an existing JSON file" in error for error in report["blockers"]))
            self.assertFalse(report["launchable"])

    def test_tinker_run_id_string_is_supported_if_artifacts_align(self) -> None:
        wandb, _, checkpoints, hf = self._valid_payloads()
        report = launch_audit.generate_launch_audit(
            self.config,
            wandb_sync=wandb,
            tinker_run="tinker-run-42",
            checkpoint_json={"checkpoints": checkpoints},
            hf_receipts=hf,
        )
        self.assertEqual(report["status"], "READY")
        self.assertEqual(report["tinker_run_id"], "tinker-run-42")

    def test_local_json_loader_is_supported(self) -> None:
        wandb, tinker, checkpoints, hf = self._valid_payloads()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wandb_path = root / "wandb.json"
            checkpoint_path = root / "checkpoint.json"
            hf_path = root / "hf.json"
            wandb_path.write_text(_json_text(wandb), encoding="utf-8")
            checkpoint_path.write_text(_json_text({"checkpoints": checkpoints}), encoding="utf-8")
            hf_path.write_text(_json_text(hf), encoding="utf-8")
            report = launch_audit.generate_launch_audit(
                self.config,
                wandb_sync=wandb_path,
                tinker_run=tinker,
                checkpoint_json=checkpoint_path,
                hf_receipts=hf_path,
            )
            self.assertEqual(report["status"], "READY")


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
