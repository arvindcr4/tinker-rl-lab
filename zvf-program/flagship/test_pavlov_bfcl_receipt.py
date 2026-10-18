from __future__ import annotations

import copy
import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import pavlov_bfcl_receipt as validator
else:
    try:
        from . import pavlov_bfcl_receipt as validator
    except ImportError:
        _TEST_DIR = Path(__file__).resolve().parent
        if str(_TEST_DIR) not in sys.path:
            sys.path.insert(0, str(_TEST_DIR))
        validator = import_module("pavlov_bfcl_receipt")


def _valid_wandb() -> dict[str, Any]:
    return {
        "online": True,
        "entity": "example-org",
        "project": "bfcl-receipts",
        "group": "receipt-closure",
        "run_id": "run-id-abc123",
        "run_url": "https://wandb.ai/example-org/bfcl-receipts/runs/run-id-abc123",
    }


def _valid_tinker() -> dict[str, Any]:
    return {"run_id": "tinker-run-001", "cost_status": "authorized"}


def _valid_checkpoint(revision: str, stage: str, visibility: str = "public") -> dict[str, Any]:
    return {
        "repo_url": "https://huggingface.co/example-org/bfcl-runner",
        "revision": revision,
        "url": f"https://huggingface.co/example-org/bfcl-runner/commit/{revision}",
        "stage": stage,
        "safe_public_artifact": True,
        "visibility": visibility,
    }


def _valid_manifest() -> dict[str, Any]:
    return {
        "suite_id": "bfcl_train",
        "category": "tool_use",
        "dataset": {
            "revision": "a" * 40,
            "source": "https://gorilla.cs.berkeley.edu/leaderboard.html",
        },
        "adapter_manifest_digest": "a" * 64,
        "scope": {"is_portfolio": False, "is_heldout": False},
        "per_example": [
            {"id": "0" * 64, "category": "tool_use", "verdict": "pass"},
            {"id": "1" * 64, "category": "tool_use", "verdict": False},
            {"id": "2" * 64, "category": "tool_use", "verdict": "error"},
        ],
        "wandb_run_identity": _valid_wandb(),
        "tinker_run_identity": _valid_tinker(),
        "hf_checkpoints": [
            _valid_checkpoint("a" * 40, "initial"),
            _valid_checkpoint("b" * 40, "periodic"),
            _valid_checkpoint("c" * 40, "final"),
        ],
        "costs": {"status": "authorized", "total_usd": 18.0},
    }


class BFCLReceiptBoundaryTests(unittest.TestCase):
    def test_valid_receipt_builds_ready_without_paid_launch(self) -> None:
        manifest = _valid_manifest()
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "READY"
        assert report["receipt_ready"]
        assert not report["paid_launch_allowed"]
        assert not report["launch"]["allowed"]
        assert report["suite_id"] == "bfcl_train"
        assert report["per_example_count"] == 3
        assert report["per_example_category"] == "tool_use"
        assert len(report["per_example_verdicts"]) == 3

    def test_non_bfcl_train_is_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["suite_id"] = "primary_eval"
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "BLOCKED"
        assert not report["receipt_ready"]
        assert "suite_id must be 'bfcl_train'" in " ".join(report["blockers"])

    def test_adapter_manifest_digest_and_dataset_revision_must_be_pinned(self) -> None:
        manifest = _valid_manifest()
        manifest["adapter_manifest_digest"] = "latest"
        digest_blocked = validator.build_receipt_record(manifest)
        assert not digest_blocked["receipt_ready"]
        assert any("adapter_manifest_digest" in blocker for blocker in digest_blocked["blockers"])

        manifest = _valid_manifest()
        manifest["dataset"]["revision"] = "b" * 39
        revision_blocked = validator.build_receipt_record(manifest)
        assert not revision_blocked["receipt_ready"]
        assert any("dataset.revision" in blocker for blocker in revision_blocked["blockers"])

    def test_rejects_non_portfolio_and_non_heldout_scope(self) -> None:
        manifest = _valid_manifest()
        manifest["scope"]["is_portfolio"] = True
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "BLOCKED"
        assert any("scope.is_portfolio" in blocker for blocker in report["blockers"])

        manifest = _valid_manifest()
        manifest["scope"]["is_heldout"] = True
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "BLOCKED"
        assert any("scope.is_heldout" in blocker for blocker in report["blockers"])

    def test_strict_examples_require_sorted_ids_unique_category_and_verdict(self) -> None:
        manifest = _valid_manifest()
        manifest["per_example"] = [
            {"id": "1" * 64, "category": "tool_use", "verdict": "pass"},
            {"id": "0" * 64, "category": "tool_use", "verdict": "pass"},
        ]
        unsorted = validator.build_receipt_record(manifest)
        assert unsorted["status"] == "BLOCKED"
        assert any("sorted deterministically" in blocker for blocker in unsorted["blockers"])

        manifest = _valid_manifest()
        manifest["per_example"][1]["category"] = "calculator"
        mismatch = validator.build_receipt_record(manifest)
        assert mismatch["status"] == "BLOCKED"
        assert any("category must be 'tool_use'" in blocker for blocker in mismatch["blockers"])

        manifest = _valid_manifest()
        manifest["per_example"][1]["verdict"] = "unknown"
        verdict_bad = validator.build_receipt_record(manifest)
        assert verdict_bad["status"] == "BLOCKED"
        assert any(
            "verdict must be pass|fail|error" in blocker for blocker in verdict_bad["blockers"]
        )

    def test_rejects_duplicated_example_ids(self) -> None:
        manifest = _valid_manifest()
        manifest["per_example"][1]["id"] = manifest["per_example"][0]["id"]
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "BLOCKED"
        assert any("duplicate ids" in blocker for blocker in report["blockers"])

    def test_rejects_synthetic_and_glaive_markers_in_receipts(self) -> None:
        manifest = _valid_manifest()
        manifest["wandb_run_identity"]["run_url"] = (
            "https://wandb.ai/glaiveai/glaive-function-calling-v2/run"
        )
        blocked = validator.build_receipt_record(manifest)
        assert blocked["status"] == "BLOCKED"
        assert any("synthetic or Glaive markers" in blocker for blocker in blocked["blockers"])

    def test_wandb_tinker_and_hf_checkpoint_evidence_is_required(self) -> None:
        manifest = _valid_manifest()
        manifest["wandb_run_identity"]["online"] = False
        wandb_block = validator.build_receipt_record(manifest)
        assert wandb_block["status"] == "BLOCKED"
        assert any(
            "wandb_run_identity.online is missing" in blocker
            or "wandb_run_identity.online must be true" in blocker
            for blocker in wandb_block["blockers"]
        )

        manifest = _valid_manifest()
        manifest["tinker_run_identity"]["cost_status"] = "pending"
        tinker_block = validator.build_receipt_record(manifest)
        assert tinker_block["status"] == "BLOCKED"
        assert any(
            "tinker_run_identity.cost_status" in blocker for blocker in tinker_block["blockers"]
        )

        manifest = _valid_manifest()
        manifest["hf_checkpoints"] = manifest["hf_checkpoints"][:2]
        hf_block = validator.build_receipt_record(manifest)
        assert hf_block["status"] == "BLOCKED"
        assert any(
            "hf_checkpoints missing required stage" in blocker for blocker in hf_block["blockers"]
        )

        manifest = _valid_manifest()
        manifest["hf_checkpoints"][1]["safe_public_artifact"] = False
        hf_visibility = validator.build_receipt_record(manifest)
        assert hf_visibility["status"] == "BLOCKED"
        assert any(
            "safe_public_artifact must be true" in blocker for blocker in hf_visibility["blockers"]
        )

    def test_costs_must_be_non_negative_number(self) -> None:
        manifest = _valid_manifest()
        manifest["costs"]["total_usd"] = -1
        blocked = validator.build_receipt_record(manifest)
        assert blocked["status"] == "BLOCKED"
        assert any(
            "costs.total_usd cannot be negative" in blocker for blocker in blocked["blockers"]
        )

    def test_network_and_credentials_are_blocked(self) -> None:
        manifest = _valid_manifest()
        manifest["requires_network"] = True
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "BLOCKED"
        assert any(
            "dataset download/network is disallowed" in blocker for blocker in report["blockers"]
        )

        manifest = _valid_manifest()
        manifest["credential_ref"] = "secret-token"
        report = validator.build_receipt_record(manifest)
        assert report["status"] == "BLOCKED"
        assert any("credential_ref is not allowed" in blocker for blocker in report["blockers"])

    def test_digests_are_deterministic(self) -> None:
        first = validator.build_receipt_record(_valid_manifest())
        second = validator.build_receipt_record(copy.deepcopy(_valid_manifest()))
        assert first["per_example_digest"] == second["per_example_digest"]
        assert first["receipt_identity_digest"] == second["receipt_identity_digest"]
        assert first["status"] == "READY"

    def test_main_returns_result_status(self) -> None:
        manifest = _valid_manifest()
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / "manifest.json"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            assert validator.main(["--manifest", str(temporary)]) == 0
            manifest["suite_id"] = "wrong_suite"
            temporary.write_text(json.dumps(manifest), encoding="utf-8")
            assert validator.main(["--manifest", str(temporary)]) == 1


if __name__ == "__main__":
    unittest.main()
