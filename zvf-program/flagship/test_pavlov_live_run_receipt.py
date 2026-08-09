from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from flagship.pavlov_live_run_receipt import (
    SCHEMA_VERSION,
    RECEIPT_TYPE,
    is_valid_live_run_receipt,
    main,
    sha256_json,
    validate_live_run_receipt,
)
from flagship.pavlovs_domain_contract import load_contract


def valid_receipt() -> dict:
    revisions = ["1" * 40, "2" * 40, "3" * 40]
    checkpoints = [
        {
            "stage": stage,
            "repo_url": "https://huggingface.co/example/xlam-809",
            "revision": revision,
            "url": f"https://huggingface.co/example/xlam-809/commit/{revision}",
            "url_verified": True,
            "visibility": "public" if stage != "periodic" else "private",
            "safe_public_artifact": stage != "periodic",
            "data_license_safe": stage != "periodic",
            "quota_safe": stage != "periodic",
            "private_artifact_safe": stage == "periodic",
            "receipt_hash": "a" * 64,
        }
        for stage, revision in zip(("initial", "periodic", "final"), revisions)
    ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "status": "OBSERVED_COMPONENT_ONLY",
        "launchable": False,
        "allocation_allowed": False,
        "provenance_ready": True,
        "scientific_evidence_status": "not_established",
        "run": {
            "component": "xLAM",
            "paid": True,
            "status": "succeeded",
            "run_id": "xlam-run-809",
        },
        "model": {"model_id": "Qwen/Qwen3.6-35B-A3B", "revision": "a" * 40, "receipt_hash": "b" * 64},
        "dataset": {"dataset_id": "xlam-component-slice", "revision": "b" * 40, "receipt_hash": "c" * 64},
        "wandb": {
            "online": True,
            "entity": "example",
            "project": "pavlov",
            "group": "xlam-809",
            "run_id": "wandb-run-809",
            "run_url": "https://wandb.ai/example/pavlov/runs/wandb-run-809",
            "state": "finished",
            "success": True,
            "receipt_hash": "d" * 64,
        },
        "tinker": {
            "provider": "Tinker",
            "run_id": "tinker-run-809",
            "status": "succeeded",
            "cost_status": "observed",
            "receipt_hash": "e" * 64,
        },
        "sampler_checkpoints": checkpoints,
        "budget": {
            "currency": "USD",
            "authorized": True,
            "authorized_cap_usd": "18.00",
            "operational_cap_usd": "16.50",
            "safety_reserve_usd": "1.50",
            "authorization_id": "f" * 40,
            "authorization_hash": "0" * 64,
            "debits": [
                {
                    "debit_id": "1" * 40,
                    "amount_usd": "1.25",
                    "status": "settled",
                    "tinker_run_id": "tinker-run-809",
                    "receipt_hash": "2" * 64,
                }
            ],
            "total_debited_usd": "1.25",
            "remaining_usd": "16.75",
        },
        "evaluator_provenance": {
            "status": "verified",
            "evaluator_id": "xlam-evaluator",
            "revision": "c" * 40,
            "dataset_revision": "b" * 40,
            "split_manifest_hash": "d" * 64,
            "task_id_hash": "e" * 64,
            "container_digest": "sha256:" + "f" * 64,
            "verifier_hash": "0" * 64,
            "receipt_id": "3" * 40,
            "provenance_hash": "4" * 64,
        },
        "claims": {
            "xlam_component_only": True,
            "portfolio_evidence": False,
            "primary_eval_heldout": False,
            "held_out": False,
            "company_usefulness": False,
        },
        "evidence": {
            "scope": "xlam_component_only",
            "status": "observed",
            "portfolio_evidence": False,
            "primary_eval_heldout": False,
            "company_usefulness": False,
        },
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


class PavlovLiveRunReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_contract()

    def test_complete_paid_xlam_receipt_is_observed_component_only(self) -> None:
        receipt = valid_receipt()
        self.assertEqual(validate_live_run_receipt(receipt, self.contract), [])
        self.assertTrue(is_valid_live_run_receipt(receipt, self.contract))

    def test_missing_fields_are_enumerated_fail_closed(self) -> None:
        errors = validate_live_run_receipt({})
        self.assertTrue(errors)
        for field in ("claims", "run", "model", "dataset", "wandb", "tinker", "sampler_checkpoints", "budget", "evaluator_provenance"):
            self.assertTrue(any(error.startswith(field) for error in errors), field)

    def test_model_and_dataset_branches_are_not_immutable_revisions(self) -> None:
        receipt = valid_receipt()
        receipt["model"]["revision"] = "main"
        receipt["dataset"]["revision"] = "v1"
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("model.revision" in error for error in errors))
        self.assertTrue(any("dataset.revision" in error for error in errors))

    def test_wandb_requires_online_run_url_and_success(self) -> None:
        for mutation in (
            lambda value: value.update({"online": False}),
            lambda value: value.update({"run_url": "https://wandb.ai/example/pavlov/runs/other"}),
            lambda value: value.update({"state": "failed", "success": False}),
        ):
            receipt = valid_receipt()
            mutation(receipt["wandb"])
            errors = validate_live_run_receipt(receipt)
            self.assertTrue(any(error.startswith("wandb") for error in errors))

    def test_tinker_run_id_is_required(self) -> None:
        receipt = valid_receipt()
        del receipt["tinker"]["run_id"]
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("tinker.run_id" in error for error in errors))

    def test_checkpoints_require_all_stages_unique_commits_and_verified_urls(self) -> None:
        receipt = valid_receipt()
        del receipt["sampler_checkpoints"][2]
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("missing stages: final" in error for error in errors))

        receipt = valid_receipt()
        receipt["sampler_checkpoints"][1]["revision"] = receipt["sampler_checkpoints"][0]["revision"]
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("duplicate HF repo+commit identity" in error for error in errors))

        receipt = valid_receipt()
        receipt["sampler_checkpoints"][0]["url_verified"] = False
        receipt["sampler_checkpoints"][0]["url"] = "https://example.invalid/checkpoint"
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("sampler_checkpoints[0].url" in error for error in errors))
        self.assertTrue(any("sampler_checkpoints[0].url_verified" in error for error in errors))

    def test_budget_debits_are_exact_and_capped(self) -> None:
        receipt = valid_receipt()
        receipt["budget"]["total_debited_usd"] = "1.24"
        errors = validate_live_run_receipt(receipt, self.contract)
        self.assertTrue(any("budget.total_debited_usd" in error for error in errors))

        receipt = valid_receipt()
        receipt["budget"]["authorized_cap_usd"] = "18.01"
        errors = validate_live_run_receipt(receipt, self.contract)
        self.assertTrue(any("authorized_cap_usd" in error for error in errors))

        receipt = valid_receipt()
        receipt["budget"]["debits"][0]["status"] = "estimated"
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("budget.debits[0].status" in error for error in errors))

    def test_evaluator_provenance_and_primary_eval_claims_are_separate(self) -> None:
        receipt = valid_receipt()
        receipt["evaluator_provenance"]["split_manifest_hash"] = "status-complete"
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("split_manifest_hash" in error for error in errors))

        receipt = valid_receipt()
        receipt["claims"]["primary_eval_heldout"] = True
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("claims.primary_eval_heldout" in error for error in errors))

        receipt = valid_receipt()
        receipt["evidence"]["portfolio_evidence"] = True
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("evidence.portfolio_evidence" in error for error in errors))

    def test_required_receipt_hash_detects_mutation(self) -> None:
        receipt = valid_receipt()
        self.assertEqual(validate_live_run_receipt(receipt), [])
        receipt["run"]["run_id"] = "tampered"
        errors = validate_live_run_receipt(receipt)
        self.assertTrue(any("receipt_hash" in error for error in errors))

    def test_adversarial_typed_booleans_caps_urls_and_safety_fail_closed(self) -> None:
        receipt = valid_receipt()
        receipt["run"]["paid"] = 1
        self.assertTrue(any("run.paid" in error for error in validate_live_run_receipt(receipt)))

        receipt = valid_receipt()
        receipt["budget"]["operational_cap_usd"] = "16.51"
        self.assertTrue(any("operational_cap_usd" in error for error in validate_live_run_receipt(receipt)))

        receipt = valid_receipt()
        receipt["wandb"]["run_url"] = "https://evil.example/wandb-run-809"
        self.assertTrue(any("wandb.run_url" in error for error in validate_live_run_receipt(receipt)))

        receipt = valid_receipt()
        del receipt["sampler_checkpoints"][0]["receipt_hash"]
        self.assertTrue(any("sampler_checkpoints[0].receipt_hash" in error for error in validate_live_run_receipt(receipt)))

        receipt = valid_receipt()
        receipt["sampler_checkpoints"][0]["visibility"] = "private"
        receipt["sampler_checkpoints"][0]["safe_public_artifact"] = False
        receipt["sampler_checkpoints"][0]["private_artifact_safe"] = False
        self.assertTrue(any("private_artifact_safe" in error for error in validate_live_run_receipt(receipt)))

    def test_cli_returns_nonzero_for_invalid_and_zero_for_valid_local_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                invalid_code = main(["--receipt", str(path)])
            self.assertEqual(invalid_code, 1)
            self.assertIn("run:", output.getvalue())

            path.write_text(json.dumps(valid_receipt()), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                valid_code = main(["--receipt", str(path)])
            self.assertEqual(valid_code, 0)
            self.assertIn("component-only receipt", output.getvalue())


if __name__ == "__main__":
    unittest.main()
