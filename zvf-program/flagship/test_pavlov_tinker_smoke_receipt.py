"""Adversarial, offline tests for the Pavlov xLAM smoke receipt gate."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from flagship.pavlov_tinker_smoke_receipt import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_REVISION,
    ReceiptValidationError,
    SMOKE_SAVE_STEPS,
    XLAM_COMPONENT_SCOPE,
    canonical_pavlov_xlam_smoke_config,
    main,
    parse_completed_checkpoint,
    validate_completed_checkpoint,
)


def _receipt(index: int, step: int | str) -> dict[str, Any]:
    revision = f"smoke-{index}-{step}"
    repo_id = f"arvindcr4/pavlov-xlam-smoke-{index}-{step}"
    commit_sha = f"{index + 1:040x}"
    repo_url = f"https://huggingface.co/{repo_id}"
    return {
        "step": step,
        "repo_id": repo_id,
        "revision": revision,
        "commit_sha": commit_sha,
        "repo_url": repo_url,
        "revision_url": f"{repo_url}/tree/{revision}",
        "commit_url": f"{repo_url}/commit/{commit_sha}",
        "source_path": f"tinker://smoke-809/sampler/step-{step}",
        "hf_revision": revision,
        "hf_commit_sha": commit_sha,
        "hf_repo_url": repo_url,
        "hf_revision_url": f"{repo_url}/tree/{revision}",
    }


def _valid_payload() -> dict[str, Any]:
    trace = [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0]
    receipts = [_receipt(index, step) for index, step in enumerate(SMOKE_SAVE_STEPS)]
    run_id = "tinker-smoke-809:train:0"
    config = canonical_pavlov_xlam_smoke_config()
    metadata = {
        key: list(config[key])
        for key in (
            "training_suite_ids",
            "heldout_suite_ids",
            "primary_evaluation_suite_ids",
            "domain_tags",
            "declared_domains",
            "training_domain_union",
            "primary_evaluation_domain_union",
        )
    }
    urls = [receipt["revision_url"] for receipt in receipts]
    commits = [receipt["commit_sha"] for receipt in receipts]
    return {
        "status": "completed",
        "final_status": "success",
        "step": 10,
        "config": config,
        "campaign_metadata": metadata,
        "wandb_run_id": "wandb-smoke-809",
        "wandb": {"run_id": "wandb-smoke-809"},
        "tinker_run_id": run_id,
        "run_id": run_id,
        "checkpoint_receipts": receipts,
        "checkpoint_urls": urls,
        "checkpoint_commit_shas": commits,
        "cost": {
            "estimated_cost_usd": 1.25,
            "maximum_authorized_cost_usd": 18.0,
            "operational_cap_usd": 16.5,
            "safety_reserve_usd": 1.5,
            "prompt_tokens": 1000,
            "sample_tokens": 2000,
            "train_tokens": 3000,
            "actual_cost_usd": 1.0,
        },
        "scope": {
            "kind": XLAM_COMPONENT_SCOPE,
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "component": "strict_tool_call",
            "claim_boundary": "component_only",
        },
        "result": {
            "seed": 809,
            "run_id": run_id,
            "reward_trace": trace,
            "avg_first5": 0.5,
            "avg_last10": 0.5,
            "peak_reward": 1.0,
            "zero_loss_steps": 0,
            "zero_reward_steps": 2,
            "heldout_reward": 0.07,
            "checkpoint_urls": urls,
            "checkpoint_commit_shas": commits,
            "checkpoint_receipts": receipts,
            "campaign_metadata": metadata,
        },
    }


class PavlovTinkerSmokeReceiptTests(unittest.TestCase):
    def assert_rejected(self, payload: dict[str, Any], message: str = "") -> None:
        with self.assertRaises(ReceiptValidationError, msg=message):
            validate_completed_checkpoint(payload)

    def test_valid_receipt_is_accepted_and_normalized_offline(self) -> None:
        payload = _valid_payload()
        accepted = validate_completed_checkpoint(
            payload,
            expected_wandb_run_id="wandb-smoke-809",
            expected_tinker_run_id="tinker-smoke-809:train:0",
        )
        self.assertEqual(accepted["wandb_run_id"], "wandb-smoke-809")
        self.assertEqual(accepted["tinker_run_id"], "tinker-smoke-809:train:0")
        self.assertEqual(accepted["scope"], XLAM_COMPONENT_SCOPE)
        self.assertEqual(
            [receipt["step"] for receipt in accepted["checkpoint_receipts"]],
            [0, 5, 10, "final"],
        )
        self.assertEqual(len(accepted["config"]["training_suite_ids"]), 12)
        self.assertEqual(len(accepted["config"]["heldout_suite_ids"]), 6)
        self.assertEqual(len(accepted["config"]["primary_evaluation_suite_ids"]), 14)
        self.assertNotEqual(
            set(accepted["config"]["heldout_suite_ids"]),
            set(accepted["config"]["primary_evaluation_suite_ids"]),
        )
        self.assertEqual(len(accepted["config"]["declared_domains"]), 16)
        self.assertEqual(
            set(accepted["config"]["training_domain_union"]),
            set(accepted["config"]["declared_domains"]),
        )
        self.assertEqual(
            set(accepted["config"]["primary_evaluation_domain_union"]),
            set(accepted["config"]["declared_domains"]),
        )
        self.assertEqual(accepted["config"]["dataset_revision"], DATASET_REVISION)
        self.assertEqual(accepted["config"]["model_revision"], MODEL_REVISION)

    def test_file_parser_round_trip_is_standard_library_only(self) -> None:
        payload = _valid_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            accepted = parse_completed_checkpoint(path)
        self.assertEqual(accepted["tinker_run_id"], payload["tinker_run_id"])

    def test_duplicate_keys_and_trailing_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"status":"completed","status":"success"}')
            with self.assertRaises(ReceiptValidationError):
                parse_completed_checkpoint(duplicate)

            trailing = Path(directory) / "trailing.json"
            trailing.write_text('{"status":"completed"}{"status":"success"}')
            with self.assertRaises(ReceiptValidationError):
                parse_completed_checkpoint(trailing)

    def test_config_is_immutable_and_exact(self) -> None:
        payload = _valid_payload()
        payload["config"]["steps"] = 11
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["config"]["unexpected"] = "must-not-be-ignored"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["config"]["dataset_revision"] = "mutable"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["config"]["domain_tags"].append("unexpected-domain-marker")
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["wandb_config"] = copy.deepcopy(payload["config"])
        payload["wandb_config"]["seed"] = 808
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["campaign_metadata"]["training_suite_ids"].pop()
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["campaign_metadata"]["domain_tags"] = ["xLAM-only"]
        self.assert_rejected(payload)

    def test_custom_expected_config_is_supported_but_still_exact(self) -> None:
        payload = _valid_payload()
        expected = canonical_pavlov_xlam_smoke_config(
            checkpoint_dir="/tmp/immutable-checkpoints"
        )
        payload["config"] = copy.deepcopy(expected)
        accepted = validate_completed_checkpoint(payload, expected_config=expected)
        self.assertEqual(accepted["config"]["checkpoint_dir"], expected["checkpoint_dir"])
        self.assert_rejected(payload)  # the default fingerprint differs

    def test_run_ids_are_required_and_aliases_must_agree(self) -> None:
        payload = _valid_payload()
        del payload["wandb_run_id"]
        del payload["wandb"]
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["wandb"]["run_id"] = "different-wandb-run"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["run_id"] = "different-tinker-run"
        self.assert_rejected(payload)

        payload = _valid_payload()
        self.assert_rejected(payload | {"tinker_run_id": "other-run"})

        payload = _valid_payload()
        payload["run_id"] = "different-tinker-run"
        self.assert_rejected(payload)
        with self.assertRaises(ReceiptValidationError):
            validate_completed_checkpoint(
                _valid_payload(), expected_wandb_run_id="wrong-wandb"
            )

    def test_all_four_checkpoint_classes_require_immutable_hf_receipts(self) -> None:
        payload = _valid_payload()
        payload["checkpoint_receipts"] = payload["checkpoint_receipts"][:-1]
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["checkpoint_receipts"][1]["step"] = 10
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["checkpoint_receipts"][2]["commit_sha"] = "not-a-commit"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["checkpoint_receipts"][0]["revision_url"] = "https://huggingface.co/owner/repo"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["checkpoint_receipts"][3]["commit_sha"] = payload["checkpoint_receipts"][0]["commit_sha"]
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["checkpoint_receipts"][0]["hf_revision"] = "different-revision"
        self.assert_rejected(payload)

    def test_checkpoint_arrays_must_match_every_receipt(self) -> None:
        payload = _valid_payload()
        payload["checkpoint_urls"][1] = payload["checkpoint_urls"][0]
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["checkpoint_commit_shas"].pop()
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["checkpoint_receipts"][0]["commit_sha"] = "0" * 40
        self.assert_rejected(payload)

    def test_reward_trace_and_aggregates_are_strict(self) -> None:
        payload = _valid_payload()
        payload["result"]["reward_trace"] = [0.0] * 9
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["reward_trace"][3] = 1.1
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["reward_trace"][3] = math.nan
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["avg_last10"] = 0.51
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["seed"] = 808
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["result"]["zero_reward_steps"] = True
        self.assert_rejected(payload)

        payload = _valid_payload()
        del payload["result"]["heldout_reward"]
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["reward_trace"] = list(payload["result"]["reward_trace"])
        payload["reward_trace"][0] = 1.0
        self.assert_rejected(payload)

    def test_cost_contract_is_strict_and_fail_closed(self) -> None:
        for key in (
            "estimated_cost_usd",
            "maximum_authorized_cost_usd",
            "operational_cap_usd",
            "safety_reserve_usd",
            "prompt_tokens",
            "sample_tokens",
            "train_tokens",
        ):
            payload = _valid_payload()
            del payload["cost"][key]
            self.assert_rejected(payload, key)

        payload = _valid_payload()
        payload["cost"]["estimated_cost_usd"] = 17.0
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["cost"]["maximum_authorized_cost_usd"] = 17.0
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["cost"]["prompt_tokens"] = True
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["cost"]["sample_tokens"] = -1
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["cost"]["train_tokens"] = 0
        payload["cost"]["prompt_tokens"] = 0
        payload["cost"]["sample_tokens"] = 0
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["cost"]["actual_cost_usd"] = float("inf")
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["cost"]["estimated_cost_usd"] = 10**10000
        self.assert_rejected(payload)

    def test_scope_is_component_only_and_cannot_be_broadened(self) -> None:
        payload = _valid_payload()
        payload["scope"]["kind"] = "full_campaign"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["scope"]["dataset"] = "another/dataset"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["scope"]["dataset_revision"] = "another-revision"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["scope"]["companies"] = ["company-1"]
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["campaign_claim"] = "all-companies"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["scope"] = "full_campaign"
        self.assert_rejected(payload)

        payload = _valid_payload()
        payload["scope"] = XLAM_COMPONENT_SCOPE
        self.assertEqual(validate_completed_checkpoint(payload)["scope"], XLAM_COMPONENT_SCOPE)

    def test_cli_accepts_valid_receipt_and_rejects_without_echoing_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completed.json"
            path.write_text(json.dumps(_valid_payload()), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            str(path),
                            "--expected-wandb-run-id",
                            "wandb-smoke-809",
                            "--expected-tinker-run-id",
                            "tinker-smoke-809:train:0",
                        ]
                    ),
                    0,
                )
            self.assertIn('"status": "accepted"', output.getvalue())
            self.assertNotIn("estimated_cost_usd", output.getvalue())
            self.assertNotIn("prompt_tokens", output.getvalue())

            payload = _valid_payload()
            payload["scope"]["companies"] = ["company-marker"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(main([str(path)]), 1)
            self.assertIn("REJECTED:", error.getvalue())
            self.assertNotIn("company-marker", error.getvalue())


if __name__ == "__main__":
    unittest.main()
