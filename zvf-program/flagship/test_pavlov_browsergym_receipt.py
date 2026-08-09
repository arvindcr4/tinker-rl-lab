"""Offline tests for the exact T3 BrowserGym result receipt validator."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

try:
    from flagship import pavlov_browsergym_receipt as receipt
except ModuleNotFoundError:  # Direct execution from the flagship directory.
    import pavlov_browsergym_receipt as receipt


def _rehash(value: dict) -> dict:
    value["receipt_hash"] = receipt.sha256_json(
        {key: item for key, item in value.items() if key != "receipt_hash"}
    )
    return value


def _retrace(value: dict) -> dict:
    value["trace_hash"] = receipt.sha256_json(
        {key: item for key, item in value.items() if key != "trace_hash"}
    )
    return value


def _structural_observed_fixture() -> dict:
    """Create safe local metadata for validator tests; this is not live evidence."""

    value = copy.deepcopy(receipt.offline_result_fixture())
    value["receipt_kind"] = receipt.RECEIPT_KIND_OBSERVED
    value["result_status"] = "OBSERVED_T3_RESULT"
    value["evidence_status"] = receipt.EVIDENCE_STATUS_OBSERVED
    episode = value["episode"]
    episode["status"] = receipt.EPISODE_STATUS_OBSERVED
    episode["evidence_status"] = receipt.EPISODE_EVIDENCE_STATUS_OBSERVED
    episode["claim_boundary"] = "T3_EPISODE_RECEIPT_ONLY"
    _retrace(episode)
    value["episode_hash"] = episode["trace_hash"]
    value["native_verifier"] = {
        "name": receipt.NATIVE_VERIFIER_NAME,
        "revision": receipt.NATIVE_VERIFIER_REVISION,
        "source": receipt.NATIVE_VERIFIER_SOURCE,
        "checked": True,
        "success": bool(episode["terminal"]["task_success"]),
        "task_id": value["task_id"],
        "episode_hash": episode["trace_hash"],
        "final_state_hash": episode["terminal"]["final_state_hash"],
        "action_count": len(episode["actions"]),
    }
    value["evidence"] = {
        "wandb": {
            "observed": True,
            "run_id": "run_t3_20260809",
            "url": "https://wandb.ai/example/pavlov/runs/run_t3_20260809",
            "project": "pavlov-t3",
            "metrics": {
                "train/browser_success_rate": 1.0,
                "train/browser_reward_mean": 1.0,
                "train/browser_action_count_mean": 1.0,
            },
        },
        "tinker": {
            "observed": True,
            "run_id": "tinker_t3_20260809",
            "config_hash": "a" * 64,
            "sampler_checkpoint": "runs/t3/sampler-step-0001",
            "final_checkpoint": "runs/t3/final-step-0001",
        },
        "hf": {
            "observed": True,
            "repo_id": "example/pavlov-t3",
            "revision": "b" * 40,
            "checkpoint": "checkpoints/step-0001",
            "checkpoint_hash": "c" * 64,
            "exported": True,
        },
    }
    value["cost"] = {
        "currency": "USD",
        "prompt_usd": 0.01,
        "sampling_usd": 0.02,
        "training_usd": 0.03,
        "other_usd": 0.0,
        "total_usd": 0.06,
        "charged_usd": 0.06,
        "cap_usd": 16.50,
        "charged": True,
        "within_cap": True,
    }
    return _rehash(value)


class BrowserGymReceiptTests(unittest.TestCase):
    def test_offline_fixture_is_valid_schema_only(self) -> None:
        value = receipt.offline_result_fixture()
        result = receipt.validate_receipt(value)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(value["suite_id"], "browsergym_train")
        self.assertEqual(value["e6_suite_id"], "webbench_eval")
        self.assertEqual(value["evidence_status"], "OFFLINE_SYNTHETIC_RECEIPT")
        self.assertFalse(value["portfolio_evidence"])
        self.assertFalse(value["main_track_claim_allowed"])
        self.assertFalse(result.metrics["all_external_evidence_observed"])
        self.assertFalse(result.metrics["native_verifier_checked_success"])

    def test_observed_shape_requires_native_and_all_external_evidence(self) -> None:
        value = _structural_observed_fixture()
        result = receipt.validate_receipt(value)
        self.assertTrue(result.ok, result.errors)
        self.assertTrue(result.metrics["all_external_evidence_observed"])
        self.assertTrue(result.metrics["native_verifier_checked_success"])
        self.assertTrue(result.metrics["cost_within_cap"])
        self.assertFalse(result.metrics["e6_substitute"])

    def test_pinned_revisions_and_task_identity_are_checked(self) -> None:
        value = receipt.offline_result_fixture()
        value["dataset_revision"] = "miniwob-plusplus@unpinned"
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("dataset_revision" in error for error in result.errors))

        value = receipt.offline_result_fixture()
        value["episode"]["task"]["task_id"] = "t3-browsergym-tampered"
        _retrace(value["episode"])
        value["episode_hash"] = value["episode"]["trace_hash"]
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("task_id" in error for error in result.errors))

    def test_episode_trace_hash_action_hash_and_state_hash_fail_closed(self) -> None:
        value = receipt.offline_result_fixture()
        value["episode"]["observations"][1]["state_hash"] = "d" * 64
        _retrace(value["episode"])
        value["episode_hash"] = value["episode"]["trace_hash"]
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("state_hash" in error or "state" in error for error in result.errors))

        value = receipt.offline_result_fixture()
        value["episode"]["actions"][0]["next_state_hash"] = "e" * 64
        _retrace(value["episode"])
        value["episode_hash"] = value["episode"]["trace_hash"]
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("next state" in error for error in result.errors))

        value = receipt.offline_result_fixture()
        value["episode"]["trace_hash"] = "f" * 64
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("trace_hash" in error for error in result.errors))

    def test_native_success_must_be_checked_for_observed_result(self) -> None:
        value = _structural_observed_fixture()
        value["native_verifier"]["checked"] = False
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("checked native" in error for error in result.errors))

        value = _structural_observed_fixture()
        value["native_verifier"]["source"] = "offline_fixture"
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("native_verifier source" in error for error in result.errors))

    def test_wandb_tinker_hf_evidence_is_required_and_syntactic_only(self) -> None:
        value = _structural_observed_fixture()
        value["evidence"]["hf"]["observed"] = False
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("missing hf evidence" in error for error in result.errors))

        value = _structural_observed_fixture()
        value["evidence"]["hf"]["revision"] = "main"
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("40-hex revision" in error for error in result.errors))

        value = _structural_observed_fixture()
        value["evidence"]["wandb"]["url"] = "https://wandb.ai/example/project?token=raw"
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(
            any("credential-like" in error or "secret-like" in error for error in result.errors)
        )

    def test_cost_is_reconciled_and_capped(self) -> None:
        value = _structural_observed_fixture()
        value["cost"]["total_usd"] = 0.99
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("component sum" in error for error in result.errors))

        value = _structural_observed_fixture()
        value["cost"]["charged_usd"] = 17.0
        value["cost"]["within_cap"] = False
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("cap" in error for error in result.errors))

    def test_e6_and_portfolio_boundaries_are_explicit(self) -> None:
        value = receipt.offline_result_fixture()
        value["suite_id"] = receipt.E6_SUITE_ID
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("suite_id" in error for error in result.errors))

        value = receipt.offline_result_fixture()
        value["e6_substitute"] = True
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("substitute" in error for error in result.errors))

        value = receipt.offline_result_fixture()
        value["portfolio_evidence"] = True
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("portfolio" in error for error in result.errors))

    def test_secret_material_is_rejected_without_redaction(self) -> None:
        value = receipt.offline_result_fixture()
        value["evidence"]["wandb"]["api_key"] = "should-not-be-present"
        _rehash(value)
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("secret-like field" in error for error in result.errors))

    def test_malformed_non_json_payload_fails_closed_without_throwing(self) -> None:
        value = receipt.offline_result_fixture()
        value["episode"]["terminal"]["non_json"] = object()
        value["receipt_hash"] = "0" * 64
        result = receipt.validate_receipt(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("cannot be hashed" in error for error in result.errors))

    def test_receipt_and_episode_hashes_are_deterministic(self) -> None:
        first = receipt.offline_result_fixture()
        second = receipt.offline_result_fixture()
        self.assertEqual(first, second)
        self.assertEqual(
            first["receipt_hash"],
            receipt.sha256_json(
                {key: item for key, item in first.items() if key != "receipt_hash"}
            ),
        )
        self.assertEqual(first["episode_hash"], first["episode"]["trace_hash"])

    def test_module_has_no_live_runtime_or_network_imports(self) -> None:
        path = Path(receipt.__file__)
        tree = ast.parse(path.read_text(), filename=str(path))
        forbidden = {"browsergym", "playwright", "tinker", "wandb", "requests", "httpx"}
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertTrue(forbidden.isdisjoint(imported), imported & forbidden)
        source = path.read_text()
        self.assertNotIn("os.environ", source)
        self.assertNotIn("subprocess", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
