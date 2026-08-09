"""Offline tests for the fail-closed T3 BrowserGym pilot config validator."""

from __future__ import annotations

import copy
import unittest

try:
    from flagship import pavlov_browsergym_launch_config as config
except ModuleNotFoundError:  # Direct execution from the flagship directory.
    import pavlov_browsergym_launch_config as config


def _valid_config() -> dict:
    return copy.deepcopy(config.build_offline_smoke_config())


class BrowserGymLaunchConfigTests(unittest.TestCase):
    def test_offline_config_is_schema_valid_but_not_paid_authorized(self) -> None:
        value = _valid_config()
        result = config.validate_t3_pilot_config(value)
        self.assertTrue(result.ok, result.errors)
        self.assertFalse(result.paid_launch_authorized)
        self.assertFalse(value["paid_launch_allowed"])
        self.assertFalse(value["receipt_gate"]["receipt_attached"])
        self.assertFalse(value["receipt_gate"]["receipts_verified"])

    def test_exact_t3_pins_and_task_manifest(self) -> None:
        value = _valid_config()
        self.assertEqual(value["suite_id"], "browsergym_train")
        self.assertEqual(value["e6_suite_id"], "webbench_eval")
        self.assertEqual(value["dataset_revision"], config.DATASET_REVISION)
        self.assertEqual(value["environment_revision"], config.ENVIRONMENT_REVISION)
        self.assertEqual(
            [task["env_id"] for task in value["task_manifest"]],
            list(config.PILOT_ENV_IDS),
        )
        self.assertEqual(
            value["split_manifest_hash"], config.task_manifest_hash(value["task_manifest"])
        )
        self.assertEqual(len(value["task_manifest"]), 3)
        for task in value["task_manifest"]:
            self.assertEqual(len(task["task_id_hash"]), 64)

    def test_task_manifest_tampering_and_revision_drift_fail_closed(self) -> None:
        value = _valid_config()
        value["task_manifest"][0]["env_id"] = "browsergym/webarena.10"
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("task_manifest" in error for error in result.errors))

        value = _valid_config()
        value["environment_revision"] = "browsergym-miniwob==latest"
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("environment_revision" in error for error in result.errors))

        value = _valid_config()
        value["split_manifest_hash"] = "0" * 64
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("split_manifest_hash" in error for error in result.errors))

    def test_stateful_action_artifact_telemetry_is_required(self) -> None:
        value = _valid_config()
        telemetry = value["telemetry"]
        for field in (
            "per_step_observation_required",
            "per_step_action_required",
            "state_hash_required",
            "action_hash_required",
            "artifact_digest_required",
            "terminal_task_success_required",
        ):
            telemetry[field] = False
        telemetry["artifact_names"] = ["browser_state"]
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("telemetry" in error for error in result.errors))

    def test_wandb_requirements_are_exact_and_online_before_tinker(self) -> None:
        value = _valid_config()
        self.assertTrue(value["wandb"]["online_required"])
        self.assertTrue(value["wandb"]["initialize_before_tinker"])
        self.assertIn("train/browser_success_rate", value["wandb"]["required_metric_keys"])
        self.assertIn("eval/browser_success_rate", value["wandb"]["required_metric_keys"])
        self.assertIn("state_hash", value["wandb"]["required_sample_fields"])
        value["wandb"]["required_metric_keys"].remove("train/browser_reward_mean")
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("required_metric_keys" in error for error in result.errors))

    def test_tinker_and_hf_checkpoint_requirements_are_not_optional(self) -> None:
        value = _valid_config()
        value["tinker"]["final_sampler_path_required"] = False
        value["hf"]["final_sampler_export_required"] = False
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("final_sampler" in error for error in result.errors))
        self.assertTrue(any("final_sampler_export" in error for error in result.errors))

    def test_receipt_gate_is_tied_to_result_schema_and_stays_closed(self) -> None:
        value = _valid_config()
        self.assertEqual(
            value["receipt_gate"]["receipt_schema_version"],
            "pavlov-browsergym-t3-result-receipt-v1",
        )
        value["receipt_gate"]["receipt_attached"] = True
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("before receipts" in error for error in result.errors))

        value = _valid_config()
        value["receipt_gate"]["receipt_schema_version"] = "wrong-schema"
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("receipt schema" in error for error in result.errors))

    def test_smoke_cost_matches_roadmap_and_stays_under_sixty_cents(self) -> None:
        estimate = config.estimate_smoke_cost()
        self.assertEqual(estimate["updates"], 10)
        self.assertEqual(estimate["batch_size"], 2)
        self.assertEqual(estimate["group_size"], 2)
        self.assertEqual(estimate["horizon"], 8)
        self.assertEqual(estimate["sequence_count"], 320)
        self.assertAlmostEqual(estimate["nominal_usd"], 0.27983872, places=8)
        self.assertAlmostEqual(estimate["conservative_envelope_usd"], 0.55967744, places=8)
        self.assertLessEqual(estimate["conservative_envelope_usd"], 0.60)
        self.assertFalse(estimate["paid_launch"])

        value = _valid_config()
        value["cost"]["conservative_envelope_usd"] = 0.61
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("cost" in error or "envelope" in error for error in result.errors))

    def test_e6_and_portfolio_claims_fail_closed(self) -> None:
        value = _valid_config()
        value["e6_substitute"] = True
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("substitute" in error for error in result.errors))

        value = _valid_config()
        value["portfolio_evidence"] = True
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("portfolio" in error for error in result.errors))

    def test_secret_like_config_fields_are_rejected(self) -> None:
        value = _valid_config()
        value["wandb"]["api_key"] = "must-not-be-configured"
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertTrue(any("secret-like field" in error for error in result.errors))

    def test_unknown_or_missing_fields_do_not_open_paid_gate(self) -> None:
        value = _valid_config()
        del value["hf"]
        value["paid_launch_allowed"] = True
        result = config.validate_t3_pilot_config(value)
        self.assertFalse(result.ok)
        self.assertFalse(result.paid_launch_authorized)
        self.assertTrue(any("missing required config field: hf" in error for error in result.errors))
        self.assertTrue(any("paid launch" in error for error in result.errors))

    def test_config_is_deterministic_and_offline_cli_is_safe(self) -> None:
        first = config.build_offline_smoke_config()
        second = config.build_offline_smoke_config()
        self.assertEqual(first, second)
        self.assertEqual(
            config.task_manifest_hash(first["task_manifest"]), first["split_manifest_hash"]
        )
        self.assertFalse(first["receipt_gate"]["receipts_verified"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
