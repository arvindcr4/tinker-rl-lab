#!/usr/bin/env python3
"""Fail-closed tests for the xLAM first paid smoke config."""

from __future__ import annotations

import copy
import json
import unittest

from flagship import pavlov_xlam_smoke_config as config


class PavlovXlamSmokeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = config.generate_smoke_config()

    def test_generated_config_is_deterministic_and_hash_stable(self) -> None:
        first = config.generate_smoke_config()
        second = config.generate_smoke_config()
        self.assertEqual(first, second)
        rendered = json.dumps(first, sort_keys=True)
        self.assertEqual(rendered, json.dumps(second, sort_keys=True))
        self.assertEqual(
            first["config_signature"],
            config._sha256({key: first[key] for key in config.ROOT_KEYS if key != "config_signature"}),
        )

    def test_generated_config_matches_required_invariants(self) -> None:
        self.assertEqual(self.config["schema_version"], config.SCHEMA_VERSION)
        self.assertEqual(self.config["smoke_id"], config.SMOKE_ID)
        self.assertEqual(self.config["model"], config.MODEL_ID)
        self.assertEqual(self.config["model_revision"], config.MODEL_REVISION)
        self.assertEqual(self.config["xlam_revision"], config.XLAM_REVISION)
        self.assertEqual(self.config["seed"], config.SEED)
        self.assertEqual(self.config["steps"], config.STEPS)
        self.assertEqual(self.config["group"], config.GROUP)
        self.assertEqual(self.config["batch"], config.BATCH)
        self.assertEqual(self.config["rank"], config.RANK)
        self.assertEqual(self.config["learning_rate"], config.LEARNING_RATE)
        self.assertEqual(self.config["temperature"], config.TEMPERATURE)
        self.assertEqual(self.config["top_p"], config.TOP_P)
        self.assertEqual(self.config["max_prompt_tokens"], config.MAX_PROMPT_TOKENS)
        self.assertEqual(self.config["max_response_tokens"], config.MAX_RESPONSE_TOKENS)
        self.assertEqual(self.config["save_every_steps"], config.SAVE_EVERY_STEPS)
        self.assertEqual(self.config["run_order"], ["wandb_init", "tinker_client"])
        self.assertEqual(self.config["wandb"], {"mode": "online", "required_before_tinker": True})
        self.assertTrue(self.config["component_only"])
        self.assertFalse(self.config["primary_eval"])
        self.assertFalse(self.config["heldout"])
        self.assertFalse(self.config["portfolio_claim"])

    def test_generated_config_passes_validation(self) -> None:
        self.assertEqual(config.validate_smoke_config(self.config), [])

    def test_unknown_root_field_fails_closed(self) -> None:
        bad = dict(self.config)
        bad["unexpected"] = "x"
        errors = config.validate_smoke_config(bad)
        self.assertTrue(any("unexpected fields" in error for error in errors))

    def test_wrong_model_rejects(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["model"] = "gpt-4"
        errors = config.validate_smoke_config(bad)
        self.assertTrue(any("model must be Qwen/Qwen3.6-35B-A3B" in error for error in errors))

    def test_wrong_revisions_reject_pinned_requirements(self) -> None:
        bad_model_revision = copy.deepcopy(self.config)
        bad_model_revision["model_revision"] = "main"
        errors = config.validate_smoke_config(bad_model_revision)
        self.assertTrue(any("model_revision must match the pinned primary model revision" in error for error in errors))

        bad_xlam_revision = copy.deepcopy(self.config)
        bad_xlam_revision["xlam_revision"] = "b" * 40
        errors = config.validate_smoke_config(bad_xlam_revision)
        self.assertTrue(any("xlam_revision must match the pinned xLAM revision" in error for error in errors))

    def test_wrong_hyperparameters_reject(self) -> None:
        bad = copy.deepcopy(self.config)
        bad["steps"] = 11
        bad["rank"] = 64
        bad["learning_rate"] = 1e-5
        errors = config.validate_smoke_config(bad)
        self.assertTrue(any("steps must be exactly 10" in error for error in errors))
        self.assertTrue(any("rank must be exactly 32" in error for error in errors))
        self.assertTrue(any("learning_rate must be exactly 2e-05" in error for error in errors))

    def test_wandb_and_order_gate_rejects_noncompliant_execution(self) -> None:
        offline = copy.deepcopy(self.config)
        offline["wandb"]["mode"] = "offline"
        errors = config.validate_smoke_config(offline)
        self.assertTrue(any("wandb mode must be online" in error for error in errors))

        wrong_order = copy.deepcopy(self.config)
        wrong_order["run_order"] = ["tinker_client", "wandb_init"]
        errors = config.validate_smoke_config(wrong_order)
        self.assertTrue(any("run_order must be [\"wandb_init\", \"tinker_client\"]" in error for error in errors))

        missing_flag = copy.deepcopy(self.config)
        missing_flag["wandb"].pop("required_before_tinker")
        errors = config.validate_smoke_config(missing_flag)
        self.assertTrue(any("wandb must contain only mode and required_before_tinker" in error for error in errors))

    def test_checkpoint_policy_rejects_missing_stages_and_wrong_steps(self) -> None:
        bad_stages = copy.deepcopy(self.config)
        bad_stages["sampler_checkpoints"]["required_stages"] = ["initial", "final"]
        errors = config.validate_smoke_config(bad_stages)
        self.assertTrue(any("required_stages must be [initial, periodic, final]" in error for error in errors))

        bad_steps = copy.deepcopy(self.config)
        bad_steps["sampler_checkpoints"]["required_steps"] = [0, 10]
        errors = config.validate_smoke_config(bad_steps)
        self.assertTrue(any("required_steps must be [0, save_every_steps, steps]" in error for error in errors))

        bad_periodic = copy.deepcopy(self.config)
        bad_periodic["sampler_checkpoints"]["periodic_every_steps"] = [4]
        errors = config.validate_smoke_config(bad_periodic)
        self.assertTrue(any("periodic_every_steps must be [5]" in error for error in errors))

        bad_visibility = copy.deepcopy(self.config)
        bad_visibility["sampler_checkpoints"]["allowed_visibility"] = ["public"]
        errors = config.validate_smoke_config(bad_visibility)
        self.assertTrue(any("allowed_visibility must be exactly ['public', 'private']" in error for error in errors))

    def test_runtime_constraints_reject_network_credentials_paid_side_effects(self) -> None:
        bad_network = copy.deepcopy(self.config)
        bad_network["runtime_constraints"]["allow_network"] = True
        errors = config.validate_smoke_config(bad_network)
        self.assertTrue(any("runtime_constraints.allow_network must be false" in error for error in errors))

        bad_credentials = copy.deepcopy(self.config)
        bad_credentials["runtime_constraints"]["allow_credentials"] = True
        errors = config.validate_smoke_config(bad_credentials)
        self.assertTrue(any("runtime_constraints.allow_credentials must be false" in error for error in errors))

        bad_paid = copy.deepcopy(self.config)
        bad_paid["runtime_constraints"]["allow_paid_run"] = True
        errors = config.validate_smoke_config(bad_paid)
        self.assertTrue(any("runtime_constraints.allow_paid_run must be false" in error for error in errors))

    def test_budget_and_signature_checks(self) -> None:
        bad_budget = copy.deepcopy(self.config)
        bad_budget["budget"]["maximum_usd"] = 17.0
        bad_budget["budget"]["operational_cap_usd"] = 17.0
        bad_budget["budget"]["reservation_usd"] = 4.0
        errors = config.validate_smoke_config(bad_budget)
        self.assertTrue(any("budget.maximum_usd must preserve $18.00 hard cap" in error for error in errors))
        self.assertTrue(any("budget.operational_cap_usd must preserve $16.50 cap" in error for error in errors))
        self.assertTrue(any("budget.reservation_usd must be exactly $0.50" in error for error in errors))

        bad_signature = dict(self.config)
        bad_signature["config_signature"] = "bad"
        errors = config.validate_smoke_config(bad_signature)
        self.assertTrue(any("config_signature is invalid or missing" in error for error in errors))

    def test_claim_flags_restrict_scope_and_exposure(self) -> None:
        bad_component = copy.deepcopy(self.config)
        bad_component["component_only"] = False
        errors = config.validate_smoke_config(bad_component)
        self.assertTrue(any("component_only must be true" in error for error in errors))

        bad_eval = copy.deepcopy(self.config)
        bad_eval["primary_eval"] = True
        errors = config.validate_smoke_config(bad_eval)
        self.assertTrue(any("primary_eval must be false" in error for error in errors))

        bad_heldout = copy.deepcopy(self.config)
        bad_heldout["heldout"] = True
        errors = config.validate_smoke_config(bad_heldout)
        self.assertTrue(any("heldout must be false" in error for error in errors))

        bad_portfolio = copy.deepcopy(self.config)
        bad_portfolio["portfolio_claim"] = True
        errors = config.validate_smoke_config(bad_portfolio)
        self.assertTrue(any("portfolio_claim must be false" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
