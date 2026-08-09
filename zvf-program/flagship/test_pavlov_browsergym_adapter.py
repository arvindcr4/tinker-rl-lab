"""Offline tests for the fail-closed Pavlov T3 BrowserGym boundary."""

from __future__ import annotations

import unittest

from flagship import pavlov_browsergym_adapter as adapter


class BrowserGymAdapterTests(unittest.TestCase):
    def test_pinned_identity_is_t3_and_not_e6(self) -> None:
        self.assertEqual(adapter.SUITE_ID, "browsergym_train")
        self.assertEqual(adapter.SUITE_ROLE, "train")
        self.assertEqual(adapter.E6_SUITE_ID, "webbench_eval")
        self.assertNotEqual(adapter.SUITE_ID, adapter.E6_SUITE_ID)
        self.assertIn("miniwob-plusplus@7fd85d71a4b60325c6585396ec4f48377d049838", adapter.PINNED_DATASET_REVISION)
        self.assertEqual(adapter.PINNED_ENVIRONMENT_REVISION, "browsergym-miniwob==0.14.3")

    def test_task_id_and_hash_are_deterministic(self) -> None:
        initial = {"url": "about:blank", "open_pages": 1, "clicked": False}
        first = adapter.TaskSpec(
            env_id="browsergym/miniwob.click-button",
            seed=7,
            goal="Click Continue.",
            initial_observation=initial,
        )
        second = adapter.TaskSpec(
            env_id="browsergym/miniwob.click-button",
            seed=7,
            goal="Click Continue.",
            initial_observation={"clicked": False, "open_pages": 1, "url": "about:blank"},
        )
        self.assertEqual(first.task_id, second.task_id)
        self.assertEqual(first.task_id_hash, second.task_id_hash)
        self.assertEqual(
            first.task_id,
            adapter.deterministic_task_id(
                env_id=first.env_id,
                seed=first.seed,
                goal=first.goal,
                initial_observation=first.initial_observation,
            ),
        )
        self.assertEqual(len(first.task_id_hash), 64)

    def test_task_tampering_is_rejected(self) -> None:
        task = adapter.TaskSpec(
            env_id="browsergym/miniwob.click-button",
            seed=1,
            goal="Click Continue.",
            initial_observation={"url": "about:blank"},
        )
        with self.assertRaises(adapter.AdapterSchemaError):
            adapter.TaskSpec(
                env_id=task.env_id,
                seed=task.seed,
                goal=task.goal,
                initial_observation=task.initial_observation,
                task_id="t3-browsergym-000000000000000000000000",
            )
        with self.assertRaises(adapter.AdapterSchemaError):
            adapter.TaskSpec(
                env_id=task.env_id,
                seed=task.seed,
                goal=task.goal,
                initial_observation=task.initial_observation,
                dataset_revision="un pinned",
            )

    def test_state_action_artifact_schema_round_trip(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        result = adapter.verify_episode(fixture)
        self.assertTrue(result.ok, result.errors)
        task = adapter.TaskSpec.from_dict(fixture["task"])
        observations = tuple(
            adapter.ObservationRecord.from_dict(item) for item in fixture["observations"]
        )
        actions = tuple(adapter.ActionRecord.from_dict(item) for item in fixture["actions"])
        artifacts = tuple(
            adapter.ArtifactRecord.from_dict(item) for item in fixture["artifacts"]
        )
        self.assertEqual(observations[0].state_hash, fixture["observations"][0]["state_hash"])
        self.assertEqual(actions[0].next_state_hash, observations[1].state_hash)
        self.assertEqual(artifacts[0].digest, fixture["artifacts"][0]["digest"])
        self.assertTrue(fixture["stateful"])
        self.assertTrue(fixture["artifact_or_side_effect"])
        self.assertEqual(task.task_id, fixture["task"]["task_id"])

    def test_fixture_is_offline_only_and_not_portfolio_evidence(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        result = adapter.verify_episode(fixture)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(fixture["evidence_status"], "OFFLINE_FIXTURE_ONLY")
        self.assertEqual(fixture["claim_boundary"], "T3_ADAPTER_VALIDATION_ONLY")
        self.assertFalse(fixture["e6_substitute"])
        self.assertFalse(fixture["portfolio_evidence"])
        self.assertFalse(result.metrics["e6_substitute"])
        self.assertFalse(result.metrics["portfolio_evidence"])
        self.assertTrue(result.metrics["task_success"])
        self.assertNotIn("reward", result.metrics)

    def test_webbench_or_efficacy_claims_fail_closed(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        fixture["suite_id"] = adapter.E6_SUITE_ID
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("E6/WebBench" in error for error in result.errors))

        fixture = adapter.offline_dry_run_fixture()
        fixture["portfolio_evidence"] = True
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("portfolio evidence" in error for error in result.errors))

    def test_hash_and_transition_tampering_fail_closed(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        fixture["task"]["task_id_hash"] = "0" * 64
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("invalid task" in error for error in result.errors))

        fixture = adapter.offline_dry_run_fixture()
        fixture["actions"][0]["next_state_hash"] = "f" * 64
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("does not link" in error for error in result.errors))

        fixture = adapter.offline_dry_run_fixture()
        fixture["artifacts"][0]["digest"] = "f" * 64
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("invalid artifacts" in error for error in result.errors))

    def test_required_artifact_and_initial_state_bindings_are_enforced(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        fixture["artifacts"] = [fixture["artifacts"][0]]
        fixture["terminal"]["artifact_hashes"] = [fixture["artifacts"][0]["digest"]]
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("missing required artifact" in error for error in result.errors))

        fixture = adapter.offline_dry_run_fixture()
        original_task = adapter.TaskSpec.from_dict(fixture["task"])
        altered_initial = dict(original_task.initial_observation)
        altered_initial["clicked"] = True
        altered_task = adapter.TaskSpec(
            env_id=original_task.env_id,
            seed=original_task.seed,
            goal=original_task.goal,
            initial_observation=altered_initial,
        )
        fixture["task"] = altered_task.to_dict()
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("initial_observation" in error for error in result.errors))

    def test_secret_guard_rejects_keys_and_values(self) -> None:
        with self.assertRaises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"api_key": "not-recorded"})
        with self.assertRaises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"headers": {"Authorization": "Bearer abcdefgh"}})
        with self.assertRaises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"note": "hf_abcd12345678"})
        with self.assertRaises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"axtree": "input token=abcdefghijk"})
        adapter.assert_secret_free(adapter.offline_dry_run_fixture())

    def test_preflight_is_separate_from_paid_authorization(self) -> None:
        readiness = adapter.preflight_readiness()
        self.assertEqual(readiness.status, "READY_OFFLINE_FIXTURE")
        self.assertTrue(readiness.fixture_valid)
        self.assertFalse(readiness.browser_launch_attempted)
        self.assertFalse(readiness.network_allowed)
        self.assertFalse(readiness.paid_launch_authorized)
        self.assertIn("browsergym", readiness.missing_runtime)
        decision = adapter.paid_launch_authorization(readiness)
        self.assertFalse(decision.authorized)
        self.assertEqual(decision.status, "NOT_AUTHORIZED")
        self.assertIn("metadata_first_adapter_never_authorizes_paid_launch", decision.reasons)

    def test_paid_authorization_stays_false_with_complete_metadata(self) -> None:
        readiness = adapter.preflight_readiness(
            runtime={
                "browsergym": True,
                "tinker": True,
                "wandb_online_gate": True,
                "model_server": True,
            }
        )
        self.assertEqual(readiness.missing_runtime, ())
        decision = adapter.paid_launch_authorization(
            readiness,
            operational_cap_usd=16.50,
            network_allowed=True,
            online_wandb_confirmed=True,
            hf_checkpoint_export_confirmed=True,
        )
        self.assertFalse(decision.authorized)
        self.assertIn("primary_campaign_owner_authorization_required", decision.reasons)

    def test_runtime_probe_is_metadata_only_and_validated(self) -> None:
        readiness = adapter.preflight_readiness(runtime={"browsergym": True})
        self.assertTrue(readiness.fixture_valid)
        self.assertFalse(readiness.runtime["tinker"])
        with self.assertRaises(adapter.AdapterSchemaError):
            adapter.preflight_readiness(runtime={"browsergym": "installed"})

    def test_fixture_is_reproducible(self) -> None:
        first = adapter.offline_dry_run_fixture()
        second = adapter.offline_dry_run_fixture()
        self.assertEqual(first, second)
        self.assertEqual(
            first["trace_hash"],
            adapter.sha256_json(
                {key: value for key, value in first.items() if key != "trace_hash"}
            ),
        )

    def test_missing_fields_and_non_contiguous_steps_are_rejected(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        del fixture["verifier"]
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("missing required field: verifier" in error for error in result.errors))

        fixture = adapter.offline_dry_run_fixture()
        fixture["observations"][1]["step"] = 3
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        self.assertFalse(result.ok)
        self.assertTrue(any("contiguous" in error for error in result.errors))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
