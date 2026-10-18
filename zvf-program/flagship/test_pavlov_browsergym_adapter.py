"""Offline tests for the fail-closed Pavlov T3 BrowserGym boundary."""

from __future__ import annotations

import unittest

from flagship import pavlov_browsergym_adapter as adapter
import pytest


class BrowserGymAdapterTests(unittest.TestCase):
    def test_pinned_identity_is_t3_and_not_e6(self) -> None:
        assert adapter.SUITE_ID == "browsergym_train"
        assert adapter.SUITE_ROLE == "train"
        assert adapter.E6_SUITE_ID == "webbench_eval"
        assert adapter.SUITE_ID != adapter.E6_SUITE_ID
        assert (
            "miniwob-plusplus@7fd85d71a4b60325c6585396ec4f48377d049838"
            in adapter.PINNED_DATASET_REVISION
        )
        assert adapter.PINNED_ENVIRONMENT_REVISION == "browsergym-miniwob==0.14.3"

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
        assert first.task_id == second.task_id
        assert first.task_id_hash == second.task_id_hash
        assert first.task_id == adapter.deterministic_task_id(
            env_id=first.env_id,
            seed=first.seed,
            goal=first.goal,
            initial_observation=first.initial_observation,
        )
        assert len(first.task_id_hash) == 64

    def test_task_tampering_is_rejected(self) -> None:
        task = adapter.TaskSpec(
            env_id="browsergym/miniwob.click-button",
            seed=1,
            goal="Click Continue.",
            initial_observation={"url": "about:blank"},
        )
        with pytest.raises(adapter.AdapterSchemaError):
            adapter.TaskSpec(
                env_id=task.env_id,
                seed=task.seed,
                goal=task.goal,
                initial_observation=task.initial_observation,
                task_id="t3-browsergym-000000000000000000000000",
            )
        with pytest.raises(adapter.AdapterSchemaError):
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
        assert result.ok, result.errors
        task = adapter.TaskSpec.from_dict(fixture["task"])
        observations = tuple(
            adapter.ObservationRecord.from_dict(item) for item in fixture["observations"]
        )
        actions = tuple(adapter.ActionRecord.from_dict(item) for item in fixture["actions"])
        artifacts = tuple(adapter.ArtifactRecord.from_dict(item) for item in fixture["artifacts"])
        assert observations[0].state_hash == fixture["observations"][0]["state_hash"]
        assert actions[0].next_state_hash == observations[1].state_hash
        assert artifacts[0].digest == fixture["artifacts"][0]["digest"]
        assert fixture["stateful"]
        assert fixture["artifact_or_side_effect"]
        assert task.task_id == fixture["task"]["task_id"]

    def test_fixture_is_offline_only_and_not_portfolio_evidence(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        result = adapter.verify_episode(fixture)
        assert result.ok, result.errors
        assert fixture["evidence_status"] == "OFFLINE_FIXTURE_ONLY"
        assert fixture["claim_boundary"] == "T3_ADAPTER_VALIDATION_ONLY"
        assert not fixture["e6_substitute"]
        assert not fixture["portfolio_evidence"]
        assert not result.metrics["e6_substitute"]
        assert not result.metrics["portfolio_evidence"]
        assert result.metrics["task_success"]
        assert "reward" not in result.metrics

    def test_webbench_or_efficacy_claims_fail_closed(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        fixture["suite_id"] = adapter.E6_SUITE_ID
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("E6/WebBench" in error for error in result.errors)

        fixture = adapter.offline_dry_run_fixture()
        fixture["portfolio_evidence"] = True
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("portfolio evidence" in error for error in result.errors)

    def test_hash_and_transition_tampering_fail_closed(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        fixture["task"]["task_id_hash"] = "0" * 64
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("invalid task" in error for error in result.errors)

        fixture = adapter.offline_dry_run_fixture()
        fixture["actions"][0]["next_state_hash"] = "f" * 64
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("does not link" in error for error in result.errors)

        fixture = adapter.offline_dry_run_fixture()
        fixture["artifacts"][0]["digest"] = "f" * 64
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("invalid artifacts" in error for error in result.errors)

    def test_required_artifact_and_initial_state_bindings_are_enforced(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        fixture["artifacts"] = [fixture["artifacts"][0]]
        fixture["terminal"]["artifact_hashes"] = [fixture["artifacts"][0]["digest"]]
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("missing required artifact" in error for error in result.errors)

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
        assert not result.ok
        assert any("initial_observation" in error for error in result.errors)

    def test_secret_guard_rejects_keys_and_values(self) -> None:
        with pytest.raises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"api_key": "not-recorded"})
        with pytest.raises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"headers": {"Authorization": "Bearer abcdefgh"}})
        with pytest.raises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"note": "hf_abcd12345678"})
        with pytest.raises(adapter.SecretMaterialError):
            adapter.assert_secret_free({"axtree": "input token=abcdefghijk"})
        adapter.assert_secret_free(adapter.offline_dry_run_fixture())

    def test_preflight_is_separate_from_paid_authorization(self) -> None:
        readiness = adapter.preflight_readiness()
        assert readiness.status == "READY_OFFLINE_FIXTURE"
        assert readiness.fixture_valid
        assert not readiness.browser_launch_attempted
        assert not readiness.network_allowed
        assert not readiness.paid_launch_authorized
        assert "browsergym" in readiness.missing_runtime
        decision = adapter.paid_launch_authorization(readiness)
        assert not decision.authorized
        assert decision.status == "NOT_AUTHORIZED"
        assert "metadata_first_adapter_never_authorizes_paid_launch" in decision.reasons

    def test_paid_authorization_stays_false_with_complete_metadata(self) -> None:
        readiness = adapter.preflight_readiness(
            runtime={
                "browsergym": True,
                "tinker": True,
                "wandb_online_gate": True,
                "model_server": True,
            }
        )
        assert readiness.missing_runtime == ()
        decision = adapter.paid_launch_authorization(
            readiness,
            operational_cap_usd=16.50,
            network_allowed=True,
            online_wandb_confirmed=True,
            hf_checkpoint_export_confirmed=True,
        )
        assert not decision.authorized
        assert "primary_campaign_owner_authorization_required" in decision.reasons

    def test_runtime_probe_is_metadata_only_and_validated(self) -> None:
        readiness = adapter.preflight_readiness(runtime={"browsergym": True})
        assert readiness.fixture_valid
        assert not readiness.runtime["tinker"]
        with pytest.raises(adapter.AdapterSchemaError):
            adapter.preflight_readiness(runtime={"browsergym": "installed"})

    def test_fixture_is_reproducible(self) -> None:
        first = adapter.offline_dry_run_fixture()
        second = adapter.offline_dry_run_fixture()
        assert first == second
        assert first["trace_hash"] == adapter.sha256_json(
            {key: value for key, value in first.items() if key != "trace_hash"}
        )

    def test_missing_fields_and_non_contiguous_steps_are_rejected(self) -> None:
        fixture = adapter.offline_dry_run_fixture()
        del fixture["verifier"]
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("missing required field: verifier" in error for error in result.errors)

        fixture = adapter.offline_dry_run_fixture()
        fixture["observations"][1]["step"] = 3
        fixture["trace_hash"] = adapter.sha256_json(
            {key: value for key, value in fixture.items() if key != "trace_hash"}
        )
        result = adapter.verify_episode(fixture)
        assert not result.ok
        assert any("contiguous" in error for error in result.errors)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
