"""Offline adversarial evidence-boundary regressions; no providers or benchmarks."""

import pytest
from flagship import pavlov_statistics as stats
from flagship import pavlov_statistical_stop_rule as stop
from flagship import pavlov_portfolio_split_registry as registry
from flagship import pavlov_lifescibench_eval_adapter as life
from flagship import pavlov_sdab_eval_adapter as sdab
from flagship import pavlov_swe_bench_pro_eval_adapter as swe
from flagship import pavlov_protocol_validator as protocol


def test_newcombe_published_method10_oracle():
    lower, upper = stats.newcombe_paired_risk_difference_interval(
        [1] * 20 + [0] * 12 + [1] * 2 + [0] * 16, [1] * 32 + [0] * 18
    )
    assert lower == pytest.approx(0.0562, abs=0.00005)
    assert upper == pytest.approx(0.3292, abs=0.00005)


def test_raw_observations_cannot_override_final_use():
    with pytest.raises(stop.StopRuleInputError, match="final split"):
        stop._normalise_arm(
            {"base_scores": [0, 0], "candidate_scores": [1, 0], "final_used_for_selection": True}
        )


def test_reported_gate_booleans_are_not_measurements():
    with pytest.raises(stop.StopRuleInputError, match="measurements"):
        stop._as_gate_result(
            {"passed": True, "no_regression": {"passed": True}, "safety": {"passed": True}}
        )


def test_role_conflicts_and_hash_aliases():
    assert registry._extract_task_hashes({"task_hashes": {"train": ["a" * 64]}}, "primary_eval")[1]
    assert registry._extract_task_hashes(
        {"primary_eval_task_hashes": ["a" * 64]}, "primary_eval"
    ) == (["a" * 64], [])


def test_manifest_rows_cannot_be_rebound():
    manifest = {"manifest_hash": "a" * 64, "rows": [{"task_id": "task", "family": "family"}]}
    with pytest.raises(life.LifeSciBenchSchemaError, match="hash does not match"):
        life.build_heldout_proof(train_manifest=manifest, eval_manifest=manifest)


def test_metadata_drops_nested_secrets_and_unknown_containers():
    metadata, _ = sdab._strip_raw_content(
        {
            "task_id": "one",
            "provider_metadata": {"hidden_tests": "secret", "nested": [{"prompt": "private"}]},
        }
    )
    assert metadata == {"task_id": "one"}


def test_empty_p2p_is_valid_but_empty_f2p_is_not():
    assert swe._official_test_names("[]", "pass_to_pass") == []
    with pytest.raises(swe.BoundaryValidationError):
        swe._official_test_names("[]", "fail_to_pass")


def test_unrelated_conditional_cannot_qualify_holdout_claim():
    assert protocol._find_unqualified_holdout_claims(
        "When the job ends, upload the log.\nAll 14 evaluation suites are held-out."
    )


def test_missing_inventory_reports_unknown_overlap():
    report = stop.validate_selection_vs_final_split("selection", "final")
    assert report["separate"] is None
    assert report["overlap_examples"] is None
