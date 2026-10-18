"""Cached evaluation claims must agree with their recorded denominator."""

from copy import deepcopy

import pytest

from platform_tinker.tinkerrl import grpo


def _prior():
    return {
        "result": {
            "seed": 42,
            "run_id": "synthetic",
            "reward_trace": [0.5],
            "avg_first5": 0.5,
            "avg_last10": 0.5,
            "peak_reward": 0.5,
            "heldout_reward": 0.5,
            "heldout_evaluation": {
                "schema_version": 1,
                "status": "partial_failure",
                "total_n": 2,
                "failed_n": 1,
                "failure_policy": "score_zero_keep_denominator",
                "uncertainty_kind": "example_bootstrap",
                "summary": {
                    "test/reward": 0.5,
                    "test/reward_n": 2.0,
                    "test/total_n": 2.0,
                    "test/failed_n": 1.0,
                    "test/reward_ci_low": 0.0,
                    "test/reward_ci_high": 1.0,
                },
            },
        }
    }


def _validate(prior):
    return grpo._validated_completed_result(prior, 42, 1, [0.5])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "completed"),
        ("total_n", True),
        ("failed_n", 3),
        ("schema_version", 2),
        ("failure_policy", "drop_failed"),
        ("uncertainty_kind", "across_seed_bootstrap"),
    ],
)
def test_cached_evaluation_rejects_contradictory_metadata(field, value):
    prior = _prior()
    prior["result"]["heldout_evaluation"][field] = value
    with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
        _validate(prior)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test/reward", 0.9),
        ("test/reward_n", 1.0),
        ("test/total_n", 1.0),
        ("test/failed_n", 0.0),
        ("test/reward_ci_low", 2.0),
        ("test/reward_ci_high", None),
    ],
)
def test_cached_evaluation_rejects_contradictory_summary(field, value):
    prior = _prior()
    prior["result"]["heldout_evaluation"]["summary"][field] = value
    with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
        _validate(prior)


def test_all_failed_evaluation_cannot_claim_nonzero_mean():
    prior = _prior()
    receipt = prior["result"]["heldout_evaluation"]
    receipt.update(status="all_failed", failed_n=2)
    receipt["summary"]["test/failed_n"] = 2.0
    receipt["summary"]["test/reward_ci_low"] = 0.0
    receipt["summary"]["test/reward_ci_high"] = 0.0
    with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
        _validate(prior)


def test_valid_evaluation_remains_exactly_unchanged():
    prior = _prior()
    before = deepcopy(prior)
    result = _validate(prior)
    assert result.heldout_evaluation == before["result"]["heldout_evaluation"]
    assert prior == before


def test_unknown_evidence_shapes_and_incomplete_intervals_fail_closed():
    candidates = [None, [], "completed"]
    incomplete = _prior()["result"]["heldout_evaluation"]
    incomplete["summary"].pop("test/reward_ci_high")
    candidates.append(incomplete)
    for evidence in candidates:
        prior = _prior()
        prior["result"]["heldout_evaluation"] = evidence
        with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
            _validate(prior)


def test_recorded_mean_only_evaluation_is_not_given_invented_uncertainty():
    prior = _prior()
    evidence = prior["result"]["heldout_evaluation"]
    evidence["uncertainty_kind"] = "not_recorded"
    evidence["summary"].pop("test/reward_ci_low")
    evidence["summary"].pop("test/reward_ci_high")
    before = deepcopy(evidence)
    assert _validate(prior).heldout_evaluation == before
    assert evidence == before


def test_evaluation_summary_uses_the_same_mean_as_the_run_result():
    # Python and NumPy summation can disagree after cancellation. The receipt
    # must not silently publish a second point estimate for the same examples.
    scores = [1e16] + [1.0] * 100 + [-1e16]
    summary = grpo.heldout_reward_summary(scores)
    assert summary["test/reward"] == grpo._reward_mean(scores)


def test_failure_and_single_example_receipts_cannot_invent_spread():
    prior = _prior()
    receipt = prior["result"]["heldout_evaluation"]
    prior["result"]["heldout_reward"] = 0.0
    receipt.update(status="all_failed", failed_n=2)
    receipt["summary"].update({"test/reward": 0.0, "test/failed_n": 2.0})
    with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
        _validate(prior)
    receipt.update(total_n=1, failed_n=1)
    receipt["summary"].update(
        {
            "test/reward_n": 1.0,
            "test/total_n": 1.0,
            "test/failed_n": 1.0,
            "test/reward_ci_high": 0.0,
        }
    )
    with pytest.raises(RuntimeError, match="Completed receipt.*held-out"):
        _validate(prior)
