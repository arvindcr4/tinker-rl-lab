"""CPU-only review regressions; all receipts are explicitly synthetic fixtures."""

import math
import pytest
from pilot.analysis import (
    AnalysisContractError,
    screening_gate,
    normalized_auc,
    balanced_candidate_report,
    _standardized_mean,
    _monte_carlo,
    _comparison_effect,
    _comparison_equivalent,
)
from pilot.test_analysis import screening_records, seal_records
from pilot.feasibility_candidate import length_feasibility_preflight
from pilot.protocol import canonical_fingerprint
from pilot.replay import ReplayContractError


def test_negative_auc_noninferior_final_fails():
    records = screening_records()
    for receipt in records:
        record = receipt["full_record"]
        if (
            record["condition"] == "intended_full"
            and record["regime"] == "filtered_variable_length"
        ):
            for row in record["evaluations"][1:-1]:
                row["accuracy"] -= 0.1
    result = screening_gate(seal_records(records))
    assert result["learning"]["final_noninferiority"]
    assert not result["learning"]["pass"]
    assert result["verdict"] == "KILL"
    assert _standardized_mean([-1, -1, -1]) == -math.inf


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.01, 1.01, True])
def test_auc_rejects_invalid_values(bad):
    record = screening_records()[0]["full_record"]
    record["evaluations"][0]["accuracy"] = bad
    with pytest.raises(AnalysisContractError):
        normalized_auc(record)


def test_receipt_integrity_identity_and_raw_rejection():
    records = screening_records()
    raw = [r["full_record"] for r in records]
    with pytest.raises(AnalysisContractError, match="receipt"):
        screening_gate(raw)
    records[0]["unit"] = {**records[0]["unit"], "seed": 99}
    with pytest.raises(AnalysisContractError, match="identity"):
        screening_gate(records)
    records = screening_records()
    records[0]["full_record"]["evaluations"][0]["accuracy"] = 0.1
    with pytest.raises(AnalysisContractError, match="fingerprint"):
        screening_gate(records)


def test_balanced_candidate_never_original_go():
    records = [
        r for r in screening_records() if r["full_record"]["regime"] == "balanced_equal_length"
    ]
    with pytest.raises(AnalysisContractError, match="matrix mismatch"):
        screening_gate(records)
    report = balanced_candidate_report(records)
    assert report["verdict"] == "DESCRIPTIVE_ONLY_PENDING_PROTOCOL_AUTHORIZATION"
    assert report["regimes"] == ["balanced_equal_length"]
    assert len(report["analysis_protocol_fingerprint"]) == 64
    assert "filtered" not in str(report["paired_seed_auc_differences"])


def test_degenerate_power_and_bad_draws():
    kwargs = dict(n=5, draws=100, alpha=0.0125, seed=1, equivalence_margin=None)
    with pytest.raises(AnalysisContractError, match="DEGENERATE"):
        _monte_carlo([1, 1, 1], **kwargs)
    kwargs["draws"] = 0
    with pytest.raises(AnalysisContractError, match="draws"):
        _monte_carlo([1, 2, 3], **kwargs)


def test_angular_metric_is_not_magnitude_equivalence():
    receipt = {"gradient_relation": "nonzero", "gradient_cosine": 1.0, "gradient_relative_l2": 0.5}
    assert _comparison_effect(receipt) == 0
    assert not _comparison_equivalent(receipt)


def test_preflight_identical_cap_lengths_stops_and_identity_binds():
    identity = dict(
        model_revision="frozen-revision",
        prompt_template_sha256="a" * 64,
        decoding={"temperature": 0.8},
        max_completion_length=512,
    )
    pool = dict(
        prompt_id="measured-fixture",
        seed=11,
        generation_fingerprint=canonical_fingerprint(identity),
        completion_lengths=[512] * 16,
    )
    report = length_feasibility_preflight([pool], generation_identity=identity)
    assert report["status"] == "STOP_INFEASIBLE_NO_FULL_ALLOCATION"
    assert not report["allocation_authorized"]
    with pytest.raises(ReplayContractError, match="identity"):
        length_feasibility_preflight(
            [pool], generation_identity={**identity, "max_completion_length": 256}
        )
