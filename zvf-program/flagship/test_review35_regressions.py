# ruff: noqa: S101 -- pytest assertions are intentional test checks
"""Synthetic metadata fixtures only; no real scores or launches."""
import copy
import pytest
import pavlov_bfcl_split_manifest as split
from test_pavlov_bfcl_split_manifest import _valid_manifest
from pavlov_bfcl_receipt import _validate_costs, BFCLReceiptBoundaryError
from test_pavlov_banker_toolbench_eval_adapter import build_valid
import pavlov_e1_e14_trained_eval_ledger as ledger
from flagship.pavlov_api_bank_rlvr_train_adapter import _validate_budget
from pavlov_binaryaudit_eval_adapter import _assign_split, BinaryAuditBoundaryError


def test_forged_ready_is_not_evidence():
    assert split.validate_split_manifest_record({"status": "READY"})
    manifest = _valid_manifest()
    record = split.build_split_manifest_record(manifest)
    assert split.validate_split_manifest_record(record)
    assert not split.validate_split_manifest_record(record, manifest)
    for field, value in [("dataset_revision", "b" * 40), ("split_digest", "sha256:" + "0" * 64), ("paid_launch_allowed", True)]:
        altered = copy.deepcopy(record)
        altered[field] = value
        assert split.validate_split_manifest_record(altered, manifest)
    manifest["split"]["primary_eval"] = manifest["split"]["train"]
    assert split.validate_split_manifest_record(record, manifest)
    assert split.validate_split_manifest_record({"schema_version": split.SCHEMA_VERSION, "status": "BLOCKED", "blockers": []})


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -float("inf"), True, -1])
def test_bfcl_cost_finite(amount):
    with pytest.raises(BFCLReceiptBoundaryError):
        _validate_costs({"status": "observed", "total_usd": amount})


def test_banker_missing_receipt_blocks_both_statuses():
    record = build_valid(result_receipts=None)
    assert record["status"] == record["boundary_status"] == "BLOCKED"


def test_unknown_binaryaudit_family_blocked():
    with pytest.raises(BinaryAuditBoundaryError):
        _assign_split("new-family-task", "new-family", False)


def training_fixture():
    return ({"run_id": "training", "sampler_path": "sampler"},
            {"commit_sha": "a" * 40, "revision": "b" * 40, "repo_url": "model"})


def result_fixture():
    return {"status": "COMPLETE", "is_model_score": True, "score": 0.0,
        "model": {"training_run_id": "training", "sampler_path": "sampler",
        "checkpoint_commit": "a" * 40, "checkpoint_revision": "b" * 40, "model_identity": "model"}}


def test_no_unvalidated_lane_can_score():
    training, checkpoint = training_fixture()
    for lane in ("E1", "E11", "E14"):
        assert ledger.validate_suite_result(lane, result_fixture(), training, checkpoint)


@pytest.mark.parametrize("field", ["training_run_id", "sampler_path", "checkpoint_commit", "checkpoint_revision", "model_identity"])
def test_all_checkpoint_bindings_required(monkeypatch, field):
    monkeypatch.setitem(ledger.SUITE_RESULT_VALIDATORS, "E11", lambda source: [])
    training, checkpoint = training_fixture()
    source = result_fixture()
    assert not ledger.validate_suite_result("E11", source, training, checkpoint)
    source["model"][field] = "other"
    assert ledger.validate_suite_result("E11", source, training, checkpoint)


@pytest.mark.parametrize("score", [None, True, "1", float("nan"), float("inf")])
def test_registered_validator_cannot_bypass_score_checks(monkeypatch, score):
    monkeypatch.setitem(ledger.SUITE_RESULT_VALIDATORS, "E11", lambda source: [])
    source = result_fixture()
    source["score"] = score
    assert ledger.validate_suite_result("E11", source, *training_fixture())


def test_missing_score_not_logged_as_zero():
    row = {"lane": "E11", "model_executed": False, "is_model_score": False, "status": "BLOCKED", "score": None}
    assert "suite/score" not in ledger.suite_log_metrics(0, row)
    row.update(model_executed=True, is_model_score=True, status="SCORED", score=0.0)
    assert ledger.suite_log_metrics(0, row)["suite/score"] == 0.0


def test_budget_normalization_never_authorizes_adapter_launch():
    budget = {"status": "AUTHORIZED_TINKER_ONLY", "provider": "Tinker", "paid_jobs_may_launch": True,
        "maximum_usd": 18, "operational_cap_usd": 16, "safety_reserve_usd": 2,
        "authorized_at": "2026-08-09T00:00:00Z", "launch_authorized": False, "launches_any_job": False}
    normalized, errors = _validate_budget(budget)
    assert not errors
    assert normalized["budget_authorized"] is True
    assert normalized["adapter_may_launch"] is normalized["paid_jobs_may_launch"] is False
    assert _validate_budget(normalized) == (normalized, [])
    for field in ("maximum_usd", "operational_cap_usd", "safety_reserve_usd"):
        bad = dict(budget, **{field: float("nan")})
        assert _validate_budget(bad)[1]


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), True])
def test_agentharm_rejects_invalid_budget(amount):
    from test_pavlov_agentharm_frontiermath_adapter import PavlovAgentharmFrontiermathAdapterTests
    from flagship.pavlov_agentharm_frontiermath_adapter import _validate_receipt_fields
    fixture = PavlovAgentharmFrontiermathAdapterTests()
    receipt = fixture._make_filled_receipt("agentharm_eval")
    receipt["budget_receipt"]["maximum_usd"] = amount
    assert any("maximum_usd" in error for error in _validate_receipt_fields("agentharm_eval", receipt))
