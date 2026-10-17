"""Matching v1 settings never become a scientific comparability certificate."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import audit_portfolio_comparability as audit_tool
from platform_tinker.tinkerrl import pavlov_portfolio_eval as evaluator
from tests import test_portfolio_eval_integrity

harness = test_portfolio_eval_integrity.harness


def _receipt(role, scores=(1.0, 0.0)):
    suites = ("api_bank_rlvr_train", "swe_gym_train")
    rows = [
        {
            "suite_id": suite,
            "source_id": str(index),
            "prompt_sha256": hashlib.sha256(suite.encode()).hexdigest(),
            "prompt_token_ids_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "prompt_tokens": 3,
            "original_prompt_tokens": 3,
            "prompt_truncated": False,
            "sampling_seed": 12 + index,
            "score": score,
            "exact": score == 1,
        }
        for index, (suite, score) in enumerate(zip(suites, scores))
    ]
    return {
        "schema_version": audit_tool.SCHEMA,
        "status": "SCORED",
        "role": role,
        "evidence_class": "portfolio_source_validation",
        "is_e1_e14_score": False,
        "is_model_score": True,
        "dataset_revision": "d" * 64,
        "source_revisions": {"source": "a" * 40},
        "model": {
            "model_id": "synthetic",
            "model_revision": "a" * 40,
            "model_revision_scope": "tokenizer_request_only",
        },
        "sampling": {
            "samples_per_problem": 1,
            "retries": 0,
            "temperature": 0.1,
            "max_tokens": 17,
            "top_p": 0.95,
            "max_prompt_tokens": 2048,
            "prompt_truncation": "prefix",
            "seed": 12,
        },
        "records": rows,
        "count": 2,
        "mean_reward": sum(scores) / 2,
        "exact_match_rate": sum(score == 1 for score in scores) / 2,
        "suites": {
            row["suite_id"]: {
                "count": 1,
                "mean_reward": row["score"],
                "exact_matches": int(row["exact"]),
                "exact_match_rate": float(row["exact"]),
            }
            for row in rows
        },
    }


def test_matching_recorded_settings_with_different_scores_remain_unknown():
    base, trained = _receipt("base"), _receipt("trained", (1.0, 1.0))
    before = deepcopy((base, trained))
    result = audit_tool.audit(base, trained)
    assert result["status"] == "unknown"
    assert result["recorded_configuration"] == "compatible"
    assert result["scientific_comparison_established"] is False
    assert result["missing_comparability_evidence"] == [
        "target_content_binding",
        "scorer_implementation_binding",
    ]
    assert "decontamination" in result["not_established"]
    assert "delta" not in result
    assert (base, trained) == before


@pytest.mark.parametrize(
    "change", ["row_order", "cohort", "seed", "temperature", "tokenizer", "dataset"]
)
def test_known_mismatch_is_incompatible(change):
    base, trained = _receipt("base"), _receipt("trained")
    if change == "row_order":
        trained["records"].reverse()
        for index, row in enumerate(trained["records"]):
            row["sampling_seed"] = trained["sampling"]["seed"] + index
    elif change == "cohort":
        trained["records"][0]["prompt_token_ids_sha256"] = "b" * 64
    elif change == "seed":
        trained["sampling"]["seed"] += 1
        for row in trained["records"]:
            row["sampling_seed"] += 1
    elif change == "temperature":
        trained["sampling"]["temperature"] = 0.8
    elif change == "tokenizer":
        trained["model"]["model_revision"] = "b" * 40
    else:
        trained["dataset_revision"] = "e" * 64
    result = audit_tool.audit(base, trained)
    assert result["status"] == result["recorded_configuration"] == "incompatible"
    assert result["mismatches"]


@pytest.mark.parametrize(
    "change",
    ["mean", "count", "exact", "suite", "role", "boolean_score", "bad_hash", "bad_metadata"],
)
def test_invalid_arithmetic_or_contract_is_rejected(change):
    base, trained = _receipt("base"), _receipt("trained")
    if change == "mean":
        trained["mean_reward"] = 0.9
    elif change == "count":
        trained["count"] = True
    elif change == "exact":
        trained["records"][0]["exact"] = False
    elif change == "suite":
        trained["suites"]["swe_gym_train"]["mean_reward"] = 0.5
    elif change == "role":
        trained["role"] = "base"
    elif change == "boolean_score":
        trained["records"][0]["score"] = True
    elif change == "bad_hash":
        trained["records"][0]["prompt_sha256"] = "not a digest"
    else:
        trained["sampling"]["max_tokens"] = "17"
    with pytest.raises(ValueError, match="trained"):
        audit_tool.audit(base, trained)


def test_missing_legacy_metadata_is_unknown_without_backfill():
    base, trained = _receipt("base"), _receipt("trained")
    for receipt in (base, trained):
        del receipt["records"][0]["prompt_token_ids_sha256"]
    result = audit_tool.audit(base, trained)
    assert result["recorded_configuration"] == "unknown"
    assert result["missing_recorded_fields"] == ["ordered_records.prompt_token_ids_sha256"]
    assert "prompt_token_ids_sha256" not in base["records"][0]


@pytest.mark.parametrize("raw", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}', "[]"])
def test_strict_json_input_rejects_ambiguous_receipts(tmp_path, raw):
    path = tmp_path / "receipt.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="duplicate|nonfinite|finite|object"):
        audit_tool.read_receipt(path)


def test_actual_evaluator_outputs_audit_readonly_and_exit_unknown(harness, tmp_path, capsys):
    assert harness.invoke() == 0
    base_path = harness.output
    trained_path = tmp_path / "trained.json"
    assert (
        evaluator.main(
            [
                "--sampler-path",
                "tinker://synthetic",
                "--hf-repo",
                "synthetic/model",
                "--hf-revision",
                "synthetic-revision",
                "--hf-commit",
                "a" * 40,
                "--per-suite",
                "1",
                "--out",
                str(trained_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    before = (base_path.read_bytes(), trained_path.read_bytes())
    assert audit_tool.main(["--base", str(base_path), "--trained", str(trained_path)]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "unknown"
    assert result["recorded_configuration"] == "compatible"
    assert result["input_sha256"]["base"] == hashlib.sha256(before[0]).hexdigest()
    command = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "tools/audit_portfolio_comparability.py"),
            "--base",
            str(base_path),
            "--trained",
            str(trained_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert command.returncode == 3
    assert json.loads(command.stdout) == result
    assert command.stderr == ""
    assert (base_path.read_bytes(), trained_path.read_bytes()) == before


def test_invalid_cli_input_returns_nonzero_json(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    assert audit_tool.main(["--base", str(missing), "--trained", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversize"])
def test_input_boundary_rejects_nonregular_or_oversize_evidence(tmp_path, monkeypatch, kind):
    import os

    path = tmp_path / "input.json"
    if kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text(json.dumps(_receipt("base")))
        before = target.read_bytes()
        path.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        monkeypatch.setattr(audit_tool, "MAX_BYTES", 32)
        path.write_bytes(b" " * 33)
    with pytest.raises((OSError, ValueError), match="regular|audit limit|symbolic"):
        audit_tool.read_receipt(path)
    if kind == "symlink":
        assert target.read_bytes() == before


@pytest.mark.parametrize("change", ["seed", "truncation", "prompt_length"])
def test_matching_but_self_contradictory_row_metadata_is_invalid(change):
    base, trained = _receipt("base"), _receipt("trained")
    for receipt in (base, trained):
        row = receipt["records"][0]
        if change == "seed":
            row["sampling_seed"] += 1
        elif change == "truncation":
            row["prompt_truncated"] = True
        else:
            row["prompt_tokens"] = 2
    with pytest.raises(ValueError, match="contradict"):
        audit_tool.audit(base, trained)


def test_different_truncation_policies_are_incomparable():
    base, trained = _receipt("base"), _receipt("trained")
    trained["sampling"]["prompt_truncation"] = "reject_over_budget"
    result = audit_tool.audit(base, trained)
    assert result["status"] == "incompatible"
    assert "sampling.prompt_truncation" in result["mismatches"]


def test_reject_over_budget_policy_cannot_claim_a_truncated_record():
    base, trained = _receipt("base"), _receipt("trained")
    for receipt in (base, trained):
        receipt["sampling"]["prompt_truncation"] = "reject_over_budget"
        record = receipt["records"][0]
        record["original_prompt_tokens"] = receipt["sampling"]["max_prompt_tokens"] + 1
        record["prompt_tokens"] = receipt["sampling"]["max_prompt_tokens"]
        record["prompt_truncated"] = True
    with pytest.raises(ValueError, match="contradictory"):
        audit_tool.audit(base, trained)
