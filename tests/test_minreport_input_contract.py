"""Declared provenance assertions must be typed and internally consistent."""

import copy
import json
from pathlib import Path

import pytest

from platform_hybrid.registry.provenance import minreport
from test_registry_provenance_review import publication  # noqa: F401


@pytest.fixture
def declared_record(request):
    # Fully synthetic v2 evidence; historical signed records remain immutable.
    record = copy.deepcopy(request.getfixturevalue("publication")["record"])
    assert minreport.verify(record)["evidence_valid"]
    assert not minreport.verify(record)["accepted"]
    return record


def failures(record):
    return [
        item
        for item in minreport.verify(record, strict=True)["checks"]
        if item["level"] != "PASS" and not item["msg"].startswith("UNAUTHENTICATED")
    ]


def test_truthy_strings_and_numeric_flags_cannot_satisfy_reporting_assertions(declared_record):
    assert failures(declared_record) == []
    for field in ("reports_ci", "reports_heldout", "heldout_disjoint"):
        for value in ("false", "0", "no", 1, False, None, [], {}):
            record = copy.deepcopy(declared_record)
            record["rigor"][field] = value
            before = copy.deepcopy(record)
            assert failures(record), (field, value)
            assert record == before


def test_seed_count_requires_matching_distinct_integer_identities(declared_record):
    for count, seeds in (
        ("3", [0, 1, 2]),
        (3.5, [0, 1, 2]),
        (3.0, [0, 1, 2]),
        (True, [0, 1, 2]),
        (None, [0, 1, 2]),
        (3, [0, 0, 0]),
        (3, [0, 1]),
        (3, [0, 1, 2, 3]),
        (3, [False, 1, 2]),
        (3, ["0", 1, 2]),
        (3, None),
        (3, {"a": 0, "b": 1, "c": 2}),
    ):
        record = copy.deepcopy(declared_record)
        record["rigor"].update(n_seeds=count, seeds=seeds)
        assert any(item["level"] == "FAIL" for item in failures(record)), (count, seeds)


def test_malformed_json_sections_produce_failures_without_crashing(declared_record):
    for value in (None, [], "yes", 1, True):
        assert failures(value)[0]["level"] == "FAIL"
        for field in ("min_report_rl", "rigor", "provenance"):
            record = copy.deepcopy(declared_record)
            record[field] = value
            assert failures(record), (field, value)
        record = copy.deepcopy(declared_record)
        record["min_report_rl"]["heldout_split"] = value
        assert failures(record)
        record = copy.deepcopy(declared_record)
        record["provenance"]["code_file"] = value
        assert failures(record)


def test_decontamination_requires_explicit_verified_assertion(declared_record):
    for value in ("false", 1, False, None):
        record = copy.deepcopy(declared_record)
        record["min_report_rl"]["decontamination"]["verified"] = value
        assert any("decontamination" in item["msg"] for item in failures(record))


def test_generator_does_not_invent_disjointness_or_decontamination(tmp_path):
    output = tmp_path / "record.json"
    record = minreport.generate("unknown-evidence", None, output)
    assert record["min_report_rl"]["heldout_split"]["disjoint_from_reward_env"] is None
    assert record["min_report_rl"]["decontamination"]["performed"] is None
    assert record["min_report_rl"]["decontamination"]["parser_robustness_probe"] is None
    assert failures(record)
    missing_output = tmp_path / "not-created.json"
    with pytest.raises(FileNotFoundError):
        minreport.generate("missing-config", tmp_path / "missing.json", missing_output)
    assert not missing_output.exists()
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        minreport.generate("bad-config", bad_config, missing_output)
    assert not missing_output.exists()


def test_live_source_outside_repository_fails_closed(declared_record, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.py")
    outside.write_text("def reward(answer):\n    return float(answer == 'correct')\n")
    try:
        declared_record["provenance"]["code_file"] = str(outside)
        assert minreport.resolve_source_path(outside) is None
        assert any("provenance source missing" in item["msg"] for item in failures(declared_record))
    finally:
        outside.unlink()


def test_strict_mode_does_not_change_score_or_mutate_frozen_record():
    path = Path(__file__).resolve().parents[1] / (
        "platform_hybrid/registry/provenance/campaign.provenance.json"
    )
    before = path.read_bytes()
    record = json.loads(before)
    ordinary = minreport.verify(record)
    strict = minreport.verify(record, strict=True)
    assert ordinary == strict
    assert not strict["accepted"]
    assert not strict["evidence_valid"]
    assert strict["grade"] == "REJECTED"
    assert failures(record)
    assert path.read_bytes() == before


def test_strict_cli_rejects_truthy_false_and_writes_a_failure_receipt(declared_record, tmp_path):
    import subprocess
    import sys

    declared_record["rigor"]["reports_ci"] = "false"
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(declared_record))
    command = [sys.executable, str(Path(minreport.__file__)), "verify", str(path), "--strict"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "PUBLICATION: REJECT" in result.stdout
    assert "Traceback" not in result.stderr
    receipt = json.loads(path.with_name(path.name + ".audit.json").read_text())
    assert not receipt["accepted"]
    assert any(item["level"] == "FAIL" for item in receipt["checks"])


def test_registry_schema_objects_and_legacy_compact_forms_are_versioned(declared_record):
    assert failures(declared_record) == []
    assert (
        minreport.canonical_min_report({"sampler_backend": "tinker"})["sampler_backend"]["backend"]
        == "tinker"
    )
    # Compact v1 declarations may be adapted during generation, but cannot
    # replace v2 evidence-bound schema objects at verification time.
    declared_record["min_report_rl"]["sampler_backend"] = "tinker"
    declared_record["min_report_rl"]["telemetry"] = ["loss"]
    assert failures(declared_record)


@pytest.mark.parametrize("field", ["git_sha", "code_hash", "reward_fn_hash"])
@pytest.mark.parametrize("value", [None, [], {}, True, 1, "", "   "])
def test_hash_presence_rejects_malformed_values(declared_record, field, value):
    declared_record["provenance"][field] = value
    assert any(
        item["level"] == "FAIL" and field in item["msg"] for item in failures(declared_record)
    )


def test_reward_hash_must_match_function_and_cannot_copy_file_hash(declared_record):
    assert "data_ref" in declared_record["provenance"]
    assert "data_hash" not in declared_record["provenance"]
    assert failures(declared_record) == []
    declared_record["provenance"]["reward_fn_hash"] = declared_record["provenance"]["code_hash"]
    assert any(
        item["level"] == "FAIL" and "reward_fn_hash missing or mismatched" in item["msg"]
        for item in failures(declared_record)
    )
    declared_record["provenance"]["reward_fn_hash"] = "sha256:" + "0" * 64
    assert any(
        "reward_fn_hash missing or mismatched" in item["msg"] for item in failures(declared_record)
    )
