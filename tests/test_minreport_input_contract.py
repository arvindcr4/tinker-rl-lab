"""Declared provenance assertions must be typed and internally consistent."""

import copy
import json
from pathlib import Path

import pytest

from platform_hybrid.registry.provenance import minreport


@pytest.fixture
def declared_record():
    path = Path(__file__).resolve().parents[1] / (
        "platform_hybrid/registry/provenance/campaign.provenance.json"
    )
    record = json.loads(path.read_text())
    # Synthetic test-only assertion, never written back to the signed record.
    record["rigor"]["reports_ci"] = True
    return record


def failures(record):
    return [
        item for item in minreport.verify(record, strict=True)["checks"] if item["level"] != "PASS"
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
        assert any(item["msg"] == "decontamination not asserted" for item in failures(record))


def test_generator_does_not_invent_disjointness_or_decontamination(tmp_path):
    output = tmp_path / "record.json"
    record = minreport.generate("unknown-evidence", None, output)
    assert record["min_report_rl"]["heldout_split"]["disjoint"] is None
    assert record["min_report_rl"]["decontamination"]["verified"] is False
    assert record["min_report_rl"]["decontamination"]["method"] is None
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


def test_live_source_outside_repository_has_a_valid_display_label(declared_record, tmp_path):
    source = tmp_path / "synthetic_source.py"
    source.write_text("# synthetic fixture\n")
    declared_record["provenance"]["code_file"] = str(source)
    declared_record["provenance"]["code_hash"] = minreport.sha256_file(source)
    assert failures(declared_record) == []


def test_strict_mode_does_not_change_score_or_mutate_frozen_record():
    path = Path(__file__).resolve().parents[1] / (
        "platform_hybrid/registry/provenance/campaign.provenance.json"
    )
    before = path.read_bytes()
    record = json.loads(before)
    ordinary = minreport.verify(record)
    strict = minreport.verify(record, strict=True)
    assert ordinary == strict
    assert strict["score_pct"] == 95.7
    assert any("rigor.reports_ci" in item["msg"] for item in failures(record))
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
    assert "Strict verification rejected" in result.stderr
    assert "Traceback" not in result.stderr
    receipt = json.loads(path.with_suffix(".audit.json").read_text())
    assert any(item["level"] == "WARN" for item in receipt["checks"])


def test_registry_schema_objects_and_legacy_compact_forms_are_supported(declared_record):
    assert failures(declared_record) == []
    declared_record["min_report_rl"]["sampler_backend"] = {
        "backend": "tinker",
        "precision": "float32",
        "temperature": 0.8,
        "top_p": 0.95,
    }
    declared_record["min_report_rl"]["telemetry"] = {
        "per_step_zvf": True,
        "per_step_gu": True,
        "source": "synthetic-fixture.jsonl",
    }
    assert failures(declared_record) == []
