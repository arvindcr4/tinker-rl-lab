"""Reject impossible accuracy and unrelated experiment identities before tolerance checks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from utils.verify_results import _match_key, verify


REFERENCE = {"demo": {"seed": 42, "last10": 1.0, "peak": 1.0}}


@pytest.mark.parametrize("value", [-0.01, 1.02])
def test_out_of_range_json_cannot_pass_even_generous_tolerances(tmp_path, value):
    (tmp_path / "demo_s42.json").write_text(json.dumps({"last10": value, "peak": value}))
    rows, failed = verify(tmp_path, REFERENCE, 2.0, 2.0)
    assert failed == 1
    assert rows[0][-1] is None


@pytest.mark.parametrize("field", ["last10", "peak"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_impossible_reference_is_rejected(tmp_path, field, value):
    expected = {"demo": {"seed": 42, "last10": 1.0, "peak": 1.0, field: value}}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        verify(tmp_path, expected, 0.05, 0.1)


@pytest.mark.parametrize("tag", ["demo_g128", "other_demo", "demo_extra_s42", "demonstration"])
def test_unknown_experiment_variant_is_not_the_headline(tag):
    assert _match_key(tag, REFERENCE) is None


@pytest.mark.parametrize("suffix", ["", "_s42", "_seed42", "_seed_42"])
def test_explicit_identity_with_recognized_seed_suffix(suffix):
    assert _match_key("demo" + suffix, REFERENCE) == "demo"


def test_unknown_variant_cannot_pass_cli_with_headline_scores(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_g128_s42.json").write_text(
        json.dumps({"last10": 0.344, "peak": 0.625})
    )
    script = Path(__file__).resolve().parents[1] / "utils/verify_results.py"
    result = subprocess.run(
        [sys.executable, str(script), "--results-dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "unknown canonical experiment identity" in result.stderr


def test_uppercase_seed_conflict_cannot_pass(tmp_path):
    (tmp_path / "demo.json").write_text(
        json.dumps({"experiment": "DEMO_S123", "seed": 42, "last10": 1.0, "peak": 1.0})
    )
    rows, failed = verify(tmp_path, REFERENCE, 0.05, 0.1)
    assert failed == 1
    assert rows[0][-1] is None


def test_exact_seed_suffixed_reference_keeps_identity():
    assert _match_key("demo_s42", {"demo_s42": {}}) == "demo_s42"


def test_duplicate_case_reference_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate case-insensitive"):
        verify(tmp_path, {"demo": REFERENCE["demo"], "DEMO": REFERENCE["demo"]}, 0.05, 0.1)


def test_uppercase_identity_matches():
    assert _match_key("DEMO_S42", REFERENCE) == "demo"
