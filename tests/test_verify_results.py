"""utils/verify_results.py must fail closed on real grpo_cli trainer logs."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "utils" / "verify_results.py"
EXPECTED = REPO / "platform_hybrid" / "paper" / "expected_results.json"


def _cli_log(last10: float, peak: float, seed: int = 42) -> str:
    # Mirrors the per-seed block printed by platform_tinker/tinkerrl/grpo_cli.py main().
    return (
        "[grpo_cli] preset=gsm8k seeds=1 steps=30\n\n"
        f"[grpo_cli] Seed {seed} done.\n"
        "  run_id        : abc\n"
        "  sampler       : tinker://x\n"
        "  avg_first5    : 0.100\n"
        f"  avg_last10    : {last10:.3f}\n"
        f"  peak_reward   : {peak:.3f}\n"
        "  zero_loss     : 0/30\n"
    )


def _run(results_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--results-dir", str(results_dir), *extra],
        capture_output=True,
        text=True,
    )


def test_cli_log_within_tolerance_passes(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.log").write_text(_cli_log(0.350, 0.600))
    r = _run(tmp_path, "--expected-results", str(EXPECTED))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1/1 experiments within tolerance" in r.stdout


def test_cli_log_outside_tolerance_fails(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.log").write_text(_cli_log(0.010, 0.050))
    r = _run(tmp_path)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "0/1 experiments within tolerance" in r.stdout


def test_multi_seed_log_checks_every_seed(tmp_path):
    text = _cli_log(0.340, 0.620, seed=42) + _cli_log(0.010, 0.050, seed=123)
    (tmp_path / "gsm8k_qwen3_8b.log").write_text(text)
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "1/2 experiments within tolerance" in r.stdout


def test_legacy_percent_format_still_parsed(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.log").write_text(
        "Last-10 avg accuracy: 34.4%\nPeak accuracy:        62.5%\n"
    )
    assert _run(tmp_path).returncode == 0


def test_no_matching_result_fails(tmp_path):
    (tmp_path / "unrelated.log").write_text("nothing here\n")
    (tmp_path / "other.json").write_text(json.dumps([1, 2, 3]))
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "FAIL" in r.stderr


def test_missing_expectations_file_fails(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.log").write_text(_cli_log(0.344, 0.625))
    r = _run(tmp_path, "--expected-results", str(tmp_path / "nope.json"))
    assert r.returncode == 2


def write_expected(tmp_path, spec):
    path = tmp_path / "expectations.fixture"
    path.write_text(json.dumps({"gsm8k_qwen3_8b": spec}))
    return ["--expected-results", str(path)]


def test_different_seed_does_not_use_headline_even_with_equal_scores(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b.log").write_text(_cli_log(0.344, 0.625, seed=123))
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "UNVERIFIED" in r.stdout
    assert "no applicable reference for seed 123" in r.stderr


def test_explicit_seed_references_accept_differing_scores(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b.log").write_text(
        _cli_log(0.344, 0.625, 42) + _cli_log(0.1, 0.2, 123)
    )
    args = write_expected(
        tmp_path,
        {"seeds": {"42": {"last10": 0.344, "peak": 0.625}, "123": {"last10": 0.1, "peak": 0.2}}},
    )
    r = _run(tmp_path, *args, "--strict")
    assert r.returncode == 0, r.stderr
    assert "2/2 experiments within tolerance" in r.stdout


def test_strict_requires_every_declared_seed(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b.log").write_text(_cli_log(0.344, 0.625))
    args = write_expected(
        tmp_path,
        {"seeds": {"42": {"last10": 0.344, "peak": 0.625}, "123": {"last10": 0.1, "peak": 0.2}}},
    )
    assert _run(tmp_path, *args).returncode == 0
    r = _run(tmp_path, *args, "--strict")
    assert r.returncode == 1
    assert "seed=123" in r.stdout


def test_missing_seed_and_unscoped_reference_are_unverified(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b.json").write_text(
        json.dumps({"experiment": "gsm8k_qwen3_8b", "last10": 0.344, "peak": 0.625})
    )
    assert "UNVERIFIED" in _run(tmp_path).stdout
    args = write_expected(tmp_path, {"last10": 0.344, "peak": 0.625})
    r = _run(tmp_path, *args)
    assert r.returncode == 1
    assert "UNVERIFIED" in r.stdout


def test_explicit_seed_independent_reference_is_opt_in(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b.log").write_text(_cli_log(0.344, 0.625, 123))
    args = write_expected(tmp_path, {"last10": 0.344, "peak": 0.625, "seed_independent": True})
    assert _run(tmp_path, *args).returncode == 0


def test_conflicting_json_seed_and_name_is_unverified(tmp_path):
    (tmp_path / "result.json").write_text(
        json.dumps(
            {"experiment": "gsm8k_qwen3_8b_s42", "seed": 123, "last10": 0.344, "peak": 0.625}
        )
    )
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "conflicting seed identities" in r.stderr


def test_bad_reference_and_bad_tolerance_are_usage_errors(tmp_path):
    args = write_expected(tmp_path, {"last10": True, "peak": 0.625, "seed": 42})
    assert _run(tmp_path, *args).returncode == 2
    assert _run(tmp_path, "--last10-tolerance", "nan").returncode == 2


def test_unknown_legacy_reference_is_not_a_seed_claim(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_g16.log").write_text(_cli_log(0.38, 0.719))
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "UNVERIFIED" in r.stdout


def test_filename_seed_conflict_is_not_hidden_by_experiment(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s123.json").write_text(
        json.dumps({"experiment": "gsm8k_qwen3_8b_s42", "last10": 0.344, "peak": 0.625})
    )
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "conflicting seed identities" in r.stderr


def test_boolean_results_are_not_numeric_accuracy(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.json").write_text(json.dumps({"last10": False, "peak": True}))
    args = write_expected(tmp_path, {"last10": 0, "peak": 1, "seed": 42})
    r = _run(tmp_path, *args)
    assert r.returncode == 1
    assert "result metrics must be finite numbers" in r.stderr


def test_incomplete_seed_block_is_not_dropped(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b.log").write_text(
        _cli_log(0.344, 0.625) + "[grpo_cli] Seed 123 done.\n  avg_last10 : 0.3\n"
    )
    r = _run(tmp_path)
    assert r.returncode == 1
    assert "1/2 experiments within tolerance" in r.stdout
    assert "UNVERIFIED" in r.stdout


def test_malformed_seed_result_cannot_be_dropped_beside_valid_result(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.log").write_text(_cli_log(0.344, 0.625))
    (tmp_path / "gsm8k_qwen3_8b_s123.json").write_text('{"last10":')
    r = _run(tmp_path)
    assert r.returncode == 2
    assert "could not parse result JSON" in r.stderr


def test_matching_wrong_shape_file_cannot_be_dropped(tmp_path):
    (tmp_path / "gsm8k_qwen3_8b_s42.log").write_text(_cli_log(0.344, 0.625))
    (tmp_path / "gsm8k_qwen3_8b_s123.json").write_text("[]")
    r = _run(tmp_path)
    assert r.returncode == 2
    assert "no parseable result" in r.stderr
