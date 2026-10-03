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
