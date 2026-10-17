"""Execute only checked child-wait fragments, never a campaign launcher."""

from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2] / "platform_hybrid/experiments/10x_structural_ceiling"


@pytest.mark.parametrize("name", ["run_round2.sh", "run_round2_resume.sh"])
def test_every_child_failure_propagates(name):
    text = (ROOT / name).read_text()
    fragment = text[text.index("failed=0") :]
    prefix = ""
    for i, label in enumerate(["1A", "1B", "1C", "2", "3A", "3B", "3C", "4"]):
        prefix += f"(exit {1 if i == 0 else 0}) & PID_{label}=$!\n"
    result = subprocess.run(
        ["bash", "-c", prefix + fragment], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 1
    assert "COMPLETE" not in result.stdout


@pytest.mark.parametrize("name", ["run_all.sh", "run_parallel.sh", "run_incomplete.sh"])
def test_counted_failure_is_nonzero(name):
    text = (ROOT / name).read_text()
    fragment = text[text.index("# A partial campaign") :]
    result = subprocess.run(
        ["bash", "-c", "FAIL_COUNT=1\n" + fragment], capture_output=True, timeout=5
    )
    assert result.returncode == 1
