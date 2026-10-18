"""The local gate must inspect staged and unstaged tracked changes."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("state", ["clean", "staged", "unstaged"])
def test_make_secret_gate_checks_both_git_diff_surfaces(tmp_path, state):
    scanner = shutil.which("gitleaks")
    if scanner is None:
        if os.environ.get("CI"):
            pytest.fail("CI requires gitleaks to exercise the local secret gate")
        pytest.skip("optional gitleaks executable is unavailable")
    if shutil.which("git") is None or shutil.which("make") is None:
        pytest.skip("git and make are required for the isolated scan contract")
    # This marker is an invented string, never a real credential. A dedicated
    # test-only rule avoids depending on a provider's current key formats.
    marker = "synthetic-scan-marker-no-credential"
    (tmp_path / "Makefile").write_bytes((ROOT / "Makefile").read_bytes())
    (tmp_path / ".gitleaks.toml").write_text(
        'title = "isolated synthetic rule"\n'
        '[[rules]]\nid = "synthetic-marker"\n'
        f"regex = '''{marker}'''\n"
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean fixture\n")

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("add", "Makefile", ".gitleaks.toml", "tracked.txt")
    git(
        "-c",
        "user.name=Synthetic Test",
        "-c",
        "user.email=synthetic@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "clean fixture",
    )
    if state != "clean":
        tracked.write_text(marker + "\n")
        if state == "staged":
            git("add", "tracked.txt")
    result = subprocess.run(
        ["make", "secrets", f"GITLEAKS={scanner}"], cwd=tmp_path, capture_output=True, text=True
    )
    if state == "clean":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0, f"{state} synthetic marker escaped the local gate"
        assert "leaks found" in result.stderr.lower()
    assert marker not in result.stdout + result.stderr


def test_ci_cannot_silently_skip_missing_scanner(tmp_path, monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(pytest.fail.Exception, match="CI requires gitleaks"):
        test_make_secret_gate_checks_both_git_diff_surfaces(tmp_path, "clean")
