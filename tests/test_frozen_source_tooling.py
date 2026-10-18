"""Style-tool regressions use temporary copies of provenance-bound source files."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def frozen_source_paths() -> list[str]:
    provenance = json.loads((ROOT / "research/public_analysis/PROVENANCE.json").read_text())
    protocol = json.loads(
        (ROOT / "zvf-program/experiments-next/rlhfbook_followup_preregistration.json").read_text()
    )
    return sorted(
        [f"research/public_analysis/{name}" for name in provenance["modules"]]
        + [protocol["scope"]["live_checkout_observation"]["path"]]
    )


@pytest.fixture
def source_copy(tmp_path: Path) -> Path:
    for name in ["pyproject.toml", *frozen_source_paths()]:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return tmp_path


def run_ruff(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_formatter_protection_names_only_the_bound_sources() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]
    assert config["force-exclude"] is True
    assert set(frozen_source_paths()) <= set(config["format"]["exclude"])


def test_format_and_unsafe_autofix_preserve_bound_source_bytes(source_copy: Path) -> None:
    paths = frozen_source_paths()
    originals = {name: (source_copy / name).read_bytes() for name in paths}
    # Candidate modules in the same package must remain formatter-managed.
    candidate = source_copy / "research/public_analysis/candidate_fixture.py"
    candidate.write_text("def example( ):return 1\n", encoding="utf-8")
    result = run_ruff(source_copy, "format", *paths, str(candidate))
    assert result.returncode == 0, result.stderr + result.stdout
    assert candidate.read_text() == "def example():\n    return 1\n"
    assert originals == {name: (source_copy / name).read_bytes() for name in paths}

    # Do not silence lint to protect provenance. Future unsafe fixes that alter
    # these definitions must fail this regression rather than refresh the hashes.
    result = run_ruff(source_copy, "check", "--fix", "--unsafe-fixes", *paths)
    assert result.returncode == 0, result.stderr + result.stdout
    assert originals == {name: (source_copy / name).read_bytes() for name in paths}


def test_formatter_protection_keeps_undefined_name_checks_enabled(source_copy: Path) -> None:
    paths = frozen_source_paths()
    for name in paths:
        with (source_copy / name).open("a", encoding="utf-8") as stream:
            stream.write("\nfrozen_source_undefined_name()\n")
    result = run_ruff(source_copy, "check", "--select", "F821", "--output-format", "json", *paths)
    assert result.returncode == 1, result.stderr + result.stdout
    diagnostics = json.loads(result.stdout)
    assert {item["code"] for item in diagnostics} == {"F821"}
    assert {
        Path(item["filename"]).relative_to(source_copy).as_posix() for item in diagnostics
    } == set(paths)
