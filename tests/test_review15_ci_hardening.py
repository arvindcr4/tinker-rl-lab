"""Offline CI trust-boundary regressions; never download or run release binaries."""

import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION_REVISIONS = {
    # Verified against the official repositories' git/ref/tags APIs.
    "actions/checkout": "3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
}
GITLEAKS_SHA256 = "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"


def secret_step():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return next(s for s in workflow["jobs"]["secret-scan"]["steps"] if s.get("name") == "gitleaks")


def test_actions_are_immutable_and_permissions_read_only():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        workflow = yaml.safe_load(path.read_text())
        assert workflow["permissions"] == {"contents": "read"}
        for job in workflow["jobs"].values():
            for step in job["steps"]:
                if "uses" not in step:
                    continue
                action, revision = step["uses"].split("@")
                assert re.fullmatch(r"[a-f0-9]{40}", revision)
                assert revision == ACTION_REVISIONS[action]


def test_gitleaks_pins_official_asset_and_checks_before_extracting():
    step = secret_step()
    assert step["env"]["GITLEAKS_SHA256"] == GITLEAKS_SHA256
    script = step["run"]
    assert (
        "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz"
        in script
    )
    assert script.index("sha256sum -c -") < script.index("tar -xzf")
    assert script.startswith("set -euo pipefail\n")


@pytest.mark.parametrize(
    "case", ["valid", "digest_mismatch", "download_failure", "invalid_archive"]
)
def test_secret_scan_stops_before_unverified_execution(tmp_path, case):
    """Run the actual workflow shell with local fixtures and no network access."""
    marker = tmp_path / "executed"
    archive = tmp_path / "fixture.tar.gz"
    payload = b'#!/bin/sh\nprintf "%s\\n" "$*" > "$EXECUTED"\n'
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("gitleaks")
        info.size = len(payload)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(payload))
    if case == "invalid_archive":
        archive.write_bytes(b"not an archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    curl = bindir / "curl"
    curl.write_text(
        '#!/bin/sh\n[ "$DOWNLOAD_FAIL" = 0 ] || exit 22\n'
        'while [ "$1" != "-o" ]; do shift; done\ncp "$FIXTURE" "$2"\n'
    )
    curl.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path),
        "GITLEAKS_SHA256": "0" * 64 if case == "digest_mismatch" else digest,
        "FIXTURE": str(archive),
        "DOWNLOAD_FAIL": "1" if case == "download_failure" else "0",
        "EXECUTED": str(marker),
        "EVENT_NAME": "pull_request",
        "PR_BASE": "a" * 40,
        "HEAD_SHA": "b" * 40,
    }
    result = subprocess.run(
        ["bash", "-c", secret_step()["run"]],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if case == "valid":
        assert result.returncode == 0, result.stderr
        assert "--redact --no-banner" in marker.read_text()
        assert f"--log-opts={'a' * 40}..{'b' * 40}" in marker.read_text()
    else:
        assert result.returncode != 0
        assert not marker.exists()
        assert not (tmp_path / "gitleaks-bin/gitleaks").exists()


def test_python_target_and_name_safety_have_no_broad_exemptions():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["project"]["requires-python"] == ">=3.11"
    ruff = config["tool"]["ruff"]
    assert ruff["target-version"] == "py311"
    for codes in ruff["lint"]["per-file-ignores"].values():
        assert not {"F821", "F811"}.intersection(codes)
    makefile = (ROOT / "Makefile").read_text()
    assert (
        "$(RUFF) check --select F821,F811 platform_* autoresearch analysis utils tests tools"
        in makefile
    )


def test_all_ci_scanner_installations_verify_before_execution():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    steps = [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "gitleaks/releases/download" in step.get("run", "")
    ]
    assert len(steps) == 3
    for step in steps:
        assert step["env"]["GITLEAKS_SHA256"] == GITLEAKS_SHA256
        script = step["run"]
        assert script.startswith("set -euo pipefail\n")
        assert script.index("sha256sum -c -") < script.index("tar -xzf")
        assert "| tar" not in script


def test_three_paper_compilation_is_owned_only_by_docs_build():
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    compiles = [
        (job, step)
        for job, body in jobs.items()
        for step in body["steps"]
        if "tectonic -X compile" in step.get("run", "")
    ]
    assert len(compiles) == 1
    owner, step = compiles[0]
    assert owner == "docs-build"
    assert step["working-directory"] == "platform_tinker/reports/final"
    for paper in (
        "grpo_agentic_llm_paper.tex",
        "grpo_agentic_llm_paper_anonymous.tex",
        "supplementary_appendix.tex",
    ):
        assert step["run"].count(paper) == 1
    assert "--outdir" in step["run"]
    quality = jobs["quality"]["steps"]
    assert any(step.get("name") == "Coverage gate" for step in quality)
    assert any(step.get("name") == "Install checksum-pinned Tectonic" for step in quality)
    assert any(
        "pytest tests/ -m latex" in step.get("run", "") for step in jobs["docs-build"]["steps"]
    )
