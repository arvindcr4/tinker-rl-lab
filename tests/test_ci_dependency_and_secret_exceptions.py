"""Offline CI dependency and exact noncredential scanner exceptions."""

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ci_dev_extra_contains_collection_dependencies_in_lock():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = project["project"]["optional-dependencies"]["dev"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    package = next(p for p in lock["package"] if p["name"] == project["project"]["name"])
    locked_dev = {p["name"] for p in package["optional-dependencies"]["dev"]}
    for name in ("tenacity", "scikit-learn", "jsonschema", "fastapi", "httpx"):
        assert any(item.startswith(name + ">=") for item in dev)
        assert name in locked_dev


@pytest.mark.parametrize(
    ("path", "needle"),
    [
        ("platform_modal/scripts/p5p8/p7_n10_replication.py", "zvf_triage_0.7"),
        ("finish/progress_reporting_audit/generate_v1.py", "foreign/duplicate native SWE task"),
        (
            "reports/runtime_followup_2026-10-11/GSM8K_TOKEN_BUDGET_AUDIT.json",
            "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        ),
    ],
)
def test_exact_noncredential_exceptions_do_not_hide_changed_values(tmp_path, path, needle):
    scanner = shutil.which("gitleaks")
    if scanner is None:
        if os.environ.get("CI"):
            pytest.fail("CI requires gitleaks for security exception regressions")
        pytest.skip("optional gitleaks executable is unavailable")
    lines = (ROOT / path).read_text().splitlines()
    line = next(line for line in reversed(lines) if needle in line)
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    report = tmp_path.parent / (tmp_path.name + "-scan.json")

    def scan(text):
        target.write_text(text + "\n")
        result = subprocess.run(
            [
                scanner,
                "dir",
                "--config",
                str(ROOT / ".gitleaks.toml"),
                "--redact",
                "--no-banner",
                "--report-format",
                "json",
                "--report-path",
                str(report),
                ".",
            ],
            capture_output=True,
            cwd=tmp_path,
            text=True,
            check=False,
        )
        return result.returncode, json.loads(report.read_text())

    code, findings = scan(line)
    assert code == 0
    assert findings == []
    # Deterministic invented credential-shaped material, never a real secret.
    synthetic = base64.b64encode(
        hashlib.sha256(b"synthetic-only-security-regression").digest()
    ).decode()
    code, findings = scan(line.replace(needle, synthetic))
    assert code == 1
    assert any(item["RuleID"] == "generic-api-key" for item in findings)
