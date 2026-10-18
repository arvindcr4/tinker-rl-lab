"""Developer harness receipts must fail closed, including skips and publish races."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import run_offline_harness as harness


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "fixture.py"
    source.write_text("# synthetic source\n")
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "SOURCES", ("fixture.py",))
    monkeypatch.setattr(harness.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda _name: "1.0")
    monkeypatch.setattr(
        harness, "_locked_versions", lambda: {name: "1.0" for name in harness.DEPENDENCIES}
    )
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        junit = Path(
            next(part.split("=", 1)[1] for part in command if part.startswith("--junitxml="))
        )
        junit.write_text(
            "<testsuites><testsuite>" + "<testcase/>" * 5 + "</testsuite></testsuites>"
        )
        return SimpleNamespace(returncode=0, stdout="synthetic test output", stderr="")

    monkeypatch.setattr(harness.subprocess, "run", run)
    return source, calls, run


def test_success_receipt_and_child_environment(tmp_path, monkeypatch, setup):
    source, calls, _ = setup
    monkeypatch.setenv("TINKER_API_KEY", "synthetic-placeholder")
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "synthetic-placeholder")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--unexpected-options")
    output = tmp_path / "receipt.json"
    assert harness.main(["--out", str(output)]) == 0
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "passed"
    assert receipt["sources_unchanged"] is True
    assert receipt["counts"] == {"tests": 5, "failures": 0, "errors": 0, "skipped": 0}
    assert set(receipt["source_sha256"]) == {source.name}
    environment = calls[0][1]["env"]
    assert "TINKER_API_KEY" not in environment
    assert "HUGGINGFACE_HUB_TOKEN" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert environment["PYTEST_ADDOPTS"] == ""
    assert not list(tmp_path.glob(".offline-harness-*"))


@pytest.mark.parametrize(
    "kind",
    [
        "short",
        "skipped",
        "error",
        "failure",
        "bad_xml",
        "dtd",
        "returncode",
        "timeout",
        "changed_source",
    ],
)
def test_missing_or_failed_evidence_cannot_pass(tmp_path, monkeypatch, setup, kind):
    source, _, normal = setup

    def run(command, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired(command, 300)
        result = normal(command, **kwargs)
        junit = Path(
            next(part.split("=", 1)[1] for part in command if part.startswith("--junitxml="))
        )
        if kind == "short":
            junit.write_text("<testsuites><testcase/></testsuites>")
        elif kind in {"skipped", "error", "failure"}:
            junit.write_text(
                f"<testsuites><testcase><{kind}/></testcase>" + "<testcase/>" * 4 + "</testsuites>"
            )
        elif kind == "bad_xml":
            junit.write_text("not XML")
        elif kind == "dtd":
            junit.write_text('<!DOCTYPE x [<!ENTITY x "no">]><testsuites/>')
        elif kind == "returncode":
            result.returncode = 1
        elif kind == "changed_source":
            source.write_text("# changed during execution\n")
        return result

    monkeypatch.setattr(harness.subprocess, "run", run)
    output = tmp_path / "receipt.json"
    assert harness.main(["--out", str(output)]) == 1
    assert json.loads(output.read_text())["status"] == "failed"


def test_publication_race_preserves_competing_receipt(tmp_path, monkeypatch, setup):
    _, _, normal = setup
    output = tmp_path / "receipt.json"

    def race(command, **kwargs):
        result = normal(command, **kwargs)
        output.write_text("competing evidence")
        return result

    monkeypatch.setattr(harness.subprocess, "run", race)
    with pytest.raises(FileExistsError):
        harness.main(["--out", str(output)])
    assert output.read_text() == "competing evidence"
    assert not list(tmp_path.glob(".offline-harness-*"))


def test_atomic_link_unavailable_stops_before_work(tmp_path, monkeypatch, setup):
    _, calls, _ = setup

    def unsupported(*_args):
        raise OSError("synthetic link unavailable")

    monkeypatch.setattr(harness.os, "link", unsupported)
    with pytest.raises(OSError, match="link unavailable"):
        harness.main(["--out", str(tmp_path / "receipt.json")])
    assert calls == []


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_output_is_rejected_before_work(tmp_path, setup, symlink):
    _, calls, _ = setup
    output = tmp_path / "receipt.json"
    if symlink:
        output.symlink_to(tmp_path / "missing.json")
    else:
        output.write_text("prior evidence")
    with pytest.raises(SystemExit) as raised:
        harness.main(["--out", str(output)])
    assert raised.value.code == 2
    assert calls == []


def test_missing_dependencies_have_actionable_error(tmp_path, monkeypatch, setup, capsys):
    _, calls, _ = setup
    monkeypatch.setattr(harness.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(SystemExit) as raised:
        harness.main(["--out", str(tmp_path / "receipt.json")])
    assert raised.value.code == 2
    assert "uv sync --frozen --extra dev --python 3.12" in capsys.readouterr().err
    assert calls == []


def test_unlocked_runtime_is_rejected_before_running(tmp_path, monkeypatch, setup, capsys):
    _, calls, _ = setup
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda _name: "2.0")
    output = tmp_path / "receipt.json"
    with pytest.raises(SystemExit) as failure:
        harness.main(["--out", str(output)])
    assert failure.value.code == 2
    assert "runtime differs from uv.lock" in capsys.readouterr().err
    assert not calls
    assert not output.exists()


def test_lock_reader_extracts_unique_required_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "DEPENDENCIES", ("torch",))
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "torch"\nversion = "2.7.1"\n')
    assert harness._locked_versions() == {"torch": "2.7.1"}
    lock.write_text('[[package]]\nname = "other"\nversion = "1.0"\n')
    with pytest.raises(ValueError, match="one version"):
        harness._locked_versions()
