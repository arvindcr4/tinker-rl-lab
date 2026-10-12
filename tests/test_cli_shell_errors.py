"""The documented module command should explain common input failures concisely."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ["negative_steps", "unknown_field", "missing_file"])
def test_module_configuration_errors_are_actionable_without_traceback(tmp_path, case):
    arguments = ["--check-config"]
    if case == "negative_steps":
        arguments += ["--steps", "-1"]
        expected = "steps must be an integer"
    else:
        path = tmp_path / "config.json"
        if case == "unknown_field":
            path.write_text('{"name":"synthetic","typo_steps":1}')
            expected = "typo_steps"
        else:
            expected = "config.json"
        arguments += ["--json-config", str(path)]
    environment = dict(os.environ)
    environment.pop("TINKER_API_KEY", None)
    environment.update(HF_HUB_OFFLINE="1", WANDB_MODE="offline", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, "-m", "platform_tinker.tinkerrl.grpo_cli", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert expected in result.stderr
    assert "ERROR:" in result.stderr
    assert "Traceback" not in result.stderr
    assert "platform_tinker.tinkerrl.grpo_cli --help" in result.stderr


@pytest.mark.parametrize("token_variable", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_shell_error_redacts_configured_secret_but_library_error_is_preserved(
    monkeypatch, capsys, token_variable
):
    from platform_tinker.tinkerrl import grpo_cli as cli

    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(token_variable, "synthetic-secret-placeholder")

    def invalid(_argv):
        raise ValueError("invalid request synthetic-secret-placeholder")

    monkeypatch.setattr(cli, "main", invalid)
    assert cli.cli_entrypoint([]) == 1
    error = capsys.readouterr().err
    assert "[redacted]" in error
    assert "synthetic-secret-placeholder" not in error
    with pytest.raises(ValueError, match="invalid request"):
        cli.main([])


def test_unexpected_runtime_failure_is_not_disguised_as_input_error(monkeypatch):
    from platform_tinker.tinkerrl import grpo_cli as cli

    def unexpected(_argv):
        raise RuntimeError("synthetic internal failure")

    monkeypatch.setattr(cli, "main", unexpected)
    with pytest.raises(RuntimeError, match="synthetic internal failure"):
        cli.cli_entrypoint([])
