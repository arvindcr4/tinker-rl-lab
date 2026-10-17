"""Inspect evaluation requests without credentials, downloads or filesystem writes."""

import json
import socket
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import pavlov_portfolio_eval as evaluator
from tests import test_portfolio_eval_integrity

harness = test_portfolio_eval_integrity.harness


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    def forbidden(*_a, **_k):
        raise AssertionError("evaluation configuration inspection must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("trained", [False, True])
def test_configuration_check_does_not_create_output_or_load_data(tmp_path, capsys, trained):
    output = tmp_path / "not-created" / "private-output.json"
    flags = [
        "--check-config",
        "--out",
        str(output),
        "--per-suite",
        "3",
        "--max-tokens",
        "17",
        "--seed",
        "12",
    ]
    if trained:
        flags += [
            "--sampler-path",
            "tinker://private-model",
            "--hf-repo",
            "private-owner/private-repo",
            "--hf-revision",
            "private-revision",
            "--hf-commit",
            "a" * 40,
        ]
    else:
        flags += ["--base-model"]
    with (
        patch.object(evaluator, "_preflight_checkpoint_storage") as storage,
        patch.object(evaluator, "make_pavlov_non_xlam_dataset") as dataset,
    ):
        assert evaluator.main(flags) == 0
    storage.assert_not_called()
    dataset.assert_not_called()
    assert not output.parent.exists()
    raw = capsys.readouterr().out
    result = json.loads(raw)
    assert result["status"] == "config_checks_passed"
    assert result["source_mode"] == ("checkpoint_path" if trained else "base_model_name")
    assert result["sampling_request"]["planned_examples"] == 6
    assert result["sampling_request"]["max_tokens"] == 17
    assert result["sampling_request"]["seed"] == 12
    assert len(result["config_sha256"]) == 64
    assert "private" not in raw
    assert str(tmp_path) not in raw
    assert "storage_writability" in result["not_checked"]
    assert "provider_weights" in result["not_checked"]


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (["--per-suite", "0"], "must be positive"),
        (["--temperature", "nan"], "finite and nonnegative"),
        (["--hf-repo", "unused"], "references require --sampler-path"),
    ],
)
def test_configuration_check_preserves_existing_input_rejections(tmp_path, flags, reason, capsys):
    with patch.object(evaluator, "make_pavlov_non_xlam_dataset") as dataset:
        with pytest.raises(SystemExit) as error:
            evaluator.main(
                ["--check-config", "--base-model", "--out", str(tmp_path / "out.json"), *flags]
            )
    assert error.value.code == 2
    dataset.assert_not_called()
    assert reason in capsys.readouterr().err


def test_configuration_check_rejects_existing_receipt_without_credentials(tmp_path, capsys):
    output = tmp_path / "existing.json"
    output.write_text("prior evidence")
    with pytest.raises(SystemExit) as error:
        evaluator.main(["--check-config", "--base-model", "--out", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "prior evidence"
    assert "choose a fresh output path" in capsys.readouterr().err


def test_inspection_fingerprint_binds_the_executed_request_and_receipt(
    harness, capsys, monkeypatch
):
    assert harness.invoke("--check-config") == 0
    inspection = json.loads(capsys.readouterr().out)
    assert harness.calls == []
    assert harness.state.tracking_config is None
    assert not harness.output.exists()
    assert harness.invoke() == 0
    receipt = json.loads(harness.output.read_text())
    assert receipt["request_sha256"] == inspection["config_sha256"]
    assert harness.state.tracking_config["request_sha256"] == receipt["request_sha256"]
    for key in ("temperature", "top_p", "max_prompt_tokens", "max_tokens", "seed"):
        assert receipt["sampling"][key] == inspection["sampling_request"][key]
    args = evaluator._parser().parse_args(
        ["--base-model", "--per-suite", "1", "--out", str(harness.output)]
    )
    monkeypatch.setattr(evaluator, "MODEL_REVISION", "b" * 40)
    assert evaluator._configuration_check(args)["config_sha256"] != receipt["request_sha256"]
