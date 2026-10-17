"""Local launch checks must never need credentials or execute an adapter."""

from dataclasses import asdict
import hashlib
import json
import socket
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo_cli as cli


def test_config_check_is_offline_and_honors_explicit_json_limits(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    path = tmp_path / "config.json"
    source = {
        "name": "synthetic-private-name",
        "steps": 1000,
        "seed": 42,
        "base_url": "https://example.test/synthetic-private-endpoint",
        "hf_owner": "synthetic-private-owner",
    }
    path.write_text(json.dumps(source))

    def blocked(*_args, **_kwargs):
        raise AssertionError("configuration check attempted network")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    flags = [
        "--check-config",
        "--json-config",
        str(path),
        "--steps",
        "1",
        "--seed",
        "17",
        "--no-resume",
    ]
    with patch.object(cli, "_build_dataset") as dataset, patch.object(cli, "run_grpo") as run:
        assert cli.main(flags) == 0
    output = capsys.readouterr().out
    receipt = json.loads(output)
    assert receipt["status"] == "config_checks_passed"
    assert receipt["schema_version"] == 1
    assert receipt["effective"]["steps_per_seed"] == 1
    assert receipt["effective"]["seed"] == 17
    assert receipt["effective"]["resume"] is False
    assert "no dataset" in receipt["scope"]
    assert "synthetic-private" not in output
    assert str(path) not in output
    config = cli.build_config(cli._parse_args(flags))
    expected = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    assert receipt["config_sha256"] == expected
    dataset.assert_not_called()
    run.assert_not_called()
    assert json.loads(path.read_text()) == source


def test_config_check_validates_revision_without_loading_factory(monkeypatch, capsys):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with patch.dict(
        cli.DATASET_FACTORIES, {"gsm8k": lambda **_kwargs: pytest.fail("factory executed")}
    ):
        assert (
            cli.main(
                ["--check-config", "--preset", "gsm8k", "--dataset-revision", "synthetic-revision"]
            )
            == 0
        )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["dataset"] == receipt["reward"] == "gsm8k"


@pytest.mark.parametrize(
    ("flags", "error", "match"),
    [
        (["--steps", "-1"], ValueError, "steps"),
        (["--dataset", "unknown"], SystemExit, "Unknown reward"),
        (["--reward", "unknown"], SystemExit, "Unknown reward"),
        (["--dataset-revision", "unused"], ValueError, "dataset revision"),
    ],
)
def test_invalid_config_check_never_reaches_execution(monkeypatch, flags, error, match):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with patch.object(cli, "_build_dataset") as dataset, patch.object(cli, "run_grpo") as run:
        with pytest.raises(error, match=match):
            cli.main(["--check-config", *flags])
    dataset.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("settings", [{"wandb_enabled": False}, {"paid_jobs_may_launch": False}])
def test_config_check_keeps_tracking_and_campaign_gates(tmp_path, monkeypatch, capsys, settings):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"name": "gated", **settings}))
    with patch.object(cli, "_build_dataset") as dataset, patch.object(cli, "run_grpo") as run:
        with pytest.raises(ValueError, match="mandatory|disables paid jobs"):
            cli.main(["--check-config", "--json-config", str(path)])
    dataset.assert_not_called()
    run.assert_not_called()
    assert not capsys.readouterr().out
