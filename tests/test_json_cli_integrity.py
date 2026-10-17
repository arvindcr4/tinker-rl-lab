"""Explicit launch limits and unambiguous JSON must reach the runner."""

import json
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo_cli as cli


def test_json_config_explicit_limits_reach_runner(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    original = json.dumps({"name": "budget", "steps": 1000, "seed": 42, "lr": 0.0002})
    path.write_text(original)
    monkeypatch.setenv("TINKER_API_KEY", "synthetic-not-a-real-key")
    with patch.object(cli, "_build_dataset", return_value=object()) as dataset:
        with patch.object(cli, "run_grpo", return_value=[]) as run:
            assert (
                cli.main(
                    ["--json-config", str(path), "--steps", "1", "--seed", "17", "--no-resume"]
                )
                == 0
            )
    config = run.call_args.args[0]
    assert (config.steps, config.seed, config.resume) == (1, 17, False)
    assert (config.name, config.lr) == ("budget", 0.0002)
    assert dataset.call_args.args[1] is config
    assert path.read_text() == original


def test_json_only_values_and_existing_opt_in_flags_are_preserved(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"name": "json-only", "steps": 7, "resume": False, "hf_public": False})
    )
    config = cli.build_config(cli._parse_args(["--json-config", str(path)]))
    assert (config.steps, config.resume, config.hf_public) == (7, False, False)
    changed = cli.build_config(
        cli._parse_args(["--json-config", str(path), "--hf-public", "--nll-aux-coef", "0.2"])
    )
    assert changed.hf_public is True
    assert changed.nll_aux_enabled is True
    assert changed.nll_aux_coef == 0.2


@pytest.mark.parametrize(
    "raw",
    [
        '{"name":"secret-marker","hf_public":false,"hf_public":true}',
        '{"name":"secret-marker","nested":{"key":1,"key":2}}',
        '{"name":"secret-marker","lr":NaN}',
        '{"name":"secret-marker","lr":Infinity}',
        '{"name":"secret-marker","lr":1e999}',
        '[{"name":"secret-marker"}]',
        '{"name":"secret-marker",}',
    ],
)
def test_ambiguous_or_invalid_json_fails_before_dataset(tmp_path, monkeypatch, raw):
    path = tmp_path / "config.json"
    path.write_text(raw)
    monkeypatch.setenv("TINKER_API_KEY", "synthetic-not-a-real-key")
    with patch.object(cli, "_build_dataset") as dataset, patch.object(cli, "run_grpo") as run:
        with pytest.raises(ValueError, match="JSON config must") as error:
            cli.main(["--json-config", str(path)])
    assert "secret-marker" not in str(error.value)
    dataset.assert_not_called()
    run.assert_not_called()
    assert path.read_text() == raw


def test_json_unknown_fields_remain_errors(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"name":"unknown","typo_steps":1}')
    with pytest.raises(TypeError, match="typo_steps"):
        cli.build_config(cli._parse_args(["--json-config", str(path)]))
