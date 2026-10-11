"""Changing adapters cannot silently reuse another preset's source pin."""

import json
from unittest.mock import Mock, patch

import pytest

from platform_tinker.tinkerrl import grpo_cli as cli


@pytest.mark.parametrize("dataset", ["pavlov_xlam", "gsm8k"])
def test_cross_adapter_override_rejects_inherited_pin(dataset):
    args = cli._parse_args(["--preset", "pavlov_portfolio", "--dataset", dataset])
    config = cli.build_config(args)
    factory = Mock()
    with patch.dict(cli.DATASET_FACTORIES, {dataset: factory}):
        with pytest.raises(ValueError, match="inherited revision"):
            cli._build_dataset(args, config)
    factory.assert_not_called()


def test_explicit_replacement_revision_is_forwarded():
    args = cli._parse_args(
        [
            "--preset",
            "pavlov_portfolio",
            "--dataset",
            "pavlov_xlam",
            "--dataset-revision",
            "explicit-xlam-revision",
        ]
    )
    config = cli.build_config(args)
    factory, kwargs = cli._dataset_request(args, config)
    assert factory is cli.DATASET_FACTORIES["pavlov_xlam"]
    assert kwargs["revision"] == "explicit-xlam-revision"


def test_same_source_alias_keeps_its_valid_inherited_pin():
    args = cli._parse_args(["--preset", "pavlov_xlam", "--dataset", "tooluse_xlam"])
    config = cli.build_config(args)
    _, kwargs = cli._dataset_request(args, config)
    assert kwargs["revision"] == cli.PRESETS["pavlov_xlam"]["dataset_revision"]


def test_json_config_revision_is_explicit_not_inherited(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "explicit-json", "dataset_revision": "json-xlam-revision"}))
    args = cli._parse_args(
        ["--preset", "pavlov_portfolio", "--dataset", "pavlov_xlam", "--json-config", str(path)]
    )
    _, kwargs = cli._dataset_request(args, cli.build_config(args))
    assert kwargs["revision"] == "json-xlam-revision"


def test_offline_check_rejects_cross_adapter_pin_before_success(monkeypatch, capsys):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="inherited revision"):
        cli.main(["--check-config", "--preset", "pavlov_portfolio", "--dataset", "pavlov_xlam"])
    assert not capsys.readouterr().out
