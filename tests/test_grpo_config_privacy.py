"""Reject ambiguous visibility flags and credential-bearing endpoint overrides."""

from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import pytest

from platform_tinker.tinkerrl import grpo


def test_boolean_options_do_not_accept_truthy_strings_or_numbers():
    for field in fields(grpo.GRPOConfig):
        if not isinstance(field.default, bool):
            continue
        for value in ("false", "true", 0, 1, None, [], {}):
            with pytest.raises(ValueError, match=field.name):
                grpo.GRPOConfig(name="typed-flags", **{field.name: value})


def test_mutated_public_flag_fails_before_repository_creation():
    config = grpo.GRPOConfig(name="private-run")
    config.hf_public = "false"
    with patch.object(
        grpo, "_prepare_hf_revision", side_effect=AssertionError("repository creation attempted")
    ) as create:
        with pytest.raises(ValueError, match="hf_public"):
            grpo._publish_checkpoint(
                config,
                42,
                "tinker://synthetic/checkpoint",
                lambda _: None,
                hf_owner="synthetic-owner",
            )
    create.assert_not_called()


@pytest.mark.parametrize(
    "url",
    [
        "https://synthetic-user:synthetic-password@example.test/api",
        "https://synthetic-user@example.test/api",
        "https://synthetic-password@[invalid",
        "https://example.test:bad",
        "https://example.test:0",
        "https://example.test:",
        "ftp://example.test/api",
        "https://",
        "",
        123,
        " https://example.test/api",
        "https://example.test/a\nb",
        "https://example.test\\api",
        "https://example.test/api#fragment",
    ],
)
def test_endpoint_rejection_never_echoes_userinfo(url):
    with pytest.raises(ValueError, match="base_url") as failure:
        grpo.GRPOConfig(name="bad-url", base_url=url)
    assert "synthetic-password" not in str(failure.value)
    assert "synthetic-user" not in str(failure.value)
    assert failure.value.__cause__ is None


@pytest.mark.parametrize(
    "url",
    [
        None,
        "https://example.test/api/v1",
        "http://localhost:8001/v1",
        "http://[::1]:8001/api",
        "https://example.test/api?region=test",
    ],
)
def test_valid_endpoint_forms_are_preserved(url):
    assert grpo.GRPOConfig(name="valid-url", base_url=url).base_url == url


def test_mutated_endpoint_fails_before_tracking_serialization(monkeypatch):
    config = grpo.GRPOConfig(name="mutated-url")
    config.base_url = "https://synthetic-user:synthetic-password@example.test/api"
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_DISABLED", raising=False)
    tracking = SimpleNamespace(init=Mock(side_effect=AssertionError("tracking attempted")))
    with patch.dict(sys.modules, {"wandb": tracking}):
        with pytest.raises(ValueError, match="base_url"):
            grpo._start_wandb(config, config.seed)
    tracking.init.assert_not_called()
    with pytest.raises(ValueError, match="base_url"):
        grpo._config_fingerprint(config, config.seed)
    dataset, reward = Mock(), Mock()
    with pytest.raises(ValueError, match="base_url"):
        grpo._run_one_seed(config, dataset, reward, tokenizer=None)
    assert dataset.mock_calls == []
    assert reward.mock_calls == []
