"""Checkpoint publication keeps the owner resolved before provider work."""

from types import SimpleNamespace
from unittest.mock import patch

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


def test_normalized_preflight_owner_is_used_for_repository_names(tmp_path):
    config = _config(tmp_path, hf_owner=" synthetic-owner ")
    api = SimpleNamespace(whoami=lambda **_kwargs: {"name": "authenticated-owner"})
    with patch.object(grpo, "_make_hf_api", return_value=(api, None)):
        resolved = grpo._preflight_hf(config)
    assert resolved == "synthetic-owner"
    repo = grpo._checkpoint_repo_id(config, resolved, config.seed, 1, "tinker://synthetic")
    assert repo.startswith("synthetic-owner/")


def test_environment_owner_change_after_preflight_cannot_redirect_publication(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_REPO_OWNER", "first-owner")
    config = _config(tmp_path, save_every=1)
    with _runtime(monkeypatch) as holder:
        original = holder["run"].log

        def change_environment(payload):
            original(payload)
            if "train/step" in payload:
                monkeypatch.setenv("HF_REPO_OWNER", "second-owner")
            return True

        holder["run"].log = change_environment
        with patch.object(grpo, "_prepare_hf_revision") as prepare:
            result = grpo._run_one_seed(
                config,
                grpo.InMemoryDataset(train=[_example()]),
                grpo.ExactMathReward(),
                _Tokenizer(),
                logger=lambda _: None,
            )
    assert len(prepare.call_args_list) == 3
    assert all(call.args[1].startswith("first-owner/") for call in prepare.call_args_list)
    assert all("/first-owner/" in url for url in result.checkpoint_urls)


def test_empty_explicit_owner_fails_before_publication(tmp_path):
    import pytest

    config = _config(tmp_path)
    with patch.object(grpo, "_prepare_hf_revision") as prepare:
        with pytest.raises(ValueError, match="resolved checkpoint owner"):
            grpo._publish_checkpoint(
                config, config.seed, "tinker://synthetic", lambda _: None, hf_owner=" "
            )
    prepare.assert_not_called()
