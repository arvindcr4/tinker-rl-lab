"""JSON-facing identity and metadata inputs must retain their declared shape."""

from pathlib import Path
import socket
from unittest.mock import patch

import pytest

from platform_tinker.tinkerrl import grpo
from tests._shared_fakes import _config, _example, _Tokenizer
from tests.test_grpo_coverage import _runtime


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("config shape tests must not use network")

    monkeypatch.setattr(socket.socket, "connect", blocked)


METADATA_FIELDS = (
    "wandb_tags",
    "training_suite_ids",
    "heldout_suite_ids",
    "held_out_suite_ids",
    "primary_evaluation_suite_ids",
    "domain_tags",
    "declared_domains",
    "training_domain_union",
    "primary_evaluation_domain_union",
)


def test_metadata_sequences_never_split_scalars_or_accept_wrong_element_types():
    for field in METADATA_FIELDS:
        for value in ("tag", {}, {"tag"}, False, [1], [""], ["   "]):
            with pytest.raises(ValueError, match=field):
                grpo.GRPOConfig(name="synthetic", **{field: value})


def test_metadata_lists_are_copied_and_optional_none_stays_empty():
    for field in METADATA_FIELDS:
        values = ["first", "second"]
        config = grpo.GRPOConfig(name="synthetic", **{field: values})
        values.append("later")
        assert getattr(config, field) == ("first", "second")
        assert getattr(grpo.GRPOConfig(name="synthetic", **{field: None}), field) == ()


def test_identity_and_optional_revision_strings_reject_ambiguous_values():
    required = ("name", "model", "checkpoint_dir")
    optional = (
        "model_revision",
        "dataset_revision",
        "hf_owner",
        "hf_repo_prefix",
        "checkpoint_name_prefix",
        "wandb_entity",
    )
    for field in required + optional:
        for value in (False, 1, "", "   ", []):
            kwargs = {"name": "synthetic", field: value}
            with pytest.raises(ValueError, match=field):
                grpo.GRPOConfig(**kwargs)
    for field in optional:
        assert getattr(grpo.GRPOConfig(name="synthetic", **{field: None}), field) is None


def test_pathlike_checkpoint_directory_has_json_compatible_config(tmp_path):
    config = grpo.GRPOConfig(name="path", checkpoint_dir=Path(tmp_path))
    assert config.checkpoint_dir == str(tmp_path)
    assert grpo._immutable_config(config, config.seed)["checkpoint_dir"] == str(tmp_path)


def test_mutated_metadata_fails_before_dataset_and_tracking(tmp_path):
    config = _config(tmp_path)
    config.domain_tags = "silently-split-before"
    dataset = grpo.InMemoryDataset(train=[_example()])
    with patch.object(grpo, "_start_wandb") as tracking:
        with patch.object(
            grpo.InMemoryDataset,
            "train_examples",
            side_effect=AssertionError("dataset read attempted"),
        ):
            with pytest.raises(ValueError, match="domain_tags"):
                grpo._run_one_seed(config, dataset, grpo.ExactMathReward(), _Tokenizer())
    tracking.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_validated_case_insensitive_mode_reaches_sdk_canonically(tmp_path, monkeypatch):
    config = _config(tmp_path, wandb_mode=" ONLINE ")
    with _runtime(monkeypatch) as holder:
        grpo._start_wandb(config, config.seed)
        import wandb

        assert wandb.init.call_args.kwargs["mode"] == "online"
        assert holder["run"] is wandb.init.return_value
