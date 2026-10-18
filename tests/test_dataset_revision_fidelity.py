"""Declared source revisions must be applied or rejected before data loading."""

import sys
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from platform_tinker.tinkerrl import grpo, grpo_cli as cli


@pytest.mark.parametrize("revision", [None, "synthetic-revision"])
def test_gsm8k_revision_reaches_both_splits(monkeypatch, revision):
    datasets = ModuleType("datasets")
    loader = Mock(return_value=[{"question": "One plus one?", "answer": "#### 2"}])
    datasets.load_dataset = loader
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    kwargs = {} if revision is None else {"revision": revision}
    data = grpo.make_gsm8k_dataset(seed=17, **kwargs)
    assert len(data.train) == len(data.test) == 1
    assert loader.call_count == 2
    for call, split in zip(loader.call_args_list, ["train", "test"], strict=True):
        assert call.args == ("openai/gsm8k", "main")
        assert call.kwargs == {"split": split, **kwargs}


@pytest.mark.parametrize("revision", [None, "synthetic-revision"])
def test_gsm8k_cli_forwards_revision_without_changing_default(revision):
    flags = ["--preset", "gsm8k", "--seed", "17"]
    if revision is not None:
        flags += ["--dataset-revision", revision]
    args = cli._parse_args(flags)
    config = cli.build_config(args)
    factory = Mock()
    with patch.dict(cli.DATASET_FACTORIES, {"gsm8k": factory}):
        cli._build_dataset(args, config)
    expected = {"seed": 17}
    if revision is not None:
        expected["revision"] = revision
    factory.assert_called_once_with(**expected)


@pytest.mark.parametrize("name", ["pavlov_portfolio", "tooluse_synth", "math100"])
def test_unsupported_revision_fails_before_factory(name):
    args = cli._parse_args(["--preset", name, "--dataset-revision", "unsupported-revision"])
    config = cli.build_config(args)
    factory = Mock()
    with patch.dict(cli.DATASET_FACTORIES, {name: factory}):
        with pytest.raises(ValueError, match="dataset revision"):
            cli._build_dataset(args, config)
    factory.assert_not_called()


def test_portfolio_preserves_its_declared_composite_revision():
    args = cli._parse_args(["--preset", "pavlov_portfolio"])
    config = cli.build_config(args)
    assert config.dataset_revision == grpo.PAVLOV_NON_XLAM_DATASET_REVISION
    factory = Mock()
    with patch.dict(cli.DATASET_FACTORIES, {"pavlov_portfolio": factory}):
        cli._build_dataset(args, config)
    factory.assert_called_once_with(seed=config.seed)
