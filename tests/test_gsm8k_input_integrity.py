"""Malformed GSM8K rows must not shrink or relabel an evaluation silently."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from platform_tinker.tinkerrl import grpo


def _source(monkeypatch, bad_row, split):
    datasets = ModuleType("datasets")
    good = {"question": "Synthetic addition?", "answer": "work #### 2"}
    loader = Mock(
        side_effect=lambda *_args, **kwargs: [good, bad_row] if kwargs["split"] == split else [good]
    )
    datasets.load_dataset = loader
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    return loader


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize(
    "row",
    [
        {"question": "Missing", "answer": "no final marker"},
        {"question": "Fraction", "answer": "#### 1/2"},
        {"question": "Ambiguous", "answer": "#### 2 or 3"},
        {"question": "Invalid grouping", "answer": "#### 1,23"},
        {"question": " ", "answer": "#### 2"},
    ],
)
def test_malformed_row_fails_with_split_and_row_without_disclosing_text(monkeypatch, split, row):
    _source(monkeypatch, row, split)
    with pytest.raises(ValueError, match=f"GSM8K {split} row 2") as error:
        grpo.make_gsm8k_dataset()
    assert row["answer"] not in str(error.value)


def test_signed_grouped_decimal_and_exponent_labels_are_complete(monkeypatch):
    datasets = ModuleType("datasets")
    values = ["+1,234.50", "-0.5", ".25", "2e3", "1."]
    datasets.load_dataset = lambda *_args, **_kwargs: [
        {"question": f"Synthetic {index}?", "answer": "work #### " + value}
        for index, value in enumerate(values)
    ]
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    result = grpo.make_gsm8k_dataset(seed=42)
    assert sorted(row.target for row in result.train) == sorted(v.replace(",", "") for v in values)
    assert result.train == result.test


@pytest.mark.parametrize("split", ["train", "test"])
def test_empty_source_is_not_reported_as_an_empty_benchmark(monkeypatch, split):
    datasets = ModuleType("datasets")
    datasets.load_dataset = lambda *_args, **kwargs: (
        [] if kwargs["split"] == split else [{"question": "Synthetic?", "answer": "#### 2"}]
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    with pytest.raises(ValueError, match=f"GSM8K {split} source has no examples"):
        grpo.make_gsm8k_dataset()
