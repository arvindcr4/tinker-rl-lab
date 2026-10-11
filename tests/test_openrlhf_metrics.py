"""Dependency-free checks of the native TensorBoard boundary; fixtures are synthetic."""

import hashlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from platform_modal.openrlhf.metrics import read_native_metrics


def fixture(tmp_path, values, tag="train/reward"):
    directory = tmp_path / "tensorboard/native"
    directory.mkdir(parents=True)
    event = directory / "events.out.tfevents.synthetic"
    event.write_bytes(b"synthetic event bytes, not real training")
    reader = Mock()
    reader.Reload.return_value = reader
    reader.Tags.return_value = {"scalars": [tag]}
    reader.Scalars.return_value = [SimpleNamespace(step=s, value=v) for s, v in values]
    module = ModuleType("tensorboard.backend.event_processing.event_accumulator")
    module.EventAccumulator = Mock(return_value=reader)
    return event, reader, module


def read(tmp_path, module, expected=2):
    with patch.dict(sys.modules, {module.__name__: module}):
        return read_native_metrics(str(tmp_path), "native", expected)


def test_records_full_trace_with_hash_and_distinct_step_semantics(tmp_path):
    event, _, module = fixture(tmp_path, [(1, 0.25), (2, 0.5)])
    result = read(tmp_path, module)
    assert result["reward_trace"] == [0.25, 0.5]
    assert result["logged_rollout_steps"] == 2
    assert result["steps"] is None
    assert result["metrics_status"] == "observed_training_rollout_rewards"
    assert result["metrics_source"]["sha256"] == hashlib.sha256(event.read_bytes()).hexdigest()
    assert result["metrics_source"]["tag"] == "train/reward"
    module.EventAccumulator.assert_called_once_with(
        str(event), size_guidance={"scalars": 0}, purge_orphaned_data=False
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [(1, 0.5)],
        [(1, 0.5), (3, 0.5)],
        [(1, 0.5), (1, 0.6)],
        [(2, 0.5), (1, 0.5)],
        [(1, float("nan")), (2, 0.5)],
        [(1, float("inf")), (2, 0.5)],
        [(1, 0.5), (2, 0.5), (3, 0.5)],
    ],
)
def test_missing_extra_duplicate_unordered_or_nonfinite_values_are_not_metrics(tmp_path, values):
    _, _, module = fixture(tmp_path, values)
    result = read(tmp_path, module)
    assert result["metrics_status"] == "not_collected"
    assert result["metrics_reason"] == "invalid_or_incomplete_reward_trace"
    assert result["reward_trace"] == []
    assert result["steps"] is None


def test_does_not_alias_unverified_reward_tags(tmp_path):
    _, _, module = fixture(tmp_path, [(1, 0.5)], tag="reward/mean")
    assert read(tmp_path, module)["metrics_reason"] == "native_reward_tag_missing"


def test_multiple_writers_are_ambiguous(tmp_path):
    event, reader, module = fixture(tmp_path, [(1, 0.5), (2, 0.5)])
    event.with_name("events.out.tfevents.other").write_bytes(b"other")
    assert read(tmp_path, module)["metrics_reason"] == "expected_one_native_event_file"
    reader.Reload.assert_not_called()


def test_events_changing_during_read_are_rejected(tmp_path):
    event, reader, module = fixture(tmp_path, [(1, 0.5), (2, 0.5)])
    reader.Scalars.side_effect = lambda tag: (
        event.write_bytes(b"changed"),
        [SimpleNamespace(step=i, value=0.5) for i in (1, 2)],
    )[1]
    assert read(tmp_path, module)["metrics_reason"] == "native_events_changed_during_read"


def test_missing_events_have_no_fake_zero_reward(tmp_path):
    result = read_native_metrics(str(tmp_path), "native", 2)
    assert result["metrics_status"] == "not_collected"
    assert result["reward_trace"] == []
    assert "logged_rollout_steps" not in result
