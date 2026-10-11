"""Optional real TensorBoard I/O smoke checks; every event is synthetic."""

import pytest

pytest.importorskip("tensorboard", reason="native TensorBoard extra is not installed")

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from platform_modal.openrlhf.metrics import read_native_metrics


@pytest.mark.parametrize(
    ("points", "observed"),
    [
        ([(1, 0.25), (2, 0.5)], True),
        ([(1, 0.25), (1, 0.5)], False),
        ([(1, 0.25), (2, float("nan"))], False),
        ([(1, 0.25)], False),
    ],
    ids=["complete", "duplicate-step", "nonfinite", "incomplete"],
)
def test_real_tensorboard_event_contract(tmp_path, points, observed):
    writer = EventFileWriter(str(tmp_path / "tensorboard/synthetic"))
    try:
        for step, value in points:
            writer.add_event(
                Event(
                    step=step,
                    summary=Summary(value=[Summary.Value(tag="train/reward", simple_value=value)]),
                )
            )
        writer.flush()
    finally:
        writer.close()

    result = read_native_metrics(str(tmp_path), "synthetic", expected_rollouts=2)
    assert result["steps"] is None
    if observed:
        assert result["metrics_status"] == "observed_training_rollout_rewards"
        assert result["reward_trace"] == [0.25, 0.5]
        assert len(result["metrics_source"]["sha256"]) == 64
    else:
        assert result["metrics_status"] == "not_collected"
        assert result["reward_trace"] == []
        assert result["metrics_reason"] == "invalid_or_incomplete_reward_trace"
