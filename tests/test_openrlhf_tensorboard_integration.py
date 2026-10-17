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


@pytest.mark.parametrize("tail", ["short-header", "bad-header-crc", "bad-data-crc"])
def test_complete_reward_prefix_does_not_hide_corrupt_event_tail(tmp_path, tail):
    import struct

    directory = tmp_path / "tensorboard/synthetic"
    writer = EventFileWriter(str(directory))
    try:
        for step in (1, 2):
            writer.add_event(
                Event(
                    step=step,
                    summary=Summary(
                        value=[Summary.Value(tag="train/reward", simple_value=0.25 * step)]
                    ),
                )
            )
        writer.flush()
    finally:
        writer.close()
    event = next(directory.glob("events.out.tfevents.*"))
    original = event.read_bytes()
    if tail == "short-header":
        suffix = b"\x01\x02"
    elif tail == "bad-header-crc":
        suffix = b"\x00" * 16
    else:
        length = struct.unpack("<Q", original[:8])[0]
        record = bytearray(original[: length + 16])
        record[-1] ^= 1
        suffix = bytes(record)
    event.write_bytes(original + suffix)
    result = read_native_metrics(str(tmp_path), "synthetic", expected_rollouts=2)
    assert result["metrics_status"] == "not_collected"
    assert result["reward_trace"] == []
    assert result["metrics_reason"] == "native_events_unreadable"
