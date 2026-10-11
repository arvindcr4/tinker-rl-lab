"""Read this invocation's native TensorBoard rewards without inferring updates."""

import hashlib
import math
from pathlib import Path
from tempfile import TemporaryDirectory


REWARD_TAG = "train/reward"


def read_native_metrics(output_dir: str, run_name: str, expected_rollouts: int) -> dict:
    """Require one stable event file and a complete finite, ordered scalar trace.

    The driver reserves a fresh output directory before launch. These are native
    training rewards indexed by rollout, never held-out scores or update counts.
    Hash binding detects file changes; it does not authenticate model execution.
    """
    result = {"metrics_status": "not_collected", "reward_trace": [], "steps": None}
    directory = Path(output_dir) / "tensorboard" / run_name
    files = sorted(directory.glob("events.out.tfevents.*"))
    if len(files) != 1 or not files[0].is_file() or files[0].is_symlink():
        return {**result, "metrics_reason": "expected_one_native_event_file"}
    event_file = files[0]
    try:
        source_bytes = event_file.read_bytes()
        before = hashlib.sha256(source_bytes).hexdigest()
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        # Parse exactly the bytes we hash. Hashing the mutable source before
        # and after parsing alone cannot detect an A -> B -> A replacement.
        with TemporaryDirectory(prefix="tinkerrl-native-events-") as scratch:
            snapshot = Path(scratch) / event_file.name
            snapshot.write_bytes(source_bytes)
            # Disable reservoir sampling and reset purging: neither may remove
            # malformed/duplicate observations before our validation.
            events = EventAccumulator(
                str(snapshot), size_guidance={"scalars": 0}, purge_orphaned_data=False
            ).Reload()
            if REWARD_TAG not in events.Tags().get("scalars", []):
                return {**result, "metrics_reason": "native_reward_tag_missing"}
            scalars = events.Scalars(REWARD_TAG)
            if (
                len(scalars) != expected_rollouts
                or not scalars
                or any(
                    item.step != index or not math.isfinite(item.value)
                    for index, item in enumerate(scalars, 1)
                )
            ):
                return {**result, "metrics_reason": "invalid_or_incomplete_reward_trace"}
            values = [float(item.value) for item in scalars]
        if hashlib.sha256(event_file.read_bytes()).hexdigest() != before:
            return {**result, "metrics_reason": "native_events_changed_during_read"}
    except (ImportError, OSError, ValueError, KeyError):
        return {**result, "metrics_reason": "native_events_unreadable"}
    return {
        **result,
        "metrics_status": "observed_training_rollout_rewards",
        "reward_trace": values,
        "logged_rollout_steps": len(values),
        "metrics_scope": "native training reward per rollout; not optimizer steps or held-out accuracy",
        "metrics_source": {
            "path": str(event_file.relative_to(output_dir)),
            "sha256": before,
            "tag": REWARD_TAG,
            "reader": "tensorboard.EventAccumulator; immutable byte snapshot; all scalars; no orphan purging",
        },
    }
