#!/usr/bin/env python3
"""Keep the frozen 40-unit E1 confirmatory campaign filled on Colab.

This supervisor does not relax the single-unit verifier.  It only starts
``run_colab_e1_confirmatory.py`` processes when a global Colab slot is free,
and considers a unit complete only when the verifier has written a strict
``results/full/<arm>-seed-<seed>.json`` record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from huggingface_hub import HfApi, get_token


REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = REPO_ROOT / "zvf-program" / "audit" / "preregistration.json"
RUNNER = REPO_ROOT / "zvf-program" / "audit" / "run_colab_e1_confirmatory.py"
EVALUATION_RUNNER = REPO_ROOT / "zvf-program" / "audit" / "run_colab_e1_evaluation.py"
RESUME_RUNNER = REPO_ROOT / "zvf-program" / "audit" / "run_colab_e1_resume.py"
AGGREGATOR = REPO_ROOT / "zvf-program" / "audit" / "aggregate_audit.py"
AGGREGATE_OUTPUT = REPO_ROOT / "zvf-program" / "audit" / "results" / "audit.json"
FULL_RESULTS = REPO_ROOT / "zvf-program" / "audit" / "results" / "full"
UNIT_RESULTS = (
    REPO_ROOT / "zvf-program" / "audit" / "results" / "colab-e1-confirmatory" / "results"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "zvf-program" / "audit" / "results" / "colab-e1-confirmatory" / "campaign"
)
RUNNER_BASENAME = RUNNER.name
EVALUATION_RUNNER_BASENAME = EVALUATION_RUNNER.name
RESUME_RUNNER_BASENAME = RESUME_RUNNER.name
Unit = tuple[str, int]
TRANSIENT_PROVIDER_MARKERS = (
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "500 server error",
    "502 server error",
    "503 server error",
    "504 server error",
    "bad gateway",
    "gateway time-out",
    "gateway timeout",
    "service unavailable",
    "read timed out",
    "readtimeout",
    "connecttimeout",
    "connection reset",
    "connection was lost",
    "temporary failure",
    "init_sys_streams",
    "bad file descriptor",
    "private repository storage limit reached",
)
NON_TRANSIENT_FATAL_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "device-side assert",
    "illegal memory access",
    "fingerprint mismatch",
    "stack fingerprint",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def latest_assignment_failure_epoch(units: list[Unit]) -> float | None:
    """Return the newest failed pre-assignment timestamp across campaign units."""
    failures: list[float] = []
    for arm, seed in units:
        result = read_json(UNIT_RESULTS / f"e1__{arm}__s{seed}__confirmatory.json")
        if (
            result is None
            or result.get("status") != "failed"
            or result.get("failed_step") != 0
            or not isinstance(result.get("completed_at"), str)
        ):
            continue
        try:
            completed_at = datetime.fromisoformat(
                result["completed_at"].replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        failures.append(completed_at.timestamp())
    return max(failures) if failures else None


def is_preassignment_failure(record: dict[str, Any] | None) -> bool:
    """Return whether a runner failed before Colab allocated a VM."""
    return bool(
        record is not None
        and record.get("status") == "failed"
        and record.get("failed_step") == 0
    )


def is_transient_provider_failure(
    record: dict[str, Any] | None,
    *,
    extra_log_path: Path | None = None,
) -> bool:
    """Return whether the final failure evidence identifies a provider outage."""
    if record is None or record.get("status") != "failed":
        return False
    text_parts = [str(record.get("error") or "")]
    log_paths: list[Path] = []
    record_log_path = record.get("log_path")
    if isinstance(record_log_path, str):
        log_paths.append(Path(record_log_path))
    if extra_log_path is not None:
        log_paths.append(extra_log_path)
    for log_path in log_paths:
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 256 * 1024))
                log_text = handle.read().decode("utf-8", errors="replace")
                if extra_log_path is not None and log_path == extra_log_path:
                    # Per-unit campaign logs are append-only across retries.
                    # Classify only the newest attempt so an old provider
                    # outage cannot taint a later deterministic failure.
                    lines = log_text.splitlines()
                    attempt_starts = [
                        index
                        for index, line in enumerate(lines)
                        if "] campaign attempt " in line
                    ]
                    if attempt_starts:
                        log_text = "\n".join(lines[attempt_starts[-1] :])
                text_parts.append(log_text)
        except OSError:
            pass
    evidence = "\n".join(text_parts).lower()
    final_evidence = "\n".join(evidence.splitlines()[-200:])
    if any(marker in final_evidence for marker in NON_TRANSIENT_FATAL_MARKERS):
        return False
    return any(marker in final_evidence for marker in TRANSIENT_PROVIDER_MARKERS)


def failure_credit_id(unit: Unit, record: dict[str, Any] | None) -> str | None:
    """Identify one immutable failed result so retry credit is applied once."""
    if record is None or record.get("status") != "failed":
        return None
    completed_at = record.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        return None
    identity = {
        "arm": unit[0],
        "seed": unit[1],
        "completed_at": completed_at,
        "failed_step": record.get("failed_step"),
        "request_path": record.get("request_path"),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def active_backoff_until(
    values: list[float | None], *, now_epoch: float
) -> float | None:
    """Return the latest future backoff deadline, ignoring expired values."""
    future = [value for value in values if value is not None and value > now_epoch]
    return max(future, default=None)


def regenerate_aggregate(*, validated_units: int, required_units: int) -> None:
    """Refresh the fail-closed frozen aggregate from independently valid records."""
    command = [
        sys.executable,
        str(AGGREGATOR),
        "--input-dir",
        str(FULL_RESULTS),
        "--output",
        str(AGGREGATE_OUTPUT),
        "--allow-incomplete",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"aggregate refresh failed with exit {completed.returncode}: "
            f"{(completed.stdout or '').strip()}"
        )
    report = read_json(AGGREGATE_OUTPUT)
    if report is None:
        raise RuntimeError("aggregate refresh did not produce valid JSON")
    if validated_units < required_units:
        expected = {
            "status": "INCOMPLETE" if validated_units else "PREREGISTERED-NOT-RUN",
            "validated_units": validated_units,
            "required_units": required_units,
            "verdicts_emitted": False,
        }
        mismatches = {
            key: {"expected": value, "observed": report.get(key)}
            for key, value in expected.items()
            if report.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"aggregate refresh disagrees with campaign: {mismatches}")
        if not isinstance(report.get("errors"), list):
            raise RuntimeError("aggregate refresh errors field is not a list")
        missing = report.get("missing_units")
        if not isinstance(missing, list) or len(missing) != required_units - validated_units:
            raise RuntimeError("aggregate refresh has the wrong missing-unit count")
    elif report.get("status") != "COMPLETE":
        raise RuntimeError("complete campaign did not produce a COMPLETE aggregate")


def load_contract() -> tuple[list[Unit], dict[str, Any]]:
    contract = read_json(PREREGISTRATION)
    if contract is None:
        raise RuntimeError(f"invalid preregistration: {PREREGISTRATION}")
    core = contract.get("core_stratum")
    if not isinstance(core, dict):
        raise RuntimeError("preregistration is missing core_stratum")
    arms = core.get("arms")
    seeds = core.get("seeds")
    if not isinstance(arms, dict) or not isinstance(seeds, list):
        raise RuntimeError("preregistration has invalid arms or seeds")
    units = [(str(arm), int(seed)) for arm in arms for seed in seeds]
    expected = {
        "heldout_n": int(core["heldout"]["n"]),
        "checkpoint_steps": list(range(5, int(core["train_steps"]) + 1, 5)),
        "treatment_changes": {
            str(arm): list(config["allowed_changes"]) for arm, config in arms.items()
        },
    }
    return units, expected


def result_path(unit: Unit) -> Path:
    arm, seed = unit
    return FULL_RESULTS / f"{arm}-seed-{seed}.json"


def unit_result_path(unit: Unit) -> Path:
    arm, seed = unit
    return UNIT_RESULTS / f"e1__{arm}__s{seed}__confirmatory.json"


def validate_result(
    unit: Unit,
    record: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    stack_fingerprint: str | None,
) -> tuple[bool, str | None]:
    """Validate the local record produced after remote W&B/HF reconciliation."""
    if record is None:
        return False, "missing or invalid JSON"
    arm, seed = unit
    checks = {
        "arm": record.get("arm") == arm,
        "seed": record.get("seed") == seed,
        "evidence_class": record.get("evidence_class") == "confirmatory",
        "heldout_n": record.get("heldout_n") == expected["heldout_n"],
        "treatment_changes": record.get("treatment_changes")
        == expected["treatment_changes"][arm],
        "fingerprint": isinstance(record.get("fingerprint"), str)
        and len(record["fingerprint"]) == 64,
        "stack_fingerprint": isinstance(record.get("stack_fingerprint"), str)
        and len(record["stack_fingerprint"]) == 64,
    }
    if stack_fingerprint is not None:
        checks["same_stack"] = record.get("stack_fingerprint") == stack_fingerprint
    manifest_ref = record.get("manifest_path")
    manifest_path = None
    if isinstance(manifest_ref, str) and manifest_ref:
        manifest_path = Path(manifest_ref)
        if not manifest_path.is_absolute():
            manifest_path = result_path(unit).parent / manifest_path
    manifest = read_json(manifest_path) if manifest_path is not None else None
    trace = manifest.get("heldout_trace") if isinstance(manifest, dict) else None
    heldout_n = expected["heldout_n"]
    checks["heldout_trace"] = isinstance(trace, list) and len(trace) == heldout_n
    if checks["heldout_trace"]:
        indices = [row.get("index") if isinstance(row, dict) else None for row in trace]
        hashes = [
            row.get("completion_sha256") if isinstance(row, dict) else None
            for row in trace
        ]
        checks["heldout_indices"] = indices == list(range(heldout_n))
        checks["heldout_hashes"] = all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        )
        checks["unique_completion_hashes"] = (
            checks["heldout_hashes"] and len(set(hashes)) == heldout_n
        )
        score = record.get("heldout_score")
        correct = sum(row.get("correct") is True for row in trace if isinstance(row, dict))
        checks["heldout_score_recomputed"] = (
            isinstance(score, (int, float))
            and math.isfinite(score)
            and math.isclose(correct / heldout_n, score, rel_tol=0.0, abs_tol=1e-12)
        )
    else:
        checks.update(
            {
                "heldout_indices": False,
                "heldout_hashes": False,
                "unique_completion_hashes": False,
                "heldout_score_recomputed": False,
            }
        )
    verification = record.get("remote_verification")
    if not isinstance(verification, dict):
        checks["remote_verification"] = False
    else:
        wandb = verification.get("wandb")
        checks["wandb_finished"] = isinstance(wandb, dict) and wandb.get("state") == "finished"
        checks["hf_checkpoints"] = verification.get("hf_checkpoint_steps") == expected[
            "checkpoint_steps"
        ]
        checks["hf_repo"] = isinstance(verification.get("hf_repo"), str)
        checks["hf_commit"] = isinstance(verification.get("hf_commit"), str)
    failed = [name for name, passed in checks.items() if not passed]
    return (not failed, ", ".join(failed) if failed else None)


def parse_process_unit(command: str) -> Unit | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens or not Path(tokens[0]).name.lower().startswith("python"):
        # A tmux server retains the command used to create its first window in
        # its own process title.  Treating that parent as another unit runner
        # permanently consumes a campaign slot after the real window exits.
        return None
    basenames = {Path(token).name for token in tokens}
    if EVALUATION_RUNNER_BASENAME in basenames or RESUME_RUNNER_BASENAME in basenames:
        try:
            source_request = Path(tokens[tokens.index("--source-request") + 1])
        except (ValueError, IndexError):
            return None
        source = read_json(source_request)
        if source is None:
            return None
        try:
            return str(source["arm"]), int(source["seed"])
        except (KeyError, TypeError, ValueError):
            return None
    if RUNNER_BASENAME not in basenames:
        return None
    try:
        if tokens[tokens.index("--mode") + 1] != "confirmatory":
            return None
        seed = int(tokens[tokens.index("--seed") + 1])
    except (ValueError, IndexError):
        return None
    try:
        arm = tokens[tokens.index("--arm") + 1]
    except ValueError:
        arm = "grpo"
    except IndexError:
        return None
    return arm, seed


def recovery_source_request(
    unit: Unit,
    *,
    checkpoint_exists: Any,
) -> Path | None:
    """Return the immutable request when HF proves evaluation can resume.

    A later Colab assignment failure can overwrite the wrapper result with
    ``failed_step=0`` even when the same content-addressed request already
    uploaded checkpoint 30.  The remote checkpoint is authoritative here.
    """
    arm, seed = unit
    result = read_json(UNIT_RESULTS / f"e1__{arm}__s{seed}__confirmatory.json")
    if result is None or result.get("status") not in {"failed", "completed"}:
        return None
    request_path = result.get("request_path")
    if not isinstance(request_path, str):
        return None
    resolved = confirmatory_source_request(Path(request_path))
    if resolved is None:
        return None
    source_path, source = resolved
    if (
        source is None
        or source.get("mode") != "confirmatory"
        or source.get("arm") != arm
        or source.get("seed") != seed
        or not isinstance(source.get("hf_repo"), str)
    ):
        return None
    if not checkpoint_exists(source["hf_repo"]):
        return None
    return source_path


def failed_source_request(
    unit: Unit,
    *,
    checkpoint_exists: Any,
) -> Path | None:
    """Return the immutable request when HF proves partial training exists."""
    arm, seed = unit
    result = read_json(UNIT_RESULTS / f"e1__{arm}__s{seed}__confirmatory.json")
    if (
        result is None
        or result.get("status") != "failed"
        or not isinstance(result.get("request_path"), str)
    ):
        return None
    resolved = confirmatory_source_request(Path(result["request_path"]))
    if resolved is None:
        return None
    source_path, source = resolved
    if (
        source is None
        or source.get("mode") != "confirmatory"
        or source.get("arm") != arm
        or source.get("seed") != seed
        or not isinstance(source.get("fingerprint"), str)
        or not isinstance(source.get("hf_repo"), str)
    ):
        return None
    if not checkpoint_exists(source["hf_repo"]):
        return None
    return source_path


def recovery_plan(
    unit: Unit,
    *,
    checkpoint5_exists: Any,
    checkpoint30_exists: Any,
) -> tuple[str, Path] | None:
    """Return the highest-value exact-source recovery proven by the Hub."""
    evaluation_request = recovery_source_request(
        unit,
        checkpoint_exists=checkpoint30_exists,
    )
    if evaluation_request is not None:
        return "evaluation-recovery", evaluation_request
    failed_request = failed_source_request(
        unit,
        checkpoint_exists=checkpoint5_exists,
    )
    if failed_request is None:
        return None
    return "exact-source-checkpoint-resume", failed_request


def confirmatory_source_request(
    request_path: Path,
    *,
    max_depth: int = 4,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve wrapper recovery requests back to the immutable training request."""
    current = request_path
    seen: set[Path] = set()
    for _ in range(max_depth):
        if current in seen:
            return None
        seen.add(current)
        request = read_json(current)
        if request is None:
            return None
        if request.get("mode") == "confirmatory":
            return current, request
        upstream = request.get("source_request")
        if not isinstance(upstream, str):
            return None
        next_path = Path(upstream).expanduser()
        current = next_path if next_path.is_absolute() else current.parent / next_path
    return None


def active_runner_processes() -> dict[int, Unit]:
    output = subprocess.run(
        ["ps", "-axo", "pid=,stat=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    active: dict[int, Unit] = {}
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or "Z" in fields[1]:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        unit = parse_process_unit(fields[2])
        if unit is not None:
            active[pid] = unit
    return active


def parse_colab_assignment_count(output: str) -> int:
    """Count server-side assignments from ``colab sessions`` output."""
    return sum(
        1
        for line in output.splitlines()
        if "Hardware:" in line and "Variant:" in line
    )


def parse_colab_session_names(output: str) -> tuple[set[str], bool]:
    """Return named sessions and whether the server still reports an unnamed row."""
    names: set[str] = set()
    has_unnamed = False
    for line in output.splitlines():
        if "Hardware:" not in line or "Variant:" not in line:
            continue
        stripped = line.strip()
        if not stripped.startswith("[") or "]" not in stripped:
            continue
        name = stripped[1 : stripped.index("]")]
        if name == "?":
            has_unnamed = True
        elif name:
            names.add(name)
    return names, has_unnamed


def remote_colab_sessions() -> tuple[int, set[str], bool] | None:
    """Return live assignment count, named sessions, and unnamed-row state."""
    try:
        result = subprocess.run(
            ["colab", "--auth=oauth2", "sessions"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    names, has_unnamed = parse_colab_session_names(result.stdout)
    return parse_colab_assignment_count(result.stdout), names, has_unnamed


def remote_colab_assignment_count() -> int | None:
    """Return live server assignment count, or ``None`` if the probe fails."""
    snapshot = remote_colab_sessions()
    return snapshot[0] if snapshot is not None else None


def active_colab_exec_transports(active_pids: set[int]) -> dict[int, tuple[int, str]]:
    """Map runner PID to its direct long-running Colab exec child and session."""
    if not active_pids:
        return {}
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,stat=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    transports: dict[int, tuple[int, str]] = {}
    for line in output.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) != 4 or "Z" in fields[2]:
            continue
        try:
            pid, parent_pid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if parent_pid not in active_pids:
            continue
        try:
            command = shlex.split(fields[3])
        except ValueError:
            continue
        if "exec" not in command or not any(
            Path(token).name == "colab" for token in command[: command.index("exec")]
        ):
            continue
        session = None
        for index, token in enumerate(command):
            if token == "--session" and index + 1 < len(command):
                session = command[index + 1]
                break
            if token.startswith("--session="):
                session = token.partition("=")[2]
                break
        if not session:
            continue
        transports[parent_pid] = (pid, session)
    return transports


def reap_missing_remote_transports(
    active_by_pid: dict[int, Unit],
    *,
    remote_names: set[str],
    has_unnamed_remote: bool,
    missing_polls: dict[int, int],
    threshold: int = 3,
) -> list[tuple[int, int, str]]:
    """Terminate stale exec transports after repeated authoritative absence.

    A single empty or partially named ``colab sessions`` response is not enough:
    the exact session must be absent for ``threshold`` successful probes, and no
    unnamed server row may still be resolving.  Killing only the CLI child lets
    the runner write its normal failure record so exact-source recovery remains
    available on the next supervisor cycle.
    """
    live_pids = set(active_by_pid)
    for runner_pid in list(missing_polls):
        if runner_pid not in live_pids:
            missing_polls.pop(runner_pid, None)
    transports = active_colab_exec_transports(live_pids)
    reaped: list[tuple[int, int, str]] = []
    for runner_pid, (transport_pid, session) in transports.items():
        if has_unnamed_remote or session in remote_names:
            missing_polls.pop(runner_pid, None)
            continue
        count = missing_polls.get(runner_pid, 0) + 1
        missing_polls[runner_pid] = count
        if count < threshold:
            continue
        os.kill(transport_pid, signal.SIGTERM)
        missing_polls.pop(runner_pid, None)
        reaped.append((runner_pid, transport_pid, session))
    return reaped


def verify_hf_launch_ready(hf_api: HfApi | None) -> None:
    """Require an authenticated, responsive Hub API before spending a slot."""
    if hf_api is None:
        raise RuntimeError("Hugging Face login is required for campaign launches")
    hf_api.whoami()


def choose_launches(
    units: list[Unit],
    completed: set[Unit],
    active: set[Unit],
    attempts: dict[str, int],
    *,
    capacity: int,
    max_attempts: int,
    last_attempt_epoch: dict[str, float] | None = None,
    now_epoch: float | None = None,
    retry_backoff_seconds: int = 0,
    recovery_kinds: dict[Unit, str] | None = None,
) -> list[Unit]:
    if capacity <= 0:
        return []
    last_attempt_epoch = last_attempt_epoch or {}
    recovery_kinds = recovery_kinds or {}
    now_epoch = time.time() if now_epoch is None else now_epoch
    eligible: list[tuple[Unit, int]] = []
    for order, unit in enumerate(units):
        key = f"{unit[0]}:{unit[1]}"
        if (
            unit in completed
            or unit in active
            or (
                attempts.get(key, 0) >= max_attempts
                and unit not in recovery_kinds
            )
        ):
            continue
        last_attempt = last_attempt_epoch.get(key)
        if (
            retry_backoff_seconds > 0
            and last_attempt is not None
            and now_epoch - last_attempt < retry_backoff_seconds
        ):
            continue
        eligible.append((unit, order))

    # Finish remotely proven work before starting another expensive unit:
    # evaluation-only recovery first, then partial-checkpoint recovery.  Among
    # everything else, preserve the anti-pinning policy of preferring untouched
    # units before ordinary retries.  A pre-assignment provider failure has no
    # recovery kind, so it cannot monopolize a campaign slot.
    recovery_rank = {
        "evaluation-recovery": 0,
        "exact-source-checkpoint-resume": 1,
    }
    eligible.sort(
        key=lambda item: (
            recovery_rank.get(
                recovery_kinds.get(item[0], ""),
                3
                if f"{item[0][0]}:{item[0][1]}" in last_attempt_epoch
                else 2,
            ),
            last_attempt_epoch.get(f"{item[0][0]}:{item[0][1]}", 0.0),
            item[1],
        )
    )
    return [unit for unit, _order in eligible[:capacity]]


def exhausted_units(
    units: list[Unit],
    completed: set[Unit],
    active: set[Unit],
    attempts: dict[str, int],
    *,
    max_attempts: int,
) -> list[Unit]:
    """Return incomplete, inactive units that consumed their retry budget."""
    return [
        unit
        for unit in units
        if unit not in completed
        and unit not in active
        and attempts.get(f"{unit[0]}:{unit[1]}", 0) >= max_attempts
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--retry-backoff-seconds", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status-once", action="store_true")
    return parser.parse_args(argv)


class CampaignLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> CampaignLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_json(self.path)
        if existing and isinstance(existing.get("pid"), int):
            try:
                os.kill(existing["pid"], 0)
            except ProcessLookupError:
                self.path.unlink(missing_ok=True)
            else:
                raise RuntimeError(f"campaign supervisor already running as pid {existing['pid']}")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"campaign lock already exists: {self.path}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started_at": utc_now()}, handle)
            handle.write("\n")
        self.acquired = True
        return self

    def __exit__(self, *_: object) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if (
        args.max_parallel <= 0
        or args.max_attempts <= 0
        or args.poll_seconds <= 0
        or args.retry_backoff_seconds < 0
    ):
        raise SystemExit("parallelism, attempts, and poll interval must be positive")
    for runner in (RUNNER, EVALUATION_RUNNER, RESUME_RUNNER, AGGREGATOR):
        if not runner.is_file():
            raise SystemExit(f"missing unit runner: {runner}")
    args.output_dir = args.output_dir.expanduser().resolve()
    units, expected = load_contract()
    state_path = args.output_dir / "campaign.json"
    lock_path = args.output_dir / "campaign.lock"
    previous = read_json(state_path) or {}
    attempts = {
        str(key): int(value)
        for key, value in (previous.get("attempts") or {}).items()
        if isinstance(value, int)
    }
    last_attempt_epoch = {
        str(key): float(value)
        for key, value in (previous.get("last_attempt_epoch") or {}).items()
        if isinstance(value, (int, float))
    }
    credited_failures = {
        str(value)
        for value in (previous.get("credited_failures") or [])
        if isinstance(value, str)
    }
    aggregate_validated_units = previous.get("aggregate_validated_units")
    if not isinstance(aggregate_validated_units, int):
        aggregate_validated_units = -1
    previous_dependency_backoff = previous.get("dependency_backoff_until_epoch")
    dependency_backoff_until = (
        float(previous_dependency_backoff)
        if isinstance(previous_dependency_backoff, (int, float))
        else None
    )
    children: dict[int, tuple[Unit, subprocess.Popen[str], Any]] = {}
    stop_requested = False
    hf_token = get_token()
    hf_api = HfApi(token=hf_token) if hf_token else None

    def checkpoint30_exists(repo_id: str) -> bool:
        if hf_api is None:
            raise RuntimeError("Hugging Face login is required for recovery selection")
        last_error: Exception | None = None
        for delay in (0, 2, 5):
            if delay:
                time.sleep(delay)
            try:
                return hf_api.file_exists(
                    repo_id=repo_id,
                    repo_type="model",
                    filename="checkpoints/checkpoint-30/trainer_state.json",
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"could not verify checkpoint 30 in {repo_id}: {last_error}")

    def checkpoint5_exists(repo_id: str) -> bool:
        if hf_api is None:
            raise RuntimeError("Hugging Face login is required for recovery selection")
        last_error: Exception | None = None
        for delay in (0, 2, 5):
            if delay:
                time.sleep(delay)
            try:
                return hf_api.file_exists(
                    repo_id=repo_id,
                    repo_type="model",
                    filename="checkpoints/checkpoint-5/trainer_state.json",
                )
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"could not verify checkpoint 5 in {repo_id}: {last_error}")

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    with CampaignLock(lock_path):
        missing_remote_polls: dict[int, int] = {}
        while True:
            for pid, (child_unit, process, handle) in list(children.items()):
                if process.poll() is not None:
                    handle.close()
                    del children[pid]

            records = {unit: read_json(result_path(unit)) for unit in units}
            failure_records = {unit: read_json(unit_result_path(unit)) for unit in units}
            stack_values = {
                record["stack_fingerprint"]
                for record in records.values()
                if record and isinstance(record.get("stack_fingerprint"), str)
            }
            if len(stack_values) > 1:
                raise RuntimeError(f"completed records disagree on stack fingerprint: {stack_values}")
            stack_fingerprint = next(iter(stack_values), None)
            validations = {
                unit: validate_result(
                    unit, records[unit], expected, stack_fingerprint=stack_fingerprint
                )
                for unit in units
            }
            completed = {unit for unit, (valid, _) in validations.items() if valid}
            if aggregate_validated_units != len(completed):
                try:
                    regenerate_aggregate(
                        validated_units=len(completed), required_units=len(units)
                    )
                except Exception as exc:
                    print(f"[aggregate-error] {exc}", file=sys.stderr, flush=True)
                else:
                    aggregate_validated_units = len(completed)
                    print(
                        f"[aggregate] refreshed validated={len(completed)}/{len(units)}",
                        flush=True,
                    )
            active_by_pid = active_runner_processes()
            active_units = set(active_by_pid.values())
            for unit, record in failure_records.items():
                if unit in active_units:
                    continue
                credit_id = failure_credit_id(unit, record)
                if credit_id is None or credit_id in credited_failures:
                    continue
                if is_preassignment_failure(record):
                    credit_reason = "pre-assignment failure"
                elif is_transient_provider_failure(
                    record,
                    extra_log_path=(
                        args.output_dir / "logs" / f"{unit[0]}-seed-{unit[1]}.log"
                    ),
                ):
                    credit_reason = "transient provider failure"
                    dependency_backoff_until = (
                        time.time() + args.retry_backoff_seconds
                        if args.retry_backoff_seconds > 0
                        else None
                    )
                else:
                    continue
                child_key = f"{unit[0]}:{unit[1]}"
                attempts[child_key] = max(0, attempts.get(child_key, 0) - 1)
                credited_failures.add(credit_id)
                print(
                    f"[retry-credit] arm={unit[0]} seed={unit[1]} "
                    f"{credit_reason} did not consume a unit attempt",
                    flush=True,
                )
                if credit_reason == "transient provider failure" and dependency_backoff_until:
                    print(
                        "[dependency-backoff] transient provider failure "
                        f"deferred launches until epoch {dependency_backoff_until:.0f}",
                        flush=True,
                    )
            remote_snapshot = remote_colab_sessions()
            if remote_snapshot is None:
                remote_assignments = None
                remote_names: set[str] = set()
                has_unnamed_remote = True
            else:
                remote_assignments, remote_names, has_unnamed_remote = remote_snapshot
                for runner_pid, transport_pid, session in reap_missing_remote_transports(
                    active_by_pid,
                    remote_names=remote_names,
                    has_unnamed_remote=has_unnamed_remote,
                    missing_polls=missing_remote_polls,
                ):
                    print(
                        "[stale-transport] remote session absent for three polls; "
                        f"terminated transport pid={transport_pid} runner={runner_pid} "
                        f"session={session}",
                        flush=True,
                    )
            occupied_slots = max(
                len(active_by_pid),
                remote_assignments if remote_assignments is not None else 0,
            )
            now_epoch = time.time()
            assignment_failure_epoch = latest_assignment_failure_epoch(units)
            assignment_backoff_until = (
                assignment_failure_epoch + args.retry_backoff_seconds
                if assignment_failure_epoch is not None
                else None
            )
            dependency_backoff_until = active_backoff_until(
                [dependency_backoff_until], now_epoch=now_epoch
            )
            effective_backoff_until = active_backoff_until(
                [assignment_backoff_until, dependency_backoff_until],
                now_epoch=now_epoch,
            )
            capacity = (
                0
                if effective_backoff_until is not None
                else max(0, args.max_parallel - occupied_slots)
            )
            if capacity > 0:
                try:
                    verify_hf_launch_ready(hf_api)
                except Exception as exc:
                    dependency_backoff_until = (
                        time.time() + args.retry_backoff_seconds
                        if args.retry_backoff_seconds > 0
                        else None
                    )
                    capacity = 0
                    print(
                        f"[defer] Hugging Face launch preflight failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if dependency_backoff_until is not None:
                        print(
                            "[dependency-backoff] Hugging Face launch preflight "
                            f"deferred until epoch {dependency_backoff_until:.0f}",
                            file=sys.stderr,
                            flush=True,
                        )
            recovery_plans: dict[Unit, tuple[str, Path]] = {}
            if capacity > 0:
                recovery_candidates = choose_launches(
                    units,
                    completed,
                    active_units,
                    {},
                    capacity=len(units),
                    max_attempts=args.max_attempts,
                    last_attempt_epoch=last_attempt_epoch,
                    now_epoch=now_epoch,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                )
                try:
                    for unit in recovery_candidates:
                        plan = recovery_plan(
                            unit,
                            checkpoint5_exists=checkpoint5_exists,
                            checkpoint30_exists=checkpoint30_exists,
                        )
                        if plan is not None:
                            recovery_plans[unit] = plan
                except Exception as exc:
                    dependency_backoff_until = (
                        time.time() + args.retry_backoff_seconds
                        if args.retry_backoff_seconds > 0
                        else None
                    )
                    capacity = 0
                    print(
                        f"[defer] recovery-priority verification failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            launches = choose_launches(
                units,
                completed,
                active_units,
                attempts,
                capacity=capacity,
                max_attempts=args.max_attempts,
                last_attempt_epoch=last_attempt_epoch,
                now_epoch=now_epoch,
                retry_backoff_seconds=args.retry_backoff_seconds,
                recovery_kinds={
                    unit: plan[0] for unit, plan in recovery_plans.items()
                },
            )

            state = {
                "schema_version": "e1-confirmatory-campaign-v1",
                "updated_at": utc_now(),
                "total_units": len(units),
                "completed_units": len(completed),
                "remaining_units": len(units) - len(completed),
                "stack_fingerprint": stack_fingerprint,
                "max_parallel": args.max_parallel,
                "max_attempts": args.max_attempts,
                "attempts": attempts,
                "last_attempt_epoch": last_attempt_epoch,
                "credited_failures": sorted(credited_failures),
                "aggregate_validated_units": aggregate_validated_units,
                "assignment_backoff_until_epoch": assignment_backoff_until,
                "dependency_backoff_until_epoch": dependency_backoff_until,
                "remote_colab_assignments": remote_assignments,
                "remote_colab_session_names": sorted(remote_names),
                "missing_remote_session_polls": {
                    str(pid): count for pid, count in sorted(missing_remote_polls.items())
                },
                "occupied_slots": occupied_slots,
                "active": [
                    {"pid": pid, "arm": unit[0], "seed": unit[1]}
                    for pid, unit in sorted(active_by_pid.items())
                ],
                "invalid_or_missing": [
                    {"arm": unit[0], "seed": unit[1], "reason": validations[unit][1]}
                    for unit in units
                    if unit not in completed
                ],
            }
            atomic_json(state_path, state)
            print(
                f"[campaign] completed={len(completed)}/{len(units)} "
                f"active={len(active_by_pid)} remote={remote_assignments} "
                f"capacity={capacity}",
                flush=True,
            )
            if len(completed) == len(units):
                return 0
            if args.status_once or stop_requested:
                return 0

            for arm, seed in launches:
                unit = (arm, seed)
                key = f"{arm}:{seed}"
                attempts[key] = attempts.get(key, 0) + 1
                last_attempt_epoch[key] = time.time()
                plan = recovery_plans.get(unit)
                if plan is None:
                    command = [
                        sys.executable,
                        str(RUNNER),
                        "--mode",
                        "confirmatory",
                        "--arm",
                        arm,
                        "--seed",
                        str(seed),
                    ]
                    launch_kind = "new-unit"
                else:
                    launch_kind, source_request = plan
                    recovery_runner = (
                        EVALUATION_RUNNER
                        if launch_kind == "evaluation-recovery"
                        else RESUME_RUNNER
                    )
                    command = [
                        sys.executable,
                        str(recovery_runner),
                        "--source-request",
                        str(source_request),
                    ]
                    if launch_kind == "evaluation-recovery":
                        command.extend(["--eval-batch-size", "8"])
                log_path = args.output_dir / "logs" / f"{arm}-seed-{seed}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", encoding="utf-8")
                handle.write(
                    f"\n[{utc_now()}] campaign attempt {attempts[key]} "
                    f"kind={launch_kind}: {shlex.join(command)}\n"
                )
                handle.flush()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                children[process.pid] = (unit, process, handle)
                print(
                    f"[launch] arm={arm} seed={seed} pid={process.pid} "
                    f"attempt={attempts[key]} kind={launch_kind} log={log_path}",
                    flush=True,
                )
            atomic_json(
                state_path,
                {
                    **state,
                    "attempts": attempts,
                    "last_attempt_epoch": last_attempt_epoch,
                    "dependency_backoff_until_epoch": dependency_backoff_until,
                },
            )

            tracked_active_units = active_units | {
                child_unit for child_unit, _process, _handle in children.values()
            }
            exhausted = exhausted_units(
                units,
                completed,
                tracked_active_units,
                attempts,
                max_attempts=args.max_attempts,
            )
            if exhausted and not active_by_pid and not children:
                print(f"[campaign] retry limit reached for {exhausted}", file=sys.stderr)
                return 1
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
