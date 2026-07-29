from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpointing import atomic_json
from .launcher import DEFAULT_OUTPUT, build_campaign_manifest, load_credentials
from .protocol import PROTOCOL_PATH, PilotProtocol, load_protocol
from .verifier import (
    VerificationError,
    verify_corpus_remote,
    verify_preflight_log,
    verify_unit_remote,
)


DEFAULT_STATE = DEFAULT_OUTPUT / "supervisor_state.json"
INFRASTRUCTURE_FAILURE_MARKERS = (
    "TooManyAssignmentsError",
    "Precondition Failed",
    "appears to be lost",
    "Connection was lost.",
    "ColabRequestError",
    "RESOURCE_EXHAUSTED",
    "429 Too Many Requests",
)


class SupervisorError(RuntimeError):
    """The persistent pilot scheduler is unauthorized or cannot make safe progress."""


def _archive_attempt_output(*, output_dir: Path, job_id: str, attempt: int) -> dict[str, Path]:
    archived: dict[str, Path] = {}
    for kind, suffix in (("logs", ".log"), ("results", ".json")):
        source = output_dir / kind / f"{job_id}{suffix}"
        if not source.is_file():
            continue
        target = output_dir / "attempts" / job_id / f"attempt-{attempt}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        if target.exists():
            if target.read_bytes() != payload:
                raise SupervisorError(f"attempt archive collision for {job_id} attempt {attempt}")
        else:
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
        archived[kind] = target
    return archived


def _classify_launcher_failure(
    *, log_path: Path | None, return_code: int, attempts: int, attempt_limit: int
) -> tuple[str, str]:
    text = ""
    if log_path is not None and log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
    marker = next(
        (candidate for candidate in INFRASTRUCTURE_FAILURE_MARKERS if candidate in text),
        None,
    )
    if marker is None:
        return (
            "failed_validation",
            f"launcher exited {return_code} without a recognized infrastructure "
            "signature; automatic retry is forbidden",
        )
    status = "pending" if attempts < attempt_limit else "failed_infrastructure"
    return status, f"launcher exited {return_code}; infrastructure marker: {marker}"


def initial_state(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "flagship-pilot-supervisor-v1",
        "manifest_fingerprint": manifest["fingerprint"],
        "jobs": {
            job["id"]: {
                "status": "pending",
                "attempts": 0,
                "pid": None,
                "last_error": None,
                "acceptance_path": None,
            }
            for job in manifest["jobs"]
        },
    }


def ready_jobs(
    manifest: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    capacity: int,
) -> list[str]:
    if capacity <= 0:
        return []
    statuses = state["jobs"]
    ready: list[str] = []
    running_corpora = sum(
        record["status"] == "running" and job["kind"] == "corpus"
        for job in manifest["jobs"]
        for record in (statuses[job["id"]],)
    )
    selected_corpora = 0
    for job in manifest["jobs"]:
        record = statuses[job["id"]]
        if record["status"] != "pending":
            continue
        if all(statuses[dependency]["status"] == "accepted" for dependency in job["depends_on"]):
            if job["kind"] == "corpus" and running_corpora + selected_corpora >= int(
                manifest["max_parallel_corpus_sessions"]
            ):
                continue
            ready.append(job["id"])
            if job["kind"] == "corpus":
                selected_corpora += 1
            if len(ready) == capacity:
                break
    return ready


def _load_or_create_state(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("manifest_fingerprint") != manifest["fingerprint"]:
            raise SupervisorError("supervisor state belongs to a different launch manifest")
        return state
    state = initial_state(manifest)
    atomic_json(path, state)
    return state


def _job_by_id(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {job["id"]: job for job in manifest["jobs"]}


def _verification_receipt(
    *,
    protocol: PilotProtocol,
    job: Mapping[str, Any],
    output_dir: Path,
    hf_api: Any,
    wandb_api: Any,
) -> Path:
    acceptance = output_dir / "acceptance" / f"{job['id']}.json"
    argv = job["argv"]
    if job["kind"] == "preflight":
        receipt = verify_preflight_log(
            protocol=protocol,
            log_path=output_dir / "logs" / f"{job['id']}.log",
            output_path=acceptance,
        )
        receipt["job_id"] = job["id"]
        atomic_json(acceptance, receipt)
        return acceptance
    if job["kind"] == "corpus":
        regime = argv[argv.index("--regime") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        manifest, _, commit = verify_corpus_remote(
            protocol=protocol,
            regime=regime,
            seed=seed,
            hf_api=hf_api,
            wandb_api=wandb_api,
        )
        receipt = {
            "schema_version": "flagship-pilot-corpus-acceptance-v1",
            "status": "accepted",
            "job_id": job["id"],
            "regime": regime,
            "seed": seed,
            "corpus_fingerprint": manifest["fingerprint"],
            "hf_commit": commit,
        }
        acceptance.parent.mkdir(parents=True, exist_ok=True)
        acceptance.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return acceptance
    condition = argv[argv.index("--condition") + 1]
    regime = argv[argv.index("--regime") + 1]
    seed = int(argv[argv.index("--seed") + 1])
    verify_unit_remote(
        protocol=protocol,
        condition=condition,
        regime=regime,
        seed=seed,
        hf_api=hf_api,
        wandb_api=wandb_api,
        output_path=acceptance,
    )
    return acceptance


def _child_command(
    *,
    job_id: str,
    output_dir: Path,
    protocol_path: Path,
    auth: str,
    timeout: int,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pilot.launcher",
        "--protocol",
        str(protocol_path),
        "--output-dir",
        str(output_dir),
        "--auth",
        auth,
        "--timeout",
        str(timeout),
        "--execute-job",
        job_id,
    ]


def run_supervisor(
    *,
    protocol: PilotProtocol,
    output_dir: Path,
    state_path: Path,
    auth: str,
    timeout: int,
    poll_seconds: float,
) -> dict[str, Any]:
    protocol.require_gpu_authorization()
    manifest = build_campaign_manifest(protocol, auth=auth, timeout=timeout)
    if not manifest["allocation_allowed"]:
        raise SupervisorError("launch manifest does not authorize allocation")
    state = _load_or_create_state(state_path, manifest)
    jobs = _job_by_id(manifest)
    credentials = load_credentials()
    os.environ.update(credentials)
    from huggingface_hub import HfApi
    import wandb

    hf_api = HfApi(token=credentials["HF_TOKEN"])
    wandb_api = wandb.Api(api_key=credentials["WANDB_API_KEY"])
    processes: dict[str, subprocess.Popen[str]] = {}

    while True:
        terminal = {"accepted", "failed_validation", "failed_infrastructure"}
        if all(record["status"] in terminal for record in state["jobs"].values()):
            return state

        for job_id, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            record = state["jobs"][job_id]
            record["pid"] = None
            del processes[job_id]
            archived = _archive_attempt_output(
                output_dir=output_dir,
                job_id=job_id,
                attempt=int(record["attempts"]),
            )
            if return_code != 0:
                record["status"], record["last_error"] = _classify_launcher_failure(
                    log_path=archived.get("logs"),
                    return_code=return_code,
                    attempts=int(record["attempts"]),
                    attempt_limit=int(manifest["max_attempts_per_job"]),
                )
                atomic_json(state_path, state)
                continue
            try:
                acceptance = _verification_receipt(
                    protocol=protocol,
                    job=jobs[job_id],
                    output_dir=output_dir,
                    hf_api=hf_api,
                    wandb_api=wandb_api,
                )
            except VerificationError as exc:
                record["status"] = "failed_validation"
                record["last_error"] = str(exc)
            except Exception as exc:
                record["last_error"] = f"verification infrastructure error: {exc}"
                record["status"] = (
                    "pending"
                    if record["attempts"] < int(manifest["max_attempts_per_job"])
                    else "failed_infrastructure"
                )
            else:
                record["status"] = "accepted"
                record["acceptance_path"] = str(acceptance)
            atomic_json(state_path, state)

        failed_validation = [
            job_id
            for job_id, record in state["jobs"].items()
            if record["status"] == "failed_validation"
        ]
        if failed_validation:
            raise SupervisorError(
                "scientific evidence validation failed for: " + ", ".join(failed_validation)
            )

        capacity = int(manifest["max_parallel_sessions"]) - len(processes)
        for job_id in ready_jobs(manifest, state, capacity=capacity):
            record = state["jobs"][job_id]
            record["attempts"] += 1
            command = _child_command(
                job_id=job_id,
                output_dir=output_dir,
                protocol_path=protocol.path,
                auth=auth,
                timeout=timeout,
            )
            process = subprocess.Popen(command, text=True)
            processes[job_id] = process
            record["status"] = "running"
            record["pid"] = process.pid
            atomic_json(state_path, state)

        if not processes and not ready_jobs(manifest, state, capacity=1):
            blocked = [
                job_id
                for job_id, record in state["jobs"].items()
                if record["status"] not in terminal
            ]
            raise SupervisorError("scheduler has no runnable jobs; blocked: " + ", ".join(blocked))
        time.sleep(poll_seconds)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent three-slot flagship pilot supervisor")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    manifest = build_campaign_manifest(protocol, auth=args.auth, timeout=args.timeout)
    if not args.run:
        print(json.dumps(initial_state(manifest), indent=2, sort_keys=True))
        return 0
    try:
        run_supervisor(
            protocol=protocol,
            output_dir=args.output_dir,
            state_path=args.state,
            auth=args.auth,
            timeout=args.timeout,
            poll_seconds=args.poll_seconds,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
