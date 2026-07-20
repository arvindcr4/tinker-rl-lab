#!/usr/bin/env python3
"""Launch one tracked, resumable E1 GRPO unit through the Colab CLI."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
import netrc
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, get_token, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_SCRIPT = REPO_ROOT / "zvf-program" / "colab-experiments" / "e1_grpo_confirmatory.py"
SECURE_EXEC = REPO_ROOT / "zvf-program" / "colab-experiments" / "secure_exec_confirmatory.py"
ENVIRONMENT_CHECK = REPO_ROOT / "zvf-program" / "colab-experiments" / "verify_colab_e1_environment.py"
AMENDMENT = REPO_ROOT / "zvf-program" / "audit" / "preregistration_colab_a100_amendment.json"
TREATMENT_SPEC = REPO_ROOT / "zvf-program" / "audit" / "preregistration_e1_treatments.json"
FULL_RESULTS = REPO_ROOT / "zvf-program" / "audit" / "results" / "full"
PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
)
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TRANSIENT_REMOTE_FAILURE_MARKERS = (
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sources(output_dir: Path, paths: list[Path]) -> dict[str, dict[str, str]]:
    """Store one content-addressed copy of every executable provenance input."""
    snapshots: dict[str, dict[str, str]] = {}
    for path in paths:
        digest = sha256_file(path)
        relative_source = str(path.relative_to(REPO_ROOT))
        relative_snapshot = Path("sources") / digest
        target = output_dir / relative_snapshot
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != digest:
                raise RuntimeError(f"content-addressed source snapshot is corrupt: {target}")
        else:
            fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=target.parent)
            try:
                with path.open("rb") as source, os.fdopen(fd, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                if sha256_file(Path(tmp_name)) != digest:
                    raise RuntimeError(f"source snapshot hash mismatch: {path}")
                os.replace(tmp_name, target)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
                raise
        snapshots[relative_source] = {
            "sha256": digest,
            "snapshot": str(relative_snapshot),
        }
    return snapshots


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "confirmatory"), required=True)
    parser.add_argument(
        "--arm", choices=("grpo", "dapo", "gspo", "drgrpo", "aero"), default="grpo"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--exec-attempts", type=int, default=3)
    parser.add_argument("--exec-retry-seconds", type=int, default=60)
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="e1-qwen3-8b-gsm8k")
    parser.add_argument("--hf-repo-prefix", default="arvindcr4/tinker-rl-lab-e1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "zvf-program" / "audit" / "results" / "colab-e1-confirmatory",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def load_credentials() -> dict[str, str]:
    hf_token = os.environ.get("HF_TOKEN") or get_token()
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        try:
            credentials = netrc.netrc()
            for machine in ("api.wandb.ai", "https://api.wandb.ai"):
                auth = credentials.authenticators(machine)
                if auth and auth[2]:
                    wandb_key = auth[2]
                    break
        except (FileNotFoundError, netrc.NetrcParseError, OSError):
            pass
    missing = []
    if not hf_token:
        missing.append("HF_TOKEN or Hugging Face CLI login")
    if not wandb_key:
        missing.append("WANDB_API_KEY or api.wandb.ai entry in ~/.netrc")
    if missing:
        raise RuntimeError("missing remote-tracking credentials: " + ", ".join(missing))
    return {"HF_TOKEN": hf_token, "WANDB_API_KEY": wandb_key}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def result_from_log(lines: list[str]) -> dict[str, Any]:
    for raw in reversed(lines):
        line = ANSI_RE.sub("", raw)
        marker = line.find("E1_RESULT ")
        if marker >= 0:
            return json.loads(line[marker + len("E1_RESULT ") :])
    raise ValueError("Colab output did not contain an E1_RESULT record")


def local_command_timeout_seconds(command: list[str], *, grace_seconds: int = 30) -> float | None:
    """Mirror a Colab CLI --timeout locally so a lost socket cannot hang forever."""
    raw_timeout: str | None = None
    for index, token in enumerate(command):
        if token == "--timeout" and index + 1 < len(command):
            raw_timeout = command[index + 1]
            break
        if token.startswith("--timeout="):
            raw_timeout = token.partition("=")[2]
            break
    if raw_timeout is None:
        return None
    try:
        timeout = float(raw_timeout)
    except ValueError:
        return None
    return timeout + grace_seconds if timeout > 0 else None


def run_logged(command: list[str], log_handle: Any, lines: list[str]) -> int:
    log_handle.write("\n[launcher command] " + shlex.join(command) + "\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    reader_errors: list[BaseException] = []

    def pump_output() -> None:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                log_handle.write(line)
                print(line, end="", flush=True)
        except BaseException as exc:  # pragma: no cover - defensive pipe failure
            reader_errors.append(exc)

    reader = threading.Thread(target=pump_output, name="e1-colab-output", daemon=True)
    reader.start()
    timeout = local_command_timeout_seconds(command)
    try:
        return_code = process.wait(timeout=timeout)
        reader.join(timeout=10)
        if reader_errors:
            raise reader_errors[0]
        return return_code
    except subprocess.TimeoutExpired:
        message = (
            f"[launcher watchdog] local timeout after {timeout:g}s; "
            "terminating stale Colab CLI transport\n"
        )
        lines.append(message)
        log_handle.write(message)
        log_handle.flush()
        print(message, end="", flush=True)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        reader.join(timeout=10)
        return 124
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        reader.join(timeout=10)
        raise


def is_transient_remote_failure(attempt_lines: list[str]) -> bool:
    """Recognize retryable provider/network failures from one command attempt."""
    output = ANSI_RE.sub("", "".join(attempt_lines[-200:])).lower()
    return any(marker in output for marker in TRANSIENT_REMOTE_FAILURE_MARKERS)


def run_logged_with_transient_retries(
    command: list[str],
    log_handle: Any,
    lines: list[str],
    *,
    attempts: int,
    retry_seconds: int,
) -> int:
    """Retry one remote invocation in-place without releasing its session."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if retry_seconds < 0:
        raise ValueError("retry_seconds cannot be negative")
    return_code = 0
    for attempt in range(1, attempts + 1):
        start = len(lines)
        return_code = run_logged(command, log_handle, lines)
        if return_code == 0:
            return 0
        if not is_transient_remote_failure(lines[start:]) or attempt == attempts:
            return return_code
        delay = retry_seconds * attempt
        message = (
            "[launcher retry] transient remote failure; preserving session and "
            f"retrying exec attempt {attempt + 1}/{attempts} in {delay}s\n"
        )
        log_handle.write(message)
        log_handle.flush()
        print(message, end="", flush=True)
        if delay:
            time.sleep(delay)
    return return_code


def stop_session(
    auth: str,
    session: str,
    log_handle: Any,
    *,
    attempts: int = 3,
    retry_seconds: int = 2,
) -> bool:
    """Release a Colab assignment even if the CLI transport exits uncleanly."""
    if attempts < 1:
        raise ValueError("cleanup attempts must be positive")
    if retry_seconds < 0:
        raise ValueError("cleanup retry seconds cannot be negative")
    command = ["colab", f"--auth={auth}", "stop", "--session", session]
    for attempt in range(1, attempts + 1):
        try:
            stopped = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=60,
            )
            return_code = stopped.returncode
            output = stopped.stdout or ""
        except subprocess.TimeoutExpired as exc:
            return_code = 124
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            output += "[launcher cleanup] Colab stop timed out after 60s\n"

        log_handle.write(
            "\n[launcher cleanup] "
            + shlex.join(command)
            + f" attempt={attempt}/{attempts} return_code={return_code}\n"
        )
        log_handle.write(output)
        log_handle.flush()
        if return_code == 0:
            return True
        if attempt < attempts:
            log_handle.write(
                f"[launcher cleanup] retrying session release in {retry_seconds}s\n"
            )
            log_handle.flush()
            if retry_seconds:
                time.sleep(retry_seconds)

    log_handle.write(
        f"[launcher cleanup warning] failed to release session {session} "
        f"after {attempts} attempts\n"
    )
    log_handle.flush()
    return False


def verify_wandb_run(api_key: str, run_url: str) -> dict[str, str]:
    parts = [part for part in urlparse(run_url).path.split("/") if part]
    runs_index = parts.index("runs")
    entity = parts[runs_index - 2]
    project = parts[runs_index - 1]
    run_id = parts[runs_index + 1]
    query = {
        "query": (
            "query Run($entity: String!, $project: String!, $run: String!) { "
            "project(name: $project, entityName: $entity) { run(name: $run) { name state } } }"
        ),
        "variables": {"entity": entity, "project": project, "run": run_id},
    }
    auth = base64.b64encode(f"api:{api_key}".encode()).decode()
    request = Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps(query).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("errors"):
        raise RuntimeError(f"W&B verification errors: {payload['errors']}")
    run = ((payload.get("data") or {}).get("project") or {}).get("run")
    if not run:
        raise RuntimeError("W&B run is not queryable")
    return {"run_id": run["name"], "state": run["state"], "url": run_url}


def verify_remote_artifacts(
    credentials: dict[str, str], result: dict[str, Any], expected_steps: list[int]
) -> dict[str, Any]:
    remote = result.get("remote") or {}
    required = ("hf_repo", "hf_commit", "wandb_run_id", "wandb_run_url")
    missing = [field for field in required if not remote.get(field)]
    if missing:
        raise RuntimeError("result lacks remote provenance: " + ", ".join(missing))
    api = HfApi(token=credentials["HF_TOKEN"])
    last_error: Exception | None = None
    for delay in (0, 2, 5, 10, 20):
        if delay:
            time.sleep(delay)
        try:
            for filename in ("run_manifest.json", "final/adapter_model.safetensors"):
                if not api.file_exists(
                    repo_id=remote["hf_repo"],
                    filename=filename,
                    repo_type="model",
                    revision=remote["hf_commit"],
                ):
                    raise RuntimeError(f"HF artifact is missing at emitted commit: {filename}")
            checkpoint_files = (
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
                "training_args.bin",
            )
            for step in expected_steps:
                for leaf in checkpoint_files:
                    filename = f"checkpoints/checkpoint-{step}/{leaf}"
                    if not api.file_exists(
                        repo_id=remote["hf_repo"],
                        filename=filename,
                        repo_type="model",
                        revision=remote["hf_commit"],
                    ):
                        raise RuntimeError(f"HF checkpoint is missing: {filename}")
            wandb_record = verify_wandb_run(credentials["WANDB_API_KEY"], remote["wandb_run_url"])
            if wandb_record["run_id"] != remote["wandb_run_id"]:
                raise RuntimeError("W&B run ID does not match emitted provenance")
            if wandb_record["state"] != "finished":
                raise RuntimeError(f"W&B run is not finished: {wandb_record['state']}")
            return {
                "verified_at": utc_now(),
                "hf_repo": remote["hf_repo"],
                "hf_commit": remote["hf_commit"],
                "hf_checkpoint_steps": expected_steps,
                "wandb": wandb_record,
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"remote verification failed: {last_error}")


def download_manifest(credentials: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    remote = result["remote"]
    path = hf_hub_download(
        repo_id=remote["hf_repo"],
        repo_type="model",
        filename="run_manifest.json",
        revision=remote["hf_commit"],
        token=credentials["HF_TOKEN"],
    )
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("remote manifest is not a JSON object")
    return value


def validate_remote_manifest(
    manifest: dict[str, Any],
    result: dict[str, Any],
    unit_fingerprint: str,
    expected_steps: list[int],
    evidence_class: str,
) -> None:
    if manifest.get("schema_version") != "e1-colab-confirmatory-run-v1":
        raise RuntimeError("remote manifest has the wrong schema version")
    if manifest.get("evidence_class") != evidence_class:
        raise RuntimeError("remote manifest has the wrong evidence class")
    run_config = manifest.get("run_config") or {}
    if run_config.get("unit_fingerprint") != unit_fingerprint:
        raise RuntimeError("remote manifest unit fingerprint does not match the request")
    if run_config.get("stack_fingerprint") != result.get("audit_record", {}).get(
        "stack_fingerprint"
    ):
        raise RuntimeError("remote manifest stack fingerprint does not match the result")
    if manifest.get("audit_record") != result.get("audit_record"):
        raise RuntimeError("remote manifest audit record does not match the emitted result")
    if manifest.get("remote_checkpoint_steps") != expected_steps:
        raise RuntimeError("remote manifest checkpoint cadence does not match the contract")
    trace = manifest.get("heldout_trace")
    heldout_n = result.get("audit_record", {}).get("heldout_n")
    if not isinstance(trace, list) or len(trace) != heldout_n:
        raise RuntimeError("remote manifest held-out trace is incomplete")
    indices = [row.get("index") if isinstance(row, dict) else None for row in trace]
    if indices != list(range(heldout_n)):
        raise RuntimeError("remote manifest held-out trace indices are not contiguous")
    completion_hashes = [
        row.get("completion_sha256") if isinstance(row, dict) else None for row in trace
    ]
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in completion_hashes
    ):
        raise RuntimeError("remote manifest held-out trace lacks completion hashes")
    if len(set(completion_hashes)) != heldout_n:
        raise RuntimeError("remote manifest held-out completion hashes are not unique")
    correct = sum(row.get("correct") is True for row in trace)
    observed_score = result["audit_record"]["heldout_score"]
    if not math.isclose(correct / heldout_n, observed_score, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("remote manifest held-out trace disagrees with heldout_score")
    manifest_run_id = (manifest.get("wandb") or {}).get("run_id")
    if manifest_run_id != result.get("remote", {}).get("wandb_run_id"):
        raise RuntimeError("remote manifest W&B run ID does not match emitted provenance")


def stack_fingerprint() -> str:
    return fingerprint(
        {
            "amendment_sha256": sha256_file(AMENDMENT),
            "runtime_packages": list(PACKAGE_PINS),
            "accelerator_class": "A100",
            "trainer_stack": "trl-grpo-transformers-generation-lora-v1",
        }
    )


def unit_contract(args: argparse.Namespace) -> tuple[int, int, int, str]:
    if args.mode == "preflight":
        return 1, 8, 1, "preflight-not-evidence"
    return 30, 500, 5, "confirmatory"


def write_validated_outputs(
    args: argparse.Namespace,
    result: dict[str, Any],
    manifest: dict[str, Any],
    unit_fingerprint: str,
    verification: dict[str, Any],
) -> dict[str, str]:
    unit = f"e1__{args.arm}__s{args.seed}"
    if args.mode == "confirmatory":
        manifest_path = FULL_RESULTS / "manifests" / f"{args.arm}-seed-{args.seed}.json"
        audit_path = FULL_RESULTS / f"{args.arm}-seed-{args.seed}.json"
        audit_record = dict(result["audit_record"])
        audit_record.update(
            {
                "manifest_path": f"manifests/{args.arm}-seed-{args.seed}.json",
                "fingerprint": unit_fingerprint,
                "evidence_class": "confirmatory",
                "remote": result["remote"],
                "remote_verification": verification,
            }
        )
        atomic_json(manifest_path, manifest)
        atomic_json(audit_path, audit_record)
        return {"audit_path": str(audit_path), "manifest_path": str(manifest_path)}

    manifest_path = args.output_dir / "manifests" / f"{unit}__{unit_fingerprint[:12]}.json"
    atomic_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path)}


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    max_steps, heldout_n, save_steps, evidence_class = unit_contract(args)
    stack_fp = stack_fingerprint()
    source_snapshots = snapshot_sources(
        args.output_dir,
        [REMOTE_SCRIPT, SECURE_EXEC, ENVIRONMENT_CHECK, AMENDMENT, TREATMENT_SPEC],
    )
    request = {
        "schema_version": "colab-e1-confirmatory-unit-v1",
        "mode": args.mode,
        "evidence_class": evidence_class,
        "arm": args.arm,
        "seed": args.seed,
        "accelerator": args.gpu,
        "script": str(REMOTE_SCRIPT.relative_to(REPO_ROOT)),
        "script_sha256": sha256_file(REMOTE_SCRIPT),
        "secure_exec_sha256": sha256_file(SECURE_EXEC),
        "environment_check_sha256": sha256_file(ENVIRONMENT_CHECK),
        "amendment_sha256": sha256_file(AMENDMENT),
        "treatment_spec_sha256": sha256_file(TREATMENT_SPEC),
        "stack_fingerprint": stack_fp,
        "runtime_packages": list(PACKAGE_PINS),
        "source_snapshots": source_snapshots,
        "run_config": {
            "max_steps": max_steps,
            "heldout_n": heldout_n,
            "max_completion_length": 1024,
            "save_steps": save_steps,
        },
        "tracking": {
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "hf_repo_prefix": args.hf_repo_prefix,
        },
    }
    unit_fingerprint = fingerprint(request)
    unit = f"e1__{args.arm}__s{args.seed}"
    hf_repo = f"{args.hf_repo_prefix}-{args.arm}-s{args.seed}-{unit_fingerprint[:8]}"
    session = f"e1-{args.mode[:3]}-{args.arm}-s{args.seed}-{unit_fingerprint[:6]}"[:40]
    wandb_run_name = f"e1-{args.mode}-{args.arm}-s{args.seed}-{unit_fingerprint[:8]}"
    result_path = args.output_dir / "results" / f"{unit}__{args.mode}.json"
    request_path = args.output_dir / "requests" / f"{unit}__{unit_fingerprint[:12]}.json"
    log_path = args.output_dir / "logs" / f"{unit}__{unit_fingerprint[:12]}.log"

    existing = read_json(result_path)
    if (
        not args.rerun
        and existing
        and existing.get("status") == "completed"
        and existing.get("fingerprint") == unit_fingerprint
    ):
        print(f"[resume] {unit}: compatible locally verified result exists", flush=True)
        return {"status": "skipped-compatible", "result_path": str(result_path)}
    if existing and existing.get("fingerprint") != unit_fingerprint:
        old_fp = str(existing.get("fingerprint") or "unknown")[:12]
        atomic_json(args.output_dir / "results" / "history" / f"{unit}__{old_fp}.json", existing)

    script_args = [
        "--arm", args.arm,
        "--seed", str(args.seed),
        "--mode", args.mode,
        "--max-steps", str(max_steps),
        "--heldout-n", str(heldout_n),
        "--max-completion-length", "1024",
        "--save-steps", str(save_steps),
        "--unit-fingerprint", unit_fingerprint,
        "--stack-fingerprint", stack_fp,
        "--hf-repo", hf_repo,
        "--wandb-project", args.wandb_project,
        "--wandb-group", args.wandb_group,
        "--wandb-run-name", wandb_run_name,
    ]
    if args.wandb_entity:
        script_args.extend(["--wandb-entity", args.wandb_entity])

    execution_plan = [
        ["colab", f"--auth={args.auth}", "new", "--gpu", args.gpu, "--session", session],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         str(REMOTE_SCRIPT), "/content/e1_grpo_confirmatory.py"],
        ["colab", f"--auth={args.auth}", "install", "--session", session, *PACKAGE_PINS],
        ["colab", f"--auth={args.auth}", "exec", "--session", session,
         "--file", str(ENVIRONMENT_CHECK), "--timeout", "120"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         "<ephemeral-secret-file>", "/content/.e1-run-secrets.json"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         "<ephemeral-request-file>", "/content/e1-confirmatory-request.json"],
        ["colab", f"--auth={args.auth}", "exec", "--session", session,
         "--file", str(SECURE_EXEC), "--timeout", str(args.timeout)],
        ["colab", f"--auth={args.auth}", "stop", "--session", session],
    ]
    launched = {
        **request,
        "fingerprint": unit_fingerprint,
        "status": "dry-run" if args.dry_run else "launching",
        "session": session,
        "hf_repo": hf_repo,
        "wandb_run_name": wandb_run_name,
        "execution_plan": execution_plan,
        "launcher_retry_policy": {
            "exec_attempts": args.exec_attempts,
            "exec_retry_seconds": args.exec_retry_seconds,
            "preserve_session": True,
        },
        "updated_at": utc_now(),
    }
    atomic_json(request_path, launched)
    if args.dry_run:
        for command in execution_plan:
            print(shlex.join(command))
        return {"status": "dry-run", "request_path": str(request_path)}

    credentials = load_credentials()
    lines: list[str] = []
    started_at = utc_now()
    return_code = 0
    failed_step = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[launch] {unit} mode={args.mode} session={session} gpu={args.gpu}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("[launcher] credentials staged out of band; values are not logged\n")
        with tempfile.TemporaryDirectory(prefix="e1-colab-unit-") as staging:
            staging_path = Path(staging)
            secret_path = staging_path / "secrets.json"
            invocation_path = staging_path / "request.json"
            secret_path.write_text(json.dumps(credentials), encoding="utf-8")
            secret_path.chmod(0o600)
            invocation_path.write_text(json.dumps({"script_args": script_args}), encoding="utf-8")
            commands = [
                execution_plan[0],
                execution_plan[1],
                execution_plan[2],
                execution_plan[3],
                ["colab", f"--auth={args.auth}", "upload", "--session", session,
                 str(secret_path), "/content/.e1-run-secrets.json"],
                ["colab", f"--auth={args.auth}", "upload", "--session", session,
                 str(invocation_path), "/content/e1-confirmatory-request.json"],
                execution_plan[6],
            ]
            try:
                for index, command in enumerate(commands):
                    if index == len(commands) - 1:
                        return_code = run_logged_with_transient_retries(
                            command,
                            log_handle,
                            lines,
                            attempts=args.exec_attempts,
                            retry_seconds=args.exec_retry_seconds,
                        )
                    else:
                        return_code = run_logged(command, log_handle, lines)
                    if return_code:
                        failed_step = index
                        break
            finally:
                stop_session(args.auth, session, log_handle)

    base_record = {
        **request,
        "fingerprint": unit_fingerprint,
        "session": session,
        "hf_repo": hf_repo,
        "started_at": started_at,
        "completed_at": utc_now(),
        "return_code": return_code,
        "failed_step": failed_step,
        "log_path": str(log_path),
        "request_path": str(request_path),
    }
    if return_code:
        failed = {**base_record, "status": "failed", "error": "colab CLI returned non-zero"}
        atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}

    try:
        result = result_from_log(lines)
        audit = result.get("audit_record") or {}
        if audit.get("arm") != args.arm or audit.get("seed") != args.seed:
            raise RuntimeError("emitted audit arm/seed does not match request")
        if result.get("evidence_class") != evidence_class:
            raise RuntimeError("emitted evidence class does not match request")
        expected_steps = list(range(save_steps, max_steps + 1, save_steps))
        verification = verify_remote_artifacts(credentials, result, expected_steps)
        manifest = download_manifest(credentials, result)
        validate_remote_manifest(
            manifest, result, unit_fingerprint, expected_steps, evidence_class
        )
        outputs = write_validated_outputs(
            args, result, manifest, unit_fingerprint, verification
        )
    except Exception as exc:
        failed = {**base_record, "status": "failed", "error": str(exc)}
        atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}

    complete = {
        **base_record,
        "status": "completed",
        "payload": result,
        "remote_verification": verification,
        "validated_outputs": outputs,
    }
    atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path), **outputs}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.exec_attempts < 1 or args.exec_retry_seconds < 0:
        raise SystemExit("exec attempts must be positive and retry seconds non-negative")
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (REMOTE_SCRIPT, SECURE_EXEC, ENVIRONMENT_CHECK, AMENDMENT, TREATMENT_SPEC):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")
    status = run_unit(args)
    print("[unit] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
