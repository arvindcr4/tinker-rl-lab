#!/usr/bin/env python3
"""Run E1 pilot units through Colab CLI with atomic, resumable records.

This launcher intentionally labels its output as a pilot.  The current open
trainer does not implement the frozen five-arm Qwen3-8B confirmatory contract.
Each paid Colab allocation executes exactly one arm/seed unit so a completed
unit can be reused without rerunning the rest of the campaign.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import netrc
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_SCRIPT = REPO_ROOT / "zvf-program" / "colab-experiments" / "e3_open_audit.py"
SECURE_EXEC = REPO_ROOT / "zvf-program" / "colab-experiments" / "secure_exec.py"
ARM_NAMES = ("grpo", "drgrpo", "dapo", "grpo_adaptiveG")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", choices=ARM_NAMES, required=True)
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--g0", type=int, default=4)
    parser.add_argument("--gmax", type=int, default=10)
    parser.add_argument("--max-new", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--inner", type=int, default=2)
    parser.add_argument("--heldout-n", type=int, default=20)
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="e1-open-audit-pilot")
    parser.add_argument("--hf-repo", default="arvindcr4/tinker-rl-lab-colab-obligations")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "zvf-program" / "audit" / "results" / "colab-e1-pilot",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_credentials() -> dict[str, str]:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            hf_token = token_path.read_text(encoding="utf-8").strip()

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
        missing.append("HF_TOKEN or ~/.cache/huggingface/token")
    if not wandb_key:
        missing.append("WANDB_API_KEY or api.wandb.ai entry in ~/.netrc")
    if missing:
        raise RuntimeError("missing remote-tracking credentials: " + ", ".join(missing))
    return {"HF_TOKEN": hf_token, "WANDB_API_KEY": wandb_key}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def result_from_log(lines: list[str]) -> dict[str, Any]:
    for raw in reversed(lines):
        line = ANSI_RE.sub("", raw)
        marker = line.find("E3_RESULT ")
        if marker >= 0:
            return json.loads(line[marker + len("E3_RESULT ") :])
    raise ValueError("Colab output did not contain an E3_RESULT record")


def stop_session(auth: str, session: str, log_handle: Any) -> None:
    command = ["colab", f"--auth={auth}", "stop", "--session", session]
    stopped = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_handle.write("\n[launcher cleanup] " + shlex.join(command) + "\n")
    log_handle.write(stopped.stdout or "")
    log_handle.flush()


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
    try:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            log_handle.write(line)
            print(line, end="", flush=True)
        return process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        raise


def verify_wandb_run(api_key: str, run_url: str) -> dict[str, str]:
    parts = [part for part in urlparse(run_url).path.split("/") if part]
    runs_index = parts.index("runs")
    entity = parts[runs_index - 2]
    project = parts[runs_index - 1]
    run_id = parts[runs_index + 1]
    query = {
        "query": (
            "query Run($entity: String!, $project: String!, $run: String!) { "
            "project(name: $project, entityName: $entity) { "
            "run(name: $run) { name state } } }"
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
    credentials: dict[str, str], result: dict[str, Any]
) -> dict[str, Any]:
    required = (
        "hf_repo",
        "hf_path",
        "hf_commit",
        "hf_checkpoint_url",
        "wandb_run_id",
        "wandb_run_url",
    )
    missing = [field for field in required if not result.get(field)]
    if missing:
        raise RuntimeError("result lacks remote provenance: " + ", ".join(missing))

    last_error: Exception | None = None
    for delay in (0, 2, 5, 10):
        if delay:
            time.sleep(delay)
        try:
            hf_api = HfApi(token=credentials["HF_TOKEN"])
            manifest_name = f"{result['hf_path']}/run_manifest.json"
            if not hf_api.file_exists(
                repo_id=result["hf_repo"],
                filename=manifest_name,
                repo_type="model",
                revision=result["hf_commit"],
            ):
                raise RuntimeError(f"HF checkpoint manifest is missing: {manifest_name}")
            wandb_record = verify_wandb_run(
                credentials["WANDB_API_KEY"], result["wandb_run_url"]
            )
            if wandb_record["run_id"] != result["wandb_run_id"]:
                raise RuntimeError("W&B run ID does not match emitted provenance")
            if wandb_record["state"] != "finished":
                raise RuntimeError(
                    f"W&B run is not finished: {wandb_record['state']}"
                )
            model_candidates = (
                f"{result['hf_path']}/model.safetensors",
                f"{result['hf_path']}/model.safetensors.index.json",
            )
            hf_model = next(
                (
                    filename
                    for filename in model_candidates
                    if hf_api.file_exists(
                        repo_id=result["hf_repo"],
                        filename=filename,
                        repo_type="model",
                        revision=result["hf_commit"],
                    )
                ),
                None,
            )
            if not hf_model:
                raise RuntimeError("HF final model weights are missing")
            return {
                "verified_at": utc_now(),
                "hf_manifest": manifest_name,
                "hf_model": hf_model,
                "hf_commit": result["hf_commit"],
                "wandb": wandb_record,
            }
        except (HTTPError, URLError, OSError, RuntimeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"remote verification failed: {last_error}")


def run_unit(args: argparse.Namespace, arm: str, seed: int, script_sha: str) -> dict[str, Any]:
    unit = f"e3_open_audit__{arm}__s{seed}"
    request = {
        "schema_version": "colab-e1-pilot-unit-v1",
        "evidence_class": "pilot-not-confirmatory",
        "arm": arm,
        "seed": seed,
        "accelerator": args.gpu,
        "model": args.model,
        "script": str(REMOTE_SCRIPT.relative_to(REPO_ROOT)),
        "script_sha256": script_sha,
        "secure_exec_sha256": sha256_file(SECURE_EXEC),
        "remote_tracking": {
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "hf_repo": args.hf_repo,
            "required": True,
        },
        "run_config": {
            "batch": args.batch,
            "g0": args.g0,
            "gmax": args.gmax,
            "max_new": args.max_new,
            "learning_rate": args.learning_rate,
            "steps": args.steps,
            "inner": args.inner,
            "heldout_n": args.heldout_n,
        },
    }
    unit_fingerprint = fingerprint(request)
    result_path = args.output_dir / "results" / f"{unit}.json"
    request_path = args.output_dir / "requests" / f"{unit}__{unit_fingerprint[:12]}.json"
    log_path = args.output_dir / "logs" / f"{unit}__{unit_fingerprint[:12]}.log"
    existing = read_json(result_path)
    if (
        not args.rerun
        and existing
        and existing.get("status") == "completed"
        and existing.get("fingerprint") == unit_fingerprint
    ):
        print(f"[resume] {unit}: compatible completed result exists", flush=True)
        return {"unit": unit, "status": "skipped-compatible", "result_path": str(result_path)}

    if existing and existing.get("fingerprint") != unit_fingerprint:
        old_fingerprint = str(existing.get("fingerprint") or "unknown")[:12]
        history_path = args.output_dir / "results" / "history" / f"{unit}__{old_fingerprint}.json"
        atomic_json(history_path, existing)

    session = f"e1p-{arm.lower().replace('_', '-')}-s{seed}-{unit_fingerprint[:6]}"[:40]
    hf_path = f"e1-pilot/{arm}/seed-{seed}/{unit_fingerprint[:12]}"
    wandb_run_name = f"e1p-{arm}-s{seed}-{unit_fingerprint[:8]}"
    script_args = [
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--model",
        args.model,
        "--batch",
        str(args.batch),
        "--g0",
        str(args.g0),
        "--gmax",
        str(args.gmax),
        "--max-new",
        str(args.max_new),
        "--learning-rate",
        str(args.learning_rate),
        "--steps",
        str(args.steps),
        "--inner",
        str(args.inner),
        "--heldout-n",
        str(args.heldout_n),
        "--unit-fingerprint",
        unit_fingerprint,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        args.wandb_group,
        "--wandb-run-name",
        wandb_run_name,
        "--hf-repo",
        args.hf_repo,
        "--hf-path",
        hf_path,
        "--require-remote-tracking",
    ]
    if args.wandb_entity:
        script_args.extend(["--wandb-entity", args.wandb_entity])
    execution_plan = [
        ["colab", f"--auth={args.auth}", "new", "--gpu", args.gpu, "--session", session],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         str(REMOTE_SCRIPT), "/content/e3_open_audit.py"],
        ["colab", f"--auth={args.auth}", "install", "--session", session, "wandb==0.25.1"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         "<ephemeral-secret-file>", "/content/.tinker-run-secrets.json"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         "<ephemeral-request-file>", "/content/tinker-unit-request.json"],
        ["colab", f"--auth={args.auth}", "exec", "--session", session,
         "--file", str(SECURE_EXEC), "--timeout", str(args.timeout)],
        ["colab", f"--auth={args.auth}", "stop", "--session", session],
    ]
    launched = {
        **request,
        "fingerprint": unit_fingerprint,
        "status": "dry-run" if args.dry_run else "launching",
        "session": session,
        "execution_plan": execution_plan,
        "wandb_run_name": wandb_run_name,
        "hf_path": hf_path,
        "updated_at": utc_now(),
    }
    atomic_json(request_path, launched)
    if args.dry_run:
        for command in execution_plan:
            print(shlex.join(command))
        return {"unit": unit, "status": "dry-run", "request_path": str(request_path)}

    credentials = load_credentials()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    started = utc_now()
    print(f"[launch] {unit}: session={session} gpu={args.gpu}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("[launcher] credentials staged out of band; values are not logged\n")
        return_code = 0
        failed_step = None
        with tempfile.TemporaryDirectory(prefix="tinker-colab-unit-") as staging:
            staging_path = Path(staging)
            secret_path = staging_path / "secrets.json"
            invocation_path = staging_path / "request.json"
            secret_path.write_text(json.dumps(credentials), encoding="utf-8")
            secret_path.chmod(0o600)
            invocation_path.write_text(
                json.dumps({"script_args": script_args}), encoding="utf-8"
            )
            commands = [
                execution_plan[0],
                execution_plan[1],
                execution_plan[2],
                ["colab", f"--auth={args.auth}", "upload", "--session", session,
                 str(secret_path), "/content/.tinker-run-secrets.json"],
                ["colab", f"--auth={args.auth}", "upload", "--session", session,
                 str(invocation_path), "/content/tinker-unit-request.json"],
                execution_plan[5],
            ]
            try:
                for index, command in enumerate(commands):
                    return_code = run_logged(command, log_handle, lines)
                    if return_code:
                        failed_step = index
                        break
            finally:
                stop_session(args.auth, session, log_handle)

    completed = utc_now()
    base_record = {
        **request,
        "fingerprint": unit_fingerprint,
        "session": session,
        "execution_plan": execution_plan,
        "started_at": started,
        "completed_at": completed,
        "return_code": return_code,
        "log_path": str(log_path),
    }
    if return_code != 0:
        failed = {
            **base_record,
            "status": "failed",
            "failed_step": failed_step,
            "error": "colab CLI returned non-zero",
        }
        atomic_json(result_path, failed)
        return {"unit": unit, "status": "failed", "result_path": str(result_path)}

    try:
        payload = result_from_log(lines)
        payload_units = payload.get("units")
        if not isinstance(payload_units, list) or len(payload_units) != 1:
            raise ValueError("expected exactly one payload unit")
        unit_payload = payload_units[0]
        if unit_payload.get("arm") != arm or unit_payload.get("seed") != seed:
            raise ValueError("payload arm/seed does not match the requested unit")
        verification = verify_remote_artifacts(credentials, unit_payload)
    except (json.JSONDecodeError, ValueError, AttributeError, RuntimeError) as exc:
        failed = {**base_record, "status": "failed", "error": str(exc)}
        atomic_json(result_path, failed)
        return {"unit": unit, "status": "failed", "result_path": str(result_path)}

    complete = {
        **base_record,
        "status": "completed",
        "payload": payload,
        "remote_verification": verification,
    }
    atomic_json(result_path, complete)
    return {"unit": unit, "status": "completed", "result_path": str(result_path)}


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not REMOTE_SCRIPT.is_file():
        raise SystemExit(f"missing Colab script: {REMOTE_SCRIPT}")
    if not SECURE_EXEC.is_file():
        raise SystemExit(f"missing secure Colab wrapper: {SECURE_EXEC}")
    script_sha = sha256_file(REMOTE_SCRIPT)
    units = [(arm, seed) for arm in dict.fromkeys(args.arm) for seed in dict.fromkeys(args.seed)]
    campaign_path = args.output_dir / "campaign.json"
    campaign = {
        "schema_version": "colab-e1-pilot-campaign-v1",
        "evidence_class": "pilot-not-confirmatory",
        "confirmatory_contract": "zvf-program/audit/preregistration.json",
        "created_at": utc_now(),
        "script_sha256": script_sha,
        "requested_units": [{"arm": arm, "seed": seed} for arm, seed in units],
        "unit_status": [],
    }
    atomic_json(campaign_path, campaign)
    failures = 0
    for arm, seed in units:
        status = run_unit(args, arm, seed, script_sha)
        campaign["unit_status"].append(status)
        campaign["updated_at"] = utc_now()
        atomic_json(campaign_path, campaign)
        failures += status["status"] == "failed"
    print(f"[campaign] {campaign_path}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
