from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import netrc
import os
import shlex
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from huggingface_hub import get_token

from .checkpointing import atomic_json
from .protocol import (
    PROTOCOL_PATH,
    REPO_ROOT,
    PilotProtocol,
    build_screening_plan,
    canonical_fingerprint,
    load_protocol,
    sha256_file,
)


BOOTSTRAP = Path(__file__).with_name("bootstrap.py")
RUNTIME_INSTALLER = Path(__file__).with_name("runtime_install.py")
DEFAULT_OUTPUT = Path(__file__).with_name("launch-v2-corpus-resume-r4-2")


class LauncherError(RuntimeError):
    """The local launch request is unsafe, unauthorized, or failed."""


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
        raise LauncherError("missing remote-tracking credentials: " + ", ".join(missing))
    return {"HF_TOKEN": hf_token, "WANDB_API_KEY": wandb_key}


def _bundle_bytes(source_bindings: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for relative, expected in sorted(source_bindings.items()):
                path = REPO_ROOT / relative
                if not path.is_file() or sha256_file(path) != expected:
                    raise LauncherError(f"source binding is stale or missing: {relative}")
                data = path.read_bytes()
                info = tarfile.TarInfo(name=f"tinker-rl-lab/{relative}")
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def write_source_bundle(path: Path, source_bindings: Mapping[str, str]) -> str:
    data = _bundle_bytes(source_bindings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _commands(
    *,
    session: str,
    packages: Sequence[str],
    auth: str,
    timeout: int,
) -> list[list[str]]:
    prefix = ["colab", f"--auth={auth}"]
    return [
        [*prefix, "new", "--gpu", "A100", "--session", session],
        [
            *prefix,
            "upload",
            "--session",
            session,
            "<source-bundle>",
            "/content/flagship-pilot-source.tar.gz",
        ],
        [*prefix, "upload", "--session", session, str(BOOTSTRAP), "/content/flagship_bootstrap.py"],
        [
            *prefix,
            "upload",
            "--session",
            session,
            "<install-request>",
            "/content/flagship-pilot-install.json",
        ],
        [
            *prefix,
            "exec",
            "--session",
            session,
            "--file",
            str(RUNTIME_INSTALLER),
            "--timeout",
            "1800",
        ],
        [*prefix, "restart-kernel", "--session", session],
        [
            *prefix,
            "upload",
            "--session",
            session,
            "<environment-request>",
            "/content/flagship-pilot-request.json",
        ],
        [*prefix, "exec", "--session", session, "--file", str(BOOTSTRAP), "--timeout", "180"],
        [
            *prefix,
            "upload",
            "--session",
            session,
            "<ephemeral-secrets>",
            "/content/.flagship-pilot-secrets.json",
        ],
        [
            *prefix,
            "upload",
            "--session",
            session,
            "<job-request>",
            "/content/flagship-pilot-request.json",
        ],
        [
            *prefix,
            "exec",
            "--session",
            session,
            "--file",
            str(BOOTSTRAP),
            "--timeout",
            str(timeout),
        ],
        [*prefix, "stop", "--session", session],
    ]


def build_campaign_manifest(
    protocol: PilotProtocol,
    *,
    auth: str = "oauth2",
    timeout: int = 86400,
) -> dict[str, Any]:
    plans = [build_screening_plan(protocol, unit) for unit in protocol.screening_units()]
    source_bindings_sha = plans[0]["protocol"]["source_bundle_sha256"]
    if any(plan["protocol"]["source_bundle_sha256"] != source_bindings_sha for plan in plans):
        raise LauncherError("unit plans do not share one source bundle")
    authorized = protocol.status == "ready_to_run" and protocol.gpu_authorized
    frozen_corpus_source = protocol.payload["corpus_reuse_binding"]
    corpus_source_bindings_sha = frozen_corpus_source["frozen_source_bindings_sha256"]
    corpus_source_archive_sha = frozen_corpus_source["frozen_source_archive_sha256"]
    packages = list(protocol.payload["runtime"]["package_pins"])
    jobs: list[dict[str, Any]] = []
    preflight_id = "preflight__a100_stack_smoke"
    preflight_session = f"fpsmoke-{source_bindings_sha[:8]}"[:40]
    jobs.append(
        {
            "id": preflight_id,
            "kind": "preflight",
            "depends_on": [],
            "session": preflight_session,
            "argv": ["smoke"],
            "source_scope": "revision_5_unit_training",
            "source_bindings_sha256": source_bindings_sha,
            "execution_plan": _commands(
                session=preflight_session,
                packages=packages,
                auth=auth,
                timeout=timeout,
            )
            if authorized
            else None,
        }
    )
    corpus_ids: dict[tuple[str, int], str] = {}
    for plan in plans:
        regime = plan["unit"]["regime"]
        seed = int(plan["unit"]["seed"])
        key = (regime, seed)
        if key in corpus_ids:
            continue
        job_id = f"corpus__{regime}__s{seed}"
        corpus_ids[key] = job_id
        session = f"fpcorp-{regime[:4]}-s{seed}-{corpus_source_bindings_sha[:4]}"[:40]
        argv = [
            "build-corpus",
            "--regime",
            regime,
            "--seed",
            str(seed),
            "--hf-repo",
            plan["identity"]["corpus_hf_repo"],
        ]
        jobs.append(
            {
                "id": job_id,
                "kind": "corpus",
                "depends_on": [preflight_id],
                "session": session,
                "argv": argv,
                "source_scope": "frozen_revision_4_corpus_generator",
                "source_bindings_sha256": corpus_source_bindings_sha,
                "source_archive_sha256": corpus_source_archive_sha,
                "execution_plan": _commands(
                    session=session,
                    packages=packages,
                    auth=auth,
                    timeout=timeout,
                )
                if authorized
                else None,
            }
        )
    for plan in plans:
        unit = plan["unit"]
        key = (unit["regime"], int(unit["seed"]))
        argv = [
            "train-unit",
            "--condition",
            unit["condition"],
            "--regime",
            unit["regime"],
            "--seed",
            str(unit["seed"]),
        ]
        jobs.append(
            {
                "id": unit["id"],
                "kind": "unit",
                "depends_on": [corpus_ids[key]],
                "session": plan["identity"]["colab_session"],
                "argv": argv,
                "unit_fingerprint": plan["fingerprint"],
                "source_scope": "revision_5_unit_training",
                "source_bindings_sha256": source_bindings_sha,
                "execution_plan": _commands(
                    session=plan["identity"]["colab_session"],
                    packages=packages,
                    auth=auth,
                    timeout=timeout,
                )
                if authorized
                else None,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": "flagship-pilot-launch-dag-v1",
        "status": "ready_to_run" if authorized else "locked_not_authorized",
        "protocol_sha256": protocol.sha256,
        "source_bindings_sha256": source_bindings_sha,
        "unit_source_bindings_sha256": source_bindings_sha,
        "corpus_source_bindings_sha256": corpus_source_bindings_sha,
        "corpus_source_archive_sha256": corpus_source_archive_sha,
        "allocation_allowed": authorized,
        "max_parallel_sessions": 3,
        "max_parallel_corpus_sessions": int(
            protocol.payload["runtime"]["execution_contract"]["corpus_checkpoint_resume_contract"][
                "max_parallel_corpus_sessions"
            ]
        ),
        "max_attempts_per_job": int(
            protocol.payload["runtime"]["execution_contract"]["corpus_checkpoint_resume_contract"][
                "attempt_limit"
            ]
        ),
        "job_count": len(jobs),
        "preflight_job_count": sum(job["kind"] == "preflight" for job in jobs),
        "corpus_job_count": sum(job["kind"] == "corpus" for job in jobs),
        "unit_job_count": sum(job["kind"] == "unit" for job in jobs),
        "jobs": jobs,
    }
    manifest["fingerprint"] = canonical_fingerprint(manifest)
    return manifest


_IDEMPOTENT_SUBCOMMANDS = frozenset({"upload", "restart-kernel", "stop"})


def _retries_for(command: Sequence[str]) -> int:
    # colab CLI control-plane calls can flake on short HTTP read timeouts;
    # only idempotent subcommands may be retried. `new` is excluded because a
    # timed-out create may still have allocated a session, and `exec` is
    # excluded because re-execution could duplicate remote side effects.
    if len(command) > 2 and command[2] in _IDEMPOTENT_SUBCOMMANDS:
        return 2
    return 0


def _run(command: Sequence[str], *, log: Any, retries: int = 0, backoff: float = 5.0) -> None:
    rendered = shlex.join(command)
    for attempt in range(1, retries + 2):
        log.write(f"$ {rendered}\n")
        log.flush()
        completed = subprocess.run(
            command, text=True, stdout=log, stderr=subprocess.STDOUT, check=False
        )
        if completed.returncode == 0:
            return
        if attempt > retries:
            break
        log.write(
            f"[launcher] attempt {attempt} failed with exit {completed.returncode}; "
            f"retrying idempotent step in {backoff:.0f}s\n"
        )
        log.flush()
        time.sleep(backoff)
    raise LauncherError(f"command failed with exit {completed.returncode}: {rendered}")


def _plan_for_job(protocol: PilotProtocol, job: Mapping[str, Any]) -> Mapping[str, Any]:
    if job["kind"] == "preflight":
        return build_screening_plan(protocol, next(protocol.screening_units()))
    if job["kind"] == "corpus":
        argv = job["argv"]
        regime = argv[argv.index("--regime") + 1]
        seed = int(argv[argv.index("--seed") + 1])
        unit = next(
            candidate
            for candidate in protocol.screening_units()
            if candidate.regime == regime and candidate.seed == seed
        )
        return build_screening_plan(protocol, unit)
    unit = next(
        candidate for candidate in protocol.screening_units() if candidate.unit_id == job["id"]
    )
    return build_screening_plan(protocol, unit)


def execute_job(
    *,
    protocol: PilotProtocol,
    manifest: Mapping[str, Any],
    job_id: str,
    output_dir: Path,
    auth: str,
) -> Path:
    protocol.require_gpu_authorization()
    jobs = {job["id"]: job for job in manifest["jobs"]}
    if job_id not in jobs:
        raise LauncherError(f"unknown launch job: {job_id}")
    job = jobs[job_id]
    if job["execution_plan"] is None:
        raise LauncherError("job contains no authorized execution plan")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results" / f"{job_id}.json"
    log_path = output_dir / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    credentials = load_credentials()
    with tempfile.TemporaryDirectory(prefix="flagship-pilot-launch-") as temporary:
        staging = Path(temporary)
        if job["kind"] == "corpus":
            source_path = (
                REPO_ROOT / protocol.payload["corpus_reuse_binding"]["frozen_source_archive_path"]
            )
            source_sha = sha256_file(source_path)
            if source_sha != job["source_archive_sha256"]:
                raise LauncherError("frozen corpus source archive changed")
            frozen_sources = _plan_for_job(protocol, job)["corpus_binding"]["source_manifest"]
            for relative, path in (
                ("zvf-program/flagship/pilot/bootstrap.py", BOOTSTRAP),
                ("zvf-program/flagship/pilot/runtime_install.py", RUNTIME_INSTALLER),
            ):
                if sha256_file(path) != frozen_sources[relative]:
                    raise LauncherError(f"external corpus bootstrap source changed: {relative}")
        else:
            source_path = staging / "source.tar.gz"
            source_sha = write_source_bundle(
                source_path,
                _plan_for_job(protocol, job)["source_bindings"],
            )
        expected_source_bindings_sha = job["source_bindings_sha256"]
        install_request = staging / "install.json"
        environment_request = staging / "environment.json"
        job_request = staging / "job.json"
        secret_path = staging / "secrets.json"
        atomic_json(
            install_request,
            {"package_pins": list(protocol.payload["runtime"]["package_pins"])},
        )
        atomic_json(
            environment_request,
            {
                "argv": ["verify-environment"],
                "source_archive_sha256": source_sha,
                "source_bindings_sha256": expected_source_bindings_sha,
            },
        )
        atomic_json(
            job_request,
            {
                "argv": job["argv"],
                "source_archive_sha256": source_sha,
                "source_bindings_sha256": expected_source_bindings_sha,
            },
        )
        atomic_json(secret_path, credentials)
        secret_path.chmod(0o600)
        commands = [list(command) for command in job["execution_plan"]]
        commands[1][-2] = str(source_path)
        commands[3][-2] = str(install_request)
        commands[6][-2] = str(environment_request)
        commands[8][-2] = str(secret_path)
        commands[9][-2] = str(job_request)
        error: str | None = None
        try:
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                for command in commands[:-1]:
                    _run(command, log=log, retries=_retries_for(command))
        except BaseException as exc:
            error = str(exc)
            raise
        finally:
            with log_path.open("a", encoding="utf-8", buffering=1) as log:
                try:
                    _run(commands[-1], log=log, retries=_retries_for(commands[-1]))
                except LauncherError:
                    if error is None:
                        raise
    if job["kind"] == "preflight":
        from .verifier import verify_preflight_log

        verify_preflight_log(protocol=protocol, log_path=log_path)
    else:
        # The Colab exec channel exits zero even when the remote script raises,
        # so fail closed here on the remote result line instead of letting the
        # supervisor mistake a silent crash for a verification retry.
        expected_prefix = (
            "FPILOT_CORPUS_RESULT " if job["kind"] == "corpus" else "FPILOT_UNIT_RESULT "
        )
        result_lines = [
            line
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith(expected_prefix)
        ]
        if len(result_lines) != 1:
            raise LauncherError(
                f"{job['kind']} log must contain exactly one result line; found {len(result_lines)}"
            )
    atomic_json(
        result_path,
        {
            "status": "completed",
            "job_id": job_id,
            "protocol_sha256": protocol.sha256,
            "source_bindings_sha256": expected_source_bindings_sha,
            "source_archive_sha256": source_sha,
            "log_path": str(log_path),
        },
    )
    return result_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flagship pilot Colab launch DAG")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--execute-job")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    manifest = build_campaign_manifest(protocol, auth=args.auth, timeout=args.timeout)
    if args.write:
        atomic_json(args.output_dir / "launch_manifest.json", manifest)
    if args.execute_job:
        try:
            result = execute_job(
                protocol=protocol,
                manifest=manifest,
                job_id=args.execute_job,
                output_dir=args.output_dir,
                auth=args.auth,
            )
        except Exception as exc:
            raise SystemExit(str(exc)) from exc
        print(result)
    elif not args.write:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
