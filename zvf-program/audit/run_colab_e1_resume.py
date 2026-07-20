#!/usr/bin/env python3
"""Resume an E1 unit with the exact content-addressed sources from its request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import tempfile
from typing import Any

import run_colab_e1_confirmatory as shared


REPO_ROOT = shared.REPO_ROOT
CANONICAL_REMOTE_SCRIPT = "zvf-program/colab-experiments/e1_grpo_confirmatory.py"
CANONICAL_SECURE_EXEC = "zvf-program/colab-experiments/secure_exec_confirmatory.py"
CANONICAL_ENVIRONMENT_CHECK = "zvf-program/colab-experiments/verify_colab_e1_environment.py"
LAUNCH_METADATA_KEYS = {
    "fingerprint",
    "status",
    "session",
    "hf_repo",
    "wandb_run_name",
    "execution_plan",
    "launcher_retry_policy",
    "updated_at",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--exec-attempts", type=int, default=3)
    parser.add_argument("--exec-retry-seconds", type=int, default=60)
    parser.add_argument(
        "--reconcile-remote",
        action="store_true",
        help=(
            "Rebuild the local accepted record from an already completed immutable "
            "HF manifest and finished W&B run without allocating Colab"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def fingerprint_payload(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if key not in LAUNCH_METADATA_KEYS}


def validate_source_request(source: dict[str, Any], source_request: Path) -> dict[str, Path]:
    if source.get("schema_version") != "colab-e1-confirmatory-unit-v1":
        raise RuntimeError("source request has the wrong schema")
    if source.get("mode") != "confirmatory" or source.get("evidence_class") != "confirmatory":
        raise RuntimeError("source request is not confirmatory evidence")
    run_config = source.get("run_config") or {}
    if (
        run_config.get("max_steps") != 30
        or run_config.get("heldout_n") != 500
        or run_config.get("max_completion_length") != 1024
        or run_config.get("save_steps") != 5
    ):
        raise RuntimeError("source request does not match the frozen 30/500 contract")
    fingerprint = source.get("fingerprint")
    if not isinstance(fingerprint, str) or shared.fingerprint(fingerprint_payload(source)) != fingerprint:
        raise RuntimeError("source request fingerprint does not validate")
    if source.get("stack_fingerprint") != shared.stack_fingerprint():
        raise RuntimeError("source request stack fingerprint is no longer compatible")
    if source.get("runtime_packages") != list(shared.PACKAGE_PINS):
        raise RuntimeError("source request package pins are no longer compatible")
    if source.get("accelerator") != "A100":
        raise RuntimeError("source request accelerator is not A100")

    snapshot_root = source_request.parent.parent
    snapshots = source.get("source_snapshots")
    if not isinstance(snapshots, dict):
        raise RuntimeError("source request lacks content-addressed snapshots")
    paths: dict[str, Path] = {}
    for canonical in (
        CANONICAL_REMOTE_SCRIPT,
        CANONICAL_SECURE_EXEC,
        CANONICAL_ENVIRONMENT_CHECK,
    ):
        record = snapshots.get(canonical)
        if not isinstance(record, dict):
            raise RuntimeError(f"source request lacks snapshot metadata for {canonical}")
        path = snapshot_root / str(record.get("snapshot"))
        digest = record.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or shared.sha256_file(path) != digest:
            raise RuntimeError(f"source snapshot is missing or corrupt: {canonical}")
        paths[canonical] = path
    if shared.sha256_file(paths[CANONICAL_REMOTE_SCRIPT]) != source.get("script_sha256"):
        raise RuntimeError("remote script snapshot disagrees with source request")
    if shared.sha256_file(paths[CANONICAL_SECURE_EXEC]) != source.get("secure_exec_sha256"):
        raise RuntimeError("secure wrapper snapshot disagrees with source request")
    if shared.sha256_file(paths[CANONICAL_ENVIRONMENT_CHECK]) != source.get(
        "environment_check_sha256"
    ):
        raise RuntimeError("environment check snapshot disagrees with source request")
    return paths


def remote_result_from_manifest(
    manifest: dict[str, Any], *, hf_repo: str, hf_commit: str
) -> dict[str, Any]:
    """Reconstruct the emitted result envelope from immutable final evidence."""
    audit_record = manifest.get("audit_record")
    wandb = manifest.get("wandb")
    if not isinstance(audit_record, dict):
        raise RuntimeError("remote manifest lacks an audit record")
    if not isinstance(wandb, dict):
        raise RuntimeError("remote manifest lacks W&B provenance")
    run_id = wandb.get("run_id")
    run_url = wandb.get("run_url")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("remote manifest lacks a W&B run ID")
    if not isinstance(run_url, str) or not run_url:
        raise RuntimeError("remote manifest lacks a W&B run URL")
    return {
        "evidence_class": manifest.get("evidence_class"),
        "audit_record": audit_record,
        "remote": {
            "hf_repo": hf_repo,
            "hf_commit": hf_commit,
            "wandb_run_id": run_id,
            "wandb_run_url": run_url,
        },
    }


def reconcile_remote_completion(
    args: argparse.Namespace,
    *,
    source: dict[str, Any],
    result_path: Path,
    source_output_dir: Path,
) -> dict[str, Any]:
    """Accept a completed remote unit after strict final-artifact verification."""
    arm = str(source["arm"])
    seed = int(source["seed"])
    unit_fingerprint = str(source["fingerprint"])
    hf_repo = str(source["hf_repo"])
    credentials = shared.load_credentials()
    info = shared.HfApi(token=credentials["HF_TOKEN"]).model_info(hf_repo)
    hf_commit = info.sha
    if not isinstance(hf_commit, str) or not hf_commit:
        raise RuntimeError("Hugging Face repository has no immutable head commit")
    manifest_file = shared.hf_hub_download(
        repo_id=hf_repo,
        repo_type="model",
        filename="run_manifest.json",
        revision=hf_commit,
        token=credentials["HF_TOKEN"],
    )
    manifest = json.loads(Path(manifest_file).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("remote manifest is not a JSON object")
    result = remote_result_from_manifest(
        manifest, hf_repo=hf_repo, hf_commit=hf_commit
    )
    audit = result["audit_record"]
    if audit.get("arm") != arm or audit.get("seed") != seed:
        raise RuntimeError("remote manifest arm/seed does not match source request")
    if result.get("evidence_class") != "confirmatory":
        raise RuntimeError("remote manifest is not confirmatory evidence")
    expected_steps = [5, 10, 15, 20, 25, 30]
    shared.validate_remote_manifest(
        manifest, result, unit_fingerprint, expected_steps, "confirmatory"
    )
    verification = shared.verify_remote_artifacts(credentials, result, expected_steps)
    args.mode = "confirmatory"
    args.arm = arm
    args.seed = seed
    args.output_dir = source_output_dir
    outputs = shared.write_validated_outputs(
        args, result, manifest, unit_fingerprint, verification
    )
    completed_at = shared.utc_now()
    complete = {
        **fingerprint_payload(source),
        "fingerprint": unit_fingerprint,
        "session": None,
        "hf_repo": hf_repo,
        "resume_source_request": str(args.source_request),
        "started_at": completed_at,
        "completed_at": completed_at,
        "return_code": 0,
        "failed_step": None,
        "log_path": None,
        "request_path": str(args.source_request),
        "status": "completed",
        "payload": result,
        "remote_verification": verification,
        "validated_outputs": outputs,
        "reconciliation": {
            "method": "immutable-remote-final-manifest",
            "hf_commit": hf_commit,
            "colab_allocated": False,
        },
    }
    shared.atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path), **outputs}


def run_resume(args: argparse.Namespace) -> dict[str, Any]:
    source = shared.read_json(args.source_request)
    if source is None:
        raise RuntimeError(f"cannot read source request: {args.source_request}")
    paths = validate_source_request(source, args.source_request)
    if args.gpu != source["accelerator"]:
        raise RuntimeError(
            f"resume accelerator {args.gpu!r} does not match source {source['accelerator']!r}"
        )
    arm = str(source["arm"])
    seed = int(source["seed"])
    unit = f"e1__{arm}__s{seed}"
    unit_fingerprint = str(source["fingerprint"])
    stack_fingerprint = str(source["stack_fingerprint"])
    hf_repo = str(source["hf_repo"])
    tracking = source["tracking"]
    run_config = source["run_config"]
    session = f"e1-resume-{arm}-s{seed}-{unit_fingerprint[:6]}"[:40]
    source_output_dir = args.source_request.parent.parent
    result_path = source_output_dir / "results" / f"{unit}__confirmatory.json"
    log_path = source_output_dir / "logs" / f"{unit}__resume__{unit_fingerprint[:12]}.log"
    if args.reconcile_remote:
        if args.dry_run:
            raise RuntimeError("--reconcile-remote and --dry-run cannot be combined")
        return reconcile_remote_completion(
            args,
            source=source,
            result_path=result_path,
            source_output_dir=source_output_dir,
        )
    script_args = [
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--mode",
        "confirmatory",
        "--max-steps",
        str(run_config["max_steps"]),
        "--heldout-n",
        str(run_config["heldout_n"]),
        "--max-completion-length",
        str(run_config["max_completion_length"]),
        "--save-steps",
        str(run_config["save_steps"]),
        "--unit-fingerprint",
        unit_fingerprint,
        "--stack-fingerprint",
        stack_fingerprint,
        "--hf-repo",
        hf_repo,
        "--wandb-project",
        tracking["wandb_project"],
        "--wandb-group",
        tracking["wandb_group"],
        "--wandb-run-name",
        source["wandb_run_name"],
    ]
    if tracking.get("wandb_entity"):
        script_args.extend(["--wandb-entity", tracking["wandb_entity"]])

    remote_script = paths[CANONICAL_REMOTE_SCRIPT]
    secure_exec = paths[CANONICAL_SECURE_EXEC]
    environment_check = paths[CANONICAL_ENVIRONMENT_CHECK]
    execution_plan = [
        ["colab", f"--auth={args.auth}", "new", "--gpu", args.gpu, "--session", session],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            str(remote_script),
            "/content/e1_grpo_confirmatory.py",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "install",
            "--session",
            session,
            *source["runtime_packages"],
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "exec",
            "--session",
            session,
            "--file",
            str(environment_check),
            "--timeout",
            "120",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            "<ephemeral-secret-file>",
            "/content/.e1-run-secrets.json",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            "<ephemeral-request-file>",
            "/content/e1-confirmatory-request.json",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "exec",
            "--session",
            session,
            "--file",
            str(secure_exec),
            "--timeout",
            str(args.timeout),
        ],
        ["colab", f"--auth={args.auth}", "stop", "--session", session],
    ]
    if args.dry_run:
        for command in execution_plan:
            print(shlex.join(command))
        return {
            "status": "dry-run",
            "source_request": str(args.source_request),
            "fingerprint": unit_fingerprint,
            "hf_repo": hf_repo,
        }

    credentials = shared.load_credentials()
    lines: list[str] = []
    return_code = 0
    failed_step = None
    started_at = shared.utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[launch-resume] {unit} session={session} fingerprint={unit_fingerprint[:12]}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("[launcher] exact source snapshots; credentials are not logged\n")
        with tempfile.TemporaryDirectory(prefix="e1-exact-resume-") as staging:
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
                [
                    "colab",
                    f"--auth={args.auth}",
                    "upload",
                    "--session",
                    session,
                    str(secret_path),
                    "/content/.e1-run-secrets.json",
                ],
                [
                    "colab",
                    f"--auth={args.auth}",
                    "upload",
                    "--session",
                    session,
                    str(invocation_path),
                    "/content/e1-confirmatory-request.json",
                ],
                execution_plan[6],
            ]
            try:
                for index, command in enumerate(commands):
                    if index == len(commands) - 1:
                        return_code = shared.run_logged_with_transient_retries(
                            command,
                            log_handle,
                            lines,
                            attempts=args.exec_attempts,
                            retry_seconds=args.exec_retry_seconds,
                        )
                    else:
                        return_code = shared.run_logged(command, log_handle, lines)
                    if return_code:
                        failed_step = index
                        break
            finally:
                shared.stop_session(args.auth, session, log_handle)

    base = {
        **fingerprint_payload(source),
        "fingerprint": unit_fingerprint,
        "session": session,
        "hf_repo": hf_repo,
        "resume_source_request": str(args.source_request),
        "started_at": started_at,
        "completed_at": shared.utc_now(),
        "return_code": return_code,
        "failed_step": failed_step,
        "log_path": str(log_path),
        "request_path": str(args.source_request),
        "launcher_retry_policy": {
            "exec_attempts": args.exec_attempts,
            "exec_retry_seconds": args.exec_retry_seconds,
            "preserve_session": True,
        },
    }
    if return_code:
        failed = {**base, "status": "failed", "error": "colab CLI returned non-zero"}
        shared.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}
    try:
        result = shared.result_from_log(lines)
        audit = result.get("audit_record") or {}
        if audit.get("arm") != arm or audit.get("seed") != seed:
            raise RuntimeError("emitted audit arm/seed does not match source request")
        if result.get("evidence_class") != "confirmatory":
            raise RuntimeError("resumed unit did not emit confirmatory evidence")
        expected_steps = [5, 10, 15, 20, 25, 30]
        verification = shared.verify_remote_artifacts(credentials, result, expected_steps)
        manifest = shared.download_manifest(credentials, result)
        shared.validate_remote_manifest(
            manifest, result, unit_fingerprint, expected_steps, "confirmatory"
        )
        args.mode = "confirmatory"
        args.arm = arm
        args.seed = seed
        args.output_dir = source_output_dir
        outputs = shared.write_validated_outputs(
            args, result, manifest, unit_fingerprint, verification
        )
    except Exception as exc:
        failed = {**base, "status": "failed", "error": str(exc)}
        shared.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}

    complete = {
        **base,
        "status": "completed",
        "payload": result,
        "remote_verification": verification,
        "validated_outputs": outputs,
    }
    shared.atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path), **outputs}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.exec_attempts < 1 or args.exec_retry_seconds < 0:
        raise SystemExit("exec attempts must be positive and retry seconds non-negative")
    args.source_request = args.source_request.expanduser().resolve()
    if not args.source_request.is_file():
        raise SystemExit(f"missing source request: {args.source_request}")
    status = run_resume(args)
    print("[resume] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
