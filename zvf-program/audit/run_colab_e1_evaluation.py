#!/usr/bin/env python3
"""Finish an E1 unit from its remote step-30 checkpoint on Colab."""

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
EVALUATOR = REPO_ROOT / "zvf-program" / "colab-experiments" / "e1_evaluate_checkpoint.py"
TRAINING_HELPERS = REPO_ROOT / "zvf-program" / "colab-experiments" / "e1_grpo_confirmatory.py"
SECURE_EXEC = REPO_ROOT / "zvf-program" / "colab-experiments" / "secure_exec_evaluation.py"
ENVIRONMENT_CHECK = shared.ENVIRONMENT_CHECK
DEFAULT_EVAL_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--exec-attempts", type=int, default=3)
    parser.add_argument("--exec-retry-seconds", type=int, default=60)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--progress-save-every", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "zvf-program" / "audit" / "results" / "colab-e1-confirmatory",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_recovery(args: argparse.Namespace) -> dict[str, Any]:
    source = shared.read_json(args.source_request)
    if not source:
        raise RuntimeError(f"cannot read source request: {args.source_request}")
    if source.get("mode") != "confirmatory" or source.get("run_config", {}).get("max_steps") != 30:
        raise RuntimeError("source request is not a 30-step confirmatory unit")
    arm = source["arm"]
    seed = source["seed"]
    unit_fingerprint = source["fingerprint"]
    stack_fingerprint = source["stack_fingerprint"]
    hf_repo = source["hf_repo"]
    tracking = source["tracking"]
    source_snapshots = shared.snapshot_sources(
        args.output_dir, [EVALUATOR, TRAINING_HELPERS, SECURE_EXEC, ENVIRONMENT_CHECK]
    )
    recovery_contract = {
        "schema_version": "e1-evaluation-recovery-v1",
        "source_unit_fingerprint": unit_fingerprint,
        "evaluator_sha256": shared.sha256_file(EVALUATOR),
        "training_helpers_sha256": shared.sha256_file(TRAINING_HELPERS),
        "secure_exec_sha256": shared.sha256_file(SECURE_EXEC),
        "environment_check_sha256": shared.sha256_file(ENVIRONMENT_CHECK),
        "eval_batch_size": args.eval_batch_size,
        "progress_save_every": args.progress_save_every,
    }
    recovery_fingerprint = shared.fingerprint(recovery_contract)
    session = f"e1-eval-{arm}-s{seed}-{recovery_fingerprint[:6]}"[:40]
    unit = f"e1__{arm}__s{seed}"
    wandb_run_name = source["wandb_run_name"]
    result_path = args.output_dir / "results" / f"{unit}__confirmatory.json"
    request_path = args.output_dir / "requests" / f"{unit}__eval__{recovery_fingerprint[:12]}.json"
    log_path = args.output_dir / "logs" / f"{unit}__eval__{recovery_fingerprint[:12]}.log"
    script_args = [
        "--arm", arm,
        "--seed", str(seed),
        "--unit-fingerprint", unit_fingerprint,
        "--stack-fingerprint", stack_fingerprint,
        "--hf-repo", hf_repo,
        "--checkpoint-step", "30",
        "--heldout-n", "500",
        "--max-completion-length", "1024",
        "--eval-batch-size", str(args.eval_batch_size),
        "--progress-save-every", str(args.progress_save_every),
        "--wandb-project", tracking["wandb_project"],
        "--wandb-group", tracking["wandb_group"],
        "--wandb-run-name", wandb_run_name,
    ]
    if tracking.get("wandb_entity"):
        script_args.extend(["--wandb-entity", tracking["wandb_entity"]])

    execution_plan = [
        ["colab", f"--auth={args.auth}", "new", "--gpu", args.gpu, "--session", session],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         str(TRAINING_HELPERS), "/content/e1_grpo_confirmatory.py"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         str(EVALUATOR), "/content/e1_evaluate_checkpoint.py"],
        ["colab", f"--auth={args.auth}", "install", "--session", session, *shared.PACKAGE_PINS],
        ["colab", f"--auth={args.auth}", "exec", "--session", session,
         "--file", str(ENVIRONMENT_CHECK), "--timeout", "120"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         "<ephemeral-secret-file>", "/content/.e1-eval-secrets.json"],
        ["colab", f"--auth={args.auth}", "upload", "--session", session,
         "<ephemeral-request-file>", "/content/e1-evaluation-request.json"],
        ["colab", f"--auth={args.auth}", "exec", "--session", session,
         "--file", str(SECURE_EXEC), "--timeout", str(args.timeout)],
        ["colab", f"--auth={args.auth}", "stop", "--session", session],
    ]
    recovery_request = {
        **recovery_contract,
        "status": "dry-run" if args.dry_run else "launching",
        "recovery_fingerprint": recovery_fingerprint,
        "source_request": str(args.source_request),
        "source_unit_fingerprint": unit_fingerprint,
        "stack_fingerprint": stack_fingerprint,
        "hf_repo": hf_repo,
        "session": session,
        "execution_plan": execution_plan,
        "launcher_retry_policy": {
            "exec_attempts": args.exec_attempts,
            "exec_retry_seconds": args.exec_retry_seconds,
            "preserve_session": True,
        },
        "source_snapshots": source_snapshots,
        "updated_at": shared.utc_now(),
    }
    shared.atomic_json(request_path, recovery_request)
    if args.dry_run:
        for command in execution_plan:
            print(shlex.join(command))
        return {"status": "dry-run", "request_path": str(request_path)}

    credentials = shared.load_credentials()
    lines: list[str] = []
    return_code = 0
    failed_step = None
    started_at = shared.utc_now()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[launch-recovery] {unit} session={session} source_checkpoint=30", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("[launcher] credentials staged out of band; values are not logged\n")
        with tempfile.TemporaryDirectory(prefix="e1-eval-recovery-") as staging:
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
                execution_plan[4],
                ["colab", f"--auth={args.auth}", "upload", "--session", session,
                 str(secret_path), "/content/.e1-eval-secrets.json"],
                ["colab", f"--auth={args.auth}", "upload", "--session", session,
                 str(invocation_path), "/content/e1-evaluation-request.json"],
                execution_plan[7],
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
        **recovery_request,
        "started_at": started_at,
        "completed_at": shared.utc_now(),
        "return_code": return_code,
        "failed_step": failed_step,
        "log_path": str(log_path),
        "request_path": str(request_path),
        "fingerprint": unit_fingerprint,
    }
    if return_code:
        failed = {**base, "status": "failed", "error": "colab CLI returned non-zero"}
        shared.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}
    try:
        result = shared.result_from_log(lines)
        expected_steps = [5, 10, 15, 20, 25, 30]
        verification = shared.verify_remote_artifacts(credentials, result, expected_steps)
        manifest = shared.download_manifest(credentials, result)
        shared.validate_remote_manifest(
            manifest, result, unit_fingerprint, expected_steps, "confirmatory"
        )
        args.mode = "confirmatory"
        args.arm = arm
        args.seed = seed
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


def main() -> int:
    args = parse_args()
    if args.exec_attempts < 1 or args.exec_retry_seconds < 0:
        raise SystemExit("exec attempts must be positive and retry seconds non-negative")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.source_request = args.source_request.expanduser().resolve()
    for path in (EVALUATOR, TRAINING_HELPERS, SECURE_EXEC, ENVIRONMENT_CHECK, args.source_request):
        if not path.is_file():
            raise SystemExit(f"missing required artifact: {path}")
    status = run_recovery(args)
    print("[recovery] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
