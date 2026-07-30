#!/usr/bin/env python3
"""Launch and verify one next-submission preflight on Kaggle GPUs."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COMMON_LAUNCHER = HERE / "run_preflight.py"
REMOTE_SCRIPT = HERE / "remote_preflight.py"
SAMPLER = HERE / "contrast_sampler.py"
TRL_ADAPTER = HERE / "trl_sampler_adapter.py"
PROTOCOL = HERE / "preregistration.json"
AUTHORIZATION = HERE / "execution_authorization.json"
DESIGN_VERIFIER = HERE / "verify_design.py"
DEFAULT_OUTPUT = HERE / "results" / "kaggle-preflight"
PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
)
TASKS = ("gsm8k", "math500")
ARMS = ("grpo_g8", "contrast_early_stop_g2_to_g8")
ACCELERATOR_TO_GPU = {
    "NvidiaTeslaA100": "A100",
    "NvidiaL4X1": "L4",
}
TERMINAL_STAGES = {"COMPLETE", "ERROR", "CANCELLED"}
STATUS_RE = re.compile(r"KernelWorkerStatus\.([A-Z]+)")


def load_common() -> Any:
    spec = importlib.util.spec_from_file_location(
        "next_submission_common_preflight", COMMON_LAUNCHER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load common preflight launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument(
        "--accelerator", choices=tuple(ACCELERATOR_TO_GPU), default="NvidiaTeslaA100"
    )
    parser.add_argument("--runtime-limit-seconds", type=int, default=7200)
    parser.add_argument("--wait-timeout-seconds", type=int, default=8100)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="next-submission-preflight-kaggle")
    parser.add_argument("--hf-repo-prefix", default="arvindcr4/tinker-rl-next-preflight-kaggle")
    parser.add_argument("--kaggle-owner", default="arvindcr")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args(argv)


def kaggle_binary() -> str:
    executable = shutil.which("kaggle")
    if executable is None:
        raise RuntimeError("kaggle CLI is unavailable")
    return executable


def embedded_sources() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in (REMOTE_SCRIPT, SAMPLER, TRL_ADAPTER)}


def build_kernel_script(script_args: list[str]) -> str:
    """Build a self-contained Kaggle entrypoint without embedding credentials."""

    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in embedded_sources().items()
    }
    return f"""import base64
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
for secret_name in ("HF_TOKEN", "WANDB_API_KEY"):
    secret_value = user_secrets.get_secret(secret_name)
    if not secret_value:
        raise RuntimeError(f"required Kaggle secret is unavailable: {{secret_name}}")
    os.environ[secret_name] = secret_value

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *{list(PACKAGE_PINS)!r},
    ],
    check=True,
)

FILES = json.loads({json.dumps(json.dumps(encoded))})
SCRIPT_ARGS = json.loads({json.dumps(json.dumps(script_args))})
source_root = Path("/kaggle/working/next-submission-preflight-source")
source_root.mkdir(parents=True, exist_ok=True)
for name, payload in FILES.items():
    (source_root / name).write_bytes(base64.b64decode(payload))
os.environ["NEXT_PREFLIGHT_SOURCE_ROOT"] = str(source_root)
os.environ["NEXT_PREFLIGHT_OUTPUT_DIR"] = "/kaggle/working/next-preflight-output"
os.environ["NEXT_PREFLIGHT_RESULT_PATH"] = "/kaggle/working/next_preflight_result.json"
os.environ["NEXT_PREFLIGHT_REPORT_TO"] = "wandb"
sys.path.insert(0, str(source_root))
script = source_root / "remote_preflight.py"
sys.argv = [str(script), *SCRIPT_ARGS]
runpy.run_path(str(script), run_name="__main__")
"""


def kernel_metadata(*, owner: str, slug: str) -> dict[str, Any]:
    return {
        "id": f"{owner}/{slug}",
        "title": slug.replace("-", " ").title()[:80],
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "machine_shape": "",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def parse_kernel_stage(output: str) -> str:
    match = STATUS_RE.search(output)
    return match.group(1) if match else "UNKNOWN"


def get_kernel_status(kaggle: str, kernel_id: str) -> tuple[str, str]:
    status = subprocess.run(
        [kaggle, "kernels", "status", kernel_id],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return parse_kernel_stage(status.stdout), status.stdout


def wait_for_kernel(
    *, kaggle: str, kernel_id: str, timeout_seconds: int, poll_seconds: int
) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    last_output = ""
    while time.monotonic() < deadline:
        stage, last_output = get_kernel_status(kaggle, kernel_id)
        print(f"[kaggle-status] {stage}", flush=True)
        if stage in TERMINAL_STAGES:
            return stage, last_output
        time.sleep(poll_seconds)
    return "TIMEOUT", last_output


def fetch_kernel_logs(kaggle: str, kernel_id: str) -> list[str]:
    logs = subprocess.run(
        [kaggle, "kernels", "logs", kernel_id],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return logs.stdout.splitlines()


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    common = load_common()
    helpers = common.load_e1_helpers()
    output_dir = args.output_dir.expanduser().resolve()
    subprocess.run([sys.executable, str(DESIGN_VERIFIER)], cwd=REPO_ROOT, check=True)
    commit = common.source_commit(require_clean=not args.dry_run)
    source_files = [
        REMOTE_SCRIPT,
        SAMPLER,
        TRL_ADAPTER,
        COMMON_LAUNCHER,
        Path(__file__).resolve(),
        PROTOCOL,
        AUTHORIZATION,
    ]
    snapshots = helpers.snapshot_sources(output_dir, source_files)
    protocol_sha256 = common.sha256_file(PROTOCOL)
    gpu = ACCELERATOR_TO_GPU[args.accelerator]
    stack_fingerprint = common.fingerprint(
        {
            "runtime_packages": list(PACKAGE_PINS),
            "provider": "kaggle",
            "hardware_flavor": args.accelerator,
            "accelerator": gpu,
            "trainer": "trl-1.8.0-custom-rollout-g8-v1",
            "sampler_sha256": common.sha256_file(SAMPLER),
            "adapter_sha256": common.sha256_file(TRL_ADAPTER),
        }
    )
    request = {
        "schema_version": "aiml-next-preflight-request-v1",
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "gpu": gpu,
        "provider": "kaggle",
        "hardware_flavor": args.accelerator,
        "source_commit": commit,
        "protocol_sha256": protocol_sha256,
        "stack_fingerprint": stack_fingerprint,
        "runtime_packages": list(PACKAGE_PINS),
        "source_snapshots": snapshots,
    }
    request["fingerprint"] = common.fingerprint(request)
    unit = f"{args.task}__{args.arm}__s{args.seed}"
    slug = f"next-preflight-{args.task}-{args.arm[:8]}-s{args.seed}-{request['fingerprint'][:8]}"
    kernel_id = f"{args.kaggle_owner}/{slug}"
    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
    hf_repo = (
        f"{args.hf_repo_prefix}-{args.task}-{args.arm[:12]}-s{args.seed}-"
        f"{request['fingerprint'][:8]}"
    )
    wandb_run_name = (
        f"next-preflight-kaggle-{args.task}-{args.arm}-s{args.seed}-{request['fingerprint'][:8]}"
    )
    result_path = output_dir / "results" / f"{unit}.json"
    request_path = output_dir / "requests" / f"{unit}__{request['fingerprint'][:12]}.json"
    bundle_dir = output_dir / "bundles" / request["fingerprint"]
    script_path = bundle_dir / "main.py"
    metadata_path = bundle_dir / "kernel-metadata.json"

    existing = helpers.read_json(result_path)
    if (
        not args.rerun
        and existing
        and existing.get("status") == "completed"
        and existing.get("fingerprint") == request["fingerprint"]
    ):
        return {"status": "skipped-compatible", "result_path": str(result_path)}
    common.archive_incompatible_result(
        output_dir=output_dir,
        unit=unit,
        existing=existing,
        new_fingerprint=request["fingerprint"],
        atomic_json=helpers.atomic_json,
    )

    script_args = [
        "--task",
        args.task,
        "--arm",
        args.arm,
        "--seed",
        str(args.seed),
        "--unit-fingerprint",
        request["fingerprint"],
        "--stack-fingerprint",
        stack_fingerprint,
        "--source-commit",
        commit,
        "--protocol-sha256",
        protocol_sha256,
        "--provider",
        "kaggle",
        "--hardware-flavor",
        args.accelerator,
        "--hf-repo",
        hf_repo,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        args.wandb_group,
        "--wandb-run-name",
        wandb_run_name,
    ]
    if args.wandb_entity:
        script_args.extend(["--wandb-entity", args.wandb_entity])

    bundle_dir.mkdir(parents=True, exist_ok=True)
    script_path.write_text(build_kernel_script(script_args), encoding="utf-8")
    helpers.atomic_json(metadata_path, kernel_metadata(owner=args.kaggle_owner, slug=slug))
    launched = {
        **request,
        "status": "dry-run" if args.dry_run else "submitting",
        "kernel_id": kernel_id,
        "kernel_url": kernel_url,
        "hf_repo": hf_repo,
        "wandb_run_name": wandb_run_name,
        "kernel_script_sha256": common.sha256_file(script_path),
        "kernel_metadata_sha256": common.sha256_file(metadata_path),
        "runtime_limit_seconds": args.runtime_limit_seconds,
        "estimated_cost_cap_usd": 0.0,
        "submitted_at": None,
        "updated_at": utc_now(),
    }
    helpers.atomic_json(request_path, launched)
    if args.dry_run:
        return {
            "status": "dry-run",
            "request_path": str(request_path),
            "bundle_dir": str(bundle_dir),
        }

    kaggle = kaggle_binary()
    push = subprocess.run(
        [
            kaggle,
            "kernels",
            "push",
            "--path",
            str(bundle_dir),
            "--timeout",
            str(args.runtime_limit_seconds),
            "--accelerator",
            args.accelerator,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if push.returncode:
        error = common.failure_summary(push.stdout.splitlines())
        rejected = {
            **launched,
            "status": "allocation-rejected",
            "failure_phase": "provider-submission",
            "allocation_started": False,
            "error": error,
            "updated_at": utc_now(),
        }
        helpers.atomic_json(request_path, rejected)
        failed = {
            **request,
            "status": "failed",
            "failure_phase": "provider-submission",
            "allocation_started": False,
            "kernel_id": kernel_id,
            "kernel_url": kernel_url,
            "request_path": str(request_path),
            "completed_at": utc_now(),
            "error": error,
        }
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path), "kernel_url": kernel_url}

    submitted = {
        **launched,
        "status": "submitted",
        "submission_output": common.failure_summary(push.stdout.splitlines()),
        "submitted_at": utc_now(),
        "updated_at": utc_now(),
    }
    helpers.atomic_json(request_path, submitted)
    print(
        "[kaggle-submitted] "
        + json.dumps(
            {"kernel_id": kernel_id, "url": kernel_url, "accelerator": args.accelerator},
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.wait:
        return {
            "status": "submitted",
            "kernel_id": kernel_id,
            "kernel_url": kernel_url,
            "request_path": str(request_path),
        }

    stage, status_output = wait_for_kernel(
        kaggle=kaggle,
        kernel_id=kernel_id,
        timeout_seconds=args.wait_timeout_seconds,
        poll_seconds=args.poll_seconds,
    )
    lines = fetch_kernel_logs(kaggle, kernel_id)
    base = {
        **request,
        "kernel_id": kernel_id,
        "kernel_url": kernel_url,
        "provider_job_stage": stage,
        "request_path": str(request_path),
        "completed_at": utc_now(),
    }
    if stage != "COMPLETE":
        failed = {
            **base,
            "status": "failed",
            "error": common.failure_summary(lines or status_output.splitlines()),
        }
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path), "kernel_url": kernel_url}

    downloads = output_dir / "downloads" / request["fingerprint"]
    downloads.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            kaggle,
            "kernels",
            "output",
            kernel_id,
            "--path",
            str(downloads),
            "--force",
            "--file-pattern",
            r".*next_preflight_result\.json$",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    remote_result_path = downloads / "next_preflight_result.json"
    try:
        if remote_result_path.is_file():
            result = json.loads(remote_result_path.read_text(encoding="utf-8"))
        else:
            result = common.result_from_log(lines)
        credentials = helpers.load_credentials()
        manifest, verification = common.verify_remote(credentials, result, request)
    except Exception as exc:
        failed = {**base, "status": "failed", "error": str(exc)}
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path), "kernel_url": kernel_url}
    complete = {
        **base,
        "status": "completed",
        "payload": result,
        "manifest": manifest,
        "remote_verification": verification,
        "fingerprint": request["fingerprint"],
    }
    helpers.atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path), "kernel_url": kernel_url}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed <= 0:
        raise SystemExit("seed must be positive")
    if args.runtime_limit_seconds <= 0 or args.wait_timeout_seconds <= 0:
        raise SystemExit("timeouts must be positive")
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise SystemExit("poll interval must be in [1, 60] seconds")
    status = run_unit(args)
    print("[kaggle-preflight] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
