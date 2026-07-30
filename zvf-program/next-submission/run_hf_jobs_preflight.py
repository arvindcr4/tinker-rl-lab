#!/usr/bin/env python3
"""Launch and verify one next-submission preflight on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from huggingface_hub import HfApi


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
COMMON_LAUNCHER = HERE / "run_preflight.py"
REMOTE_SCRIPT = HERE / "remote_preflight.py"
SAMPLER = HERE / "contrast_sampler.py"
TRL_ADAPTER = HERE / "trl_sampler_adapter.py"
PROTOCOL = HERE / "preregistration.json"
AUTHORIZATION = HERE / "execution_authorization.json"
DESIGN_VERIFIER = HERE / "verify_design.py"
DEFAULT_OUTPUT = HERE / "results" / "hf-jobs-preflight"
TRACKIO_VERSION = "0.34.0"
PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
    f"trackio=={TRACKIO_VERSION}",
)
TASKS = ("gsm8k", "math500")
ARMS = ("grpo_g8", "contrast_early_stop_g2_to_g8")
FLAVOR_TO_GPU = {
    "l4x1": "L4",
    "a10g-large": "A10G",
    "a100-large": "A100",
    "h200": "H200",
}


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
    parser.add_argument("--flavor", choices=tuple(FLAVOR_TO_GPU), default="a100-large")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--wait-timeout", default="2h15m")
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="next-submission-preflight-hf-jobs")
    parser.add_argument("--hf-repo-prefix", default="arvindcr4/tinker-rl-next-preflight-hf")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args(argv)


def embedded_sources() -> dict[str, bytes]:
    """Return the exact executable sources materialized inside the ephemeral job."""

    return {path.name: path.read_bytes() for path in (REMOTE_SCRIPT, SAMPLER, TRL_ADAPTER)}


def build_job_script() -> str:
    """Build a self-contained PEP 723 entrypoint without embedding credentials."""

    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in embedded_sources().items()
    }
    dependency_lines = "\n".join(f'#   "{pin}",' for pin in PACKAGE_PINS)
    return f"""# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
{dependency_lines}
# ]
# ///

import base64
import json
import os
from pathlib import Path
import runpy
import sys

FILES = json.loads({json.dumps(json.dumps(encoded))})
source_root = Path("/tmp/next-submission-preflight-source")
source_root.mkdir(parents=True, exist_ok=True)
for name, payload in FILES.items():
    (source_root / name).write_bytes(base64.b64decode(payload))
os.environ["NEXT_PREFLIGHT_SOURCE_ROOT"] = str(source_root)
os.environ["NEXT_PREFLIGHT_REPORT_TO"] = "wandb,trackio"
sys.path.insert(0, str(source_root))
script = source_root / "remote_preflight.py"
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
"""


def job_stage(job: Any) -> str:
    stage = getattr(getattr(job, "status", None), "stage", None)
    return str(getattr(stage, "value", stage or "unknown")).upper()


def sanitize_provider_error(exc: Exception, credentials: dict[str, str]) -> str:
    """Return a bounded provider error that cannot contain submitted secrets."""

    message = str(exc)
    for value in credentials.values():
        if value:
            message = message.replace(value, "<redacted>")
    return message


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
    gpu = FLAVOR_TO_GPU[args.flavor]
    stack_fingerprint = common.fingerprint(
        {
            "runtime_packages": list(PACKAGE_PINS),
            "provider": "huggingface_jobs",
            "hardware_flavor": args.flavor,
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
        "provider": "huggingface_jobs",
        "hardware_flavor": args.flavor,
        "source_commit": commit,
        "protocol_sha256": protocol_sha256,
        "stack_fingerprint": stack_fingerprint,
        "runtime_packages": list(PACKAGE_PINS[:-1]),
        "tracking_packages": {"trackio": TRACKIO_VERSION},
        "source_snapshots": snapshots,
    }
    request["fingerprint"] = common.fingerprint(request)
    unit = f"{args.task}__{args.arm}__s{args.seed}"
    hf_repo = (
        f"{args.hf_repo_prefix}-{args.task}-{args.arm[:12]}-s{args.seed}-"
        f"{request['fingerprint'][:8]}"
    )
    wandb_run_name = (
        f"next-preflight-hf-{args.task}-{args.arm}-s{args.seed}-{request['fingerprint'][:8]}"
    )
    result_path = output_dir / "results" / f"{unit}.json"
    request_path = output_dir / "requests" / f"{unit}__{request['fingerprint'][:12]}.json"
    script_path = output_dir / "scripts" / f"{unit}__{request['fingerprint'][:12]}.py"

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
        "huggingface_jobs",
        "--hardware-flavor",
        args.flavor,
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

    job_script = build_job_script()
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(job_script, encoding="utf-8")
    launched = {
        **request,
        "status": "dry-run" if args.dry_run else "submitting",
        "hf_repo": hf_repo,
        "wandb_run_name": wandb_run_name,
        "job_script_sha256": common.sha256_file(script_path),
        "timeout": args.timeout,
        "estimated_cost_cap_usd": 5.0
        if args.flavor == "a100-large" and args.timeout == "2h"
        else None,
        "submitted_at": None,
        "updated_at": utc_now(),
    }
    helpers.atomic_json(request_path, launched)
    if args.dry_run:
        return {
            "status": "dry-run",
            "request_path": str(request_path),
            "script_path": str(script_path),
        }

    credentials = helpers.load_credentials()
    api = HfApi(token=credentials["HF_TOKEN"])
    try:
        job = api.run_uv_job(
            str(script_path),
            script_args=script_args,
            flavor=args.flavor,
            timeout=args.timeout,
            secrets=credentials,
            python="3.12",
        )
    except Exception as exc:
        error = common.failure_summary(sanitize_provider_error(exc, credentials).splitlines())
        rejected = {
            **launched,
            "status": "allocation-rejected",
            "failure_phase": "provider-submission",
            "provider_job_id": None,
            "provider_job_url": None,
            "allocation_started": False,
            "error": error,
            "updated_at": utc_now(),
        }
        helpers.atomic_json(request_path, rejected)
        failed = {
            **request,
            "status": "failed",
            "failure_phase": "provider-submission",
            "provider_job_id": None,
            "provider_job_url": None,
            "allocation_started": False,
            "request_path": str(request_path),
            "completed_at": utc_now(),
            "error": error,
        }
        helpers.atomic_json(result_path, failed)
        return {
            "status": "failed",
            "result_path": str(result_path),
            "job_url": None,
        }
    submitted = {
        **launched,
        "status": "submitted",
        "provider_job_id": job.id,
        "provider_job_url": job.url,
        "submitted_at": utc_now(),
        "updated_at": utc_now(),
    }
    helpers.atomic_json(request_path, submitted)
    print(
        "[hf-job-submitted] "
        + json.dumps(
            {"job_id": job.id, "url": job.url, "flavor": args.flavor, "timeout": args.timeout},
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.wait:
        return {
            "status": "submitted",
            "job_id": job.id,
            "job_url": job.url,
            "request_path": str(request_path),
        }

    wait = subprocess.run(
        ["hf", "jobs", "wait", "--timeout", args.wait_timeout, job.id],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    final_job = api.inspect_job(job_id=job.id)
    lines = list(api.fetch_job_logs(job_id=job.id))
    base = {
        **request,
        "provider_job_id": job.id,
        "provider_job_url": job.url,
        "provider_job_stage": job_stage(final_job),
        "request_path": str(request_path),
        "completed_at": utc_now(),
        "wait_return_code": wait.returncode,
    }
    if wait.returncode or job_stage(final_job) != "COMPLETED":
        failed = {**base, "status": "failed", "error": common.failure_summary(lines)}
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path), "job_url": job.url}
    try:
        result = common.result_from_log(lines)
        manifest, verification = common.verify_remote(credentials, result, request)
    except Exception as exc:
        failed = {**base, "status": "failed", "error": str(exc)}
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path), "job_url": job.url}
    complete = {
        **base,
        "status": "completed",
        "payload": result,
        "manifest": manifest,
        "remote_verification": verification,
        "fingerprint": request["fingerprint"],
    }
    helpers.atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path), "job_url": job.url}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed <= 0:
        raise SystemExit("seed must be positive")
    status = run_unit(args)
    print("[hf-jobs-preflight] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
