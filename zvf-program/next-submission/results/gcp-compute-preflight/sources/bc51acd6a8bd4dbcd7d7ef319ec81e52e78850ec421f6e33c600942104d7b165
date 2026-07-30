#!/usr/bin/env python3
"""Launch and verify one next-submission preflight on a Spot GCP A100 VM."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
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
DEFAULT_OUTPUT = HERE / "results" / "gcp-compute-preflight"
PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
)
PROJECT = "electric-armor-388216"
ZONE = "us-central1-a"
MACHINE_TYPE = "a2-highgpu-1g"
GPU = "A100"
HARDWARE_FLAVOR = "a2-highgpu-1g-spot"
IMAGE_PROJECT = "deeplearning-platform-release"
IMAGE = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260730"
SERVICE_ACCOUNT = "webarena-runner@electric-armor-388216.iam.gserviceaccount.com"
SECRET_NAMES = {"HF_TOKEN": "hf-token", "WANDB_API_KEY": "wandb-api-key"}
SPOT_HOURLY_USD = 1.92802
DEFAULT_MAX_RUN_DURATION = "90m"
DEFAULT_MAX_RUN_HOURS = 1.5
ESTIMATED_COST_CAP_USD = 3.0
TASKS = ("gsm8k", "math500")
ARMS = ("grpo_g8", "contrast_early_stop_g2_to_g8")
TERMINAL_STAGES = {"TERMINATED", "ABSENT"}


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
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--zone", default=ZONE)
    parser.add_argument("--service-account", default=SERVICE_ACCOUNT)
    parser.add_argument("--max-run-duration", default=DEFAULT_MAX_RUN_DURATION)
    parser.add_argument("--wait-timeout-seconds", type=int, default=6300)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="next-submission-preflight-gcp")
    parser.add_argument("--hf-repo-prefix", default="arvindcr4/tinker-rl-next-preflight-gcp")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args(argv)


def gcloud_binary() -> str:
    executable = shutil.which("gcloud")
    if executable is None:
        raise RuntimeError("gcloud CLI is unavailable")
    return executable


def embedded_sources() -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in (REMOTE_SCRIPT, SAMPLER, TRL_ADAPTER)}


def build_entry_script(
    *, script_args: list[str], project: str, secret_names: dict[str, str]
) -> str:
    """Build the VM entrypoint; secret values never enter the local bundle."""

    encoded = {
        name: base64.b64encode(content).decode("ascii")
        for name, content in embedded_sources().items()
    }
    return f"""import base64
import json
import os
from pathlib import Path
import runpy
import sys
import urllib.parse
import urllib.request

METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
METADATA_HEADERS = {{"Metadata-Flavor": "Google"}}


def get_json(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


token_payload = get_json(
    METADATA_ROOT + "/instance/service-accounts/default/token", METADATA_HEADERS
)
access_token = token_payload["access_token"]
secret_headers = {{"Authorization": "Bearer " + access_token}}
for environment_name, secret_name in {secret_names!r}.items():
    encoded_name = urllib.parse.quote(secret_name, safe="")
    payload = get_json(
        "https://secretmanager.googleapis.com/v1/projects/"
        + {project!r}
        + "/secrets/"
        + encoded_name
        + "/versions/latest:access",
        secret_headers,
    )
    os.environ[environment_name] = base64.b64decode(payload["payload"]["data"]).decode(
        "utf-8"
    )

FILES = json.loads({json.dumps(json.dumps(encoded))})
SCRIPT_ARGS = json.loads({json.dumps(json.dumps(script_args))})
source_root = Path("/opt/next-submission-preflight-source")
source_root.mkdir(parents=True, exist_ok=True)
for name, payload in FILES.items():
    (source_root / name).write_bytes(base64.b64decode(payload))
os.environ["NEXT_PREFLIGHT_SOURCE_ROOT"] = str(source_root)
os.environ["NEXT_PREFLIGHT_OUTPUT_DIR"] = "/var/lib/next-preflight/output"
os.environ["NEXT_PREFLIGHT_RESULT_PATH"] = "/var/lib/next-preflight/result.json"
os.environ["NEXT_PREFLIGHT_REPORT_TO"] = "wandb"
sys.path.insert(0, str(source_root))
script = source_root / "remote_preflight.py"
sys.argv = [str(script), *SCRIPT_ARGS]
runpy.run_path(str(script), run_name="__main__")
"""


def build_startup_script(entry_script: str) -> str:
    encoded = base64.b64encode(entry_script.encode("utf-8")).decode("ascii")
    requirements = " ".join(PACKAGE_PINS)
    return f"""#!/bin/bash
set -uo pipefail
umask 077
mkdir -p /opt/next-submission-preflight /var/lib/next-preflight
exec > >(tee -a /var/log/next-preflight.log /dev/ttyS0) 2>&1
echo NEXT_PREFLIGHT_BOOT_STARTED
printf '%s' '{encoded}' | base64 -d > /opt/next-submission-preflight/main.py
if [ -x /opt/conda/bin/python ]; then
  PYTHON_BIN=/opt/conda/bin/python
else
  PYTHON_BIN=$(command -v python3)
fi
"$PYTHON_BIN" -m pip install --disable-pip-version-check --no-input {requirements}
INSTALL_EXIT=$?
if [ "$INSTALL_EXIT" -eq 0 ]; then
  "$PYTHON_BIN" /opt/next-submission-preflight/main.py
  RUN_EXIT=$?
else
  RUN_EXIT=$INSTALL_EXIT
fi
echo NEXT_PREFLIGHT_EXIT_CODE=$RUN_EXIT
sync
sleep 10
shutdown -h now
exit "$RUN_EXIT"
"""


def create_command(
    *,
    gcloud: str,
    instance: str,
    project: str,
    zone: str,
    service_account: str,
    max_run_duration: str,
    startup_script: Path,
) -> list[str]:
    return [
        gcloud,
        "compute",
        "instances",
        "create",
        instance,
        f"--project={project}",
        f"--zone={zone}",
        f"--machine-type={MACHINE_TYPE}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
        f"--max-run-duration={max_run_duration}",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--image={IMAGE}",
        f"--image-project={IMAGE_PROJECT}",
        "--boot-disk-size=100GB",
        "--boot-disk-type=pd-balanced",
        f"--service-account={service_account}",
        "--scopes=cloud-platform",
        "--network=default",
        "--labels=purpose=next-preflight,evidence=none",
        f"--metadata-from-file=startup-script={startup_script}",
        "--quiet",
    ]


def instance_status(gcloud: str, *, instance: str, project: str, zone: str) -> str:
    described = subprocess.run(
        [
            gcloud,
            "compute",
            "instances",
            "describe",
            instance,
            f"--project={project}",
            f"--zone={zone}",
            "--format=value(status)",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if described.returncode:
        return "ABSENT"
    return described.stdout.strip().upper() or "UNKNOWN"


def wait_for_instance(
    *,
    gcloud: str,
    instance: str,
    project: str,
    zone: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        stage = instance_status(gcloud, instance=instance, project=project, zone=zone)
        print(f"[gcp-status] {stage}", flush=True)
        if stage in TERMINAL_STAGES:
            return stage
        time.sleep(poll_seconds)
    return "TIMEOUT"


def serial_output(gcloud: str, *, instance: str, project: str, zone: str) -> tuple[int, str]:
    output = subprocess.run(
        [
            gcloud,
            "compute",
            "instances",
            "get-serial-port-output",
            instance,
            f"--project={project}",
            f"--zone={zone}",
            "--port=1",
            "--start=0",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return output.returncode, output.stdout


def serial_output_with_retry(
    gcloud: str,
    *,
    instance: str,
    project: str,
    zone: str,
    attempts: int = 12,
    delay_seconds: int = 5,
) -> tuple[int, str]:
    """Wait for GCP to publish the final stopped-VM serial buffer."""

    last = (1, "serial output was never requested")
    for attempt in range(attempts):
        last = serial_output(gcloud, instance=instance, project=project, zone=zone)
        code, output = last
        if code == 0 and (
            "NEXT_PREFLIGHT_EXIT_CODE=" in output or "NEXT_PREFLIGHT_RESULT " in output
        ):
            return last
        if attempt + 1 < attempts:
            time.sleep(delay_seconds)
    return last


def delete_instance(gcloud: str, *, instance: str, project: str, zone: str) -> dict[str, Any]:
    deleted = subprocess.run(
        [
            gcloud,
            "compute",
            "instances",
            "delete",
            instance,
            f"--project={project}",
            f"--zone={zone}",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    absent = instance_status(gcloud, instance=instance, project=project, zone=zone) == "ABSENT"
    return {
        "delete_return_code": deleted.returncode,
        "instance_absent_verified": absent,
    }


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
    stack_fingerprint = common.fingerprint(
        {
            "runtime_packages": list(PACKAGE_PINS),
            "provider": "gcp_compute",
            "hardware_flavor": HARDWARE_FLAVOR,
            "accelerator": GPU,
            "machine_type": MACHINE_TYPE,
            "image": IMAGE,
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
        "gpu": GPU,
        "provider": "gcp_compute",
        "hardware_flavor": HARDWARE_FLAVOR,
        "project": args.project,
        "zone": args.zone,
        "machine_type": MACHINE_TYPE,
        "image": IMAGE,
        "service_account": args.service_account,
        "credential_source": "gcp-secret-manager-existing-references",
        "secret_names": SECRET_NAMES,
        "source_commit": commit,
        "protocol_sha256": protocol_sha256,
        "stack_fingerprint": stack_fingerprint,
        "runtime_packages": list(PACKAGE_PINS),
        "source_snapshots": snapshots,
    }
    request["fingerprint"] = common.fingerprint(request)
    unit = f"{args.task}__{args.arm}__s{args.seed}"
    instance = (
        f"next-preflight-{args.task}-{args.arm[:8]}-s{args.seed}-{request['fingerprint'][:8]}"
    )
    console_url = (
        f"https://console.cloud.google.com/compute/instancesDetail/zones/{args.zone}/"
        f"instances/{instance}?project={args.project}"
    )
    hf_repo = (
        f"{args.hf_repo_prefix}-{args.task}-{args.arm[:12]}-s{args.seed}-"
        f"{request['fingerprint'][:8]}"
    )
    wandb_run_name = (
        f"next-preflight-gcp-{args.task}-{args.arm}-s{args.seed}-{request['fingerprint'][:8]}"
    )
    result_path = output_dir / "results" / f"{unit}.json"
    request_path = output_dir / "requests" / f"{unit}__{request['fingerprint'][:12]}.json"
    bundle_dir = output_dir / "bundles" / request["fingerprint"]
    entry_path = bundle_dir / "main.py"
    startup_path = bundle_dir / "startup.sh"
    serial_path = output_dir / "logs" / f"{unit}__{request['fingerprint'][:12]}.log"

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
        "gcp_compute",
        "--hardware-flavor",
        HARDWARE_FLAVOR,
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

    entry_script = build_entry_script(
        script_args=script_args, project=args.project, secret_names=SECRET_NAMES
    )
    startup_script = build_startup_script(entry_script)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(entry_script, encoding="utf-8")
    startup_path.write_text(startup_script, encoding="utf-8")
    launched = {
        **request,
        "status": "dry-run" if args.dry_run else "submitting",
        "instance": instance,
        "console_url": console_url,
        "hf_repo": hf_repo,
        "wandb_run_name": wandb_run_name,
        "entry_script_sha256": common.sha256_file(entry_path),
        "startup_script_sha256": common.sha256_file(startup_path),
        "max_run_duration": args.max_run_duration,
        "spot_hourly_usd_at_design": SPOT_HOURLY_USD,
        "estimated_cost_cap_usd": ESTIMATED_COST_CAP_USD,
        "cost_estimate_source": "https://cloud.google.com/spot-vms/pricing",
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

    gcloud = gcloud_binary()
    created = subprocess.run(
        create_command(
            gcloud=gcloud,
            instance=instance,
            project=args.project,
            zone=args.zone,
            service_account=args.service_account,
            max_run_duration=args.max_run_duration,
            startup_script=startup_path,
        ),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if created.returncode:
        error = common.failure_summary(created.stdout.splitlines())
        partial_stage = instance_status(
            gcloud, instance=instance, project=args.project, zone=args.zone
        )
        cleanup = (
            delete_instance(gcloud, instance=instance, project=args.project, zone=args.zone)
            if partial_stage != "ABSENT"
            else {"delete_return_code": None, "instance_absent_verified": True}
        )
        rejected = {
            **launched,
            "status": "allocation-rejected",
            "failure_phase": "provider-submission",
            "allocation_started": False,
            "cleanup": cleanup,
            "error": error,
            "updated_at": utc_now(),
        }
        helpers.atomic_json(request_path, rejected)
        failed = {
            **request,
            "status": "failed",
            "failure_phase": "provider-submission",
            "allocation_started": False,
            "cleanup": cleanup,
            "instance": instance,
            "console_url": console_url,
            "request_path": str(request_path),
            "completed_at": utc_now(),
            "error": error,
        }
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path), "console_url": None}

    submitted_at = utc_now()
    submitted = {
        **launched,
        "status": "submitted",
        "submission_output": common.failure_summary(created.stdout.splitlines()),
        "submitted_at": submitted_at,
        "updated_at": submitted_at,
    }
    helpers.atomic_json(request_path, submitted)
    print(
        "[gcp-submitted] "
        + json.dumps(
            {"instance": instance, "url": console_url, "cost_cap_usd": 3.0},
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.wait:
        return {
            "status": "submitted",
            "instance": instance,
            "console_url": console_url,
            "request_path": str(request_path),
        }

    outcome: dict[str, Any]
    try:
        stage = wait_for_instance(
            gcloud=gcloud,
            instance=instance,
            project=args.project,
            zone=args.zone,
            timeout_seconds=args.wait_timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        serial_return_code, serial = serial_output_with_retry(
            gcloud, instance=instance, project=args.project, zone=args.zone
        )
        serial_path.parent.mkdir(parents=True, exist_ok=True)
        serial_path.write_text(serial, encoding="utf-8")
        base = {
            **request,
            "instance": instance,
            "console_url": console_url,
            "provider_job_stage": stage,
            "request_path": str(request_path),
            "serial_log_path": str(serial_path),
            "serial_return_code": serial_return_code,
            "completed_at": utc_now(),
            "allocation_started": True,
        }
        if stage != "TERMINATED" or serial_return_code:
            outcome = {
                **base,
                "status": "failed",
                "failure_phase": "vm-runtime",
                "error": common.failure_summary(serial.splitlines()),
            }
        else:
            try:
                result = common.result_from_log(serial.splitlines())
                credentials = helpers.load_credentials()
                manifest, verification = common.verify_remote(credentials, result, request)
                outcome = {
                    **base,
                    "status": "completed",
                    "payload": result,
                    "manifest": manifest,
                    "remote_verification": verification,
                    "fingerprint": request["fingerprint"],
                }
            except Exception as exc:
                outcome = {
                    **base,
                    "status": "failed",
                    "failure_phase": "receipt-verification",
                    "error": str(exc),
                }
    except Exception as exc:
        outcome = {
            **request,
            "status": "failed",
            "failure_phase": "launcher-runtime",
            "instance": instance,
            "console_url": console_url,
            "request_path": str(request_path),
            "completed_at": utc_now(),
            "error": str(exc),
        }
    cleanup = delete_instance(gcloud, instance=instance, project=args.project, zone=args.zone)
    outcome["cleanup"] = cleanup
    if not cleanup["instance_absent_verified"]:
        outcome["pre_cleanup_status"] = outcome["status"]
        outcome["status"] = "failed"
        outcome["failure_phase"] = "cleanup"
        outcome["error"] = "temporary GCP instance deletion was not verified"
    helpers.atomic_json(result_path, outcome)
    terminal_request = {
        **submitted,
        "status": outcome["status"],
        "provider_job_stage": outcome.get("provider_job_stage"),
        "completed_at": outcome["completed_at"],
        "cleanup": cleanup,
        "updated_at": utc_now(),
    }
    if outcome["status"] == "failed":
        terminal_request.update(
            {
                "failure_phase": outcome.get("failure_phase"),
                "error": outcome.get("error"),
            }
        )
    helpers.atomic_json(request_path, terminal_request)
    return {
        "status": outcome["status"],
        "result_path": str(result_path),
        "console_url": console_url,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.seed <= 0:
        raise SystemExit("seed must be positive")
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        raise SystemExit("poll interval must be in [1, 60] seconds")
    if args.wait_timeout_seconds <= 0:
        raise SystemExit("wait timeout must be positive")
    if not args.dry_run and not args.wait:
        raise SystemExit("live GCP runs require --wait so cleanup can be verified")
    if args.max_run_duration != DEFAULT_MAX_RUN_DURATION:
        raise SystemExit(f"max run duration must stay frozen at {DEFAULT_MAX_RUN_DURATION}")
    if SPOT_HOURLY_USD * DEFAULT_MAX_RUN_HOURS >= ESTIMATED_COST_CAP_USD:
        raise SystemExit("frozen Spot compute estimate exceeds the receipt cost cap")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    status = run_unit(args)
    print("[gcp-preflight] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
