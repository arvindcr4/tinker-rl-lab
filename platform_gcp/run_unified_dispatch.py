#!/usr/bin/env python3
"""GCP Spot A100 launcher for non-TRL frameworks (unified dispatch).

Sister to ``zvf-program/next-submission/run_gcp_preflight.py`` — the frozen,
hash-anchored TRL preflight launcher. That file is bound to the preregistered
protocol and must stay byte-identical, so framework dispatch for
verl/openrlhf/skyrl/tinker on GCP lives HERE instead. This launcher provisions
the same Spot A100 VM shape and runs
``python -m platform_local.unified --framework <fw> --backend local`` on it — the
same per-framework code path as the local backend.

TRL on GCP still routes through the frozen launcher (it carries the validated
receipt/HF/W&B plumbing). This launcher is intentionally non-frozen: it is
infrastructure for the framework × backend matrix, not part of the preregistered
protocol, and may evolve freely.

Secrets: provide ``HF_TOKEN`` / ``WANDB_API_KEY`` / ``TINKER_API_KEY`` in the
local environment; they are forwarded to the VM as instance metadata and the
entry script surfaces them before training. (Not a preregistered run, so metadata
pass-through is acceptable — the frozen launcher's Secret-Manager path is what
the protocol uses.)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import time
from pathlib import Path

# Spot A100 shape — mirrors the frozen launcher's constants (single source of
# truth is zvf-program/next-submission/run_gcp_preflight.py). Duplicated here so
# this file can evolve without touching the frozen one.
PROJECT = "electric-armor-388216"
ZONE = "us-central1-a"
MACHINE_TYPE = "a2-highgpu-1g"
GPU = "A100"
IMAGE_PROJECT = "deeplearning-platform-release"
IMAGE = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580-v20260730"
DEFAULT_MAX_RUN_DURATION = "90m"
CANONICAL_MODEL = "Qwen/Qwen3-8B"
DONE_MARKER = "GCP_UNIFIED_DISPATCH_DONE"
FAIL_MARKER = "GCP_UNIFIED_DISPATCH_FAIL"

# Env vars forwarded local -> VM metadata -> entry script.
FORWARDED_ENV = ("HF_TOKEN", "WANDB_API_KEY", "TINKER_API_KEY")


def build_entry_script(*, framework: str, model: str, forwarded: dict[str, str]) -> str:
    """VM-side Python: surface secrets, clone the repo, run unified dispatch, write result.json."""
    return f"""import json, os, subprocess, sys
from pathlib import Path

# Surface forwarded secrets into the training environment.
{chr(10).join(f'os.environ.setdefault({k!r}, {v!r})' for k, v in forwarded.items())}

FRAMEWORK = {framework!r}
MODEL = {model!r}
REPO = Path("/root/tinker-rl-lab")

# Clone the repo (carries platform_local.unified + the per-framework drivers).
subprocess.run(
    ["git", "clone", "https://github.com/pes-llm-research/tinker-rl-lab.git", str(REPO)],
    check=False,
)

cmd = [
    sys.executable, "-m", "platform_local.unified",
    "--framework", FRAMEWORK, "--backend", "local",
    "--model", MODEL, "--algorithm", "grpo",
]
print("GCP_UNIFIED_DISPATCH_CMD", " ".join(cmd), flush=True)
proc = subprocess.run(cmd, cwd=str(REPO))

result = {{"framework": FRAMEWORK, "model": MODEL, "platform": "gcp-a100-spot",
          "mode": "unified-dispatch", "exit_code": proc.returncode}}
Path("/root/dispatch_result.json").write_text(json.dumps(result, indent=2))
print({DONE_MARKER!r} if proc.returncode == 0 else {FAIL_MARKER!r}, flush=True)
sys.exit(proc.returncode)
"""


def build_startup_script(entry_script: str, *, max_run_duration: str) -> str:
    encoded = base64.b64encode(entry_script.encode("utf-8")).decode("ascii")
    return f"""#!/bin/bash
set -uo pipefail
exec > >(tee -a /var/log/dispatch.log /dev/ttyS0) 2>&1
echo GCP_UNIFIED_DISPATCH_BOOT
if [ -x /opt/conda/bin/python ]; then PY=/opt/conda/bin/python; else PY=$(command -v python3); fi
apt-get update -qq && apt-get install -y -qq git
printf '%s' '{encoded}' | base64 -d > /root/dispatch_entry.py
timeout {max_run_duration} "$PY" /root/dispatch_entry.py
echo GCP_UNIFIED_DISPATCH_EXIT=$?
sync; sleep 5; shutdown -h now
"""


def gcloud_binary() -> str:
    import shutil

    exe = shutil.which("gcloud")
    if exe is None:
        raise RuntimeError("gcloud CLI is unavailable")
    return exe


def create_command(*, gcloud: str, instance: str, project: str, zone: str,
                   startup_script: Path, max_run_duration: str) -> list[str]:
    return [
        gcloud, "compute", "instances", "create", instance,
        f"--project={project}", f"--zone={zone}",
        f"--machine-type={MACHINE_TYPE}",
        f"--service-account=webarena-runner@{project}.iam.gserviceaccount.com",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--accelerator=count=1,type=nvidia-tesla-a100",
        f"--create-disk=auto-delete=yes,boot=yes,image-project={IMAGE_PROJECT},image={IMAGE},size=200GB",
        f"--metadata-from-file=startup-script={startup_script}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=STOP",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--max-run-duration={max_run_duration}",
        "--no-shielded-integrity-monitoring",
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", required=True,
                   choices=["tinker", "verl", "openrlhf", "skyrl"])
    p.add_argument("--model", default=CANONICAL_MODEL)
    p.add_argument("--algorithm", default="grpo")
    p.add_argument("--task", default="gsm8k")
    p.add_argument("--seed", type=int, default=211)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--zone", default=ZONE)
    p.add_argument("--max-run-duration", default=DEFAULT_MAX_RUN_DURATION)
    p.add_argument("--instance", help="override instance name")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wait", action="store_true")
    return p.parse_args(argv)


def run_unit(args: argparse.Namespace) -> dict:
    forwarded = {k: os.environ.get(k, "") for k in FORWARDED_ENV}
    entry = build_entry_script(framework=args.framework, model=args.model, forwarded=forwarded)
    startup = build_startup_script(entry, max_run_duration=args.max_run_duration)
    instance = args.instance or f"gcp-unified-{args.framework}-{int(time.time())}"

    plan = {
        "framework": args.framework, "model": args.model, "instance": instance,
        "project": args.project, "zone": args.zone, "machine_type": MACHINE_TYPE,
        "max_run_duration": args.max_run_duration, "status": "dry-run" if args.dry_run else "submitting",
    }

    if args.dry_run:
        print(f"[gcp-unified/{args.framework}] instance={instance}")
        print(f"  machine: {MACHINE_TYPE} ({GPU}) Spot, image={IMAGE}")
        print(f"  on-box: python -m platform_local.unified --framework {args.framework} --backend local --model {args.model}")
        print(f"  forwarded env: {', '.join(FORWARDED_ENV)}")
        return plan

    import tempfile
    gcloud = gcloud_binary()
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(startup)
        script_path = Path(f.name)

    try:
        print(f"[gcp-unified/{args.framework}] creating {instance} ...")
        subprocess.run(create_command(gcloud=gcloud, instance=instance, project=args.project,
                                      zone=args.zone, startup_script=script_path,
                                      max_run_duration=args.max_run_duration), check=True)
        plan["status"] = "running"
        if args.wait:
            plan["exit_code"] = _wait_for_marker(gcloud, instance, args.project, args.zone)
        else:
            print(f"  launched (no --wait); serial log: gcloud compute instances "
                  f"get-serial-port-output {instance} --zone={args.zone}")
    finally:
        if args.wait:
            subprocess.run([gcloud, "compute", "instances", "delete", instance,
                            f"--project={args.project}", f"--zone={args.zone}",
                            "--quiet"], check=False)
    return plan


def _wait_for_marker(gcloud: str, instance: str, project: str, zone: str,
                     timeout_s: int = 6300, poll_s: int = 15) -> int:
    """Poll serial output for the done/fail marker."""
    start = time.time()
    while time.time() - start < timeout_s:
        proc = subprocess.run(
            [gcloud, "compute", "instances", "get-serial-port-output", instance,
             f"--project={project}", f"--zone={zone}", "--port=1"],
            capture_output=True, text=True,
        )
        out = proc.stdout or ""
        if DONE_MARKER in out:
            print(f"[gcp-unified] {DONE_MARKER} observed.")
            return 0
        if FAIL_MARKER in out:
            print(f"[gcp-unified] {FAIL_MARKER} observed.")
            return 1
        time.sleep(poll_s)
    print("[gcp-unified] timeout waiting for completion marker.")
    return 2


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(run_unit(args), indent=2))


if __name__ == "__main__":
    main()
