#!/usr/bin/env python3
"""Non-frozen GCP dispatch protocol v2: secret references and owned resources.

Only Secret Manager resource names enter metadata. Use a narrowly scoped service
account. Receipts are process receipts, not independently verified science results.
No provider access occurs during dry-run. Remote runtime dependencies still need a
reviewed image; the dispatcher does not claim cross-framework equivalence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

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
FORWARDED_ENV = ("HF_TOKEN", "WANDB_API_KEY", "TINKER_API_KEY")
PROTOCOL = "unified-dispatch-v2"
GCLOUD_TIMEOUT_S = 120
MAX_RUNTIME_S = 24 * 3600


def build_entry_script(
    *,
    framework: str,
    model: str,
    forwarded: dict | None = None,
    secret_refs: dict | None = None,
    project: str = PROJECT,
    algorithm: str = "grpo",
    task: str = "gsm8k",
    seed: int = 211,
    run_id: str = "",
    spec_hash: str = "",
    repo_revision: str = "",
) -> str:
    if forwarded and any(forwarded.values()):
        raise ValueError("Raw credential forwarding is prohibited; use secret_refs")
    refs = secret_refs or {}
    for key, ref in refs.items():
        if key not in FORWARDED_ENV or not re.fullmatch(
            r"projects/[\w-]+/secrets/[\w-]+/versions/[0-9]+", ref
        ):
            raise ValueError(
                "Expected an allowed environment name and versioned Secret Manager reference"
            )
    return f"""import json, os, subprocess, sys
from pathlib import Path
for key, resource in {refs!r}.items():
    _, project, _, secret, _, version = resource.split("/")
    value = subprocess.check_output(["gcloud", "secrets", "versions", "access", version,
        "--secret", secret, "--project", project], text=True).strip()
    if not value:
        raise RuntimeError("Required credential is empty")
    os.environ[key] = value
repo = Path("/root/tinker-rl-lab")
subprocess.run(["git", "clone", "https://github.com/pes-llm-research/tinker-rl-lab.git", str(repo)], check=True)
subprocess.run(["git", "checkout", "--detach", {repo_revision!r}], cwd=repo, check=True)
cmd = [sys.executable, "-m", "platform_local.unified", "--framework", {framework!r},
       "--backend", "local", "--model", {model!r}, "--algorithm", {algorithm!r},
       "--task", {task!r}, "--seed", {str(seed)!r}]
proc = subprocess.run(cmd, cwd=repo)
receipt = {{"protocol": {PROTOCOL!r}, "run_id": {run_id!r}, "spec_hash": {spec_hash!r},
           "repo_revision": {repo_revision!r}, "exit_code": proc.returncode,
           "status": "process_completed" if proc.returncode == 0 else "failed",
           "completed_steps": None, "artifact_hashes": {{}}}}
path = Path("/root/dispatch_result.json")
path.with_suffix(".tmp").write_text(json.dumps(receipt))
path.with_suffix(".tmp").replace(path)
sys.exit(proc.returncode)
"""


def build_startup_script(entry_script: str, *, max_run_duration: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*[smh]", max_run_duration):
        raise ValueError("max-run-duration must be a positive duration such as 90m")
    seconds = int(max_run_duration[:-1]) * {"s": 1, "m": 60, "h": 3600}[max_run_duration[-1]]
    if seconds > MAX_RUNTIME_S:
        raise ValueError("max-run-duration exceeds the 24-hour safety ceiling")
    encoded = base64.b64encode(entry_script.encode()).decode()
    return f"""#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/dispatch.log /dev/ttyS0) 2>&1
if [ -x /opt/conda/bin/python ]; then PY=/opt/conda/bin/python; else PY=$(command -v python3); fi
apt-get update -qq && apt-get install -y -qq git
printf '%s' '{encoded}' | base64 -d > /root/dispatch_entry.py
status=0
timeout {max_run_duration} "$PY" /root/dispatch_entry.py || status=$?
echo GCP_UNIFIED_DISPATCH_EXIT=$status
# The controller collects the receipt. The provider-side DELETE deadline is the
# independent crash/timeout backstop; do not stop before receipt collection.
exit "$status"
"""


def gcloud_binary() -> str:
    import shutil

    exe = shutil.which("gcloud")
    if not exe:
        raise RuntimeError("gcloud CLI is unavailable")
    return exe


def create_command(
    *,
    gcloud: str,
    instance: str,
    project: str,
    zone: str,
    startup_script: Path,
    max_run_duration: str,
    run_id: str = "",
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
        f"--service-account=webarena-runner@{project}.iam.gserviceaccount.com",
        "--scopes=https://www.googleapis.com/auth/cloud-platform",
        "--accelerator=count=1,type=nvidia-tesla-a100",
        f"--create-disk=auto-delete=yes,boot=yes,image-project={IMAGE_PROJECT},image={IMAGE},size=200GB",
        f"--metadata-from-file=startup-script={startup_script}",
        f"--labels=dispatch-run={run_id}",
        "--provisioning-model=SPOT",
        "--instance-termination-action=DELETE",
        "--maintenance-policy=TERMINATE",
        "--no-restart-on-failure",
        f"--max-run-duration={max_run_duration}",
        "--format=json",
    ]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", required=True, choices=["tinker", "verl", "openrlhf", "skyrl"])
    p.add_argument("--model", default=CANONICAL_MODEL)
    p.add_argument("--algorithm", default="grpo", choices=["grpo"])
    p.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    p.add_argument("--seed", type=int, default=211)
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--zone", default=ZONE)
    p.add_argument("--max-run-duration", default=DEFAULT_MAX_RUN_DURATION)
    p.add_argument("--instance")
    p.add_argument(
        "--repo-revision", help="Reviewed 40-character Git commit (required for execution)"
    )
    p.add_argument(
        "--secret-env", action="append", default=[], metavar="ENV=VERSIONED_SECRET_RESOURCE"
    )
    p.add_argument("--receipt-dir", type=Path, default=Path("dispatch_receipts"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args(argv)


def _owned(gcloud, instance, project, zone, run_id, instance_id):
    proc = subprocess.run(
        [
            gcloud,
            "compute",
            "instances",
            "describe",
            instance,
            f"--project={project}",
            f"--zone={zone}",
            "--format=json",
        ],
        capture_output=True,
        text=True,
        timeout=GCLOUD_TIMEOUT_S,
    )
    if proc.returncode:
        return False
    obj = json.loads(proc.stdout)
    return (
        str(obj.get("id")) == str(instance_id)
        and obj.get("labels", {}).get("dispatch-run") == run_id
    )


def _wait_for_receipt(
    gcloud, instance, project, zone, run_id, spec_hash, timeout_s=6300, poll_s=15
):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        proc = subprocess.run(
            [
                gcloud,
                "compute",
                "ssh",
                instance,
                f"--project={project}",
                f"--zone={zone}",
                "--quiet",
                "--command=sudo cat /root/dispatch_result.json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        try:
            receipt = json.loads(proc.stdout) if proc.returncode == 0 else None
        except ValueError:
            receipt = None
        if receipt is not None:
            if (
                receipt.get("protocol") != PROTOCOL
                or receipt.get("run_id") != run_id
                or receipt.get("spec_hash") != spec_hash
                or type(receipt.get("exit_code")) is not int
                or receipt.get("status") not in ("process_completed", "failed")
            ):
                raise ValueError(
                    "Receipt identity or terminal status does not match this invocation"
                )
            return receipt
        time.sleep(poll_s)
    raise TimeoutError("No authenticated terminal receipt collected before deadline")


def _wait_for_marker(*args, **kwargs):
    """Serial strings are untrusted diagnostics and can never prove completion."""
    raise RuntimeError("Serial markers cannot verify completion; collect a terminal receipt")


def run_unit(args):
    if args.framework in ("skyrl", "verl") and not args.dry_run:
        raise NotImplementedError(
            "SkyRL/verl lack a validated scientific training receipt adapter; rejected before provisioning"
        )
    if args.algorithm != "grpo" or args.task != "gsm8k":
        raise ValueError("Unsupported experiment; rejected before provisioning")
    revision = getattr(args, "repo_revision", None)
    if not args.dry_run and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Execution requires --repo-revision with a reviewed full commit hash")
    refs = dict(value.split("=", 1) for value in getattr(args, "secret_env", []))
    run_id = uuid.uuid4().hex
    spec = {key: getattr(args, key) for key in ("framework", "model", "algorithm", "task", "seed")}
    spec.update(protocol=PROTOCOL, repo_revision=revision)
    spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    entry = build_entry_script(
        **{k: spec[k] for k in ("framework", "model", "algorithm", "task", "seed")},
        secret_refs=refs,
        run_id=run_id,
        spec_hash=spec_hash,
        repo_revision=revision or "",
    )
    startup = build_startup_script(entry, max_run_duration=args.max_run_duration)
    instance = args.instance or f"gcp-unified-{args.framework}-{run_id[:12]}"
    plan = dict(
        spec,
        instance=instance,
        run_id=run_id,
        spec_hash=spec_hash,
        status="dry-run" if args.dry_run else "submitting",
    )
    if args.dry_run:
        return plan
    gcloud = gcloud_binary()
    created = False
    instance_id = None
    receipt_saved = False
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as handle:
        handle.write(startup)
        script_path = Path(handle.name)
    try:
        proc = subprocess.run(
            create_command(
                gcloud=gcloud,
                instance=instance,
                project=args.project,
                zone=args.zone,
                startup_script=script_path,
                max_run_duration=args.max_run_duration,
                run_id=run_id,
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=GCLOUD_TIMEOUT_S,
        )
        created = True
        instance_id = json.loads(proc.stdout)[0]["id"]
        plan["status"] = "running"
        if args.wait:
            receipt = _wait_for_receipt(
                gcloud, instance, args.project, args.zone, run_id, spec_hash
            )
            receipt_dir = Path(getattr(args, "receipt_dir", "dispatch_receipts"))
            receipt_dir.mkdir(parents=True, exist_ok=True)
            path = receipt_dir / f"{run_id}.json"
            path.with_suffix(".tmp").write_text(json.dumps(receipt, indent=2))
            path.with_suffix(".tmp").replace(path)
            receipt_saved = True
            plan.update(exit_code=receipt["exit_code"], status=receipt["status"], receipt=str(path))
    finally:
        script_path.unlink(missing_ok=True)
        # Never delete on failed/ambiguous create, identity mismatch, or before
        # durable collection. The provider deadline bounds orphan retention.
        if created and args.wait and receipt_saved and instance_id is not None:
            if _owned(gcloud, instance, args.project, args.zone, run_id, instance_id):
                subprocess.run(
                    [
                        gcloud,
                        "compute",
                        "instances",
                        "delete",
                        instance,
                        f"--project={args.project}",
                        f"--zone={args.zone}",
                        "--quiet",
                    ],
                    check=True,
                    timeout=GCLOUD_TIMEOUT_S,
                )
    return plan


def main(argv=None):
    result = run_unit(parse_args(argv))
    print(json.dumps(result, indent=2))
    return result.get("exit_code", 0)


if __name__ == "__main__":
    raise SystemExit(main())
