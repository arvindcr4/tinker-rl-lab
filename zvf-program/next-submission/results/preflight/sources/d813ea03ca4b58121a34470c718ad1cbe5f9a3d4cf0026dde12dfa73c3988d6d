#!/usr/bin/env python3
"""Launch and independently verify one next-submission Colab preflight."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
REMOTE_SCRIPT = HERE / "remote_preflight.py"
SAMPLER = HERE / "contrast_sampler.py"
TRL_ADAPTER = HERE / "trl_sampler_adapter.py"
SECURE_EXEC = HERE / "secure_exec_preflight.py"
ENVIRONMENT_CHECK = HERE / "verify_preflight_environment.py"
PROTOCOL = HERE / "preregistration.json"
AUTHORIZATION = HERE / "execution_authorization.json"
DESIGN_VERIFIER = HERE / "verify_design.py"
DEFAULT_OUTPUT = HERE / "results" / "preflight"
PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
)
ARMS = ("grpo_g8", "contrast_early_stop_g2_to_g8")
TASKS = ("gsm8k", "math500")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def load_e1_helpers() -> Any:
    path = REPO_ROOT / "zvf-program/audit/run_colab_e1_confirmatory.py"
    spec = importlib.util.spec_from_file_location("e1_colab_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the audited Colab helper module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_commit(*, require_clean: bool) -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        if status:
            raise RuntimeError(
                "live preflight requires a clean committed source tree; commit the task-scoped files first"
            )
    return commit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--auth", choices=("oauth2", "adc"), default="oauth2")
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="next-submission-preflight")
    parser.add_argument("--hf-repo-prefix", default="arvindcr4/tinker-rl-next-preflight")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args(argv)


def result_from_log(lines: list[str]) -> dict[str, Any]:
    for raw in reversed(lines):
        line = ANSI_RE.sub("", raw)
        marker = line.find("NEXT_PREFLIGHT_RESULT ")
        if marker >= 0:
            value = json.loads(line[marker + len("NEXT_PREFLIGHT_RESULT ") :])
            if not isinstance(value, dict):
                break
            return value
    raise ValueError("Colab output did not contain a NEXT_PREFLIGHT_RESULT record")


def verify_wandb_run(api_key: str, run_url: str) -> dict[str, Any]:
    match = re.fullmatch(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", run_url)
    if match is None:
        raise RuntimeError("unexpected W&B run URL")
    entity, project, run_id = match.groups()
    query = {
        "query": "query Run($entity: String!, $project: String!, $run: String!) { project(name: $project, entityName: $entity) { run(name: $run) { name state } } }",
        "variables": {"entity": entity, "project": project, "run": run_id},
    }
    authorization = base64.b64encode(f"api:{api_key}".encode()).decode()
    request = Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps(query).encode(),
        headers={"Authorization": f"Basic {authorization}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    run = ((payload.get("data") or {}).get("project") or {}).get("run") or {}
    if run.get("name") != run_id:
        raise RuntimeError("W&B API did not return the requested run")
    return {"run_id": run_id, "state": run.get("state"), "run_url": run_url}


def validate_manifest(
    manifest: dict[str, Any],
    result: dict[str, Any],
    request: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != "aiml-next-preflight-run-v1":
        raise RuntimeError("remote manifest schema mismatch")
    if (
        manifest.get("status") != "complete"
        or manifest.get("evidence_class") != "preflight-not-evidence"
    ):
        raise RuntimeError("remote manifest evidence status mismatch")
    if result.get("schema_version") != "aiml-next-preflight-result-v1":
        raise RuntimeError("remote result schema mismatch")
    if result.get("evidence_class") != "preflight-not-evidence":
        raise RuntimeError("remote result was promoted beyond preflight")
    run_config = manifest.get("run_config") or {}
    expected = {
        "task": request["task"],
        "arm": request["arm"],
        "seed": request["seed"],
        "unit_fingerprint": request["fingerprint"],
        "stack_fingerprint": request["stack_fingerprint"],
        "source_commit": request["source_commit"],
        "protocol_sha256": request["protocol_sha256"],
        "max_steps": 1,
        "heldout_n": 8,
    }
    wrong = {
        key: (run_config.get(key), value)
        for key, value in expected.items()
        if run_config.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"remote run config mismatch: {wrong}")
    if manifest.get("audit_record") != result.get("audit_record"):
        raise RuntimeError("manifest and result audit records differ")
    audit = manifest["audit_record"]
    if audit.get("evidence_tier") != "preflight_not_scientific_evidence":
        raise RuntimeError("preflight evidence tier drift")
    if audit.get("heldout_n") != 8 or not 0 <= audit.get("heldout_correct", -1) <= 8:
        raise RuntimeError("held-out counts are invalid")
    if not math.isclose(
        audit["heldout_accuracy"], audit["heldout_correct"] / 8, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("held-out accuracy disagrees with counts")
    groups = audit.get("rollout_groups")
    generated = audit.get("generated_rollouts")
    charged = audit.get("charged_generated_tokens")
    if (
        type(groups) is not int
        or groups <= 0
        or type(generated) is not int
        or type(charged) is not int
    ):
        raise RuntimeError("sampler count telemetry is invalid")
    fractions = [audit.get(f"{label}_fraction") for label in ("all_wrong", "all_correct", "mixed")]
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1
        for value in fractions
    ):
        raise RuntimeError("sampler fraction telemetry is invalid")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("sampler fractions do not partition groups")
    if request["arm"] == "grpo_g8" and generated != groups * 8:
        raise RuntimeError("baseline did not generate G8 for every prompt")
    if request["arm"] == "contrast_early_stop_g2_to_g8":
        mixed_groups = round(fractions[2] * groups)
        if generated != groups * 2 + mixed_groups * 6:
            raise RuntimeError("intervention generation count violates the G2-to-G8 rule")
    if charged <= 0:
        raise RuntimeError("charged generated token count is not positive")
    trace = manifest.get("heldout_trace")
    if not isinstance(trace, list) or len(trace) != 8:
        raise RuntimeError("held-out trace is incomplete")
    if [row.get("index") for row in trace] != list(range(8)):
        raise RuntimeError("held-out trace indices are not contiguous")
    if sum(row.get("correct") is True for row in trace) != audit["heldout_correct"]:
        raise RuntimeError("held-out trace disagrees with audit count")
    source_files = manifest.get("source_files") or {}
    expected_source_files = {
        path.name: sha256_file(path) for path in (REMOTE_SCRIPT, SAMPLER, TRL_ADAPTER)
    }
    if source_files != expected_source_files:
        raise RuntimeError("remote executable source hashes differ from local source")


def verify_remote(
    credentials: dict[str, str],
    result: dict[str, Any],
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    remote = result.get("remote") or {}
    api = HfApi(token=credentials["HF_TOKEN"])
    for filename in ("run_manifest.json", "final/adapter_model.safetensors"):
        if not api.file_exists(
            repo_id=remote["hf_repo"],
            repo_type="model",
            filename=filename,
            revision=remote["hf_commit"],
        ):
            raise RuntimeError(f"remote artifact missing: {filename}")
    manifest_path = hf_hub_download(
        repo_id=remote["hf_repo"],
        repo_type="model",
        filename="run_manifest.json",
        revision=remote["hf_commit"],
        token=credentials["HF_TOKEN"],
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(manifest, result, request)
    wandb_record = verify_wandb_run(credentials["WANDB_API_KEY"], remote["wandb_run_url"])
    if wandb_record["run_id"] != remote["wandb_run_id"] or wandb_record["state"] != "finished":
        raise RuntimeError("W&B run is absent, mismatched, or unfinished")
    return manifest, {
        "verified_at": utc_now(),
        "hf_repo": remote["hf_repo"],
        "hf_commit": remote["hf_commit"],
        "wandb": wandb_record,
    }


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    helpers = load_e1_helpers()
    output_dir = args.output_dir.expanduser().resolve()
    subprocess.run([sys.executable, str(DESIGN_VERIFIER)], cwd=REPO_ROOT, check=True)
    commit = source_commit(require_clean=not args.dry_run)
    sources = [
        REMOTE_SCRIPT,
        SAMPLER,
        TRL_ADAPTER,
        SECURE_EXEC,
        ENVIRONMENT_CHECK,
        PROTOCOL,
        AUTHORIZATION,
        Path(__file__).resolve(),
    ]
    snapshots = helpers.snapshot_sources(output_dir, sources)
    protocol_sha256 = sha256_file(PROTOCOL)
    stack_fingerprint = fingerprint(
        {
            "runtime_packages": list(PACKAGE_PINS),
            "accelerator": args.gpu,
            "trainer": "trl-1.8.0-custom-rollout-g8-v1",
            "sampler_sha256": sha256_file(SAMPLER),
            "adapter_sha256": sha256_file(TRL_ADAPTER),
        }
    )
    request = {
        "schema_version": "aiml-next-preflight-request-v1",
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "gpu": args.gpu,
        "source_commit": commit,
        "protocol_sha256": protocol_sha256,
        "stack_fingerprint": stack_fingerprint,
        "runtime_packages": list(PACKAGE_PINS),
        "source_snapshots": snapshots,
    }
    request["fingerprint"] = fingerprint(request)
    unit = f"{args.task}__{args.arm}__s{args.seed}"
    session = f"next-pre-{args.task[:4]}-{args.arm[:8]}-{request['fingerprint'][:6]}"[:40]
    hf_repo = f"{args.hf_repo_prefix}-{args.task}-{args.arm[:12]}-s{args.seed}-{request['fingerprint'][:8]}"
    wandb_run_name = (
        f"next-preflight-{args.task}-{args.arm}-s{args.seed}-{request['fingerprint'][:8]}"
    )
    result_path = output_dir / "results" / f"{unit}.json"
    request_path = output_dir / "requests" / f"{unit}__{request['fingerprint'][:12]}.json"
    log_path = output_dir / "logs" / f"{unit}__{request['fingerprint'][:12]}.log"

    existing = helpers.read_json(result_path)
    if (
        not args.rerun
        and existing
        and existing.get("status") == "completed"
        and existing.get("fingerprint") == request["fingerprint"]
    ):
        return {"status": "skipped-compatible", "result_path": str(result_path)}

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

    execution_plan = [
        ["colab", f"--auth={args.auth}", "new", "--gpu", args.gpu, "--session", session],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            str(REMOTE_SCRIPT),
            "/content/remote_preflight.py",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            str(SAMPLER),
            "/content/contrast_sampler.py",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            str(TRL_ADAPTER),
            "/content/trl_sampler_adapter.py",
        ],
        ["colab", f"--auth={args.auth}", "install", "--session", session, *PACKAGE_PINS],
        [
            "colab",
            f"--auth={args.auth}",
            "exec",
            "--session",
            session,
            "--file",
            str(ENVIRONMENT_CHECK),
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
            "/content/.next-preflight-secrets.json",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "upload",
            "--session",
            session,
            "<ephemeral-request-file>",
            "/content/next-preflight-request.json",
        ],
        [
            "colab",
            f"--auth={args.auth}",
            "exec",
            "--session",
            session,
            "--file",
            str(SECURE_EXEC),
            "--timeout",
            str(args.timeout),
        ],
        ["colab", f"--auth={args.auth}", "stop", "--session", session],
    ]
    launched = {
        **request,
        "status": "dry-run" if args.dry_run else "launching",
        "session": session,
        "hf_repo": hf_repo,
        "wandb_run_name": wandb_run_name,
        "execution_plan": execution_plan,
        "updated_at": utc_now(),
    }
    helpers.atomic_json(request_path, launched)
    if args.dry_run:
        for command in execution_plan:
            print(shlex.join(command))
        return {"status": "dry-run", "request_path": str(request_path)}

    credentials = helpers.load_credentials()
    lines: list[str] = []
    return_code = 0
    failed_step = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("[launcher] credentials staged out of band; values are not logged\n")
        with tempfile.TemporaryDirectory(prefix="next-colab-preflight-") as staging:
            staging_path = Path(staging)
            secret_path = staging_path / "secrets.json"
            invocation_path = staging_path / "request.json"
            secret_path.write_text(json.dumps(credentials), encoding="utf-8")
            secret_path.chmod(0o600)
            invocation_path.write_text(json.dumps({"script_args": script_args}), encoding="utf-8")
            commands = [
                *execution_plan[:6],
                [
                    "colab",
                    f"--auth={args.auth}",
                    "upload",
                    "--session",
                    session,
                    str(secret_path),
                    "/content/.next-preflight-secrets.json",
                ],
                [
                    "colab",
                    f"--auth={args.auth}",
                    "upload",
                    "--session",
                    session,
                    str(invocation_path),
                    "/content/next-preflight-request.json",
                ],
                execution_plan[8],
            ]
            try:
                for index, command in enumerate(commands):
                    return_code = helpers.run_logged(command, log_handle, lines)
                    if return_code:
                        failed_step = index
                        break
            finally:
                helpers.stop_session(args.auth, session, log_handle)

    base = {
        **request,
        "session": session,
        "hf_repo": hf_repo,
        "completed_at": utc_now(),
        "return_code": return_code,
        "failed_step": failed_step,
        "log_path": str(log_path),
        "request_path": str(request_path),
    }
    if return_code:
        failed = {**base, "status": "failed", "error": "colab CLI returned non-zero"}
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}
    try:
        result = result_from_log(lines)
        manifest, verification = verify_remote(credentials, result, request)
    except Exception as exc:
        failed = {**base, "status": "failed", "error": str(exc)}
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}
    complete = {
        **base,
        "status": "completed",
        "payload": result,
        "manifest": manifest,
        "remote_verification": verification,
        "fingerprint": request["fingerprint"],
    }
    helpers.atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed <= 0 or args.timeout <= 0:
        raise SystemExit("seed and timeout must be positive")
    status = run_unit(args)
    print("[next-preflight] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
