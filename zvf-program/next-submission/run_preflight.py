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
import time
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
SETUP_SCRIPT = HERE / "setup_colab_preflight.py"
PROTOCOL = HERE / "preregistration.json"
AUTHORIZATION = HERE / "execution_authorization.json"
DESIGN_VERIFIER = HERE / "verify_design.py"
DEFAULT_OUTPUT = HERE / "results" / "preflight"
PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "jinja2==3.1.6",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
    "huggingface_hub==1.26.0",
)
ARMS = ("grpo_g8", "contrast_early_stop_g2_to_g8")
TASKS = ("gsm8k", "math500")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DECODER_CONTRACT = {
    "enable_thinking": False,
    "training": {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    },
    "evaluation": {"do_sample": False},
}


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
    parser.add_argument("--setup-timeout", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--wandb-project", default="tinker-rl-lab")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group", default="next-submission-preflight")
    parser.add_argument("--hf-repo-prefix", default="arvindcr4/tinker-rl-next-preflight")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.seed <= 0 or args.setup_timeout <= 0 or args.timeout <= 0:
        raise SystemExit("seed and timeouts must be positive")
    if "/" not in args.hf_repo_prefix or args.hf_repo_prefix.endswith("/"):
        raise SystemExit("hf-repo-prefix must include a namespace and repository prefix")


def colab_cli_version() -> str:
    completed = subprocess.run(
        ["colab", "version"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    match = re.search(r"\bVersion:\s*([0-9]+(?:\.[0-9]+){2})\b", completed.stdout)
    if match is None:
        raise RuntimeError("could not parse the Colab CLI version")
    return match.group(1)


def wandb_graphql(api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    authorization = base64.b64encode(f"api:{api_key}".encode()).decode()
    request = Request(
        "https://api.wandb.ai/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"Basic {authorization}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read())
    if payload.get("errors"):
        raise RuntimeError("W&B API rejected the verification query")
    return payload


def verify_tracking_credentials(
    credentials: dict[str, str],
    *,
    hf_repo_prefix: str,
) -> dict[str, str]:
    api = HfApi(token=credentials["HF_TOKEN"])
    identity = api.whoami()
    hf_identity = identity.get("name")
    namespace = hf_repo_prefix.split("/", 1)[0]
    organizations = {
        organization.get("name")
        for organization in identity.get("orgs", [])
        if isinstance(organization, dict)
    }
    if hf_identity != namespace and namespace not in organizations:
        raise RuntimeError("Hugging Face credential cannot write the requested namespace")

    payload = wandb_graphql(
        credentials["WANDB_API_KEY"],
        "query Viewer { viewer { username } }",
        {},
    )
    wandb_identity = ((payload.get("data") or {}).get("viewer") or {}).get("username")
    if not isinstance(wandb_identity, str) or not wandb_identity:
        raise RuntimeError("W&B credential did not resolve an authenticated viewer")
    return {"hf_identity": hf_identity, "wandb_identity": wandb_identity}


def result_from_log(lines: list[str]) -> dict[str, Any]:
    for raw in reversed(lines):
        line = ANSI_RE.sub("", raw)
        marker = line.find("NEXT_PREFLIGHT_RESULT ")
        if marker >= 0:
            value = json.loads(line[marker + len("NEXT_PREFLIGHT_RESULT ") :])
            if not isinstance(value, dict):
                break
            return value
    raise ValueError("remote output did not contain a NEXT_PREFLIGHT_RESULT record")


def failure_summary(lines: list[str], *, limit: int = 12) -> str:
    """Return a bounded ANSI-free failure tail for self-contained receipts."""

    cleaned = [ANSI_RE.sub("", line).strip() for line in lines]
    nonempty = [line for line in cleaned if line]
    return " | ".join(nonempty[-limit:]) or "colab CLI returned non-zero without output"


def archive_incompatible_result(
    *,
    output_dir: Path,
    unit: str,
    existing: dict[str, Any] | None,
    new_fingerprint: str,
    atomic_json: Any,
) -> Path | None:
    """Preserve an older unit receipt before a hardware/fingerprint retry."""

    if not existing or existing.get("fingerprint") == new_fingerprint:
        return None
    old_fingerprint = existing.get("fingerprint")
    if (
        not isinstance(old_fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", old_fingerprint) is None
    ):
        old_fingerprint = fingerprint(existing)
    history_path = output_dir / "results" / "history" / f"{unit}__{old_fingerprint[:12]}.json"
    atomic_json(history_path, existing)
    return history_path


def verify_wandb_run(
    api_key: str,
    run_url: str,
    *,
    expected_project: str | None,
    expected_entity: str | None,
) -> dict[str, Any]:
    match = re.fullmatch(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", run_url)
    if match is None:
        raise RuntimeError("unexpected W&B run URL")
    entity, project, run_id = match.groups()
    if (expected_project is not None and project != expected_project) or (
        expected_entity is not None and entity != expected_entity
    ):
        raise RuntimeError("W&B run URL differs from the requested tracking destination")
    query = {
        "query": "query Run($entity: String!, $project: String!, $run: String!) { project(name: $project, entityName: $entity) { run(name: $run) { name state } } }",
        "variables": {"entity": entity, "project": project, "run": run_id},
    }
    payload = wandb_graphql(api_key, query["query"], query["variables"])
    run = ((payload.get("data") or {}).get("project") or {}).get("run") or {}
    if run.get("name") != run_id:
        raise RuntimeError("W&B API did not return the requested run")
    return {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "state": run.get("state"),
        "run_url": run_url,
    }


def expected_hf_repo(request: dict[str, Any]) -> str:
    tracking = request.get("tracking")
    if not isinstance(tracking, dict):
        raise RuntimeError("request lacks a tracking contract")
    prefix = tracking.get("hf_repo_prefix")
    if not isinstance(prefix, str) or "/" not in prefix:
        raise RuntimeError("request lacks a valid Hugging Face repository prefix")
    return (
        f"{prefix}-{request['task']}-{request['arm'][:12]}-"
        f"s{request['seed']}-{request['fingerprint'][:8]}"
    )


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
    if result.get("status") != "completed":
        raise RuntimeError("remote result status is not completed")
    if result.get("evidence_class") != "preflight-not-evidence":
        raise RuntimeError("remote result was promoted beyond preflight")
    run_config = manifest.get("run_config") or {}
    if result.get("run_config") != run_config:
        raise RuntimeError("manifest and result run configs differ")
    expected = {
        "task": request["task"],
        "arm": request["arm"],
        "seed": request["seed"],
        "unit_fingerprint": request["fingerprint"],
        "stack_fingerprint": request["stack_fingerprint"],
        "source_commit": request["source_commit"],
        "protocol_sha256": request["protocol_sha256"],
        "provider": request["provider"],
        "hardware_flavor": request["hardware_flavor"],
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
    if request.get("decoder") != DECODER_CONTRACT or run_config.get("decoder") != DECODER_CONTRACT:
        raise RuntimeError("remote decoder contract differs from the frozen request")
    tracking_backends = run_config.get("tracking_backends")
    if not isinstance(tracking_backends, list) or "wandb" not in tracking_backends:
        raise RuntimeError("remote run config lacks required W&B tracking")
    tracking = request.get("tracking")
    if tracking is not None:
        if not isinstance(tracking, dict) or run_config.get("tracking") != {
            "wandb_project": tracking.get("wandb_project"),
            "wandb_entity": tracking.get("wandb_entity"),
            "wandb_group": tracking.get("wandb_group"),
        }:
            raise RuntimeError("remote run config differs from the requested tracking contract")
    if request["provider"] == "huggingface_jobs" and "trackio" not in tracking_backends:
        raise RuntimeError("Hugging Face Jobs run config lacks required Trackio tracking")
    if manifest.get("audit_record") != result.get("audit_record"):
        raise RuntimeError("manifest and result audit records differ")
    audit = manifest["audit_record"]
    if audit.get("status") != "complete":
        raise RuntimeError("preflight audit status is not complete")
    identity = {
        "task_id": request["task"],
        "arm_id": request["arm"],
        "seed": request["seed"],
    }
    wrong_identity = {
        key: (audit.get(key), value) for key, value in identity.items() if audit.get(key) != value
    }
    if wrong_identity:
        raise RuntimeError(f"preflight audit identity mismatch: {wrong_identity}")
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
    updated_groups = audit.get("updated_groups")
    if (
        type(groups) is not int
        or groups <= 0
        or type(generated) is not int
        or type(charged) is not int
        or type(updated_groups) is not int
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
    mixed_groups = round(fractions[2] * groups)
    if updated_groups != mixed_groups or not 0 <= updated_groups <= groups:
        raise RuntimeError("updated-group count disagrees with mixed-group telemetry")
    if request["arm"] == "contrast_early_stop_g2_to_g8":
        if generated != groups * 2 + mixed_groups * 6:
            raise RuntimeError("intervention generation count violates the G2-to-G8 rule")
    if charged < generated:
        raise RuntimeError("charged generated tokens are below the generated-rollout count")
    clipped = audit.get("completion_clipped_fraction")
    if not isinstance(clipped, (int, float)) or not math.isfinite(clipped) or not 0 <= clipped < 1:
        raise RuntimeError("completion clipping telemetry is invalid or all completions clipped")
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
    runtime_versions = manifest.get("runtime_versions")
    if not isinstance(runtime_versions, dict) or result.get("runtime_versions") != runtime_versions:
        raise RuntimeError("manifest and result runtime versions are missing or differ")
    runtime_packages = request.get("runtime_packages")
    if (
        not isinstance(runtime_packages, list)
        or not runtime_packages
        or not all(
            isinstance(pin, str) and re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", pin)
            for pin in runtime_packages
        )
    ):
        raise RuntimeError("request runtime package pins are invalid")
    expected_packages = dict(pin.split("==", 1) for pin in runtime_packages)
    if len(expected_packages) != len(runtime_packages):
        raise RuntimeError("request runtime package pins contain duplicate names")
    wrong_versions = {
        name: (runtime_versions.get(name), version)
        for name, version in expected_packages.items()
        if runtime_versions.get(name) != version
    }
    if wrong_versions:
        raise RuntimeError(f"remote runtime package mismatch: {wrong_versions}")
    if request["provider"] == "huggingface_jobs":
        tracking_packages = request.get("tracking_packages")
        if not isinstance(tracking_packages, dict) or runtime_versions.get(
            "trackio"
        ) != tracking_packages.get("trackio"):
            raise RuntimeError("remote Trackio version differs from the provider request")
    observed_gpu = runtime_versions.get("gpu")
    requested_gpu = request.get("gpu")
    if (
        not isinstance(observed_gpu, str)
        or not isinstance(requested_gpu, str)
        or re.search(rf"\b{re.escape(requested_gpu)}\b", observed_gpu, re.IGNORECASE) is None
    ):
        raise RuntimeError(
            f"observed GPU {observed_gpu!r} does not match requested accelerator {requested_gpu!r}"
        )


def verify_remote(
    credentials: dict[str, str],
    result: dict[str, Any],
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    remote = result.get("remote") or {}
    api = HfApi(token=credentials["HF_TOKEN"])
    tracking = request.get("tracking")
    expected_repo = expected_hf_repo(request) if tracking is not None else remote.get("hf_repo")
    if not isinstance(expected_repo, str) or remote.get("hf_repo") != expected_repo:
        raise RuntimeError("remote Hugging Face repository differs from the request")
    required_files = ("run_manifest.json", "final/adapter_model.safetensors")
    for filename in required_files:
        if not api.file_exists(
            repo_id=expected_repo,
            repo_type="model",
            filename=filename,
            revision=remote["hf_commit"],
        ):
            raise RuntimeError(f"remote artifact missing: {filename}")
    model_info = api.model_info(
        expected_repo,
        revision=remote["hf_commit"],
        files_metadata=True,
    )
    if model_info.private is not True:
        raise RuntimeError("remote Hugging Face repository is not private")
    if model_info.sha != remote["hf_commit"]:
        raise RuntimeError("remote Hugging Face revision did not resolve to the reported commit")
    sibling_names = {sibling.rfilename for sibling in model_info.siblings or []}
    if not set(required_files).issubset(sibling_names):
        raise RuntimeError("remote Hugging Face commit lacks required files")
    manifest_path = hf_hub_download(
        repo_id=expected_repo,
        repo_type="model",
        filename="run_manifest.json",
        revision=remote["hf_commit"],
        token=credentials["HF_TOKEN"],
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(manifest, result, request)
    wandb_record = verify_wandb_run(
        credentials["WANDB_API_KEY"],
        remote["wandb_run_url"],
        expected_project=tracking.get("wandb_project") if isinstance(tracking, dict) else None,
        expected_entity=tracking.get("wandb_entity") if isinstance(tracking, dict) else None,
    )
    if wandb_record["run_id"] != remote["wandb_run_id"] or wandb_record["state"] != "finished":
        raise RuntimeError("W&B run is absent, mismatched, or unfinished")
    return manifest, {
        "verified_at": utc_now(),
        "hf_repo": expected_repo,
        "hf_commit": remote["hf_commit"],
        "hf_private": True,
        "hf_files": list(required_files),
        "wandb": wandb_record,
    }


def recover_result_from_remote(
    credentials: dict[str, str],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct a dropped Colab result marker from final private artifacts."""

    repo = expected_hf_repo(request)
    api = HfApi(token=credentials["HF_TOKEN"])
    model_info = api.model_info(repo, files_metadata=True)
    if model_info.private is not True or not isinstance(model_info.sha, str):
        raise RuntimeError(
            "cannot recover from a non-private or unresolved Hugging Face repository"
        )
    names = {sibling.rfilename for sibling in model_info.siblings or []}
    required = {"run_manifest.json", "final/adapter_model.safetensors"}
    if not required.issubset(names):
        raise RuntimeError("cannot recover because final Hugging Face artifacts are incomplete")
    manifest_path = hf_hub_download(
        repo_id=repo,
        repo_type="model",
        filename="run_manifest.json",
        revision=model_info.sha,
        token=credentials["HF_TOKEN"],
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    wandb = manifest.get("wandb")
    if not isinstance(wandb, dict):
        raise RuntimeError("cannot recover because the manifest lacks W&B identity")
    return {
        "schema_version": "aiml-next-preflight-result-v1",
        "status": "completed",
        "evidence_class": manifest.get("evidence_class"),
        "audit_record": manifest.get("audit_record"),
        "run_config": manifest.get("run_config"),
        "runtime_versions": manifest.get("runtime_versions"),
        "remote": {
            "hf_repo": repo,
            "hf_commit": model_info.sha,
            "hf_manifest_path": "run_manifest.json",
            "hf_final_adapter_path": "final/adapter_model.safetensors",
            "wandb_run_id": wandb.get("run_id"),
            "wandb_run_url": wandb.get("run_url"),
        },
    }


def result_from_log_or_remote(
    lines: list[str],
    credentials: dict[str, str],
    request: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Parse streamed output, falling back to fully verifiable final artifacts."""

    try:
        return result_from_log(lines), None
    except ValueError as exc:
        return recover_result_from_remote(credentials, request), {
            "reason": str(exc),
            "source": "exact-private-hf-commit-and-finished-wandb-run",
        }


def build_execution_plan(
    args: argparse.Namespace,
    *,
    session: str,
    secret_source: str,
    invocation_source: str,
) -> list[list[str]]:
    prefix = ["colab", f"--auth={args.auth}"]
    return [
        [*prefix, "new", "--gpu", args.gpu, "--session", session],
        [
            *prefix,
            "upload",
            "--session",
            session,
            str(REMOTE_SCRIPT),
            "/content/remote_preflight.py",
        ],
        [
            *prefix,
            "upload",
            "--session",
            session,
            str(SAMPLER),
            "/content/contrast_sampler.py",
        ],
        [
            *prefix,
            "upload",
            "--session",
            session,
            str(TRL_ADAPTER),
            "/content/trl_sampler_adapter.py",
        ],
        [
            *prefix,
            "exec",
            "--session",
            session,
            "--file",
            str(SETUP_SCRIPT),
            "--timeout",
            str(args.setup_timeout),
        ],
        [
            *prefix,
            "upload",
            "--session",
            session,
            invocation_source,
            "/content/next-preflight-request.json",
        ],
        [
            *prefix,
            "exec",
            "--session",
            session,
            "--file",
            str(ENVIRONMENT_CHECK),
            "--timeout",
            "120",
        ],
        [
            *prefix,
            "upload",
            "--session",
            session,
            secret_source,
            "/content/.next-preflight-secrets.json",
        ],
        [
            *prefix,
            "exec",
            "--session",
            session,
            "--file",
            str(SECURE_EXEC),
            "--timeout",
            str(args.timeout),
        ],
        [*prefix, "stop", "--session", session],
    ]


def stop_session_verified(
    auth: str,
    session: str,
    log_handle: Any,
    *,
    attempts: int = 3,
    delay_seconds: float = 2.0,
) -> dict[str, Any]:
    if attempts <= 0:
        raise ValueError("cleanup attempts must be positive")
    stop_codes: list[int] = []
    sessions_codes: list[int] = []
    absent = False
    for attempt in range(1, attempts + 1):
        stop_command = ["colab", f"--auth={auth}", "stop", "--session", session]
        stopped = subprocess.run(
            stop_command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stop_codes.append(stopped.returncode)
        log_handle.write("\n[launcher cleanup] " + shlex.join(stop_command) + "\n")
        log_handle.write(stopped.stdout or "")

        sessions_command = ["colab", f"--auth={auth}", "sessions"]
        sessions = subprocess.run(
            sessions_command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        sessions_codes.append(sessions.returncode)
        log_handle.write("\n[launcher cleanup verify] " + shlex.join(sessions_command) + "\n")
        log_handle.write(sessions.stdout or "")
        log_handle.flush()
        absent = sessions.returncode == 0 and session not in (sessions.stdout or "")
        if absent:
            break
        if attempt < attempts:
            time.sleep(delay_seconds)
    return {
        "attempts": len(stop_codes),
        "stop_return_codes": stop_codes,
        "sessions_return_codes": sessions_codes,
        "session_absent_verified": absent,
    }


def run_unit(args: argparse.Namespace) -> dict[str, Any]:
    helpers = load_e1_helpers()
    output_dir = args.output_dir.expanduser().resolve()
    subprocess.run([sys.executable, str(DESIGN_VERIFIER)], cwd=REPO_ROOT, check=True)
    commit = source_commit(require_clean=not args.dry_run)
    cli_version = colab_cli_version()
    sources = [
        REMOTE_SCRIPT,
        SAMPLER,
        TRL_ADAPTER,
        SECURE_EXEC,
        ENVIRONMENT_CHECK,
        SETUP_SCRIPT,
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
            "provider": "colab",
            "colab_cli_version": cli_version,
            "trainer": "trl-1.8.0-custom-rollout-g8-v1",
            "sampler_sha256": sha256_file(SAMPLER),
            "adapter_sha256": sha256_file(TRL_ADAPTER),
            "decoder": DECODER_CONTRACT,
        }
    )
    tracking = {
        "hf_repo_prefix": args.hf_repo_prefix,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "wandb_group": args.wandb_group,
    }
    request = {
        "schema_version": "aiml-next-preflight-request-v1",
        "task": args.task,
        "arm": args.arm,
        "seed": args.seed,
        "gpu": args.gpu,
        "provider": "colab",
        "hardware_flavor": args.gpu,
        "auth_strategy": args.auth,
        "colab_cli_version": cli_version,
        "source_commit": commit,
        "protocol_sha256": protocol_sha256,
        "stack_fingerprint": stack_fingerprint,
        "runtime_packages": list(PACKAGE_PINS),
        "decoder": DECODER_CONTRACT,
        "tracking": tracking,
        "source_snapshots": snapshots,
    }
    request["fingerprint"] = fingerprint(request)
    unit = f"{args.task}__{args.arm}__s{args.seed}"
    session = f"next-pre-{args.task[:4]}-{args.arm[:8]}-{request['fingerprint'][:6]}"[:40]
    hf_repo = expected_hf_repo(request)
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
    archive_incompatible_result(
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
        "colab",
        "--hardware-flavor",
        args.gpu,
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

    execution_plan = build_execution_plan(
        args,
        session=session,
        secret_source="<ephemeral-secret-file>",
        invocation_source="<ephemeral-request-file>",
    )
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
    try:
        tracking_preflight = verify_tracking_credentials(
            credentials,
            hf_repo_prefix=args.hf_repo_prefix,
        )
    except Exception as exc:
        failed = {
            **request,
            "status": "failed",
            "failure_phase": "tracking-credential-preflight",
            "allocation_started": False,
            "error": str(exc),
            "request_path": str(request_path),
            "completed_at": utc_now(),
        }
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}
    lines: list[str] = []
    return_code = 0
    failed_step = None
    allocation_started = False
    cleanup = {
        "attempts": 0,
        "stop_return_codes": [],
        "sessions_return_codes": [],
        "session_absent_verified": False,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write("[launcher] credentials staged out of band; values are not logged\n")
        with tempfile.TemporaryDirectory(prefix="next-colab-preflight-") as staging:
            staging_path = Path(staging)
            secret_path = staging_path / "secrets.json"
            invocation_path = staging_path / "request.json"
            secret_path.write_text(json.dumps(credentials), encoding="utf-8")
            secret_path.chmod(0o600)
            invocation_path.write_text(
                json.dumps({"expected_gpu": args.gpu, "script_args": script_args}),
                encoding="utf-8",
            )
            commands = build_execution_plan(
                args,
                session=session,
                secret_source=str(secret_path),
                invocation_source=str(invocation_path),
            )[:-1]
            try:
                for index, command in enumerate(commands):
                    return_code = helpers.run_logged(command, log_handle, lines)
                    if index == 0 and return_code == 0:
                        allocation_started = True
                    if return_code:
                        failed_step = index
                        break
            finally:
                cleanup = stop_session_verified(args.auth, session, log_handle)

    if not cleanup["session_absent_verified"]:
        return_code = return_code or 1
        failed_step = "cleanup"
        lines.append("Colab session absence could not be verified after stop")

    base = {
        **request,
        "session": session,
        "hf_repo": hf_repo,
        "completed_at": utc_now(),
        "return_code": return_code,
        "failed_step": failed_step,
        "allocation_started": allocation_started,
        "cleanup": cleanup,
        "tracking_preflight": tracking_preflight,
        "log_path": str(log_path),
        "request_path": str(request_path),
    }
    if return_code:
        failed = {**base, "status": "failed", "error": failure_summary(lines)}
        helpers.atomic_json(result_path, failed)
        return {"status": "failed", "result_path": str(result_path)}
    try:
        result, recovery = result_from_log_or_remote(lines, credentials, request)
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
    if recovery is not None:
        complete["recovery"] = recovery
    helpers.atomic_json(result_path, complete)
    return {"status": "completed", "result_path": str(result_path)}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    status = run_unit(args)
    print("[next-preflight] " + json.dumps(status, sort_keys=True), flush=True)
    return 1 if status["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
