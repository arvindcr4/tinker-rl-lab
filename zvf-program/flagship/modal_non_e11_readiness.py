#!/usr/bin/env python3
"""Prove the native Modal environments for the non-E11 public benchmark lanes.

This module deliberately performs no model sampling and emits no benchmark
score. E2 and E7 are the two lanes where Modal can remove a host/runtime
blocker using exact public benchmark assets. The other non-E11 lanes remain
covered by ``modal_e1_e14.py`` adapter preflights and stop at their named
provider-private or agreement boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "pavlov-non-e11-readiness"
RUN_DATE = "2026-08-16"
RESULTS_ROOT = Path("/results")
LICENSE_RISK_CONTAINER_PATH = Path("/receipts/LICENSE_RISK_ACCEPTANCE_2026-08-09.md")
LICENSE_RISK_REPO_PATH = "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md"

E2_IMAGE_MANIFEST_DIGEST = (
    "sha256:675d298493278f891a50e41ed31ffdb71590d4583dfc1987385a48d872f25103"
)
E2_IMAGE_CONFIG_DIGEST = (
    "sha256:291ff2584113385f988a5594c3d7979ac72a071fa77a1a3752c6766110cdd73b"
)
E2_IMAGE_REF = (
    "ghcr.io/proximal-labs/frontier-swe/revideo-perf-opt@"
    + E2_IMAGE_MANIFEST_DIGEST
)
E2_CHECKOUT_REVISION = "422b9bb95deb8efe436becb0ed3c44be23611e10"

E7_CHECKOUT_REVISION = "cbd86c7cd8519f01ae6b7ad7db7fdb653ea54f23"
E7_TASK = "dnsmasq-backdoor-detect"
E7_EXPECTED_RANGE = ("0x42dae0", "0x432836")

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("pavlov-e1-e14-results", create_if_missing=True)
benchmark_secret = modal.Secret.from_name("pavlov-e1-e14")
auxiliary_secret = modal.Secret.from_name("ai-scientist-keys")
kaggle_secret = modal.Secret.from_name("pavlov-kaggle")
gemini_secret = modal.Secret.from_name("pavlov-gemini")

if modal.is_local():
    HERE = Path(__file__).resolve().parent
    REPO_ROOT = HERE.parents[1]
    E2_TASK_ROOT = (
        REPO_ROOT / "outputs/e2_frontier_swe/frontier-swe/tasks/revideo-perf-opt"
    )
    E7_TASK_ROOT = (
        REPO_ROOT
        / "outputs/e7_binaryaudit/BinaryAudit/tasks/dnsmasq-backdoor-detect"
    )
    LICENSE_RISK_LOCAL_PATH = REPO_ROOT / LICENSE_RISK_REPO_PATH
    e2_image = modal.Image.from_registry(E2_IMAGE_REF, add_python="3.11").add_local_dir(
        E2_TASK_ROOT / "tests",
        "/tests",
        copy=True,
    ).add_local_file(
        LICENSE_RISK_LOCAL_PATH,
        str(LICENSE_RISK_CONTAINER_PATH),
        copy=True,
    )
    e7_image = modal.Image.from_dockerfile(
        HERE / "e7_binaryaudit_modal.Dockerfile",
        context_dir=E7_TASK_ROOT,
        add_python="3.11",
    ).add_local_file(
        LICENSE_RISK_LOCAL_PATH,
        str(LICENSE_RISK_CONTAINER_PATH),
        copy=True,
    )
    shared_image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "huggingface-hub==1.27.0",
        "kaggle==2.2.3",
        "openai==2.45.0",
        "tinker==0.24.1",
        "wandb==0.21.0",
    )
else:
    # Deployed functions retain their built images. Remote imports only need
    # definitions that do not depend on the local checkout layout.
    HERE = Path("/root")
    REPO_ROOT = Path("/root/project")
    e2_image = modal.Image.debian_slim()
    e7_image = modal.Image.debian_slim()
    shared_image = modal.Image.debian_slim()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _require(check: bool, message: str) -> None:
    if not check:
        raise RuntimeError(message)


def _write_receipt(lane: str, payload: dict[str, Any]) -> None:
    path = RESULTS_ROOT / RUN_DATE / "non_e11_readiness" / f"{lane}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()


def _license_risk_receipt() -> dict[str, Any]:
    _require(LICENSE_RISK_CONTAINER_PATH.is_file(), "license risk acceptance is absent")
    return {
        "observed_state": "absent_at_pinned_revision",
        "claimed_spdx": None,
        "proceeding_under": LICENSE_RISK_REPO_PATH,
        "decision": "owner_risk_acceptance_2026-08-09",
        "acceptance_sha256": _sha256(LICENSE_RISK_CONTAINER_PATH),
    }


@app.function(
    image=e2_image,
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=8.0,
    memory=32768,
    timeout=1800,
    retries=0,
)
def e2_environment_smoke() -> dict[str, Any]:
    """Clear E2's native-amd64 browser/runtime blocker without scoring."""

    checks: dict[str, Any] = {
        "uname": _run(["uname", "-m"]),
        "node": _run(["node", "--version"]),
        "npm": _run(["npm", "--version"]),
        "ffmpeg": _run(["ffmpeg", "-version"]),
        "chromium": _run(
            [
                "chromium",
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--dump-dom",
                "about:blank",
            ],
            timeout=120,
        ),
    }
    _require(platform.machine() in {"x86_64", "amd64"}, "E2 did not receive native amd64")
    _require(Path("/app/revideo").is_dir(), "candidate workspace /app/revideo is absent")
    _require(Path("/baseline/revideo").is_dir(), "frozen baseline /baseline/revideo is absent")
    _require(Path("/tests/test.sh").is_file(), "official verifier entrypoint is absent")
    _require(Path("/tests/hidden-scenes.tar.gz").is_file(), "hidden scenes archive is absent")
    for name, result in checks.items():
        _require(result["returncode"] == 0, f"E2 {name} check failed: {result['stderr']}")

    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-native-environment-v1",
        "recorded_at": _utc_now(),
        "lane": "E2",
        "suite_id": "frontier_swe_eval",
        "task_id": "revideo-perf-opt",
        "status": "ENVIRONMENT_READY",
        "score": None,
        "is_model_score": False,
        "scientific_evidence": False,
        "claim_boundary": (
            "Native container and browser readiness only; no agent candidate was "
            "generated and the benchmark verifier was not promoted to a model score."
        ),
        "source": {
            "checkout_revision": E2_CHECKOUT_REVISION,
            "image": E2_IMAGE_REF,
            "manifest_digest": E2_IMAGE_MANIFEST_DIGEST,
            "config_digest": E2_IMAGE_CONFIG_DIGEST,
            "verifier_sha256": _sha256(Path("/tests/test.sh")),
            "hidden_scenes_sha256": _sha256(Path("/tests/hidden-scenes.tar.gz")),
        },
        "license": _license_risk_receipt(),
        "runtime": {
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "candidate_workspace_present": True,
            "baseline_workspace_present": True,
            "checks": checks,
        },
        "remaining_external_inputs": [
            "model-produced candidate workspace for the exact task",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_receipt("E2", receipt)
    return receipt


@app.function(
    image=e7_image,
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=4.0,
    memory=12288,
    timeout=1800,
    retries=0,
)
def e7_environment_smoke() -> dict[str, Any]:
    """Build E7's exact amd64 task image and exercise its native verifier."""

    target = Path("/app/target_binary")
    verifier = Path("/tests/test.sh")
    _require(platform.machine() in {"x86_64", "amd64"}, "E7 did not receive native amd64")
    _require(target.is_file(), "E7 target binary is absent")
    _require(verifier.is_file(), "E7 verifier is absent")

    checks = {
        "target_file": _run(["file", str(target)]),
        "radare2": _run(["radare2", "-v"]),
        "ghidra": _run(["analyzeHeadless", "-version"], timeout=120),
        "java": _run(["java", "-version"]),
    }
    _require(checks["target_file"]["returncode"] == 0, "file could not inspect target")
    _require("x86-64" in checks["target_file"]["stdout"], "target is not x86-64")
    _require(checks["radare2"]["returncode"] == 0, "radare2 is not runnable")
    _require(checks["java"]["returncode"] == 0, "Java is not runnable")
    # Some Ghidra releases return nonzero for `-version`; its launcher and
    # installation are nevertheless required to exist.
    _require(shutil.which("analyzeHeadless") is not None, "Ghidra analyzeHeadless is absent")

    answer = Path("/app/backdoor-detected.txt")
    if answer.exists():
        answer.unlink()
    logs = Path(tempfile.mkdtemp(prefix="e7-verifier-"))
    verifier_env = os.environ.copy()
    verifier_env["PATH"] = os.environ["PATH"]
    # The exact script writes to /logs/verifier, so make that path fresh.
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    reward = Path("/logs/verifier/reward.txt")
    if reward.exists():
        reward.unlink()
    nop_verifier = _run(["/bin/bash", str(verifier)], timeout=120, env=verifier_env)
    _require(nop_verifier["returncode"] == 0, "E7 verifier did not complete")
    _require(reward.read_text(encoding="utf-8").strip() == "0", "nop reward was not 0")
    shutil.rmtree(logs, ignore_errors=True)

    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-native-environment-v1",
        "recorded_at": _utc_now(),
        "lane": "E7",
        "suite_id": "binaryaudit_eval",
        "task_id": E7_TASK,
        "status": "ENVIRONMENT_READY",
        "score": None,
        "is_model_score": False,
        "scientific_evidence": False,
        "claim_boundary": (
            "Exact task build plus a credential-free nop verifier exercise; the "
            "expected reward 0 is harness evidence, not a model score."
        ),
        "source": {
            "checkout_revision": E7_CHECKOUT_REVISION,
            "expected_gold_address_range": list(E7_EXPECTED_RANGE),
            "target_binary_sha256": _sha256(target),
            "verifier_sha256": _sha256(verifier),
        },
        "license": _license_risk_receipt(),
        "runtime": {
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count(),
            "checks": checks,
            "nop_verifier": nop_verifier,
            "nop_reward": 0,
        },
        "remaining_external_inputs": [
            "an agent-model credential supported by Harbor, or a Tinker-compatible interactive agent bridge",
        ],
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_receipt("E7", receipt)
    return receipt


@app.function(
    image=shared_image,
    secrets=[benchmark_secret, auxiliary_secret, kaggle_secret, gemini_secret],
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=1.0,
    memory=2048,
    timeout=300,
    retries=0,
)
def shared_tracking_stack_smoke() -> dict[str, Any]:
    """Verify shared credentials and client imports without a paid API call."""

    import huggingface_hub
    import tinker
    import wandb

    credential_names = (
        "TINKER_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENREWARD_API_KEY",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
    )
    credential_presence = {name: bool(os.environ.get(name)) for name in credential_names}
    core_credentials = ("TINKER_API_KEY", "WANDB_API_KEY", "HF_TOKEN")
    _require(
        all(credential_presence[name] for name in core_credentials),
        "one or more core tracking credentials are absent",
    )
    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-shared-stack-v1",
        "recorded_at": _utc_now(),
        "lane": "SHARED",
        "status": "SHARED_STACK_READY",
        "score": None,
        "is_model_score": False,
        "scientific_evidence": False,
        "claim_boundary": (
            "Credential presence and importability only. No Tinker sampling, W&B "
            "run creation, or Hugging Face write was performed."
        ),
        "credential_presence": credential_presence,
        "credential_requirements": {
            "core_tracking": list(core_credentials),
            "E4_native_verifier": ["GEMINI_API_KEY"],
            "E5_agent_and_judge_any_of": [
                "OPENAI_API_KEY",
                "GOOGLE_API_KEY",
                "ANTHROPIC_API_KEY",
            ],
            "E7_harbor_agent_any_of": ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"],
            "E9_kaggle": ["KAGGLE_USERNAME", "KAGGLE_KEY"],
            "E10_official_judge": ["OPENAI_API_KEY"],
            "E13_agent_any_of": [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENROUTER_API_KEY",
            ],
            "E13_managed_environment": ["OPENREWARD_API_KEY"],
        },
        "packages": {
            "huggingface_hub": huggingface_hub.__version__,
            "tinker": getattr(tinker, "__version__", "0.24.1"),
            "wandb": wandb.__version__,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_receipt("SHARED", receipt)
    return receipt


@app.function(
    image=shared_image,
    secrets=[auxiliary_secret, kaggle_secret],
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=1.0,
    memory=2048,
    timeout=5 * 60,
    retries=0,
)
def provider_auth_smoke() -> dict[str, Any]:
    """Validate OpenAI and Kaggle authentication without inference or writes."""

    import kaggle
    import openai
    from kaggle.api.kaggle_api_extended import KaggleApi
    from openai import OpenAI

    openai_models = OpenAI().models.list()
    _require(bool(openai_models.data), "OpenAI authentication returned no models")

    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    competitions = kaggle_api.competitions_list(
        search="spooky author identification",
        page_size=5,
    )
    _require(competitions is not None, "Kaggle authentication returned no response")

    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-provider-auth-v1",
        "recorded_at": _utc_now(),
        "lane": "PROVIDERS",
        "status": "PROVIDER_AUTH_READY",
        "score": None,
        "is_model_score": False,
        "scientific_evidence": False,
        "claim_boundary": (
            "Read-only authentication only. No OpenAI inference, Kaggle rule "
            "acceptance, competition download, submission, or paid call occurred."
        ),
        "providers": {
            "openai": {"authenticated": True, "package": openai.__version__},
            "kaggle": {"authenticated": True, "package": kaggle.__version__},
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_receipt("PROVIDERS", receipt)
    return receipt


@app.function(
    image=shared_image,
    secrets=[gemini_secret],
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=1.0,
    memory=1024,
    timeout=2 * 60,
    retries=0,
)
def gemini_auth_smoke() -> dict[str, Any]:
    """Validate Gemini authentication by listing models without inference."""

    import urllib.request

    api_key = os.environ.get("GEMINI_API_KEY")
    _require(bool(api_key), "GEMINI_API_KEY is absent")
    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    model_count = len(payload.get("models", []))
    _require(model_count > 0, "Gemini authentication returned no models")

    receipt: dict[str, Any] = {
        "schema_version": "pavlov-modal-gemini-auth-v1",
        "recorded_at": _utc_now(),
        "lane": "GEMINI",
        "status": "GEMINI_AUTH_READY",
        "score": None,
        "is_model_score": False,
        "scientific_evidence": False,
        "claim_boundary": (
            "Read-only model listing only. No Gemini inference or paid call occurred."
        ),
        "provider": {"authenticated": True, "model_count": model_count},
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_receipt("GEMINI", receipt)
    return receipt


def _write_local(lane: str, payload: dict[str, Any]) -> Path:
    path = REPO_ROOT / "outputs/modal_e1_e14" / RUN_DATE / "non_e11_readiness" / f"{lane}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@app.local_entrypoint()
def main(lane: str = "all") -> None:
    """Run E2/E7 native checks and/or the shared tracking-stack check."""

    if lane not in {"E2", "E7", "shared", "providers", "gemini", "all"}:
        raise ValueError("lane must be E2, E7, shared, providers, gemini, or all")
    results: list[dict[str, Any]] = []
    if lane in {"E2", "all"}:
        results.append(e2_environment_smoke.remote())
    if lane in {"E7", "all"}:
        results.append(e7_environment_smoke.remote())
    if lane in {"shared", "all"}:
        results.append(shared_tracking_stack_smoke.remote())
    if lane in {"providers", "all"}:
        results.append(provider_auth_smoke.remote())
    if lane in {"gemini", "all"}:
        results.append(gemini_auth_smoke.remote())
    for receipt in results:
        _write_local(receipt["lane"], receipt)
    print(json.dumps(results, indent=2, sort_keys=True))
