#!/usr/bin/env python3
"""Run the Pavlov E1-E14 evaluation gates on Modal.

The lane preflight functions are deliberately cheap and fail closed.  They
validate the exact suite contract, exercise the lane-specific adapter tests,
probe the authoritative source without downloading benchmark data, and retain
the existing lane receipt as provenance.  A preflight is never a model score.

Non-E11 readiness is derived from each receipt instead of being hard-coded to a
single lane. No related benchmark is substituted for a blocked E-suite.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "pavlov-e1-e14"
REMOTE_ROOT = Path("/root/project")
RESULTS_ROOT = Path("/results")
RUN_DATE = "2026-08-16"

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
FINAL_SAMPLER_PATH = (
    "tinker://cf0ad8c1-1f1b-5ff3-8bd7-2a0bf232657b:train:0/"
    "sampler_weights/seed809_final"
)
FINAL_HF_REPO = (
    "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-"
    "tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
)
FINAL_HF_REVISION = "checkpoint-seed809-stepfinal-9f777c4018b6"
FINAL_HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"

E11_DATASET_REVISION = "c498220d0a52248f8e3fdffe279075215bde2da6"
E11_IVERILOG_REVISION = "4fd5291632232fbe1ba49b2c26bb6b2bf1c6c9cf"
E11_DATASETS = ("code-complete-iccad2023", "spec-to-rtl")
E11_KNOWN_UNSCOREABLE = ("verilog_eval/spec-to-rtl/Prob099_m2014_q6c",)

USD_PER_M_PREFILL = 0.54
USD_PER_M_CACHED_PREFILL = 0.108
USD_PER_M_SAMPLE = 1.335
E11_MAX_PROJECTED_TINKER_USD = 0.60
TINKER_OPERATIONAL_CAP_USD = 16.50
TINKER_SAFETY_RESERVE_USD = 1.50

LANES: dict[str, dict[str, Any]] = {
    "E1": {
        "suite_id": "swe_bench_pro_eval",
        "receipt": "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro/seed1818/receipt.json",
        "tests": ("flagship.test_pavlov_swe_bench_pro_eval_adapter",),
    },
    "E2": {
        "suite_id": "frontier_swe_eval",
        "receipt": "outputs/e2_frontier_swe/e2_frontier_swe_lane_receipt_2026-08-09.json",
        "tests": ("flagship.test_frontier_swe_eval",),
    },
    "E3": {
        "suite_id": "sdab_eval",
        "receipt": "outputs/e3_sdab/receipt_2026-08-09.json",
        "tests": (
            "flagship.test_pavlov_sdab_eval_adapter",
            "flagship.test_eval_pavlov_sdab",
        ),
    },
    "E4": {
        "suite_id": "banker_toolbench_eval",
        "receipt": "outputs/modal_e1_e14/2026-08-16/e4_recovery_pass16_receipt.json",
        "tests": ("flagship.test_pavlov_banker_toolbench_harbor_rerun",),
    },
    "E5": {
        "suite_id": "apex_agents_eval",
        "receipt": "outputs/e5_apex_agents/lane_receipt_2026-08-09.json",
        "tests": ("flagship.test_eval_apex_agents_ingestion",),
    },
    "E6": {
        "suite_id": "webbench_eval",
        "receipt": "outputs/e6_webbench/e6_lane_receipt_2026-08-09.json",
        "tests": ("flagship.test_pavlov_webbench_eval_adapter",),
    },
    "E7": {
        "suite_id": "binaryaudit_eval",
        "receipt": "outputs/e7_binaryaudit/e7_binaryaudit_receipt_2026-08-09.json",
        "tests": ("flagship.test_pavlov_binaryaudit_eval_adapter",),
    },
    "E8": {
        "suite_id": "lifescibench_eval",
        "receipt": "outputs/e8_lifescibench/lane_receipt_2026-08-09.json",
        "tests": ("flagship.test_pavlov_lifescibench_eval_adapter",),
    },
    "E9": {
        "suite_id": "mle_bench_eval",
        "receipt": "outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json",
        "tests": ("flagship.test_mle_bench_eval",),
    },
    "E10": {
        "suite_id": "agentharm_eval",
        "receipt": "outputs/e10_agentharm/receipt_2026-08-09.json",
        "tests": ("flagship.test_pavlov_agentharm_frontiermath_adapter",),
    },
    "E11": {
        "suite_id": "verilog_eval",
        "receipt": "outputs/modal_e1_e14/2026-08-16/e11/launch_preflight_receipt.json",
        "tests": (
            "flagship.test_e11_verilog_eval_local_runner",
            "flagship.test_pavlov_verilog_eval_split_manifest",
        ),
    },
    "E12": {
        "suite_id": "appbench_eval",
        "receipt": "outputs/e12_appbench/local_receipt.json",
        "tests": ("flagship.test_pavlov_appbench_openreward_games_adapter",),
    },
    "E13": {
        "suite_id": "openreward_games_eval",
        "receipt": "outputs/e13_openreward_games/receipt_2026-08-09.json",
        "tests": (
            "flagship.test_e13_openreward_games_local_runner",
            "flagship.test_e13_openreward_games_tinker_train",
        ),
    },
    "E14": {
        "suite_id": "frontiermath_eval",
        "receipt": "outputs/e14_frontiermath/receipt_2026-08-09.json",
        "tests": (
            "flagship.test_pavlov_agentharm_frontiermath_adapter",
            "flagship.test_e14_frontiermath_public_samples",
        ),
    },
}

NON_E11_ACTIONS: dict[str, dict[str, Any]] = {
    "E1": {
        "owner": "local_then_campaign",
        "next_action": "Run only structurally valid one-task patches, then build the exact 731-task campaign runner.",
        "can_improve_without_external_input": True,
    },
    "E2": {
        "owner": "local_then_maintainer",
        "next_action": "Re-run the saved candidate with numeric-only Harbor rewards; obtain an explicit benchmark license receipt before publication.",
        "can_improve_without_external_input": True,
    },
    "E3": {
        "owner": "provider",
        "next_action": "Obtain the immutable private 80-task bundle, split, runtime, and native grader from the provider.",
        "can_improve_without_external_input": False,
    },
    "E4": {
        "owner": "local_then_provider_judge",
        "next_action": "Preserve the pass-16 recovery receipt, then require a clean one-pass scored task before expanding the campaign.",
        "can_improve_without_external_input": True,
    },
    "E5": {
        "owner": "local_then_budget",
        "next_action": "Build the Archipelago environment and pre-register a small stratified subset before any agent or judge calls.",
        "can_improve_without_external_input": True,
    },
    "E6": {
        "owner": "provider",
        "next_action": "Obtain Halluminate task authorization, deterministic live environment, ground truth, and native verifier.",
        "can_improve_without_external_input": False,
    },
    "E7": {
        "owner": "maintainer",
        "next_action": "Obtain a license artifact or written authorization bound to the pinned BinaryAudit revision.",
        "can_improve_without_external_input": False,
    },
    "E8": {
        "owner": "provider",
        "next_action": "Obtain the immutable LifeSciBench task package, license, manifest, environment, verifier, and disjointness receipt.",
        "can_improve_without_external_input": False,
    },
    "E9": {
        "owner": "account_holder_then_local",
        "next_action": "Accept the remaining Kaggle competition agreements, then pin the image, submission artifact, and disjointness receipt.",
        "can_improve_without_external_input": False,
    },
    "E10": {
        "owner": "provider",
        "next_action": "Obtain AISI private held-out files and the approved policy-grader specification.",
        "can_improve_without_external_input": False,
    },
    "E12": {
        "owner": "provider_then_humans",
        "next_action": "Obtain the licensed AppBench package and implement its native GUI/deployment verifier with two independent graders.",
        "can_improve_without_external_input": False,
    },
    "E13": {
        "owner": "provider",
        "next_action": "Obtain an official held-out suite, deployment binding, license grant, and OpenReward credential.",
        "can_improve_without_external_input": False,
    },
    "E14": {
        "owner": "epoch_ai",
        "next_action": "Arrange an Epoch-hosted evaluation against the immutable model endpoint; no local substitute is valid.",
        "can_improve_without_external_input": False,
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalise_blocker(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("detail", "reason", "message", "code", "name"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                code = value.get("code")
                if key != "code" and isinstance(code, str) and code.strip():
                    return f"{code}: {item.strip()}"
                return item.strip()
        return _stable_json(value)
    return None


def _collect_blockers(receipt: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for key in ("blockers", "outstanding_blockers", "missing", "errors"):
        value = receipt.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            blocker = _normalise_blocker(item)
            if blocker:
                blockers.append(blocker)
    return sorted(set(blockers))


def _receipt_launch_allowed(receipt: dict[str, Any]) -> bool:
    """Read the launch flag from the receipt schemas used across the lanes."""

    launch = receipt.get("launch")
    gates = receipt.get("gates")
    authorization = gates.get("authorization") if isinstance(gates, dict) else None
    candidates = (
        receipt.get("launch_allowed"),
        receipt.get("paid_launch_allowed"),
        launch.get("allowed") if isinstance(launch, dict) else None,
        authorization.get("launch_authorized")
        if isinstance(authorization, dict)
        else None,
    )
    return any(value is True for value in candidates)


def _receipt_ready(receipt: dict[str, Any]) -> bool:
    status = str(receipt.get("status") or receipt.get("final_status") or "").upper()
    return status in {"READY", "COMPLETE", "COMPLETED", "SCORED"}


def _readiness_class(
    *,
    adapter_passed: bool,
    source_has_model_score: bool,
    launch_ready: bool,
) -> str:
    if not adapter_passed:
        return "ADAPTER_BLOCKED"
    if source_has_model_score:
        return "RECORDED_MODEL_RESULT"
    if launch_ready:
        return "READY_FOR_FULL_MODAL_EVAL"
    return "EXTERNAL_OR_EXECUTION_INPUT_REQUIRED"


def _contract_primary_eval() -> list[tuple[str, dict[str, Any]]]:
    contract_path = REMOTE_ROOT / "zvf-program/flagship/pavlovs_domain_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return [
        (suite_id, suite)
        for suite_id, suite in contract["suite_registry"].items()
        if suite.get("role") == "primary_eval"
    ]


def _probe_url(url: str) -> dict[str, Any]:
    headers = {
        "User-Agent": "pavlov-e1-e14-modal-preflight/1.0",
        "Range": "bytes=0-0",
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.1",
    }
    started = time.monotonic()
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read(1)
            return {
                "reachable": True,
                "status": response.status,
                "final_url": response.geturl(),
                "content_type": response.headers.get("Content-Type"),
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            }
    except urllib.error.HTTPError as exc:
        return {
            "reachable": True,
            "status": exc.code,
            "final_url": exc.geturl(),
            "error": str(exc),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }
    except Exception as exc:  # remote DNS/TLS/provider failures are evidence
        return {
            "reachable": False,
            "status": None,
            "final_url": None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }


def _run_tests(modules: tuple[str, ...]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "/root/project/zvf-program:/root/project/zvf-program/flagship"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "-q", *modules],
        cwd=REMOTE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    summary = next(
        (line.strip() for line in reversed(combined.splitlines()) if line.strip() in {"OK"} or line.startswith("FAILED")),
        None,
    )
    ran = next(
        (line.strip() for line in reversed(combined.splitlines()) if line.startswith("Ran ")),
        None,
    )
    return {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "modules": list(modules),
        "ran": ran,
        "summary": summary,
        "output_tail": combined[-4000:],
    }


def _write_volume_json(relative: str, payload: dict[str, Any]) -> None:
    path = RESULTS_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results_volume.commit()


def _source_image() -> modal.Image:
    root = _repo_root()
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install(
            "huggingface-hub==1.11.0",
            "numpy==2.4.2",
            "pydantic==2.12.5",
            "PyYAML==6.0.3",
            "requests==2.32.5",
        )
        .add_local_dir(
            root / "zvf-program/flagship",
            "/root/project/zvf-program/flagship",
            copy=True,
            ignore=[
                "**/__pycache__/**",
                "**/*.pyc",
                "**/*.log",
                "**/*.stdout",
                "**/*.stderr",
                "paper/**",
            ],
        )
    )
    for relative in sorted({entry["receipt"] for entry in LANES.values()}):
        image = image.add_local_file(root / relative, f"/root/project/{relative}", copy=True)
    image = image.add_local_file(
        root / "outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
        "/root/project/outputs/_setup/LICENSE_RISK_ACCEPTANCE_2026-08-09.md",
        copy=True,
    )
    return image


if modal.is_local():
    preflight_image = _source_image()

    e11_image = (
        modal.Image.debian_slim(python_version="3.13")
        .apt_install(
            "autoconf",
            "bison",
            "build-essential",
            "ca-certificates",
            "flex",
            "git",
            "gperf",
            "libreadline-dev",
            "make",
            "zlib1g-dev",
        )
        .run_commands(
            "git clone https://github.com/steveicarus/iverilog.git /opt/iverilog-src",
            f"cd /opt/iverilog-src && git checkout {E11_IVERILOG_REVISION}",
            "cd /opt/iverilog-src && sh autoconf.sh && ./configure --prefix=/opt/iverilog",
            "cd /opt/iverilog-src && make -j4 && make install",
            "ln -s /usr/bin/make /usr/local/bin/gmake",
            "git clone https://github.com/NVlabs/verilog-eval.git /opt/verilog-eval",
            f"cd /opt/verilog-eval && git checkout {E11_DATASET_REVISION}",
            "/opt/iverilog/bin/iverilog -V | head -n 1",
        )
        .pip_install(
            "jinja2==3.1.6",
            "tinker==0.24.1",
            "transformers==5.5.4",
            "wandb==0.21.0",
        )
        .add_local_file(
            _repo_root() / "outputs/e11_verilog_eval/e11_model_run.py",
            "/root/project/outputs/e11_verilog_eval/e11_model_run.py",
            copy=True,
        )
        .add_local_file(
            _repo_root()
            / "outputs/modal_e1_e14/2026-08-16/e11/launch_preflight_receipt.json",
            "/root/project/outputs/modal_e1_e14/2026-08-16/e11/launch_preflight_receipt.json",
            copy=True,
        )
    )
else:
    # The deployed function retains its built image; remote imports only need stubs.
    preflight_image = modal.Image.debian_slim()
    e11_image = modal.Image.debian_slim()

results_volume = modal.Volume.from_name("pavlov-e1-e14-results", create_if_missing=True)
secret = modal.Secret.from_name("pavlov-e1-e14")
app = modal.App(APP_NAME, include_source=True)


@app.function(
    image=preflight_image,
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=1.0,
    memory=2048,
    timeout=300,
    retries=0,
)
def run_lane_preflight(lane: str) -> dict[str, Any]:
    """Execute one exact-suite gate without sampling or benchmark mutation."""

    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}")
    spec = LANES[lane]
    primary = _contract_primary_eval()
    index = int(lane[1:]) - 1
    suite_id, contract = primary[index]
    if suite_id != spec["suite_id"]:
        raise RuntimeError(
            f"contract label drift: {lane} expected {spec['suite_id']}, found {suite_id}"
        )

    receipt_path = REMOTE_ROOT / spec["receipt"]
    receipt_bytes = receipt_path.read_bytes()
    source = json.loads(receipt_bytes)
    tests = _run_tests(spec["tests"])
    source_score = source.get("score")
    source_is_model_score = source.get("is_model_score") is True and source_score is not None
    blockers = _collect_blockers(source)
    launch_ready = _receipt_ready(source) and _receipt_launch_allowed(source)
    if not launch_ready and not source_is_model_score and not blockers:
        blockers.append("exact suite has no launch authorization or completed model-score receipt")

    status = (
        "RECORDED_MODEL_RESULT"
        if source_is_model_score
        else "READY_FOR_FULL_MODAL_EVAL"
        if launch_ready and not blockers
        else "BLOCKED"
    )
    if not tests["passed"]:
        status = "BLOCKED"
        blockers.append("lane adapter tests failed in the Modal image")

    result: dict[str, Any] = {
        "schema_version": "pavlov-modal-e1-e14-preflight-v1",
        "recorded_at": _utc_now(),
        "lane": lane,
        "suite_id": suite_id,
        "suite_role": "primary_eval",
        "status": status,
        "score": None,
        "is_model_score": False,
        "claim_boundary": "Modal preflight is infrastructure evidence, never a benchmark score.",
        "contract": {
            "source_url": contract.get("url"),
            "split": contract.get("split"),
            "stateful": contract.get("stateful"),
            "artifact_or_side_effect": contract.get("artifact_or_side_effect"),
            "domains": contract.get("domains"),
        },
        "source_receipt": {
            "path": spec["receipt"],
            "sha256": _sha256_bytes(receipt_bytes),
            "status": source.get("status") or source.get("final_status"),
            "contains_model_score": source_is_model_score,
            "launch_allowed": source.get("launch_allowed") is True,
        },
        "modal": {
            "app": APP_NAME,
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "authoritative_source_probe": _probe_url(str(contract.get("url"))),
        "adapter_tests": tests,
        "readiness_class": _readiness_class(
            adapter_passed=tests["passed"],
            source_has_model_score=source_is_model_score,
            launch_ready=launch_ready and not blockers,
        ),
        "actionability": NON_E11_ACTIONS.get(lane),
        "blockers": sorted(set(blockers)),
    }
    result["receipt_sha256"] = _sha256_bytes(_stable_json(result).encode("utf-8"))
    _write_volume_json(f"{RUN_DATE}/preflight/{lane}.json", result)
    return result


def _e11_load_prompts(dataset: str) -> list[tuple[str, str]]:
    directory = Path("/opt/verilog-eval") / f"dataset_{dataset}"
    return [
        (path.name[: -len("_prompt.txt")], path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*_prompt.txt"))
    ]


def _e11_extract_module(response: str) -> str | None:
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    fence = re.compile(r"```(?:systemverilog|verilog|sv)?\s*\n(.*?)```", re.I | re.S)
    module = re.compile(r"\bmodule\s+TopModule\b.*?\bendmodule\b", re.S)
    for candidate in reversed(fence.findall(response)):
        match = module.search(candidate)
        if match:
            return match.group(0).strip()
    match = module.search(response)
    return match.group(0).strip() if match else None


def _e11_wandb_id_path(run_key: str) -> Path:
    return RESULTS_ROOT / RUN_DATE / "e11" / run_key / "wandb_run_id.txt"


def _e11_checkpoint_path(run_key: str, dataset: str, problem_id: str) -> Path:
    return RESULTS_ROOT / RUN_DATE / "e11" / run_key / "samples" / dataset / f"{problem_id}.json"


def _e11_project_cost(prompts: dict[str, list[tuple[str, str]]], max_tokens: int) -> dict[str, Any]:
    prompt_chars = sum(len(text) for rows in prompts.values() for _, text in rows)
    count = sum(len(rows) for rows in prompts.values())
    prefill_tokens = round(prompt_chars / 3.0)
    output_tokens = count * max_tokens
    usd = prefill_tokens / 1e6 * USD_PER_M_PREFILL + output_tokens / 1e6 * USD_PER_M_SAMPLE
    return {
        "problem_count": count,
        "prompt_chars": prompt_chars,
        "projected_prefill_tokens": prefill_tokens,
        "projected_output_tokens": output_tokens,
        "projected_usd": round(usd, 6),
        "function_cap_usd": E11_MAX_PROJECTED_TINKER_USD,
        "within_function_cap": usd <= E11_MAX_PROJECTED_TINKER_USD,
    }


def _e11_configure(dataset: str, build: Path) -> None:
    build.mkdir(parents=True, exist_ok=True)
    command = [
        "/opt/verilog-eval/configure",
        f"--with-task={dataset}",
        "--with-model=manual-rtl-coder",
        "--with-examples=0",
        "--with-samples=1",
        "--with-temperature=0",
        "--with-top-p=0.01",
    ]
    result = subprocess.run(command, cwd=build, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(f"E11 configure failed for {dataset}: {result.stderr[-2000:]}")


def _e11_write_sample(build: Path, problem_id: str, record: dict[str, Any]) -> None:
    problem_dir = build / problem_id
    problem_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{problem_id}_sample01"
    source = record.get("module_source") or ""
    (problem_dir / f"{stem}.sv").write_text(source + ("\n" if source else ""), encoding="utf-8")
    (problem_dir / f"{stem}-sv-generate.log").write_text(
        f"model = {MODEL_ID}\n"
        f"revision = {MODEL_REVISION}\n"
        f"sampler_path = {FINAL_SAMPLER_PATH}\n"
        f"prompt_tokens = {record['prompt_tokens']}\n"
        f"resp_tokens = {record['response_tokens']}\n"
        f"cost = {record['estimated_cost_usd']:.8f}\n",
        encoding="utf-8",
    )


def _e11_run_harness(build: Path, problem_ids: list[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PATH"] = "/opt/iverilog/bin:/usr/local/bin:/usr/bin:/bin"
    tested = subprocess.run(
        ["gmake", "-j", "8", "sv-iv-test", "VERBOSE=1"],
        cwd=build,
        env=env,
        capture_output=True,
        text=True,
        timeout=10800,
    )
    analyzed = subprocess.run(
        ["gmake", "sv-iv-analyze", "VERBOSE=1"],
        cwd=build,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    results: dict[str, bool] = {}
    log_receipts: dict[str, dict[str, Any]] = {}
    for problem_id in problem_ids:
        log = build / problem_id / f"{problem_id}_sample01-sv-iv-test.log"
        text = log.read_text(encoding="utf-8") if log.is_file() else ""
        passed = bool(re.search(r"(?m)^Mismatches:\s+0\s+in\s+\d+\s+samples\s*$", text))
        results[problem_id] = passed
        log_receipts[problem_id] = {
            "present": log.is_file(),
            "sha256": _sha256_bytes(log.read_bytes()) if log.is_file() else None,
            "verdict": "PASS" if passed else "FAIL",
        }
    summary = build / "summary.csv"
    return {
        "test_exit": tested.returncode,
        "analyze_exit": analyzed.returncode,
        "summary_csv_present": summary.is_file(),
        "summary_csv_sha256": _sha256_bytes(summary.read_bytes()) if summary.is_file() else None,
        "direct_results": results,
        "log_receipts": log_receipts,
        "test_output_tail": (tested.stdout + tested.stderr)[-3000:],
        "analyze_output_tail": (analyzed.stdout + analyzed.stderr)[-3000:],
    }


def _e11_score(results: dict[str, bool]) -> dict[str, Any]:
    total = len(results)
    passes = sum(results.values())
    excluded = [task for task in E11_KNOWN_UNSCOREABLE if task in results]
    corrected_total = total - len(excluded)
    corrected_passes = passes - sum(bool(results[task]) for task in excluded)
    return {
        "raw": {
            "denominator": total,
            "passes": passes,
            "pass_at_1": passes / total if total else None,
        },
        "corrected": {
            "denominator": corrected_total,
            "passes": corrected_passes,
            "pass_at_1": corrected_passes / corrected_total if corrected_total else None,
            "excluded": excluded,
        },
        "reporting_rule": "Raw and corrected denominators are always reported together.",
    }


@app.function(
    image=e11_image,
    secrets=[secret],
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=8.0,
    memory=32768,
    timeout=86400,
    retries=0,
)
def run_e11_full(
    *,
    seed: int = 1816,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    billed_tinker_usd_before: float,
) -> dict[str, Any]:
    """Run one resumable, full, final-checkpoint E11 pass@1 evaluation."""

    launch_gate_path = (
        REMOTE_ROOT
        / "outputs/modal_e1_e14/2026-08-16/e11/launch_preflight_receipt.json"
    )
    launch_gate_bytes = launch_gate_path.read_bytes()
    launch_gate = json.loads(launch_gate_bytes)
    if launch_gate.get("status") != "READY" or launch_gate.get("launch_allowed") is not True:
        raise RuntimeError("E11 launch receipt is not READY")
    if launch_gate.get("model", {}).get("sampler_path") != FINAL_SAMPLER_PATH:
        raise RuntimeError("E11 launch receipt sampler identity drifted")
    if launch_gate.get("decontamination", {}).get("receipt_id") != (
        "ddc32a42a42ec6ecebfbbf6335ba16ecddbe3c89"
    ):
        raise RuntimeError("E11 decontamination receipt identity drifted")

    if billed_tinker_usd_before < 0:
        raise ValueError("billed_tinker_usd_before must be non-negative")
    spendable = TINKER_OPERATIONAL_CAP_USD - TINKER_SAFETY_RESERVE_USD
    if billed_tinker_usd_before >= spendable:
        raise RuntimeError("Tinker spendable cap is already exhausted")

    prompts = {dataset: _e11_load_prompts(dataset) for dataset in E11_DATASETS}
    projection = _e11_project_cost(prompts, max_tokens)
    if not projection["within_function_cap"]:
        raise RuntimeError(f"E11 projection exceeds function cap: {projection}")
    if billed_tinker_usd_before + projection["projected_usd"] > spendable:
        raise RuntimeError("E11 projection would consume the protected Tinker reserve")

    run_key = f"final_full_seed{seed}_mt{max_tokens}_t{str(temperature).replace('.', 'p')}"
    wandb_id_path = _e11_wandb_id_path(run_key)
    wandb_id_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_MODE"] = "online"
    import wandb

    if wandb_id_path.is_file():
        wandb_id = wandb_id_path.read_text(encoding="utf-8").strip()
    else:
        wandb_id = wandb.util.generate_id()
        wandb_id_path.write_text(wandb_id + "\n", encoding="utf-8")
        results_volume.commit()
    wandb_run = wandb.init(
        id=wandb_id,
        resume="allow",
        entity="arvindcr4-pes-university",
        project="tinker-rl-lab-pavlov",
        group="pavlov-e1-e14-modal-20260816",
        job_type="primary-evaluation",
        name=f"e11_verilog_eval_final_full_modal_seed{seed}",
        tags=["e11", "verilog_eval", "pass@1", "trained-final", "modal"],
        mode="online",
        config={
            "suite_id": "verilog_eval",
            "suite_role": "primary_eval",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampler_path": FINAL_SAMPLER_PATH,
            "hf_repo": FINAL_HF_REPO,
            "hf_revision": FINAL_HF_REVISION,
            "hf_commit": FINAL_HF_COMMIT,
            "dataset_revision": E11_DATASET_REVISION,
            "iverilog_revision": E11_IVERILOG_REVISION,
            "samples_per_problem": 1,
            "retries_per_problem": 0,
            "max_response_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "problems": projection["problem_count"],
            "verifier_type": "native upstream sv-iv-test with Icarus Verilog 12.0",
            "is_model_score": True,
            "projected_usd": projection["projected_usd"],
            "billed_tinker_usd_before": billed_tinker_usd_before,
        },
        reinit=True,
    )
    if wandb_run is None or not getattr(wandb_run, "id", None):
        raise RuntimeError("W&B online initialization failed before Tinker")

    import tinker
    import tinker.types as T
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    service = tinker.ServiceClient(
        user_metadata={
            "campaign": "pavlov-e1-e14-modal",
            "stage": "primary-evaluation",
            "suite_id": "verilog_eval",
            "wandb_run_id": wandb_run.id,
        }
    )
    sampler = service.create_sampling_client(model_path=FINAL_SAMPLER_PATH)

    sampled_records: dict[str, dict[str, dict[str, Any]]] = {}
    new_samples = 0
    resumed_samples = 0
    total_prompt_tokens = 0
    total_response_tokens = 0
    extraction_failures = 0

    for dataset, rows in prompts.items():
        sampled_records[dataset] = {}
        for problem_index, (problem_id, prompt_text) in enumerate(rows):
            checkpoint = _e11_checkpoint_path(run_key, dataset, problem_id)
            if checkpoint.is_file():
                record = json.loads(checkpoint.read_text(encoding="utf-8"))
                resumed_samples += 1
            else:
                chat = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_ids = tokenizer.encode(chat, add_special_tokens=False)
                response = sampler.sample(
                    T.ModelInput.from_ints(prompt_ids),
                    num_samples=1,
                    sampling_params=T.SamplingParams(
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.95,
                        seed=seed,
                    ),
                ).result()
                sequence = response.sequences[0]
                response_tokens = list(sequence.tokens)
                response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                module_source = _e11_extract_module(response_text)
                estimated_cost = (
                    len(prompt_ids) / 1e6 * USD_PER_M_PREFILL
                    + len(response_tokens) / 1e6 * USD_PER_M_SAMPLE
                )
                record = {
                    "dataset": dataset,
                    "problem_id": problem_id,
                    "prompt_sha256": _sha256_bytes(prompt_text.encode("utf-8")),
                    "response_sha256": _sha256_bytes(response_text.encode("utf-8")),
                    "module_source": module_source,
                    "module_sha256": (
                        _sha256_bytes(module_source.encode("utf-8")) if module_source else None
                    ),
                    "extracted_module": module_source is not None,
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(response_tokens),
                    "estimated_cost_usd": estimated_cost,
                    "sample_index": 1,
                    "sampling_retry_count": 0,
                    "sampled_at": _utc_now(),
                }
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                new_samples += 1
                if new_samples % 8 == 0:
                    results_volume.commit()
                wandb_run.log(
                    {
                        "sampling/completed": new_samples + resumed_samples,
                        "sampling/new": new_samples,
                        "sampling/resumed": resumed_samples,
                        "sampling/extraction_failure": int(module_source is None),
                        "cost/estimated_sample_usd": estimated_cost,
                    },
                    step=new_samples + resumed_samples,
                )
            sampled_records[dataset][problem_id] = record
            total_prompt_tokens += int(record["prompt_tokens"])
            total_response_tokens += int(record["response_tokens"])
            extraction_failures += int(not record["extracted_module"])
        results_volume.commit()

    merged_results: dict[str, bool] = {}
    verifier_receipts: dict[str, Any] = {}
    for dataset, rows in prompts.items():
        build = Path(tempfile.mkdtemp(prefix=f"e11-modal-{dataset}-"))
        _e11_configure(dataset, build)
        for problem_id, _ in rows:
            _e11_write_sample(build, problem_id, sampled_records[dataset][problem_id])
        harness = _e11_run_harness(build, [problem_id for problem_id, _ in rows])
        for problem_id, passed in harness["direct_results"].items():
            merged_results[f"verilog_eval/{dataset}/{problem_id}"] = passed
        verifier_receipts[dataset] = {
            key: value for key, value in harness.items() if key != "direct_results"
        }

    score = _e11_score(merged_results)
    actual_estimated_usd = (
        total_prompt_tokens / 1e6 * USD_PER_M_PREFILL
        + total_response_tokens / 1e6 * USD_PER_M_SAMPLE
    )
    receipt: dict[str, Any] = {
        "schema_version": "e11-modal-full-pass-at-1-receipt-v1",
        "recorded_at": _utc_now(),
        "lane": "E11",
        "suite_id": "verilog_eval",
        "suite_role": "primary_eval",
        "status": "SCORED",
        "is_model_score": True,
        "evidence_class": "primary_evaluation/pass@1",
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "sampler_path": FINAL_SAMPLER_PATH,
            "hf_repo": FINAL_HF_REPO,
            "hf_revision": FINAL_HF_REVISION,
            "hf_commit": FINAL_HF_COMMIT,
        },
        "dataset": {
            "repo": "NVlabs/verilog-eval",
            "revision": E11_DATASET_REVISION,
            "license": "apache-2.0",
            "problem_count": len(merged_results),
            "task_framings": list(E11_DATASETS),
        },
        "decontamination": {
            "status": "verified",
            "receipt_id": launch_gate["decontamination"]["receipt_id"],
            "receipt_sha256": launch_gate["decontamination"]["receipt_sha256"],
            "launch_gate_sha256": _sha256_bytes(launch_gate_bytes),
            "boundary": (
                "All task-bearing fields from every row in both pinned training-source "
                "snapshots were compared against every E11 prompt and reference."
            ),
        },
        "sampling": {
            "samples_per_problem": 1,
            "retries_per_problem": 0,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "seed": seed,
            "new_samples": new_samples,
            "resumed_samples": resumed_samples,
            "extraction_failures": extraction_failures,
        },
        "score": score["corrected"]["pass_at_1"],
        "pass_at_1": score,
        "verifier": {
            "identity": "NVlabs/verilog-eval sv-iv-test",
            "iverilog_version": "12.0",
            "iverilog_revision": E11_IVERILOG_REVISION,
            "receipts": verifier_receipts,
        },
        "cost": {
            "projection": projection,
            "estimated_actual_usd": round(actual_estimated_usd, 6),
            "prompt_tokens": total_prompt_tokens,
            "response_tokens": total_response_tokens,
            "billed_tinker_usd_before": billed_tinker_usd_before,
            "operational_cap_usd": TINKER_OPERATIONAL_CAP_USD,
            "protected_reserve_usd": TINKER_SAFETY_RESERVE_USD,
        },
        "wandb": {
            "run_id": wandb_run.id,
            "url": wandb_run.url,
            "project": "tinker-rl-lab-pavlov",
        },
        "modal": {
            "app": APP_NAME,
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "function_call_id": os.environ.get("MODAL_FUNCTION_CALL_ID"),
            "cpu_count": os.cpu_count(),
            "memory_request_mib": 32768,
        },
        "outstanding_blockers": [],
    }
    receipt["receipt_sha256"] = _sha256_bytes(_stable_json(receipt).encode("utf-8"))
    _write_volume_json(f"{RUN_DATE}/e11/{run_key}/final_receipt.json", receipt)
    wandb_run.log(
        {
            "test/pass_at_1_raw": score["raw"]["pass_at_1"],
            "test/pass_at_1_corrected": score["corrected"]["pass_at_1"],
            "test/passes_raw": score["raw"]["passes"],
            "test/denominator_raw": score["raw"]["denominator"],
            "test/extraction_failures": extraction_failures,
            "cost/estimated_actual_usd": round(actual_estimated_usd, 6),
        },
        step=projection["problem_count"] + 1,
    )
    wandb_run.summary.update(
        {
            "status": "SCORED",
            "score": receipt["score"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
    )
    wandb_run.finish(exit_code=0)
    return receipt


def _write_local_json(relative: str, payload: Any) -> Path:
    path = _repo_root() / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


@app.local_entrypoint()
def main(
    mode: str = "non_e11",
    seed: int = 1816,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    billed_tinker_usd_before: float = 1.638457,
) -> None:
    """Run preflights or E11, with non-E11 preflight as the safe default."""

    if mode not in {"non_e11", "preflight", "e11", "all"}:
        raise ValueError("mode must be non_e11, preflight, e11, or all")
    out_root = f"outputs/modal_e1_e14/{RUN_DATE}"
    preflights: list[dict[str, Any]] = []
    if mode in {"non_e11", "preflight", "all"}:
        selected_lanes = (
            [lane for lane in LANES if lane != "E11"]
            if mode == "non_e11"
            else list(LANES)
        )
        preflights = list(run_lane_preflight.map(selected_lanes, order_outputs=True))
        for receipt in preflights:
            _write_local_json(f"{out_root}/preflight/{receipt['lane']}.json", receipt)
        ready = [
            item["lane"] for item in preflights if item["status"] == "READY_FOR_FULL_MODAL_EVAL"
        ]
        recorded = [
            item["lane"] for item in preflights if item["status"] == "RECORDED_MODEL_RESULT"
        ]
        adapter_ready = [
            item["lane"]
            for item in preflights
            if item["adapter_tests"]["passed"]
            and item["authoritative_source_probe"]["reachable"]
        ]
        summary = {
            "schema_version": "pavlov-modal-e1-e14-summary-v1",
            "recorded_at": _utc_now(),
            "status": "PARTIAL" if ready or recorded else "BLOCKED",
            "adapter_status": (
                "READY"
                if len(adapter_ready) == len(preflights)
                else "PARTIAL"
            ),
            "score": None,
            "is_model_score": False,
            "lane_count": len(preflights),
            "adapter_ready": adapter_ready,
            "ready_for_full_modal_eval": ready,
            "recorded_model_results": recorded,
            "readiness_classes": {
                item["lane"]: item["readiness_class"] for item in preflights
            },
            "locally_improvable": [
                item["lane"]
                for item in preflights
                if (item.get("actionability") or {}).get(
                    "can_improve_without_external_input"
                )
            ],
            "blocked": [item["lane"] for item in preflights if item["status"] == "BLOCKED"],
            "receipts": {item["lane"]: item["receipt_sha256"] for item in preflights},
        }
        path = _write_local_json(f"{out_root}/preflight_summary.json", summary)
        print(json.dumps({**summary, "local_path": str(path)}, indent=2))

    if mode in {"e11", "all"}:
        if not preflights:
            e11_preflight = run_lane_preflight.remote("E11")
            _write_local_json(f"{out_root}/preflight/E11.json", e11_preflight)
            preflights = [e11_preflight]
        if preflights:
            e11 = next(item for item in preflights if item["lane"] == "E11")
            if e11["status"] != "READY_FOR_FULL_MODAL_EVAL":
                raise RuntimeError("E11 Modal preflight did not authorize the full evaluation")
        receipt = run_e11_full.remote(
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            billed_tinker_usd_before=billed_tinker_usd_before,
        )
        path = _write_local_json(f"{out_root}/e11_full_receipt.json", receipt)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "lane": receipt["lane"],
                    "score": receipt["score"],
                    "raw": receipt["pass_at_1"]["raw"],
                    "corrected": receipt["pass_at_1"]["corrected"],
                    "cost": receipt["cost"],
                    "wandb": receipt["wandb"],
                    "local_path": str(path),
                },
                indent=2,
            )
        )
