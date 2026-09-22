"""Separate 75-minute Omni continuation. No allocation on import.

Same immutable native actor, payloads, BF16 merge, CUDA graph and journal helpers.
Only this launch profile and its evidence identity differ. Root owns reservations.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

FROZEN_MODAL_SHA256 = "1f1c1b6b78c74a25a6e3927094540767ec5511f3a627e9456fc6e58a073a3232"
FROZEN_ACTOR_SHA256 = "968462cee08c2089b7aec016ba52836505ad898dcf6b7a553b5e0cb1970818dd"
if os.environ.get("PUBLIC_RUNTIME_INCLUDE_AGENTDOJO") == "1":
    raise ValueError("Group runtime is static-batch only; unset PUBLIC_RUNTIME_INCLUDE_AGENTDOJO")


def frozen_modal_source_path(source_file, is_local):
    if not is_local:
        return Path("/root/modal_public_portfolio_fast_runtime.py")
    return Path(source_file).resolve().with_name("modal_public_portfolio_fast_runtime.py")


FROZEN_MODAL_PATH = frozen_modal_source_path(__file__, modal.is_local())
if hashlib.sha256(FROZEN_MODAL_PATH.read_bytes()).hexdigest() != FROZEN_MODAL_SHA256:
    raise ValueError("Frozen fast Modal helper/image source hash mismatch")
_spec = importlib.util.spec_from_file_location("public_group_frozen_modal", FROZEN_MODAL_PATH)
frozen = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = frozen
_spec.loader.exec_module(frozen)


def runtime_profile(name="fullbatch75"):
    if name != "fullbatch75":
        raise ValueError("This separate launcher accepts only fullbatch75")
    return {"name": "group-fullbatch75", "purpose": "fullbatch75", "reserved_usd": 9.0,
            "total_seconds": 4500, "startup_seconds": 60, "function_seconds": 4440, "max_requests": 4430}


PROFILE = runtime_profile()
APP_NAME = "pavlov-public-portfolio-group-runtime-" + PROFILE["purpose"]
MODEL_POINTER = frozen.MODEL_POINTER
REMOTE_RUNTIME = "/root/public_colab_runtime_group.py"
REMOTE_COORDINATOR = "/root/public_runtime_journal_group_commit.py"

def runtime_source_path(source_file, is_local):
    if not is_local:
        return Path(REMOTE_RUNTIME)
    return Path(source_file).resolve().parents[2] / ".codex-run/public_colab_runtime_group.py"


RUNTIME_SCRIPT = runtime_source_path(__file__, modal.is_local())
COORDINATOR_SCRIPT = RUNTIME_SCRIPT.with_name("public_runtime_journal_group_commit.py")
FAST_ACTOR_SCRIPT = RUNTIME_SCRIPT.with_name("public_colab_runtime_fast.py")
if hashlib.sha256(FAST_ACTOR_SCRIPT.read_bytes()).hexdigest() != FROZEN_ACTOR_SHA256:
    raise ValueError("Frozen fast actor source hash mismatch")
for dependency, expected in ((RUNTIME_SCRIPT, "6755e4e41d9ebb06ad7febdc93eb71f092e98e910d81048bca34cbffffdc5118"),
        (COORDINATOR_SCRIPT, "3493b27ff29fa80f3b931e224458e727d546305833e2b22d35441aae6dd716a4")):
    if hashlib.sha256(dependency.read_bytes()).hexdigest() != expected:
        raise ValueError("Frozen actor/coordinator dependency changed")
CONTAINER_IMPORTED_AT = time.monotonic()
CPU_CORES, MEMORY_GIB = frozen.CPU_CORES, frozen.MEMORY_GIB
FUNCTION_SECONDS, STARTUP_SECONDS = PROFILE["function_seconds"], PROFILE["startup_seconds"]
TOTAL_RESERVED_SECONDS, RESERVED_USD = PROFILE["total_seconds"], PROFILE["reserved_usd"]
INCLUSIVE_HOURLY_USD = frozen.INCLUSIVE_HOURLY_USD
ESTIMATED_MAX_COMPUTE_USD = TOTAL_RESERVED_SECONDS * INCLUSIVE_HOURLY_USD / 3600
hf_cache, secret = frozen.hf_cache, frozen.secret
app = modal.App(APP_NAME)
# frozen.image already contains the exact reviewed fast actor. These source-only
# layers reuse every CUDA/package/compile layer and add all group dependencies.
image = (frozen.image
         .add_local_file(str(FROZEN_MODAL_PATH), "/root/modal_public_portfolio_fast_runtime.py", copy=True)
         .add_local_file(str(COORDINATOR_SCRIPT), REMOTE_COORDINATOR, copy=True)
         .add_local_file(str(RUNTIME_SCRIPT), REMOTE_RUNTIME, copy=True))
load_runtime = frozen.load_runtime


def source_manifest():
    paths = [Path(__file__).resolve(), FROZEN_MODAL_PATH, RUNTIME_SCRIPT, COORDINATOR_SCRIPT, FAST_ACTOR_SCRIPT]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


ORIGINAL_EXPORT_SHA256 = "8c9822bfcb52d9e14398da9c9057790ec169f4f754cc37e4b00b89ff7b3b6206"
RESUME_HELPER_SHA256 = "7dceaeab22f75ce4749fc58023245c5efe4556f68e94e8b4c19eed5febf51150"
RESUME_COLLECTOR_SHA256 = "81eee19f4761d0829abbd215c7fdb48d137e0a77762cb10c298955187a6c2614"
LIMITS = {"cpu": [4, 4], "memory_mib": [131072, 131072]}


def validate_profile_reservation(profile, reservation, request_count):
    frozen.validate_profile_reservation(profile, reservation, request_count)
    if (profile != runtime_profile() or reservation.get("runtime_profile") != "group-fullbatch75"
            or reservation.get("max_hourly_units") != 7.2
            or reservation.get("explicit_resource_limits") != LIMITS
            or reservation.get("source_dependencies_sha256") != source_manifest()):
        raise ValueError("Exact75-minute profile, five reviewed sources and hard4CPU/128GiB required")


def validate_inputs(requests_jsonl, reservation, prior_wandb_receipt, performance_smoke=False, resume_receipt=None):
    rows = load_runtime().parse_request_jsonl(requests_jsonl)
    if performance_smoke is not False or not rows:
        raise ValueError("Omni continuation requires original native rows and forbids throughput smokes")
    validate_profile_reservation(PROFILE, reservation, len(rows) + 2)
    if (prior_wandb_receipt.get("mode") != "online"
            or prior_wandb_receipt.get("initialized_before_model_work") is not True
            or not prior_wandb_receipt.get("run_id")):
        raise ValueError("Prior online W&B evidence required")
    if (not isinstance(resume_receipt, dict) or resume_receipt.get("schema") != "omni-actor-resume-review-v1"
            or resume_receipt.get("status") != "PREPARED_NOT_DISPATCHED"
            or resume_receipt.get("score") is not None or resume_receipt.get("native_ingestion_performed") is not False
            or resume_receipt.get("original_export_sha256") != ORIGINAL_EXPORT_SHA256
            or resume_receipt.get("source_sha256", {}).get("preparer") != RESUME_HELPER_SHA256
            or resume_receipt.get("source_sha256", {}).get("collector") != RESUME_COLLECTOR_SHA256
            or resume_receipt.get("full_expected_total") != 4428
            or resume_receipt.get("automatic_resampling_allowed") is not False
            or resume_receipt.get("byte_identical_original_rows") is not True
            or resume_receipt.get("new_native_claims") != 0 or resume_receipt.get("model_calls") != 0
            or resume_receipt.get("provider_calls") != 0):
        raise ValueError("Original reviewed untouched-row selection required")
    digest = hashlib.sha256(requests_jsonl.encode("utf-8")).hexdigest()
    receipt_digest = hashlib.sha256((json.dumps(resume_receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)+"\n").encode()).hexdigest()
    if (digest != reservation.get("requests_sha256") or digest != resume_receipt.get("continuation_requests_sha256")
            or receipt_digest != reservation.get("resume_receipt_sha256")
            or reservation.get("original_export_sha256") != ORIGINAL_EXPORT_SHA256):
        raise ValueError("Continuation bytes/selection/reservation identity mismatch")
    ids = [row.get("task_id") for row in rows]
    excluded = resume_receipt.get("excluded_started_task_ids", [])
    if (ids != resume_receipt.get("continuation_task_ids") or len(ids) != resume_receipt.get("continuation_count")
            or len(set(ids)) != len(ids) or len(set(excluded)) != len(excluded)
            or set(ids).intersection(excluded) or len(ids)+len(excluded) != 4428):
        raise ValueError("Selection contains duplicate/previously started tasks or an incomplete inventory")
    for row in rows:
        if (row.get("kind") != "actor" or row.get("api_path") != "/v1/chat/completions"
                or row.get("contract_sha256") != resume_receipt.get("contract_sha256")):
            raise ValueError("Only selected original Omni actor rows are allowed")
    now = dt.datetime.now(dt.timezone.utc)
    observed = dt.datetime.fromisoformat(resume_receipt["provider_observed_at"].replace("Z", "+00:00"))
    deadline = dt.datetime.fromisoformat(resume_receipt["review_valid_until"].replace("Z", "+00:00"))
    if observed.tzinfo is None or deadline.tzinfo is None or not observed <= now <= deadline or (deadline-observed).total_seconds() > 300:
        raise ValueError("Fresh complete terminal inventory required before dispatch")


def stop_marker_path(reservation_id):
    return "/public-portfolio-group-stops/" + load_runtime().stable_hash(reservation_id)[:24] + ".json"


def write_stop_marker(reservation_id):
    """Only a control-plane file write; never starts a function or GPU."""
    if not reservation_id:
        raise ValueError("An explicit active reservation id is required")
    path = stop_marker_path(reservation_id)
    payload = json.dumps({"reservation_id": reservation_id, "action": "STOP_AT_IDLE_BOUNDARY"}).encode()
    with hf_cache.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(payload), path)
    return path


@app.function(image=image, gpu="H200", cpu=(4, 4), memory=(131072, 131072),
              volumes={"/cache": hf_cache}, secrets=[secret],
              env={"PUBLIC_GROUP_RUNTIME_PROFILE": "fullbatch"},
              timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS,
              retries=0, max_containers=1, min_containers=0,
              single_use_containers=True)
def run_batch(requests_jsonl: str, reservation: dict, prior_wandb_receipt: dict,
              served_model_name: str = "pavlov-public-portfolio-bf16", performance_smoke: bool = False, resume_receipt: dict | None = None) -> dict:
    """One GPU session: hash-check, smoke text+image, batch, terminate and return."""
    validate_inputs(requests_jsonl, reservation, prior_wandb_receipt, performance_smoke, resume_receipt)
    if served_model_name != "pavlov-public-portfolio-bf16":
        raise ValueError("Exact original served model identity required")
    runtime = load_runtime()
    run_id = runtime.stable_hash(reservation["reservation_id"])[:24]
    directory = Path("/cache/public-portfolio-group-runtime") / run_id
    if directory.exists():
        raise ValueError("Reservation already has a runtime directory; refusing implicit retry")
    directory.mkdir(parents=True)
    requests_path = directory / "requests.jsonl"
    requests_path.write_text(requests_jsonl)
    runtime.write_json(directory / "reservation.json", reservation)
    runtime.write_json(directory / "prior_wandb.json", prior_wandb_receipt)
    runtime.write_json(directory / "resume_receipt.json", resume_receipt)
    source_sha256 = source_manifest()
    pointer = json.loads(Path(MODEL_POINTER).read_text())
    # One allowance covers bootstrap time, all26hashes, loading, requests and teardown.
    remaining = int(FUNCTION_SECONDS - (time.monotonic() - CONTAINER_IMPORTED_AT) - 35)
    if remaining < 120:
        raise TimeoutError("Too little reserved GPU session time remains before loading")
    command = [sys.executable, REMOTE_RUNTIME,
               "--model-dir", pointer["merged_path"], "--merge-receipt", MODEL_POINTER,
               "--reservation", str(directory / "reservation.json"),
               "--wandb-receipt", str(directory / "prior_wandb.json"),
               "--output-dir", str(directory / "result"), "--cpu-offload-gib", "0",
               "--wall-seconds", str(remaining), "--startup-seconds", str(min(500, remaining - 40)),
               "--request-seconds", "180", "--max-model-len", "32768", "--max-num-seqs", "32",
               "--commit-volume", "pavlov-e1-qwen36-hf-cache",
               "--stop-marker-volume-path", stop_marker_path(reservation["reservation_id"]),
               "--requests-jsonl", str(requests_path), "--served-model-name", served_model_name]
    if performance_smoke:
        command += ["--performance-smoke"]
    process = None
    status = "RPC_FAILED"
    try:
        with (directory / "supervisor.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process.wait(timeout=remaining + 20)
        status = "RPC_RETURNED" if process.returncode == 0 else "RPC_RUNTIME_FAILED"
    except subprocess.TimeoutExpired:
        status = "RPC_SUPERVISOR_TIMEOUT"
    finally:
        runtime.stop_process(process)
        runtime.write_json(directory / "rpc_receipt.json", {
            "status": status, "score": None, "reservation_id": reservation["reservation_id"],
            "runtime_profile": PROFILE,
            "gpu": "H200", "function_timeout_seconds": FUNCTION_SECONDS,
            "startup_timeout_seconds": STARTUP_SECONDS, "single_use_container": True,
            "max_compute_usd_at_published_rates": ESTIMATED_MAX_COMPUTE_USD,
            "reserved_usd": RESERVED_USD, "pricing_source": "https://modal.com/pricing",
            "source_sha256": source_sha256,
            "explicit_resource_limits": LIMITS,
            "resume_receipt_sha256": reservation["resume_receipt_sha256"],
            "stop_marker_volume_path": stop_marker_path(reservation["reservation_id"]),
            "optional_performance_smoke_count": 32 if performance_smoke else 0,
            "finished_at": runtime.utcnow()})
        hf_cache.commit()
    result_path = directory / "result/runtime_receipt.json"
    responses_path = directory / "result/responses.jsonl"
    started_path = directory / "result/started_requests.jsonl"
    def tail(path: Path) -> str:
        if not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - 24000))
            return f.read().decode(errors="replace")
    return {"rpc_receipt": json.loads((directory / "rpc_receipt.json").read_text()),
            "receipt": json.loads(result_path.read_text()) if result_path.exists() else {"status": status, "score": None},
            "responses": [json.loads(x) for x in responses_path.read_text().split("\n") if x.strip()] if responses_path.exists() else [],
            "started_requests": [json.loads(x) for x in started_path.read_text().split("\n") if x.strip()] if started_path.exists() else [],
            "supervisor_log_tail": tail(directory / "supervisor.log"),
            "server_log_tail": tail(directory / "result/server.log"),
            "journal_epoch_fences": [json.loads(x) for x in (directory / "result/journal_epochs.jsonl").read_text().split("\n") if x.strip()] if (directory / "result/journal_epochs.jsonl").exists() else [],
            "volume_output_directory": str(directory)}


@app.local_entrypoint()
def main(reservation: str, wandb_receipt: str, requests: str, resume_receipt: str, output: str):
    """Explicit root-owned dispatch, with cleanup for uncertain spawn and all failures."""
    funds = json.loads(Path(reservation).read_text())
    wandb = json.loads(Path(wandb_receipt).read_text())
    selection = json.loads(Path(resume_receipt).read_text())
    data = Path(requests).read_text()
    validate_inputs(data, funds, wandb, False, selection)
    destination = Path(output)
    if destination.exists():
        raise ValueError("Fresh output path required")
    sources = source_manifest()
    call = None
    try:
        call = run_batch.spawn(data, funds, wandb, "pavlov-public-portfolio-bf16", False, selection)
        result = call.get(timeout=TOTAL_RESERVED_SECONDS)
        result["client_source_dependencies_sha256_before_dispatch"] = sources
        result["explicit_resource_limits"] = LIMITS
        result["client_dispatch"] = {"app_id": app.app_id, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                     "requests_sha256": funds["requests_sha256"], "resume_receipt_sha256": funds["resume_receipt_sha256"]}
        with destination.open("x") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps({"status": result["receipt"]["status"], "output": str(destination), "responses": len(result["responses"]), "score": None}))
    except BaseException as exc:
        failure = {"status": "CLIENT_RPC_FAILED", "error_type": type(exc).__name__, "score": None, "app_id": app.app_id,
                   "reservation_id": funds["reservation_id"], "source_dependencies_sha256_before_dispatch": sources}
        try:
            if not destination.exists():
                with destination.open("x") as handle:
                    json.dump(failure, handle, indent=2)
        finally:
            try:
                if call is not None:
                    call.cancel(terminate_containers=True)
            finally:
                subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app.app_id],
                               timeout=30, capture_output=True, check=False)
        raise
