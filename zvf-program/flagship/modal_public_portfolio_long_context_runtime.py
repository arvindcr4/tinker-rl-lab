"""Separate65536-context tail runtime:223 native LAB rows plus2 basic smokes.

Reuse the frozen fast CUDA image and unchanged group coordinator. No allocation
on import. Root owns terminal evidence, reservation, dispatch and reconciliation.
All original fast/group/Omni entrypoints remain unchanged.
"""
from __future__ import annotations

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

TAIL_REQUEST_SHA256 = "d65035e77134f684a5244f32b4cf17b463facb626847faea77d66f364d3d84c2"
TAIL_SELECTION_SHA256 = "fcd7852328140660d2f20b0a6c127f475f294739f1f24ad9aa86e52d0167988b"
FROZEN_MODAL_SHA256 = "1f1c1b6b78c74a25a6e3927094540767ec5511f3a627e9456fc6e58a073a3232"
FROZEN_ACTOR_SHA256 = "6171da92f0c15049b8c55658681f2f82b80356334e7ce7c35998d324e7c41c19"
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


def runtime_profile():
    return {"name": "group-long-context-tail", "purpose": "long-context-tail", "reserved_usd": 4.0,
            "total_seconds": 1800, "startup_seconds": 60, "function_seconds": 1740, "max_requests": 225,
            "native_requests": 223, "max_model_len": 65536}


PROFILE = runtime_profile()

APP_NAME = "pavlov-public-portfolio-group-runtime-" + PROFILE["purpose"]
MODEL_POINTER = frozen.MODEL_POINTER
REMOTE_RUNTIME = "/root/public_colab_runtime_group_long_context.py"
REMOTE_COORDINATOR = "/root/public_runtime_journal_group_commit.py"

def runtime_source_path(source_file, is_local):
    if not is_local:
        return Path(REMOTE_RUNTIME)
    return Path(source_file).resolve().parents[2] / ".codex-run/public_colab_runtime_group_long_context.py"


RUNTIME_SCRIPT = runtime_source_path(__file__, modal.is_local())
COORDINATOR_SCRIPT = RUNTIME_SCRIPT.with_name("public_runtime_journal_group_commit.py")
FAST_ACTOR_SCRIPT = RUNTIME_SCRIPT.with_name("public_colab_runtime_long_context.py")
if hashlib.sha256(FAST_ACTOR_SCRIPT.read_bytes()).hexdigest() != FROZEN_ACTOR_SHA256:
    raise ValueError("Frozen fast actor source hash mismatch")
CONTAINER_IMPORTED_AT = time.monotonic()
CPU_CORES, MEMORY_GIB = 4, 128
RESOURCE_LIMITS = {"cpu": [4, 4], "memory_mib": [131072, 131072]}
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
         .add_local_file(str(RUNTIME_SCRIPT), REMOTE_RUNTIME, copy=True)
         .add_local_file(str(FAST_ACTOR_SCRIPT), "/root/public_colab_runtime_long_context.py", copy=True))
def load_runtime():
    path = FAST_ACTOR_SCRIPT
    spec = importlib.util.spec_from_file_location("public_long_context_tail_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_manifest():
    paths = [Path(__file__).resolve(), FROZEN_MODAL_PATH, RUNTIME_SCRIPT, COORDINATOR_SCRIPT, FAST_ACTOR_SCRIPT]
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def validate_profile_reservation(profile, reservation, request_count):
    load_runtime().check_reservation(reservation, profile["total_seconds"], request_count)
    if (reservation.get("provider") != "modal" or reservation.get("unit") != "USD"
            or reservation.get("runtime_profile") != profile["name"]
            or reservation.get("reserved_units") != 4 or reservation.get("max_wall_seconds") != 1800
            or reservation.get("max_requests") != 225 or request_count != 225
            or reservation.get("max_hourly_units") != 8
            or reservation.get("explicit_resource_limits") != RESOURCE_LIMITS
            or reservation.get("declared_context_window") != 65536):
        raise ValueError("Tail reservation must bind4USD/1800s/225requests, CPU4/RAM128GiB and65536context")


def validate_inputs(requests_jsonl, reservation, prior_wandb_receipt, performance_smoke=False):
    rows = load_runtime().parse_request_jsonl(requests_jsonl)
    if (len(rows) != PROFILE["native_requests"] or performance_smoke
            or len({row.get("task_id") for row in rows}) != len(rows)
            or any(not isinstance(row.get("task_id"), str) or row.get("is_runtime_smoke") for row in rows)):
        raise ValueError("Context tail requires223 native rows and only2 basic capability smokes")
    if (reservation.get("requests_sha256") != hashlib.sha256(requests_jsonl.encode()).hexdigest()
            or reservation.get("requests_sha256") != TAIL_REQUEST_SHA256
            or reservation.get("tail_selection_receipt_sha256") != TAIL_SELECTION_SHA256):
        raise ValueError("Tail reservation input bytes mismatch")
    if reservation.get("source_dependencies_sha256") != source_manifest():
        raise ValueError("Tail reservation source hashes mismatch")
    validate_profile_reservation(PROFILE, reservation, len(rows) + 2)
    if (prior_wandb_receipt.get("mode") != "online"
            or prior_wandb_receipt.get("initialized_before_model_work") is not True
            or not prior_wandb_receipt.get("run_id")):
        raise ValueError("Prior W&B online receipt must exist before GPU dispatch")


def stop_marker_path(reservation_id):
    return "/public-portfolio-long-context-stops/" + load_runtime().stable_hash(reservation_id)[:24] + ".json"


def write_stop_marker(reservation_id):
    """Only a control-plane file write; never starts a function or GPU."""
    if not reservation_id:
        raise ValueError("An explicit active reservation id is required")
    path = stop_marker_path(reservation_id)
    payload = json.dumps({"reservation_id": reservation_id, "action": "STOP_AT_IDLE_BOUNDARY"}).encode()
    with hf_cache.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(payload), path)
    return path


@app.function(image=image, gpu="H200", cpu=(CPU_CORES, CPU_CORES), memory=(MEMORY_GIB * 1024, MEMORY_GIB * 1024),
              volumes={"/cache": hf_cache}, secrets=[secret],
              env={"PUBLIC_RUNTIME_INCLUDE_AGENTDOJO": "0"},
              timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS,
              retries=0, max_containers=1, min_containers=0,
              single_use_containers=True)
def run_batch(requests_jsonl: str, reservation: dict, prior_wandb_receipt: dict,
              served_model_name: str = "pavlov-public-portfolio-bf16", performance_smoke: bool = False) -> dict:
    """One GPU session: hash-check, smoke text+image, batch, terminate and return."""
    validate_inputs(requests_jsonl, reservation, prior_wandb_receipt, performance_smoke)
    runtime = load_runtime()
    run_id = runtime.stable_hash(reservation["reservation_id"])[:24]
    directory = Path("/cache/public-portfolio-long-context-runtime") / run_id
    if directory.exists():
        raise ValueError("Reservation already has a runtime directory; refusing implicit retry")
    directory.mkdir(parents=True)
    requests_path = directory / "requests.jsonl"
    requests_path.write_text(requests_jsonl)
    runtime.write_json(directory / "reservation.json", reservation)
    runtime.write_json(directory / "prior_wandb.json", prior_wandb_receipt)
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
               "--request-seconds", "180", "--max-model-len", "65536", "--max-num-seqs", "32",
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
            "explicit_resource_limits": RESOURCE_LIMITS,
            "context_addendum": {"original_context_window": 32768, "declared_context_window": 65536,
                "native_capacity": 262144, "config_sha256": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
                "native_prompts_unchanged": True, "native_sampling_unchanged": True, "numerical_parity_asserted": False,
                "full_tail_tokenization_before_native_generation": True, "selected_tail_sha256": TAIL_REQUEST_SHA256,
                "root_selection_receipt_sha256": TAIL_SELECTION_SHA256},
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
            "tail_tokenization_preflight": json.loads((directory / "result/tail_tokenization_preflight.json").read_text()) if (directory / "result/tail_tokenization_preflight.json").exists() else None,
            "explicit_resource_limits": RESOURCE_LIMITS,
            "volume_output_directory": str(directory)}


@app.local_entrypoint()
def main(reservation: str, wandb_receipt: str, output: str, requests: str = "",
         served_model_name: str = "pavlov-public-portfolio-bf16", performance_smoke: bool = False):
    """Explicit dispatch only; root owns paid ledger and reconciliation."""
    reservation_data = json.loads(Path(reservation).read_text())
    wandb_data = json.loads(Path(wandb_receipt).read_text())
    requests_data = Path(requests).read_text() if requests else ""
    validate_inputs(requests_data, reservation_data, wandb_data, performance_smoke)
    output_path = Path(output)
    if output_path.exists():
        raise ValueError("Output already exists; use a new reservation and output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_sources = source_manifest()
    call = None
    try:
        # A spawn exception can leave a remote call admitted but its ID unknown.
        call = run_batch.spawn(requests_data, reservation_data, wandb_data, served_model_name, performance_smoke)
        result = call.get(timeout=TOTAL_RESERVED_SECONDS)
        result["client_source_dependencies_sha256_before_dispatch"] = local_sources
        result["client_dispatch"] = {"app_id": app.app_id, "requests_sha256": hashlib.sha256(requests_data.encode()).hexdigest(),
                                     "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        with output_path.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        print(json.dumps({"result_path": str(output_path.resolve()), "status": result["receipt"]["status"],
                          "responses": len(result["responses"]), "score": None}, sort_keys=True))
    except BaseException as exc:
        failure = {"status": "CLIENT_RPC_FAILED", "error_type": type(exc).__name__,
                   "score": None, "app_id": app.app_id, "runtime_profile": PROFILE,
                   "reservation_id": reservation_data["reservation_id"],
                   "source_dependencies_sha256_before_dispatch": local_sources,
                   "client_timeout_seconds": TOTAL_RESERVED_SECONDS}
        try:
            # Never overwrite a partially published original result. Disk failure
            # cannot bypass provider cleanup; the missing/partial file stays unresolved.
            if not output_path.exists():
                with output_path.open("x") as handle:
                    json.dump(failure, handle, indent=2)
                    handle.flush(); os.fsync(handle.fileno())
        except BaseException:
            pass
        finally:
            try:
                if call is not None:
                    call.cancel(terminate_containers=True)
            finally:
                if app.app_id:
                    subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app.app_id],
                                   check=False, timeout=30, capture_output=True)
        raise
