"""Private, single-use H200 RPC for the public replacement portfolio runtime.

No deployment, no implicit allocation on import, no automatic retries. The local
entrypoint requires a pre-existing ledger reservation before GPU dispatch.
Uses the already materialized exact BF16 model in the existing HF-cache volume.
"""
from __future__ import annotations

import importlib.util
import base64
import io
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time
import tarfile

import modal

def runtime_profile(name: str) -> dict:
    profiles = {
        "canary": {"name": "canary", "reserved_usd": 2.0, "total_seconds": 900,
                   "startup_seconds": 60, "max_requests": 1969},
        "fullbatch": {"name": "fullbatch", "reserved_usd": 8.0, "total_seconds": 3600,
                      "startup_seconds": 60, "max_requests": 1969},
    }
    if name not in profiles:
        raise ValueError("PUBLIC_RUNTIME_PROFILE must be canary or fullbatch")
    profile = dict(profiles[name])
    profile["function_seconds"] = profile["total_seconds"] - profile["startup_seconds"]
    return profile


PROFILE = runtime_profile(os.environ.get("PUBLIC_RUNTIME_PROFILE", "canary"))
APP_NAME = "pavlov-public-portfolio-runtime-" + PROFILE["name"]
MODEL_POINTER = "/cache/e1-qwen36-seed809-merged-pointer.json"
REMOTE_RUNTIME = "/root/public_colab_runtime.py"
def runtime_source_path(source_file: str, is_local: bool) -> Path:
    # Remote Modal modules are flattened into /root; never index their parents.
    if not is_local:
        return Path(REMOTE_RUNTIME)
    return Path(source_file).resolve().parents[2] / ".codex-run/public_colab_runtime.py"


RUNTIME_SCRIPT = runtime_source_path(__file__, modal.is_local())
CONTAINER_IMPORTED_AT = time.monotonic()
RESERVED_USD = PROFILE["reserved_usd"]
GPU_RATE = 0.001261
CPU_RATE = 0.0000131
RAM_RATE = 0.00000222
CPU_CORES = 4
MEMORY_GIB = 128
FUNCTION_SECONDS = PROFILE["function_seconds"]
STARTUP_SECONDS = PROFILE["startup_seconds"]
TOTAL_RESERVED_SECONDS = FUNCTION_SECONDS + STARTUP_SECONDS
INCLUSIVE_HOURLY_USD = 3600 * (GPU_RATE + CPU_CORES * CPU_RATE + MEMORY_GIB * RAM_RATE)
ESTIMATED_MAX_COMPUTE_USD = TOTAL_RESERVED_SECONDS * INCLUSIVE_HOURLY_USD / 3600
CUDA_PREFLIGHT_SOURCE = """#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
__global__ void codex_sm90_probe(__nv_bfloat16* output) {
    using Reduction = cub::BlockReduce<float, 32>;
    __shared__ typename Reduction::TempStorage storage;
    float value = Reduction(storage).Sum(1.0f);
    if (threadIdx.x == 0) output[0] = __float2bfloat16(value);
}
int main() { int version = 0; return cudaRuntimeGetVersion(&version); }
"""
CUDA_PREFLIGHT_WRITE = "from pathlib import Path; Path('/tmp/codex_cuda_sm90_probe.cu').write_text(" + repr(CUDA_PREFLIGHT_SOURCE) + ")"

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("pavlov-e1-qwen36-hf-cache", create_if_missing=False)
secret = modal.Secret.from_name("pavlov-e1-e14")
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-runtime-ubuntu22.04", add_python="3.12")
    .uv_pip_install(
        "vllm==0.28.0+cu129", "torch==2.13.0+cu129", "transformers==5.16.1",
        "huggingface-hub==1.30.0", "safetensors==0.8.0", "wandb==0.29.0",
        extra_index_url="https://wheels.vllm.ai/0.28.0/cu129",
        extra_options="--index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129",
    )
    .apt_install("build-essential")
    .run_commands("cc --version", "python -c 'import shutil; assert shutil.which(\"cc\"), \"Triton host C compiler missing\"'")
    # These compiler components do not depend on CUDA's cuBLAS meta-packages,
    # whose newer versions conflict with the runtime base image's cuBLAS pins.
    .apt_install("cuda-nvcc-12-9=12.9.86-1", "cuda-cudart-dev-12-9=12.9.79-1",
                 "cuda-cccl-12-9=12.9.27-1", "cuda-crt-12-9=12.9.86-1",
                 "cuda-nvvm-12-9=12.9.86-1")
    .run_commands(
        "python -c " + shlex.quote(CUDA_PREFLIGHT_WRITE),
        "/usr/local/cuda-12.9/bin/nvcc --version",
        "/usr/local/cuda-12.9/bin/nvcc -std=c++17 -arch=sm_90 --cudart shared /tmp/codex_cuda_sm90_probe.cu -o /tmp/codex_cuda_sm90_probe",
        "test -x /tmp/codex_cuda_sm90_probe",
        # Compile/link only: image building does not need or allocate a GPU.
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_TELEMETRY": "1",
          "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": "/usr/local/cuda-12.9",
          "CUDACXX": "/usr/local/cuda-12.9/bin/nvcc", "CC": "gcc", "CXX": "g++"})
    .apt_install("libcurand-dev-12-9=10.3.10.19-1")
    .run_commands(
        "test -f /usr/local/cuda-12.9/include/curand.h",
        "FLASHINFER_CUDA_ARCH_LIST=9.0a timeout 300 python -c 'from flashinfer.jit.sampling import gen_sampling_module; module = gen_sampling_module(); module.build(); print(\"FLASHINFER_SAMPLING_BUILD_PASSED\")'",
    )
    .add_local_file(str(RUNTIME_SCRIPT), REMOTE_RUNTIME, copy=True)
)

# AgentDojo stays in its own locked CPU environment; it does not alter vLLM's
# torch/transformers dependencies. All source files are pinned by its manifest.
AGENTDOJO_SETUP = RUNTIME_SCRIPT.parents[1] / "outputs/public_portfolio_2026-09-05/agentdojo_setup" if modal.is_local() else Path("/opt/agentdojo")
AGENTDOJO_WRAPPER = RUNTIME_SCRIPT.parents[1] / "zvf-program/flagship/public_agentdojo_native.py" if modal.is_local() else Path("/opt/agentdojo/public_agentdojo_native.py")
AGENTDOJO_ENABLED = os.environ.get("PUBLIC_RUNTIME_INCLUDE_AGENTDOJO") == "1"
agentdojo_image = image
if AGENTDOJO_ENABLED:
    agentdojo_image = (
        image.uv_pip_install("uv==0.12.10")
        .add_local_dir(str(AGENTDOJO_SETUP / "source"), "/opt/agentdojo/source", copy=True,
                       ignore=["**/__pycache__/**", "**/.venv/**"])
        .run_commands("UV_PROJECT_ENVIRONMENT=/opt/agentdojo/.venv uv sync --project /opt/agentdojo/source --frozen --no-dev",
                      "uv pip install --python /opt/agentdojo/.venv/bin/python wandb==0.29.0")
        .env({"PUBLIC_RUNTIME_INCLUDE_AGENTDOJO": "1"})
        .add_local_file(str(AGENTDOJO_WRAPPER), "/opt/agentdojo/public_agentdojo_native.py")
    )
    for metadata_name in ("selected_source_manifest.json", "source_receipt.json", "manifest.json"):
        agentdojo_image = agentdojo_image.add_local_file(str(AGENTDOJO_SETUP / metadata_name), "/opt/agentdojo/" + metadata_name)


def load_runtime():
    path = Path(REMOTE_RUNTIME) if Path(REMOTE_RUNTIME).exists() else RUNTIME_SCRIPT
    spec = importlib.util.spec_from_file_location("public_exact_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_profile_reservation(profile: dict, reservation: dict, request_count: int) -> None:
    load_runtime().check_reservation(reservation, profile["total_seconds"], request_count)
    if reservation.get("provider") != "modal" or reservation.get("unit") != "USD":
        raise ValueError("H200 reservation must be Modal USD")
    if float(reservation["reserved_units"]) != profile["reserved_usd"]:
        raise ValueError(f"Profile {profile['name']} requires an explicit USD{profile['reserved_usd']:g} reservation")
    if int(reservation["max_wall_seconds"]) != profile["total_seconds"]:
        raise ValueError("Reservation wall limit must exactly bind the selected runtime profile")
    if request_count > profile["max_requests"] or int(reservation["max_requests"]) > profile["max_requests"]:
        raise ValueError("Runtime profile permits at most1969 generation requests including smoke")
    if profile["name"] == "fullbatch" and reservation.get("runtime_profile") != "fullbatch":
        raise ValueError("Fullbatch requires an explicit runtime_profile:fullbatch ledger binding")
    if reservation.get("runtime_profile", profile["name"]) != profile["name"]:
        raise ValueError("Reservation names a different runtime profile")
    if float(reservation["max_hourly_units"]) < INCLUSIVE_HOURLY_USD:
        raise ValueError("Reservation hourly rate omits GPU, CPU, or memory cost")


def validate_inputs(requests_jsonl: str, reservation: dict, prior_wandb_receipt: dict, extra_requests: int = 0) -> None:
    count = len([line for line in requests_jsonl.split("\n") if line.strip()])
    validate_profile_reservation(PROFILE, reservation, count + 2 + extra_requests)
    if prior_wandb_receipt.get("mode") != "online" or prior_wandb_receipt.get("initialized_before_model_work") is not True:
        raise ValueError("W&B online receipt must predate the GPU request")


@app.function(image=image, gpu="H200", cpu=CPU_CORES, memory=MEMORY_GIB * 1024,
              volumes={"/cache": hf_cache}, secrets=[secret],
              env={"PUBLIC_RUNTIME_PROFILE": PROFILE["name"]},
              timeout=FUNCTION_SECONDS, startup_timeout=STARTUP_SECONDS,
              retries=0, max_containers=1, min_containers=0,
              single_use_containers=True)
def run_batch(requests_jsonl: str, reservation: dict, prior_wandb_receipt: dict,
              served_model_name: str = "pavlov-public-portfolio-bf16", native_driver_spec: dict | None = None) -> dict:
    """One GPU session: hash-check, smoke text+image, batch, terminate and return."""
    validate_inputs(requests_jsonl, reservation, prior_wandb_receipt,
                    int(native_driver_spec["max_requests"]) if native_driver_spec else 0)
    runtime = load_runtime()
    run_id = runtime.stable_hash(reservation["reservation_id"])[:24]
    directory = Path("/cache/public-portfolio-runtime") / run_id
    if directory.exists():
        raise ValueError("Reservation already has a runtime directory; refusing implicit retry")
    directory.mkdir(parents=True)
    requests_path = directory / "requests.jsonl"
    requests_path.write_text(requests_jsonl)
    runtime.write_json(directory / "reservation.json", reservation)
    runtime.write_json(directory / "prior_wandb.json", prior_wandb_receipt)
    source_sha256 = {"runtime_script": runtime.digest(Path(REMOTE_RUNTIME)),
                     "modal_module": runtime.digest(Path(__file__))}
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
               "--request-seconds", "180", "--max-model-len", "32768", "--max-num-seqs", "8",
               "--commit-volume", "pavlov-e1-qwen36-hf-cache",
               "--requests-jsonl", str(requests_path), "--served-model-name", served_model_name]
    if native_driver_spec:
        runtime.write_json(directory / "native_driver_spec.json", native_driver_spec)
        command += ["--enable-tool-calling", "--native-driver-spec", str(directory / "native_driver_spec.json")]
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
            "responses": [json.loads(x) for x in responses_path.read_text().splitlines()] if responses_path.exists() else [],
            "started_requests": [json.loads(x) for x in started_path.read_text().splitlines()] if started_path.exists() else [],
            "supervisor_log_tail": tail(directory / "supervisor.log"),
            "server_log_tail": tail(directory / "result/server.log"),
            "native_driver_log_tail": tail(directory / "result/native_driver.log"),
            "volume_output_directory": str(directory)}


@app.function(image=agentdojo_image, gpu="H200", cpu=CPU_CORES, memory=MEMORY_GIB * 1024,
              volumes={"/cache": hf_cache}, secrets=[secret], timeout=FUNCTION_SECONDS,
              env={"PUBLIC_RUNTIME_PROFILE": PROFILE["name"]},
              startup_timeout=STARTUP_SECONDS, retries=0, max_containers=1,
              min_containers=0, single_use_containers=True)
def run_agentdojo(reservation: dict, prior_wandb_receipt: dict, campaign_id: str,
                  max_tasks: int = 1, max_generation_requests: int = 100,
                  served_model_name: str = "pavlov-public-portfolio-bf16") -> dict:
    validate_inputs("", reservation, prior_wandb_receipt, max_generation_requests)
    if not campaign_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in campaign_id):
        raise ValueError("Campaign id must be a safe alphanumeric slug")
    if not 1 <= max_tasks <= 97:
        raise ValueError("AgentDojo max_tasks must be between one and97")
    runtime = load_runtime()
    directory = Path("/cache/public-portfolio-agentdojo") / campaign_id
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"model_id": runtime.BASE, "model_revision": runtime.BASE_REVISION,
                "hf_repo": runtime.ADAPTER, "hf_commit": runtime.ADAPTER_REVISION,
                "served_model_id": served_model_name, "runtime_limits": {
                    "per_request_max_output_tokens": 4096, "temperature": 0,
                    "max_context_tokens": 32768, "wall_time_seconds": TOTAL_RESERVED_SECONDS,
                    "max_generation_requests": max_generation_requests, "budget_usd": RESERVED_USD,
                    "runtime_profile": PROFILE["name"],
                    "boundary": "Explicit experimental inference condition, not upstream default"}}
    identity_path = directory / "model_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Native campaign identity or runtime limits drift")
    runtime.write_json(identity_path, identity)
    url = prior_wandb_receipt["url"].split("/")
    spec = {"driver_id": "agentdojo_native", "max_requests": max_generation_requests,
            "default_max_tokens": 4096,
            "argv": ["/opt/agentdojo/.venv/bin/python", "/opt/agentdojo/public_agentdojo_native.py",
                     "--setup", "/opt/agentdojo", "run", "--output", str(directory / "native"),
                     "--model-identity", str(identity_path), "--wandb-entity", url[3],
                     "--wandb-project", url[4], "--wandb-run-id", "agentdojo-" + campaign_id,
                     "--max-tasks", str(max_tasks)]}
    result = run_batch.local("", reservation, prior_wandb_receipt, served_model_name, spec)
    hf_cache.commit()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and not path.is_symlink():
                tar.add(path, arcname=str(path.relative_to(directory)), recursive=False)
    result["native_artifacts_targz_base64"] = base64.b64encode(archive.getvalue()).decode()
    result["native_campaign_volume_path"] = str(directory)
    return result


@app.local_entrypoint()
def main(reservation: str, wandb_receipt: str, output: str, requests: str = "",
         served_model_name: str = "pavlov-public-portfolio-bf16", workload: str = "batch",
         campaign_id: str = "", max_tasks: int = 1, max_generation_requests: int = 100):
    """Explicit dispatch only; callers own ledger reservation and reconciliation."""
    reservation_data = json.loads(Path(reservation).read_text())
    wandb_data = json.loads(Path(wandb_receipt).read_text())
    requests_data = Path(requests).read_text() if requests else ""
    if workload not in {"batch", "agentdojo"}:
        raise ValueError("Workload must be batch or agentdojo")
    if workload == "agentdojo" and not AGENTDOJO_ENABLED:
        raise ValueError("Set PUBLIC_RUNTIME_INCLUDE_AGENTDOJO=1 to include the isolated native image")
    validate_inputs(requests_data, reservation_data, wandb_data,
                    max_generation_requests if workload == "agentdojo" else 0)
    output_path = Path(output)
    if output_path.exists():
        raise ValueError("Output already exists; use a new reservation and output path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Provider retries of container imports can occur outside function retries.
    # Bound the caller's wait and explicitly stop this transient app on timeout.
    call = (run_agentdojo.spawn(reservation_data, wandb_data, campaign_id, max_tasks,
                               max_generation_requests, served_model_name) if workload == "agentdojo"
            else run_batch.spawn(requests_data, reservation_data, wandb_data, served_model_name))
    try:
        result = call.get(timeout=TOTAL_RESERVED_SECONDS)
    except Exception as exc:
        failure = {"status": "CLIENT_RPC_FAILED", "error_type": type(exc).__name__,
                   "score": None, "app_id": app.app_id,
                   "runtime_profile": PROFILE,
                   "reservation_id": reservation_data["reservation_id"],
                   "client_timeout_seconds": TOTAL_RESERVED_SECONDS}
        output_path.write_text(json.dumps(failure, indent=2) + "\n")
        try:
            call.cancel(terminate_containers=True)
        finally:
            subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app.app_id],
                           check=False, timeout=30, capture_output=True)
        raise
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result_path": str(output_path.resolve()), "status": result["receipt"]["status"],
                      "responses": len(result["responses"]), "score": None}, sort_keys=True))
