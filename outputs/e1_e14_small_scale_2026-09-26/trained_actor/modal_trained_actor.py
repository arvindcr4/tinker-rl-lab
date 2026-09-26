"""OpenAI-compatible vLLM endpoint for the TRAINED E1-E14 actor (seed809 stepfinal LoRA).

Reuses the original campaign's serving route (zvf-program/flagship/modal_public_portfolio_runtime.py +
zvf-program/e1_wave10/recovered/public_colab_runtime_fast.py::server_command):
  * LoRA already merged into exact BF16 weights (streaming_lora_delta_merge_v1, 2026-08-17), cached in
    Modal Volume `pavlov-e1-qwen36-hf-cache` at the path named by e1-qwen36-seed809-merged-pointer.json;
  * vLLM 0.28.0 (cu129) image, same pins; H200; served model id `pavlov-public-portfolio-bf16`.

Deploy (scale-to-zero, max 1 container):  modal deploy modal_trained_actor.py
Stop:                                      modal app stop pavlov-trained-actor-e1e14
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

import modal

APP_NAME = "pavlov-trained-actor-e1e14"
if os.environ.get("TRAINED_ACTOR_WEIGHTS", "trained") == "base":
    APP_NAME += "-basectl"  # adapter-effect control only; never used by lanes
SERVED_MODEL = "pavlov-public-portfolio-bf16"
MODEL_POINTER = "/cache/e1-qwen36-seed809-merged-pointer.json"
BASE_SNAPSHOT_GLOB = "/cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0"
PORT = 8000
# Same compilation config as the original campaign (no torch.compile, decode-only CUDA graphs -> fast cold start).
GRAPH_CONFIG = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32], "max_cudagraph_capture_size": 32}
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
api_secret = modal.Secret.from_name("pavlov-trained-actor-api")  # TRAINED_ACTOR_API_KEY

# Image copied verbatim from modal_public_portfolio_runtime.py (minus the batch-runtime script).
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
    .apt_install("cuda-nvcc-12-9=12.9.86-1", "cuda-cudart-dev-12-9=12.9.79-1",
                 "cuda-cccl-12-9=12.9.27-1", "cuda-crt-12-9=12.9.86-1",
                 "cuda-nvvm-12-9=12.9.86-1")
    .run_commands(
        "python -c " + shlex.quote(CUDA_PREFLIGHT_WRITE),
        "/usr/local/cuda-12.9/bin/nvcc --version",
        "/usr/local/cuda-12.9/bin/nvcc -std=c++17 -arch=sm_90 --cudart shared /tmp/codex_cuda_sm90_probe.cu -o /tmp/codex_cuda_sm90_probe",
        "test -x /tmp/codex_cuda_sm90_probe",
    )
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": "/usr/local/cuda-12.9",
          "CUDACXX": "/usr/local/cuda-12.9/bin/nvcc", "CC": "gcc", "CXX": "g++"})
    .apt_install("libcurand-dev-12-9=10.3.10.19-1")
    .run_commands(
        "test -f /usr/local/cuda-12.9/include/curand.h",
        "FLASHINFER_CUDA_ARCH_LIST=9.0a timeout 300 python -c 'from flashinfer.jit.sampling import gen_sampling_module; module = gen_sampling_module(); module.build(); print(\"FLASHINFER_SAMPLING_BUILD_PASSED\")'",
    )
)

# Knobs (read at deploy time and baked into the function env).
MAX_MODEL_LEN = int(os.environ.get("TRAINED_ACTOR_MAX_MODEL_LEN", "32768"))
MAX_NUM_SEQS = int(os.environ.get("TRAINED_ACTOR_MAX_NUM_SEQS", "16"))
SCALEDOWN_SECONDS = int(os.environ.get("TRAINED_ACTOR_SCALEDOWN_SECONDS", "300"))
WEIGHTS = os.environ.get("TRAINED_ACTOR_WEIGHTS", "trained")  # "base" only for the adapter-effect control


def model_dir() -> str:
    if WEIGHTS == "base":
        return BASE_SNAPSHOT_GLOB
    return json.load(open(MODEL_POINTER))["merged_path"]


def server_command(path: str) -> list[str]:
    # Mirrors public_colab_runtime_fast.server_command (+ tool calling, API key, public bind).
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", path, "--tokenizer", path,
            "--served-model-name", SERVED_MODEL if WEIGHTS == "trained" else "qwen36-base-bf16",
            "--host", "0.0.0.0", "--port", str(PORT),
            "--dtype", "bfloat16", "--seed", "809",
            "--compilation-config", json.dumps(GRAPH_CONFIG, separators=(",", ":")),
            "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
            "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.88",
            "--limit-mm-per-prompt", '{"image":4,"video":0}',
            "--mm-encoder-attn-backend", "FLASH_ATTN", "--mm-processor-cache-gb", "0",
            "--reasoning-parser", "qwen3", "--no-enable-prefix-caching",
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
            "--max-logprobs", "20",
            "--api-key", os.environ["TRAINED_ACTOR_API_KEY"]]


@app.function(image=image, gpu="H200", cpu=4, memory=64 * 1024,
              volumes={"/cache": hf_cache.read_only()}, secrets=[api_secret],
              env={"TRAINED_ACTOR_WEIGHTS": WEIGHTS, "TRAINED_ACTOR_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
                   "TRAINED_ACTOR_MAX_NUM_SEQS": str(MAX_NUM_SEQS)},
              min_containers=0, max_containers=1, scaledown_window=SCALEDOWN_SECONDS,
              timeout=24 * 3600)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1200)
def serve():
    path = model_dir()
    print("serving", WEIGHTS, "weights from", path, flush=True)
    subprocess.Popen(server_command(path))
