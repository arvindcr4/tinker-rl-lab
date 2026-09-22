"""Bounded exact-merged BF16 serving candidate; a canary is never a suite score.

Requires a separately supplied budget reservation and the existing exact-merge
receipt. Does not allocate hardware, train, quantize, or silently drop images.
Run --help for the CLI. Imports are deliberately light until all local gates pass.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import signal
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "Qwen/Qwen3.6-35B-A3B"
BASE_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
ADAPTER = "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1-1f1b-5ff-9f777c4018b6"
ADAPTER_REVISION = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"
MERGE_METHOD = "streaming_lora_delta_merge_v1"
ARM = "pavlov_exact_merged_bf16_vllm028_decode_graphs32"
SERVED_MODEL = "pavlov-public-portfolio-bf16"
GIB = 1024**3
BATCH_TEARDOWN_SECONDS = 35
GRAPH_CONFIG = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY",
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32], "max_cudagraph_capture_size": 32}
PERFORMANCE_SMOKE_COUNT = 32


def stop_requested(args) -> bool:
    """Read a committed stop marker without reloading open volume journals."""
    marker = getattr(args, "stop_marker", None)
    if marker and Path(marker).exists():
        return True
    volume_path = getattr(args, "stop_marker_volume_path", None)
    if not volume_path:
        return False
    import modal
    try:
        # Content is immaterial: existence requests a drain. Read at most one chunk.
        for unused in modal.Volume.from_name(args.commit_volume).read_file(volume_path):
            return True
        return True
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False
    # Other control-plane errors propagate: do not start work with unknown stop state.


def native_boundary_guard(args, deadline) -> str | None:
    if stop_requested(args):
        return "STOP_MARKER_REQUESTED"
    if deadline is not None and deadline - time.monotonic() < 2 * args.request_seconds + BATCH_TEARDOWN_SECONDS:
        return "INSUFFICIENT_TIME_FOR_NEXT_REQUEST"
    return None


def performance_smoke_requests(served_model: str) -> list[dict]:
    """Synthetic throughput diagnostic; contains no benchmark task or gold answer."""
    return [{"custom_id": f"runtime-performance-smoke-{i}", "is_runtime_smoke": True,
             "payload": {"model": served_model, "messages": [{"role": "user", "content":
                f"Synthetic runtime exercise {i+1}. Write a detailed, continuous tutorial of at least 1600 words about organizing a fictional community garden, covering soil, watering, planting, paths, tools, seasons and recordkeeping. Continue with concrete examples until the output limit. This is a throughput diagnostic, not an evaluation question."}],
                "temperature": 0, "top_p": 1, "max_tokens": 1024, "n": 1, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}}
            for i in range(PERFORMANCE_SMOKE_COUNT)]


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def parse_request_jsonl(text: str) -> list[dict]:
    # JSONL records use LF; U+2028/U+2029 inside JSON strings are payload text.
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def check_reservation(value: dict, wall_seconds: int, requests: int) -> None:
    if value.get("status") != "RESERVED" or not value.get("reservation_id"):
        raise ValueError("A real RESERVED budget reservation is required")
    if value.get("provider") not in {"colab", "modal"}:
        raise ValueError("Reservation provider must name the actual colab or modal allocation")
    # Currency can be USD or provider compute units; never invent an exchange rate.
    if value.get("unit") not in {"USD", "colab_compute_units"}:
        raise ValueError("Reservation unit must be USD or colab_compute_units")
    reserved = float(value["reserved_units"])
    hourly = float(value["max_hourly_units"])
    if not all(math.isfinite(x) and x > 0 for x in (reserved, hourly)):
        raise ValueError("Reservation amounts and conservative hourly rate must be positive")
    if not 1 <= wall_seconds <= min(7200, int(value["max_wall_seconds"])):
        raise ValueError("Requested wall time exceeds reservation or two-hour process limit")
    if requests > int(value["max_requests"]) or requests < 2:
        raise ValueError("Request count must include both smoke requests and fit reservation")
    if wall_seconds * hourly / 3600 > reserved + 1e-9:
        raise ValueError("Reserved amount does not cover requested maximum wall time")
    expiry = dt.datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=wall_seconds):
        raise ValueError("Reservation must remain valid through the whole requested run")


def check_merge_identity(receipt: dict, directory: Path) -> dict:
    expected = {"base_model": BASE, "base_commit": BASE_REVISION,
                "adapter_repo": ADAPTER, "adapter_commit": ADAPTER_REVISION,
                "merge_method": MERGE_METHOD, "all_adapter_tensors_consumed": True,
                "adapter_tensor_count": 862, "adapter_module_count": 431,
                "weight_file_count": 26}
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"Merge provenance mismatch: {key}")
    hashes = receipt.get("weight_shard_sha256", {})
    if len(hashes) != 26 or any(Path(x).name != x for x in hashes):
        raise ValueError("Expected 26 safe shard names in immutable merge receipt")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(hashes):
        raise ValueError("Weight index and receipt disagree")
    if set(p.name for p in directory.glob("*.safetensors")) != set(hashes):
        raise ValueError("Merged model has extra or missing weight shards")
    for name, expected_hash in sorted(hashes.items()):
        if digest(directory / name) != expected_hash:
            raise ValueError(f"Merged shard hash mismatch: {name}")
        print(f"VERIFIED_SHARD {name}", flush=True)
    config = json.loads((directory / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("Quantization is forbidden in this exact BF16 runtime arm")
    if "Qwen3_5MoeForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("Expected the multimodal Qwen3.5 MoE architecture")
    return {**expected, "weight_shard_sha256": hashes, "receipt_sha256": stable_hash(receipt)}


def check_live_hf(directory: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    base_info = api.model_info(BASE, revision=BASE_REVISION)
    adapter_info = api.model_info(ADAPTER, revision=ADAPTER_REVISION)
    if base_info.sha != BASE_REVISION or adapter_info.sha != ADAPTER_REVISION:
        raise ValueError("Live immutable Hugging Face revision verification failed")
    # No tokenizer or vision-preprocessor drift hidden behind unchanged weight hashes.
    files = {x.rfilename for x in base_info.siblings}
    aux_hashes = {}
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "preprocessor_config.json", "processor_config.json", "chat_template.jinja",
                     "generation_config.json"):
        if filename not in files:
            continue
        source = Path(hf_hub_download(BASE, filename, revision=BASE_REVISION))
        if not (directory / filename).is_file() or digest(source) != digest(directory / filename):
            raise ValueError(f"Pinned tokenizer/config/processor file mismatch: {filename}")
        aux_hashes[filename] = digest(source)
    if "preprocessor_config.json" not in aux_hashes:
        raise ValueError("Pinned vision preprocessing config is required")
    return {"base_commit": base_info.sha, "adapter_commit": adapter_info.sha,
            "auxiliary_sha256": aux_hashes, "verified_at": utcnow()}


def check_hardware(offload_gib: float) -> dict:
    import psutil
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    gpu = torch.cuda.get_device_properties(0)
    ram = psutil.virtual_memory()
    if gpu.total_memory < 38 * GIB:
        raise RuntimeError("This profile requires at least an A100 40GB")
    if offload_gib > 0 and ram.available < (offload_gib + 12) * GIB:
        raise RuntimeError("Insufficient available host RAM for offloaded weights plus 12GiB margin")
    if offload_gib == 0 and gpu.total_memory < 84_000_000_000:
        raise RuntimeError("BF16 without offload requires at least 84GB memory")
    if gpu.total_memory < 80 * GIB and offload_gib < 44:
        raise RuntimeError("A100 40GB requires this candidate's 44-48GiB offload profile")
    return {"gpu_name": gpu.name, "gpu_total_bytes": gpu.total_memory,
            "cpu_available_bytes": ram.available, "cpu_total_bytes": ram.total,
            "torch_cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability(0))}


def package_versions() -> dict:
    versions = {p: importlib.metadata.version(p) for p in
                ("vllm", "torch", "transformers", "huggingface-hub", "safetensors", "wandb")}
    for p, v in {"vllm": "0.28.0", "torch": "2.13.0", "transformers": "5.16.1"}.items():
        if versions[p].split("+")[0] != v:
            raise RuntimeError(f"Package drift: {p} requires {v}")
    return versions


def server_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(args.model_dir), "--tokenizer", str(args.model_dir),
               "--served-model-name", args.served_model_name, "--host", "127.0.0.1", "--port", str(args.port),
               "--dtype", "bfloat16", "--seed", "809",
               "--compilation-config", json.dumps(GRAPH_CONFIG, separators=(",", ":")),
               "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
               "--max-num-batched-tokens", "8192", "--gpu-memory-utilization", "0.88",
               "--limit-mm-per-prompt", '{"image":4,"video":0}',
               "--mm-encoder-attn-backend", "FLASH_ATTN", "--mm-processor-cache-gb", "0",
               "--reasoning-parser", "qwen3", "--no-enable-prefix-caching"]
    if args.cpu_offload_gib:
        command += ["--offload-backend", "uva", "--cpu-offload-gb", str(args.cpu_offload_gib),
                    "--cpu-offload-params", "experts"]
    if args.enable_tool_calling:
        command += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    return command


def request_json(url: str, payload: dict | None, timeout: float, return_raw: bool = False):
    request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        try:
            raw = response.read()
        except Exception as exc:
            raise HTTPBodyReadFailure(response.status, getattr(exc, "partial", b""), type(exc).__name__) from exc
        # Generation callers persist this transport evidence before decoding.
        return (None, raw, response.status) if return_raw else json.loads(raw)


class HTTPBodyReadFailure(Exception):
    def __init__(self, status, raw, error_type):
        super().__init__("HTTP response received but body read failed")
        self.status, self.raw, self.error_type = status, raw, error_type


def persist_http_response(result, output, lock, commit_volume=None):
    """Keep transport bytes durable even if JSON decoding/validation crashes."""
    with lock:
        path = Path(output.name).with_name("http_responses.jsonl")
        with path.open("a") as transport:
            transport.write(json.dumps(result) + "\n")
            transport.flush()
            os.fsync(transport.fileno())
        sync_volume(commit_volume)


def smoke_requests(served_model: str = SERVED_MODEL) -> list[dict]:
    from PIL import Image
    stream = io.BytesIO()
    Image.new("RGB", (224, 224), (255, 0, 0)).save(stream, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
    common = {"model": served_model, "max_tokens": 32, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}}
    return [{**common, "messages": [{"role": "user", "content": "Reply with the single word READY."}]},
            {**common, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is the main color? Reply with one word."},
                {"type": "image_url", "image_url": {"url": uri}}]}]}]


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    # The server and engine workers live in an isolated process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def sync_volume(name: str | None) -> None:
    if name:
        import modal
        modal.Volume.from_name(name, create_if_missing=False).commit()


def execute_request(index: int, row: dict, args: argparse.Namespace, base_url: str,
                    output, started_output, lock: threading.Lock) -> dict:
    """Count exact rendered tokens, durably mark started, save exact HTTP bytes."""
    payload = dict(row.get("payload", row.get("body", row)))
    if payload.get("model") != args.served_model_name:
        raise ValueError(f"Request {index} model does not match declared served model")
    if payload.get("stream", False) is not False or int(payload.get("n", 1)) != 1:
        raise ValueError("Bounded batch accepts one nonstreaming completion per task")
    if "messages" not in payload:
        raise ValueError(f"Request {index} lacks messages")
    if row.get("api_path", "/v1/chat/completions") != "/v1/chat/completions":
        raise ValueError("This actor runtime only accepts chat completions; native Omni-Judge needs its separate pinned model")
    max_tokens = int(payload.get("max_completion_tokens", payload.get("max_tokens", 0)))
    if max_tokens < 1:
        raise ValueError("Each request requires an explicit positive completion token cap")
    tokenization = {"model": payload["model"], "messages": payload["messages"], "add_generation_prompt": True}
    for key in ("chat_template", "chat_template_kwargs", "tools", "documents"):
        if key in payload:
            tokenization[key] = payload[key]
    token_info = request_json(base_url + "/tokenize", tokenization, args.request_seconds)
    count = int(token_info["count"])
    if count + max_tokens > args.max_model_len:
        raise ValueError(f"Request {index}: {count} prompt tokens + {max_tokens} requested tokens exceed context {args.max_model_len}; no truncation permitted")
    t = time.monotonic()
    result = {"request_index": index, "custom_id": row.get("custom_id", row.get("task_id")),
              "is_runtime_smoke": index < 2 or row.get("is_runtime_smoke") is True, "payload_sha256": stable_hash(payload),
              "started_at": utcnow(), "prompt_token_count": count,
              "max_completion_tokens": max_tokens, "score": None}
    result.update({k: row[k] for k in ("task_id", "batch_id", "contract_sha256", "request_sha256", "kind", "api_path") if k in row})
    # Commit STARTED before a paid generation. Interrupted IDs remain visible to
    # the coordinator, and cannot be mistaken for unattempted work.
    with lock:
        started_output.write(json.dumps({**result, "status": "GENERATION_STARTED"}) + "\n")
        started_output.flush()
        os.fsync(started_output.fileno())
        sync_volume(args.commit_volume)
    try:
        _, raw, http_status = request_json(base_url + "/v1/chat/completions", payload, args.request_seconds, return_raw=True)
        result.update({"http_status": http_status,
                       "raw_body_base64": base64.b64encode(raw).decode(), "status": "HTTP_RESPONSE_RECORDED"})
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception as read_exc:
            raw = getattr(read_exc, "partial", b"")
            result.update({"body_read_failed": True, "error_type": type(read_exc).__name__})
        finally:
            exc.close()
        result.update({"http_status": exc.code, "raw_body_base64": base64.b64encode(raw).decode(),
                       "status": "HTTP_BODY_READ_FAILED" if result.get("body_read_failed") else "HTTP_ERROR_RECORDED"})
    except HTTPBodyReadFailure as exc:
        result.update({"http_status": exc.status, "raw_body_base64": base64.b64encode(exc.raw).decode(),
                       "status": "HTTP_BODY_READ_FAILED", "body_read_failed": True, "error_type": exc.error_type})
    except Exception as exc:
        result.update({"http_status": 0, "raw_body_base64": "", "status": "REQUEST_INTERRUPTED_OUTCOME_UNKNOWN",
                       "error_type": type(exc).__name__})
    result.update({"received_at": utcnow(), "elapsed_seconds": time.monotonic() - t})
    if result["http_status"]:
        persist_http_response(result, output, lock, args.commit_volume)
    if result["status"] == "HTTP_RESPONSE_RECORDED":
        try:
            response = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
            result.update({"status": "MALFORMED_JSON_RESPONSE", "error_type": type(exc).__name__})
        else:
            result["response"] = response
            choices = response.get("choices") if isinstance(response, dict) else None
            if (not isinstance(response, dict) or response.get("model") != args.served_model_name
                    or not isinstance(choices, list) or len(choices) != 1
                    or not isinstance(choices[0], dict) or choices[0].get("index") != 0
                    or not isinstance(choices[0].get("message"), dict)):
                result["status"] = "INVALID_SERVER_ENVELOPE"
            elif result["is_runtime_smoke"]:
                content = choices[0]["message"].get("content")
                if not isinstance(content, str) or not content.strip():
                    result["status"] = "EMPTY_SMOKE_GENERATION"
    with lock:
        output.write(json.dumps(result) + "\n")
        output.flush()
        os.fsync(output.fileno())
        sync_volume(args.commit_volume)
    return result


def execute_batch_waves(rows, args, base_url, output, started_output, lock, receipt, run, deadline):
    """Return False only at an idle wave boundary with insufficient reserved time."""
    executor = ThreadPoolExecutor(max_workers=args.max_num_seqs)
    clean_stop = False
    try:
        for first in range(0, len(rows), args.max_num_seqs):
            remaining = deadline - time.monotonic()
            required = 2 * args.request_seconds + BATCH_TEARDOWN_SECONDS
            marker_stop = stop_requested(args)
            if marker_stop or remaining < required:
                clean_stop = True
                receipt["status"] = "RUNTIME_BATCH_PARTIAL_CLEAN_STOP"
                receipt["clean_stop"] = {"reason": "STOP_MARKER_REQUESTED" if marker_stop else "INSUFFICIENT_TIME_FOR_NEXT_WAVE",
                    "remaining_seconds": max(0, remaining), "required_seconds": required,
                    "request_timeout_seconds": args.request_seconds,
                    "tokenization_allowance_seconds": args.request_seconds,
                    "generation_allowance_seconds": args.request_seconds,
                    "teardown_seconds": BATCH_TEARDOWN_SECONDS,
                    "next_unstarted_row_index": first, "unstarted_rows": len(rows) - first,
                    "active_requests": 0}
                with lock:
                    for handle in (output, started_output):
                        handle.flush()
                        os.fsync(handle.fileno())
                    write_json(args.output_dir / "runtime_receipt.json", receipt)
                    sync_volume(args.commit_volume)
                print(f"RUNTIME_BATCH_PARTIAL_CLEAN_STOP unstarted_rows={len(rows)-first}", flush=True)
                return False
            offset = getattr(args, "request_index_offset", 2)
            futures = [executor.submit(execute_request, i + offset, rows[i], args, base_url, output, started_output, lock)
                       for i in range(first, min(first + args.max_num_seqs, len(rows)))]
            wave_failed = False
            wave_error = None
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    # Drain already dispatched peers before stopping the server.
                    # Their intent/raw journals remain independently durable.
                    wave_error = wave_error or exc
                    continue
                receipt["requests_completed"] = receipt.get("requests_completed", 0) + 1
                wave_failed |= result["status"] != "HTTP_RESPONSE_RECORDED"
                run.log({"requests/completed": receipt["requests_completed"], "runtime/request_seconds": result["elapsed_seconds"]})
            if wave_error is not None:
                raise wave_error
            if wave_failed:
                raise RuntimeError("Batch contains recorded failure; refusing automatic retry/new work")
        return True
    finally:
        executor.shutdown(wait=clean_stop, cancel_futures=True)


def run_native_driver(spec: dict, args: argparse.Namespace, base_url: str, receipt: dict, deadline=None) -> bool:
    """Run a native tool loop behind a metered, private, same-container gateway."""
    reason = native_boundary_guard(args, deadline)
    if reason:
        receipt["status"] = "RUNTIME_BATCH_PARTIAL_CLEAN_STOP"
        receipt["clean_stop"] = {"reason": reason, "boundary": "BEFORE_NATIVE_DRIVER", "active_requests": 0}
        write_json(args.output_dir / "runtime_receipt.json", receipt)
        sync_volume(args.commit_volume)
        return False
    command = spec.get("argv")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("Native driver argv must be a nonempty list of strings")
    limit = int(spec["max_requests"])
    default_cap = int(spec["default_max_tokens"])
    if limit < 1 or not 1 <= default_cap < args.max_model_len:
        raise ValueError("Native driver requires positive request and generation caps")
    credential = secrets.token_urlsafe(32)
    durable_lock, budget_lock = threading.Lock(), threading.Lock()
    state = {"requests": 0, "cache_hits": 0, "rejected": 0}
    cache = {}
    in_flight = set()
    outputs = (args.output_dir / "native_driver_responses.jsonl").open("x")
    starts = (args.output_dir / "native_driver_started_requests.jsonl").open("x")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            return

        def send(self, status, raw):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def error_response(self, status, message):
            self.send(status, json.dumps({"error": {"message": message, "type": "runtime_guard"}}).encode())

        def do_GET(self):
            if self.path != "/v1/runtime/episode-boundary":
                self.error_response(404, "Unsupported endpoint")
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + credential):
                self.error_response(401, "Invalid private runtime credential")
                return
            try:
                reason = native_boundary_guard(args, deadline)
                self.send(200, json.dumps({"may_start_episode": reason is None, "reason": reason}).encode())
            except Exception as exc:
                self.error_response(503, "Episode boundary unavailable: " + type(exc).__name__)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.error_response(404, "Unsupported endpoint")
                return
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + credential):
                self.error_response(401, "Invalid private runtime credential")
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 20 * 1024**2:
                self.error_response(413, "Request body outside runtime limit")
                return
            try:
                payload = json.loads(self.rfile.read(size))
                original_sha = stable_hash(payload)
                task_id = self.headers.get("X-Public-Task-ID")
                native_run_id = self.headers.get("X-Public-Run-ID")
                # Cache only when a native episode identity is explicitly bound.
                key = (native_run_id, task_id, original_sha) if task_id else None
                with budget_lock:
                    if key is not None and key in cache:
                        state["cache_hits"] += 1
                        self.send(*cache[key])
                        return
                    if key is not None and key in in_flight:
                        self.error_response(503, "Identical task request still in flight; no duplicate generation")
                        return
                    if state["requests"] >= limit:
                        state["rejected"] += 1
                        self.error_response(403, "Reserved native model request count exhausted")
                        return
                    # A stop marker drains the current episode; it is checked by
                    # the cooperating wrapper before its next immutable intent.
                    if deadline is not None and deadline - time.monotonic() < 2 * args.request_seconds + BATCH_TEARDOWN_SECONDS:
                        state["rejected"] += 1
                        self.error_response(503, "Insufficient reserved time for another request; no generation started")
                        return
                    state["requests"] += 1
                    index = state["requests"] + 1
                    if key is not None:
                        in_flight.add(key)
                if "max_tokens" not in payload and "max_completion_tokens" not in payload:
                    payload["max_tokens"] = default_cap
                row = {"payload": payload, "custom_id": task_id or f"native-call-{index-1}",
                       "request_sha256": original_sha}
                if task_id:
                    row["task_id"] = task_id
                if native_run_id:
                    row["batch_id"] = native_run_id
                result = execute_request(index, row, args, base_url, outputs, starts, durable_lock)
                status = result["http_status"] or 503
                raw = base64.b64decode(result["raw_body_base64"]) if result["raw_body_base64"] else json.dumps({"error": {"message": "Generation outcome unknown; do not resample", "type": "runtime_guard"}}).encode()
                with budget_lock:
                    if key is not None:
                        cache[key] = (status, raw)
                        in_flight.discard(key)
                self.send(status, raw)
            except Exception as exc:
                # Infrastructure guards are503, not400 context errors which native
                # AgentDojo can count as a failed benchmark answer.
                self.error_response(503, "Native runtime request rejected: " + type(exc).__name__)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    receipt["native_driver"] = {"driver_id": spec.get("driver_id"), "argv": command,
                                "request_limit": limit, "default_max_tokens": default_cap,
                                "tool_parser": "qwen3_coder" if args.enable_tool_calling else None,
                                "request_cap_is_runtime_protocol": True}
    write_json(args.output_dir / "native_driver_contract.json", receipt["native_driver"])
    env = {**os.environ, **spec.get("env", {}), "OPENAI_COMPATIBLE_BASE_URL": gateway_url,
           "OPENAI_COMPATIBLE_API_KEY": credential,
           "PUBLIC_RUNTIME_EPISODE_BOUNDARY_URL": gateway_url + "/runtime/episode-boundary",
           "PUBLIC_RUNTIME_EPISODE_STOP_RECEIPT": str(args.output_dir / "native_episode_clean_stop.json")}
    driver = None
    try:
        with (args.output_dir / "native_driver.log").open("w") as log:
            driver = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while driver.poll() is None:
                time.sleep(1)
        receipt["native_driver"]["returncode"] = driver.returncode
        if driver.returncode:
            raise RuntimeError("Native driver failed or exhausted its reservation; inspect native_driver.log")
    finally:
        stop_process(driver)
        server.shutdown()
        server.server_close()
        outputs.close()
        starts.close()
        receipt["native_driver"].update(state)
        write_json(args.output_dir / "native_driver_receipt.json", receipt["native_driver"])
        sync_volume(args.commit_volume)
    episode_stop = args.output_dir / "native_episode_clean_stop.json"
    if episode_stop.exists():
        receipt["status"] = "RUNTIME_BATCH_PARTIAL_CLEAN_STOP"
        receipt["clean_stop"] = {**json.loads(episode_stop.read_text()), "boundary": "NATIVE_EPISODE", "active_requests": 0}
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--merge-receipt", type=Path, required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--wandb-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests-jsonl", type=Path)
    parser.add_argument("--wall-seconds", type=int, default=1800)
    parser.add_argument("--startup-seconds", type=int, default=900)
    parser.add_argument("--request-seconds", type=int, default=300)
    parser.add_argument("--serve-seconds", type=int, default=0,
                        help="Keep the local endpoint available after batch, bounded by total wall deadline")
    parser.add_argument("--cpu-offload-gib", type=float, default=0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--performance-smoke", action="store_true", help="32 synthetic concurrent max1024 generations; no benchmark score")
    parser.add_argument("--stop-marker", type=Path, help="Container-local marker: stop at the next idle boundary")
    parser.add_argument("--stop-marker-volume-path", help="Committed marker path in --commit-volume; read without mount reload")
    parser.add_argument("--commit-volume", help="Mounted Modal volume to commit after each started/result record")
    parser.add_argument("--enable-tool-calling", action="store_true",
                        help="Use official Qwen3.6 qwen3_coder parser; capability remains unverified until an actual tool call")
    parser.add_argument("--native-driver-spec", type=Path,
                        help="Native workload argv/env plus request/cap limits; executes through private guarded gateway")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default=SERVED_MODEL)
    parser.add_argument("--validate-only", action="store_true", help="No GPU import, HF access, W&B init, or model load")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "runtime_receipt.json"
    if receipt_path.exists():
        raise ValueError("Use a fresh output directory; runtime receipts are immutable")
    arm = ARM
    receipt = {"schema_version": "public-colab-exact-runtime-v1", "arm_id": arm,
               "status": "PREPARING", "started_at": utcnow(), "score": None,
               "full_suite_scores": 0, "vision_capability": "UNVERIFIED",
               "tool_call_parser": "qwen3_coder" if args.enable_tool_calling else None,
               "claim_boundary": "Runtime recovery only; full suite scores require separate native evaluator receipts. Exact merged BF16 weights; backend parity with Tinker is not asserted."}
    process = None
    run = None
    log = None
    started = time.monotonic()
    try:
        reservation = json.loads(args.reservation.read_text())
        rows = parse_request_jsonl(args.requests_jsonl.read_text()) if args.requests_jsonl else []
        driver_spec = json.loads(args.native_driver_spec.read_text()) if args.native_driver_spec else None
        extra_requests = int(driver_spec["max_requests"]) if driver_spec else 0
        extra_requests += PERFORMANCE_SMOKE_COUNT if args.performance_smoke else 0
        check_reservation(reservation, args.wall_seconds, len(rows) + 2 + extra_requests)
        if args.performance_smoke and (rows or driver_spec or args.max_num_seqs != 32):
            raise ValueError("Performance smoke is synthetic-only with exactly32 concurrent requests")
        if driver_spec and not args.enable_tool_calling:
            raise ValueError("Native tool-loop driver requires --enable-tool-calling")
        if args.serve_seconds != 0:
            raise ValueError("Use --requests-jsonl: unmetered external endpoint requests are disabled")
        if not 1 <= args.max_num_seqs <= 32:
            raise ValueError("Fast candidate concurrency must be between one and32")
        if args.stop_marker_volume_path and not args.commit_volume:
            raise ValueError("Remote stop marker requires its committed volume")
        if args.cpu_offload_gib:
            raise ValueError("Fast CUDA-graph candidate is bounded to full-GPU BF16; UVA compatibility not established")
        if not 0 <= args.cpu_offload_gib <= 64 or not 2048 <= args.max_model_len <= 32768:
            raise ValueError("Offload/context outside this bounded candidate profile")
        early = json.loads(args.wandb_receipt.read_text())
        if early.get("mode") != "online" or early.get("initialized_before_model_work") is not True or not early.get("run_id"):
            raise ValueError("Prior online W&B receipt must precede model work")
        receipt["reservation"] = {k: reservation[k] for k in
                                  ("reservation_id", "provider", "unit", "reserved_units", "max_hourly_units", "max_wall_seconds", "max_requests", "expires_at")}
        receipt["prior_wandb_receipt_sha256"] = digest(args.wandb_receipt)
        receipt["requests_input_sha256"] = digest(args.requests_jsonl) if args.requests_jsonl else None
        receipt["serving_condition"] = {"compilation_config": GRAPH_CONFIG, "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": 8192, "OMP_NUM_THREADS": "1", "performance_smoke_count": PERFORMANCE_SMOKE_COUNT if args.performance_smoke else 0,
            "sampling_unchanged": True, "cross_backend_numerical_equivalence_asserted": False}
        if args.validate_only:
            receipt["status"] = "LOCAL_GATES_VALIDATED_NOT_LOADED"
            return 0
        def expired(signum: int, frame: object) -> None:
            raise TimeoutError("Reserved process wall time exhausted")
        signal.signal(signal.SIGALRM, expired)
        signal.signal(signal.SIGTERM, expired)
        signal.signal(signal.SIGINT, expired)
        signal.alarm(args.wall_seconds)
        import wandb
        url_parts = early["url"].split("/")
        run = wandb.init(entity=url_parts[3], project=url_parts[4],
                         id="colab-" + stable_hash(reservation["reservation_id"] + str(args.output_dir))[:12],
                         name="public-exact-bf16-runtime", job_type="runtime-recovery",
                         mode="online", resume="never", dir=str(args.output_dir),
                         config={"arm": arm, "base_revision": BASE_REVISION,
                                 "adapter_revision": ADAPTER_REVISION, "score": None})
        if run is None or run.settings.mode != "online" or not run.url:
            raise RuntimeError("Live W&B online initialization failed before model work")
        receipt["wandb"] = {"run_id": run.id, "url": run.url, "mode": "online", "initialized_at": utcnow()}
        write_json(args.output_dir / "wandb_receipt.json", receipt["wandb"])
        receipt["packages"] = package_versions()
        receipt["hardware"] = check_hardware(args.cpu_offload_gib)
        receipt["merge"] = check_merge_identity(json.loads(args.merge_receipt.read_text()), args.model_dir)
        receipt["huggingface"] = check_live_hf(args.model_dir)
        command = server_command(args)
        receipt["server_command"] = command
        receipt["status"] = "LOADING"
        write_json(receipt_path, receipt)
        env = {**os.environ, "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        # Notebook stdout hacks are unnecessary: standalone server gets a real log file.
        log = (args.output_dir / "server.log").open("w")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        base_url = f"http://127.0.0.1:{args.port}"
        loading_started = time.monotonic()
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM exited before ready with code {process.returncode}; see server.log")
            if time.monotonic() - loading_started > args.startup_seconds:
                raise TimeoutError("vLLM startup deadline exceeded")
            try:
                request_json(base_url + "/v1/models", None, 5)
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        receipt["load_seconds"] = time.monotonic() - loading_started
        receipt["endpoint"] = base_url + "/v1"
        lock = threading.Lock()
        with (args.output_dir / "responses.jsonl").open("x") as output, (args.output_dir / "started_requests.jsonl").open("x") as started_output:
            for index, row in enumerate(smoke_requests(args.served_model_name)):
                result = execute_request(index, row, args, base_url, output, started_output, lock)
                if result["status"] != "HTTP_RESPONSE_RECORDED":
                    raise RuntimeError(f"Smoke request {index} failed; see durable response")
                run.log({"requests/completed": index + 1, "runtime/request_seconds": result["elapsed_seconds"]})
                receipt["requests_completed"] = index + 1
                if index == 1:
                    receipt["vision_capability"] = "IMAGE_URL_SMOKE_ACCEPTED"
                    receipt["status"] = "SMOKE_PASSED"
                    write_json(receipt_path, receipt)
                    print("TEXT_AND_IMAGE_SMOKE_PASSED", flush=True)
            if args.performance_smoke:
                perf_start = time.monotonic()
                perf_rows = performance_smoke_requests(args.served_model_name)
                if not execute_batch_waves(perf_rows, args, base_url, output, started_output,
                                           lock, receipt, run, started + args.wall_seconds):
                    receipt["clean_stop"]["boundary"] = "PERFORMANCE_SMOKE"
                    return 0
                receipt["performance_smoke"] = {"count": len(perf_rows), "max_tokens_each": 1024,
                    "elapsed_seconds": time.monotonic() - perf_start, "score": None, "synthetic_only": True}
                perf_results = [r for r in parse_request_jsonl((args.output_dir / "responses.jsonl").read_text())
                                if str(r.get("custom_id", "")).startswith("runtime-performance-smoke-")]
                counts = [r.get("response", {}).get("usage", {}).get("completion_tokens") for r in perf_results]
                if len(counts) == PERFORMANCE_SMOKE_COUNT and all(type(n) is int for n in counts):
                    receipt["performance_smoke"].update({"completion_tokens": sum(counts),
                        "min_completion_tokens_per_request": min(counts),
                        "aggregate_completion_tokens_per_second": sum(counts) / receipt["performance_smoke"]["elapsed_seconds"]})
                write_json(receipt_path, receipt)
                args.request_index_offset = 2 + len(perf_rows)
            # Submit one bounded wave at a time. Failures stop new waves; completed
            # and started IDs have already been committed by each worker.
            batch_complete = execute_batch_waves(rows, args, base_url, output, started_output,
                                                lock, receipt, run, started + args.wall_seconds)
        if not batch_complete:
            return 0
        if driver_spec:
            if not run_native_driver(driver_spec, args, base_url, receipt, started + args.wall_seconds):
                return 0
        receipt["status"] = "NATIVE_DRIVER_FINISHED_EVALUATOR_RECEIPTS_REQUIRED" if driver_spec else "RUNTIME_BATCH_COMPLETE" if rows else "RUNTIME_SMOKE_COMPLETE"
        return 0
    except Exception as exc:
        receipt["status"] = "RUNTIME_FAILED"
        # Do not persist credentials or provider response bodies in exception traces.
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)[:1000] if not isinstance(exc, urllib.error.HTTPError) else f"HTTP status {exc.code}"
        print(f"RUNTIME_FAILED {receipt['error_type']}: {receipt['error']}", flush=True)
        return 1
    finally:
        signal.alarm(0)
        stop_process(process)
        if log:
            log.close()
        receipt["server_process_stopped"] = process is None or process.poll() is not None
        receipt["provider_session_release_required"] = True
        receipt["finished_at"] = utcnow()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if "reservation" in receipt:
            receipt["conservative_units_used"] = receipt["elapsed_seconds"] / 3600 * receipt["reservation"]["max_hourly_units"]
        write_json(receipt_path, receipt)
        if run is not None:
            run.summary.update({"runtime_status": receipt["status"], "score": None})
            run.finish(exit_code=0 if receipt["status"] in {"RUNTIME_BATCH_COMPLETE", "RUNTIME_BATCH_PARTIAL_CLEAN_STOP", "RUNTIME_SMOKE_COMPLETE", "NATIVE_DRIVER_FINISHED_EVALUATOR_RECEIPTS_REQUIRED"} else 1)
        print(f"RECEIPT_PATH {receipt_path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
