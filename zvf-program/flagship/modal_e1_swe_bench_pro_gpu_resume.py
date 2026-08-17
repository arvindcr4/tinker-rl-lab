"""Resume the exact E1 campaign on one or more Modal A100-80GB lanes.

The first portion of the campaign used the pinned Tinker sampler. This runner
continues only tasks without a terminal generation receipt, using the exact
immutable Hugging Face LoRA export on its exact base-model revision. It never
resamples terminal tasks and records the mixed inference backends explicitly.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import modal

try:
    from . import modal_e1_swe_bench_pro_full as base
except ImportError:
    import modal_e1_swe_bench_pro_full as base


APP_LANE = os.environ.get("E1_MODAL_LANE", "main")
if not APP_LANE or len(APP_LANE) > 32 or not all(
    char.isalnum() or char in "-_" for char in APP_LANE
):
    raise RuntimeError("E1_MODAL_LANE must be a short alphanumeric slug")
APP_NAME = f"pavlov-e1-swe-bench-pro-full-gpu-{APP_LANE}"
GPU_TYPE = "A100-80GB"
GPU_RATE_USD_PER_SECOND = 0.000694
GPU_PRICING_SOURCE = "https://modal.com/pricing"
MAX_MODEL_LEN = 49_152
MAX_GPU_PROJECTED_USD = 30.0
PROJECTED_LOAD_SECONDS = 600
PROJECTED_TASK_SECONDS = 60
GENERATION_BACKEND = "modal_gpu_vllm_merged_peft"
VLLM_VERSION = "0.19.0"
MERGE_METHOD = "streaming_lora_delta_merge_v1"
MERGE_CPU_CORES = 16.0
MERGE_MEMORY_GIB = 128.0
CPU_RATE_USD_PER_CORE_SECOND = 0.0000131
MEMORY_RATE_USD_PER_GIB_SECOND = 0.00000222
MERGED_POINTER_PATH = Path("/cache/e1-qwen36-seed809-merged-pointer.json")

secret = modal.Secret.from_name("pavlov-e1-e14")
hf_cache = modal.Volume.from_name(
    "pavlov-e1-qwen36-hf-cache", create_if_missing=True
)
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .uv_pip_install(
        "vllm==0.19.0",
        "transformers==4.57.6",
        "peft==0.18.1",
        "huggingface_hub[hf_xet]>=0.34.0,<1",
        "wandb==0.21.0",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file(
        str(Path(__file__).with_name("modal_e1_swe_bench_pro_full.py")),
        "/root/modal_e1_swe_bench_pro_full.py",
    )
)
merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0",
        "numpy==2.4.3",
        "huggingface_hub>=1.2.1,<2",
        "safetensors==0.7.0",
        "wandb==0.21.0",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_file(
        str(Path(__file__).with_name("modal_e1_swe_bench_pro_full.py")),
        "/root/modal_e1_swe_bench_pro_full.py",
    )
)
app = modal.App(APP_NAME, include_source=True)


def _gpu_projection(pending_count: int) -> float:
    seconds = PROJECTED_LOAD_SECONDS + pending_count * PROJECTED_TASK_SECONDS
    return round(seconds * GPU_RATE_USD_PER_SECOND, 6)


def _augment_receipt(
    receipt: dict[str, Any],
    generations: list[dict[str, Any]],
    gpu_backend_receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    if not gpu_backend_receipts:
        raise RuntimeError("no Modal GPU backend receipts found")
    backend_counts = Counter(
        str(item.get("generation_backend") or "tinker_remote")
        for item in generations
    )
    task_gpu_usd = sum(
        float(item.get("estimated_modal_gpu_usd") or 0.0)
        for item in generations
    )
    startup_gpu_usd = sum(
        float(item.get("estimated_modal_gpu_usd") or 0.0)
        for item in gpu_backend_receipts
    )
    merge_receipts: dict[str, dict[str, Any]] = {}
    for item in gpu_backend_receipts:
        merge = dict(item.get("merge") or {})
        identity_fields = {
            key: merge.get(key)
            for key in (
                "base_commit",
                "adapter_commit",
                "merge_method",
                "merged_path",
                "weight_bytes",
                "weight_shard_sha256",
            )
        }
        if any(value in (None, "", {}) for value in identity_fields.values()):
            raise RuntimeError("GPU merge receipt lacks immutable artifact identity")
        identity = base._sha256_text(base._stable_json(identity_fields))
        merge_receipts[identity] = merge
    if len(merge_receipts) != 1:
        raise RuntimeError("GPU lanes do not share one immutable merge receipt")
    merge_cpu_usd = float(
        next(iter(merge_receipts.values())).get(
            "estimated_modal_cpu_memory_usd", 0.0
        )
    )
    receipt["claim_boundary"] += (
        " The campaign used one immutable model checkpoint through two inference "
        "backends: the original Tinker sampler and a direct Modal GPU vLLM "
        "continuation from an exact, tensor-accounted LoRA delta merge. Backend "
        "counts and versions "
        "are recorded; no terminal task "
        "was resampled."
    )
    receipt["sampling"].update(
        {
            "backend_counts": dict(sorted(backend_counts.items())),
            "mixed_inference_backends": len(backend_counts) > 1,
            "modal_gpu_type": GPU_TYPE,
            "modal_gpu_count": 1,
            "modal_gpu_lane_startups": len(gpu_backend_receipts),
            "modal_gpu_engine": f"vllm=={VLLM_VERSION}",
            "modal_gpu_checkpoint_format": MERGE_METHOD,
            "modal_gpu_adapter_format": "peft_lora_safetensors",
            "modal_gpu_max_model_len": MAX_MODEL_LEN,
            "modal_gpu_adapter_commit": base.HF_COMMIT,
            "terminal_task_resampling_performed": False,
        }
    )
    receipt["cost"].update(
        {
            "estimated_modal_gpu_startup_usd": round(startup_gpu_usd, 8),
            "estimated_modal_gpu_task_usd": round(task_gpu_usd, 8),
            "estimated_modal_gpu_total_usd": round(
                startup_gpu_usd + task_gpu_usd, 8
            ),
            "estimated_modal_merge_cpu_memory_usd": round(merge_cpu_usd, 8),
            "estimated_modal_total_compute_usd": round(
                startup_gpu_usd + task_gpu_usd + merge_cpu_usd, 8
            ),
            "modal_gpu_rate_usd_per_second": GPU_RATE_USD_PER_SECOND,
            "modal_gpu_pricing_source": GPU_PRICING_SOURCE,
        }
    )
    backend_paths = [str(Path(item["path"])) for item in gpu_backend_receipts]
    receipt["artifacts"]["gpu_backend_receipt"] = backend_paths[0]
    receipt["artifacts"]["gpu_backend_receipts"] = backend_paths
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = base._sha256_text(base._stable_json(receipt))
    return receipt


def _load_gpu_backend_receipts(
    root: Path, run_dir: Path
) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("gpu_backend_receipt*.json"))
    if not paths:
        raise RuntimeError("no GPU backend receipt artifacts found")
    receipts: list[dict[str, Any]] = []
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("status") != "READY":
            raise RuntimeError(f"GPU backend is not READY: {path}")
        expected_path = str(path.relative_to(root))
        if receipt.get("path") != expected_path:
            raise RuntimeError(f"GPU backend receipt path drift: {path}")
        expected_sha256 = str(receipt.get("receipt_sha256") or "")
        unhashed = dict(receipt)
        unhashed.pop("receipt_sha256", None)
        if expected_sha256 != base._sha256_text(base._stable_json(unhashed)):
            raise RuntimeError(f"GPU backend receipt hash drift: {path}")
        merge = dict(receipt.get("merge") or {})
        merge_sha256 = str(merge.pop("receipt_sha256", "") or "")
        if merge_sha256 != base._sha256_text(base._stable_json(merge)):
            raise RuntimeError(f"GPU merge receipt hash drift: {path}")
        receipts.append(receipt)
    return receipts


def _merge_compute_cost(seconds: float) -> float:
    per_second = (
        MERGE_CPU_CORES * CPU_RATE_USD_PER_CORE_SECOND
        + MERGE_MEMORY_GIB * MEMORY_RATE_USD_PER_GIB_SECOND
    )
    return round(seconds * per_second, 8)


def _adapter_module_plan(module: str) -> tuple[str, str, str | None]:
    prefix = "base_model.model.model."
    if not module.startswith(prefix):
        raise ValueError(f"unexpected adapter module prefix: {module}")
    relative = module[len(prefix) :]
    if relative == "unembed_tokens":
        return "lm_head.weight", "simple", None
    target_prefix = "model.language_model."
    if relative.endswith(".linear_attn.in_proj_q"):
        stem = relative.removesuffix("in_proj_q")
        return target_prefix + stem + "in_proj_qkv.weight", "qkv", "q"
    if relative.endswith(".linear_attn.in_proj_k"):
        stem = relative.removesuffix("in_proj_k")
        return target_prefix + stem + "in_proj_qkv.weight", "qkv", "k"
    if relative.endswith(".linear_attn.in_proj_v"):
        stem = relative.removesuffix("in_proj_v")
        return target_prefix + stem + "in_proj_qkv.weight", "qkv", "v"
    if relative.endswith(".mlp.experts.w1"):
        stem = relative.removesuffix("w1")
        return target_prefix + stem + "gate_up_proj", "gate_up", "w1"
    if relative.endswith(".mlp.experts.w3"):
        stem = relative.removesuffix("w3")
        return target_prefix + stem + "gate_up_proj", "gate_up", "w3"
    if relative.endswith(".mlp.experts.w2"):
        stem = relative.removesuffix("w2")
        return target_prefix + stem + "down_proj", "simple", None
    return target_prefix + relative + ".weight", "simple", None


@app.function(
    image=merge_image,
    secrets=[secret],
    volumes={"/cache": hf_cache},
    cpu=1.0,
    memory=4096,
    timeout=10 * 60,
)
def inspect_checkpoint_layout() -> dict[str, Any]:
    """Return tensor metadata only; no weight tensor is materialized."""
    import re

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    base_path = Path(
        snapshot_download(
            base.MODEL_ID,
            revision=base.MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
        )
    )
    adapter_path = Path(
        snapshot_download(
            base.HF_REPO,
            revision=base.HF_REVISION,
            token=os.environ["HF_TOKEN"],
        )
    )
    adapter_layout: dict[str, set[tuple[int, ...]]] = {}
    with safe_open(
        adapter_path / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        for key in handle.keys():
            normalized = re.sub(r"\.layers\.\d+\.", ".layers.N.", key)
            adapter_layout.setdefault(normalized, set()).add(
                tuple(handle.get_slice(key).get_shape())
            )

    index = json.loads(
        (base_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    selected = [
        key
        for key in index["weight_map"]
        if key == "lm_head.weight"
        or ".layers.0." in key
        or ".layers.3." in key
    ]
    by_file: dict[str, list[str]] = {}
    for key in selected:
        by_file.setdefault(index["weight_map"][key], []).append(key)
    base_layout: dict[str, tuple[int, ...]] = {}
    for filename, keys in by_file.items():
        with safe_open(base_path / filename, framework="pt", device="cpu") as handle:
            for key in keys:
                base_layout[key] = tuple(handle.get_slice(key).get_shape())
    result = {
        "adapter": {
            key: sorted(shapes) for key, shapes in sorted(adapter_layout.items())
        },
        "base": dict(sorted(base_layout.items())),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


@app.function(
    image=merge_image,
    secrets=[secret],
    volumes={"/cache": hf_cache},
    cpu=MERGE_CPU_CORES,
    memory=int(MERGE_MEMORY_GIB * 1024),
    timeout=2 * 60 * 60,
)
def merge_checkpoint() -> dict[str, Any]:
    """Materialize an exact merged checkpoint when dynamic LoRA is unsupported."""
    import gc
    import hashlib
    import shutil
    from collections import defaultdict

    import torch
    import wandb
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    started_at = base._utc_now()
    started = time.perf_counter()
    run = wandb.init(
        entity="arvindcr4-pes-university",
        project="tinker-rl-lab-pavlov",
        group="e1-swe-bench-pro-full-seed1818",
        job_type="modal-cpu-checkpoint-merge",
        name="e1_full_exact_lora_delta_merge",
        tags=["e1", "swe_bench_pro", "modal", "peft", "checkpoint"],
        mode="online",
        config={
            "base_model": base.MODEL_ID,
            "base_commit": base.MODEL_REVISION,
            "adapter_repo": base.HF_REPO,
            "adapter_revision": base.HF_REVISION,
            "adapter_commit": base.HF_COMMIT,
            "merge_method": MERGE_METHOD,
            "cpu_cores": MERGE_CPU_CORES,
            "memory_gib": MERGE_MEMORY_GIB,
        },
        reinit=True,
    )
    if run is None or not getattr(run, "id", None):
        raise RuntimeError("W&B online initialization failed before checkpoint merge")
    try:
        expected = {
            "base_commit": base.MODEL_REVISION,
            "adapter_commit": base.HF_COMMIT,
            "merge_method": MERGE_METHOD,
        }
        if MERGED_POINTER_PATH.is_file():
            cached = json.loads(MERGED_POINTER_PATH.read_text(encoding="utf-8"))
            merged_path = Path(str(cached.get("merged_path") or ""))
            if (
                all(cached.get(key) == value for key, value in expected.items())
                and merged_path.is_dir()
                and (merged_path / "config.json").is_file()
                and list(merged_path.glob("*.safetensors"))
            ):
                run.summary.update({"status": "READY", "cache_hit": True})
                run.finish(exit_code=0)
                result = dict(cached)
                result.update(
                    {
                        "status": "READY",
                        "cache_hit": True,
                        "cache_check_wandb_run_id": str(run.id),
                        "cache_check_wandb_url": str(run.url),
                    }
                )
                return result

        api = HfApi(token=os.environ["HF_TOKEN"])
        base_info = api.model_info(base.MODEL_ID, revision=base.MODEL_REVISION)
        adapter_info = api.model_info(base.HF_REPO, revision=base.HF_REVISION)
        if base_info.sha != base.MODEL_REVISION:
            raise RuntimeError(f"base-model revision drift: {base_info.sha}")
        if adapter_info.sha != base.HF_COMMIT:
            raise RuntimeError(f"adapter revision drift: {adapter_info.sha}")
        base_path = snapshot_download(
            base.MODEL_ID,
            revision=base.MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
        )
        adapter_path = snapshot_download(
            base.HF_REPO,
            revision=base.HF_REVISION,
            token=os.environ["HF_TOKEN"],
        )
        base_path = Path(base_path)
        adapter_path = Path(adapter_path)
        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        rank = int(adapter_config["r"])
        alpha = float(adapter_config["lora_alpha"])
        scaling = alpha / rank
        if adapter_config.get("peft_type") != "LORA":
            raise RuntimeError("checkpoint is not a PEFT LoRA adapter")
        if adapter_config.get("bias") != "none":
            raise RuntimeError("adapter bias merge is unsupported")
        if bool(adapter_config.get("use_rslora", False)):
            raise RuntimeError("RS-LoRA scaling is unsupported")

        adapter_tensor_path = adapter_path / "adapter_model.safetensors"
        plans: dict[str, list[dict[str, str | None]]] = defaultdict(list)
        with safe_open(adapter_tensor_path, framework="pt", device="cpu") as handle:
            adapter_keys = set(handle.keys())
        a_suffix = ".lora_A.weight"
        b_suffix = ".lora_B.weight"
        a_keys = sorted(key for key in adapter_keys if key.endswith(a_suffix))
        if len(a_keys) * 2 != len(adapter_keys):
            raise RuntimeError("adapter contains non-LoRA or unpaired tensors")
        for a_key in a_keys:
            module = a_key[: -len(a_suffix)]
            b_key = module + b_suffix
            if b_key not in adapter_keys:
                raise RuntimeError(f"missing LoRA B tensor for {module}")
            target, strategy, slot = _adapter_module_plan(module)
            plans[target].append(
                {
                    "module": module,
                    "a_key": a_key,
                    "b_key": b_key,
                    "strategy": strategy,
                    "slot": slot,
                }
            )

        index_path = base_path / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = dict(index["weight_map"])
        missing_targets = sorted(set(plans) - set(weight_map))
        if missing_targets:
            raise RuntimeError(
                f"adapter targets are absent from pinned base: {missing_targets[:10]}"
            )
        merged_path = Path(
            f"/cache/e1-qwen36-seed809-merged-{uuid.uuid4().hex[:12]}"
        )
        merged_path.mkdir(parents=True)
        for source in base_path.iterdir():
            if source.is_file() and not source.name.endswith(".safetensors"):
                shutil.copy2(source, merged_path / source.name)

        targets_by_file: dict[str, list[str]] = defaultdict(list)
        for target in plans:
            targets_by_file[weight_map[target]].append(target)

        consumed_modules: set[str] = set()
        shard_sha256: dict[str, str] = {}
        shard_names = sorted(set(weight_map.values()))
        with safe_open(
            adapter_tensor_path, framework="pt", device="cpu"
        ) as adapter_handle:
            for shard_number, filename in enumerate(shard_names, start=1):
                source_path = base_path / filename
                tensors = load_file(source_path, device="cpu")
                with safe_open(
                    source_path, framework="pt", device="cpu"
                ) as source_handle:
                    metadata = source_handle.metadata()

                def delta_for(item: dict[str, str | None]) -> torch.Tensor:
                    a = adapter_handle.get_tensor(str(item["a_key"]))
                    b = adapter_handle.get_tensor(str(item["b_key"]))
                    delta = torch.matmul(b.float(), a.float()).mul_(scaling)
                    if not bool(torch.isfinite(delta).all()):
                        raise RuntimeError(
                            f"non-finite LoRA delta for {item['module']}"
                        )
                    consumed_modules.add(str(item["module"]))
                    return delta

                for target in sorted(targets_by_file.get(filename, [])):
                    items = plans[target]
                    strategies = {str(item["strategy"]) for item in items}
                    if len(strategies) != 1:
                        raise RuntimeError(f"mixed merge strategies for {target}")
                    strategy = strategies.pop()
                    if strategy == "simple":
                        if len(items) != 1:
                            raise RuntimeError(f"duplicate simple adapter for {target}")
                        delta = delta_for(items[0])
                    elif strategy == "qkv":
                        by_slot = {str(item["slot"]): item for item in items}
                        if set(by_slot) != {"q", "k", "v"}:
                            raise RuntimeError(f"incomplete QKV adapter for {target}")
                        delta = torch.cat(
                            [delta_for(by_slot[slot]) for slot in ("q", "k", "v")],
                            dim=0,
                        )
                    elif strategy == "gate_up":
                        by_slot = {str(item["slot"]): item for item in items}
                        if set(by_slot) != {"w1", "w3"}:
                            raise RuntimeError(f"incomplete expert adapter for {target}")
                        delta = torch.cat(
                            [delta_for(by_slot[slot]) for slot in ("w1", "w3")],
                            dim=1,
                        )
                    else:
                        raise RuntimeError(f"unknown merge strategy {strategy}")
                    base_tensor = tensors[target]
                    if tuple(delta.shape) != tuple(base_tensor.shape):
                        raise RuntimeError(
                            f"delta shape mismatch for {target}: "
                            f"{tuple(delta.shape)} != {tuple(base_tensor.shape)}"
                        )
                    tensors[target] = base_tensor + delta.to(base_tensor.dtype)
                    del delta

                output_path = merged_path / filename
                save_file(tensors, output_path, metadata=metadata)
                digest = hashlib.sha256()
                with output_path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                shard_sha256[filename] = digest.hexdigest()
                del tensors
                gc.collect()
                print(
                    f"merged and hashed shard {shard_number}/{len(shard_names)}"
                )

        expected_modules = {str(item["module"]) for values in plans.values() for item in values}
        if consumed_modules != expected_modules:
            missing = sorted(expected_modules - consumed_modules)
            raise RuntimeError(f"unconsumed adapter modules: {missing[:10]}")
        weight_files = sorted(merged_path.glob("*.safetensors"))
        if len(weight_files) != len(shard_names):
            raise RuntimeError("merged checkpoint shard count mismatch")
        merge_seconds = time.perf_counter() - started
        estimated_usd = _merge_compute_cost(merge_seconds)
        receipt = {
            "schema_version": "pavlov-e1-exact-lora-merge-v1",
            "status": "READY",
            "cache_hit": False,
            "started_at": started_at,
            "finished_at": base._utc_now(),
            "merge_seconds": round(merge_seconds, 6),
            "estimated_modal_cpu_memory_usd": estimated_usd,
            "cpu_cores": MERGE_CPU_CORES,
            "memory_gib": MERGE_MEMORY_GIB,
            "pricing_source": GPU_PRICING_SOURCE,
            "base_model": base.MODEL_ID,
            "base_commit": base_info.sha,
            "adapter_repo": base.HF_REPO,
            "adapter_commit": adapter_info.sha,
            "merge_method": MERGE_METHOD,
            "lora_rank": rank,
            "lora_alpha": alpha,
            "lora_scaling": scaling,
            "adapter_tensor_count": len(adapter_keys),
            "adapter_module_count": len(expected_modules),
            "merged_target_count": len(plans),
            "all_adapter_tensors_consumed": True,
            "merged_path": str(merged_path),
            "weight_file_count": len(weight_files),
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
            "weight_shard_sha256": shard_sha256,
            "wandb_run_id": str(run.id),
            "wandb_url": str(run.url),
        }
        MERGED_POINTER_PATH.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        hf_cache.commit()
        run.log(
            {
                "merge/seconds": merge_seconds,
                "merge/weight_bytes": receipt["weight_bytes"],
                "merge/adapter_module_count": len(expected_modules),
                "merge/merged_target_count": len(plans),
                "cost/estimated_modal_cpu_memory_usd": estimated_usd,
            },
            step=1,
        )
        run.summary.update(
            {
                "status": "READY",
                "cache_hit": False,
                "merged_path": str(merged_path),
            }
        )
        run.finish(exit_code=0)
        return receipt
    except Exception as exc:
        run.summary.update(
            {"status": "INFRA_ERROR", "error_type": type(exc).__name__}
        )
        run.finish(exit_code=1)
        raise


@app.cls(
    image=gpu_image,
    gpu=GPU_TYPE,
    secrets=[secret],
    volumes={"/cache": hf_cache},
    cpu=8.0,
    memory=65_536,
    timeout=60 * 60,
    scaledown_window=5 * 60,
    max_containers=1,
)
class ExactCheckpointGenerator:
    @modal.enter()
    def load(self) -> None:
        import wandb
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        from vllm import LLM

        started_at = base._utc_now()
        started = time.perf_counter()
        run = wandb.init(
            entity="arvindcr4-pes-university",
            project="tinker-rl-lab-pavlov",
            group="e1-swe-bench-pro-full-seed1818",
            job_type="modal-gpu-model-load",
            name="e1_full_gpu_backend_load",
            tags=["e1", "swe_bench_pro", "modal", "gpu", "preflight"],
            mode="online",
            config={
                "model_id": base.MODEL_ID,
                "model_revision": base.MODEL_REVISION,
                "adapter_repo": base.HF_REPO,
                "adapter_revision": base.HF_REVISION,
                "adapter_commit": base.HF_COMMIT,
                "gpu_type": GPU_TYPE,
                "gpu_count": 1,
                "vllm_version": VLLM_VERSION,
                "max_model_len": MAX_MODEL_LEN,
                "generation_backend": GENERATION_BACKEND,
            },
            reinit=True,
        )
        if run is None or not getattr(run, "id", None):
            raise RuntimeError("W&B online initialization failed before GPU load")
        try:
            api = HfApi(token=os.environ["HF_TOKEN"])
            base_info = api.model_info(
                base.MODEL_ID, revision=base.MODEL_REVISION
            )
            adapter_info = api.model_info(
                base.HF_REPO, revision=base.HF_REVISION
            )
            if base_info.sha != base.MODEL_REVISION:
                raise RuntimeError(f"base-model revision drift: {base_info.sha}")
            if adapter_info.sha != base.HF_COMMIT:
                raise RuntimeError(f"adapter revision drift: {adapter_info.sha}")

            if not MERGED_POINTER_PATH.is_file():
                raise RuntimeError("exact merged-checkpoint pointer is missing")
            merge_receipt = json.loads(
                MERGED_POINTER_PATH.read_text(encoding="utf-8")
            )
            if merge_receipt.get("base_commit") != base.MODEL_REVISION:
                raise RuntimeError("merged checkpoint base revision drift")
            if merge_receipt.get("adapter_commit") != base.HF_COMMIT:
                raise RuntimeError("merged checkpoint adapter revision drift")
            if merge_receipt.get("merge_method") != MERGE_METHOD:
                raise RuntimeError("merged checkpoint method drift")
            if merge_receipt.get("all_adapter_tensors_consumed") is not True:
                raise RuntimeError("merged checkpoint has unconsumed adapter tensors")
            merged_path = Path(str(merge_receipt.get("merged_path") or ""))
            if not merged_path.is_dir() or not list(
                merged_path.glob("*.safetensors")
            ):
                raise RuntimeError("exact merged checkpoint is incomplete")
            self.tokenizer = AutoTokenizer.from_pretrained(
                merged_path, local_files_only=True
            )
            self.llm = LLM(
                model=str(merged_path),
                tokenizer=str(merged_path),
                dtype="bfloat16",
                max_model_len=MAX_MODEL_LEN,
                max_num_seqs=1,
                gpu_memory_utilization=0.98,
                enforce_eager=True,
                language_model_only=True,
                trust_remote_code=False,
            )
            gpu_name = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            load_seconds = time.perf_counter() - started
            estimated_usd = load_seconds * GPU_RATE_USD_PER_SECOND
            run.log(
                {
                    "gpu/load_seconds": load_seconds,
                    "cost/estimated_modal_gpu_usd": estimated_usd,
                },
                step=1,
            )
            run.summary.update(
                {
                    "status": "READY",
                    "gpu_name": gpu_name,
                    "base_commit": base_info.sha,
                    "adapter_commit": adapter_info.sha,
                }
            )
            self.load_receipt = {
                "schema_version": "pavlov-e1-modal-gpu-backend-v1",
                "status": "READY",
                "started_at": started_at,
                "finished_at": base._utc_now(),
                "load_seconds": round(load_seconds, 6),
                "estimated_modal_gpu_usd": round(estimated_usd, 8),
                "gpu_type_requested": GPU_TYPE,
                "gpu_runtime": gpu_name,
                "gpu_count": 1,
                "generation_backend": GENERATION_BACKEND,
                "vllm_version": VLLM_VERSION,
                "max_model_len": MAX_MODEL_LEN,
                "base_model": base.MODEL_ID,
                "base_commit": base_info.sha,
                "adapter_repo": base.HF_REPO,
                "adapter_commit": adapter_info.sha,
                "merged_checkpoint": str(merged_path),
                "merge_method": merge_receipt["merge_method"],
                "wandb_run_id": str(run.id),
                "wandb_url": str(run.url),
                "pricing_source": GPU_PRICING_SOURCE,
                "gpu_rate_usd_per_second": GPU_RATE_USD_PER_SECOND,
            }
            run.finish(exit_code=0)
        except Exception as exc:
            run.summary.update(
                {"status": "INFRA_ERROR", "error_type": type(exc).__name__}
            )
            run.finish(exit_code=1)
            raise

    def _render_prompt(
        self, task: dict[str, Any], sources: dict[str, str]
    ) -> tuple[str, str]:
        prompt = base._build_prompt(task, sources)
        rendered = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic source-code patch generator. "
                        "Reply with the requested patch only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return prompt, rendered

    @modal.method()
    def health(self) -> dict[str, Any]:
        import wandb
        from vllm import SamplingParams

        run = wandb.init(
            entity="arvindcr4-pes-university",
            project="tinker-rl-lab-pavlov",
            group="e1-swe-bench-pro-full-seed1818",
            job_type="modal-gpu-lora-health",
            name="e1_full_gpu_lora_health",
            tags=["e1", "modal", "gpu", "lora", "preflight"],
            mode="online",
            config={
                "generation_backend": GENERATION_BACKEND,
                "adapter_commit": base.HF_COMMIT,
                "max_tokens": 1,
            },
            reinit=True,
        )
        if run is None or not getattr(run, "id", None):
            raise RuntimeError("W&B online initialization failed before GPU health")
        started = time.perf_counter()
        try:
            output = self.llm.generate(
                ["Return the word OK."],
                SamplingParams(max_tokens=1, temperature=0.0),
                use_tqdm=False,
            )[0]
            seconds = time.perf_counter() - started
            result = dict(self.load_receipt)
            result.update(
                {
                    "lora_health": "PASSED",
                    "health_token_count": len(output.outputs[0].token_ids),
                    "health_seconds": round(seconds, 6),
                    "health_estimated_modal_gpu_usd": round(
                        seconds * GPU_RATE_USD_PER_SECOND, 8
                    ),
                    "health_wandb_run_id": str(run.id),
                    "health_wandb_url": str(run.url),
                }
            )
            result["estimated_modal_gpu_usd"] = round(
                float(result["estimated_modal_gpu_usd"])
                + seconds * GPU_RATE_USD_PER_SECOND,
                8,
            )
            run.summary.update({"status": "PASSED"})
            run.finish(exit_code=0)
            return result
        except Exception as exc:
            run.summary.update(
                {"status": "INFRA_ERROR", "error_type": type(exc).__name__}
            )
            run.finish(exit_code=1)
            raise

    @modal.method()
    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        import wandb
        from vllm import SamplingParams

        task = dict(payload["task"])
        sources = dict(payload["sources"])
        seed = int(payload["seed"])
        index = int(payload["index"])
        max_tokens = int(payload["max_tokens"])
        temperature = float(payload["temperature"])
        instance_id = str(task.get("instance_id") or "")
        started_at = base._utc_now()
        run: Any = None
        sample_started = False
        response_text = ""
        try:
            if set(task) != set(base.GENERATION_FIELDS):
                raise RuntimeError("unexpected fields reached GPU generation")
            prompt, rendered = self._render_prompt(task, sources)
            prompt_ids = self.tokenizer.encode(
                rendered, add_special_tokens=False
            )
            if len(prompt_ids) + max_tokens > MAX_MODEL_LEN:
                raise RuntimeError(
                    f"prompt plus output exceeds GPU context: "
                    f"{len(prompt_ids)}+{max_tokens}>{MAX_MODEL_LEN}"
                )
            run = wandb.init(
                entity="arvindcr4-pes-university",
                project="tinker-rl-lab-pavlov",
                group=f"e1-swe-bench-pro-full-seed{seed}",
                job_type="primary-evaluation-exact-suite-modal-gpu",
                name=f"e1_full_gpu_{index:04d}_seed{seed}",
                tags=[
                    "e1",
                    "swe_bench_pro",
                    "exact-suite",
                    "pass@1",
                    "modal",
                    "gpu",
                ],
                mode="online",
                config={
                    "suite_id": "swe_bench_pro_eval",
                    "scope": "exact_731_task_test_split",
                    "task_index": index,
                    "instance_id": instance_id,
                    "dataset_revision": base.DATASET_REVISION,
                    "evaluator_revision": base.EVALUATOR_REVISION,
                    "model_id": base.MODEL_ID,
                    "model_revision": base.MODEL_REVISION,
                    "adapter_repo": base.HF_REPO,
                    "adapter_revision": base.HF_REVISION,
                    "adapter_commit": base.HF_COMMIT,
                    "generation_backend": GENERATION_BACKEND,
                    "vllm_version": VLLM_VERSION,
                    "gpu_type": GPU_TYPE,
                    "samples_per_problem": 1,
                    "sampling_retries": 0,
                    "thinking_enabled": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.95,
                    "seed": seed,
                    "source_sha256": base._sha256_text(
                        base._stable_json(sources)
                    ),
                    "prompt_sha256": base._sha256_text(prompt),
                },
                reinit=True,
            )
            if run is None or not getattr(run, "id", None):
                raise RuntimeError("W&B online initialization failed before GPU sample")

            sampling = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                seed=seed,
            )
            started = time.perf_counter()
            sample_started = True
            output = self.llm.generate(
                [rendered],
                sampling,
                use_tqdm=False,
            )[0]
            seconds = time.perf_counter() - started
            tokens = list(output.outputs[0].token_ids)
            response_text = self.tokenizer.decode(
                tokens, skip_special_tokens=True
            )
            patch, validation_reason = base._extract_diff(response_text)
            status = "GENERATED" if patch else "GENERATION_FAILED"
            estimated_usd = seconds * GPU_RATE_USD_PER_SECOND
            run.log(
                {
                    "generation/prompt_tokens": len(prompt_ids),
                    "generation/response_tokens": len(tokens),
                    "generation/patch_structurally_valid": int(bool(patch)),
                    "gpu/generation_seconds": seconds,
                    "cost/estimated_modal_gpu_usd": estimated_usd,
                },
                step=1,
            )
            run.summary.update(
                {
                    "status": status,
                    "candidate_patch_sha256": (
                        base._sha256_text(patch) if patch else None
                    ),
                    "patch_validation_reason": validation_reason,
                }
            )
            run_id = str(run.id)
            run_url = str(run.url)
            run.finish(exit_code=0 if patch else 1)
            run = None
            return {
                "instance_id": instance_id,
                "index": index,
                "status": status,
                "phase": "complete",
                "sample_started": True,
                "sample_completed": True,
                "patch": patch,
                "patch_sha256": base._sha256_text(patch) if patch else None,
                "patch_validation_reason": validation_reason,
                "response_text": response_text,
                "response_sha256": base._sha256_text(response_text),
                "prompt_sha256": base._sha256_text(prompt),
                "source_sha256": base._sha256_text(
                    base._stable_json(sources)
                ),
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(tokens),
                "estimated_tinker_usd": 0.0,
                "estimated_modal_gpu_usd": round(estimated_usd, 8),
                "modal_gpu_seconds": round(seconds, 6),
                "generation_backend": GENERATION_BACKEND,
                "gpu_type": GPU_TYPE,
                "vllm_version": VLLM_VERSION,
                "hf_commit": base.HF_COMMIT,
                "wandb_run_id": run_id,
                "wandb_url": run_url,
                "started_at": started_at,
                "finished_at": base._utc_now(),
            }
        except Exception as exc:
            run_id = str(getattr(run, "id", "") or "")
            run_url = str(getattr(run, "url", "") or "")
            if run is not None:
                try:
                    run.summary.update(
                        {
                            "status": "INFRA_ERROR",
                            "error_type": type(exc).__name__,
                        }
                    )
                    run.finish(exit_code=1)
                except Exception:
                    pass
            return {
                "instance_id": instance_id,
                "index": index,
                "status": "INFRA_ERROR",
                "phase": "sampling" if sample_started else "pre_sampling",
                "sample_started": sample_started,
                "sample_completed": False,
                "response_text": response_text,
                "response_sha256": (
                    base._sha256_text(response_text) if response_text else None
                ),
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
                "generation_backend": GENERATION_BACKEND,
                "gpu_type": GPU_TYPE,
                "wandb_run_id": run_id or None,
                "wandb_url": run_url or None,
                "started_at": started_at,
                "finished_at": base._utc_now(),
            }


def _validate_existing_run(
    root: Path, run_dir: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    raw_path = run_dir / "dataset_test_731.jsonl"
    sanitized_path = run_dir / "generation_manifest_731.jsonl"
    rows, tasks = base._validate_existing_dataset(raw_path, sanitized_path)
    image_manifest = json.loads(
        (run_dir / "image_manifest.json").read_text(encoding="utf-8")
    )
    if image_manifest.get("count") != base.EXPECTED_TASK_COUNT:
        raise RuntimeError("saved image manifest is incomplete")
    evaluator_dir = root / "outputs/e1_swe_bench_pro/evaluator"
    evaluator_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=evaluator_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if evaluator_commit != base.EVALUATOR_REVISION:
        raise RuntimeError(f"evaluator revision drift: {evaluator_commit}")
    return rows, tasks, image_manifest


def _save_generation_result(result: dict[str, Any], tasks_dir: Path) -> None:
    instance_id = str(result["instance_id"])
    task_dir = tasks_dir / instance_id
    task_dir.mkdir(parents=True, exist_ok=True)
    response_text = str(result.pop("response_text", "") or "")
    (task_dir / "generation_response.txt").write_text(
        response_text, encoding="utf-8"
    )
    base._write_json(task_dir / "generation.json", result)


def _evaluate_completed_run(
    *,
    root: Path,
    run_dir: Path,
    state_path: Path,
    rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    image_manifest: dict[str, Any],
    gpu_backend_receipts: list[dict[str, Any]],
    seed: int,
    max_tokens: int,
    temperature: float,
    evaluator_workers: int,
) -> dict[str, Any]:
    candidates_path = run_dir / "candidates.json"
    candidates, generations = base._build_candidates(
        tasks, run_dir / "tasks", candidates_path
    )
    if len(generations) != base.EXPECTED_TASK_COUNT:
        raise RuntimeError("evaluation requires 731 terminal generation receipts")
    base._write_json(
        state_path,
        {
            "status": "EVALUATING",
            "updated_at": base._utc_now(),
            "terminal_generations": len(generations),
            "valid_candidates": len(candidates),
            "generation_backend": GENERATION_BACKEND,
        },
    )
    raw_path = run_dir / "dataset_test_731.jsonl"
    image_manifest_path = run_dir / "image_manifest.json"
    base._run_evaluator(
        root,
        raw_path,
        candidates_path,
        image_manifest_path,
        run_dir / "evaluation",
        workers=max(1, evaluator_workers),
    )
    receipt = base._aggregate_receipt(
        root=root,
        run_dir=run_dir,
        rows=rows,
        tasks=tasks,
        generations=generations,
        candidates=candidates,
        image_manifest=image_manifest,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    receipt = _augment_receipt(receipt, generations, gpu_backend_receipts)
    base._write_json(run_dir / "receipt.json", receipt)
    base._write_json(
        state_path,
        {
            "status": "SCORED",
            "updated_at": base._utc_now(),
            "score": receipt["score"],
            "receipt": str((run_dir / "receipt.json").relative_to(root)),
        },
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "scope": receipt["scope"],
                "score": receipt["score"],
                "score_percent": receipt["score_percent"],
                "coverage": receipt["coverage"],
                "cost": receipt["cost"],
                "receipt": str(run_dir / "receipt.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return receipt


@app.local_entrypoint()
def main(
    seed: int = 1818,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    evaluator_workers: int = 20,
    index_min: int = 0,
    index_max: int = 730,
    generate_only: bool = False,
    evaluation_only: bool = False,
    lane: str = "main",
) -> None:
    if not (0 <= index_min <= index_max < base.EXPECTED_TASK_COUNT):
        raise RuntimeError(
            f"invalid task-index range {index_min}..{index_max}"
        )
    if not lane or len(lane) > 32 or not all(
        char.isalnum() or char in "-_" for char in lane
    ):
        raise RuntimeError("lane must be a short alphanumeric slug")
    if lane != APP_LANE:
        raise RuntimeError(
            f"lane argument {lane!r} must match E1_MODAL_LANE={APP_LANE!r}"
        )
    if evaluation_only and (generate_only or lane != "main"):
        raise RuntimeError(
            "evaluation-only requires the main lane without --generate-only"
        )
    if (index_min, index_max) != (0, base.EXPECTED_TASK_COUNT - 1):
        if not generate_only:
            raise RuntimeError("a partial index range requires --generate-only")
        artifact_suffix = f"_{lane}"
    else:
        artifact_suffix = "" if lane == "main" else f"_{lane}"

    root = base._repo_root()
    run_dir = (
        root
        / f"outputs/modal_e1_e14/{base.RUN_DATE}/"
        f"e1_swe_bench_pro_full/seed{seed}"
    )
    if not run_dir.is_dir():
        raise RuntimeError(f"cannot GPU-resume missing exact-suite run: {run_dir}")
    tasks_dir = run_dir / "tasks"
    state_path = run_dir / f"run_state{artifact_suffix}.json"
    rows, tasks, image_manifest = _validate_existing_run(root, run_dir)
    base._recover_interrupted_attempts(tasks, tasks_dir)
    all_pending, _ = base._generation_inputs(
        tasks,
        tasks_dir,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    pending = [
        payload
        for payload in all_pending
        if index_min <= int(payload["index"]) <= index_max
    ]
    if evaluation_only:
        if all_pending:
            raise RuntimeError(
                "evaluation-only requires 731 terminal generation receipts; "
                f"{len(all_pending)} remain"
            )
        _evaluate_completed_run(
            root=root,
            run_dir=run_dir,
            state_path=state_path,
            rows=rows,
            tasks=tasks,
            image_manifest=image_manifest,
            gpu_backend_receipts=_load_gpu_backend_receipts(root, run_dir),
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
            evaluator_workers=evaluator_workers,
        )
        return
    projection = _gpu_projection(len(pending))
    if projection > MAX_GPU_PROJECTED_USD:
        raise RuntimeError(
            f"GPU projection ${projection:.4f} exceeds "
            f"${MAX_GPU_PROJECTED_USD:.2f}"
        )
    preflight = {
        "schema_version": "pavlov-e1-modal-gpu-resume-preflight-v1",
        "status": "READY",
        "recorded_at": base._utc_now(),
        "task_count": base.EXPECTED_TASK_COUNT,
        "terminal_generations_preserved": (
            base.EXPECTED_TASK_COUNT - len(all_pending)
        ),
        "global_pending_generations_at_start": len(all_pending),
        "pending_gpu_generations": len(pending),
        "lane": lane,
        "index_min": index_min,
        "index_max": index_max,
        "generate_only": generate_only,
        "terminal_task_resampling_performed": False,
        "generation_backend": GENERATION_BACKEND,
        "gpu_type": GPU_TYPE,
        "gpu_count": 1,
        "vllm_version": VLLM_VERSION,
        "base_model": base.MODEL_ID,
        "base_commit": base.MODEL_REVISION,
        "adapter_repo": base.HF_REPO,
        "adapter_commit": base.HF_COMMIT,
        "projected_modal_gpu_usd": projection,
        "modal_gpu_cap_usd": MAX_GPU_PROJECTED_USD,
        "pricing_source": GPU_PRICING_SOURCE,
    }
    preflight["receipt_sha256"] = base._sha256_text(
        base._stable_json(preflight)
    )
    base._write_json(
        run_dir / f"gpu_resume_preflight{artifact_suffix}.json", preflight
    )
    base._write_json(
        state_path,
        {
            "status": "GPU_GENERATING",
            "updated_at": base._utc_now(),
            "pending_generations": len(pending),
            "lane": lane,
            "index_min": index_min,
            "index_max": index_max,
            "generation_backend": GENERATION_BACKEND,
            "gpu_type": GPU_TYPE,
        },
    )
    print(json.dumps(preflight, indent=2, sort_keys=True))

    merge_receipt = merge_checkpoint.remote()
    merge_receipt_path = run_dir / f"gpu_merge_receipt{artifact_suffix}.json"
    local_merge_receipt = dict(merge_receipt)
    local_merge_receipt["receipt_sha256"] = base._sha256_text(
        base._stable_json(local_merge_receipt)
    )
    base._write_json(merge_receipt_path, local_merge_receipt)
    print(json.dumps(local_merge_receipt, indent=2, sort_keys=True))

    generator = ExactCheckpointGenerator()
    backend_receipt = generator.health.remote()
    backend_receipt["merge"] = local_merge_receipt
    backend_receipt_path = run_dir / f"gpu_backend_receipt{artifact_suffix}.json"
    backend_receipt["path"] = str(backend_receipt_path.relative_to(root))
    backend_receipt["receipt_sha256"] = base._sha256_text(
        base._stable_json(backend_receipt)
    )
    base._write_json(backend_receipt_path, backend_receipt)
    print(json.dumps(backend_receipt, indent=2, sort_keys=True))

    for completed, payload in enumerate(pending, start=1):
        generation_path = (
            tasks_dir / str(payload["task"]["instance_id"]) / "generation.json"
        )
        if generation_path.exists():
            raise RuntimeError(
                "terminal generation appeared after lane preflight; refusing "
                f"to resample {payload['task']['instance_id']}"
            )
        result = generator.generate.remote(payload)
        _save_generation_result(result, tasks_dir)
        if result["status"] == "INFRA_ERROR":
            base._write_json(
                state_path,
                {
                    "status": "GPU_GENERATION_ERROR",
                    "updated_at": base._utc_now(),
                    "completed_gpu_generations": completed - 1,
                    "failed_instance_id": result["instance_id"],
                    "phase": result["phase"],
                    "sample_started": result["sample_started"],
                },
            )
            raise RuntimeError(
                f"GPU generation failed for {result['instance_id']}: "
                f"{result.get('error')}"
            )
        if completed % 10 == 0 or completed == len(pending):
            print(f"saved {completed}/{len(pending)} Modal GPU generations")

    if generate_only:
        lane_result = {
            "status": "GPU_GENERATION_RANGE_COMPLETE",
            "updated_at": base._utc_now(),
            "lane": lane,
            "index_min": index_min,
            "index_max": index_max,
            "completed_gpu_generations": len(pending),
            "terminal_task_resampling_performed": False,
        }
        base._write_json(state_path, lane_result)
        print(json.dumps(lane_result, indent=2, sort_keys=True))
        return

    _evaluate_completed_run(
        root=root,
        run_dir=run_dir,
        state_path=state_path,
        rows=rows,
        tasks=tasks,
        image_manifest=image_manifest,
        gpu_backend_receipts=[backend_receipt],
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        evaluator_workers=evaluator_workers,
    )
