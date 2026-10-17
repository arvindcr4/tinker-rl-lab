"""Observed runtime provenance. Unknown values stay null; no historical backfill."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .samestack_protocol import canonical_hash, file_sha256


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def capture_source(repo_root=None):
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain").decode()
        diff = git("diff", "--binary", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "dirty_hash": None,
                "source_files": {}, "status": "unavailable"}
    paths = [Path(__file__), Path(__file__).with_name("samestack_protocol.py"),
             Path(__file__).with_name("samestack_training.py"),
             Path(__file__).with_name("samestack_contract.py"),
             Path(__file__).with_name("samestack_dependencies.py"),
             Path(__file__).parent / "modal/modal_samestack_gsm8k_cot_v2.py",
             root / "platform_colab/samestack_v2/colab_run_arms.py",
             root / "platform_colab/samestack_v2/drive_sessions.py"]
    files = {str(path.relative_to(root)): file_sha256(path) for path in paths if path.is_file()}
    return {"commit": commit, "dirty": bool(status),
            "dirty_hash": hashlib.sha256(diff + canonical_hash(files).encode()).hexdigest(),
            "dirty_hash_scope": "tracked HEAD diff plus execution source file hashes",
            "source_files": files, "status": "observed"}


def verify_source_files(source, repo_root):
    """Verify the execution payload matches provenance before loading weights."""
    root = Path(repo_root)
    required = [f"platform_hybrid/experiments/{name}.py" for name in
                ("samestack_manifest", "samestack_protocol", "samestack_training",
                 "samestack_contract", "samestack_dependencies")]
    required.append("platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot_v2.py")
    for relative in required:
        path = root / relative
        if not path.is_file() or source.get("source_files", {}).get(relative) != file_sha256(path):
            raise ValueError(f"Source payload hash mismatch or missing file: {relative}")


def resolve_revisions(model_id, dataset_id, model_revision="main", tokenizer_revision=None,
                      dataset_revision="main"):
    from huggingface_hub import HfApi
    api = HfApi()
    model_sha = api.model_info(model_id, revision=model_revision).sha
    tokenizer_sha = api.model_info(model_id, revision=tokenizer_revision or model_sha).sha
    dataset_sha = api.dataset_info(dataset_id, revision=dataset_revision).sha
    import re
    if not all(isinstance(s, str) and re.fullmatch(r"[0-9a-f]{40}", s)
               for s in (model_sha, tokenizer_sha, dataset_sha)):
        raise ValueError("Model, tokenizer and dataset must resolve to immutable commit SHA revisions")
    return {"model": {"id": model_id, "revision": model_sha},
            "tokenizer": {"id": model_id, "revision": tokenizer_sha},
            "dataset": {"id": dataset_id, "config": "main", "revision": dataset_sha}}


def hardware_manifest(torch, model, model_dtype=None):
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        driver = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "driver_versions": driver, "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "attention_backend": getattr(model.config, "_attn_implementation", None),
            "model_dtype": model_dtype or str(next(model.parameters()).dtype), "logprob_dtype": "torch.float32",
            "value_head_dtype": "torch.float32", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def example_ids(dataset, split):
    return [f"{split}:{index}:{canonical_hash({'question': row['question'], 'answer': row['answer']})}"
            for index, row in enumerate(dataset)]
