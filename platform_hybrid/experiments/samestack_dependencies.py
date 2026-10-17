"""One exact candidate runtime lock for Modal and Colab.

Version equality is necessary, not a claim of numerical/GPU compatibility. The
lock remains candidate-not-gpu-validated until an explicitly reported GPU test.
Importing this module neither imports training libraries nor installs packages.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json

LOCK_ID = "samestack-v3-runtime-candidate-1"
LOCK = {
    "torch": "2.7.1", "transformers": "5.5.4", "peft": "0.19.1",
    "accelerate": "1.13.0", "datasets": "4.8.4", "huggingface-hub": "1.11.0",
    "numpy": "2.2.6", "safetensors": "0.8.0", "tokenizers": "0.22.2",
}
REQUIREMENTS = tuple(f"{name}=={version}" for name, version in LOCK.items())


def runtime_versions():
    versions = {}
    for name in LOCK:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def verify_runtime(versions=None):
    versions = runtime_versions() if versions is None else dict(versions)
    mismatches = []
    for name, expected in LOCK.items():
        actual = versions.get(name)
        # CUDA/CPU local build suffixes are recorded, not conflated with versions.
        base = actual.split("+", 1)[0] if name == "torch" and isinstance(actual, str) else actual
        if base != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual!r}")
    if mismatches:
        raise ValueError("same-stack runtime lock mismatch: " + "; ".join(mismatches))
    return versions


def lock_manifest():
    digest = hashlib.sha256(json.dumps(LOCK, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"lock_id": LOCK_ID, "status": "candidate-not-gpu-validated", "versions": dict(LOCK), "hash": digest}
