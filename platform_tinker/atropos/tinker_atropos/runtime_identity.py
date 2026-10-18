"""Runtime identity checks: tokenizer metadata is not a backend weight revision."""

import hashlib
import importlib.metadata
import json


def verify_tokenizer(backend, selected):
    if backend.get_vocab() != selected.get_vocab():
        raise ValueError("Selected tokenizer vocabulary does not match backend tokenizer")
    if getattr(backend, "chat_template", None) != getattr(selected, "chat_template", None):
        raise ValueError("Selected chat template does not match backend tokenizer")


def runtime_manifest(model, tokenizer, sampling=None):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    versions = {}
    for package in ("tinker", "transformers", "torch", "pydantic", "fastapi"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "version": "runtime-v2",
        "backend_model": model,
        "backend_weight_revision": None,
        "backend_weight_revision_status": "unverified",
        "vocabulary_sha256": digest(tokenizer.get_vocab()),
        "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None)),
        "dependencies": versions,
        "sampling": sampling,
        "loss": "importance_sampling",
        "normalization": "group_centered_or_teacher_minus_behavior_v2",
    }
