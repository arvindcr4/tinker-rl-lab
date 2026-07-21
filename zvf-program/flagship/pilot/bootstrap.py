from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path


SOURCE_BUNDLE = Path("/content/flagship-pilot-source.tar.gz")
SOURCE_ROOT = Path("/content/tinker-rl-lab")
SECRET_PATH = Path("/content/.flagship-pilot-secrets.json")
REQUEST_PATH = Path("/content/flagship-pilot-request.json")


class BootstrapError(RuntimeError):
    """The staged source, request, or credential bundle is unsafe or incomplete."""


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise BootstrapError(f"source archive escapes destination: {member.name}")
            if member.issym() or member.islnk():
                raise BootstrapError(f"source archive contains a link: {member.name}")
        bundle.extractall(destination, filter="data")


def load_secrets(path: Path) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if set(payload) != {"HF_TOKEN", "WANDB_API_KEY"}:
            raise BootstrapError("secret payload fields are incomplete or unexpected")
        for key, value in payload.items():
            if not isinstance(value, str) or not value:
                raise BootstrapError(f"secret value is empty: {key}")
            os.environ[key] = value
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    # The frozen protocol enables torch deterministic algorithms; every CUDA
    # GEMM then requires this cuBLAS workspace configuration to exist before
    # the first cuBLAS handle is created inside the remote kernel.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not SOURCE_BUNDLE.is_file() or not REQUEST_PATH.is_file():
        raise BootstrapError("source bundle or request is missing")
    safe_extract(SOURCE_BUNDLE, Path("/content"))
    load_secrets(SECRET_PATH)
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    if set(request) != {"argv", "source_archive_sha256", "source_bindings_sha256"}:
        raise BootstrapError("request fields are incomplete or unexpected")
    actual_archive_sha = hashlib.sha256(SOURCE_BUNDLE.read_bytes()).hexdigest()
    if actual_archive_sha != request["source_archive_sha256"]:
        raise BootstrapError("source archive SHA-256 does not match request")
    argv = request["argv"]
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        raise BootstrapError("request argv must be a string list")
    sys.path.insert(0, str(SOURCE_ROOT / "zvf-program/flagship"))
    from pilot.protocol import build_screening_plan, load_protocol

    protocol = load_protocol()
    plan = build_screening_plan(protocol, next(protocol.screening_units()))
    if plan["protocol"]["source_bundle_sha256"] != request["source_bindings_sha256"]:
        raise BootstrapError("extracted source bindings do not match request")
    from pilot.remote_unit import main as remote_main

    return remote_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
