#!/usr/bin/env python3
"""Build the deterministic, offline-review artifact for the flagship paper."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parents[2]
BUNDLE = PAPER_DIR / "review_bundle.zip"
OUTER_DIGEST = PAPER_DIR / "REVIEW_BUNDLE.sha256"
ZIP_TIME = (2026, 7, 27, 0, 0, 0)

PAPER_FILES = (
    "main.tex",
    "main.pdf",
    "flagship.bib",
    "CLAIM_AUDIT.md",
    "ADVERSARIAL_REVIEW.md",
    "REVIEW_BUNDLE.md",
    "verify_claims.py",
    "build_review_bundle.py",
)

REPOSITORY_FILES = (
    "pyproject.toml",
    "uv.lock",
    "zvf-program/flagship/pilot_preregistration.json",
    "zvf-program/flagship/pilot/README.md",
    "zvf-program/flagship/pilot/artifacts.py",
    "zvf-program/flagship/pilot/bootstrap.py",
    "zvf-program/flagship/pilot/checkpointing.py",
    "zvf-program/flagship/pilot/evaluation.py",
    "zvf-program/flagship/pilot/flops.py",
    "zvf-program/flagship/pilot/objective.py",
    "zvf-program/flagship/pilot/protocol.py",
    "zvf-program/flagship/pilot/remote_core.py",
    "zvf-program/flagship/pilot/remote_training.py",
    "zvf-program/flagship/pilot/remote_unit.py",
    "zvf-program/flagship/pilot/replay.py",
    "zvf-program/flagship/pilot/runtime_install.py",
    "zvf-program/flagship/pilot/training.py",
)

REPOSITORY_TREES = (
    "zvf-program/flagship/s1",
    "zvf-program/flagship/pilot/provenance",
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/acceptance",
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/results",
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/recovery",
)

CAMPAIGN_FILES = (
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/supervisor_state.json",
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/launch_manifest.json",
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/execution-notes.md",
    "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/logs/corpus__filtered_variable_length__s11.log",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def allowed(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
        and path.name != ".DS_Store"
    )


def collect() -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for name in PAPER_FILES:
        path = PAPER_DIR / name
        if not path.is_file():
            raise SystemExit(f"required paper file is missing: {path}")
        payloads[name] = path.read_bytes()

    for name in (*REPOSITORY_FILES, *CAMPAIGN_FILES):
        path = REPO_ROOT / name
        if not path.is_file():
            raise SystemExit(f"required evidence file is missing: {path}")
        payloads[f"repository/{name}"] = path.read_bytes()

    for name in REPOSITORY_TREES:
        root = REPO_ROOT / name
        if not root.is_dir():
            raise SystemExit(f"required evidence tree is missing: {root}")
        for path in sorted(root.rglob("*")):
            if allowed(path):
                relative = path.relative_to(REPO_ROOT).as_posix()
                payloads[f"repository/{relative}"] = path.read_bytes()
    return payloads


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def main() -> None:
    payloads = collect()
    manifest = "".join(
        f"{sha256_bytes(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("utf-8")

    with zipfile.ZipFile(BUNDLE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads):
            archive.writestr(zip_info(name), payloads[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        archive.writestr(
            zip_info("MANIFEST.sha256"),
            manifest,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )

    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    OUTER_DIGEST.write_text(f"{digest}  {BUNDLE.name}\n", encoding="utf-8")
    print(f"{BUNDLE} ({BUNDLE.stat().st_size} bytes)")
    print(f"sha256 {digest}")
    print(f"payload files {len(payloads)}")


if __name__ == "__main__":
    main()
