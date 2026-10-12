#!/usr/bin/env python3
"""Run the fixed CPU developer workflow and publish a fresh local evidence receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_cpu_online_learning_smoke.py",
    "tests/test_cpu_gspo_multiepoch.py",
    "tests/test_cpu_critic_resume_smoke.py",
    "tests/test_cpu_heldout_workflow.py",
)
EXPECTED_TESTS = 5
DEPENDENCIES = ("pytest", "torch", "numpy", "transformers", "tinker")
SOURCES = (
    "platform_tinker/tinkerrl/grpo.py",
    "platform_tinker/tinkerrl/grpo_cli.py",
    "platform_tinker/tinkerrl/critic.py",
    "tests/_shared_fakes.py",
    "tests/test_grpo_coverage.py",
    "tests/test_cpu_causal_lm_smoke.py",
    "tests/test_tinker_sdk_loss_integration.py",
    "tools/run_offline_harness.py",
    *TESTS,
)


def _junit_counts(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("JUnit evidence must not contain document/entity declarations")
    # Owned, locally generated pytest output; DTD/entity declarations rejected.
    cases = ET.fromstring(data).findall(".//testcase")  # noqa: S314
    return {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="Fresh local JSON receipt path (never overwritten)"
    )
    args = parser.parse_args(argv)
    output = args.out.absolute()
    if os.path.lexists(output):
        parser.error("--out already exists; choose a fresh path to preserve previous evidence")
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    if missing:
        parser.error(
            "missing dependencies: "
            + ", ".join(missing)
            + "; from the repository root install the locked dev environment with "
            "`uv sync --frozen --extra dev`, then run `.venv/bin/python tools/run_offline_harness.py --out NEW_PATH`"
        )
    fingerprints = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
    }
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    environment = dict(os.environ)
    environment.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="offline",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    # Tests supply offline doubles and forbid socket.connect. No real credentials
    # are needed; do not forward known provider/tracking credentials to the child.
    for name in (
        "TINKER_API_KEY",
        "WANDB_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "OPENAI_API_KEY",
    ):
        environment.pop(name, None)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".offline-harness-", dir=output.parent) as temporary:
        directory = Path(temporary)
        probe = directory / "publication-probe"
        probe.write_bytes(b"{}")
        os.link(probe, directory / "publication-link")
        junit = directory / "checks.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            f"--junitxml={junit}",
            *TESTS,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            returncode = completed.returncode
            diagnostic = (completed.stdout + completed.stderr)[-6000:]
        except subprocess.TimeoutExpired:
            returncode = 124
            diagnostic = "Offline CPU workflow exceeded 300 seconds; inspect the environment before retrying."
        counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        try:
            counts = _junit_counts(junit)
        except (OSError, ET.ParseError, ValueError):
            diagnostic += "\nMissing or invalid test evidence; the workflow cannot pass."
        unchanged = all(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in fingerprints.items()
        )
        passed = (
            returncode == 0
            and counts == {"tests": EXPECTED_TESTS, "failures": 0, "errors": 0, "skipped": 0}
            and unchanged
        )
        receipt = {
            "schema_version": 1,
            "status": "passed" if passed else "failed",
            "scope": "Fixed random tiny CPU models with in-memory provider/persistence doubles; no pretrained weights, LoRA, live provider, canonical-backend equivalence or benchmark-quality claim.",
            "checks": [
                "CLI config inspection and execution",
                "sampled learning",
                "fresh actor/Adam resume",
                "independent multi-epoch GSPO objective",
                "critic resume",
                "final-policy held-out failure denominator",
                "completed-cache replay",
            ],
            "test_files": list(TESTS),
            "expected_tests": EXPECTED_TESTS,
            "counts": counts,
            "returncode": returncode,
            "sources_unchanged": unchanged,
            "source_sha256": fingerprints,
            "versions": versions,
            "python": sys.version.split()[0],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "diagnostic_tail": diagnostic,
        }
        staged = directory / "receipt.json"
        with staged.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(staged, output)  # Atomic, same-filesystem, no clobbering fallback.
    print(
        f"OFFLINE HARNESS: {'PASS' if passed else 'FAIL'} ({counts['tests']} checks; {counts['skipped']} skipped)"
    )
    print(f"Receipt: {output}")
    if not passed:
        print(diagnostic, file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, importlib.metadata.PackageNotFoundError) as exc:
        print(
            f"Offline harness could not complete: {exc}. Choose a writable fresh output path and verify the locked dev environment.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
