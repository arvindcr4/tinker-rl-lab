#!/usr/bin/env python3
"""Install the exact Colab preflight stack with a bounded long-running exec."""

from importlib import metadata
import json
import subprocess
import sys


PACKAGE_PINS = (
    "trl==1.8.0",
    "transformers==5.13.1",
    "jinja2==3.1.6",
    "datasets==4.8.5",
    "peft==0.19.1",
    "torchao==0.17.0",
    "wandb==0.28.0",
    "huggingface_hub==1.26.0",
)
REMOVED_UNUSED_OPTIONAL_PACKAGES = ("torchaudio", "torchvision")


subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *PACKAGE_PINS,
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "--yes",
        *REMOVED_UNUSED_OPTIONAL_PACKAGES,
    ],
    check=True,
)

observed = {
    name: metadata.version(name)
    for name in (
        "trl",
        "transformers",
        "jinja2",
        "datasets",
        "peft",
        "torchao",
        "wandb",
        "huggingface_hub",
    )
}
expected = dict(pin.split("==", 1) for pin in PACKAGE_PINS)
if observed != expected:
    raise RuntimeError(
        "installed package mismatch: "
        + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
    )

print(
    "NEXT_PREFLIGHT_SETUP_OK "
    + json.dumps(
        {
            "versions": observed,
            "removed_unused_optional_packages": list(REMOVED_UNUSED_OPTIONAL_PACKAGES),
        },
        sort_keys=True,
    ),
    flush=True,
)
