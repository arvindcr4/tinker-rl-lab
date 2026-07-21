from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence


REQUEST_PATH = Path("/content/flagship-pilot-install.json")
PIN = re.compile(r"([A-Za-z0-9_.-]+)==([^=\s]+)")
ABI_COUPLED_PACKAGES = (
    "torchvision",
    "torchao",
    "torchaudio",
    "torchtext",
    "fastai",
    "torchdata",
)


class RuntimeInstallError(RuntimeError):
    """The remote install request or resulting runtime violates the contract."""


def load_install_request(path: Path = REQUEST_PATH) -> tuple[str, ...]:
    if not path.is_file():
        raise RuntimeInstallError(f"runtime install request is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"package_pins"}:
        raise RuntimeInstallError("runtime install request fields are incomplete or unexpected")
    pins = payload["package_pins"]
    if not isinstance(pins, list) or not pins:
        raise RuntimeInstallError("runtime package pins must be a non-empty list")
    if not all(isinstance(pin, str) and PIN.fullmatch(pin) for pin in pins):
        raise RuntimeInstallError("every runtime package must use one exact name==version pin")
    names = [PIN.fullmatch(pin).group(1).lower() for pin in pins]
    if len(names) != len(set(names)):
        raise RuntimeInstallError("runtime package pins contain duplicate names")
    return tuple(pins)


def install_runtime(
    pins: Sequence[str],
    *,
    runner: Callable[..., Any] = subprocess.run,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    spec_finder: Callable[[str], Any] = importlib.util.find_spec,
) -> dict[str, str]:
    command = [
        "uv",
        "pip",
        "install",
        "--system",
        "--reinstall-package",
        "numpy",
        *pins,
    ]
    completed = runner(command, check=False)
    if completed.returncode != 0:
        fallback = [sys.executable, "-m", "pip", "install", *pins]
        runner(fallback, check=True)
    observed: dict[str, str] = {}
    for pin in pins:
        match = PIN.fullmatch(pin)
        assert match is not None
        name, expected = match.groups()
        actual = version_getter(name)
        if actual != expected:
            raise RuntimeInstallError(
                f"runtime version mismatch after install for {name}: "
                f"expected {expected}, got {actual}"
            )
        observed[name] = actual
    # Colab base images ship torch-ABI-coupled wheels built for a different
    # torch; importing them raises errors such as `operator torchvision::nms
    # does not exist` or PEFT's `incompatible version of torchao` ImportError
    # and breaks the pinned stack. None of them belongs to the frozen
    # scientific stack, and the reference isolated environment contains none
    # of them, so fail closed if any remain importable after uninstall.
    runner(
        ["uv", "pip", "uninstall", "--system", *ABI_COUPLED_PACKAGES],
        check=False,
    )
    for name in ABI_COUPLED_PACKAGES:
        if spec_finder(name) is not None:
            raise RuntimeInstallError(
                f"{name} remains importable after uninstall; the remote "
                "environment does not match the pinned scientific stack"
            )
    return observed


def main() -> int:
    pins = load_install_request()
    observed = install_runtime(pins)
    print(
        "FPILOT_INSTALL_RESULT "
        + json.dumps({"status": "install_pass", "runtime_versions": observed}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
