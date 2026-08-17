#!/usr/bin/env python3
"""Validate or launch the E2/E4/E7 Harbor jobs through the Tinker bridge.

The default action is configuration-only and cannot sample. Execution requires
an explicit acknowledgement and an authorized total matching the bridge's
persistent server-side cap.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_BASE_URL = (
    "https://arvindcr4--pavlov-tinker-openai-bridge-"
    "tinkeropenaibridge-web.modal.run"
)
BRIDGE_API_BASE = f"{BRIDGE_BASE_URL}/v1"
MODEL_ALIAS = "pavlov-qwen36-tinker"
HF_COMMIT = "64444133c55d88c3f1bf0df8a2f5d7ac646125c8"
HARBOR_EXT_ROOT = REPO_ROOT / "outputs/e2_frontier_swe/frontier-swe"


@dataclass(frozen=True)
class Lane:
    checkout: Path
    config: Path
    harbor_command: tuple[str, ...] = ("harbor",)
    env_defaults: tuple[tuple[str, str], ...] = ()


LANES = {
    "E2": Lane(
        checkout=REPO_ROOT / "outputs/e2_frontier_swe/frontier-swe",
        config=Path("tasks/revideo-perf-opt/job-tinker-bridge.yaml"),
    ),
    "E4": Lane(
        checkout=REPO_ROOT / "outputs/e4_banker_toolbench/official_repo_ff6db552",
        config=Path("job-tinker-bridge.yaml"),
        env_defaults=(("GEMINI_API_KEY", "modal-secret-injected"),),
    ),
    "E7": Lane(
        checkout=REPO_ROOT / "outputs/e7_binaryaudit/BinaryAudit",
        config=Path("configs/tinker-bridge-dnsmasq-docker.yaml"),
    ),
}


class LaunchGateError(RuntimeError):
    """Raised when a launch would violate a provenance or spending gate."""


def load_dotenv(path: Path, env: dict[str, str]) -> None:
    """Load missing values from a simple dotenv file without printing secrets."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in env:
            env[key] = value.strip().strip('"').strip("'")


def resolve_bridge_api_key(env: dict[str, str]) -> str:
    """Read the bridge-only credential from the environment or macOS Keychain."""
    if api_key := env.get("TINKER_BRIDGE_API_KEY"):
        return api_key
    account_result = subprocess.run(
        ["id", "-un"],
        check=False,
        capture_output=True,
        text=True,
    )
    account = account_result.stdout.strip()
    if account_result.returncode != 0 or not account:
        raise LaunchGateError("could not resolve the Keychain account")
    key_result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            account,
            "-s",
            "pavlov-tinker-openai-bridge",
            "-w",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    api_key = key_result.stdout.strip()
    if key_result.returncode != 0 or not api_key:
        raise LaunchGateError(
            "TINKER_BRIDGE_API_KEY is absent and the bridge Keychain item is unavailable"
        )
    return api_key


def fetch_health(api_key: str, *, timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{BRIDGE_BASE_URL}/health",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise LaunchGateError(f"bridge health returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise LaunchGateError(f"bridge health is unreachable: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise LaunchGateError("bridge health returned a non-object payload")
    return payload


def validate_health(payload: dict[str, Any]) -> None:
    if payload.get("status") != "READY":
        raise LaunchGateError("bridge status is not READY")
    if payload.get("model") != MODEL_ALIAS:
        raise LaunchGateError("bridge model alias drifted")
    if payload.get("hf_commit") != HF_COMMIT:
        raise LaunchGateError("bridge immutable HF commit drifted")
    if payload.get("evidence_class") != "infrastructure_not_model_score":
        raise LaunchGateError("bridge evidence boundary is missing")
    wandb_url = payload.get("wandb_url")
    if not isinstance(wandb_url, str) or not wandb_url.startswith("https://wandb.ai/"):
        raise LaunchGateError("bridge has no online W&B receipt URL")


def _money(value: Any, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise LaunchGateError(f"bridge {name} is not a decimal amount") from exc
    if result < 0:
        raise LaunchGateError(f"bridge {name} is negative")
    return result


def validate_execute_budget(payload: dict[str, Any], authorized_total: Decimal) -> None:
    budget = payload.get("budget")
    if not isinstance(budget, dict):
        raise LaunchGateError("bridge budget receipt is missing")
    maximum = _money(budget.get("maximum_usd"), "maximum_usd")
    charged = _money(budget.get("charged_usd"), "charged_usd")
    reserved = _money(budget.get("reserved_usd"), "reserved_usd")
    if authorized_total <= 0:
        raise LaunchGateError("authorized total must be positive")
    if maximum.quantize(Decimal("0.01")) != authorized_total.quantize(Decimal("0.01")):
        raise LaunchGateError(
            "authorized total must exactly match the persistent bridge maximum"
        )
    if charged + reserved >= maximum:
        raise LaunchGateError("bridge has no remaining authorized budget")


def build_harbor_env(api_key: str, lane: Lane) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = BRIDGE_API_BASE
    for key, value in lane.env_defaults:
        env.setdefault(key, value)
    python_paths = [str(lane.checkout), str(HARBOR_EXT_ROOT)]
    if existing_pythonpath := env.get("PYTHONPATH"):
        python_paths.extend(existing_pythonpath.split(os.pathsep))
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_paths))
    return env


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--lane", choices=sorted(LANES), required=True)
    result.add_argument(
        "--execute",
        action="store_true",
        help="run the Harbor job; without this flag only --print-config is evaluated",
    )
    result.add_argument(
        "--acknowledge-paid-run",
        action="store_true",
        help="required with --execute",
    )
    result.add_argument(
        "--authorized-total-usd",
        type=Decimal,
        help="required with --execute and must match the server-side total cap",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args, harbor_args = parser().parse_known_args(argv)
    env = os.environ.copy()
    load_dotenv(REPO_ROOT / ".env", env)
    api_key = resolve_bridge_api_key(env)

    lane = LANES[args.lane]
    config_path = lane.checkout / lane.config
    if not config_path.is_file():
        raise LaunchGateError(f"lane config is absent: {config_path}")

    health = fetch_health(api_key)
    validate_health(health)
    command = [*lane.harbor_command, "run", "-c", str(lane.config)]
    if args.execute:
        if not args.acknowledge_paid_run or args.authorized_total_usd is None:
            raise LaunchGateError(
                "--execute requires --acknowledge-paid-run and --authorized-total-usd"
            )
        validate_execute_budget(health, args.authorized_total_usd)
        command.extend(harbor_args)
    else:
        if args.acknowledge_paid_run or args.authorized_total_usd is not None:
            raise LaunchGateError("paid-run flags are only valid with --execute")
        command.append("--print-config")

    completed = subprocess.run(
        command,
        cwd=lane.checkout,
        env=build_harbor_env(api_key, lane),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaunchGateError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
