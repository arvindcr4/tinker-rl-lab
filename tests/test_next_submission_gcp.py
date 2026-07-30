from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/run_gcp_preflight.py"
SPEC = importlib.util.spec_from_file_location("next_submission_gcp", PATH)
assert SPEC is not None and SPEC.loader is not None
GCP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GCP
SPEC.loader.exec_module(GCP)


def embedded_payload(script: str) -> dict[str, str]:
    match = re.search(r"FILES = json\.loads\((.+)\)\n", script)
    assert match is not None
    encoded_json = json.loads(match.group(1))
    return json.loads(encoded_json)


def test_entry_embeds_exact_sources_and_only_existing_secret_references():
    script = GCP.build_entry_script(
        script_args=["--task", "gsm8k"],
        project=GCP.PROJECT,
        secret_names=GCP.SECRET_NAMES,
    )
    embedded = embedded_payload(script)

    assert {
        name: base64.b64decode(value) for name, value in embedded.items()
    } == GCP.embedded_sources()
    assert "metadata.google.internal" in script
    assert "secretmanager.googleapis.com" in script
    assert GCP.SECRET_NAMES == {
        "HF_TOKEN": "hf-token",
        "WANDB_API_KEY": "wandb-api-key",
    }
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", script)


def test_startup_script_has_frozen_packages_serial_receipt_and_shutdown():
    script = GCP.build_startup_script("print('entry')\n")

    for requirement in GCP.PACKAGE_PINS:
        assert requirement in script
    assert "/dev/ttyS0" in script
    assert "NEXT_PREFLIGHT_EXIT_CODE" in script
    assert "sleep 10" in script
    assert "shutdown -h now" in script
    assert "set -x" not in script


def test_create_command_is_spot_bounded_and_uses_exact_temporary_instance(tmp_path):
    startup = tmp_path / "startup.sh"
    command = GCP.create_command(
        gcloud="gcloud",
        instance="exact-temp-instance",
        project=GCP.PROJECT,
        zone=GCP.ZONE,
        service_account=GCP.SERVICE_ACCOUNT,
        max_run_duration=GCP.DEFAULT_MAX_RUN_DURATION,
        startup_script=startup,
    )

    assert command[:5] == [
        "gcloud",
        "compute",
        "instances",
        "create",
        "exact-temp-instance",
    ]
    assert "--provisioning-model=SPOT" in command
    assert "--instance-termination-action=STOP" in command
    assert "--max-run-duration=90m" in command
    assert "--machine-type=a2-highgpu-1g" in command
    assert f"--image={GCP.IMAGE}" in command
    assert f"--service-account={GCP.SERVICE_ACCOUNT}" in command


def test_cost_and_hardware_contract_remain_bounded():
    assert GCP.GPU == "A100"
    assert GCP.HARDWARE_FLAVOR == "a2-highgpu-1g-spot"
    assert GCP.SPOT_HOURLY_USD * GCP.DEFAULT_MAX_RUN_HOURS < 3.0
    assert GCP.ESTIMATED_COST_CAP_USD == 3.0


def test_live_parser_requires_wait_for_cleanup():
    args = GCP.parse_args(["--task", "gsm8k", "--arm", "contrast_early_stop_g2_to_g8"])

    assert args.dry_run is False
    assert args.wait is False
    with pytest.raises(SystemExit, match="live GCP runs require --wait"):
        GCP.validate_args(args)


def test_serial_receipt_retries_until_final_marker(monkeypatch):
    responses = iter(
        [
            (1, "resource is not ready"),
            (0, "boot output without final marker"),
            (0, "NEXT_PREFLIGHT_EXIT_CODE=1"),
        ]
    )
    monkeypatch.setattr(GCP, "serial_output", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(GCP.time, "sleep", lambda seconds: None)

    result = GCP.serial_output_with_retry(
        "gcloud",
        instance="exact-temp-instance",
        project=GCP.PROJECT,
        zone=GCP.ZONE,
        attempts=3,
        delay_seconds=0,
    )

    assert result == (0, "NEXT_PREFLIGHT_EXIT_CODE=1")
