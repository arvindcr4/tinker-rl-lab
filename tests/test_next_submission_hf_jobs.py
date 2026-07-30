from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/run_hf_jobs_preflight.py"
SPEC = importlib.util.spec_from_file_location("next_submission_hf_jobs", PATH)
assert SPEC is not None and SPEC.loader is not None
HF_JOBS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HF_JOBS
SPEC.loader.exec_module(HF_JOBS)


def embedded_payload(script: str) -> dict[str, str]:
    match = re.search(r"FILES = json\.loads\((.+)\)\nsource_root", script)
    assert match is not None
    encoded_json = json.loads(match.group(1))
    return json.loads(encoded_json)


def test_job_script_embeds_exact_executable_sources_without_credentials():
    script = HF_JOBS.build_job_script()
    embedded = embedded_payload(script)
    expected = HF_JOBS.embedded_sources()

    assert set(embedded) == set(expected)
    assert {name: base64.b64decode(value) for name, value in embedded.items()} == expected
    assert "HF_TOKEN" not in script
    assert "WANDB_API_KEY" not in script


def test_job_script_pins_scientific_stack_and_trackio():
    script = HF_JOBS.build_job_script()
    for requirement in HF_JOBS.PACKAGE_PINS:
        assert f'"{requirement}"' in script
    assert 'NEXT_PREFLIGHT_REPORT_TO"] = "wandb,trackio"' in script


def test_supported_flavors_have_unambiguous_observed_gpu_labels():
    assert HF_JOBS.FLAVOR_TO_GPU == {
        "l4x1": "L4",
        "a10g-large": "A10G",
        "a100-large": "A100",
        "h200": "H200",
    }


def test_provider_error_sanitizer_removes_every_submitted_secret():
    credentials = {"HF_TOKEN": "hf_example-secret", "WANDB_API_KEY": "wandb-secret"}
    exc = RuntimeError("failed hf_example-secret and wandb-secret")

    sanitized = HF_JOBS.sanitize_provider_error(exc, credentials)

    assert sanitized == "failed <redacted> and <redacted>"
