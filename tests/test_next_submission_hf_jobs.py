from __future__ import annotations

import base64
from pathlib import Path

from tests._shared_fakes import embedded_payload, load_module


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/run_hf_jobs_preflight.py"
HF_JOBS = load_module("next_submission_hf_jobs", PATH)


def test_job_script_embeds_exact_executable_sources_without_credentials():
    script = HF_JOBS.build_job_script()
    embedded = embedded_payload(script, r"FILES = json\.loads\((.+)\)\nsource_root")
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
