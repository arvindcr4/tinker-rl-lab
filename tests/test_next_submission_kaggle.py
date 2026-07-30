from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "zvf-program/next-submission/run_kaggle_preflight.py"
SPEC = importlib.util.spec_from_file_location("next_submission_kaggle", PATH)
assert SPEC is not None and SPEC.loader is not None
KAGGLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = KAGGLE
SPEC.loader.exec_module(KAGGLE)


def embedded_payload(script: str) -> dict[str, str]:
    match = re.search(r"FILES = json\.loads\((.+)\)\n", script)
    assert match is not None
    encoded_json = json.loads(match.group(1))
    return json.loads(encoded_json)


def test_kernel_script_embeds_exact_sources_pins_and_secret_lookup():
    script = KAGGLE.build_kernel_script(["--task", "gsm8k"])
    embedded = embedded_payload(script)
    expected = KAGGLE.embedded_sources()

    assert {name: base64.b64decode(value) for name, value in embedded.items()} == expected
    for requirement in KAGGLE.PACKAGE_PINS:
        assert requirement in script
    assert "UserSecretsClient" in script
    assert 'NEXT_PREFLIGHT_RESULT_PATH"]' in script
    assert 'SCRIPT_ARGS = json.loads("[\\"--task\\", \\"gsm8k\\"]")' in script
    assert not re.search(r"hf_[A-Za-z0-9]{30,}", script)


def test_kernel_metadata_is_private_gpu_and_internet_enabled():
    metadata = KAGGLE.kernel_metadata(owner="owner", slug="slug")

    assert metadata["id"] == "owner/slug"
    assert metadata["code_file"] == "main.py"
    assert metadata["kernel_type"] == "script"
    assert metadata["is_private"] == "true"
    assert metadata["enable_gpu"] == "true"
    assert metadata["enable_internet"] == "true"


def test_accelerator_labels_and_status_parser_are_unambiguous():
    assert KAGGLE.ACCELERATOR_TO_GPU == {
        "NvidiaTeslaA100": "A100",
        "NvidiaL4X1": "L4",
    }
    assert (
        KAGGLE.parse_kernel_stage('kernel has status "KernelWorkerStatus.COMPLETE"') == "COMPLETE"
    )
    assert KAGGLE.parse_kernel_stage("unexpected") == "UNKNOWN"
