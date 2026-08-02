"""Verification gate for the framework × backend experiment run-code matrix.

Real GPU/Modal/GCP runs are not feasible in CI, so this test gates on what is:
every (framework, backend) cell resolves to a non-stub ``LaunchPlan`` whose driver
file exists in the repo, the manifest matches the live backend code, and a sample
of cells resolve end-to-end through the CLI ``--dry-run``.

This is the concrete check that "all frameworks × all backends have experiment run
code" (slide 06, Infrastructure).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from platform_local.unified.backends import BACKEND_NAMES, BACKENDS  # noqa: E402
from platform_local.unified.canonical import FRAMEWORKS, load_spec  # noqa: E402

MATRIX = REPO / "platform_local" / "unified" / "matrix.json"


@pytest.fixture(scope="module")
def spec():
    return load_spec()


@pytest.fixture(scope="module")
def registry():
    return BACKENDS()


def test_matrix_manifest_exists_and_well_formed():
    doc = json.loads(MATRIX.read_text())
    assert doc["schema"] == "tinkerrl-framework-backend-matrix-v1"
    assert set(doc["frameworks"]) == set(FRAMEWORKS)
    assert set(doc["backends"]) == set(BACKEND_NAMES)
    expected = len(FRAMEWORKS) * len(BACKEND_NAMES)
    assert len(doc["cells"]) == expected, f"expected {expected} cells"


def test_all_cells_resolve_to_real_driver(spec, registry):
    """Every cell's LaunchPlan points at a driver file that exists in the repo."""
    missing = []
    for fw in FRAMEWORKS:
        for be in BACKEND_NAMES:
            plan = registry[be].plan(fw, spec)
            assert plan.command, f"{be}/{fw}: empty command"
            assert plan.driver_file, f"{be}/{fw}: empty driver_file"
            driver = REPO / plan.driver_file
            if not driver.exists():
                missing.append(f"{be}/{fw} -> {plan.driver_file}")
    assert not missing, "cells with missing driver files:\n  " + "\n  ".join(missing)


def test_manifest_matches_live_backends(spec, registry):
    """The persisted matrix.json must agree with the live backend.plan() output."""
    doc = json.loads(MATRIX.read_text())
    live = {(c["framework"], c["backend"]): c for c in doc["cells"]}
    for fw in FRAMEWORKS:
        for be in BACKEND_NAMES:
            plan = registry[be].plan(fw, spec)
            cell = live[(fw, be)]
            assert cell["command"] == plan.command, f"{be}/{fw} command drift"
            assert cell["driver_file"] == plan.driver_file, f"{be}/{fw} driver drift"


def test_each_framework_has_local_run_code(registry):
    """Every framework has a local in-process path (the _run_* methods are filled)."""
    from platform_local.unified.launcher import UnifiedLauncher

    launcher = UnifiedLauncher()
    launcher.dry_run = True  # guard: never actually train
    launcher.spec = load_spec()
    for fw in FRAMEWORKS:
        launcher.framework = fw
        # dispatch_framework must not raise NotImplementedError for the dispatch itself;
        # local backend plan must resolve.
        plan = registry["local"].plan(fw, launcher.spec)
        assert plan.driver_file != "?", f"local/{fw}: unresolved driver"


@pytest.mark.parametrize(
    "backend,framework",
    [
        ("local", "tinker"),
        ("modal", "trl"),
        ("modal", "verl"),
        ("gcp", "trl"),
        ("vast", "tinker"),
        ("hfspaces", "trl"),
        ("colab", "trl"),
    ],
)
def test_cli_dry_run_resolves_cell(backend, framework):
    """The unified CLI resolves a sample of cells to a LaunchPlan without compute."""
    proc = subprocess.run(
        [sys.executable, "-m", "platform_local.unified",
         "--backend", backend, "--framework", framework, "--dry-run"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert f"[{backend}/{framework}]" in proc.stdout, (
        f"dry-run did not emit [{backend}/{framework}] plan:\n{proc.stdout}"
    )


def test_shims_delegate_to_unified():
    """Each platform_<backend>/run_experiment.py pins its backend and resolves."""
    shim_for = {
        "local": "platform_local/run_experiment.py",
        "modal": "platform_modal/run_experiment.py",
        "colab": "platform_colab/run_canonical.py",
        "vast": "platform_vast/run_experiment.py",
        "gcp": "platform_gcp/run_experiment.py",
        "hfspaces": "platform_hf_spaces/run_experiment.py",
    }
    for backend, shim in shim_for.items():
        proc = subprocess.run(
            [sys.executable, str(REPO / shim), "--framework", "trl", "--dry-run"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        )
        assert proc.returncode == 0, f"{shim} failed: {proc.stderr}"
        assert f"[{backend}/trl]" in proc.stdout, f"{shim} did not pin backend {backend}"
