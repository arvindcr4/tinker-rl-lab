"""Packaging / shippability sanity checks."""

import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

import zvf_triage

_PKG_DIR = Path(zvf_triage.__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent  # src/zvf_triage -> src -> repo root


def test_py_typed_marker_present():
    assert (_PKG_DIR / "py.typed").is_file(), "PEP 561 py.typed marker missing"


def test_version_is_exposed():
    assert zvf_triage.__version__ == "0.1.0"


def test_public_api_exports_resolve():
    for name in zvf_triage.__all__:
        assert hasattr(zvf_triage, name), f"__all__ lists missing symbol {name!r}"


def test_quickstart_example_runs(capsys):
    """The example must execute end-to-end and reach its auto-stop assertions."""
    example = _REPO_ROOT / "examples" / "quickstart.py"
    if not example.is_file():
        pytest.skip("examples/quickstart.py not found in this layout")
    argv = sys.argv
    sys.argv = ["quickstart.py"]  # default args -> deterministic seed=0 run
    try:
        runpy.run_path(str(example), run_name="__main__")
    except SystemExit as exc:  # main() returns 0 -> SystemExit(0)
        assert exc.code == 0
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "Auto-stop fired" in out
    assert "Demo complete" in out
