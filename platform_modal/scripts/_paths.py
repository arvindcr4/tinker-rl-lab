"""Repository-relative paths for the platform_modal analysis scripts.

Replaces the old hard-coded ``/home/claude/tinker-rl-lab-minimax`` root so the
scripts run from any clone. Importing this module has no side effects: it only
resolves paths and never creates directories.

Usage from a script in ``platform_modal/scripts/`` or one level below::

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from _paths import REPO_ROOT
"""

from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """Walk up from ``start`` to the first directory containing pyproject.toml."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"no pyproject.toml found above {start}")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
HYBRID_ROOT = REPO_ROOT / "platform_hybrid"
RESULTS_DIR = HYBRID_ROOT / "experiments" / "results"
P5P8_RESULTS_DIR = RESULTS_DIR / "p5p8"
REGISTRY_DIR = HYBRID_ROOT / "registry"
PAPER_FIGURES_DIR = HYBRID_ROOT / "paper" / "figures"
