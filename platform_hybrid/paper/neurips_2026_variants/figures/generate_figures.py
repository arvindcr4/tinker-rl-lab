#!/usr/bin/env python3
"""Compatibility entry point for the canonical paper figure set."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from platform_hybrid.paper.figure_module import render_legacy_figure


if __name__ == "__main__":
    render_legacy_figure("paper", Path(__file__).resolve().parent)
