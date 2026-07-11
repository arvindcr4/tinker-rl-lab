#!/usr/bin/env python3
"""Compatibility entry point for the canonical Wave 6 sensitivity figure."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from platform_hybrid.paper.figure_module import render_legacy_figure


if __name__ == "__main__":
    render_legacy_figure("wave6", Path(__file__).resolve().parent)
