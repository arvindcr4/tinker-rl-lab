"""Tests for the repo-wide stale-verdict gate (tools/check_stale_verdicts.py)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "tools" / "check_stale_verdicts.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_stale_verdicts", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("check_stale_verdicts", module)
    spec.loader.exec_module(module)
    return module


def test_gate_passes_on_current_tree():
    gate = _load_gate()
    assert gate.main() == 0


def test_unmarked_occurrence_is_flagged(tmp_path):
    gate = _load_gate()
    target = tmp_path / "notes.md"
    target.write_text(
        "Status update\n\nThe frozen aggregate says DAPO `DISAPPEARS` today.\n",
        encoding="utf-8",
    )
    problems = gate.find_unmarked_occurrences(target, "notes.md")
    assert len(problems) == 1
    assert "notes.md:3" in problems[0]


def test_marked_occurrence_passes(tmp_path):
    gate = _load_gate()
    target = tmp_path / "notes.md"
    target.write_text(
        "Status update\n\nThe former DAPO `DISAPPEARS` verdict is superseded.\n",
        encoding="utf-8",
    )
    assert gate.find_unmarked_occurrences(target, "notes.md") == []


def test_banner_covers_dated_historical_entries(tmp_path):
    gate = _load_gate()
    lines = ["> Correction banner: DAPO entries below are superseded.", ""]
    lines.extend(f" filler line {index}" for index in range(80))
    lines.append("2026-07-20 entry: aggregate emitted DAPO `DISAPPEARS`.")
    target = tmp_path / "status.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert gate.find_unmarked_occurrences(target, "status.md") == []


def test_allowlist_matching():
    gate = _load_gate()
    assert gate.is_allowlisted("zvf-program/audit/aggregate_audit.py")
    assert gate.is_allowlisted("archive/anything.md")
    assert not gate.is_allowlisted("outputs/new_deck.py")
