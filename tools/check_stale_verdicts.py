#!/usr/bin/env python3
"""Repo-wide gate against quoting the superseded DAPO ``DISAPPEARS`` verdict.

On 2026-08-02 the E1 analysis was corrected (exact noncentral paired-t MDE
0.0101159 > 0.01 margin; the preregistered Benjamini-Hochberg step had never
run). All four arms are INCONCLUSIVE. Per
``zvf-program/audit/STATISTICAL_REANALYSIS.md``, any text that still presents
the DAPO ``DISAPPEARS`` verdict as current "is stale and must not be used in
a manuscript, review response, talk, or abstract."

This script scans tracked text files and fails when a ``DISAPPEARS``
occurrence is not clearly marked as superseded. An occurrence is acceptable
when any of the following holds:

1. the file is explicitly allowlisted below (rule definitions, the correction
   documents themselves, synthetic test fixtures, or dated quarantined run
   directories covered by a SUPERSEDED note);
2. a quarantine marker (e.g. "superseded", "stale", "former") appears within
   ``WINDOW`` lines of the occurrence;
3. the file's first ``BANNER_LINES`` lines carry a correction banner pointing
   at the reanalysis (covers dated historical entries in status logs).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEEDLE = "DISAPPEARS"
MARKERS = (
    "superseded",
    "stale",
    "quarantin",
    "former",
    "corrected",
    "correction",
    "must not",
    "invalidat",
    "no longer",
    "do not quote",
    "unsafe",
)
WINDOW = 5
BANNER_LINES = 60
BANNER_MARKERS = ("STATISTICAL_REANALYSIS", "superseded")
TEXT_SUFFIXES = {".md", ".tex", ".py", ".json", ".txt", ".yaml", ".yml", ".rst"}

# Files that may mention the verdict without an inline marker, with the
# reason. Keep this list short; prefer inline markers in edited files.
ALLOWLISTED_FILES = {
    "tools/check_stale_verdicts.py",  # this gate
    "tests/test_stale_verdict_gate.py",  # gate tests
    "zvf-program/audit/aggregate_audit.py",  # preregistered rule implementation
    "zvf-program/audit/preregistration.json",  # preregistered rule definition
    "zvf-program/audit/test_aggregate_audit.py",  # synthetic fixtures
    "zvf-program/audit/STATISTICAL_REANALYSIS.md",  # the correction itself
    "autoresearch/deli-neurips-tmlr-260802/audits/statistical_refutation.md",
    "autoresearch/reason-260728-0744/E1_STATISTICAL_REAUDIT.md",
    "autoresearch/reason-260728-0744/handoff.json",
    "autoresearch/reason-260728-0744/lineage.md",
    "autoresearch/reason-260727-2155/WANDB_EVIDENCE_AUDIT.md",
    "autoresearch/improve-260714-1806/SUPERSEDED.md",  # the quarantine note
    "tests/test_next_submission_design.py",  # prior-evidence-boundary fixture
    "zvf-program/flagship/paper/MAY_REVIEW_RESPONSE_AND_ACL_PLAN.md",  # invalidation notice
    "zvf-program/flagship/research/decision_synthesis.md",  # line 209 discusses verdict vocabulary only; result lines corrected 2026-08-02
}

# Directories preserved as dated history; each carries or is covered by a
# superseded/quarantine note at the directory or file level.
ALLOWLISTED_PREFIXES = (
    "archive/",
    "autoresearch/improve-260714-1806/",  # dated 2026-07-14 run dir; see SUPERSEDED.md there
)


def tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8")
        path = ROOT / rel
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            paths.append(path)
    return paths


def is_allowlisted(relative: str) -> bool:
    if relative in ALLOWLISTED_FILES:
        return True
    return any(relative.startswith(prefix) for prefix in ALLOWLISTED_PREFIXES)


def find_unmarked_occurrences(path: Path, relative: str) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    banner = "\n".join(lines[:BANNER_LINES])
    if any(marker in banner for marker in BANNER_MARKERS):
        return []
    problems = []
    for index, line in enumerate(lines):
        if NEEDLE not in line:
            continue
        lo = max(0, index - WINDOW)
        hi = min(len(lines), index + WINDOW + 1)
        context = "\n".join(lines[lo:hi]).lower()
        if any(marker in context for marker in MARKERS):
            continue
        problems.append(f"{relative}:{index + 1}: unmarked stale verdict: {line.strip()[:120]}")
    return problems


def main() -> int:
    problems: list[str] = []
    for path in tracked_text_files():
        relative = path.relative_to(ROOT).as_posix()
        if is_allowlisted(relative):
            continue
        problems.extend(find_unmarked_occurrences(path, relative))
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        print(
            "stale-verdict gate: FAILED — mark superseded DAPO `DISAPPEARS` "
            "occurrences inline (see zvf-program/audit/STATISTICAL_REANALYSIS.md) "
            "or justify an allowlist entry"
        )
        return 1
    print("stale-verdict gate: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
