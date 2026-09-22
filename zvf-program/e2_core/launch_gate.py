#!/usr/bin/env python3
"""Offline E2 launch gate: validates amendment acceptance + surviving harness.

Exit 0 = gates present (still requires fresh IAM + reservation before launch).
Exit 1 = missing prerequisites, with reasons on stderr.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
reasons = []


def need(path: str, what: str) -> None:
    p = REPO / path
    if not p.is_file():
        reasons.append(f"missing {what}: {path}")


need("outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/amendment_acceptance_2026-09-19.json",
     "amendment acceptance record")
need("outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/native-harness/main.py",
     "surviving CORE-Bench harness entry")
need("outputs/PES_Phase2_Review_2026-09-12/finish/e2_completion/native-harness/benchmark/benchmark.py",
     "surviving CORE-Bench benchmark module")

if reasons:
    print("\n".join(reasons), file=sys.stderr)
    sys.exit(1)
print("E2 launch gate: amendment accepted, harness present. IAM + fresh reservation remain launch-time prerequisites.")
