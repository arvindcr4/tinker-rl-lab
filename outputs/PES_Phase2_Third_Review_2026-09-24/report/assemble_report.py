#!/usr/bin/env python3
"""
Assemble the Phase 2 Third Review written report from its section files.

Front matter and the appendices live here; sections 1-5 come from
section_*.md written alongside this file.

Outputs:
  Phase2_Third_Review_Report_ArvindCR.md    -- canonical markdown
  Phase2_Third_Review_Report_ArvindCR.docx  -- via pandoc, if available
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "Phase2_Third_Review_Report_ArvindCR"
MD_OUT = os.path.join(HERE, BASE + ".md")
DOCX_OUT = os.path.join(HERE, BASE + ".docx")

FRONT_MATTER = """---
title: "Tinker RL Lab: E1–E14 Held-Out Benchmark Evaluation of a Tinker-Trained Qwen3.6-35B-A3B Actor"
author: "Arvind C R"
date: "24 September 2026"
---

# Phase 2 Third Review — Written Report

**Programme:** M.Tech Data Science and Machine Learning, PES University
**Student:** Arvind C R  ·  SRN PES2PGE24DS140
**Project guide:** Ramesh Prakash Guledgudd
**Review:** Phase 2, Third Review
**Date:** 24 September 2026

---

## Abstract

This report documents a held-out evaluation campaign conducted across fourteen
agentic and reasoning benchmark suites, using a single Tinker-trained actor:
`Qwen/Qwen3.6-35B-A3B` with the `pavlov-portfolio-qwen36-seed809-stepfinal`
adapter, in bfloat16. Each suite was graded by its own native evaluator at a
pinned revision, under a receipt model in which every reported figure traces to
a surviving evidence file.

Four lanes carry strictly complete graded scopes: E8 (LAB-Bench public split,
1,967/1,967 questions across eight categories), E10 (AgentDojo benign-utility
split, 97/97 episodes), E11 (VerilogEval, 312/312 tasks, 129/312 = 41.35%
pass@1), and E14 (Omni-MATH, all 4,428 dispositions recorded, official accuracy
2271/4428 = 51.31% reproduced). Two lanes carry full verified scores under their
original contracts: E1 (SWE-bench Pro, 2/731 = 0.274% pass@1, with all eighteen
non-native outcomes retained in the denominator) and E11.

The remaining lanes are reported as partial, quantity-blocked, or externally
blocked, each with an explicit terminal state and a named reopen condition.
Original-contract and replacement-scope results are never pooled, no
cross-suite aggregate is computed or claimed, and no claim of improvement over
a baseline is made anywhere in this work. All structural claims were
re-verified by deterministic code checks on 19 September 2026 (11/11 PASS).

---

## Contents

1. Introduction and scope
2. Method and evaluated system
3. Results
4. Discussion
5. Verification, integrity and limitations
6. Conclusions
   - Appendix A — Terminal state of every lane
   - Appendix B — Evidence index

---
"""

CONCLUSIONS = """## 6. Conclusions

The E1–E14 campaign is complete as an evidence artifact. All fourteen lanes hold
a recorded terminal state, each bound to a surviving receipt, and each
adjudicated by a deterministic check that a reviewer can re-run without access
to the original compute environment.

The work establishes four strictly complete graded scopes and two full verified
original-contract scores for a single actor, under sampling conditions that
were preserved rather than tuned per suite. It establishes, with equal clarity,
that seven suites could not be scored, and it names the specific gate blocking
each one — a credential, a quota, or a third party's refusal to grant access.
Those gates are matters of authorisation and access rather than of experimental
design, and closing every lane that is not externally blocked is estimated at
roughly $108 of already-authorised budget.

The methodological contribution is the reporting discipline. Distinguishing
coverage from execution from score, refusing to substitute an adjacent
benchmark under an original benchmark's name, and retaining failed generations
inside the denominator rather than excluding them all cost the campaign
visually. They are what make its numbers defensible under questioning, and they
are the part of this work that would transfer most directly to another
evaluation effort.

---

## Appendix A — Terminal state of every lane

| Lane | Suite | Verified result | Coverage | Terminal state |
|---|---|---|---|---|
| E1 | SWE-bench Pro | 2/731 = 0.274% pass@1 | 713/731 native evals (97.54%) | REBUILD_READY_LAUNCH_PENDING |
| E2 | FrontierSWE | 1/17 tasks; replay normalised 0.8628 | 1/17 | AMENDMENT_ACCEPTED_LAUNCH_PENDING |
| E3 | SDAB (private) | no score | 0 | CLOSED_EXTERNAL |
| E4 | BankerToolBench | 1/100 tasks; recovery 0.3115 | 1/100 | CLOSED_PARTIAL |
| E5 | APEX-Agents | 7/480 native-scored; prefix mean 0.050505 | 7/480 | REBUILD_READY_LAUNCH_PENDING |
| E6 | WebArena | no score | 0/812 graded | PENDING_QUOTA |
| E7 | BinaryAudit | no grade; verifier reward 0.0 | 1/46 attempted | CLOSED_EXTERNAL |
| E8 | LAB-Bench (public) | 8 categories complete | 1967/1967 | COMPLETE |
| E9 | MLE-bench | suite score null | 40/75 (53.33%) | PENDING_QUOTA |
| E10 | AgentDojo (benign) | benign-utility scope complete | 97/97 | COMPLETE |
| E11 | VerilogEval | 129/312 = 41.35% pass@1 | 312/312 | COMPLETE |
| E12 | AppBench | no score | 0 | CLOSED_EXTERNAL |
| E13 | BALROG | — | 13/255 | AMENDMENT_ACCEPTED_LAUNCH_PENDING |
| E14 | Omni-MATH | 2271/4428 = 51.31% official accuracy | 4428/4428 | COMPLETE |

Original-contract results and replacement-scope results are listed in separate
sections of this report and are never combined into a single figure.

---

## Appendix B — Evidence index

| Artifact | Role |
|---|---|
| `outputs/E1_E14_FINAL_RESULTS_2026-09-19.md` | Consolidated per-lane results ledger |
| `outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json` | Deterministic integrity check — 11/11 PASS |
| `outputs/UNBLOCK_CARRYOUT_2026-09-21.md` | Most recent per-lane gate status |
| `outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md` | Terminal-state ledger of record |
| `outputs/PES_Phase2_Review_2026-09-12/finish/DECISION_PACKAGE_2026-09-19.md` | Governing decision package (D1–D6) |
| `outputs/PES_Phase2_Review_2026-09-12/finish/external_closures_2026-09-19/` | Terminal close-out records for externally blocked lanes |
| `outputs/PES_Phase2_Review_2026-09-12/finish/authorization_unlimited_v1.json` | Budget authority (unlimited mode, per-run bounds retained) |
| `zvf-program/flagship/` | Surviving native runners for the replacement lanes |
| `submission/demo/` | Self-contained offline demo artifact and defence runbook |

All paths are relative to the project repository root.
"""

SECTION_FILES = [
    "section_1_intro.md",
    "section_2_method.md",
    "section_3_results.md",
    "section_4_discussion.md",
    "section_5_integrity.md",
]


def main() -> int:
    parts = [FRONT_MATTER]
    missing = []
    for name in SECTION_FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path, encoding="utf-8") as fh:
            body = fh.read().strip()
        parts.append("\n" + body + "\n\n---\n")
    parts.append("\n" + CONCLUSIONS)

    if missing:
        print("MISSING SECTIONS: " + ", ".join(missing), file=sys.stderr)

    md = "\n".join(parts)
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write(md)
    words = len(md.split())
    print(f"wrote {MD_OUT}  ({words} words, {len(md)} chars)")

    if shutil.which("pandoc"):
        try:
            subprocess.run(
                ["pandoc", MD_OUT, "-o", DOCX_OUT, "--toc", "--toc-depth=2",
                 "-V", "geometry:margin=1in", "--standalone"],
                check=True, capture_output=True, timeout=120,
            )
            print(f"wrote {DOCX_OUT}  ({os.path.getsize(DOCX_OUT)/1024:.0f} KB)")
        except subprocess.CalledProcessError as exc:
            print("pandoc failed: " + exc.stderr.decode()[:400], file=sys.stderr)
    else:
        print("pandoc not found — markdown only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
