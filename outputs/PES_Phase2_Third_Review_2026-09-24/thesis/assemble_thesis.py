#!/usr/bin/env python3
"""
Assemble the full M.Tech thesis report from its chapter files.

Front matter lives here. Chapters 1-10 and the appendices come from the
ch*.md files written alongside this script.

Outputs:
  Thesis_Report_ArvindCR.md    -- canonical markdown
  Thesis_Report_ArvindCR.docx  -- via pandoc, if available
  Thesis_Report_ArvindCR.html  -- via pandoc, if available
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "Thesis_Report_ArvindCR"
MD_OUT = os.path.join(HERE, BASE + ".md")
DOCX_OUT = os.path.join(HERE, BASE + ".docx")
HTML_OUT = os.path.join(HERE, BASE + ".html")

TITLE = ("Tinker RL Lab — A Multi-Framework Benchmark and Study of "
         "GRPO-Style Reinforcement-Learning Post-Training of Large Language Models")

FRONT_MATTER = f"""---
title: "{TITLE}"
author: "Arvind C R"
date: "24 September 2026"
---

# {TITLE}

**M.Tech Project — Phase 2 Report**

| | |
|---|---|
| **Candidate** | Arvind C R |
| **SRN** | PES2PGE24DS140 |
| **Programme** | M.Tech Data Science & Artificial Intelligence |
| **Institution** | PES University |
| **Project guide** | Ramesh Prakash Guledgudd |
| **Date** | 24 September 2026 |

---

## Declaration

I declare that this report is my own work. The multi-framework benchmark
infrastructure and the initial GRPO baseline experiments were produced by the
Semester 3 project group, of which I was a member. The research direction,
experimental expansion, statistical analyses, the P1–P8 paper series, and the
E1–E14 held-out evaluation campaign reported here are my individual
contribution, carried out under the guidance of Ramesh Prakash Guledgudd. All
sources of data and prior work are cited. Where results are incomplete,
blocked, or negative, they are reported as such rather than omitted.

The boundary between inherited and individual work is stated precisely in
`platform_hybrid/sem 4 work/PROVENANCE.md`, and is summarised in Chapter 1
§1.6 and Appendix B.

---

## Abstract

This report presents a multi-framework benchmark for group-relative
reinforcement-learning post-training of large language models, together with a
study of that post-training's behaviour across libraries, scales, group sizes
and datasets, and a held-out evaluation of a single trained actor against
fourteen agentic and reasoning benchmark suites.

The central methodological commitment is attribution: every comparative claim
is made with the entire training stack held fixed except the one factor under
test. Four de-confound pillars — a matched same-stack PPO/GRPO contrast, the
Zero-Variance Fraction as a signal-starvation diagnostic, a group-size sweep,
and a length-bias and held-out-generalisation study — are reported under that
commitment, alongside a cross-library and cross-scale scaling study, a minimum
reporting standard, a machine-readable stack registry, an adaptive group-size
controller derived from the ZVF diagnostic, and an applied credit-card fraud
case study.

The principal results are deliberately conservative. Across 70+ runs spanning
roughly 2.4 orders of magnitude in model scale, no reliable scaling law is
recovered from reward curves. A matched same-stack comparison of PPO and GRPO
finds the two estimators statistically indistinguishable (Welch p = 0.7605). A
single-seed group-size sweep does not establish an optimum, and trainability is
non-monotone in group size. Four of eleven GRPO runs peak before 65% of
training and then decay, consistent with length bias rather than capability
gain.

The E1–E14 campaign evaluates one Tinker-trained actor
(`Qwen/Qwen3.6-35B-A3B` with the `pavlov-portfolio-qwen36-seed809-stepfinal`
adapter) using each suite's own native evaluator at a pinned revision. Four
suites carry strictly complete graded scopes: LAB-Bench public (1,967/1,967),
AgentDojo benign utility (97/97), VerilogEval (312/312, 129/312 = 41.35%
pass@1), and Omni-MATH (all 4,428 dispositions recorded, official accuracy
2,271/4,428 = 51.31% reproduced). Two carry full verified scores under their
original contracts: SWE-bench Pro (2/731 = 0.274% pass@1, with all eighteen
non-native outcomes retained in the denominator) and VerilogEval. The remaining
lanes are reported as partial, quota-blocked, or externally blocked, each with
an explicit terminal state and a named reopen condition.

Original-contract and replacement-scope benchmark results are never pooled, no
cross-suite aggregate is computed, and no claim of improvement over a baseline
is made anywhere in this work. Structural claims were re-verified by
deterministic code checks on 19 September 2026 (11/11 PASS).

---

## Acknowledgements

I thank my project guide, Ramesh Prakash Guledgudd, for direction throughout
this work, and the Semester 3 project group for the shared benchmark
infrastructure on which the Semester 4 studies build. Compute for the
Semester 4 experiments was provided through the Tinker platform, Modal, and
Google Cloud Platform.

---

## Contents

**Part I — Foundations**

1. Introduction
2. Literature Survey
3. System Requirements Specification

**Part II — Design and Implementation**

4. Proposed Methodology
5. Implementation Details

**Part III — Results**

6. Results I: Core GRPO Studies (P1–P4)
7. Results II: Reporting, Registry and Controller (P5–P7)
8. Results III: Applied Study (P8)
9. Results IV: The E1–E14 Held-Out Benchmark Campaign

**Part IV — Synthesis**

10. Cross-Study Synthesis, Threats to Validity, Conclusions and Future Work

**Appendices**

- Appendix A. Complete Run Registry
- Appendix B. Reproducibility and Provenance
- Appendix C. Notation, Symbols and Abbreviations

---
"""

RUNNING_FOOT = """
---

*End of report.*
"""

CHAPTER_FILES = [
    ("ch01_introduction.md", "Part I — Foundations"),
    ("ch02_literature.md", None),
    ("ch03_requirements.md", None),
    ("ch04_methodology.md", "Part II — Design and Implementation"),
    ("ch05_implementation.md", None),
    ("ch06_results_core.md", "Part III — Results"),
    ("ch07_results_infra.md", None),
    ("ch08_results_fraud.md", None),
    ("ch09_results_campaign.md", None),
    ("ch10_synthesis_conclusions.md", "Part IV — Synthesis"),
    ("ch_back_run_registry.md", "Appendices"),
    ("ch_back_reproducibility.md", None),
    ("ch_back_notation.md", None),
]

PART_HEADING = {
    "Part I — Foundations": "# Part I — Foundations",
    "Part II — Design and Implementation": "# Part II — Design and Implementation",
    "Part III — Results": "# Part III — Results",
    "Part IV — Synthesis": "# Part IV — Synthesis",
    "Appendices": "# Appendices",
}


def main() -> int:
    parts = [FRONT_MATTER]
    missing = []
    present = []

    for name, part in CHAPTER_FILES:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        if part:
            parts.append("\n\n" + PART_HEADING[part] + "\n\n---\n")
        with open(path, encoding="utf-8") as fh:
            body = fh.read().strip()
        parts.append("\n\n" + body + "\n\n---\n")
        present.append(name)
    parts.append(RUNNING_FOOT)

    md = "".join(parts)
    with open(MD_OUT, "w", encoding="utf-8") as fh:
        fh.write(md)

    words = len(md.split())
    print(f"wrote {MD_OUT}")
    print(f"  chapters present: {len(present)}/{len(CHAPTER_FILES)}")
    print(f"  {words:,} words  (~{words/500:.0f} printed pages at 500 w/page)")
    if missing:
        print("  MISSING: " + ", ".join(missing), file=sys.stderr)

    if shutil.which("pandoc"):
        common = ["--toc", "--toc-depth=2", "--standalone",
                  "--metadata", f"title={TITLE}",
                  "--metadata", "author=Arvind C R"]
        for out, extra in ((DOCX_OUT, []), (HTML_OUT, ["--embed-resources", "--css", "/dev/null"])):
            try:
                subprocess.run(["pandoc", MD_OUT, "-o", out] + common + extra,
                               check=True, capture_output=True, timeout=180)
                print(f"wrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                err = getattr(exc, "stderr", b"") or b""
                print(f"pandoc failed for {os.path.basename(out)}: "
                      f"{err.decode()[:300]}", file=sys.stderr)
    else:
        print("pandoc not found — markdown only", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
