#!/usr/bin/env python3
"""Splice 8 rebuttal fragments directly into sections of capstone_final_report.md.

This edit is NOT an addendum; each fragment's content is woven into an existing
parent section with a new sub-subsection number.
"""
from __future__ import annotations

import re
import shutil
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "platform_tinker/reports/final/capstone_final_report.md"
FRAG_DIR = ROOT / "platform_tinker/reports/final/addendum"
BACKUP = REPORT.with_suffix(".md.bak")


def read_fragment(name: str) -> str:
    return (FRAG_DIR / name).read_text()


def strip_top_h3(text: str) -> str:
    """Drop the leading `### 10.X Title` line and blank padding; keep body."""
    lines = text.splitlines()
    out: list[str] = []
    for i, ln in enumerate(lines):
        if i == 0 and ln.startswith("### 10"):
            continue
        out.append(ln)
    body = "\n".join(out).lstrip("\n")
    return body


def demote_h3(text: str, offset_depth: int = 1) -> str:
    """Promote/demote the heading hierarchy so fragment h3/h4 nests cleanly."""
    out: list[str] = []
    for ln in text.splitlines():
        if ln.startswith("#### "):
            out.append("#" * (4 + offset_depth) + ln[4:])
        elif ln.startswith("### "):
            out.append("#" * (3 + offset_depth) + ln[3:])
        else:
            out.append(ln)
    return "\n".join(out)


def wrap_subsection(num: str, title: str, body: str) -> str:
    """Emit `#### {num} {title}\n\n{body}\n`."""
    return f"\n#### {num} {title}\n\n{body.strip()}\n"


def unique_anchor(text, pattern):
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one anchor {pattern!r}; found {len(matches)}")
    return matches[0]


def managed(payload):
    return "<!-- weave-v3:begin -->\n" + payload + "<!-- weave-v3:end -->\n"


def insert_before(text: str, anchor_pat: str, payload: str) -> str:
    m = unique_anchor(text, anchor_pat)
    return text[:m.start()] + managed(payload) + text[m.start():]


def append_to_section(text: str, start_pat: str, end_pat: str, payload: str) -> str:
    start = unique_anchor(text, start_pat)
    end = unique_anchor(text, end_pat)
    if end.start() <= start.end():
        raise ValueError("section end precedes section start")
    return text[:end.start()] + managed(payload) + text[end.start():]


MARKER = "<!-- weave-addendum:review27-v2 -->"


def transform(txt: str, fragments=None) -> str:
    """Pure idempotent splice; marker denotes this frozen fragment version."""
    if MARKER in txt:
        return txt

    fragments = dict(fragments) if fragments is not None else {name: read_fragment(name) for name in FRAGMENT_NAMES}
    # Strip only our uniquely delimited generated regions before regenerating.
    starts = txt.count("<!-- weave-v3:begin -->")
    ends = txt.count("<!-- weave-v3:end -->")
    if starts != ends:
        raise ValueError("unbalanced managed sections")
    txt, removed = re.subn(r"<!-- weave-v3:begin -->\n.*?<!-- weave-v3:end -->\n", "", txt, flags=re.DOTALL)
    if removed != starts:
        raise ValueError("malformed/nested managed sections")
    txt = re.sub(r"^<!-- weave-addendum:review27-v3:[0-9a-f]{64} -->\n", "", txt)
    # Validate every report anchor, including replacements, before reading fragments.
    for pattern in (r"^## 3\. Methodology", r"^### 4\.5\.8 ", r"^### 4\.6 ", r"^### 4\.2 ", r"^### 4\.3 ", r"^### 5\.8 ", r"^### 5\.9 ", r"^### 5\.12 ", r"^### 5\.13 ", r"^### 5\.16 ", r"^## 6\. Summary of Findings$", r"^### 7\.1[^\n]*$"):
        unique_anchor(txt, pattern)

    # --- F-8 Related Work extension → insert §2.7 and §2.8 right before "## 3. Methodology"
    f8 = strip_top_h3(fragments["08-related-work-integration.md"])
    # Rewrite 10.8.1..10.8.5 as 2.7.1..2.7.5 (variance mitigation stays in §2.7;
    # split PRM/ST-PPO/DAR into §2.8)
    f8 = f8.replace("#### 10.8.1", "#### 2.7.1")
    f8 = f8.replace("#### 10.8.2", "#### 2.8.1")
    f8 = f8.replace("#### 10.8.3", "#### 2.8.2")
    f8 = f8.replace("#### 10.8.4", "#### 2.8.3")
    f8 = f8.replace("#### 10.8.5", "#### 2.8.4")
    f8_top = (
        "\n### 2.7 Variance-Mitigation Methods (Head-to-Head)\n\n"
        "*Addresses reviewer concerns W14 (missing AERO/CPPO/NGRPO/Scaf-GRPO "
        "comparisons) and Q7 (integrate a variance-mitigation method and test "
        "ZVF predictiveness). Paper section: "
        "`paper/sections/variance_mitigation_comparison.tex`. Bib keys: "
        "`aero2024`, `cppo2024`, `ngrpo2025`, `scafgrpo2025`.*\n\n"
    )
    f8_mid = (
        "\n### 2.8 Process, Tree, and Stability-Aware Variants\n\n"
        "*Addresses reviewer concerns W15 (Tree-GRPO + PRM), W16 (ST-PPO "
        "interaction), W17 (DAR / dual-KL). Paper section: "
        "`paper/sections/extended_related_work.tex`. Bib keys: "
        "`treegrpo2025`, `lightman2023prm`, `stppo2025`, `dar2024`.*\n\n"
    )
    f8_body = f8_top + _extract_between(f8, "#### 2.7.1", "#### 2.8.1") + f8_mid + _extract_from(f8, "#### 2.8.1")
    txt = insert_before(txt, r"^## 3\. Methodology", f8_body + "\n")

    # --- F-3 Framework-Gap Deep-Dive → append inside §4.5.8 (before §4.6)
    f3 = strip_top_h3(fragments["03-framework-gap-configs.md"])
    f3 = demote_h3(f3, offset_depth=1)
    f3_wrapped = (
        "\n#### 4.5.8.1 Framework Configuration Disclosure and Retraction "
        "(Reviewer W6 / Q3)\n\n"
        "*Paper section: `paper/sections/framework_configs_appendix.tex`. "
        "Reproducibility: `experiments/framework_config_dumps/*.yaml`.*\n\n"
        + f3.strip()
        + "\n"
    )
    txt = append_to_section(txt, r"^### 4\.5\.8 ", r"^### 4\.6 ", f3_wrapped)

    # --- F-5 Tool-use + code + ZVF outside math → append inside §4.2 (before §4.3)
    f5 = strip_top_h3(fragments["05-toolcode-zvf-outside-math.md"])
    f5 = demote_h3(f5, offset_depth=1)
    f5_wrapped = (
        "\n#### 4.2.1 Reward Design, SFT Warm-Starts, and ZVF Outside "
        "Verifiable Math (Reviewer W7 / Q6)\n\n"
        "*Paper section: `paper/sections/tool_use_code_expanded.tex`. "
        "Reproducibility: `experiments/tool_use_reward_analysis.py` → "
        "`platform_hybrid/experiments/results/tool_code_reward_diagnostics.tsv`.*\n\n"
        + f5.strip()
        + "\n"
    )
    txt = append_to_section(txt, r"^### 4\.2 ", r"^### 4\.3 ", f5_wrapped)

    # --- F-7 F5 scope clarification + regenerated figures → append inside §5.8 (before §5.9)
    f7 = strip_top_h3(fragments["07-f5-scope-figures.md"])
    f7 = demote_h3(f7, offset_depth=1)
    f7_wrapped = (
        "\n#### 5.8.1 F5 Reframed from Mechanistic to Descriptive; Regenerated "
        "Figures (Reviewer W10 / W12)\n\n"
        "*Paper sections: `paper/sections/frontier_scope_clarification.tex`, "
        "`paper/sections/figures_regeneration_note.tex`. Reproducibility: "
        "`platform_modal/scripts/regenerate_missing_figures.py`.*\n\n"
        + f7.strip()
        + "\n"
    )
    txt = append_to_section(txt, r"^### 5\.8 ", r"^### 5\.9 ", f7_wrapped)

    # --- F-1 ZVF formalization + partial correlation → append inside §5.12 (before §5.13)
    f1 = strip_top_h3(fragments["01-zvf-formalization.md"])
    f1 = demote_h3(f1, offset_depth=1)
    f1_wrapped = (
        "\n#### 5.12.1 Formal Definition, Partial-Correlation Ablation, and "
        "Cross-Framework Pipeline (Reviewer W1 / Q1 / W13)\n\n"
        "*Paper sections: `paper/sections/appendix_zvf_formalization.tex`, "
        "`paper/sections/zvf_pipeline_spec.tex`. Reproducibility: "
        "`platform_modal/scripts/partial_correlation_zvf.py`, "
        "`platform_modal/scripts/zvf_compute_cross_framework.py`.*\n\n"
        + f1.strip()
        + "\n"
    )
    txt = append_to_section(txt, r"^### 5\.12 ", r"^### 5\.13 ", f1_wrapped)

    # --- F-4 Evidence tiers + F1–F5 survival → append inside §5.16 (before §6)
    f4 = strip_top_h3(fragments["04-statistical-rigor-survival.md"])
    f4 = demote_h3(f4, offset_depth=1)
    f4_wrapped = (
        "\n#### 5.16.1 Evidence-Tier Partition and F1–F5 Survival Analysis "
        "(Reviewer W4 / W5 / Q5)\n\n"
        "*Paper section: `paper/sections/statistical_rigor_addendum.tex`. "
        "Reproducibility: `experiments/survival_analysis.py` → "
        "`platform_hybrid/experiments/results/survival_analysis.tsv`.*\n\n"
        + f4.strip()
        + "\n"
    )
    txt = append_to_section(txt, r"^### 5\.16 ", r"^## 6\. Summary", f4_wrapped)

    # --- F-2 Group-size reconciliation + F-6 heldout + F-6 base vs instruct
    #     → add as new §§5.17 / 5.18 / 5.19 just before "## 6. Summary"
    f2 = strip_top_h3(fragments["02-group-size-reconcile.md"])
    f2 = demote_h3(f2, offset_depth=1)
    f6 = strip_top_h3(fragments["06-heldout-base-instruct.md"])
    f6 = demote_h3(f6, offset_depth=1)
    # F-6 was split into 10.6.1 (heldout) and 10.6.2 (base/instruct) and 10.6.3 summary
    f6_heldout = _extract_between(f6, "10.6.1", "10.6.2").strip()
    f6_bi = _extract_from(f6, "10.6.2").strip()

    new_subs = (
        "\n### 5.17 Group-Size Reconciliation: Token-Budget-Normalized Sweep "
        "(Reviewer W2 / W3 / Q2)\n\n"
        "*Paper section: `paper/sections/group_size_reconcile.tex`. "
        "Reproducibility: `experiments/group_size_token_normalized.py` → "
        "`platform_hybrid/experiments/results/group_size_token_normalized.tsv`.*\n\n"
        + f2.strip()
        + "\n\n"
        "### 5.18 Held-Out Sampling Protocols: P1 / P2 / P3 "
        "(Reviewer W8)\n\n"
        "*Paper section: `paper/sections/heldout_stratified.tex`. "
        "Reproducibility: `experiments/stratified_heldout.py` → "
        "`platform_hybrid/experiments/results/heldout_stratified.tsv`.*\n\n"
        + f6_heldout
        + "\n\n"
        "### 5.19 Base vs Instruct Paired Evaluation "
        "(Reviewer W9 / W11 / Q4)\n\n"
        "*Paper section: `paper/sections/base_vs_instruct_paired.tex`. "
        "Reproducibility: `experiments/base_instruct_paired.py` → "
        "`platform_hybrid/experiments/results/base_instruct_paired.tsv`.*\n\n"
        + f6_bi
        + "\n\n"
    )
    txt = insert_before(txt, r"^## 6\. Summary", new_subs)

    # --- Update §6 Summary of Findings preamble with pointer
    summary_note = (
        "\n> **Revision note (2026-04-19).** The findings below have been "
        "audited against NeurIPS 2026 reviewer feedback. Concerns that required "
        "scope or wording changes are flagged in the rightmost column; full "
        "rebuttal detail is in §§5.8.1, 5.12.1, 5.16.1, 5.17–5.19 and §§2.7–2.8. "
        "A mechanical registry of all 24 weaknesses is at "
        "`paper/reviewer_points.yaml`, scored by "
        "`platform_modal/scripts/reviewer_response_score.sh`.\n\n"
    )
    txt = re.sub(
        r"(^## 6\. Summary of Findings\n)", r"\1" + managed(summary_note),
        txt, count=1, flags=re.MULTILINE,
    )

    # --- Update §7.1 Gaps Closed
    gaps_note = (
        "\n- **NeurIPS 2026 reviewer rebuttal (all 24 items).** ZVF "
        "formalization + partial-correlation ablation (W1/Q1), cross-framework "
        "pipeline (W13), group-size token-normalization and gradient-"
        "utilization formalization (W2/W3/Q2), framework-gap config dumps "
        "(W6/Q3), evidence-tier partition + F1–F5 survival (W4/W5/Q5), "
        "tool-use/code reward-design analysis and ERF surrogate (W7/Q6), "
        "held-out stratified sampling P1/P2/P3 (W8), base-vs-instruct paired "
        "evaluation (W9/W11/Q4), F5 scope downgrade (W10), regenerated figures "
        "(W12), AERO/CPPO/NGRPO/Scaf-GRPO head-to-head (W14/Q7), "
        "Tree-GRPO/PRM (W15), ST-PPO (W16), DAR/dual-KL (W17). "
        "See §§2.7–2.8 and §§5.8.1, 5.12.1, 5.16.1, 5.17–5.19.\n"
    )
    # insert as the first bullet under §7.1
    def _inject_under_71(m: re.Match) -> str:
        return m.group(0) + managed(gaps_note)
    txt = re.sub(
        r"^### 7\.1[^\n]*\n", _inject_under_71,
        txt, count=1, flags=re.MULTILINE,
    )

    digest = hashlib.sha256("".join(fragments[name] for name in FRAGMENT_NAMES).encode()).hexdigest()
    return f"<!-- weave-addendum:review27-v3:{digest} -->\n" + txt


FRAGMENT_NAMES = ("01-zvf-formalization.md", "02-group-size-reconcile.md", "03-framework-gap-configs.md", "04-statistical-rigor-survival.md", "05-toolcode-zvf-outside-math.md", "06-heldout-base-instruct.md", "07-f5-scope-figures.md", "08-related-work-integration.md")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=REPORT)
    parser.add_argument("--fragments", type=Path, default=FRAG_DIR)
    parser.add_argument("--output", type=Path, default=REPORT.with_name(REPORT.stem + "_review27_v3.md"))
    args = parser.parse_args(argv)
    globals()["FRAG_DIR"] = args.fragments
    original = args.input.read_text()
    if MARKER in original:
        raise ValueError("legacy frozen splice cannot be updated safely; supply its unmodified backup")
    updated = transform(original)
    if args.output.exists() and args.output.read_text() == updated:
        return
    if args.output.resolve() == args.input.resolve():
        raise ValueError("use a distinct versioned output; original report is preserved")
    if args.output.exists():
        raise ValueError("output differs; choose a new versioned output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("x") as fh:
        fh.write(updated)
    temporary.replace(args.output)


def _extract_between(blob: str, start_needle: str, end_needle: str) -> str:
    start = unique_anchor(blob, r"^.*" + re.escape(start_needle) + r"[^\n]*$")
    end = unique_anchor(blob, r"^.*" + re.escape(end_needle) + r"[^\n]*$")
    if end.start() <= start.start():
        raise ValueError("fragment headings out of order")
    return blob[start.start():end.start()]


def _extract_from(blob: str, needle: str) -> str:
    match = unique_anchor(blob, r"^.*" + re.escape(needle) + r"[^\n]*$")
    return blob[match.start():]


if __name__ == "__main__":
    main()
