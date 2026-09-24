#!/usr/bin/env python3
"""
Build the M.Tech thesis report PDF.

Pipeline:
  1. pandoc converts each chapter markdown -> LaTeX fragment
     (--top-level-division=chapter so "#" becomes \\chapter)
  2. figures are injected at anchor headings, falling back to end-of-chapter
  3. preamble + front matter + chapters + back matter -> thesis_master.tex
  4. tectonic compiles thesis_master.tex -> PDF

Run:  python3 build_thesis.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
MASTER = os.path.join(HERE, "thesis_master.tex")
BUILD = os.path.join(HERE, "build")
PDF = os.path.join(HERE, "Thesis_Report_ArvindCR.pdf")

CHAPTERS = [
    ("ch01_introduction.md", "Part I --- Foundations"),
    ("ch02_literature.md", None),
    ("ch03_requirements.md", None),
    ("ch04_methodology.md", "Part II --- Design and Implementation"),
    ("ch05_implementation.md", None),
    ("ch06_results_core.md", "Part III --- Results"),
    ("ch07_results_infra.md", None),
    ("ch08_results_fraud.md", None),
    ("ch09_results_campaign.md", None),
    ("ch10_synthesis_conclusions.md", "Part IV --- Synthesis"),
]

APPENDICES = [
    ("ch_back_run_registry.md", "Appendices"),
    ("ch_back_reproducibility.md", None),
    ("ch_back_notation.md", None),
]

# figure name -> (caption, label, [anchor substrings tried in order])
FIGURES: dict[str, list[tuple[str, str, list[str]]]] = {
    "ch01_introduction.md": [
        ("fig_confounding",
         "The confounding problem this work addresses. A single reported reward "
         "figure sits on top of an unstated stack; without fixing and reporting "
         "that stack the number is not attributable to the method under test.",
         "fig:confounding", ["Motivating", "Problem Statement", "Introduction"]),
        ("fig_grpo_loop",
         "The GRPO training loop. A prompt is sampled $G$ times, each completion "
         "is scored by a reward function, advantages are formed relative to the "
         "group, and the policy is updated. When all $G$ rewards coincide the "
         "group-relative advantage collapses to zero and no gradient flows.",
         "fig:grpoloop", ["Background", "Introduction"]),
        ("fig_contribution_map",
         "Boundary between the inherited Semester-3 group infrastructure and the "
         "individual Semester-4 contribution reported in this document.",
         "fig:contribution", ["Contribution", "Scope"]),
    ],
    "ch02_literature.md": [
        ("fig_lineage",
         "Algorithmic lineage from RLHF through PPO and DPO to GRPO and its "
         "variants. Each transition removes a component or substitutes a "
         "cheaper estimator; GRPO removes the critic and replaces it with a "
         "group-relative baseline.",
         "fig:lineage", ["RLHF and PPO to Group-Relative", "Direct Preference Optimisation"]),
        ("fig_research_questions",
         "The eight studies P1--P8 grouped by theme, with the question each one "
         "addresses.",
         "fig:questions", ["Positioning", "Literature"]),
    ],
    "ch04_methodology.md": [
        ("fig_attribution",
         "The attribution design. Exactly one factor of the training stack is "
         "varied while the remainder is held fixed, so a measured difference "
         "cannot be attributed to an unstated framework difference.",
         "fig:attribution", ["Design Commitment", "Methodology"]),
        ("fig_zvf_definition",
         "Definition of the Zero-Variance Fraction. One prompt produces $G$ "
         "completions with scalar rewards; when the within-group reward standard "
         "deviation is zero the advantage is identically zero and the sample "
         "contributes no gradient. ZVF is the fraction of such groups.",
         "fig:zvfdef", ["Zero-Variance", "Formalis", "Methodology"]),
        ("fig_gradient_utilisation",
         "Gradient utilisation, $1-\\zvf$, as a function of group size $G$. "
         "Increasing $G$ reduces the zero-variance fraction with saturating "
         "benefit, which is why larger groups are not uniformly better.",
         "fig:gu", ["Zero-Variance", "Methodology"]),
        ("fig_telemetry",
         "Per-step telemetry recorded by the unified harness, and the "
         "destinations each record flows to.",
         "fig:telemetry", ["Telemetry", "Methodology"]),
        ("fig_four_pillars",
         "The four de-confound pillars. Each holds the whole stack fixed and "
         "varies exactly one factor.",
         "fig:pillars", ["Methodology", "Portfolio"]),
        ("fig_campaign_design",
         "Design of the E1--E14 held-out evaluation campaign: one frozen actor, "
         "each suite graded by its own native evaluator at a pinned revision, "
         "with original-contract and replacement-scope results kept separate.",
         "fig:campaign", ["Campaign", "Methodology"]),
    ],
    "ch05_implementation.md": [
        ("fig_architecture",
         "Six reinforcement-learning backends behind one Tinker-style API. A "
         "single task, reward and decoding configuration drives any backend, so "
         "differences in outcome are attributable to the stack.",
         "fig:arch", ["Architecture", "Implementation"]),
        ("fig_harness",
         "Evaluation harness internals. The harness is fail-closed: when a "
         "precondition cannot be satisfied the lane records \\textsc{blocked} "
         "rather than substituting an adjacent benchmark.",
         "fig:harness", ["Harness", "Implementation"]),
        ("fig_receipt_pipeline",
         "The evidence chain from sealed request through execution and native "
         "grading to a deterministic ledger check. A break anywhere in the "
         "chain means no number is reported for that lane.",
         "fig:receipts", ["Receipt", "Implementation"]),
    ],
    "ch06_results_core.md": [
        ("fig_zvf_by_library",
         "Mean Zero-Variance Fraction by backend library. Signal starvation is "
         "large and varies across stacks, which is the empirical case for "
         "reporting the stack alongside any reward figure.",
         "fig:zvflib", ["Zero-Variance", "P2", "Results"]),
        ("fig_group_size",
         "Trainability as a function of group size $G$. The relationship is "
         "non-monotone: intermediate group sizes outperform both the smallest "
         "and the largest tested, and a single-seed sweep does not establish an "
         "optimum.",
         "fig:groupsize", ["Group Size", "P3", "Results"]),
        ("fig_length_bias",
         "Length bias. Four of eleven GRPO runs peak before 65\\% of training "
         "and then decay, consistent with the policy optimising response length "
         "rather than correctness. Dr.\\ GRPO does not exhibit the pattern.",
         "fig:lengthbias", ["Length Bias", "P4", "Results"]),
        ("fig_scaling_null",
         "Cross-scale behaviour of reward across the studied model range. No "
         "reliable monotone scaling trend is recovered; the conservative "
         "negative is the result.",
         "fig:scaling", ["Scaling", "P1", "Results"]),
    ],
    "ch09_results_campaign.md": [
        ("fig_lane_status",
         "Terminal status of all fourteen E1--E14 lanes. Each cell names the "
         "suite and its state, so status is legible without relying on colour "
         "alone.",
         "fig:lanes", ["Campaign", "Results"]),
        ("fig_e8_categories",
         "LAB-Bench public split, per-category accuracy, all 1{,}967 questions "
         "graded across eight categories. Accuracy is strongly "
         "category-dependent; no pooled cross-category average is claimed.",
         "fig:e8", ["LAB-Bench", "E8", "complete scopes"]),
        ("fig_e11_decomposition",
         "VerilogEval decomposition: two native framings of 156 tasks, "
         "67/156 code-completion and 62/156 spec-to-RTL, summing to "
         "$129/312 = 41.35\\%$ pass@1.",
         "fig:e11", ["VerilogEval", "E11", "complete scopes"]),
        ("fig_terminal_taxonomy",
         "Terminal-state taxonomy used to close every lane. Each state has a "
         "defined meaning and a named reopen condition where one exists.",
         "fig:taxonomy", ["Terminal", "blocked lanes", "externally blocked"]),
        ("fig_e4_diagnostic",
         "The E4 rerun diagnostic. Across 100 trials agents terminate at a "
         "median of roughly six steps without producing deliverables, and more "
         "than a quarter end in role-token degeneration. The 0.0 mean reward "
         "therefore measures tool-dialogue collapse, not finance reasoning.",
         "fig:e4", ["BankerToolBench", "E4", "rerun"]),
    ],
    "ch10_synthesis_conclusions.md": [
        ("fig_threats",
         "Threats to validity grouped as internal, construct and external, each "
         "with the mitigation applied in this work.",
         "fig:threats", ["Threats", "Validity", "Synthesis"]),
    ],
}


# Headings in the chapter markdown carry manual numbers ("## 3.2 Foo").
# LaTeX numbers them itself, so strip the manual ones to avoid "3.2 3.2 Foo".
_HEAD_NUM = re.compile(r"^(#{1,6})\s+(\d+(?:\.\d+)*)\.?\s+(.*)$")
_APPENDIX_NUM = re.compile(r"^(#{1,2})\s+(Appendix\s+[A-Z]\.?)\s+(.*)$", re.I)


def _strip_heading_numbers(md_text: str) -> str:
    out = []
    for line in md_text.split("\n"):
        m = _HEAD_NUM.match(line)
        if m:
            line = f"{m.group(1)} {m.group(3)}"
        else:
            m2 = _APPENDIX_NUM.match(line)
            if m2:
                line = f"{m2.group(1)} {m2.group(2)} {m2.group(3)}"
        out.append(line)
    return "\n".join(out)


def pandoc_md_to_tex(md_path: str, tex_path: str) -> bool:
    """Convert a chapter markdown file to a LaTeX fragment."""
    if not shutil.which("pandoc"):
        return False
    try:
        raw = open(md_path, encoding="utf-8").read()
        # keep the H1 as the chapter title (pandoc -> \chapter) with its number
        # removed too, since \chapter supplies its own.
        tmp = md_path + ".numbered.md"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(_strip_heading_numbers(raw))
        subprocess.run(
            ["pandoc", tmp, "-t", "latex", "--top-level-division=chapter",
             "-o", tex_path],
            check=True, capture_output=True, timeout=120,
        )
        os.remove(tmp)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        err = getattr(exc, "stderr", b"") or b""
        print(f"  pandoc FAILED for {os.path.basename(md_path)}: "
              f"{err.decode()[:200]}", file=sys.stderr)
        return False


def available_figures() -> set[str]:
    if not os.path.isdir(FIGDIR):
        return set()
    return {f[:-4] for f in os.listdir(FIGDIR) if f.endswith(".pdf")}


def inject_figures(tex: str, chapter_md: str, have: set[str]) -> tuple[str, int, list[str]]:
    """Insert \\fig{...} blocks at anchors. Returns (tex, n_inserted, missing)."""
    specs = FIGURES.get(chapter_md, [])
    if not specs:
        return tex, 0, []
    inserted, missing = 0, []
    pending = []
    for name, caption, label, anchors in specs:
        if name not in have:
            missing.append(name)
            continue
        pending.append((name, caption, label, anchors))

    for name, caption, label, anchors in pending:
        block = (f"\n\\fig{{{name}}}\n{{{caption}}}\n{{{label}}}\n\n")
        safe_caption = caption.replace("\n", " ")
        block = f"\n\n\\fig{{{name}}}\n{{{safe_caption}}}\n{{{label}}}\n\n"
        placed = False
        for anchor in anchors:
            # find a heading line containing the anchor
            pat = re.compile(r"^(\\(?:chapter|section|subsection)\*?\{[^}]*"
                             + re.escape(anchor) + r"[^}]*\})", re.M)
            m = pat.search(tex)
            if m:
                # insert after the end of that heading's first paragraph block
                end = tex.find("\n\n", m.end())
                end = end if end != -1 else m.end()
                tex = tex[:end] + "\n" + block + tex[end:]
                placed = True
                break
        if not placed:
            tex = tex.rstrip() + "\n" + block
        inserted += 1
    return tex, inserted, missing


def main() -> int:
    os.makedirs(BUILD, exist_ok=True)
    have = available_figures()
    print(f"figures available: {len(have)}")

    pieces = [open(os.path.join(HERE, "preamble.tex"), encoding="utf-8").read()]
    pieces.append(open(os.path.join(HERE, "frontmatter.tex"), encoding="utf-8").read())

    total_figs, all_missing, missing_ch, empty_ch = 0, set(), [], []

    def emit(md_name: str, part: str | None) -> None:
        nonlocal total_figs
        md_path = os.path.join(HERE, md_name)
        if not os.path.exists(md_path):
            missing_ch.append(md_name)
            return
        if os.path.getsize(md_path) < 400:
            empty_ch.append(md_name)
        if part:
            pieces.append(f"\n\\part{{{part}}}\n")
        tex_path = os.path.join(BUILD, md_name[:-3] + ".tex")
        if not pandoc_md_to_tex(md_path, tex_path):
            missing_ch.append(md_name + " (pandoc)")
            return
        tex = open(tex_path, encoding="utf-8").read()
        tex, n, miss = inject_figures(tex, md_name, have)
        total_figs += n
        all_missing.update(miss)
        pieces.append("\n" + tex + "\n")

    for md_name, part in CHAPTERS:
        emit(md_name, part)
    for md_name, part in APPENDICES:
        emit(md_name, part)

    pieces.append("\n\\end{document}\n")
    master = "\n".join(pieces)
    with open(MASTER, "w", encoding="utf-8") as fh:
        fh.write(master)

    words = len(re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", master).split())
    print(f"wrote {MASTER}  (~{words:,} words, {total_figs} figures injected)")
    if missing_ch:
        print(f"  MISSING/FAILED chapters: {', '.join(missing_ch)}", file=sys.stderr)
    if empty_ch:
        print(f"  SUSPICIOUSLY SHORT: {', '.join(empty_ch)}", file=sys.stderr)
    if all_missing:
        print(f"  figures referenced but NOT built: {', '.join(sorted(all_missing))}",
              file=sys.stderr)

    if not shutil.which("tectonic"):
        print("tectonic not found — .tex written but not compiled", file=sys.stderr)
        return 1

    print("compiling with tectonic ...")
    try:
        r = subprocess.run(["tectonic", "-X", "compile", MASTER,
                            "--outdir", BUILD, "--keep-logs"],
                           capture_output=True, timeout=1200, cwd=HERE)
        out = (r.stdout + r.stderr).decode(errors="replace")
        tail = "\n".join(out.strip().splitlines()[-18:])
        print(tail)
        built = os.path.join(BUILD, "thesis_master.pdf")
        if os.path.exists(built):
            shutil.copy(built, PDF)
            pages = out.count("Writing")  # rough
            print(f"OK -> {PDF}  ({os.path.getsize(PDF)/1024:.0f} KB)")
            return 0
        print("no PDF produced", file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        print("tectonic timed out", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
