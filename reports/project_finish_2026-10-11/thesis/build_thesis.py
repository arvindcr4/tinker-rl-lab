#!/usr/bin/env python3
"""
Build the M.Tech thesis report PDF.

Pipeline:
  1. pandoc converts each chapter markdown -> LaTeX fragment
     (--top-level-division=chapter so "#" becomes \\chapter)
  2. figures are injected at anchor headings, falling back to end-of-chapter
  3. preamble + front matter + chapters + back matter -> thesis_master.tex
  4. tectonic or latexmk compiles thesis_master.tex -> PDF

Run: ./build_portable.sh (recommended), or python3 build_thesis.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
MASTER = os.path.join(HERE, "thesis_master.tex")
BUILD = os.path.join(HERE, "build")
PDF = os.path.join(HERE, "Thesis_Report_ArvindCR_2026-10-18_REVISED_REVIEW.pdf")

CHAPTERS = [
    ("ch01_introduction.md", "Foundations"),
    ("ch02_literature.md", None),
    ("ch03_requirements.md", None),
    ("ch04_methodology.md", "Design and Implementation"),
    ("ch05_implementation.md", None),
    ("ch06_results_core.md", "Results"),
    ("ch07_results_infra.md", None),
    ("ch09_results_campaign.md", None),
    ("ch10_synthesis_conclusions.md", "Synthesis"),
]

APPENDICES = [
    ("ch_back_run_registry.md", "Appendices"),
    ("ch_back_reproducibility.md", None),
    ("ch_back_notation.md", None),
    ("ch_back_campaign_detail.md", None),
    ("ch_back_errata.md", None),
    ("ch08_results_fraud.md", None),
    ("ch_back_c1_evidence.md", None),
]

# the evidence map is generated from the collected source notes and
# emitted as the last appendix
EVIDENCE_MAP = os.path.join(BUILD, "ch_back_evidence_map.md")

# figure name -> (caption, label, [anchor substrings tried in order])
FIGURES: dict[str, list[tuple[str, str, list[str]]]] = {
    "ch01_introduction.md": [
        ('fig_confounding',
         'Historical framework records and the attribution gap. Tinker/Base and TRL/Instruct are different treatments; their recorded training rewards do not isolate a framework effect. Dry-run rows have no measured outcome. Marginal effect-size bars are removed rather than interpreted as a variance decomposition.',
         'fig:confounding', ['Motivating', 'Problem Statement', 'Introduction']),
        ('fig_grpo_loop',
         'The group-relative loop and exact-tie mechanism. The Phase-1 P2 batch (0.72--0.77, Qwen3.5-4B, GSM8K, G=8, one seed) used mean-centred advantages r minus group mean without standard-deviation normalisation. Exact ties give zero reward-relative advantages in both that form and standardised GRPO. The arithmetic G sweep is training telemetry only, not estimator or held-out-accuracy evidence.',
         'fig:grpoloop', ['Background', 'Introduction']),
        ("fig_contribution_map",
         "Boundary between the inherited Semester-3 group infrastructure and the "
         "individual Semester-4 contribution reported in this document. Both "
         "columns are drawn from the repository's own provenance record (the "
         "Semester-3 tag and boundary commit, the post-boundary commit and file "
         "counts, the solo-authorship commit, and the Semester-4 README with its "
         "paper-to-evidence map); where the record does not settle a claim, the "
         "claim is left out rather than inferred.",
         "fig:contribution", ["Contribution", "Scope"]),
    ],
    "ch02_literature.md": [
        ("fig_lineage",
         "Algorithmic lineage from RLHF through PPO to GRPO, with DPO as an "
         "orthogonal branch. Each transition removes a component or substitutes a "
         "cheaper estimator; GRPO removes the critic and replaces it with a "
         "group-relative baseline. DPO (dashed frame) removes the on-policy loop "
         "and the reward model rather than simplifying the critic, trading online "
         "exploration for stability. Dates are first publication or first use at "
         "scale. GRPO variants (DAPO, Dr.~GRPO and others) are discussed in the "
         "text and not drawn.",
         "fig:lineage", ["RLHF and PPO to Group-Relative", "Direct Preference Optimisation"]),
        ('fig_research_questions',
         'Seven core RL studies and their present evidence boundaries. Missing controls and invalid designs are left open, rather than labelled successful null experiments. The archived non-RL side study is outside the core argument.',
         'fig:questions', ['Positioning', 'Literature']),
    ],
    "ch04_methodology.md": [
        ("fig_attribution",
         "The target attribution design: vary one factor while holding the "
         "remaining stack fixed. Executed comparisons did not always meet "
         "that ideal: checkpoint identity, prompt exposure and other residual "
         "differences are disclosed with each result in Chapters~6 and~7.",
         "fig:attribution", ["Design Commitment", "Methodology"]),
        ('fig_zvf_definition',
         'Exact ties versus near-ties under the population-variance reporting convention (ddof=0). The denominator stabiliser and the variance threshold are different quantities. Historical sample-variance analyses are retained with their convention in the text; near-ties need not have exactly zero advantages. Measured composition is the three-seed Qwen3-8B GSM8K sampling diagnostic.',
         'fig:zvfdef', ['Zero-Variance', 'Formalis', 'Methodology']),
        ('fig_gradient_utilisation',
         'Measured group-contrast fraction on the fixed-prompt arithmetic sweep. Points are seed means, not a fitted law, a matched-token efficiency frontier or a held-out generalisation result. The four values use G=2,4,8,16 with three seeds each; larger G consumes more completions.',
         'fig:gu', ['Zero-Variance', 'Methodology']),
        ("fig_telemetry",
         "Intended per-step telemetry schema of the unified harness; actual availability is runner-specific. The diagram shows the "
         "destinations each record flows to.",
         "fig:telemetry", ["Telemetry", "Methodology"]),
        ('fig_four_pillars',
         'Intended comparison axes and the controls still needed to identify them. The leaked arithmetic estimator comparison is excluded. A working launcher or an illustrative model is not evidence that an intended control was executed.',
         'fig:pillars', ['Methodology', 'Portfolio']),
        ("fig_campaign_design",
         "Historical lane-scope map for the harness audit. Original contracts and exploratory protocol-override replacements are kept separate. Replacement completion labels describe retained coverage, not satisfaction of the prospective fail-closed gates. Missing or private source payloads limit independent reproduction; a source hash is not execution authentication. Full metric definitions and denominators are in Chapter~8 and Table~C.3.",
         "fig:campaign", ["Campaign", "Methodology"]),
    ],
    "ch05_implementation.md": [
        ("fig_architecture",
         "Six RL frameworks behind one Tinker-style API. Five of the six are in "
         "the dispatch matrix (5 frameworks $\\times$ 6 compute backends); only two "
         "executed (Tinker, TRL: last-10 mean training reward on Qwen/Qwen3-8B "
         "GSM8K, from \\texttt{framework\\_comparison.json}), two are dry-run "
         "launch plans and two were not executed. One configuration drives every "
         "framework, but the executed comparison bundled a different base "
         "checkpoint (Chapter~7), so outcome differences are reported with the "
         "stack rather than attributed to it.",
         "fig:arch", ["Architecture", "Implementation"]),
        ('fig_harness',
         'Evaluation-pipeline stages and their evidence boundaries. Prospective fail-closed gates are distinguished from the recorded historical executions, including six exploratory protocol-override replacements on 27 September. Offline gate tests do not authenticate lost training source or prove that a gate governed a measured run.',
         'fig:harness', ['Harness', 'Implementation']),
        ("fig_receipt_pipeline",
         "The evidence chain from sealed request through execution and native "
         "grading to a deterministic ledger check. A break anywhere in the "
         "chain prevents promotion to a fully execution-verified result. "
         "Recovered summaries remain separately labelled with their limits. Upper row: the chain; "
         "the 11/11 PASS badge is the 2026-09-19 deterministic ledger check of eleven named claims, not of every figure "
         "(\\texttt{LEDGER\\_CODE\\_CHECK\\_2026-09-19.json}). Lower row: the E1 "
         "wave-10 recovery, where the lost execution source stops the chain.",
         "fig:receipts", ["Receipt", "Implementation"]),
    ],
    "ch06_results_core.md": [
        ('fig_zvf_by_library',
         'Measured exact-tie fractions recomputed from the three stored Qwen3-8B GSM8K sampling tensors (200 prompts per seed). The 95/600 pooled count is descriptive; no synthetic variant panel, imputed AUROC or predictive validation is presented.',
         'fig:zvflib', ['Zero-Variance', 'P2', 'Results']),
        ('fig_group_size',
         'Measured reward-contrast summaries and completion counts under the fixed-prompt arithmetic sweep. No reconstructed $G\\ge32$ performance grid or contaminated arithmetic accuracy comparison is shown. Stored values are unchanged.',
         'fig:groupsize', ['Group Size', 'P3', 'Results']),
        ('fig_length_bias',
         'Evidence boundaries for the length-bias question. Both GSM8K caps bind; the raised-cap first-five-step means are training telemetry, not pretraining baselines. No held-out raised-cap evaluation or causal mask comparison is available.',
         'fig:lengthbias', ['Length Bias', 'P4', 'Results']),
        ('fig_scaling_null',
         'Logged-horizon inventory of the twelve historical cross-scale anchors, from the retained cross-scale TSV. Five interrupted traces are excluded from scaling fits. The remaining seven are still heterogeneous single-seed records, not a controlled scaling experiment; no regression or asymptotic trend is asserted.',
         'fig:scaling', ['Scaling', 'P1', 'Results']),
    ],
    "ch09_results_campaign.md": [
        ('fig_lane_status',
         'Scope-specific lane accounting, with the September 27 replacements explicitly marked as exploratory protocol-override runs. Their receipts retain evidence, but these runs did not satisfy the original prospective acceptance path. A complete coverage label does not establish complete grading, execution-source provenance or training benefit. Original contracts and replacements are never pooled.',
         'fig:lanes', ['Campaign', 'Results']),
        ("fig_e8_categories",
         "LAB-Bench public split, per-category accuracy, all 1{,}967 questions "
         "graded across eight categories. Each bar is correct/total for one "
         "category (labels give the counts). Accuracy is strongly "
         "category-dependent, so no pooled cross-category average is drawn or "
         "claimed. CloningScenarios is a historically reported zero (0/33). "
         "The original diagnostic file was recovered and byte-verified on "
         "3 October (Appendix B); its aggregate counts agree with these bars.",
         "fig:e8", ["LAB-Bench", "E8", "complete scopes"]),
        ("fig_e11_decomposition",
         "VerilogEval decomposition: two native framings of 156 tasks, "
         "67/156 code-completion and 62/156 spec-to-RTL, summing to "
         "$129/312 = 41.35\\%$ pass@1. The dashed top bar is the sum of the two "
         "rows, not a third measurement. Graded by the suite's native harness "
         "(receipt \\texttt{e11\\_full\\_receipt.json}); the $129/311 = 41.48\\%$ "
         "value that excludes \\texttt{Prob099\\_m2014\\_q6c} is a non-canonical "
         "sensitivity only.",
         "fig:e11", ["VerilogEval", "E11", "complete scopes"]),
        ("fig_terminal_taxonomy",
         "Terminal-state taxonomy used to close every lane. Each state has a "
         "defined meaning and a named reopen condition where one exists. The "
         "seven states are the ledger's full vocabulary; 17 lane-states cover 14 "
         "lanes because E8, E10 and E14 carry separate original and replacement "
         "rows. States classify lanes, not results: an unrecognised state is "
         "recorded as blocked rather than absorbed into a percentage, and "
         "original-contract and replacement-scope numbers are never pooled. The "
         "completion gate's \\texttt{PARTIAL\\_EXACT}/\\texttt{PARTIAL\\_RECOVERY} "
         "labels are a separate evidence-class vocabulary (Chapter~5). Lane "
         "assignments are those of 2026-09-19; the six lanes in the three "
         "pending leaves were all run and scored on 2026-09-27 (Chapter~8).",
         "fig:taxonomy", ["Terminal", "blocked lanes", "externally blocked"]),
        ('fig_replacement_results',
         'Exploratory protocol-override replacement outcomes recorded on 27 September 2026. Rate labels give exact numerator/denominator and one-decimal percentages; E6 counts 108 ungraded tasks as failures, so 90/812 is a lower bound. E13 is mean progression across six environment means, not a binary success fraction. These measurements do not validate the proposed fail-closed controllers.',
         'fig:replacement', ['Replacement-scope lanes completed', 'Replacement-scope']),
    ],
    # ch10: the threats figure (fig_threats) was replaced by Table 9.1
}


# Headings in the chapter markdown carry manual numbers ("## 3.2 Foo").
# LaTeX numbers them itself, so strip the manual ones to avoid "3.2 3.2 Foo".
_HEAD_NUM = re.compile(r"^(#{1,6})\s+([A-Z]?\d*(?:\.\d+)*)\.?\s+(.*)$")
# Appendix chapters are lettered by \appendix, so drop the manual "Appendix A."
_APPENDIX_NUM = re.compile(r"^(#{1,2})\s+Appendix\s+[A-Z]\.?\s+(.*)$", re.I)


def _strip_heading_numbers(md_text: str) -> str:
    out = []
    for line in md_text.split("\n"):
        m2 = _APPENDIX_NUM.match(line)
        if m2:
            line = f"{m2.group(1)} {m2.group(2)}"
        else:
            m = _HEAD_NUM.match(line)
            # only strip tokens that really are numbers ("3.2", "A.1"), not words
            if m and re.fullmatch(r"(\d+|[A-Z]\.\d+)(\.\d+)*", m.group(2)):
                line = f"{m.group(1)} {m.group(3)}"
        out.append(line)
    return "\n".join(out)


# Inline "(source: path)" / "(sources: a, b)" evidence notes interrupt the
# prose (~450 of them). Remove them from the running text and collect them,
# per section, into a generated evidence-map appendix. A note that opens its
# own paragraph (a table's source line) stays visible under its table. Table
# rows and headings are left alone.
EVIDENCE: list[tuple[str, list[str]]] = []   # (heading line, [source bodies])
_SRC_OPEN = re.compile(r"\s?\((sources?):\s")


def _sources_to_footnotes(md_text: str) -> tuple[str, int]:
    out, n = [], 0
    for line in md_text.split("\n"):
        if line.startswith("#"):
            EVIDENCE.append((line, []))
        if line.lstrip().startswith(("|", "#")):
            out.append(line)
            continue
        buf, i = [], 0
        while True:
            m = _SRC_OPEN.search(line, i)
            if not m:
                buf.append(line[i:])
                break
            # find the matching close paren, respecting nesting and `code`
            j, depth, in_code = m.end(), 1, False
            while j < len(line) and depth:
                ch = line[j]
                if ch == "`":
                    in_code = not in_code
                elif not in_code and ch == "(":
                    depth += 1
                elif not in_code and ch == ")":
                    depth -= 1
                j += 1
            if depth:                       # unbalanced: leave as-is
                buf.append(line[i:])
                break
            label = "Sources" if m.group(1) == "sources" else "Source"
            body = line[m.end():j - 1].strip()
            if m.start() == 0 and not line[:m.start()].strip():
                # a source note that opens its own paragraph (typically under a
                # table) has no text to anchor a footnote mark; print it inline
                buf.append(f"*{label}: {body.rstrip('.')}.*")
                if line[j:j + 1] == ".":
                    j += 1
            else:
                buf.append(line[i:m.start()])
                if EVIDENCE:
                    EVIDENCE[-1][1].append(body)
            n += 1
            i = j
        out.append("".join(buf))
    return "\n".join(out), n


_LEADIN = re.compile(r"^\*\*Table ([A-Z0-9]+\.[A-Z0-9]+) [—-]+ (.+?)\*\*(.*)$")


def _table_leadins(md_text: str) -> str:
    """Tables carry a bold "**Table 9.A — Title.** note" lead-in.

    If the table below it also has a pandoc caption (": ..."), LaTeX numbers
    that caption, so the lead-in keeps its text but drops the number. If not,
    the lead-in is the table's only caption: keep its number, set it as a
    caption line and add it to the List of Tables.
    """
    lines = md_text.split("\n")
    for k, line in enumerate(lines):
        m = _LEADIN.match(line)
        if not m:
            continue
        j = k + 1
        # the lead-in paragraph may wrap onto further lines
        while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith("|"):
            j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        num, title, rest = m.group(1), m.group(2).strip(), m.group(3)
        if j < len(lines) and lines[j].startswith(": "):
            lines[k] = f"**{title}**{rest}"
        else:
            toc = title.rstrip(".").replace("\\", "").replace("`", "")
            toc = re.sub(r"[$*_]", "", toc)
            lines[k] = (f"`\\addcontentsline{{lot}}{{table}}{{\\protect\\numberline{{{num}}}{toc}}}`{{=latex}}"
                        f"**Table {num}.** {title}{rest}")
            lines[k] = "`\\Needspace{10\\baselineskip}`{=latex}" + lines[k]
    return "\n".join(lines)


_PATHLIKE = re.compile(r"/|\.(?:md|tex|json|tsv|py|csv|ya?ml|txt|jsonl|pdf|bib|pptx|ipynb)\b")


def write_evidence_map(path: str) -> int:
    """Write the evidence-map appendix from the collected source notes."""
    out = ["# Appendix H. Source Reference Map", "",
           "This map collects source references retained in the sanitized manuscript. "
           "A reference is not a promise that its payload is bundled or publicly available. "
           "Repository-relative scientific paths are retained where safe; private operational "
           "identifiers and source-export paths are omitted. The public availability manifest "
           "distinguishes included document sources from withheld evidence categories. "
           "Original records have not been regenerated to fill gaps, and no complete public "
           "provenance or universal experiment-coverage claim is made. Table sources remain "
           "under their tables.", ""]
    def split_items(body: str) -> list[str]:
        # split on ";" or "," outside `code`, drop "source(s):" prefixes
        parts, cur, code = [], "", False
        for ch in body:
            if ch == "`":
                code = not code
            if ch in ";," and not code:
                parts.append(cur); cur = ""
            else:
                cur += ch
        parts.append(cur)
        clean = []
        for p in parts:
            p = re.sub(r"^\s*(?:and\s+)?(?:sources?:\s*)?", "", p).strip().rstrip(".").strip()
            if not p:
                continue
            # a qualifier split off its path ("x.tex, Table y", "x.json, corrected")
            # is not a source of its own: keep it on the preceding item
            if not _PATHLIKE.search(p):
                if clean:
                    clean[-1] += f", {p}"
                continue
            if p.startswith("`"):
                clean.append(p)
            elif " " not in p:
                clean.append(f"`{p}`" if re.search(r"[/_]", p) else p)
            else:  # a path inside a phrase: code-format the path so it can break
                parts = re.split(r"(`[^`]*`)", p)
                clean.append("".join(x if x.startswith("`") else
                                     re.sub(r"(?<![\w`])([\w.\-]+/[\w./\-]*\w)", r"`\1`", x)
                                     for x in parts))
        return clean

    n, chapter_head, emitted_head = 0, None, None
    for heading, bodies in EVIDENCE:
        title = re.sub(r"^#+\s*", "", heading)
        if heading.startswith("# "):
            mm = re.match(r"(?:Appendix\s+)?([A-Z]|\d+)\.\s+(.*)", title)
            if mm:
                kind = "Appendix" if mm.group(1).isalpha() else "Chapter"
                title = f"{kind} {mm.group(1)}: {mm.group(2)}"
            chapter_head = title
        if not bodies:
            continue
        seen, items = set(), []
        for b in bodies:
            for part in split_items(b):
                if part not in seen:
                    seen.add(part)
                    items.append(part)
        if chapter_head != emitted_head:
            out += [f"## {chapter_head}", ""]
            emitted_head = chapter_head
        if not heading.startswith("# "):
            # Keep the long campaign evidence heading with its first source.
            if title.startswith("8.4 Replacement-scope lanes completed"):
                out += [r"`\Needspace{8\baselineskip}`{=latex}", ""]
            out += [f"**{title}**", ""]
        out += [f"- {x}" for x in items] + [""]
        n += len(items)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return n


def pandoc_md_to_tex(md_path: str, tex_path: str) -> bool:
    """Convert a chapter markdown file to a LaTeX fragment."""
    if not shutil.which("pandoc"):
        return False
    tmp = None
    try:
        with open(md_path, encoding="utf-8") as fh:
            raw = fh.read()
        # Keep the chapter title; LaTeX supplies its number.
        body, _ = _sources_to_footnotes(raw)
        body = _table_leadins(_strip_heading_numbers(body))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".numbered.md", dir=os.path.dirname(md_path),
            encoding="utf-8", delete=False,
        ) as fh:
            tmp = fh.name
            fh.write(body)
        # --natbib emits citations for the IEEE-style reference list.
        subprocess.run(
            ["pandoc", tmp, "-t", "latex", "--top-level-division=chapter",
             "--natbib", "-o", tex_path],
            check=True, capture_output=True, timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        err = getattr(exc, "stderr", b"") or b""
        detail = err.decode(errors="replace") if err else str(exc)
        print(f"  pandoc FAILED for {os.path.basename(md_path)}: "
              f"{detail[:200]}", file=sys.stderr)
        return False
    finally:
        if tmp is not None:
            os.remove(tmp)


def available_figures() -> set[str]:
    if not os.path.isdir(FIGDIR):
        return set()
    return {f[:-4] for f in os.listdir(FIGDIR)
            if f.endswith(".pdf") and os.path.isfile(os.path.join(FIGDIR, f))
            and os.path.getsize(os.path.join(FIGDIR, f)) > 0}


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
        safe_caption = caption.replace("\n", " ")
        # short caption (first sentence) keeps the List of Figures readable
        short = re.split(r"(?<=[.])\s", safe_caption, maxsplit=1)[0].rstrip(".")
        block = f"\n\n\\fig{{{name}}}\n{{{short}}}\n{{{safe_caption}}}\n{{{label}}}\n\n"
        ref = f" (Figure~\\ref{{{label}}})"
        placed = False
        for anchor in anchors:
            # find a heading line containing the anchor
            pat = re.compile(r"^(\\(?:chapter|section|subsection)\*?\{[^}]*"
                             + r"\s+".join(map(re.escape, anchor.split())) + r"[^}]*\})", re.M | re.I)
            # the chapter title may also contain the anchor and be followed
            # directly by a section, so try every matching heading in turn
            for m in pat.finditer(tex):
                # cite the figure at the end of the first complete prose
                # paragraph of the section, then place the float after it
                nxt = re.compile(r"^\\(?:chapter|section)\*?\{", re.M).search(tex, m.end())
                limit = nxt.start() if nxt else len(tex)
                pos = tex.find("\n\n", m.end())
                target = None
                while pos != -1 and pos < limit:
                    end = tex.find("\n\n", pos + 2)
                    end = end if end != -1 and end <= limit else limit
                    para = tex[pos:end].strip()
                    if (para.endswith(".")
                            and not re.match(r"\\(begin|fig|section|subsection|label)\b", para)
                            and "\\end{" not in para[-40:]):
                        target = (pos, end)
                        break
                    pos = tex.find("\n\n", pos + 2)
                if target is None:
                    continue
                pos, end = target
                para = tex[pos:end].rstrip()
                prev = re.search(r" \(Figures?~\\ref\{([^}]*)\}(?: and~\\ref\{[^}]*\})*\)\.$", para)
                if prev:                     # merge with a reference already there
                    labels = re.findall(r"\\ref\{([^}]*)\}", prev.group(0)) + [label]
                    refs = " and~".join(f"\\ref{{{l}}}" for l in labels)
                    para = para[:prev.start()] + f" (Figures~{refs})."
                else:
                    para = para[:-1] + ref + "."
                # keep figure order: go past floats already placed here
                tail = tex[end:]
                fm = re.match(r"(\s*\\fig\{[^}]*\}\n\{[^\n]*\}\n\{[^\n]*\}\n\{[^}]*\}\n)+", tail)
                skip = fm.end() if fm else 0
                tex = tex[:pos] + para + tail[:skip] + "\n" + block + tail[skip:]
                placed = True
                break
            if placed:
                break
        if not placed:
            tex = tex.rstrip() + f"\n\nFigure~\\ref{{{label}}} summarises this chapter.\n" + block
        inserted += 1
    return tex, inserted, missing


def breakable_long_tokens(master: str) -> tuple[str, int]:
    """Rewrite over-wide \\texttt{...} arguments to \\longtok{...}.

    Pandoc emits code spans as \\texttt{...}. Long unbroken tokens (paths,
    SHA-256 hashes, HF/tinker URIs, receipt status constants) contain no
    spaces, so TeX cannot break them and the line runs past the right
    margin. \\longtok wraps the content in \\seqsplit so it may break at
    any character.

    Only tokens whose *longest unbroken run* exceeds THRESHOLD chars are
    rewritten: short code spans keep plain \\texttt (no visual change) and
    spans containing spaces or display math are left untouched, since
    \\seqsplit would mangle them.

    Brace matching is done by depth counting so nested groups such as
    \\texttt{a{b}c} survive intact.
    """
    THRESHOLD = 28          # chars; ~the width of the text column in lmtt
    # Inside narrow table columns even short identifiers overflow, so use a
    # much lower bar there. \_ renders as a wide underscore and cannot be
    # broken, so a 15-char token like CLOSED\_EXTERNAL is already too wide
    # for a 9-column table cell.
    TABLE_THRESHOLD = 8
    out: list[str] = []
    i, n = 0, len(master)
    rewritten = 0
    OPEN = "\\texttt{"

    # Pre-compute which character ranges lie inside a longtable/tabular body
    # so the threshold can be lowered there.
    in_table = bytearray(n)
    for tm in re.finditer(
            r"\\begin\{(?:longtable|tabular)\}.*?\\end\{(?:longtable|tabular)\}",
            master, re.S):
        for k in range(tm.start(), tm.end()):
            in_table[k] = 1

    while True:
        j = master.find(OPEN, i)
        if j == -1:
            out.append(master[i:])
            break
        out.append(master[i:j])

        # walk forward to the matching close brace
        k = j + len(OPEN)
        depth = 1
        while k < n and depth:
            ch = master[k]
            if ch == "\\":          # skip escaped char (e.g. \_ \{ )
                k += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1

        if depth != 0:              # unbalanced; leave as-is
            out.append(master[j:])
            break

        arg = master[j + len(OPEN):k]      # inner content
        # seqsplit is byte-oriented under pdfLaTeX; preserve other Unicode
        # in ordinary texttt and render truncated identifiers with ASCII dots.
        arg = arg.replace("…", "...")
        longest = max((len(t) for t in re.split(r"\s+", arg)), default=0)
        has_space = bool(re.search(r"\s", arg))
        has_math = "$" in arg

        thr = TABLE_THRESHOLD if in_table[j] else THRESHOLD
        # escaped chars render wider than their source length
        longest_rendered = longest - 2 * arg.count("\\_")

        if has_math or not arg.isascii():
            out.append(master[j:k + 1])
        elif longest_rendered > thr and not has_space:
            # pure long token (path/hash/URI): seqsplit breaks it anywhere
            out.append(f"\\longtok{{{arg}}}")
            rewritten += 1
        elif longest > thr and has_space:
            # Spaced content (JSON blob, or a token containing escaped spaces
            # such as  model.sampler\_path\ =\ withheld-sampler-route  ). seqsplit would
            # destroy the spacing, so instead insert explicit zero-width break
            # points after the delimiter characters that appear inside long
            # runs. Covers JSON punctuation, path separators, RFC-3986 URI
            # scheme separators, hyphens (UUIDs / slugs) and underscore
            # escapes, which is where these tokens can legally wrap.
            patched = re.sub(
                r'(&quot;|"|,|:|/|\\}|\[|\]|=|-|\+|@|~|%|\\_)(?=[^\s])',
                lambda mm: mm.group(1) + "\\allowbreak{}",
                arg)
            out.append(f"\\texttt{{{patched}}}")
            rewritten += 1
        else:
            out.append(master[j:k + 1])

        i = k + 1

    return "".join(out), rewritten


def fix_table_widths(master: str) -> tuple[str, int]:
    """Correct pandoc's tabcolsep arithmetic in longtable column specs.

    Pandoc emits each column as
        >{\\raggedright\\arraybackslash}p{(\\linewidth - N\\tabcolsep) * \\real{f}}
    where N should be 2 * (number of columns) -- two \\tabcolsep per column.
    In practice pandoc under-counts by 2 on every table in this corpus
    (e.g. a 9-column table gets N=16 instead of 18), which makes the table
    wider than \\linewidth and pushes the last column past the right margin
    (measured: the appendix run-registry tables overflowed by ~11pt).

    This pass recomputes N = 2 * ncols for every longtable whose spec is
    wrong. Tables that already agree are left untouched.
    """
    LT = re.compile(r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}", re.S)
    MULT = re.compile(r"\\(?:linewidth|columnwidth)\s*-\s*(\d+)\\tabcolsep")
    fixed = 0

    def repl(m: re.Match) -> str:
        nonlocal fixed
        spec = m.group(1)
        ncols = spec.count("p{")
        gaps = MULT.findall(spec)
        if not gaps or ncols == 0:
            return m.group(0)
        used = int(gaps[0])
        need = 2 * ncols
        if used == need:
            return m.group(0)
        new_spec = MULT.sub(
            lambda mm: f"\\linewidth - {need}\\tabcolsep", spec)
        fixed += 1
        return "\\begin{longtable}[]{@{}" + new_spec + "@{}}"

    return LT.sub(repl, master), fixed


def fix_wide_tables(master: str) -> tuple[str, int]:
    """Shrink padding and font for many-column tables so they fit the margin.

    Even with correct column-width arithmetic, a 9- or 10-column table has
    so little room per cell that identifiers like CLOSED\\_EXTERNAL cannot fit
    on a line and the row pokes past the right margin (measured: 543.7pt
    against a 523.3pt limit). Wrapping the table in \\scriptsize and halving
    \\tabcolsep reclaims enough width to fit (measured: 518.2pt).

    Only tables with >= 4 columns are touched. The trigger is column COUNT,
    not column width, because a 4-column table whose last column holds
    21-character status constants (e.g. Table 13.3, REBUILD\\_READY\\_LAUNCH\\_)
    overflows just as badly as a 9-column one -- measured 602.1pt on a
    595.3pt page, i.e. off the paper. Narrower tables are left at natural size.
    """
    MIN_COLS = 4
    fixed = 0

    def repl(m: re.Match) -> str:
        nonlocal fixed
        whole = m.group(0)
        spec = m.group(1)
        ncols = spec.count("p{")
        if ncols < MIN_COLS:
            return whole
        fixed += 1
        return ("\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}\n"
                + whole + "\n\\endgroup")

    pat = re.compile(
        r"\\begin\{longtable\}\[\]\{@\{\}(.*?)@\{\}\}.*?\\end\{longtable\}",
        re.S)
    return pat.sub(repl, master), fixed


def break_bare_identifiers(master: str) -> tuple[str, int]:
    """Add break opportunities to long bare identifiers anywhere in the source.

    Some status constants appear as PLAIN TEXT -- not inside \\texttt{} --
    either in running prose or as a bare table cell (e.g. Table 13.3 lists
    REBUILD\\_READY\\_LAUNCH\\_PENDING with no \\texttt wrapper). Those cannot
    break at all and ran off the paper (measured: 596-602pt on a 595.3pt
    page). Insert a zero-width break after each \\_ in any run of >= 3
    underscore-escaped segments.

    This deliberately DOES cover longtable bodies. The earlier version skipped
    them, on the assumption that the \\texttt pass handled table cells -- but
    bare (non-\\texttt) identifiers in table cells fell through the gap
    between the two passes and were the last remaining source of off-paper
    text. \\allowbreak inside a p{} column is harmless: it only offers a
    break, it does not force one.
    """
    MIN_SEGMENTS = 3

    # identifier candidate: word chars joined by escaped underscores.
    # Guard (?<![\\{]) so we do not match inside an existing macro name or
    # immediately after a backslash.
    ident = re.compile(
        r"(?<![\\{])([A-Za-z0-9][A-Za-z0-9]*(?:\\_[A-Za-z0-9]+){"
        + str(MIN_SEGMENTS - 1) + r",})")

    counter = 0

    def rep(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        return m.group(1).replace("\\_", "\\_\\allowbreak{}")

    return ident.sub(rep, master), counter


def main() -> int:
    EVIDENCE.clear()
    engine = os.environ.get("THESIS_TEX_ENGINE") or ("tectonic" if shutil.which("tectonic") else "latexmk")
    if engine not in {"tectonic", "latexmk"}:
        raise ValueError("THESIS_TEX_ENGINE must be tectonic or latexmk")
    missing_tools = [name for name in ("pandoc", engine) if not shutil.which(name)]
    if missing_tools:
        print("missing required tools: " + ", ".join(missing_tools), file=sys.stderr)
        return 1

    have = available_figures()
    chapters = [name for name, _ in CHAPTERS + APPENDICES] + ["ch_back_coverage_audit.md",
                                                            "ch_back_provenance.md"]
    required = chapters + ["preamble.tex", "frontmatter.tex", "references.bib",
                           "figures/pes_logo.png"]
    missing = [name for name in required
               if not os.path.isfile(os.path.join(HERE, name))
               or os.path.getsize(os.path.join(HERE, name)) == 0]
    short = [name for name in chapters if name not in missing
             and os.path.getsize(os.path.join(HERE, name)) < 400]
    missing_figures = {name for chapter in chapters
                       for name, *_ in FIGURES.get(chapter, []) if name not in have}
    if missing:
        print("missing/empty required inputs: " + ", ".join(missing), file=sys.stderr)
    if short:
        print("SUSPICIOUSLY SHORT chapters: " + ", ".join(short), file=sys.stderr)
    if missing_figures:
        print("figures referenced but NOT built: " + ", ".join(sorted(missing_figures))
              + "; run compile_figures.py --force first", file=sys.stderr)
    if missing or short or missing_figures:
        return 1

    os.makedirs(BUILD, exist_ok=True)
    print(f"figures available: {len(have)}")

    preamble = open(os.path.join(HERE, "preamble.tex"), encoding="utf-8").read()
    # numeric, sorted citations for the IEEE reference list; natbib must load
    # before hyperref
    preamble = preamble.replace(
        "\\usepackage[hidelinks]{hyperref}",
        "\\usepackage{needspace}\n\\usepackage[numbers,sort&compress]{natbib}\n\\usepackage[hidelinks]{hyperref}", 1)
    # \fig{name}{short caption}{caption}{label}
    preamble = re.sub(r"\\newcommand\{\\fig\}\[3\]\{%.*?\\end\{figure\}\}",
                      lambda _: ("\\newcommand{\\fig}[4]{%\n"
                                 "  \\begin{figure}[htbp]\\centering\n"
                                 "  \\includegraphics[width=0.94\\linewidth]{figures/#1.pdf}%\n"
                                 "  \\caption[#2]{#3}\\label{#4}%\n"
                                 "  \\end{figure}}"),
                      preamble, count=1, flags=re.S)
    # front matter in roman numerals; arabic restarts at Part I / Chapter 1
    preamble = preamble.replace("\\begin{document}", "\\begin{document}\n\\pagenumbering{roman}", 1)
    pieces = [preamble]
    pieces.append(open(os.path.join(HERE, "frontmatter.tex"), encoding="utf-8").read())
    pieces.append("\n\\cleardoublepage\n\\pagenumbering{arabic}\n\\hypersetup{pageanchor=true}\n")

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
    # IEEE-style numbered reference list, before the appendices (PES template)
    pieces.append("\n\\cleardoublepage\n\\phantomsection\n"
                  "\\addcontentsline{toc}{chapter}{References}\n"
                  f"\\bibliographystyle{{{os.environ.get('THESIS_BIB_STYLE', 'IEEEtran')}}}\n\\bibliography{{references}}\n")
    pieces.append("\n\\appendix\n")
    for md_name, part in APPENDICES:
        emit(md_name, part)
    n_ev = write_evidence_map(EVIDENCE_MAP)
    EVIDENCE.clear()
    emit(EVIDENCE_MAP, None)
    emit("ch_back_coverage_audit.md", None)
    # generated by tools/audit_thesis_numbers.py
    emit("ch_back_provenance.md", None)
    print(f"evidence map: {n_ev} source entries")

    pieces.append("\n\\end{document}\n")
    master = "\n".join(pieces)
    # Combining accents in prose (e.g. p-hat) are absent from the report's
    # T1 text font. Use explicit TeX accents in both text and math contexts.
    for accent, command in (("\u0302", "hat"), ("\u0304", "bar")):
        master = re.sub(r"([A-Za-zτλ])" + accent,
                        lambda m: "\\ensuremath{\\" + command + "{" + m[1] + "}}",
                        master)

    # ---- long-token pass: make over-wide \texttt args breakable ----------
    # Paths, hashes, URIs and receipt constants emitted by pandoc as
    # \texttt{...} contain no spaces and cannot break, so they run past the
    # right margin (measured: 72/217 pages overflowed before this pass).
    # Route the long ones through \longtok = \texttt{\seqsplit{...}}.
    # Order matters: the bare-identifier pass must run BEFORE the \\texttt
    # pass so that bare table cells (which the \\texttt pass cannot see) get
    # their break opportunities too.
    master, n_bare = break_bare_identifiers(master)
    print(f"bare-identifier pass: added breaks to {n_bare} identifier(s)")

    master, n_longtok = breakable_long_tokens(master)
    print(f"long-token pass: rewrote {n_longtok} \\texttt argument(s) to \\longtok")

    master, n_tables = fix_table_widths(master)
    # "Qwen/Qwen3.5-4B" has no break point before its hyphen; allow one after
    # the org slash in plain text (not inside \longtok/\texttt arguments)
    # TeX never breaks the first word of a paragraph, so a long first token in
    # a narrow cell overflows into the next column; zero glue lifts that.
    master = master.replace(r">{\raggedright\arraybackslash}p{",
                            r">{\raggedright\arraybackslash\hspace{0pt}}p{")
    org = re.compile(r"(?<![{\w/])((?:Qwen|meta-llama|nvidia|deepseek-ai|mistralai|google)/)(?=\w)")
    master = re.sub(r"\\begin\{longtable\}.*?\\end\{longtable\}",
                    lambda m: org.sub(r"\1\\allowbreak{}", m.group(0)), master, flags=re.S)
    print(f"table-width pass: corrected {n_tables} longtable column spec(s)")

    master, n_wide = fix_wide_tables(master)
    print(f"wide-table pass: shrank {n_wide} many-column table(s)")

    if missing_ch:
        print(f"  MISSING/FAILED chapters: {', '.join(missing_ch)}", file=sys.stderr)
    if empty_ch:
        print(f"  SUSPICIOUSLY SHORT: {', '.join(empty_ch)}", file=sys.stderr)
    if all_missing:
        print(f"  figures referenced but NOT built: {', '.join(sorted(all_missing))}",
              file=sys.stderr)

    if missing_ch or empty_ch or all_missing:
        return 1

    with open(MASTER, "w", encoding="utf-8") as fh:
        fh.write(master)

    words = len(re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", master).split())
    print(f"wrote {MASTER}  (~{words:,} words, {total_figs} figures injected)")
    print(f"compiling with {engine} ...")
    try:
        # Keep output isolated from old build PDFs, on the publication filesystem.
        with tempfile.TemporaryDirectory(prefix=".thesis-", dir=os.path.dirname(PDF)) as outdir:
            command = (["tectonic", "-X", "compile", MASTER, "--outdir", outdir,
                        "--keep-logs", "--only-cached", "--untrusted"] if engine == "tectonic" else
                       ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        "-file-line-error", "-no-shell-escape", f"-outdir={outdir}", MASTER])
            r = subprocess.run(command, capture_output=True, timeout=1200, cwd=HERE)
            out = (r.stdout + r.stderr).decode(errors="replace")
            print("\n".join(out.strip().splitlines()[-18:]))
            log = os.path.join(outdir, "thesis_master.log")
            if os.path.isfile(log):
                shutil.copyfile(log, os.path.join(BUILD, "thesis_master.log"))
            if r.returncode != 0:
                print(f"{engine} failed (exit {r.returncode})", file=sys.stderr)
                return 1
            for suffix in ("aux", "toc", "lot", "lof", "bbl", "blg"):
                artifact = os.path.join(outdir, "thesis_master." + suffix)
                if os.path.isfile(artifact):
                    shutil.copyfile(artifact, os.path.join(BUILD, "thesis_master." + suffix))
            built = os.path.join(outdir, "thesis_master.pdf")
            if not os.path.isfile(built) or os.path.getsize(built) == 0:
                print("no nonempty PDF produced", file=sys.stderr)
                return 1
            os.replace(built, PDF)
        print(f"OK -> {PDF}  ({os.path.getsize(PDF)/1024:.0f} KB)")
        return 0
    except subprocess.TimeoutExpired:
        print(f"{engine} timed out", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"thesis build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
