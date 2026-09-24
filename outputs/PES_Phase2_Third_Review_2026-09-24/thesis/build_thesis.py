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
        longest = max((len(t) for t in re.split(r"\s+", arg)), default=0)
        has_space = bool(re.search(r"\s", arg))
        has_math = "$" in arg

        thr = TABLE_THRESHOLD if in_table[j] else THRESHOLD
        # escaped chars render wider than their source length
        longest_rendered = longest - 2 * arg.count("\\_")

        if has_math:
            out.append(master[j:k + 1])
        elif longest_rendered > thr and not has_space:
            # pure long token (path/hash/URI): seqsplit breaks it anywhere
            out.append("\\longtok{%s}" % arg)
            rewritten += 1
        elif longest > thr and has_space:
            # Spaced content (JSON blob, or a token containing escaped spaces
            # such as  model.sampler\_path\ =\ tinker://...  ). seqsplit would
            # destroy the spacing, so instead insert explicit zero-width break
            # points after the delimiter characters that appear inside long
            # runs. Covers JSON punctuation, path separators, RFC-3986 URI
            # scheme separators, hyphens (UUIDs / slugs) and underscore
            # escapes, which is where these tokens can legally wrap.
            patched = re.sub(
                r'(&quot;|"|,|:|/|\\}|\[|\]|=|-|\+|@|~|%|\\_)(?=[^\s])',
                lambda mm: mm.group(1) + "\\allowbreak{}",
                arg)
            out.append("\\texttt{%s}" % patched)
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
    MULT = re.compile(r"\\linewidth\s*-\s*(\d+)\\tabcolsep")
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
            lambda mm: "\\linewidth - %d\\tabcolsep" % need, spec)
        fixed += 1
        return "\\begin{longtable}[]{@{}%s@{}}" % new_spec

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
        r"(?<![\\{])([A-Za-z0-9][A-Za-z0-9]*(?:\\_[A-Za-z0-9]+){%d,})"
        % (MIN_SEGMENTS - 1))

    counter = 0

    def rep(m: re.Match) -> str:
        nonlocal counter
        counter += 1
        return m.group(1).replace("\\_", "\\_\\allowbreak{}")

    return ident.sub(rep, master), counter


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
    print(f"table-width pass: corrected {n_tables} longtable column spec(s)")

    master, n_wide = fix_wide_tables(master)
    print(f"wide-table pass: shrank {n_wide} many-column table(s)")

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
