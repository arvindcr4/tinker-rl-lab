#!/usr/bin/env python3
"""
Build the PES M.Tech Phase 2 THIRD REVIEW deck for the Tinker RL Lab project.

Submission: M.Tech Project, "Phase 2 Third Review" (40 marks), due 2026-09-24.
Deliverable: PPT/PPTX, <= 100 MB.

Every number on these slides is sourced from a repository file; see the SRC
comments. The authoritative consolidated ledger is
outputs/E1_E14_FINAL_RESULTS_2026-09-19.md, gated by the deterministic check
outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json (11/11 PASS).

Design rule carried from build_review_pptx.py: original-contract and
replacement-scope numbers are NEVER pooled, and no cross-suite average or
baseline-gain claim is made anywhere.

Usage:
    python build_third_review_pptx.py            # writes the deck next to this file
    python build_third_review_pptx.py --check    # also round-trip loads the result
"""
from __future__ import annotations

import os
import sys

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "ArvindCR_Phase2_Third_Review_2026-09-24_v2.pptx")

# ---------------------------------------------------------------------------
# Palette / typography  (matches the established PES review-deck style)
# ---------------------------------------------------------------------------
PES_BLUE = RGBColor(0x1F, 0x4E, 0x79)     # dark navy title
ACCENT = RGBColor(0x2E, 0x74, 0xB5)       # lighter blue rule / accents
INK = RGBColor(0x22, 0x22, 0x22)          # body text
MUTED = RGBColor(0x5A, 0x5A, 0x5A)        # captions / footer
LIGHT_BG = RGBColor(0xF2, 0xF5, 0xF9)     # panel fill
GREEN = RGBColor(0x1E, 0x7D, 0x4F)        # complete
AMBER = RGBColor(0xB0, 0x6A, 0x00)        # partial
RED = RGBColor(0xA6, 0x2B, 0x2B)          # blocked
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RULE = RGBColor(0xD0, 0xD9, 0xE4)
FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
M = Inches(0.62)                          # left/right margin
FOOTER = "Arvind C R  ·  PES2PGE24DS140  ·  Phase 2 Third Review  ·  24 September 2026"
TITLE_LABEL = "Phase 2 Third Review  ·  Tinker RL Lab"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def _blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rect(slide, x, y, w, h, fill=None, line=None, line_w=Pt(0.75)):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w
    shp.shadow.inherit = False
    return shp


def _text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
          space_after=Pt(4), line_spacing=1.0):
    """runs: list of (text, size_pt, bold, color[, italic]) or a plain string."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0

    if isinstance(runs, str):
        runs = [(runs, 14, False, INK)]

    first = True
    for item in runs:
        text, size, bold, color = item[0], item[1], item[2], item[3]
        italic = item[4] if len(item) > 4 else False
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = space_after
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = FONT
    return tb


BULLET_COLOR = ACCENT


def _bullets(slide, x, y, w, h, items, size=14, gap=6, line_spacing=1.05):
    """items: list of str, or (text, color) / (text, color, bold)."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for it in items:
        if isinstance(it, str):
            text, color, bold = it, INK, False
        elif len(it) == 2:
            text, color, bold = it[0], it[1], False
        else:
            text, color, bold = it[0], it[1], it[2]
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = "•  " + text
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
        r.font.name = FONT
    return tb


def _header(slide, title, subtitle=None, kicker=None):
    """Standard slide header: kicker label, navy title, accent rule."""
    y = Inches(0.34)
    if kicker:
        _text(slide, M, y, SLIDE_W - 2 * M, Inches(0.26),
              [(kicker.upper(), 10.5, True, ACCENT)], space_after=Pt(0))
        y += Inches(0.26)
    _text(slide, M, y, SLIDE_W - 2 * M, Inches(0.52),
          [(title, 26, True, PES_BLUE)], space_after=Pt(0))
    y += Inches(0.56)
    _rect(slide, M, y, SLIDE_W - 2 * M, Pt(1.6), fill=ACCENT)
    y += Inches(0.12)
    if subtitle:
        _text(slide, M, y, SLIDE_W - 2 * M, Inches(0.30),
              [(subtitle, 12.5, False, MUTED)], space_after=Pt(0))
        y += Inches(0.34)
    return y + Inches(0.10)


def _footer(slide, n):
    _rect(slide, M, SLIDE_H - Inches(0.50), SLIDE_W - 2 * M, Pt(0.75), fill=RULE)
    _text(slide, M, SLIDE_H - Inches(0.42), Inches(10.2), Inches(0.26),
          [(FOOTER, 9, False, MUTED)], space_after=Pt(0))
    _text(slide, SLIDE_W - M - Inches(0.9), SLIDE_H - Inches(0.42), Inches(0.9),
          Inches(0.26), [(str(n), 9, True, MUTED)], align=PP_ALIGN.RIGHT,
          space_after=Pt(0))


def _src(slide, text):
    """Small source/evidence line pinned to the bottom of the content area."""
    _text(slide, M, SLIDE_H - Inches(0.76), SLIDE_W - 2 * M, Inches(0.24),
          [("Sources: " + text, 8.5, False, MUTED, True)], space_after=Pt(0))


STATUS_COLOR = {
    "COMPLETE": GREEN,
    "REPLACEMENT": GREEN,
    "TERMINAL": GREEN,
    "PARTIAL": AMBER,
    "BLOCKED": RED,
}


def _table(slide, x, y, w, rows, widths, size=11.5, header_size=11,
           row_h=Inches(0.34), status_col=None, header_h=Inches(0.40)):
    """Draw a simple, well-ruled table with a navy header row.

    rows[0] is the header. widths are relative fractions summing to 1.0.
    status_col: index of a column whose cell text maps to a status colour.
    """
    n_rows, n_cols = len(rows), len(rows[0])
    total = sum(widths)
    w_emu = int(w)
    col_w = [Emu(int(w_emu * frac / total)) for frac in widths]

    # header
    cx = x
    _rect(slide, x, y, w, header_h, fill=PES_BLUE)
    for j, cell in enumerate(rows[0]):
        _text(slide, cx + Inches(0.08), y + Inches(0.055), col_w[j] - Inches(0.16),
              header_h, [(str(cell), header_size, True, WHITE)],
              anchor=MSO_ANCHOR.MIDDLE, space_after=Pt(0))
        cx += col_w[j]
    ry = y + header_h

    for i, row in enumerate(rows[1:]):
        if i % 2 == 1:
            _rect(slide, x, ry, w, row_h, fill=LIGHT_BG)
        _rect(slide, x, ry, w, Pt(0.5), fill=RULE)
        cx = x
        for j, cell in enumerate(row):
            col = INK
            bold = False
            if status_col is not None and j == status_col:
                col = STATUS_COLOR.get(str(cell).upper(), INK)
                bold = True
            _text(slide, cx + Inches(0.08), ry, col_w[j] - Inches(0.16), row_h,
                  [(str(cell), size, bold, col)], anchor=MSO_ANCHOR.MIDDLE,
                  space_after=Pt(0))
            cx += col_w[j]
        ry += row_h
    _rect(slide, x, ry, w, Pt(0.75), fill=RULE)
    return ry


def _stat(slide, x, y, w, h, value, label, color=PES_BLUE, value_size=30):
    _rect(slide, x, y, w, h, fill=LIGHT_BG, line=RULE)
    _rect(slide, x, y, Pt(3.2), h, fill=color)
    _text(slide, x + Inches(0.18), y + Inches(0.16), w - Inches(0.30), Inches(0.56),
          [(value, value_size, True, color)], space_after=Pt(0))
    _text(slide, x + Inches(0.18), y + Inches(0.74), w - Inches(0.30), h - Inches(0.80),
          [(label, 10.5, False, MUTED)], space_after=Pt(0), line_spacing=1.05)


def _callout(slide, x, y, w, h, heading, body, color=ACCENT):
    _rect(slide, x, y, w, h, fill=LIGHT_BG, line=RULE)
    _rect(slide, x, y, Pt(3.2), h, fill=color)
    _text(slide, x + Inches(0.18), y + Inches(0.12), w - Inches(0.34), Inches(0.28),
          [(heading, 12, True, color)], space_after=Pt(0))
    _text(slide, x + Inches(0.18), y + Inches(0.44), w - Inches(0.34), h - Inches(0.58),
          [(body, 11.5, False, INK)], space_after=Pt(0), line_spacing=1.06)


# ---------------------------------------------------------------------------
# Sourced campaign facts  (see SRC: file paths in the slides)
# ---------------------------------------------------------------------------
MODEL_LINE = ("Qwen/Qwen3.6-35B-A3B  (base commit 995ad96e)  +  adapter "
              "arvindcr4/pavlov-portfolio-qwen36-seed809-stepfinal-tinker-cf0ad8c1 "
              "(commit 64444133), bfloat16")
HEADLINE = ("One Tinker-trained actor evaluated against 14 held-out agent and RL "
            "benchmark suites under strict receipt, budget and decontamination governance.")

# Lane, suite, verified status, headline number, coverage
LANES_A = [  # E1-E7
    ["Lane", "Suite", "Status", "Verified result", "Coverage"],
    ["E1", "SWE-bench Pro", "COMPLETE", "2/731 = 0.274% pass@1", "713/731 native"],
    ["E2", "FrontierSWE", "PARTIAL", "1/17 tasks; replay norm. 0.8628", "1/17"],
    ["E3", "SDAB (private)", "BLOCKED", "no result path without private bundle", "-"],
    ["E4", "BankerToolBench", "PARTIAL", "1/100 tasks; recovery 0.3115", "1/100"],
    ["E5", "APEX-Agents", "PARTIAL", "7/480 native-scored; prefix 0.050505", "7/480"],
    ["E6", "WebArena", "BLOCKED", "0/812 graded — AWS quota (1/16 vCPU)", "0/812"],
    ["E7", "BinaryAudit", "BLOCKED", "1/46 attempted; reward 0.0, no grade", "1/46"],
]

LANES_B = [  # E8-E14
    ["Lane", "Suite", "Status", "Verified result", "Coverage"],
    ["E8", "LAB-Bench (public)", "REPLACEMENT", "8 categories, all COMPLETE", "1967/1967"],
    ["E9", "MLE-bench", "PARTIAL", "40/75 natively graded; suite score null", "53.33%"],
    ["E10", "AgentDojo (benign)", "REPLACEMENT", "benign-utility evaluation complete", "97/97"],
    ["E11", "VerilogEval", "COMPLETE", "129/312 = 41.35% pass@1", "312/312"],
    ["E12", "AppBench", "BLOCKED", "deployment access unavailable", "-"],
    ["E13", "BALROG", "PARTIAL", "13/255 episodes", "13/255"],
    ["E14", "Omni-MATH", "REPLACEMENT", "51.31% official accuracy (2271/4428)", "4428/4428"],
]


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------
def slide_title(prs, n):
    s = _blank(prs)
    _rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, fill=WHITE)
    _rect(s, Inches(0), Inches(0), SLIDE_W, Inches(0.30), fill=PES_BLUE)
    _rect(s, Inches(0), Inches(0.30), SLIDE_W, Pt(2.5), fill=ACCENT)

    _text(s, M, Inches(1.55), SLIDE_W - 2 * M, Inches(0.34),
          [("PES UNIVERSITY  ·  M.TECH DATA SCIENCE & AI", 12, True, ACCENT)],
          space_after=Pt(0))
    _text(s, M, Inches(2.02), SLIDE_W - 2 * M, Inches(0.70),
          [("Phase 2 Third Review", 44, True, PES_BLUE)], space_after=Pt(0))
    _text(s, M, Inches(2.80), SLIDE_W - 2 * M, Inches(0.86),
          [("Tinker RL Lab — E1–E14 held-out benchmark evaluation of a "
            "Tinker-trained Qwen3.6-35B-A3B actor", 17, False, INK)],
          space_after=Pt(0), line_spacing=1.14)
    _rect(s, M, Inches(3.86), Inches(2.6), Pt(2.5), fill=ACCENT)

    _text(s, M, Inches(4.20), SLIDE_W - 2 * M, Inches(1.20),
          [("Arvind C R   ·   SRN PES2PGE24DS140", 14.5, True, INK),
           ("Project Guide:  Ramesh Prakash Guledgudd", 13, False, MUTED),
           ("Department of Computer Science  ·  24 September 2026", 13, False, MUTED)],
          space_after=Pt(7))
    _text(s, M, Inches(6.30), SLIDE_W - 2 * M, Inches(0.30),
          [("Every figure in this deck is bound to a surviving receipt and was "
            "re-verified by deterministic code checks (11/11 PASS).", 10.5, False, MUTED, True)],
          space_after=Pt(0))
    return s


def slide_agenda(prs, n):
    s = _blank(prs)
    y = _header(s, "Agenda", kicker="Phase 2 · Third Review")
    left = [
        "1.  Campaign scope and governance",
        "2.  The evaluated model and grader",
        "3.  E1–E7 results",
        "4.  E8–E14 results",
        "5.  Fully verified suites (E8, E10, E11, E14)",
        "6.  Verification and integrity apparatus",
    ]
    right = [
        "7.   Work completed since the second review",
        "8.   Runtime repairs and re-implementations",
        "9.   Externally blocked lanes — the honest boundary",
        "10. Evidence and reproduction pointers",
        "11. What the measurements establish",
        "12. Conclusions and next steps",
    ]
    _bullets(s, M, y, Inches(5.9), Inches(4.4), left, size=14, gap=13)
    _bullets(s, M + Inches(6.15), y, Inches(5.9), Inches(4.4), right, size=14, gap=13)
    _src(s, "this deck")
    _footer(s, n)
    return s


def slide_glance(prs, n):
    s = _blank(prs)
    y = _header(s, "Campaign at a glance",
                subtitle=HEADLINE,
                kicker="Scope and governance")
    w = (SLIDE_W - 2 * M - Inches(0.36)) / 3
    _stat(s, M, y, w, Inches(1.34), "14", "held-out benchmark suites, each graded by its own native evaluator")
    _stat(s, M + w + Inches(0.18), y, w, Inches(1.34), "4", "suites with a strictly complete replacement scope (E8, E10, E11, E14)")
    _stat(s, M + 2 * (w + Inches(0.18)), y, w, Inches(1.34), "11/11", "deterministic integrity checks PASS on 2026-09-19")
    y2 = y + Inches(1.56)
    _callout(s, M, y2, SLIDE_W - 2 * M, Inches(1.16), "Two rules held throughout",
             "Original-contract and replacement-scope numbers are never pooled. No cross-suite average, "
             "no baseline-gain claim, and no backend numerical-parity claim is made anywhere in this work.")
    _callout(s, M, y2 + Inches(1.32), SLIDE_W - 2 * M, Inches(1.16),
             "Honest partial and blocked lanes",
             "Lanes that could not be completed are reported as PARTIAL or BLOCKED with immutable evidence "
             "trails — not silently dropped. §9 lists every externally blocked lane and its reopen condition.",
             color=AMBER)
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md; outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json")
    _footer(s, n)
    return s


def slide_model(prs, n):
    s = _blank(prs)
    y = _header(s, "The evaluated model and its grader",
                subtitle="A single actor, preserved sampling conditions.",
                kicker="Method")
    _callout(s, M, y, SLIDE_W - 2 * M, Inches(1.10), "Model under evaluation", MODEL_LINE)
    y2 = y + Inches(1.26)
    _bullets(s, M, y2, SLIDE_W - 2 * M, Inches(2.4), [
        ("One actor only. No per-lane fine-tuning, no prompt tuning, no evaluator-specific adaptation.", INK),
        ("Native evaluators. Every suite is scored by its own upstream grader at a pinned revision — "
         "no substituted or re-implemented metric.", INK),
        ("Frozen sampling. Prompts, sampling parameters and preserved original outputs are unchanged from "
         "the run that produced them.", INK),
        ("Graded scopes kept separate. E8, E10 and the new E14 used the merged BF16 artifact; the retained "
         "E11 result came from the original Tinker sampler. No numerical parity between the two is claimed.", AMBER),
    ], size=13.5, gap=11)
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §1, §6")
    _footer(s, n)
    return s


def slide_results_a(prs, n):
    s = _blank(prs)
    y = _header(s, "Results — E1 to E7",
                subtitle="Original-contract suites. \"—\" means no usable graded partial exists.",
                kicker="Results")
    _table(s, M, y, SLIDE_W - 2 * M, LANES_A, [0.06, 0.21, 0.15, 0.42, 0.16],
           status_col=2, row_h=Inches(0.40))
    _text(s, M, SLIDE_H - Inches(1.02), SLIDE_W - 2 * M, Inches(0.30),
          [("E1 denominator note: 14 generation failures and 4 artifact losses remain in the denominator "
            "rather than being silently excluded.", 10, False, MUTED, True)], space_after=Pt(0))
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §3; outputs/UNBLOCK_CARRYOUT_2026-09-21.md")
    _footer(s, n)
    return s


def slide_results_b(prs, n):
    s = _blank(prs)
    y = _header(s, "Results — E8 to E14",
                subtitle="Replacement-scope lanes are labelled REPLACEMENT and never pooled with the above.",
                kicker="Results")
    _table(s, M, y, SLIDE_W - 2 * M, LANES_B, [0.06, 0.21, 0.17, 0.40, 0.16],
           status_col=2, row_h=Inches(0.40))
    _text(s, M, SLIDE_H - Inches(1.02), SLIDE_W - 2 * M, Inches(0.30),
          [("E9 note: the modal_streaming arm produced 40/75 native grades; the merged-vLLM arm is a "
            "separate arm with one valid grade and is never merged into the same figure.", 10, False, MUTED, True)],
          space_after=Pt(0))
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §2, §4")
    _footer(s, n)
    return s


def slide_e8(prs, n):
    s = _blank(prs)
    y = _header(s, "E8 — LAB-Bench (public split)",
                subtitle="All 1,967 questions graded across 8 categories. Strictly complete.",
                kicker="Verified suite 1 of 4")
    cats = [("TableQA", "0.787"), ("FigQA", "0.514"), ("ProtocolQA", "0.250"),
            ("SuppQA", "0.183"), ("LitQA2", "0.176"), ("DbQA", "0.131"),
            ("SeqQA", "0.033"), ("CloningScenarios", "0.000")]
    # horizontal bars
    bx, by = M + Inches(1.65), y + Inches(0.06)
    bw, bh, gap = Inches(6.4), Inches(0.30), Inches(0.155)
    maxv = 0.80
    for i, (name, val) in enumerate(cats):
        ry = by + i * (bh + gap)
        _text(s, M, ry - Inches(0.015), Inches(1.55), bh, [(name, 11.5, False, INK)],
              anchor=MSO_ANCHOR.MIDDLE, space_after=Pt(0))
        _rect(s, bx, ry, bw, bh, fill=LIGHT_BG, line=RULE)
        frac = float(val) / maxv
        if frac > 0:
            _rect(s, bx, ry, Inches(bw.inches * frac), bh,
                  fill=ACCENT if float(val) >= 0.15 else RGBColor(0x9D, 0xBC, 0xDC))
        _text(s, bx + bw + Inches(0.12), ry - Inches(0.015), Inches(0.75), bh,
              [(val, 11.5, True, INK)], anchor=MSO_ANCHOR.MIDDLE, space_after=Pt(0))
    px = M + Inches(9.0)
    _callout(s, px, y, SLIDE_W - M - px, Inches(1.62), "What this shows",
             "Accuracy is strongly category-dependent. Structured-table reading is the relative "
             "strength; sequence-level and cloning-scenario reasoning are near floor.")
    _callout(s, px, y + Inches(1.78), SLIDE_W - M - px, Inches(1.62), "Scope limit",
             "Public split only. This is a capability profile, not a leaderboard position — "
             "no baseline comparison is claimed.", color=AMBER)
    _text(s, M, SLIDE_H - Inches(1.02), SLIDE_W - 2 * M, Inches(0.30),
          [("Bars are scaled to a 0.80 axis. Each category uses its own complete native denominator.",
            10, False, MUTED, True)], space_after=Pt(0))
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §2")
    _footer(s, n)
    return s


def slide_e10_e11(prs, n):
    s = _blank(prs)
    y = _header(s, "E10 AgentDojo  ·  E11 VerilogEval",
                subtitle="Two further strictly complete scopes.", kicker="Verified suites 2 and 3 of 4")
    half = (SLIDE_W - 2 * M - Inches(0.24)) / 2

    _rect(s, M, y, half, Inches(3.5), fill=WHITE, line=RULE)
    _text(s, M + Inches(0.20), y + Inches(0.16), half - Inches(0.40), Inches(0.40),
          [("E10  ·  AgentDojo", 17, True, PES_BLUE)], space_after=Pt(0))
    _text(s, M + Inches(0.20), y + Inches(0.62), half - Inches(0.40), Inches(0.62),
          [("90.72%", 34, True, GREEN), ("benign task utility  ·  88 / 97 tasks succeeded", 11.5, False, MUTED)],
          space_after=Pt(4))
    _bullets(s, M + Inches(0.20), y + Inches(1.42), half - Inches(0.40), Inches(1.9), [
        ("97/97 tasks evaluated — complete benign-utility scope.", INK),
        ("This measures helpfulness on benign tasks only.", INK),
        ("Attack robustness and adversarial security are NOT measured by this scope.", AMBER),
    ], size=11.5, gap=9)

    x2 = M + half + Inches(0.24)
    _rect(s, x2, y, half, Inches(3.5), fill=WHITE, line=RULE)
    _text(s, x2 + Inches(0.20), y + Inches(0.16), half - Inches(0.40), Inches(0.40),
          [("E11  ·  VerilogEval", 17, True, PES_BLUE)], space_after=Pt(0))
    _text(s, x2 + Inches(0.20), y + Inches(0.62), half - Inches(0.40), Inches(0.62),
          [("41.35%", 34, True, GREEN), ("pass@1  ·  129 / 312", 11.5, False, MUTED)],
          space_after=Pt(4))
    _bullets(s, x2 + Inches(0.20), y + Inches(1.42), half - Inches(0.40), Inches(1.9), [
        ("Two native framings of 156 tasks: 67/156 code-completion, 62/156 spec-to-RTL.", INK),
        ("Retained result from the original Tinker sampler (not the merged artifact).", INK),
        ("This result does not establish improvement over any baseline.", AMBER),
    ], size=11.5, gap=9)

    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §2")
    _footer(s, n)
    return s


def slide_e14(prs, n):
    s = _blank(prs)
    y = _header(s, "E14 — Omni-MATH",
                subtitle="Official accuracy reproduced under the native scoring protocol.",
                kicker="Verified suite 4 of 4")
    w = (SLIDE_W - 2 * M - Inches(0.36)) / 3
    _stat(s, M, y, w, Inches(1.30), "51.31%", "official accuracy — 2,271 correct of 4,428 in the scorer's own denominator")
    _stat(s, M + w + Inches(0.18), y, w, Inches(1.30), "4426/4428", "accepted reports (99.95% parser coverage)")
    _stat(s, M + 2 * (w + Inches(0.18)), y, w, Inches(1.30), "4428/4428", "dispositions recorded — 4,426 accepted + 2 recorded skips")
    y2 = y + Inches(1.52)
    _callout(s, M, y2, SLIDE_W - 2 * M, Inches(1.30), "The two excluded rows, resolved",
             "Both Omni-Judge outputs were truncated before any \"## Equivalence Judgement\" section, so the "
             "judge emitted no verdict. The native scorer's omission of missing-verdict rows is correct official "
             "behaviour, not a parser bug. Score impact: none — both rows already sit in the denominator as "
             "not-correct. E14 therefore closes terminal-complete with no re-judge, since re-judging would "
             "deviate from the official scorer with no score effect.")
    _text(s, M, y2 + Inches(1.48), SLIDE_W - 2 * M, Inches(0.30),
          [("Replacement scope for the original FrontierMath contract, which remains externally blocked (§9).",
            10.5, False, MUTED, True)], space_after=Pt(0))
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §4, §9.1")
    _footer(s, n)
    return s


def slide_e1(prs, n):
    s = _blank(prs)
    y = _header(s, "E1 — SWE-bench Pro",
                subtitle="The largest original-contract run in the campaign, reported in full.",
                kicker="Original-contract detail")
    w = (SLIDE_W - 2 * M - Inches(0.36)) / 3
    _stat(s, M, y, w, Inches(1.30), "0.274%", "pass@1 — 2 of 731 terminal generations resolved")
    _stat(s, M + w + Inches(0.18), y, w, Inches(1.30), "731", "terminal generations; 713 reached native evaluation")
    _stat(s, M + 2 * (w + Inches(0.18)), y, w, Inches(1.30), "18", "generation failures (14) and artifact losses (4) retained in the denominator")
    y2 = y + Inches(1.52)
    _callout(s, M, y2, SLIDE_W - 2 * M, Inches(1.28), "Why the denominator is reported this way",
             "Failures are not quietly dropped to inflate the rate. Keeping all 731 in the denominator means "
             "the reported figure is a lower bound on the true resolve rate under these sampling conditions — "
             "a deliberately conservative choice.")
    _callout(s, M, y2 + Inches(1.44), SLIDE_W - 2 * M, Inches(1.16), "Status and next step",
             "A $16 wave-10 re-implementation is complete and verified 10/10 offline against the lost-suite "
             "record, replacing execution source lost with an earlier workspace deletion. The launch itself "
             "remains lead-authorised and pending.", color=AMBER)
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §3, §9.5")
    _footer(s, n)
    return s


def slide_verification(prs, n):
    s = _blank(prs)
    y = _header(s, "Verification and integrity",
                subtitle="Nothing in this deck rests on an unchecked assertion.",
                kicker="Evidence discipline")
    _bullets(s, M, y, Inches(7.4), Inches(3.6), [
        ("11/11 deterministic checks PASS (2026-09-19) — exact divisions, receipt presence, "
         "sealed-artifact hash equality, and lost-source absence probes.", GREEN, True),
        ("Every number is bound to a surviving receipt file in outputs/, and the consolidated ledger "
         "states its own source for each row.", INK),
        ("Paid launches are lead-authorised and gated per-launch by a technical-chain existence check "
         "and a preflight — not by a cumulative cap.", INK),
        ("Blocked lanes carry terminal close-out records with explicit reopen conditions, "
         "rather than being left silently pending.", INK),
        ("The 2026-09-12 halt was clean: no orphan processes, deployments or sessions; disk recovered.", INK),
    ], size=13, gap=12)

    px = M + Inches(7.75)
    pw = SLIDE_W - M - px
    _callout(s, px, y, pw, Inches(1.36), "Budget as counted",
             "$5.6301 of the $50 additional cap counted. The E5 $80 successor reservation is held "
             "separately and remains unreserved. No spend occurred on 2026-09-19.")
    _callout(s, px, y + Inches(1.52), pw, Inches(1.64), "Known evidence loss",
             "Execution source for the finish-era recovery code was lost with a workspace deletion. "
             "It is confined to recovery/build code; the replacement-lane native runners survive. "
             "Sealed requests, validation receipts and test logs pin any faithful re-implementation.",
             color=AMBER)
    _src(s, "outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json; "
            "outputs/PES_Phase2_Review_2026-09-12/finish/STATE_RECONCILIATION_2026-09-19.md")
    _footer(s, n)
    return s


def slide_since(prs, n):
    s = _blank(prs)
    y = _header(s, "Work completed since the second review",
                subtitle="12 September → 24 September 2026.", kicker="Progress")
    rows = [
        ["Area", "What changed", "State"],
        ["E1 wave-10", "Lost execution source re-implemented under tracked paths; verified 10/10 offline", "COMPLETE"],
        ["E5 successor", "Chain re-verified; duplicate-kwarg dispatch bug fixed; halt test now 52/52", "COMPLETE"],
        ["E2 CORE-Bench", "Orchestration driver rebuilt — 12/12 offline tests on schedule math, cutoff, IAM gate", "COMPLETE"],
        ["E13 BALROG", "Hosted-supervisor chain verified 17/17 offline", "COMPLETE"],
        ["E4 rerun", "Full 100-trial base-model rerun executed; 100/100 trials, mean reward 0.0", "COMPLETE"],
        ["E14", "Terminal note resolves both parser-exclusion rows; closes terminal-complete", "COMPLETE"],
        ["External lanes", "Six terminal close-out records issued with reopen conditions", "COMPLETE"],
        ["E6 / E9", "AWS quota re-checked live; still NO-GO in both regions", "BLOCKED"],
    ]
    _table(s, M, y, SLIDE_W - 2 * M, rows, [0.16, 0.68, 0.16], status_col=2,
           row_h=Inches(0.40), size=11)
    _text(s, M, SLIDE_H - Inches(1.00), SLIDE_W - 2 * M, Inches(0.30),
          [("The E4 rerun's 0.0 mean measures tool-dialogue collapse in the base model, not finance "
            "reasoning: 27+ of 100 trajectories ended in role-token degeneration before producing deliverables.",
            10, False, MUTED, True)], space_after=Pt(0), line_spacing=1.05)
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §9; outputs/UNBLOCK_CARRYOUT_2026-09-21.md")
    _footer(s, n)
    return s


def slide_blocked(prs, n):
    s = _blank(prs)
    y = _header(s, "Externally blocked lanes — the honest boundary",
                subtitle="Access requested; no provider response. Recorded as terminal, not pending.",
                kicker="Limitations")
    _bullets(s, M, y, Inches(6.6), Inches(3.5), [
        ("E3 SDAB — private task bundle; no result path without it.", INK),
        ("E7 BinaryAudit — private payload; issue #22 opened upstream 2026-09-21.", INK),
        ("E8-original — LifeSciBench package is private.", INK),
        ("E10-original — AgentHarm held-out tasks are private.", INK),
        ("E12 AppBench — deployment access unavailable.", INK),
        ("E13-original — OpenReward held-out games are private.", INK),
        ("E14-original — FrontierMath requires a hosted provider run.", INK),
    ], size=12.5, gap=9.5)

    px = M + Inches(6.95)
    pw = SLIDE_W - M - px
    _callout(s, px, y, pw, Inches(1.52), "Why this is a valid submission state",
             "Reporting a lane as BLOCKED with a null score, rather than substituting a different "
             "benchmark and presenting it as the original, is the honest outcome. Replacement scopes "
             "are always labelled as replacements and never pooled with original-contract numbers.")
    _callout(s, px, y + Inches(1.68), pw, Inches(1.52), "Reopen conditions",
             "Each close-out record names the single external action that would reopen its lane. "
             "No lane is closed on the basis of silence alone — only on a recorded absence of "
             "provider response after repeated access requests.", color=AMBER)
    _src(s, "outputs/PES_Phase2_Review_2026-09-12/finish/external_closures_2026-09-19/; "
            "outputs/UNBLOCK_CARRYOUT_2026-09-21.md")
    _footer(s, n)
    return s


def slide_establish(prs, n):
    s = _blank(prs)
    y = _header(s, "What the measurements establish",
                subtitle="And what they deliberately do not.", kicker="Interpretation")
    half = (SLIDE_W - 2 * M - Inches(0.24)) / 2
    _rect(s, M, y, half, Inches(3.3), fill=WHITE, line=RULE)
    _rect(s, M, y, Pt(3.2), Inches(3.3), fill=GREEN)
    _text(s, M + Inches(0.20), y + Inches(0.16), half - Inches(0.44), Inches(0.34),
          [("Established", 15, True, GREEN)], space_after=Pt(0))
    _bullets(s, M + Inches(0.20), y + Inches(0.60), half - Inches(0.44), Inches(2.5), [
        ("A complete, receipt-backed evaluation of one open-weights actor across 14 held-out suites.", INK),
        ("Four suites carry strictly complete graded scopes (E8, E10, E11, E14).", INK),
        ("Every claim is independently re-checkable from the repository without re-running anything.", INK),
        ("Partial and blocked lanes are reported as such, with their denominators intact.", INK),
    ], size=11.5, gap=9)

    x2 = M + half + Inches(0.24)
    _rect(s, x2, y, half, Inches(3.3), fill=WHITE, line=RULE)
    _rect(s, x2, y, Pt(3.2), Inches(3.3), fill=AMBER)
    _text(s, x2 + Inches(0.20), y + Inches(0.16), half - Inches(0.44), Inches(0.34),
          [("Not claimed", 15, True, AMBER)], space_after=Pt(0))
    _bullets(s, x2 + Inches(0.20), y + Inches(0.60), half - Inches(0.44), Inches(2.5), [
        ("No cross-suite average or aggregate score of any kind.", INK),
        ("No improvement over a baseline or over the base model.", INK),
        ("No claim that these results generalise beyond the frozen sampling conditions.", INK),
        ("No numerical-parity claim between the merged BF16 artifact and the original Tinker sampler.", INK),
        ("No adversarial-robustness claim — the E10 scope is benign utility only.", INK),
    ], size=11.5, gap=8)
    _src(s, "outputs/E1_E14_FINAL_RESULTS_2026-09-19.md §1, §2, §6")
    _footer(s, n)
    return s


def slide_evidence(prs, n):
    s = _blank(prs)
    y = _header(s, "Evidence and reproduction",
                subtitle="Every figure in this deck traces to one of these.",
                kicker="Pointers")
    rows = [
        ["Artifact", "Role"],
        ["outputs/E1_E14_FINAL_RESULTS_2026-09-19.md", "Consolidated per-lane results ledger — the source for every table in this deck"],
        ["outputs/verification/LEDGER_CODE_CHECK_2026-09-19.json", "Deterministic integrity check — 11/11 PASS"],
        ["outputs/UNBLOCK_CARRYOUT_2026-09-21.md", "Most recent per-lane gate status and outstanding actions"],
        [".../finish/DECISION_PACKAGE_2026-09-19.md", "The six user-level decisions governing remaining funded work"],
        [".../finish/external_closures_2026-09-19/", "Six terminal close-out records with reopen conditions"],
        ["zvf-program/flagship/", "Surviving native runners for the replacement lanes"],
        ["submission/demo/", "Self-contained offline demo artifact and defense runbook"],
    ]
    _table(s, M, y, SLIDE_W - 2 * M, rows, [0.42, 0.58], row_h=Inches(0.40), size=11)
    _src(s, "repository paths as listed")
    _footer(s, n)
    return s


def slide_conclusions(prs, n):
    s = _blank(prs)
    y = _header(s, "Conclusions and next steps", kicker="Wrap-up")
    _bullets(s, M, y, SLIDE_W - 2 * M, Inches(2.2), [
        ("The E1–E14 campaign is complete as an evidence artifact: all fourteen lanes hold a recorded, "
         "sourced terminal state — complete, partial, or externally blocked.", GREEN, True),
        ("Four suites are strictly complete and independently re-checkable; the remainder are honestly "
         "bounded with their denominators and access failures documented.", INK),
        ("No headline claim in this work relies on a substituted benchmark, a pooled score, or an "
         "unverified receipt.", INK),
    ], size=13.5, gap=14)

    y2 = y + Inches(2.32)
    _text(s, M, y2, SLIDE_W - 2 * M, Inches(0.30),
          [("Next steps", 15, True, PES_BLUE)], space_after=Pt(0))
    _bullets(s, M, y2 + Inches(0.40), SLIDE_W - 2 * M, Inches(1.7), [
        ("Execute the four prepared launches that currently fail only on authorisation: E1 wave-10 ($16), "
         "E5 successor ($80), E2 CORE-Bench (~$4), E13 BALROG ($8).", AMBER),
        ("Escalate the two open AWS quota cases, or rebuild the E9 runtime locally.", AMBER),
        ("Pursue the outstanding external access requests; each blocked lane already names its reopen condition.", INK),
        ("Consolidate the campaign as the evaluation chapter of the thesis submission.", INK),
    ], size=12.5, gap=9)
    _src(s, "outputs/UNBLOCK_CARRYOUT_2026-09-21.md")
    _footer(s, n)
    return s


def slide_thanks(prs, n):
    s = _blank(prs)
    _rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, fill=WHITE)
    _rect(s, Inches(0), Inches(0), SLIDE_W, Inches(0.30), fill=PES_BLUE)
    _rect(s, Inches(0), Inches(0.30), SLIDE_W, Pt(2.5), fill=ACCENT)
    _text(s, M, Inches(2.45), SLIDE_W - 2 * M, Inches(0.80),
          [("Thank you", 42, True, PES_BLUE)], space_after=Pt(0))
    _text(s, M, Inches(3.36), SLIDE_W - 2 * M, Inches(0.44),
          [("Questions & discussion", 20, False, INK)], space_after=Pt(0))
    _rect(s, M, Inches(4.00), Inches(2.6), Pt(2.5), fill=ACCENT)
    _text(s, M, Inches(4.34), SLIDE_W - 2 * M, Inches(1.20),
          [("Arvind C R   ·   SRN PES2PGE24DS140", 14.5, True, INK),
           ("M.Tech Data Science & AI, PES University", 13, False, MUTED),
           ("Project Guide:  Ramesh Prakash Guledgudd", 13, False, MUTED)],
          space_after=Pt(7))
    return s


def build() -> str:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        slide_title, slide_agenda, slide_glance, slide_model,
        slide_results_a, slide_results_b,
        slide_e8, slide_e10_e11, slide_e14, slide_e1,
        slide_verification, slide_since, slide_blocked,
        slide_establish, slide_evidence, slide_conclusions, slide_thanks,
    ]
    for i, fn in enumerate(builders, start=1):
        fn(prs, i)

    prs.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    path = build()
    size_mb = os.path.getsize(path) / 1048576
    print(f"wrote {path}  ({size_mb:.2f} MB)")
    if "--check" in sys.argv:
        rp = Presentation(path)
        print(f"round-trip OK: {len(rp.slides)} slides")
