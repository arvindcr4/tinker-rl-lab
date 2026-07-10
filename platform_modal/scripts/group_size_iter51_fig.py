#!/usr/bin/env python3
"""Iter 51 — 2-panel figure: (left) reward-vs-G curves at 4 budgets, with Wu 97.6% retention
line on the right axis; (right) retention G=4/G=32 vs budget, with argmax G annotated.

Stdlib-only PDF renderer (mirrors scripts/group_size_iter47_fig.py style).
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# stdlib PDF writer (single page, line segments only)
# ---------------------------------------------------------------------------


class _Obj:
    def __init__(self, n):
        self.n = n
        self.lines = []

    def add(self, s):
        self.lines.append(s)


def _stream(objs):
    out = []
    for o in objs:
        if not o.lines:
            continue
        s = "\n".join(o.lines)
        out.append(f"<< /Length {len(s)} >>\nstream\n{s}\nendstream")
    return out


def write_pdf(path, objs, page_objs, page_w=595, page_h=420):
    # objs: list of _Obj; page_objs: list of obj indices that are pages
    with open(path, "wb") as f:
        out = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets = [0]
        # Reserve obj numbers 1..N
        n = max(o.n for o in objs) + 1
        # Build content streams first to know lengths
        content_streams = {}
        for o in objs:
            if o.lines and "/Length" not in (o.lines[0] if o.lines else ""):
                s = "\n".join(o.lines)
                content_streams[o.n] = s
        # Write objects
        for i in range(1, n):
            f.write(f"{i} 0 obj\n".encode())
            if i in content_streams:
                s = content_streams[i]
                f.write(f"<< /Length {len(s)} >>\nstream\n".encode())
                f.write(s.encode("latin-1"))
                f.write(b"\nendstream\nendobj\n")
            else:
                f.write(b"endobj\n")
            offsets.append(f.tell())
        # xref
        xref_pos = f.tell()
        f.write(f"xref\n0 {n}\n0000000000 65535 f \n".encode())
        for off in offsets[1:n]:
            f.write(f"{off:010d} 00000 n \n".encode())
        # trailer
        f.write(
            f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
        )


# ---------------------------------------------------------------------------
# Plotter: 2-panel landscape, no external deps
# ---------------------------------------------------------------------------


def _new_pdf_page(lines, w=842, h=595):
    """Wrap plot lines into a single PDF page stream."""
    # Default font: Helvetica 10pt
    head = "q 0.5 0.5 0.5 RG 0.5 w\n"
    body = head + "\n".join(lines) + "\nQ"
    return body


def _panel_left(top=540, bottom=80, left=70, right=420):
    """Panel 1: reward-vs-G curves at 4 budgets."""
    lines = []
    # Frame
    lines.append(f"{left} {bottom} {right-left} {top-bottom} re S")
    # Title
    lines.append("BT /F1 12 Tf 70 555 Td (Iter 51 (a): reward vs G per budget) Tj ET")
    lines.append("BT /F1 8 Tf 70 545 Td (Qwen2.5-0.5b / arithmetic; 5 G values x 4 budgets) Tj ET")
    # Axes labels
    lines.append("BT /F1 9 Tf 245 50 Td (group size G) Tj ET")
    lines.append("BT /F1 9 Tf 20 280 Td (heldout acc) Tj ET")
    # Grid
    for acc in [0.2, 0.4, 0.6, 0.8, 1.0]:
        y = bottom + (acc - 0.2) / 0.8 * (top - bottom)
        lines.append(f"{left} {y:.1f} m {right} {y:.1f} l S")
        lines.append(f"BT /F1 7 Tf 50 {y+2:.1f} Td ({acc:.1f}) Tj ET")
    for Gx in [4, 8, 16, 32, 64]:
        x = left + (math.log2(Gx) - 2) / 6 * (right - left)
        lines.append(f"{x:.1f} {bottom} m {x:.1f} {bottom+3} l S")
        lines.append(f"BT /F1 7 Tf {x-4:.1f} {bottom-10:.1f} Td ({Gx}) Tj ET")
    # Plot 4 budget curves
    rows = []
    with open(RES / "group_size_iter51_reward_vs_G.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            row = dict(zip(header, line.rstrip("\n").split("\t")))
            rows.append(row)
    budgets = sorted({int(r["T_tokens"]) for r in rows})
    colors = {1_000_000: "0.2 0.4 0.8", 4_000_000: "0.0 0.6 0.4", 16_000_000: "0.9 0.5 0.0", 64_000_000: "0.8 0.1 0.3"}
    for T in budgets:
        cell = sorted([r for r in rows if int(r["T_tokens"]) == T], key=lambda r: int(r["G"]))
        for i, r in enumerate(cell[:-1]):
            x1 = left + (math.log2(int(r["G"])) - 2) / 6 * (right - left)
            x2 = left + (math.log2(int(cell[i + 1]["G"])) - 2) / 6 * (right - left)
            y1 = bottom + (float(r["acc"]) - 0.2) / 0.8 * (top - bottom)
            y2 = bottom + (float(cell[i + 1]["acc"]) - 0.2) / 0.8 * (top - bottom)
            lines.append(f"{colors[T]} RG 1.6 w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")
        # Markers + error bars
        for r in cell:
            x = left + (math.log2(int(r["G"])) - 2) / 6 * (right - left)
            y = bottom + (float(r["acc"]) - 0.2) / 0.8 * (top - bottom)
            ylo = bottom + (float(r["ci_lo"]) - 0.2) / 0.8 * (top - bottom)
            yhi = bottom + (float(r["ci_hi"]) - 0.2) / 0.8 * (top - bottom)
            lines.append(f"{colors[T]} RG 0.8 w {x:.1f} {ylo:.1f} m {x:.1f} {yhi:.1f} l S")
            # marker filled
            lines.append(f"{colors[T]} rg {x-2.5:.1f} {y-2.5:.1f} 5 5 re f")
    # Wu 97.6% horizontal line, computed from argmax G=32 at T=64M as the "100% reference"
    g32_t64 = next(r for r in rows if int(r["T_tokens"]) == 64_000_000 and int(r["G"]) == 32)
    ref = float(g32_t64["acc"])
    yref = bottom + (ref * 0.976 - 0.2) / 0.8 * (top - bottom)
    lines.append(f"0.5 0.5 0.5 RG 0.6 w 0.6 0.6 0.6 RG [2 3] 0 d {left} {yref:.1f} m {right} {yref:.1f} l S 0.4 w 0 d")
    lines.append(f"BT /F1 7 Tf {right-110:.1f} {yref+3:.1f} Td (Wu 2025: 97.6% of G=32@T=64M) Tj ET")
    # Legend
    ly = top - 20
    for T, c in colors.items():
        lines.append(f"{c} RG 1.6 w {left+10:.1f} {ly:.1f} m {left+30:.1f} {ly:.1f} l S")
        lines.append(f"BT /F1 8 Tf {left+34:.1f} {ly-3:.1f} Td (T={T//1_000_000}M) Tj ET")
        ly -= 12
    return lines


def _panel_right(top=540, bottom=80, left=460, right=812):
    """Panel 2: retention G=4/G=32 vs budget + argmax G as bars."""
    lines = []
    lines.append(f"{left} {bottom} {right-left} {top-bottom} re S")
    lines.append("BT /F1 12 Tf 460 555 Td (Iter 51 (b): retention G=4/G=32 vs budget) Tj ET")
    lines.append("BT /F1 8 Tf 460 545 Td (Wu 2025 threshold + argmax G bars) Tj ET")
    # Gridlines
    for r in [0.5, 0.7, 0.8, 0.9, 0.976, 1.0]:
        y = bottom + (r - 0.4) / 0.6 * (top - bottom)
        if 0.4 <= r <= 1.0:
            lines.append(f"{left} {y:.1f} m {right} {y:.1f} l S")
            lines.append(f"BT /F1 7 Tf {left-30:.1f} {y+2:.1f} Td ({r:.3f}) Tj ET")
    # Wu 97.6% band
    yref = bottom + (0.976 - 0.4) / 0.6 * (top - bottom)
    lines.append(f"0.5 0.5 0.5 RG 0.6 w [2 3] 0 d {left} {yref:.1f} m {right} {yref:.1f} l S 0 d")
    lines.append(f"BT /F1 7 Tf {right-180:.1f} {yref+3:.1f} Td (Wu 2025: 97.6% retention) Tj ET")
    # X axis (log budget)
    budgets_M = [1, 4, 16, 64]
    for i, b in enumerate(budgets_M):
        x = left + i / (len(budgets_M) - 1) * (right - left)
        lines.append(f"{x:.1f} {bottom} m {x:.1f} {bottom+3} l S")
        lines.append(f"BT /F1 7 Tf {x-12:.1f} {bottom-10:.1f} Td (T={b}M) Tj ET")
    # Read retention from lit_compare
    rows = []
    with open(RES / "group_size_iter51_lit_compare.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    # Plot retention line (red)
    for i, r in enumerate(rows[:-1]):
        x1 = left + i / (len(rows) - 1) * (right - left)
        x2 = left + (i + 1) / (len(rows) - 1) * (right - left)
        y1 = bottom + (float(r["ours_retention"]) - 0.4) / 0.6 * (top - bottom)
        y2 = bottom + (float(rows[i + 1]["ours_retention"]) - 0.4) / 0.6 * (top - bottom)
        lines.append(f"0.8 0.1 0.3 RG 2.0 w {x1:.1f} {y1:.1f} m {x2:.1f} {y2:.1f} l S")
    for i, r in enumerate(rows):
        x = left + i / (len(rows) - 1) * (right - left)
        y = bottom + (float(r["ours_retention"]) - 0.4) / 0.6 * (top - bottom)
        above = r["above_wu_threshold"] == "yes"
        c = "0.0 0.6 0.2" if above else "0.8 0.1 0.3"
        lines.append(f"{c} rg {x-4:.1f} {y-4:.1f} 8 8 re f")
    # Read peak shift; show as bar annotations along bottom
    pk = []
    with open(RES / "group_size_iter51_peak_shift.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            pk.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    for i, p in enumerate(pk):
        x = left + i / (len(pk) - 1) * (right - left)
        gp = int(p["argmax_G"])
        barh = (math.log2(gp) - 2) / 6 * 18  # tiny bar showing argmax
        lines.append(f"0.2 0.4 0.8 RG 0.8 w {x-2:.1f} {bottom-25:.1f} m {x-2:.1f} {bottom-25+barh:.1f} l S")
        lines.append(f"BT /F1 6 Tf {x-8:.1f} {bottom-32:.1f} Td (peak={gp}) Tj ET")
    # Legend
    lines.append("BT /F1 8 Tf 460 525 Td (red line: retention G=4/G=32; green=above Wu, red=below) Tj ET")
    lines.append("BT /F1 8 Tf 460 515 Td (blue ticks below axis: argmax G at each budget) Tj ET")
    return lines


def main():
    left = _panel_left()
    right = _panel_right()
    body = _new_pdf_page(left + right)
    out = FIG / "group_size_iter51.pdf"
    # Single-page: just write raw stream
    with open(out, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        body_bytes = body.encode("latin-1")
        # 3 objects: catalog, pages, content
        offsets = [0]
        f.write(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        offsets.append(f.tell())
        f.write(b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n")
        offsets.append(f.tell())
        f.write(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 842 595] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n")
        offsets.append(f.tell())
        f.write(f"4 0 obj\n<< /Length {len(body_bytes)} >>\nstream\n".encode())
        f.write(body_bytes)
        f.write(b"\nendstream\nendobj\n")
        offsets.append(f.tell())
        f.write(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
        offsets.append(f.tell())
        xref_pos = f.tell()
        f.write(b"xref\n0 6\n0000000000 65535 f \n")
        for off in offsets:
            f.write(f"{off:010d} 00000 n \n".encode())
        f.write(b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n")
        f.write(f"{xref_pos}\n".encode())
        f.write(b"%%EOF\n")
    # Mirror to paper/figures
    with open(out, "rb") as fin, open(PAPER_FIG / "group_size_iter51.pdf", "wb") as fout:
        fout.write(fin.read())
    print(f"Wrote {out} and {PAPER_FIG / 'group_size_iter51.pdf'}")


if __name__ == "__main__":
    main()
