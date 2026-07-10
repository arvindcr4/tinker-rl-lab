#!/usr/bin/env python3
"""Build a self-contained per-paper Overleaf bundle for each of P1..P8.

Each bundle = the paper's own main .tex + the FULL sections/, figures/, tikz/
directories + references.bib + the neurips_*.sty files. Copying the full
figures/ dir (rather than resolving per-figure references) guarantees Overleaf
renders exactly what the local `pdflatex` build produces — no figure-path
guesswork, and missing figures degrade to placeholder boxes via the preamble
fallback just as they do locally.

Output: <out>/P1 .. <out>/P8, ready to mirror into each Overleaf git repo.
"""
import os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, "..", ".."))   # repo .../paper
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "stage")

PAPERS = {
    "P1": "paper_P1_scaling.tex",
    "P2": "paper_P2_zvf.tex",
    "P3": "paper_P3_group_size.tex",
    "P4": "paper_P4_length_bias.tex",
    "P5": "paper_P5_minreport.tex",
    "P6": "paper_P6_registry.tex",
    "P7": "paper_P7_zvf_controller.tex",
    "P8": "paper_P8_fraud.tex",
}
STY = ("neurips_2024.sty", "neurips_2025.sty", "neurips_2026.sty")
DIRS = ("sections", "figures", "tikz")


def build(pid, mainfile):
    stage = os.path.join(OUT, pid)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    shutil.copy2(os.path.join(PAPER, mainfile), os.path.join(stage, mainfile))
    shutil.copy2(os.path.join(PAPER, "references.bib"), os.path.join(stage, "references.bib"))
    for sty in STY:
        p = os.path.join(PAPER, sty)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(stage, sty))
    for d in DIRS:
        src = os.path.join(PAPER, d)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(stage, d))
    nsec = len(os.listdir(os.path.join(stage, "sections")))
    nfig = sum(len(fs) for _, _, fs in os.walk(os.path.join(stage, "figures")))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(stage) for f in fs) / 1e6
    print(f"{pid}: {mainfile}  sections={nsec} figures={nfig} size={size:.1f}MB")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for pid, mainfile in PAPERS.items():
        build(pid, mainfile)
    print("stages under", OUT)
