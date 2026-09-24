#!/usr/bin/env python3
"""
Compile every figures/*.tex to figures/*.pdf with tectonic, in parallel.

Tectonic's package cache is shared, so after the first figure the remaining
compiles are fast. Reports per-figure status and a summary.

Usage:  python3 compile_figures.py [--only name1,name2] [-j N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")


def compile_one(name: str) -> tuple[str, bool, str]:
    tex = os.path.join(FIGDIR, name + ".tex")
    pdf = os.path.join(FIGDIR, name + ".pdf")
    if os.path.exists(pdf) and os.path.getmtime(pdf) > os.path.getmtime(tex):
        return name, True, "cached"
    try:
        r = subprocess.run(
            ["tectonic", "-X", "compile", tex, "--outdir", FIGDIR],
            capture_output=True, timeout=600, cwd=FIGDIR,
        )
    except subprocess.TimeoutExpired:
        return name, False, "timeout"
    if r.returncode == 0 and os.path.exists(pdf):
        return name, True, f"{os.path.getsize(pdf)//1024}KB"
    err = (r.stderr or r.stdout).decode(errors="replace")
    # pull the first genuinely useful error line
    lines = [l for l in err.splitlines()
             if l.strip().startswith("!") or "error" in l.lower()]
    msg = lines[0][:160] if lines else err.strip().splitlines()[-1][:160] if err.strip() else "failed"
    return name, False, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("-j", type=int, default=6)
    args = ap.parse_args()

    if not os.path.isdir(FIGDIR):
        print("no figures dir", file=sys.stderr)
        return 1
    names = sorted(f[:-4] for f in os.listdir(FIGDIR) if f.endswith(".tex"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        names = [n for n in names if n in wanted]
    if not names:
        print("no figure .tex files found", file=sys.stderr)
        return 1

    print(f"compiling {len(names)} figures with {args.j} workers ...")
    ok, bad = [], []
    with cf.ThreadPoolExecutor(max_workers=args.j) as ex:
        for name, good, msg in ex.map(compile_one, names):
            (ok if good else bad).append((name, msg))
            print(f"  {'OK  ' if good else 'FAIL'} {name:32s} {msg}")

    print(f"\n{len(ok)}/{len(names)} compiled")
    if bad:
        print("failed: " + ", ".join(n for n, _ in bad), file=sys.stderr)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
