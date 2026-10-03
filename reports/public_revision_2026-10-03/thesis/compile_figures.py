#!/usr/bin/env python3
"""
Compile every figures/*.tex to figures/*.pdf with tectonic, in parallel.

Tectonic's package cache is shared, so after the first figure the remaining
compiles are fast. Reports per-figure status and a summary.

Usage:  python3 compile_figures.py [--only name1,name2] [-j N] [--force]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import subprocess
import sys
import tempfile
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")


def compile_one(name: str, force: bool = False) -> tuple[str, bool, str]:
    tex = os.path.join(FIGDIR, name + ".tex")
    pdf = os.path.join(FIGDIR, name + ".pdf")
    if not os.path.isfile(tex):
        return name, False, "missing source"
    if (not force and os.path.isfile(pdf) and os.path.getsize(pdf) > 0
            and os.path.getmtime(pdf) > os.path.getmtime(tex)):
        return name, True, "cached"
    try:
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=FIGDIR) as outdir:
            r = subprocess.run(
                ["tectonic", "-X", "compile", tex, "--outdir", outdir],
                capture_output=True, timeout=600, cwd=FIGDIR,
            )
            built = os.path.join(outdir, name + ".pdf")
            if r.returncode == 0:
                if not os.path.isfile(built) or os.path.getsize(built) == 0:
                    return name, False, "no nonempty PDF produced"
                size = os.path.getsize(built)
                os.replace(built, pdf)
                return name, True, f"{size//1024}KB"
    except subprocess.TimeoutExpired:
        return name, False, "timeout"
    except OSError as exc:
        return name, False, str(exc)
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
    ap.add_argument("--force", action="store_true", help="rebuild even if the PDF is newer")
    args = ap.parse_args()
    if args.j < 1:
        ap.error("-j must be at least 1")
    if not shutil.which("tectonic"):
        print("missing required tool: tectonic", file=sys.stderr)
        return 1

    if not os.path.isdir(FIGDIR):
        print("no figures dir", file=sys.stderr)
        return 1
    names = sorted(f[:-4] for f in os.listdir(FIGDIR) if f.endswith(".tex"))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        missing = wanted - set(names)
        if missing:
            print("missing figure sources: " + ", ".join(sorted(missing)), file=sys.stderr)
            return 1
        names = [n for n in names if n in wanted]
    if not names:
        print("no figure .tex files found", file=sys.stderr)
        return 1

    print(f"compiling {len(names)} figures with {args.j} workers ...")
    ok, bad = [], []
    with cf.ThreadPoolExecutor(max_workers=args.j) as ex:
        for name, good, msg in ex.map(partial(compile_one, force=args.force), names):
            (ok if good else bad).append((name, msg))
            print(f"  {'OK  ' if good else 'FAIL'} {name:32s} {msg}")

    print(f"\n{len(ok)}/{len(names)} compiled")
    if bad:
        print("failed: " + ", ".join(n for n, _ in bad), file=sys.stderr)
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
