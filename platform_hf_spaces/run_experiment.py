#!/usr/bin/env python3
"""HF SPACES experiment entry — physical parity shim.

HF Spaces hosts the results demo (no GPU training). This entry delegates to the
unified launcher's hfspaces backend, which fetches experiment outputs produced on
the GPU-capable backends into the Space's dashboard.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BACKEND = "hfspaces"


def main():
    from platform_local.unified.__main__ import main as unified_main

    argv = sys.argv[1:]
    if "--backend" not in argv and "-b" not in argv:
        argv = ["--backend", BACKEND] + argv
    sys.argv = [sys.argv[0]] + argv
    unified_main()


if __name__ == "__main__":
    main()
