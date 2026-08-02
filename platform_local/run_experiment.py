#!/usr/bin/env python3
"""LOCAL experiment entry — physical parity shim.

Every platform_<backend>/ dir carries this canonical entry point. It delegates to
the unified launcher with this backend pinned, so the framework × backend matrix
shares one source of training logic (no duplication).
"""
import sys

BACKEND = "local"


def main():
    from platform_local.unified.__main__ import main as unified_main

    argv = sys.argv[1:]
    if "--backend" not in argv and "-b" not in argv:
        argv = ["--backend", BACKEND] + argv
    sys.argv = [sys.argv[0]] + argv
    unified_main()


if __name__ == "__main__":
    main()
