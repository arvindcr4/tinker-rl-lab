#!/usr/bin/env python3
"""MODAL experiment entry — physical parity shim (delegates to the unified launcher)."""
import sys

BACKEND = "modal"


def main():
    from platform_local.unified.__main__ import main as unified_main

    argv = sys.argv[1:]
    if "--backend" not in argv and "-b" not in argv:
        argv = ["--backend", BACKEND] + argv
    sys.argv = [sys.argv[0]] + argv
    unified_main()


if __name__ == "__main__":
    main()
