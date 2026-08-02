#!/usr/bin/env python3
"""COLAB experiment entry — physical parity shim (delegates to the unified launcher).

On a Colab A100 runtime this drives the canonical GSM8K GRPO run. The interactive
alternative remains ``advanced_rl_colab.ipynb``.
"""
import sys

BACKEND = "colab"


def main():
    from platform_local.unified.__main__ import main as unified_main

    argv = sys.argv[1:]
    if "--backend" not in argv and "-b" not in argv:
        argv = ["--backend", BACKEND] + argv
    sys.argv = [sys.argv[0]] + argv
    unified_main()


if __name__ == "__main__":
    main()
