#!/usr/bin/env python3
"""Compatibility entry point for parameterized GSM8K GRPO runs."""

import sys

from tinkerrl.grpo_cli import legacy_main


if __name__ == "__main__":
    raise SystemExit(legacy_main("gsm8k", sys.argv[1:]))
