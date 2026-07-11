#!/usr/bin/env python3
"""Compatibility entry point for Experiment A's consolidated GRPO run."""

from tinkerrl.grpo_cli import legacy_main


if __name__ == "__main__":
    raise SystemExit(legacy_main("tooluse_baseline"))
