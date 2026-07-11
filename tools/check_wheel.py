#!/usr/bin/env python3
"""Verify that a built wheel contains every supported first-party module."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


REQUIRED_MEMBERS = {
    "platform_local/__init__.py",
    "platform_local/trl_integrations/config.py",
    "platform_local/trl_integrations/trainer.py",
    "platform_local/unified/__main__.py",
    "platform_local/unified/launcher.py",
    "platform_tinker/__init__.py",
    "platform_tinker/tinkerrl/grpo.py",
    "utils/seed.py",
    "utils/stats.py",
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_wheel.py DIST_WHEEL [...]", file=sys.stderr)
        return 2

    for raw_path in sys.argv[1:]:
        wheel = Path(raw_path)
        with zipfile.ZipFile(wheel) as archive:
            members = set(archive.namelist())
            entry_points_name = next(
                (name for name in members if name.endswith(".dist-info/entry_points.txt")),
                None,
            )
            entry_points = (
                archive.read(entry_points_name).decode("utf-8") if entry_points_name else ""
            )
        missing = sorted(REQUIRED_MEMBERS - members)
        if missing:
            print(
                f"{wheel}: missing supported modules: {', '.join(missing)}",
                file=sys.stderr,
            )
            return 1
        expected_entry_point = "tinkerrl = platform_local.unified.__main__:main"
        if expected_entry_point not in entry_points:
            print(f"{wheel}: missing console entry point", file=sys.stderr)
            return 1
        print(f"{wheel}: verified {len(REQUIRED_MEMBERS)} supported module files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
