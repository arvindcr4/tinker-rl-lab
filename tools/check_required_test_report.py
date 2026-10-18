"""Fail if a required scientific test module was absent, skipped, or failed."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def check_report(path: Path, required: list[str]) -> dict[str, int]:
    data = path.read_bytes()
    if len(data) > 10_000_000 or b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("test report contains forbidden XML declarations or exceeds 10 MB")
    cases = list(ET.fromstring(data).iter("testcase"))  # noqa: S314 (bounded, DTD-free report)
    counts = {}
    for module in required:
        selected = [case for case in cases if case.get("classname", "").split(".")[-1] == module]
        if not selected:
            raise ValueError(f"required module absent from executed report: {module}")
        if any(
            case.find(tag) is not None
            for case in selected
            for tag in ("skipped", "failure", "error")
        ):
            raise ValueError(f"required module skipped or unsuccessful: {module}")
        counts[module] = len(selected)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args(argv)
    try:
        counts = check_report(args.report, args.modules)
    except (OSError, ValueError, ET.ParseError) as exc:
        parser.error(str(exc))
    for module, count in counts.items():
        print(f"{module}: {count} executed, no skips/failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
