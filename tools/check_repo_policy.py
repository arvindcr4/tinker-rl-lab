#!/usr/bin/env python3
"""Fast repository hygiene checks shared by CI and pre-commit."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 100 * 1024 * 1024
REQUIRED_FILES = {
    ".editorconfig",
    ".github/workflows/ci.yml",
    ".pre-commit-config.yaml",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "REPRODUCE.md",
    "SECURITY.md",
    "pyproject.toml",
    "uv.lock",
}
TEXT_SUFFIXES = {".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
SECRET_SUFFIXES = {".key", ".p12", ".pem"}


def tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]


def main() -> int:
    issues: list[str] = []
    missing = sorted(name for name in REQUIRED_FILES if not (ROOT / name).is_file())
    issues.extend(f"missing required file: {name}" for name in missing)

    for path in tracked_paths():
        if not path.exists():
            continue  # a deliberate working-tree deletion is reviewed in its diff
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_TRACKED_BYTES:
            issues.append(f"tracked file exceeds 100 MiB: {relative}")
        if path.name == ".env" or (
            path.suffix.lower() in SECRET_SUFFIXES and path.name != ".env.example"
        ):
            issues.append(f"secret-like file is tracked: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES and "archive" not in relative.parts:
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(line.startswith(("<<<<<<< ", ">>>>>>> ")) for line in text.splitlines()):
                issues.append(f"merge-conflict marker found: {relative}")

    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1
    print("repository policy: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
