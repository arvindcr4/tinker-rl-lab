#!/usr/bin/env python3
"""Build the NeurIPS 2026 blind submission bundle.

Inputs assembled from the live repo:
  reports/final/grpo_agentic_llm_paper_anonymous.pdf -> submission/contents/paper_anon.pdf
  blind_review/tinker-rl-lab-anon.tar.gz -> submission/contents/code.tar.gz

Refreshes checksums.sha256 / MANIFEST.md, then zips the whole bundle.
"""
from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENTS = ROOT / "submission" / "contents"
ZIP_PATH = ROOT / "submission" / "neurips2026_tinker_rl_lab.zip"

SOURCES = [
    (
        ROOT / "reports" / "final" / "grpo_agentic_llm_paper_anonymous.pdf",
        CONTENTS / "paper_anon.pdf",
    ),
    (ROOT / "blind_review" / "tinker-rl-lab-anon.tar.gz", CONTENTS / "code.tar.gz"),
]

# Bundle files that live in contents/ already.
EXISTING_MEMBERS = [
    "REVIEWER_README.md",
    "SUBMISSION_README.md",
]

BUNDLE_ORDER = [
    "REVIEWER_README.md",
    "SUBMISSION_README.md",
    "code.tar.gz",
    "paper_anon.pdf",
]

STALE_MEMBERS = [
    "paper.pdf",
    "report.pdf",
    "grpo_agentic_llm_paper.pdf",
    "grpo_agentic_llm_paper_anonymous.pdf",
    "presentation.pptx",
    "ethics_statement.pdf",
    "supporting_data.tar.gz",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    CONTENTS.mkdir(parents=True, exist_ok=True)
    for name in STALE_MEMBERS:
        p = CONTENTS / name
        if p.exists():
            p.unlink()
            print(f"removed stale blind-package member {p.relative_to(ROOT)}")
    for src, dst in SOURCES:
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        shutil.copy2(src, dst)
        print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    for name in EXISTING_MEMBERS:
        p = CONTENTS / name
        if not p.exists():
            raise SystemExit(f"missing bundle member: {p}")

    sums = {}
    for name in BUNDLE_ORDER:
        p = CONTENTS / name
        if not p.exists():
            raise SystemExit(f"missing bundle member: {p}")
        sums[name] = sha256(p)

    # Write checksums.sha256 (the authoritative sha256sum -c file).
    checksum_path = CONTENTS / "checksums.sha256"
    checksum_path.write_text(
        "".join(f"{sums[n]}  {n}\n" for n in BUNDLE_ORDER)
    )
    print(f"wrote {checksum_path.relative_to(ROOT)}")

    # Update MANIFEST.md.
    manifest_path = CONTENTS / "MANIFEST.md"
    header = "# Submission Manifest\n\n| File | Size | SHA-256 |\n|---|---:|---|\n"
    body = ""
    for name in BUNDLE_ORDER:
        p = CONTENTS / name
        size_mib = p.stat().st_size / (1024 * 1024)
        body += f"| `{name}` | {size_mib:.2f} MiB | `{sums[name]}` |\n"
    footer = "\nVerify with:\n\n```bash\nsha256sum -c checksums.sha256\n```\n"
    manifest_path.write_text(header + body + footer)
    print(f"wrote {manifest_path.relative_to(ROOT)}")

    # Build the zip deterministically (sorted names, no absolute paths).
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(
        [CONTENTS / n for n in BUNDLE_ORDER]
        + [checksum_path, manifest_path]
    )
    with zipfile.ZipFile(
        ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zf:
        for m in members:
            zf.write(m, arcname=m.name)
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    zip_sha = sha256(ZIP_PATH)
    print(
        f"wrote {ZIP_PATH.relative_to(ROOT)} ({size_mb:.1f} MB), "
        f"sha256={zip_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
