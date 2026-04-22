#!/usr/bin/env python3
"""Build the NeurIPS 2026 D&B submission bundle.

Inputs assembled from the live repo:
  paper/main.pdf              -> submission/contents/paper.pdf
  paper/main_anon.pdf         -> submission/contents/paper_anon.pdf
  paper/ethics_wrapper.pdf    -> submission/contents/ethics_statement.pdf
  reports/final/capstone_final_report.pdf -> submission/contents/report.pdf
  reports/final/grpo_agentic_llm_paper.pdf -> submission/contents/grpo_agentic_llm_paper.pdf
  reports/final/grpo_agentic_llm_paper_anonymous.pdf -> submission/contents/grpo_agentic_llm_paper_anonymous.pdf
  reports/final/submission_uploads/final_capstone_presentation.pptx
                                -> submission/contents/presentation.pptx
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
PRESENTATION_SOURCE = (
    ROOT / "reports" / "final" / "submission_uploads" / "final_capstone_presentation.pptx"
)
PRESENTATION_TARGET = CONTENTS / "presentation.pptx"

PRESENTATION_REPLACEMENTS = {
    b"Final Capstone Presentation | Group 6 | PES University": (
        b"Final Capstone Presentation | Anonymous Submission"
    ),
    b"Guides: Prof. Narayana Darapaneni and Mr. Anwesh Reddy Paduri": (
        b"Advisors: Anonymous"
    ),
    b"PES University": b"Anonymous Institution",
    b"Group 6": b"Anonymous Group",
    b"Narayana Darapaneni": b"Anonymous Advisor",
    b"Anwesh Reddy Paduri": b"Anonymous Advisor",
}


SOURCES = [
    (ROOT / "paper" / "main.pdf", CONTENTS / "paper.pdf"),
    (ROOT / "paper" / "main_anon.pdf", CONTENTS / "paper_anon.pdf"),
    (ROOT / "paper" / "ethics_wrapper.pdf", CONTENTS / "ethics_statement.pdf"),
    (
        ROOT / "reports" / "final" / "capstone_final_report.pdf",
        CONTENTS / "report.pdf",
    ),
    (
        ROOT / "reports" / "final" / "grpo_agentic_llm_paper.pdf",
        CONTENTS / "grpo_agentic_llm_paper.pdf",
    ),
    (
        ROOT / "reports" / "final" / "grpo_agentic_llm_paper_anonymous.pdf",
        CONTENTS / "grpo_agentic_llm_paper_anonymous.pdf",
    ),
    (
        ROOT / "blind_review" / "tinker-rl-lab-anon.tar.gz",
        CONTENTS / "code.tar.gz",
    ),
]

# Bundle files that live in contents/ already.
EXISTING_MEMBERS = [
    "REVIEWER_README.md",
    "SUBMISSION_README.md",
    "data_statement.md",
    "supporting_data.tar.gz",
]

BUNDLE_ORDER = [
    "REVIEWER_README.md",
    "SUBMISSION_README.md",
    "code.tar.gz",
    "data_statement.md",
    "ethics_statement.pdf",
    "grpo_agentic_llm_paper.pdf",
    "grpo_agentic_llm_paper_anonymous.pdf",
    "paper.pdf",
    "paper_anon.pdf",
    "presentation.pptx",
    "report.pdf",
    "supporting_data.tar.gz",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_presentation(src: Path, dst: Path) -> None:
    """Copy the final deck while removing blind-review identifiers in XML parts."""
    if not src.exists():
        raise SystemExit(f"missing source: {src}")
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(
        dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith((".xml", ".rels")):
                for old, new in PRESENTATION_REPLACEMENTS.items():
                    data = data.replace(old, new)
            out_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            out_info.compress_type = zipfile.ZIP_DEFLATED
            out_info.external_attr = info.external_attr
            zout.writestr(out_info, data)


def main() -> int:
    CONTENTS.mkdir(parents=True, exist_ok=True)
    for src, dst in SOURCES:
        if not src.exists():
            raise SystemExit(f"missing source: {src}")
        shutil.copy2(src, dst)
        print(f"copied {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    sanitize_presentation(PRESENTATION_SOURCE, PRESENTATION_TARGET)
    print(
        f"sanitized {PRESENTATION_SOURCE.relative_to(ROOT)} -> "
        f"{PRESENTATION_TARGET.relative_to(ROOT)}"
    )

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
