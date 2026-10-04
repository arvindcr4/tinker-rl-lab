#!/usr/bin/env python3
"""Check that a recorded review applies to the selected PDF, without changing it.

This is a receipt-binding gate, separate from frozen-publication file integrity.
It does not perform a visual review, authenticate reviewers, parse PDF pages, or
establish scientific validity. A matching hash binds the recorded page count and
review to exact bytes; it does not independently verify those recorded claims.
"""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from ._strict import require
    from .check_public_release import HEX256, ROOT, safe_file, sha256, strict_json
else:
    from _strict import require
    from check_public_release import HEX256, ROOT, safe_file, sha256, strict_json


DEFAULT_RECEIPT = "reports/public_revision_2026-10-03/thesis/public_quality_receipt.json"
DEFAULT_PDF = (
    "reports/public_revision_2026-10-03/thesis/"
    "Thesis_Report_ArvindCR_C1_Coverage_Derived_2026-10-03_PUBLIC.pdf"
)
SCOPE = (
    "Exact PDF binding and recorded review completion only; no independent visual, "
    "page-count, scientific, privacy, or reviewer-authenticity verification."
)


def verify_receipt(root, receipt_path=DEFAULT_RECEIPT, pdf_path=DEFAULT_PDF):
    """Fail closed on missing, malformed, unfinished, or stale review receipts.

    The PDF is selected by the caller, never by the receipt's filename. This
    prevents a stale receipt from silently selecting a different reviewed file.
    """
    root = Path(root)
    receipt = strict_json(safe_file(root, receipt_path))
    require(isinstance(receipt, dict), "quality receipt must be an object")
    require(receipt.get("schema") == "public-document-quality-v1", "unknown quality schema")
    pdf = safe_file(root, pdf_path)
    require(pdf.suffix.lower() == ".pdf", "selected document must be a PDF")
    require(receipt.get("pdf_filename") == pdf.name, "quality receipt PDF filename mismatch")
    size = receipt.get("bytes")
    require(type(size) is int and size > 0, "receipt bytes must be a positive integer")
    digest = receipt.get("sha256")
    require(isinstance(digest, str) and HEX256.fullmatch(digest), "invalid receipt SHA-256")
    require(pdf.stat().st_size == size, "stale quality receipt: PDF byte count mismatch")
    require(sha256(pdf) == digest, "stale quality receipt: PDF SHA-256 mismatch")
    with pdf.open("rb") as stream:
        require(stream.read(5) == b"%PDF-", "selected document lacks a PDF header")

    pages = receipt.get("pages")
    require(type(pages) is int and pages > 0, "receipt pages must be a positive integer")
    reviewed = receipt.get("rendered_review_pages")
    require(isinstance(reviewed, list) and reviewed, "receipt has no rendered review pages")
    require(
        all(type(page) is int and 1 <= page <= pages for page in reviewed),
        "rendered review page is outside the recorded document",
    )
    require(len(set(reviewed)) == len(reviewed), "duplicate rendered review pages")
    for field in ("visual_review", "independent_review"):
        value = receipt.get(field)
        require(
            isinstance(value, str) and value.startswith("pass:") and value[5:].strip(),
            f"{field} does not record a completed passing review",
        )
    # v1 has no required status field; reject explicit nonfinal states if supplied.
    if "status" in receipt:
        require(receipt["status"] in ("final", "reviewed", "pass"), "receipt status is not final")
    for field in ("all_page_text_bounds_failures", "replacement_character_pages"):
        require(receipt.get(field) == [], f"{field} must record no failures")
    for field in ("embedded_file_count", "privacy_pattern_failure_count"):
        require(type(receipt.get(field)) is int and receipt[field] == 0, f"{field} must be zero")
    require(
        receipt.get("scientific_case_ledger_unchanged") is True,
        "receipt does not record an unchanged scientific case ledger",
    )
    return {"pdf": pdf_path, "sha256": digest, "bytes": size, "recorded_pages": pages}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--receipt", default=DEFAULT_RECEIPT)
    parser.add_argument("--pdf", default=DEFAULT_PDF)
    args = parser.parse_args(argv)
    try:
        result = verify_receipt(args.root, args.receipt, args.pdf)
    except (OSError, ValueError) as exc:
        print(f"FAIL: PDF quality receipt: {exc}")
        print(SCOPE)
        return 1
    print(f"PASS: PDF quality receipt binds {result['pdf']} ({result['bytes']} bytes)")
    print(SCOPE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
