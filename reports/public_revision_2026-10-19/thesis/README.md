# Thesis sources — ARS-corrected revision, 19 October 2026

Additive corrected copy of the frozen 11 October 2026 thesis sources. The frozen 317-page PDF was **not** copied here and remains only in `reports/public_revision_2026-10-11/` (SHA-256 `83d8ee3a73a7073f17b80d260595b2d11a27f8d750c369b784326d94734980e0`); figure PDFs were likewise not copied, so a rebuild must first regenerate figures. Headline claim language was shrunk to body tier (see `../ERRATA.md`); no experimental result was regenerated.

Build with `python3 build_thesis.py` in this directory (Python 3.11+, Pandoc and Tectonic 0.17.0 with populated TeX cache), after regenerating any missing figure PDFs. Rebuilding creates new bytes: it requires fresh checksums, manifests, and independent human review before any quality claim. **Checksums and quality receipts for this copy are pending** — the 11 October `public_quality_receipt.json` was removed from this copy because it certifies only the old bytes, and no replacement was fabricated.

`public_source_availability.json` and `provenance_audit.json` are historical records of their original scopes, not fresh certification of this copy. The public-checkout number audit of the 11 October sources is INCOMPLETE (970: 828 PASS, 22 CONTEXT, 120 ARTIFACT_MISSING). Run the current auditor with explicit thesis-dir and output paths outside both public-revision directories to see currently available evidence.

Status: pending independent human review. Nothing here is an institutional submission or human academic approval.
