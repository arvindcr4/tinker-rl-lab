# blind_review/ — INDEX

**Purpose:** Anonymized blind-review submission package for NeurIPS 2026 (Datasets & Benchmarks). Built from the canonical `paper/main.tex` + repo code by idempotent anonymization scripts; the non-anon working tree is never modified. Includes the anon PDF/TeX, anon code tarball, change logs, and audit records.

**Key files:**
- `main_anon.pdf` / `main_anon.tex` — anonymized paper (51 pages, authors/affils/URLs scrubbed)
- `tinker-rl-lab-anon.tar.gz` — anonymized code tarball (~97 MB here; ~27 MB in manifest — regenerate to match)
- `anonymize_paper.py` / `anonymize_code.py` — idempotent scrubbers (reviewers can re-run); post-scan asserts zero residual identifiers
- `paper_changes.log` / `code_changes.log` — per-rule / per-file replacement counts
- `AUDIT.md` — authoritative Task-11 anonymization audit (deliverables table, audit-script results)
- `SUBMISSION_MANIFEST.md` — every bundle file + SHA-256, rebuild + verify instructions
- `.gitignore` — excludes large regenerated artifacts

**Subfolders:**
- `audit_logs/` — raw output of the `*_audit.py` scripts (see its INDEX.md)

**Find it fast:**
- to rebuild the anon bundle → run both `anonymize_*.py`, then `latexmk main_anon.tex`
- to verify integrity → `sha256sum -c` against `SUBMISSION_MANIFEST.md`
- anonymization guarantees / token blocklist → `SUBMISSION_MANIFEST.md` §2, `AUDIT.md`
