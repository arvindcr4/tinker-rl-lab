# submission/contents/ — INDEX

**Purpose:** Reviewer-facing metadata for the packaged NeurIPS 2026 D&B submission zip. Describes and checksums the bundle contents (paper PDFs, code tarball, ethics + data statements).

**Key files:**
- `MANIFEST.md` — every file in `neurips2026_tinker_rl_lab.zip` with SHA-256; 7-file bundle composition
- `checksums.sha256` — authoritative machine-readable list for `sha256sum -c`
- `REVIEWER_README.md` — reviewer entry point: file guide, reproducibility smoke-test (`platform_modal/scripts/smoke_test.sh`, `check_qwen3_8b_claim.py`), anonymity guarantees
- `data_statement.md` — dataset provenance, licensing, PII / offensive-content notes

**Find it fast:**
- to verify bundle integrity → `sha256sum -c checksums.sha256`
- what a reviewer opens first → `REVIEWER_README.md`
