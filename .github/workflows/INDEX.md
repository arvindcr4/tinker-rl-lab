# .github/workflows/ — INDEX

**Purpose:** GitHub Actions CI for TinkerRL (runs on push to main/develop and PRs to main).

**Key files:**
- `ci.yml` — jobs: `lint` (ruff check + format), `test-core` (pytest matrix py3.9–3.12), `test-environments`, `reproducibility-check`, `docs-check`.

**Find it fast:**
- to change lint/test/repro gates → `ci.yml`
