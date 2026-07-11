# .github/workflows/ — INDEX

**Purpose:** GitHub Actions CI for TinkerRL (runs on push to main/develop and PRs to main).

**Key files:**
- `ci.yml` — jobs: `lint`, locked `test-core` matrix, wheel `package`,
  `test-environments`, `reproducibility-check`, `docs-check`, and strict
  MIN-REPORT provenance verification.

**Find it fast:**
- to change lint/test/repro gates → `ci.yml`
