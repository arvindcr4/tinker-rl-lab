# Phase 2 submission verification record

## Checkout repair verification — 2026-10-02 (UTC)

The artifact cleanup broke the supported source checkout after the original
verification below. The repairs were checked from base revision `c402f376b`
plus the current fixes, not from ignored files left in the developer checkout.
A separate snapshot contained 7,180 tracked paths (about 326 MiB). Unrelated
concurrent edits to `AGENTS.md` and `CLAUDE.md` were excluded from that snapshot.

| Check | Result | Scope |
|---|---|---|
| Required source restoration | PASS | 72 required files, 940,263 bytes, matched pre-cleanup revision `649c6beb3` byte for byte on restoration. Subsequent edits only update this note and remove trailing whitespace from one TeX comment. Large run outputs remain excluded. |
| Lockfile consistency | PASS | Seven root dependency metadata entries synchronized; no resolved package versions changed. `uv lock --check --offline` and wheel checks pass. |
| Complete source-snapshot `make check` | **435 passed, 2 skipped, 17 subtests passed** | Lint, format, repository policy, all core tests including the two real compiler integrations, wheel build and documentation checks. |
| Fresh locked environment | **435 passed, 2 skipped, 17 subtests passed** | Complete `make check` repeated after an isolated install of the declared locked development dependencies. |
| Compiler-free unit selection | **433 passed, 2 skipped, 2 deselected** | All six checked TeX tool names were absent from PATH. Only the two marked real LaTeX integrations were deselected. |
| Fresh-process trainer imports | PASS | TRL and verl imports reject CodeCarbon initialization and leave no working-directory artifacts. No shutdown logging errors appeared. |
| Native thesis/package rebuild | PASS | Rebuilt all 24 required figure sources and the full 265-page A4 thesis from the snapshot. ZIP integrity passed for 122 selected files plus manifest/checksums. |
| Extracted review package | **58 passed, 1 skipped**; **21/21 evidence groups PASS** | The sole skip is the Git-index check in an archive without `.git`; required-file presence is still asserted. Demo self-tests also pass. |
| New PDF automated checks | PASS | No detected text outside the page or past the right body margin; no undefined references/citations, duplicate labels or missing-character diagnostics. Existing small box/font warnings remain. |
| Evidence and artifact preservation | PASS | All 22 measured JSON receipts and the published PDF in the original checkout remain unchanged. No tracked snapshot files changed during `make check`; only the snapshot PDF changed during the submission build. |
| Independent code review | PASS | Whole-fix review and separate LaTeX/CI review found no blockers. |

A fresh isolated environment was also created with
`uv sync --locked --extra dev --no-python-downloads --python <Python-3.12.13>`.
The initial offline attempt could not find a cached PyTorch wheel; the normal
locked install downloaded standard dependencies and installed 123 packages.
It did not change the original checkout environment or any resolved versions.

The two existing core skips need live service/authentication access or the
optional `rliable` package with heavyweight matrices. They are not compiler
skips. One expected SciPy constant-input precision warning remains. The existing
locked `build==1.5.1` yanked-version warning was not addressed by a dependency
upgrade. Pandoc 3.11 and real Tectonic 0.17.0 were used; cached TeX packages were
available. No training or benchmark grading was launched.

Hosted CI is **not verified**: the GitHub account billing lock prevented jobs
from starting (audited run: https://github.com/arvindcr4/tinker-rl-lab/actions/runs/37055147422).
The owner must resolve that account issue before obtaining a remote CI result.
The rebuilt ZIP/PDF were confined to the verification snapshot; the published
checkout PDF and the earlier local release ZIP were not replaced. These repair
changes have not been committed or pushed by this verification.

## Original package verification — historical checkpoint

Prepared on 2026-10-02 (UTC) from the then-current working tree,
base revision `c0be4a313`, before publication at `649c6beb3`. The checks below
record that original package, not the post-cleanup checkout. Neither section
establishes institutional approval or reproduces all experiments.

### Verified locally

| Check | Result | Scope |
|---|---|---|
| `make check` | PASS; **386 tests passed** | Locked core environment; Ruff lint/format, repository/stale-verdict policy, tests, wheel build and supported-module checks, documentation entrypoints. |
| Selected evidence checker | **21/21 groups PASS**, 22 JSON inputs | M1b summary/raw outcomes, six replacement receipts, fourteen paired lane receipts. See `selected_evidence_check.json` for checked and unchecked fields. |
| Canonical scientific audits | **9/9 PASS** | Historical `platform_tinker/reports/final` scope, compiled with cached native Tectonic in temporary output. Not a substitute for current-thesis checks. |
| Current builder regression tests | **30 PASS** | Missing/empty inputs, conversion/compile failure, stale/partial output, timeout, atomic publication, forced figure rebuild. Included in the 386 tests. |
| Extracted-package tests | **58 PASS** | Evidence/build/package tests plus seven demo tests; these overlap the core test count. Run without a source checkout in a temporary directory. |
| Extracted-package commands | PASS | Evidence checker, deterministic offline demo and self-test, all SHA256SUMS, rebuild of all 24 packaged figures and full thesis. No GPU or provider credentials needed. |
| Native source package build | PASS | Forces 27 local figure sources; thesis uses 24 mapped figures. Fresh nonempty PDF, then allowlisted ZIP and checksum. |
| ZIP integrity | PASS | Exact membership, SHA-256, sizes, manifest and checksums; no extraction/execution during verifier operation. |
| M1b statistical spot-check | PASS | SciPy independently reproduces paired t-test and CI; exact five-seed sign-flip p=0.0625 against PPO, 0.75 against G2. |
| Independent code reviews | PASS; no blocking findings | Gemini 3.1 Pro reviewed package/checker, build hardening, audit preservation, and final small changes. |
| CI configuration | YAML and pinned asset verified | Tectonic 0.17.0 Linux release SHA-256 and archive member checked locally. Remote GitHub Actions were **not run**. |

`uv sync --locked --extra dev` repaired the project environment without changing
the lock. The final check used Python 3.12.13, pytest 9.0.3, Ruff 0.15.11,
SciPy 1.17.1, and the repository's locked dependencies. Pandoc and Tectonic are
OS tools, not Python dependencies. Their TeX package cache was warmed with real
builds; the canonical scientific audit then ran cached-only. The older audit
was also checked not to modify any of 62 existing report artifacts.

### PDF quality checks

The rebuilt thesis is **265 A4 pages**. Automated word-bound checks covered all
pages: no text off the page and no text beyond the right body margin. The final
LaTeX log has no undefined citations/references, multiply-defined labels, or
missing-character diagnostics. Small font/box diagnostics remain; compiler
warnings are not silently called a clean log.

Visual samples covered the title/certificate/declarations/abstract, overview
table, two Chapter 6 plots, campaign-status figure, paired-results table,
references, and final evidence-map page. An overwide overview-table source
column and one repeated heading destination were corrected. This is sampled
visual QA, not an institutional formatting or editorial sign-off.

### Scientific integrity and exclusions

- **No training or benchmark grading was launched. No stored results were changed.**
- E13's manuscript parenthetical now says 1.8752 pp, matching direct per-item
  equal-environment aggregation. The stored 1.875 value was already correct.
- E3/E7/E12 follow their original shared generator's staged-rounding rule. For
  E12, the stored difference remains -0.0332; direct per-item delta is -5/151.
  Both are exposed by the checker; no arbitrary tolerance hides the distinction.
- The checker allows historical NaN only for PPO step-log ZVF, which is undefined
  at G=1. Checked scores and every other non-finite location remain fail-closed.
- E1's 1/190 and prior 4/110 are not pooled. E6's 90/812 is a lower bound with
  108 ungraded judge-dependent tasks. External closures stay external closures.
- The ZIP is selected review evidence, not the full repository, training data,
  weight checkpoints, private benchmark keys, or complete run provenance.
- Secret-pattern screening of selected text files found no recognizable API
  tokens or private keys. It is not a comprehensive security audit. The package
  intentionally contains identifying academic details and the existing signature.

### Remaining actions

The author must approve the AI-tool declaration and claims, obtain required
institutional signatures/certificates, confirm the portal's file/size/page rules,
and submit. External benchmark contracts need provider access and separate
permission/budget for any rerun. No institutional upload was made. This original
record predates the later publication commit `649c6beb3`; the repair verification
above is separately scoped.
