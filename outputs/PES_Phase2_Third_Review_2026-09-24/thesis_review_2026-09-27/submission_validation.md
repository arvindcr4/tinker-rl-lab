# Phase 2 submission verification record

Prepared on 2026-10-02 (UTC) from the current working tree,
base revision `c0be4a313`. These checks validate the identified Phase 2 thesis
review package. They do not establish institutional approval or reproduce all
experiments in the repository.

## Verified locally

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

## PDF quality checks

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

## Scientific integrity and exclusions

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

## Remaining actions

The author must approve the AI-tool declaration and claims, obtain required
institutional signatures/certificates, confirm the portal's file/size/page rules,
and submit. External benchmark contracts need provider access and separate
permission/budget for any rerun. Nothing was uploaded, committed, or pushed.
