# Evaluator-review validation — 11 October 2026

The preceding evaluator review's complete local test run passed **986 tests**, with **1 platform skip** and
**1 documented expected failure**. Coverage was **82.05%**, above the repository's
77% gate. Both native LaTeX integration tests and the property tests were included.
`delivery_checks.json` binds the commands, final source files and local log hashes.

| Check | Result | Evidence |
|---|---|---|
| Complete suite with coverage | 986 passed, 1 skipped, 1 xfailed; 82.05% | `evaluator_review/full_coverage.log` |
| Ruff, formatting and mypy | PASS | `evaluator_review/static.log` |
| Lock consistency, required docs, changed-file secret scan | PASS | `evaluator_review/static.log` |
| Number-audit regression tests | 21 passed again after Colab corrections | `colab_followup/audit_tests.log` |
| Frozen source-analysis tests | 4 passed | `colab_followup/analysis_tests.log` |
| Colab CPU reanalysis | PASS; local/remote numeric agreement within 1e-12; runtime released | `../colab_validation/execution_receipt.json` |
| Selected thesis number audit | 967 occurrences: 945 pass, 22 context; 34 printed-count calculations | `../thesis/provenance_audit.json` |
| PDF build | PASS; 312 pages | `colab_followup/thesis_build.log` |
| All-page PDF text and physical bounds | PASS | `pdf_checks.json` |
| PDF visual review | Earlier all-page layout thumbnails; current changed pages and full-size spots | `pdf_checks.json` |
| Twelve-slide presentation | Structure, geometry, font policy and import PASS; full-size visual review | `deck_checks.json` |

The full 986-test suite was not repeated for the later document and package update.
The 21 audit regression tests and four new analysis tests were rerun; the selected
thesis audit, PDF/deck checks and clean-extraction acceptance cover this revision.

The platform skip requires MPS `EmbeddingBag`. The strict expected failure records
an approximately 1.4e-9 advantage residual for a constant fractional reward group.
It remains documented rather than silently treated as exact zero.

## Scientific and audit corrections

`../EVALUATOR_REVIEW.md` records the mathematical, statistical and presentation
corrections. The audit now directly recomputes seed-paired mechanism statistics
from the retained per-run TSV, and checks the historical coefficient-product
ratios against their stored rows without endorsing causal mediation. Missing,
duplicate, unexpected and nonfinite rows fail explicitly. The corrected E12
binding checks its descriptive fraction without endorsing an independent-item
Wilson interval for clustered rubric items.

The earlier software fixes remain covered: stable Student-t limits, signed
constant differences, paired TOST boundaries, all-tied Mann–Whitney and explicit
nonfinite-input rejection. No stored model score or raw experiment was edited.

## Commands and scope

```sh
COVERAGE_FILE=reports/project_finish_2026-10-11/validation/.coverage_evaluator \
uv run --no-sync --with hypothesis==6.168.3 python -m pytest tests/ \
  -q -rsx --cov --cov-config=.coveragerc --cov-report=term
make lint format-check typecheck lock-check docs-check secrets
uv run --no-sync python tools/audit_thesis_numbers.py --check \
  --thesis-dir reports/project_finish_2026-10-11/thesis
```

Hypothesis uses the locked version in a temporary uv overlay without modifying the
shared environment. The full logs, rendered pages and finalization receipts remain
in the checkout; compact checks are bundled in the review archive. The archive is
verified after clean extraction with the manifest checker, all seven offline demo
self-tests, the default demo, the registry example (expected R5/exit 5), and a
replay of the frozen Colab analysis with its four acceptance tests.
`archive_checks.json` records those exact archive checks outside the archive itself.

Earlier logs and `evaluator_review/prior_delivery_checks.json` preserve the earlier
979-test, 305-page delivery snapshot. The original diagnostics include an invalid
WANDB_MODE override and a removed root-README change caught by frozen-manifest
tests. Those logs are historical; the current records above are authoritative.

Checks establish local software behavior and document consistency. They do not
authenticate every historical run, independently validate every scientific claim,
certify native PowerPoint rendering, or complete institutional submission.
