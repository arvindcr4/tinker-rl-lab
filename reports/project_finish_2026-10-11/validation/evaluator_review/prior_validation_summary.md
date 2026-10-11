# Project finish validation — 11 October 2026

The final complete local test run passed **979 tests**, with **1 platform skip**
and **1 documented expected failure**. Measured coverage was **81.84%**, above the
repository's 77% gate. The run included the two native LaTeX integration tests
and the property tests. See `delivery_full_coverage.log` and `delivery_checks.json`. The ZIP includes compact receipts; detailed logs remain in the full local checkout.

## Final gates

| Check | Result | Evidence |
| --- | --- | --- |
| Complete pytest suite with coverage | 979 passed, 1 skipped, 1 xfailed; 81.84% coverage | `delivery_full_coverage.log` |
| Public release/integrity tests | 211 passed | `final_public_check.log` |
| Ruff and repository policy | PASS | `delivery_static.log` |
| Formatting | PASS, 116 files | `delivery_static.log` |
| Mypy | PASS, 18 source files | `delivery_static.log` |
| Lock consistency and required documentation | PASS | `delivery_static.log` |
| Current-change redacted secret scan | PASS | `delivery_static.log` |
| Independent final numerical regression review | 47 targeted tests passed; 18 additional nonfinite combinations rejected | `delivery_checks.json` |
| Final number-audit regression tests | 14 passed, included in full suite | `delivery_full_coverage.log` |
| Nine structured submission/scientific audits | PASS, zero issues | `audit_suite.log` |
| Selected thesis arithmetic | PASS, all 21 checks | `submission_check.log` |
| Offline demo self-tests | 7 passed | `submission_check.log` |

The only skipped test requires `EmbeddingBag` on MPS, where it is unavailable.
The remaining strict expected failure records an existing approximately
1.4e-9 advantage residual for a constant fractional reward group; this task
left that small numerical residue unchanged.

## Numerical corrections

- The same-stack v2 Student-t CDF handles tiny nonzero statistics without
  evaluating `log(0)`. The limiting CDF agrees with the analytic Cauchy formula
  at one degree of freedom and with the SciPy reference at the other tested
  degrees of freedom.
- Identical paired arms now return a two-sided p-value of 1 and t-statistic 0.
  Nonzero constant differences preserve the t-statistic's sign. The same
  zero-variance branch rejects equivalence at or beyond the prespecified TOST
  margin; fewer than two shared seeds fail explicitly.
- All-tied Mann–Whitney inputs return U = n*m/2 and p = 1, avoiding a NaN from
  the zero-variance asymptotic calculation.
- Three strict expected-failure regressions are now ordinary passing tests.
  New core tests exercise these cases even without Hypothesis installed.
- Paired differences and Mann–Whitney observations reject nonfinite inputs before
  calculating significance, preventing invalid data from entering degenerate branches.

These changes correct future calculations at degenerate inputs. Historical
results, native benchmark definitions, scores, and receipts were not regenerated
or edited by this validation work.

## Reproduction and environment

The existing project environment has Python 3.12.13, pytest 9.1.1, Ruff 0.16.10,
mypy 2.4.0, and Tectonic 0.17.0 on macOS arm64. Full details are in
`environment.json`; the final edited source hashes are in `delivery_checks.json`. Earlier source snapshots are preserved as earlier validation evidence.
Hypothesis was absent from the shared `.venv`, so the final run used its exact
`uv.lock` version in a temporary uv overlay without changing that environment:

```bash
COVERAGE_FILE=reports/project_finish_2026-10-11/validation/.coverage_delivery \
uv run --no-sync --with hypothesis==6.168.3 python -m pytest tests/ \
  -q -rsx --cov --cov-config=.coveragerc --cov-report=term
make lint format-check typecheck lock-check docs-check secrets
make public-check
```

No paid model evaluation or training was launched. Arithmetic/integrity checks
do not independently verify historical model performance or establish a native
benchmark score. Submission building is handled separately from these logs.

## Preserved diagnostics

`core_coverage.log` is a superseded initial diagnostic. The harness had injected
`WANDB_MODE=offline`, causing 21 mocked tracking tests to reject the environment
before reaching their intended checks. The canonical rerun without that
override passed 898 tests at 81.06% coverage before the numerical fixes and
expanded final run. This was a harness error, not a production regression.

`property_tests.log` records the original 29 passes and four expected failures.
The final suite resolves the three statistical failures. The first targeted
numerical run also exposed cancellation in SciPy's near-zero df=1 reference;
the final regression uses the exact Cauchy CDF for that case. Earlier logs are
retained for transparency; `delivery_checks.json` and the delivery logs above are the
authoritative completion evidence. The earlier `final_results.json` describes the
972-test snapshot before the final seven tests were added.

The first delivery run caught a README navigation addition that changed a frozen
publication-manifest entry (two integrity tests failed). The addition was removed;
the original README and publication checks now pass. The review pack has its own
entry guide. The full rerun passed all 979 non-skipped/non-expected-failure tests.
