# Review 27 early scaling repairs

Scope: `scaling_law_iter45/49/53/57/61/65/69.py` and `_review27_rank.py`.

## Corrected implementation

- Iter45 requires explicit trace file, SHA-256 and run ID; verifies source bytes, identity and ordered finite length; rejects explicitly synthetic/reconstructed evidence. Summary statistics cannot manufacture or initialize a measured trajectory fit. Semi-log slopes are named as such. Allocation optimum checks are unavailable.
- Iter49 rechecks original source provenance and excludes duplicate identities, synthetic evidence and invalid targets. Its additive model is descriptive; candidate ranking does not imply a budget-dependent optimum.
- Shared statistics implement average-rank Spearman, deterministic permutation inference, SciPy tau-b, and unavailable constant-input results. Iter53 uses average ranks for rank-displacement tables and renders current outcomes, including unavailable statistics.
- Iter57 uses explicit failure statuses, named plot/table records, correctly directed gap predicates and consistent base-10 exponent fits. Its quantile/late-window quantity is a finite-trace reward summary, not a validated asymptote. Bounds are optimizer diagnostics, not identifiability proof.
- Iter61 distinguishes extreme step means from recorded group ZVF, excludes failed fits from denominators, and keeps missing measurements unavailable. Cross-cohort comparisons cannot pass without matched protocols. Generated prose follows current records.
- Iter65 fits continuous hinge segments against actual line SSE, requires signed improvement, distinguishes three slope counts from six shape conditions, exposes reachable drift/collapse classes, and preserves fixed short-trace schemas. Conditional boundary resampling uses residual blocks around the fitted trend and a repeatable seed. Ineligible short traces remain in per-trace outputs but are excluded from phase denominators and plotted boundary alignment.
- Iter69 uses an initial-positive paired ACF sequence, keeps constant-trace ESS unavailable, fits saturation in original reward space, and excludes bound hits from rate-correlation inference. Retrospective results are descriptive, not a validated forecasting proxy.
- Imports do not create result directories. Each main accepts explicit source/output roots, defaults to a separate `review27_v3` candidate directory, and writes input/script/helper checksums, revision and dependency versions. Historical stored outputs were not edited.

## Validation

CPU-only regression command:

```
MPLCONFIGDIR=/tmp/review27-mpl /tmp/tinker-review-venv/bin/python -m pytest -q \
  tests/reviews/test_review27.py tests/reviews/test_review27_rank.py \
  tests/reviews/test_review27_early_evidence.py \
  tests/reviews/test_review27_early_narratives.py
```

Result: **72 passed, 1 skipped** (the verified LaTeX-format limitation below).

Includes explicit identity/hash/path/length failures, synthetic-source rejection, summary-independence, constant/tied/reversed ranks, failed optimizer mocks, signed continuous phase fixtures, residual-block repeatability, short-first-row serialization, exact saturation fits, plot-record checks, reversed report outcomes, import side-effect checks, and temporary-only full synthetic pipelines. Compileall and `git diff --check` are additional checks.

Real-input CPU smoke runs of iter57/61/65/69 wrote only to temporary directories; resampling counts were reduced for smoke validation. These are execution checks, not publication-grade uncertainty reruns. Iter45 correctly stopped because legacy frontier summaries do not provide explicit measured-trace identities. No training or paid jobs ran.

## Remaining evidence requirements

1. The legacy frontier TSV has no explicit trace file/hash/run-ID mapping. Iter45 and downstream measured-fit analyses cannot be promoted until that mapping is independently supplied and verified. Filename/cohort guesses are deliberately forbidden.
2. Recorded group-ZVF fields and a separate G=8 summary do not establish matched model, task, split, group size, training protocol or weighting. The cross-pillar claim remains unavailable.
3. Compute proxies do not establish actual FLOP/token accounting, feasible allocations, a budget-dependent optimum, or an architectural causal effect.
4. Short nonstationary traces, interior optimizer solutions, finite-trace quantiles and retrospective thresholds do not establish parameter identifiability, asymptotic reward, preregistration, equivalence, calibrated boundary coverage or held-out forecast performance. These require protocol/evidence decisions rather than code-generated claims.
5. `pdflatex` exists but its `pdflatex.fmt` is missing. The optional compilation regression is skipped for that verified environment limitation; escaping, table shape and row endings are still tested. No successful TeX compilation is claimed.
6. Corrected candidate results have not replaced or validated historical publication artifacts. Full-size reproducibility runs and scientific review remain necessary before promotion.
