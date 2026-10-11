# Linux VM validation and fail-closed follow-up — 11 October 2026

This is an additive software-validation note. It does not replace the frozen
thesis, authenticate a training run, supply missing scientific provenance, or
record independent thesis approval.

## Source and environment

The Linux checkout was fetched from the published repository, with both branch
tips verified before validation:

- `fix/provenance-openrlhf-20261011`:
  `ca799b872f8e65f7f76b887b99a3d0f967c2985a` (validation base)
- `review/local-pes-source-20261011`:
  `71c60b6a90643557a8611e6ec1906b07e242c891` (verified, not merged)

Runtime: Linux x86_64, Python 3.12.14, uv 0.12.23. Development dependencies were
installed from the unchanged lock with `uv sync --locked --extra dev`. No Mac
environment or credentials were copied. No GPU training, provider evaluation,
model upload, or frozen-result regeneration was performed.

## Confirmed software corrections

### Duplicate result identities

The verifier previously accepted duplicate records for the same experiment and
canonical seed as two successful experiments. This occurred both within one
trainer log and across JSON/log copies. All observations sharing a duplicated
identity are now `UNVERIFIED`; no arbitrary first observation wins. Distinct
seeds remain distinct even with an explicitly seed-independent reference.

### Single-seed uncertainty and seed ingestion

One score retains its descriptive mean but cannot estimate its across-run
standard error. The bootstrap helper now returns NaN interval bounds for one
score, and standard error returns NaN instead of zero. Tables, CLI output, and
plot legends explicitly say uncertainty is not estimable; a single-seed curve
has no confidence band. CLI exports include an `uncertainty_status` column.

This deliberately changes the old zero-width-interval behavior. Consumers must
handle unavailable uncertainty rather than treating it as zero. Having two or
more values does not itself establish independence or adequate statistical power.
See the [SciPy standard-error contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sem.html).

Malformed `seed_*` entries are now errors rather than silently omitted data.
Seed names must contain a nonnegative decimal integer; canonical aliases still
trigger the existing duplicate-source guard. Empty/missing sources remain errors.
Bootstrap resample counts must be positive integers, not booleans.

### Synthetic OpenRLHF wrapper

The synthetic reward and loss streams now depend on the configured seed.
Repeated identical configurations reproduce the same synthetic traces; different
seeds change them. The wrapper raises rather than silently downgrading an
available real runtime to synthetic execution. Step exceptions propagate rather
than returning a partial trace as successful completion.

These are plumbing checks, not estimates of real model reward or comparisons
between frameworks. Canonical native `max_steps` remains unsupported.

### Test isolation

Mocked online-tracking tests now select their intended fake online mode explicitly.
An inherited `WANDB_MODE=offline` no longer prevents their downstream guards from
being exercised. Provider implementations remain mocked. All 122 tests in that
module pass under an externally supplied offline setting.

## Validation checkpoint

- Baseline CPU suite: **1,413 passed, one expected skip, two LaTeX tests
  deselected; 84.74% configured coverage**.
- Changed-tree CPU suite: **1,451 passed, one expected skip, two LaTeX tests
  deselected; 84.76% configured coverage**, in 131.75 seconds.
- Focused result-identity/OpenRLHF tests: **92 passed**.
- Focused statistics/analysis tests: **131 passed**.
- Repository Ruff lint/format and configured 20-file mypy checks: passed.
- Rebuilt wheel and nine-module content verification: passed.
- All 237 stdlib public-release checks, current selected thesis arithmetic, and
  the seven-test offline demo: passed.

The CPU run used `pytest tests/ -q -m 'not latex' --cov
--cov-config=.coveragerc --cov-report=term`, with CUDA hidden, Hugging Face offline,
and W&B offline. This passes the existing 77% coverage gate; it is not a full
native-document validation result.

## Native-document blocker

The installed `pdflatex` executable lacks `pdflatex.fmt`, so its first compilation
fails before processing the paper. The CI-pinned Tectonic 0.17.0 executable was
installed and checked against SHA-256
`8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`.
Its official default bundle request returned HTTP 403 at
`https://relay.fullyjustified.net/default_bundle_v33.tar.index.gz`.

No access restriction was bypassed. Native LaTeX checks remain blocked; the
scientific audit cannot be reported as passing on this VM. Strict provenance
and independent-thesis-review gaps are unchanged.
