# Analysis input and validation contracts

The October 2026 tooling fixes tighten ambiguous input handling without changing
frozen publication data, reference numbers, manuscript figures or release receipts.

- `load_multi_seed_results` accepts one JSONL file per canonical integer seed.
  The existing single-file filename flexibility remains. Multiple files, including
  files in alias directories such as `seed_042` and `seed_42`, now raise an error
  naming both sources. JSONL records must be objects. Files are never concatenated
  or selected by alphabetical preference.
- Trainer-log accuracy values are complete decimal tokens, optionally signed and
  in scientific notation. CLI values are fractions; legacy final-report values
  must carry `%`. Values must be finite and within [0, 1] after percentage
  conversion. Trailing text and incomplete tokens fail instead of matching a
  numeric prefix. Existing decimal trainer output remains supported.
- `welch_ttest` still uses SciPy's unequal-variance Welch test for its p-value.
  Its `effect_size_cohens_d` uses the sample-size-weighted pooled sample variance:
  `((n_a-1)*s_a^2 + (n_b-1)*s_b^2)/(n_a+n_b-2)`, with sample variances. Identical
  constant groups report zero effect; distinct constant groups report signed
  infinity. Each group requires at least two finite observations.
- `plot_learning_curves_with_ci` requires finite real metric values, including
  NumPy integer/floating scalars. Missing values and booleans are rejected. When
  `step` is present, every observation must provide it, steps must be strictly
  increasing, and every seed within an algorithm must share exactly the same grid.
  Different schedules, duplicate or reordered steps and unequal run lengths fail;
  no truncation, zero imputation or interpolation occurs. Entirely step-less
  legacy input is still supported on equal-length grids with an "Observation
  index" axis. Explicit steps and observation indices cannot share a plot. Bands
  use mean ± 1.96 standard errors, a normal approximation.
- Public package tests now use standard-library unittest, including temporary
  files, exception checks and parameter subtests. `make public-check` actually
  collects them without installing pytest.

The statistics CLI reaches the seed loader. The effect-size and plotting changes
also affect direct API users; repository caller inspection found no production
callers of those two functions outside their definitions. The verification CLI
uses the corrected log parser. No effect on published stored results was
established by the counterexamples or caller review, so none were regenerated.
