# Review 42 resolution

Source: review_42_zvf_misc_2.md. The provided source cuts off mid-sentence after
“Generate the relevant documentation tables from the”. Its remaining prose and
five-first list are missing. This resolution covers only visible findings.

## P0
1. Resume: exact requested checkpoint equality, required max-prompt/rank/tokenizer
   fields and full versioned identity. Reject mismatches before environment,
   dataset or provider setup. Reuse original resolved sampler for base snapshots
   too; persist identity before first sampling. Validate banked rows before reuse.
2. Truncation: explicit completeness/actual/planned/exploratory metadata; default
   theory loaders and seed aggregates reject partial/legacy-unverified pools.
   Explicit quality `--allow-partial` propagates its exploratory classification.

## P1
- Correlated T1: exact binary hypergeometric conditional truth; complete strata
  including remainder, population weights and conservative weighted Hoeffding CI
  instead of unjustified one-draw Wilson; Monte Carlo coverage intervals.
- Inputs: finite non-boolean binary rewards, G/count/grid domains, matching token
  counts, prompt IDs and p-hat consistency. Invalid domains fail before output.
- Builder count: dataset shortfall fails before sampler creation; full counts and
  per-prompt rewards/tokens required before completion.
- T2: exact stratum ZVF, explicit censored trials, nearest-rank quantiles, DKW
  quantile/ratio intervals, observable budget-event coverage CI. Unidentified
  quantities are null; the point check never establishes model validation.
- T3: distinct objective IDs and proxy-only/no-unique-learning-optimum scope;
  remove argmax-as-validation wording; regression for J(2)=J(3) points/mixtures.
- Provenance: pool-analysis-v2 requires all reviewed identity fields and compares
  the entire contract in seed aggregation. Dataset/tokenizer revisions are passed
  to loaders; parser/config/prompt hashes are derived rather than guessed.

## P2 and interpretation
The pipeline's binary-exact-v1 specification is shared and tested: finite binary
rewards, population variance, zero epsilon, K>=2. Continuous triage is explicitly
a different declared contract, not silently redefined. GU is reward contrast,
not proven gradient use. The executable T1 decision contract is Wilson/shuffled/
M>=64, exploratory, with curriculum global inference and readiness still blocked.
Prospective documentation supersedes old launch interpretations while preserving
historical slides, logs, stored results and manuscripts unchanged.

## Validation
13 synthetic offline regression tests pass:
`python -m unittest discover -s zvf-program/experiments-next -p test_analysis_contract.py`.
Includes end-to-end offline T1/T2/T3 entrypoints with in-memory synthetic pools and
mocked writes, pre-provider resume rejection, shortfall, every provenance-field
mismatch, censoring, domain failures and proxy ties. Compileall and diff-check pass.
No training, provider calls, paid jobs, stored result edits or generated artifacts.

## Explicit limits
Operator-declared model/checkpoint immutable identities still need verification
against provider records; this API offers no independent content verifier. New
seed aggregation intentionally requires identical prompt manifests; varying-prompt
replication needs a separately frozen contract. Calibration/finite-pool inference
is exploratory and cannot retroactively establish external-population coverage or
controller readiness. No missing review tail is inferred or claimed resolved.
