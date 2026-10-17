# Pro review batches: offline validation and delivery boundary

Requested base: `baaeae72ff86b2f626354d4e2c17c3394dabacb7`.
Branch: `fix/pro-review-slices`. This combines reviews 01–12, 22 and 29, with
at least one separate commit per supplied review. Reviews 22 and 29 are partial;
missing continuation is not inferred. Per-review resolution documents distinguish
repaired code from unavailable execution evidence and unresolved scientific claims.

## Final selected regression gate

Code commit tested: `b7915c7050dcf48591e751dfeb581735c28dae62`.
The follow-up adds documentation and removes one trailing blank source line; it does not change tested behavior.

- 104 selected test modules: **2,628 passed, 30 skipped, zero failed**, seven warnings
- Skips: 10 absent retained E12 asset fixtures; 6 optional Atropos dependency tests;
  11 explicitly gated paid Tinker integration tests; 3 absent historical flagship
  audit fixtures. Synthetic admission/guardrail regressions run separately within
  the passing set. No paid test was enabled
- Existing `tests/test_samestack_statistics.py`: all 32 cases included and passed
- Hypothesis 6.170.1 is installed and both property modules ran; properties and
  numerical objective regressions passed after preserving float64 observation
  precision in detached snapshots
- Python syntax parsing and isolated Ruff F821/F822/F823 checks cover changed
  Python sources; targeted full lint/format checks also passed where documented
- `git diff --check` passed

Reproduce the selected suite using the paths in `pro_review_test_modules.txt`:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/finish/e12_admission:$PWD/finish/progress_reporting_audit:$PWD/zvf-program:$PWD/platform_tinker/atropos:$PWD/zvf-program/audit:$PWD/zvf-program/next-submission:$PWD/zvf-program/sweep:$PWD/zvf-program/e1_wave10:$PWD/zvf-program/zvf-triage/src:$PWD"
python -m pytest -q -ra --import-mode=importlib -m 'not integration' $(cat docs/reviews/pro_review_test_modules.txt)
```

Interpreter: Python 3.12.14, CPU Torch 2.7.1, pytest 9.1.1. Statistical checks use
SciPy 1.18.1/statsmodels 0.15.0. These are local verification versions, not a
claim that a prospective GPU/provider training environment has been validated.
This is a selected regression gate, not an assertion that every repository test
or installed-wheel/native-document CI job was executed.

## Required demo checks

The slices branch does not modify `submission/demo/web` sources.
`npm run build` passed; its generated tracked HTML was restored unchanged.
`node tests/grade_all.mjs` graded **71,226 stored completions from 27 runs with zero
disagreements**. No stored scores or generated demo data were rewritten.
The separate same-stack PR contains its own corrected grader/demo changes and
validation; this branch does not silently merge that PR.

## Thesis source-only boundary

Thesis review 12 has separate source-only commits. Only chapter `.md` files and
`frontmatter.tex` changed under the authorized thesis directory. Generated
`thesis_master.tex`, PDFs, figures, and stored evidence remain unchanged.
Table 6.2 is byte-identical to the starting snapshot (SHA256
`ba3cbee0e5f2555cc1d65101498c12ffa757bd5f0b0ce6d65a35cbb060f245cc`), and the existing
v2 result paragraph's numeric sequence is preserved. The O8 sentence is unchanged;
the other session's unit-conversion and pending-platform paragraph are not duplicated.
All chapter Markdown parsed with Pandoc to temporary JSON; scoped source and table
checks passed. Synthetic arithmetic checks cover the loss/gradient counterexample,
Jensen counterexample, E11 sensitivity, correlation direction and finite-group
advantage identity. No rebuilt PDF or visual-layout approval is claimed.

## Deliberate unresolved boundaries

- No training, model sampling, paid cloud/provider job, main push, or merge
- No historical scientific results, weights, checkpoints, datasets, signatures,
  or registry entries changed. JSON changes are source schemas/configuration and
  derivative-code provenance; historical source-definition hashes are preserved
- Incomplete retained evidence cannot be fabricated. New validators may reject
  old records until a separately reviewed migration supplies actual bindings
- Historical manuscript claim spans remain unbound where not authored, and 11
  ambiguous stale-verdict passages remain flagged rather than blanket-allowlisted
- Invalid SkyRL/unsupported real veRL, missing trusted HumanEval isolation and
  incomplete WebArena service/evaluator contracts fail closed
- Candidate generation/reward/normalization/masking changes have prospective
  protocol identities; historical experiments were not rerun or relabeled
- Excluded `platform_calibration.py` and `Platform.jsx` were not imported from
  newer main. Main advancement and overlap with the separate same-stack PR need
  explicit reconciliation before merging; no automatic merge is authorized
- The 22/29 aggregate-only synthesis paths cannot publish new confirmatory claims
  without the missing independent-unit and metric contracts
