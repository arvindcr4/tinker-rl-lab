# Reviewed branch integration

This integration preserves slices `ec466b4f`, same-stack integration `b8ca50d7`,
and native-contract integration `2eb0f414` on main `72812969`.
The original batch-validation document records the historical slices-only gate;
this document records the subsequent merge reconciliation.

## Preserved contracts

- Thesis source reconciliation combines review12's interim-inference limits
  with main's percentage-point conversions. Main's O8 and pending execution
  platform paragraphs are retained exactly. Table 6.2 retains SHA-256
  `ba3cbee0e5f2555cc1d65101498c12ffa757bd5f0b0ce6d65a35cbb060f245cc`.
  All chapter Markdown parses successfully with Pandoc. No generated manuscript,
  PDF, or historical evidence is regenerated.
- Registry retains schema validation, strict JSON, v3 substantive-leaf coverage,
  union-of-declarations unknown coverage, exact finite numeric comparison,
  authenticated provenance and validator implementation binding. The earlier
  branch's idempotence tests are retained. Producer fixtures supply the required
  implementation/version receipt; no binding gate is relaxed. Native-contract
  input guards are exercised with synthetic authenticated v2 artifacts.
- Canonical same-stack v3 and explicitly named legacy review10/11 modules retain
  distinct evidence/statistical contracts. See `samestack_contract_integration.md`.
  No legacy record is silently promoted into canonical verified evidence.
- Native TRL/OpenRLHF paths use explicit execution-contract selectors, retaining
  fail-closed candidate/scientific modes and validated native execution paths.
  Resume tests preserve source/dataset/checkpoint identity and critic/optimizer/RNG
  state. Final actor receipts retain the validated published checkpoint path.
- Statistics retain explicit measurement contracts, finite-value guards,
  Student-t intervals, rank trimming, duplicate rejection and nonestimability
  for a single independent unit. Bootstrap resampling is overflow-safe.

## Verification

The combined offline gate passed **3,742 tests, 34 skipped, zero failed**
(seven warnings) across 176 modules in `pro_review_integration_test_modules.txt`.
The skips comprise ten missing retained E12 fixtures, six optional Atropos tests,
eleven explicitly disabled paid Tinker tests, three missing historical flagship
audit fixtures, three optional gitleaks checks, and one optional TensorBoard test.
All 490 changed Python sources parse; isolated Ruff F821/F822/F823 and whitespace
checks pass. Reproduce the selected gate with:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD/tests:$PWD/finish/e12_admission:$PWD/finish/progress_reporting_audit:$PWD/zvf-program:$PWD/platform_tinker/atropos:$PWD/zvf-program/audit:$PWD/zvf-program/next-submission:$PWD/zvf-program/sweep:$PWD/zvf-program/e1_wave10:$PWD/zvf-program/zvf-triage/src:$PWD"
python -m pytest -q -ra --import-mode=importlib -m 'not integration' $(cat docs/reviews/pro_review_integration_test_modules.txt)
```

The demo tree is byte-identical to tested PR20 `b8ca50d7`; its 25 JS tests,
build and strict audit of 71,226 stored completions therefore remain applicable.
No browser QA is claimed: Chromium startup was unavailable in that environment.
Historical results, registry entries, generated thesis artifacts, excluded
`Platform.jsx` match main `72812969` byte-for-byte.
`platform_calibration.py` differs from that reference: it rejects an empty
platform set, reports unavailable inference for fewer than two pairs, retains
empty contrasts, and reserves unavailable tests in the Holm family without
presenting their placeholders as test results.
No training, provider jobs, or historical result migration is part of this work.
