# Same-stack Pro review resolution

Base: `baaeae72ff86b2f626354d4e2c17c3394dabacb7` (`origin/main`).
Branch: `fix/pro-review-samestack-v2`.

This is a code and offline regression repair, not a new experiment. No training,
paid service jobs, historical run records, stored verdicts, or historical analysis
JSONs were changed. The corrected execution protocol is
`samestack-gsm8k-cot-v3-policy-exact-eos`; its runtime lock is explicitly a
candidate, not a GPU-validated environment. Historical inputs cannot be upgraded
into prospectively registered or fully reproduced evidence by this patch.

## Finding-by-finding disposition

- C1 — fixed: explicit neutral sampling/evaluation configurations and inherited-default clearing; tiny random CPU-model tests compare generation scores with training log probabilities
- C2 — fixed: exact rational equality, valid balanced boxed syntax, safe zero denominators, and shared full-string formatting fixtures; independent Python and JavaScript audits retain all original scores (71,226 comparisons across 27 runs, zero disagreements)
- C3 — fixed: loss and attention include the first EOS, including pad=EOS, and exclude subsequent padding; both stop IDs, immediate stopping, truncation and left padding are tested
- C4 — fixed: finite/minimum-sample validation, explicit degenerate-statistic behavior, zero-discordance McNemar, and validated Holm inputs across the publication paths
- C5 — fixed: shared schema checks run identity, resolved configuration, ordered arrays, metrics, hashes and provenance; contradictory and empty staging inputs fail closed
- C6 — fixed: producer failure cannot reuse a stale clean audit; strict validation requires a fresh invocation-specific audit bound to unchanged registry inputs
- S1 — fixed: a canonical analysis artifact binds the planned cohort and effective stack identity, reports exclusions/missing runs, and supplies the demo's paired comparisons
- S2 — skipped: `platform_calibration.py` and `Platform.jsx` are absent from this base; no frozen-evaluator or crossover experiment was run
- S3 — fixed: shared policy prompt schedules and separate warm-up streams; policy-update exposure is distinguished from total prompt presentations/generations
- S4 — fixed: three fixed primary hypotheses retain missing p=1 slots; G2–PPO is exploratory; retrospective/interim analyses cannot claim prospective confirmation
- S5 — partly: 90% equivalence and 95% difference intervals, family-adjusted TOST and directional labels are implemented; scientific justification of ±2 percentage points and a prospectively registered final claim are not supplied
- S6 — fixed: seed-level inference is primary; item-only and pooled diagnostics retain their distinct targets; sign-flip assumptions and finite-seed resolution are explicit
- S7 — partly: two-sided noncentral-t power, input guards and adaptive MDE roots are repaired; final-design family/TOST/interaction power needs justified variance/correlation assumptions and remains unclaimed
- S8 — partly: the implemented prompt-end detached linear terminal-reward PPO-style baseline and constant-mean MSE are named accurately; broader algorithm/task/model replication requires new experiments and was not performed
- R1 — partly: future runs bind exact candidate dependencies, immutable revisions, effective settings, source/runtime/item identities and schedules; historical unknowns remain unknown and GPU/PEFT compatibility is not asserted
- R2 — fixed: immutable identities, validated complete-bundle reuse, temporary transfers, hash verification, adapter/tokenizer preservation and atomic completion markers; optimizer/RNG training resume is explicitly unsupported
- R3 — fixed: unsupported defaults remain unknown, stale generated outcomes are not inherited, and extraction lineage is recorded; historical entries are deliberately not regenerated
- R4 — fixed: revision exemption removed and matching evidence expanded; no-known-differences, completeness and satisfaction of matching requirements are distinct
- Q1 — partly: neutral protocol/training/schema helpers and canonical publication inference are exercised directly, with independent cross-language grading fixtures and bounded enumeration; legacy runner AST compatibility tests remain, and real platform execution is untested
- Q2 — fixed: guarded analysis/Colab entry points, pure interfaces, effective configuration reporting and strict JSON; unexpected nonfinite values fail instead of being silently repaired
- Q3 — fixed: source literals are extracted or rejected on mismatch, and coverage definitions distinguish reporting completeness from implementation presence
- D1 — fixed: unconditional equivalence/improvement claims are replaced by validated canonical status and per-arm descriptive gains
- D2 — fixed for available files: v1 is superseded without unsupported causal attribution; absent Platform.jsx wording is skipped
- D3 — fixed for available files: empty/missing/one-pair/overlap states are guarded and collection is separated from analyzability; absent Platform.jsx calibration badges are skipped
- D4 — partly: recomputation claims are corrected and full available-input hashes plus missing provenance are exported; absent historical dependencies cannot be reconstructed from summary hashes

## Deliberate fail-closed outcomes

- Legacy records lack the new immutable manifest and contain 210 nonstandard NaN
  diagnostic tokens. They remain descriptive historical evidence rather than
  verified primary runs. Independent read-only audits explicitly handle those
  legacy diagnostic tokens without rewriting files or modifying quoted text.
- The retrospective plan cannot establish prospective registration or a justified
  equivalence margin. No unconditional final equivalence verdict is emitted.
- Registry generator `--check` reports 12 stale historical entries because the
  user requested unchanged stored evidence. The validator's fresh-output and
  schema checks pass independently; these are different checks.
- Live browser smoke is blocked by this environment's browser/sandbox launcher;
  build, export, server-side rendering and contract tests provide the available
  UI validation. No browser-security restrictions were bypassed.

## Historical registry entries intentionally not regenerated

`colab-open_dapo_e3`, `colab-open_drgrpo_e3`,
`colab-open_grpo-adaptiveg_e3`, `colab-open_grpo_e3`,
`openrlhf_grpo_qwen3-8b_gsm8k`, `tinker_dapo_qwen3.5-4b_gsm8k`,
`tinker_drgrpo_qwen3.5-4b_gsm8k`, `tinker_grpo_qwen3-8b_gsm8k`,
`tinker_grpo_qwen3.5-4b_gsm8k`, `tinker_gspo_qwen3.5-4b_gsm8k`,
`trl_grpo_qwen3-8b_gsm8k`, `verl_grpo_qwen3-8b_gsm8k`.

## Final local validation (17 October 2026)

- Python: 281 passed across `tests/test_samestack_statistics.py`,
  `tests/test_samestack_artifact_contract.py`, `tests/test_samestack_analysis_review.py`,
  `tests/test_samestack_protocol.py`, `tests/test_registry_stackdiff.py`,
  `tests/test_registry_evidence.py`, and `tests/test_provenance.py`
- `npm run build`: passed against a fresh temporary canonical export supplied
  through `DEMO_DATA`; tracked historical data/HTML retained unchanged
- `node --test tests/test_*.mjs`: 24 passed (math/grading, Python parity,
  canonical artifact contracts and server-side rendering)
- `python tests/test_export.py`: 5 passed
- `node tests/grade_all.mjs`: 71,226 completions from 27/27 runs; zero
  disagreements, missing sidecars, invalid inputs, length mismatches or invalid scores
- `python platform_hybrid/registry/query.py validate-strict`: PASS;
  28 entries, 18 deltas, zero schema failures/errors/warnings, 54 informational findings
- Ruff on changed/new Python code and `git diff --check`: passed
- Vendor-secret pattern scan over 49 changed/new files: no matches; this bounded
  check is not a claim of a comprehensive security audit

The full repository test suite and live GPU/browser execution were not claimed.
The preserved historical source and result files have no Git diff.
