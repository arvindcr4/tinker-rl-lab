# Review 01 resolution

Source: user-supplied `review_01_modal_training.md` (all numbered findings, design
concerns, reproductions and five priority changes reviewed).

Protocol: [candidate v2](../protocols/modal_training_v2.md). No training, paid jobs,
model uploads, credential/log access, or stored measured-result edits. No affected
scientific evidence was regenerated. Exact-image OpenRLHF/GPU integration remains
unvalidated; the independent 0.8.4 Modal launcher is unchanged.

## Numbered findings

1. **Fixed** — real is the public default; missing dependencies raise; explicit seeded simulation has a distinct fixture schema, no assumed trend/framework ranking, and empirical guards and the direct framework-comparison consumer reject it; the consumer no longer launches fallback training/simulation and requires revalidation of untyped legacy artifacts; tests cover routing, seed changes and rejection
2. **Fixed** — checked subprocess exit/timeout, atomic unique attempt directories, failed/incomplete/completed states and null unavailable metrics; no upload/stale-checkpoint reuse; exit-17 and missing-history regressions verify no accepted result
3. **Fixed within the candidate driver** — one validated effective config controls command and metadata; GRPO-only/unsupported settings fail early; seed/group/clipping/temperature/batches/lengths/optimizer/data/project/topology/budget are wired; full installed-backend acceptance is pending
4. **Fixed in the reviewed driver** — environment authentication and a nonsecret enable value keep the key out of argv/logs/manifests; captured-output regression passes; historical log inspection and credential rotation were not performed
5. **Partly fixed** — explicitly new 0.8.0 candidate pin, no unsupported ref_pretrain flag, absolute local reward callback, option-surface CLI preflight and actual callback fixtures; full parser/distributed callback validation in a pinned image is unavailable and not claimed
6. **Partly fixed** — exact unique child W&B ID, finished-state check, full scan of train/reward, validated finite contiguous steps and local attempt-bound metrics mirror; worker-local source-of-truth/counters require backend instrumentation and remain unavailable
7. **Fixed at the launcher contract** — independent configured rollout/prompt/completion/update budgets, exact divisibility and retained-prompt cap; the 30-rollout/240-update counterexample is tested; actual optimizer updates/generated tokens remain null rather than inferred measurements
8. **Fixed** — shared tested repository/results roots, platform_hybrid input paths, versioned output directories, required-input checks and import-safe modules; regression imports prohibit directory creation
9. **Fixed conservatively** — task intervals stay primary; task means alone produce no aggregate CI/significance, singleton ±.05 and fake bootstrap metadata removed; P6 never edits registry verdicts; all three toy effects tested
10. **Partly fixed** — StateFlow no longer computes retention from future suffix rewards or claims stop-policy effectiveness; retrospective labels/held-out checkpoint-only helper and censored events; actual stopped-checkpoint evaluation is unavailable; [0.8]+[1.0]*39 fixture never claims retained-policy reward
11. **Fixed** — correct frozen-reference beta/2 local gradient scaling, pair-dependent sigmoid/tied-reference caveat, IRPO optimum/feasibility unmeasured; no copied optimum or GU-ratio feasibility claim; analytic/autograd regressions
12. **Fixed** — unique matched-seed dataset drives paired statistics, sample-SD Cohen d, sorted duplicate/gap-validated terminal windows and optional explicit final-step check; unknown final budgets stay INCOMPLETE in JSON/TSV with per-run completion status; sign-reversal, -sqrt(2), ordering, duplicate and short-prefix fixtures
13. **Fixed** — tied Spearman constants return unavailable, unique budgets and high-budget G*=32 with a real SATURATED token are required, CFR fits actual observed T and abs(delta)*T, NIAH censoring retained, StateFlow requires all transitions; empty/constant/missing/censored adversarial tests
14. **Fixed** — weighted out-strength PPR with fixed restart/dangling handling, total-mass correlation removed, true BM25@25 and stable digest baseline; triangle 1/2/50-step, dangling, rank-15 and cross-process-hash tests
15. **Fixed with explicit design limitation** — bounded warmup, candidate identities, equal nonexhaustive budgets, true per-anchor quantiles and stated early-discovery count; impossible nine-anchors-in-eight-global-evaluations criterion is flagged, never replaced by exhaustive maxima
16. **Fixed** — shlex-joined absolute argv and separately quoted display text, explicit supported backend and bounded pilot; v3 attempts verify actual YAML bytes, command/seed, trainer/runner source and runtime identity before success skip, reject stale receipts, and isolate each retry config/checkpoint/W&B identity; hostile paths, stale-command/config/source/version and retry regressions pass

## Design concerns

D1. **Partly fixed** — objective normalization/epsilon/masking/token weighting/clipping/KL/reference/update schedule recorded and CPU contract/autograd fixtures added; these specify intended behavior, not verification of the unavailable upstream optimizer/image
D2. **Partly fixed** — native tokenizer template, pinned revision requirement, template/data hashes/example identities, no silent missing label/truncation, actual callback correct/incorrect/malformed/truncated/multibox fixtures; real tokenizer revision and distributed decoding contract cannot be validated without the image/model snapshot
D3. **Partly fixed** — ZVF and iid sanity-check interpretation corrected, no herding/CFR/convergence inference or “gradient death” label; raw per-group rewards/task-gradient/KL worker telemetry remains unavailable and is not fabricated
D4. **Fixed for the supplied producers** — paired 90% t-equivalence interval with an explicitly candidate margin, exact small sign enumeration and inconclusive degeneracy; other thresholds relabeled descriptive/exploratory with no multiplicity-adjusted or confirmatory claim; unjustified P6 proxy signs removed
D5. **Partly fixed** — attempt identity/effective seed/command/config, available repo/package/GPU/platform metadata, image digest when supplied and input/template/parser hashes; unavailable worker seeds/backend commit/image/actual work counters stay unknown; resume is explicitly unsupported rather than pretending state restoration
D6. **Partly fixed** — topology/memory/batch validation, timeout, configurable checkpoints, fresh-attempt-only policy, no model-reload upload, no import-time CodeCarbon/parent-worker VRAM confusion; bounded resumable-by-success pilot plans; exact worker instrumentation/full resume/cost screening need future backend and campaign work
D7. **Fixed** — synthetic heuristic is honestly named and seeded, equal-budget comparison, descriptive score dispersion, nonoverlapping NIAH target, P6 promises match outputs, versioned evidence/method/input-hash manifests cover every generated artifact; no generated claims enter registry automatically

## Follow-up independent audit

The follow-up commit also includes the root agent's verified fixes; none were reverted.

- **Fixed, sweep identity/isolation (16/D5/D6):** generated config/seed pairs have distinct checkpoint roots and W&B names. The runner checks the exact argv, actual YAML SHA-256, actual seed, trainer and runner source SHA-256, interpreter hash/version and relevant installed package versions before checking a prior success. Changed or legacy identities require a new manifest. Retry configs/checkpoints/W&B IDs are unique, and a source/input change during execution cannot produce a reusable success. Offline mocks cover every path; no trainer was launched. This is process deduplication, not complete checkpoint resume or empirical validation; package-version metadata is not a container lock.
- **Fixed, enclosing provenance (1/8/D7):** StateFlow, NIAH and paired-statistics loaders check both the enclosing JSON object and each run, including explicit synthetic `evidence_type`. Root-only synthetic, failed and dry-run tags cannot be lost by extracting `runs`; rejection tests preserve input bytes and create no output.
- **Fixed, input coverage (D7):** paired-statistics summary and every artifact manifest hash all used `samestack_ppo_grpo.json`, `variance_mitigation.tsv` and `group_size_advantage_variance.tsv` inputs. Missing optional and unrelated files are excluded.
- **Fixed, completion propagation (12/D7):** the steps-0..4 counterexample retains per-run `unverified_final_budget` and produces H2/H4 `INCOMPLETE`, with a reason in JSON/TSV. Prefix statistics remain available as descriptive observations. A caller-supplied planned final step is required for a verified terminal-window threshold; that scalar is recorded and is not inferred from the observed endpoint.
- **Fixed, root audit patches (1/3/6/13):** saved framework aggregates recompute summaries from the raw reward trace; configured/inherited W&B entity is used consistently; AlphaStar saturation requires the stated high-budget group and exact token, and duplicate/contradictory budgets raise.

## Verification and limitations

- Verification result: **241 tests passed** across the new and relevant existing suites (125 review-01 regression cases, including 41 follow-up cases; two expected SciPy warnings on pre-existing degenerate-data tests). All touched Python files pass Ruff; `git diff --check` passes.
- New offline tests: `tests/test_pro_review_01_training.py` and `tests/test_pro_review_01_analysis.py`; all reviewer-reproduced counterexamples have synthetic regressions
- Relevant existing suites: pydantic configuration support, launcher forwarding, shared-analysis boundaries, analysis validation and stats
- `ruff check` on all touched Python files and `git diff --check`; compileall syntax check on changed producer/driver modules
- No installed OpenRLHF backend, CUDA image, actual model/tokenizer snapshot, complete worker-local metrics protocol or authenticated remote run was available/tested. CPU objective fixtures are not upstream parity evidence
- Legacy results/registry entries have not been rewritten or reclassified in place. The new versioned producers deliberately cannot certify historical evidence provenance
- The local environment initially lacked pydantic and pandas/plotting dependencies; they were installed only into the temporary test environment, not repository dependencies or a training image

Reproduction command (CPU/offline fixtures):

```sh
/tmp/tinker-review-venv/bin/python -m pytest -q \
  tests/test_pro_review_01_training.py tests/test_pro_review_01_analysis.py \
  tests/test_pydantic_support.py tests/test_launcher_forwarding.py \
  tests/test_shared_analysis_boundaries.py tests/test_analysis_validation.py tests/test_stats.py
```
