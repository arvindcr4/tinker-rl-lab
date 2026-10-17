# Review 07 resolution: hybrid runs and frameworks

Review input: `review_07_hybrid_runs_frameworks.md` (source review: https://chatgpt.com/c/6ad36050-403c-83ec-98fe-2d5b9fe7898b)

Implementation base: `16cde307`, isolated branch `fix/pro-review-slice07-work`.

Scope: offline code, tests and protocol documentation only. Historical result/config/checkpoint artifacts have not been edited or regraded. No training, provider requests, paid jobs, uploads, push or merge occurred. Versioned prospective changes are documented in `pro_review_07_protocol_v3.md`.

## Findings

| # | Priority | Resolution | Code, evidence and remaining limits |
|---|---|---|---|
| 1 | P0 | Fixed prospectively | N10 uses explicit response/fixed-token and full rollout-batch denominators; retains zero-advantage samples in the reduction; unused clip option removed and unclipped IS objective versioned. Fixed-length gradient tests plus actual mocked N10 datums verify weights for unequal response lengths. Historical N10 interpretation is unchanged |
| 2 | P0 | Fixed | Cell runner exports implemented surrogate objective IDs and false canonical-equivalence metadata; custom losses use response spans. Modal best-of-four update renamed to success-filtered SFT. Exhaustive binary G=2..4 test proves the advantage-clipped surrogate equals the baseline there. Canonical DAPO/GSPO are not falsely implemented |
| 3 | P0 | Fixed in listed custom-PG paths | Live probe, Campaign, Wave 6, parallel/world-class and named legacy GRPO loops refresh the sampler after every update. Legacy state-save conditions gate persistence only. Mock live run verifies s0/s1/s2/final sampler sequence with state-save cadence 2. N10 also refreshes after updated rollouts, retaining its behavior-logprob IS path |
| 4 | P0 | Fixed | Shared real-SDK datum builder shifts input/targets without a fabricated tail; explicit response masks are consumed by all listed custom losses. Real TensorData test verifies token IDs and zero prompt gradients. Empty sampled responses fail explicitly |
| 5 | P0 | Fixed | Axolotl adapts string or single-message conversational completions and validates batch cardinality. Reproduction and missing-label tests pass |
| 6 | P0 | Fixed in reviewed grader paths | Axolotl, Modal aliases/TRL callback, live/Campaign/Wave6, N10/N8/Mega and primary Tinker suite math scorers use the same v3 numeric scorer. Contradictory/malformed boxes and hidden thinking tests agree across adapters. Live/N10 persist raw completions and scorer identity for prospective regrading. Historical regrading skipped without authoritative raw evidence |
| 7 | P0 | Fixed | Parallel/world-class use group-level ZVF; N10 logs fully starved rollouts and successful update counters, returning mean ZVF 1 in the all-starved reproduction. N2 population variance and Mega scaled plug-in variance use distinct names |
| 8 | P0 | Fixed failure semantics; runtime evidence unavailable | OpenRLHF/verl stop on subprocess failure, never substitute missing reward history with [0], use a fresh exact trainer ID, and remain partial when observed update/checkpoint evidence is unavailable. Parallel errors no longer silently continue; completion receipts and resume checks replace peak-based success. Held-out failures receive bounded retries and mark the whole fixed evaluation incomplete with null accuracy. Mock evaluator failure test preserves the population |
| 9 | P0 | Fixed | Registry preserves model namespaces/base/instruction identity, requires explicit training ownership and a single consistent candidate, and quarantines conflicts without metrics. No invented :train:0 or UUID promotion. HF metadata evidence is file/revision-local; all labels must agree. Wrong-newer-model reproduction is quarantined |
| 10 | P1 | Fixed for explicit settings; full resolved revisions partial | Live validates before completed-return and preserves prior traces on failed resumed initialization; Wave6 deduplicates to 12 jobs and uses original explicit horizon plus immutable treatment validation; N8 binds samples to a manifest and content-derived item IDs; Mega hashes generation/scorer settings and refuses legacy family-only skip evidence. Missing legacy specifications fail closed. Exact external dataset/model revisions remain unavailable and block controlled equivalence |
| 11 | P1 | Fixed | Independent Campaign/Wave6/parallel experiments use process isolation and returned W&B Run objects, modern create_new semantics, per-run RNGs and no global wandb.log patch. Main-path logging/init failures are visible in result receipts. Static regression confirms no global patches in these launchers |
| 12 | P1 | Fixed reporting defects; scientific support remains separate | H2H is side-effect-free at import, requires manifest membership, rejects conflicting duplicates, uses objective IDs and missing-aware metric counts, and has no fixed cross-stack conclusion. Claim-table get_step reports observed progress (3, not configured 200); requested counts remain in receipts. Structural validation is explicitly SCHEMA_PASS_ONLY |
| 13 | P1 | Fixed export contract; live export verification skipped | Shared serial/parallel exporter has collision-resistant source branches, exact checkpoint/config/model-revision prerequisites, receipt/hash idempotency, subprocess success checks, immutable commit verification and success-only index. Both use the same source root. Metadata-only exports use hf_results_repo and honest cards. Offline test rejects wrong-source existing weights and checks immutable commit return. No live HF/Tinker calls or credentials used |
| 14 | P1 | Partly fixed; execution intentionally blocked | HumanEval script assembly now concatenates unindented components and compile-checks a known valid function. A trusted isolated evaluator is absent; runtime evaluation raises IsolatedEvaluatorUnavailable rather than converting harness/infrastructure errors to reward zero. Candidate execution and passing/failing runtime fixtures were skipped for this explicit blocker |
| 15 | P2 | Fixed code-level coverage contracts; external inventory not run | Modal inventory uses structured file types, traverses directories independent of dots/extensions, propagates listing failure and records requested/listed/discovered/fetched/failed/truncated coverage. HF iterates paginated models and exported branch/tag commits without arbitrary file caps and retains file-local coverage errors. Backfill identity hashes source URI and bytes; edited content is no longer skipped by name. Offline structured-name/failure and changed-content tests pass |
| 16 | P0 | Partly fixed; controlled execution skipped | Added framework-neutral strict comparison admission contract covering effective objective/config, adaptation, exposure, work counters and selection; wrappers explicitly identify native recipes and are comparison-ineligible. No controlled training was authorized, so no empirical framework-isolation claim is made |
| 17 | P0/P1 | Partly fixed | TRL dependency pinned to 1.1.0; loss_type=grpo, beta=0 and seed/data_seed explicit. TRL records initialized argument and PEFT config separately from request hash. Other wrappers retain unavailable-resolution markers rather than invented defaults. Complete dependency/image/model/tokenizer/data lock and historical initialized-config receipts remain unavailable. The full checkout does contain nominal dumps, the delegated CLI and SkyRL YAML: source inspection found placeholder hashes/managed nulls, Tinker preset G4/LR3e-5 before overrides, and duplicate SkyRL trainer keys plus a Qwen2.5-1.5B declaration. Wrappers now record Tinker source/preset identity and reject duplicate/mismatched SkyRL recipes before launch; no historical runtime settings are inferred from these sources |
| 18 | P1 | Partly fixed | Primary Tinker runners pass initialization seeds and deterministic per-request sampling seeds; Campaign/Wave6 use private Random instances in separate processes; N8/Mega sample requests use stable IDs. Live/N10 retain prompt-token identities. Legacy seed coverage and managed trainable-scope/determinism cannot be asserted as universally verified; protocol requires explicit resolved scope before comparison |
| 19 | P1 | Partly fixed | TRL development data moved to disjoint train[500:600], checkpoint policy made final-checkpoint and unvalidated early stopping disabled; metric key becomes eval_reward. Held-out tokenizer failure never silently substitutes a mirror. Comparison contract requires disjoint frozen item sets and exact tokenizer/template identities. Historical exposure/rendering and native verl/Tinker/SkyRL equivalence remain unresolved; no locked-test evaluation run was authorized |
| 20 | P1/P2 | Partly fixed | Tinker Modal wrapper no longer allocates/reports local H100 as managed training hardware; orchestrator CPU and unavailable managed hardware are separate. Claim-table schema validity is distinct from provenance/completion/comparison/claim support. Audit aggregation partitions estimands and treatment hashes, excludes incomplete runs, and reports per-metric counts/failure coverage. Provider hardware/resource measurements were not available |

## Validation

All tests are offline; service interactions in behavioral tests are fakes.

Final validation:

```text
/tmp/tinker-review-venv/bin/python -m pytest -q tests/test_modal_common.py tests/reviews/test_hybrid_core_review.py tests/reviews/test_hybrid_public_evidence_paths.py tests/test_pro_review_02_modal_analysis.py tests/reviews/test_hybrid_framework_review.py --disable-warnings --maxfail=3
257 passed, 1 warning

/tmp/tinker-review-venv/bin/python -m pytest -q tests/reviews/test_hybrid_framework_review.py --disable-warnings --maxfail=2
36 passed (including a final targeted rerun after strengthening immutable export-commit verification)

/tmp/tinker-review-venv/bin/python -m compileall -q platform_hybrid/experiments/framework_contracts.py platform_hybrid/experiments/tinker-runs platform_hybrid/experiments/modal platform_hybrid/experiments/axolotl
PASS

/tmp/tinker-review-venv/bin/python -m ruff check --isolated --select F821,F822,F823 $(git diff --name-only -- '*.py') platform_hybrid/experiments/framework_contracts.py tests/reviews/test_hybrid_framework_review.py --output-format concise
All checks passed

git diff --check
PASS
```

An added subprocess-failure regression exposed that this installed W&B version lacks `wandb.util.generate_id`; trainer attempt IDs now use stdlib UUIDs, and the regression passes.

- New regression suite covers shared reward policy, conversational cardinality, N10 loss/gradient normalization and real constructed datums, response masking, binary-surrogate equivalence, group ZVF, starved N10 rollouts, sampler/persistence independence, completed-resume mismatches, failed-resume trace preservation, identity quarantine, Wave6 dedup/horizon, Mega identity/HumanEval assembly, observed steps, export receipts, structured inventory, missing metrics, audit pooling, held-out failures, tokenizer fallback, backfill content identity and controlled comparison admission
- Existing Modal scorer tests were updated to the explicitly changed v3 policy; the former wrong-box “rescue” behavior is now rejected
- Compileall and isolated Ruff undefined-name checks cover all edited Python modules

## Five-first recommendation outcomes

1. Immutable request/receipt/comparison/export contracts implemented; externally resolved scientific settings remain explicitly unavailable
2. Objective normalization, masks, policy freshness and implementation identities repaired before any future training
3. Shared scoring and prompt-group diagnostics wired into the reviewed primary paths
4. Failure, identity, export and report evidence are fail-closed instead of silently promoted
5. Controlled multi-framework training/final evaluation skipped as explicitly outside authorization; it requires the missing resolved environment/provider evidence and an isolated code evaluator before expansion

## Independent runtime-audit follow-up

The initial regression set missed several production edge cases. A separate follow-up commit repairs all confirmed runtime reproductions without altering historical artifacts:

| Finding | Follow-up fix | Offline production-path evidence |
|---|---|---|
| 9 | Conflicting explicit `tinker_run_id` / `training_run_id` aliases are quarantined for both candidate owners. Matching aliases or a sole valid alias can own one run only; malformed aliases fail closed | Actual `correlate` tests assert no shared metric/URL for conflicting or malformed owners, with positive controls |
| 10 | Wave6 initialization failures, late failures and launcher/resumer worker crashes now retain v3 specifications; known progress/traces are retained and crash progress remains unknown | Actual `run_one`, `launch`, `wave6_resume.main` and `_compute_rerun_plan` tests verify retryability, original horizon and observed update count |
| 10 | Live resume without a saved training state refuses before initialization, preserving the original receipt bytes; a restart needs a new attempt tag | Actual `run` test forbids dataset/W&B initialization and checks original bytes unchanged |
| 10 | N8 smoke/full outputs have independent immutable manifest bindings; existing shared legacy manifests are validated read-only, not overwritten | Actual `run_all` alternates smoke/full attempts and verifies both bindings, existing samples and legacy bytes; an orphan manifest also rejects changed treatment |
| 7, 8 | Mega rejects missing prompt groups, empty/under/overfilled completion groups and empty completion tokens before complete artifacts. Campaign/Wave6 enforce the same response cardinality. N8 validates each chunk and stages it before accumulation | Actual `Campaign._run_with_sem`/`run_cell`, Campaign/Wave6 training functions and N8 `process_prompt` tests assert no invalid optimizer update/completed row; bounded recovery admits only complete chunks |
| 14 | HumanEval body-only/fenced-body indentation and nested helper definitions are retained | Assembly compile-only tests for all three forms reach the explicit unavailable-isolation blocker; no generated code executes |

These are validity/evidence repairs within v3, documented in the protocol amendment. The provider/runtime-evidence and trusted-isolation limitations above remain unchanged.

Follow-up validation (all offline; fake service boundaries):

```text
/tmp/tinker-review-venv/bin/python -m pytest -q tests/test_modal_common.py tests/reviews/test_hybrid_core_review.py tests/reviews/test_hybrid_public_evidence_paths.py tests/test_pro_review_02_modal_analysis.py tests/reviews/test_hybrid_framework_review.py --disable-warnings --maxfail=3
292 passed, 1 warning

/tmp/tinker-review-venv/bin/python -m pytest -q tests/reviews/test_hybrid_framework_review.py --disable-warnings --maxfail=3
71 passed (35 additional runtime-audit regressions)

/tmp/tinker-review-venv/bin/python -m compileall -q platform_hybrid/experiments/framework_contracts.py platform_hybrid/experiments/tinker-runs tests/reviews/test_hybrid_framework_review.py
PASS

/tmp/tinker-review-venv/bin/python -m ruff check --isolated --select F821,F822,F823 $(git diff --name-only -- '*.py') --output-format concise
All checks passed

git diff --check
PASS
```
