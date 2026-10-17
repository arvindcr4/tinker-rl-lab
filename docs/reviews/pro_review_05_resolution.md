# Review 05 resolution: ZVF side programs

Reviewed baseline: `baaeae72` (origin/main). Scope: `review_05_zvf_side_programs.md`,
including all 20 findings, D1–D6, consolidation targets, and maintenance notes.
This change does not alter retained results, preregistrations, or amendment files.
All verification below is local CPU/synthetic or mocked. No model downloads,
training, paid runs, cloud jobs, deployments, or external-agent calls were made.

## Admission and versioning

- Audit admission now shares `audit_evidence_contract.record_errors`; common
  scientific source enters the `trl-grpo-transformers-generation-lora-v2` hash.
- Preflight requests retain recomputable `preflight-evidence-v2` manifests.
  Telemetry is `generated-and-optimizer-observed-v2`; GSM grading is
  `complete-marked-integer-v2`. Historical A004 eligibility counters are explicitly
  classified as infrastructure-only and never prove an applied update.
- Sweeps use `sweep-effective-v2`, specification-hashed IDs, mandatory measured
  resource fields and durable attempt reconciliation. Old rows are not relabeled.
- Triage is version 0.2.0. JEV retained/routing receipts use v2 semantics. The
  theory document is a corrected v2 contract with withdrawn claims identified.
- The existing next-submission protocol pins old source hashes and the former
  numerical backend. Its execution verifier intentionally rejects current drift.
  A prospective reviewed amendment is required before any future execution.
  Tests update hashes only in synthetic in-memory fixtures to test the remaining
  gate logic; frozen on-disk hashes are not relaxed or rewritten.

## Finding-by-finding disposition

“Fixed” means the reported code path and local reproduction are covered. It does
not certify remote provider behavior or historical scientific results. Partial
items describe the remaining dependency precisely.

| ID | Status | Change, regression coverage, and remaining limits |
|---|---|---|
| 1 P0 | Fixed | Campaign, aggregation and independent verification share metric/identity/evidence-class/trace checks; reject missing measurements, impossible ranges/counts, GU mismatch, absent exact HF/W&B provenance, and mismatched manifests. Campaign success requires aggregate success. Cache includes preregistration, records and referenced manifest bytes. Tests reproduce missing fields, pilot manifest, failure exit, and same-count changes. Existing consumer-specific checkpoint checks remain additional restrictions. |
| 2 P0 | Fixed locally | Matrix calls the launcher's pure manifest validation plus shared identity/accounting checks: positive generations and optimizer steps, G8/G2-to-G8 physical counts, model/revision, objective, dataset, package pins and GPU, accepted protocol hash and recomputed retained manifests. Synthetic four-cell matrix rejects the zero-count/unrelated-model/missing-revision/forged-hash reproduction. Historical manifests without retained identity cannot pass merely by repeating hashes. Unpinned torch/CUDA environment remains D2 partial. |
| 3 P0 | Partial, runner blocked | Both sweep entrypoints use the same resolved matched spec and command builder; distinct GRESO/CI filters, DAPO clips/dynamic sampling, selected arms, batch size and token cap are forwarded and hashed. Audit cells also bind effective specs. Missing command placeholders fail locally. Returned spec/resource accounting is mandatory. The configured `experiments/tinker-runs/cell_runner.py` is absent; the separate platform runner is not a compatible replacement and was not edited. End-to-end runner enforcement of v2 settings/budgets requires that missing adapter. |
| 4 P0 | Fixed | Removed aggregate-to-artificial-reward reconstruction. Summary-only observations retain .9/.96 exactly and cannot mutate per-prompt histories or apply controls. Raw groups require known cardinality; TRL log events are deduplicated by actual global step. Tests cover the supplied distortion and repeated-observation reproduction. |
| 5 P0 | Fixed locally; runtime validation pending | Rollout telemetry names eligibility separately. Genuine completion stop/length/unknown reasons are counted before EOS placeholders; clipping is cumulative genuine truncated/generated. An injectable optimizer observer records active loss tokens (including environment masks), finite gradient norm, actual parameter change, finite parameters and optimizer boundary. Fake trainers without optimization cannot report an applied update; synthetic tensors test no-change/no-token/nonfinite-gradient/changed cases. No pinned GPU/TRL runtime was executed; changed source stays behind the protocol gate. |
| 6 P1 | Fixed | Batch group IDs never imply stable prompt identity. Explicit aligned prompt IDs enable recommendations; repeated stable IDs across separate batch groups are rejected. Diagnostics retain all observed rewards. Empty pool stops explicitly; cold-start stopping is independent of drop recommendations. Nonfinite/out-of-range data fails before history mutation. Tests cover censoring, batch reuse, cold-stop and invalid values. |
| 7 P1 | Fixed conservatively; provider validation pending | GCP separates ABSENT from LOOKUP_FAILED; only structured exact-resource NOT_FOUND/404 establishes absence. Auth/transport/malformed errors remain unresolved. CLI inspection/deletion have timeouts; cleanup is in finally, including interruption after allocation, except explicit successful no-wait ownership transfer. Tests reproduce dual auth failures and wrong-resource absence. Real gcloud error serialization is not validated without credentials; text-only not-found intentionally remains unresolved. |
| 8 P1 | Fixed | Exclusive fsynced attempt claims precede dispatch. Native workers echo nonce; preexisting results and unresolved attempts require reconciliation. Completed generation reuse validates retained request/model/deployment/source and receipt hash; it does not dispatch twice. Shared attempt store also serves sweeps. Local child-process and mocked network tests cover stale success, exclusive claim and uncertain-call retry. |
| 9 P1 | Fixed locally | Launch, resume and aggregate use the same horizon/spec/metric contract. Resume/aggregate additionally require nonce-bound durable COMPLETED receipts and matching current result hashes; master CSV is never admission evidence. Partial/missing-status/short results fail, stale files cannot be overwritten, required-cell failures yield nonzero. Synthetic runner tests cover acceptance then tamper rejection. Missing external runner limits actual execution (#3). |
| 10 P1 | Fixed locally; remote verification pending | All four provider entrypoints use shared request sealing (decoder/tracking before hashing) and shared completed receipt envelopes. HF Jobs/Kaggle use observed terminal resource state, without claiming deletion; GCP maps actual absence into common cleanup fields. Four provider-shaped synthetic matrices pass the same validator and constructors; source checks assert each entrypoint uses them. No allocation, credentials, upload or provider lifecycle integration was attempted. |
| 11 P1 | Fixed by fail-closed rejection | GCP rejects every framework except the complete TRL adapter before loading helpers or allocating; framework is part of request identity. No unsupported framework is launched under a misleading identity. Mock regression confirms early rejection. Implementing alternate frameworks is explicitly deferred. |
| 12 P1 | Partial | Native-worker controls have a separate parser; evaluator argv follows `--`. Worker receipt success is separate from embedded evaluator return code; top-level success requires EXITED and code 0. Local subprocess test executes an evaluator returning 7 and retains that fact. Required evaluator artifacts cannot be validated without a supplied evaluator/artifact schema; process success is not claimed as scientific grading success. |
| 13 P1 | Partial, fail closed | Evaluation recovery invokes frozen-source validation and exact GPU checks; campaign recovery consumes the same finite attempt budget (100 prior attempts cannot bypass max 3). Historical v1 recovery is disabled because it does not bind an approved evaluator/checkpoint/progress contract. No new approved v2 recovery artifact exists, so the old runnable path is not reopened or presented as complete. A future adapter must bind exact checkpoint revision and evaluator/progress provenance. No recovery ran. |
| 14 P1 | Fixed, grader v2 | Complete final-marker integer grammar rejects decimal/fraction/negation/malformed commas; valid sign/comma forms and multiple-marker handling have adversarial tests. Changed semantics are versioned; historical outputs are untouched and not recomputed. |
| 15 P1 | Fixed | Central-t/noncentral-t quantiles/power use SciPy with finite validation; dependency declared in `audit/requirements-analysis.txt`. Tests compare small/moderate degrees of freedom, df=75 constant expectation=1, and finite n=76 power. Existing frozen design binding rejects numerical implementation drift until amended. |
| 16 P1 | Fixed with exploratory scope | Normalize loss alias. Missing measurements are excluded/countable rather than invented zero. Pairing requires the full model/dataset/split/budget/parser/stack block plus seed; duplicate arm/block records fail. Cross-model pairs are excluded. Three unanimous signs report p=.25. Without a registered primary-comparison/attempt-selection plan, outputs are labeled exploratory. |
| 17 P1 | Fixed locally; historical ledger not rerun | JEV probes execute while temporary inputs exist. Typed routing failures persist fallback outcomes. Exact structured status enums prevent INCOMPLETE/UNSCORED/NOT_TERMINAL/UNGRADED acceptance. Pure ledger checks read nonnegative component counts, reject missing evidence, and produce nonzero on FAIL. Tests use mocks and synthetic retained components. Referenced historical ledger inputs are unavailable; no replacement facts or verification output were invented. |
| 18 P1 | Partial, unsupported hooks rejected | Removed fit fallback; installation requires a real recognized reward/advantage hook. Separate recommended/applied group-size fields; applied remains unknown without acknowledgement. Shaped/unbounded reward scales are diagnostic-only. No supported-version integration or sampler control acknowledgement was demonstrated; verl stop/group changes remain recommendations, not applied actions. Unsupported trainers fail installation. |
| 19 P1 | Fixed locally | Modal serve is a resource-registered ASGI function; no blocking/terminating subprocess lifecycle. HTTP contract validates exact served model and greedy/nonstreaming/text/template options, rejects unsupported requests, and reports real token counts and EOS/cap finish reason. AST registration and FastAPI fake-model tests pass. No deployment, model fetch or GPU serving was attempted. |
| 20 P1 | Fixed document and mathematical regressions | Corrected scalar surrogate/gradient distinction and conditional score-norm bound; withdrew unsupported mutual-information improvement/absorption claims. Corrected grading-map equivalence, entropy-gap sign, discrete binary variance, exact group-size inequality and sufficient two-tail bound. Table thresholds regenerated numerically; endpoint scaling described correctly. Tests reproduce .03125 counterexample, rescaled-logit MI failure, G=2 failure and .840625 threshold. |

## Design concerns and maintenance

| ID | Status | Resolution / boundary |
|---|---|---|
| D1 | Partial, prototype remains non-admission | Monotonic deadlines and identified supervisor boot govern lease expiry. Restart fails closed for reconciliation. Only supervisor-admitted, supervisor-signed accepted renewals affect a lease. Manual booleans generate MANUAL_ATTESTATION_UNVERIFIED and cannot produce a terminal cleanup receipt. Tests reproduce frozen-wall expiry and requester-written future renewal. There is no deployed provider observer with retained exact-resource evidence, so real cleanup admission remains unavailable. |
| D2 | Partial | Audit common source now contributes to scientific hash. Preflight retains separately recomputable scientific and execution manifests; destination/tracking details remain unit identity rather than scientific hash. Package pins and GPU are checked. Historical audit stack strings cannot retrospectively prove retained source equality; old requests lack recomputable manifests. Torch/CUDA versions are observed but lack an approved pin/compatibility policy. No arbitrary environment pins were invented. |
| D3 | Partial, claims limited | Module now calls rollout×cap a planned upper bound, not measured equality. v2 admission requires attempted/accepted groups, generated/training tokens, optimizer steps and GPU time, with accounting and token-cap checks. True runtime enforcement including rejected generation work needs the absent external runner and a prospective comparison resource rule. No post-hoc winner claim is authorized. |
| D4 | Fixed locally | Raw verifier scores precede KL-shaped rewards; expose verifier_zvf and effective_training_reward_zvf independently, along with source/reduction/scale/variance convention/tolerance. GU is reward-contrast fraction, not an observed optimizer gradient. Unbounded/shaped controls disabled; optimizer evidence is separate preflight telemetry. |
| D5 | Partial | Repeated completion hashes are allowed; unique contiguous example indices and outcome hashes are checked. Prospective diagnostic default is 1009, and launchers reject overlap with confirmatory and reserve seeds before setup. Old seed-211 artifacts are unchanged. No locked dataset-row manifest establishes diagnostic/confirmatory example disjointness; positional indices are only the existing trace convention. A prospective amendment must freeze diagnostic example IDs, permitted infrastructure-only decisions, and independent confirmatory evaluation. This cannot be reconstructed from missing manifests. |
| D6 | Partial | Exact state bytes retained even for large input, requested versus resolved model separated (unresolved is explicit, not jev-latest asserted as fact), CLI path/binary digest where available, typed answer, routing policy/version/thresholds/outcome retained. UUID/exclusive atomic publication plus file/directory fsync removes racy overwrite. Dated lane facts are explicitly historical manual fixtures. Upstream CLI may not disclose resolved model identity, and missing source receipts are not dynamically verified. |
| Maintenance: imports | Fixed | Required process-group import failures fail tests instead of being converted to skips. Offline integration error branch is tested explicitly rather than skipped based on optional installed packages. |
| Maintenance: slides | Skipped: missing inputs | No base `.pptx` exists in this worktree; enrichment scripts refer to unavailable author-machine absolute paths. No deck was fabricated or overwritten. A load-once/apply-all/save-once pipeline and rendering regression remain pending receipt of a real base deck and intended transformation order. |

### Consolidation delivered

Shared admission cores now serve audit consumers, preflight launcher/matrix,
provider request/result sealing, and all sweep completion phases. Sweeps share
resolved settings/commands. E1 and sweeps share durable local attempts. GCP and
supervisor evidence keep request/termination/absence/unverified states distinct;
there is no claim that all cloud lifecycle implementations have been unified.
Grader and telemetry definitions are versioned; validated SciPy handles audit
power while pairing/missing-data rules are explicit in head-to-head analysis.

## Verification

Environment: Python 3.12 in `/tmp/tinker-review-venv`, existing installed packages;
CPU only. Test temporary outputs and logs are under `/tmp`/pytest temporary paths.

```sh
PYTHONPATH=zvf-program/audit:zvf-program/next-submission:zvf-program/sweep:zvf-program/e1_wave10:zvf-program/zvf-triage/src \
/tmp/tinker-review-venv/bin/python -m pytest -q \
  zvf-program/audit zvf-program/zvf-triage/tests zvf-program/e1_wave10 \
  zvf-program/e13_balrog/test_hosted_supervisor.py \
  tests/test_next_submission_*.py tests/test_pro_review_05_regressions.py \
  --import-mode=importlib --disable-warnings
```

Final result: **346 passed, 0 skipped, 3 warnings in 23.72 seconds**. The dedicated
review reproduction file contributes 60 tests. Test warnings are FastAPI's
startup-event deprecation; fake HTTP calls do not initialize or download a model.
Additional checks: changed Python files pass Ruff `F,E9`; `git diff --check`
passes. Full-repository tests and live service contracts are not represented as
validated by this scoped test command.

## Integration notes

Edits are scoped to side-program source, its tests, and this resolution. There
are no changes under `zvf-program/flagship` or `platform_hybrid`. Coordination
with flagship and hybrid owners found no intended file overlaps. Protocol,
amendment, results, output CSV/JSON, and slide binary artifacts remain untouched.
