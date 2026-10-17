# Prospective evidence boundary v2

This is a prospective protocol amendment dated 2026-10-17. It changes no stored result, historical experiment, provider authorization, or frozen preregistration. The generated phase summary is sourced from `campaign_plan_v2.PLAN`; both manifest builders consume the same plan and hash.

## Stages and trusted inputs

- INPUTS_VALID: schema and content integrity only; provider-package validation retains `VALIDATED_INPUTS_ONLY_NOT_A_SCORE_OR_LAUNCH_AUTHORIZATION`
- LAUNCH_READY: affirmative signed provider decisions, independently provisioned issuer roots, exact campaign/phase/suite/run/checkpoint/plan binding, rendered token ceilings and a shared durable reservation are all required
- EXECUTION_COMPLETE: exact task coverage, paired identities, actual per-batch composition, disjoint selection/final IDs, complete due exports and no unresolved infrastructure errors
- SCIENTIFIC_CLAIM_PASS: unavailable until approved cluster analysis, metric normalization, suite/domain/safety guards and production reward-path calibration are implemented and validated. A source status, provenance hash or finite score cannot promote this stage

The v2 receipt predicates accept separately supplied trusted issuer keys and exact expected context. The legacy manifest builder interfaces have no trusted issuer provisioning and therefore intentionally remain BLOCKED. Do not copy issuer keys from the receipt itself. Production provisioning is a deployment prerequisite, not a test fixture to be committed. Licensed text or an arbitrary receipt identifier is insufficient approval.

## E11

Candidate stdout is diagnostic only, including apparently valid `Mismatches:` lines. Exactly one signed supervisor outcome per frozen task is necessary. The supervisor must execute outside the candidate sandbox, sign exact generated source/log digests and distinguish model compile/test failure from harness failure. This repository has no provisioned supervisor or issuer for that runtime; the native `gmake` path cannot currently emit SCORED under v2. A nonzero make exit is not automatically a model or infrastructure failure.

The signed execution authorization must bind the frozen task/prompt manifest and separate Modal/Tinker permissions. Generation caches bind checkpoint, sampler, tokenizer, dataset, prompt and sampler controls. Raw response integrity is required; extraction is always reapplied and grading is new. Historical records without raw generation evidence are inadmissible, not overwritten. Writes use fsync plus atomic replace.

## Financial state

The authoritative request ledger uses transactional SQLite with integer nanodollars, explicit provisioning of reconciled prior spend, immutable request IDs and no automatic initialization. Missing established state blocks serving. Sampling settlement records an estimate and retains the full reservation until independently verified final provider billing. Tracking exceptions cannot charge twice. The accounting calculator preserves remaining running commitments after partial billing and conservatively uses the maximum applicable explicit/token estimate.

SQLite is suitable only when every paid consumer shares the same durable transactional filesystem. Modal Dict is removed. The repository does not provision such a cross-provider store, durable filesystem mount, or credentialed billing reconciler. Bridge and E11 require an established shared ledger path and trusted execution identity; no fallback resets spending. Do not deploy multiple private copies or copy a reconciled database as if it were shared. Integration across other paid entrypoints and provider costs remains a launch blocker. No remote calls were made to validate this change.

## Lifecycle and estimand

Launch requires the initial artifact and publication capability, not future final artifacts. Running requires every checkpoint due through the current logical step; completion requires the exact scheduled set and final step, all bound to the same run. Historical final exports cannot satisfy a new run.

The v2 estimand is fixed-adapter paired task-cluster performance conditional on the recorded training seed. This is not a training-method/retraining-variability estimate. Independent cluster counts, cluster membership, paired results, shared suite resamples across overlapping domains, metric range/direction/transformation/reference and missing/error policy must be frozen before final outcomes are inspected.

## Safety feasibility and guards

The prospective safety planning check uses a conservative one-sided exact binomial bound on newly harmful independent pairs. At zero newly harmful events, the upper bound is `1 - 0.05**(1/n)`: 100 pairs gives approximately 2.95%; 149 pairs are required for <=2%. This bound conservatively upper-bounds the net increase without treating repairs as evidence cancelling new harms. With clustered observations, individual episodes cannot be counted as independent; freeze a valid cluster design and project its cost before launch. An all-zero bootstrap is not an acceptable safety interval.

A suite regression must be displayed and must satisfy its own predeclared guard; a positive domain average cannot erase it. Suite margins and the cluster analysis still require protocol approval, so scientific eligibility remains false. Conjunctive no-regression decisions are intersection–union tests; this amendment does not silently add Bonferroni or promise simultaneous individual-interval coverage.

Scalar reward and binary task_success are distinct. Wilson history is a predeclared bounded history window of task_success observations, not arbitrary fractional rewards. The current reference uses endpoint success only; a full-triage interior graded plateau is recheck, not informative. Degeneracy everywhere uses active-row sample standard deviation, correction one, ATOL.

## Theory/calibration boundary

The proposition is conditional action ambiguity after the particular all-zero observation. It is not equality of future-outcome distributions. The marker control detects the demonstrated marker failure in a local analogue only. A source hash does not prove semantic equivalence. Production calibration must execute versioned controls through the actual reward path and report injected-failure coverage, false-positive/false-negative rates and charged cost; those empirical claims remain unvalidated.
