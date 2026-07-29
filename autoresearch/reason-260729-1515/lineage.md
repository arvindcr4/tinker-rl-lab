# Independent holdout lineage

Task: determine whether the post-training foundations follow-up is ready to converge after the fixed 16-case mutation score reached 16/16.

## Candidate A: accept the current contract as converged

Rationale: the pinned working metric reached 16/16, all focused tests passed, the full suite passed, and the frozen review bundle remained valid.

## Cold-start critiques

Three reviewers inspected the repository without access to the Autoresearch evaluator.

### Contract-robustness reviewer

Verdict: fail. Missing or wrongly typed nested structures still produce raw exceptions; scientific semantics can be replaced by placeholder text; duplicate answer checks satisfy the length-only rule; the reported protocol hash can refer to the file rather than the verified in-memory payload; and the recorded live-source observation is not checked against the computed hash.

### Research-method reviewer

Verdict: fail. The alignment AUC can be undefined in one-class homogeneous strata; the statistical design lacks a primary estimand, cluster unit, confidence procedure, effect threshold, and non-identifiability rule; prose stages have no receipts; the theorem ledger is absent; byte-identical completions conflict with on-policy arms; and the error-budget quantities lack a common estimand.

### Source-and-assumption reviewer

Verdict: fail. The pinned sources are real and directionally support the audits, but source claims have no machine-readable theorem locators or assumption rows. The current verifier is a contract linter, not an evidence promotion gate, and numerical execution thresholds remain intentionally unspecified.

## Candidate B: verifier hardening only

Add strict structure validation, canonical payload hashing, exact distinct evaluation checks, and semantic bindings. This addresses software robustness but leaves the research protocol non-operational.

## Candidate C: synthesized winner

Keep Candidate B's verifier hardening and add the smallest executable scientific packet:

1. rename the current verdict to contract-lint pass and keep promotion false;
2. add a machine-readable source/assumption ledger;
3. add an offline S0-S2 packet with immutable artifact slots, a completion-level independent target, prompt-clustered cross-fitting and bootstrap, a one-class `NOT_IDENTIFIABLE` rule, and a minimum-effect/power field;
4. replace byte-identical realized completions with byte-identical prompt schedules plus arm-specific on-policy completion hashes;
5. leave training thresholds and GPU execution behind a separate amendment rather than inventing values now.

Candidate C wins because it fixes both the software contract and the cheapest scientifically decisive experiment without broadening into unauthorized training.
