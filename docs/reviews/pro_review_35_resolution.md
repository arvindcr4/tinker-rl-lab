# Review 35 resolution (partial source)

The supplied text ends at item 16's filename: missing continuation explicitly skipped. No real benchmark execution, training, paid jobs or stored-result edits.

## Implemented / verified
1. BFCL split v2 requires separately supplied trusted-source input and recomputes the complete READY record. Forged status, digest, revision, launch flags and changed partitions cannot pass. Structural consistency is explicitly not an authenticity claim.
2/15. Ledger v2 has a suite validator registry and mandatory finite-score/completion/final-training-run/sampler/checkpoint revision+commit/model identity checks. No native result validator was established by this slice; the registry is intentionally empty. Legacy E11 flags cannot become scores. This is conservative blocking, not a claim that all 14 native integrations now exist.
3. BFCL receipt v2 rejects non-finite money; API-Bank rejects non-finite budgets/reserves and AgentHarm rejects non-finite/Boolean budget caps.
4. BankerToolBench v2 readiness fields share the complete error decision, including missing receipts.
6. BinaryAudit already emits family, language, classification rule, marker list and content hash; its unknown-family rejection is regression-tested. Lane-constructed classification is not official benchmark certification.
8. API-Bank v2 separates campaign budget_authorized from adapter_may_launch. Legacy input authorization is normalized to paid_jobs_may_launch=false and adapter_may_launch=false; normalization is idempotent.
9. AgentHarm direct scoring prohibition is preserved; signed collection remains the only supported route.
10. AppBench explicitly distinguishes trusted provider-signed validity from independent benchmark verification.
12. Harbor rerun remains permanently BLOCKED preflight; credentials do not promote it to execution evidence.
14. Missing ledger scores are omitted from W&B numeric series; a true zero remains numeric. Categorical status and validity remain visible.

## Remaining design/evidence work
Items 5, 7, 11 and 13 need authenticated external execution receipts, exact official evaluator fixtures and independent authority. This patch does not manufacture those artifacts, invent an official metric, or assert native scoring equivalence. Common cross-adapter heldout/contamination evidence bindings, official BFCL per-category differential scoring/verdict provenance, and FrontierSWE differential verification remain open. Item 12's separate post-execution schema is deferred until its real execution contract exists; preflight is still explicitly unusable as a score. Item 16 is truncated and skipped.

## Verification
Focused existing BFCL split/receipt, BankerToolBench and API-Bank tests plus synthetic review regressions: 67 passed. Further signed-provider/AgentHarm tests are run separately; no credentialed execution is implied. Test-only injected ledger validators establish registry and checkpoint behavior, not native validity.

Final combined offline verification across review 33/35 fixtures and affected adapters/runners: 281 passed, 4 toolchain-dependent skipped, 5 absent-archive E9 tests deselected. `git diff --check` passed. This is not a full repository or native-benchmark pass.
