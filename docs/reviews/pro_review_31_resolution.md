# Review 31: flagship pilot

Source: review_31_flagship_pilot_1.md, partial. Complete findings 1–9 addressed below; finding 10 and the continuation are missing. No unseen findings are inferred.

## Scientific and authorization boundaries

Frozen preregistration, corpus bindings, existing stored results and generated analyses are unchanged. No provider, training, GPU allocation, or paid job was run. A candidate analysis is not authorization to amend or execute the pilot. Code changes naturally create a new source identity for any future plan; historical acceptance records retain their original source identity.

1. Added `balanced_candidate_report`, schema `flagship-balanced-analysis-candidate-v1` with its own canonical protocol fingerprint. It requires exactly the balanced four-condition/three-seed matrix, matched corpus and compute, and returns DESCRIPTIVE_ONLY_PENDING_PROTOCOL_AUTHORIZATION. It cannot substitute for original GO. Full two-regime screening still rejects balanced-only data. No existing v2 result or protocol was relabeled; corrected-direction analysis has a new v3 output schema.
2. Positive intended-minus-native AUC differences are required. Standardized means retain sign, including negative constant effects. Regression covers negative intermediate AUC with noninferior final accuracy.
3. Replaced confirmatory approval with explicitly exploratory, regime-specific sensitivity reporting: null/half/observed mean, observed/doubled SD, normal-model variance CI, configured Student critical values, finite/positive argument validation, and explicit DEGENERATE_VARIANCE. Three-seed plug-in results never authorize execution or claim adequate confirmatory power. Scientifically meaningful effects, equivalence margins, and final sample sizes remain decisions requiring preregistration.
4. Removed averaging across regimes in power sensitivity. Balanced candidate has only the balanced estimand and never synthesizes filtered observations.
5. Added an unexecuted measured-length preflight candidate: validates fixed generation fingerprint, unique prompt/seed identities, sixteen measured lengths, frozen six-row selection CV, and an explicit safety margin. Constant-cap lengths return STOP_INFEASIBLE_NO_FULL_ALLOCATION. This read-only helper cannot allocate; requiring it in production is part of the future authorized amendment, not a silent change to frozen generation.
6. Corrected screening reports retain paired differences and now report seed SD/SE, absolute checkpoint evaluations and counts; candidate balanced report adds an explicitly exploratory normal seed-mean CI. Small-n screening, practical margins and generalization limits are explicit.
7. Analysis validates 128 examples, finite bounded accuracy and correct-count agreement. Reports retain counts and explain 1/128 resolution, paired-example uncertainty, repeated-evaluation limitations and need for a separate final set. No larger evaluation or sample-size claim is fabricated.
8. Names the angular attribution metric explicitly. Balanced reporting exposes cosine, relative L2, norms and zero-relation counts separately; equal-direction/different-magnitude regression shows angular effect is not vector equivalence. Threshold science remains exploratory, not causal proof.
9. Analysis rejects raw unit records and requires verifier acceptance wrappers; validates full-record and plan hashes, wrapper/unit/plan identities, consistent protocol/source bindings, finite gradient domains/norms, matched replay group identities, finite accuracies/counts, and positive token charges. Receipt hashes establish internal integrity, not cryptographic authenticity: production receipts must still come from independent remote verification. Existing synthetic tests explicitly seal their fixtures; they are not experimental receipts.

## Validation

50 CPU tests pass across pilot analysis/review regressions, replay, evaluation, artifacts, verifier, and review-30 regressions using `/tmp/tinker-review-venv/bin/python`. Regression fixtures are synthetic. No existing result files were modified.

Remaining scientific prerequisites are explicit rather than papered over: authorization for a balanced-only protocol, justified effects/margins and sample size, independently verified real evidence, representative measured length preflight, and any missing continuation of the review.
