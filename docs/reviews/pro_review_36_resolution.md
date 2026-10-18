# Review 36 resolution

Scope: visible review 36; offline synthetic validation only. No training, provider execution, historical results, or generated analyses changed. SDAB result and LifeSciBench result contracts advance to v2; older envelopes must be regenerated from original evidence, never relabelled.

| Finding | Resolution |
|---|---|
|1|Require SDAB boundary digest before normalization; no authenticated result verifier exists, so held-out promotion fails closed rather than trusting checksum receipts.|
|2|Require nonempty SDAB task grades, complete task coverage, bounded native component scores, first-attempt identity, and derived composite/Pass@1. LifeSciBench requires strict booleans, rewards/action counts, 70% threshold and aggregate equality.|
|3|Recompute both manifest hashes, reject duplicate IDs; retain complete manifests in proof and rebuild during receipt validation.|
|4|Reject task-list/suite role disagreement in registry and collision index; remove singleton fallback.|
|5|Check final-use flag before raw/summary branching.|
|6|Recompute serialized domain gates from domain measurements under frozen thresholds; reject missing measurements and contradictory summaries.|
|7|Retain paired baseline count/mean through normalization and compare against frozen base before eligibility. Full ordered task/checkpoint joining remains an upstream evidence requirement.|
|8|Set audit rejects duplicate run/debit identities and aggregate charges above campaign cap. Reservations/provider-account settlement are not represented by this legacy schema and remain an operational evidence limitation.|
|9|Implement continuity-corrected paired Newcombe method 10; published (20,12,2,16) oracle gives [0.0561564,0.3292070]. Historical intervals untouched.|
|10|Allow truthful unsuccessful attempts including missing test output; derive verifier aggregates from task evidence.|
|11|Hash complete SDAB provider input before sanitization; downstream bundle identity includes it.|
|12|Replace shallow filtering with scalar metadata allowlist, excluding nested arbitrary content.|
|13|Preserve upstream xLAM blockers and non-ready status in both paths.|
|14|All held-out indicators require identical registry/runtime proof; flags strictly boolean.|
|15|Allow empty P2P but retain nonempty F2P.|
|16|Pass verified root registry digest explicitly.|
|17|Require exact immutable HF tree/commit URL, rejecting fragments/query tricks; evaluator dataset must match declared dataset in both validators.|
|18|Independent auditor checks schema/type and duplicate debit identities.|
|19|Highest-step final selected deterministically; conflicting ties rejected.|
|20|Fix tuple alias lookup and contradictory aliases.|
|21|Qualification scoped to claim sentence rather than neighboring lines.|

Design concerns A–D: official source authenticity, exact inventory anchoring, deployed timeout/cancellation, signed completed grading, and published golden fixtures cannot be established by local checks. Existing native signed-result and launch gates remain intact. A structurally valid local receipt is not a provider attestation. SDAB explicitly refuses held-out promotion pending such verification. Instrumented scientific-answer records remain a Pavlov protocol, not a claim that optional official artifacts are mandatory upstream. Source labels alone do not establish official inventory membership. E: absent selection/final inventories now report unknown separation and unknown overlap.

Validation: targeted new adversarial regressions plus existing statistics, selection, registry, resolver, adapter, and receipt tests. Float oracle assertion uses tolerance to avoid a platform-specific last-bit Wilson failure. Tests are synthetic and never call a provider. Remaining limitations above are not represented as completed official benchmark runs.
