# Review 40 resolution

The source ends at “The five changes to make first”; the missing list is not reconstructed. No training, provider calls, paid jobs, real benchmark execution or stored-result edits were performed.

## Implemented
1. WebBench v2 uses independent positive finite agent/verifier timeouts, POSIX process-group cleanup (including grandchildren), and persisted scoreless TIMED_OUT/FAILED receipts. CLI timeout parameters are explicit.
2/8. One strict integer-ID validator covers manifests, selection, hashes and verdicts. Duplicate identical/conflicting verdicts, absent/malformed IDs, extras and empty CLI components are rejected.
3. Local evidence v2 requires artifact paths, hashes actual screenshot/DOM/trajectory bytes, rejects traversal/symlink escapes, binds the trajectory to the actual agent output and records immutable environment/verifier revision declarations. A fresh output directory prevents stale artifact reuse. The evidence manifest preserves task/artifact bindings; it does not authenticate an external issuer or establish official observation semantics.
4. Every accepted observation is explicitly smoke/subset scoped, with denominator and subset_score. Generic score and benchmark_score stay null; score_is_claimable and is_full_benchmark_result stay false. Even a caller-provided full-length inventory cannot manufacture an official benchmark score.
5. Review36 already derives SWE verifier aggregate booleans from task results; those tests pass with this change.
6/9. LifeSci result receipt v3 retains review36's complete manifest/aggregation checks but reports local manifest_disjointness_valid separately. Removing a synthetic marker cannot set receipt_proven_heldout; evidence_authenticity remains unverified_local_structure. Cross-suite complete training/source authentication is still not established.
7. Paid execution fails closed pending actual provider-bound reservation/settlement enforcement. Versioned blocked policy gates WebBench launch, SDAB sampler creation, E1 full/GPU dispatch and generation, and E9 launch/replay/regrade paths before provider submission. There is no CLI bypass or caller-asserted “enforced” flag. Pure validators and archival collection remain available. Synthetic tests inject fake gates only alongside fake providers. No atomic ledger was fabricated or claimed.
10. Verilog adapter v3 / receipt v2 require public checkpoints to have safe_public_artifact=true and private checkpoints false; cross-validator synthetic checks cover both.
12. Timeout versus failure remains explicit and scoreless with requested denominator. A universal cross-adapter official-outcome mapping is not invented.

The xLAM component receipt/config schema is bumped to v2 for review34's manifest protocol; exact content overlap with the loaded training slice is rejected when that inventory is available. This is not proof of complete training-corpus noncontamination.

## Remaining evidence/architecture obligations
A trusted external spending reservation/settlement service must be integrated before the blocked paid launch gates can be replaced. No flags or unverified receipt fields should unlock them. Pinned official conformance corpora and independent authenticity verification remain required for benchmark-level claims. Local artifact validation does not independently certify grader implementation or official observation semantics. Generalized inventory completeness, platform-specific non-POSIX cleanup and standardized state mapping for every adapter remain future work. Review34's unknown SDAB score scale remains unresolved; execution is now blocked before paid sampling.

## Validation and dependencies
Review40 was developed on review36 (591b42ca) and review37 (8919015a), preserving their shared helper changes. The parent should apply those commits before this one. Local cherry-picks de09cd2c/82d494e5 are duplicates, not additional review work.

400 passed, 4 optional/archive-dependent skipped in the combined targeted offline regression set, followed by protocol-only adjustments and repeated focused tests. New tests include real local child-process timeout cleanup, fake sampling requests, semantic re-signing and actual synthetic artifact bytes. These are not real benchmark executions. Group75 remains unexecuted because its historical runtime file is absent (review39 report). `git diff --check` passes.
