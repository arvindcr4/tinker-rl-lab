# Review 43 — prospective repairs and evidence quarantine

Base repair: `34698759`. No paid jobs, training, benchmark runs, historical
rescoring, or stored result-table rewrites were performed. Protocol v2 outputs
must not be presented as a revalidation of historical v1 runs.

## Resolution by finding

1. Fixed: independent archive inodes, atomic linked-file redaction, overlap rejection
2. Fixed exact numeric / typed-JSON counterexamples; symbolic MATH equivalence remains unsupported (no credit), not a general MATH verifier
3. Fixed: every successful optimizer update refreshes sampling independently of awaited recovery saves; completion-only sequence-sum objective is explicit. Actual `train()` wiring is tested over 11 updates, including callback masking. No final held-out evaluation is claimed
4. Quarantined: original statistical tables are preserved; reconstructed seed clouds and dependent trace-step inference are explicitly unsupported as empirical independent-run uncertainty
5. Fixed Python failure propagation, ownership persistence, setup gating, default cleanup, cleanup result checking and shell consolidation. **Still requires independent provider-side lease/watchdog before unattended use**; host failure cannot be covered by Python finally
6. Fixed: strict parse failure, evaluation/task/seed identity, conflicting duplicate rejection, explicit expected coverage gate. Unmanifested summaries say coverage_unverified
7. Corrected offer search/new_contract/show-instance lifecycle, supported SFTP writes, configured client key, default host-key verification. Mocked, not live-provider validated; pinned client deployment remains an operator prerequisite
8. Quarantined contradictory historical claims; generated config_identity_v2.json captures actual released YAML values and explicitly unknown historical run/revision. Existing names remain historical aliases for compatibility, not model evidence
9–10. Fixed in base repair: duplicate YAML rejection and every-child campaign exit propagation
11. Fixed advertised CLI controls and rejection of unknown options. Fresh v3 run directories eliminate unsafe cache reuse entirely. Validate arm model/seed/completed steps and evaluate API-returned sampler paths with explicit arm mapping. Immutable model revisions remain unknown
12. Fixed action-error validity, explicit errors, max-step truncation and environment-creation failures; no cumulative partial-reward promotion to success. Maximum task reward is a versioned local contract, not an official cross-benchmark scorer claim
13. Fixed stable task-based seeds, effective config hash, strict resume identity and prior-row summary accounting. Explicit retry policy retains recorded error attempts; retry into a new evaluation output. Existing non-resume output is rejected
14. Fixed requested-package import failure, unsupported concurrency rejection, effective receipts and publication checks. Cards require complete error-free manifest coverage, matching model/benchmark receipt, and render actual generation settings. Package revisions remain unknown
15. Corrected prompt-conditional expectation, heterogeneity and finite-sample randomness
16. Corrected documented KL direction, response-token normalization and gradient semantics
17. Added linked erratum keyed to frozen TeX git blob; original frozen TeX unchanged. Four historical contradictions remain unresolved pending exact observations/run records
18. Archival generator requires --archival-build and visibly watermarks every slide; removes equivalence/mechanism/executed-demo claims. --check remains structural only, not semantic validation
19. Illustrative chart positions now derive from labeled numbers and display no-run/no-seed caveat; convergence/perfect-generalization claims removed. It remains illustrative, not a named empirical trace
20. Fixed optional W&B import, no-card upload metadata, unsupported rental controls, SSH key usage, unique nondeleting DPO directories, presentation output CLI, per-matrix LoRA math, P8 inventory and workshop source link
21. Scoped causal/predictive/length/architecture interpretations; require disjoint predictive windows and incremental reward-only comparison. No empirical claim was recomputed
22. Added explicit claim-lineage requirements and unresolved structured records; submission packages require receipt evidence before being called submitted
23. Awaited saves and 120-second sampling-future timeout added. **Aggregate archive byte/wall-clock enforcement across third-party downloads remains unresolved**; existing Tinker selection estimate is not an enforceable total archive cap. No claim that archiving or rental execution is now safe unattended
24. Clearly separated comparison templates from effective runtime receipts; actual trainer is custom Tinker, not native SkyRL. Pinned Axolotl batching and v1/v2 external routing still need runtime validation; unknown revisions remain unknown

## Offline validation

37 focused review-43 tests pass, including real mocked train and evaluation driver
wiring, wrong rewards, corruption/conflicts, shard/resume identity, rental
ownership/setup cleanup, unsupported controls, and model-card publication gates.
Changed Python modules compile; changed shell scripts pass bash -n; git diff
--check passes. No provider, GPU, browser benchmark, deck-render, full integration
suite or historical reanalysis validation is implied.
