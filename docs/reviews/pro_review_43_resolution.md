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
23. Awaited saves and 120-second sampling-future timeout retained. Archive protocol `owned_archive_v3_aggregate_budget` closes unsupported remote transfers fail-closed: W&B metadata/history/files, HF remote metadata and Tinker listing/downloads return explicit `unsupported_bounded_remote_transfer` without SDK/CLI calls. Owned local W&B/HF copies share an actual cumulative-write quota with redaction, README and a reserved/countable terminal receipt. Deadline checks precede/follow local chunks; stalled filesystem syscalls are not hard-interruptible. No claim that archiving or rental execution is safe unattended
24. Clearly separated comparison templates from effective runtime receipts; actual trainer is custom Tinker, not native SkyRL. Pinned Axolotl batching and v1/v2 external routing still need runtime validation; unknown revisions remain unknown

## Offline validation

37 focused review-43 tests pass, including real mocked train and evaluation driver
wiring, wrong rewards, corruption/conflicts, shard/resume identity, rental
ownership/setup cleanup, unsupported controls, and model-card publication gates.
Changed Python modules compile; changed shell scripts pass bash -n; git diff
--check passes. No provider, GPU, browser benchmark, deck-render, full integration
suite or historical reanalysis validation is implied.

## R43.23 aggregate archive follow-up

- Fresh output roots only; source/destination overlap and shared-inode protections
  retained. Copies never fall back to unbounded `shutil.copy2`; each actual chunk
  is admitted before writing. Metadata sizes are preflight hints, not evidence
  of transferred bytes. Symlinks are skipped instead of exporting external data.
- `--archive-budget-gb` and `--archive-timeout-seconds` govern the shared local
  operation. Invalid/nonfinite values fail before output creation. Existing
  provider-specific CLI options are retained for compatibility but cannot enable
  disabled remote paths. No override for unbounded remote transfers is supplied.
- Accounting is conservative cumulative writes, including temporary files,
  redaction replacements and failed copies; removal never refunds the quota.
  Failed trees and temporary files are removed without touching evidence. A
  top-level failure removes copied payload, including any unredacted content.
- The terminal JSON receipt uses a pre-reserved 64 KiB (included in the cap),
  padded with JSON-compatible whitespace, and records itself in charged bytes.
  Only this bounded status write/cleanup is allowed after deadline exhaustion.
  This is a logical-write ceiling, not a filesystem-block/RSS quota or a currency
  budget; local blocking system calls cannot be interrupted by cooperative checks.
- 21 focused archive regressions pass: original inode/redaction preservation,
  aggregate W&B/HF multi-file exhaustion, stale size hints, preflight oversize,
  before/during-write deadlines, failure cleanup, redaction budget, exact terminal
  receipt accounting, existing-root rejection, and mocked provider no-call gates.
  Complete review-43 regression selection: 107 passed (one existing datetime
  deprecation warning); compile, Ruff F checks and diff whitespace checks pass.
  No real credentials, live provider calls, paid jobs or stored results were accessed.
