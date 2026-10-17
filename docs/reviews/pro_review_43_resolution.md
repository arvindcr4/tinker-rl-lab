# Review 43 — implementation in progress

The prospective bridge protocol is `skyrl-tinker-exact-onpolicy-v2`. No training or historical rescoring was performed.

- 1 fixed: archives own their inodes; legacy linked redaction replaces the destination atomically; overlapping source/output paths are rejected
- 2 partly: exact numeric boxed/signed final answers and whole typed JSON tool arguments replace partial-string/key-only rewards; general symbolic MATH equivalence remains unsupported and receives no credit
- 3 partly: sampler refresh follows every successful optimizer step independently of durable checkpoint cadence; state saves are awaited and objective/reward measurement are versioned; provider/GPU behavior remains unvalidated
- 4 pending: historical reconstructed-seed and step-level inference must be quarantined without rewriting stored tables
- 5–8 pending: provider lifecycle/adapter, episode identity and model/config attribution repairs
- 9 fixed: duplicate YAML keys fail validation; released trainer mappings retain model, optimizer and logging
- 10 fixed: all child exit statuses contribute to campaign failure; no COMPLETE claim after a failed child
- 11–24 pending: remaining launcher, evaluation, notes, presentation and provenance repairs

Validation so far: 24 offline synthetic regressions; no service jobs, real benchmark execution or stored artifact edits.
