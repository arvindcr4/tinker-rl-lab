# Progress reporting audit — 2026-09-12

Implemented and tested the offline 14-lane report generator. Only the two owned
reporting paths were written. Parent live state and all existing lane receipts
were read-only. No cloud/model calls, downloads, benchmark runs, or replay.

The retained snapshot is `snapshot_v3/report.json` and `snapshot_v3/report.md`
under `.codex-run/finish_20260912/progress_reporting`. Earlier local development
snapshots v1/v2 are retained; v3 is the reviewed delivery, not a parent promotion.

## Findings

- Snapshot time: 2026-09-12T15:38:21.232177+00:00; 164 consumed files have SHA-256 provenance.
- Replacement score progress: 7/14 (E1, E5, E8, E10, E11, E13, E14). Partial scores count, including zero; this is not original-suite completion or aggregate accuracy.
- Complete named portfolio scopes: E8, E10, E11. E8 is public LAB-Bench; E10 is benign AgentDojo; E11 is retained Verilog.
- E1: 110/300 terminal attempts versus 35/300 native reports. Native-report subset success is 4/35; terminal-attempt success with errors-as-zero is separately retained. Parent's larger percentage must be labelled terminal-attempt coverage.
- E5 snapshots remain separate: E5-native-20260912-03: 1/5 successes, 5/97 coverage; E5-native-20260912-04: 1/4 successes, 4/97 coverage; E5-native-20260912-05: 1/3 successes, 3/97 coverage; E5-native-20260912-09: 5/27 successes, 27/97 coverage; E5-native-20260912-10: 0/16 successes, 16/97 coverage. No pooling.
- E7 original BinaryAudit denominator is 28 primary-eval tasks; 46 is inventory (28 primary, 10 heldout-labelled, 8 train). The historical errored task's native reward 0.0 is preserved without inventing a clean accuracy. Replacement CyberGym is separately 1,507 tasks.
- E9 original MLE-bench coverage is unique graded competitions in the historical legacy index, not competition accuracy. Fresh merged-arm pilots remain separate. Replacement MLDevBench's 34 released configurations do not establish full private scope.
- E12 six public AppBench artifacts do not establish the original heldout denominator. Replacement VisualAgentBench paper scope is separately 746; no score is inferred from prepared assets.
- E13 13/13 BabyAI success covers 13/255 BALROG episodes; it says nothing about unavailable original OpenReward Games.
- E14 2,271/4,426 is accepted-report accuracy. Coverage is 4,426/4,428 with two parser exclusions; strict full-scope score remains null. No private FrontierMath claim.
- Historical receipt context is labelled as such. Unmeasured selected scopes display unverified, not a made-up zero.

## Validation

25 unittest cases passed. Tests include scope/denominator drift, unknown-vs-zero,
zero-score counting, duplicate SWE/Tau identities, terminal errors, nonterminal
and nonfinite rewards, public/private separation, the original E7 split,
canonical exclusion behavior, source hash tampering, receipt-discovery races,
path escape, refusal to overwrite prior reports, rejection of parent output
paths, and the 6 GiB floor. The final generator ran successfully on local receipts.

This is a bounded receipt-level audit, not a full transitive artifact or live-job
recheck. The generator records its own source hash and all source membership.
Active runs may add new records immediately after publication; rerun to a new
owned snapshot. Missing/changed evidence fails closed instead of promoting a
stale status. No source file was altered to make validation pass.

## Concrete parent next action

Review the retained snapshot and use the generator from README_v1.md for the next
reporting tick, with a new output directory. In the parent's human-facing report,
name E1 terminal-attempt coverage separately from native evaluation coverage,
retain E7's 28-task primary denominator, and always display the suite name beside
its E-label. Parent can promote a reviewed snapshot separately; this task did not
modify `progress_reporting_state.json` or `progress_report_latest.md`.
