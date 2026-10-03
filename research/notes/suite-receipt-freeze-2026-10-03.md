---
title: 'Suite receipt freeze 2026-10-03'
id: suite-receipt-freeze-2026-10-03
tags:
- rl-verification
created: '2026-10-03T16:00:00Z'
status: review
tier: institutional
type: note
content_type: review
summary: 'Frozen 4 of 8 pending suite receipts from live canonical sources (MLE-bench, SWE-bench-Pro, VerilogEval, WebBench); 4 stay pending with documented blockers.'
---

# Suite receipt freeze (2026-10-03)

Context: [[final_report_rl-verification]]. Four research agents read live
canonical sources (repos, HF datasets, papers) for the 8 non-held-out
suites; pins landed in `_PUBLIC_SUITE_RECEIPTS` (`grpo.py`). Frozen 6→10.

Frozen (split + pin + license + runtime verified):
- `mle_bench_eval` — openai/mle-bench HEAD `507f92e`, MIT-code/Kaggle-data.
- `swe_bench_pro_eval` — HF `2d52cb3`, V2 642 tasks, Harbor+Docker runtime.
- `verilog_eval` — tags v2.0.0/v1.0.0, MIT, iverilog harnesses.
- `webbench_eval` — repo + HF pins, MIT, human-graded live sites.

Still pending (fail-closed via `require_frozen_suite_receipt`):
- `binaryaudit_eval` — license unpinned (claimed Apache-2.0, no LICENSE
  file); HEAD-only pin, no tags. Unblocks on upstream LICENSE/tag.
- `frontier_swe_eval` — suite-ID mapping ambiguous (candidate: SWE-bench
  Verified). Unblocks on a mapping decision, then the Verified pin applies.
- `lifescibench_eval` — paper-only benchmark, no public artifact (release
  restricted). Unblocks on a public release.
- `sdab_eval` — provider-run, no public paper/repo/dataset. Unblocks on
  public specs or provider access.

Rule: `None` in a record is a verified absence, never a skipped lookup.
Raw per-field findings with reasons were captured in the research pass;
this note is the durable decision log.
