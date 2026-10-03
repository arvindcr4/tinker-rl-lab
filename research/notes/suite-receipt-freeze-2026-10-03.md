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
  file); HEAD-only pin, no tags. Re-verified 2026-10-03: HEAD unchanged,
  still no tags, API license still null. Unblocks on upstream LICENSE/tag.
- `frontier_swe_eval` — MAPPING DECIDED 2026-10-03: `Proximal-Labs/
  frontier-swe`, not SWE-bench Verified. The in-repo eval
  (`zvf-program/flagship/frontier_swe_eval.py`) speaks only to that repo
  at `422b9bb` and refuses substitutes; the pinned commit and all 17
  task IDs were verified live. Stays pending on license (verified
  absent: no LICENSE/COPYING file, API null, no README statement).
  Unblocks on an upstream license or explicit maintainer authorization.
- `lifescibench_eval` — paper-only benchmark (bioRxiv DOI verified via
  Crossref 2026-10-03; CC-BY-NC preprint). No public artifact: 0 GitHub
  repos, 0 HF datasets, no data/code links on the paper page.
  Unblocks on a public release.
- `sdab_eval` — provider-run, page re-verified live 2026-10-03 (80
  tasks, Apr 2026, samples upon request). No public paper/repo/dataset/
  license. Unblocks on public specs or provider access.

Rule: `None` in a record is a verified absence, never a skipped lookup.
Raw per-field findings with reasons were captured in the research pass;
this note is the durable decision log.
