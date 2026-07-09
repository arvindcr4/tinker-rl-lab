# Autoresearch Dashboard: Paper Improvement (Discovery Report)

**Segment 8 | Runs:** 34-55 | **Kept:** 22 | **Discarded:** 0 | **Crashed:** 0
**Baseline:** reviewer_issues: 7 (#34)
**Best:** reviewer_issues: 0 (#35+, all audits passing)

## Current Status
- **Paper:** 11 pages (9 content + 2 refs), 7 tables, 2 figures, 17 citations
- **All audits passing:** 30/30 paper checks, 0 claim/sync/abstract issues
- **Held-out evals:** 5 x 100-problem GSM8K evals running via Tinker API
  - s137 GRPO, s256 GRPO, s512 GRPO, 4B GRPO, base Qwen3-8B
  - ETA: ~6 hours from launch
- **Preliminary:** 10-problem pilot shows 90% for both GRPO and base

## Key Runs
| # | commit | description |
|---|--------|-------------|
| 35 | ac62de1 | All 17 issues resolved: related work, datasets, future plan |
| 36 | 3e596f0 | KL/entropy diagnostics table from arithmetic logs |
| 37 | e0f343b | Reward limitations + safety discussion |
| 40 | 7368f31 | 4B model data: capacity threshold narrowed to 3B-4B |
| 41 | 52f409f | 5-seed multi-seed table, abstract caveats synced |
| 42 | bf00a22 | Answered 4 reviewer questions substantively |
| 43 | a145bd0 | Per-run compute budget table |
| 47 | 57f5d32 | 2 figures: capacity threshold + diagnostics |
| 48 | 658b93a | Fixed eval script for Qwen3 CoT; 10-problem pilot: 90% |
| 54 | 133fd5e | Paper compiles cleanly to 11 pages |

## Discovery Report Progress: 26/37 done, 1 running, 10 need compute
