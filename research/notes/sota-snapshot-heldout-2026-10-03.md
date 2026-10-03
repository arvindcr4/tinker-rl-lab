---
title: 'SOTA snapshot for held-out suites 2026-10-03'
id: sota-snapshot-heldout-2026-10-03
tags:
- rl-verification
- sota
created: '2026-10-03T15:20:00Z'
status: review
tier: institutional
type: note
content_type: review
summary: 'SOTA snapshot 2026-10-03: only FrontierMath and APEX-Agents have live public leaderboards; 4 of 6 held-out suites have none. Re-verify at paper time.'
---

# SOTA snapshot — held-out suites (2026-10-03, live sources only)

Context: [[final_report_rl-verification]]. Numbers below were read from live
pages on 2026-10-03 — never from memory. **Re-verify at paper time.**

| Suite | Top model | Score | Date | Source |
|---|---|---|---|---|
| AgentHarm | NO_PUBLIC_LEADERBOARD | — | 2026-10-03 | docs-only page, no ranked board |
| FrontierMath (Tier 4 v2) | GPT-6 Astra (OpenAI) | 97.6% | 2026-10-02 | benchlm.ai Epoch-mirror; vendor confirms ~98% |
| AppBench | NO_PUBLIC_LEADERBOARD | — | 2026-10-03 | repo + static EMNLP'24 paper only |
| banker_toolbench | NO_PUBLIC_LEADERBOARD | — | 2026-10-03 | no such public suite/leaderboard |
| APEX-Agents (v1.1 Pass@1) | Gemini 4 Argon (High) | 82.2% ±4.4% | 2026-10-03 | official Mercor leaderboard |
| OpenReward games | NO_PUBLIC_LEADERBOARD | — | 2026-10-03 | 382 per-env boards, no unified board |

Caveats: Epoch AI's official FrontierMath page is JS-rendered (no
server-side rows); the figure comes via the benchlm.ai mirror citing Epoch
plus OpenAI's launch page. AgentHarm/AppBench have canonical static paper
results but no live ranked leaderboard. 4-of-6 suites lacking boards is
itself a finding: held-out claims on those suites cannot cite a moving
frontier.
