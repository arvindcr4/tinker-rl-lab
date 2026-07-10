# Adversarial Review — `reports/TinkerRL-Bench_status_Ramesh_2026-07-04.pptx`

**Reviewer:** Kimi Code CLI (subagent)  
**Date:** 2026-07-04  
**Sources checked:**
- Slide text extracted via `python-pptx`
- `experiments/launch_log.md`
- `experiments/results/{N12,A2,A4}_20260704/` (TSVs, JSONs, figures)
- `experiments/results/n2_reward_tensor/`, `n2_reward_tensor_resume/`, `n10_seed_expansion/`, `mega_20260704/`
- Process status for PIDs 1063289, 1063290, 1063291, 1002298
- `minimax_autoresearch/state/progress.json`, `state_berkeley/progress.json`, heartbeat/iteration logs
- Worktree `/home/claude/tinker-rl-lab-minimax/` git status and LaTeX build logs
- `minimax_autoresearch/state/task_spec.md`, worktree `paper/REVIEW_codex.md`

---

## Summary

The deck is directionally accurate about the week's headline events: the Tinker billing block was cleared, N2/N10/mega were relaunched, the three zero-cost re-analyses (N12, A2, A4) completed with the numbers shown, and the main MiniMax run finished at 137 iterations / 13 findings. However, several status figures are hardcoded stale values that no longer match the live repo, and a few claims overstate the evidence-hygiene process. The most serious problems are on **slide 3 (active-experiment progress)**, where N2 and N10 are described with outdated step counts, and **slides 3 & 5 (Berkeley run)**, which under-report iteration, findings, and spend versus `state_berkeley/progress.json`.

---

## Claims Checked

| Slide | Claim | Evidence | Verdict |
|-------|-------|----------|---------|
| 1 | Deck title/date/presenter | Hardcoded in source | OK |
| 2 | Billing block resolved; N2, N10, mega relaunched | `launch_log.md` lines 97–100; PIDs 1063289, 1063290, 1063291 are running | **OK** |
| 2 | N12, A2, A4 produced outputs | Directories exist; launch_log lines 5–95 | **OK** |
| 2 | MiniMax main: 137 iters, 13 findings | `state/progress.json`: iter 137, findings 13 | **OK** |
| 2 | 3 re-analyses, 3 runs relaunched | Matches launch_log | **OK** |
| 2 | Citation audit removed 2 fabricated entries | `task_spec.md` and `REVIEW_codex.md` name `liu2026gdpo` and `nimmaturi2025scalinglaws` | **Plausible / not independently logged** |
| 3 | N2 PID 1063289 running | Confirmed via `ps` | **OK** |
| 3 | N2 status "grpo 26/40, then aero/gift/areal" | Log shows grpo 40/40 done; aero is running (~10/40 at review time) | **STALE / WRONG** |
| 3 | N10 PID 1063290 running | Confirmed via `ps` | **OK** |
| 3 | N10 "grpo seed 42 step 12/15" | Log shows s42 completed 15/15; s179 is in progress | **STALE / WRONG** |
| 3 | Mega PID 1063291, 506 runnable cells, concurrency 3 | `ps` and log line `runnable_cells=506` confirm | **OK** |
| 3 | Berkeley PID 1002298 running | Confirmed via `ps` | **OK** |
| 3 | Berkeley "iter 18 running" | `state_berkeley/progress.json`: iter 19; heartbeat.json: iter 19 | **WRONG** |
| 4 | N12 AUROC 0.929 [0.824, 0.994] | `n12_meta.json` / `n12_axis_aurocs.tsv`: 0.929 [0.8238, 0.9938] | **OK (rounded)** |
| 4 | A2 slope −1.09 ± 1.52, p = 0.74 | `a2_scaling_fit.tsv`: −1.0855 ± 1.5159, p = 0.7406 | **OK (rounded)** |
| 4 | A4 NDE +0.005, NIE +0.009, TE +0.014, GER 0.63 | `mediation_estimates.json`: 0.005, 0.009, 0.014, 0.632 | **OK (rounded)** |
| 5 | Main 35 h run completed | Iteration log spans 34.51 h (2026-07-02 11:47 → 2026-07-03 22:17) | **OK (≈35 h)** |
| 5 | Main run $415 spend | `state/progress.json`: $415.2282 | **OK (rounded)** |
| 5 | Main run "no guardrail violations" | 32 iterations logged `error_max_turns`; no evidence of forbidden ops, but "no violations" is unverified | **OVERSTATED / CAVEAT** |
| 5 | Worktree "27 commits ahead of main" | `git rev-list --count main..HEAD` = 29 | **WRONG** |
| 5 | Berkeley "iter 18, 25 findings, $55 spend" | `state_berkeley/progress.json`: iter 19, 27 findings, $61.0886 | **WRONG** |
| 6 | Adversarial review pass: every prose claim traced | `REVIEW_codex.md` exists and lists fixes; "every" is too strong without a full re-audit | **OVERSTATED** |
| 6 | Citation audit removed 2 fabricated entries | Supported by `REVIEW_codex.md`, but no standalone audit log with timestamps | **PARTIALLY VERIFIED** |
| 6 | "All 4 pillar papers rebuild at 0 errors / 0 undefined citations after every agent iteration" | Latest `paper/build/` logs for P1–P4 are clean; task spec says "rebuild at least one affected paper" per iteration, not all 4 after every iteration | **OVERSTATED** |
| 7 | Billing incident details and N2 resume writes to separate dir | `launch_log.md` and `experiments/results/n2_reward_tensor_resume/` confirm | **OK** |
| 8 | Next steps / questions | Aspirational; no factual check required | OK |

---

## Issues Found

### 1. Slide 3 — N2 progress is stale (severity: **major**)
- **Claim:** "grpo 26/40, then aero/gift/areal"
- **Evidence:** `experiments/tinker-runs/logs/n2_reward_tensor_resume_20260704.out` shows `[n2:grpo] 40/40` completed and `[n2:aero] 10/40` (and climbing) at review time.
- **Impact:** Understates progress and makes ETA "~2 h" look longer than it is.
- **Suggested fix:** Change to `"grpo 40/40 complete; aero 10/40 in progress; gift/areal pending"` and update ETA.

### 2. Slide 3 — N10 progress is stale and contradicted by manifest (severity: **major**)
- **Claim:** "grpo seed 42 step 12/15; 8 seeds total"
- **Evidence:** Log shows `[n10_grpo_s42] 15/15` completed; `experiments/results/n10_seed_expansion/n10_grpo_s179.json` exists and shows step 2/15 in progress.
- **Additional repo issue:** `experiments/results/n10_seed_expansion/n10_manifest_20260704.json` still records all runs as `failed` with a Tinker API key error from an earlier launch attempt. This stale artifact contradicts the running process and successful JSON outputs.
- **Suggested fix:** Change slide text to `"grpo seed 42 15/15 complete; seed 179 in progress (step 2/15); 8 seeds total"`. Also regenerate or delete the stale manifest so evidence is self-consistent.

### 3. Slides 3 & 5 — Berkeley run numbers are wrong and internally inconsistent with repo state (severity: **major**)
- **Slide 3 claim:** "iter 18 running, heartbeat fresh"
- **Slide 5 claim:** "Current: iter 18, 25 findings, $55 spend"
- **Evidence:** `minimax_autoresearch/state_berkeley/progress.json` reads `iteration: 19`, `findings: 27`, `total_cost_usd: 61.0886`; `heartbeat.json` confirms iteration 19.
- **Impact:** Both slides understate progress and spend; a reader checking the state files will not trust the deck.
- **Suggested fix (slide 3):** `"iter 19 running, heartbeat fresh"`.  
  **Suggested fix (slide 5):** `"Current: iter 19, 27 findings, $61.09 spend"`.

### 4. Slide 5 — Worktree commit count is wrong (severity: **minor**)
- **Claim:** "worktree now 27 commits ahead of main"
- **Evidence:** In `/home/claude/tinker-rl-lab-minimax`, `git rev-list --count main..HEAD` = 29.
- **Suggested fix:** Change to `"29 commits ahead of main"`.

### 5. Slide 6 — LaTeX build discipline is overstated (severity: **moderate**)
- **Claim:** "All 4 pillar papers rebuild at 0 errors / 0 undefined citations after every agent iteration"
- **Evidence:** Latest `paper/build/paper_P{1,2,3,4}_*.log` files are clean (0 `!` errors, 0 undefined citations), but the task spec (`minimax_autoresearch/state/task_spec.md`) says *"After each iteration, rebuild at least one affected `paper_P{N}_*.tex`"* — not all four after every iteration. Also, `paper/paper_P1_scaling.log` (outside `build/`) still contains 3 `! Extra alignment tab` errors and undefined citations from an earlier build.
- **Suggested fix:** `"All 4 pillar papers build at 0 LaTeX errors / 0 undefined citations in the latest `paper/build/` logs; the affected paper is rebuilt after each agent iteration."`

### 6. Slide 5 — "No guardrail violations" is unverifiable / potentially misleading (severity: **moderate**)
- **Claim:** "stopped cleanly on budget, no guardrail violations"
- **Evidence:** The main run did stop on budget (`status: stopped`). However, 32 of 137 iterations logged `subtype: error_max_turns`. These are handled by the watchdog, but they are iterations that failed to complete normally.
- **Suggested fix:** Qualify as `"stopped cleanly on budget; watchdog recovered 32 max-turns iterations, no forbidden-operation guardrail hits detected"`.

### 7. Slide 6 — "Every prose claim traced" and citation audit claims are too absolute (severity: **minor**)
- **Claim:** "Every prose claim traced to a TSV/JSON artifact" and "Live arXiv lookups verified every reference."
- **Evidence:** `REVIEW_codex.md` documents a real adversarial review and two fabricated citations, but there is no standalone audit artifact or timestamped log for the 2026-07-04 pass.
- **Suggested fix:** `"Adversarial review pass (2026-07-03, REVIEW_codex.md) traced headline prose claims to TSV/JSON artifacts and corrected discrepancies; citation audit removed 2 fabricated entries named in task_spec.md."`

### 8. Slide 3 / repo — Mega campaign progress is much smaller than implied (severity: **minor / caveat**)
- **Claim:** "506 runnable cells, concurrency 3; ETA: days"
- **Evidence:** The log confirms 506 runnable cells, but only 5 cells are recorded in `cells_done.jsonl` (≈1%). The stale `campaign_summary.json` still says `"completed_this_process": 2` from the pre-billing-block process.
- **Impact:** Not a false claim, but the slide gives no progress indicator; a reviewer might assume the campaign is well underway.
- **Suggested fix:** Add progress: `"506 runnable cells, 5 completed since relaunch, concurrency 3, ETA: days"`.

---

## Corrected Text for Key Slides

### Slide 3 — Active Experiments

```
N2 · Reward-tensor instrumentation
PID 1063289
grpo 40/40 done; aero ~10/40 in progress; gift/areal pending
ETA: ~1 h (grpo+aero done)

N10 · gsm8k_cot seed expansion
PID 1063290
grpo seed 42 15/15 done; seed 179 step 2/15; 8 seeds total
ETA: ~3–6 h

Mega · 500-cell sampling campaign
PID 1063291
506 runnable cells, 5 completed since relaunch, concurrency 3
ETA: days

MiniMax · Berkeley mining
PID 1002298
iter 19 running, heartbeat fresh
ETA: 8 h budget
```

### Slide 5 — Autonomous Research Engine (Berkeley card)

```
Berkeley run — live now
• Mining 3 Berkeley RDI agents courses
• 4 round-robin threads: F24, SP25, F25, synthesis
• Output contract: ranked ledger (proposed → prototyped → validated)
• Current: iter 19, 27 findings, $61.09 spend
```

### Slide 5 — Worktree commits

```
worktree now 29 commits ahead of main
```

### Slide 6 — Evidence Integrity

```
LaTeX build discipline
All 4 pillar papers build at 0 errors / 0 undefined citations in the
latest paper/build/ logs; the affected paper is rebuilt after each agent
iteration.
```

---

## Overall Verdict

**Status:** Needs revision before being sent to Ramesh.

The deck is not unreliable at the headline level, but it contains multiple stale hardcoded numbers that misrepresent active-experiment progress and Berkeley-run spend/iteration counts. Because these are exactly the kinds of operational details a status meeting relies on, they should be corrected. The underlying evidence (launch log, progress JSONs, running PIDs, and build logs) supports a positive story once the stale values are refreshed. I recommend regenerating the deck from the live JSON/TSV/log sources rather than hardcoding status numbers, and cleaning up the stale `n10_manifest_20260704.json` so the repo does not contradict the slide narrative.
