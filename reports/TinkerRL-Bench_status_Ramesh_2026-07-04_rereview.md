# Re-Review — TinkerRL-Bench Status Slide Deck & Transcript

**Files reviewed:**
- `reports/TinkerRL-Bench_status_Ramesh_2026-07-04.pptx`
- `reports/TinkerRL-Bench_status_Ramesh_2026-07-04_transcript.md`

**Evidence checked:**
- `experiments/launch_log.md`
- `experiments/results/{N12,A2,A4}_20260704/` (TSVs, JSONs)
- `experiments/results/n2_reward_tensor_resume/`, `n10_seed_expansion/`, `mega_20260704/`
- Live process status for PIDs 1063289 (N2), 1063290 (N10), 1063291 (mega), 1002298 (Berkeley)
- `minimax_autoresearch/state/progress.json`, `state_berkeley/progress.json`, iteration/heartbeat logs
- Worktree LaTeX build logs: `/home/claude/tinker-rl-lab-minimax/paper/build/paper_P{1,2,3,4}_*.log`
- `paper/references.bib`, `paper/main.tex`, `paper/main_anon.tex`, `paper/sections/related_work_v2.tex`
- `blind_review/paper_changes.log`

---

## Summary

The update is directionally accurate on the big items: the main MiniMax block finished at 137 iterations / 13 findings / ~$415, the three zero-cost re-analyses (N12, A2, A4) match their reported numbers, the Tinker billing block was cleared and N2/N10/mega were relaunched, and the Berkeley run is live.

Several of the *previous* review issues were addressed:
- Berkeley counters were refreshed (iter 19 / 27 findings / ~$61 in the deck; the repo is now at iter 20).
- The worktree commit count was corrected to 29 commits ahead of `main`.
- The N10 slide was reworded to "seed 42 complete" rather than a stale step count.
- The citation audit claim was softened from "removed" to "identified two problematic entries."
- The billing incident is no longer mentioned on the slides.

However, the two most severe integrity claims from the first review are **still not fixed**, and the active-run numbers have drifted stale again because the deck/transcript were generated from a snapshot rather than live logs.

---

## Top remaining issues

### 1. LaTeX build discipline claim is still false
**Location:** Slide 2, Slide 6, transcript Slide 2 / Slide 6.
**Claim:** "All 4 pillar papers build at 0 errors / 0 undefined citations in the latest paper/build/ logs."
**Evidence:** Worktree build logs show:
- `paper_P1_scaling.log`: undefined citation `arXiv2507.18014` and undefined references to `sec:zvf` / `sec:zvf-cross-experiment`.
- `paper_P2_zvf.log`: undefined references to `sec:variance-honesty`, `sec:group-size-iter31`, `sec:zvf-gradient`, `sec:extended-related-work`.
- `paper_P4_length_bias.log`: undefined references to `sec:group-size`, `sec:lb-iter28`, `sec:lb-iter36`, `sec:lb-iter40`.
- Only `paper_P3_group_size.log` is clean among the four.

The deck and transcript both repeat this as a headline integrity claim, but the logs contradict it.

### 2. Citation audit claim remains unsupported
**Location:** Slide 6, Slide 7, transcript Slide 2 / Slide 6.
**Claim:** "A citation audit identified two problematic entries, and cleanup is tracked in `paper_changes.log`."
**Evidence:**
- The two entries named in the task spec — `liu2026gdpo` and `nimmaturi2025scalinglaws` — are still present in `paper/references.bib`, `paper/main.tex`, `paper/main_anon.tex`, and `paper/sections/related_work_v2.tex` (and in the worktree build artifacts).
- `blind_review/paper_changes.log` is the **anonymization** change log; it contains no citation-cleanup entries at all. There is no repo file that tracks cleanup of these two references.
So the softer "identified" wording is accurate, but the "tracked in `paper_changes.log`" clause is not.

### 3. Active-run status numbers are stale again
**N2:** The resume log shows AERO at step ~30/40 (and the process is still running) with 69 rows in `n2_reward_tensor_resume/n2_metrics.tsv`. The slide says "AERO step 9/40 · 50 rows logged."

**Mega campaign:** `experiments/results/mega_20260704/cells_done.jsonl` contains 5 completed cells (3 since relaunch). The slide says "6 cells done," and the transcript says "five cells completed since relaunch" — both overstate the actual post-relaunch count.

**Berkeley:** `minimax_autoresearch/state_berkeley/progress.json` now shows iteration 20; the deck and transcript still report iteration 19.

### 4. N10 manifest contradicts the live run
The running process has completed `n10_grpo_s42` 15/15 and is now running `n10_grpo_s179` (step 6/15 at last log tail). However, `experiments/results/n10_seed_expansion/n10_manifest_20260704.json` still records all runs as `failed` with a missing `TINKER_API_KEY` from an earlier launch attempt. The slide/transcript narrative is consistent with the live log, but the stale manifest undermines it.

### 5. Mega campaign artifacts are internally inconsistent
`campaign_summary.json` reports `"completed_this_process": 2`, while `cells_done.jsonl` has 5 entries and the process log shows 3 `[done]` lines. The repo needs a single, reconciled progress source.

---

## Verdict

**Needs more work** before being sent to Ramesh.

The headline story is fine, but the integrity narrative is still damaged by the false clean-build claim and the unresolved problematic citations. On top of that, the active-run numbers have aged out again. I recommend:
1. Regenerating the deck and transcript from live logs/JSONs rather than hardcoded snapshots.
2. Correcting the build claim to reflect the actual warnings in `paper/build/paper_P{1,2,4}_*.log` (or fixing the undefined refs/citations first).
3. Either removing the two problematic references or rewording the citation-audit claim to say they were identified but cleanup is still in progress — and pointing to a real tracker, not `paper_changes.log`.
4. Regenerating or deleting the stale `n10_manifest_20260704.json` and `mega_20260704/campaign_summary.json` so the repo does not contradict the live processes.
