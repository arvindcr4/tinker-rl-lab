# Re-Review 2 — TinkerRL-Bench Status Slide Deck & Transcript

**Files reviewed:**
- `reports/TinkerRL-Bench_status_Ramesh_2026-07-04.pptx`
- `reports/TinkerRL-Bench_status_Ramesh_2026-07-04_transcript.md`
- `platform_modal/scripts/generate_ramesh_status_slides_20260704.py`

**Evidence checked:**
- `experiments/launch_log.md`
- `experiments/results/{N12,A2,A4}_20260704/` (TSVs, JSONs)
- `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` and live log
- `experiments/results/n10_seed_expansion/` (manifest + per-seed JSONs + log)
- `experiments/results/mega_20260704/` (`campaign_summary.json`, `cells_done.jsonl`, `cells.tsv`, log)
- `minimax_autoresearch/state/progress.json`, `minimax_autoresearch/state_berkeley/progress.json`
- Worktree LaTeX build logs: `/home/claude/tinker-rl-lab-minimax/paper/build/paper_P{1,2,3,4}_*.log`
- `paper/references.bib`, `paper/main.tex`, `paper/main_anon.tex`, `paper/sections/related_work_v2.tex`
- `blind_review/paper_changes.log`

---

## Summary

The update fixes the most serious integrity problem from the first re-review: the LaTeX/build claim is no longer false. The deck and transcript now correctly disclose that P1, P2, and P4 compile to PDF but still carry undefined references/citations being cleaned. Berkeley is appropriately framed as background/in-progress, the billing incident is gone, and the slide count is exactly 8.

However, several live-run numbers have aged out again, and the citation-audit tracking claim is still unsupported. The deck cannot be sent to Ramesh without correction.

---

## Item-by-item check

### 1. LaTeX build claims — FIXED
**Slide 2, Slide 7, transcript Slide 2 / Slide 7.**
The deck now says: "All compile to PDF; P1/P2/P4 still carry LaTeX warnings/undefined refs that are being cleaned" and "Latest builds compile to PDF for all 4 pillars; P1/P2/P4 still carry undefined references/citations that are being fixed."

This matches the worktree build logs:
- `paper_P1_scaling.log`: undefined citation `arXiv2507.18014`, undefined refs `sec:zvf`, `sec:zvf-cross-experiment`.
- `paper_P2_zvf.log`: undefined refs `sec:variance-honesty`, `sec:group-size-iter31`, `sec:zvf-gradient`, `sec:extended-related-work`.
- `paper_P4_length_bias.log`: undefined refs `sec:group-size`, `sec:lb-iter28`, `sec:lb-iter36`, `sec:lb-iter40`, `sec:zvf-iter126`.
- `paper_P3_group_size.log`: clean.

### 2. Citation cleanup tracking — STILL UNSUPPORTED
**Slide 7, generator script line 210.**
The slide says: "Citation audit identified 2 problematic entries; cleanup is in progress and tracked in the worktree change log."

Problems:
- The only worktree change log found is `blind_review/paper_changes.log`, which is the **anonymization** change log and contains no citation-cleanup entries at all.
- The two problematic entries (`liu2026gdpo` and `nimmaturi2025scalinglaws`) are still present in `paper/references.bib` (lines 298 and 1021), `paper/main.tex` (lines 816, 837, 2076), `paper/main_anon.tex`, and `paper/sections/related_work_v2.tex` (lines 48 and 184). No actual cleanup has occurred.

The transcript avoids naming `paper_changes.log`, but the deck's "worktree change log" wording is effectively the same false claim.

### 3. Live numbers — PARTIALLY STALE

#### N2 reward-tensor instrumentation — WRONG
**Slide 4, generator script line 126, transcript Slide 4.**
Claim: "GRPO 40/40 done · AERO ~33/40 in progress."

Evidence (`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` and live log):
- GRPO: 40/40 done ✅
- AERO: 40/40 done (steps 0–39 in TSV, log shows "[n2:aero] 40/40") ❌
- GIFT: started but paused at step ~22/40 due to Tinker 402 billing-block errors (log ends with billing-block messages) ❌

The correct status should be something like: "GRPO 40/40 done · AERO 40/40 done · GIFT ~22/40 paused on billing block."

#### N10 seed expansion — MOSTLY ACCURATE, MANIFEST STALE
**Slide 4, generator script line 127, transcript Slide 4.**
Claim: "Seed 42 15/15 done · seed 179 in progress."

Evidence:
- `experiments/results/n10_seed_expansion/n10_grpo_s42.json` shows 15 steps completed, heldout_acc 0.375 ✅
- Live log shows `n10_grpo_s179` at step 12/15 before the log ends, so "in progress" is accurate ✅
- However, `experiments/results/n10_seed_expansion/n10_manifest_20260704.json` still records all runs as `failed` with a missing `TINKER_API_KEY` from an earlier launch attempt. This stale manifest contradicts the slide narrative and should be regenerated.

#### Mega 500-cell campaign — UNDERCOUNTED / STALE
**Slide 4, generator script line 128, transcript Slide 4.**
Claim: "5 cells done · 500+ planned · concurrency 3."

Evidence:
- `experiments/results/mega_20260704/cells_done.jsonl` has 8 entries.
- `experiments/results/mega_20260704/cells.tsv` has 8 data rows (9 lines including header).
- The mega log has 6 `[done]` markers.
- `experiments/results/mega_20260704/campaign_summary.json` says `"completed_this_process": 2`, which is inconsistent with the other two sources.

The slide undercounts the completed cells. A reconciled number is at least 6–8, not 5. The repo also needs a single source of truth for mega progress.

#### Autonomous research main block — ACCURATE
**Slide 2, Slide 6, transcript Slide 2 / Slide 6.**
Claim: 137 iterations, 13 verified findings, ~$415 tracked spend, worktree 29 commits ahead.

Evidence:
- `minimax_autoresearch/state/progress.json`: iteration 137, findings 13, total_cost_usd 415.2282 ✅
- `git rev-list --count main..HEAD` in the worktree returns 29 ✅

#### Berkeley curriculum run — ACCURATE AND APPROPRIATELY FRAMED
**Slide 4 background box, Slide 6, transcript Slide 4 / Slide 6.**
Claim: iteration 20, in progress, no validated outputs yet.

Evidence:
- `minimax_autoresearch/state_berkeley/progress.json`: iteration 20, status "running", total_cost_usd 64.2139, findings 27, last_summary is `error_max_turns` ✅
- The deck/transcript consistently treat Berkeley as background and explicitly say "no validated outputs yet." ✅

### 4. Conference deadline claims — REALISTIC
**Slide 8, transcript Slide 8 and appendix lines 29–36.**
The deck and transcript state that NeurIPS 2026 E&D is past and list ICLR 2027 (~Sept/Oct 2026), ICML 2027 D&B (~Jan 2027), and NeurIPS 2027 E&D (~May 2027) as realistic targets. These align with the usual annual deadline cycles. The script's "clear path to a NeurIPS D&B submission" (line 79) is ambiguous but reads as NeurIPS 2027 in context.

### 5. Berkeley de-emphasized — YES
Berkeley is confined to a background card on Slide 4 and an "in progress" card on Slide 6, with explicit caveats that it has no validated outputs. This is appropriate.

### 6. No billing-block content — YES
Neither the deck nor the transcript mentions the July 4 billing block. Note, however, that the N2/GIFT run is currently paused because of a Tinker 402 billing-block error, so the "let N2 ... run to completion" next-step item is contingent on resolving payment.

### 7. Slide count — EXACTLY 8
Verified via `python-pptx`: `len(prs.slides) == 8`.

---

## Verdict

**Needs more work.**

The integrity framing around LaTeX builds is now honest, Berkeley is handled correctly, the billing incident is absent, and the slide count is right. But the deck still goes to Ramesh with stale/misleading operational numbers and an unsupported citation-tracking claim.

---

## Required fixes

1. **Correct N2 status** (`platform_modal/scripts/generate_ramesh_status_slides_20260704.py` line 126; Slide 4; transcript Slide 4). AERO is finished; GIFT is the active arm and is paused on a Tinker billing block. Suggested wording: "GRPO 40/40 · AERO 40/40 · GIFT ~22/40 paused on billing block."

2. **Reconcile and update mega count** (`platform_modal/scripts/generate_ramesh_status_slides_20260704.py` line 128; Slide 4; transcript Slide 4). Use a single reconciled source (`cells_done.jsonl` or `cells.tsv`) and report the actual count (currently 8 completed cells, with `campaign_summary.json` showing 2 and needing regeneration).

3. **Remove or reword the citation-tracking claim** (`platform_modal/scripts/generate_ramesh_status_slides_20260704.py` line 210; Slide 7). Either say "identified but not yet removed" or, if cleanup is genuinely in progress, point to the actual tracker. Do not say it is tracked in `paper_changes.log` or the "worktree change log."

4. **Regenerate `n10_manifest_20260704.json`** so it reflects the successful `n10_grpo_s42` run and in-progress `n10_grpo_s179` run rather than the stale `failed`/`TINKER_API_KEY` state.

5. **Regenerate `mega_20260704/campaign_summary.json`** so it agrees with `cells_done.jsonl` and `cells.tsv`.

After these fixes, the deck and transcript can be re-reviewed for a ready verdict.
