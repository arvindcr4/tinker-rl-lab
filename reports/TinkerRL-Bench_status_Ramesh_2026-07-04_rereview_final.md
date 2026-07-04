# Final Adversarial Review — Ramesh Status Deck & Transcript

**Files reviewed:**
- `reports/TinkerRL-Bench_status_Ramesh_2026-07-04.pptx`
- `reports/TinkerRL-Bench_status_Ramesh_2026-07-04_transcript.md`
- `scripts/generate_ramesh_status_slides_20260704.py`

**Evidence checked:**
- `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` + live log + running PID
- `experiments/results/n10_seed_expansion/` manifest + per-seed JSONs + running PID
- `experiments/results/mega_20260704/cells_done.jsonl`, `cells_failed.jsonl`, `campaign_summary.json`
- `minimax_autoresearch/state/progress.json`, `minimax_autoresearch/state_berkeley/progress.json`
- Worktree LaTeX build logs: `/home/claude/tinker-rl-lab-minimax/paper/build/paper_P{1,2,3,4}_*.log`

---

## Verdict

**NEEDS_MORE_WORK**

The deck and transcript are materially stale on two live-run counts. Everything else checks out, but the operational numbers are the heart of the "what got done this week" narrative and cannot go to Ramesh as-is.

---

## Remaining issues

### 1. N2 GIFT progress is badly under-counted — MUST FIX
**Slide 4 / generator line 126 / transcript Slide 4.**
- Claim: "GRPO 40/40 · AERO 40/40 · GIFT 6/40 in progress"
- Live evidence:
  - `n2_metrics.tsv`: GRPO 40 rows, AERO 40 rows, **GIFT 19 rows (steps 0–18)**
  - Live log tail: GIFT has reached **18/40** and is still running (PID `1063289`)
- Required fix: update to current state, e.g. "GRPO 40/40 · AERO 40/40 · GIFT ~18/40 in progress". Re-running the generator or hand-editing the deck is not enough if the script hard-codes the stale number.

### 2. N10 "2/8 seeds done" claim is unsupported — MUST FIX
**Slide 4 / generator line 127 / transcript Slide 4.**
- Claim: "GRPO 2/8 seeds done · remaining seeds + Dr.GRPO in progress"
- Live evidence:
  - `n10_grpo_s42.json`: 15 steps, heldout_acc 0.375 → **s42 genuinely done** ✅
  - `n10_grpo_s179.json`: only **1 step completed**, heldout_acc 0.25 → **not done** ❌
  - Manifest was manually regenerated and marks s179 "ok" with steps 15 / heldout null, but the per-seed JSON is the ground truth and contradicts it.
  - PID `1063290` is still running the N10 script.
- Required fix: report only s42 as done and s179 as in progress, e.g. "GRPO 1/8 done (seed 42); seed 179 + remaining + Dr.GRPO in progress". Do not rely on the manually overridden manifest until the process finishes and rewrites it.

### 3. Berkeley iteration number has aged
**Slide 4 background box / Slide 6 / transcript Slide 4 & Slide 6.**
- Claim: "Iteration 20, no validated outputs yet"
- Live evidence: `minimax_autoresearch/state_berkeley/progress.json` shows **iteration 21**, status `running`, last_summary `error_max_turns`.
- Fix: update to "Iteration 21, no validated outputs yet." Low severity, but it shows the deck was not regenerated from live files.

---

## What now checks out

- **Mega campaign:** `cells_done.jsonl` = 11 entries, `cells_failed.jsonl` = 8 entries, 3 cell_ids overlap with done → 3 unique failures. Matches Slide 4. `campaign_summary.json` has also been reconciled ("completed_this_process": 11).
- **MiniMax main block:** 137 iterations, 13 findings, $415.23 tracked, 29 commits ahead. Matches Slide 2 / Slide 6.
- **LaTeX builds:** P3 log is clean; P1, P2, P4 logs still show undefined references/citations. Deck/transcript now honestly disclose this instead of claiming clean builds.
- **Citation audit claim:** Slide 7 now says the 2 problematic entries are "identified" and "cleanup is in progress (not yet removed from source)". The citations (`liu2026gdpo`, `nimmaturi2025scalinglaws`) are indeed still present in `paper/references.bib`, `main.tex`, `main_anon.tex`, and `sections/related_work_v2.tex`, so the wording is now accurate.
- **Slide count:** exactly 8 slides.
- **Conference timeline:** realistic; NeurIPS 2026 E&D correctly noted as past.
- **No billing-block content:** neither deck nor transcript mentions the July 4 billing block.

---

## Required fixes before READY

1. Update `scripts/generate_ramesh_status_slides_20260704.py` line 126 and regenerate the deck/transcript with the live N2 GIFT count (~18/40 and running).
2. Update generator line 127 and prose to reflect that only N10 GRPO seed 42 is complete; seed 179 is still running (per-seed JSON has only 1/15 steps).
3. Update Berkeley iteration count from 20 to 21.
4. Re-run the generator and re-export the PPTX, or hand-edit both deck and transcript to keep them identical.

After these changes the package can be sent to Ramesh.
