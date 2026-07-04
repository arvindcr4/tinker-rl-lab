# Adversarial Review — TinkerRL-Bench Status Transcript

**File reviewed:** `reports/TinkerRL-Bench_status_Ramesh_2026-07-04_transcript.md`  
**Slide deck reviewed:** `reports/TinkerRL-Bench_status_Ramesh_2026-07-04.pptx`  
**Evidence reviewed:** `experiments/launch_log.md`, `experiments/results/{N12,A2,A4}_20260704/`, `minimax_autoresearch/state{,_berkeley}/progress.json`, live process status, run logs, worktree build logs, submission manifest.  
**Date of review:** 4 July 2026

---

## Summary

The transcript is well-organized, appropriately cautious in places, and generally aligned with the slide deck. However, several factual claims are stale or unsupported by the evidence currently in the repo. The most severe issue is an overstatement of evidence hygiene: the four pillar papers do **not** currently compile with zero undefined citations/references. There are also significant stale status claims about the live N2 and N10 runs, and the Berkeley autoresearch counter is understated in the slide deck. The transcript should be refreshed against current logs and the strongest hygiene claims should be softened or corrected before delivery to Ramesh.

---

## Claims Checked

| Claim (source) | Evidence | Verdict |
|---|---|---|
| Tinker 402 billing block cleared; N2, N10, mega relaunched (Slide 2, transcript) | `experiments/launch_log.md` billing section; live PIDs 1063289, 1063290, 1063291 | ✅ Correct |
| N12/A2/A4 zero-cost re-analyses completed (Slide 2, transcript) | Output dirs and `launch_log.md` entries exist and are timestamped 2026-07-04 | ✅ Correct |
| MiniMax M3 main run: 137 iterations, 13 findings, ~$415, stopped cleanly (Slide 2, 5, transcript) | `minimax_autoresearch/state/progress.json`: iter 137, findings 13, cost $415.23, status `stopped` | ✅ Correct |
| N2 is on GRPO step 26/40 (Slide 3, transcript Slide 3) | Resume log shows GRPO 40/40 completed; currently on AERO step 9/40 (PID 1063289) | ❌ False / stale |
| N10 is on seed 42 step 12/15 (Slide 3, transcript Slide 3) | Log shows grpo seed 42 already at 15/15; process is running but manifest is inconsistent | ❌ False / stale |
| Mega campaign: 506 runnable cells, concurrency 3 (Slide 3, transcript) | Script and log confirm 506 cells, concurrency 3 | ✅ Correct (but only ~5 cells finished) |
| Berkeley run: iter 18, 25 findings, $55 spend (Slide 5) | `state_berkeley/progress.json`: iter 19, findings 27, cost $61.09 | ❌ Stale |
| N12 4-channel AUROC 0.929, unchanged (Slide 4, transcript) | `n12_meta.json`: 0.929 [0.824, 0.994] for `zvf_risk_max_4ch` | ✅ Correct |
| A2 slope −1.09 ± 1.52, p = 0.74 (Slide 4, transcript) | `a2_scaling_fit.tsv` / `a2_meta.json`: −1.085 ± 1.516, p = 0.741 | ✅ Correct |
| A4 NDE +0.005, NIE +0.009, TE +0.014, GER 0.63, TE not significant (Slide 4, transcript) | `mediation_estimates.json`: matches; GER CI −5.1 to +6.6 | ✅ Correct |
| Main run appended 13 verified findings to `AUTORESEARCH_FINDINGS.jsonl`; worktree 27 commits ahead (Slide 5) | Worktree file has 27 lines total (not all from main run); `git rev-list --count main..HEAD` = 29 commits ahead | ⚠️ Partially misleading / wrong commit count |
| All four pillar papers compile at 0 errors / 0 undefined citations (Slide 2, 6, transcript) | `paper/build/paper_P{1,2,3,4}*.log`: P1 has undefined citations; P1, P2, P4 have undefined references | ❌ False |
| Citation audit removed two fabricated references (Slide 2, 6, transcript) | `liu2026gdpo` and `nimmaturi2025scalinglaws` still present in `/home/claude/tinker-rl-lab/paper/references.bib`, `main.tex`, `main_anon.tex`, and worktree `main_anon.tex`, `p5_related.tex` | ❌ False / incomplete |
| NeurIPS 2026 D&B bundle assembled (Slide 6, transcript) | `submission/contents/MANIFEST.md` and `checksums.sha256` exist, but the actual files they reference (`paper.pdf`, `paper_anon.pdf`, `code.tar.gz`, etc.) are not in the repo | ⚠️ Misleading |
| Limitations disclosed (Slide 7, transcript) | Listed limitations match task spec and known issues | ✅ Correct |

---

## Issues Found

### 1. **SEVERE — False claim that all four pillar papers compile cleanly**
- **Location:** Slide 2 ("Evidence hygiene locked"), Slide 6 ("LaTeX build discipline"), transcript Slide 2 and Slide 6.
- **Current wording:** "All four pillar papers compile with zero LaTeX errors and zero undefined citations" / "All 4 pillar papers rebuild at 0 errors / 0 undefined citations after every agent iteration."
- **Evidence:** `paper/build/paper_P1_scaling.log` contains `Package natbib Warning: There were undefined citations.` and `LaTeX Warning: There were undefined references.`; `paper_P2_zvf.log` and `paper_P4_length_bias.log` also report undefined references. Only `paper_P3_group_size.log` is clean among the four.
- **Suggested rephrase:** "The four pillar papers build to PDF, but P1 currently has undefined citations and P1/P2/P4 have undefined references that need to be cleaned up before submission. P3 is clean."
- **Severity:** High — this is an integrity claim made directly to Ramesh and it is contradicted by the build logs.

### 2. **SEVERE — False claim that fabricated citations were removed**
- **Location:** Slide 2 ("Citation audit removed 2 fabricated entries"), Slide 6, transcript Slide 2 and Slide 6.
- **Current wording:** "the citation audit caught and removed two fabricated references" / "2 fabricated entries found and removed."
- **Evidence:** The entries `liu2026gdpo` and `nimmaturi2025scalinglaws` still exist in the main repo's `paper/references.bib`, `paper/main.tex`, and `paper/main_anon.tex`, and in the worktree's `paper/main_anon.tex` and `paper/sections/p5_related.tex`. The build warning about undefined citations in P1 is consistent with these still being present.
- **Suggested rephrase:** "A citation audit identified two fabricated/problematic entries (`liu2026gdpo`, `nimmaturi2025scalinglaws`). They have been removed from some files but still appear in `main.tex`, `main_anon.tex`, and `p5_related.tex`; cleanup is in progress."
- **Severity:** High — repeating "removed" when the entries are still live weakens trust in the hygiene process.

### 3. **MAJOR — N2 status is stale (not on GRPO step 26/40)**
- **Location:** Slide 3, transcript Slide 3.
- **Current wording:** "N2 is the reward-tensor instrumentation run. It is on the GRPO arm, step 26 of 40, and will then move through AERO, GIFT, and AREAL."
- **Evidence:** The resume log (`experiments/tinker-runs/logs/n2_reward_tensor_resume_20260704.out`) shows GRPO completed 40/40 and the run is currently executing AERO (step 9/40 at time of review). PID 1063289 is alive and writing to `experiments/results/n2_reward_tensor_resume/`.
- **Suggested rephrase:** "N2 is the reward-tensor instrumentation run. GRPO finished 40/40; it is now on AERO (step ~9/40) and will continue through GIFT and AREAL."
- **Severity:** Major — status of a headline active experiment is wrong.

### 4. **MAJOR — N10 status is stale/internally inconsistent**
- **Location:** Slide 3, transcript Slide 3.
- **Current wording:** "N10 is the gsm8k_cot seed expansion ... on seed 42, step 12 of 15."
- **Evidence:** The same log shows `n10_grpo_s42` already completed 15/15. The manifest (`n10_manifest_20260704.json`) contradicts the log, reporting all runs as failed due to a missing `TINKER_API_KEY`. The on-disk JSONs (`n10_grpo_s{42,179}.json`) are from earlier short runs (3 and 1 steps). The live PID 1063290 is running with `--seeds 8`, but the current seed/step is unclear.
- **Suggested rephrase:** "N10 is the gsm8k_cot seed expansion. The live run (PID 1063290) has finished grpo seed 42/15 steps and is proceeding through the 8-seed panel; the manifest file is stale and needs regeneration."
- **Severity:** Major — the claim is contradicted by the primary log, and the repo contains conflicting artifacts that should be reconciled before the meeting.

### 5. **MAJOR — Berkeley run counters are stale in the slide deck**
- **Location:** Slide 5.
- **Current wording:** "Current: iter 18, 25 findings, $55 spend."
- **Evidence:** `minimax_autoresearch/state_berkeley/progress.json` shows iteration 19, findings 27, total_cost_usd 61.09. The transcript Slide 5 says "currently at iteration 18 with a fresh heartbeat," which is also one iteration behind.
- **Suggested rephrase:** "Current: iter 19, 27 findings, ~$61 spend."
- **Severity:** Moderate–Major — the slide is hardcoded with stale numbers; the transcript repeats the stale iteration count.

### 6. **MODERATE — Submission bundle claim overstates readiness**
- **Location:** Slide 6, transcript Slide 6.
- **Current wording:** "NeurIPS 2026 Datasets and Benchmarks submission bundle is assembled with manifest, reviewer README, and checksums."
- **Evidence:** `submission/contents/MANIFEST.md` and `checksums.sha256` exist, but the files they list (`paper.pdf`, `paper_anon.pdf`, `code.tar.gz`, `ethics_statement.pdf`) are not present in the repo. Only the manifest and checksum files are assembled.
- **Suggested rephrase:** "The NeurIPS D&B manifest and checksum file are assembled; the constituent files referenced in the manifest still need to be generated and verified."
- **Severity:** Moderate — "assembled" implies the bundle is ready to submit.

### 7. **MODERATE — Worktree commit count is wrong**
- **Location:** Slide 5.
- **Current wording:** "worktree now 27 commits ahead of main."
- **Evidence:** `git rev-list --count main..HEAD` in `/home/claude/tinker-rl-lab-minimax` returns 29.
- **Suggested rephrase:** "worktree is 29 commits ahead of main."
- **Severity:** Minor — easily corrected, but shows the deck was not generated from live data.

### 8. **MODERATE — "Verified findings" / AUTORESEARCH_FINDINGS.jsonl count is ambiguous**
- **Location:** Slide 5.
- **Current wording:** "13 verified findings appended to AUTORESEARCH_FINDINGS.jsonl" (for the main 35 h run).
- **Evidence:** The main run did produce 13 findings (`state/progress.json`). However, the worktree `AUTORESEARCH_FINDINGS.jsonl` now contains 27 lines, mixing main-run and Berkeley-run findings. There is no `"verified": true` field in the JSONL, so "verified" is not programmatically distinguishable.
- **Suggested rephrase:** Keep the main-run 13 finding count, but clarify that the ledger file also contains Berkeley findings and that verification status is implicit (or add a verified field).
- **Severity:** Moderate — potential for Ramesh to over-count verified output from the main run.

### 9. **MODERATE — Missing caveats on re-analysis approximations**
- **Location:** Slide 4, transcript Slide 4.
- **Current wording:** "A2 re-plotted the scaling null using a frontier-proposed contrastive-yield abscissa, C_eff... A4 ran a causal mediation estimator on existing group-tensor rollouts."
- **Evidence:** `a2_meta.json` explicitly lists caveats: `p_x` is proxied by per-step mean reward and `KL_t` is proxied by `|loss|`. `mediation_estimates.json` notes the GER point estimate is unstable (CI −5.1 to +6.6). The transcript does mention GER instability for A4 but omits the A2 proxy caveats.
- **Suggested rephrase:** For A2, add: "C_eff is computed with available step-level proxies for per-prompt pass probability and KL; the null is robust to these proxies on the current evidence base."
- **Severity:** Moderate — the re-analyses are sold as "zero-cost" (true) but the approximation chain is not fully transparent.

### 10. **MINOR — Tone / potential oversell in autonomous-agent framing**
- **Location:** Slide 5, transcript Slide 5.
- **Current wording:** "The autonomous engine is the main productivity multiplier" / "the agent can run unattended and still produce committable, evidence-backed artifacts."
- **Evidence:** The agent has produced 13 main-run findings and 27 Berkeley findings, and guardrails are in place (writes confined to worktree, push/upload/destructive ops blocked). However, this morning's billing incident shows the *Tinker-dependent* loop is not fully unattended, and "committable" does not mean "committed/pushed".
- **Suggested rephrase:** "The agent runs unattended on the analysis and writing loop; API billing still needs human monitoring, and commits remain local until manually reviewed."
- **Severity:** Minor — not false, but slightly too bullish for a status update that also flags a billing incident.

---

## Overall Verdict

**Do not deliver the transcript as-is.** The status update is mostly accurate on completed work (re-analyses, MiniMax M3 main run, billing incident) but contains multiple stale or false claims about live state and evidence hygiene. The two severe issues — the false clean-build claim and the unremoved fabricated citations — directly undermine the integrity narrative that is central to the deck. Refresh the active-run status from live logs, rerun the LaTeX build check and clean up undefined references/citations, and reconcile the N10 manifest before presenting to Ramesh.

**Top 3 most important issues to fix:**
1. **False claim that the four pillar papers compile with zero undefined citations/references** — build logs show P1 has undefined citations and P1/P2/P4 have undefined references.
2. **False claim that two fabricated citations were removed** — `liu2026gdpo` and `nimmaturi2025scalinglaws` still appear in the main repo and worktree source files.
3. **Stale N2 status** — the transcript says GRPO step 26/40, but the resume log shows GRPO finished and AERO is already running.
