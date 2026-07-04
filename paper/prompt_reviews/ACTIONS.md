# ACTIONS — Cross-Paper Prioritized Action List (Review Round 2026-07-04)

Synthesized from `P1_REVIEW.md`–`P4_REVIEW.md` and the four `*_decisive_experiment.md`
files in this directory. Launch-ready specs for the four decisive experiments live at
`/home/claude/tinker-rl-lab/experiments/NEW_EXPERIMENTS_FROM_STRESS_TESTS.md` (ids SD1–SD4);
dedup against `experiments/NEW_EXPERIMENTS_PLAN.md` is recorded there, not here.

**State of the queue at time of writing:** plan items N2 (reward-tensor-instrumented runs)
and N8 (per-prompt pass-rate spectrum) are already launched — smoke + first artifacts under
`experiments/results/n2_reward_tensor/` and `experiments/results/n8_passrate_spectrum/`
dated 2026-07-04 11:23–11:32. No `experiments/launch_log.md` exists.

Effort key: `d` = person-day (zero compute unless noted). Risk-reduction is the expected
drop in reviewer-rejection probability for the affected paper(s).

## Ranked action table

| Rank | Paper | Action | Type | Effort | Expected review-risk reduction |
|------|-------|--------|------|--------|-------------------------------|
| 1 | P2 | Re-label ALL `variance_mitigation.tsv`-derived numbers as synthetic/dry-run projections (iter126 H1–H3, `tab:zvf-by-library`, zvf-dynamics pooled table, iter130 rows built on the 45 synthetic trajectories) and apply the 7 claim-lint rewrites | fix prose | ~1 d | **Very high — integrity-level.** Unconditional in every branch of every experiment; publishing byte-identical simulator output as measured data is a desk-reject if a reviewer regenerates `synthesize_rows()`. |
| 2 | P4 | Disclose the `MAX_NEW = 200` cap + step-0 saturation in the GSM8K-CoT paragraphs; fix the 4 unsupported claims (iter120 citation, `tab:reward-shape` Theil–Sen column, iter32 CI, iter128 counts); rescope the W claims | fix prose | 2–4 h | **Very high.** The undisclosed cap is one line of released code away from a validity-level reject ("no length inflation" is true by construction). Disclosure is required even if the SD4/cap-512 batch later moots it. |
| 3 | P1 | SD1 Stage A — label audit of the 5 scaling anchors (trace `model` field vs `base_instruct_paired.tsv` vs iter129 labels); file the two errata (Qwen3-8B labeled base but is Instruct; Nemotron A12B MoE labeled dense) | re-analysis | ~0.5 h script | **Very high for P1.** Pre-check says the hard-kill fires: the bimodality/capability-class clause — P1's only affirmative causal claim — is void as stated until re-derived from verified labels. Gates what ranks 6 and 9 may say. |
| 4 | P3+P4 (+P2) | Zero-compute Stage-0 batch: P3 token-matched G=4@4s vs G=16@s re-slice of `groupsize_zvf_sweep.json` (SD3 Stage 0) + P4 exhaustive 256-assignment joint sign-flip permutation of the iter136 p=0.0039 headline (SD4 Stage 0); archive P2's already-executed Arm A table alongside | re-analysis | ~1 d CPU total | **High, two papers at once.** Each can kill or rescue a headline claim for free: P3's equivalence claim dies before any compute if R0 fires; P4's GLOBAL_REJECT either survives an honest null or is downgraded today. **Multi-paper action.** |
| 5 | P1 | Fix the 8 contradicted claims + 5 mislabels ("five orders of magnitude" → ≈2.4; Δ₁T paragraph; PPO control numbers; `tab:scaling-cross` CI; iter-133 0/7→2/7; etc. per lint list) | fix prose | ~1 d | **High.** All reviewer-reproducible from released files; required under every branch of SD1. |
| 6 | P1 | Stage-0 detection-floor paragraph: promote iter121 late−early gain-proxy regression + synthetic-recovery power table (≤26% power) into the build; add level-vs-gain decomposition; rescope headline to "no slope measurable in the saturated regime at T ≤ 30" | add ablation (free) | ~1 d | **High.** Converts P1's biggest rejection argument ("the null is unfalsifiable as measured") from a silent flaw into a scoped, disclosed claim — without GPU spend. |
| 7 | P3 | Fix prose: gradient-validation claim (3×, no artifact), impossible p<10⁻²⁴ at n=4, half-saturation column off 2.7×, tag `FALLBACK_ROWS` grid "illustrative, reconstructed", accuracy-vs-reward retention wording, honest run counts | fix prose | ~0.5 d | **High.** One impossible statistic or the synthetic-grid-as-measured-sweep alone can sink credibility of the otherwise-solid non-monotonicity half. |
| 8 | P2 | Reward-trace + entropy-trace risk-index baselines with leave-one-method-out CV; splice `zvf_iter134_heldout.tsv` (100% false positives on real converged runs) into the paper; pre-committed rule: ZVF keeps claimed value iff LOMO-AUROC(ZVF) − LOMO-AUROC(reward) > 0, CI excluding 0 | re-analysis | ~1 d, one stdlib script | **High.** Decides whether P2 has any positive contribution beyond signals every dashboard already logs — the reviewers' most probable rejection argument. |
| 9 | P1 | SD1 Stages B+C — harness-replica vs harness-fixed sampling of Qwen3-8B (600 completions, step 0) + cluster recomputation | run experiment (sampling-only) | ~0.5 d, 0 training steps | **Medium-high.** Settles harness-vs-capability attribution for the incapable cluster. **Multi-serving:** Stage B doubles as a baseline-offset c measurement for the saturation refit (plan item N5), and cross-checks against the already-launched N8 pass-rate spectrum on the same model. |
| 10 | P4 | SD4/Action-1 combined 12-arm batch: 6 arms @ cap 200 with `save_weights_for_sampler` (replication gate + truncation-retention sweep) + 6 arms @ cap 512 (censoring ablation) | run experiment | ~10 GPU-h / <9M tokens | **High per GPU-hour.** One batch retires P4's two biggest threats (undisclosed censoring; untested equivalence clause) and unblocks plan item A3 by reconstituting the missing Dr.GRPO checkpoints. **Multi-serving action** (P4 Actions 1+3, plan A3). |
| 11 | P2 | SD2 Arm B — one destabilized real run (Llama-3.2-1B, lr 1e-3 = 10× safe, 40 steps) with external collapse rule; decides CSD claim's fate against the existing real safe trace | run experiment | 1 run, ~15–30 min + 1 h CPU | **Medium.** Conditional-only: Arm A already forces the rank-1 relabeling regardless; this run just picks which of two pre-written endings the CSD section gets, and gates whether plan item N1 (10-seed prospective validation) is ever worth launching. |
| 12 | P3 | SD3 Stage 1 — measured token-matched G=4 vs G=32 pair at T=4M, 3 seeds, Qwen3.5-4B, frontier-difficulty pool | run experiment | 6 runs, 24M tokens; **needs G>8 cap exception** | **High if run.** Every version of the equivalence claim currently rests on a synthetic grid; this cell fires exactly one of R1/R2/R3. Batch the exception request with plan items N7/B1 (same large-G blocker). Only if Stage 0 (rank 4) passes. |
| 13 | P3 | Gradient-vector residual-isomorphism ablation: per-batch cos(V_GRPO − V_mDPO, V_KL) ≈ 1 on the existing 0.5B/arithmetic setup | add ablation | ~1 d eng + trivial compute | **Medium.** Upgrades the rank-7 hedged wording back to a genuine validation claim; without it the demotion stands (acceptable for submission). |
| 14 | P1 | Stage-1 headroom-controlled cross-scale gain positive control (3–4 dense anchors, p0 ∈ [0.2,0.6] subsets, G=8, T=150, 3 seeds) | run experiment | 40–80 H100-h | **Medium (camera-ready).** Upgrades P1's null from vacuous to substantive ("H > 0, slope ≈ 0, tight CI"). Exceeds the ≤40-step cap → post-audit / camera-ready window; design must inherit SD1's verified labels. |

## Actions serving multiple papers / queue items

- **Rank 4 (Stage-0 batch)** — one CPU day, decides headline claims for P3 *and* P4 and
  archives P2's Arm A; cheapest risk reduction on the entire list after the prose fixes.
- **Rank 9 (SD1 Stage B)** — P1 attribution + N5 c-measurement + N8 cross-check in one
  600-completion sampling job.
- **Rank 10 (12-arm batch)** — P4 cap ablation + P4 equivalence probe + unblocks plan A3
  (previously "deferred until Dr.GRPO ckpt confirmed" — no checkpoints ever existed, so
  reconstitution is the only path).
- **Ranks 1/2/5/7 (prose passes)** — same editorial skill set, 24 lint-flagged claims across
  the 4 papers; can be executed as one 2–3-day cross-paper claim-lint sprint, ideally by one
  person for consistency of hedging register.
- **Already-launched N2** (reward-tensor instrumentation) will eventually let P2 *replace*
  the rank-1 synthetic rows with real tensor-level traces — but relabeling now is still
  mandatory; do not wait on N2.

## Sequencing note

Ranks 1–7 are all zero-GPU and jointly form the "submission-safe floor": no paper should be
submitted before its rows in 1–7 are done. Ranks 9–12 decide which *claims* survive; run
rank 4 and rank 3 first because their outcomes rewrite what ranks 9–12 must test (SD3 Stage 1
is skipped entirely if its Stage 0 kills equivalence; SD1 Stage B/C wording depends on the
Stage A branch).
