# P4 — Minimal Decisive Experiment (Length Bias / Dr.GRPO Equivalence & Coupling, Disconfirming Check)

Prompt contract: `research_prompts/design/minimal-decisive-experiment.md`
Feeds from: Hypothesis Stress Test (P4_stress_test.md). Feeds into: Section Drafter from Notes.
Date: 2026-07-04

## Input (placeholders filled)

- **Research question** `{{research_question}}`: Does the "GRPO and Dr.GRPO are practically equivalent on GSM8K-CoT" clause survive constrained decoding — i.e., do the two converged policies (Qwen2.5-1.5B-Instruct, 3 seeds each) retain held-out accuracy equally when the generation cap is swept below the natural mean completion length (~180–190 tokens), T_max ∈ {64, 96, 128, 160, 192, 256, 512}? Companion: does the iter136 GLOBAL_REJECT sign-test (p=0.0039) survive a structure-respecting joint permutation that flips the algorithm label per seed (exhaustive 2^5 × 2^3 sign-flips, all four statistics recomputed per task), instead of treating the 8 (task × hypothesis) cells as independent?
- **Decision needed** `{{decision_needed}}`: (1) If GRPO's mean retention falls ≥ 10 pp below Dr.GRPO's at any cap T ≤ 160 with the paired-seed bootstrap 95% CI excluding zero (equivalently pooled McNemar p < 0.05 at that cap) → rescope "practically equivalent" to "equivalent only at unconstrained decoding". (2) If the joint permutation yields p ≥ 0.05 → downgrade GLOBAL_REJECT/p=0.0039 to "directionally consistent, not significant". (3) Symmetric strengthening: if retention curves stay within ±3 pp at every cap down to 64 → equivalence clause behaviorally confirmed; reword the coupling clause to "statistically detectable but behaviorally inert at this horizon".
- **Resource limit** `{{resource_limit}}`: Tinker API only (no local GPU); model ≤ 8B; ≤ 40 training steps per run or sampling-only; ≤ 600 unique prompts; prefer pure re-analysis of `experiments/results/` or reuse of existing checkpoints/logs.

---

## 0) Feasibility audit (what actually exists — changes the design)

Audited 2026-07-04 against the repo:

- **No checkpoints exist.** The GSM8K-CoT runs were produced by `experiments/modal/modal_drgrpo_gsm8k_cot.py` (Modal A10G, HF + LoRA r=16, not Tinker). The script saves **only JSON metrics** (`{algo}_s{seed}.json`, lines 201–204); the LoRA adapters were never written. The handoff's premise "take the already-trained converged checkpoints and re-evaluate" is therefore **not executable as stated** — the minimal faithful instantiation must reconstitute the 6 arms, this time saving sampler weights.
- **The eval set was 200 items, not 500.** `N_EVAL = 200`, `test.shuffle(seed=0).select(range(200))`, greedy, `MAX_NEW = 200`. Per-item binary arrays `pre_correct`/`post_correct` for all 6 runs are stored in `experiments/results/drgrpo_gsm8k_cot_full.json` (grpo post: .255/.260/.275; dr_grpo post: .250/.265/.250; mean_comp_len_last5 ≈ 170–191 tokens — consistent with the ~184–189 natural-length premise).
- **The original cap was already 200 tokens**, so caps {256, 512} probe *beyond* the training-time decoding regime, not below it; the decisive region T ≤ 160 is unaffected.
- **The companion is pure re-analysis and fully executable now.** Inputs: `drgrpo_gsm8k_cot_full.json` (gsm8k_cot: 3 seeds × 2 algos, 30-step step_log) and `drgrpo_vs_grpo.json` (arithmetic_easy: 5 seeds × 2 algos, 40-step step_log); statistic definitions in `scripts/length_bias_iter136.py`; observed cells in `experiments/results/length_bias_iter136_paired_tests.tsv` and `_summary.tsv`.
- **Greedy decoding is prefix-stable**: the cap-T greedy output is exactly the first T tokens of the cap-512 greedy output (and identical if EOS fires before T). So the 7-cap sweep needs **one** 512-cap generation per (arm, item), with lower caps scored by prefix truncation + answer re-extraction. This collapses the handoff's 21,000 generations to 6 × 200 = **1,200**.

## 1) Setup

One experiment, two stages. Stage 0 is CPU-minutes, zero new compute, always runs first, and decides clause (2) by itself. Stage 1 is the decisive measured sweep for clauses (1)/(3). Single dataset per stage, single primary metric per stage.

### Stage 0 (companion, zero new compute): structure-respecting joint sign-flip permutation of the iter136 global sign test

- **Data:** step logs above; recompute the four iter136 statistics (H1 |ρ(ΔR,ΔL)|, H2 |ρ_len lag-1|, H3 late efficiency, H4 ρ(ΔZ,ΔL)) per run exactly as in `scripts/length_bias_iter136.py`.
- **Null:** for each seed, flip the GRPO/Dr.GRPO label **jointly** — the flip swaps that seed's pair of runs simultaneously for *all four* hypotheses within its task, respecting shared-seed/shared-trajectory dependence. Exhaustive enumeration: 2^5 (arithmetic seeds) × 2^3 (gsm8k seeds) = **256 joint assignments** — exact, no Monte-Carlo error.
- **Statistic:** the observed global count, K = number of the 8 (task × hypothesis) paired cells whose mean paired delta lies in the pre-registered direction (observed K = 8/8).
- **Output:** exact one-sided p = #{assignments with K′ ≥ 8} / 256. Attainable floor: p_min = 1/256 ≈ 0.0039 (only if the identity flip is the unique maximizer). Write `experiments/results/length_bias_iter140_joint_perm.tsv` (or next free iter slot).

### Stage 1 (decisive, ~3M sampled tokens): Tinker reconstitution + single-pass truncation-retention sweep

- **Reconstitution (required because no checkpoints were saved):** re-run the exact regime of `modal_drgrpo_gsm8k_cot.py` on the Tinker API: Qwen/Qwen2.5-1.5B-Instruct, LoRA r=16, 30 steps (≤ 40 ✓), 8 prompts/step × G=8, K=2 clipped epochs (ε=0.2), lr 1e-5, max_new = 200, GRPO = std-normalized advantages / Dr.GRPO = centered-only with fixed 1/(B·MAX_NEW) token normalizer, seeds {42,123,456} × {grpo, dr_grpo} = 6 arms. **The one change vs the original: call `save_weights_for_sampler` at the final step.** If Qwen2.5-1.5B-Instruct is not served by Tinker, fall back to re-running the original Modal script with adapter-saving added (identical numerics to the original artifact; note the backend in the writeup).
- **Prompt budget:** 8 prompts/step × 30 steps = 240 unique train prompts per run + the **identical 200 held-out test items** (`gsm8k test.shuffle(seed=0)[:200]`) = 440 ≤ 600 ✓. Deviation from handoff: 200 eval items, not 500 — this respects the prompt cap *and* makes the full-cap endpoint directly comparable to the stored per-item arrays. Pooled across 3 seeds this still gives 600 paired GR-vs-DR item observations per cap.
- **Replication gate (validity precondition, not a trigger):** each retrained algo-mean must land within ±3 pp of the stored post accuracy (GR .263, DR .255) at cap 200, and mean completion length within ±15 tokens of the stored last-5 means (~180–187). Violation voids the truncation readout (the reconstitution failed to reproduce the regime the claim is about).
- **Sweep (sampling-only):** one greedy pass per (arm, item) at T_max = 512, storing token IDs → 1,200 generations (≈ 0.6M tokens). For each cap T ∈ {64, 96, 128, 160, 192, 256, 512}: truncate to the first T tokens, decode, re-run the `\boxed{}` / last-number extractor, score against gold. Exactness follows from greedy prefix-stability; verify on a 20-item spot-check by actually generating at T=128.
- **Primary metric (single):** retention ret_A(T) = Acc_A(T) / Acc_A(512) per (algo, seed); paired per-seed difference Δret(T) = ret_DR(T) − ret_GR(T). Inference: 10k paired-seed bootstrap on mean Δret(T); confirmatory pooled McNemar (600 item pairs, GR vs DR correctness at cap T, same items within seed) at each cap.
- **Cost:** training ≈ 6 × 30 × 64 completions × ≤200 tok ≈ 2.3M sampled tokens; sweep ≈ 0.6M; Stage 0 ≈ CPU-minutes. Total < 3M sampled tokens, no run exceeds 30 training steps.

## 2) Decision rule

Evaluate in this order; Stage 0 and Stage 1 decide different clauses and both readouts are reported.

- **R0 (coupling clause, from Stage 0 — always fires one way):** if exact joint-permutation p ≥ 0.05 → downgrade the iter136 headline from "GLOBAL_REJECT, p=0.0039" to "directionally consistent (8/8), not significant under a structure-respecting null (p = k/256)". If p < 0.05 (i.e., K′ ≥ 8 in ≤ 12 of 256 assignments) → the coupling clause stands, now under the honest null; replace the reported p with the joint-permutation p.
- **R1 (equivalence fails):** if at **any** cap T ∈ {64, 96, 128, 160}: mean Δret(T) ≥ +10 pp (Dr.GRPO over GRPO) **and** the paired-seed bootstrap 95% CI on Δret(T) excludes 0 (or pooled McNemar p < 0.05 at that cap, same direction) → rescope the paper's clause to "practically equivalent **only at unconstrained decoding**; under generation caps ≤ 0.85× natural length GRPO degrades faster by ≥ 10 pp retention".
- **R2 (symmetric strengthening):** if |mean Δret(T)| ≤ 3 pp at **every** cap down to 64 → the equivalence clause is behaviorally confirmed under constrained decoding; reword the coupling clause (whatever R0 decided) to "statistically detectable but behaviorally inert at this horizon".
- **R3 (grey zone):** any other pattern (gaps in (3, 10) pp, or ≥ 10 pp with CI straddling 0, or direction favoring GRPO) → not decisive for clause (1); see 4).

## 3) Success threshold

The experiment *succeeds as an experiment* (is decisive) iff:

- **Stage 0:** always — the permutation is exhaustive (256 assignments), so R0 yields an exact yes/no on clause (2) with zero sampling error; and
- **Stage 1:** R1 or R2 fires. Concretely: at the pooled n = 600 item pairs per cap, a true 10 pp retention gap at base accuracy ~0.26 yields an expected discordant-pair count comfortably powered for McNemar at α = 0.05, so a grey-zone outcome reflects a genuinely small/unstable effect, not under-powering at the 10 pp threshold (the ±3 pp band, by contrast, is a descriptive equivalence band, not a powered TOST).

Validity preconditions (violation voids Stage 1, fires no trigger): replication gate passes (±3 pp accuracy, ±15 tok length, all 6 arms complete 30 steps); identical 200 eval items and prompt template across all arms; greedy decoding; prefix-stability spot-check (20 items at T=128) shows 0 mismatches; Acc_A(512) ≥ 0.15 for every arm (retention ratios with near-zero denominators are not interpretable).

## 4) Failure interpretation

- **Replication gate fails** (Tinker port does not reproduce the Modal endpoint): the truncation readout cannot be attributed to the original claim's artifacts. Fallback action (one): re-run via the original Modal script with the single added `save_pretrained` call, then execute the identical sweep offline on the saved adapters — same budget, backend-matched.
- **R3 grey zone** (e.g., 3–10 pp gaps, or unstable sign across seeds): the equivalence clause is neither refuted nor confirmed under constrained decoding at this n. Pre-registered fallback action (one): amend the paper's claim wording to "practically equivalent at the training-time decoding budget (T_max = 200); retention under tighter caps showed a non-significant gap of X pp (95% CI [a, b]) and is left unresolved" — no further compute; do not silently keep the unqualified claim.
- **Stage 0 returns p ≥ 0.05 while Stage 1 fires R2:** both clauses convert to the weakest joint form — "directionally consistent step-level coupling, not significant under the joint null, and behaviorally inert under decoding constraints" — i.e., the pillar's contribution is descriptive, and the paper's P4 section must lead with the equivalence result, not the coupling statistic.
- **Stage 0 returns p < 0.05 while Stage 1 fires R1:** strongest joint outcome — coupling is real under the honest null *and* has behavioral teeth; promote the truncation curve to a main figure and the joint-permutation p to the headline statistic.
