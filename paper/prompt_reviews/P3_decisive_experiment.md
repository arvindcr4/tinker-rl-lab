# P3 — Minimal Decisive Experiment (Group-Size Equivalence, Disconfirming Check)

Prompt contract: `research_prompts/design/minimal-decisive-experiment.md`
Feeds from: Hypothesis Stress Test (P3_stress_test.md). Feeds into: Section Drafter from Notes.
Date: 2026-07-04

## Input (placeholders filled)

- **Research question** `{{research_question}}`: Does token-normalized equivalence between small and large group sizes (G=4 vs G=32) survive on a *measured*, token-matched pair at T = 4M optimizer-visible tokens per arm, on a capable model (≤8B) at mid-difficulty (initial pass rate p0 ∈ [0.3, 0.6], the learning frontier) — the smallest budget at which the paper's reconstruction predicts non-equivalence (predicted Δ = +0.11)? And does the same run identify the mechanism (contrast starvation via ZVF vs gradient noise σ²_R/G via GU)?
- **Decision needed** `{{decision_needed}}`: Whether to (1) demote the equivalence claim to "budget-conditional, under-trained regime only (T ≤ T_equiv ≈ 1M)", (2) demote the preference-density-dial framing to the contrast-saturated near-ceiling regime, (3) withdraw/re-derive the FALLBACK_ROWS-derived artifacts (iter107/111/115/135 retention curve, TOST table, compute-cost projection, T* extrapolations), or (4) keep the current claims. Note: the "iter138 contrast-yield analysis" referenced upstream does not exist in this worktree (max iter135); the 4–5× contrast-yield figure is iter115's (`group_size_iter115_zvf_linkage.tsv`, GU ratio 4.15–5.03×).
- **Resource limit** `{{resource_limit}}`: Tinker API only (no local GPU); model ≤ 8B; ≤ 40 training steps per run or sampling-only; ≤ 600 unique prompts total; prefer reuse of existing checkpoints/logs under `experiments/results/`. Hard cap for the measured stage: 6 runs × 4M tokens = 24M sampled tokens.

---

## 1) Setup

One experiment, two gated stages. Stage 0 costs zero compute and can kill the hypothesis by itself (trigger 4); Stage 1 runs **only if Stage 0 passes** and is the decisive measured pair (triggers 1–3). Single dataset per stage, single primary metric per stage.

### Stage 0 (gate, zero new compute): token-matched re-slice of existing logs

- **Data:** `experiments/results/groupsize_zvf_sweep.json` (Qwen2.5-0.5B, arithmetic_correctness, G ∈ {2,4,8,16}, seeds {42,123,456}, 40 steps, 16 prompts/step, per-step `mean_reward`, `zvf`, `advantage_variance`, `grad_norm`). Verified present and complete.
- **Slice:** token-matched pairs — G=4 at step 4s vs G=16 at step s (equal cumulative completions at equal prompts/step), restricted to the mid-training window where G=16 train `mean_reward` ∈ [0.3, 0.8].
- **Feasibility (audited 2026-07-04):** window = G=16 steps s ∈ {2..6} (seed 456: {2..7}); 5–6 matched pairs per seed × 3 seeds = 16 paired points, split into two disjoint sub-windows W1 = s ∈ {2,3,4}, W2 = s ∈ {5,6(,7)}.
- **Statistic:** per-window mean paired difference d = reward(G=16@s) − reward(G=4@4s), paired within seed; sign consistency across W1, W2.
- **Scope caveat (why Stage 1 exists):** Stage 0 tests equivalence only within the measured G range (4 vs 16), on a 0.5B model and a near-saturating task; it cannot confirm the hypothesis at G=32 / 4M tokens — it can only refute it cheaply.

### Stage 1 (decisive, 24M sampled tokens): measured token-matched G=4 vs G=32 pair on Tinker

- **Model:** Qwen/Qwen3.5-4B via Tinker API (repo's current ≤8B Qwen cell after Tinker dropped Qwen3-4B-Instruct-2507; fallback Qwen3-8B-Base if 4B unavailable).
- **Task / prompts (600 total, hard cap respected):**
  - *Calibration (sampling-only):* draw k = 4 completions at T = 1.0 for 600 GSM8K train prompts; measure mean completion length L̂ and per-prompt pass rate. Keep the 300 prompts with per-prompt pass ∈ [0.25, 0.6] as the **frontier train pool** (aggregate p0 must land in [0.3, 0.6]; if aggregate p0 > 0.6, tighten the keep-band to [0.2, 0.5]).
  - *Eval:* 300 fixed held-out GSM8K test prompts, greedy decoding, evaluated once per finished arm per seed. (300 train + 300 eval = 600 unique prompts.)
- **Arms (both ≤ 40 steps):** GRPO-style RL, identical LR/schedule/max-len across arms.
  - G=4: batch ≈ ⌈4M / (40 · 4 · L̂)⌉ prompts/step (≈ 84 at L̂ = 300), sampled with replacement from the 300-prompt pool (~11 epochs — log epoch count).
  - G=32: batch ≈ ⌈4M / (40 · 32 · L̂)⌉ prompts/step (≈ 10 at L̂ = 300), same 40 steps → step-matched *and* token-matched.
  - Token accounting in-loop: stop each arm at the first step where cumulative optimizer-visible tokens ≥ 4M; record exact T per run (arms must land within 5% of each other).
- **Seeds:** {42, 123, 456} per arm → 6 runs. Cost: 6 × 4M ≈ 24M sampled tokens.
- **Primary metric (single):** held-out accuracy; Δ = acc(G=32) − acc(G=4), paired at the prompt level within seed (900 paired observations), 10k-resample paired bootstrap for the 95% CI, plus TOST at ε ∈ {0.024, 0.05} (two one-sided 90% CIs). Power note: paired SE ≈ 0.018, so the primary trigger at predicted Δ = +0.11 has ≈ 4.8 SD of margin, and TOST at ε = 0.05 is well-powered; TOST at ε = 0.024 is *underpowered* at this n and is reported as directional only.
- **Mechanism telemetry (same run, no extra cost):** per-step, per-group log of advantage variance, ZVF (zero-variance fraction of groups), and GU (non-zero-gradient contrast yield per group), mirroring the `groupsize_zvf_sweep` step-log schema, so the mechanism readout uses the identical fields as iter115.

## 2) Decision rule

Evaluate in this order; first firing rule decides.

- **R0 (gate, from Stage 0):** if |mean paired d| > 0.02 mean reward in a window, sign-consistent across ≥ 2 of the sub-windows W1/W2 (paired across all 3 seeds) → token-normalized equivalence fails *within the measured G range*; revise the hypothesis **before any new compute** and do not run Stage 1 as designed (re-scope it to locate T_equiv instead). Otherwise proceed to Stage 1.
- **R1 (primary):** if the 95% paired-bootstrap CI lower bound of Δ at token-matched T = 4M is > +0.024 (equivalently retention < 90% with CI upper bound < 0.976) → revise the hypothesis to "equivalence is budget-conditional and holds only in the under-trained regime, T ≤ T_equiv ≈ 1M tokens on this benchmark."
- **R2 (mechanism kill-shot):** if simultaneously GU(G=4)/GU(G=32) ≥ 2 while retention < 90% → the preference-density mechanism is falsified for this regime (gradient noise σ²_R/G is binding; confirms iter115 Finding 2 on measured data); demote the "preference-density dial" framing to the contrast-saturated near-ceiling regime only.
- **R3 (symmetric trigger):** if TOST passes at ε = 0.05 (p < 0.05) at T = 4M where the reconstruction predicted Δ = +0.11 → the FALLBACK_ROWS illustrative table and all downstream artifacts (iter107/111/115/135 retention curve, TOST table, compute-cost projection, T* extrapolations) are withdrawn or re-derived from the measured runs.
- **R4 (keep):** if neither R1 nor R3 fires (CI straddles the band and TOST fails) → the run is not decisive at this n; see 4) Failure interpretation.

## 3) Success threshold

The experiment *succeeds as an experiment* (i.e., is decisive) iff exactly one of these obtains:

- **Equivalence refuted:** Δ 95% CI lower bound > +0.024 (R1), with or without the R2 mechanism finding; or R0 fires at Stage 0 with |d| > 0.02 and ≥ 2/2 sign-consistent windows.
- **Reconstruction refuted:** TOST at ε = 0.05 passes with p < 0.05 (R3) — i.e., the measured pair is statistically equivalent exactly where the paper's reconstruction predicted a +0.11 gap.

Validity preconditions (any violation voids the readout, does not fire a trigger): aggregate p0 of the frontier pool ∈ [0.3, 0.6]; per-arm realized T within 4M ± 5% and arms within 5% of each other; all 6 runs complete ≥ 35/40 steps; held-out eval on the identical 300 fixed prompts, greedy, for every run.

## 4) Failure interpretation

- **R4 outcome (inconclusive band: CI lower bound ≤ 0.024 but TOST at 0.05 fails):** the effect at 4M is real-but-small or noise-limited. **Fallback action (one, pre-committed):** do *not* add runs; re-run the identical eval on the 3 existing checkpoint pairs with the eval set enlarged to the full GSM8K test split (sampling-only, no new training) to shrink the paired SE ~2×, then re-apply R1/R3 once. If still in the band, downgrade the paper's equivalence claim from "holds at practical budgets" to "not distinguishable from a ≤ 5-point gap at T = 4M," and gate all T*-extrapolation artifacts behind that caveat.
- **Stage 0 fires but Stage 1 was expected to pass:** treat the existing FALLBACK_ROWS-derived retention curve as unvalidated in *both* regimes; hypothesis revision happens on Stage 0 evidence alone (zero-compute kill), and Stage 1's budget is re-scoped to a T_equiv bisection (1M vs 4M, single seed pair) rather than the 6-run design.
- **Validity precondition violated (e.g., p0 lands > 0.6 — model at ceiling):** the run says nothing about the frontier regime; re-select the frontier pool with the tighter keep-band and re-run only the affected arms — the design, thresholds, and triggers are frozen and are not revised post hoc.

## Handoff (per prompt-pack convention)

- → Section Drafter `{{raw_notes}}`: the Stage 0/Stage 1 setup above + measured results tables (Δ, CI, TOST p-values, GU ratio, ZVF trajectories).
- → Section Drafter `{{must_keep_points}}`: decision rules R0–R4 verbatim, incl. the 0.024/0.05 thresholds and the GU ≥ 2 mechanism criterion.
