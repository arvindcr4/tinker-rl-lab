# NEW_EXPERIMENTS_FROM_STRESS_TESTS — SD1–SD4 launch-ready specs

Date: 2026-07-04. Source: the four per-paper decisive-experiment contracts
(P1–P4), synthesized from the corresponding per-paper stress-test reviews.

**Dedup policy:** cross-referenced against `experiments/NEW_EXPERIMENTS_PLAN.md`
(read-only — that file is NOT edited by this doc). No `experiments/launch_log.md` exists;
launch status was verified from result directories: plan items **N2** and **N8** are
LAUNCHED (smoke + first artifacts in `experiments/results/n2_reward_tensor/` and
`experiments/results/n8_passrate_spectrum/`, 2026-07-04 11:23–11:32). An SD entry is
marked MERGED-WITH only where it duplicates a plan item; cross-links that merely share
infrastructure or double as partial coverage are noted without merging.

Caps applied to every launch stage: Tinker API only, model ≤ 8B, ≤ 40 training steps per
run, **G ≤ 8**, ≤ 600 unique prompts — or pure re-analysis. (Note: the G ≤ 8 cap is
tighter than the per-paper contracts assumed; SD3 Stage 1 is affected, see its entry.)

---

## SD1 — P1: Harness-vs-capability audit of the "incapable" scaling cluster

**Status: NEW** (no plan-item duplicate; Stage B doubles as partial coverage of plan
item N5, and cross-checks the launched N8 — see dedup note).

- **Goal:** Determine whether the low-R̄ cluster in the P1 scaling analysis (Qwen3-8B
  anchor R̄ = 0.2854; Nemotron-120B R̄ = 0.1750) is a capability-class (instruct-vs-base)
  effect or a harness artifact (inverted checkpoint labels + hardcoded ChatML template +
  512-token cap on a thinking-mode model).
- **Disconfirming logic:** The bimodality/capability-class clause is P1's only affirmative
  causal claim. If the anchor labels are inverted (Stage A pre-check indicates they are:
  the trace's `model = "Qwen/Qwen3-8B"` is the **Instruct** `model_id` in
  `base_instruct_paired.tsv`, scoring 0.2925 pre-RL, while `Qwen/Qwen3-8B-Base` scores
  0.8250 — yet iter129 labels the anchor `dense-base`), or if zero-reward completions are
  predominantly harness-attributable, the clause is void/re-attributed. A clean audit +
  |B2 − B1| ≤ 0.05 lets the clause survive.
- **Config (within caps; 0 training steps):**
  - Stage A (re-analysis, ~30 min): join the 5 anchors in
    `scaling_law_iter117_meta.json` → trace `model` field → `base_instruct_paired.tsv`
    `model_id` → `scaling_law_iter129_capability_scaling.tsv` label; emit a 5-row
    agree/contradict table.
  - Stage B (sampling-only Tinker job, ≤ 600 completions): `Qwen/Qwen3-8B`, 150 GSM8K
    train prompts (seed 42, loader of `tinker_parallel_runner.py`), 2 samples/prompt ×
    2 conditions — B1 harness replica (hardcoded ChatML + `SYSTEM_PROMPT_MATH`,
    max_tokens 512, T=0.8, top_p 0.95) vs B2 harness fixed (native chat template,
    thinking disabled, fallback max_tokens 2048). Score with the original `reward_math`;
    classify zero-reward B1 completions (correct-but-unparsed / truncated-at-cap /
    wrong / degenerate).
  - Stage C (re-analysis, ~1 h): substitute corrected Qwen3-8B value; recompute largest
    R_max gap (vs 0.5313), dip statistic (vs 0.5216, p = 0.056), cross-scale slope.
    Nemotron (120B > cap): trace forensics only; flag unresolved within budget.
- **Expected artifact:** iter-style meta JSON + Stage-A verification table + B1/B2
  per-completion scores/classifications + recomputed gap/dip/slope under
  `experiments/results/` (suggested: `p1_harness_audit/`).
- **Revision trigger (pre-registered):** Hard-kill if Stage A confirms label inversion →
  bimodality clause void as stated, all class claims re-derived. Revision if ≥30%
  zero-reward B1 completions correct-but-unparsed, or B2 ≥ B1 + 0.15, or recomputed gap
  < 0.25 / any anchor crosses the 0.2854↔0.8167 boundary → re-attribute dominant axis to
  harness compatibility, withdraw bimodality clause. Indeterminate band (10–30% /
  0.05–0.15) → descriptive downgrade ("pre-RL harness-measured accuracy separates the
  anchors"). In every branch: fix the two label errata (Qwen3-8B class; Nemotron = A12B
  MoE, not dense).
- **Dedup vs plan:** Not in NEW_EXPERIMENTS_PLAN. Stage B's B2 condition doubles as a
  direct baseline-offset c measurement for the 8B anchor (plan item **N5**, itself partly
  covered by the launched **N8** K=64 spectrum on the same model). Cross-check: N8's
  pass-rate histogram for Qwen/Qwen3-8B under a correct template independently
  arbitrates B1-vs-B2 before Stage B even runs — consume its `passrates.jsonl` in the
  Stage-C writeup.

---

## SD2 — P2: One destabilized real run to decide the ZVF critical-slowing-down claim

**Status: NEW** (related to plan item **N1**, not a duplicate — SD2 is the ~1-run gate
that decides whether N1's 10-seed prospective validation is ever launched).

- **Goal:** Test on a *real* GRPO run whether detrended matched-window rolling lag-1
  autocorrelation (w=15) of the per-step ZVF trace rises before an *externally defined*
  collapse with an actionable (≥10-step) τ-crossing lead — versus the existing real safe
  trace (`experiments/results/arithmetic_metrics.jsonl`, 100 steps, lr 1e-4).
- **Disconfirming logic:** 100% of the published CSD evidence is simulator output
  (`synthesize_rows()` reproduces `variance_mitigation.tsv` byte-identically); Arm A
  (already executed, read-only) shows the simulator's timescale off ~20×, detrending
  erases its "CSD" signal, and τ=0.4 false-fires at step 2 of a healthy run. Any single
  failure of the three survival conditions executes the pre-registered demotion of the
  CSD claim to "untested hypothesis" and reframes ZVF as a saturation/compute-waste
  diagnostic. Passing all three at n=1 only earns "preliminary real-data support".
- **Config (within caps — the only new spend):**
  `tinker_cookbook.recipes.math_rl.train`, verifiable arithmetic, Llama-3.2-1B (same
  family as the safe trace), G=4, groups_per_batch=100, **lr = 1e-3 (10× safe)**,
  **40 steps**, fixed pool 480 train + 120 held-out = 600 prompts. Per step: log
  ZVF_t = `frac_all_good + frac_all_bad`; sample the 120 held-out prompts. Sampler
  checkpoints every 10 steps (pattern: `arithmetic_checkpoints.jsonl`). ~15–30 min
  wall-clock + ~1 h CPU analysis.
- **Expected artifact:** run log (ZVF_t, held-out acc_t), external-collapse step s_c,
  matched-window lag-1 stats vs the safe trace, τ∈{0.4–0.7} lead-time grid with k=5
  persistence + safe-run false-alarm check; suggested
  `experiments/results/sd2_zvf_csd_destab/`.
- **Revision trigger (pre-registered, any one fires demotion):** T-a no external collapse
  (held-out acc ≥20% rel. below running peak for ≥5 steps) within 40 steps at 10× lr;
  T-b/c pre-collapse detrended lag-1 ≤ μ_safe + 2σ_safe or ≤ matched-window safe max;
  T-d no τ gives a persistent crossing ≥10 steps before s_c without a persistent safe-run
  false alarm. **Every branch:** the synthetic re-labeling of all
  `variance_mitigation.tsv`-derived numbers and `tab:zvf-by-library` ("simulation
  projection") executes unconditionally, now.
- **Dedup vs plan:** N1 (prospective CSD validation, 10 seeds, deferred) stays deferred
  and is launched only if SD2 passes all three conditions. The launched **N2**
  (tensor-instrumented runs) de-synthetizes the variance-mitigation table but produces no
  collapse trace, so it does not cover SD2.

---

## SD3 — P3: Measured token-matched small-G vs large-G equivalence test

**Status: Stage 0 NEW-ANALYSIS (launch now, zero compute); Stage 1 NEW but
**G-cap-blocked** — requires the same large-G exception as plan items N7/B1 (not a
duplicate of either: N7 tests the native Wu claim at G=2 vs G=16, B1 is the 2×2
super-group design; SD3 Stage 1 tests token-normalized equivalence at fixed T=4M).**

- **Goal:** Test whether token-normalized equivalence G=4 ≈ G=32 survives on a
  *measured*, token-matched pair at T = 4M optimizer-visible tokens/arm, capable model,
  mid-difficulty (p0 ∈ [0.3, 0.6]) — the smallest budget where the paper's own
  reconstruction (`FALLBACK_ROWS`) predicts non-equivalence (Δ = +0.11). Same run reads
  out the mechanism (contrast starvation GU/ZVF vs gradient noise σ²_R/G).
- **Disconfirming logic:** Every version of the equivalence claim rests on a synthetic
  grid that itself contradicts equivalence at every T ≥ 4M, and iter115 falsifies the
  preference-density mechanism in that regime. A measured cell must fire exactly one of:
  equivalence refuted (R1), mechanism refuted (R2), or reconstruction refuted → withdraw
  the FALLBACK_ROWS artifacts (R3). Stage 0 can kill equivalence for free within the
  already-measured G range.
- **Config:**
  - **Stage 0 (re-analysis, within caps, runnable today):** re-slice
    `experiments/results/groupsize_zvf_sweep.json` (Qwen2.5-0.5B, arithmetic,
    G∈{2,4,8,16}, seeds {42,123,456}, 40 steps) into token-matched pairs G=4@step 4s vs
    G=16@step s, window where G=16 train mean_reward ∈ [0.3, 0.8] (s∈{2..6}; seed 456
    {2..7}); ≈16 paired points, sub-windows W1 = s∈{2,3,4}, W2 = s∈{5,6(,7)}; statistic:
    per-window mean paired d = reward(G=16@s) − reward(G=4@4s), sign consistency.
  - **Stage 1 (as-designed: gated on Stage 0 pass AND a G>8 exception):** Qwen/Qwen3.5-4B
    (fallback Qwen3-8B-Base); calibration k=4 @ T=1.0 on 600 GSM8K train prompts → keep
    300 with per-prompt pass ∈ [0.25, 0.6]; eval = 300 fixed held-out GSM8K prompts,
    greedy; arms G=4 (≈84 prompts/step) vs G=32 (≈10 prompts/step), both ≤40 steps,
    stop at cumulative 4M ± 5% tokens; seeds {42,123,456} → 6 runs ≈ 24M sampled tokens;
    metric Δ = acc(G=32) − acc(G=4), 900 prompt-level pairs, 10k paired bootstrap +
    TOST ε∈{0.024, 0.05}; free telemetry: per-step advantage variance, ZVF, GU
    (iter115 schema).
  - **Within-cap fallback if the exception is denied:** run the same design as G=4 vs
    G=8 token-matched (reduced contrast; reconstruction predicts a smaller Δ — reduced
    decisiveness, R2 mechanism readout still valid).
- **Expected artifact:** Stage 0 paired-difference table; Stage 1 Δ/CI/TOST table + GU
  ratio + ZVF trajectories; suggested `experiments/results/sd3_g4_vs_g32_tokmatch/`.
- **Revision trigger (first firing rule decides):** R0 (Stage 0): |d| > 0.02,
  sign-consistent in ≥2/2 sub-windows across 3 seeds → equivalence fails within measured
  G range; do NOT run Stage 1 as designed (re-scope to T_equiv bisection). R1: Δ CI lower
  bound > +0.024 → demote to "budget-conditional, under-trained only (T ≤ ~1M)".
  R2: GU(G=4)/GU(G=32) ≥ 2 while retention < 90% → preference-density mechanism demoted
  to near-ceiling regime. R3: TOST@0.05 passes → withdraw/re-derive FALLBACK_ROWS and all
  downstream artifacts (iter107/111/115/135 curve, TOST table, T* extrapolations).
  R4 inconclusive: no new runs; re-eval existing checkpoints on full GSM8K test split,
  re-apply once, else downgrade to "not distinguishable from a ≤5-point gap at T=4M".
  Validity preconditions (p0 band, T ± 5%, ≥35/40 steps, fixed eval set) void the
  readout without firing a trigger.
- **Dedup vs plan:** Not a duplicate of N7/B1/N9 (different question), but Stage 1 shares
  their G>8 launch blocker — file one combined exception request for the
  SD3-Stage-1 / N7 / B1 batch. Stage 0 has no plan counterpart (B2 in the plan is the
  difficulty-stratified re-analysis blocked on N8; unrelated slice).

---

## SD4 — P4: Joint-permutation null + truncation-retention sweep (GRPO vs Dr.GRPO)

**Status: Stage 0 NEW-ANALYSIS (CPU-minutes, run today). Stage 1 **MERGED-WITH plan item
A3** ("Length-adversarial truncation test", listed NEW-RUN deferred pending a converged
Dr.GRPO checkpoint) — SD4 Stage 1 *is* A3, and resolves its blocker: no checkpoints were
ever saved anywhere (the Modal script wrote JSON metrics only), so the arms must be
reconstituted with weight-saving; do not wait on checkpoint confirmation.**

- **Goal:** (a) Does the iter136 8/8 sign-test headline (p = 0.0039) survive a
  dependence-respecting null that flips the GRPO/Dr.GRPO label jointly per seed?
  (b) Does GRPO ≈ Dr.GRPO practical equivalence survive constrained decoding, or is
  GRPO's held-out accuracy length-mediated?
- **Disconfirming logic:** The p = 0.0039 treats 8 (task × hypothesis) cells sharing
  seeds/trajectories as independent (effective draws ≈ 2 tasks; 7/8 cells individually
  null) — the exhaustive 256-assignment permutation gives the exact honest p. The
  equivalence clause was never formally tested and only at length-marginalized altitude;
  a retention-vs-cap sweep is the behavioral test it cannot dodge. Either stage can
  demote its clause; jointly they decide whether P4 leads with equivalence or coupling.
- **Config (within caps — G=8, 30 steps, 440 prompts/run, <3M tokens total):**
  - **Stage 0 (re-analysis):** recompute iter136 H1–H4 per run from
    `drgrpo_gsm8k_cot_full.json` (3 seeds × 2 algos) + `drgrpo_vs_grpo.json`
    (5 seeds × 2 algos) per `scripts/length_bias_iter136.py`; exhaustive 2⁵ × 2³ = 256
    joint sign-flip assignments; statistic K = #cells in pre-registered direction
    (observed 8/8); exact one-sided p.
  - **Stage 1 (Tinker reconstitution + sweep):** Qwen/Qwen2.5-1.5B-Instruct, LoRA r=16,
    30 steps, 8 prompts/step × G=8, K=2 clipped epochs (ε=0.2), lr 1e-5, max_new=200;
    GRPO = std-normalized vs Dr.GRPO = centered-only with fixed 1/(B·MAX_NEW) normalizer;
    seeds {42,123,456} × 2 algos = 6 arms; **one change: `save_weights_for_sampler` at
    final step** (fallback: original Modal script + `save_pretrained`). Replication gate:
    ±3 pp of stored post acc (GR .263 / DR .255), ±15 tok of stored lengths. Sweep: one
    greedy 512-cap pass per (arm × 200 held-out items) = 1,200 generations (~0.6M tok);
    score prefix-truncations at T ∈ {64, 96, 128, 160, 192, 256, 512} (greedy
    prefix-stability; 20-item spot-check at T=128 must show 0 mismatches). Metric:
    ret_A(T) = Acc_A(T)/Acc_A(512); Δret(T) = ret_DR − ret_GR; 10k paired-seed bootstrap
    + pooled McNemar (600 item pairs/cap).
  - **Batching note:** launch as one 12-arm batch with the P4 cap-512 censoring ablation
    (identical script, `MAX_NEW` 200 vs 512) — combined <~10 GPU-h / ~9M tokens, retires
    both P4 validity threats in one run (see ACTIONS.md rank 10).
- **Expected artifact:** `experiments/results/length_bias_iter140_joint_perm.tsv` (exact
  joint p) + retention curves/Δret tables per cap + saved sampler weights; suggested run
  dir `experiments/results/sd4_truncation_retention/`.
- **Revision trigger (pre-registered):** R0: joint p ≥ 0.05 → downgrade "GLOBAL_REJECT,
  p = 0.0039" to "directionally consistent (8/8), not significant under a
  structure-respecting null"; p < 0.05 → replace the headline p with the joint p.
  R1: at any T ∈ {64–160}, mean Δret ≥ +10 pp with CI excluding 0 (or McNemar p < 0.05)
  → rescope to "equivalent only at unconstrained decoding" (the null becomes evidence
  *for* length-mediated success under GRPO). R2: |Δret| ≤ 3 pp at every cap down to 64 →
  equivalence behaviorally confirmed; coupling reworded "statistically detectable but
  behaviorally inert at this horizon". R3 grey zone: no further compute; pre-registered
  hedged wording with the measured CI. Replication-gate failure voids the sweep (fires
  no trigger; fall back to Modal backend).
- **Dedup vs plan:** Stage 1 = **MERGED-WITH A3** (supersedes its "confirm checkpoint"
  precondition — confirmed impossible; reconstitution is the design). Stage 0 has no
  plan counterpart. A4 (CLMP length-mediation re-analysis) and N10 (seed expansion)
  remain separate and are unaffected.

---

## Queue summary

| id | paper | status vs NEW_EXPERIMENTS_PLAN | launchable now | new compute |
|----|-------|-------------------------------|----------------|-------------|
| SD1 | P1 | NEW (Stage B doubles as partial N5; cross-check launched N8) | yes (all stages) | 0 train steps; ≤600 sampled completions |
| SD2 | P2 | NEW (gates deferred N1) | yes | 1 run × 40 steps, 1B model |
| SD3 | P3 | NEW (Stage 1 shares N7/B1's G>8 exception; not a duplicate) | Stage 0 yes; Stage 1 gated | Stage 1: 24M tokens (or within-cap G=4-vs-8 fallback) |
| SD4 | P4 | Stage 1 **MERGED-WITH A3**; Stage 0 NEW | yes (batch with cap-512 ablation) | <3M tokens (≈9M with the 12-arm batch) |

Recommended launch order: SD4 Stage 0 + SD3 Stage 0 + SD1 Stage A (all zero-compute,
today) → SD1 Stages B/C + SD2 + SD4 Stage 1 batch (small spends) → SD3 Stage 1 only if
its Stage 0 passes and the large-G exception is granted.
