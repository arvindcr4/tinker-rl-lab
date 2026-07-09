# P4 REVIEW — Synthesis of the Four Contract Outputs (Length Bias / GRPO vs Dr.GRPO)

Synthesized 2026-07-04 from `P4_stress_test.md`, `P4_ablation_gap.md`, `P4_claim_lint.md`,
`P4_decisive_experiment.md`. Paper: `paper/paper_P4_length_bias.tex` + `paper/sections/p4_*.tex`,
`length_bias*.tex`, `frontier_synthesis_length_bias.tex`.

Central claim under review: *in short-horizon GSM8K-CoT, GRPO and Dr.GRPO are practically
equivalent (similar held-out gains, no length inflation, no verbosity-trap signature), while
step-level length–reward coupling still differs in the predicted direction (8/8 cells,
sign-test p = 0.0039).*

---

## 1) Verdict on the central claim's robustness

The claim's numerical core is solid — 32 of 49 audited claims reproduce exactly from
`experiments/results/`, including the 0/16 trap flags and all-negative length trends — but the
claim as stated is **not robust**. Its most serious defect is undisclosed right-censoring: the
entire GSM8K-CoT cell was trained *and* evaluated under a 200-token generation cap that the
pre-RL policy already saturates (step-0 mean length 189–196 of 200, `modal_drgrpo_gsm8k_cot.py:45`),
so "neither inflates length" is true by construction and the negative length–reward coupling
that drives ~30 analysis iterations is exactly what censoring predicts mechanically. The
equivalence clause ("statistically similar held-out gains") was never formally tested and rests
solely on length-marginalized statistics at n = 3 seeds — the one measurement altitude
insensitive to the length-mediated-success confound the paper itself concedes in
`p4_conclusion.tex`. The coupling clause's headline p = 0.0039 treats 8 (hypothesis × task)
cells as independent when they share seeds and trajectories (effective independent draws ≈ 2
tasks; 7/8 cells individually null), and four quoted statistics (the iter120 citation, the
`tab:reward-shape` Theil–Sen column, the iter32 "CI non-overlapping" assertion, and two iter128
details) are contradicted by or absent from the result files. Net verdict: the scoped null is
plausibly correct but currently unfalsifiable as framed — it becomes defensible only after the
cap ablation (Action 1), the prose corrections (Action 2), and the decisive experiment (§3).

---

## 2) Top 3 prioritized actions

### Action 1 — ADD ABLATION (run it first): uncensored generation cap, `MAX_NEW` 200 → 512

**The #1 action.** The 200-token cap is the only *undisclosed, validity-level* weakness — a
reviewer finds it in one line of released code, it converts the paper's negative-control framing
into circularity ("length is already controlled" — by the authors' sampler config), and the
paper applies exactly this criticism to its own 100-step arithmetic run ("the trap is
mechanically impossible") while never applying it to the GSM8K cell. Every other known weakness
is disclosed and scoped; this one is reject-bait. Either outcome converts the confound into a
result: if length still stays ~185–195 tokens with 2.5× headroom, the null becomes
reviewer-proof; if GRPO drifts upward and Dr.GRPO attenuates it, the headline inverts and must
be rewritten before submission, not after.

- **Effort:** one-line config change (`MAX_NEW: 200 → 512`); 6 runs (Qwen2.5-1.5B-Instruct,
  30 steps, seeds {42, 123, 456} × {GRPO, Dr.GRPO}); ~35 min/run → **~3.5–4 GPU-hours total**
  on the existing Modal A10G setup. Add two log fields: per-completion lengths and per-step
  fraction-at-cap. `scripts/length_bias*.py` rerun unchanged on the new traces. Optional
  strengthening at ~3× cost: 100 steps (~12 GPU-h), which also closes most of the horizon
  objection.
- **Touches:** `experiments/modal/modal_drgrpo_gsm8k_cot.py` (line 45); new
  `experiments/results/` artifacts (cap-512 arm of every headline table);
  `paper/sections/length_bias.tex` (new cap-ablation subsection re-emitting fraction-at-cap,
  ρ triples, trap flag, decile E[R|L], paired deltas, held-out McNemar);
  `paper/sections/p4_abstract.tex` + `p4_results_intro.tex` if numbers move.

### Action 2 — FIX PROSE (zero compute, do in parallel): disclose the cap; correct the 4 unsupported claims; rescope the overbroad ones

**Effort: 2–4 hours, no compute** (plus one CPU-minutes script if adding the formal held-out test).

1. **Disclose the cap now** (independently of Action 1's outcome): state `MAX_NEW = 200` and
   step-0 saturation in the GSM8K paragraphs of `sections/length_bias.tex`, and scope the
   abstract/results-intro "neither inflates length (≈193 → ≈188)" claim as cap-bounded —
   `sections/p4_abstract.tex`, `sections/p4_results_intro.tex`. Also replace "≈193 → ≈188"
   with a pair citable from one artifact (pooled 194.4 → 183.5, or Dr.GRPO 193.9 → 186.8).
2. **Fix the four unsupported (U) claims** flagged by the lint:
   - iter120 citation ("pooled ρ = +0.556, p < 10⁻³, 32 points") in
     `sections/length_bias_iter132.tex` **and** `length_bias_iter136.tex` → actual file gives
     ρ = 0.50 (n = 20, p = 0.025, arith) and ρ = 0.56 (n = 12, p ≈ 0.06, GSM8K); no pooled stat exists.
   - `tab:reward-shape` Theil–Sen slope column in `sections/length_bias.tex` → replace with the
     `curve_stats` values (−0.117 / −0.107 / −0.0076 / −0.0127) already used in the figure caption.
   - iter32 "CI non-overlapping" (CV reduction) → CIs overlap ([0.0233, 0.0519] vs
     [0.0216, 0.0285]); downgrade to directional.
   - iter128 details → pair count is 32 not 40; Dr.GRPO ΔL spans 4.5–8.9 tokens, not "−7 to −9".
3. **Rescope the weakly-supported (W) claims** per the lint's rewrite list: "statistically
   similar held-out gains" → "indistinguishable at our seed budget" (or better, run the
   one-script paired test / TOST on the six `dacc_heldout` values in
   `length_bias_iter128_efficiency_frontier.tsv`); P5 "monotonic" → "decreasing with one local
   exception (decile 4)"; I44 CI re-quote + one-sided framing; I68 scope to GSM8K-only; I72b add
   ns qualifier; C1 in `sections/p4_conclusion.tex` scoped to the near-ceiling Qwen3-8B eval
   (the 1.5B cell gains on 3/3 seeds); D1 name which comparison is single-seed. Also remove the
   dangling `\ref`s to iter108/120/124 sections not `\input` in `paper_P4_length_bias.tex`.

### Action 3 — RUN EXPERIMENT: the two-stage decisive experiment (§3 below)

Stage 0 (joint sign-flip permutation) is **CPU-minutes, zero new compute, runnable today** and
by itself decides whether the p = 0.0039 headline survives an honest null. Stage 1 (truncation-
retention sweep) is < 3M sampled tokens (~4 GPU-h equivalent) and behaviorally tests the
equivalence clause the marginal statistics cannot — note **no checkpoints were saved**, so it
requires reconstituting the 6 arms with `save_weights_for_sampler` added.

- **Effort:** Stage 0 today; Stage 1 one launcher batch (can share infrastructure with Action 1
  — see launcher note in §3).
- **Touches:** new `scripts/length_bias_iter140_joint_perm.py` +
  `experiments/results/length_bias_iter140_joint_perm.tsv`;
  `sections/length_bias_iter136.tex` (replace p = 0.0039 with the exact joint p);
  `sections/p4_conclusion.tex` and `frontier_synthesis_length_bias.tex` (the A3 truncation test
  stops being "the single highest-value follow-up ... never run").

---

## 3) The decisive experiment (launcher-ready)

**Objective.** Test both clauses of the central claim under nulls/probes they can actually fail:
(a) does GRPO ≈ Dr.GRPO equivalence survive constrained decoding, or is GRPO's held-out accuracy
length-mediated; (b) does the 8/8 sign-test survive a dependence-respecting permutation.

**Resource envelope.** Tinker API (fallback: original Modal script), model ≤ 8B, ≤ 30 training
steps/run, ≤ 440 unique prompts/run, total < 3M sampled tokens. Stage 0 is CPU-only.

### Stage 0 — joint sign-flip permutation (run first; zero compute; decides clause b alone)

- **Inputs:** `experiments/results/drgrpo_gsm8k_cot_full.json` (3 seeds × 2 algos, 30-step logs),
  `drgrpo_vs_grpo.json` (5 seeds × 2 algos, 40-step logs); statistic definitions in
  `scripts/length_bias_iter136.py`.
- **Procedure:** recompute the four iter136 statistics (H1–H4) per run; for the null, flip the
  GRPO/Dr.GRPO label **jointly per seed** (the flip swaps that seed's pair for all four
  hypotheses within its task). Exhaustive: 2⁵ × 2³ = 256 assignments, exact p, no MC error.
- **Statistic:** K = number of the 8 (task × hypothesis) cells with mean paired delta in the
  pre-registered direction (observed K = 8). Output exact one-sided p = #{K′ ≥ 8}/256 to
  `experiments/results/length_bias_iter140_joint_perm.tsv`.
- **Decision R0:** p ≥ 0.05 → downgrade "GLOBAL_REJECT, p = 0.0039" to "directionally consistent
  (8/8), not significant under a structure-respecting null (p = k/256)". p < 0.05 → clause
  stands; report the joint p in place of (1/2)⁸.

### Stage 1 — reconstitution + single-pass truncation-retention sweep (~3M tokens)

- **Reconstitution (required — no checkpoints exist; original script saved only JSON metrics):**
  re-run the exact regime of `experiments/modal/modal_drgrpo_gsm8k_cot.py` on Tinker:
  Qwen/Qwen2.5-1.5B-Instruct, LoRA r=16, 30 steps, 8 prompts/step × G=8, K=2 clipped epochs
  (ε=0.2), lr 1e-5, `max_new=200`; GRPO = std-normalized advantages, Dr.GRPO = centered-only
  with fixed 1/(B·MAX_NEW) normalizer; seeds {42,123,456} × {grpo, dr_grpo} = 6 arms.
  **One change: call `save_weights_for_sampler` at the final step.** If the model is not served
  by Tinker, fall back to the original Modal script with adapter-saving added.
- **Replication gate (validity precondition):** each retrained algo-mean within ±3 pp of stored
  post accuracy (GR .263, DR .255) at cap 200, and mean completion length within ±15 tokens of
  stored last-5 means (~180–187). Violation voids the sweep (fallback: Modal rerun, same sweep
  offline on saved adapters).
- **Sweep (sampling-only, exploits greedy prefix-stability):** one greedy pass per (arm, item)
  at T_max = 512 on the **identical 200 held-out items** (`gsm8k test.shuffle(seed=0)[:200]`),
  storing token IDs → 1,200 generations (~0.6M tokens). For each cap
  T ∈ {64, 96, 128, 160, 192, 256, 512}: truncate to the first T tokens, decode, re-run the
  `\boxed{}`/last-number extractor, score. Spot-check prefix-stability by truly generating 20
  items at T = 128 (must be 0 mismatches). Require Acc(512) ≥ 0.15 for every arm.
- **Primary metric:** retention ret_A(T) = Acc_A(T)/Acc_A(512) per (algo, seed); paired
  Δret(T) = ret_DR(T) − ret_GR(T). Inference: 10k paired-seed bootstrap on mean Δret(T) +
  confirmatory pooled McNemar (600 item pairs per cap).
- **Decision rules (pre-registered):**
  - **R1 (equivalence fails):** at any T ∈ {64, 96, 128, 160}, mean Δret(T) ≥ +10 pp with the
    bootstrap 95% CI excluding 0 (or pooled McNemar p < 0.05, same direction) → rescope the claim
    to "practically equivalent only at unconstrained decoding"; the null becomes evidence *for*
    length-mediated success under GRPO.
  - **R2 (strengthening):** |mean Δret(T)| ≤ 3 pp at every cap down to 64 → equivalence
    behaviorally confirmed; reword the coupling clause to "statistically detectable but
    behaviorally inert at this horizon".
  - **R3 (grey zone, 3–10 pp or unstable sign):** no further compute; amend the claim to
    "practically equivalent at the training-time decoding budget (T_max = 200); retention under
    tighter caps showed a non-significant gap of X pp (95% CI [a,b]) and is left unresolved".
- **Joint readout:** Stage 0 ns + R2 → pillar contribution is descriptive; lead P4 with the
  equivalence result. Stage 0 sig + R1 → strongest outcome; promote the truncation curve to a
  main figure and the joint-permutation p to the headline statistic.

**Launcher note (batching with Action 1).** Stage 1's reconstitution and Action 1's cap ablation
use the same script and differ only in `MAX_NEW` (200 vs 512) — launch as one 12-arm batch:
6 arms @ cap 200 with weight-saving (replication gate + truncation sweep) and 6 arms @ cap 512
(the censoring ablation). Combined cost stays under ~10 GPU-hours / ~9M sampled tokens and
retires the paper's two biggest threats in a single run.
