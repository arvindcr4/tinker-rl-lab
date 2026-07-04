# P2 (ZVF) — Synthesis Review

Synthesized 2026-07-04 from the four contract outputs in this directory:
`P2_stress_test.md`, `P2_ablation_gap.md`, `P2_claim_lint.md`, `P2_decisive_experiment.md`.
Paper: `paper/paper_P2_zvf.tex` — "The Zero-Variance Fraction: A Descriptive Diagnostic
for Signal Starvation in GRPO".

---

## 1) Verdict on the central claim (5 sentences)

The central claim splits into two halves, and only one is robust. The
critical-slowing-down early-warning half is **not**: every trajectory-level number
behind it (0.609 vs 0.415 lag-1, d=7.3, 19–39-step leads, `tab:zvf-by-library`) is
byte-identical dry-run simulator output (`synthesize_rows()` reproduces
`variance_mitigation.tsv` 100/100 rows) with quota-assigned, ZVF-defined collapse
labels, the gap vanishes under length-matched windows, and the repo's one *real*
trajectory shows the τ=0.4 alarm false-firing at step 2 of a healthy run that ends at
100% accuracy. The descriptive-diagnostic half survives, but only narrowly (strictly
binary rewards + fixed G + one stack): the negative results (mastery/incapacity
aliasing, 1e-4-jitter flatline, ρ≈0.27 to held-out outcome) all reproduce from real
GSM8K tensors, while the flagship positive number — risk-index AUROC 0.929 — is tuned
and scored on the same 52 rows, has no reward- or entropy-trace comparator despite
ZVF correlating with both at |r| ≈ 0.8–0.9, and is contradicted by an in-repo but
unpublished held-out check (`zvf_iter134_heldout.tsv`: 100% false positives on the
real converged Qwen3-8B/GSM8K runs). Claim-level hygiene is mid: 34/53 claims
reproduce exactly, but 7 are contradicted by their own cited artifacts, worst being
the "5–15 step lead" sentence whose cited file says lead = 1 step. Net: the paper is
salvageable as "cheap saturation diagnostic + honest negative results" after
mandatory re-labeling, but the early-warning and cross-library headline claims must
be demoted now and can only be re-earned on real data.

## 2) Top 3 prioritized actions

### Action 1 — FIX PROSE (mandatory, unconditional): re-label all simulator-derived numbers as synthetic projections and correct the 7 unsupported claims
- **Type:** fix prose. **Effort: ~1 day, zero compute.** Forced NOW by stress-test
  findings D0/D1 regardless of any future experiment outcome.
- **What:** (a) Re-mark every number sourced from
  `experiments/results/variance_mitigation.tsv` as "synthetic/dry-run projection" or
  delete: iter126 H1/H2/H3, `tab:zvf-by-library`, the zvf-dynamics pooled table, and
  the iter130 risk-index rows built on those 45 trajectories. (b) Apply the linter's
  ready-made rewrites for the 7 unsupported claims (exact replacement text in
  `P2_claim_lint.md` §3).
- **Files/sections touched:**
  - `paper/sections/zvf_iter126.tex` — H1/H2 CSD numbers → synthetic label or delete.
  - `paper/sections/zvf.tex` — R5 (residual β₂ sign contradicted by
    `zvf_iter26_residual.tsv`: actual +0.80), R6 (delete "5–15 rollout steps"; cited
    `zvf_leadtime_summary.tsv` says 1 step), R8 (regenerate iter130 method-ranking
    table from `zvf_iter130_method_risk.tsv`, restore MCGRPO row),
    `tab:zvf-by-library` caption → "simulation projection".
  - `paper/sections/zvf_cross_experiment_diagnostic.tex` — R10/R16 stale
    counts/medians vs regenerated `zvf_summary.tsv` (7/5/41/27, plateau median 0.2219).
  - `paper/sections/zvf_scaling.tex` — R23b ("+1.0 by construction" → 0.79).
  - `paper/sections/zvf_iter34.tex` — R28b (p = 0.195, not significant).
  - `paper/sections/zvf_iter62.tex` — R37b ("all nine" → eight of nine; SCAFGRPO 0.235).
  - `paper/sections/p2_abstract.tex` + `p2_results_intro.tex` — A4 early-phase range
    scope fix (G=2/4/8 start at 0.25–0.61, not 0.04–0.13).
  - Easy win while there: cite `pcd_vs_zvf_summary.tsv` for the jitter flatline (C4),
    converting a thought experiment into a measured result.

### Action 2 — ADD ABLATION (claim-load-bearing): reward/entropy-trace risk-index baseline + leave-one-method-out CV + splice in the held-out check
- **Type:** add ablation (pure re-analysis). **Effort: ~1 day, one stdlib script,
  zero GPU / zero new training.**
- **What:** Compute the identical three alarm channels (level, rolling lag-1 w=15,
  drift slope) from the per-step `reward_mean` trace (all 45+52-row panels;
  `variance_mitigation.tsv`) and the `entropy` trace (12 G-sweep cells;
  `groupsize_zvf_sweep.json` `step_log`), max-fuse identically, and report
  Mann-Whitney AUROC + B=2000 bootstrap CI side-by-side with `zvf_risk_max` — under
  leave-one-method-out CV for both indices. Splice `zvf_iter134_heldout.tsv`
  (including the tinker_gsm8k false positives) into the paper as the transfer check.
  Pre-commit the decision rule: ZVF keeps its claimed value iff
  LOMO-AUROC(ZVF) − LOMO-AUROC(reward) > 0 with CI excluding 0.
- **Files/sections touched:** new `scripts/zvf_baseline_alarm.py` (pattern:
  `scripts/pcd_vs_zvf.py`); `paper/sections/zvf.tex` iter130 subsection (AUROC table
  + thresholds 0.30/0.55); `paper/sections/p2_conclusion.tex` ("report ZVF alongside
  mean reward", "earliest cheap diagnostic" sentences live or die on this result).
- **Why #2:** this is the reviewers' most probable rejection argument — the paper's
  entire positive contribution is incremental value over signals every dashboard
  already logs, and the repo's own data (Spearman(mean_reward, outcome)=+0.95 vs
  ZVF's +0.56) points the wrong way.

### Action 3 — RUN EXPERIMENT: the one destabilized real run that decides the CSD claim's fate (Arm B of the decisive experiment)
- **Type:** run experiment. **Effort: one Tinker run, 40 training steps,
  ≈15–30 min wall-clock + ~1 h CPU analysis (Arm A re-analysis already executed).**
- **What:** see §3 below — full launcher-ready spec.
- **Files/sections touched:** new run under `experiments/` (reuse
  `experiments/run_tinker.sh` pattern); outcome rewrites
  `paper/sections/zvf_iter126.tex` and the collapse language in
  `paper/sections/zvf_dynamics.tex` (either "preliminary real-data support" or
  demotion to "untested hypothesis" + reframe as saturation/compute-waste diagnostic).
- **Why #3:** decisive but conditional — Actions 1–2 execute unconditionally;
  this run determines only which of two pre-written endings the CSD section gets.

## 3) The single decisive experiment (launcher-ready)

**Question.** On a real GRPO run of the paper's verifiable-arithmetic task, does
detrended matched-window rolling lag-1 autocorrelation (w=15) of the per-step ZVF
trace rise before an *externally defined* collapse, with an actionable (≥10-step)
τ-crossing lead — versus the existing real safe run?

**Launch config (Arm B — the only new spend):**
- Recipe: `tinker_cookbook.recipes.math_rl.train` (same as run id `39aa5eb2-…:train:0`).
- Model: Llama-3.2-1B (same family as the existing safe trace, so safe-vs-collapse is
  a within-model contrast). Task: verifiable arithmetic.
- G=4, groups_per_batch=100, **lr=1e-3** (10× the safe run's 1e-4 — the cheapest
  destabilizer), **40 training steps**.
- Prompts: fixed pool of 480 training + 120 held-out = 600.
- Logging per step: ZVF_t = `env/all/by_group/frac_all_good` +
  `frac_all_bad`; held-out accuracy by sampling the 120 held-out prompts
  (sampling-only). Checkpoint sampler weights every 10 steps
  (pattern: `experiments/results/arithmetic_checkpoints.jsonl`).
- Reference safe trace (no cost, already on disk):
  `experiments/results/arithmetic_metrics.jsonl` (100 steps, lr=1e-4).

**Pre-registered collapse rule (external — never via ZVF):** held-out accuracy falls
≥20% relative below its running peak and stays there ≥5 consecutive steps; s_c = first
step of that streak.

**Analysis:** mean detrended rolling lag-1 (w=15) on [0, s_c) of Arm B vs the
distribution over all length-s_c contiguous windows of the safe trace
(μ_safe, σ_safe, max); τ-crossing lead times for τ ∈ {0.4, 0.5, 0.6, 0.7} with k=5
persistence, plus false-alarm check on the safe run.

**Decision rule (any single failure ⇒ pre-registered demotion executes):**
- **T-a:** no external collapse within 40 steps at 10× lr → demote CSD claim to
  "untested hypothesis" immediately; schedule no further runs.
- **T-b/c:** collapse occurs but Arm-B pre-collapse detrended lag-1 ≤ μ_safe + 2σ_safe,
  or ≤ the matched-window safe maximum (band overlap) → demote.
- **T-d:** no τ ∈ {0.4–0.7} gives a k=5-persistent crossing ≥10 steps before s_c with
  zero persistent false alarm on matched early safe windows → demote.
- **All pass:** claim survives only as "preliminary real-data support in one
  destabilized run"; full 5-seed D2 (150+ steps/seed) becomes justified follow-up.
- **Every branch:** the Action-1 synthetic re-labeling executes regardless, and
  `tab:zvf-by-library` stays labeled "simulation projection" (AERO is not
  implemented on this stack, so it cannot be regenerated at any budget here).

**Already-executed Arm A context the launcher inherits:** the real safe run shows no
collapse (max drawdown 1.75%), ZVF>0.4 at step 2 (vs simulator's step 43–46 — 20×
timescale mismatch), raw lag-1 0.045 (vs simulator 0.38–0.46), and linear detrending
erases the "CSD" signal inside the simulator itself. Reproduction snippet:
`P2_decisive_experiment.md` appendix (~1 min, read-only).

---

## Cross-document consistency notes

- All four documents agree the negative/descriptive core (aliasing, jitter flatline,
  ρ≈0.27, sticky drift) reproduces from real artifacts and survives.
- All four agree the CSD/early-warning and cross-library headline numbers trace to
  the same synthetic TSV; no document found any real per-step ZVF trajectory beyond
  `arithmetic_metrics.jsonl`.
- The linter's R5/R27 finding (residual sign +0.80, contradicting `zvf.tex`'s
  "negative β₂") and the ablation report's point 4 independently flag the same
  incremental-validity soft spot that Action 2 tests head-on.
