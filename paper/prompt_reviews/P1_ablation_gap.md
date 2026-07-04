# P1 (Scaling Laws) — Ablation Gap Review

Executed per `research_prompts/revision/ablation-gap-finder.md` (pre-submission ablation
reviewer contract) against `paper/paper_P1_scaling.tex` and its `\input` sections.
Date: 2026-07-04. All file paths are repo-relative to `/home/claude/tinker-rl-lab-minimax/`.

## Inputs (placeholder fill)

- **Main claim** (`sections/p1_abstract.tex:9-13`): *"across five orders of magnitude in
  parameter count, the cross-scale slope of the GRPO reward gain is statistically
  indistinguishable from zero, and no single saturation exponent is identifiable"* — the
  defensible object is a local, stack-conditioned saturation law, not a power law in N.
- **Method components**: (1) per-trace saturation fit R(t)=R_max(1−e^{−λt}) on 5 frontier
  anchors (T ≤ 30 steps, GSM8K, Tinker API, single seed); (2) cross-scale OLS of per-trace
  statistics on log10 N (n=5, extended to n=12); (3) pre-registered three-phase
  falsification battery; (4) identifiability/AIC/holdout audits; (5) Nemotron-120B collapse
  autopsy; (6) same-stack PPO-vs-GRPO control (Qwen2.5-0.5B, 5 seeds).
- **Current ablations** (verified in `experiments/results/` and the P1 build): functional-form
  ablation (iter17/37: constant/linear/saturation/logistic/Hill), anchor-pool ablation
  (n=5 → n=12, iter13), parametric vs rank tests (Spearman backup), MoE-vs-dense
  stratification (iter13/21/45), compute-proxy abscissa (iter45/49), truncation stability
  (iter41), changepoint + permutation nulls (iter17), leave-one-out residuals (iter49/53),
  ZVF-conditioning (iter61), phase-conformity geometry (iter65), same-stack estimator
  control (iter29).

---

## 1) Missing ablation

**A headroom-controlled measurement of the actual GRPO *gain* (Δ held-out, post − pre) at
more than one model scale** — i.e., the positive control that shows the benchmark could
detect a nonzero gain slope if one existed.

The headline claim is about the scaling of the GRPO reward **gain**, but no experiment in
the P1 build (or anywhere in `experiments/results/`) measures the gain across scale:

- The headline regression `experiments/results/scaling_law_cross_scale.tsv` regresses
  **absolute levels** (`mean_reward`, `peak`, `var_reward`, `R_max`, `t_80`) on log10 N
  (n=5). No Δ-over-base metric appears in it or in the n=12 extension
  (`scaling_law_power_law.tsv`).
- Most anchors have **no headroom to express a gain**: R(1)=1.000 for Qwen3.5-4B and
  Llama-3.1-8B-Instruct, R(1)=0.875 for DeepSeek-V3.1 (`sections/scaling_laws.tex:70-77`);
  repo-wide, 7/12 anchors reach t50 at step 1 (`sections/scaling_law_iter109.tex:119`,
  section not in the P1 build). With traces starting at ceiling and T ≤ 30 steps, a flat
  cross-scale slope of the *gain* is guaranteed **by construction**, independent of whether
  GRPO benefits scale.
- The only true pre/post-RL held-out gain measurement in the entire results tree is at
  **exactly one scale point**: `experiments/results/base_instruct_paired.tsv` — Qwen3-8B
  Instruct, held-out 0.820 → 0.833 (Δ=+0.013, paired p=0.256, 5 seeds); Qwen3-8B-Base and
  Llama-3.1-8B rows are train-only, n=1; the 0.6B/1.7B rows are `source-missing`. One
  scale point cannot support any statement about a cross-scale slope of the gain.
- The repo's own synthetic calibration proves the current design has near-zero detection
  power: `experiments/results/scaling_law_iter121_synthetic_recovery.tsv` shows that with
  the observed noise, a **true** slope of 0.1/decade is recovered only 26% of the time at
  n=5 anchors and 4% at n=12; even 0.2/decade is recovered ≤ 40% of the time. (This
  calibration and the late−early gain-proxy regression, `scaling_law_iter121_late_early.tsv`,
  exist as artifacts but are **not** part of `paper_P1_scaling.tex`, which inputs only
  iters 29–65.)
- Conversely, `sections/scaling_law_iter29.tex` shows that when headroom *does* exist
  (Qwen2.5-0.5B, 40-step same-stack runs, mean R far from ceiling), AICc prefers the
  saturation form on **5/5 GRPO seeds** — learning is detectable in this benchmark when the
  policy is below ceiling. This is internal evidence that the frontier flatness is a
  ceiling artifact, not a property of GRPO.

## 2) Why reviewers will ask for it (highest-risk gap)

This gap sits directly under the paper's title claim, not under a secondary analysis. A
NeurIPS reviewer's likely wording:

> "The abstract claims the cross-scale slope of the *GRPO reward gain* is indistinguishable
> from zero over five orders of magnitude, but Table `tab:scaling-cross` regresses absolute
> training-reward levels, not gains, and four of five anchors start at or near their reward
> ceiling at step 1 with a ≤30-step budget. The measurement therefore has no sensitivity to
> the quantity the paper is about: the null is baked in by task saturation. There is no
> positive control demonstrating the benchmark could detect a scaling law if one existed,
> and held-out pre/post gain is reported at only one scale. The central negative result is
> unfalsifiable as measured. Reject unless a headroom-controlled gain measurement is added."

Why it outranks every other candidate gap:

1. **Claim dependence is total.** Contributions (i)–(iii) (`sections/p1_intro.tex:26-34`)
   and the conclusion all derive from the flat-slope null. If the null is an artifact of
   measuring saturated levels, the paper has no headline. The other hedges (single-seed
   frontier runs, GSM8K-only scope) are already explicitly Tier-C-scoped in the text;
   this one is not — the abstract states the null as the "central result" without the
   ceiling qualifier.
2. **It is a confound the paper's own artifacts document but the P1 build never confronts.**
   iter121's power calibration and the R(1)-starts-at-ceiling observations exist in the
   repo, yet `paper_P1_scaling.tex` stops at iter65 and ships the null unqualified by them.
   A reviewer who finds the released `scaling_law_iter121_synthetic_recovery.tsv` (the paper
   promises "we release all code, logs" in the abstract) will read the flat slope as a
   known-underpowered design presented as evidence.
3. **The existing ablation battery is orthogonal to it.** Every current ablation (functional
   form, anchor pool, abscissa choice, MoE/dense, truncation, changepoints) re-analyzes the
   same saturated level traces. None changes the ordinate to a gain or the regime to one
   with headroom, so no amount of them answers the objection.
4. **The backlog concedes the gap but ships nothing.** `experiments/FRONTIER_EXPERIMENT_BACKLOG.md`
   C2 ("Preregistered curve-collapse scaling law", H(N,T) = (A−A0)/(1−A0)) is *exactly* the
   headroom-normalized gain measurement — flagged "Expensive… the honest current claim is
   taxonomic" and not run. A2 (contrastive-yield re-plot, Tier A) changes only the abscissa
   (C_eff instead of N) while keeping the saturated-level ordinate, so it does not close
   this gap.

## 3) Minimal way to run it (cheapest credible version)

Two stages; Stage 0 is free, Stage 1 is the actual ablation and is a deliberately small
slice of backlog item C2 (gain-slope only, no curve-collapse law claim).

**Stage 0 — re-analysis + scoping (no GPU, ~1 day):**
- Promote the existing gain-proxy regression (Δ = late-window mean − R(1) vs log10 N,
  already computed at n=5 in `scaling_law_iter121_late_early.tsv`) into the P1 build,
  extended to the 12-anchor pool, alongside the iter121 synthetic-recovery power table as
  an explicit "detection floor" paragraph.
- Add the level-vs-gain decomposition: the R(1) slope (+0.060/decade, iter13) is
  statistically identical to the mean-reward slope (+0.081/decade), i.e., essentially all
  cross-scale variance is base-model level, not RL. State it.
- This makes the confound explicit but does **not** rescue the headline — it only converts
  a silent flaw into a scoped claim ("no slope measurable in the saturated regime").

**Stage 1 — headroom-matched cross-scale gain (the ablation; ≈40–80 H100-hours + minor
API cost, reuses existing infra):**
1. *Headroom matching (inference only):* for 3–4 dense anchors spanning ~2 orders of
   magnitude (Qwen3-0.6B, Qwen3.5-4B, Qwen3-8B, optionally Qwen3.5-27B), score per-prompt
   base pass rate p0 on GSM8K (G=8 samples/prompt); select a per-anchor prompt subset with
   p0 ∈ [0.2, 0.6] (≈500 train / 500 held-out per anchor). Every anchor now starts
   materially below ceiling with matched headroom.
2. *Training:* one fixed stack (the existing TRL LoRA r=32 pipeline with the 5-seed
   manager, or Tinker), GRPO, G=8, T=150 steps — 5× the current frontier budget; 3 seeds at
   ≤8B, 1–2 seeds at 27B.
3. *Endpoint:* normalized held-out gain H = (A_post − A_pre)/(1 − A_pre) per run (the C2
   estimand); regress H on log10 N with seed-level bootstrap CIs.

This directly reuses `base_instruct_paired.tsv`'s pre/post evaluation path (which already
proved the pipeline at one scale) and the Tier-A seed machinery from
`sections/_shared_methods.tex`; the only new artifact is the per-anchor difficulty filter.

## What result would change the paper's conclusion

- **H > 0 at ≥2 scales and slope(H, log10 N) ≠ 0** (bootstrap CI excluding zero, either
  sign): the headline null is falsified as a ceiling/budget artifact. Abstract's "central
  result," contribution (i), and the conclusion must be rewritten — the surviving claim
  shrinks to "no slope in the saturated GSM8K regime at T ≤ 30."
- **H > 0 but slope ≈ 0 with a tight CI**: strongest possible outcome — the positive
  control works (RL gain is detectable), yet the gain still does not scale. The null
  upgrades from vacuous to substantive and the reviewer attack is fully defused.
- **H ≈ 0 at all scales even with headroom and T=150**: a different and larger problem —
  GRPO produces no held-out gain at all in this stack, which undercuts not just P1 but the
  premise of the benchmark; the paper would need to report this as its finding.

The current text is compatible with all three outcomes, which is precisely why a reviewer
will not accept the headline as it stands.

## Cross-check: ablations that already exist (and why they don't cover this)

| Existing artifact | What it covers | Why it doesn't close the gap |
|---|---|---|
| `scaling_law_cross_scale.tsv`, `scaling_law_power_law.tsv` | level-vs-N regressions, n=5/12 | ordinate is absolute reward level, saturated at step 1 |
| `scaling_law_iter121_late_early.tsv` | within-trace late−early proxy vs N (n=5) | proxy ≈ 0 by ceiling construction; not in the P1 build |
| `base_instruct_paired.tsv` | true held-out pre/post gain, 5 seeds | one scale point only (Qwen3-8B); 0.6B/1.7B rows source-missing |
| `scaling_law_iter121_synthetic_recovery.tsv` | detection-power calibration | proves the design can't detect a slope; not in the P1 build |
| iter29 same-stack 40-step runs | learning detectable below ceiling | single scale (0.5B); used for estimator equivalence, not scaling |
| Backlog A2 (C_eff abscissa) | alternative x-axis | keeps saturated-level y-axis |
| Backlog C2 (curve-collapse H(N,T)) | exactly the right estimand | flagged Tier-C, expensive, not run — Stage 1 above is its minimal slice |

## Runner-up gaps (lower risk, for completeness)

1. **Step-budget ablation alone** (T ≤ 30 → 150+ at fixed anchors): subsumed by Stage 1;
   on its own it does not fix the ceiling confound for the instruct anchors.
2. **Base-vs-instruct initialization mix in the anchor pool** (the 12 anchors mix base and
   instruct checkpoints; `scaling_law_iter129_capability_scaling.tsv` shows capability
   class, not size, separates the pool): partially covered by iter109b/129 artifacts, but
   those sections are not in the P1 build — worth importing, though it sharpens rather
   than rescues the null.
3. **Multi-seed frontier anchors** (Nemotron collapse on n=1): already hedged as Tier-C
   descriptive case studies in `_shared_methods.tex` and `p1_conclusion.tex`; reviewers
   generally accept scoped case-study framing, so this is a weakness but not the rejection
   trigger.
