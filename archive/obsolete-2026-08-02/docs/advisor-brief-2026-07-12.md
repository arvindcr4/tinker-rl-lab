# Advisor Brief — M.Tech Project Review, 2026-07-12

**Student:** Arvind C R · **Project:** Zero-Variance Fraction (ZVF) in Group-Relative RL Post-Training (tinker-rl-lab)

## The contribution I am defending

> **ZVF — the fraction of prompt groups whose completions all receive identical
> reward, and therefore contribute zero gradient — is a cheap, online diagnostic
> for signal starvation in GRPO-style training, and group size G controls which
> end of training starves: small G exhausts its own signal as accuracy rises,
> large G avoids that wall at the same rollout budget.**

Scope is stated honestly everywhere: Qwen3-8B, GSM8K, Tinker managed API
(LoRA rank 4), 1–3 seeds per result. These are the limits, not footnotes.

## Evidence in hand (all runs logged to W&B `zvf-training`, artifacts in-repo)

1. **Matched-budget group-size panel (new this week).** 2,560 rollouts/arm,
   two seeds each: the G=2×160-step arms master the sampled pool (train
   reward ≈ 0.9–1.0) and end at ZVF ≈ 0.75–1.0 — the *all-correct*
   zero-variance wall, their final steps spent on zero-gradient groups —
   while the G=16×20-step arms end mid-learning (≈ 0.3–0.5) with
   ZVF ≤ 0.25 and signal intact.
   Reward alone reads "success"; ZVF, read together with reward, shows the
   updates have stopped carrying signal (ZVF alone aliases mastery with
   incapacity — the reward coordinate disambiguates). This is the clearest
   demonstration that ZVF carries information mean reward cannot.
2. **Loss-form panel (corrected).** Six uncapped 1,024-token arms, 3 seeds per
   loss: no length inflation in either GRPO or Dr.GRPO (all arms *shrink*
   3.8–12.2%), and no late-ZVF separation between the losses at this scale.
3. **Theory checks (with today's corrections owned openly).** ZVF's
   confidence interval is calibrated (Wilson, all tested settings). The
   geometric waiting-time quantile G·⌈ln δ/ln ZVF⌉ matched the observed
   quantile in all six strata (ratio ≈ 1.00) — correctly read as a
   *reliability budget*, not a minimum (direction fixed 2026-07-11). The
   signal-per-rollout objective's optimum turns out to be the universal
   G ∈ {2,3} for any prior (an algebraic identity, found in external review
   2026-07-11) — so the matching empirical argmax is a model-fit check, not
   a validated prediction; meanwhile the panel above shows small G inverts
   in the endgame. Group size is a schedule question.
4. **pass@k baselines.** Qwen3-8B base: GSM8K pass@1 30.4% / pass@32 91.0%
   (clustered 95% CIs); MATH-500 and MBPP panels for the post-RL comparison
   are complete or near-complete.

## The negative result I am reporting, not hiding

A six-arm GRPO-vs-Dr.GRPO panel was **invalidated** by an unwired `--loss`
flag: both arms silently trained the same objective, and no reward, length, or
ZVF trace revealed it — only reading the runner did. The panel was rerun with
the fix the same day; invalid artifacts are preserved under explicit
`.invalid_actually_grpo.json` names. This incident is now a thesis
methods-chapter case study (with the reporting checklist and registry it
motivated) on why stack identity must be verified, not inferred from outputs.

## Plan to completion (thesis-first, validated by external review)

The 17 drafted documents were consolidated under a reviewed plan (two external
model reviews + a three-model adversarial council). Decisions:

- **Primary deliverable: the thesis.** Pillars → chapters (P8, an unrelated
  fraud-detection study, is out-of-program and excluded); contribution as
  stated above; limitations chapter owns the scope statement.
- **Conference submissions are post-degree and gated.** The flagship paper
  goes out only if a pre-registered, compute-matched bakeoff shows the
  ZVF-triggered adaptive-G controller strictly beats static G=16 and naive
  heuristics; otherwise the descriptive-diagnostic version goes to a
  workshop venue. The reproducibility audit is deferred until it can run on
  an open stack with tooling that would have auto-caught the flag bug.
- **The benchmark harness ships as a versioned artifact** (code, manifests,
  validation scripts, technical report), not an archival paper claim.

## By defense date

1. Thesis draft: chapters assembled from the existing corpus (writing, not new
   experiments) — the binding activity from here.
2. Remaining compute (credit-dependent, ~already-budgeted scale): 3-seed
   replication of the two headline panels only; no new model families or task
   families before the degree.
3. Frozen artifact release: bench + registry snapshot + run manifests.

## What I need from this review

1. Agreement on the thesis contribution statement and its stated scope.
2. Confirmation that the post-degree gating of conference submissions is
   acceptable (vs. attempting a submission before defense).
3. Chapter-structure feedback: diagnostic → theory → group-size schedule →
   reproducibility case study → benchmark artifact.
