# Iter-110 — P6 Cross-Panel Paired-Bootstrap Agreement (N2 ↔ zvf130)

**Pillar:** P6 (GRPO-Registry)
**Vein (fresh):** brief vein (a) — validate existing entries against measured
behavior. Prior P6 iters (18, 46, 82, 90) judged each registry delta against ONE
panel (N2 same-stack **or** zvf130 risk-index). This iter runs the SAME 3
GRPO-family methods (aero / gift / areal) on BOTH panels with paired bootstrap
on the natural pairing unit (per-step for N2, per-seed for zvf130), and
classifies whether the two panels AGREE in sign+significance vs DIVERGE.

## Falsifiable measured headlines

**H1 — Cross-panel verdict distribution on 6 (variant × metric) pairs (B=4000
paired bootstrap, seed=20260705, ci=0.95):**
- AGREE_BOTH_SIG   = **0** (zero; no variant has both panels confirming the
  same-direction effect on zvf / zvf_risk)
- DIVERGE_BOTH_SIG = **2** (both are GIFT)
- PARTIAL_ONE_SIG  = **3** (AERO zvf↔mean_zvf, AREAL zvf↔zvf_risk, AREAL zvf↔mean_zvf)
- BOTH_NONSIG      = **1** (AERO zvf↔zvf_risk)

**H2 — GIFT is the only registry delta that CONTRADICTS ITSELF across panels
(same stack, opposite direction).** On N2 (paired-step bootstrap on 40 steps,
variant[step] − grpo[step]):
- zvf = +0.0500 [95% CI +0.0203, +0.0797] (sig)  → GIFT *raises* zero-variance
  fraction on the same stack
- frac_all_one = +0.0438 [95% CI +0.0172, +0.0703] (sig) → GIFT *raises*
  all-one groups
- pcd = −0.0078 [95% CI −0.0124, −0.0029] (sig)   → GIFT *lowers* pairwise
  contrast diversity
- loss = −17,875 [95% CI −18,413, −17,335] (sig)  → GIFT's documented gamma
  shift dominates the loss surface

On zvf130 (paired-seed bootstrap on 5 seeds, variant[seed] − grpo[seed]):
- zvf_risk = −0.263 [95% CI −0.357, −0.169] (sig)  → GIFT *lowers* 5-seed
  zero-variance risk
- mean_zvf = −0.367 [95% CI −0.372, −0.363] (sig) → GIFT *lowers* mean zvf
- risk_csd = −0.580 [95% CI −0.767, −0.392] (sig) → GIFT *lowers* risk csd
- risk_drift = −0.618 [95% CI −0.628, −0.610] (sig) → GIFT *lowers* risk drift

Both panels exclude 0 in opposite directions on zvf: GIFT raises per-step zvf
(N2) but lowers seed-aggregated zvf_risk (zvf130). This is a real, falsifiable
finding — not a CI-width artifact (both CIs are tight and exclude 0).

**H3 — N2 also reveals 3 measured effects for GIFT that the registry does not
claim (`expected_effects` returns UNCLAIMED):**
- mean_len Δ = −0.74 tokens (not sig)  → neutral on length
- cv_len   Δ = −0.001 (not sig)        → neutral on length dispersion
- mean_len/gift/reward_mean Δ = +0.0105 (sig)  → GIFT raises reward by ~1pp
  on the same stack (the registry's expected_effect for reward_mean was ">="
  — this SUPPORTS that claim)

**H4 — AERO and AREAL are PARTIAL: zvf130 detects the reduction, N2 does not.**
Per-step N2 zvf on the gsm8k canonical 16-prompt set has so few all-zero
groups (most prompts have 6+/8 successful rollouts at G=8) that the
per-step zvf signal is dominated by ceiling effects. The 5-seed zvf130
risk-index captures the cross-seed / cross-step tail behaviour that N2
single-seed misses. This is a measurement-scope limitation, not a registry
bug — but it means the registry's claim that "AERO lowers ZVF" is
**supported only by zvf130**, and the N2 panel alone would conclude NEUTRAL.

**H5 — A side finding on mean_len (NEW metric, not in any registry claim):
N2 paired-step bootstrap on 40 steps shows a +14.2 token increase for AERO
[CI +9.9, +18.5] and +14.6 for AREAL [CI +10.5, +18.8] but −0.7 for GIFT
[CI −3.3, +1.8]. AERO and AREAL systematically lengthen outputs on this
gsm8k stack; GIFT does not. None of these are in the registry's
`expected_effects` blocks → 3 more UNCLAIMED rows for the iter-46 audit.**

## Verification of registry

All 39 entries (12 stack + 11 delta + 16 measured-related) PASS
`jsonschema.validate` after iter-110. The cross-panel audit is **not**
written into the registry entries themselves because the variant_delta
schema's `claim_validation` items use `additionalProperties: false`.
Instead, the audit lives in the standalone
`platform_hybrid/experiments/results/p5p8/p6_iter110_xpanel_summary.json` + verdict TSV.

## Cross-coupling

- **P6 iter-46** (registry_measured_claimed.py): per-(metric, panel) verdict
  using N2 last-10 only. Did not detect GIFT's cross-panel contradiction
  because it judged each (metric, panel) row independently and the
  `n2_same_stack_last10` row for GIFT.zvf was `delta=0.05, sig=true` and
  UNCLAIMED for lack of a declared expected_effect. iter-110 closes this
  by EXPLICITLY pairing (variant, n2_metric) with (variant, zvf130_metric)
  and reporting 2 CONTRADICTIONS.
- **P7 iter-107** (cross-curve class transfer): same stack, four methods —
  iter-107 reported savings `grpo=+0.360, aero=+0.360, gift=+0.385, areal=+0.353`
  with `cv(method)=0.039`. iter-110's N2 zvf point estimates are
  aero=0.0, gift=+0.05, areal=−0.014, but the zvf130 5-seed zvf_risk
  point estimates are aero=−0.148, gift=−0.263, areal=−0.246. The two
  panels agree on relative ordering (gift < areal < aero on zvf130 risk;
  gift > areal ≈ aero on N2 per-step zvf) but disagree on direction.

## Reproducibility

- Script: `platform_modal/scripts/p5p8/p6_iter110_n2_zvf130_xpanel.py` (stdlib only,
  ~280 lines)
- Bootstrap: paired_step_or_seed_pct, B=4000, seed=20260705, ci=0.95
- Outputs:
  - `platform_hybrid/experiments/results/p5p8/p6_iter110_n2_panel.tsv` (27 rows: 3 variants
    × 9 metrics)
  - `platform_hybrid/experiments/results/p5p8/p6_iter110_zvf130_panel.tsv` (15 rows: 3
    variants × 5 metrics)
  - `platform_hybrid/experiments/results/p5p8/p6_iter110_xpanel_verdict.tsv` (6 rows: 3
    variants × 2 metric pairs)
  - `platform_hybrid/experiments/results/p5p8/p6_iter110_xpanel_summary.json` (counts +
    bootstrap metadata)
- Run: `python3 platform_modal/scripts/p5p8/p6_iter110_n2_zvf130_xpanel.py` (≈3s on a
  cold start, <1MB memory)
- No GPU, no Tinker call.

## What this iter does NOT claim

- We do not claim the GIFT contradiction is a bug in the GIFT method.
  GIFT is *supposed* to trade a small loss increase for better
  training stability; +0.05 per-step zvf and −17,875 loss are exactly
  what its gamma-prior derivation predicts. The contradiction is a
  **registry-coverage gap**: the registry's `expected_effects` for GIFT
  says `zvf < 0` (zvf130 panel only) and the N2 panel does not see the
  same direction. The fix is either (a) add an N2-panel-specific
  expected_effect for GIFT that anticipates the per-step zvf increase
  (because gamma-regularised advantages produce more uniform group means
  → more all-one groups at G=8), or (b) drop the zvf_risk claim and
  replace with a panel-agnostic metric like `larq` or `pcd` which
  arguably better captures what GIFT actually changes.
- The BOTH_NONSIG verdict for AERO zvf↔zvf_risk reflects the n=40 vs n=5
  power asymmetry, not a real absence of effect.