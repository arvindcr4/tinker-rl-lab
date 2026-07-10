# Iter 74 — P6 DrGRPO Measured Evidence + Zero-Evidence Delta Audit

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Vein picked:** hybrid of brief veins (a)+(d) — measured-vs-claimed audit on
existing entries, plus add measured evidence for `delta_drgrpo` from a
pre-existing real panel (length_bias_iter60), and characterise the
remaining 4 zero-evidence deltas with explicit panel requirements.

## What this iteration delivers

1. **`platform_modal/scripts/p5p8/p6_drplatform_tinker/grpo_measured_evidence.py`** — populates
   `delta_drgrpo.json` with 3 measured rows + 3 expected_effects + 3
   claim_validation rows, sourced from
   `platform_hybrid/experiments/results/length_bias_iter60_platform_tinker/grpo_vs_drgrpo.tsv`
   (Welch pooled-task CI, B=4000, seed=20260705, 2 tasks).

2. **`platform_modal/scripts/p5p8/p6_zero_evidence_audit.py`** — catalogs the 4
   remaining zero-evidence delta entries
   (`delta_dapo`, `delta_gspo`, `delta_liteppo`, `delta_reinforce`)
   with per-delta (core_claim, needed_panel, closest_proxy,
   verdict_status), and produces the registry-wide evidence summary.

3. **Registry patch** — `platform_hybrid/registry/entries/delta_drgrpo.json` now
   carries 3 measured rows / 3 expected_effects / 3 claim_validation
   rows. All additive; existing fields retained with `"| iter-74"`
   suffix on `notes`. Schema validation: **34/34 PASS** before and
   after.

4. **Paper section** — `platform_hybrid/paper/sections/p6_iter74_drgrpo.tex` (new),
   wired into `platform_hybrid/paper/paper_P6_registry.tex`. Rebuilds cleanly to
   **37 pages / 0 errors / 0 undefined citations** (was 36, +1 page).

## Falsifiable headline (re-run 2026-07-05)

**The registry's predicted sign for DrGRPO's `neg_frac` is CONTRADICTED
on the live corpus.**

- **H1 (predicted <0, measured >0 with sign-detectable CI):**
  pooled Δ_neg_frac = +0.1028 [+0.0646, +0.1409] (welch-pooled-task
  over arithmetic_easy [n=5 paired] + gsm8k_cot [n=3 paired]).
- **Verdict: CONTRADICTS** (first CONTRADICTS verdict on a
  length-bias metric with sign-detectable CI).
- **Sharp reading:** DrGRPO on this corpus does NOT remove length
  bias — it inverts it. DrGRPO has more negative-elasticity steps
  than GRPO, opposite the registry-listed prediction.

## Two non-headline findings

- **H2 (`pos_frac`):** pooled Δ = -0.1001 [-0.2087, +0.0085], CI
  includes zero → **NEUTRAL**. The inversion is concentrated in the
  negative-elasticity direction; the positive-elasticity axis is
  not statistically detectable on this panel at n=2 task means.

- **H3 (`L_star`):** pooled Δ = +44.25 [-27.85, +116.35], CI
  includes zero → **NEUTRAL**. The optimal-length curvature fit
  shifts toward longer responses for DrGRPO (consistent with
  neg_frac rising), but the CI is wide on 2 task means.

## Zero-evidence delta audit (4 of 14 deltas remaining)

| delta_id | n_components | core claim (one-line) | Needed panel |
| --- | --- | --- | --- |
| `delta_dapo` | 5 | asymmetric clip + dyn-samp + token-loss + KL off | N2-same-stack 5-method (DAPO replacing GRPO) |
| `delta_gspo` | 2 | sequence-level importance ratio + clip | N2-same-stack 5-method (GSPO replacing GRPO) |
| `delta_liteppo` | 2 | remove value head + clip advantages | N2-same-stack 5-method (LitePPO replacing GRPO) |
| `delta_reinforce` | 2 | vanilla REINFORCE with group-relative baseline | N2-same-stack 5-method (REINFORCE replacing GRPO) |

All four entries are PPO-family GRPO-variants that require
extending the N2 same-stack 4-method run (grpo/aero/areal/gift) to a
5+ method panel where the named variant replaces GRPO while every
other stack axis is held fixed. This is an **experimental-design
constraint, not a registry-schema constraint**.

## Registry-wide evidence tally (post iter-74)

- **14 delta entries**; **10 with measured evidence** (was 9, +1)
- **27 measured rows** total (was 24, +3 for DrGRPO)
- **27 claim_validation rows** total (was 24, +3)
- Verdict distribution: **10 SUPPORTS, 3 CONTRADICTS, 6 NEUTRAL, 8
  UNCLAIMED** (was 10/2/4/8; CONTRADICTS count rises 2→3 on
  DrGRPO's neg_frac).
- Schema validation: **34/34 PASS**

## Cross-paper coupling

- **P7 (Pillar 3):** the iter-66 δ_div contrastive-yield residual
  remains the non-CONTRADICT axis; all 4 measured variants on that
  axis have predicted sign matching observation (Δ_div > 0 for
  aero/areal/gift; p_two_sided < 0.001).
- **P5 (Pillar 1):** the iter-69 placebo-replacement feasibility
  found the 4/7 placebo problem is corpus-design not schema-design;
  this iter sharpens the parallel for P6 — the 4/14 zero-evidence
  problem is also corpus-design (no N2-same-stack rollout for
  DAPO/GSPO/LitePPO/REINFORCE), not schema-design (the schema has
  supported `measured[]`, `expected_effects[]`, and
  `claim_validation[]` since iter-34).
- **iter-58 (P6 #58 measured-coverage):** this iter closes 1 of 5
  zero-evidence deltas catalogued there (DrGRPO); the remaining 4
  are recharacterised in
  `platform_hybrid/experiments/results/p5p8/p6_zero_evidence_audit.tsv`.

## Reproduction

```
python3 platform_modal/scripts/p5p8/p6_drplatform_tinker/grpo_measured_evidence.py   # exit 0; patches delta_drgrpo.json
python3 platform_modal/scripts/p5p8/p6_zero_evidence_audit.py       # exit 0; 4-row audit + summary
python3 platform_hybrid/registry/query.py validate                   # 34/34 PASS
cd paper && pdflatex paper_P6_registry && bibtex paper_P6_registry && pdflatex ×2
```

## Outputs

- `platform_hybrid/experiments/results/p5p8/p6_drplatform_tinker/grpo_measured.tsv`
  (per-task × per-metric rows with paired CIs)
- `platform_hybrid/experiments/results/p5p8/p6_zero_evidence_audit.tsv`
  (4 rows: dapo/gspo/liteppo/reinforce characterisation)
- `platform_hybrid/experiments/results/p5p8/p6_registry_evidence_summary.tsv`
  (14 rows: per-delta evidence tally)
- `platform_hybrid/experiments/results/p5p8/p6_zero_evidence_audit_summary.json`
  (machine-readable headline numbers)
- `platform_hybrid/registry/entries/delta_drgrpo.json` (patched additively)
- `platform_hybrid/paper/sections/p6_iter74_drgrpo.tex` (new section)

## Backward compatibility

`platform_hybrid/registry/schema.json` unchanged. `delta_drgrpo.json` patched
additively. All 33 other entries unchanged. Schema validation
**34/34 PASS** before and after.