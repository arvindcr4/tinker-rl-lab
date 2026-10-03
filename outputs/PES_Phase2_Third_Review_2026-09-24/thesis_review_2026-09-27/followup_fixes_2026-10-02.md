# Follow-up fixes, 2026-10-02

These close the items left open by the 2026-09-27 review fix logs, plus the claimed fixes that a re-verification found had not landed.

## New experiment: M1(b)

`platform_hybrid/experiments/modal/modal_samestack_gsm8k_cot.py` was run on Modal: 15 runs on A10G. The setup was Qwen2.5-1.5B-Instruct on GSM8K-CoT with a 200-token cap, 64 generations per step and 30 steps. There were five paired seeds, evaluated on 200 GSM8K test items. The analysis plan was fixed in the script before launch.

| Arm | Accuracy before → after | Change from pre-training (95% CI) |
|---|---|---|
| GRPO, G = 8 | 0.200 → 0.245 | +0.045 [+0.033, +0.057] |
| GRPO, G = 2 | 0.200 → 0.250 | +0.050 [+0.026, +0.074] |
| PPO, value head | 0.200 → 0.195 | −0.005 [−0.033, +0.023] |

Paired contrasts after training:
- **GRPO G8 − PPO:** +0.050 [+0.015, +0.085], p = 0.016. All five seeds are positive, so the sign-flip test gives p = 0.0625, its floor at n = 5.
- **GRPO G8 − GRPO G2:** −0.005 [−0.038, +0.028], p = 0.69. TOST at ±0.02 gives p = 0.13, so equivalence is not established.

The PPO arm is minimal and untuned: a fresh linear value head, no warm-up, no GAE. The GRPO–PPO gap reflects that this PPO arm does not train. It is not a comparison against a tuned critic.

Results files:
- `platform_hybrid/experiments/results/samestack_gsm8k_cot.json`
- `platform_hybrid/experiments/results/samestack_gsm8k_cot_full.json`

Where the result is reported in the thesis:
- new §6.5.5 and its table
- §6.6
- Table 1.1
- the §1.2 same-stack paragraph
- the RQ2 entries in ch01 and ch10
- §9.4 future work
- the abstract
- Appendix A §A.3.2 and Table A.9

## Statistics: M16 and M18

The recomputation is in `stats_followup/`.

**M18.** The source's ZVF and PCD values (ρ = 0.529 and 0.541) are not reproducible. Its own script gives 0.407 and 0.336.
- ch07 now reports ρ = +0.44, +0.39 and +0.28 over all 4,753 pairs.
- A Mantel test gives p < 10⁻³ for each.
- Added to Table E.1.

**M16.** The intervals resample steps, not prompt-steps.
- Exact sign-flip tests, Holm-adjusted, give p = 0.047 (AERO), 0.047 (AREAL) and 0.23 (GIFT).
- At prompt level, the AERO interval includes zero.
- ch07 is corrected and the change is added to Table E.1.

## Claimed fixes that had not landed

- **"Held-out" wording:** removed from the abstract, ch01, ch02, and the §4.5 and Chapter 8 headings. The keyword "held-out evaluation" became "evaluation governance".
- **11/11 PASS:** each mention is now scoped to "eleven named checks". Appendix B states what the checks do not cover.
- **Counts:** the `zvf-audit` model count is now six labels, not seven. The registry snapshot behind 31/31 versus 44/48 is now named.
- **E14:** the Wilson CI [0.498, 0.528] was added.
- **E10:** the registry row now shows utility 88/97.
- **Research-question references:** the RQ4 entry in ch01 now points to Tables 8.A/8.B, not 9.A/9.B.
- **Citations and intervals:**
  - The 505-task audit is now cited to `p3_abstract.tex`.
  - The two sources for the scaling-slope interval (+0.313 and +0.323) are reconciled.
- **Wording:** the ch06 length-inflation sentence is fixed. The abstract now states that the same-stack evaluation uses in-distribution draws.
- **Figures:**
  - Fig 1.2 panel D is now descriptive only, with no CIs, p-values or power.
  - Fig 4.1: Pillar 2 now says all nine rows are a projection. Pillar 3 now gives the training-reward ratio, not retention. "Fixed in advance" became "declared, not pre-registered".
  - Fig 2.2 is fixed: verdict counts, the G = 16 equivalence statement, and the P2 association.
  - Fig 1.1 now names the sweep model as Qwen2.5-0.5B.

## Layout

- **Fig 10.1 (threats):** converted to Table 9.1 in §9.2.
- **Table A.7:** the constant Algo, G, Seed and Steps columns are dropped, which removes the cell overlap.
- **Appendix A tables:** proportional column widths. The build now adds zero glue at the start of each cell and allows a break after the org slash in model IDs. Remaining overfull boxes are all under 10 pt.
- **Evidence map (Appendix G):** cross-reference fragments are now attached to their source path instead of listed as separate sources.
