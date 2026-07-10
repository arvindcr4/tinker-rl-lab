# Iter 89 — P5 N2 algorithm-axis bootstrap CIs + LOMO stability

## Proposal (brief vein (b) + (c))

Iter 85 row 101 reported point estimates of pooled algorithm-axis
$\eta^2_{\mathrm{algo}}$ on the N2 four-method same-stack panel
(40 steps × 4 methods × 1 seed). At $n{=}40$ steps the point
estimates can be sample-noise unstable, particularly on right-skewed
channels like `zvf` and `pcd`. Brief vein (c) asks for **bootstrap
CIs on every P5 headline number**, which iter 85 did not close.

This iteration applies paired-step bootstrap ($B{=}4000$, seed
20260705) on the same N2 panel, plus a leave-one-method-out (LOMO)
stability audit on the algorithm axis.

## Measured result

### H1 — bootstrap CIs on pooled $\eta^2_{\mathrm{algo}}$ per metric

| Channel        | $\eta^2_{\mathrm{pt}}$ | $\eta^2_{\mathrm{lo}}$ | $\eta^2_{\mathrm{hi}}$ | ≤0.05? | ≤0.10? |
| -------------- | ---------------------: | ---------------------: | ---------------------: | :----: | :----: |
| `zvf`          | 0.0454 | 0.0145 | 0.1127 | ✗ | ✗ |
| `pcd`          | 0.0357 | 0.0153 | 0.0807 | ✗ | ✓ |
| `larq`         | 0.0010 | 0.0003 | 0.0136 | ✓ | ✓ |
| `reward_mean`  | 0.0075 | 0.0019 | 0.0244 | ✓ | ✓ |
| `mean_len`     | 0.0631 | 0.0333 | 0.1264 | ✗ | ✗ |
| `cv_len`       | 0.0457 | 0.0235 | 0.0914 | ✗ | ✓ |
| `loss`         | 0.9867 | 0.9828 | 0.9908 | ✗ | ✗ |

**Correction to iter 85 row 101's strict-pass headline**: only
**2 of 4 channels** that iter 85 reported as strict-pass survive
bootstrap-strict (`larq`, `reward_mean`). Loose Ivison (UB ≤ 0.10)
survives on **4 of 7 non-loss channels** (the same as iter 85's
loose verdict).

### H2 — pair-wise algorithm-pair decomposition

The 6-pair decomposition isolates which methods carry the
between-method variance:

- On `zvf`, `pcd`, `mean_len`, `cv_len`: **GIFT-containing pairs**
  have $\eta^2_{\mathrm{pair}} \geq 0.04$ (CI UB up to 0.16); the
  **three non-GIFT pairs** have $\eta^2_{\mathrm{pair}} \leq 0.005$
  (CI UB ≤ 0.04).
- On `larq`, `reward_mean`: all 6 pairs stay ≤ 0.04.

The algorithm-axis signal on contrast-yield and length-variance
channels is **GIFT-driven**, not a generic GRPO-family property.

### H3 — leave-one-method-out (LOMO) stability on `zvf`

| Omit   | Remaining             | $\eta^2_{\mathrm{pt}}$ | $\eta^2_{\mathrm{lo}}$ | $\eta^2_{\mathrm{hi}}$ |
| ------ | --------------------- | ---------------------: | ---------------------: | ---------------------: |
| grpo   | aero, areal, gift     | 0.0555 | 0.0167 | 0.1370 |
| aero   | areal, gift, grpo     | 0.0539 | 0.0149 | 0.1301 |
| areal  | aero, gift, grpo      | 0.0429 | 0.0099 | 0.1086 |
| gift   | aero, areal, grpo     | **0.0038** | 0.0003 | 0.0392 |

**Removing GIFT collapses algorithm-axis $\eta^2$ from 0.0454 to
0.0038 (a 12× drop)**. Removing any other method leaves $\eta^2$
in [0.043, 0.056]. GIFT is uniquely load-bearing on the `zvf`
axis.

### H4 — bootstrap CI on GIFT dominance (Cohen's d, last-10 step)

- `zvf`: $d{=}+2.353$, CI $[+1.521, +4.155]$, **lower bound ≥ 1.0**.
- `reward_mean`: $d{=}+0.465$, CI $[+0.038, +0.975]$, excludes 0.
- `pcd`: $d{=}-1.712$, CI $[-3.099, -1.076]$, excludes 0.

GIFT's `zvf` effect is statistically indistinguishable from a
≥1.5-SD excess.

## Cross-paper coupling

1. **P5 iter 85 row 101** — direct bootstrap correction of
   iter 85's strict-pass headline. The qualitative thesis survives
   on the loose threshold.
2. **P6 iter 86 row 102** — iter 86 measured `Y_obs` ranking
   (AREAL > AERO ≈ GRPO > GIFT); iter 89 isolates **GIFT as the
   lone variance driver** on the algorithm axis, recovering iter 86's
   GIFT-is-lowest ranking at the algorithm-axis layer.
3. **P7 iter 87 row 103** — iter 87's hysteresis controller treats
   GIFT as the worst case (H5). iter 89's pair-wise finding
   confirms why: GIFT's contrast-yield excess comes at the price
   of larger length variance, which the hysteresis rule must absorb.
4. **FRONTIER_INSIGHTS Round 2 (Gemini Deep Think)** — the frontier
   synthesis's $\delta_{\mathrm{div}} \in [0.13, 0.23]$ range is
   consistent with GIFT contributing nearly all of that range;
   iter 89's LOMO evidence (drop 0.0454 → 0.0038 on GIFT removal)
   is the **first isolation** of the structural bonus as
   GIFT-specific.

## Operational recommendation

Report the N2 four-method same-stack panel as **algorithm-equivalent
on 2 of 7 channels at $\eta^2 \leq 0.05$ (strict, bootstrap-corrected)
and on 4 of 7 at $\eta^2 \leq 0.10$ (loose)**. Adopt the **GIFT-aware**
reporting convention: when reporting algorithm-axis decomposition on
contrast-yield / length-variance channels, isolate GIFT-vs-rest as a
separate axis; the algorithm axis is dominated by GIFT, not by generic
GRPO-family differences.

## Files

- `platform_modal/scripts/p5p8/p5_iter89_n2_unpacking_bootstrap.py` (~280 LoC, stdlib only)
- `experiments/results/p5p8/p5_n2_unpacking_boot.tsv` (7 rows)
- `experiments/results/p5p8/p5_n2_unpacking_pair.tsv` (36 rows)
- `experiments/results/p5p8/p5_n2_unpacking_lomo.tsv` (4 rows)
- `experiments/results/p5p8/p5_n2_unpacking_boot_summary.json`
- `paper/sections/p5_iter89_n2_bootstrap.tex` (~80 lines, 7 paragraphs + 2 tables)
- `paper/paper_P5_minreport.tex` extended with `\input{sections/p5_iter89_n2_bootstrap}`
- `paper/paper_P5_minreport.pdf` rebuilds to 44 pages / 0 errors / 0 undefined citations

## Verified citations

- Ivison et al., 2024. Unpacking DPO and PPO: Disentangling Best
  Practices for Learning from Preference Feedback. NeurIPS 2024.
  arXiv:2406.09279. (already cited in iter 85)
- Cohen, J. (1988). Statistical power analysis for the behavioral
  sciences. (Cohen's $d$ definition; standard reference)