# 60 — P5 Stack-Conditioning Generalisation to the Mega-Cell Corpus

**Vein (fresh, not in prior ledger)**: extend the iter-45 η² stack-conditioning
framework from the 4-method same-stack N2 panel to the 98-cell mega
corpus where the algorithm is **fixed** (GRPO across all 98 cells) and
the stack varies across {model, task, G, temperature, seed}. The P5
thesis "Report the Stack, Not the Label" was defended at the algo-axis
level by iter-45 (η²(method) = 0.0005 on N2); this iter defends it at
the stack-axis level on a much larger and more heterogeneous corpus
(98 cells, 2 models × 3 tasks × 5 G × 2 temperatures × 2 seeds).

## Method

Read `experiments/results/mega_20260704/cells.tsv` (98 GRPO cells).
For each cell, take the **per-cell mean_reward** and **per-cell zvf**
as outcomes; compute one-way η² for the five stack axes; bootstrap
95% percentile CI on each axis (B=2000, cell-level resample, seed
20260704). Headline: sum of η²(stack axes) on mean_reward, with each
clamped at ≥0; the threshold for "stack dominates" is 0.50 (when only
the stack varies, the stack explains >50% of outcome variance).

## Falsifiable headline

```
sum(eta^2 stack axes) on mean_reward = 0.7560  [PASS, threshold 0.50]
sum(eta^2 stack axes) on zvf         = 0.9268  [PASS, threshold 0.50]
```

Per-axis decomposition (95% bootstrap CI, B=2000):

```
outcome    axis          k   eta^2     CI_lo    CI_hi
mean_reward model        2   0.4527    0.3144   0.6328
mean_reward task         3   0.2729    0.1880   0.4148
mean_reward G            5   0.0304    0.0115   0.1766
mean_reward temperature  2   0.0000    0.0000   0.0556
mean_reward seed         2   0.0000    0.0000   0.0513

zvf         task         3   0.4687    0.3675   0.5895
zvf         G            5   0.4437    0.2675   0.6296
zvf         temperature  2   0.0092    0.0000   0.0802
zvf         model        2   0.0053    0.0000   0.0781
zvf         seed         2   0.0000    0.0000   0.0533
```

## Sharpest finding — the two outcomes are dominated by different stack axes

- **mean_reward is dominated by MODEL (η² = 0.4527)**: the meta-llama-3.2-3B
  base model **fails completely on 34/35 humaneval_subset cells** (mean_reward=0);
  the Qwen3.5-4B base model scores much higher on humaneval but not on
  gsm8k. So the choice of MODEL is the dominant predictor of whether
  GRPO can produce any reward signal on humaneval at all.

- **zvf is dominated by TASK (η² = 0.4687) and G (η² = 0.4437)**: contrast
  starvation (zero-variance fractions) is determined by the task
  difficulty distribution and the group size G, not by the model.
  This is consistent with the iter-45 N2 finding (step+prompt variance
  dominates) and the iter-47 finding (per-prompt headroom is the
  per-step ZVF decomposition axis).

- **TEMPERATURE and SEED are near-zero on both outcomes** (η² < 0.01,
  CIs all overlapping 0). This is a controlled negative result that
  sharpens the iter-45 finding: once model+task+G are fixed, neither
  sampling temperature (within {0.6, 1.0}) nor seed (within {0, 1})
  moves the outcome by more than 1pp of variance.

## Connection to iter-45 (cross-paper coupling)

- Iter-45 measured η² on a 4-method same-stack panel (N2: grpo, aero,
  gift, areal) and found η²(method) = 0.0005. That is the *algo-axis*
  result: with the stack held fixed, the algorithm label is invisible.
- This iter (60) measures η² on a 1-algorithm varying-stack panel
  (mega: only GRPO, but stack varies across 2×3×5×2×2). The
  *stack-axis* result is the inverse: with the algorithm held fixed,
  the stack explains 75.6% of mean_reward variance and 92.7% of zvf
  variance.
- Together the two iterations form a 2×2 design: **(algo, stack) ×
  (vary-algo, vary-stack)**. Both cells of the design point to the
  same conclusion: **variance lives on the stack axis, not the
  algorithm axis**. This is the operational form of the P5 thesis
  "Report the Stack, Not the Label" at the multi-cell scale.

## Zero-cluster (controlled)

34/98 cells have mean_reward = 0; 34 of those are humaneval_subset
cells (Llama-3.2-3B base model fails entirely on the subset); 1 is a
gsm8k_easy cell. The η²(model) on mean_reward is inflated by this
zero-cluster, but the **zvf decomposition is NOT inflated** — the zvf
varies continuously across all 98 cells (range [0.34, 1.00]) and the
task+G axes cleanly explain 91.2% of its variance. This is the
strongest possible defense against the "the result is just the
zero-cluster" critique: even on the non-degenerate outcome, the
stack-axis dominance holds.

## Falsifiable predictions (parked)

If the P5 thesis holds, then on a future mega harvest with **more
models** (k_model ≥ 3) and **more temperatures** (k_temp ≥ 3):

- η²(model) on mean_reward should grow proportionally to the model's
  base capability spread on the included tasks;
- η²(temperature) on zvf should remain near zero unless temperatures
  span {0.6, 1.0, 1.5+} (current {0.6, 1.0} is too narrow to
  discriminate).

## Artifacts

- `platform_modal/scripts/p5p8/p5_stack_conditioning_mega.py` (~250 LoC, stdlib only)
- `experiments/results/p5p8/p5_stack_conditioning_mega.tsv` (10 rows: 5 axes × 2 outcomes)
- `experiments/results/p5p8/p5_stack_conditioning_mega_boot.tsv` (10 rows: bootstrap CIs)
- `experiments/results/p5p8/p5_stack_conditioning_mega_summary.json`

## Why this matters for P5

Iter-45 (η²(method) = 0.0005) was a clean N=4 same-stack measurement
that defended "Report the Stack, Not the Label" at the algo-axis.
Iter-60 (η²(stack sum) = 0.7560 / 0.9268) is the complementary N=98
varying-stack measurement that defends the same thesis at the
stack-axis. Together they form a 2×2 cross-design — neither alone is
sufficient because (vary-algo, fix-stack) and (fix-algo, vary-stack)
are complementary surfaces. The paper section can now point to BOTH
cells as evidence of the P5 thesis, not just the algo-axis side.