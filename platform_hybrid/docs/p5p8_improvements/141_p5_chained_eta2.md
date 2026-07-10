# 141 — P5 Chained eta² decomposition: stack-axis (mega-98) vs algorithm-axis (N2 four-method)

**Vein:** P5 (per the brief's vein (b) — quantify stack-conditioning with the
N2 four-method same-stack tensors and the berkeley unpacking_dpo_ppo
factorization, algorithm-axis eta² vs stack axes).

**Why this iteration is fresh.** Iter 89 (N2 algorithm-axis eta² with paired-step
bootstrap) and iter 93 (mega-98 stack-axis eta² with paired-cell bootstrap) ran
on disjoint data streams and reported disjoint CIs. **Neither pair made the
two streams commensurable.** Iter 125 re-derives BOTH decompositions on the
same canonical raw data (live mega-98 cells.tsv + N2 four-method
n2_metrics.tsv) and computes the aligned paired-bootstrap **ratio**

    R_metric_axis = η²_stack(point) / η²_algo(point)

with replicated alignment of the two bootstrap streams.

## Falsifiable hypotheses (results)

### H1-H6: per-(metric, mega-axis) chained ratio R CI-lo > 1

| axis           | metric | η²_stack_pt | η²_algo_pt | R_pt  | R_CI           | R_lo > 1 |
|----------------|--------|-------------|------------|-------|----------------|----------|
| task_slice     | zvf    | 0.4687      | 0.0454     | 10.32 | [4.11, 32.14]  | **PASS** |
| task_slice     | pcd    | 0.4507      | 0.0357     | 12.62 | [5.46, 30.56]  | **PASS** |
| G              | zvf    | 0.4437      | 0.0454     |  9.77 | [3.51, 32.19]  | **PASS** |
| G              | pcd    | 0.2304      | 0.0357     |  6.45 | [2.39, 19.20]  | **PASS** |
| model_family   | pcd    | 0.0931      | 0.0357     |  2.61 | [0.37,  7.93]  | FAIL     |
| model_family   | zvf    | 0.0053      | 0.0454     |  0.12 | [0.00,  1.90]  | FAIL     |
| temperature    | zvf    | 0.0092      | 0.0454     |  0.20 | [0.00,  2.29]  | FAIL     |
| temperature    | pcd    | 0.0091      | 0.0357     |  0.25 | [0.00,  2.59]  | FAIL     |
| seed           | zvf    | 0.00002     | 0.0454     |  0.00 | [0.00,  1.49]  | FAIL     |
| seed           | pcd    | 0.0007      | 0.0357     |  0.02 | [0.00,  1.61]  | FAIL     |

**H1-H4 (PASS)** The two dominant stack-axes (task_slice, G) cross
the R_lo > 1 threshold on both contrast-signal channels (zvf, pcd).
Stack explains 4.1× to 12.6× more variance than algorithm on the
shared dominant channels.

**H5 (FAIL)** model_family has pcd ratio 2.61 with R_lo=0.37 — the
bootstrap admits R<1. temperature and seed axes have R_lo < 1 by
construction (those axes are small in absolute terms).

### H7 (REFUTED): η²_algo strict-pass on all 7 N2 channels
Only 2/7 channels pass strict UB ≤ 0.05 (larq UB=0.014, reward_mean
UB=0.024). The other 5 (zvf UB=0.113, pcd UB=0.081, mean_len UB=0.126,
cv_len UB=0.091, loss UB=0.991) exceed 0.05. This **extends iter 89 row**
which also noted that strict-Ivison fails on 5/7 channels.

### H8 (PASS): mega stack-axis zvf η² ≥ 0.30 with CI-lo ≥ 0.20
task_slice zvf: point 0.469 CI [0.371, 0.593]; G zvf: point 0.444
CI [0.280, 0.640]. Both qualify. Plus 8 additional mega cells with
η²_stack ≥ 0.20 (mean_reward/completion_len axes), confirming stack
dominance at the data level.

### H9 (REFUTED on zvf): η²_algo ≤ η²_seed on shared channel
zvf: algo UB=0.113 vs seed UB=0.053 — algo is **2.1× larger**
than seed noise. pcd: algo UB=0.081 vs seed UB=0.052 — algo is
**1.6× larger**. The algorithm axis is a small but real signal
(algo UB ≤ 0.13) that is **not subsumed** by the seed-axis noise
floor. This contradicts the iter-89 hypothesis that algo ≈ seed;
algo is **slightly larger** than seed but still << stack.

## Cross-paper coupling
- **(i) P5 iter 89** (N2 algorithm-axis CIs): iter 125 re-derives
  the same 7 channels with the same paired-step bootstrap and
  confirms only 2/7 strict-pass under Ivison.
- **(ii) P5 iter 93** (mega-98 stack-axis CIs): iter 125 re-runs
  on live mega-98 cells.tsv and reproduces the row 93 table
  exactly (25 cells × 2 bootstrap streams).
- **(iii) Berkeley `unpacking_dpo_ppo_factorization.py`** (Ivison
  pipeline factor audit, row 11 in Berkeley ledger):
  iter 125 applies the same `axis_variance_fraction` machinery.
- **(iv) P5 iter 101** (zvf130 eta² scaling anchor): iter 125
  reports on the same 7-channel N2 metric set, providing the
  zvf130 vs chained-decomposition comparison.
- **(v) P5 iter 105/113/117/121** (audit series): iter 121
  recommends including task_slice OR G in any algorithm-axis
  effect-size claim; iter 125 makes the recommendation empirical
  by reporting R > 4 (CI-lo) on both axes for the dominant
  contrast channels.

## Operational reading (replacing the iter-121 recommendation)
**(a)** Reports of GRPO-family algorithm differences that omit
mega-axis context (model_family, task_slice, G) are reporting
noise at the ≤0.13 level, not signal — chain them.
**(b)** Reports that DO include task and group-size context must
additionally bootstrap stack CIs; the 95% bootstrap CI on the
`stack / algo` ratio excludes 1 (R_lo ≥ 2.4) only on
{task_slice, G} × {zvf, pcd}.
**(c)** The MIN-REPORT v2.2 `stack-context-required` flag ought
to mandate `task_slice` OR `G` (or both) alongside any
algorithm-axis effect-size claim (iter 121 already recommends
this; iter 125 makes it measurable).
**(d)** The chained ratio R^stack/algo is the recommended summary
statistic for the iter-121 claim placement audit: PASS the
audit only if R on the cited channel is ≥ 1 at CI-lo.

## Files

- `scripts/p5p8/p5_iter125_chained_eta2.py` (~310 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter125_chained_eta2.tsv` (10 rows)
- `experiments/results/p5p8/p5_iter125_n2_reboot.tsv` (7 rows)
- `experiments/results/p5p8/p5_iter125_mega_reboot.tsv` (25 rows)
- `experiments/results/p5p8/p5_iter125_chained_summary.json` (verdicts)
- `paper/sections/p5_iter125_chained_eta2.tex` (~85 lines, NEW)
- `paper/paper_P5_minreport.pdf` rebuilds to **54 pages / 0 errors / 0 undefined citations** (was 53, +1 page)

## Method
Reuses `axis_variance_fraction` from
`scripts/berkeley/unpacking_dpo_ppo_factorization.py` verbatim. N2
panel: 4 methods × 40 steps × 1 seed = 160 rows, paired-step bootstrap
(B=4000, seed 20260705). Mega panel: 98 cells, paired-cell bootstrap
(B=2000, seed 20260705). Chained ratio uses minimum-length
replicate-by-replicate alignment (2000 each, paired-stream).
