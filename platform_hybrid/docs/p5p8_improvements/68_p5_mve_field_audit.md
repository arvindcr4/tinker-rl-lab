# Iter 57 — P5 Item-8 MVE field-level distributional sanity audit (#68)

## Vein

Fresh P5 vein (not in prior ledger): the iter-53 #64 sub-field MVE
analysis recommended a 5-field continuous-telemetry item-8 (mean_reward,
zvf, pcd, mean_completion_len, std_completion_len) and iter-56 #67
validated empirically that all 5 fields are populated in cells.tsv,
delivering a deterministic +20 badge uplift. This iter moves ONE LEVEL
DEEPER to per-FIELD distributional sanity:

1. per-MVE-field distribution (min, max, median, IQR, std, %-constant,
   %-NaN) on the 98 cells
2. per-MVE-field cross-axis η² over the 5 stack axes (model_family,
   task_slice, G, temperature, seed)
3. per-pair Pearson r between the 5 MVE fields
4. per-cell 5-tuple fingerprint uniqueness
5. outlier detection (zvf ∈ {0,1}, reward ∈ {0,1}, std = 0)
6. per-task MVE distribution (gsm8k_easy / gsm8k_hard / humaneval_subset)

## Method

`scripts/p5p8/p5_mve_field_audit.py` (~290 LoC, stdlib + matplotlib).
Loads `experiments/results/mega_20260704/cells.tsv` (98 cells).
For each of 5 MVE fields:

- **Distribution** — standard 5-number summary + %-NaN + %-constant +
  %-saturated-at-min + %-saturated-at-max
- **Per-axis η²** — one-way SS_between / SS_total on each of the 5 stack
  axes; block-bootstrap 95% CI (B=2000, seed 20260704) on cell-level
  resampling with replacement
- **Cross-axis sum η²** — sum of the 5 axis-level η² values per field

Per-pair Pearson r on the 5 MVE fields (aligned n=98 cells). Per-cell
5-tuple fingerprint uniqueness. Outlier detection by saturation flag.
Per-task distribution (gsm8k_easy: 24 cells, gsm8k_hard: 40 cells,
humaneval_subset: 34 cells).

## Headline findings (falsifiable)

1. **All 5 MVE fields PASS the 0.50 stack-dominance threshold** (cross-axis
   sum η² ∈ [0.7310, 0.9269], every field's upper CI bound on the dominant
   axis > 0.50). The 8-item MIN-REPORT stack conditioning is preserved
   at per-field granularity: each field is at least 73% stack-explained.

2. **Each MVE field has a DIFFERENT dominant stack axis** — the 5-tuple
   is not redundant:
   - zvf: task_slice (0.4687) + G (0.4437) → 0.9269 sum (highest)
   - pcd: task_slice (0.4507) + G (0.2304) → 0.7840
   - mean_reward: model_family (0.4527) + task_slice (0.2729) → 0.7560
   - mean_completion_len: model_family (0.3006) + temperature (0.2066) → 0.7339
   - std_completion_len: task_slice (0.4135) + model_family (0.1592) → 0.7310
   The 5 fields measure DIFFERENT signal: temperature matters for length
   but not for zvf/reward; G matters for zvf but not for reward.

3. **Seed axis is uniformly near-zero with CIs that exclude 0.10 on
   every field** (max upper CI = 0.0609 on std_completion_len, max
   upper CI = 0.0572 on pcd/mean_reward). The seed is the noise axis
   — confirming iter-32's field-sufficiency finding at per-MVE-field
   granularity.

4. **Pairwise correlations are NOT near-1** — the strongest pair is
   zvf↔pcd at r=-0.8878 (anti-correlated, both bounded [0,1] and
   complementary by definition). All other pairs have |r| < 0.82:
   reward↔std_len -0.8221 (high std = confused sampling = low reward);
   pcd↔std_len -0.7169. The 5-tuple is informationally distinct.

5. **100.0% unique 5-tuples** (98 unique / 98 cells) — every cell is
   informationally distinct on the 5 MVE fields, validating the
   iter-53 "distinct profiles 15→98" claim at per-cell granularity.

6. **Outlier density** — 36/98 cells have zvf ∈ {0, 1} (36.7%
   saturated); 35/98 have reward ∈ {0, 1} (35.7% saturated, all on
   humaneval_subset). 0/98 cells have zero std.

7. **humaneval_subset is COMPLETELY DEGENERATE on 3/5 fields** —
   34 cells with mean_reward = 0.0, zvf = 1.0, pcd = 0.0 (zero
   variance). The 34 cells still distinguish themselves on
   mean_completion_len (std=66.6) and std_completion_len (std=19.1).
   The 5-tuple on humaneval_subset reduces to a 2-tuple
   (length, std) plus 3 constants — the iter-53 audit's "15 distinct
   profiles" headroom is dominantly driven by the humaneval cells
   splitting on length.

## Why this matters

The iter-53 sub-field MVE recommendation (#64) was theoretical
(cardinality analysis); iter-56 (#67) validated it on the empirical
cells.tsv. This iter (a) shows the 5 MVE fields are NOT a redundant
re-encoding of the same signal (each has a distinct dominant axis and
non-trivial pairwise r), (b) quantifies per-field stack conditioning
to license the 8-item MIN-REPORT standard at the field level rather
than just the item level, and (c) sharpens the iter-32 load-bearing
finding that the 5 continuous-telemetry fields are the load-bearing
layer of MIN-REPORT — not just one of several equally-informative
sources.

## Outputs

- `scripts/p5p8/p5_mve_field_audit.py` (~290 LoC, stdlib + matplotlib)
- `experiments/results/p5p8/p5_mve_field_audit.tsv` (5 rows × N cols)
- `experiments/results/p5p8/p5_mve_field_eta2.tsv` (5 fields × 6 rows: 5 axes + cross_sum)
- `experiments/results/p5p8/p5_mve_field_corr.tsv` (5×5 Pearson matrix)
- `experiments/results/p5p8/p5_mve_field_summary.json`
- `experiments/results/p5p8/figures/p5_mve_field_dist.{png,pdf}`

## Cross-coupling

- Closes iter-53 #64 sub-field MVE at the per-FIELD level
- Sharpens iter-56 #67 empirical MVE validation by reporting
  per-field structure rather than aggregate badge uplift
- Confirms iter-32 #32 load-bearing finding that seed is a noise axis
  (every per-MVE-field seed η² CI includes 0 at the 0.10 level)
- Sharpens iter-49 #60 stack-conditioning mega (which used only 2 of
  the 5 MVE outcomes) to all 5 fields

## Reproduction

```
python3 scripts/p5p8/p5_mve_field_audit.py
```

~3s on a single core. Outputs to `experiments/results/p5p8/`.