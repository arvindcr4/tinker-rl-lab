# 102 — P6 cross-stack delta_div / y_obs matrix on the N2 same-stack corpus (iter 86)

## Vein picked
Brief vein from iter 83 P5P8 synthesis section "fresh veins not yet on the ledger":
> **P6**: from the registry's `tinker_*_qwen3.5-4b_gsm8k.json`
> `outcomes.zvf_antiherding` block, compute the cross-stack comparison of
> `zvf_antiherding` between the 4 same-stack methods (grpo/aero/gift/areal);
> this would add a per-method measure of contrast-yield preservation
> independent of the reward/zvf scalar pair.

Fresh, not in 101 prior rows. Closes the iter-66 row 77 single-delta-vs-grpo
claim into a 6-pair machine-readable matrix.

## Falsifiable headlines

- **H1 — only 1/6 (17%) method-pairs reach significance on $\delta_{\mathrm{div}}$.**
  Among the four same-stack N2 methods, only
  **AREAL > GIFT by +0.014 (CI [+0.0016, +0.0268], p=0.0275)** survives
  the paired-step bootstrap at 95%. The iter-66 row 77 range statement
  ``$\delta_{\mathrm{div}} \in [0.039, 0.053]$'' therefore holds as a
  *range*, not as a *rank*.

- **H2 — 3/6 (50%) pairs reach significance on $\mathrm{Y\_obs}$ (Contrastive Yield), all of them involving GIFT.**
  GIFT reduces Contrastive Yield by 5.0–6.4pp relative to GRPO/AERO/AREAL.
  GRPO and AERO are tied at $\mathrm{Y\_obs}=0.2797$ on every step
  (no significant difference on any axis). The cross-stack ranking
  on $\mathrm{Y\_obs}$ is sharp; the ranking on $\delta_{\mathrm{div}}$ is flat.

- **H3 — GIFT is the only outlier in contrast-preservation on the same stack.**
  AREAL is the best ($\mathrm{Y\_obs}=0.2938$), AERO/GRPO tied
  ($0.2797$), GIFT is the worst ($0.2297$). Iter-66 row 77's
  ``all four share a structural diversity bonus'' reads as
  *anti-herding*, not as *contrast-yield-preservation*. The two are
  not interchangeable.

- **H4 — registry's `outcomes.zvf_antiherding` block is best read on the
  $\mathrm{Y\_obs}$ axis for cross-stack claims**, not on the
  $\delta_{\mathrm{div}}$ axis. Cross-method divergence on
  $\mathrm{Y\_obs}$ is 3× more often significant than on
  $\delta_{\mathrm{div}}$ (3/6 vs 1/6).

## Per-method ranking

| Rank | Method | $\mathrm{Y\_obs}$ (40-step mean) | $\delta_{\mathrm{div}}$ (40-step mean) |
|---|---|---:|---:|
| 1 | AREAL | 0.2938 | 0.0532 |
| 2 | AERO  | 0.2797 | 0.0453 |
| 3 | GRPO  | 0.2797 | 0.0497 |
| 4 | GIFT  | 0.2297 | 0.0394 |

## Six-pair matrix (paired-step bootstrap, B=4000, seed 20260705, n=40)

| pair | $\Delta\delta_{\mathrm{div}}$ | CI$_{95\%}$ | sig | $\Delta\mathrm{Y\_obs}$ | CI$_{95\%}$ | sig |
|---|---:|---|:---:|---:|---|:---:|
| grpo -- aero  | +0.0044 | [-0.0048,+0.0138] | NS  |  0.0000 | [-0.0234,+0.0234] | NS  |
| grpo -- areal | -0.0035 | [-0.0153,+0.0078] | NS  | -0.0141 | [-0.0422,+0.0125] | NS  |
| grpo -- gift  | +0.0103 | [-0.0007,+0.0210] | NS  | +0.0500 | [+0.0203,+0.0797] | **SIG** |
| aero -- areal | -0.0079 | [-0.0209,+0.0049] | NS  | -0.0141 | [-0.0484,+0.0188] | NS  |
| aero -- gift  | +0.0059 | [-0.0058,+0.0177] | NS  | +0.0500 | [+0.0188,+0.0813] | **SIG** |
| areal -- gift | +0.0138 | [+0.0016,+0.0268] | **SIG** | +0.0641 | [+0.0281,+0.1031] | **SIG** |

## Mechanism

The iter-66 row 77 audit gave four point estimates of $\delta_{\mathrm{div}}$
on a panel of 40 steps × 16 prompts × G=8. The four
single-delta self-references were summarized as ``$\delta_{\mathrm{div}}
\in [0.039, 0.053]$'' — a single scalar range, NOT a paired
significance test.

The iter-86 script computes the full 6-pair matrix on the same per-step
data. Two effect-magnitude differences (on $\delta_{\mathrm{div}}$ and
$\mathrm{Y\_obs}$) are exposed:

1. **$\delta_{\mathrm{div}}$ has high within-method variance**, so the CIs
   on 5/6 pairs include zero. The 4 methods do share a structural sampling
   diversity bonus $\in [0.039, 0.053]$, but the **ranking** among them on
   that bonus is not statistically sharp.
2. **$\mathrm{Y\_obs}$ has smaller within-method variance**, so 3/6
   paired-step CIs exclude zero -- every cross-stack pair involving
   GIFT. The ranking on $\mathrm{Y\_obs}$ (AREAL > AERO ≈ GRPO > GIFT)
   is the one that survives 95% significance testing.

## Artifacts

| File | Rows / size | Notes |
|---|---|---|
| `scripts/p5p8/p6_cross_stack_delta_div_matrix.py` | ~290 LoC | stdlib only; paired-step bootstrap B=4000, seed 20260705; idempotent |
| `experiments/results/p5p8/p6_cross_stack_delta_div_matrix.tsv` | 6 rows (6 method-pairs × 24 columns) | the matrix |
| `experiments/results/p5p8/p6_cross_stack_delta_div_matrix.json` | summary + per-method rank + per-pair verdict | full JSON |
| `experiments/results/p5p8/p6_cross_stack_delta_div_matrix_summary.json` | per-method rank + headline counts | condensed rank summary |
| `paper/sections/p6_iter86_cross_stack_matrix.tex` | 6 paragraphs + 2 tables + 4 cross-coupling bullets | new `\subsection{...iter 86}` for paper_P6_registry |

## Cross-paper coupling

- **(i) P6 iter-66 row 77** --- the iter-66 single-delta-vs-grpo claim is a
  4-self-reference; the iter-86 matrix shows that only 1/6 $\delta_{\mathrm{div}}$
  comparisons are significant, but the $\mathrm{Y\_obs}$ axis still
  produces a sharp GIFT-worst ranking. Iter-86 is the **machine-readable**
  correction of iter-66's narrative summary.
- **(ii) P6 iter-78 row 92** --- field-coverage audit; the iter-86
  matrix is the first analysis that uses every existing
  `zvf_antiherding` block in *pairwise* fashion rather than as four
  independent self-references. Iter-86 is consistent with iter-78's
  schema coverage without requiring a schema bump.
- **(iii) P7 iter-83 row 98 / iter-79 row 93** --- the joint controller
  fires on every GIFT step ($\mathrm{Y\_obs}$ lowest); the iter-86 result
  confirms that the controller's per-method savings-per-rollout ranking
  in iter-83 row 98 should be sorted by $\mathrm{Y\_obs}$ rather than
  $\delta_{\mathrm{div}}$. No update to the controller is needed; the
  iter-86 result is a *post-hoc* ranking alignment.
- **(iv) P5 iter-81 row 96 / P5P8-SYNTH row 95** --- the per-cell
  Items 14-17 yield-residual block is the cell-granularity analog of
  the iter-86 per-method contrast-yield ranking. Cell- and step-level
  contrasts share the same anti-herding structure: AREAL best,
  GIFT worst. The two analyses differ only by aggregation unit
  (prompt vs group).

## Operational recommendation

For downstream consumers of `outcomes.zvf_antiherding`:

1. **Filter cross-method significance on $\mathrm{Y\_obs}$, not on
   $\delta_{\mathrm{div}}$.** 3/6 $\mathrm{Y\_obs}$ pairs are
   significant; 1/6 $\delta_{\mathrm{div}}$ pairs are significant.
   The ranking on the structural-diversity-bonus axis is much
   noisier than the ranking on the contrast-yield axis.
2. **The registry's `outcomes.zvf_antiherding.delta_div_mean` field
   should NOT be read as a cross-stack significance claim.**
   It is a per-method self-reference against the iid baseline;
   the cross-stack significance is on a separate axis and
   lives in
   `experiments/results/p5p8/p6_cross_stack_delta_div_matrix.tsv`.
3. **No schema bump is required.** The matrix is a derived
   aggregate that reads from the existing four `zvf_antiherding`
   blocks without altering the schema. Future iterations may want
   to add a `cross_stack_y_obs_rank` enum field per stack entry,
   but this iter defers that schema bump until another signal
   cross-stack reading (e.g., paper P6's table) requires it.

## Status

validated — iter 86
