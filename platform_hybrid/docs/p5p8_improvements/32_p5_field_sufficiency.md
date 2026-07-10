# P5 item 32 — MIN-REPORT field predictive-sufficiency ("load-bearing" test)

**Pillar:** P5 (Report the Stack, Not the Label) · **Class:** T1+T2 · **Iter:** 25

## Idea (not in prior ledger)

Exhibits 5/7 (items 03/11) establish that **stack axes** dominate the variance
of every telemetry channel via *univariate* one-way $\eta^2$. That answers
"how much does each axis explain in isolation" but not the question a
practitioner actually faces when deciding what to disclose: **if I omit this
one MIN-REPORT field, how much of my ability to predict the reported outcome
do I lose?** That is a *joint, multivariate* question — it must account for
field interactions and redundancy, which univariate $\eta^2$ cannot.

We operationalize "Report the Stack, Not the Label" as a **predictive-sufficiency
test**: fit a joint predictor of per-cell telemetry (`zvf`, `mean_reward`) from
the disclosed MIN-REPORT stack fields, then measure the **regret of omission**
$\Delta R^2 = R^2_\text{full} - R^2_\text{drop\ }f$ for each field $f$, with
paired cluster-over-cell bootstrap CIs. A field is **load-bearing** iff its
$\Delta R^2$ CI excludes zero. The **label-only** baseline (all 98 cells share
the same sampling label) has $R^2=0$ by construction — the punchline in one
number: the label predicts nothing; the stack predicts most of the variance.

## Method

- Data: `experiments/results/mega_20260704/cells.tsv`, $n=98$ cells
  ($2$ models $\times 3$ tasks $\times 5$ $G \times 2$ temps $\times 2$ seeds).
- Predictors: `model_family`, `task_slice` (one-hot), `G` (log2), `temperature`.
  `seed` is a nuisance control (added on top of full stack).
- Model: `RandomForestRegressor` (300 trees, `min_samples_leaf=3`), out-of-fold
  predictions via 8-fold CV averaged over 12 fold seeds for stability.
- CIs: paired case-resampling bootstrap over cells ($B=2000$) on $R^2$ and every
  $\Delta R^2$.
- Script: `scripts/p5p8/p5_field_sufficiency.py`; outputs
  `experiments/results/p5p8/p5_field_sufficiency{.tsv,_summary.json}`.

## Result

Run: `n=98` cells, 8-fold CV averaged over 8 fold-seeds, $B=2000$ paired
bootstrap. All numbers from `p5_field_sufficiency_summary.json`.

**The label predicts nothing; the stack predicts almost everything.**

| target | label-only $R^2$ | full-stack $R^2$ (95% CI) |
|---|---|---|
| `zvf` | **0.000** | 0.832 [0.772, 0.879] |
| `mean_reward` | **0.000** | 0.993 [0.990, 0.996] |

**Per-field regret of omission $\Delta R^2$ (95% paired-bootstrap CI); load-bearing = CI excludes 0:**

| field | `zvf` $\Delta R^2$ | load? | `mean_reward` $\Delta R^2$ | load? |
|---|---|---|---|---|
| model\_family | 0.025 [0.007, 0.049] | ✔ | **0.942 [0.809, 1.129]** | ✔ |
| task\_slice   | **0.513 [0.356, 0.752]** | ✔ | **0.656 [0.497, 0.889]** | ✔ |
| G             | **0.441 [0.324, 0.581]** | ✔ | 0.001 [-0.001, 0.002] | ✗ |
| temperature   | 0.011 [-0.003, 0.025] | ✗ | 0.001 [0.000, 0.001] | ✔ (negligible) |
| **+seed (nuisance)** | **-0.015 [-0.022, -0.006]** | strictly harmful | -0.004 [-0.006, -0.002] | strictly harmful |

Three findings:

1. **Load-bearing-ness is outcome-specific.** For the contrastive-signal channel
   (`zvf`), `task_slice` and `G` dominate ($\Delta R^2 = 0.51, 0.44$); for
   accuracy (`mean_reward`), `model_family` and `task_slice` dominate
   ($0.94, 0.66$). `G` is load-bearing for ZVF but **statistically inert for
   reward** ($\Delta R^2 = 0.0005$, CI straddles 0) — exactly consistent with
   Exhibit 7's univariate finding that $G$ drives ZVF but not reward, now shown
   jointly with interaction structure. A single "report the important fields"
   rule is therefore ill-posed: which stack fields are load-bearing depends on
   which telemetry channel a claim rests on. This is the multivariate refinement
   the univariate $\eta^2$ table could not deliver.

2. **The seed axis is not merely uninformative — it is actively harmful to
   condition on.** Adding `seed` on top of the full stack *lowers* CV $R^2$ by
   0.015 (ZVF) / 0.004 (reward), CI excludes 0 in both channels: the predictor
   wastes capacity fitting run-to-run noise. This is the sharpest possible form
   of "seed is not a stack axis."

3. $\Delta R^2$ values do **not** sum to full $R^2$ (e.g. ZVF: 0.025+0.513+0.441
   +0.011 = 0.99 > 0.832) because leave-one-field-out on a joint RF captures
   redundancy/interactions, not an orthogonal ANOVA partition. This is expected
   and is precisely why the joint test adds information over Exhibit 7.

Figure: `experiments/results/p5p8/figures/p5_field_sufficiency.{png,pdf}`.

## Verified citations (reused, already in references.bib)

- `henderson2018deep` — Henderson et al., *Deep Reinforcement Learning That
  Matters* (AAAI 2018): implementation/hyperparameter choices dominate reported
  RL performance. Our field-omission regret makes this claim measurable at the
  individual reporting-field level for RL-for-LLMs.

## Verdict

**validated.** Adds Exhibit 13 to `paper/sections/p5_evidence.tex`
(`tab:p5-field-suff`). Directly answers the open question Exhibit 7 flagged
("which axes are load-bearing"), quantifies the paper's title claim in one
number (label $R^2=0$ vs stack $R^2=0.83$–0.99), and hardens the
"seed is not a stack axis" position with a signed, CI-bounded statement.
