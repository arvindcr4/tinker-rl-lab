# 72 — P5 MIN-REPORT item discriminative-power audit

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label).

**Vein (fresh, not in prior ledger):** the iter-13 #18 MIN-REPORT auditor
produces a 0-100 badge by **weighting 7 items** (10/10/20/20/10/10/20).
Two open questions were left un-quantified:
1. Is the **ranking** of manifests robust to small perturbations of
   those weights, or do the weights dominate the ordering?
2. Does each individual item carry any measurable outcome-correlation
   on a real measured corpus, or are several items informational
   noise?

This iteration closes both questions on the canonical
n=103 audit sample (98 mega + 5 quick) and on the n=98 mega corpus
that joins to `cells.tsv` measured outcomes.

## Method

`scripts/p5p8/p5_item_discriminative_power.py` (~290 LoC, stdlib only):

1. **Load** `experiments/results/p5p8/minreport_audit.tsv` (n=103 rows,
   pre-existing from iter-13 #18) and
   `experiments/results/mega_20260704/cells.tsv` (n=98 measured cells).
2. **Measure (A) Multi-rater weight robustness** — for B=500
   weight perturbations drawn from symmetric Dirichlet(α=2) on the
   7-item weight vector, renormalised to sum to 100, recompute the
   badge and compute Spearman ρ with the canonical badge ranking
   across the 98 mega cells.  95% bootstrap percentile CI.
3. **Measure (B.1) Variance share** — for each of the 7 items, the
   population variance of the subscore on the full n=103 audit,
   normalised by total subscore variance — the share of *which item
   distinguishes which manifest*.
4. **Measure (B.2) Per-item × outcome Spearman ρ** on the 98 mega
   cells, bootstrap n=1000, for each (item × outcome ∈
   {mean_reward, zvf, pcd}).  Reports NaN for items that are
   constant on the mega subset.
5. **Measure (C) Inter-item redundancy** — Pearson r between every
   pair of items across the 103-row audit.

## Headline findings (falsifiable)

**Finding 1 — (A) Ranking is *trivially* robust (ρ = 1.000 over
B=500 weight perturbations)** — but the reason is that the
corpus has only 2-3 unique values per item, so any monotone
weight perturbation preserves the ordering.  Weight perturbation
**cannot rescue the auditor from its 2-tier degeneracy** on a
homogeneous corpus.

```
mean ranking Spearman ρ over B=500 perturbations = 1.0000
                                                [1.0000, 1.0000]
interpretation: robust (trivially — see B.1)
```

**Finding 2 — (B.1) Variance share is concentrated in 3 items.**  Of
the full-103 audit, items 3 (sampler/backend), 5 (G-schedule), and
7 (decontam/parser) account for **91% of the badge variance**;
items 1 (loss), 2 (KL), 4 (zvf-path), and 6 (held-out split) are
**near-constant** at ≤3% share each (each has only 2 unique values
across 103 rows).

```
item  share_of_variance   n_unique_values
1          0.0072           2       (loss: all 'n/a-sampling' or 'grpo')
2          0.0072           2       (KL:   all 'n/a'              or 'kl-…')
3          0.5406           2       (sampler/backend: tinker-closed or default)
4          0.0475           2       (zvf-path:  validates against the per-tensor JSON or n/a)
5          0.0540           3       (G:    fixed-G=N varies over the 5 unique N)
6          0.0288           2       (heldout: 'gsm8k_easy/hard/humaneval_subset' on mega; '')
7          0.3147           3       (decontam/parser: 13.33 vs 16.67 vs 10.0)
```

**Finding 3 — (B.2) Item 7 is the only outcome-correlated item on
the n=98 mega corpus.**  Items 1-6 are *constants* on the mega
corpus (every cell was emitted from the same template with the
same declared values), so their Spearman ρ is structurally
undefined.  **Item 7 (decontam + parser probe)** carries the entire
measurable outcome-correlation: its sub-score on the mega cells
predicts **mean_reward ρ = +0.831 [+0.752, +0.893]**, predicts
**zvf ρ = -0.820 [-0.883, -0.733]** (negative because higher
parser-probe score = lower ZVF on this corpus), and predicts
**pcd ρ = +0.819 [+0.735, +0.885]**.

The decontam sub-score differentiates between cells where the
parser-probe field is reported-but-not-validated (score 10.0)
versus reported-and-validated (score 13.33 or 16.67); both
states correlate with substantially different measured
outcomes — exactly the "Report the Stack, Not the Label" effect
at item-level granularity.

**Finding 4 — (C) Items 1, 2, 3 are perfectly redundant; (5, 6)
are negatively redundant.**  Items 1, 2, 3 each take only 2 unique
values across the audit and are perfectly Pearson-correlated
(r = +1.000 / -1.000); on the full sample items (5, 6) have r
= -0.623 (group-size schedule and heldout split are inversely
varying across the manifests — small-G cells tend to live on
held-out slices that are missing the corpus's own slice identity).

```
inter-item Pearson r — top 5 by |r|
items (1,3)         +1.000        -- perfectly redundant
items (2,3)         +1.000        -- perfectly redundant
items (3,6)         -1.000        -- perfectly anti-redundant
items (1,2)         +1.000        -- perfectly redundant
items (1,6)         -1.000        -- perfectly anti-redundant
items (5,6)         -0.623        -- meaningful
items (5,4)         +0.943        -- driven by tinker-closed path key
items (6,7)         +0.463        -- meaningful
```

## What this means for paper_P5

1. **The auditor's weight vector is over-parameterised.**  Items 1
   and 2 contribute ≤1% of badge variance each; they could be
   folded into a single "stack declaration" item without changing
   the ranking meaningfully (see § "Findings → auditor refactor" of
   the iter-13 #18 doc).
2. **Item 7 carries the entire measurable outcome-correlation** for
   the auditor on the mega corpus.  The "Report the Stack" thesis
   is sharpest at item-7 granularity on this corpus.
3. **The auditor would benefit from being run on a more
   heterogeneous corpus.**  The mega+quick combined sample has
   only 2 unique values on 5 of 7 items, suggesting the next
   manifest-emitter iteration should randomise loss/KL/sampler
   declarations to widen the auditor's discriminatory range.
4. **Weight-perturbation robustness is not informative** on this
   homogeneous corpus — the ρ = 1.0 finding is structurally
   guaranteed by the lack of item variance and does not validate
   the weight choices.  Future audits targeting weight robustness
   must first widen the item variance.

## Operational recommendation

For the next manifest-emitter iteration:
- emit `loss_form` from a controlled distribution (e.g.
  multinomial over {`grpo`, `dapo`, `drgrpo`, `n/a-sampling`})
- emit `ref_policy_kl` from a controlled distribution
- emit `sampler_backend_precision` from a controlled distribution
  (currently a single value across all cells)
- emit `group_size_schedule` so that the G values cover the full
  set {2, 4, 8, 16, 32} on a per-corpus basis

This will lift items 1-3 from 2 unique values to 4-7, restoring
the auditor's discriminative power and making any future
weight-perturbation audit interpretable.

## Artifacts

- `experiments/results/p5p8/p5_item_discriminative_power.tsv`
  (62 rows: 1 ranking rho + 7 weight-swing SD + 14 variance/n_unique
  + 21 outcome rho + 21 Pearson pairs)
- `experiments/results/p5p8/p5_item_discriminative_power_summary.json`
  (machine-readable shape)
- `scripts/p5p8/p5_item_discriminative_power.py` (≤300 LoC, stdlib)
- appended to `AUTORESEARCH_FINDINGS.jsonl` (pillar P5).

No paper update this iteration — the finding is a **scoping** audit
(informs the next iter-62 manifest-emitter work) rather than a
new claim for the §4 evidence section.  The next iteration will
fold item-1 and item-2 into a single stack-declaration item and
re-run the auditor on the widened corpus.
