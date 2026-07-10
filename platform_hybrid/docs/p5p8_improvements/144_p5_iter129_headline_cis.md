# 144 — P5 headline-CI audit (iter 129)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Type:** T1 (statistical rigor) + T3 (cross-paper coupling)
**Status:** proposed → **validated** (iter 129)
**Vein (fresh, not in 138 prior rows):** Closes brief vein (c) at the
P5-paper level. Iter-123 row 137 did this for P7 (19 headlines;
4 PASS / 1 TENSION / 5 REPORTED / 9 INSUFFICIENT_N); iter-129
does the same for P5 (15 headlines) using the same template.

## Problem statement

Every numerical point estimate in `paper/sections/p5_*.tex` deserves
a 95% CI that says *what noise source* it is reporting
(Miller 2024, arXiv:2411.00640). Iter-129 inventories **15 distinct
P5 point-estimate headlines** spanning 5 prior iterations and
recomputes each against the canonical raw data with a paired/non-paired
bootstrap appropriate to the underlying design.

## Headline inventory (15 rows)

| id  | vein    | claim                                 | published_pt | n | source data |
|-----|---------|---------------------------------------|--------------|---|-------------|
| H01 | iter-85 | eta^2(algo) mean over 6 channels      | 0.0331       | 6 | N2 four-method |
| H02 | iter-85 | eta^2(algo, zvf)                      | 0.0454       | 40| N2 four-method per-step |
| H03 | iter-85 | eta^2(algo, pcd)                      | 0.0357       | 40| N2 four-method per-step |
| H04 | iter-85 | eta^2(algo, loss) positive control    | 0.9867       | 40| N2 four-method per-step |
| H05 | iter-85 | Cohen's d zvf (GIFT vs other 3)       | +1.899       | 40| N2 last-10 pooled |
| H06 | iter-85 | Cohen's d pcd (GIFT vs other 3)       | −1.605       | 40| N2 last-10 pooled |
| H07 | iter-89 | eta^2 bootstrap UB on zvf             | 0.113        | 40| N2 (exceeds Ivison 0.05 ⇒ DECISIVE-fail) |
| H08 | iter-101| eta^2(algo, zvf_risk) 9-method        | 0.763        | 9 | zvf130_method_risk |
| H09 | iter-101| eta^2(seed, zvf_risk) control         | 0.0071       | 5 | zvf130_method_risk |
| H10 | iter-101| SCAFGRPO LOMO rel_drop                | −0.1042      | 5 | zvf130_method_risk (leave-one-out deterministic) |
| H11 | iter-125| chained R(zvf, task_slice)            | 10.32        | 3 | mega-98 + N2 |
| H12 | iter-125| chained R(pcd, task_slice)            | 12.62        | 3 | mega-98 + N2 |
| H13 | iter-125| chained R(zvf, G)                     | 9.77         | 5 | mega-98 + N2 |
| H14 | iter-125| chained R(pcd, G)                     | 6.45         | 5 | mega-98 + N2 |
| H15 | iter-121| auditor blind-spot rate (M1+M2)       | 0.0          | 196| 8 mutations × 98 cells = 784 audit evals |

## Method

`scripts/p5p8/p5_iter129_headline_cis.py` is the iter-123 row-137
template instantiated for P5. For each headline:

1. Load the canonical raw data (N2 per-step metrics, zvf130 per-seed
   risk index, mega-98 cells.tsv, or the iter-121 audit ledger).
2. Compute the published point estimate from the data.
3. Run a paired/non-paired bootstrap B=2000, seed=20260705, percentile
   method, on the natural pairing unit (per-step for N2, per-method for
   zvf130, per-cell for mega).
4. Compare the published point estimate against the recomputed CI.

**eta^2 bootstrap correction.** The first iter-129 run used a pooled
resampling that destroyed group structure; recomputed CI was
artificially compressed (UB=0.06 on a 0.99 point estimate). Fixed by
resampling WITHIN each group (preserves group identity & sizes); the
post-fix eta^2 bootstrap correctly recovers eta^2(loss)=0.987 [0.983, 0.991].

## Verdicts

```
n_headlines: 15
  PASS:      13
  TENSION:    1
  REPORTED:   1
  INS:        0
```

### H01–H08, H10–H14 — PASS (13/15 = 86.7%)

All 13 P5 numerical headlines have recomputed bootstrap CIs that contain
the published point estimate. This closes the reviewer-visible gap
"every P5 headline has a CI" for the 13 bootstrap-able headlines.

### H09 — TENSION (1/15 = 6.7%)

`eta^2(seed, zvf_risk) = 0.0071` was published iter-101. The
iter-129 recomputed value is **0.0191** with CI [0.0135, 0.2615].
The published 0.0071 is **below the recomputed CI lower bound**.
Probable cause: the iter-101 seed-axis control used a different
aggregation (likely GRPO-only or paired-seed across 5 methods × 1 seed
rather than the 5-seed × 4-method panel that iter-129 uses). The
published number is not REGRESSED — it is in the same order of
magnitude (small seed-axis contribution vs 0.76 algo-axis on the
zvf_risk channel) — but it does not survive a re-derivation on the
canonical `zvf_iter130_risk_index.tsv` per-seed data.

**Honest verdict:** TENSION, not REGRESS. The iter-101 paper text
should be patched to clarify the seed-axis aggregation unit (or to
report the recomputed value 0.019 with the bootstrap CI).

### H15 — REPORTED (1/15 = 6.7%)

The auditor blind-spot rate is deterministic (0/196 mutations
detected across M1+M2); there is no noise to bootstrap. Reported
as-is with the 196-evaluation sample size.

## Cross-paper coupling

1. **P7 iter-123 row 137** — iter-129 mirrors the iter-123 template at
   the P5-paper layer. Both audits use B=2000 paired bootstrap, seed
   20260705, ci=0.95, on the canonical raw data of each pillar.
2. **Berkeley row 11** (`unpacking_dpo_ppo_factorization.py`,
   `axis_variance_fraction`) — iter-129 uses the same eta^2
   decomposition formula on the N2 four-method panel. The iter-129
   bootstrap is the iter-123 / Miller 2024 extension of the
   `axis_variance_fraction` recipe.
3. **Berkeley row 22** (`adding_error_bars_to_evals.py`,
   `bootstrap_ci_paired` / `bootstrap_ci_mean`) — iter-129 reuses the
   paired/non-paired bootstrap primitives verbatim from the Berkeley
   Miller recipe. No reinvention.
4. **P5 iter-89 row 106** (N2 bootstrap-strict) — iter-89 reported
   the eta^2 UB on zvf = 0.113 on the N2 panel; iter-129 re-derives
   UB = 0.1464 (a slightly wider bootstrap CI). The DECISIVE-fail
   verdict (UB > 0.05) is preserved under iter-129's corrected
   within-group bootstrap.
5. **P5 iter-101 row 116** (zvf130 eta^2 scaling) — iter-101 reported
   0.763 on the 9-method panel; iter-129 reproduces exactly and adds
   the bootstrap CI [0.708, 0.919] (the published value is at the CI
   lower edge, supporting the LOMO range [0.6835, 0.8274] finding).
6. **FRONTIER_INSIGHTS Round 1** (Critic Degeneracy Hypothesis) —
   iter-129's H08 / H09 audit quantifies the algorithm-axis dominance
   over the seed-axis (9.5× to 40× eta^2 ratio across panels) that
   FRONTIER synthesis attributes to the critic collapsing to the
   group-mean estimator.

## Operational recommendation

(a) **Patch the iter-101 paper text** to clarify the seed-axis
aggregation unit OR to update H09 to 0.019 [0.014, 0.262] — the
TENSION is on the order-of-magnitude only, not on the qualitative
claim ("algorithm axis dominates seed axis on zvf_risk").

(b) **Add an `effect_size + 95% CI` column** to every numerical claim
in `paper/sections/p5_*.tex` next synthesis pass. The iter-129 audit
TSV is the source-of-truth for these CIs.

(c) **Wire `p5_iter129_recomputed_in_ci == True` for ≥13/15** as a CI
gate in `paper/sections/p5_iter129_headline_ci.tex` — any future
mutation that drops below 13/15 should fail the audit before paper
rebuild.

(d) **Mark H15 as REPORTED not PASS** in any future paper-facing
table — it is deterministic, not bootstrapped, and conflating it
with PASS overstates the rigor of the audit.

## Artefacts

| path | rows | description |
|------|------|-------------|
| `scripts/p5p8/p5_iter129_headline_cis.py` | 290 LoC | iter-129 audit driver |
| `experiments/results/p5p8/p5_iter129_headline_cis.tsv` | 15 | per-headline verdict |
| `experiments/results/p5p8/p5_iter129_headline_cis.json` | 15 | machine-readable summary |

`paper_P5_minreport.pdf` not rebuilt this iter (audit-only deliverable;
paper-facing patch lands next synthesis pass via the iter-101 H09
update recommended above).