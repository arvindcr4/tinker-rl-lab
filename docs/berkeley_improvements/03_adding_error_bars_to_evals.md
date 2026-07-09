# 03 — Adding Error Bars to Evals (Berkeley F25 L8, Sida Wang)

**Source lecture.** F25 L8 — Sida Wang (Meta), "Predictable Noise in LLMs / Adding
Error Bars to Evals." The lecture synthesizes two papers: (a) Evan Miller's
2024 short-note "Adding Error Bars to Evals" recipe for LLM evals, and
(b) Wang's own December 2025 "Measuring all the noises of LLM Evals."

**Target.** A1 — Statistical rigor of the benchmark (every headline number in
the 4 pillar papers should carry a defensible 95% CI; the original headline
should name the noise source, the paired/unpaired design, and an equiv-region
test for any `~=` claim).

**Citations (verified via WebFetch on arxiv.org, no fabrication).**

- Miller, Evan (2024). *Adding Error Bars to Evals: A Statistical Approach to
  Language Model Evaluations.* arXiv:2411.00640 (cs.CL / stat.AP), submitted
  1 Nov 2024. https://arxiv.org/abs/2411.00640 — confirmed title/authors/year/
  categories. Single-author "statistics for researchers" recipe.
- Wang, Sida (2025). *Measuring all the noises of LLM Evals.* arXiv:2512.21326
  (cs.LG / cs.AI / cs.CL / stat.ML), submitted 24 Dec 2025; v2 dated 29 Mar
  2026. https://arxiv.org/abs/2512.21326 — confirmed title/authors/abstract.
  Defines **prediction noise** (variation across generated answers for a
  fixed question) and **data noise** (variation across questions). Empirically:
  each eval has a characteristic total noise level; **paired prediction noise
  exceeds paired data noise** — so averaging predictions boosts power.

**Mapping onto Pillar evidence.** Of the seven Pillar 1/2/3/4 headline
numbers that cross-check the 4 pillar papers, several were reported as point
estimates with no error bars (H1, H2, H5), one was a paired test with a
declared SE (H4), one was a slope-in-log-log-space with a reported CI (H3),
one was an AUROC with a DeLong CI (H6), and one was an effect-size-CI pair
(H7). We re-derive each under Miller's recipe and grade the verdict as
DECISIVE / SUGGESTIVE / NULL.

## Prototype

`scripts/berkeley/adding_error_bars_to_evals.py` (stdlib-only, ~800 lines)
ingests the seven TSV/JSON inputs and writes:

- `experiments/results/berkeley/adding_error_bars_audit.tsv` — flattened table
  (headline, pillar, n_for_CI, point_estimate, method, current_SE_or_CI,
  propagated_CI95, width_to_point, noise_source, verdict, explanation).
- `experiments/results/berkeley/adding_error_bars_summary.json` — full
  structure with audit row dicts + aggregate verdict.

Run: `python3 scripts/berkeley/adding_error_bars_to_evals.py` (~2 s).

The script implements: bootstrap percentile CIs on the mean (B=10 000),
paired-difference bootstrap CIs, OLS-slope bootstrap CIs in log10 space, and
Miller's equiv-region (TOST) test for any null=0 claim.

## Sharp findings on real Pillar evidence

| id | pillar | headline | n | current SE/CI | bootstrap CI95 | verdict |
|---|---|---|---|---|---|---|
| H1 | P3 | iter115 GU_ratio(G=4/G=32) at T=1M = 5.03 | 4 | NONE | [4.16, 4.81] | **DECISIVE** |
| H2 | P3 | iter131 retention(T) slope (log10 T) | 4 | NONE (point only) | [-0.237, -0.038] | **DECISIVE (slope<0)** |
| H3 | P3 | iter123 SNR slope in G | 4 | [+0.148, +0.583] | [+0.148, +0.583] | NULL (CI contains theory +0.500) |
| H4 | P3 | iter135 Native-Wu paired G=2 ~ G=16 | 3 | +/-0.0088 | [-0.0205, +0.0139] | NULL (equiv-region straddled) |
| H5 | P1 | iter137 R_max_2p vs log10(N) slope | 5 | OLS-only | [-0.796, +0.323] | NULL (no cross-anchor law) |
| H6 | P2 | iter130 AUROC(zvf_risk_max) | 52 | [+0.83, +1.00] | [+0.83, +1.00] | **DECISIVE** (excludes 0.5) |
| H7 | P4 | iter136 Dr.GR vs GR H3 late-eff | 5 | p_param=0.0312 | Cohen's d=+2.68 | **DECISIVE (p<0.05)** |

**Aggregate verdict: 4 DECISIVE, 0 SUGGESTIVE, 3 NULL.**

**Sharpest contributions.**

1. The Pillar 1 cross-anchor scaling law is officially NULL on this evidence
   base. After bootstrapping the OLS slope of `R_max_2p` against
   `log10(params_B)` across the 5 iter137 anchors, the 95% CI is
   [-0.796, +0.323] — it includes 0 by a factor of 2x, so the headline
   "+0.507 +/- 0.718" should be re-stated as "no detectable cross-scale law in
   the present anchor pool." Miller's rule (n=5 anchors cannot declare
   significance at 2*SE) makes the published OLS-only claim under-powered.

2. The Pillar 3 Native-Wu test (iter135) is the cleanest case where Miller's
   recipe PROHIBITS the headline conclusion. With n=3 paired seeds the 95% CI
   on the G=16-G=2 difference is [-0.0205, +0.0139]. The Wu 97.6% retention
   band corresponds to a difference of -0.024 — *below* the lower CI bound.
   The native-Wu claim "G=2 ~ G=16" is supported in **direction only**, not in
   magnitude. The headline should be downgraded to "SUGGESTIVE (equiv-region
   straddled)."

3. The Pillar 2 AUROC(zvf_risk_max) is the **only headline** in the audit
   that survives Miller's recipe cleanly because it already has a 52-seed
   pooled DeLong CI excluding 0.5 by 0.33. This is the model for the other 6
   claims.

4. Wang's own finding (prediction noise > data noise on most evals) explains
   why H4 (n=3 paired seeds, prediction-noise dominant) cannot match
   H6 (n=52 pooled seeds, data+prediction noise effectively averaged out).

## Recommended action

- **GO.** Apply the audit row table to the paper as a "Statistical Rigor
  Appendix" (one row per headline), and update the H1/H2/H5 headlines in the
  Pillar 1/3 text to declare their CIs.
- For H4 specifically: replace "Wu claim holds natively" with "Wu claim holds
  in direction (paired n=3, equiv-region straddled)." This converts a banner
  claim into a properly-CI'd result.
- For H3: keep the headline but add "CI for SNR slope contains theory +0.500;
  consistent with but does not establish the sqrt(G) law." (Miller's exact
  prescription: a CI containing the theoretical point is "consistent" not
  "validated.")

## Reproducibility

```bash
python3 scripts/berkeley/adding_error_bars_to_evals.py
# writes experiments/results/berkeley/adding_error_bars_audit.tsv (+ summary.json)
```

— end —
