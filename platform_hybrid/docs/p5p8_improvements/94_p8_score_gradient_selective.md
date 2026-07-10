# 94 — P8 score-stream gradient selective-LLM rule (iter 80 JOB A)

## Falsifiable headlines

- **H1 — gradient-band rule at g_thr=0.0001 (XGB-20raw backbone) catches 141 of 144 positives at K=2% recall@97.92% using 9 LLM calls.** The iter-76 row 89 absolute-band rule at width=0.10 catches the same 141 positives using **21 LLM calls** — gradient-band uses **57% fewer LLM invocations at matched recall**.
- **H2 — paired-bootstrap CI on Δrecall (gradient - absolute) at K=2% is [−0.030, +0.035]**, INDISTINGUISHABLE on recall itself. The LLM-call saving is the binding differential, not the recall.
- **H3 — cost-per-fraud-caught under gradient rule = $7.15e-3 vs absolute-band = $7.23e-3** (1.07% cheaper). Cost-per-decision = $0.0001008/dec vs $0.0001019/dec. The iter-76 row 89 "LLM-as-scribe surrogate statistically significantly more expensive at every K" finding is now broken: under selective gradient-band invocation, the LLM-as-scribe regime is statistically and economically equivalent to XGB-only.
- **H4 — score-stream gradient distribution on the top-2% (200 rows) of the score-sorted stream is statistically equivalent across backbones**: XGB-20raw mean grad=0.0047, XGB-24full mean grad=0.0047. The cheap backbone and the LLM-as-scribe surrogate rank the test rows with comparable steepness on the alert band. This is a **negative result** sharpening iter-76 row 89: there is no structure on the BACKBONE axis that the absolute-band rule could exploit but gradient-band cannot.

## Per-axis result table (XGB-20raw backbone)

| g_thr | n_llm | recall@K=2% | AUC | cost/dec | cost/fraud |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | 9 | 141/144 (97.92%) | 0.99977 | $1.008e-4 | $7.150e-3 |
| 0.0005 | 29 | 142/144 (98.61%) | 0.99977 | $1.026e-4 | $7.226e-3 |
| 0.001 | 50 | 142/144 (98.61%) | 0.99979 | $1.045e-4 | $7.359e-3 |
| 0.005 | 138 | 143/144 (99.31%) | 0.99984 | $1.124e-4 | $7.862e-3 |
| 0.01 | 174 | 143/144 (99.31%) | 0.99983 | $1.157e-4 | $8.088e-3 |
| 0.05 | 199 | 143/144 (99.31%) | 0.99984 | $1.179e-4 | $8.245e-3 |

## Cross-paper coupling

- (i) **iter-76 row 89 absolute-band baseline (P8)** — gradient-band rule scores the same recall with strictly fewer LLM calls; the iter-76 row 89 mint recommendation is closed.
- (ii) **iter-72 row 85 joint controller (P7)** — the gradient-band rule generalizes the joint controller's per-step logic to fraud detection: invoke expensive computation only where the cheap backbone's "knowledge plateau" indicates uncertainty.
- (iii) **iter-68 row 79 single-sensor ablation (P8)** — selective gradient-band rule replaces 4 single-aggregate features (V_mean, V_std, V_max, V_min) with a *single* gradient signal; consistent with iter-68 finding that V_std/V_max carry most of the signal.
- (iv) **iter-75 row 88 exact finite-pool contrast-preservation (P7)** — gradient rule approximates the "preserve contrast" intuition on the fraud axis: only invoke LLM where the cheap signal is too noisy to distinguish rows.

## Operational recommendation

Deploy **gradient-band (XGB-20raw backbone, g_thr=0.0001)** with the following properties on this corpus:
- AUC = 0.9998 (statistically equivalent to always-LLM)
- Recall@K=2% = 97.92% (141 of 144 positives caught)
- n_llm_calls = 9 of 10000 (0.09% invocation rate)
- Cost per decision = $0.000101 (vs $0.0001 XGB-only baseline)

## Reproducibility

- Script: `platform_modal/scripts/p5p8/p8_score_gradient_selective.py` (290 lines)
- Outputs:
  - `platform_hybrid/experiments/results/p5p8/p8_score_gradient_distribution.tsv` (398 rows)
  - `platform_hybrid/experiments/results/p5p8/p8_score_gradient_selective.tsv` (14 rows)
  - `platform_hybrid/experiments/results/p5p8/p8_score_gradient_vs_absband.tsv` (12 rows)
  - `platform_hybrid/experiments/results/p5p8/p8_score_gradient_boot.tsv` (7 rows)
  - `platform_hybrid/experiments/results/p5p8/p8_score_gradient_summary.json`
  - `platform_hybrid/experiments/results/p5p8/figures/p8_score_gradient.{png,pdf}`
- Seed: 20260705 (paired bootstrap B=600, 95% CI)

## Paper-facing text

Lifted into `platform_hybrid/paper/sections/p8_iter80_score_gradient.tex`.
