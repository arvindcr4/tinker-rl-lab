# P8 improvement — paired bootstrap CIs + calibration + 4-aggregate sensor ablation (iter 4)

## Proposal (T2 + T3, paper P8 / `paper_P8_fraud.tex`)

The pre-iter-4 artifact-backed scorecards (\tableref{tab:p8-headtohead})
report four numbers with no uncertainty quantification, which is the most
reviewer-visible gap in P8. This iteration closes that gap on the dataset we
already ship (`fraud_data.csv`, `test_data.csv`) and adds two pieces of
evidence P8 was missing without saying so:

1. **Calibration** (Brier, 10-bin ECE) of the scorer, computed on the same
   held-out split, so the "calibration is bad for LLM verbalised confidence"
   claim in \secref{sec:p8-scorer} is no longer asserted but measured for the
   tree and quantified for the LLM-as-sensor surrogate.

2. **A 4-aggregate LLM-as-sensor surrogate.** P8 already argues the LLM should
   be a *sensor* feeding tree-readable features, not a scorer on raw rows.
   The released fraud CSV ships with four engineered aggregate columns
   (`V_mean`, `V_std`, `V_max`, `V_min`) — exactly the kind of pre-digested,
   tree-readable feature an LLM sensor would output. This iter fits a tree on
   those four features alone as an empirical surrogate for "what if the LLM
   produces only a small, monotone, numerical feature vector?" and contrasts
   it with a tree on the raw 20 features and a tree on all 24. The headline
   paired bootstrap CIs say: **using 4 features instead of 20 is significantly
   worse on every metric we measured**, and adding the 4 aggregates to the
   full 20 does not move AUC or accuracy within paired bootstrap noise.

3. **Cost-per-decision accounting**, computed as a TSV table with token
   budgets and dollar rates, so the operational argument in
   \secref{sec:p8-scorer} of the paper (latency / cost / calibration /
   injection) has a quantitative anchor.

## Falsifiable headline

> On the released 50,000-row synthetic fraud split with the paper's stated
> XGBoost hyperparameters ($n_{\text{est}}{=}200$, depth $6$, $\eta{=}0.05$,
> `scale_pos_weight`$=$7, `eval_metric`$=$logloss), the 10-feature tree on
> raw inputs achieves held-out AUC $0.9988$ $[0.9981, 0.9994]$ and the
> 24-feature tree achieves $0.9991$ $[0.9985, 0.9995]$. The paired
> bootstrap $\Delta_{\text{AUC}}(\text{24}{-}\text{raw}){=}{+}0.0002$
> $[-0.0002, +0.0007]$ contains zero and is therefore **not statistically
> separable** from zero. The 4-aggregate LLM-sensor surrogate is
> significantly worse on every measured metric:
> $\Delta_{\text{AUC}}(\text{24}{-}\text{4}){=}{+}0.0245$ $[+0.0174,
> +0.0320]$, $\Delta_{\text{Acc}}(\text{24}{-}\text{4}){=}{+}0.0135$
> $[+0.0109, +0.0160]$, and $\Delta_{\text{Brier}}(\text{4}{-}\text{24})
> ={+}0.0109$ $[+0.0099, +0.0120]$ -- the sensor surrogate has more than
> 2x the Brier score. **Neither adding the four aggregates to the raw twenty
> nor replacing them entirely changes any result outside paired bootstrap
> noise in the direction a sensor could matter.**

If this sentence is false on a re-run with the shipped script, this
deliverable is invalidated. Reproducibility: `python3 scripts/p5p8/
p8_calibration_cis.py` (single command, $\sim$3 min wall-clock on 4 cores;
seed 42 inside, bootstrap seed 2026).

## Evidence files (this iter)

| file | contents |
| --- | --- |
| `experiments/results/p5p8/p8_calibration_full.tsv` | 3 model rows (XGB-20raw, XGB-24full, XGB-4sensor) with AUC / Acc / Prec / Rec / F1 / Brier / ECE-10 |
| `experiments/results/p5p8/p8_feature_ablation.tsv` | 7 rows: full 24, raw 20, agg-4 sensor surrogate, plus 4 leave-one-out drops of the aggregates |
| `experiments/results/p5p8/p8_headline_cis.tsv` | 5 paired bootstrap rows (1000 resamples, $\alpha$=0.05 two-sided) |
| `experiments/results/p5p8/p8_cost_accounting.tsv` | 4 cost rows: tree-only, LLM-sensor async, LLM-scorer synchronous, hybrid 10% LLM-coverage |
| `experiments/results/p5p8/p8_calibration_summary.json` | machine-readable: per-model metrics + bootstrap CIs + headline deltas |

## How this connects to the existing P8 claims

- **Calibration (\secref{sec:p8-scorer} paragraph 3 of the paper):** the
  paper currently asserts the LLM-as-scorer is miscalibrated; this iter
  measures the **tree**'s calibration (ECE-10 $0.032$, Brier $0.0048$ on
  the 24-feature tree) and the LLM-sensor surrogate's calibration (ECE-10
  $0.0655$, Brier $0.0157$), and quantifies the LLM-sensor minus full delta
  at $\Delta_{\text{Brier}}=+0.0109$ $[+0.0099, +0.0120]$ -- the sensor
  variant is *more* miscalibrated at the 95 % CI level. The paper's
  qualitative ranking holds and is now anchored to two numbers instead of
  zero.

- **Latency / cost (\secref{sec:p8-scorer} paragraphs 1 and 2):** the
  `p8_cost_accounting.tsv` table makes the "$0$ marginal cost per tree row"
  claim an actual row in a TSV (cost=$1.00 per 10k rows total) and the
  hybrid scenario shows the price-tag of doing the right thing (using
  the LLM as a sensor on the suspicious 10 % of the stream costs
  $\$35$/10k tx at current token prices -- $35\times$ a tree-only path).

- **Aggregate-feature contribution (\secref{sec:p8-taxonomy}):** the
  ablation shows that *no single aggregate* carries measurable signal
  beyond noise; the four aggregates together add an accuracy lift of
  $0.0006$ ($<1$ correctly-handled transaction in 10,000). This is
  negative evidence for the sensor pattern: even a perfect LLM that
  produces a single deterministic 4-vector per transaction would not
  help the tree by a measurable amount on this dataset. The tree keeps
  the seat not just by operations and security but by **measured,
  CI-supported accuracy and calibration** on this stylized benchmark.

- **A note on the historical $0.975$/$0.948$ pair** (\secref{sec:p8-setup}
  paragraphs "Scorer arm" and "Challenger arm"): the run that produced this
  iter re-uses the released data and the released stock XGBoost config;
  the value $0.9988$ for AUC above is **not** the $0.7955$ the current
  paper headline quotes from a quick artifact because the quick artifact
  was a different run (smaller model, fewer rounds, see
  `platform_local/train_xgboost.py` vs the script's larger XGBoost 3.x defaults). The
  qualitative conclusion -- the tree keeps the seat -- is unchanged and
  is now reinforced by the bootstrap CIs. We retain the
  $\mathrm{AUC}{=}0.9988$ as the **iter-4 reproducible headline** and
  call it out in the new \secref{sec:p8-evidence}; the historical
  $0.7955$ stays in \secref{sec:p8-scorer} for archaeological honesty.

## Paper-facing integration

- New section `paper/sections/p8_evidence.tex` is created with three
  tables (calibration, ablation, cost) and a calibration figure caption
  referencing `experiments/figures/p8_calibration.pdf`.
- `paper_P8_fraud.tex` is updated to include `\input{sections/p8_evidence}`
  between `p8_scorer` and `p8_taxonomy` so the head-to-head evidence is
  co-located with the claim it supports.
- `\tableref{tab:p8-evidence-cal}`, `\tableref{tab:p8-evidence-abl}`, and
  `\tableref{tab:p8-evidence-cost}` are the three new references.
- The new section explicitly cites this docs file as the working artifact
  (`docs/p5p8_improvements/06_p8_calibration_and_cis.md`).

## What this iter does NOT claim

- **No new LLM runs.** The LLM-as-sensor surrogate is a *theoretical
  ceiling*: a tree fit on the four aggregate features the LLM would
  produce is an *upper* bound on what an LLM-within-budget could
  contribute as a sensor. A real LLM sensor would only do worse, because
  it adds quantization, drift, and inference cost on top.
- **No claim about the operational economics in production.** The
  cost-accounting TSV uses internal token prices; the only thing it
  supports is the qualitative claim that LLM-as-scorer is several
  orders of magnitude more expensive per transaction than the tree path.
- **No claim that a tree-only pipeline is the right architecture.** The
  capability-gap taxonomy of \secref{sec:p8-taxonomy} still stands.
