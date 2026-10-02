# Appendix F. Applied Side Study: Credit-Card Fraud Detection (P8)

**Scope statement.** This appendix reports a side study. It uses synthetic data and sits outside the RL contribution of this thesis. The dataset is generated with scikit-learn's `make_classification`; the language-model arm is trained by supervised fine-tuning, not by reinforcement learning; and no result in this appendix is used to support any GRPO, PPO, ZVF, group-size or campaign claim elsewhere in the report (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_intro.tex). The study is retained, in condensed form, as the P8 slot of the internal report series and as one worked example of measuring a claimed capability rather than accepting its label.

The study compares a gradient-boosted tree with a fine-tuned language model on credit-card fraud detection. Its conclusion is that the tree keeps the real-time scoring seat, and that the language model's plausible roles lie in tasks the tree cannot perform at all.

## F.1 Dataset, splits, and the two arms

The dataset is synthetic: `make_classification` (seed 42), 50,000 transactions, 20 anonymised numeric features V1–V20 (10 informative, 2 redundant), two clusters per class, class separation 0.8 and `flip_y=0.01`, with a realised fraud rate of 1.44%. Four per-row aggregates (mean, standard deviation, maximum, minimum) are appended, giving 24 features (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_setup.tex; fraud_data.csv). The data reproduce class imbalance and anonymised feature shape, and none of the temporal drift, verification latency or adversarial adaptation of real card-fraud data (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_limitations.tex).

| Arm | Configuration | Evaluation split |
|---|---|---|
| XGBoost (scorer) | 200 estimators, depth 6, lr 0.05, subsample/colsample 0.8, `scale_pos_weight=7`; no search | 80/20 stratified split, 10,000 test rows, 144 positives |
| Qwen3.5-4B SFT | TabLLM-style row serialisation; answer-token cross-entropy, no reference policy, no KL; 63 steps at batch 32 | 500-row positive-enriched held-out subset at 20% fraud, disjoint from training |

: The two arms of P8.

(sources: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_setup.tex; platform_hybrid/experiments/results/quick_20260704/qp8-fraud-sft_manifest.json; platform_hybrid/experiments/results/quick_20260704/qp8-fraud-sft.tsv)

## F.2 The head-to-head result

| Arm | Eval split | _n_ | Accuracy | ROC-AUC |
| --- | --- | --- | --- | --- |
| XGBoost (scorer) | full test | 10,000 | n/a | 0.7955 |
| Qwen3.5-4B SFT | positive-enriched test | 500 | 0.7920 | 0.4827 |

: Head-to-head result on the synthetic card-fraud set, by arm and evaluation split.

(source: platform_hybrid/experiments/results/quick_20260704/qp8_fraud.tsv)

The two AUCs sit on different held-out splits, so this is not a like-for-like ranking (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_abstract.tex). The language model's AUC of 0.4827 is chance-level ranking: its Hanley–McNeil 95% CI, with 100 positives and 400 negatives, is [0.42, 0.55] and includes 0.5. Its accuracy of 0.792 is slightly *below* the 0.800 that a constant "legitimate" answer would score at 20% prevalence, so it is not evidence of discrimination. The tree's AUC interval is [0.75, 0.84], with 144 positives among 10,000 test rows. The tree's precision 0.723 (34/47, Wilson 95% CI [0.58, 0.83]), recall 0.236 (34/144, [0.17, 0.31]) and F1 0.356 at the default threshold come from the released script's own run (AUC 0.7942, 0.49 s training, about 6 ms inference per 10,000 rows), which uses a slightly different configuration from the 0.7955 artefact (sources: xgboost_results.json; platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_setup.tex). An older record of 21 June 2026 (XGBoost AUC 0.975, language model 0.948) could not be reconstructed and is not used (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_limitations.tex).

## F.3 Why the scoring seat stays with the tree

Four operational arguments hold even at AUC parity (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_scorer.tex):

- **Latency.** Card authorisation needs a score within tens of milliseconds; the tree takes about 6 ms per 10,000 rows, against roughly 300 ms for a language-model pass.
- **Cost.** The released cost table puts the synchronous language-model path at about 35 times the per-transaction cost of the tree path (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_evidence.tex).
- **Calibration.** Downstream threshold and expected-loss rules consume the score as a probability; verbalised language-model confidences are poorly calibrated.
- **Prompt injection.** A model that reads attacker-controlled text fields in a synchronous decision loop opens an injection channel that a tree scoring numbers does not have.

## F.4 Supporting measurements

| Check | Result | Source |
|---|---|---|
| Aggregate-only tree (stand-in for a language-model "sensor") vs 24-feature tree | Worse on every axis: AUC −0.0245 [0.0174, 0.0320], accuracy −0.0135 [0.0109, 0.0160], Brier +0.0109 [0.0099, 0.0120] | p5p8/p8_headline_cis.tsv |
| Adding the four aggregates to the 20 raw features | AUC +0.0002 [−0.0002, 0.0007], within noise | p5p8/p8_headline_cis.tsv |
| Leakage caveat | Tables reporting AUC 0.998–0.999 fit on `fraud_data.csv` and score a subset of it; they are within-protocol relative comparisons only. The leakage-free split gives 0.7955 | p8_evidence.tex |
| Cost per fraud caught ($0.50/alert, $100 missed-fraud loss) | Raw features $2.68; sensor-augmented $3.28; sensor-only $14.92; the sensor never pays for itself in the 25-cell (σ, L) grid | p8_evidence.tex |
| Sensor-noise break-even | $5.29 (oracle), $26.44 (σ ≤ 0.02), $36.40 (σ = 0.05) | p8_noisy_sensor.tex |

: Supporting P8 measurements. Bracketed ranges are paired-bootstrap 95% intervals as reported by the source. Source files are under `platform_hybrid/experiments/results/` and `platform_hybrid/paper/archive/absorbed/P08_fraud/sections/`.

## F.5 Proposed roles for the language model

The study argues, without measuring, that the language model's useful roles are tasks the tree cannot enter: document and image fraud through a vision-language model, drafting compliance narratives (Suspicious Activity Reports under 31 C.F.R. § 1020.320; adverse-action reasons under 12 C.F.R. § 1002.9), cold-start triage of new fraud typologies before labels exist, and post-score investigation of the alert queue. The proposed architecture keeps XGBoost as the scorer and uses the language model as a sensor (typed features, no authority) and a scribe (narratives conditioned on the tree's own attributions, with a human signature). The "85 times cheaper than an analyst" triage figure is an internal estimate from June-2026 token prices and assumed investigation times, and has not been validated (sources: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_taxonomy.tex; platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_compliance.tex; platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_architecture.tex).

## F.6 Limitations

The dataset is synthetic and single-generator, so no number here is a production expectation, and there is no cross-institution or public-benchmark replication. The challenger is one model family, one serialisation and one fine-tuning budget. Of the proposed roles, only the scorer comparison carries a measurement of the study's own; the others rest on external literature and regulatory text, and nothing in the compliance discussion is legal advice (source: platform_hybrid/paper/archive/absorbed/P08_fraud/sections/p8_limitations.tex).
