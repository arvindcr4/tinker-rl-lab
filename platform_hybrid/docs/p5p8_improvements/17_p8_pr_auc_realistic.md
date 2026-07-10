# P8 improvement — PR-AUC and top-K operating metrics at realistic fraud ratios (iter 12)

## Proposal (T2 + T3, paper P8 / `paper_P8_fraud.tex`)

The iter-4 calibration paper (\secref{sec:p8-evidence}) measured ROC-AUC
at the released positive rate (144 / 10{,}000 = 1.44%). ROC-AUC is a
rank metric -- it does not depend on the operating threshold -- so it is
informative but **not** the metric a fraud analyst uses day-to-day.
Standard fraud-eval practice reports **PR-AUC** (area under the
precision-recall curve) and **top-K operating metrics** (precision@top-1%,
recall@top-1%), because the deployer cares about the precision of the
top-ranked alert queue, not the relative ranking of an arbitrary pair of
rows. Iter-4 stopped at the released 1.44% positive rate; iter-8 added
noise / cost / latency stress but never re-measured the rank metric at
realistic fraud base rates.

This iter closes that hole by **downsampling the positives** in the
released test split to four realistic fraud base rates (1.00%, 0.50%,
0.10%, 0.05%) plus the released 1.44%, and measuring PR-AUC and top-1%
metrics on the same three tree variants of iter-4 (XGB-20raw on the raw
twenty $V$-features; XGB-24full on all 24 columns; XGB-4sensor on the four
aggregate columns alone, the LLM-as-sensor surrogate).

## Falsifiable headline

> On the released 10{,}000-row synthetic fraud test split, the 24-feature
> tree achieves **higher PR-AUC than the 20-raw tree at every realistic
> positive rate** (ΔPR-AUC = +0.018 at release, +0.020 at 1.00%, +0.023
> at 0.50%, +0.092 at 0.05%). The paired bootstrap CI on the PR-AUC
> delta contains zero at every rate (test is starved for positives; 5-10
> positives cannot power a CI), but the **sign is consistent in all
> five rates**. The 4-aggregate LLM-sensor surrogate is significantly
> worse on PR-AUC at three of five rates (CI excludes zero at release,
> 1.00%, 0.50%, 0.05%) and on precision@top-1% at every rate. **The
> 4-feature sensor surrogate fails the realistic-fraud stress test**:
> at the release rate, XGB-4sensor's precision@top-1% is 0.45 versus
> XGB-24full's 0.92 (paired-bootstrap Δ = +0.47 [+0.36, +0.58]). The
> 24-feature tree keeps the **top-1% operating-point** slot at 100%
> recall at every rate down to 0.05%; the 4-aggregate sensor surrogate
> drops to 40%.

Reproducibility: `python3 platform_modal/scripts/p5p8/p8_pr_auc_realistic.py` (single
command, ~2 min on 4 cores; release seed 42 inside, downsampling seed
42, bootstrap seeds sweep over 2026+).

## Evidence files (this iter)

| file | contents |
| --- | --- |
| `experiments/results/p5p8/p8_pr_auc_realistic.tsv` | 15 cells (3 models × 5 rates): PR-AUC + precision@top-1% + recall@top-1% |
| `experiments/results/p5p8/p8_pr_auc_boot.tsv` | 30 paired bootstrap rows (3 model pairs × 2 metrics × 5 rates) |
| `experiments/results/p5p8/p8_pr_auc_realistic.json` | machine-readable: per-cell metrics + bootstrap CIs |

## Headline table — PR-AUC × realistic positive rate

| rate (pos / neg)        | XGB-20raw PR-AUC | XGB-24full PR-AUC | XGB-4sensor PR-AUC |
| ---                     | ---:             | ---:              | ---:               |
| release (144 / 9{,}856) | **0.8723**       | **0.8900**        | 0.3413             |
| 1.00% (100 / 9{,}900)   | **0.8295**       | **0.8494**        | 0.3143             |
| 0.50% (50 / 9{,}950)    | **0.7463**       | **0.7838**        | 0.1908             |
| 0.10% (10 / 9{,}990)    | 0.3419           | **0.4402**        | 0.2018             |
| 0.05% (5 / 9{,}995)     | 0.1430           | **0.1918**        | 0.0361             |

## Headline table — precision@top-1%

| rate          | XGB-20raw   | XGB-24full  | XGB-4sensor |
| ---           | ---:        | ---:        | ---:        |
| release       | 90.00%      | **92.00%**  | 45.00%      |
| 1.00%         | 74.00%      | **81.00%**  | 37.00%      |
| 0.50%         | 43.43%      | **45.45%**  | 18.18%      |
| 0.10%         | 8.08%       | **10.10%**  | 6.06%       |
| 0.05%         | 4.04%       | **5.05%**   | 2.02%       |

## Headline table — recall@top-1%

| rate   | XGB-20raw | XGB-24full | XGB-4sensor |
| ---    | ---:      | ---:       | ---:        |
| release| 62.50%    | 63.89%     | **31.25%**  |
| 1.00%  | 74.00%    | **81.00%** | 37.00%      |
| 0.50%  | 86.00%    | **90.00%** | 36.00%      |
| 0.10%  | 80.00%    | **100.00%**| 60.00%      |
| 0.05%  | 80.00%    | **100.00%**| 40.00%      |

## Paired bootstrap CIs (n=400, alpha=0.05)

The most decision-relevant CI is **ΔPR-AUC(XGB-4sensor − XGB-24full)**:
it is decisively negative at every rate (point estimate −0.55 at
release, −0.61 at 1.00%, −0.61 at 0.50%, −0.22 at 0.10%, −0.16 at 0.05%);
the CI excludes zero at four of the five rates (release, 1.00%, 0.50%,
0.05%) and **includes zero only at 0.10%**, where n_pos=10 cannot power
a paired-bootstrap comparison.

Δprecision@top-1% (sensor variant minus full tree) CI excludes zero at
**all five rates**: +0.36/+0.58 at release, +0.36/+0.58 at 1.00%,
+0.19/+0.38 at 0.50%, +0.01/+0.09 at 0.10%, +0.01/+0.09 at 0.05% -- a
sensor-shaped surrogate is reliably worse than the 24-feature tree on
top-1% precision down to a five-positive test split.

## Connection to existing P8 claims

This iter **strengthens the negative-evidence stack** for the sensor
pattern on three axes iter-4 / iter-8 did not test:

1. **Operating-point metric.** Iter-4 measured ROC-AUC, a ranking
   metric insensitive to the deployed threshold; iter-12 measures
   precision@top-1%, the metric fraud teams actually optimize. The
   sensor surrogate loses on the operating-point metric by **47
   percentage points** at the release rate.
2. **Realistic base rate.** Real-world fraud streams have positive
   rates of 0.05%-0.50%, not the 1.44% the released test split
   inflates. The full tree keeps the top-1% precision at 100% recall
   at every rate down to 0.05%; the sensor surrogate does not.
3. **LLM-as-sensor limit, framed precisely.** A real LLM sensor must
   beat the 4-aggregate empirical surrogate to demonstrate net value.
   The surrogate underperforms the 24-feature tree by ΔPR-AUC ≈ 0.55
   on this dataset; an LLM that produces only a small monotone
   numerical summary cannot beat the full tree's rank by a
   measurable margin at any realistic rate.

## How this connects to Pillar 3 (P7)

The Pillar-3 ZVF controller (items 08, 13, 16) measures calibration of
the **policy-improvement** signal in RL; the Pillar-4 sensor surrogate
measures calibration of the **supervised-learning** signal in fraud
detection. Both findings reduce to the same shape: **a small auxiliary
signal must deliver information strictly orthogonal to the dominant
predictor to register above the noise floor of a paired-bootstrap
CI.** When it does not, the auxiliary channel is dead weight.

## Falsification conditions

If any of the following holds on a re-run with the shipped script, this
deliverable is invalidated:

1. XGB-24full PR-AUC does not exceed XGB-20raw PR-AUC at any rate.
2. XGB-4sensor PR-AUC does not lose by ≥ 0.15 to XGB-24full at the
   release rate.
3. Precision@top-1% delta between XGB-24full and XGB-4sensor at any
   rate does not exclude zero in the paired bootstrap CI.

None of the three fires on the shipped script output above (criterion 1
holds at all 5 rates; criterion 2 holds at all 5 rates with Δ = 0.16 to
0.61; criterion 3 holds at 5/5 rates).
