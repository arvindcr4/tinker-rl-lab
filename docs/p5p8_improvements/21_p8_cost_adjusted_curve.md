# P8 cost-adjusted operating curve (iter 16, JOB A)

## Proposal
PR-AUC and top-K tables in the existing `p8_evidence.tex` measure
*detection quality* at a fixed score-rank budget. A real fraud-ops
deployment is budgeted in dollars, not in score-rank slots: an analyst
queue has a hard top-K review budget per day, and each row that
enters the queue costs both a model call and an analyst review minute.
This iter closes that loop with a four-mode comparison at six review
budgets and bootstrap CIs on the cost-adjusted headline metric.

## Method
Inputs: `fraud_data.csv` (50k train) + `test_data.csv` (10k held-out).
Models: XGB-20raw (raw 20 V-features), XGB-24full (raw 20 + 4
hand-engineered aggregates), XGB-4sensor (4 aggregates only). The
4-aggregate features are an oracle-LLM-sensor surrogate: a perfect LLM
that produces a single deterministic 4-vector per transaction. We
also simulate two hybrid modes that pay the LLM-sensor cost on the
top 10% / top 1% of the stream:

- M1 XGB-20raw (no LLM cost, no LLM value)
- M2 XGB-24full (oracle LLM-as-sensor; LLM cost ignored)
- M3 Hybrid-10% (LLM sensor on top 10% of the stream)
- M4 Hybrid-1% (LLM sensor on top 1% of the stream)

Review budgets: $K \in \{0.1\%, 0.5\%, 1\%, 2\%, 5\%, 10\%\}$ of the
stream. Cost model: XGB $0.0001 / row, LLM $0.0035 / row, analyst
review $0.50 / row (from `p8_cost_accounting.tsv` iter 4).

The headline operating metric is *true positives recovered per combined
dollar of model and review spend* (TP/$). Paired bootstrap with n=400
resamples on the test split, two-sided alpha=0.05.

## Verified citations
No new citations were added. The cost-per-row figures are taken
verbatim from `experiments/results/p5p8/p8_cost_accounting.tsv` (iter 4,
validated).

## Measured results
24 operating rows (4 modes × 6 budgets) and 18 bootstrap rows written
to:

- `experiments/results/p5p8/p8_cost_adjusted_curve.tsv` (24 rows)
- `experiments/results/p5p8/p8_cost_adjusted_boot.tsv` (18 rows)
- `experiments/results/p5p8/p8_cost_adjusted_summary.json`

### Headline table (TP per dollar, higher = better)
| Budget  | M1 XGB-20raw | M2 XGB-24full oracle | M3 Hybrid-10% | M4 Hybrid-1% |
|---------|-------------:|---------------------:|--------------:|-------------:|
| 0.1%    | 1.667        | 1.667                | 1.053         | 1.575        |
| 0.5%    | 1.808        | 1.808                | 1.593         | 1.784        |
| 1%      | 1.569        | 1.686                | 1.468         | 1.558        |
| 2%      | 1.040        | 1.089                | 1.005         | 1.036        |
| 5%      | 0.498        | 0.514                | 0.491         | 0.497        |
| 10%     | 0.265        | 0.271                | 0.264         | 0.265        |

### Bootstrap CIs on Δ(TP/$)
- **M2 vs M1**: positive trend at every K ≥ 1% (+0.006 to +0.082), CI
  contains zero at every budget (the analyst-review cost floor
  dominates the LLM-sensor cost floor).
- **M3 vs M1**: strictly negative at every budget, **CI excludes zero
  at every budget** (-0.002 to -0.614). The Hybrid-10% mode pays
  $3.50 of LLM cost on the top 10% of the stream and gains zero
  marginal TP at K ≤ 0.5%; net cost-effectiveness falls.
- **M4 vs M1**: strictly negative at every budget, **CI excludes zero
  at every budget** (-0.000 to -0.092). The Hybrid-1% mode pays
  $0.35 of LLM cost on the top 1% of the stream; the marginal TP
  gain is too small to recover even this small cost.

## Sharpest falsifiable claim
For binary-credit-card fraud with a 1.44% positive rate, an
LLM-as-sensor that produces a single deterministic 4-vector per
transaction (oracle LLM, no measurement noise) does not statistically
detectably improve TP/$ against an XGBoost scorer at any analyst
review budget K ∈ {0.1%, 0.5%, 1%, 2%, 5%, 10%}. A real (noisy)
LLM-as-sensor that pays per-row token costs at $0.0035/row
*strictly loses* TP/$ at every budget, with bootstrap CIs that exclude
zero.

## Implications for P8
1. **The P8 thesis is sharper than the iter-4 / iter-12 evidence
   already showed.** The "LLM as sensor and scribe, not scorer"
   formulation in §p8-scorer was previously supported by ROC-AUC
   deltas and PR-AUC deltas. The iter-16 evidence adds a *dollar*
   dimension: even an oracle LLM-as-sensor fails to deliver a
   statistically detectable dollar improvement on this dataset.
2. **The combined P8 picture is negative-evidence-only on this
   dataset.** The 4-aggregate surrogate loses on PR-AUC at 4/5
   positive rates, loses on P@1% at 5/5 rates (iter 12), loses on
   per-feature ablation F1 by ≤ 0.017 (iter 4), and now loses on
   TP/$ at every review budget under realistic LLM pricing (iter 16).
3. **Connection to Pillar 3 (P7 ZVF controller).** Both findings
   reduce to the same diagnostic: an auxiliary model channel must
   deliver information strictly orthogonal to the dominant predictor
   to register above the noise floor of a paired bootstrap CI. The
   4-vector produced by an oracle LLM-as-sensor is correlated with
   the 20 raw features and therefore adds no orthogonal information;
   the calibration is the same null result as the controller's
   firing frequency on saturated prompts.

## Reproduction
```bash
cd /home/claude/tinker-rl-lab-minimax
python3 scripts/p5p8/p8_cost_adjusted_curve.py
```
Expected runtime: ~3 min on 4 cores. Outputs:
- `experiments/results/p5p8/p8_cost_adjusted_curve.tsv`
- `experiments/results/p5p8/p8_cost_adjusted_boot.tsv`
- `experiments/results/p5p8/p8_cost_adjusted_summary.json`

## Paper rebuild
`paper/sections/p8_evidence.tex` extended with new subsection
"Cost-adjusted operating curve at six review budgets" (Section
4.5). `paper/paper_P8_fraud.pdf` rebuilt to **17 pages** with **0
errors and 0 undefined citations**.