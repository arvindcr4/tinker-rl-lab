# Item 40 — P8 noisy-sensor cost-optimal robustness (iter 32)

## Pick (vein (c) of brief, not in prior ledger)

The cost-optimal frame in item 35 (iter 28) used an *oracle* 4-aggregate
sensor surrogate (the LLM emits the exact $V_\text{mean}, V_\text{std},
V_\text{max}, V_\text{min}$ per row). Item 10 (iter 4) certifies a
sensor-noise budget $\sigma \le 0.02$ on those aggregates: a real LLM
sensor cannot emit the exact value, only an estimate within roughly
$0.02$ standard units. Iter 28's reviewer-facing headline ("the
oracle cost-optimal break-even is at $L^\star \approx \$5.5$") does
not address whether that break-even survives the noise a real LLM
sensor must contend with. This iter closes the loop by re-running the
entire cost-optimal frame at five sensor-noise levels
$\sigma \in \{0.0, 0.005, 0.010, 0.020, 0.050\}$ on the same held-out
split, refitting \textsc{24full} on a noisy training set and scoring on
a noisy test set, and re-computing the cost-optimal threshold and the
break-even fraud loss $L^\star$.

## Method

Inputs:

- `fraud_data.csv` — 50,000 synthetic fraud rows (24 numeric features + Class)
- `test_data.csv` — 10,000 held-out rows (same schema + Class)

Per $\sigma$:

1. Add Gaussian noise $N(0, \sigma)$ to the four aggregate columns
   ($V_\text{mean}, V_\text{std}, V_\text{max}, V_\text{min}$) on
   *both* the training and the test set (deterministic seed
   `20260704`).
2. Refit `XGBClassifier(n_estimators=200, max_depth=4, lr=0.1,
   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1, tree_method=hist,
   random_state=42)` on the noisy 24-feature training set.
3. Score the noisy 24-feature held-out set.

Per $(\sigma, \rho)$, where $\rho = L / C_\text{inv}$ is the cost ratio
(we use $C_\text{inv} = \$0.50$/alert, $c_\text{sense} = \$0.0035$/row
for any sensor-augmented tree):

1. For each tree, run `optimal_cut` (item 35, iter 28) to find the
   rank cutoff minimizing $C_\text{inv} \cdot \text{alerts} + L \cdot
   \text{FN}$ on the held-out split.
2. Compute $\Delta$ cost/dec = (cost of noisy-24full + sensing
   charge) − cost of 20raw (no sensing charge).
3. Run paired bootstrap ($n=400$, seed `20260704`) over row indices,
   re-fitting the optimal cut at each replicate.

We additionally compute the break-even $L^\star$ on the *exact*
(no-bootstrap) cost-min gap: the smallest $L$ at which the
sensor-augmented tree becomes cheaper than the raw-features tree.

## Headline (item 40)

| $\sigma$ | $L^\star$ (USD) | $\rho^\star$ | sensor wins (CI) | sensor loses (CI) | neither |
|---------:|----------------:|-------------:|-----------------:|------------------:|--------:|
| 0.000    | **5.29**        | 10.6         | 0/8              | 2/8               | 6/8     |
| 0.005    | **5.09**        | 10.2         | 0/8              | 2/8               | 6/8     |
| 0.010    | **5.09**        | 10.2         | 0/8              | 2/8               | 6/8     |
| 0.020    | **26.44**       | 52.9         | 0/8              | 2/8               | 6/8     |
| 0.050    | **36.40**       | 72.8         | 0/8              | 2/8               | 6/8     |

**Headline falsifiable claim:** *The oracle cost-optimal break-even
$L^\star \approx \$5.3$ (iter 28) does not survive item-10's noise
budget.* At $\sigma=0.02$ (the noise floor a real LLM must contend
with per item 10), the break-even jumps to $\$26.4$ — a **5× rise**.
At $\sigma=0.05$ it climbs further to $\$36.4$ — a **7× rise**. The
CI signature (0/8 ρ certify sensor wins; 2/8 certify sensor loses at
low ρ) is *unchanged* across $\sigma$, so the noise-induced
break-even drift is invisible to the paired-bootstrap CI at discrete
$\rho$ — a paper that only reports CIs would have missed it.

## What this adds to the iter-28 result

Iter 28's "What the cost-optimal frame settles" paragraph ended with
*"the LLM sensor is worth deploying only where the missed-fraud loss
is large and the raw scorer is weak."* Iter 32 sharpens that sentence:
the missed-fraud loss must be *much* larger than the oracle break-even
suggests — specifically, 5× larger under item-10's noise budget. A
fraud-ops team that reads only iter 28 might deploy the sensor when
$L \ge \$5$; a team that reads iter 32 should not deploy until
$L \ge \$26$ (under $\sigma \le 0.02$). This is the first item in the
P8 ledger that quantifies the cost-of-noise directly through the
break-even $L^\star$.

## Cross-paper connection

Item 9's sensor-noise budget $\sigma \le 0.02$ was the *prior* on
real-LLM emission quality. Iter 32 turns that prior into a
*posterior* on the cost-optimal break-even and shows the prior matters
operationally. The same methodology transfers to P7's controller: if a
saturating-on-real-data controller uses an LLM-estimated ZVF, the
trigger threshold $\tau$ should be inflated to absorb the sensor
noise. Future P7 iters may use this analysis as a template.

## Artifacts

- `scripts/p5p8/p8_noisy_sensor_cost.py` (~245 LoC, stdlib +
  numpy + pandas + xgboost + matplotlib)
- `experiments/results/p5p8/p8_noisy_sensor.tsv` (40 rows: 5 σ × 8 ρ
  exact cost-optimal)
- `experiments/results/p5p8/p8_noisy_sensor_boot.tsv` (40 rows: same
  shape with paired bootstrap CI)
- `experiments/results/p5p8/p8_noisy_sensor_breakeven.tsv` (5 rows: L*
  per σ)
- `experiments/results/p5p8/p8_noisy_sensor_summary.json` (machine-readable)
- `experiments/results/p5p8/figures/p8_noisy_sensor.{png,pdf}`
  (Δ-cost vs ρ, one curve per σ)
- `paper/sections/p8_noisy_sensor.tex` (new §4.7, ~150 lines)
- `paper/paper_P8_fraud.pdf` rebuilds to 22 pages / 0 errors / 0
  undefined citations

## Verified citations

No new citations needed — this item reuses the iter-28 cost matrix
and the iter-4 sensor-noise budget.