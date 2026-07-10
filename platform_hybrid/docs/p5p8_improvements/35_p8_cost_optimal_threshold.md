# P8 expected-cost-optimal decision threshold (iter 28, JOB A)

## Proposal
Every prior P8 cost artifact fixes *how much* fraud review to buy and then
measures detection quality: item 21 reports TP-per-dollar at six fixed top-K
review budgets; item 27 sweeps 20 thresholds for precision/recall/F1; items 9/17
report ROC/PR-AUC and top-K. None answers the question a fraud-ops lead actually
optimises: **given a cost matrix (investigation cost `C_inv` per alert, loss `L`
per *missed* fraud), which threshold `tau* = argmin_tau E[cost]` minimises
dollars-per-decision — and does the LLM-as-sensor aggregate block lower that
minimum, net of the per-row LLM sensing cost?**

This closes the decision-theoretic loop: it selects the operating point by
expected monetary cost, sweeps the cost ratio `rho = L / C_inv`, and puts paired
bootstrap CIs on the *cost advantage* of the sensor feature sets.

## Method
- Inputs: `fraud_data.csv` (50k train), `test_data.csv` (10k held-out, 1.44% pos).
- Three XGBoost trees (`n_estimators=200, max_depth=4, lr=0.1, subsample=0.8`,
  seed 42 — identical to the iter-4/20 recipe):
  - `XGB-20raw`  : V1..V20 (no LLM cost)
  - `XGB-24full` : V1..V20 + 4 aggregates (oracle LLM sensor surrogate; pays
    `c_sense`)
  - `XGB-4sensor`: 4 aggregates only (pays `c_sense`)
- Cost per decision (USD): `C(m,tau) = c_sense_m + [C_inv*(TP+FP) + L*FN]/N`.
  Caught fraud costs `C_inv` (investigation); a false alert costs `C_inv`; a
  **missed** fraud costs `L`. `tau*` is found by exact argmin over all 10k rank
  cutoffs (O(N) via cumulative TP/FP).
- Cost figures from `p8_cost_accounting.tsv` (iter 4): `C_inv = $0.50`/alert
  (analyst review), `c_sense = $0.0035`/row (Qwen3.5-4B sensor).
- `rho` grid = {2,5,10,20,50,100,200,500}; break-even `L*` from a fine
  L∈[0.5,400] step-$0.50 sweep. Paired bootstrap n=400 (seed 2026, α=0.05
  two-sided): fix each model's `tau*` (the deployed policy), resample the 10k
  test rows, recompute realised per-decision cost, take the paired difference.

## Verified citations
No new citations added. Cost-per-row and tree config are taken verbatim from
already-validated worktree artifacts (`p8_cost_accounting.tsv`,
`p8_threshold_calibration.py`).

## Measured results (`p8_cost_optimal{,_boot,_breakeven}.tsv`, `_summary.json`)

**1. The cost-optimal threshold is strongly cost-ratio-dependent — a fixed
threshold is never optimal.** For `XGB-24full`, `tau*` falls monotonically from
0.175 at `rho=2` to 0.014 at `rho=500`, and optimal recall climbs 0.61 → 0.97.
Higher missed-fraud loss pushes the optimum toward more alerts, exactly as
decision theory predicts. The paper should report the operating point as a
function of `rho`, not a single number.

**2. The 4 LLM-sensor aggregates alone are cost-catastrophic — they cannot carry
the scoring load.** `XGB-4sensor` is *detectably* more expensive than both other
trees at **8/8** cost ratios (paired-bootstrap CI excludes zero at every `rho`).
At `rho=100` it forces a 39.3% alert rate ($0.240/decision) versus `XGB-24full`'s
10.8% ($0.088/decision) — **2.7× more expensive** — for the same 94–96% recall.
This is the sharpest quantitative statement yet of P8's thesis: the aggregates
are a *sensor*, not a *scorer*.

**3. The LLM-sensor *increment* on top of raw features never earns a
CI-certified net win.** `XGB-24full` vs `XGB-20raw`: the paired-bootstrap CI
excludes zero at only 2/8 ratios — and both are the **low-loss** cases (`rho`=2,5)
where `XGB-20raw` is detectably *cheaper* (you pay $0.0035/row sensing for no
gain: at `rho=2`, $0.0142 vs $0.0112/decision). The first break-even is
`L* ≈ $5.50` (`rho*≈11`), but the advantage is razor-thin (peak $0.0008/decision
near `L=$6.50`), oscillates in sign with the discrete `tau*` cutoff, and **no
`rho` yields a bootstrap CI that certifies a 24full net saving.** Honest scope:
on this data the oracle LLM sensor, as a supplement to a competent raw-feature
scorer, is not worth its per-row cost.

**4. Raw features are the scorer backbone.** `XGB-20raw` beats `XGB-4sensor` at
8/8 ratios (CIs exclude zero), e.g. Δ=$0.143/decision at `rho=100`.

## Falsifiable predictions
- On any dataset where a competent raw-feature scorer exists, the LLM-sensor
  aggregate block will (a) be cost-dominant as a stand-alone scorer at every
  `rho`, and (b) fail to produce a bootstrap-CI-significant net-cost reduction
  as a supplement unless `L` exceeds a break-even that scales with the sensing
  cost `c_sense`.
- Raising `c_sense` (larger LLM / synchronous scoring) shifts `L*` upward
  proportionally; the 4-sensor cost penalty is `c_sense`-independent (driven by
  detection loss, not sensing spend).

## Status: validated (real 10k held-out split, exact argmin, paired bootstrap CIs)
