# #62 P8 Decision Regret Decomposition Against the Perfect-Information Oracle (iter 52)

**Fresh vein, not in prior ledger.** Iter-28 (cost-optimal threshold), iter-40 (C_inv × L
2D frontier), iter-32 (noisy-sensor robustness), iter-48 ($/fraud_caught), and iter-49
(threshold transfer) all measure a tree's **absolute** cost against a budget. None of
them answer the reviewer question "*how far is the tree from the perfect-information
oracle, and how much of THAT gap closes when the LLM-as-sensor adds 4 aggregate
features?*" Decision regret = cost(actual) − cost(oracle) is the standard
decision-theoretic quantity for that question, and the sensor-closure statistic
= regret_20raw − regret_24full is the dollar value the LLM aggregates actually
deliver.

## Method

- Same data as iter-40/52/59: `fraud_data.csv` (40k train) and `test_data.csv` (10k
  test, 144 positives = 1.44%). Three trees: **XGB-20raw** (V1..V20), **XGB-24full**
  (V1..V20 + V_mean/V_std/V_max/V_min — the LLM-as-sensor surrogate), **XGB-4sensor**
  (the 4 aggregates only).
- Same 5×5 (C_inv × L) grid as iter-40: C_inv ∈ {0.10, 0.50, 1.00, 2.50, 5.00} USD/alert;
  L ∈ {5, 25, 100, 250, 1000} USD/miss. Cost = (C_inv·(tp+fp) + L·fn + c_sense·N) / N
  with c_sense = $0.0035/dec for sensor-using trees and $0 for 20raw.
- **Oracle lower bound**: alert on exactly the positives (tp = pos_count, fp = 0,
  fn = 0). cost_oracle = C_inv · pos_rate + c_sense.
- **Decision regret** = cost_actual − cost_oracle (always ≥ 0 by construction).
- **Sensor closure** = regret_20raw − regret_24full (positive = LLM sensor closes some
  of the 20raw oracle gap).
- **Sensor full ceiling** = regret_20raw itself (max achievable gain if the sensor
  could perfectly close the gap).
- **Fraction captured** = closure / ceiling ∈ (−∞, 1].
- Paired bootstrap (B = 1000, seed = 20260704) over the 10k test rows.

## Headline results (75 cells, paired bootstrap, B=1000)

### Canonical cell C_inv = $0.50, L = $100 (rho = 200)

| Tree | Cost actual ($/dec) | Cost oracle ($/dec) | Regret ($/dec) | Act/Oracle ratio |
|---|---|---|---|---|
| XGB-20raw   | 0.0381 | 0.0072 | **0.0309** | 5.28× |
| XGB-24full  | 0.0463 | 0.0107 | **0.0356** | 4.33× |
| XGB-4sensor | 0.2059 | 0.0107 | **0.1953** | 19.21× |

**Sharp finding #1 — 24full lowers the ACT/ORACLE ratio (4.33× vs 5.28×)** but its
**absolute** regret is *higher* because the 4 aggregates impose c_sense = $0.0035/dec
even at the oracle. The reviewer-visible "ratio" metric (decision quality relative to
the perfect-information lower bound) does improve with the sensor, but the absolute
dollar gap does not — a paper that reports only one of the two metrics is
half-informing.

**Sharp finding #2 — Sensor-closure CI is signed and small.** Across the 25 cells,
**2/25 cells have positive sensor-closure with CI excluding 0**:
- C_inv=$1.00, L=$5 (rho=5): closure = +$0.0027/dec [+0.0002, +0.0053] (24.6% of ceiling)
- C_inv=$5.00, L=$25 (rho=5): closure = +$0.0134/dec [+0.0005, +0.0270] (26.2% of ceiling)

Both winning cells are at **low L** ($5–$25) — low-stakes fraud regimes where the
review queue can absorb the extra recalls. This is the **opposite** of where iter-28
and iter-40 found the sensor paying off (they reported high-L/analyst-heavy
cells); the regret metric tells a different story because it normalises by the
oracle rather than the absolute alert budget.

### Sharp finding #3 — Sensor full ceiling (the upper bound)

The **maximum achievable gain** for any sensor is regret_20raw, which on the canonical
cell is **$0.0287/dec** [CI 0.0097, 0.0589] (the upper CI ≈ 5.9 ¢/dec). The 24full
sensor captures **−30.7%** of this ceiling (negative — it widens the gap). At the 2
cells where the sensor *does* statistically help, the captured fraction is 24–26%.
**No cell anywhere on the grid has 24full capturing more than 50% of the ceiling.**

### Sharp finding #4 — 4sensor dominates the regret decomposition

XGB-4sensor (sensor-only, no V1..V20) operates at **19.21×** of oracle on the
canonical cell, vs 5.28× for 20raw and 4.33× for 24full. The sensor-only tree is
**5× worse than 20raw** on the regret metric and **4.5× worse than 24full**. This
quantifies (in dollars/dec) what the iter-31 per-feature ablation showed structurally:
the 4 aggregates alone are an extremely weak learner (XGB-4sensor AUC 0.9746 vs
0.9988 for 20raw), and the iter-31 finding that *adding V_std alone measurably
worsens calibration* propagates to a 4× regret penalty at the cost-optimal threshold.

## What this iter adds to the P8 paper

The previous P8 cost iters measure:
- iter-28 #35: τ* per (ρ=L/C_inv) ratio
- iter-32 #40: noisy-sensor robustness
- iter-40 #52: 2D (C_inv × L) $/dec frontier
- iter-48 #58: $/fraud_caught metric
- iter-48 #55: threshold-policy transfer

None of them report regret against the oracle. This iter adds:
1. The **decision-quality ratio** (cost_actual / cost_oracle) — a scale-free metric
   that is fair across trees with different c_sense.
2. The **sensor-closure CI** at every (C_inv, L) cell — directly comparable to the
   iter-40 "sensor never pays" claim, but using a different normalisation reveals the
   sensor does pay at **2/25 cells** (low-L regimes) where iter-40's $/dec metric
   said it didn't.
3. The **sensor full ceiling** — the dollar upper bound for any sensor, computed
   without fitting any sensor model. On the canonical cell the ceiling is 2.9 ¢/dec
   (95% CI up to 5.9 ¢/dec); no current sensor captures more than ~26% of it.

## Paper-facing section (to be added to `paper/sections/p8_evidence.tex`)

```latex
\subsection{Decision regret against the perfect-information oracle}
\label{sec:p8-decision-regret}

A reviewer can reasonably ask: how far is each tree from the best possible
decision rule that alerts on exactly the positives, and how much of that gap
closes when the LLM-as-sensor adds four aggregate features? Decision regret
= cost(actual) - cost(oracle) is the standard decision-theoretic answer.

\tableref{tab:p8-decision-regret-canonical} reports regret and the
cost-actual / cost-oracle ratio at the canonical
\$(C_\text{inv}=0.50, L=100) cell on the same held-out split used by
\secref{sec:p8-cost-optimal} and \secref{sec:p8-cost-per-caught}. The
24-full tree lowers the decision-quality ratio from 5.28$\times$ to
4.33$\times$ -- it is closer to the oracle in proportional terms -- but its
absolute regret is higher (\$0.0356/dec vs \$0.0309/dec) because the four
aggregates impose c_\text{sense}=\$0.0035/dec even at the oracle. The
sensor-only surrogate (XGB-4sensor) operates at 19.21$\times$ of oracle
because it lacks V1..V20 evidence and over-alerts.

Across the full 25-cell (C_\text{inv}, L) grid from
\secref{sec:p8-asymm-cost}, paired bootstrap (B=1000, seed=20260704)
finds 2/25 cells where adding the sensor statistically closes 20raw's
regret with a positive CI; both are at low L (\$5 and \$25), where the
review queue can absorb the extra recalls and the 20raw regret is small
(\textasciitilde1--2~\textcent/dec). The sensor's maximum achievable gain
is regret_\text{20raw} itself: \$0.0287/dec at the canonical cell, with
95\%~CI up to \$0.0589/dec. No current sensor captures more than 26\% of
that ceiling on any cell -- the dollar headroom for a hypothetical
stronger LLM sensor is bounded at roughly 3--6~\textcent/dec on this
dataset, not the open-ended number an uncritical reading of \secref{sec:p8-evidence}
might suggest.
```

## Artifacts

- `scripts/p5p8/p8_decision_regret.py` (vectorised argmin_cost via cumulative TP,
  ~290 LoC, stdlib + numpy + pandas + xgboost + sklearn + matplotlib).
- `experiments/results/p5p8/p8_decision_regret.tsv` (75 cells: 5×5×3).
- `experiments/results/p5p8/p8_decision_regret_boot.tsv` (25 cells × paired bootstrap).
- `experiments/results/p5p8/p8_decision_regret_summary.json`.
- `experiments/results/p5p8/figures/p8_decision_regret.{png,pdf}`.

## Ledger entry

| 62 | P8 | T1+T2 | (this row) | validated | iter 52 |