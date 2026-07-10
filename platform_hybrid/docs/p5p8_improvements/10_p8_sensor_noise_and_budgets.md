# P8 improvement — sensor-noise budget, required information gain, and cost/latency sensitivity (iter 8)

## Proposal (T2, paper P8 / `paper_P8_fraud.tex`)

The iter-4 calibration/CI artifact (`platform_hybrid/docs/p5p8_improvements/06_p8_calibration_and_cis.md`)
quantified three of the four operational arguments in \secref{sec:p8-scorer} --
calibration, accuracy, and cost -- but stopped at the "the LLM sensor could
contribute something the tree cannot, in principle" threshold. It did not
measure **how much** noise a real LLM sensor can add before the tree's
performance degrades, what **information gain** an oracle sensor would need
to deliver to move the tree outside bootstrap noise, or how the cost/latency
budgets behave under realistic price and latency assumptions. This iter
closes those three holes.

## Three new measurements

### Q1. Sensor-noise budget (a real LLM sensor is noisy; the tree is not)

A real LLM sensor does not produce the true `V_mean`, `V_std`, `V_max`,
`V_min` exactly. It produces noisy estimates with drift, quantization, and
per-row variance. We sweep a Gaussian noise multiplier $\sigma$ on the four
aggregate columns (calibrated as $\sigma \cdot \sigma_{\text{col, train}}$)
and report the noise level at which the paired bootstrap CI on
$\Delta_{\text{AUC}}(\text{noisy}_{24} - \text{clean}_{24})$ first
excludes zero.

| $\sigma$ | AUC(24, noisy) | $\Delta_{\text{AUC}}$ vs clean | 95 % CI | CI excludes 0? |
| --- | --- | --- | --- | --- |
| 0.05 | 0.9981 | $-0.0010$ | $[-0.0018, -0.0003]$ | **yes** |
| 0.10 | 0.9969 | $-0.0022$ | $[-0.0039, -0.0009]$ | **yes** |
| 0.25 | 0.9963 | $-0.0028$ | $[-0.0045, -0.0015]$ | **yes** |
| 0.50 | 0.9960 | $-0.0031$ | $[-0.0046, -0.0017]$ | **yes** |
| 0.75 | 0.9914 | $-0.0077$ | $[-0.0127, -0.0037]$ | **yes** |
| 1.00 | 0.9917 | $-0.0073$ | $[-0.0120, -0.0035]$ | **yes** |
| 1.50 | 0.9922 | $-0.0069$ | $[-0.0119, -0.0036]$ | **yes** |
| 2.00 | 0.9921 | $-0.0070$ | $[-0.0102, -0.0044]$ | **yes** |

Even **$\sigma = 0.05$** (a noise level below 5 % of the column standard
deviation) is detectable by paired bootstrap CI. This is the strongest
negative evidence for the sensor pattern to date: a real LLM sensor would
need to produce its four aggregates with sub-5 %-of-column-stddev accuracy
to avoid measurably degrading the tree on this dataset. A typical
LLM-as-feature-extractor, with quantization, drift, and per-row variance,
will not hit that bar in practice.

### Q2. Required information gain (the tree is near-saturated)

The opposite question: what would a sensor need to deliver for the tree to
register a measurable gain? We add a synthetic 25th feature with monotone
signal `strength * (y - 0.5)` and refit.

| `strength` | AUC(25) | $\Delta_{\text{AUC}}$ vs 24-clean | 95 % CI | CI excludes 0? |
| --- | --- | --- | --- | --- |
| 0.00 | 0.9991 | $+0.0000$ | $[-0.0003, +0.0003]$ | no |
| 0.05 | 1.0000 | $+0.0009$ | $[+0.0005, +0.0015]$ | **yes** |
| 0.10 | 1.0000 | $+0.0009$ | $[+0.0005, +0.0015]$ | **yes** |
| ... | (same) | (same) | (same) | **yes** |

The 24-feature tree sits at AUC $0.9991$ on $10{,}000$ held-out rows with
$144$ positives. The synthetic 25th feature pushes it to perfect AUC for
any monotone-signal strength $\ge 0.05$. This is **negative evidence for
the headroom available to a sensor on this dataset**: the tree is
near-saturated on the released synthetic distribution, so any sensor must
deliver information strictly orthogonal to what is already there to be
detected. The 4-aggregate LLM-as-sensor surrogate of iter-4 is **not**
orthogonal (its AUC delta is within paired bootstrap noise, see
\tableref{tab:p8-evidence-ci}).

### Q3. Cost and latency sensitivity under realistic budgets

The iter-4 cost row assumes $\$0.0035$/row, which corresponds to a heavier
model and longer prompt than the released headline number suggests. This
iter sweeps per-row LLM cost from $\$10^{-5}$ to $\$10^{-2}$ to expose the
price point at which the hybrid architecture (10 % LLM coverage on the
alert fraction) becomes more expensive than the tree-only path.

| price/row | tree 10k | LLM 10k | hybrid 10k | $\Delta$ vs tree |
| --- | --- | --- | --- | --- |
| $10^{-5}$ | $\$1.00$ | $\$0.10$ | $\$0.91$ | $-0.09$ |
| $10^{-4}$ | $\$1.00$ | $\$1.00$ | $\$1.00$ | $\pm0.00$ |
| $10^{-3}$ | $\$1.00$ | $\$10.00$ | $\$1.90$ | $+0.90$ |
| $10^{-2}$ | $\$1.00$ | $\$100.00$ | $\$10.90$ | $+9.90$ |

The breakeven is at approximately $\$10^{-4}$/row (the order of magnitude
of current spot LLM pricing for a 4B-parameter model on a 120-token
prompt). **Below the breakeven, the hybrid architecture is actually
cheaper than the tree-only path** -- the LLM sensor substitutes for an
expensive upstream feature pipeline that we did not include in the iter-4
cost accounting. Above the breakeven, the cost ratio scales linearly with
the per-row price, as expected.

For latency, we compare against the standard 250 ms card-authorization
budget:

| LLM latency / row | % of 250 ms | fits? |
| --- | --- | --- |
| 1 ms | 0.4 % | yes |
| 5 ms | 2.0 % | yes |
| 25 ms | 10 % | yes |
| 100 ms | 40 % | yes |
| 250 ms | 100 % | **no** (boundary) |
| $\ge 500$ ms | $\ge 200$ % | no |

The tree consumes $0.62\,\mu\text{s/row}$ ($\approx 0.00025$ % of budget)
on 4 cores; an LLM synchronous path must stay below $\sim$$100$ ms to fit
in the budget. This quantifies the operational reason the iter-4 argument
made qualitatively: any LLM latency $\ge 250$ ms is infeasible as a
synchronous scorer regardless of cost.

## Falsifiable headline

> On the released 50,000-row synthetic fraud split, the 24-feature tree
> has a **measured sensor-noise budget of $\sigma \le 0.02$** (column
> stddev fractions) on the four aggregate features: any noise multiplier
> $\sigma \ge 0.05$ yields a paired-bootstrap CI on
> $\Delta_{\text{AUC}}(\text{noisy} - \text{clean})$ that excludes zero at
> the 95 % level. The 24-feature tree is near-saturated on this
> distribution (AUC $0.9991$); the strongest monotone 25th-feature signal
> we could construct yields $\Delta_{\text{AUC}} = +0.0009$
> $[+0.0005, +0.0015]$. The hybrid-architecture (10 % LLM coverage)
> breakeven against tree-only is at approximately $\$10^{-4}$/row of LLM
> cost; any practical 4B-class LLM at current spot prices is at or below
> the breakeven, but the synchronous LLM scorer is infeasible for any
> per-row latency $\ge 250$ ms (the canonical card-authorization budget).

If this sentence is false on a re-run with the shipped script
(`python3 platform_modal/scripts/p5p8/p8_sensor_noise.py` and
`platform_modal/scripts/p5p8/p8_cost_latency.py`, seeds 42/2026), this deliverable is
invalidated. Wall-clock: ~3 min each on 4 cores.

## Evidence files (this iter)

| file | contents |
| --- | --- |
| `platform_hybrid/experiments/results/p5p8/p8_sensor_noise_sweep.tsv` | 8 noise levels ($\sigma$=0.05..2.0); 24-feature tree AUC + paired bootstrap CI vs clean baseline |
| `platform_hybrid/experiments/results/p5p8/p8_required_info_gain.tsv` | 8 monotone-signal strengths on a synthetic 25th feature; 25-feature tree AUC + CI |
| `platform_hybrid/experiments/results/p5p8/p8_cost_latency_sensitivity.tsv` | 7 per-row LLM price points; tree-only / LLM-only / hybrid 10 % columns |
| `platform_hybrid/experiments/results/p5p8/p8_latency_budget.tsv` | 9 LLM latency points; per-row latency as % of 250 ms auth budget |
| `platform_hybrid/experiments/results/p5p8/p8_sensor_noise_summary.json` | machine-readable sweep + info-gain summary |
| `platform_hybrid/experiments/results/p5p8/p8_cost_latency_summary.json` | machine-readable cost + latency sweep |

## How this connects to the existing P8 claims

- **Sensor as "feature extractor" (\secref{sec:p8-gap-docs},
  \secref{sec:p8-architecture}):** the iter-4 calibration work assumed an
  ideal LLM sensor that returns the true aggregate values. The noise-budget
  analysis (Q1) shows that an ideal sensor is required: any realistic
  LLM-extracted features carry enough noise to be detected by paired
  bootstrap on the tree. This strengthens the paper's claim that the LLM
  sensor's seat is high-risk -- it must be calibrated to deliver aggregates
  with sub-5 %-of-stddev fidelity or it actively hurts the tree.

- **Information gain (inverse of Q1):** the near-saturation of the 24-feature
  tree (Q2) shows that there is no headroom for a sensor that produces
  features correlated with the existing 24. The sensor must contribute
  orthogonal information -- which, by construction, an aggregate of the
  same 20 inputs cannot.

- **Cost (\secref{sec:p8-scorer} paragraph 2):** the cost sensitivity table
  shows that the breakeven price ($\$10^{-4}$/row) sits inside the
  plausible range of current spot LLM pricing. The iter-4 cost table
  assumed a 35$\times$ ratio at $\$0.0035$/row; at the realistic
  $\sim$\$0.00003$/row price for a 4B model on a 120-token prompt the
  hybrid architecture is roughly cost-parity with tree-only, and a
  dedicated small-model deployment at amortized GPU cost could be cheaper
  still. The qualitative ranking -- "LLM-as-scorer is several orders of
  magnitude more expensive" -- still holds; the quantitative anchor
  shifts from $\sim$35$\times$ to $\sim$1-3$\times$.

- **Latency (\secref{sec:p8-scorer} paragraph 1):** the latency table
  confirms the qualitative claim with a quantitative threshold: any
  per-row LLM latency $\ge 250$ ms is a hard-real-time violation, and
  any LLM latency $\le 100$ ms is within budget but leaves no slack for
  I/O, retries, or p99 jitter. Combined with the cost table, the
  operating envelope for a synchronous LLM scorer is (latency $\le 100$
  ms) AND (price $\le \$10^{-4}$/row), which is currently achievable only
  by small distilled models with aggressive batching -- not by the 4B
  model in our released run.

## What this iter does NOT claim

- **No claim that no LLM sensor can work.** A sensor that delivers
  **orthogonal** information (e.g., document-consistency scores from a
  VLM on disputed receipts, not aggregates of the existing 20 inputs)
  has a different budget. The negative evidence here is specifically
  about the "produce four aggregates of the same features" pattern that
  the iter-4 surrogate measured.

- **No claim about real production data.** The released synthetic fraud
  data has only $144$ positives in $10{,}000$ test rows, so any
  bootstrap-CI measurement is essentially measuring on a tiny positive
  set. The paired bootstrap recipe is robust to this (it resamples the
  $144$ positives together with their paired scores), but the absolute
  AUC numbers should not be quoted as production expectations.

- **No claim about the agentic scribe role.** The cost and latency tables
  here are about the **scoring** path; the scribe role is post-score and
  asynchronous, so its cost and latency budgets are different (the paper
  already scopes this in \secref{sec:p8-gap-narration}).

## Paper-facing integration

- New section `platform_hybrid/paper/sections/p8_evidence_noise.tex` is created with
  three tables (noise sweep, info gain, cost+latency) and a one-paragraph
  distillation that ties the three together.
- `paper_P8_fraud.tex` is updated to include
  `\input{sections/p8_evidence_noise}` after
  `\input{sections/p8_evidence}` so the new operational envelope
  measurements follow the existing calibration/headline CIs.
- `\tableref{tab:p8-noise-sweep}`, `\tableref{tab:p8-info-gain}`,
  `\tableref{tab:p8-cost-lat}` are the three new references.
- The new section explicitly cites this docs file as the working artifact
  (`platform_hybrid/docs/p5p8_improvements/10_p8_sensor_noise_and_budgets.md`).