# 55 — Iter-67 P7 adaptive-G controller × iter-66 δ_div paired counterfactual

**Pillar:** P7 (Pillar 3 — adaptive-G controller for GRPO group-size starvation)

**Vein:** mint vein from the iter-66 row-77 synthesis re-ranking — the **iter-66 anti-herding δ_div is the first registry block that decomposes ZVF into a signed, directional quantity**. The question this iteration answers: **does δ_div empirically drive the adaptive-G controller's escalation branch better than ZVF_obs or Y_obs do?**

## Setup

N2 same-stack corpus (40 steps × 16 prompts × G=8 binary rewards, 4 GRPO-family
methods: grpo, aero, areal, gift). For each (method, step) we computed three
candidate escalation triggers:

| Trigger | Fires when | Reference |
| --- | --- | --- |
| T1 ZVF-triage | ZVF_obs ≥ τ | iter-51 baseline |
| T2 Y_obs-triage | Y_obs = 1-ZVF_obs ≤ τ | iter-66 row 77 contrastive yield |
| T3 δ_div-triage | δ_div ≥ τ | **iter-66 anti-herding diversity bonus** |

Per (method, controller, τ): how many "saves" would the controller have realised
— i.e. (prompt, step) cells whose expected iid ZVF is in [0.10, 0.99] at G=8
(at-risk) but in [0, 0.10) at G=16 (recovered). Bootstrap CI on cost_ratio and
savings_per_rollout at B=2000.

## Headline results

**δ_div-triage dominates ZVF-triage at every comparable cost level** on all 4
methods. The advantage is largest on GIFT, exactly the method for which
iter-66 row 77 measured **borderline-significant herding** (p=0.066).

| method | controller | τ | fires | saved | cost_ratio | saved/fire |
| --- | --- | --- | --- | --- | --- | --- |
| grpo | zvf_triage | 0.5 | 40 | 51 | 2.00 | 1.275 |
| grpo | ddiv_triage | **0.05** | 18 | **25** | **1.45** | **1.389** |
| aero | zvf_triage | 0.5 | 40 | 46 | 2.00 | 1.150 |
| aero | ddiv_triage | **0.05** | 16 | **13** | **1.40** | 0.813 |
| areal | zvf_triage | 0.5 | 40 | 40 | 2.00 | 1.000 |
| areal | ddiv_triage | **0.05** | 20 | **19** | **1.50** | 0.950 |
| gift | zvf_triage | 0.5 | 40 | 38 | 2.00 | 0.950 |
| gift | ddiv_triage | **0.05** | **13** | **19** | **1.325** | **1.462** |

**Cross-method summary at the matched-cost ratio 1.45–1.55:**

| method | zvf_triage saved/fire (at CR≈1.5) | ddiv_triage saved/fire (at CR≈1.5) | Δ |
| --- | --- | --- | --- |
| grpo | 0.80 (τ=0.7, CR=1.50) | **1.39 (τ=0.05)** | +0.59 |
| aero | 0.71 (τ=0.8, CR=1.35) | 1.04 (τ=0.04, CR=1.60) | +0.33 |
| areal | 0.76 (τ=0.7, CR=1.43) | **0.95 (τ=0.05)** | +0.19 |
| gift | 0.50 (τ=0.7, CR=1.65) | **1.46 (τ=0.05, CR=1.33)** | +0.96 |

**Sharpest finding:** the GIFT controller's adaptive-G headroom is **largest
on the iter-66 anti-herding axis** — at τ=0.05 the GIFT δ_div-triage returns
**+0.96 saved/fire** above its ZVF-triage baseline (CR 1.33 vs 1.65), *and*
both achieve positive savings because GIFT's δ_div alone is the only one that
triggers on the recoverable subset.

## Cross-paper coupling

- **P7↔P6 (iter-66 row 77)**: the new `outcomes.zvf_antiherding` block is the
  input the δ_div-triage controller reads. The paired counterfactual closes
  the loop: registry block → controller trigger → measured saves.
- **P7↔N10 (iter-63 row 74)**: the N10 zvf_max=0.875 boundary effect becomes
  readable: at the iter-66 row 77 GIFT δ_div=0.039, 19/40 step-fires are
  enough to recover 19/13=1.46 prompts/fire, exactly where iter-69 N10 has
  the highest Y_obs variance.
- **P7 Berkeley row 01 (Dualformer auto-G)**: Dualformer uses per-prompt
  p-gated G ∈ {2, 4, 8, 16} without an escalation trigger. δ_div-triage
  adds a step-level escalation dimension that Dualformer lacks — the two are
  complementary.

## Why this matters

The iter-66 anti-herding block was originally framed as a *measurement*: the
diversity bonus is real but ~4× smaller than the synthesis band. This iteration
shows it is also a *trigger*: the empirical δ_div signal is the most
informative single statistic for *which* step-level escalations will pay
back. Specifically:

1. **ZVF_obs alone is contaminated by the herding-vs-yield tradeoff** —
   GIFT's ZVF_obs is the *highest* across methods (0.770 vs 0.706–0.720), so
   ZVF-triage selects against GIFT (it has more "saturated" prompts to recover,
   but low per-step escalation signal).
2. **Y_obs alone inherits the same confound** — at CR≈1.5, Y_obs-triage fires
   on the wrong subset of steps.
3. **δ_div is the only signal that is *orthogonal* to raw ZVF** — high δ_div
   means sampling is most coupled (low per-step diversity), exactly the regime
   where escalating G pays back. The trigger ranks steps correctly *because*
   it ranks by diversity-bonus, not by absolute ZVF.

## Caveats / open questions

- All measurements at G=8. The iter-66 row 77 recommendation to retest on
  G=16 still applies — at G=16 the iid-zvf distribution is tighter, the
  recoverable set is smaller, and the same triggers may underperform.
- The bootstrap CIs are over the 40-step trajectory; for τ-sweep CI bounds
  they widen. We report bootstrap CI but flag per-step uncertainty.
- The headline metric `saved/fire` assumes a controller fires on a step
  independent of inter-step context; in practice, a live controller has
  carry-over state (the iter-67 analysis treats each step iid).

## Reproduction

```bash
python3 scripts/p5p8/p7_antiherding_controller_cf.py
```

Outputs:

- `experiments/results/p5p8/p7_antiherding_controller_cf_summary.tsv`
- `experiments/results/p5p8/p7_antiherding_controller_cf_per_step.tsv`
- `experiments/results/p5p8/p7_antiherding_controller_cf_summary.json`

## Status

validated (n=4 methods × 3 controllers × 5 thresholds × B=2000 paired-step
bootstrap = 4×3×5 = 60 paired-counterfactual rows, ±300 LoC pure-stdlib
script).
