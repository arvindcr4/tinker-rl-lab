# Iter 193 — P5 Algorithm-axis vs Stack-axis variance ratio (MIN-REPORT)

**Pillar:** P5 (Report the Stack, Not the Label / MIN-REPORT)
**Vein:** brief (b) — stack-conditioning via variance decomposition. Fresh vs
iter-141 (algorithm axis alone) and iter-189 (stack axis alone): iter-193 puts
the two axes **side by side** on shared outcome channels and reports the
**stack-to-label variance ratio** with CI-disjointness.

## Question
"Report the Stack, Not the Label" claims the algorithm *label* is a weak
predictor of outcomes relative to the *stack*. How much outcome variance does
the algorithm axis (4 GRPO-family methods on one fixed stack, N2) explain versus
the stack axes (G / temperature / task / model / seed, mega)? Reuses the Berkeley
Ivison-et-al. factorization recipe (`axis_variance_fraction`), adds unbiased
ω² and stratified bootstrap CIs (B=2000, seed 20260706, 95%).

## Data (real, in-repo)
- **Algorithm axis** — `n2_reward_tensor_resume/n2_metrics.tsv`: 4 methods
  (grpo/aero/gift/areal) × ~40 steps, identical stack (Qwen3-4B, GSM8K-easy,
  G=8, T=0.6, seed 0). 160 (method×step) rows per channel.
- **Stack axes** — `mega_20260704/cells.tsv`: 98 cells, factors
  model_family / task_slice / G / temperature / seed.
- **Shared channels:** zvf, reward, completion length.

## Headline result (`p5_iter193_ratio.tsv`)

| channel | algo η² [95% CI] | top stack axis (η²) | stack:label ratio | CI-disjoint |
|---|---|---|---|---|
| zvf | 0.0454 [0.009, 0.144] | task_slice (0.469) | **10.3×** | yes |
| reward | 0.0075 [0.003, 0.078] | model_family (0.453) | **60.6×** | yes |
| len | 0.0631 [0.019, 0.161] | model_family (0.301) | 4.8× | yes |

- **Algorithm label explains ≤6.3% of variance on every channel**; on reward it
  explains **0.75%** and its unbiased ω² is **−0.0115 < 0** — i.e. after
  small-sample correction the algorithm label explains **effectively zero**
  same-stack reward variance. This is the sharpest empirical form of the
  frontier **Estimator-Equivalence Principle** (Round 1).
- Stack axes explain **45–47%** of reward/zvf variance. Every stack-vs-label CI
  pair is **disjoint** (algo CI upper bound < top-stack CI lower bound).

## Hypotheses (4 PASS / 1 sharp FAIL)
- **H1 PASS** — algorithm η² < 0.10 on all 3 channels.
- **H2 PASS** — top-stack CI strictly above algorithm CI (disjoint) for zvf & reward.
- **H3 FAIL → SHARP** — for zvf the single top stack axis is **task_slice
  (0.469), not G (0.444)** — the two co-dominate and are nearly tied, refining
  the frontier/iter-189 emphasis on G alone: zvf's censored-contrast structure is
  driven by **task difficulty and group size jointly**.
- **H4 PASS** — seed η² < 0.10 on all channels (seed noise ≪ stack).
- **H5 PASS** — stack:label ratio > 3× on zvf and reward.

## Per-channel stack profile (`p5_iter193_axis_eta2.tsv`)
- **zvf** is co-dominated by task_slice (0.469) and G (0.444); temperature/model/seed ≈ 0.
- **reward** is dominated by model_family (0.453) then task_slice (0.273); G only 0.030.
- **len** spreads across model_family (0.301), task_slice (0.210), temperature (0.207).

Different outcomes are governed by different stack knobs — no single axis is "the
stack." This strengthens the MIN-REPORT case: a label plus one knob is insufficient.

## Cross-paper coupling
- **iter-141 (algorithm axis alone)** — iter-193 supplies the missing stack-side
  denominator and the ratio; iter-141's ω²≈0 reward result is reproduced.
- **iter-189 (manifest predictive power)** — iter-189's η²(zvf)=0.872 for the
  3-field discriminating set is consistent with zvf being stack-governed here.
- **iter-161 (stack factorization)** / **iter-133 (n10 η²)** — same recipe, new
  joint framing.
- **Frontier Round 1 (Estimator-Equivalence / Critic Degeneracy)** — reward
  ω²<0 is the quantitative expression of estimator equivalence at fixed stack.

## Operational
- **REPORT** stack:label ratios (60.6× reward, 10.3× zvf) as the P5 headline.
- **ADD** `tab:p5-iter193-ratio` to `paper_P5_minreport.tex`.
- **WIRE** `scripts/p5p8/p5_iter193_algo_vs_stack_eta2.py` as a CI gate — fails
  if any algorithm η² ≥ 0.10 or any stack-vs-label CI pair stops being disjoint.

## Reproduce
`python3 scripts/p5p8/p5_iter193_algo_vs_stack_eta2.py`
Outputs: `p5_iter193_axis_eta2.tsv` (18 rows), `p5_iter193_ratio.tsv` (3 rows),
`p5_iter193_summary.json`.
