# Iter 165 — P5 per-step algorithm-axis η² trajectory on N2 four-method panel

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label / MIN-REPORT)
**Vein:** brief vein (b) at per-step granularity; closes the open question
raised by iter-161 row 176 ("is the algorithm-axis variance constant or
decaying across training steps?") and extends iter-89/106 with a new
operational finding.
**Status:** validated + 4/5 falsifiable headlines PASS, 1 honest negative.

## Why this iteration

Iter-161 row 176 reports η²(method, zvf)=0.0075, η²(method, reward_mean)=0.0075
on the pooled 160-row N2 panel (DECISIVE). The open question is whether the
algorithm-axis variance is CONSTANT or DECAYING across training steps.

If constant, the iter-161 headline is trajectory-robust and any per-step
analysis is redundant. If decaying, the algorithm-axis is a TRANSIENT
signal-availability artifact at training start — sharpening the stack-vs-label
thesis to "the label matters at the start, then the stack takes over."

For each of 40 training steps, compute η²(method | step, prompt) on the
4 method × 16 prompt = 64 obs per step per channel (reward_mean, mean_len,
cv_len). Add paired-prompt bootstrap CIs (B=2000, seed=20260705) on each cell.

## Method (terse)

1. Load the 4-method × 40-step × 16-prompt × G=8 reward tensor from
   `experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl`.
2. For each (method, step), compute 16 prompt-mean values per channel
   (reward_mean = mean of 8 rollouts; mean_len = mean length; cv_len = coefficient of variation).
3. For each (channel, step): η²(method | step) = SS_method / SS_total on the
   4 method groups of 16 prompt-mean values.
4. Bootstrap: for each cell, sample 16 prompts with replacement (paired across
   4 methods), recompute η²; B=2000. Report point + (lo, hi, mean_boot).
5. Step-band summary: split 40 steps into early / mid / late (14/13/13).
6. Test 5 falsifiable hypotheses (H1-H5 below).

## 5 falsifiable hypotheses settled (4/5 PASS)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** per-step mean η²(method|step) ≤ 0.05 on ≥ 2/3 channels (DECISIVE) | **PASS** | 3/3: reward 0.0056, mean_len 0.0139, cv_len 0.0405 |
| **H2** trajectory |Spearman| ≤ 0.5 on 5/6 channels (TRAJECTORY STATIONARY) | **FAIL** | 2/3: reward +0.114, mean_len +0.875, cv_len +0.401 |
| **H3** pooled η²(method, reward_mean) matches iter-161 within ±0.005 | **PASS** | pooled=0.00747 vs iter-161=0.0075 (Δ=3.4e-5, exact replication) |
| **H4** GIFT dominates algorithm axis: LOMO(GIFT)/full < 0.5 on reward_mean | **PASS** | LOMO(GIFT)=0.181× full (82% drop), confirming iter-89/106 H3 |
| **H5** \|mean(early η²_reward_mean) − mean(late η²_reward_mean)\| ≤ 0.02 (stationarity on reward) | **PASS** | Δ=0.0004 (<< 0.02) — reward is trajectory-stationary |

## Sharpest paper-grade findings

1. **Algorithm axis is trajectory-stationary on reward_mean** (H5 PASS).
   Per-step η²(method, reward_mean) mean is 0.0056 across all 40 steps; the
   early-band (steps 0-13) and late-band (steps 27-39) means differ by
   0.0004 — well below the 0.02 stationarity bar. The iter-161 pooled
   headline of η²=0.0075 is robust to trajectory binning.

2. **Algorithm axis GROWS on mean_len** (H2 FAIL on this channel only).
   Per-step Spearman ρ(steps, η²(mean_len))=+0.875 — eta² on mean_len
   rises from 0.002 in the early band to 0.031 in the late band (15× growth).
   This is the **first P5 finding of a training-trajectory trend in the
   algorithm-axis decomposition**. Cross-check: cv_len Spearman +0.401 is
   positive but smaller.

3. **Pooled eta² replication to 4 decimal places** (H3 PASS).
   Iter-165 re-derives the iter-161 pooled η²(method, reward_mean) from
   the canonical n2_metrics.tsv (160 rows) and gets 0.0074664, matching
   iter-161's 0.0075 within 3.4e-5 (0.45%). The two implementations
   (iter-165 reads per-(method, prompt) means; iter-161 reads the same
   panel at per-(method, step) terminal stats) converge to the same answer.

4. **GIFT uniquely load-bearing on reward_mean** (H4 PASS).
   Leave-one-method-out on reward_mean: removing GIFT collapses η² from
   0.000503 to 0.0000911 (0.181×, 82% drop). Removing GRPO RAISES η²
   to 1.326× full (the 3 non-GRPO methods are more dispersed); removing
   AERO gives 0.962× full; removing AREAL gives 1.086× full. **Only GIFT
   removal shrinks the algorithm axis** — the structural diversity bonus
   iter-66 row 77 measured is concentrated in GIFT at the reward level
   (not just zvf as iter-89/106 showed).

5. **Per-step η² trajectories are bounded by 0.13 on every channel**.
   max(η²(method, cv_len)) = 0.1292 at step 39 (late band); max on
   mean_len = 0.0518 at step 38; max on reward_mean = 0.0306 at step 0.
   The **algorithm axis never crosses the 0.15 threshold** on any
   prompt-level channel — consistent with iter-89/106's strict-pass
   (η² ≤ 0.05 on 2/7 channels, ≤ 0.10 on 4/7).

## Per-step band summary (mean η² per band per channel)

| band | n_steps | channel | mean η² | min | max |
|---|---|---|---|---|---|
| early | 14 | reward_mean | 0.0056 | 0.0015 | 0.0306 |
| early | 14 | mean_len | 0.0019 | 0.0002 | 0.0054 |
| early | 14 | cv_len | 0.0213 | 0.0013 | 0.0581 |
| mid | 13 | reward_mean | 0.0053 | 0.0003 | 0.0149 |
| mid | 13 | mean_len | 0.0096 | 0.0022 | 0.0348 |
| mid | 13 | cv_len | 0.0466 | 0.0006 | 0.1239 |
| late | 13 | reward_mean | 0.0060 | 0.0002 | 0.0125 |
| late | 13 | mean_len | 0.0312 | 0.0078 | 0.0518 |
| late | 13 | cv_len | 0.0549 | 0.0017 | 0.1292 |

**The mean_len trajectory**: early 0.0019 → mid 0.0096 → late 0.0312
is a 16× monotone growth. This is the training-time emergence of the
algorithm-axis signal on length-controlled channels.

## Cross-paper coupling

- **P5 iter-161 row 176** — exact replication of pooled η²(method, reward_mean)
  at 4-decimal precision (0.00747 vs 0.0075). Per-step analysis sharpens the
  pooled headline: the iter-161 finding holds trajectory-stably on reward_mean
  but underestimates the algorithm-axis growth on mean_len (where pooled η²
  collapses to 0.0139 mean).
- **P5 iter-89/106 rows 101/106** — H4 PASS confirms GIFT dominates the
  algorithm axis on reward_mean (LOMO 0.181×), extending the iter-89
  zvf-finding to the reward channel at per-(method, prompt) granularity.
- **P5 iter-141 row 159** — iter-141 produced the N2 reward tensor with
  per-(method × step) terminal stats; iter-165 reads the same tensor at
  the per-(method × step × prompt) granularity, exposing the trajectory.
- **P5P8-SYNTH iter-164 D13** — iter-164 showed per-prompt reward stability
  is structurally unreachable at G=8; iter-165 confirms per-prompt values
  are well-defined (mean η²(method) on 64 obs per step is the right
  measurement at this granularity).
- **FRONTIER_INSIGHTS Round 2** (ZVF = signal availability) — the per-step
  trajectory finding on mean_len shows the algorithm axis is not just
  static "label noise" but a training-time emergent signal: as the model
  learns, the method differences in completion length diverge.

## Operational recommendations

1. **Adopt per-step trajectory analysis** as a third P5 evaluation
   protocol alongside iter-161 pooled η² and iter-89 bootstrap CI on the
   N2 panel. The trajectory view exposes monotone trends the pooled view
   averages away.
2. **Report H5 stationarity as a falsifiable check** for any new
   algorithm-axis claim: |early − late| ≤ 0.02 on the canonical
   channel. Iter-165 PASS at 0.0004 sets the bar.
3. **Cite iter-165 H4 LOMO evidence** when reporting algorithm-axis
   decomposition: GIFT removal collapses the axis 82%on reward_mean,
   so a "GRPO-family equivalent" claim is valid only AFTER excluding GIFT.

## Deliverables

- `scripts/p5p8/p5_iter165_per_step_eta2_trajectory.py` (~325 LoC, stdlib only)
- `experiments/results/p5p8/p5_iter165_per_step_eta2.tsv` (120 rows: 40 steps × 3 channels)
- `experiments/results/p5p8/p5_iter165_per_step_eta2_boot.tsv` (120 rows: with bootstrap CIs)
- `experiments/results/p5p8/p5_iter165_step_band_summary.tsv` (9 rows: 3 bands × 3 channels)
- `experiments/results/p5p8/p5_iter165_summary.json` (machine-readable H1-H5 verdicts)
- `paper/sections/p5_iter165_per_step_trajectory.tex` (NEW §sec:p5-iter165)
- 1 line in `findings_ledger.jsonl` (pillar P5, iter 165)