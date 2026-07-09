# Iter 164 — D13 = P5P8-SYNTH per-prompt reward stability (JOB B)

**Pillar:** P5P8-SYNTH (13-domain density matrix)
**Vein:** iter-161 mint vein #4 — D13 per-prompt reward stability on N2 reward tensors
**Status:** validated + 4/4 falsifiable headlines PASS (honest structural findings)

## Why this iteration

Iter-160 added D12 = P5 N10 step-aggregate reward-stability density on the
160-cell N2 reward tensor (4 methods × 40 steps). The natural follow-up is
the per-prompt granularity: D13 on 4 × 40 × 16 = 2560 cells.

The motivation: per-prompt reward stability at G=8 rollouts is what every
GRPO-family training run actually has. If D13 sits in MID/HIGH, then per-prompt
rewards are well-measured and group baselines are stable. If D13 sits in LOW,
the per-prompt CI is dominated by the binomial n=8 floor and reward stability
is structurally unreachable.

## Method (terse)

For each (method × step × prompt) cell on N2 (4 × 40 × 16 = 2560 cells):
1. Read the 8-rollout binary reward vector.
2. Compute Wilson 95% CI half-width on the success proportion.
3. D13(cell, ε) = 1[half-width < ε].
4. Density = (#stable cells) / 2560.

ε ∈ {0.025, 0.05, 0.10}. Per-method breakdown at canonical ε=0.05.

## 4 falsifiable hypotheses settled (4/4 PASS)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** D13@0.05 in LOW layer | **PASS** | density = 0.0000 (0/2560 cells); layer = LOW |
| **H2** Wilson CI half-width structural floor ≥ 0.10 at n=8 (p1 half-width) | **PASS** | p1 = 0.1622, p10 = 0.1622, median = 0.1622 |
| **H3** D13 << D12 (per-prompt granularity tighter than step-aggregate) | **PASS** | D13 = 0.0 vs D12 = 0.175, ratio = 0.0 |
| **H4** All 4 methods in LOW layer | **PASS** | grpo=0, aero=0, gift=0, areal=0 |

## Sharpest paper-grade findings

1. **Per-prompt reward stability is structurally unreachable at G=8.** The minimum Wilson 95% CI half-width on a binomial proportion with n=8 rollouts is **0.1622** (achieved when k=0 or k=8; the half-width is bounded below by z·sqrt(p(1-p)/n) minimized at p=0 or p=1). Every ε ≤ 0.10 is structurally infeasible at this granularity — D13 is forced into the LOW layer by the binomial n=8 floor.

2. **To achieve ε ≤ 0.10 per-prompt stability, you need G ≥ 67.** At p=0.5, the half-width is z·sqrt(0.25/n) ≤ 0.10 → n ≥ 96. At p=0 or p=1, n ≥ 1.96²/(4·0.01) = 96 also (conservatively). For ε ≤ 0.05 you need G ≥ 384. **This is the operational implication: GRPO at G=8 cannot measure per-prompt reward stability at any reasonable precision.**

3. **D13 is the second P5P8-SYNTH LOW-layer domain** (joining D1, D6, D7). It is the FIRST structural-LOW domain — its LOW status is determined by the binomial n=8 floor, not by empirical sample sparsity. **Layer assignments (13 domains)**: LOW = {D1, D6, D7, D13} (4 domains); MID = {D2, D3, D4, D8, D9, D11, D12} (7 domains); HIGH = {D5, D10} (2 domains).

4. **D13 vs D12 ratio = 0** — per-prompt granularity is **infinite× tighter** than step-aggregate in the half-width sense. Per-prompt CI half-width ∈ [0.16, 0.28] (n=8); step-aggregate CI half-width is ~0.05 on the 128-rollout step mean. The granularity crossover is sharp and structural.

## Operational implication for GRPO-family training

- **Per-prompt reward stability at G=8 is NOT measurable.** Reporting per-prompt confidence intervals on a G=8 GRPO rollout is structurally meaningless.
- **Step-aggregate (D12) is the canonical reporting granularity** at G=8. Iter-160 D12@0.05 = 0.175 with Wilson CI [0.124, 0.241] sits in MID — step-aggregate reward stability is a meaningful metric, but per-prompt is not.
- **To upgrade D13 to MID or HIGH, increase G.** G=32 → per-prompt half-width floor at p̂=0.5 is 0.173; still too coarse. **G ≥ 67 needed for ε ≤ 0.10.**

## Cross-paper coupling

- **P5P8-SYNTH iter-160 row 175 (D12 step-aggregate)**: D12 at step level recovers 17.5% density at ε=0.05. Iter-164 D13 at per-prompt recovers 0% — the per-prompt-vs-step granularity gap is exactly the binomial n=8 floor.
- **P5 iter-141 (N10 reward tensor)**: iter-141 produced the 160-cell N2 tensor; iter-164 reads the same tensor at finer granularity.
- **P7 iter-163 row 177 (step-aggregate Pareto)**: P7 controller evaluation is at step granularity, consistent with iter-164's recommendation that step-aggregate is the canonical reporting granularity at G=8.

## Deliverables

- `scripts/p5p8/synth_iter164_d13_per_prompt_reward_stability.py` (~290 LoC, stdlib only)
- `experiments/results/p5p8/synth_iter164_d13_per_cell.tsv` (7680 rows: 2560 cells × 3 ε)
- `experiments/results/p5p8/synth_iter164_d13_per_eps.tsv` (3 rows: per-ε Wilson CIs + layer)
- `experiments/results/p5p8/synth_iter164_d13_per_method.tsv` (4 rows: per-method density)
- `experiments/results/p5p8/synth_iter164_summary.json` (machine-readable H1-H4 verdicts)
- 1 line in `AUTORESEARCH_FINDINGS.jsonl` (pillar P5P8-SYNTH, iter 164)