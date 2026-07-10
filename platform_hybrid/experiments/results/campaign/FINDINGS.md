# Multi-seed campaign — the powered re-test (Tinker, Qwen3.5-4B, 8 steps, GSM8K, 2026-07-06)

9 process-parallel runs (`experiments/openings/campaign.py`, W&B group `campaign`). Purpose: re-test the single-seed P3/curriculum claims with statistical power (kimi flagged them underpowered). **NOTE:** `zero_loss_frac` in this batch is invalid (empty-metrics-dict bug, since fixed) — held-out gain and oversample are the valid signals. Held-out n=12 (each gain quantized to ~0.083).

## Baseline vs Curriculum @ G=4, 3 seeds
| seed | baseline gain | curriculum gain (oversample) |
|---|---|---|
| 0 | 0.000 | +0.167 (6.0x) |
| 1 | +0.167 | -0.083 (5.2x) |
| 2 | -0.083 | -0.167 (5.3x) |
| **mean** | **+0.028** (sd 0.10) | **-0.028** (sd 0.14) |

## Group size @ seed 0 (baseline)
G2 +0.083 · G4 0.000 · G8 +0.083 · G16 +0.083

## Findings (honest, mostly NEGATIVE — which is the value)
1. **Naive curriculum does NOT beat baseline.** Mean gains +0.028 vs -0.028; difference ~0.056, SE ~0.10 (t~0.6) => not significant. The promising single-seed curriculum result (+0.167 at s0) was **noise** (s1,s2 negative). And it costs ~5-6x the sampling. => the *naive* "drop collapsed groups + oversample" lever is not worth it; P2/P3 must find a better token-budget lever.
2. **No robust "G=4 sweet spot."** At seed 0, G4 was the *worst* single point (0.0) while G2/G8/G16 all +0.083. The earlier single-seed "G=4 best" claim does not survive.
3. **Held-out RL gains on this setup are within noise** (all |gains| <= 0.167 = 1-2 of 12 examples). Consistent with the honest "diagnostic benchmark, not a GRPO-wins leaderboard" framing.

## Why this matters
This is the payoff of the verify->multi-seed->honest loop: kimi predicted the single-seed results were underpowered; the powered re-test confirms the effects are noise. Reporting these negatives (not the cherry-picked single-seed positives) is what makes the portfolio defensible. Next lever for P2/P3: token-budget-optimal allocation with staleness bounds (not naive filtering), tested multi-seed from the start.
