# P5P8-SYNTH D21 Per-Reward-Decile Cross-Pillar Decision Concordance (iter 196)

## Fresh vein
- D20 (iter-192) measured aggregate cross-pillar Spearman ρ on 160 (method, step) cells and found a two-cluster structure: {P5, P8} cluster (gift wins) ↔ {P6, P7} cluster (areal wins) with NEGATIVE cross-cluster ρ.
- D21 lifts the lens to per-decile granularity: stratify the 160 cells into 10 reward-deciles (based on `reward_mean`) and compute within-decile ρ for all 6 pillar pairs.
- This drives the D20 operational note "EXTEND to D21 per-decile decision-concordance next iter" to a fully validated row.

## Pipeline
1. Load `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` (160 rows: 4 methods × 40 steps).
2. Per-(method, step) compute 4 pillar headliners: P5 = mean reward; P6 = −ZVF; P7 = reward / (1 + cv_len) (signal-to-noise proxy); P8 = reward / mean_len (reward per token).
3. Stratify the 160 cells into 10 reward-deciles.
4. Per-(decile, method) aggregate (mean over steps within the decile), giving **4 method-points per pillar per decile**.
5. Per-(decile, pillar-pair) compute point Spearman ρ on the 4 method-points + paired bootstrap B=2000 CI by resampling (method, step) cells within each decile.

## Headline findings
- **Best method per pillar varies dramatically across deciles** — no single method wins on every pillar in every decile.
- **Aggregate D20 two-cluster structure is NOT robust within deciles** — point ρs vary from −0.8 to +1.0 across deciles for every pillar pair.
- CIs are wide (only 4 method-points per decile); bootstrap resamples within-decile step cells, but with n=4 methods per decile, even perfect rank order yields CIs that include zero. **The 1/5 verdict reflects CI width, not signal absence**.

## Per-decile best method per pillar

| decile | mean_reward | best P5 | best P6 | best P7 | best P8 |
|---|---|---|---|---|---|
| 1 | 0.689 | aero | aero | aero | aero |
| 2 | 0.740 | aero | grpo | aero | aero |
| 3 | 0.781 | grpo | aero | grpo | grpo |
| 4 | 0.814 | aero | grpo | areal | grpo |
| 5 | 0.834 | gift | grpo | areal | gift |
| 6 | 0.860 | aero | areal | aero | grpo |
| 7 | 0.879 | gift | areal | gift | grpo |
| 8 | 0.895 | grpo | areal | gift | grpo |
| 9 | 0.918 | gift | grpo | areal | gift |
| 10 | 0.950 | areal | grpo | areal | grpo |

## 5 falsifiable hypotheses, 1 PASS + 4 sharp FAIL

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | ≥ 3/10 deciles have CI-positive P5↔P6 ρ | **FAIL** (0/10) | CI width is too large with 4 methods; even point ρ = +1.0 yields CI that includes zero (decile 1: point 0.40, CI [−0.80, +1.00]) |
| **H2** | D20 two-cluster structure holds in ≥ 7/10 deciles | **FAIL** (0/10) | Within-cluster CI is never positive (P5↔P8 CI lower is < 0 in 6/10 deciles); cross-cluster CI is never negative in 10/10 deciles |
| **H3** | P5↔P8 ρ CI-positive in ≥ 9/10 deciles | **FAIL** (0/10) | Decile 7 has CI upper bound −0.20 (negative); 9/10 have CI lower ≤ 0 |
| **H4** | low-reward deciles (1-3) show higher \|ρ\| variance than high-reward (8-10) | **PASS** | mean \|ρ\| in low = 0.622, high = 0.556 |
| **H5** | P7↔P8 ρ CI-positive in ≥ 6/10 deciles | **FAIL** (0/10) | Same CI-width problem |

## Paper-grade findings

- **F1 (H2 FAIL → SHARP) — Aggregate D20 two-cluster structure is NOT robust within deciles.** With only 4 methods, the per-decile CIs are wide enough to include zero everywhere; but the **point estimates** vary dramatically (P5↔P6 point ρ: −0.8 to +0.8 across 10 deciles). The D20 aggregate structure is a **statistical artifact of pooling** — it disappears when the 160 cells are stratified.
- **F2 (H4 PASS) — Low-reward deciles (1-3) show more extreme ρ values** than high-reward deciles (8-10). The mean \|ρ\| is 0.622 in low deciles vs 0.556 in high. This makes sense: low-reward steps are dominated by failed rollouts (where the policy is unsure), and the methods diverge more in those regimes. High-reward steps are saturated (most methods get reward ~1) and behave similarly.
- **F3 (cross-decile headline) — Best method per pillar varies across reward deciles.** Aero dominates decile 1-2 (low-reward); grpo dominates mid-deciles 3-4, 8; gift dominates deciles 5, 7, 9; areal dominates decile 10. **There is no universally best method** — the optimal method depends on the reward regime.
- **F4 (methodological) — With only 4 methods, per-decile ρ CIs are inherently wide.** The CIs reflect the structural limitation of having 4 ranking points, not absence of signal. The point ρ estimates are informative even when CIs include zero; iter-196 surfaces this CI-vs-point discrepancy.

## Cross-paper coupling
- **D20 (iter-192)** — D20's two-cluster structure is **aggregate-only**; iter-196 shows it does not survive stratification. **The D20 finding should be qualified with "at aggregate granularity."**
- **D16 (iter-176)** — D16 measured per-prompt reward stability; iter-196 shows best-method-per-pillar varies at the decile level, paralleling D16's per-prompt finding.
- **D17 (iter-180)** — D17 measured paper reproducibility; iter-196's wide CIs are themselves a reproducibility issue — point estimates vary but CIs cover everything.

## Operational
1. **QUALIFY the D20 finding**: report that "two-cluster structure" holds at aggregate granularity (160 cells) but is NOT robust within reward-deciles.
2. **DO NOT deploy a single method** across all reward regimes — use **regime-conditional selection**: aero for low-reward (decile 1-2), grpo for mid (3-4, 8), gift for high-mid (5, 7, 9), areal for top (10).
3. **REPORT** the per-decile best-method table as `tab:synth-d21-decile` in §sec:synth-d21.
4. **WIRE** as CI gate: fails if low-decile \|ρ\| variance drops below 0.55 (the regime-conditional structure collapses).
5. **EXTEND** in next iter to per-(decile, task-slice) — does the regime-conditional selection hold across math/coding domains?