# Findings from Modal volume artifacts (primary-source re-analysis)

Computed 2026-07-11 directly from the harvested per-seed JSONs in
`modal_artifacts/` (generator `modal_inventory.py`; 127 metrics files).
These are the runs the claim-to-run table could previously cite only as
bare `local:` IDs — they are now resolvable `modal://` artifacts WITH
held-out metrics.

## samestack (P5-C3 substrate) — 5 seeds, held-out backed

`modal://tinkerrl-results/samestack/{grpo,ppo}_s{42,123,456,789,1024}.json`
Qwen/Qwen2.5-0.5B, 40 steps, n_gen=128.

| arm | n | held-out acc (mean ± t95) | last-10 reward |
|---|---|---|---|
| GRPO | 5 | 0.990 ± 0.010 | 0.979 ± 0.019 |
| PPO  | 5 | 0.992 ± 0.008 | 0.918 ± 0.138 |

Read: on one fixed stack, GRPO and PPO are statistically indistinguishable
held-out (saturated-task regime); PPO's training reward is far noisier
across seeds. Caveats: 40 steps < the repo's Tier-B threshold (50), and the
task is near ceiling — this is a same-stack *stability* exhibit, not a
capability ranking.

## drgrpo_gsm8k (P4-C1 primary record) — the `local:` IDs resolved

`modal://tinkerrl-results/drgrpo_gsm8k/{grpo,dr_grpo}_s{42,123,456}.json`
Qwen/Qwen2.5-1.5B-Instruct, n_eval=200/seed.

| arm | n | pre → post held-out | Δ (mean ± t95) | comp len first5→last5 | mean ZVF |
|---|---|---|---|---|---|
| GRPO    | 3 | 0.2017 → 0.2633 | +0.062 ± 0.019 | 195 → 180 | 0.335 |
| Dr.GRPO | 3 | 0.2050 → 0.2550 | +0.050 ± 0.022 | 194 → 187 | 0.354 |

Read: matches P4's published numbers exactly — this is the primary record.
Overlapping CIs (indistinguishable gains), no length inflation under the
200-token cap, mean ZVF ≈ 0.34 both arms. **Confirms the P4 model conflict:
the runs are Qwen2.5-1.5B-Instruct; the abstract's Qwen3-8B claim must be
fixed or the runs supplied.**

## zvf-open (P5-C2 open-audit arm)

`modal://tinkerrl-zvf-open-results/zvf_gsm8k_qwen25_0_5b_G{4,8}_seed42.json`
Qwen2.5-0.5B, 30 logged steps: G=4 mean ZVF 0.80 (first5 0.80), last-10
reward 0.10; G=8 mean ZVF 0.53 (first5 1.00), last-10 0.19. G=32 directory
exists with per-step logs. Single-seed — diagnostic, not comparative.

## Cross-framework zoo (`tinker-results`, 66 files)

summary.json: 5 seeds (42/123/456/789/1024) per framework; cleanrl_ppo,
pufferlib, rl_games, trl_* successful 5/5; **d3rlpy 0/5** (honestly
recorded). This is the F3 substrate; note the statistical-rigor addendum's
warning still applies (different frameworks ≠ matched algorithms).

## Claim-table consequences

- **P4-C1**: run IDs upgrade from `local:drgrpo_gsm8k_cot:*` to the
  `modal://` paths above (Tier stays X only because of the abstract/model
  conflict — the evidence itself is now artifact-backed, 3 seeds, held-out).
- **P5-C3**: samestack now has 5-seed held-out evidence at `modal://` paths
  (stays sub-Tier-B on the 40 < 50 step threshold).
- **P5-C2**: open-audit arm resolvable at `modal://` paths (single-seed).
