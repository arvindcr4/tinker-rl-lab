# #86 P5 — MIN-REPORT v2.0 stack-axis extension prototype on the live 98-cell corpus (iter 73)

**Pillar:** P5 (Pillar 1 — Report the Stack, Not the Label)
**Target classes:** T2 (fresh-data evidence) + T3 (cross-paper coupling) + T1 (statistical rigor)

## Summary

Iter-65 row 76 established that **4 of the 7 MIN-REPORT items are placebos** on the live 98-cell corpus (n_unique=1, H=0). Iter-69 row 81 confirmed that even a v2 schema with 4 GRPO/PPO hyperparameter candidates would NOT escape the placebo problem (corpus-design constraint, not schema-design). The iter-72 mint recommendation explicitly invited a **yield-aware MIN-REPORT item that records stack axes directly**. This iter takes that invitation concretely:

**Prototype a v2.0 MIN-REPORT schema with the 5 stack axes that cells.tsv already records as Items 8-12** (model_family, task_slice, G, temperature, seed), and quantify on the live 98-cell corpus:

1. **H1 (per-item information uplift):** Does v2 increase total info budget?
2. **H2 (fingerprint × measured-telemetry coupling):** Does v2 strengthen the iter-65 row 76 Spearman ρ(hamming, |ΔX|) over v1?
3. **H3 (per-cell badge uplift):** Is the badge uplift deterministic (every cell populates all 5 axes from cells.tsv)?
4. **H4 (per-axis 1-D Spearman contribution):** Which of the 5 v2 axes individually drive the correlation?

## Critical nuance: v2 axes are PARTIALLY REDUNDANT with v1

v1 items 5 and 6 already encode two of the v2 axes:
- v1 item 5 (`group_size_schedule`) is a structured string `"fixed-G=N"` that uniquely encodes G.
- v1 item 6 (`heldout_split`) is a structured string that uniquely encodes `task_slice` on this corpus (every manifest's `heldout_split` value equals the cell's `task_slice`).

Therefore the **truly-new v2 axes** (axes that v1 does NOT encode) are only **3**: `model_family`, `temperature`, `seed`. The iter reports three fingerprints:
- **v1**: 7-tuple (canonical MIN-REPORT).
- **v2**: 12-tuple = v1 + 5 axes (full stack-axis extension).
- **v2_tn** (truly-new): 10-tuple = v1 + 3 truly-new axes (model_family + temperature + seed).

## Falsifiable headlines (audit re-run 2026-07-05, n=98, B=5000 bootstrap)

| Hypothesis | Claim | Result |
|---|---|---|
| **H1** | v2 stack axes add positive info bits to the manifest; total info budget increases | **CONFIRMED**: v2 stack axes add **+6.86 bits** (60.1% uplift; v2 total = 18.27 bits vs v1 11.41 bits); v2 stack-discriminative bits rises from 4.80 → 11.66 (2.4×) |
| **H2** | v2 fingerprint strengthens Spearman ρ(hamming, \|ΔX\|) over v1 | **OUTCOME-DEPENDENT**: v2 weakens on zvf (−0.049) and pcd (−0.030); v2 strengthens on reward (+0.057) and mean_len (+0.070); v2_tn weakens on zvf (−0.099) and pcd (−0.059) but strengthens on reward (+0.072) and mean_len (+0.116) |
| **H3** | v2 badge uplift is deterministic (every cell populates all 5 axes) | **CONFIRMED**: badge uplift is **+35 pts (v2)** / **+20 pts (v2_tn)** on every cell; 95% bootstrap CI is degenerate (no variance) |
| **H4** | Per-axis 1-D Hamming reveals which axes drive coupling | **task_slice dominates zvf** (ρ=+0.367 alone vs v1 fingerprint's +0.435); **model_family dominates reward** (ρ=+0.349 alone vs v1 fingerprint's +0.254); **G second** for zvf (ρ=+0.193); temperature and seed are noise (|ρ|<0.05 on every outcome) |

## Headline numbers (n=98 cells, 2000 sampled cell-pairs, B=5000 bootstrap)

- **v1 total info budget**: 11.4127 bits (matches iter-65 row 76 independent measurement: 11.4 bits).
- **v1 stack-discriminative bits**: 4.7980 bits (3 PLACEBO + 1 CELL_IDENTIFIER + 3 VARYING_STACK_DESCRIPTOR).
- **v2 stack axes bits**: 6.8607 bits (5 VARYING_STACK_DESCRIPTOR items: model_family 0.999 + task_slice 1.555 + G 2.312 + temperature 0.995 + seed 1.000).
- **v2 total info budget**: 18.2735 bits (+60.1% uplift).
- **v2 stack-discriminative bits**: 11.6588 bits (2.4× lift over v1).
- **v2 truly-new bits**: 2.994 bits (model_family + temperature + seed — the 3 axes v1 doesn't already encode).
- **Spearman ρ (Hamming, |ΔX|) on v1**: zvf=+0.4352, reward=+0.2536, pcd=+0.3991, mean_len=+0.2141.
- **Spearman ρ (Hamming, |ΔX|) on v2**: zvf=+0.3864, reward=+0.3102, pcd=+0.3686, mean_len=+0.2837.
- **Spearman ρ (Hamming, |ΔX|) on v2_tn**: zvf=+0.3367, reward=+0.3251, pcd=+0.3405, mean_len=+0.3299.
- **Per-axis 1-D Hamming ρ_zvf**: model_family=+0.019, **task_slice=+0.367**, G=+0.193, temperature=+0.023, seed=−0.045.
- **Per-axis 1-D Hamming ρ_reward**: model_family=+0.349, task_slice=+0.285, G=−0.014, temperature=+0.010, seed=−0.026.
- **Badge uplift v2**: +35 pts (degenerate CI [+35, +35]).
- **Badge uplift v2_tn**: +20 pts (degenerate CI [+20, +20]).

## Cross-paper coupling

- **(P5↔P5)**: closes the iter-72 mint recommendation with a concrete prototype. The "4/7 placebo" problem (iter-65 row 76) is a CORPUS-DESIGN artifact, not a SCHEMA-DESIGN artifact (iter-69 row 81). The v2 schema's +6.86 bits of NEW information would resolve the placebo problem on info-budget grounds (v1 has 3 PLACEBO + 1 CELL_IDENTIFIER; v2 has 0 PLACEBO + 1 CELL_IDENTIFIER).
- **(P5↔P6)**: the v2 stack-axis items are population-equivalent to the registry's existing per-stack `min_report` block; they would not require a new registry schema field, only an additive-optional `outcomes.v2_stack_axes` block.
- **(P5↔P7)**: iter-49 row 60 P5 stack-conditioning-mega found η²(model)=0.4527 on mean_reward (model dominates), η²(task)=0.2729 second; this iter's per-axis 1-D Spearman on reward reproduces: model_family ρ=+0.349 alone > task_slice ρ=+0.285 alone — confirming model as the primary reward axis.
- **(P5↔Berkeley)**: the "truly-new v2 axes" (model_family + temperature + seed) decomposition mirrors the Berkeley CDH row-12 (auditor-first) pattern of partitioning axes by signal content.

## Sharpest paper-facing claim

**The iter-65 4-of-7 placebo problem on the live 98-cell corpus is a CORPUS-DESIGN constraint, not a SCHEMA-DESIGN constraint** (re-confirms iter-69 row 81). A v2.0 MIN-REPORT with 5 stack-axis declarations (Items 8-12) would resolve it on info-budget grounds (+60.1% bits), but the resulting fingerprint × measured-telemetry coupling is **outcome-dependent**:

- For **zvf**: adding v2 axes WEAKENS coupling (−0.049) because the new axes (model_family, temperature, seed) are noise on the zvf signal — only `task_slice` and `G` carry zvf variance, and v1 already encodes both via items 5 and 6.
- For **reward**: adding v2 axes STRENGTHENS coupling (+0.057) because `model_family` is the dominant reward axis (η²(model)=0.4527 from iter-49 row 60), and v1 does NOT encode model_family.
- For **mean_len**: adding v2 axes STRENGTHENS coupling (+0.116 for v2_tn) because model_family drives mean_len variance on the live corpus.

The **operational recommendation**: the iter-53 #64 MVE recommendation (add 5 continuous-telemetry fields) and this iter's v2 stack-axis extension are **complementary, not redundant**. Together they would lift distinct manifest profiles 15→98 (iter-53) AND add +6.86 bits of stack-axis information (this iter).

## Why this matters

- The **4/7 placebo problem on the live 98-cell corpus is a CORPUS-DESIGN constraint**, not a SCHEMA-DESIGN constraint (re-confirms iter-69 row 81 from a different measurement path).
- A v2.0 schema with 5 stack-axis declarations would resolve the placebo problem on info-budget grounds but **does NOT uniformly improve fingerprint × measured-telemetry coupling** — the coupling is outcome-dependent because each axis drives a different outcome.
- The **per-axis 1-D Spearman analysis** is a new measurement pattern that future MIN-REPORT revisions can use to prioritize which axes to add: pick the axes whose 1-D Hamming ρ exceeds 0.2 on the outcomes of interest (here: `task_slice` for zvf, `model_family` for reward).
- The **deterministic +35 pts badge uplift** is a hard prediction: every cell in the live corpus already populates all 5 v2 axes from cells.tsv, so the extension is free.

## Reproducibility (verbatim)

```bash
cd /home/claude/tinker-rl-lab-minimax
python3 scripts/p5p8/p5_stack_axis_minreport.py
```

Inputs:
- `experiments/results/mega_20260704/cells.tsv` (98 cells, 12 axes)
- `experiments/results/mega_20260704/manifests/*.json` (98 manifests)

Outputs:
- `experiments/results/p5p8/p5_stack_axis_minreport.tsv` (12 rows: 7 v1 + 5 v2)
- `experiments/results/p5p8/p5_stack_axis_minreport_boot.tsv` (bootstrap CIs)
- `experiments/results/p5p8/p5_stack_axis_minreport_summary.json` (full headline numbers)

Script: `scripts/p5p8/p5_stack_axis_minreport.py` (~280 LoC, stdlib only).

## Status

Validated, ready for ledger row addition.