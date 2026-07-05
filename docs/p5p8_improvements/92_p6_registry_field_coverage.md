# 92 — P6 registry field-level coverage audit + Vein (d) backfill (iter 78)

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Vein:** brief veins (b)+(d) — coverage audit on field-level granularity across all 35 entries,
plus transparent entry for the missing-from-registry `PPO` method (1 wandb run,
tinker-rl-lab-world-class / v9eesnqc / Qwen3.5-4B / gsm8k / G=30), plus
`expected_effects` backfill on DAPO and GSPO (the two zero-evidence deltas whose
source papers are arXiv-verified and make explicit per-axis predictions).

## What was built

1. `scripts/p5p8/p6_registry_field_coverage.py` (~280 LoC, stdlib only)
   — reads `registry/entries/*.json`, classifies per `(record_type, framework)`,
   normalizes method labels (case + punctuation stripped) so `PPO` ==
   `ppo` == `pporeinforce`, emits:
   - `experiments/results/p5p8/registry_field_coverage_matrix.tsv`
     (35 entries × 12 cols: 8 optional-block flags + n_populated + n_total +
     coverage_pct + citation_arxiv)
   - `experiments/results/p5p8/registry_field_coverage_gaps.tsv`
     (entries with ≥2 null optional blocks, sorted by #null)
   - `experiments/results/p5p8/registry_method_coverage_matrix.tsv`
     (framework × method registration matrix; cells = '+' if registered,
     '*' if in-ledger-only, empty otherwise)
   - `experiments/results/p5p8/registry_field_coverage_summary.json`
   - `experiments/results/p5p8/registry_method_coverage_summary.json`

2. **New entry:** `registry/entries/delta_ppo.json` (Vein d) — PPO entry with
   4 named techniques (value_head, gae_lambda, ratio_clip, epoch_k) and
   3 `expected_effects` rows (zvf/reward_mean/zvf_risk_mean). Measured block
   intentionally null with explicit same-stack-arm criterion (only 1 wandb
   PPO run exists; cross-stack relative to the N2 GRPO reference).

3. **Backfilled:** `registry/entries/delta_dapo.json` and
   `registry/entries/delta_gspo.json` — each gained 3 `expected_effects` rows
   sourced from the respective source papers (Yu et al. 2025 arXiv:2503.14476;
   Qwen team 2025 arXiv:2507.18071).

4. New `paper/sections/p6_iter78_field_coverage.tex` (1 table, 4 paragraphs,
   no new equations) added to `paper/paper_P6_registry.tex`.

5. `P5P8_IMPROVEMENTS.md` row 92 appended.

6. `AUTORESEARCH_FINDINGS.jsonl` line appended.

## Headline findings (falsifiable)

### H1 — registry methods grew from 14 → 17 normalized labels

After adding `delta_ppo.json` (and the 2 expected-effects backfills on
DAPO and GSPO), the registry now covers **17 distinct normalized methods**
(was 14): 12 stack labels (aero/areal/cppo/dapo/drgrpo/es/gift/grpo/gspo/
mcgrpo/ngrpo/scafgrpo) + 15 variant names, with overlap of 10 stack labels
shared with variant names + 5 variant-only (Adaptive-G, DAPO, Dr.GRPO,
GSPO, LitePPO).

### H2 — measured-evidence coverage unchanged at 10/15 (the experimental-design constraint)

Variants with at least one `measured[]` row stayed at **10/15**. The 5
zero-measured variants (DAPO, GSPO, LitePPO, REINFORCE, PPO) all fail the
**same-stack-arm criterion** that the iter-74 row 87 audit identified: a
single same-stack run with only the named variant change applied. The
coverage-audit tool makes this structurally explicit: each zero-measured
entry now carries (a) the `expected_effects` block (where the source
paper makes a per-axis prediction) and (b) a notes-block explanation of
the gap. **Information gain**: 3 of 5 zero-measured variants (DAPO, GSPO,
PPO) now have `expected_effects` rows that the future auditor can score
against the measured block once a same-stack run lands.

### H3 — average optional-block coverage: variant_delta 52.0%, stack 90.0%

Two record-type signatures differ markedly. The 20 stack records carry
the 4 `outcomes` sub-fields (mean_last10_train_reward / mean_zvf /
heldout_delta / rollouts) plus min_report, notes, variant_deltas_applied
— **90% field-population average**. The 15 variant-delta records carry the
5 optional blocks (measured / expected_effects / claim_validation /
measured_yield_residual / controller_predicted_savings_per_rollout) —
**52% field-population average**, driven entirely by the iter-46 extension
(measured, expected_effects, claim_validation) and iter-66/iter-72
extension (measured_yield_residual, controller_predicted_savings_per_rollout).

### H4 — methods_only_in_ledger reduced from 6 → 5

| H | Method in ledger | Status before | Status after iter 78 |
|---|---|---|---|
| 1 | grpo / GRPO | registered (tinker_grpo + trl_grpo + verl_grpo + 3 colab) | unchanged |
| 2 | ppo | **missing** | **closed** (delta_ppo.json) |
| 3 | reinforce / ppo_reinforce | registered (delta_reinforce, no measured) | unchanged |
| 4 | trl-grpo | not registered as variant | open (TRL is a framework, not a variant) |
| 5 | per-group regression; continuous reward; population-standardized advantage | one-off custom algo | open (transparent miss) |
| 6 | UNKNOWN / '' | unparseable rows | open (data-quality issue) |

Closing PPO lifts **methods overlap from 2 → 3** of the 9 ledger raw labels
(grpo, ppo, reinforce). The remaining 5 ledger-only labels are either
framework-axes (TRL-GRPO), one-off custom experiments (per-group
regression), or data-quality issues (UNKNOWN / empty).

### H5 — gaps table identifies exactly 2 fully-null entries

`registry_field_coverage_gaps.tsv` reports the 9 entries with ≥2 null
optional blocks; the 2 fully-null entries (LitePPO and REINFORCE) carry
all 5 blocks null. The 7 partially-populated entries (Adaptive-G, CPPO,
DrGRPO, ES, DAPO, GSPO, PPO) carry 1 populated block each (typically
`measured[]` or `expected_effects[]`).

## Cross-paper coupling

- **P5 iter-69 row 81 placebo-replacement** — independently confirmed:
  the 14→15 variants gap was driven by **registry-side design**
  (PPO never had an entry even though the worktree had a run), not
  by **ledger-side coverage** (PPO was always in the experiment ledger).
- **P5 iter-77 row 91 cross-corpus portability** — independently sharpened:
  the C1/C2/C3 corpora carry the algorithm-axis (Item 2 KL = ref_policy_kl)
  that the registry's `expected_effects` block records. C4–C7 corpora
  would benefit from a paired-difference row entry-type — not a
  registry schema change.
- **P6 iter-74 row 87 DrGRPO measured evidence** — the same same-stack-arm
  criterion that closed DrGRPO's zero-evidence status also gates the
  remaining 5 zero-measured variants. Closing them requires **new
  Tinker 5+ method panel** (DAPO/GSPO/LitePPO/REINFORCE/PPO with only
  the named variant change), which is the experimental-design
  recommendation the iter-74 audit surfaced.
- **P5 iter-65 row 76 manifest × telemetry coupling** — the field-level
  coverage audit provides the same completeness measurement at the
  registry level that the manifest × telemetry coupling provides at the
  experiment level. Both surface "schema is fine; corpus is the gap."

## Sharp operational recommendation

1. **Add `delta_trl-grpo.json`** if TRL-GRPO is to be elevated from
   framework-axis (TRL is in `framework.name`) to variant-axis (TRL-GRPO
   is a base GRPO variant under the TRL framework). Currently TRL-GRPO
   is captured implicitly via the `tinker_grpo_*` + `trl_grpo_qwen3-8b_gsm8k`
   stack records; promoting it to a variant_delta entry would close the
   ledger-only gap.
2. **Run a 5+ method same-stack panel** (DAPO, GSPO, LitePPO, REINFORCE,
   PPO with only the named variant change vs the N2 GRPO reference) on
   Qwen3.5-4B / gsm8k / G=8. This single Tinker run would close 5 of the
   5 remaining zero-measured variant deltas in one batch. **Estimated
   cost**: ~5× of the N2 baseline run (≤ 5× Tinker-hours, well under
   the 35h weekly budget).
3. **Adopt `expected_effects` as required** for any newly-added
   variant-delta record. Currently it is optional; making it required
   would force new entries to surface the source-paper predictions
   rather than letting the entry hide behind an empty measured block.

## Reproducibility

```
cd /home/claude/tinker-rl-lab-minimax
python3 scripts/p5p8/p6_registry_field_coverage.py
python3 scripts/p5p8/registry_validate.py
```

Validation gate: all 35 entries PASS schema validation; 35/35 badges
score 96+ (stack records) or 100 (variant records).

## Status

Validated.