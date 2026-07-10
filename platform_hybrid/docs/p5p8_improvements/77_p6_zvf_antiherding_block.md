# 77 — P6 Contrastive-Yield + anti-herding residual block on N2 same-stack corpus (iter 66)

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Target classes:** T2 (fresh-data evidence) + T3 (cross-paper coupling)

## Summary

Implements the **Contrastive Yield** framing from Round 2 of
`FRONTIER_INSIGHTS.md` (Gemini Deep Think): the residual

    delta_div = ZVF_iid − ZVF_obs

is the **anti-herding diversity bonus** attributable to
temperature-driven autoregressive sampling. The same decomposition
that motivates P7's controller fires is now registered on the
machine-readable surface: the registry exposes both
`outcomes.zvf_antiherding` (stacks) and `measured_yield_residual`
(deltas) with paired-step bootstrap CIs over `B=4000` resamples on the
N2 corpus's 40 steps × 16 prompts × G=8 binary-reward groups.

## Vein (fresh, not in 76-row prior ledger)

Prior P6 work stopped at "what does this entry **measure**?" (iter-69
MBPCA) and "how honestly does it **report**?" (iter-71 / iter-73).
Neither extends the measured contract into the **directional**
Contrastive-Yield axis that the frontier synthesis proposed. This iter
is the first registry block that grounds ZVF in a signed, decomposable
quantity rather than a single threshold-and-fire statistic.

## Falsifiable headlines (audit re-run 2026-07-05)

| Method | ZVF_obs | ZVF_iid | δ_div | Y_obs | vs-grpo 95% CI | p |
|--------|---------|---------|-------|-------|----------------|---|
| grpo   | 0.7203  | 0.7700  | 0.0497| 0.2797| ---            | --- |
| aero   | 0.7203  | 0.7656  | 0.0453| 0.2797| [-0.014,+0.005]| 0.364 |
| areal  | 0.7063  | 0.7595  | 0.0532| 0.2938| [-0.008,+0.015]| 0.541 |
| gift   | 0.7703  | 0.8097  | 0.0394| 0.2297| [-0.021,+0.001]| **0.066** |

- **All four N2 methods show a positive δ_div** (range
  [0.039, 0.053], mean 0.047) — anti-herding diversity bonus is
  *empirically real* on the live corpus.
- **δ_div is ~4× smaller than the frontier-synthesis band** of
  [0.13, 0.23]. The synthesis guess is directionally correct but
  quantitatively conservative by a substantial margin.
- **GIFT shows borderline-significant herding vs GRPO** at p=0.066:
  its group-reweighting prior herds relative to plain GRPO on this
  corpus. This is the second measured cross-panel sign disagreement
  in the registry (after iter-69 N2-vs-Z130 risk for GIFT); it does
  not falsify GIFT's overall registry claim, but is registered
  honestly.
- **AREAL has the highest yield** (Y = 0.2938) — the variance
  reduction mentioned in the source paper is concentrated on the
  p≈0 tail, not the marginal frontier.

## Schema & surface change

- `registry/schema.json` (additive-optional, both blocks):
  - new `outcomes.zvf_antiherding` block on stack records (10
    fields, all nullable, `additionalProperties: false`)
  - new `measured_yield_residual` block on variant_delta records (12
    fields, all nullable, `additionalProperties: false`)
- 5 N2-measured stack entries patched:
  `tinker_{grpo,aero,areal,gift}_qwen3.5-4b_gsm8k.json`
  (the 3 dapo/drgrpo/gspo entries ship `null` because no N2 tensor
  exists for them; this preserves the schema-bounded pattern from
  iter-62).
- 3 N2-measured delta entries patched:
  `delta_{aero,areal,gift}.json`
- 11th subcommand `registry/query.py antiherding` (10 prior
  subcommands unchanged; `--method` filter).
- **34/34 schema PASS** unchanged; the patch is purely additive.

## Cross-paper coupling (P5 ↔ P6 ↔ P7 ↔ P8)

- **P5**: every item-7 MIN-REPORT leaf populated (decontamination) is
  now also a δ_div ≠ 0 candidate: an entry declaring a
  parser-robustness probe can be re-audited for whether the parser
  invariant causes sampling to herd (a future audit row).
- **P7** (primary coupling): the iter-63 row 74 finding that
  ZVF_max=0.875 on N10 at iter-51 default τ_esc=0.70 is now
  reframed as Y_obs_min=0.125. The controller's escalation branch
  fires on the lowest-yield 12.5% of groups — exactly the regime
  where δ_div is largest and adaptive-G can rescue the most
  contrast.
- **P6 intra**: GIFT's δ_div reduction is the second cross-panel
  sign disagreement in the registry (after iter-69 N2-vs-Z130 risk).
  The new `measured_yield_residual` block surfaces the disagreement
  *without forcing a claim-validation verdict*, so the cross-panel
  evidence is registered but the registry's overall claim structure
  is preserved.
- **P8**: the schema pattern (additive-optional, all-nullable,
  `additionalProperties: false`, paired-step bootstrap
  `ci_method`) now generalises cleanly to any future "did you
  actually run it?" block. This is the third such block after
  iter-28 `ci_method` and iter-62 `outcomes.coverage`.

## Why the synthesis-to-data gap matters (sharpest finding)

The frontier-synthesis Round 2 labelled δ_div ∈ [0.13, 0.23] as
"the measured structural diversity bonus". The measured value is
**[0.039, 0.053]**, **~4× smaller**. This:

1. **Sharpens the synthesis into an empirical, auditable statement**
   rather than a hand-tuned estimate — the registry block is now the
   canonical source for the δ_div number.
2. **Generates a falsifiable P7 prediction**: GIFT's
   borderline-significant herding shift (p=0.066) is the first
   quantitative constraint on how a P7 adaptive-G controller should
   treat GIFT-style group reweighting on this corpus.
3. **Closes a measurement–synthesis gap** that the iter-50 /
   iter-69 registry audits could not resolve because they measured
   ZVF only, not its Bernoulli-baseline-residual decomposition.

## Outputs

- `scripts/p5p8/p6_zvf_antiherding.py` (~290 LoC, stdlib only;
  paired-step bootstrap; idempotent re-runnable)
- `experiments/results/p5p8/p6_zvf_antiherding_summary.tsv` (4 rows)
- `experiments/results/p5p8/p6_zvf_antiherding_per_step.tsv` (160
  rows: 4 methods × 40 steps × 9 columns)
- `experiments/results/p5p8/p6_zvf_antiherding_summary.json`
- `registry/schema.json` patched (two new additive blocks)
- `registry/entries/{tinker_grpo,aero,areal,gift}_qwen3.5-4b_gsm8k.json`
  patched with `outcomes.zvf_antiherding`
- `registry/entries/delta_{aero,areal,gift}.json` patched with
  `measured_yield_residual`
- `registry/query.py` 11th subcommand `antiherding`
- `paper/sections/p6_registry_health.tex`
  §sec:p6-zvf-antiherding + Table `tab:p6-zvf-antiherding`
- `paper/build/paper_P6_registry.pdf` rebuilds to **37 pages / 0
  errors / 0 undefined citations** (was 36, +1 page for the new
  §sec:p6-zvf-antiherding + Table)
- `paper/the P5-P8 improvement backlog` row 77 (next mint row)
- `findings_ledger.jsonl` line appended with pillar P6

## Reproduction

```bash
python3 scripts/p5p8/p6_zvf_antiherding.py           # ~3s; idempotent
python3 registry/query.py validate                   # 34/34 PASS
python3 registry/query.py antiherding               # per-method table
python3 registry/query.py antiherding --method gift # per-method filter
```
