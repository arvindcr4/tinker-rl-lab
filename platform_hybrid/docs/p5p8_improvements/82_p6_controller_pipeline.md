# Iter-70 — P6 controller-predicted-savings block (closes registry→controller pipeline)

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Vein:** fresh vein (iter-66 row 77 mint recommendation + iter-67 row 78)
**Date:** 2026-07-05

## What this iteration does

Extends the iter-66 row 77 `measured_yield_residual` block (a *measurement*)
with a third additive-optional block that lifts the iter-67 row 78 controller
counterfactual (an *intervention*) into the registry. After iter-70 a
downstream consumer can ask *which trigger the iter-51 adaptive-G controller
should use on this stack* directly from the registry without re-running
iter-67.

## Headlines

- **Schema additive-only** — new optional property
  `controller_predicted_savings_per_rollout` on both `stack_record.outcomes`
  and `variant_delta_record`. All fields nullable, `additionalProperties:
  false`, no `required` clause → **34/34 PASS** unchanged.
- **7 entries populated** — 4 stack entries (`tinker_{grpo,aero,areal,gift}_qwen3.5-4b_gsm8k.json`)
  + 3 variant-delta entries (`delta_{aero,areal,gift}.json`). Remaining 27
  entries ship `null` (unreported → auditor scores as gap).
- **60 predictions lifted** — 4 methods × 3 triggers × 5 thresholds of
  `trigger/threshold/fires/saved/missed/saved_per_fire/savings_per_rollout_{pt,lo,hi}/cost_ratio_pt`.
- **Sharpest registry-readable finding** — at matched cost ratio
  1.45–1.55, GIFT under `ddiv_triage@τ=0.05` is the **highest
  saved-per-fire** entry in the entire registry
  (19 saves / 13 fires / `saved_per_fire=1.4615`).

## Why this matters

The block is the **third additive-optional extension** in the iter-28 →
iter-62 → iter-66 → iter-70 chain. Together they implement a
*measurement → trigger → registry* pipeline:

1. iter-28 `outcomes.ci_method` — *how the CI was computed*.
2. iter-62 `outcomes.coverage` — *how the entry is reporting-covered*.
3. iter-66 row 77 `measured_yield_residual` — *what the signed yield-residual is*.
4. iter-70 `controller_predicted_savings_per_rollout` (this iter) —
   *which controller trigger dominates on the N2 corpus*.

Each block is independently auditable, nullable on entries that have not
been N2-measured, and backward-compatible. Reviewers can audit any of the
three blocks in isolation without trusting the others.

## Cross-paper coupling

- **P7**: the iter-51 controller trigger is now machine-readable per method.
  A future P7 audit can ask *"on which methods does the optimal τ vary by
  >0.02?"* directly from the registry.
- **P6**: the three-block family is the first operational decomposition of
  *how a registry entry earns trust* (CI provenance + signed measurement +
  controller prediction).
- **P5**: every N2 entry now carries a stack axis *and* an outcome axis,
  partially counterweighting the iter-65 row 76 4-of-7 placebo finding.

## Reproduction

```bash
python3 scripts/p5p8/p6_controller_predicted_savings.py --apply   # idempotent
python3 registry/query.py validate                                # 34/34 PASS
```

## Outputs

- `scripts/p5p8/p6_controller_predicted_savings.py` (~290 LoC, stdlib only)
- `experiments/results/p5p8/p6_controller_predicted_savings.tsv` (60 rows)
- `experiments/results/p5p8/p6_controller_predicted_savings_summary.json`
- patched `registry/schema.json` (2 new additive-optional properties)
- patched 4 `registry/entries/tinker_*_qwen3.5-4b_gsm8k.json` stack entries
- patched 3 `registry/entries/delta_{aero,areal,gift}.json` variant deltas
- patched `registry/query.py` (validator path unchanged)
- patched `paper/sections/p6_controller_pipeline.tex` (new section)
- patched `paper/paper_P6_registry.tex` (single `\input` line)
- `paper/paper_P6_registry.pdf` rebuilds to **38 pages / 0 errors / 0
  undefined citations** (was 37, +1 page)
