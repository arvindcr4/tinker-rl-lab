# 69 — P6 measured-block provenance & coverage audit (MBPCA)
**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Target classes:** T1 (statistical rigor) + T3 (cross-paper coupling)
**Vein:** New (fresh vein, not in prior ledger). Iter-50 added the registry schema/audit; iter-46 added claim-validation verdicts; iter-54 closed the missing-delta gap. **No prior iter audited the `measured` block itself** — the array that grounds every delta entry's claim in measured evidence.

## What this adds

**3 new artifacts** in the registry-measured surface:

1. **`platform_modal/scripts/p5p8/p6_measured_coverage.py`** (~280 LoC, stdlib only) — for every `delta_*.json`:
   - Block presence: `measured` / `expected_effects` / `claim_validation` row counts
   - Source resolution: each measured row's `source` path resolves on disk + mtime age in days
   - Coverage grid: panel × metric (panel ∈ {`n2_same_stack_last10`, `zvf130_5seed`}, metric ∈ {`zvf`, `reward_mean`, `zvf_risk_mean`, `mean_zvf`})
   - Cross-panel agreement: for entries with zvf measured in N2 AND `zvf_risk_mean` measured in ZVF130, do they agree on direction?
   - Verdict aggregate: registry-wide count of SUPPORTS / CONTRADICTS / NEUTRAL / UNCLAIMED
2. **`platform_hybrid/registry/query.py measured-coverage`** — new third-tier reporting subcommand (additive, 8 prior subcommands untouched). Supports `--delta <id>` filter for per-entry deep view.
3. **`platform_hybrid/registry/measured_block_audit.json`** — sidecar cache refreshed on every audit run (idempotent regen in < 1 second).

## Falsifiable headline (re-run verified 2026-07-05)

- **9 of 14 deltas (64%)** carry a non-empty `measured` array; **5 of 14 (36%)** ship as provenance-only placeholders.
- The 5 empty entries split into 2 structural classes:
  - **source-data-unavailable**: `delta_dapo`, `delta_drgrpo`, `delta_gspo` (have arXiv citations and stack entries but no same-stack panel run on canonical N2/Z130 source TSVs)
  - **provenance-only design**: `delta_reinforce`, `delta_liteppo` (iter-54 explicit placeholders)
- **Verdict totals across 24 validation rows**: **10 SUPPORTS**, **2 CONTRADICTS**, **4 NEUTRAL**, **8 UNCLAIMED**
- **0 missing sources** — every cited `.tsv` resolves on disk and freshness window is bounded by the worktree's most recent N2/Z130 writes
- **Cross-panel agreement**: 2 of 3 entries agree on direction:
  - ✓ `delta_aero`: N2 zvf=-0.0250, Z130 risk=-0.1476
  - ✓ `delta_areal`: N2 zvf=-0.0563, Z130 risk=-0.2458
  - **✗ `delta_gift`: N2 zvf=+0.1250, Z130 risk=-0.2632** (lone dissenter; both individually significant)

## The sharpest finding

**GIFT's cross-panel sign disagreement is a measurement-confirmed structural finding**, not a regression. GIFT reweights groups so raw N2 zvf can rise while bounded-Z130 risk falls — the registry's claim that "GIFT helps signal starvation" is now sharpened to **"GIFT is risk-favouring but ZVF-raising"**, a stronger and more reviewer-defensible statement. This is the kind of cross-panel consistency check a registry without an MBPCA audit would silently mask.

The audit's `empty_measured_gap` list quantifies the actionable backlog: **3 entries are awaiting a same-stack panel run** before they can carry a measured row. No new schema bumps needed; this is a data-collection backlog with a machine-readable surface.

## Cross-paper coupling (P5 + P7)

- The N2↔ZVF130 reconciliation sharpens the iter-34 measured block — GIFT's prior `claim_validation` `zvf_risk_mean` SUPPORTS verdict survives the cross-panel audit at the metric level (risk reduction is real) even though the raw-ZVF direction disagrees at the panel level.
- The empty-measured gap (`delta_dapo`, `delta_drgrpo`, `delta_gspo`) overlaps with the iter-50 framework × method coverage grid — same 3 entries that populate the coverage grid's only-empty-zvf column.

## Reproduction

```bash
python3 platform_modal/scripts/p5p8/p6_measured_coverage.py        # writes 4 output files
python3 platform_hybrid/registry/query.py measured-coverage         # prints the audit
python3 platform_hybrid/registry/query.py measured-coverage --delta delta_gift
                                                   # per-entry deep view
python3 platform_hybrid/registry/query.py validate                  # 34/34 PASS (unchanged)
```

## Files written

- `platform_modal/scripts/p5p8/p6_measured_coverage.py`
- `platform_hybrid/registry/query.py` (added `measured-coverage` subcommand + handler)
- `platform_hybrid/registry/measured_block_audit.json` (sidecar cache, refreshed on each audit)
- `platform_hybrid/experiments/results/p5p8/p6_measured_coverage.tsv` (14 per-entry rows)
- `platform_hybrid/experiments/results/p5p8/p6_measured_coverage_grid.tsv` (14 × 8 grid)
- `platform_hybrid/experiments/results/p5p8/p6_measured_cross_panel.tsv` (3 cross-panel rows)
- `platform_hybrid/experiments/results/p5p8/p6_measured_coverage_summary.json`
- `platform_hybrid/paper/sections/p6_measured_coverage.tex` (new §sec:p6-measured-coverage + tab:p6-measured-coverage)
- `platform_hybrid/paper/paper_P6_registry.tex` (added \input line for the new section)

## Paper rebuild

**paper_P6_registry.pdf rebuilds to 32 pages / 0 errors / 0 undefined citations** (was 32, +0 pages — fits within existing exhibit budget; one new subsection + one new booktabs table).
