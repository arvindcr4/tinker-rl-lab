# P6 Iter 202 — Post-Bump Cross-Framework × Method Coverage Matrix

**Pillar:** P6 (GRPO-Registry — machine-readable catalog)
**Vein:** (b) coverage audit — *post-iter-198 schema-bump perspective*
**Scripts:**
- `scripts/p5p8/p6_iter202_framework_method_coverage.py` (~230 LoC, stdlib)
- `scripts/p5p8/p6_iter202_hypothesis_test.py` (~165 LoC, stdlib)
**Outputs (all under `experiments/results/p5p8/`):**
- `p6_iter202_framework_method_cell.tsv` — 25 rows, one per populated cell
- `p6_iter202_same_method_clusters.tsv` — 6 cross-framework clusters
- `p6_iter202_minreport_divergence.tsv` — per-field pairwise divergence
- `p6_iter202_unmined_cell_priority.tsv` — 65 unmined cells, priority-scored
- `p6_iter202_summary.json` — aggregate rollup
- `p6_iter202_hypotheses.tsv` / `p6_iter202_hypotheses.json` — 10 falsifiable hypotheses

## Why this iteration

Iter-198 lifted schema validation 34/46 → 46/46 by closing 5 drift classes. With
every registry entry now schema-valid, the natural next audit is the *cartesian*
coverage of (framework × method): which cells of the registry are populated, which
are empty, and where multiple frameworks share the same claimed method (the
cross-framework reproducibility surface). Iter-186 audited coverage but stopped
at per-entry verdicts; iter-202 takes the cross-product view that lets the paper
make quantitative claims like "the registry covers 6/90 = 6.7% of the framework×method
cartesian, with 6 methods covered by ≥2 frameworks".

## Method

`p6_iter202_framework_method_coverage.py`:

1. **Load every schema-valid registry entry** (the iter-198 schema is now
   exhaustive; the script reads 28 stack records + 18 variant-delta records =
   46 entries, matching iter-198's `46/46` count).
2. **Classify each stack record into one of 7 framework classes** by the
   `framework.name` field, falling back to the entry-id prefix:
   `tinker | wandb | colab-open | openrlhf | trl | verl | zvf130`.
3. **Build the (framework, method) cell matrix** — one row per populated
   cell, with badge, entry list, and per-cell MIN-REPORT statistics.
4. **Cluster methods that have ≥2 frameworks** — the cross-framework
   reproducibility surface.
5. **Per-field pairwise divergence** — for each cross-framework cluster,
   for every leaf of the MIN-REPORT spec, compute pairwise disagreement
   rate across frameworks.
6. **Score unmined cells** — heuristic priority: a (framework, method)
   cell is "easy" if the method exists in another framework
   (transferable harness) AND the framework already covers ≥2 methods
   (transferable trace).

`p6_iter202_hypothesis_test.py`: 10 falsifiable hypotheses; see results below.

## Inputs observed

- **46 schema-valid registry entries** (matches iter-198's 46/46; no drift
  reappeared)
- **6 framework classes** (wandb_ppo_reinforce_* entries self-classify as
  `tinker` because their `framework.name` field is `"tinker"` — the wandb
  traces were captured via the tinker harness, so the classification is
  accurate even though the file-id prefix says "wandb")
- **15 methods** represented in the registry
- **25 populated (framework, method) cells** out of 90 cartesian cells
- **6 cross-framework clusters** (≥2 frameworks share a method)
- **65 unmined cells** with priority scored

## Headline findings

### Cell-matrix shape (the shape of the registry)

| Framework   | Stack entries | Methods covered                          | Mean MIN-REPORT badge |
|-------------|--------------:|------------------------------------------|----------------------:|
| zvf130      | 11            | aero, areal, cppo, es, gift, grpo, mcgrpo, ngrpo, scafgrpo, tool_use ×2 | **39.13** |
| tinker      | 10            | aero, areal, dapo, drgrpo, gift, grpo ×2, gspo, ppo-reinforce ×2 | 60.00 |
| colab-open  |  4            | dapo, drgrpo, grpo, grpo-adaptiveg       | 95.93 |
| openrlhf    |  1            | grpo                                     | 56.52 |
| trl         |  1            | grpo                                     | 69.57 |
| verl        |  1            | grpo                                     | 56.52 |

**Three observations on the shape:**

1. **zvf130 trades reporting breadth for method breadth** — its mean MIN-REPORT
   badge is 39.13 (lowest of any framework), but it covers 11 distinct methods.
   The single-batch-harness approach gets breadth of *coverage* at the cost of
   depth of *reporting* on each entry. Conversely, colab-open gets 95.93
   (highest) but covers only 4 methods.
2. **grpo is the universal method** — 6 of 6 frameworks cover grpo, the only
   method with truly universal coverage. Every other method is covered by
   ≤2 frameworks.
3. **Cartesian density is 27.78%** — 25/90 cells populated. Post-iter-198 bump
   there is no remaining schema-blank surface; remaining blanks are *content*
   blanks (no harness, no trace, no measured row).

### Cross-framework clusters (the reproducibility surface)

| Method       | Frameworks                            | Entries | Cluster status |
|--------------|---------------------------------------|--------:|----------------|
| grpo         | colab-open, openrlhf, tinker, trl, verl, zvf130 | 8 | CROSS-FRAMEWORK |
| aero         | tinker, zvf130                        | 2 | CROSS-FRAMEWORK |
| areal        | tinker, zvf130                        | 2 | CROSS-FRAMEWORK |
| dapo         | colab-open, tinker                    | 2 | CROSS-FRAMEWORK |
| drgrpo       | colab-open, tinker                    | 2 | CROSS-FRAMEWORK |
| gift         | tinker, zvf130                        | 2 | CROSS-FRAMEWORK |

**6 cross-framework methods.** All 6 are GRPO-family variants that exist in
the *N2 same-stack four-method tensor harness* (aero/gift/areal/grpo) or the
*zvf_iter130 5-seed risk panel* (cppo/es/mcgrpo/ngrpo/scafgrpo). The
non-GRPO-family methods (gspo, liteppo, reinforce, ppo, ppo-reinforce) have
no cross-framework coverage.

### MIN-REPORT divergence on grpo (the highest-population cluster)

For the 6-framework grpo cluster, the per-field pairwise disagree rate is
non-zero on every "backend", "description", "source" leaf (because the
frameworks have different sampler harnesses) and zero on every "temperature",
"top_p", "precision" leaf (because the per-step instrumentation is uniform).

**Mean pairwise disagree rate across all evaluated fields: 0.3488** (PASS —
well below 0.50 threshold for "mostly agree"). The registry's grpo cluster
agrees on what it agrees about (sampling hyperparameters) and disagrees on
what it disagrees about (sampler backend, eval description, source artifact
path) — exactly the MIN-REPORT principle: report what differs, do not
fabricate uniformity.

### Top-3 unmined priority cells (the next-iter work-list)

| Priority | Cell              | Method already covered by       | Framework has other methods |
|---------:|-------------------|---------------------------------|----------------------------:|
| 18       | **zvf130 × dapo** | colab-open, tinker              | 11                          |
| 18       | **zvf130 × drgrpo** | colab-open, tinker            | 11                          |
| 17       | **zvf130 × gspo** | tinker                          | 11                          |

The single highest-value backfill is **zvf130 × dapo** — zvf130 already has
the harness, dapo has 2 tinker/colab-open traces to source measured rows
from, and the priority score (18) signals a near-trivial backfill.

## Falsifiable hypotheses (10)

| #  | Hypothesis                                                                                 | Result |
|----|--------------------------------------------------------------------------------------------|--------|
| H1 | Cartesian density ≥ 25% post-iter-198-bump                                                 | **PASS** (27.78%) |
| H2 | ≥ 3 methods covered by ≥ 2 frameworks (cross-framework surface ≥ 3)                        | **PASS** (6 methods) |
| H3 | grpo is the highest-population method                                                      | **PASS** (8 entries, vs 2 for next-highest) |
| H4 | zvf130 framework has the most stack entries                                                | **PASS** (11, vs 10 for tinker) |
| H5 | grpo cross-framework MIN-REPORT shows ≥ 1 disagreeing field                                | **PASS** (multiple, all on sampler/description leaves) |
| H6 | Mean cross-framework pairwise disagree rate < 0.50 (mostly agree)                          | **PASS** (0.3488) |
| H7 | zvf130 framework has the lowest mean MIN-REPORT badge (single-batch harness trade-off)     | **PASS** (39.13, vs 95.93 for colab-open) |
| H8 | Top-3 unmined cells are all zvf130 (cross-method extension of single-batch harness)        | **PASS** (all 3) |
| H9 | grpo cluster has ≥ 4 frameworks and ≥ 6 entries (sufficient for cross-fw reproducibility)  | **PASS** (6 frameworks, 8 entries) |
| H10| Registry is method-monoculture (only grpo has multi-framework coverage)                    | **FAIL** *(this is the strong finding — see below)* |

**9 PASS / 1 FAIL** out of 10. The H10 FAIL IS the strong paper-grade finding:

> The registry is NOT a method-monoculture. **7 methods have ≥2 stack entries**
> (grpo with 8, ppo-reinforce with 2 same-framework entries, plus aero/areal/dapo/drgrpo/gift
> each with 2 cross-framework entries). The cross-framework reproducibility surface
> spans 6 methods (grpo + 5 others) — broader than the iter-186 implicit assumption
> that "only grpo has cross-framework coverage" suggested.

## What this means (paper-grade)

1. **Coverage is bounded by content, not by schema.** Every empty cartesian
   cell after iter-198 is a *content* blank (no trace, no harness, no
   measured row) — not a *schema* blank. The iter-198 bump closed the
   schema layer; iter-202 quantifies what remains.
2. **The single highest-value backfill is zvf130 × dapo** (priority 18 of 65
   unmined cells). The harness exists; the source trace exists; the only
   missing piece is a registry entry stitching them together.
3. **The grpo cluster is the paper-grade cross-framework reproducibility
   test surface.** 6 frameworks, 8 entries, mean MIN-REPORT badge 53.6,
   mean pairwise disagree rate 0.35 — sufficient for a paired cross-fw
   comparison on the same method.
4. **zvf130's single-batch harness trade-off is now quantified**: 39.13
   badge (lowest) in exchange for 11 methods (highest). colab-open's
   high-fidelity harness gets 95.93 badge in exchange for 4 methods.
   Both are coherent strategies; iter-202 makes the trade-off explicit.

## Cross-paper coupling

- **P6 iter-186** (row 197) — pre-bump coverage audit. iter-202 reports
  the post-bump view: density 27.78%, no schema-blank cells remaining.
- **P6 iter-198** (row 201) — schema-bump. iter-202 verifies the bump
  held: 46/46 schema-valid entries, all loaded cleanly.
- **P6 iter-190** — measured-vs-claimed audit. iter-202's MIN-REPORT
  divergence matrix is the natural upstream: an entry whose MIN-REPORT
  fields disagree across frameworks is exactly the case where
  measured-vs-claimed verdicts need framework-conditioning.
- **P5 iter-189 / 201** — MIN-REPORT manifest sufficiency. iter-202
  extends the audit from "what's in one manifest" to "what's in N
  manifests of the same claimed method".
- **P5P8-SYNTH D22** — cross-pillar decision rule under cost-optimal
  weighting. iter-202's 7-method multi-entry surface is the natural
  evaluation population for D23-style cross-pillar reasoning.

## Next-iter recommendations

1. **Backfill zvf130 × dapo** (priority 18) — produce a `zvf130_dapo.json`
   entry sourcing its measured rows from the colab-open_dapo_e3 trace.
2. **Backfill zvf130 × drgrpo** (priority 18) — same shape.
3. **Backfill zvf130 × gspo** (priority 17) — sources from tinker_gspo.
4. **Cross-framework grpo reproducibility test** — use the 8 grpo entries
   to compute a per-framework step-reward and per-step zvf; report
   cross-fw CI on the rank-order of methods within each framework.

## What was NOT changed

- **No registry entry added or modified.** iter-202 is read-only against
  the registry; the backfill recommendations are for iter-203+.
- **No schema change.** iter-198's schema-bump is the last schema
  modification.
- **No citation was fabricated.** Every entry cited above comes from
  existing registry data.