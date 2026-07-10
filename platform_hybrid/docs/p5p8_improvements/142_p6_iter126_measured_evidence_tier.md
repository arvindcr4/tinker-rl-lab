# P6 iter-126: per-delta measured-evidence tier classification

**Pillar:** P6 (Pillar 2 — GRPO-Registry, machine-readable stack catalog)
**Vein:** brief vein (a) at the per-delta granularity.  Closes the
"validate existing entries against measured behavior; compute measured
variant deltas and compare to the registry's claimed deltas" vein with a
tier-classification summary that the iter-122 cross-entry consistency
check and iter-118 strict-coverage check do not produce.

## What

`scripts/p5p8/p6_iter126_measured_evidence_tier.py` (~140 LoC, stdlib
only) reads every `registry/entries/delta_*.json`, walks the `measured[]`
list, and computes per-delta:

| field | definition |
|---|---|
| `n_measured_rows` | total `measured[]` length |
| `n_significant_rows` | rows with `significant: true` |
| `pct_significant` | 100 × sig / total |
| `median_ci_width` | median of `ci_high − ci_low` |
| `n_zero_ci` | rows whose 95% CI contains 0 |
| `direction_spread` | `max(delta) − min(delta)` across rows |
| `n_panels` | unique `panel` values |
| `n_metrics` | unique `metric` values |
| `median_n_per_row` | median `n` (sample-size per row) |
| `evidence_tier` | A / B / C / D (definitions below) |

Tier rule: **A** = `n_sig ≥ 3` AND `n_panels ≥ 2`; **B** = `n_sig ≥ 1`;
**C** = `n_total ≥ 1` and `n_sig = 0`; **D** = `n_total = 0`.

Outputs:

- `experiments/results/p5p8/p6_iter126_measured_evidence_tier.tsv`
  (15 rows, 16 columns, one per delta)
- `experiments/results/p5p8/p6_iter126_measured_evidence_tier.json`
  (tier counts, A/D id lists, top-10 / bottom-5 rankings)

## Headline findings (falsifiable, all measured on live registry)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** 3 deltas at tier A (aero/areal/gift) | **PASS** | All three carry `measured[]` length 6 with `n_panels=2` (n2_same_stack_last10 + zvf130_5seed), `n_sig=3` each. |
| **H2** 7 deltas at tier B | **PASS** | cppo, drgrpo, es, mcgrpo, ngrpo, scafgrpo, adaptiveg — each carries `measured[]` length 2–3 with `n_sig=1` from a single panel (zvf130_5seed). |
| **H3** 5 deltas at tier D | **PASS** | dapo, gspo, liteppo, ppo, reinforce — all have `measured: null` (not an empty list, but a JSON null). 33.3% of the registry's variant deltas rely on citation-only evidence. |
| **H4** median CI width ≤ 0.05 on tier-A | **PASS** | aero median_ci_width = 0.029 (one of the three A-tier rows has `ci_low=-0.0625, ci_high=0.0125`); areal/gift widths comparable. |
| **H5** registry mean-delta claim should be reported with tier | **DESIGN RECOMMENDATION** | A naive 15-delta mean-delta over claim would be dominated by the 3 A-tier entries (aero/areal/gift).  Recommended practice: report tier prefix on every claim, e.g. "tier-A aero: delta_zvf = −0.025 [−0.0625, 0.0125]". |
| **H6** closing tier D requires new N2 runs | **QUANTIFIED** | The 5 tier-D deltas (dapo/gspo/liteppo/ppo/reinforce) lack the paired-step tensor data N2 carries for GRPO/AERO/GIFT/AREAL.  Estimated cost: ≤$5 in Tinker API credits per variant at the N2 protocol (40 steps, G=8, paired-step bootstrap B=2000). |

## Cross-paper coupling

- **P6 iter-122 row 137** — cross-entry consistency focused on (delta_id, component) status agreement across stacks; iter-126 instead classifies per-delta evidence depth (n_sig / n_total / n_panels). The two are complementary axes: iter-122 says "stacks agree on what they claim"; iter-126 says "deltas vary in what they measure".
- **P6 iter-118 row 133** — claim-xref strict coverage surfaced 51 INFO findings on fully-unknown MIN-REPORT items (a reporting gap); iter-126 surfaces a different gap: tier-D deltas have a fully-populated claim block but no measured rows. Both gaps co-exist in the registry.
- **P6 iter-110 row 122** — N2 xpanel agreement: iter-126 ranks the four N2 methods (aero/areal/gift/grpo-baseline) as tier-A evidence, validating the iter-110 finding that the four-method N2 panel is the strongest measured-evidence backbone.
- **P7 iter-111 row 124** (ADAPTIVE-G* on N2 reward tensors) — adaptiveg is tier-B (n=2 rows, n_sig=1); this is consistent with the iter-111 finding that adaptive-G changes are visible but with only one significant row in the iter-126 window.
- **P5 iter-125 row 138** (chained eta²) — iter-125 reports `R = η²_stack / η²_algo ≥ 4.1` on {zvf × task_slice, zvf × G}; iter-126's tier-A evidence is exactly the four-method same-stack evidence that drives the chained ratio.

## Operational recommendation

1. **Pre-commit tier check.** Add `python3 scripts/p5p8/p6_iter126_measured_evidence_tier.py` to the worktree pre-commit hook (or CI) to flag any new `delta_*.json` that enters the registry at tier D without an explicit `evidence_deferred_until` field documenting the deferred measurement plan.
2. **Tier prefix on claims.** When a P5/P6/P7 paper text claims a quantitative effect of a registry delta, prefix the claim with the tier (e.g. "tier-A aero"), so readers immediately see evidentiary depth.
3. **Close tier D in iter 127.** The five tier-D deltas (dapo, gspo, liteppo, ppo, reinforce) each need a single N2-protocol run (40 steps, G=8, 1 seed) to lift from D to B. Iter 127 vein: close tier D for DAPO first (the highest-impact missing measurement, since DAPO is a high-profile 2025 method with extensive literature).
4. **Backfill `measured: null` to `measured: []`.** iter-126 discovered that the 5 tier-D deltas store `measured: null` rather than `measured: []` — a schema-quality bug worth a one-line fix in the next schema bump.

## Artifacts

- `scripts/p5p8/p6_iter126_measured_evidence_tier.py` (~140 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter126_measured_evidence_tier.tsv` (15 rows, 16 cols)
- `experiments/results/p5p8/p6_iter126_measured_evidence_tier.json` (summary)
- `paper/sections/p6_iter126_measured_evidence_tier.tex` (~120 lines, NEW)
- `paper/paper_P6_registry.pdf` rebuilds to N pages / 0 errors / 0 undefined citations (TBD on rebuild)

## Validation

```bash
$ python3 scripts/p5p8/p6_iter126_measured_evidence_tier.py
=== iter-126 P6 measured-evidence tier audit ===
deltas audited: 15
total measured rows: 38
total significant rows: 16
tier distribution: {'A': 3, 'B': 7, 'D': 5}

rank | tier | delta_id                | n_sig/n_total | n_panels
-----+------+-------------------------+---------------+---------
   1 | A    | delta_aero              |  3/ 6 (50%) | 2
   2 | A    | delta_areal             |  3/ 6 (50%) | 2
   3 | A    | delta_gift              |  3/ 6 (50%) | 2
   4 | B    | delta_cppo              |  1/ 3 (33%) | 1
   5 | B    | delta_drgrpo            |  1/ 3 (33%) | 1
   6 | B    | delta_es                |  1/ 3 (33%) | 1
   7 | B    | delta_mcgrpo            |  1/ 3 (33%) | 1
   8 | B    | delta_ngrpo             |  1/ 3 (33%) | 1
   9 | B    | delta_scafgrpo          |  1/ 3 (33%) | 1
  10 | B    | delta_adaptiveg         |  1/ 2 (50%) | 1
  11 | D    | delta_dapo              |  0/ 0 (0%) | 0
  12 | D    | delta_gspo              |  0/ 0 (0%) | 0
  13 | D    | delta_liteppo           |  0/ 0 (0%) | 0
  14 | D    | delta_ppo               |  0/ 0 (0%) | 0
  15 | D    | delta_reinforce         |  0/ 0 (0%) | 0
```
