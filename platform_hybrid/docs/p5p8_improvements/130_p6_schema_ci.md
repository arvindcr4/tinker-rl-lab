# Iter 130 — P6 Schema CI Validator + Stale-CI Auto-Patch

**Pillar:** P6 (GRPO-Registry, machine-readable catalog of stacks and variant
deltas) — Vein (c) **schema validation script + CI-style check that every
entry parses**, combined with vein (a) **validate existing entries against
measured behavior**.

## Goal

Ship a deterministic, stdlib-only validator that any registry edit must pass.
Auto-patch the stale `ci_method=point_no_perseed_sd` rows that the iter-128
audit flagged, so the registry reaches a validated steady state.

## What shipped

1. **`scripts/p5p8/p6_iter130_schema_ci.py`** — single-pass CI for every
   `registry/entries/*.json`:
   - `parse_ok` (file loads as JSON)
   - `schema_ok` (oneOf-branch required fields present, MIN-REPORT items
     populated or explicitly null)
   - `badge` A/B/C/D from measured[] count + significance + staleness
   - emits
     `experiments/results/p5p8/p6_iter130_schema_ci.{tsv,json}` and a patch plan.

2. **`scripts/p5p8/p6_iter130_patch_stale_mag.py`** — recomputes the 5 stale
   `mag_mean` rows (`delta_cppo`, `delta_es`, `delta_mcgrpo`, `delta_ngrpo`,
   `delta_scafgrpo`) from `zvf_iter130_risk_index.tsv` per-seed `mean_zvf`
   with paired-seed bootstrap (B=2000, seed=20260705), writes patched JSON
   entries in place, and logs the diff to
   `experiments/results/p5p8/p6_iter130_patch_log.{tsv,json}`.

## Findings (real data, deterministic)

Pre-patch and post-patch validator pass:

| metric | pre | post |
| --- | --- | --- |
| n_entries | 39 | 39 |
| parse_ok | 39/39 | 39/39 |
| schema_ok | 39/39 | 39/39 |
| badge A | 4 | **9** |
| badge B | 1 | 1 |
| badge C (stale) | 5 | **0** |
| badge D (no measured) | 29 | 29 |
| stale_method_rows | 5 | **0** |
| total measured rows | 38 | 38 |
| significant rows | 16 | **21** |

Schema health: **PASS** (39/39). 5 entries gained significant rows from the
recompute.

### Recomputed `mag_mean` deltas (paired-seed bootstrap, B=2000)

| entry | method | Δ (vs grpo) | 95% CI | sig |
| --- | --- | --- | --- | --- |
| delta_cppo | CPPO | -0.18558 | [-0.18788, -0.18318] | ✓ |
| delta_es | ES | -0.36656 | [-0.37150, -0.36304] | ✓ |
| delta_mcgrpo | MCGRPO | -0.33474 | [-0.34166, -0.32892] | ✓ |
| delta_ngrpo | NGRPO | -0.18042 | [-0.18166, -0.17916] | ✓ |
| delta_scafgrpo | SCAFGRPO | -0.16414 | [-0.16728, -0.16076] | ✓ |

All five `mag_mean` deltas favor `grpo` (negative Δ on the
risk-weighted-difference scale). The paired-seed CIs are very tight because
the per-seed `mean_zvf` is highly stable across seeds; this matches the
frontier-synthesis insight that ZVF is observed signal availability, not latent
difficulty.

### Coverage notes

- 29 entries (75%) carry **no measured rows** (badge D): every catalog-record
  `stack` record (`tinker_*`, `verl_*`, `trl_*`, `openrlhf_*`, `colab-open_*`
  and the 9 `zvf130_*` per-method batch harnesses) is by design a stack
  manifest, not a measured-effect row — measured effect rows live on the
  `delta_*` records.
- 9 entries (badge A) now sit at the strongest tier: `delta_aero`,
  `delta_areal`, `delta_drgrpo`, `delta_gift`, `delta_cppo`, `delta_es`,
  `delta_mcgrpo`, `delta_ngrpo`, `delta_scafgrpo`. The first four were
  already at A pre-iter-130; the latter five gained A by patching the
  stale `mag_mean` row.
- `delta_adaptiveg` stays at B (only 2 measured rows; needs an N2
  same-stack pass to earn A — proposed follow-up).

## How to re-run

```bash
python3 scripts/p5p8/p6_iter130_schema_ci.py
python3 scripts/p5p8/p6_iter130_patch_stale_mag.py
python3 scripts/p5p8/p6_iter130_schema_ci.py     # re-validate
```

Deterministic (B=2000, seed=20260705). Stdlib only.

## Files added

- `scripts/p5p8/p6_iter130_schema_ci.py`
- `scripts/p5p8/p6_iter130_patch_stale_mag.py`
- `experiments/results/p5p8/p6_iter130_schema_ci.tsv`
- `experiments/results/p5p8/p6_iter130_schema_ci.json`
- `experiments/results/p5p8/p6_iter130_schema_ci_patch_plan.tsv`
- `experiments/results/p5p8/p6_iter130_patch_log.tsv`
- `experiments/results/p5p8/p6_iter130_patch_log.json`
- `registry/entries/delta_{cppo,es,mcgrpo,ngrpo,scafgrpo}.json`  (5 patched)

## Next veins to consider

- Promote `delta_adaptiveg` to badge A by adding an N2 same-stack
  `mag_mean` row, closing the Adaptive-G measurement gap.
- Move the validator into a CI step (e.g. `registry/test_schema.py`)
  so registry edits cannot merge without parse_ok + schema_ok.
- Add a `measured_panel_concordance` check that flags same-panel,
  same-method delta rows whose CIs disagree (e.g. `last10` vs
  `full40` zvf under-sign flip).
