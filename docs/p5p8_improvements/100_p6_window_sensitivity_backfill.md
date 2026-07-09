# 100 — P6 window-sensitivity schema backfill (iter 84 JOB B / SYNTH)

## Falsifiable headlines

- **H1 — schema is extended additively with 2 new optional fields** on `measured_delta`: `window_sensitivity` (string enum: `STABLE` / `STABLE-DIRECTION-MAG-SHIFT` / `FRAGILE-SIGN-FLIP`; default `STABLE-DIRECTION-MAG-SHIFT`) and `robust_panel` (string enum: `full40` / `last20` / `last10` / `last5` / `none`). Backward compatible: 35/35 entries still `PASS` jsonschema validation post-bump (`ok=35, fail=0`).
- **H2 — 3 delta entries backfilled (`delta_aero`, `delta_gift`, `delta_areal`), 6 measured[] rows carry the new fields**, computed from the iter-82 row 97 panel-by-panel re-measurement (B=2000 paired-step bootstrap, seed=20260705). Every backfilled row's `window_sensitivity` is `STABLE-DIRECTION-MAG-SHIFT` (same sign at every panel; magnitude varies panel-by-panel) — the iter-82 default, no FRAGILE-SIGN-FLIPs found once we apply the noise-floor rule (|δ|>0.005) that the iter-82 data requires.
- **H3 — `robust_panel` resolves as**: aero.zvf→`none` (no same-sign panel sig), aero.reward_mean→`last10`, gift.zvf→`full40`, gift.reward_mean→`full40`, gift.mean_len→`last5`, areal.reward_mean→`last10`. **The 4/6 cells that ARE robust resolve to last10 or full40** — confirming the iter-82 row 97 finding that the registry's panel=`n2_same_stack_last10` choice was a *defensible* default, not a cherry-pick.
- **H4 — `additionalProperties: false` preserved**; schema bump is purely additive (no required-field changes, no removals, no enum tightening).

## Per-cell classification table

| method | metric | full40 | last10 | last5 | window_sensitivity | robust_panel |
|---|---|---:|---:|---:|---|---|
| aero | zvf | δ=0.000 NS | -0.025 NS | -0.063 **SIG** | STABLE-DIRECTION-MAG-SHIFT | none |
| aero | reward_mean | -0.007 NS | -0.014 **SIG** | -0.020 **SIG** | STABLE-DIRECTION-MAG-SHIFT | last10 |
| gift | zvf | +0.050 **SIG** | +0.125 **SIG** | +0.113 **SIG** | STABLE-DIRECTION-MAG-SHIFT | full40 |
| gift | reward_mean | +0.011 **SIG** | +0.016 NS | +0.023 NS | STABLE-DIRECTION-MAG-SHIFT | full40 |
| gift | mean_len | -0.74 NS | -3.89 NS | -11.61 **SIG** | STABLE-DIRECTION-MAG-SHIFT | last5 |
| areal | reward_mean | -0.005 NS | -0.020 **SIG** | -0.020 **SIG** | STABLE-DIRECTION-MAG-SHIFT | last10 |

(`bold SIG` = CI excludes zero; NS = CI includes zero. Noise floor |δ|>0.005 applied to sign comparisons: full40 δ=0.0 (aero.zvf) is NOT counted as a sign flip.)

## Why this is the right SYNTH top item

| # | evidence | impact | readiness | rank |
|---|---|---|---|---|
| 1 | this iter (P6 window-sensitivity schema backfill) | **HIGH** (closes iter-82 mint recommendation; lands the deferred schema bump; backfills 3 entries; paper-facing) | **READY** (additive-only, 35/35 entries PASS; data exists from iter-82) | **1st** |
| 2 | P7 Iso-G fired-step registry backfill | medium (pure schema) | not ready (data not yet exported in registry format) | 2nd |
| 3 | P5 v2.2 MIN-REPORT schema formalization (Items 14-17) | high (most MIN-REPORT uplift) | not ready (would require v2.2 schema bump + v1→v2.2 migration) | 3rd |

## Cross-paper coupling

- (i) **P6 iter-82 row 97** — this iter closes the iter-82 mint recommendation. The iter-82 paper section promised a deferred schema patch: this iter lands it.
- (ii) **P6 iter-66 row 77 / iter-70 row 82 / iter-78 row 92** — registry coverage audits as a class; the iter-84 window-sensitivity field is the FIRST coverage-style field that scores "did this entry actually survive a panel-by-panel remeasurement?" rather than "does this entry self-report it was measured?".
- (iii) **P5 iter-65 row 23 / iter-65 row 76 / iter-80 row 95 / iter-81 row 96** — the iter-65 eta^2 recipe on the algorithm axis is the analog of the iter-82 row 97 window-sensitivity verb at the panel axis. Both audit "what does the same entry look like under an alternative measurement?"
- (iv) **P8 iter-80 row 94 / iter-84 row 99** — the fraud-detection cohort-calibration audit (P8 row 99) and the registry window-sensitivity backfill (P6 row 100) share the structural question "does the same model hold up under a different assessment window?". The P8 answer is "NO — every cohort breap the 0.10 ECE threshold"; the P6 answer is "the same sign holds on every panel, but the magnitude varies".

## Operational recommendation

1. **Adopt the iter-84 schema bump** (`window_sensitivity` + `robust_panel` on `measured_delta`). It is additive-optional; existing entries are unaffected.
2. **Backfill the same fields on all remaining `delta_*.json` entries** (`delta_adaptiveg`, `delta_cppo`, `delta_dapo`, `delta_drgrpo`, `delta_es`, `delta_gspo`, `delta_liteppo`, `delta_mcgrpo`, `delta_ngrpo`, `delta_ppo`, `delta_reinforce`, `delta_scafgrpo`) once their panel-by-panel re-measurement is run. **Defer**: only the 3 N2-same-stack remeasured variants are backfilled this iter.
3. **Audit downstream consumers**: any consumer of `measured[]` rows that filters on `panel="n2_same_stack_last10"` should now also check `robust_panel` and refilter on `robust_panel ∈ {last10, full40, last20}`.

## Reproducibility

- Script: `scripts/p5p8/p6_window_sensitivity_schema.py` (250 lines; stdlib + jsonschema; idempotent — re-running does not double-patch)
- Outputs:
  - `experiments/results/p5p8/p6_window_sensitivity_backfill.json` (machine-readable)
  - `experiments/results/p5p8/p6_window_sensitivity_backfill.log`
  - `registry/schema.json` (2 new optional fields on `measured_delta`)
  - `registry/entries/delta_aero.json`, `delta_gift.json`, `delta_areal.json` (backfilled)
- Input: `experiments/results/p5p8/p6_n2_window_deltas.tsv` (iter-82)
- Validation gate: 35/35 entries pass jsonschema (Draft-2020-12)

## Paper-facing text

Lifted into `paper/sections/p6_iter84_window_sensitivity.tex`.
