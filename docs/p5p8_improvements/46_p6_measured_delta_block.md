# 46 — P6: measured-delta block grounds every GRPO-variant record in real data

**Pillar 2 (P6). Target class: T1 (statistical rigor) + T3 (cross-paper coupling) + schema (vein c).**
**Status: validated.** Iteration 34.

## Gap

The `variant_delta_record` cataloged what a GRPO variant *claims* to change
(`deltas[]` = component/field/change) plus a verified `citation`, but carried
**no measured outcome**. The registry has known this hole since the iter-28 and
iter-33 synthesis notes ("variant_delta has no outcomes block"). Measured deltas
existed only in loose TSVs (iter-18 `registry_measured_claimed.tsv`, iter-22
`registry_variant_coupling.tsv`); they were never first-class, schema-validated,
provenance-backed fields on the records themselves.

## What was built

1. **Schema extension** (`registry/schema.json`, additive + optional, backward
   compatible): a new `measured` array on `variant_delta_record`, each element a
   `$defs/measured_delta` = `{metric, panel, base, delta, ci_low, ci_high, n,
   significant, ci_method, source, note}`. `panel` and `source` are **required**
   so no delta can be recorded without saying *where* it was measured; `ci_method`
   reuses the existing iter-28 `$defs/ci_method`. 31/31 entries still PASS
   (additive-optional pattern, same discipline as the iter-28 `ci_method` bump).

2. **`scripts/p5p8/p6_measured_delta_block.py`** (≤300 LoC, stdlib + jsonschema)
   computes `variant − grpo` on two provenanced panels and writes the blocks:
   - **`n2_same_stack_last10`** — N2 four-method same-stack reward tensors
     (`n2_metrics.tsv`), metrics `{zvf, reward_mean}`, last 10 of 40 steps,
     **paired-by-step percentile bootstrap** (n_boot=2000, seed=20260704).
     Populated for aero/gift/areal.
   - **`zvf130_5seed`** — 5-seed `zvf_iter130_method_risk.tsv`, metric
     `zvf_risk_mean` with a Welch normal-approx CI (both arms carry per-seed SD),
     plus `mean_zvf` as an honestly-flagged point estimate (per-seed SD for
     `mag_mean` is not stored → CI marked unmeasurable, `significant:false`).
     Populated for all 8 measured variants.

## Measured results (`experiments/results/p5p8/p6_measured_delta_block.tsv`)

- **8 delta records populated, 22 measured rows**; full registry re-validates
  **31/31 PASS** against the bumped schema.
- **zvf130 5-seed, `zvf_risk_mean`: 8/8 variants significantly BELOW grpo** (every
  CI excludes 0; deltas −0.131 to −0.352). The iter-18 directional "ZVF below
  grpo" reading is now a CI-backed, schema-validated registry field for every
  method in the risk panel.
- **N2 same-stack reward: aero −0.0141 [−0.0234,−0.0055]✓, areal −0.0195
  [−0.0320,−0.0078]✓, gift +0.0164 [−0.0070,+0.0398] n.s.** The two significant
  reward deltas are ≤2pp — tiny, corroborating the Pillar-1 "algorithm barely
  moves the outcome once the stack is fixed" reading while being honest that the
  CI does exclude 0 at this sample.

## Headline falsifiable finding — the ZVF delta is *panel-conditional*

The **sign of a variant's ZVF effect flips with the panel**, which is precisely
why `panel` is a required field on the measured block:

| variant | N2 same-stack `zvf` Δ | zvf130 `mean_zvf` Δ |
| --- | --- | --- |
| gift | **+0.125** (CI [+0.081,+0.175], excludes 0) | **−0.367** |
| aero | −0.025 (n.s.) | −0.261 |
| areal | −0.056 (n.s.) | −0.360 |

On the N2 same-stack run GIFT *raises* per-step zero-variance fraction
(+0.125, significant) — its γ-style likelihood prior shifts the advantage — yet
on the canonical zvf130 distribution every variant including GIFT sits *below*
grpo. A single "ZVF delta" number for a variant is therefore ill-posed; the
registry now records the panel so the two facts cannot be conflated. This is a
concrete instance of the P5/P6 thesis (report the stack/panel, not the label) at
the level of the variant catalog itself.

## Reproduce

```
python3 scripts/p5p8/p6_measured_delta_block.py
python3 -c "import json,jsonschema,pathlib;s=json.load(open('registry/schema.json'));\
V=jsonschema.Draft202012Validator(s);\
print(sum(not list(V.iter_errors(json.load(open(p)))) for p in pathlib.Path('registry/entries').glob('*.json')),'PASS')"
```

## Citations
No new citations. The variant citations (le2025rlzvp / nan2025ngrpo / …) were
verified in prior iterations and are unchanged; this item adds only measured
provenance, not claims.
