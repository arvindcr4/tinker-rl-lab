# P6 item 10: self-reported CI provenance (`outcomes.ci_method`) — iter 28 JOB B/SYNTH

## Why now (re-ranking outcome)
Re-ranking the ledger by (impact × evidence × readiness) this synthesis
iteration, **item 10 is the only genuinely open `proposed` row** — every other
row is `validated` or `rejected` with a recorded reason. Its readiness was
blocked by one fear: "a schema bump forces re-validating all 31 entries." This
iter dissolves that blocker by making the addition *backward-compatible*.

## What item 10 asked for
Add an `outcomes.ci_method` field so a registry entry can self-report the
provenance of the confidence interval on its telemetry (paired-bootstrap
`n_boot`, `seed`, `ci_level`, `source`) — closing the gap where an entry gives
a point outcome with no way to know how (or whether) a CI was computed.

## Method (backward-compatible schema addition)
1. Added a reusable `$defs/ci_method` object to `registry/schema.json`
   (`method`, `n_boot`, `seed`, `ci_level`, `source`; `additionalProperties:
   false`, all leaves nullable) and referenced it as an **optional** property
   inside the already-permissive `outcomes` object. No `required` clause →
   every existing entry validates unchanged.
2. Populated `ci_method` truthfully on the **7** tinker entries whose outcomes
   were derived from the N2 same-stack four-method run (aero, areal, dapo,
   drgrpo, grpo, gspo, gift). Their CI methodology is the paired bootstrap in
   `scripts/p5p8/registry_validate.py::bootstrap_paired_diff` — verified in
   source: `n_boot=2000`, `seed=0`, percentile 2.5/97.5 (95%). The remaining 24
   entries leave it `null` (unreported).
3. Re-validated all 31 entries with `jsonschema.Draft202012Validator` and
   re-ran the canonical `registry_validate.py`.

Everything is in one idempotent, stdlib+jsonschema script
`scripts/p5p8/add_ci_method.py` (safe to re-run; asserts no regression).

## Measured result
- **31/31 entries schema PASS** after the edit (no regression) — confirmed by
  both the new validator and the canonical `registry_validate.py`.
- **7/31 entries now self-report `ci_method`**; the schema itself is still a
  valid Draft 2020-12 meta-schema (`check_schema` passes); `registry/query.py`
  CLI unaffected.
- Outputs: `experiments/results/p5p8/registry_ci_method_coverage.tsv` (31 rows),
  `registry_ci_method_summary.json`.

## Paper-facing integration
`paper/sections/p6_schema.tex` gains a "Self-reported CI provenance" paragraph
stating the optional field, the 31/31 backward-compatibility guarantee, and the
7/31 coverage. `paper_P6_registry.pdf` rebuilds to **23 pages / 0 errors / 0
undefined citations**.

## Status transition: proposed (iter 2) → **validated** (iter 28)
The thread stops cycling: the schema now carries CI-provenance self-report, the
fear of a breaking bump is resolved (additive-optional pattern), and a future
contributor can record `ci_method` on any new entry.

## Verified citations
None added (schema/config change only; all provenance points to in-repo
artifacts verified in source).
