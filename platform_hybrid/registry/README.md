# GRPO-Registry

A machine-readable catalog of group-relative RL (GRPO-family) training stacks
and variant deltas. This directory is the deliverable of the resource paper
`paper/paper_P6_registry.tex`.

## Layout

- `schema.json` — JSON Schema (draft 2020-12) for the two record types:
  - `stack`: one deployed GRPO-family training stack, described by the
    seven-item MIN-REPORT-RL block plus provenance and per-delta
    implementation status;
  - `variant_delta`: the set of changes a named variant (DAPO, Dr. GRPO,
    GSPO, ...) makes to base GRPO, verified against the source paper.
- `entries/*.json` — 31 records (20 stack + 11 variant-delta). Iter 6 added
  the 8 GRPO-family methods present in the worktree
  (aero/gift/areal/ngrpo/cppo/mcgrpo/es/scafgrpo); iter 15 verified the 8 new
  citations.
- `query.py` — stdlib-only reference CLI: `list`, `query`, `badge`,
  `stackdiff`.

## Schema invariants enforced (post iter-26 stress test)

The iter-26 stress test surfaced three latent gaps in the original schema.
They are now closed:

1. `variant_delta_record.id` carries the same `^[a-z0-9][a-z0-9_.-]*$`
   pattern as `stack_record.id` (previously: no pattern check on the
   variant-delta side).
2. Every leaf under each of the 7 MIN-REPORT items is in the item's
   `required` list, so a "leaf dropped" entry fails schema validation
   instead of silently passing (previously: leaves were optional; the
   iter-14 audit's per-leaf coverage was a value-null check, not a
   key-presence check).
3. `variant_deltas_applied[].delta_id` is constrained by an `enum` that
   enumerates every `registry/entries/delta_*.json` stem. When a new delta
   is added, run
   `python3 scripts/p5p8/regenerate_schema_delta_enum.py` to refresh
   the enum and re-validate all 31 entries.

The iter-26 stress test (`scripts/p5p8/registry_stress_test.py`) applies
3,230 adversarial perturbations across 13 categories × 31 entries × 10
mutations and reports a 100.0% recovery rate (95% paired bootstrap CI
[1.000, 1.000]) on the bumped schema.

## Field convention

`null` means **unreported**. Explicit `false` / `0.0` / `"none"` means
**reported-as-absent**. The MIN-REPORT badge scores reporting coverage,
not configuration virtue.

## Quick start

```bash
python3 registry/query.py list
python3 registry/query.py badge
python3 registry/query.py query --item reference_kl
python3 registry/query.py stackdiff colab-open_dapo_e3 tinker_dapo_qwen3.5-4b_gsm8k
python3 registry/query.py claim-validation            # iter-46: (delta, metric, panel) audit verdicts
```

## Validation

```bash
python3 -c "import json, glob, jsonschema; s=json.load(open('registry/schema.json')); \
  [jsonschema.validate(json.load(open(p)), s) for p in glob.glob('registry/entries/*.json')]"
```

## Seed-entry provenance

| Entries | Source |
|---|---|
| `trl/verl/openrlhf/tinker_grpo_qwen3-8b_gsm8k` | `experiments/framework_config_dumps/*.yaml` (44-field fairness-audit dumps) |
| `colab-open_*_e3` | E3 open-trainer audit (W&B `zvf-colab-experiments`) — toy scale, directional |
| `tinker_*_qwen3.5-4b_gsm8k` | 12-cell Tinker head-to-head (W&B `zvf-audit`), 3 seeds/arm |
