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
- `entries/*.json` — 12 seed stack records + 3 variant-delta records.
- `query.py` — stdlib-only reference CLI: `list`, `query`, `badge`,
  `stackdiff`.

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
