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
- `entries/*.json` — 46 queryable records (28 stack + 18 variant-delta), plus
  two provenance amendment sidecars. Iter 6 added
  the 8 GRPO-family methods present in the worktree
  (aero/gift/areal/ngrpo/cppo/mcgrpo/es/scafgrpo); iter 15 verified the 8 new
  citations.
- `generate_entries.py` — regenerates stack entries from run sources (see Usage).
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
  the enum and re-validate all queryable entries.

The iter-26 stress test (`scripts/p5p8/registry_stress_test.py`) applied
3,230 adversarial perturbations to the then-current 31-entry snapshot
(13 categories × 31 entries × 10 mutations) and reported a 100.0% recovery
rate (95% paired bootstrap CI [1.000, 1.000]) on the bumped schema. This
historical result does not imply that later entries were in that snapshot.

## Field convention

`null` means **unreported**. Explicit `false` / `0.0` / `"none"` means
**reported-as-absent**. The MIN-REPORT badge scores reporting coverage,
not configuration virtue.

## Usage

Stdlib only (`validate` / `validate-strict` additionally need `jsonschema`).
Paths resolve from the script location, so run from anywhere — e.g. the repo root:

```bash
python3 platform_hybrid/registry/query.py list
python3 platform_hybrid/registry/query.py badge                     # MIN-REPORT badge + spec_coverage
python3 platform_hybrid/registry/query.py stackdiff tinker_grpo_qwen3-8b_gsm8k trl_grpo_qwen3-8b_gsm8k
python3 platform_hybrid/registry/query.py stackdiff colab-open_dapo_e3 tinker_dapo_qwen3.5-4b_gsm8k --sources -v
python3 platform_hybrid/registry/query.py stackdiff --all [--generated]   # R-level distribution
python3 platform_hybrid/registry/query.py validate                  # schema
python3 platform_hybrid/registry/query.py validate-strict           # schema + orphan/source gate
python3 platform_hybrid/registry/generate_entries.py [--check|--dry-run]
python3 -m pytest tests/test_registry_stackdiff.py tests/test_provenance.py
```

### stackdiff severity scheme

stackdiff compares a fixed field catalogue (`FIELDS` in `query.py`). A level is
assigned **only** from fields known on both sides that differ; a field unknown on
either side never raises the level. The verdict is the maximum severity:

| Level | Meaning | Fields |
|---|---|---|
| R5 | outcome-critical | `execution.mode`, `checkpoint.{id,variant,revision}`, `reward.id`, `prompt.{template_sha256,chat_template}`, `data.dataset`, `eval.{metric,heldout}` |
| R4 | algorithm | `loss.*` (objective, ratio, clip, aggregation, advantage norm, token mask, length norm, dynamic sampling, reward shaping, inner epochs), `kl.*`, variant-delta rows declared on both sides |
| R3 | hyperparameter / budget | `decoding.*`, `batch.*`, `train.{steps,sampler_refresh}`, `data.{subset,selection}`, `optimizer.*`, `lora.*` |
| R2 | runtime | `trainer.name`, `runtime.{backend,precision}`, `reward.fn_sha256` (byte-level implementation hash; the semantic `reward.id` is R5) |
| R1 | nuisance | `trainer.version`, `seeds` |
| R0 | no known differences | — |

Missing information is reported separately:

- `coverage` — fraction of compared fields known on both sides;
- `unknown` — each field unknown on one or both sides, with its severity, and
  `max hidden severity` (the worst level that could still be hiding);
- `UNDERSPECIFIED` — `coverage < 0.8`, or any R5 field unknown (except
  `checkpoint.revision`, which no run script in this repo pins);
- `undeclared` — a variant-delta row present on one side only. It is neither a
  difference nor counted in coverage (a baseline lists no deltas).

`checkpoint.id` and `data.dataset` are compared after stripping trailing
parenthetical annotations, lowercasing, and aliasing `gsm8k` -> `openai/gsm8k:main`
(hand-entered values carry free text). Exit status is 0; pass `--exit-level` to
exit with the R-level (0..5) for scripted gating.

Field values come from an entry's `run_config` when present, otherwise from its
hand-entered top-level fields and `min_report` (JSON null = unknown).

### Generated entries (`run_config`)

`generate_entries.py` rebuilds 12 stack entries from the run sources (pinned
git blobs of the run scripts at the commit live on the run date, plus the
results JSONs) and writes, per field, `{value, status, source}`:

| status | meaning | source format |
|---|---|---|
| `logged` | read from a results/log artifact | `<path>#<json key>` |
| `code` | constant in the pinned run script | `<path>@<commit>:L<n>` |
| `default` | argparse/library default; launcher not in repo | `<path>@<commit>:L<n>` |
| `inferred` | stated rule (e.g. `-Base` suffix) | rule text |
| `unknown` | not derivable (value null) | reason |

`provenance.source_artifacts` lists each source with its full sha256.
The generator fails if an anchor regex no longer matches. `--check` (also run by
the tests) fails if entries on disk are stale. `regen_changes.json` records the
leaf-level before/after against the hand-written versions.

| Entries | Run sources |
|---|---|
| `tinker_grpo_qwen3-8b_gsm8k` | `experiments/tinker-runs/campaign_v2.py@bf5b1db9f` + `framework_comparison.json` (Tinker row) |
| `trl_grpo_qwen3-8b_gsm8k` | `experiments/modal/modal_grpo_trl.py@bf5b1db9f` + `framework_comparison.json` (TRL row) |
| `verl_/openrlhf_grpo_qwen3-8b_gsm8k` | `framework_comparison.json` — `mode: dryrun`, synthetic trace, no training ran |
| `tinker_{grpo,drgrpo,dapo,gspo}_qwen3.5-4b_gsm8k` | `live_zvf_probe.py` + `cell_runner.py` @`b0ac85bfb` + `results/week_h2h/*.json` |
| `colab-open_*_e3` | `zvf-program/colab-experiments/e3_open_audit.py@71e6f09ee` + `results/e3_open_audit.json` |

The other 16 stack entries (`zvf130_*`, `wandb_*`, N2 `tinker_{aero,areal,gift}_*`)
are still hand-entered and carry no `run_config`.

## Seed-entry provenance

| Entries | Source |
|---|---|
| `trl/verl/openrlhf/tinker_grpo_qwen3-8b_gsm8k` | generated (see above). The `experiments/framework_config_dumps/*.yaml` previously cited describe neither the Tinker nor the TRL run (e.g. LoRA rank 16 / temp 0.7 / 384 tokens vs rank 32 / 512 tokens actually used) and carry a placeholder `model_sha256` |
| `colab-open_*_e3` | generated (see above) — toy scale, directional |
| `tinker_{grpo,drgrpo,dapo,gspo}_qwen3.5-4b_gsm8k` | generated (see above), 3 seeds/arm |
