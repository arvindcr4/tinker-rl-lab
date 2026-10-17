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
- `query.py` — schema-validating reference CLI: `list`, `query`, `badge`,
  `stackdiff`.

## Schema invariants enforced (post iter-26 stress test)

The iter-26 stress test surfaced three latent gaps in the original schema.
They are now closed:

1. `variant_delta_record.id` carries the same `^[a-z0-9][a-z0-9_.-]*$`
   pattern as `stack_record.id` (previously: no pattern check on the
   variant-delta side).
2. Core leaves under the 7 MIN-REPORT items are required. Later additive
   leaves remain optional for historical compatibility, but the v3 badge uses
   every substantive schema leaf as a fixed denominator. Deleting a leaf
   cannot improve the score. Source/description metadata earns no credit.
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

All registry commands require `jsonschema` and validate records before indexing.
Duplicate JSON properties, nonfinite numbers, duplicate record IDs and repeated
delta declarations fail. Historical amendments are explicitly classified by
filename and digest in `sidecars.json`; arbitrary discriminator-less JSON fails.
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
python3 -m pytest tests/test_registry_stackdiff.py tests/test_registry_evidence.py tests/test_provenance.py
```

### stackdiff severity scheme

stackdiff compares a fixed field catalogue (`FIELDS` in `query.py`). A level is
assigned **only** from fields known on both sides that differ; a field unknown on
either side never raises the level. The verdict is the maximum severity:

| Level | Meaning | Fields |
|---|---|---|
| R5 | outcome-critical | `execution.mode`, `checkpoint.{id,variant,revision}`, `reward.id`, `prompt.{template_sha256,chat_template}`, `data.{dataset,revision,ordered_example_identity}`, `tokenizer.{id,revision}`, `eval.{metric,heldout}` |
| R4 | algorithm | `loss.*` (objective, ratio, clip, aggregation, advantage norm, token mask, length norm, dynamic sampling, reward shaping, inner epochs), `kl.*`, variant-delta rows declared on both sides |
| R3 | hyperparameter / budget | `decoding.*`, `batch.*`, `train.{steps,sampler_refresh}`, `data.{subset,selection}`, `optimizer.*`, `lora.*` |
| R2 | runtime | `trainer.name`, `runtime.{backend,precision,gpu,environment}`, `implementation.identity`, `reward.fn_sha256` (byte-level implementation hash; the semantic `reward.id` is R5) |
| R1 | seed / version / reporting | `trainer.version`, `seeds` |
| R0 | no known differences | — |

Missing information is reported separately:

- `coverage` (`stackdiff_known_both_v2`) — fraction of compared fields known on both sides;
- `unknown` — each field unknown on one or both sides, with its severity, and
  `max hidden severity` (the worst level that could still be hiding);
- `UNDERSPECIFIED` — `coverage < 0.8`, or any R5 field unknown, including
  checkpoint revision
- `undeclared` — a unilateral delta declaration, counted as unknown R4 evidence
  in the union-of-declarations coverage denominator
- `comparable` — independently false when required fields or declarations are
  unresolved, defaults/inferences are needed, or an implementation/version
  difference needs semantic review
- `evidence_coverage` — observed (`logged`), source-defined (`code`), assumed
  (`default`, `inferred`, legacy `registry`), and unresolved fractions, each
  divided by the full catalogue
- `evidence_complete` — no unknown comparison fields or one-sided delta declarations
- `matching_requirements.satisfied` — all required fields are observed and equal, with no unresolved declarations, conditional evidence, or implementation differences needing semantic review

`checkpoint.id` and `data.dataset` are compared after stripping trailing
parenthetical annotations, lowercasing, and aliasing `gsm8k` -> `openai/gsm8k:main`
(hand-entered values carry free text). Exit status is 0; pass `--exit-level` to
exit with the R-level when different, or exit 6 for R0 without established
comparability (including empty/all-unknown populations). R-levels classify
fields; they do not measure causal effect. Version and reward implementation
changes require semantic review irrespective of their descriptive R-level.
Integers and finite floats are compared exactly, with no display rounding.

Observed `effective_manifest` fields override an entry's `run_config`, which
in turn overrides hand-entered top-level fields and `min_report`. Explicit nulls
in the manifest override historical assumptions as unknown. The extractor
supports canonical field names and the shared runtime manifest's `revisions`,
`hardware.gpu_names`, `environment_hash`, `source`, `generation.train`, and
ordered `train_item_ids` / `test_item_ids`. A provider name alone is not an
environment identity, and one ordered split alone is not both split identities.
`decoding.*` includes top-k and repetition penalty; no absent setting is assumed
to equal a library default. Per-entry specification coverage is separately
versioned as `stack_spec_known_fields_v2`.

### Generated entries (`run_config`)

`generate_entries.py` rebuilds 12 stack entries from the run sources (pinned
git blobs of the run scripts at the commit live on the run date, plus the
results JSONs) and writes, per field, `{value, status, source}`:

| status | meaning | source format |
|---|---|---|
| `logged` | read from a results/log artifact | `<path>#<json key>` |
| `code` | constant in the pinned run script | `<path>@<commit>:L<n>` |
| `default` | explicitly recorded assumption; conditional comparison only | `<path>@<commit>:L<n>` |
| `inferred` | stated rule (e.g. `-Base` suffix) | rule text |
| `unknown` | not derivable (value null) | reason |

`provenance.source_artifacts` lists each source with its full sha256.
The generator fails if an anchor regex no longer matches or a numeric
assertion contradicts captured values. Numeric optimizer settings use AST
extraction; unresolved launcher/library defaults remain unknown. Semantic labels
anchored only by a regex location are marked inferred, so comparisons relying
on them are conditional until independently established.
Batch prompts are calculated from actual generation batch and group sizes, and Adam betas are extracted independently.
`--check` fails if entries on disk are stale; disposable projections verify idempotence. `regen_changes.json` records the
leaf-level before/after against the hand-written versions.

Generation v3 rebuilds the entire generated projection, including outcomes,
from source specifications. Only the explicit `provenance.manual_annotations`
namespace survives. Every comparison field is populated, with explicit unknowns
where evidence is absent; generated records never fall back to legacy MIN-REPORT.
Dry-run plans live in `intended_config`; execution identities stay unknown and
synthetic diagnostics are separate. Disabled ratio clipping uses
`clipping_applied: false` with null, inapplicable epsilons. Unsupported
decontamination assertions stay null.

The reward fingerprint `python-reward-assignment-and-module-v2` hashes complete
AST reward assignments, the parser, and the full source module conservatively.
H2H aggregate rewards and ZVF are recomputed from distinct completed per-seed
logs with input digests. Other extracted summary statistics identify their
upstream computation as unreproduced. Selected source hashes do not prove that
historical executions used those bytes; absent execution receipts remain unknown.

`reported_leaf_coverage_v3` is the mean of seven per-item substantive-leaf
fractions, exactly the unrounded badge divided by 100.
`item_presence_coverage_v3` is the fraction of seven items with any substantive
leaf reported. These are reporting completeness metrics, not scientific rigor.

The 12 historical generated entries are preserved unchanged during this code
review. They deliberately fail the v3 freshness check until a separately
reviewed migration is authorized; no new evidence is inferred for them.

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

## Fail-closed audit snapshots

`validate-strict` requires successful child execution, a unique temporary audit,
the exact current schema/entry/sidecar input digest, and the current validator
protocol/version and implementation digest. Missing or malformed output fails.
`--include-warn` uses the same decision for printed verdict and exit status.

`measured-coverage`, `coverage`, and `antiherding` only reuse cached output with a
`registry-report-cache-v1` receipt binding input manifest, output bytes and
producer digest. Legacy caches have no freshness assurance and fail explicitly.
Use `--refresh` to recompute in a temporary isolated copy; stored results,
registry entries, schemas and credentials are not rewritten. Output identifies
the precise snapshot digest. This establishes reporting freshness, not validity
of the original experimental claims or the historical statistical methods.

For authenticated publication, use the v2 file-level gate described in
[`provenance/SPEC.md`](provenance/SPEC.md). A reporting badge never substitutes
for trusted authentication, full evidence verification, or statistical analysis.

## Historical v2 coverage migration

The generator does not infer instruct from an absent Base suffix,
assume an absent launch command used CLI defaults, or report decontamination /
parser robustness as false without evidence. Unknown remains JSON null.
Generated outcomes are a deep copy of the current source-derived outcomes;
old CI/controller statistics are not carried over. Explicit annotations belong
in `provenance.manual_annotations`. Numeric outcome lineage records a source
pointer, input SHA-256 and extraction computation version under
`provenance.outcome_provenance`. This identifies extraction of a source-reported
statistic, not a new independent reproduction of its upstream computation.

Historical v2 coverage names distinguish the denominators and claims. New
generation uses the v3 substantive-leaf names above for MIN-REPORT coverage;
the versioned delta fractions below remain current:

- `min_report_item_presence_v1`: fraction of seven items with any reported leaf
- `min_report_leaf_mean_v2`: equal-weight mean of each item's reported-leaf
  fraction under the historical v2 badge definition
- `declared_delta_reporting_completeness_v2`: fraction of rows documented as
  implemented, surrogate, or absent; unknown is incomplete reporting
- `declared_delta_implementation_fraction_v1`: fraction actually marked implemented
- `declared_delta_surrogate_fraction_v1`: fraction marked surrogate

No declared rows yields null for the three delta fractions. Documented absence
counts toward reporting completeness and contributes zero implementation
presence. Coverage inputs and computation version are separately recorded in
`provenance.coverage_provenance`. Legacy fields remain schema-readable for
historical records and must not be confused with these versioned definitions.

The code-only review intentionally leaves the twelve historical generated
records and stored results unchanged. Consequently `generate_entries.py --check`
reports those records as stale until a separately reviewed regeneration occurs.
A schema/strict-validation pass establishes structural/source-reference validity,
not correctness or freshness of every historical scientific claim.
