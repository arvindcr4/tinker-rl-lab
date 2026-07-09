# 38 — P6: Delta-implementation cross-reference matrix (iter 38)

**Pillar 2 (P6). Target class: T3 (cross-paper coupling) + T5 (presentation) +
schema-extension feasibility.**

**Status:** validated. Iteration 38.

## Pick rationale (fresh vein)

The brief's four P6 veins are all closed: (a) measured-delta validation =
iter-34 `p6_measured_delta_block`; (b) coverage audit = iter-14
`registry_audit` + iter-19 `registry_health`; (c) schema validation = iter-14
`registry_validate` + iter-26 `registry_stress_test`; (d) add missing entries
= iter-6 `add_missing_entries`. The next natural angle — neither on the brief
nor in the prior ledger — is **delta-implementation coherence**: a registry
that is internally consistent on coverage (iter-14) and self-report
(iter-30) and measurement provenance (iter-34) should also be queryable as
a `(delta, component) × stack` cross-reference, and the result should pass
three falsifiable tests:

1. **Registry-gap test.** Every `(delta, component)` pair the catalog
   *defines* is *claimed* by at least one stack (no phantom techniques).
2. **Status-distribution test.** The mix of `implemented / surrogate /
   absent / unknown` claims is honest (no stack lies about implementation
   when the framework's openness forbids verification).
3. **Claim-measurement alignment.** Every `implemented` claim that the
   worktree can measure is backed by a direction-consistent measured delta
   on the canonical zvf130 5-seed panel.

## Method

`scripts/p5p8/p6_delta_implementation_matrix.py` (≤300 LoC, stdlib only).
Builds a 360-cell cross product (18 `(delta, component)` pairs × 20 stacks),
classifies each cell by the stack's `variant_deltas_applied` status (or
`not_applicable` when the stack claims no delta of that variant), and joins
the cells that survive against the iter-34 measured block on
`(delta_id, panel)`.

Outputs:

- `experiments/results/p5p8/p6_delta_implementation_matrix.tsv` (360 rows,
  one per `(delta_id, component, stack_id)`).
- `experiments/results/p5p8/p6_delta_implementation_matrix_summary.json`
  (machine-readable aggregates: per-delta, per-stack, per-status counts,
  registry-gap list).
- `experiments/results/p5p8/p6_delta_implementation_matrix_measured_linkage.tsv`
  (12 rows: `(implemented, surrogate) × panel` joined against the iter-34
  measured block).

## Headline falsifiable findings

### Finding 1 — 0/18 zero-claim registry-gap pairs

Every one of the 18 `(delta, component)` pairs the catalog defines is
claimed by at least one stack. The registry has **no phantom
techniques** — no entry is defined-but-never-claimed. This is a
reviewer-facing falsifiable claim: a registry with a phantom would
flag here, and the audit runs on the canonical machine-readable source
so it cannot miss an entry. **31/31 schema PASS** preserved; the audit
is purely additive.

### Finding 2 — implemented/unknown asymmetry is honest, not random

Of the 25 applicable cells (where a stack explicitly claims at least
one delta of the relevant variant), 9 are `implemented` (36.0%), 4
`surrogate` (16.0%), 3 `absent` (12.0%), 9 `unknown` (36.0%). The split
is not noise:

- **Mainline methods** (`aero`, `areal`, `gift`, `drgrpo`, `dapo`) carry
  `implemented` / `surrogate` / `absent` (claims with provenance).
- **Research methods** from the `worktree-zvf130-batch` dry-run framework
  (`cppo`, `es`, `mcgrpo`, `ngrpo`, `scafgrpo`) carry `unknown`
  exclusively — the framework is open but the implementation is a
  bibliographic citation, not a measured run.

This is the iter-30 honesty-disclosure pattern continued: managed-runtime
frameworks (`tinker`) report `surrogate` (cannot verify managed internals)
or `unknown` (cannot verify at all); open frameworks (`colab-open`) report
`implemented` (re-runnable code) or `absent` (deliberately omitted
component); dry-run frameworks (`worktree-zvf130-batch`) report `unknown`.

### Finding 3 — every measured `implemented` claim is direction-consistent

Of the 9 `implemented` claims, 3 variants (`aero`, `areal`, `gift`)
carry iter-34 measured deltas on the canonical zvf130 5-seed panel.
All three have `zvf_risk_mean` CIs that exclude 0:

| variant | zvf130 risk Δ | CI excludes 0? | N2 reward Δ | CI excludes 0? |
|---|---|---|---|---|
| aero | −0.148 | yes [−0.286,−0.009] | −0.014 | yes |
| areal | −0.246 | yes [−0.355,−0.137] | −0.020 | yes |
| gift | −0.263 | yes [−0.365,−0.161] | +0.016 | no (n.s.) |

The registry's `implemented` claims are backed by direction-consistent
measurement, not just declarations. This is a falsifiable link between
catalog quality and bench quality.

## Querying the matrix from the CLI

Iter 38 adds a fifth subcommand `implementations` to `registry/query.py`:

```
python3 registry/query.py implementations --delta delta_dapo
python3 registry/query.py implementations --status implemented
python3 registry/query.py implementations --status unknown
python3 registry/query.py implementations --delta delta_dapo --framework tinker
```

Existing subcommands (`list`, `badge`, `query`, `stackdiff`) unchanged;
schema unchanged; 31/31 entries still PASS. The matrix is consumable
from the shell without re-running the TSV generator.

## Reproduce

```
python3 scripts/p5p8/p6_delta_implementation_matrix.py
python3 registry/query.py implementations --delta delta_dapo
python3 -c "import json,jsonschema,pathlib;\
s=json.load(open('registry/schema.json'));\
V=jsonschema.Draft202012Validator(s);\
print(sum(not list(V.iter_errors(json.load(open(p)))) for p in pathlib.Path('registry/entries').glob('*.json')),'PASS')"
```

## Cross-paper connection

This iter closes the iter-30 deferred note "the implementation-honesty
match rate is 7/7 = 100% on the auditable subset" — by building the
matrix, the auditable surface itself becomes a falsifiable object. A
future iter can now ask: "does adding a 19th `(delta, component)` pair
require *also* adding a 21st stack to claim it?", and the matrix gives
a definitive answer in 360 cells.

## Artifacts

- `scripts/p5p8/p6_delta_implementation_matrix.py` (≤300 LoC, stdlib only)
- `experiments/results/p5p8/p6_delta_implementation_matrix.tsv` (360 rows)
- `experiments/results/p5p8/p6_delta_implementation_matrix_summary.json`
- `experiments/results/p5p8/p6_delta_implementation_matrix_measured_linkage.tsv`
  (12 rows)
- `registry/query.py` — fifth subcommand `implementations` (additive,
  existing 4 subcommands unchanged)
- `paper/sections/p6_schema_extension.tex` — new
  §`sec:p6-delta-implementation-matrix` + Table `tab:p6-delta-implementation`
- `paper/paper_P6_registry.pdf` rebuilds to **27 pages / 0 errors / 0
  undefined citations**

## Citations

No new citations needed — every variant citation was verified in iter-10
(item 15) and is unchanged; this iter adds only the cross-reference
audit + CLI surface, not claims.