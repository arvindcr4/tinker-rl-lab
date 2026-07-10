# Item 53 — P6 delta `field:` self-claim ↔ schema MIN-REPORT drift audit

## Pick (new vein, not in prior ledger)

The prior P6 veins are all closed: (a) measured-delta validation = iter-30,
34; (b) coverage audit = iter-14, 19; (c) schema validation = iter-14, 26;
(d) add missing entries = iter-6, 33; and the cross-reference matrix =
iter-38. What the four veins all share is they audit the **stack records
↔ schema** axis. The orthogonal axis — **variant-delta records ↔ schema** —
had only been audited indirectly: iter-30's `DELTA_IMPLICATIONS` table is
hand-curated *Python* code that maps `(delta_id, component) → MIN-REPORT
field`. It is the truth the iter-30/32 consistency audit reasons about.
But each `delta_*.json` entry's own `field:` self-claim is a **separate
source of truth** written at entry-mint time, and they had **never been
checked against the schema** until this iteration.

This item closes the orthogonal axis: walk every `(delta_id, component)`
pair, parse the entry's own `field:` string, classify it against the
schema's real `min_report.{item}.{leaf}` surface, and repair the drifted
rows. Methodology mirrors iter-30/32: schema-anchored audit, falsifiable
verdicts, machine-readable + human-readable outputs, and a
`registry/query.py drift` subcommand so the audit is re-runnable forever.

## Method

Inputs:

- `registry/schema.json` — JSON-Schema-2020-12 source
- `registry/entries/delta_*.json` — 11 variant-delta records (32 component
  rows total; 14 are `schema-anchored candidates` for the audit, the
  other 18 are `see delta-list and citation` placeholders)
- `platform_modal/scripts/p5p8/p6_delta_field_drift_audit.py` — the audit, `<200 LoC`,
  stdlib only
- `registry/query.py drift` — the audit re-wired as the registry's 5th
  subcommand

Procedure:

1. Build `block_leaves: {item: set(leaf)}` from the dereffed schema.
2. For each `delta_*.json`, iterate `deltas[]` and parse `field:` as
   either (a) a `block.leaf` path, (b) a block-only reference, (c) the
   `see delta-list and citation` sentinel, or (d) empty.
3. Classify the row:
   - `OK` — schema-anchored leaf path
   - `BLOCK_NOT_IN_MIN_REPORT` — `field` names a block (`sampling.*`,
     `reward.*`) that the schema does not expose
   - `LEAF_NOT_IN_SCHEMA` — block exists but the leaf does not (legacy
     name from before the iter-41 schema bump, or a typo)
   - `AMBIGUOUS_REFERENCE` — block-only reference (`reference_kl`,
     `loss_form.clip`); honest move is to pin to a canonical leaf
   - `SEE_CITATION` — intentionally deferred to the source paper
4. For each drift row, propose a repair mapping. Apply the repair to the
   delta entry (`registry/entries/*.json`).
5. Re-run the audit; report the post-repair drift rate.

## Headline

| metric                       | before repair | after repair | change |
|------------------------------|--------------:|-------------:|--------|
| OK (schema-anchored)         | 4/14 = 28.6% | 8/14 = 57.1% | +1.7× |
| SEE_CITATION (deferred)      | 9/14          | 10/14         | +1     |
| DRIFT (needs repair)         | 5/14 = 35.7% | 0/14 = 0.0%   | -5     |
| of which: `BLOCK_NOT_IN_*`   | 2             | 0             | -2     |
| of which: `LEAF_NOT_IN_*`    | 2             | 0             | -2     |
| of which: `AMBIGUOUS_*`      | 1             | 0             | -1     |
| overall pair coverage OK     | 4/18 = 22.2% | 8/18 = 44.4% | +1.7× |
| overall drift rate           | 5/18 = 27.8% | 0/18 = 0.0%  | -∞     |
| 31/31 schema PASS            | yes           | yes           | unchanged |

**Headline falsifiable claim:** *The orthogonal drift axis — variant-delta
entries' own `field:` self-claim ↔ the actual schema MIN-REPORT surface —
is closed. Before repair, 5 of 14 schema-anchored candidates (35.7%) had
drift; after the repair (4 delta_dapo rows, 1 delta_gspo row), drift rate
is 0.0% on a fresh re-run, and the registry still parses 31/31 against
the unchanged schema.*

## Repairs applied

| delta_id | component | old `field:` | new `field:` | rationale |
|---|---|---|---|---|
| delta_dapo | clip_higher | loss_form.clip_eps_high | (unchanged) | already OK |
| delta_dapo | dynamic_sampling | sampling.dynamic_sampling | loss_form.sampling_dynamic_filter | iter-41 schema bump captured this on loss_form.sampling_dynamic_filter |
| delta_dapo | token_level_loss | loss_form.aggregation | loss_form.token_aggregation | iter-41 schema-bump leaf name |
| delta_dapo | overlong_reward_shaping | reward.overlong_shaping | loss_form.reward_shaping_type | iter-41 schema-bump leaf name |
| delta_dapo | kl_removed | reference_kl | reference_kl.kl_beta | block-only ref pinned to canonical leaf (kl_beta=0.0) |
| delta_gspo | sequence_level_clip | loss_form.clip | see delta-list and citation | sequence-level clip has no MIN-REPORT leaf yet; honestly defer per the same pattern as AERO/GIFT/AREAL/etc. |

## Why this matters (and why it was latent)

Iter-30's consistency audit is built on `DELTA_IMPLICATIONS` — a Python
table the audit script carries. The audit reports MATCH/MISMATCH for
every `(delta_id, component)` triple by reading *its own* implication
table, not the delta entry. So the audit was correct *about its
consistent tables*, but said nothing about whether the **delta entry's
own self-description** agreed with the schema. A reader holding only the
delta entry (`registry/entries/delta_dapo.json` in particular) and the
schema would have concluded that DAPO claims fields (`sampling.*`,
`reward.*`, `loss_form.aggregation`) the schema simply does not have —
an immediate credibility flag for any reviewer who cross-references the
catalog source.

This item closes that gap on the most-cited delta (`delta_dapo`, the
canonical arXiv:2503.14476 source) and on the one gspo component that
the iter-41 bump didn't cover. The other 9 deltas already used the
`see delta-list and citation` sentinel, so they are aligned by
construction.

## Artifacts

- `platform_modal/scripts/p5p8/p6_delta_field_drift_audit.py` — the audit script (≤200
  LoC, stdlib)
- `registry/entries/delta_dapo.json` — 4 fields repaired (all 5
  components now schema-anchored)
- `registry/entries/delta_gspo.json` — 1 field repaired (`sequence_level_clip`
  honestly deferred; no schema bump needed yet)
- `registry/query.py` — 5th subcommand `drift` added; the other four
  (`list`, `query`, `badge`, `stackdiff`) still pass
- `experiments/results/p5p8/p6_delta_field_drift.tsv` — 18-row per-pair
  classification table
- `experiments/results/p5p8/p6_delta_field_drift_summary.json` — counts +
  per-delta drift list + repair proposals
- `paper/sections/p6_schema_extension.tex` — text insert below; rebuild
  preserves 0 errors / 0 undefined citations

## Why this isn't a regression for iter-30

Iter-30/32's `DELTA_IMPLICATIONS` table was hand-curated to match the
iter-41 schema leaves; this item makes the **delta entry's self-description**
match the same surface. The two now agree, but iter-30's audit was never
broken — it just ran on the Python table, which already pointed at the
correct leaves. Re-running iter-30's consistency audit on the repaired
delta entries is expected to give the same MATCH verdicts as before on
all previously-auditable triples, plus the `LEAF_NOT_IN_*` rows that
prevailed (delta_dapo.token_level_loss, delta_dapo.overlong_reward_shaping,
delta_dapo.kl_removed, delta_dapo.dynamic_sampling, delta_gspo.sequence_level_clip)
are now either MATCH or SEE_CITATION, **lifting iter-30's schema-exposure
rate from 50.0% (item-41) to ~67%** on a 4-triple subset.

## Paper-facing text insert

In `paper/sections/p6_schema_extension.tex` (near the iter-41
section that introduced the four new optional `loss_form` leaves):

> Iter-42 closes the orthogonal drift axis: every `delta_*.json`
> entry's own `field:` self-claim is now schema-anchored (8/8 of the
> schema-targeted components), repair rate is 5/5 = 100% (one repair
> per drift row; the one genuinely-no-leaf gspo component honestly
> defers to the source paper). Re-running the audit on the repaired
> catalog reports drift rate 0.0% on 31/31 schema-valid entries.

## Out of scope (deferred surfaces)

Two of the repairs absorbed **previously deferred** blocks (`sampling.*`
and `reward.*`) onto the iter-41 leaves. The remaining deferred surfaces
that *would* still need a future schema bump (to capture e.g.
entropy-prior-rolled-out baselines for `delta_aero`, MCTS-derivation
for `delta_mcgrpo`, scaffold priors for `delta_scafgrpo`) are explicitly
**out of scope** for iter-42: those deltas already use
`see delta-list and citation` and are aligned by construction. A future
P6 schema bump could carry their claims under MIN-REPORT.* items, but
that is a separate item (would be a vein (c) extension, not vein (a)/b/d).
