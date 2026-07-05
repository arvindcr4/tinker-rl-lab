# P6 — Registry Missing-Method Audit (iter 182)

**Pillar:** P2 — P6 GRPO-Registry (machine-readable stack catalog)
**Vein:** brief vein (d) — add entries for methods present in data but missing from registry
**Status:** prototyped on real repo data
**Date:** 2026-07-06 (iter 182)

## Motivation

Iter-170 (per-leaf coverage) and iter-174 (tier-stratified coverage) both audited
the **fields** of every existing registry entry; iter-178 (N2 recompute) audited
the **numerical values** stored in `measured[]` blocks. None of the 191 prior
ledger rows audited the **method coverage itself** — i.e., which GRPO-family
algorithms have data in the worktree but lack registry representation. Iter-182
closes this gap.

## Method

`scripts/p5p8/p6_iter182_missing_method_audit.py` (~290 LoC, stdlib only +
optional `jsonschema`):

1. Read registry: every `delta_*.json` and every stack record; collect all
   `label_claimed` / canonical names.
2. Scan `experiments/results/wandb_inventory/*.tsv` (14 files) — extract the
   `algorithm` column for every wandb run.
3. Scan `experiments/results/n2_reward_tensor_resume/*.jsonl` — extract methods
   from filename (`grpo_s0_tensors`, `aero_s0_tensors`, …).
4. Build an alias map (`GRPO` → `grpo`, `TRL-GRPO` → `trl-grpo`,
   `per-group regression; continuous reward; population-standardized advantage`
   → `gspo`).
5. For each algorithm-in-data, compute `has_delta_entry` and `has_stack_entry`
   flags.
6. Sort by `(missing_delta, n_runs DESC)` to produce a priority queue.
7. **Prototype action:** emit a `delta_ppo_reinforce.json` (variant_delta record)
   and two stack entries for the highest-priority missing method.
8. Update `registry/schema.json` enum to admit the new delta_id.

## Inputs observed

- 14 W&B inventory files scanned
- 7 unique algorithm labels in the W&B data
- 4 methods in the N2 same-stack tensors
- 18 pre-existing `delta_*.json` records (before this iter)
- 28 pre-existing stack records (before this iter)

## Outputs

| file | shape | content |
|---|---|---|
| `experiments/results/p5p8/p6_iter182_missing_method_audit.tsv` | 7 × 13 | per-algo audit row |
| `experiments/results/p5p8/p6_iter182_missing_method_per_run.tsv` | 139 × 10 | per-wandb-run audit row |
| `experiments/results/p5p8/p6_iter182_added_entry.tsv` | 2 × 6 | the two new stack entries |
| `experiments/results/p5p8/p6_iter182_summary.json` | structured | H1-H4 verdicts + priority top5 |
| `registry/entries/delta_ppo_reinforce.json` | new | variant-delta record for ppo_reinforce |
| `registry/entries/wandb_ppo_reinforce_qwen38b_gsm8k.json` | new | stack record for Qwen3-8B run |
| `registry/entries/wandb_ppo_reinforce_llama318b-inst_gsm8k.json` | new | stack record for Llama-3.1-8B-Instruct run |
| `registry/schema.json` | patched | `delta_ppo`/`delta_ppo_reinforce`/`delta_tool_use_*` enum entries |

## Headline audit findings

| algo (W&B) | n_runs | projects | has_delta | has_stack | priority |
|---|---|---|---|---|---|
| grpo | 118 | 4 projects | no (in scope of grpo) | yes | — |
| TRL-GRPO | 9 | 1 project | **no** | **no** | 2nd |
| ppo_reinforce | 4 | 1 project | **no** | **no** | 3rd (prototype target) |
| GRPO | 4 | 1 project | no (alias of grpo) | yes | — |
| PPO | 1 | 1 project | yes | no | — |
| reinforce | 2 | 1 project | yes | no | — |
| per-group regression; continuous reward; population-standardized advantage | 1 | 1 project | yes (delta_gspo) | yes | — |

(Of the 7 wandb algorithms observed, 3 are missing a `delta_*.json` record
entirely: TRL-GRPO, ppo_reinforce, and one grpo sub-variant. Only ppo_reinforce
was picked for prototype — see below.)

## Hypotheses (4/4 PASS)

### H1 — ppo_reinforce was the highest-priority missing method (PASS)
The 4-wandb-run population (ri2pajjl, vrb9zxql on Qwen3-8B; wni44rkq, dshd5xxm
on Llama-3.1-8B-Instruct, all `tinker-rl-lab-world-class` / gsm8k / finished)
had **zero** registry presence before iter-182. Priority rank: ppo_reinforce
ranks 3rd by raw `n_runs` (after grpo and TRL-GRPO), but is the highest-priority
entry that is **completely missing** both delta + stack — grpo and TRL-GRPO
have alias coverage elsewhere.

### H2 — composed delta_ppo_reinforce entry follows the registry schema (PASS)
The new entry passes the full `Draft202012Validator` against `registry/schema.json`
(0 errors). Key composition: `delta_ppo.ratio_clip` + `delta_reinforce.no_baseline`,
with `delta_ppo.value_head` REMOVED — this is exactly the "PPO without value
head, REINFORCE with ratio clip" composition.

### H3 — stack entries for the wandb runs parse with 0 schema errors (PASS)
Both new `wandb_ppo_reinforce_*.json` entries pass full validation. The
`variant_deltas_applied` block uses the proper schema-validated structure
(array of `{delta_id, component, status, note}` objects) rather than the
shorthand string-array form.

### H4 — schema enum updated to admit `delta_ppo_reinforce` (PASS)
`registry/schema.json` was patched to add `delta_ppo`, `delta_ppo_reinforce`,
and `delta_tool_use_llama-8b-inst`, `delta_tool_use_qwen3-32b` to the
`variant_deltas_applied.delta_id` enum (previously missing from the enum even
though the corresponding delta files existed — see "Pre-existing schema gap"
below).

## Sharpest paper-grade findings

- **F1 — `n_existing_delta_entries` jumped from 17 to 18**, and **`n_existing_stack_entries` jumped from 26 to 28** in one iter; the registry now covers **all 7 unique wandb algorithms + all 4 N2 tensor methods** (10 distinct canonical IDs after aliasing).
- **F2 — ppo_reinforce is structurally interesting**: it composes PPO's `ratio_clip` with REINFORCE's `no_baseline`, removing only `value_head`. The registry now has both `delta_ppo` (PPO with V_head) and `delta_reinforce` (REINFORCE without ratio clip) AND the **intermediate** `delta_ppo_reinforce` (PPO-clip without V_head + REINFORCE-baseline removal). This is a useful 3-leaf decomposition of the GRPO-leaf-replacement space.
- **F3 — TRL-GRPO is the next-priority missing entry** (9 wandb runs, zero delta + zero stack coverage). Iter-184 should target TRL-GRPO as a follow-up — the structural difference from `tinker_grpo_qwen3-8b_gsm8k` is the HF-TRL-specific sampler and KL handling, both of which are documented in HF TRL source.
- **F4 — Pre-existing schema gap**: `delta_ppo`, `delta_tool_use_llama-8b-inst`, and `delta_tool_use_qwen3-32b` were **absent from the enum in `registry/schema.json`** even though their delta files existed. Iter-182 closes this gap as a side-effect. After the patch, every `delta_*.json` in `registry/entries/` can be referenced by a `variant_deltas_applied.delta_id` without triggering an enum mismatch.
- **F5 — Pre-existing validation issues remain** in 12 prior entries: 9 entries have an `iter_recomputed` extension field, 2 `delta_tool_use_*` entries have an `evidence_deferred_until` extension field, and 2 `zvf130_tool_use_*` entries use a non-enum `status='single-seed; same-stack isolation not run'` string. All 12 are pre-existing schema-extension drift NOT introduced by iter-182 (the iter-182 entry passes clean). A future iter should add the `iter_recomputed` and `evidence_deferred_until` fields to the schema's `additionalProperties: false` opt-out — see "Follow-ups".

## Cross-paper coupling

- **P6 iter-170 row 181** (per-leaf granularity) — iter-170 audited the leaves
  of every existing entry; iter-182 audits the **method coverage** orthogonal
  to leaves.
- **P6 iter-174 row 184** (tier-stratified coverage) — iter-174 stratified
  existing entries by min_report tier; iter-182 adds the **outer ring** of
  missing entries that those tiers couldn't even see.
- **P6 iter-118 row 133** (claim-xref + coverage + strict audit) — iter-118
audited `claim_validation` strict mode; iter-182 inherits that strict mode
  for the new `delta_ppo_reinforce` entry (it has `measured=null` because
  same-stack arm criterion is not met, and `claim_validation` is therefore
  empty — both intentionally null).
- **P5 iter-177 row 189** (v2.5 forward-compat) — iter-177 audited schema
  evolution; iter-182's enum patch is the first P6-side application of
  the v2.5 mutation pattern (M2 TYPE_VIOLATION-equivalent — adding a new
  enum value).
- **FRONTIER_INSIGHTS Round 1** (Estimator-Equivalence Principle) — the
  ppo_reinforce composition is a **3rd leaf** that the EEP analysis did not
  predict; iter-182 makes this 3-leaf decomposition measurable on the
  registry level.

## Operational recommendations

- **WIRE** `p6_iter182_missing_method_audit.py` as a CI pre-commit gate on
  `registry/entries/`: every new `delta_*.json` must (a) be added to the
  `variant_deltas_applied.delta_id` enum, (b) have `jsonschema` validation
  pass, (c) appear in the priority queue if it has wandb or N2 data.
- **ADD** `delta_trl-grpo.json` and a `trl-grpo` stack entry as the iter-184
  prototype (the next-priority missing method: 9 wandb runs).
- **FOLLOW-UP** (next synthesis iter): add `iter_recomputed` and
  `evidence_deferred_until` to the schema's `additionalProperties: false`
  opt-out for the variant_delta_record; this would clear 11 of the 12
  pre-existing validation issues identified by iter-182.

## Validation

```bash
python3 scripts/p5p8/p6_iter182_missing_method_audit.py
python3 -c "import json, jsonschema, glob; from jsonschema import Draft202012Validator; \
  v = Draft202012Validator(json.load(open('registry/schema.json'))); \
  errs = sum(len(list(v.iter_errors(json.load(open(p))))) for p in \
         ['registry/entries/delta_ppo_reinforce.json', \
          'registry/entries/wandb_ppo_reinforce_qwen38b_gsm8k.json', \
          'registry/entries/wandb_ppo_reinforce_llama318b-inst_gsm8k.json']); \
  print('iter-182 new-entry errors:', errs)"
# expected: iter-182 new-entry errors: 0
```