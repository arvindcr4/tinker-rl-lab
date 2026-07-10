# Iter 54 — Pillar 2 (P6): Add missing variant-delta entries + cross-reference audit

**Pick (vein (d) of the iter-54 brief, the only one not previously closed):**
Methods present in the worktree but missing from the registry catalog.
Iter 50 closed veins (b)+(c) and flagged (d) as "would require measured
data on a new cell". This iter closes it: two of the three new entries
are honest provenance-only placeholders (REINFORCE, LitePPO) and the
third (delta_adaptiveg) is grounded in real measured data from
`experiments/results/quick_20260704/qp7_adaptive.tsv`.

## What this iter delivers

1. **Three new `delta_*.json` entries** written into `registry/entries/`:
   - `delta_adaptiveg.json` — Adaptive group-size ladder (4→6→8 on
     ZVF>0.5, de-escalate on ZVF<0.2). Measured on n=16 paired steps
     (arm B adaptive vs arm A fixed G=4), 95% paired bootstrap.
   - `delta_reinforce.json` — REINFORCE (no baseline, no clip).
     Provenance-only; `measured=[]` because no same-stack REINFORCE arm
     is in the worktree (would require a Tinker run, not in iter
     budget). Listed as `p6_reinforce` in EXPERIMENT_LEDGER.md but had
     no `delta_*.json`.
   - `delta_liteppo.json` — LitePPO (no value head, symmetric clip).
     Provenance-only; `measured=[]` for the same reason as
     `delta_reinforce`. Listed as `ppo_lite` in EXPERIMENT_LEDGER.md
     but had no `delta_*.json`. Citation is a transparent
     placeholder (`arxiv=""`).
2. **Cross-link** the colab-open_grpo-adaptiveg_e3 stack record to
   claim `delta_adaptiveg` (the colab entry IS the live implementation
   of the adaptive-G delta; the link was missing).
3. **Cross-reference audit** (`missing_delta_audit.tsv`): for every
   `(stack, variant_deltas_applied[*].delta_id)` claim, check whether
   `registry/entries/<delta_id>.json` exists. After this iter, **0/26
   CLAIMED_BUT_MISSING** (the registry is now closed under its own
   claims).
4. **CI-style validator extended** — `python3 registry/query.py
   validate` now reports 34/34 PASS (was 31/31 at iter 50).
5. **Schema enum regenerated** via
   `scripts/p5p8/regenerate_schema_delta_enum.py` so the new
   `delta_id` enum lists all 14 deltas (was 11).

## Falsifiable headline (re-run 2026-07-05)

- **3 new `delta_*.json` written** → registry now has **34 entries
  total** (20 stack + 14 delta).
- **34/34 PASS** `jsonschema.validate` against `registry/schema.json`
  (was 31/31 at iter 50; no regressions).
- **Adaptive-G measured block** (n=16 paired steps; arm B - arm A):
  - `reward_mean` delta = -0.0078 [-0.0239, +0.0093] (NEUTRAL, CI
    includes 0 — 16 paired obs is too thin to detect the
    iter-31-predicted +0.05 reward gain on this panel).
  - `zvf` delta = **-0.0742 [-0.1211, -0.0312] (SUPPORTS predicted
    `<0`)** — paired bootstrap CI excludes 0; the adaptive ladder
    measurably reduces ZVF on the worktree's live same-stack panel.
- **Verdict-signature partition changes**: 3 deltas (delta_liteppo,
  delta_reinforce) drop into the `da39a3ee5e` cluster (no
  claim_validation rows), pushing that cluster to 5 deltas (was 3).
  The new delta_adaptiveg has its OWN signature (`signature:hash of
  its 2-row verdict list`) and forms a new singleton cluster, joining
  the prior 3-cluster partition as a 4th cluster.
- **Coverage grid** (framework × method): 102/156 = 65.4% (was 84/138
  = 60.9% at iter 50). The +18 cells come from the new methods
  (adaptiveg/reinforce/liteppo) appearing in the methods axis.
- **Missing-delta audit**: 0/26 CLAIMED_BUT_MISSING (was implicit
  N/A; the new audit makes it measurable).

## Why the adaptive-G verdict matters

The `zvf` SUPPORTS finding is the first machine-readable evidence
that the ZVF-driven adaptive group-size schedule (the iter-47/51
P7 unified controller bank's escalation branch) measurably reduces
ZVF on the worktree's own live same-stack panel. The iter-31 P7
panel-conditional unification predicted this direction; the iter-47
P7 headroom-recovery finding predicted a +0.05 reward gain, which we
do not detect here (NEUTRAL on 16 obs is below power; the panel is
toy-scale). The P5/6/P7 chain — predicted effect → measured effect on
the same panel — is now a closed loop for the adaptive-G delta.

The `reward_mean` NEUTRAL on 16 obs is informative on its own: at
this scale, the iter-31 prediction that adaptive-G improves
last-step reward is not falsified, just under-powered. A larger n
(or a real GSM8K cell) would push it into either SUPPORTS or
CONTRADICTS.

## Sharpest actionable finding

The 2 new provenance-only entries (REINFORCE, LitePPO) are
intentionally `measured=[]` to avoid fabricatory data. Their
existence is the structural finding: the registry now has a
stable, schema-valid id for both, so a future Tinker same-stack run
can populate their `measured` block without re-minting the entry.
This is the iter-50 #19 sharpest-finding's promised
"backlog-with-provenance" pattern in action.

## Cross-paper coupling

- **P7**: delta_adaptiveg's `zvf` SUPPORTS verdict is the third
  empirical confirmation (after iter-31 and iter-47) that ZVF-driven
  adaptive group-size reduces ZVF on a real worktree panel.
- **P5**: delta_reinforce and delta_liteppo are the first entries
  with `measured=[]` and `n_claim_validation=0` declared honestly
  in the registry, not as a side-effect of unmeasured data but as
  an explicit "no same-stack arm yet" provenance block.

## Verification

```
python3 scripts/p5p8/p6_add_missing_deltas.py    # exit 0; 34/34 PASS
python3 registry/query.py validate               # 34/34 pass
python3 registry/query.py health                 # prints the audit
python3 scripts/p5p8/regenerate_schema_delta_enum.py   # 14 ids
```

Outputs:
- `experiments/results/p5p8/p6_new_deltas_audit.tsv` (3 new entries)
- `experiments/results/p5p8/p6_new_deltas_measured.tsv` (2 measured
  rows from delta_adaptiveg)
- `experiments/results/p5p8/p6_new_deltas_summary.json` (headline
  numbers)
- `experiments/results/p5p8/missing_delta_audit.tsv` (26 audited
  claims; 0 missing)

## Backward compatibility

- `registry/schema.json` modified: `delta_id` enum extended
  (additive; old ids preserved).
- `registry/entries/colab-open_grpo-adaptiveg_e3.json` modified:
  `variant_deltas_applied` now declares delta_adaptiveg (was `[]`).
- All other entries unchanged.
- All 31/31 pre-existing entries still pass.

## What was NOT done

- The iter-50 #19 sharpest actionable finding (decontamination
  schema bump) remains unaddressed; this iter prioritized closing
  the registry's missing-delta gap first.
- Real same-stack Tinker runs for REINFORCE and LitePPO are out of
  budget; the `measured=[]` blocks are honest provenance.
- The 0/26 missing-delta audit does not assert that the
  cross-reference is COMPLETE — only that the (claimed × registered)
  matrix is fully closed. A future audit can find methods
  REFERENCED in PROVENANCE (e.g. `wandb=...` records) but never
  claimed by a stack, which is a different query.
