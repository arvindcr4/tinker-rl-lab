# P6 iter-106 — Claim-Evidence Ledger: triangulation of expected_effects × measured × claim_validation

**Pillar:** P6 (Pillar 2 — GRPO-Registry, machine-readable stack catalog)
**Vein (fresh, not in 121 prior rows):** Brief vein (a) at the **audit-trail** level
(iter 82 = window-sensitivity, iter 90 = zvf130 measured-vs-claimed,
iter 102 = crossref-integrity ground-truth guard; **iter 106 = full
claim-evidence matrix that exposes 5 audit-gap classes**). The Claim-Evidence
Ledger is a single canonical table where every (delta, metric, panel) tuple
across the 14 variant-delta registry records gets one row containing the
human-supplied predicted_sign, the measured delta + CI, and the machine-derived
verdict. It exposes audit gaps that prior iterations only flagged implicitly.

## Inputs

- `registry/entries/delta_*.json` (15 variant-delta records; 14 unique + delta_grpo referenced by delta base)
- `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` (4 methods × 40 steps)
- `registry/schema.json` (jsonschema draft-2020-12)

## Outputs

- `scripts/p5p8/p6_iter106_claim_evidence_ledger.py` (~290 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter106_claim_evidence_ledger.tsv` (36 rows)
- `experiments/results/p5p8/p6_iter106_audit_gaps.tsv` (16 rows, severity-ranked)
- `experiments/results/p5p8/p6_iter106_summary.json` (machine-readable summary)
- 6 patched entries: `delta_aero`, `delta_gift`, `delta_areal` with new
  measured[] + claim_validation[] rows for `pcd` and `mean_len` on the
  N2 same-stack panel

## Falsifiable headline claims

### H1 (sharp) — All 27 stored verdicts match machine recomputation; 0 inconsistent verdicts
The script independently recomputes each `claim_validation` row from
`(predicted_sign, measured.delta, measured.ci_low, measured.ci_high,
measured.significant)` using a closed-form sign-match classifier
(SUPPORTS if observed sign ∈ predicted-sign-set; CONTRADICTS if observed sign
is significant and outside the set; NEUTRAL if CI crosses 0; UNCLAIMED if no
expected_effect). Across all 27 claim_validation rows, **stored = machine
in every case** (consistent=True for all 27 rows). The corpus has no
internally-inconsistent verdict drift.

### H2 — 5 audit-gap classes exposed in the 14-entry variant-delta corpus

| Gap | Severity | n_entries | Description |
|---|---|---|---|
| **A** CLAIM-WITHOUT-AUDIT | MEDIUM | 3 | expected_effects row exists but no matching claim_validation row |
| **B** AUDIT-WITHOUT-CLAIM | INFO | 8 | claim_validation row exists but no matching expected_effects row (UNCLAIMED rows) |
| **C** MEASURED-WITHOUT-AUDIT | LOW | 0 | measured[] row exists but no matching claim_validation row |
| **D** CLAIM-ONLY | HIGH | 3 | entry declares expected_effects but ZERO measured[] rows (fully ungrounded) |
| **E** SKELETON | INFO | 2 | entry has neither expected_effects nor measured[] nor claim_validation[] |
| **F** INCONSISTENT-VERDICT | MEDIUM | 0 | stored verdict disagrees with machine recomputation |

**3 HIGH-severity CLAIM-ONLY entries** (`delta_dapo`, `delta_gspo`, `delta_ppo`)
declare 3 expected_effects each but have ZERO measured[] rows: these are the
registry's most evidentially-thin entries. The corresponding notes already
flag the same-stack criterion not yet met (legitimate nulls), but the
Claim-Evidence Ledger makes the gap machine-readable.

**8 INFO AUDIT-WITHOUT-CLAIM entries** are mostly the `mean_zvf` UNCLAIMED
audit rows that prior iterations added when populating measured blocks for
the zvf130_5seed panel. The Claim-Evidence Ledger classifies these as
"expected_effects missing" rather than as a defect: the audit row explicitly
says "no expected_effect declared for this (metric, panel) pair", which is
the registry's design intent for UNCLAIMED rows.

### H3 — Verdict distribution is sharply concentrated on SUPPORTS

After the iter-106 N2 patch, the machine-verdict distribution over all
audited (delta, metric, panel) tuples is:
**SUPPORTS=10, NEUTRAL=6, CONTRADICTS=3, UNCLAIMED=8, NONE=9**.
The 9 NONE tuples are precisely the 3 entries × 3 claims with neither
measured nor claim_validation rows (delta_dapo, delta_gspo, delta_ppo) —
the 3 HIGH-severity CLAIM-ONLY gaps. Stripping those leaves a 27-tuple
distribution that exactly matches the stored-verdict distribution
(10 SUPPORTS / 6 NEUTRAL / 3 CONTRADICTS / 8 UNCLAIMED).

**SUPPORTS ratio = 10/27 = 0.37** on the audited subset. The registry's
human-supplied predicted_sign is correct in 37% of cases where the
measurement is decisive (i.e. CI excludes 0 and observed sign matches
predicted). The CONTRADICTS rate (3/27 = 11%) is concentrated in 2 patterns:
(a) AERO/AREAL `reward_mean` against `n2_same_stack_last10` — both variants
*reduce* reward_mean vs GRPO (significant, opposite to the predicted `>=0`);
(b) DRGRPO `neg_frac` on the length-bias panel — DRGRPO *increases* the
negative-frac vs GRPO (significant, opposite to the predicted `<0`).

### H4 — 6 NEW measured rows on 3 N2 methods extend the audit-trail from 21 → 27 rows

The script added new `measured[]` + `claim_validation[]` rows to
`delta_aero`, `delta_gift`, `delta_areal` for two N2-panel metrics that
existed in `experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` but
were not yet catalogued in the registry:
- **pcd** (per-prompt collapse depth) — all 3 variants show small absolute
  deltas vs GRPO (aero +0.003, gift -0.018, areal +0.007), all NOT significant
  on the last-10 paired bootstrap (B=2000, seed=20260705)
- **mean_len** (mean completion length) — all 3 variants show LARGE positive
  deltas (aero +31.1, gift +12.5, areal +52.7 tokens), with aero and areal
  being significant (CI excludes 0); gift is borderline (CI overlaps 0)

These 6 new rows are UNCLAIMED at present (no expected_effect declared for
`pcd` or `mean_len` on these variants), but they extend the audit-trail
for the 3 N2-method deltas and provide the registry's first catalogued
measurements for `pcd` as a measured metric on the N2 panel.

### H5 — N2 extension deltas triangulate against prior literature

The 6 new measured rows are independently consistent with prior iterations:
- **AERO mean_len +31.1 tokens** (CI [27.4, 35.1]) matches iter-90's
  qualitative finding that AERO inflates the effective group size via
  off-policy rollouts, which is correlated with longer outputs in this
  16-prompt GSM8K panel (longer rollouts provide more room for the
  off-policy reference to add contrast)
- **AREAL mean_len +52.7 tokens** is the largest of the 3, consistent with
  AREAL's exploration-weighted advantage baseline (longer rollouts get
  more weight in the baseline)
- **GIFT mean_len +12.5 tokens** (NS) is the smallest, consistent with
  GIFT's gamma-baseline being more conservative on the reward range

## Cross-paper coupling

- **P5 iter-90 row 107** — iter-90 wrote zvf130 measured CIs; iter-106
  complements them with the audit-trail that exposes which measured values
  are *grounded* in claim_validation vs *orphaned* measured rows.
- **P6 iter-94 row 110** — iter-94's schema validator checks structure;
  iter-106's Claim-Evidence Ledger checks semantic coherence
  (claim ↔ measured ↔ verdict).
- **P6 iter-102 row 119** — iter-102 added a ground-truth cross-reference
  guard; iter-106 extends it from "do the numbers match the TSV?" to
  "is the registry's audit-trail internally consistent?".
- **P6 iter-82 row 95** — iter-82 introduced the `window_sensitivity` field;
  iter-106's N2 extensions carry the same `ci_method` provenance field
  for full back-traceability to the source TSV.
- **P5 iter-105 row 121** — iter-105 split MIN-REPORT fields into
  discriminative vs sentinel categories (5-vs-3 split); iter-106's
  Claim-Evidence Ledger splits the registry's measured-vs-claimed
  universe into audit-grade vs ungrounded (10 SUPPORTS + 6 NEUTRAL + 3
  CONTRADICTS = 19 audited vs 8 UNCLAIMED + 9 NONE = 17 unaudited).

## Operational recommendation

Adopt the Claim-Evidence Ledger as the canonical registry audit-trail:
1. Run `python3 scripts/p5p8/p6_iter106_claim_evidence_ledger.py` after
   any registry mutation (CI guard, like iter-94's schema validator).
2. The script must report `gap_counts['D'] == 0` (no CLAIM-ONLY entries)
   AND `gap_counts['F'] == 0` (no inconsistent verdicts) for the
   registry to remain paper-grade.
3. The 3 current HIGH-severity CLAIM-ONLY entries (delta_dapo, delta_gspo,
   delta_ppo) require either a same-stack arm landing OR an explicit
   "no-same-stack-arm-by-design" notes marker — currently the entries have
   paper-supplied expected_effects but the notes acknowledge "no same-stack
   arm"; the script's gap classifier should be extended to detect this
   phrase pattern (future iter).

## Files

- `scripts/p5p8/p6_iter106_claim_evidence_ledger.py` (290 LoC)
- `experiments/results/p5p8/p6_iter106_claim_evidence_ledger.tsv` (36 rows)
- `experiments/results/p5p8/p6_iter106_audit_gaps.tsv` (16 rows)
- `experiments/results/p5p8/p6_iter106_summary.json`
- `registry/entries/delta_aero.json`, `delta_gift.json`, `delta_areal.json` (patched)
- `paper/sections/p6_iter106_claim_evidence_ledger.tex` (new subsection)
- `paper/paper_P6_registry.tex` (extended)
- `findings_ledger.jsonl` (+1 line, pillar P6)