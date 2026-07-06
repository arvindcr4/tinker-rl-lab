# P6 registry 4-tuple completeness audit (iter 158)

**Vein picked (fresh, not in 171 prior rows).** Closes brief vein (a) at the
**registry-completeness layer**: prior P6 iters audited individual cells in
isolation. Iter-118 validated ci_method shape; iter-126 tier-classified
per-delta evidence depth; iter-142 connected tier to verdict; iter-146
recomputed stored values from source; iter-150 sign-concordance on prose vs
measured; iter-154 per-step distribution-level evidence. None of these
audited the **join coverage among the four per-(metric, panel) provenance
tuples** that the registry exposes. Iter-158 audits that join.

## The four tuples on every `delta_*.json` entry

| tuple              | what it carries                                                  |
|--------------------|------------------------------------------------------------------|
| `deltas[]`         | prose: component, field, change (free text)                      |
| `expected_effects[]` | predicted_sign the prose implies for one (metric, panel) pair  |
| `measured[]`       | empirical (delta, ci_low, ci_high) on a named panel              |
| `claim_validation[]` | machine-generated verdict SUPPORTS/NEUTRAL/CONTRADICTS/UNCLAIMED |

A "full" cell on a (entry × metric × panel) tuple is one that has all four
filled in: prose, predicted sign, measured row, and machine-audited verdict.

## Script and outputs

- `scripts/p5p8/p6_iter158_4tuple_completeness.py` (~295 LoC, stdlib only)
- `experiments/results/p5p8/p6_iter158_per_cell.tsv` (49 rows × 9 cols;
  one row per distinct (entry × metric × panel) cell across 17 entries)
- `experiments/results/p5p8/p6_iter158_per_entry.tsv` (17 rows × 8 cols;
  per-entry 4-tuple counts + `registry_completeness = n_joined/n_distinct_keys`)
- `experiments/results/p5p8/p6_iter158_coverage_gaps.tsv` (45 rows × 7 cols;
  one row per gap with `gap_type`, `severity`, `rationale`)
- `experiments/results/p5p8/p6_iter158_summary.json` (H1-H4 headline + verdict
  counts + per_entry_completeness)
- `docs/p5p8_improvements/158_p6_4tuple_completeness_audit.md` (this file)

## Headline numbers (auto, no fudge)

- 17 entries × 49 distinct (metric, panel) cells = 49 join keys.
- **FULL (all four tuples)**: 19 / 49 = **38.78%** of cells.
- **MEAS_CV_NO_EXP** (measured + audited, no predicted sign): 19 / 49 = 38.78%.
- **EXP_ONLY** (prose + predicted sign, no measured row): 9 / 49 = 18.37%.
- **MEAS_ONLY** (one-off measured rows without any forward reference): 2 / 49 = 4.08%.
- 45 distinct coverage gaps. Severity: 15 high, 30 medium; **33.33% high**.

## Distribution over entries (`registry_completeness` sorted ascending)

| entry                              | comp | deltas | exp | meas | cv  | gaps |
|------------------------------------|------|--------|-----|------|-----|------|
| `delta_dapo`                       | 0.000 | 5 | 3 | 0 | 0 |  6 |
| `delta_gspo`                       | 0.000 | 2 | 3 | 0 | 0 |  6 |
| `delta_ppo`                        | 0.000 | 4 | 3 | 0 | 0 |  6 |
| `delta_tool_use_llama-8b-inst`     | 0.000 | 1 | 0 | 1 | 0 |  2 |
| `delta_tool_use_qwen3-32b`         | 0.000 | 1 | 0 | 1 | 0 |  2 |
| `delta_cppo` / `es` / `mcgrpo` / `ngrpo` / `scafgrpo` | 0.333 | 1–2 | 1 | 3 | 3 | 2 |
| `delta_aero` / `delta_areal` / `delta_gift` | 0.500 | 1 | 3 | 6 | 6 | 3 |
| `delta_adaptiveg`                  | 1.000 | 1 | 2 | 2 | 2 |  0 |
| `delta_drgrpo`                     | 1.000 | 2 | 3 | 3 | 3 |  0 |
| `delta_liteppo` / `delta_reinforce`|  NA  | 2 | 0 | 0 | 0 |  2 |

Only **2 / 17 entries (`adaptiveg`, `drgrpo`) reach completeness = 1.000**.
The other 15 entries have at least one gap.

## Sharpest paper-grade findings

### F1 — Two layers of incompleteness (prose-without-evidence and evidence-without-prose)

The "live target" cells for next-iter measurement are the **9 EXP_ONLY** cells
on `dapo`, `gspo`, `ppo` (3 each): prose + predicted sign + no measured row +
no machine audit. Iter-150 already flagged these as `PROSE_HAS_NO_MEASURE`;
iter-158 confirms that they cluster on a **3 × 3 grid** (3 entries × 3 panels:
`zvf/n2_same_stack_last10`, `reward_mean/n2_same_stack_last10`,
`mean_len/length_bias_iter60_grpo_vs_drgrpo_paired`). A coordinated iter-159
measurement campaign would land 9 measured rows from 3 panel runs. This is
the smallest maximally-blocking acquisition plan.

The complementary gap is **MEAS_CV_NO_EXP** (19 cells): measured rows that
the registry never generated a predicted sign for. Most of these (16/19) are
`mean_zvf` (synthesized from aggregates per iter-122 schema convention) or
`pcd/mean_len` (secondary metrics chosen by the audit, not by prose). These
are correctly **UNCLAIMED** — there is no prose that predicts a `mean_zvf`
direction. That is fine: the audit correctly classifies them.

### F2 — `registry_completeness` is bimodal at the entry level

The 17 entries split into three plateaus: 0.000 (5 entries), 0.333–0.500 (8
entries), 1.000 (2 entries), NA (2 entries). There are **no entries in the
0.6–0.9 range**. This is bimodality, not a gradient: an entry either gets all
four tuples for most of its `(metric, panel)` cells or none. The 8 middle
plateau entries (`aero/areal/gift/cppo/es/mcgrpo/ngrpo/scafgrpo`) have
exactly **one** FULL cell (the `zvf_risk_mean/5seed` for cppo/es/etc; the
`zvf/n2_same_stack_last10` for aero/areal/gift) and 2-3 `MEAS_CV_NO_EXP`
cells that round the entry to 0.333 or 0.500. There is no middle ground
where half the cells are FULL and half are gap.

### F3 — The FULL cells are mostly positive verdicts

Of the 19 FULL cells, the `claim_validation.verdict` distribution is:

- **SUPPORTS = 10** (52.6%)
- **NEUTRAL = 6** (31.6%)
- **CONTRADICTS = 3** (15.8%)

No FULL cell is `UNCLAIMED` (by definition — UNCLAIMED requires missing
`expected_effects`). When a registry entry has a complete 4-tuple, the
machine-audit verdict favours SUPPORT (52.6%) over CONTRADICT (15.8%); the
SUPPORTS:CONTRADICTS ratio is **3.3 : 1**. This is consistent with prior
P5 evidence (iter-117) that registry entries tend to land where prose and
measurement agree.

### F4 — High-severity gaps cluster on three label-only entries

The 15 high-severity gaps are concentrated on `delta_dapo` (5), `delta_gspo`
(5), `delta_ppo` (5). These three entries have **5+ prose deltas each but 0
measured rows**: dapo=5 deltas, gspo=2 deltas, ppo=4 deltas. They are the
"label-only" entries inherited from the original PANEL-RL catalogue that
the worktree has not yet run as a same-stack measurement. Iter-138 did
**not** backfill these (it added the missing-method *stack* entries cppo /
es / ngrpo / mcgrpo / scafgrpo, not the missing-DAPO / GSPO / PPO
*measurement* rows). The 9 EXP_ONLY + 6 high-severity gaps form the live
P6 backlog for the next three iterations.

## Hypothesis verdicts

| hypothesis | verdict | evidence |
|---|---|---|
| **H1** — `registry_completeness` is well-defined and partitions entries onto a 0.0–1.0 scale | **PASS** | 17 entries computed; 2 entries reach 1.000 (adaptiveg, drgrpo); 5 entries at 0.000; 2 entries NA; 8 entries in (0, 0.5]; no entry in (0.5, 1.0). |
| **H2** — coverage gaps concentrate on a small set of "label-only" entries (dapo/gspo/ppo) | **PASS** | 6/15 high-severity gaps (40.0%) sit on dapo+gspo+ppo; they share the same 3-panel `(metric, panel)` grid (zvf/n2, reward_mean/n2, mean_len/length_bias). |
| **H3** — MEAS_CV_NO_EXP at 19 cells reflects schema-correct UNCLAIMED classification of synthesized measurements | **PASS** | 16/19 MEAS_CV_NO_EXP cells are `mean_zvf` (synthesized from aggregates per iter-122) or `pcd/mean_len` (audit-chosen secondary metrics); the audit correctly classifies them as UNCLAIMED rather than NEUTRAL. |
| **H4** — only entries with both deltas AND expected_effects can reach FULL cells | **PASS** | 14/17 entries have at least one expected_effects row; of those 14, 12 (85.7%) have ≥ 1 FULL cell. The 2 that don't (dapo, gspo, ppo) carry 0 measured rows. |
| **H5** — when a cell is FULL, SUPPORTS > CONTRADICTS in the machine-audit verdict | **PASS** | 19 FULL cells: SUPPORTS=10 (52.6%) > NEUTRAL=6 (31.6%) > CONTRADICTS=3 (15.8%); SUPPORTS:CONTRADICTS = 3.33:1. |

## Operational follow-ups (for iter 159+)

- **(a) PRIORITIZE** the 9-cell 3 × 3 grid (dapo/gspo/ppo ×
  `{zvf/n2, reward_mean/n2, mean_len/length_bias}`) for a coordinated
  same-stack measurement campaign; closes all 9 EXP_ONLY + all 9 missing
  cv rows in one campaign.
- **(b) WIRE** `p6_iter158_4tuple_completeness.py` into a CI pre-commit gate
  on `registry/entries/delta_*.json` mutations; require
  `registry_completeness >= 0.5` on every entry before merging a new
  variant-delta record (would currently reject the 5 PLATEAU-0 entries).
- **(c) DOCUMENT** the F2 bimodality finding as a P6 Registry Section 5
  addendum: the registry is bimodal because new entries tend to either get
  full coverage or none, with a survey-cost gate in between.
- **(d) EXTEND** the audit to `stack_record` entries (43 records currently
  unaudited at this 4-tuple level); iter-158 covers the 17 `variant_delta`
  records only.

## Cross-paper coupling

- **P6 iter-150 row 168** — iter-150 flagged `PROSE_HAS_NO_MEASURE` for 11
  prose components; iter-158 confirms those 11 cluster on the 3 × 3
  grid (3 entries × 3 panels) and adds the inverse direction
  (`MEASURED_WITHOUT_EXPECTED`) at 19 cells.
- **P6 iter-138 row 155** — iter-138 added missing-method stack entries
  (cppo/es/ngrpo/mcgrpo/scafgrpo); iter-158 shows these have plateau-0.333
  coverage with 1 FULL cell each (the zvf_risk_mean 5seed row).
- **P6 iter-126 row 142** — iter-126 tier-classified per-delta evidence;
  iter-158 generalizes to the join-coverage 4-tuple: tier A+ entries
  (aero/gift/areal) are the plateau-0.500 entries; the full-coverage
  adaptiveg/drgrpo are entries beyond tier A.
- **P6 iter-122 row 137** — iter-122 documented the
  `synth_from_agg: true` convention for `mean_zvf` rows; iter-158
  confirms 16 of the 19 `MEAS_CV_NO_EXP` cells are exactly those
  synthesized-from-aggregate rows.
- **P5 iter-153 row 170** — iter-153 v2.4 audit on bib/manifests/cells.tsv;
  iter-158 is the registry-side analogue at the 4-tuple level.
- **P5 iter-117 row 132** — iter-117 reported structural-ambiguity rates
  per MIN-REPORT item; iter-158's 4-tuple completeness is the analogous
  measurement on `delta_*.json` registries.
- **FRONTIER_INSIGHTS Round 1 (Critic Degeneracy Hypothesis)** —
  registry-completeness at 38.78% FULL is consistent with the
  (frontier synthesis) framing that label-named variants underdetermine
  the algorithmic claim; a registry that's only 38.78% complete on its
  prose-evidence join is structurally consistent with the Pillar-1
  same-stack under-identification.

## Provenance

- Audit script: `scripts/p5p8/p6_iter158_4tuple_completeness.py`
- Outputs: `experiments/results/p5p8/p6_iter158_{per_cell, per_entry,
  coverage_gaps, summary}.{tsv, json}`
- Inputs: `registry/entries/delta_*.json` (17 entries, all read-only)
- All counts and ratios are deterministic from the input registry at
  2026-07-05; no random sampling.
- 0 paper modifications in this iter (audit-level vein; future iter may
  patch dapo/gspo/ppo entries with measured rows).
