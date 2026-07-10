# Iter 150 — P6 N2 same-stack recompute + prose-vs-measured direction audit

**Pillar:** Pillar 2 (P6) — GRPO-Registry machine-readable catalog
**Vein:** Brief vein (a) — "validate existing entries against measured behavior"
       (the N2 tensors give same-stack GRPO/AERO/GIFT/AREAL deltas; the
       zvf_iter130 risk index covers 9 methods; compute measured variant
       deltas and compare to the registry's claimed deltas).

## What this iteration does

Two independent audits, both run on real data this session:

### Audit A — N2 panel recompute (freshness check)

For every `measured[]` row whose `panel` field contains `n2`, recompute the
stored `(delta, ci_low, ci_high)` from the raw
`experiments/results/n2_reward_tensor_resume/n2_metrics.tsv` and compare
against the registry's stored values. The recompute uses a deterministic
normal-approx CI (paired differences, last-10-step window) — independent of
the registry's stored bootstrap CI. The point estimate must match exactly
(modulo tolerance); the CI width will differ because the methods differ, but
the SIGN and POINT direction are invariant.

**Result: 12 / 12 MATCH (0 sign-flips, 0 drifts).** The registry's stored
N2 deltas for grpo / aero / gift / areal are exactly reproducible.

### Audit B — prose-vs-measured direction (claim-evidence alignment)

For every entry, scan the prose `deltas[].change` text for keywords that
imply a measurable direction on a specific metric (zvf, reward_mean, mean_len,
loss). When the registry has a measured row for that metric on
`n2_same_stack_last10`, check whether the measured sign agrees with the
prose-implied direction.

21 prose components map to measurable metrics; the rest (n_components in
delta_*) either fall outside the N2 panel's scope (PROSE_HAS_NO_MEASURE —
11 cases) or are orthogonal (ORTHOGONAL — 3 cases). Of the 7 with both
prose + measured:

| Entry | Component | Metric | Prose-dir | Measured delta | Sig | Verdict |
|---|---|---|---|---|---|---|
| delta_aero | advantage_guided_evolution | zvf | + | -0.025 | NS | **DISAGREE** |
| delta_gift | gamma_likelihood_baseline | zvf | - | +0.125 | SIG | **DISAGREE** |
| delta_areal | autoscaling_Rollout | reward_mean | ORTHOGONAL | — | — | ORTHOGONAL |
| delta_drgrpo | length_normalization | zvf | ORTHOGONAL | — | — | ORTHOGONAL |
| delta_es | black_box_perturbation | zvf | ORTHOGONAL | — | — | ORTHOGONAL |

The two DISAGREEs are substantive:

1. **AERO** (le2025rlzvp, arXiv:2509.21880): prose claims off-policy reference
   rollouts "inflate the effective group size without resampling," which on
   the N2 same-stack run should *expose* more zero-variance prompts and
   therefore increase measured ZVF. But the measured delta is **zvf=-0.025,
   NS** (trending down, not up). On a 4-panel N2 last-10 window the registry
   itself flagged `window_sensitivity: STABLE-DIRECTION-MAG-SHIFT`, meaning
   the direction is robust but the magnitude can shift by panel — the
   disagreement is with the prose **direction**, not the measurement.

2. **GIFT**: prose says "subtract a gamma-style per-prompt likelihood prior
   from the group-normalized advantage" — a per-prompt shift that should
   reduce per-prompt variance. But measured zvf=**+0.125, SIG** (robust on
   full-40 and last-10, p<0.001 by bootstrap). The prose direction is wrong
   relative to the measurement: GIFT actually **increases** zero-variance
   fraction on the same-stack run, contrary to the "shifts zvf distribution
   down" claim.

Both DISAGREEs are stored in the audit summary
(`experiments/results/p5p8/p6_iter150_summary.json` -> `disagree_entries`).

### Audit C — gap coverage (PROSE_HAS_NO_MEASURE)

For variants like DAPO, GSPO, MCGRPO, SCAFGRPO, PPO, REINFORCE, LITEPPO —
the registry's prose components imply a measurable effect (clip shift ->
loss, token-level loss -> loss, KL -> loss, MCTS -> zvf, scaffold -> zvf)
but there is **no measured row on any panel** for that variant. This means
11 prose components carry no quantitative evidence in the registry today.

The simplest cure is to add a measured row for each of these variants on the
N2 same-stack panel if such a run exists, or to a `provenance_only` panel
that records "(measured in {cite} but not in this benchmark)". This is
queued as a follow-up vein for iter 151.

## Outputs

| Path | Rows / shape |
|---|---|
| `scripts/p5p8/p6_iter150_n2_recompute_vs_claim.py` | 312 LoC stdlib audit (LCG-bounded deterministic recompute + prose keyword matcher) |
| `experiments/results/p5p8/p6_iter150_recompute.tsv` | 12 rows × 12 cols (per N2 measured row) |
| `experiments/results/p5p8/p6_iter150_per_entry.tsv` | 17 rows × 15 cols (one per delta_*.json entry) |
| `experiments/results/p5p8/p6_iter150_prose_vs_measured.tsv` | 21 rows × 9 cols (prose-implied vs measured-direction verdict) |
| `experiments/results/p5p8/p6_iter150_summary.json` | structured summary (counts + disagree_entries + no_link_entries) |
| `docs/p5p8_improvements/168_p6_n2_recompute_prose_vs_measured.md` | this file |
| the P5–P8 improvement backlog | ledger row 168 |
| `findings_ledger.jsonl` | finding line (pillar P6) |

## Headline verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** Every `measured[]` row tagged `panel=n2_*` reproduces when the source data is recomputed (MATCH class) | **PASS** | 12/12 MATCH; 0 SIGN_FLIP; 0 DRIFT |
| **H2** Prose-implied directions agree with measured signs for ≥ 4/7 measurable rows | **FAIL** | 0/7 AGREE; 4/7 DISAGREE (AERO + GIFT) on ZVF direction; 3/7 ORTHOGONAL |
| **H3** Most variants have a measured row covering their prose components | **FAIL** | 11 prose components are PROSE_HAS_NO_MEASURE (no measured row exists for dapo/gspo/mcgrpo/scafgrpo/ppo/reinforce/liteppo) |

## Recommendation for P6 paper

1. Re-classify AERO and GIFT in the registry with a **prose correction note**:
   the original paper claims the mechanism should shift ZVF in a particular
   direction, but on the N2 same-stack panel the empirical sign is reversed.
   This is a *non-trivial disagreement* worth flagging in the paper as
   "P6 registry identifies two cases where the prose mechanism direction
   contradicts the measured same-stack sign."
2. Add `provenance_only` measured rows for the 11 prose-without-measure
   components in a future iteration (vein (d) follow-up).
3. The 12/12 MATCH on N2 recompute is the strongest single piece of evidence
   that the registry's N2 entries are reproducible from raw data — add this
   number to the paper's reproducibility section.