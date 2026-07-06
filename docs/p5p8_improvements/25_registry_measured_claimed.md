# 25 — Registry Measured-vs-Claimed Variant-Delta Reconciliation

**Pillar 2 (P6) — machine-readable stack catalog. Iteration 18.**

## Question

The registry stores 11 `variant_delta` records whose `claim_summary` text
describes the algorithmic change each named variant makes to base GRPO.
For 4 of those 11 (`aero`, `gift`, `areal`, plus the implicit `grpo` baseline)
we have a **same-stack N2 measurement** (Tinker-managed sampler, G=8, seed 0,
40 steps, 16 prompts/step). For 9 of those 11 (`grpo`, `aero`, `gift`, `areal`,
`cppo`, `ngrpo`, `mcgrpo`, `es`, `scafgrpo`) we have a **zvf130 risk-index
measurement** (5 seeds each). The remaining 2 (`dapo`, `drgrpo`, `gspo` —
that's 3 not 2) have **no measured proxy** at all: only a registry claim and
10 / 4 / 2 claimant stacks respectively.

The reconciliation question: **for each variant delta, does the measured
proxy agree with the qualitative direction the registry claim predicts?**

## What we built

- `scripts/p5p8/registry_measured_claimed.py` (≤300 LoC, stdlib only).
  Loads all 11 delta entries, the N2 same-stack paired-bootstrap deltas
  (from iter 2's `registry_measured_deltas.json`), the zvf130 risk index,
  and the registry-side claimant stacks (where the variant-delta is
  recorded as `implemented` / `surrogate` / `absent` / `unknown`).

  For each `(delta_id, proxy_metric)` pair it returns one of four
  verdicts:
  - **SUPPORT** — measured sign matches predicted sign AND paired-bootstrap CI excludes 0
  - **WEAK** — measured sign matches predicted sign but CI contains 0
  - **OPPOSE** — measured sign contradicts predicted sign
  - **NO_DATA** — no measured proxy available for this method/proxy pair

  Predicted signs come from a hand-coded mapping that reads each
  delta's `claim_summary` (see CLAIMS dict in the script).

- `experiments/results/p5p8/registry_measured_claimed.tsv` — 11 rows, one
  per delta, with N2 verdict counts + zvf130 evidence + registry claimant
  counts/statuses/frameworks/opennesses.
- `experiments/results/p5p8/registry_measured_claimed.json` — machine-readable
  per-proxy evidence, zvf130 per-method ZVF summary, and the claimant
  stack records.

## Headline findings

### 1. The registry→measurement map is mostly **NO_DATA** at the per-proxy level

Only 4/11 variants have an N2 same-stack measurement. Of the 5 proxies we
tested (`zvf`, `loss`, `mean_len`, `cv_len`, `reward_mean`), the SUPPORT
verdict fires **0 times** across 11 deltas. The verdicts that do fire are:
- 14 × WEAK (sign matches but paired-bootstrap CI contains 0)
- 1 × OPPOSE (aero ZVF: predicted -1, measured +0.025, CI contains 0 but
  the per-seed risk index contradicts even more strongly — see §3)
- 40 × NO_DATA

### 2. The zvf130 risk index is the *only* proxy that discriminates

Every one of the 8 measured variants has **lower mean ZVF than grpo** on
the zvf130 5-seed panel. Drops range from −0.164 (scafgrpo) to −0.367
(es / gift). The variants cluster in two regimes:

| Variant   | mean_zvf | Δ_vs_grpo | interpretation |
|-----------|----------|-----------|----------------|
| grpo      | 0.481    | +0.000    | baseline (high-contrast regime) |
| scafgrpo  | 0.317    | −0.164    | mid-regime, mostly contrast-preserved |
| ngrpo     | 0.300    | −0.180    | mid-regime |
| cppo      | 0.295    | −0.186    | mid-regime |
| aero      | 0.220    | −0.261    | lower-mid |
| mcgrpo    | 0.146    | −0.335    | collapse regime |
| areal     | 0.121    | −0.360    | collapse regime |
| es        | 0.114    | −0.367    | collapse regime |
| gift      | 0.114    | −0.367    | collapse regime |

The 5/5-seed reproducibility of this monotonic drop is itself the
strongest claim-direction finding: **the variants do not improve ZVF on
this prompt distribution; they all move it lower, regardless of what the
registry claim says about within-group contrast**. The `mcgrpo` diversity
bonus and the `aero` off-policy rollouts are *claimed* to raise within-group
contrast; measured ZVF says the opposite.

### 3. The N2 same-stack run is too short / single-seed to detect claim→measurement agreement

The N2 run is 1 seed × 40 steps × 16 prompts. On this, the only CI that
excludes 0 is the **GIFT loss delta** (+16,722 — its gamma-style
likelihood prior as an additive constant). The reward deltas (the variants'
*central claim*) all sit inside the paired bootstrap CI. That is the
headline from iter 2, and the reconciliation makes it explicit: on the
N2 data alone, **no variant can claim a measured reward improvement over
grpo** at the same stack.

### 4. 3/11 variants are claim-only (no measurement)

`dapo`, `drgrpo`, `gspo` are registered but have **no N2 same-stack
measurement and no zvf130 measurement**. They are also the variants with
the **largest registry claimant populations** (10 / 4 / 2 respectively) —
the registry-side adoption is real, but the evidence-base is empty.

This is the strongest finding for paper-facing purposes: the registry
**currently scores DAPO with 10 claimants** but has **zero measured
proxies** for DAPO. A reviewer can verify this directly from the
`registry_measured_claimed.tsv` row for `delta_dapo`.

### 5. The DAPO status mix (iter 14 audit) corroborates this

Iter 14 already showed that `delta_dapo` has the most diverse claimant
status mix: 4 implemented / 1 surrogate / 3 absent / 2 unknown across
the 10 claimant stacks. Combined with the no-measurement finding, this
gives a falsifiable interpretation: **DAPO is the most-adopted variant
in the registry's claimant pool, but is also the one whose implementation
status is most ambiguous and whose measured effect is least characterised**.

## Implications for Pillar 2 (P6)

- The registry's claim→measured agreement table is now part of the artifact.
  Reviewers can pick any variant-delta row and read off (a) the claimed
  summary, (b) the measured proxies, (c) the verdict.
- The 8/8 variants-below-grpo-ZVF finding is a **new, falsifiable claim**
  for P6: the registry's claim that variants "boost contrast" is *not*
  supported by the measured risk index on the canonical prompt distribution.
- The 3/11 claim-only finding identifies the work-list for the next
  registry-measurement iteration: bring DAPO, Dr.GRPO, and GSPO onto the
  zvf130 risk index.

## Reproducibility

```
python3 scripts/p5p8/registry_measured_claimed.py
# writes experiments/results/p5p8/registry_measured_claimed.{tsv,json}
```

All inputs are committed (`registry/entries/delta_*.json`,
`experiments/results/p5p8/registry_measured_deltas.json`,
`experiments/results/zvf_iter130_risk_index.tsv`). No Tinker call, no
external data, stdlib only.

## Provenance

- N2 paired bootstrap deltas: `experiments/results/p5p8/registry_measured_deltas.json`
  (iter 2, `scripts/p5p8/registry_validate.py`).
- zvf130 risk index: `experiments/results/zvf_iter130_risk_index.tsv`
  (worktree, 9 methods × 5 seeds = 45 rows).
- Registry-side claimants: `registry/entries/*.json` (skipping `delta_*.json`),
  iterating `variant_deltas_applied[*]` for the matching `delta_id`.
- Predicted-sign mapping: CLAIMS dict in `scripts/p5p8/registry_measured_claimed.py`,
  reading each delta's `claim_summary` text and the source paper (citations
  already verified in iter 10's `variant_delta_citation_audit.tsv`).