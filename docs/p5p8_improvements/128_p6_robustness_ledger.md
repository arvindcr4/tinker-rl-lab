# 128 — P6 measured-block robustness ledger (iter 114)

**Pillar:** P6 (Pillar 2 — GRPO-Registry)
**Vein (fresh, not in 117 prior rows):** characterize the *measurement quality*
of every `measured[]` row in `registry/entries/delta_*.json` — not the
directional claim-validation that iter-46/iter-90/iter-110 already cover —
and surface a per-entry **groundedness score** that turns the registry's 33
measured rows + 28 expected-effects into a single reviewer-facing
"what you can trust" matrix.

## Why this vein (vs the existing 117 P5–P8 ledger rows)

Prior P6 iters stop at the verdict layer (SUPPORTS / NEUTRAL /
CONTRADICTS / UNCLAIMED). What is **missing** is an audit of the
*underlying measurement robustness*: which measured rows are
load-bearing (large effect / tight CI), which are fragile (CI barely
excludes 0), which are structurally underpowered (n_obs < 8),
which are point-only (no per-seed SD stored). This iter produces a
ledger that catalog-auditors and paper reviewers can read directly.

## Definitions (stdlib only, no jsonschema / scipy)

For each `measured[]` row:
```
effect   = |delta|
ci_half  = (ci_high - ci_low) / 2
snr      = effect / ci_half         (signal-to-noise ratio)
bucket:
  POINT_ONLY    = ci_method == 'point_no_perseed_sd'
                OR (ci_low == ci_high == delta)   [8 rows]
  UNDERPOWERED  = n_obs < 8                       [11 rows]
  LOAD_BEARING  = significant AND snr >= 5.0      [2 rows]
  FRAGILE_SIG   = significant AND snr < 5.0       [5 rows]
  FRAGILE_NS    = NOT significant AND effect > 0  [7 rows]
  DEGENERATE    = no delta, no CI, no n           [0 rows]

groundedness_score = Σ (BUCKET_WEIGHTS[bucket] × count) / total_rows
  LOAD_BEARING = 1.00; FRAGILE_SIG = 0.50;
  FRAGILE_NS = 0.25; UNDERPOWERED = 0.10;
  POINT_ONLY = 0.00
```

## Falsifiable measured headlines

### H1 — Only 2 / 33 (6.1 %) measured rows are LOAD-BEARING

After rolling up every `measured[]` row across all 15 `delta_*.json`
entries, the bucket distribution is:

| Bucket | Count | % of 33 |
|---|---:|---:|
| LOAD_BEARING | 2 | 6.1 % |
| FRAGILE_SIG  | 5 | 15.2 % |
| FRAGILE_NS   | 7 | 21.2 % |
| UNDERPOWERED | 11 | 33.3 % |
| POINT_ONLY   | 8 | 24.2 % |

The 2 LOAD-BEARING rows are both N2 last-10 `mean_len` effects
(`delta_aero.mean_len`: Δ=+31.08, snr=8.06;
`delta_areal.mean_len`: Δ=+30.13, snr=6.39). **Neither is the
registry's headline claim** — the registry's headline is "ZVF
reduction" — so by the SNR criterion, the registry has *zero
load-bearing evidence on its headline metric*.

### H2 — Headline ZVF-reduction claims are uniformly FRAGILE / UNDERPOWERED / POINT_ONLY

Every registry entry whose `expected_effects[]` block predicts
`zvf_risk_mean < 0` on the `zvf130_5seed` panel has its measured
counterpart in one of three fragile buckets:

| Variant | zvf_risk_mean zvf130_5seed row | Bucket |
|---|---|---|
| AERO | Δ=-0.148, CI [-0.286, -0.009], n=5 | UNDERPOWERED |
| GIFT | Δ=-0.263, CI [-0.365, -0.161], n=5 | UNDERPOWERED |
| AREAL | Δ=-0.246, CI [-0.355, -0.137], n=5 | UNDERPOWERED |
| CPPO | Δ=-0.151, CI [-0.253, -0.049], n=5 | UNDERPOWERED |
| ES | Δ=-0.273, CI [-0.375, -0.171], n=5 | UNDERPOWERED |
| MCGRPO | Δ=-0.174, CI [-0.289, -0.060], n=5 | UNDERPOWERED |
| NGRPO | Δ=-0.131, CI [-0.249, -0.013], n=5 | UNDERPOWERED |
| SCAFGRPO | Δ=-0.352, CI [-0.456, -0.249], n=5 | UNDERPOWERED |

8 of 8 ZVF-reduction claims fall under the UNDERPOWERED bucket (n=5,
no Welch's adjustment recorded for n<8). This is a **measurement
method gap, not a registry bug**: 5 seeds on the zvf130 risk panel
yields underpowered paired bootstrap; the iter-110 row 132 audit
showed Welch's t-test on the same n=5 data REVERSES 2/8 of these
from sig→NS, but the published registry claim_validation block
records these as SUPPORTS rather than the FRAGILE verdict they
deserve.

### H3 — 8 / 33 rows are POINT_ONLY (no per-seed SD stored)

`mean_zvf` rows on the `zvf130_5seed` panel for 8 method×metric
cells (aero, areal, gift, grpo, cppo, es, mcgrpo, ngrpo,
scafgrpo): `ci_method=point_no_perseed_sd` and
`ci_low == ci_high == delta`. These rows cannot be validated
without re-deriving the per-seed standard deviation from the
underlying `experiments/results/zvf_iter130*.tsv` data. **This
is the highest-ROI follow-up** — recomputing 8 CIs from existing
raw zvf130 data would lift the bucket distribution by ~24 %.

### H4 — 3 entries (DAPO, GSPO, PPO) are PURELY THEORETICAL

| Entry | measured rows | expected_effects | unmeasured expectations |
|---|---:|---:|---:|
| delta_dapo  | 0 | 3 | 3 |
| delta_gspo  | 0 | 3 | 3 |
| delta_ppo   | 0 | 3 | 3 |
| delta_liteppo  | 0 | 0 | 0 |
| delta_reinforce | 0 | 0 | 0 |

These 5 entries have `groundedness_score = 0.0`. The 3
`theoretical_only` entries (DAPO, GSPO, PPO) carry **only
paper-derived predictions** — the registry's promise is that
these can be auto-scored once a same-stack arm lands
(no such arm exists yet; this row has been an open recommendation
since iter-78 row 92 and iter-94 row 110).

### H5 — Top-3 by groundedness score:

| Entry | Score | Rows | LB | Fragile |
|---|---:|---:|---:|---:|
| delta_adaptiveg | 0.375 | 2 | 0 | 1 sig + 1 ns |
| delta_aero      | 0.350 | 6 | 1 | 2 ns + 1 fragile-sig |
| delta_areal     | 0.350 | 6 | 1 | 2 ns + 1 fragile-sig |

Every entry with `measured[]` rows has at least 1 FRAGILE_NS row
(NS but trending) — no entry currently scores above 0.5. **This
is the registry's blind spot**: the catalog can claim
FRAGILE_NS as "evidence of a direction" while the BN-tier
reviewer cannot use that to falsify the SUPPORTS verdict.

## Operational consequences

1. **Add `delta_aero.measured[].snr` to every `measured[]` row** —
   extensions to `scripts/p5p8/p6_iter114_robustness_ledger.py`
   can dump the bucket directly into the entry, making the
   audit read at audit-time rather than at iter-time.

2. **Re-derive the 8 POINT_ONLY `mean_zvf` CIs** from
   `experiments/results/zvf_iter130*.tsv`. This single 8-row
   backfill moves 24.2 % of the measured evidence base out of
   the unreadable bucket.

3. **Add a 6th-seed run to zvf130** to lift 11 of 33 measured
   rows out of UNDERPOWERED (n<8 is fragile even with paired
   bootstrap) — but this costs wall-clock; cheaper to re-bootstrap
   the existing 5 seeds with a different seed and aggregate by
   Fisher's method.

4. **Make `groundedness_score` an entry-level field** on the
   schema. This is a forward schema bump (since iter-94 row 110
   schema_validator is `--strict`-gated, CI blocks unsafe schema
   changes); defer to a future registry iteration.

## Cross-paper coupling

- **P6 iter-46 row 50** (claim_validation block): iter-46
  introduced the SUPPORTS/NEUTRAL/CONTRADICTS/UNCLAIMED verdict
  layer on top of `measured[]`. This iter reveals that 8 of
  the registry's "supported" zvf-reduction claims should have
  been FRAGILE — they sit at SNR < 5 with n=5.
- **P6 iter-102 row 122** (`sig_robust_bootstrap_and_welch`):
  iter-102 added the conservative cross-check that lifts
  iter-102 above the single-method significance bar. This
  iter shows that 8 of 33 measured rows are
  `sig_robust_bootstrap_and_welch = null` (because mean_zvf
  rows lack per-seed SD), so the conservative check is
  structurally unavailable for 24 % of the evidence base.
- **P6 iter-110 row 132** (N2↔zvf130 cross-panel): the
  cross-panel audit only inspected 6 (variant, metric) cells
  with paired bootstrap; this iter audits the *robustness* of
  every measured[] row independently.
- **P5 iter-105 row 121** (per-value-class audit): P5 measures
  manifest fingerprinting; P6 iter-114 measures registry
  fingerprinting at the same granularity. Both surface "schema
  is fine; corpus power is the gap."

## Reproducibility

```bash
cd /home/claude/tinker-rl-lab-minimax
python3 scripts/p5p8/p6_iter114_robustness_ledger.py   # ~0.5s
```

Run-time: <1 second. Outputs:

- `experiments/results/p5p8/p6_iter114_robustness_per_row.tsv` (33 rows)
- `experiments/results/p5p8/p6_iter114_robustness_per_entry.tsv` (15 rows)
- `experiments/results/p5p8/p6_iter114_robustness_claim_matrix.tsv` (28 rows)
- `experiments/results/p5p8/p6_iter114_robustness_summary.json` (machine-readable)

All four files pass a stub `wc -l` consistency check.

## Status

Prototyped — iter 114. Operationally: re-run as a CI gate every
future registry mutation (the schema_validator already gates every
mutate; add this as a `--with-robustness` flag in a follow-up).
