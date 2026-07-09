# 58 — P8 cost-per-fraud-caught ($/fraud_caught) accounting on the iter-40 grid

**Vein (fresh, not in prior ledger; complementary metric to iter-28/40).**
iter-28, iter-32, iter-36, iter-40 all use cost/decision ($/dec) on the
held-out fraud split. $/dec is dominated by alert count — a noisy
over-alerter looks "cheap per stream" but may be paying $50/alerts ×
many to catch a single fraud. The complementary metric fraud-ops
*actually* reports is $/fraud_caught = total_cost / true_positives at
the cost-optimal threshold. This iter is the first P8 vein to lift
the $/dec convention and report $/fraud_caught as a separate axis.

## Question (operational, falsifiable)

For the same three trees {XGB-20raw, XGB-24full, XGB-4sensor} on the
same iter-40 (C_inv × L) grid:

> Q1. At the cost-optimal threshold, what is the $/fraud_caught for
> each tree at the canonical cell (C_inv=$0.50, L=$100)?
>
> Q2. Does the LLM-sensor surrogate (4sensor) catch fraud *cheaper per
> catch* despite its larger $/dec, or is the claim limited to $/dec?
>
> Q3. Does adding the 20 raw features back (24full vs 4sensor) lower
> $/fraud_caught with bootstrap CI exclusion of zero?

## Method

For each (model, C_inv, L) cell:

1. Find τ*(model, C_inv, L) by minimising cost/decision (iter-28).
2. Compute $/dec at τ* and $/fraud_caught = (c_sense·N + C_inv·(tp+fp) +
   L·fn) / tp.
3. Paired bootstrap (B=1000, seed=20260704) over test rows; compute
   Δ(24full - 20raw) and Δ(4sensor - 20raw) on $/fraud_caught.

Grid: same (C_inv, L) as iter-40; 5 × 5 = 25 cells × 3 trees = 75 cells.

## Headline findings

### Q1 — canonical cell ($/fraud_caught)

| model | $/fraud_caught | ratio vs 20raw |
|---|---|---|
| **XGB-20raw** | **$2.68** | 1.00× (baseline) |
| XGB-24full | $3.28 | 1.22× (sensor adds 22%) |
| XGB-4sensor | $14.92 | 5.57× (sensor-only is 5.6× more expensive *per catch*) |

The canonical cost-optimal threshold for raw features catches a fraud
for $2.68 in fully-loaded cost; the sensor-augmented tree catches the
same fraud for $3.28; the sensor-only surrogate pays $14.92 per catch.
The sensor-only cost catastrophic finding from iter-28 is therefore
**not a $/dec artefact** — it persists when the comparison is normalised
by successful catches.

### Q2 — sensor + raw (24full) vs raw alone

On the 25 (C_inv, L) cells:

- Mean Δ(24full − 20raw) on $/fraud_caught ranges from −$1.59 to +$35.
- **Bootstrap CI excludes zero in only 2/25 cells** — both at low C_inv
  ($0.10, $0.50) × low L ($5).
- Best-case mean Δ = −$1.59 (24full beats 20raw); worst-case mean
  Δ = +$35 (one cell has degenerate τ* collapse).

**Conclusion**: the iter-28/40 $/dec finding generalises to
$/fraud_caught: adding the sensor block to a tree that already has
the 20 raw features yields no detectable win on cost-per-catch.

### Q3 — sensor-only (4sensor) vs raw (20raw)

On the **same 25 cells**:

- Mean Δ(4sensor − 20raw) on $/fraud_caught ranges from +$1.87 to +$81.80.
- **Bootstrap CI excludes zero in 25/25 cells** — every cell carries a
  statistically detectable *cost penalty* on $/fraud_caught for the
  sensor-only tree.
- Min penalty = $1.87/caught (4sensor at C_inv=$0.10, L=$5).
- Max penalty = $81.80/caught (4sensor at C_inv=$2.50, L=$5); the CI
  excludes zero with mean +82 [+29, +231].

**Conclusion**: the sensor-only surrogate is a 1.9×–5.6× more
expensive fraud catcher *per catch*. The "sensor pays for itself"
reading is FALSE on this evidence base when the comparison is
normalised by true positives.

### Counter-intuitive / sharpest reviewer-facing claim

> For binary credit-card fraud on the held-out 10k split, the
> LLM-as-sensor surrogate (XGB-4sensor) catches fraud at
> **$14.92/caught** at the canonical cost-optimal threshold vs the
> XGB-20raw tree's **$2.68/caught** — a 5.57× per-catch penalty with
> bootstrap CI excluding zero in **25/25** (C_inv × L) cells. The
> iter-28/40 "sensor never pays" claim was previously defended on $/dec;
> this iter shows the same conclusion holds when the denominator is
> successful catches, not stream events. The sensor block's value is
> ranking lift in a narrow operating band (iter-24, iter-31); it does
> NOT translate to either cheaper fraud catching per stream event OR
> cheaper fraud catching per catch. Reframing the operational metric
> from $/dec to $/fraud_caught sharpens — rather than inverts — the
> sensor-not-scorer thesis.

## Artifacts

- `scripts/p5p8/p8_cost_per_caught.py` (~210 LoC, stdlib + numpy + pandas + xgboost + matplotlib)
- `experiments/results/p5p8/p8_cost_per_caught.tsv` (75 cells: 25 × 3 trees)
- `experiments/results/p5p8/p8_cost_per_caught_boot.tsv` (25 paired-bootstrap rows)
- `experiments/results/p5p8/p8_cost_per_caught_summary.json`
- `experiments/results/p5p8/figures/p8_cost_per_caught.{png,pdf}`

## Cross-paper coupling

- Iter-28 (cost-optimal threshold): parent — both veins share the same
  τ* search but use $/fraud_caught instead of $/dec.
- Iter-31 / iter-47 P7 (sensor as ranking lift): orthogonal — those
  veins measured contrast/ZVF in the *training* rollout; this vein
  measures catch cost on the *test* stream.
- Iter-40 (asymmetric cost frontier): SIBLING same-grid; this iter
  resolves the question "does the same grid's $/dec ordering invert
  on $/fraud_caught?" with **NO** — sensor+raw is still slightly
  worse (2/25 CI excludes zero in either direction) and sensor-only is
  decisively worse (25/25 CI excludes zero in the cost direction).

## Reproduction

`python3 scripts/p5p8/p8_cost_per_caught.py` (~2 min on 4 cores).

## Status

Validated at iter 48 (2026-07-04). Cross-checked with the iter-28
cost-optimal frame (same τ*; same trees; same held-out split).
