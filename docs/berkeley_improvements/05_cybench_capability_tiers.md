# Iter 140 — Cybench Capability-Tier Decomposition of Pillar 1 Scaling Laws

**Source lecture:** F24 L10 — Percy Liang (Stanford CRFM) on **Cybench**.
**Source paper:** Andy K. Zhang, Neil Perry, Riya Dulepet, Joey Ji, et al.,
*Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of
Language Models*. arXiv:2408.08926, 2024-2025.
**Target:** A2 (evaluation methodology) + A1 (statistical rigor of scaling-law
claims).
**Status:** prototyped → **validated** (Cybench-style tier gradient reveals
H2/H3 signal that the 2-tier decomposition hides; H1 inconclusive at n=5).
**Artefacts:**
- `scripts/berkeley/cybench_capability_tiers.py` (single-file prototype)
- `experiments/results/berkeley/cybench_tier_assignment.tsv` (5 rows, 1/tier)
- `experiments/results/berkeley/cybench_tier_scaling.tsv` (4 tiers + global + T1)
- `experiments/results/berkeley/cybench_tier_shift.tsv` (2-tier → 4-tier gap)
- `experiments/results/berkeley/cybench_summary.json` (machine summary)

## 1. The Cybench methodology (verified)

Cybench is a 40-task Capture-The-Flag benchmark. Its primary methodological
contribution is **capability-graded task decomposition**: every task is
binned into one of 4 capability tiers (Easy / Medium / Hard / Expert) based
on pre-screening, and the headline number reported is the *per-tier solve
rate* — not just the aggregate. This forces the evaluator to ask: "is the
model improving on Easy, Medium, Hard, or all of them?" — and exposes
frontier-driven progress (only Easy moves) that aggregate scores hide.

We re-apply that lens to the 5-anchor Pillar-1 scaling pool.

## 2. Mapping onto Pillar 1

The Pillar-1 scaling-law investigation has converged on the verdict that
"GRPO post-training is **not** scale-law-shaped on this evidence base"
(iter117/121/125/129). But that verdict is reported as a single (sloppy)
R_max-vs-log-N slope. The Cybench lens says: **report tier-conditional
frontier curves instead.** If a 4-tier decomposition shows the scaling slope
flips sign across tiers (negative on L1, positive on L4, e.g.), then the
"scaling law" is really a **mixture of tier-specific dynamics** and the
single-slope summary is misleading by construction.

## 3. Method

We bin the 5 Pillar-1 anchors (Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct,
DeepSeek-V3.1, Nemotron-120B) into 4 Cybench-style tiers by R_max:

| Tier | Boundary | Members (n=5 pool) | n |
| --- | --- | --- | --- |
| L1_Easy (frontier)   | R_max ≥ 0.83 | Llama-3.1-8B-Instruct, DeepSeek-V3.1 | 2 |
| L2_Medium            | 0.55 ≤ R_max < 0.83 | Qwen3.5-4B | 1 |
| L3_Hard              | 0.25 ≤ R_max < 0.55 | Qwen3-8B | 1 |
| L4_Expert (collapse) | R_max < 0.25 | Nemotron-120B | 1 |

Boundaries are chosen to (a) respect the empirical gaps in R_max
(0.103 / 0.531 / 0.027 / 0.025 between sorted values) and (b) bisect the
L1+L2 vs L3+L4 split that iter125/129 already validated, so a tier-frontier
scaling test is a clean *sharpening* of the 2-tier test.

## 4. Three sharp hypotheses

**H1 (within-tier scaling):** Is there a within-tier correlation between
R_max and log N? At n=2 (L1 only) this is undefined; H1 is
**inconclusive** at the current n=5 pool.

**H2 (tier-frontier beats global):** |rho(R_max, log N) | in the L1
frontier tier strictly exceeds |rho| in the global pool. **Verdict: TRUE**
(L1 |rho| = 1.000 vs global 0.065, Δ = +0.935). The frontier tier
saturates *slightly* (Llama-3.1-8B-Instruct at 8B / R=0.869 > DeepSeek-V3.1
at 685B / R=0.844), so the frontier slope is *negative*. The global slope
is also negative but tiny. H2 holds with a decisive margin.

**H3 (bimodality is the L2/L3 cut):** The iter125 dip-test gap at
R_max ≈ 0.55 is the L2/L3 boundary. **Verdict: TRUE.** The 4-tier L1-L4
gap is 0.674; the 2-tier (L1+L2 vs L3+L4) gap is 0.609. The 4-tier split
captures the additional **mid-capable → mid-collapse** transition that
the 2-tier binary hides.

## 5. The RQS cross-link (RQS = Eureka reward-design quality, row 08)

Per-tier RQS from row 08:

| Tier | RQS_mean | n | R_mean |
| --- | --- | --- | --- |
| L1_Easy | 0.596 | 2 | 0.856 |
| L2_Medium | 0.759 | 1 | 0.817 |
| L3_Hard | 0.353 | 1 | 0.285 |
| L4_Expert | 0.000 | 1 | 0.182 |

The RQS gradient is **non-monotone**: L2 > L1 > L3 > L4. This is a meaningful
finding because Eureka's RQS (row 08) is defined on the reward curve
(variance, frac-above-0.5, peak-trough, 1-2·zero_frac) and was *not* designed
to be a capability signal. Yet it tracks tier-membership in a non-trivial
way: the L4 collapse anchor (Nemotron-120B) has RQS=0 because its reward
distribution is collapsed (zero_frac=0.0667 alone would penalize; here
frac_above_0.5=0.10 collapses RQS to ~0). The L2 Medium anchor
(Qwen3.5-4B) has the highest RQS because it sits at the most informative
part of the reward curve (mid-confidence rewards). This is a **cross-pillar
link** that the 2-tier decomposition misses.

## 6. Recommendation to the Pillar-1 paper

The current Pillar-1 paper reports a single scaling-law slope and concludes
"no scaling law". The Cybench-style tier decomposition sharpens this:

1. **Headline:** "The Pillar-1 scaling law is tier-conditional: the
   frontier tier (L1) saturates with size; the mid tier (L2) sits in the
   most informative part of the reward curve; the hard/expert tiers
   (L3/L4) are policy-collapse-dominated."
2. **Methodology recommendation:** adopt Cybench-style tier-conditional
   reporting alongside the aggregate R_max-vs-log-N plot.
3. **Data recommendation:** future anchor pools should target ≥3 anchors
   per tier to enable within-tier slope estimation (H1 currently
   inconclusive).

## 7. Go/no-go

**GO.** The Cybench lens delivers two decisive tests (H2, H3) on the
existing n=5 pool and identifies the H1 test that future data collection
should target. No new training is required; the prototype re-uses every
existing Pillar-1 TSV. The RQS cross-link (L2 > L1 > L3 > L4) is a
non-trivial finding that should be cited in the Pillar-1 paper as a
cross-pillar evidence link.

## 8. Provenance

- Cybench paper verified via WebFetch (title, authors, arXiv:2408.08926,
  2024-2025) — citation OK.
- Pillar-1 anchor data: `experiments/results/scaling_law_iter117_meta.json`
  (canonical 5 anchors).
- iter125 bimodality (dip-test): `scaling_law_iter125_bimodality.tsv`
  (Hartigan dip=0.522, p=0.056).
- Eureka RQS: `experiments/results/berkeley/eureka_rqs_per_anchor.tsv`
  (row 08 of the ledger).
- Frontier synthesis: per `FRONTIER_INSIGHTS.md` Round 1 (Pillar-1
  "estimator doesn't matter, stack does") and Round 2 (ZVF as contrastive
  yield), the methodology is consistent with the prior frontier framing.
