# Row 24 (B-F25) — Multiplicity & Winner's-Curse Audit

**Course vein:** Berkeley F25 "Agentic AI" L8, Sida Wang, *"Adding Error Bars to
Evals."* Canonical reference: **Evan Miller, "Adding Error Bars to Evals: A
Statistical Approach to Language Model Evaluations," arXiv:2411.00640 (2024)**
(citation verified 2026-07-04 via arxiv.org). Multiplicity procedures: **Holm
(1979, Scand. J. Stat. 6:65–70)**; **Benjamini & Hochberg (1995, JRSS-B 57:289–300,
FDR)**.

**Target:** A1 (statistical rigor of the benchmark) — the *family-level* gap left
open by rows 20–23.

## The untried gap

Rows 20–23 each put a CI / power bound / noise-robustness envelope on an
**individual** headline number. But TinkerRL-Bench reports **many simultaneous
headline claims** across 4 papers, and it highlights **selected extrema** — the
"+24pp group-size swing" is a *max over a sweep*; the "best G" is an *argmax*.
Two classical failure modes that a per-number CI cannot see:

1. **Multiplicity** — with `K` simultaneous looks at α=0.05, the chance of ≥1
   false positive under the global null is `1−(1−α)^K`, not 0.05.
2. **Winner's curse** — the reported value of a *selected* extremum is biased
   upward: `E[max_j(θ_j+ε_j)] > max_j θ_j`.

Neither had been quantified here. This row closes both on **real in-repo data**.

## Data (all real, in-repo)

- `experiments/results/berkeley/headline_ci_clustering.tsv` (row-20 output):
  point + **seed-clustered** CI → SE for each headline.
- `experiments/results/group_size_effect.tsv`: per-G heldout accuracy + SE
  (G∈{2,4,8,16}, 3 seeds) and the reward~log10(G) slope p.
- `experiments/results/group_size_g4_vs_g32_broader_scale.tsv`: G32-vs-G4
  accuracy gap + CI at 4 compute budgets (the "+24pp swing").

## Hypotheses & results — 4/4 decisive

### H1 — Family-wise error rate (`mwc_h1_fwer.tsv`)
The reported family is **16 simultaneous looks** (6 effect claims + 6 pairwise
group-size comparisons + 4 broader-scale budget comparisons). Under the global
null at per-look α=0.05:

| family | K | FWER (uncorrected) |
|---|---|---|
| headline effect claims | 6 | 0.265 |
| **total reported family** | **16** | **0.560** |

→ if every headline were truly null, there is a **56% chance of ≥1 "significant"
result by chance alone**. A per-number CI (rows 20–23) is blind to this.

### H2 — Bonferroni / Holm / BH-FDR on the family (`mwc_h2_multiplicity.tsv`)
Raw two-sided p recomputed from the **clustered** (row-20) point/SE, then
corrected across the K=6 family:

| claim | effect | raw p | Bonferroni | BH-FDR q | verdict |
|---|---|---|---|---|---|
| P2 ZVF decay G2→G16 | 0.207 | 2e-166 | 1e-165 | 1e-165 | **survives** |
| P3 +24pp swing (G32 vs G4, 64M) | 0.240 | 5e-15 | 3e-14 | 1e-14 | **survives** |
| P3 reward↑ G2→G16 | 0.033 | 8e-7 | 5e-6 | 2e-6 | **survives** |
| P3 reward~log10(G) slope | — | 0.076 | 0.46 | 0.12 | **fragile** |
| P1 GRPO≠PPO paired | 0.061 | 0.13 | 0.76 | 0.15 | *equivalence — immune* |
| P4 tool-use dense>sparse | 0.074 | 0.19 | 1.00 | 0.19 | **fragile** |

The three **strong physical effects survive even Bonferroni** (the most
conservative correction). The two **marginal** claims — the reward/log-G slope
(already p=0.076) and the dense>sparse tool-use gap (p=0.19) — do **not** survive
and must be labelled **exploratory**, not confirmatory. Raw-significant count 3 →
Bonferroni 3 → BH 3 (no change): the benchmark's confirmatory backbone is
multiplicity-robust; only two soft claims were riding on uncorrected p.

### H3 — Winner's curse on the "best G" (`mwc_h3_winners_curse.tsv`)
Selecting `argmax_G heldout_acc` (G=8, 0.990) over 4 close, noisy arms is
upward-biased. A deterministic 4-point Gauss grid (256 joint nodes, seedless)
gives selection bias **0.052 pp** → debiased best-G **0.9895**. Worst case, if all
4 arms were *truly tied*: `E[max of 4]·SE_mean = 1.029·0.0037 = 0.39 pp` —
**negligible**. So the benchmark's selected best-G is **trustworthy**, *because
the per-G SEs are tiny* (low-variance 3-seed runs).

**But the same argmax on the 2-seed tool-use pillar** (SE ≈ 0.039, ~10× larger)
would inflate a selected-best by `0.564·0.039 = 2.2 pp` under the null —
**comparable to the 7.4 pp effect it selects**. Winner's curse bites exactly
where SNR is low, which is precisely the fragile tool-use pillar flagged in H2.
The two diagnostics converge on the same weak spot.

### H4 — Winner's curse on the "+24pp swing" (`mwc_h4_swing_selection.tsv`)
The swing is `max_budget |acc(G32)−acc(G4)|`, selected at 64M. The 4 budget arms
(0.01, 0.11, 0.21, 0.24) are **well-separated** (monotone growth), so the
selection bias is only **0.004** → debiased swing **0.236**. The +24pp magnitude
is **robust to selection** — unlike the best-G *point*, the swing *gap* is a
large, cleanly-identified effect.

## The asymmetry (headline conceptual result)

Multiplicity correction only raises the bar to **reject** a null. It therefore
**cannot touch the benchmark's flagship equivalence/null claims** (GRPO≈PPO,
p=0.75 heldout; row-20 clustered paired-Δ straddles 0). Those are
**multiplicity-immune**: their credibility can only *rise* as we control false
discovery among the *positive* family. So the net effect of this audit is
asymmetric and favourable — it **strengthens** the benchmark's headline nulls
while **demoting** exactly two marginal positive claims to exploratory. This is
the honest, defensible posture for a NeurIPS-caliber benchmark: report every
positive discovery with a family-corrected q-value and every selected extremum
with a selection-adjusted value.

## Cross-pillar link

The same per-task/per-prompt dispersion `σ²_p` that governs row-23 `pass^k`
reliability and Pillar-2 ZVF collapse also governs the winner's-curse magnitude
(tight arms → small bias). Rows 20/21/22 **widen each CI** (clustering, power,
verifier noise); this row **controls the family and the selection** on top of
those widened CIs. Together they form a complete "Adding Error Bars" stack:
per-number CI → power → noise-robustness → **family + selection**.

## Paper-facing recommendation (not yet patched)

Add a short "Multiple comparisons & selection" paragraph to the eval-protocol
section: (i) report BH-FDR q-values for the family of positive claims; (ii) mark
the reward/log-G slope and dense>sparse tool-use gap as **exploratory**; (iii)
report the best-G accuracy with its selection-adjusted value; (iv) note the
+24pp swing is selection-robust. Apply only after a maintainer confirms the
family definition (which claims count as one family).

## Artifacts
- `scripts/berkeley/multiplicity_winners_curse.py`
- `experiments/results/berkeley/mwc_h1_fwer.tsv`
- `experiments/results/berkeley/mwc_h2_multiplicity.tsv`
- `experiments/results/berkeley/mwc_h3_winners_curse.tsv`
- `experiments/results/berkeley/mwc_h4_swing_selection.tsv`
- `experiments/results/berkeley/mwc_summary.json`
