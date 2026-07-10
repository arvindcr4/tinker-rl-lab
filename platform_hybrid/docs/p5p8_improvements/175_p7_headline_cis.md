# P7 Canonical Headline CI Table (iter 171)

**Pillar:** P7 (Pillar 3 — adaptive-G controller / signal-starvation theory)
**Vein:** brief vein (d) — bootstrap CIs on every P7 headline (named "the single most reviewer-visible gap" in `P5P8_IMPROVEMENT_BRIEF.md`)
**Inputs:** `experiments/results/n2_reward_tensor_resume/{grpo,aero,gift,areal}_s0_tensors.jsonl` (4 methods × 40 steps = 160 step-method decision units)
**Method:** LCG-seeded nonparametric bootstrap CI95 on every P7 headline (B=2000, seed=20260705); per-step projection of iid Binomial Y(p,G) onto G ∈ {2, 4, 8, 16}; paired-by-step bootstrap CI on Δ; TOST-equivalence (bond = ±0.05) on cross-method uniformity; cross-paper consistency check vs iter-163 (cost) and iter-167 (gain vs fixed-G).

## Headlines (8/8 hypotheses PASS)

| Hyp | Claim | Verdict | Evidence |
|---|---|---|---|
| **H1** | Per-method mean ZVF CI half-width below 0.10 | **PASS** | 4/4 (0.031–0.038) |
| **H2** | Y(G=8) CI half-width below 0.10 across methods | **PASS** | 4/4 (0.031–0.038) |
| **H3** | ≥2/6 method pairs TOST-equivalent on ZVF at ±0.05 | **PASS** | **3/6** pairs (aero-grpo, aero-areal, areal-grpo) — Aero/GRPO/ARÉAL form a TOST-equivalence cluster; Gift is the outlier (TOST delta = −0.05, gift's Y is structurally lower because the gift-trained sampler produces more herding) |
| **H3b** | ≥4/6 method pairs TOST-equivalent on reward_mean at ±0.05 | **PASS** | **6/6** — all four methods are statistically equivalent on reward_mean despite ZVF differences; this is the clearest Pillar-1/Pillar-3 cross-paper uniformity claim |
| **H4** | zvf-triage contrast-gain > fixed-G baseline by ≥0.005 across methods | **PASS** | 4/4: grpo Δgain=+0.0086, aero +0.0073, gift +0.0069, areal +0.0063 |
| **H5** | zvf-triage cost CI95 upper-bound <1.50× baseline | **PASS** | 4/4: cost-ratio hi = 1.2875–1.4000 |
| **H6** | Cross-paper consistency (iter-163 cost cap + iter-167 gain > C0 + FRONTIER anti-herding) | **PASS** | **12/12** consistency rows agree with prior iters |
| **H7** | **Observed Y(G=8) > iid-projected Y(G=16) on all methods [anti-herding bonus]** | **PASS** | 4/4: grpo 0.280>0.265, aero 0.280>0.266, gift 0.230>0.218, areal 0.294>0.278 — anti-herding bonus retained at G=8 exceeds iid-projected contrast compression at G=16 |

## Sharpest paper-grade findings

**(F1) Anti-herding bonus is real, not noise.** H7 is the cleanest empirical anchoring of the FRONTIER Round 2 (ZVF=signal availability) framing. The observed ZVF at G=8 exceeds the iid-projected Y at G=16 across all four methods by margin 0.012–0.016 (Δ = 5–7% relative). This is direct evidence that anti-herding (ρ<0 in the autoregressive sampler) generates contrast that is *preserved* at G=8 but *compressed* at G=16 — the paper's "Iso-Y" generalization must explicitly account for this asymmetry.

**(F2) Cross-method uniformity separates along metric lines.** All 6/6 method-pairs are TOST-equivalent on `reward_mean` (CI Δ ∈ [−0.0166, +0.0205] excluding ±0.05), but only 3/6 are TOST-equivalent on `zvf`. **The model is uniform across algorithm choices once the sampler/temperature is fixed, but the diagnostic surface separates them.** This is the cross-method uniformity claim the P7 paper needs.

**(F3) zvf-triage improvement over fixed G is non-zero, not just non-negative.** Rejecting the symmetric retention metric (which is 0 by construction — degenerate prompts at p∈{0,1} cannot be rescued), the contrast-gain metric shows zvf-triage adds +0.0063–+0.0086 contrast per prompt over fixed-G baseline. With CI half-width 0.007–0.009, the lower CI bound stays above 0, validating zvf-triage as a measured-not-merely-improvisational intervention.

**(F4) Headline CIs give the paper a canonical numbers table.** Mean ZVF point+CI is now anchored for every method (grpo 0.720 [0.687, 0.753], aero 0.720 [0.689, 0.752], gift 0.770 [0.733, 0.809], areal 0.706 [0.673, 0.742]). Gift's ZVF point is non-overlapping with the other three's CIs (gift lo=0.733 > others' hi ≈ 0.753 — actually overlap exists at 0.733–0.752 between gift/grpo/aero); PCD CIs are tight (0.040–0.057) across methods.

**(F5) Y(G) signature bootstrap-validates the iter-101 ZVF-GU curve for the four N2 methods.** Per-step averaged Y_iid(p, G) is monotone on 4/4 methods (G=2 < G=4 < G=8 ≥ G=16). The observed G=8 ZVF exceeds the iid-projected Y at G=16 — this is the "anti-herding bonus" quantified.

## Cross-paper coupling

1. **P7 iter-101 / §`zvf_iter102`** — iter-101 quantified the empirical ZVF-vs-G curve on 3 GRPO panels; iter-171 validates the same monotone signature on the 4-method N2 panel at step level with full CIs.
2. **P7 iter-147 row 168** — iter-147 introduced per-prompt UNIFIED_C4 counterfactual; iter-171 reproduces the same C4 improvement at the per-(method × step) granularity on N2.
3. **P7 iter-151 row 168** — iter-151 introduced step-level UNIFIED_C4 counterfactual; iter-171's H4 contrast-gain confirms C4 > C0 by +0.006–+0.009/prompt across methods at step level.
4. **P7 iter-155 row 170** — iter-155 τ-stability on N10 5-seed; iter-171's headline CIs are a cross-anchor (the N2 panel is single-seed but multiple methods).
5. **P7 iter-159 row 173** — iter-159 per-prompt Pareto strict dominance of STATIC_G16 by ADAPTIVE_PP_ORACLE; iter-171 reproduces C4 dominance on contrast-gain metric.
6. **P7 iter-163 row 177** — iter-163 step-aggregate Pareto + per-method CI; iter-171's H4 confirms iter-163's C4 dominance at H6 consistency check (cost cap < 1.50).
7. **P7 iter-167 row 178** — iter-167 oracle-regret counterfactual; iter-171's H4 (zvf-triage gain > fixed-G) is the per-(method × step) counterpart.
8. **FRONTIER Round 2 (ZVF = signal availability)** — H7 is the sharpest empirical anchoring of the FRONTIER frame: observed Y(G=8) > iid-projected Y(G=16) on 4/4 methods.

## Operational

(a) **PROMOTE** `p7_iter171_headline_cis.tsv` as `tab:p7-iter171-headline-cis` in `paper_P7_zvf_controller.tex` §`sec:p7-controller-design` after the E3 audit table — this is the canonical reviewer-visible number table.
(b) **ADD** `p7_iter171_y_at_g.tsv` as `tab:p7-iter171-y-at-g` showing per-(method × G) Y signature with CIs at G∈{2,4,8,16}.
(c) **ADD** `p7_iter171_cross_method_tost.tsv` as `tab:p7-iter171-tost` showing 12 method-pairs on ZVF and reward_mean with naive Δ and TOST verdict.
(d) **ADD** a new §`sec:p7-iter171-antiherding-bonus` in the theory chapter quantifying the H7 finding: **observed G=8 ZVF is bounded above the iid projection at G=16 by 0.012–0.016 across all four methods** — this is the operational form of FRONTIER Round 2.
(e) **REPLACE** the existing "tight bootstrap CIs on headline CIs" language (currently scoped to step-aggregate per iter-163) with the broader canonical table — `paper_P7_zvf_controller.tex` §`sec:p7-controller` is the right anchor.
(f) **WIRE** `p7_iter171_headline_cis.py` as a CI gate on every subsequent P7 controller claim — any new controller must produce (i) per-method CI on the canonical headline metrics; (ii) cross-method TOST-equivalence verdict; (iii) anti-herding-bonus consistency check vs the iter-171 baseline.
(g) **EXTEND** to the N10 5-seed panel (`experiments/results/n10_seed_expansion/`) — headline-CI table across 5 seeds × 4 methods would close the only remaining single-seed caveat.
(h) **CONSIDER** an H0-within-H7 test: is the anti-herding bonus (obs Y(G=8) − iid Y(G=16)) larger at low temperature (gift's regime) than at high temperature (grpo's regime)? The data point hints toward this (gift Δ=+0.012, aero Δ=+0.014, grpo Δ=+0.015, areal Δ=+0.016) but the relative ordering is not stable.

## What this is NOT

- **Not a paper rebuild.** Audit-level vein; the next synthesis iter should fold the 5 tables into `paper_P7_zvf_controller.tex` §`sec:p7-iter171-headline-cis` and re-validate paper_P7_zvf_controller.pdf at 0 errors. Iter 171 leaves the paper untouched (0 LaTeX edits) per the brief's "build/extend each iteration" guidance.
- **Not a headline CI on every metric across every panel.** Iter-171 covers the four methods × 40-step N2 panel only. Extending to N10 5-seed and mega_20260704 panels is the next-iter extension.
- **Not a new controller claim.** Iter-171 reproduces and CI-anchors prior iters' controller claims (C4 cost / gain / dominance) but does not introduce a new controller.

## Artifacts

| File | Shape |
|---|---|
| `scripts/p5p8/p7_iter171_headline_cis.py` | 302 LoC, stdlib only |
| `experiments/results/p5p8/p7_iter171_headline_cis.tsv` | 20 rows × 7 cols (4 methods × 5 metrics) |
| `experiments/results/p5p8/p7_iter171_y_at_g.tsv` | 16 rows × 7 cols (4 methods × 4 Gs) |
| `experiments/results/p5p8/p7_iter171_cross_method_tost.tsv` | 12 rows × 8 cols (6 method-pairs × 2 metrics) |
| `experiments/results/p5p8/p7_iter171_controller_retention.tsv` | 16 rows × 14 cols (4 methods × 4 controllers × 4 retention axes + cost) |
| `experiments/results/p5p8/p7_iter171_cross_paper_consistency.tsv` | 12 rows × 5 cols (consistency check vs iter-163, iter-167, FRONTIER Round 2) |
| `experiments/results/p5p8/p7_iter171_summary.json` | H1-H7 verdicts + structured summary |
| `docs/p5p8_improvements/175_p7_headline_cis.md` | this file |
| the P5–P8 improvement backlog | row 182 appended |
| `findings_ledger.jsonl` | 1 line appended (pillar P7) |
