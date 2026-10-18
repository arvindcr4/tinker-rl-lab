# P2 ZVF — Deep Review 6c172429 disposition (2026-10-18)

Source: https://research.google/deepreview/documents/6c172429-8530-4670-ab66-7ee55a624715 (`review.md`, `comments.md`).
Recomputations: `platform_modal/scripts/zvf_review_6c17_audit.py` → `platform_hybrid/experiments/results/zvf_review_6c17_*.tsv`.
Build (tectonic): P2 28 pp and P1 51 pp, 0 errors and 0 undefined refs/cites.

## Resolved: synthetic data removed, replaced with e1 real runs

**Finding.** `experiments/results/variance_mitigation.tsv` (9 methods × 5 seeds × 100 steps) and its aggregates were produced by `experiments/variance_mitigation_integration.py --dry-run` (commits `1d40de6da`, 2026-04-19, and `6e2929550`); the docstring states it emits "synthetic per-step rows whose aggregate statistics match the projections in Table variance-head2head of the paper", and the rows contain negative `reward_mean` for a binary reward.

**Resolution (author decision).** No number derived from that file remains in P2.
- **Replaced with real data.** §4.1 is now a matched 8-seed comparison from the e1 confirmatory campaign (`experiments/results/e1_confirmatory_wandb_export.tsv`; W&B `arvindcr4-pes-university/tinker-rl-lab`, `e1-confirmatory-*`; Qwen3-8B, TRL GRPOTrainer, G=4, 30 steps, lr 1e-6, β=0; same seeds 11,23,37,53,71,89,107,131 in every arm; 500 held-out GSM8K problems). Analysis: `platform_modal/scripts/e1_variance_mitigation_analysis.py` → `e1_variance_mitigation_summary.tsv`, `figures/e1_variance_mitigation.png`. Held-out: GRPO 0.633 [0.625,0.641]; AERO −0.001 (paired p=0.86), DAPO +0.001 (0.76), Dr.GRPO −0.002 (0.67), GSPO +0.005 (0.21); all TOST-equivalent at ε=0.02 (p ≤ 0.003). Mean ZVF 0.69–0.71 except DAPO 0 (filters zero-variance groups; 1,734 vs 480 rollouts). AERO uses 44% of GRPO's generated tokens. Limits stated: 30 steps at lr 1e-6, no base-model held-out score.
- **Updated to real rows only.** Abstract, intro contributions, results opener, §4.2–4.4, §5 failure table (35 measured rows: 30 converged, 3 collapse, 1 drift, 1 plateau), §5 recipe and AERO/RL-ZVP subsection, §10.15 signed ZVF (11 measured runs; ZVF⁻ AUC 1.0 vs raw 0.71, with the r̄ confound stated), Discussion and Conclusion, frontier-synthesis cross-references.
- **Deleted (no real counterpart).** §6 dynamics, §8 bootstrap/lead-time, §9 forecasting, §10.10 early-trace, §10.14 lagged cross-correlation, §10.16 difficulty-stratified ZVF, §10.18 eval-protocol, the iter130 risk-index stub, the withdrawn δ_div iso-yield/iso-G and classifier sections, the inference-baselines critique-pass paragraph (iter130-based), the synthetic Figures (`zvf_by_library`, `zvf_vs_failure`, `zvf_signed_decomposition`), the 0.56/0.62 collapse association, and every claim about CPPO, Scaf-GRPO, NGRPO, MC-GRPO, ES, AReaL and GIFT results (no completed real runs; now stated as not evaluated).

## Fixed

| Review item | Change |
|---|---|
| §4.5 fused-risk index left active (withdrawn in §10.18) | Section deleted (its panel mixed simulated trajectories with five scaling anchors carrying identical hand-set features, ZVF 0.95 and lag-1 0.85, all labelled `collapse`). HippoRAG paragraph deleted. |
| Taxonomy Eqs. 1–4 not mutually exclusive; tool-use p=0 "collapse" | Rewritten as the ordered cascade in `zvf_diagnostic.py::classify`, including its first rule ℓ≤0.001 (which is how the tool-use cells are labelled). |
| §7 Spearman text +1.0 vs table 0.791 | Text now 0.791 (recomputed); "frac≥0.1 isolates collapse" corrected (Qwen3-8B = 0.133); post-hoc-threshold caveat. |
| §10.2–10.3 LOO centroid leakage; reversed permutation logic | Strict LOO: 7/12 (0.583), collapse 0/1, majority 0.417, label-permutation p=0.07. Stored predictions don't reproduce. p≥0.195 now read correctly (no feature distinguishable from chance). Wrong per-phase `zvf_direction` values corrected; Nemotron label conflict (§7 vs iter33) stated. |
| zvf_discriminator 0.333 vs 0.427 | Formula in text now shows the [0,1] clip used by the code (gives 0.333). |
| §10.7, §10.9, §10.11–10.13 retracted δ_div analyses left intact | Sections deleted from P2 (withdrawn analyses). |
| §10.8 classifier circularity | Sections deleted from P2 (withdrawn analyses). |
| §10.18 iter130 seed-stability paragraph | Sections deleted from P2 (withdrawn analyses). |
| §10.15 odds-split "conservative lower bound"; r̄ confound | Not a bound (over-imputes when p<0.5); per-step application stated; recomputed on the 11 measured runs, where r̄ alone separates 7 healthy (r̄≥0.67) from 4 unhealthy (r̄≤0.265). |
| Appendix A.4 retracted 0.28 plug-in | Replaced by h₈(p̄)=0.054, 2.9×, beta-binomial 0.170 (matches `p2_delta_artifact.json`); "opposite-sign residual" removed (0.5B residual is positive). K≡G note. |
| Eq. 11 PCD variance notation | Biased group variance made explicit. |
| §10.19 KL caveat scope; 0.845; "Section the companion paper" | Scoped to reward-level β folded before standardisation with ZVF on task reward; 0.838; reference fixed. Old Δ_G ∈ [−0.23,−0.13] corrected to the same-rollout +0.01…+0.11 (P3 audit). |
| Appendix B empty 12-row table; ε 1e-4 vs 1e-6 | Table replaced by design paragraph; ε=10⁻⁶ (script default); min non-zero variance 2.5×10⁻³ at G=16 noted. |
| Appendix C "rewards scaled to [0,1]" | Script does no scaling; text now says inputs must be pre-scaled with fixed task bounds. |
| Ref [18] arXiv 2510.20015 (an IOL physics paper) | Entry corrected to Mukherjee et al., arXiv 2505.11711; P2 now cites Dr. GRPO for the normalisation variants. |
| Related work missing He et al. (ACR/AVSPO), Nie et al. | Added, plus DAPO, GRESO, AERO, VCRL, online difficulty filtering, RL-ZVP. |
| Missing bib: libraries, Scheffer; [16] venue; [19] "J. Jang"; [23] GIFT et al. | SB3, CleanRL, Tianshou, d3rlpy, PufferLib, Scheffer added (verified); RL-ZVP as ICLR 2026; Jyh-Shing Roger Jang; GIFT single author. NGRPO/Scaf-GRPO/MC-GRPO/AReaL entries already exist. |
| Empty §2.1/§2.4, F3, verl, BH "0", ‡, planned-tests markers, family count, Fig 2 N→G | Fixed in shared methods/TikZ (P3 pass); this pass added library cites, renamed "read carefully" heading, d3rlpy note on Table 3, "Last-10" definition, σ_r+ε in Fig. 2, HumanEval wording aligned with Fig. 1. |
| Absorbed-satellite repository note in abstract | Removed. |
| "appendix" self-references; #36320; duplicate empty 5.6; placeholder paragraph | Fixed (placeholder → "not run"). |
| Table 10 never cited | Citing paragraph added; notes the advantage-variance proxy is near-mechanical. |
| Typography in synthetic sections | Superseded: those sections were deleted when the synthetic data was removed. |

## Refuted / not changed

- Table 21 sweep rows "don't match the formula": the script applies the odds-split per step at the step-mean reward and averages (`zvf_signed_decomposition.py`), which reproduces 0.072/0.003; now stated in the caption.
- Discriminator "should be 0.427": the implementation clips the factor to [0,1]; 0.333 is correct for the code (formula text fixed).
- "*Project guide" footnote: legitimate author-role attribution; kept.
- Table 4 MDE values: verified correct by the reviewer and by the P3 audit.

## Needs author compute

Longer-horizon (more steps, higher learning rate) and harder-task versions of the e1 comparison, plus a base-model held-out score, to bound effects the 30-step null cannot see; real runs of the other variance-mitigation methods if they are to be discussed; per-group reward logging on runs with comparable mean reward for a signed-ZVF test; the reward-shape counterfactual (Appendix B); the prospective ZVF prediction test (§5.8).
