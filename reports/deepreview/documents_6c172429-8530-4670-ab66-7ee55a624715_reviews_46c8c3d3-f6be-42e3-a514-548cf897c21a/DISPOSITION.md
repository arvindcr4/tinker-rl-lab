# P2 ZVF — Deep Review 6c172429 disposition (2026-10-18)

Source: https://research.google/deepreview/documents/6c172429-8530-4670-ab66-7ee55a624715 (`review.md`, `comments.md`).
Recomputations: `platform_modal/scripts/zvf_review_6c17_audit.py` → `platform_hybrid/experiments/results/zvf_review_6c17_*.tsv`.
Build (tectonic): P2 42 pp, P3 30 pp, P1 51 pp; all 0 errors and 0 undefined refs/cites.

## Blocking: the variance-mitigation corpus is a synthetic dry run (held for author)

`experiments/results/variance_mitigation.tsv` (9 methods × 5 seeds × 100 steps) and everything aggregated from it (`zvf_by_library.tsv`, iter22/30/42/50/58/62 artifacts) come from `experiments/variance_mitigation_integration.py --dry-run`, added in `1d40de6da` (2026-04-19, "autoresearch rebuttal") and extended to MCGRPO/GIFT/AReaL/ES in `6e2929550`. The script docstring: "pass --dry-run to emit synthetic per-step rows whose aggregate statistics match the projections in Table variance-head2head of the paper"; its "real" path notes the launcher "only exposes simulated trajectories". The rows also contain negative `reward_mean` for a binary reward.

The abstract and §5.5 already call it a simulation projection, but these parts treat it as measured and are **untouched pending the author**: §4.1/§4.3 ("we ran on Tinker rollout workers…", "real, same-stack measurement"), §5.3 MCGRPO/GIFT/AReaL/ES drift paragraph, §5.5 bullets, §6 (dynamics, ρ₁≈0.94, 100- vs 300-step inconsistency), §8 (bootstrap CIs on 0/5, n=5 normal-approx Wilcoxon p=0.043, pre-/post-window overlap), §9 (forecasting AUC 0.93, base-rate 0.264, post-collapse windows, between-method confounding, f2/f3 collinearity, missing Fig. 11 graphic), §10.10, §10.14 (undetrended cross-correlation; slope-feature contradiction with §9.2), §10.15 AERO/mitigation paragraph, §10.16 (training-stage vs prompt-difficulty binning, step-level bootstrap), intro contributions (i)–(ii), results-opener "60 measured runs" and "0.04–0.13", Table 6/8 collapse-tally definitions, related figure rendering issues. Only pure typography was touched there (below).

## Fixed

| Review item | Change |
|---|---|
| §4.5 fused-risk index left active (withdrawn in §10.18) | Body replaced by withdrawal notice; verified all five scaling anchors have identical hand-set features (ZVF 0.95, lag-1 0.85) and label `collapse` in `zvf_iter130_risk_index.tsv`. HippoRAG paragraph deleted. Scheffer 2009 cited. |
| Taxonomy Eqs. 1–4 not mutually exclusive; tool-use p=0 "collapse" | Rewritten as the ordered cascade in `zvf_diagnostic.py::classify`, including its first rule ℓ≤0.001 (which is how the tool-use cells are labelled). |
| §7 Spearman text +1.0 vs table 0.791 | Text now 0.791 (recomputed); "frac≥0.1 isolates collapse" corrected (Qwen3-8B = 0.133); post-hoc-threshold caveat. |
| §10.2–10.3 LOO centroid leakage; reversed permutation logic | Strict LOO: 7/12 (0.583), collapse 0/1, majority 0.417, label-permutation p=0.07. Stored predictions don't reproduce. p≥0.195 now read correctly (no feature distinguishable from chance). Wrong per-phase `zvf_direction` values corrected; Nemotron label conflict (§7 vs iter33) stated. |
| zvf_discriminator 0.333 vs 0.427 | Formula in text now shows the [0,1] clip used by the code (gives 0.333). |
| §10.7, §10.9, §10.11–10.13 retracted δ_div analyses left intact | Replaced by withdrawal notes (labels kept). |
| §10.8 classifier circularity | Withdrawn: `zvf_iter38_crosslibrary.py` L163 uses mean ZVF in the truth label (the TSV header comment saying otherwise is wrong); 9/14 rows simulated; never predicts collapse/drift. |
| §10.18 iter130 seed-stability paragraph | Withdrawn (seeds are simulator noise; index withdrawn). |
| §10.15 odds-split "conservative lower bound"; r̄ confound | Not a bound (over-imputes when p<0.5); per-step application stated; added that r̄ alone separates 7 healthy (r̄>0.5) from 13 unhealthy (r̄<0.5), 9/13 simulated; "15 non-GSM8K runs" → 17. |
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
| Typography in synthetic sections | Drafting notes removed (§6.2, §8.1); [0.93,0.93]→[0.928,0.932]; θ in math; (ZVF−0.5)⁺; "Four pre-registered predictions"; "−0.480 and −0.476"; "(ii)~flattener"; "ZVF lead-time" captions renamed peak-to-collapse lag. No numbers or claims changed. |

## Refuted / not changed

- Table 21 sweep rows "don't match the formula": the script applies the odds-split per step at the step-mean reward and averages (`zvf_signed_decomposition.py`), which reproduces 0.072/0.003; now stated in the caption.
- Discriminator "should be 0.427": the implementation clips the factor to [0,1]; 0.333 is correct for the code (formula text fixed).
- "*Project guide" footnote: legitimate author-role attribution; kept.
- Table 4 MDE values: verified correct by the reviewer and by the P3 audit.

## Needs author compute

Real (non-dry-run) multi-seed runs of GRPO vs. AERO/CPPO/NGRPO/Scaf-GRPO/MC-GRPO/GIFT/AReaL/ES if §4–§10 are to keep empirical claims; per-group reward logging on runs with r̄ on both sides of 0.5 for a signed-ZVF test; the reward-shape counterfactual (Appendix B); the prospective ZVF prediction test (§5.8).
