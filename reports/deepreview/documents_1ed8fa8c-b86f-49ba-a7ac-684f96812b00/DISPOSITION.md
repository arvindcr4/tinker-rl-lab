# P3 group-size — Deep Review 1ed8fa8c disposition (2026-10-18)

> **Merge note.** A parallel pass (commit `23cda282b`) landed on main first and its
> P3 text is the base of the merged version. This branch re-applies on top of it
> only what that pass lacked: (1) the logged "SNR" is sqrt(f/(1-f)) of the
> non-degenerate-step fraction, so every "52% of sqrt(G)" claim is withdrawn and
> Table 14/Fig. 7B are relabelled; (2) CDH H3 recomputed from raw runs (paired t,
> nu=4, p=0.27, CI [-0.072, 0.194]; the 0.156/0.0958 summary does not match the
> raw runs); (3) RQS described correctly (per-model reward-trace composite, partly
> circular with r_mean); (4) classic-RL bandit actions are {0..198}, not [-99, 99].
> The table below describes this branch's original pass; where it conflicts with
> `23cda282b` (e.g. figure regeneration, FALLBACK_ROWS disclosure), the merged
> text follows `23cda282b`.

Source: https://research.google/deepreview/documents/1ed8fa8c-b86f-49ba-a7ac-684f96812b00 (`review.md`).
Recomputations: `platform_modal/scripts/group_size_review_1ed8_audit.py` → `platform_hybrid/experiments/results/group_size_review_1ed8_*.tsv`, `figures/group_size_iter27_{measured.pdf,grid_panels.png}`.
Build: `paper_P3_group_size.tex` 27 pp, 0 undefined refs/cites; `paper_P1_scaling.tex` still builds (its one undefined ref, `sec:exp-design`, predates this change).

## Resolved (2026-10-18): reconstructed G×T grid removed, replaced by measured runs

The hand-entered `FALLBACK_ROWS` grid (no training runs behind it; script comment
says the values were written to match the appendix table; `GU_SCALE` hand-calibrated)
is removed with everything built on it: grid tables/figures, T_crit and R(T) fits,
bootstrap CIs, iter23 section, reconcile section, GE_tok table, Iterative-RPO G*,
Dualformer savings, Paper2Agent joint fit, and the "budget-dependent optimum near
G=32" claim and title. Replacements, all measured:

- **Iso-compute sweep on an unsaturated task** (Colab L4, `platform_colab/p3_isocompute_groupsize.py`;
  40 runs in `experiments/results/p3_isocompute/`): Qwen2.5-0.5B, two-digit multiplication
  (base 0.198), 128 rollouts/step for every G∈{2,4,8,16,32}, 60 steps, 5 seeds whitened +
  3 seeds unwhitened. Held-out 0.598/0.612/0.599/0.597/0.580; only G4−G32 has p<0.05
  (0.032, Bonferroni 0.13). ZVF 0.826→0.353, far above the batch-mean closed form. Whitening
  effect null (paired p≥0.26).
- **Qwen3-8B/GSM8K adapters at G∈{2,4,16,32}** (Tinker, 30 steps, lr 1e-5, one seed each;
  W&B `campaign_w2_*`): pass@1 0.301–0.325, pass@32 0.905–0.915 vs base 0.266–0.306 /
  0.865–0.890 (`experiments/results/w2_groupsize_passk_wandb_export.tsv`).

New title: "Group Size in GRPO: Zero-Variance Mechanics, Measured Sweeps, and the Bridge to DPO".

## Fixed

| Review item | Change |
|---|---|
| ZVF residual sign / negative-correlation claim (Critical) | Closed form re-evaluated at per-step training pass rate on the same rollouts: residuals now +0.010…+0.109 (Jensen/heterogeneity). Text, Table 9, abstract, intro corrected. |
| TOST p<1e-9 at ν=2 (Major) | Seed-level Welch TOST: G4 p=0.0008, G8 0.025, G16 0.043 (paired df=2: 0.012/0.043/0.056). Table 14 rewritten; G2-vs-G2 CI now exactly 1. |
| — (not raised by reviewer) "SNR(adv) = 52% of √G" | Statistic is √(f/(1−f)) of the fraction of steps with ≥1 non-degenerate group, not a gradient SNR. Removed every √G/52% claim; Fig 7B replot shows f. |
| 505-task audit undefined | Defined in new §"Conditional utility audit" (cohort, Y_G/√G objective, √G as declared proxy vs linear cost). |
| Difficulty-stratified / TOST undefined | Contribution and abstract rescoped; ZVF and TOST defined at first use; Gemini bin numbers removed as untested. |
| g_GRPO ∝ 1/[GK(G−K)] | Defined S_pair, ḡ_pair; correct prefactors K(G−K)/G (unwhitened), √(K(G−K))/G (whitened). |
| argmax Y_G/G | Stated it is {2,3} for all p; interior optimum needs a constraint or sublinear proxy; p_min=min(p,1−p). |
| ±½ vs ±1, population vs Bessel σ, V_mDPO normalisers, index folding, BEI subscripts, p_x notation | All fixed in frontier section; prior-literature attribution added. |
| 2×2 super-group protocol | Explicit common-super-group sampling protocol. |
| CDH: G=128, grad-norm content, H3 equivalence, H4 direction/confound, H5 | Stack facts from `modal_samestack_ppo_grpo.py`; H3 paired t p=0.27 (old 0.156 was normal approx), inconclusive; H4 marked confounded by ZVF floor; H5 RQS defined (trace composite, partly circular). Verdict softened; VinePPO/VAPO cited. |
| CDH-echo contradictions, Dr.GRPO form, row-08/12/16 refs | Criterion made consistent; normaliser written out (no σ divisor + 1/(B·L_max)); dangling ledger refs removed; verdict = weak echo. |
| Table 15 "Null" verdicts | Relabelled inconclusive; 5–8.3 pp quantisation; G=4 gain is 1/8 single seed; G=2 regime contrast added. |
| η²=0.54 / d=−1.47 vs equivalence; "GU-ratio" | Metric is last-10 reward on 12-run sweep (Δ=0.0066); renamed to GE_tok ratio; H3 SNR slope declared uninterpretable. |
| Related work | Dr.GRPO, DAPO, RLOO, REINFORCE++, VinePPO, VAPO, Vojnovic–Yun, IsoCompute, Tan et al. (new bib entries verified on arXiv). |
| Empty §2.1/§2.4, 199-action spec | Task Suite body added; classic env is a contextual bandit with actions {0..198} (was wrongly [−99,99]); LLM-arithmetic parser stated. |
| F3, Task-4, Wave-6, missing appendices | Defined inline; appendix pointers → released `.tex` paths. |
| Library roster / GRPO hyperparameters / Table 3 "native names" | Classic libraries described as non-LLM bandit learners; P3 sweep protocol paragraph (LoRA r16, lr 1e-4, 16 prompts, T=1.0, 10 tokens, no KL/clip); Table 3 wording corrected. |
| Planned-tests markers, BH "0", family count, GPT-OSS-20B, verl, Fig 1 legend/arith node, Fig 2 N→G and font | Fixed (shared methods / TikZ). |
| Dangling "52.1%, Table 6", .json/.tsv, Appendix 6, may saturates, C=0, residual range, Table 8/9 precision, "inside the band", ±2.7%, Pillar-3 cite [21], p vs p_eff, forward-cite, IPO/GIFT bib | Fixed. |
| Matched-budget er2b panel | lr 3e-5 both arms, not rescaled; no held-out eval; confound stated. |
| Redundant measured tables | Duplicate measured table in reconcile section replaced by a pointer. Further consolidation of grid tables is pending with the grid decision. |

## Not changed (reviewer incorrect)

- **Table 4 col. 4 power values**: `statsmodels.TTestIndPower(nobs1=5, ratio=3)` reproduces the printed 0.066…0.955 and MDE 1.530 exactly (`group_size_review_1ed8_power.tsv`).
- **Section 9 / Figures 8–17**: not present in the current source (removed before this review run).

## Needs author compute

Unsaturated matched-budget multi-seed sweep (MATH-500/AIME), unwhitened/RLOO/DAPO control arms, gradient-vector logging, critic diagnostics. Not fabricated; scoped as limitations.
