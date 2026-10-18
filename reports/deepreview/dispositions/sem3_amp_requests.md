# Sem3 cluster: requests for files owned by the Amp session / coordinator

No edits were made to `platform_tinker/reports/combined/combined_capstone*.tex`,
`platform_tinker/reports/esa_phase1/*.tex`, or `platform_hybrid/paper/references.bib`.

## 1. Duplicate-citation fix: no request needed

The sem-3 capstone duplicate pairs (`rafailov2023dpo`/`rafailov2024direct`,
`schulman2017ppo`/`schulman2017proximal`, `liu2025understanding`/`liu2025r1zerocritical`)
are real in the source, not stale `.aux`/`.bbl` artifacts. `citation_audit.py` computes
`dup_titles` over cited entries in the `.tex`/`.bib`, not over the `.bbl`. The report
looked flaky because the HEAD version of `platform_hybrid/sem 3 work/source/capstone_final_report.tex`
had broken `\input` paths: the literature-survey sections (which cite `rafailov2023dpo`) were
not resolved, so the pairs appeared or disappeared depending on whether those inputs were
found. The fix (unified keys in the capstone `.tex`, plus removing the 23 keys that
`final_missing_references.bib` duplicated from `capstone-literature-survey/references.bib`)
was already uncommitted in the main checkout. Branch commit 8e79a54e2 ports it byte-identically,
so merging it cannot conflict. The audit is clean for both capstone roots, `base_final.tex`, and `paper.tex`
(paper.tex `trl2020` was changed to `vonwerra2022trl` in `ethics_statement.tex`, the same change as in main).

## 2. `combined_capstone.tex` has the same unsourced content that was removed from `base_final.tex`

`combined_capstone.tex` contains the Group 6 report chapters. The following items are
unsourced or invalid, as established in `reports/deepreview/dispositions/Sem3_Group6_Original_Report.jsonl`.
The replacement text is the corresponding block of `platform_tinker/reports/combined/base_final.tex` on this branch.

| Item | combined_capstone.tex (approx. line) | Action |
|---|---|---|
| Fig. "Illustrative training curves comparing PPO, DPO, and GRPO" | ~4846 | remove (synthetic; finding 1046) |
| "Total experimental runs: 19 conditions ... = 57 runs" | ~4942 | reword as proposed and not executed (1030) |
| Table "GRPO Experiment Results Summary", GSM8K/MATH pgfplots curves, HumanEval "Qwen3-3.8B" 32->40% | ~5000-5100 | remove; replace with `tab:measured_runs` block from base_final.tex (1023-1028) |
| "Statistical Analysis of Experimental Results" (bootstrap, z-tests, Spearman, Chow, Mann-Whitney, Bonferroni) | ~5150-5270 | remove (1031-1033, 1048) |
| 80-variant catalog (31/18/8/23, 39%, kappa 0.85, pie chart, 52/28/20% architectures) | search `methodN` | remove (1039) |
| Table 3.1 unattributed rows, OSWorld "Mobile-Agent-v3 73.3" (AndroidWorld), "2-29x"/"4-29x" multipliers | search `73.3`, `29$\times$` | replace with verified Table 3.1 from base_final.tex (1019-1022, 1041, 1042) |

## 3. Possible unresolved inconsistency in the capstone (not edited)

The capstone sentence "paired per-prompt p=0.539 (evaluated on a random 50-problem subset)"
conflicts with `platform_hybrid/experiments/p0_gsm8k_paired.md`, which reports the 0.539
paired t-test on n = 200 paired prompts (the same 200 prompts as the per-seed files
`platform_tinker/reports/final/gsm8k_heldout_seed*.json`). REVIEW43_ERRATUM.md already lists
this as unresolved. The capstone is outside this cluster, so it is reported here for the owner.
