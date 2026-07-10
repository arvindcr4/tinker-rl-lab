# Hostile Reviewer Rectification Log

Source: anonymous hostile reviewer memo recommending reject, confidence high.
Scope: `paper/main.tex`, `paper/main_anon.tex`, `reports/final/capstone_final_report.tex`,
`submission/contents/*`, `blind_review/*`.

| # | Reviewer concern | Status | Rectification |
|---:|---|---|---|
| 1 | 54-page PDF far over NeurIPS 9-page main-text limit. | **Partial (format work required)** | No content edit shrinks 54 pages to 9; remaining plan is a dedicated NeurIPS submission branch that collapses §6--§12 content into appendix-only inputs and compiles with `\usepackage[final,nonatbib]{neurips_2026}` in submission mode. Flagged as a known format gap in the submission README. |
| 2 | Double-blind: bundle contains non-anon `paper.pdf`/`report.pdf` with names. | **Documented, rebuild pending** | `blind_review/main_anon.pdf` + `blind_review/tinker-rl-lab-anon.tar.gz` are the anonymized artifacts. Submission bundle currently ships both named and anon PDFs; reviewer-facing bundle should be rebuilt from `main_anon.tex` + anon report only. Action: regenerate `submission/contents/` with anon-only PDFs and strip `paper.pdf`, `report.pdf`, `grpo_agentic_llm_paper.pdf` from the reviewer zip. |
| 3 | Runner is not canonical GRPO. | **Fixed** | Title retitled to *"When Does GRPO-Style Training Have a Learning Signal? A Claim-to-Evidence Audit of a Critic-Free Group-Relative Runner"* in both `main.tex` and `main_anon.tex`. Scope boundary paragraph added at the top of "Claims We Explicitly Do Not Make" repeating the prompt-token-loss diagnostic ($61.6$--$89.6\%$ of loss from prompt tokens) and the four canonical-GRPO omissions (ratio/clip, frozen ref, single optimizer step, completion mask). |
| 4 | Only clean held-out result is null ($82.0\!\to\!83.3$, $p{=}0.26$). | **Fixed (framing)** | Scope-boundary paragraph states explicitly that "$82.0\%\!\to\!83.3\%$ ($p=0.26$) is the only clean capability result" and that the paper is an engineering diagnostic, not a benchmark. Abstract, intro thesis paragraph, §Results §GSM8K, and capstone report all already carry this framing. |
| 5 | MATH-500 / HumanEval pools loaded from `test` splits → contamination. | **Fixed (consolidated)** | Scope-boundary paragraph repeats the "reward-environment probes, not generalization tests" language explicitly. Abstract and introduction already carry the same disclaimer. |
| 6 | Reward environments are proxy/broken (sandbox removes builtins; tool reward scores JSON only; MATH mixes boxed-format with answer). | **Fixed (consolidated)** | Scope-boundary paragraph restates the three broken/proxy environment properties. Abstract and introduction already say so. |
| 7 | Under-powered, confounded, 30-step, single-seed, backend/sampler/optimizer/LoRA/checkpoint-confounded. | **Fixed (consolidated)** | Scope-boundary paragraph enumerates the confounds; Limitations §Algorithm-Label-Effects-Not-Identified and §Single-Seed-Tinker-Experiments already carry this. Paper does not claim PPO vs GRPO; it claims stack non-identifiability. |
| 8 | BH adjustment: "19 of 20 survive" treats per-step autocorrelated rewards as independent. | **Fixed** | `main.tex` §Statistical Methodology now says "19 of 20 tests pass BH at FDR $=0.05$, but we caution against reading this as 19 independently replicated findings: several tests use per-step trajectory rewards that are autocorrelated ... BH controls FDR \emph{within} this pre-registered set of 20 tests on trace-level observations; it does not upgrade trace-level descriptive effect sizes into inferential evidence about distinct seeds, runs, or stacks." BH table caption mirrors the caveat in both paper variants. |
| 9 | Artifact inconsistency: `heldout_gsm8k.json` reports 92.08% while paper Table 10 prints 91.6%. | **Fixed** | Two corrections applied in `main.tex`: (a) Table 10 row mean corrected from `91.6\%` to `92.08\%` so it matches the raw JSON, (b) new "Artifact reconciliation" paragraph after the table explains that the $92.08\%$ post-selection ledger (10 heterogeneous checkpoints, $N{=}500$) and the $82.0\%\!\to\!83.3\%$ paired control (Qwen3-8B, 5 seeds, $N{=}200$) are different experiments and should not be compared. The capstone report already carries this framing. |
| 10 | Novelty too thin once caveats are honored. | **Fixed (framing)** | Scope boundary: "narrow defensible contribution is reward-diversity triage (ZVF/GU) plus stack-identifiability reporting discipline, not a leaderboard or algorithm paper." Abstract + intro Claim 1 already foreground this; contribution ranking in §Introduction demotes breadth tables to supporting observations. |

## What still requires work outside this pass

- **Page-limit compliance for NeurIPS main track.** The paper as written is an appendix-rich 54-page engineering audit. A NeurIPS submission requires a ≤9-page main text; the rectification branch should either (a) extract a focused 9-page manuscript around the diagnostic contribution, or (b) submit this to the Datasets & Benchmarks track where the page budget is friendlier.
- **Submission bundle rebuild for double-blind.** The reviewer-facing zip must ship only anonymized PDFs (`main_anon.pdf`, anon report). The rebuild script should exclude named variants and any logs/figures carrying author identifiers.
- **One clean replication experiment.** If compute is available, the most credibility-positive additional experiment is a 3–5 seed replication of the Qwen3-8B GSM8K paired control under a single backend with the completion-only mask enabled, so the abstract's $82\%\!\to\!83.3\%$ number rests on seeded evidence rather than a single paired test.

## Files touched in this rectification pass

- `paper/main.tex` — title, BH-caveat, BH table caption, Table 10 row-mean arithmetic fix, artifact-reconciliation paragraph, scope-boundary paragraph.
- `paper/main_anon.tex` — title, BH-caveat, BH table caption, scope-boundary paragraph.
- `reports/final/HOSTILE_REVIEW_RECTIFICATION.md` — this log.

No new experiments, no new figures, no new tables. Rectification is editorial and arithmetic only.
