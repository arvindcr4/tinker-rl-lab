# Document Review and Remediation Report: Paper P6 (Submission Build)

**Assigned Manuscript**: `platform_hybrid/paper/paper_P6_submission.tex`  
**Artifact Evaluated**: GRPO-Registry Resource and Submission Variant  
**Evaluator**: Document Review Analyst & Remediator  
**Date**: October 18, 2026  
**Status**: Remediated & Verified Clean  

---

## 1. Executive Summary

This report presents the technical audit and remediation of `paper_P6_submission.tex`. The manuscript presents **GRPO-Registry**, a machine-readable catalog of group-relative reinforcement learning (RL) training stacks and variant deltas. The submission build targets a conference datasets/resources or position-artifact track under a strict 9-page limit for the main text.

The audit verified the submission framing, mathematical formulations, citation hygiene, and empirical claims. Key deficiencies identified during initial review included:
1. An unfinished placeholder in Section 5.2 that referenced pending Claim-C experiments;
2. Float placement warnings (`[h]` changed to `[ht]`) across appendix tables;
3. A conceptual inversion in Section 8.3 Finding A regarding Zero Variance Fraction (ZVF) and within-group contrast;
4. Inconsistent cross-references between the submission body and appendix;
5. Outdated stack entry counts in Section 3 text.

All identified issues were directly remediated in the LaTeX source files. The manuscript was compiled using the official project toolchain (`tectonic`). The build completed with **zero errors, zero undefined references, zero BibTeX errors, and zero float warnings**. The main text spans exactly 8 pages (excluding references and appendices), strictly adhering to conference submission constraints.

---

## 2. Key Strengths & Contributions

1. **Rigorous Machine-Readable Schema**:
   - The JSON schema (`registry/schema.json`) defines strict specifications for training stacks and variant deltas.
   - It incorporates the 7-field run manifest of the MIN-REPORT-RL standard.
   - It establishes explicit semantics for missingness: JSON `null` indicates unreported fields, while explicit values (`false`, `0.0`, `"none"`) indicate verified absence.

2. **Falsification of Nominal Benchmarking**:
   - The paper demonstrates that nominal algorithm labels severely underdetermine executed update rules.
   - It exposes how two runs labelled identically ("GRPO, Qwen3-8B, GSM8K") diverged $17\times$ in final performance ($85.6\%$ vs $5.0\%$) due to an unannounced base checkpoint switch ($64\%$ of gap) and confounded runtime harnesses.

3. **Diagnostic R0--R5 Severity Ladder**:
   - The `stackdiff` tool maps paired configuration differences to a principled risk ladder.
   - R5 isolates outcome-critical confounds (checkpoints, prompt templates, reward extractors, eval protocols).
   - R4 through R1 isolate algorithmic changes, hyperparameters, runtime environments, and random seeds.

4. **Verifiable Seed Population**:
   - The registry includes 28 concrete stack records and 20 variant deltas derived from actual experimental runs.
   - The paper avoids hypothetical entries, testing the schema against real-world reporting gaps.

5. **Exhaustive Raw-Data Recomputation**:
   - Section 9 (iter-190 audit) recomputes deltas from raw step-level TSVs rather than trusting stored values.
   - It detects zero silent drift ($\epsilon \le 10^{-4}$) between source TSVs and registry entries.

---

## 3. Issues Identified & Remediation Details

### Issue 1: Unfinished Placeholder in Section 5.2 [Critical - Fixed]
- **Observation**: Section 5.2 in `sections/p6s_worked_examples.tex` contained verbatim draft text: `\textbf{[Placeholder --- results pending in \texttt{experiments/results/publication\_fixes/}.]}`.
- **Root Cause**: The text was held open for the Claim-C experiment analyzing matched TRL vs Tinker runs.
- **Verification of Data**: The experiment was completed and recorded in `platform_hybrid/experiments/results/publication_fixes/matched_trl_tinker/matched_trl_tinker_result.json`. Across 3 seeds (42, 123, 456) and 30 steps with identical base checkpoints, prompts, parsers, and batch shapes, the training gap between Tinker and TRL had mean $+0.050$ ($95\%$ bootstrap CI $[-0.156, +0.294]$, spanning zero). The held-out accuracy gap had mean $+0.012$ ($95\%$ bootstrap CI $[-0.050, +0.078]$, spanning zero).
- **Remediation**: Replaced the placeholder block with a formal subsection titled *"Controlled Validation: Matched TRL-vs-Tinker Evaluation"*, documenting these empirical confidence intervals and confirming that matching R5/R3 parameters reduces flip risk to R2.

### Issue 2: Conceptual Inversion in Section 8.3 Finding A [Major - Fixed]
- **Observation**: In `sections/p6_measured_claimed.tex` (lines 93--97 and 131--135), the text claimed: *"no measured variant has lifted the within-group contrast that the variant claims to lift"* because variants had lower mean ZVF than GRPO.
- **Root Cause**: The text confused ZVF (Zero Variance Fraction) with contrast. ZVF measures the fraction of groups with zero variance (contrast starvation). Lower ZVF implies *higher* within-group contrast.
- **Remediation**: Corrected Finding A and the summary bullet to state accurately that all 8 measured variants reduce ZVF (reducing contrast starvation by $0.16$ to $0.37$). Added the critical distinction that this reduction in ZVF does not translate into positive reward deltas on the same stack.

### Issue 3: Inconsistent Section vs. Appendix Cross-References [Major - Fixed]
- **Observation**:
  - In `sections/p6s_population.tex` line 79, text referred to `(Section~\ref{sec:p6-measured})`. In the submission build, `sec:p6-measured` is placed in the Appendix.
  - In `sections/p6_query_auditor.tex` line 47, text referenced `Table~\ref{tab:p6-entries}` without noting that the full per-entry table was moved to Appendix B in the submission cut-list.
- **Remediation**:
  - Updated `p6s_population.tex` to use `\ifdefined\SUBMISSIONBUILD(Appendix~\ref{sec:p6-measured})\else(Section~\ref{sec:p6-measured})\fi`.
  - Updated `p6_query_auditor.tex` to cite summary Table 1 in the main body and direct readers to Table 2 in Appendix B for individual entries.
  - Clarified Table 2 references in `sections/p6_conclusion.tex`.

### Issue 4: Outdated Entry Counts in Section 3 [Minor - Fixed]
- **Observation**: In `sections/p6_schema.tex` (line 120), the text referred to "all 31 existing entries" and stated that 24 entries leave `ci_method` null.
- **Root Cause**: The 31-entry figure was an earlier milestone count (20 stacks + 11 deltas). The active registry contains 28 stack entries, 7 of which have `ci_method` populated.
- **Remediation**: Corrected text to state that 7 entries have `ci_method` populated while the remaining 21 of 28 stack entries leave it null.

### Issue 5: TeX Float Specifier Warnings [Minor - Fixed]
- **Observation**: Compilation emitted `LaTeX Warning: 'h' float specifier changed to 'ht'` for tables in `sections/p6s_appendix_entries.tex` and `sections/p6_iter190_raw_recompute_audit.tex`.
- **Remediation**: Updated `\begin{table}[h]` to `\begin{table}[ht]` in both files.

### Issue 6: Misleading Build Comment in Manuscript Header [Minor - Fixed]
- **Observation**: Header comment in `paper_P6_submission.tex` read `Build: tectonic -X compile paper_P6_registry.tex`.
- **Remediation**: Corrected header to state `Build: tectonic -X compile paper_P6_submission.tex`.

---

## 4. Empirical Claims & Ablation Audit

| Claim in Text | Evidence Source | Status | Audit Assessment |
| :--- | :--- | :--- | :--- |
| **$17\times$ Nominal Performance Gap** | TRL vs Tinker GSM8K comparison | **Verified** | Real benchmark artifact ($85.6\%$ vs $5.0\%$). Accurately characterized as an unpinned confound (Base vs Instruct checkpoint explaining $64\%$ of gap, plus template/parser confounds), not an algorithm superiority result. |
| **Controlled Matched Equivalence** | `matched_trl_tinker_result.json` | **Verified** | Under identical checkpoints and harnesses, Tinker minus TRL training reward gap is $+0.050$ ($95\%$ CI spans 0) and held-out gap is $+0.012$ ($95\%$ CI spans 0). |
| **Pairwise Registry Flip-Risk Distribution** | `p6_registry_numbers.json` via CLI | **Verified** | 378 total pairs: R0: 39, R1: 0, R2: 1, R3: 10, R4: 26, R5: 302. Underspecified pairs: 342/378. Median coverage: 0.19. Generated pair subset: 66 pairs. |
| **Same-Stack Variant Reward Invariance** | N2 4-method tensor runs | **Verified** | AERO regresses reward by $-0.014$ (CI $[-0.023, -0.005]$); AREAL regresses reward by $-0.020$ (CI $[-0.032, -0.008]$). GIFT reward delta $+0.016$ spans 0. All reward deltas $\le 2$pp. |
| **Completion Length Redistribution** | N2 last-10 step tensor data | **Verified** | AERO and AREAL lengthen completions by $+31.1$ and $+30.1$ tokens respectively while reducing length-CV by $-0.034$ and $-0.038$. |
| **Raw TSV Recompute Consistency** | iter-190 audit of raw source TSVs | **Verified** | Maximum drift between stored registry values and recomputed bootstrap deltas is $\le 10^{-4}$. Zero silent drift detected. |

---

## 5. Compilation Verification Log

Compilation was executed in `/Users/arvind/Developer/agentic_repos/tinker-rl-lab/platform_hybrid/paper` using:
```bash
/opt/homebrew/bin/tectonic -X compile paper_P6_submission.tex --keep-logs --keep-intermediates --untrusted --print
```

### Build Transcript Summary:
- **Engine**: XeTeX / Tectonic v2 CLI
- **Exit Code**: `0` (Success)
- **Document Class**: `article` (NeurIPS format geometry: textwidth 397.48pt, textheight 650.43pt)
- **Output Files**:
  - `paper_P6_submission.pdf` (165.96 KiB, 20 pages total)
  - `paper_P6_submission.log` (30.94 KiB)
  - `paper_P6_submission.blg` (209 B)
  - `paper_P6_submission.aux` (23.42 KiB)
- **Page Budget Audit**:
  - Pages 1--8: Main text (Abstract, Intro, Overview, Schema, Population, Query/Auditor, Worked Examples, Synthesis, Conclusion)
  - Pages 9--10: References (`paper_P6_submission.bbl`)
  - Pages 11--20: Appendix (Artifact Reference, Seed Tables, Related Resources, Measured Evidence, Recompute Audit, Governance)
  - **Main Body Budget Compliance**: Exactly 8 pages ($\le 9$ pages required).
- **Warning Audit**:
  - Undefined references: **0**
  - Undefined citations: **0**
  - Multiply-defined labels: **0**
  - LaTeX float placement warnings: **0**
  - BibTeX warnings: **0**

---

## 6. Final Recommendation

**Decision**: **ACCEPT / READY FOR SUBMISSION**

The manuscript `paper_P6_submission.tex` has been remediated to publication standards. All placeholders and draft markers have been eliminated. Empirical findings are strictly aligned with machine-checked data in the repository. The document compiles cleanly with zero warnings or errors.
