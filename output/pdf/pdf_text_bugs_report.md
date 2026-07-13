# Repository PDF text and layout bug review

## Scope and method

- Project PDFs inventoried: **99** files / **2,199** pages.
- Text-bearing PDFs visually reviewed: **44** files / **2,144** pages.
- Purely graphical plot PDFs skipped after preview validation: **55** files / **55** pages.
- Rendered review images: **2,144** grayscale PNGs at 120 DPI under `output/pdf/review_images/`.
- Every text-bearing page was examined visually in contact sheets; suspicious pages were then opened at original resolution. Text extraction, spelling scans, unresolved-reference scans, and edge-bound checks were secondary aids only.
- Original PDFs were not edited or deleted. SHA-256 integrity is recorded in `pdf_review_manifest.tsv` and verified separately in `original_pdf_integrity_check.txt`.
- The local `.venv/` dependency tree is outside project-document scope. Its eleven Matplotlib toolbar/icon PDFs are vendored GUI assets, not repository-authored documents.

Supporting artifacts:

- `output/pdf/pdf_inventory.tsv` - authoritative project-PDF inventory and classification.
- `output/pdf/pdf_review_manifest.tsv` - page/image counts, render settings, directories, and source hashes.
- `output/pdf/pdf_plot_only_skips.tsv` - visually previewed plot-only exceptions.
- `output/pdf/pdf_review_validation.txt` - final coverage and integrity validation.

## Per-PDF findings

### `output/pdf/Phase1_Project_Report_ZVF.pdf`

Pages visually inspected: **49**.

- No text or layout bugs found after visual inspection of all 49 rendered pages. The automated duplicate-word candidate on page 2 is a false positive caused by the two-column signature block; the rendered certificate is correct.

### `platform_hybrid/paper/acm_main.pdf`

Pages visually inspected: **11**.

- Page 1: the ACM reference block still contains the template DOI `https://doi.org/10.1145/nnnnnnn.nnnnnnn`.
- Page 1: `Manuscript submitted to ACM` is printed twice at the lower left, once directly below the copyright block and again in the page footer.
- No other text or layout bugs found on pages 2-11.

### `platform_hybrid/paper/consolidated_artifacts/2026-07-09-research-scratch/main.pdf`

Pages visually inspected: **78**.

- No other text or layout bugs found.

Additional cross-cutting QA:

- Pages 36-37: the Salesforce Hugging Face URL is split across the page boundary with a broken hyperlink annotation. On page 36 its border stretches through the footer and encloses the page number; on page 37 it continues as a long empty border at the top, with a tiny clipped annotation fragment at the extreme left edge.

### `platform_hybrid/paper/consolidated_artifacts/2026-07-09-research-scratch/main_anon.pdf`

Pages visually inspected: **76**.

- Page 1: the author block retains literal template placeholders: `Anonymous Author(s)`, `Affiliation`, `Address`, and `email`.
- No other text or layout bugs found.

### `platform_hybrid/paper/main.pdf`

Pages visually inspected: **273**.

- Pages 13, 15, 69, 73, 75, 81, 185, 211, and 214: ten figures are missing and rendered as placeholder boxes/raw filenames instead of graphics:
  - Page 13: `performance_profiles.pdf`.
  - Page 15: `scaling_law_figure.pdf` and `scaling_params_figure.pdf`.
  - Page 69: `wave6_sensitivity.pdf`.
  - Page 73: `effect_sizes_forest.pdf`.
  - Page 75: `zvf_heatmap.pdf`.
  - Page 81: `reward_stability.pdf`.
  - Page 185: `figures/zvf_signed_decomposition.pdf`.
  - Page 211: `figures/group_size_iter23.pdf`.
  - Page 214: `figures/group_size_iter27.pdf`.
- Page 273: a single final `As authoring/editing aids` checklist bullet is orphaned at the top of an otherwise almost entirely blank page, indicating poor final-page pagination.
- No other text or layout bugs were found beyond the items listed below.

Additional cross-cutting QA:

- Page 46: the three-part slope-constraint equation is clipped at the right edge. It visibly ends at `PLATEAU_SLOPE_MAX = 0.0`; the source tail `15` (the intended value `0.015`) is outside the page.
- Page 58: `The home-anchor MAE MAE(A -> A) is the within-anchor over-fit floor` duplicates `MAE`; one occurrence is redundant (or punctuation is missing before the function notation).
- Page 91: Table 83 extends beyond the right edge. The last header `Dr.GRPO sig. (rho_betaL,betaR, p)` and every value in that column are cut off after the opening fragment / `p=`.
- Page 163: two consecutive headings, `A8.6 Practical Diagnostic Recipe` and `A8.7 Practical Diagnostic Recipe`, have the same title; A8.6 contains no intervening content.
- Page 243: the fourth confidence interval in the displayed delta sequence is clipped at the right edge. It visibly ends `Delta = +0.242 [+0.236, +`; the source tail is `0.248]`.

### `platform_hybrid/paper/main_anon.pdf`

Pages visually inspected: **65**.

- Pages 12, 14, 15, 20, 22, 27, and 36: nine figures are absent and replaced by the literal box text `Figure omitted in this draft.`:
  - Page 12: Figure 6.
  - Page 14: Figures 8 and 9.
  - Page 15: Figure 11.
  - Page 20: Figure 13.
  - Page 22: Figure 15.
  - Page 27: Figure 18.
  - Page 36: Figure 20.
- There are 154 unresolved `??` cross-references across 50 pages. Page/count map: 5/3, 6/1, 8/3, 9/3, 10/5, 11/2, 12/1, 14/1, 15/2, 16/2, 18/4, 19/4, 20/1, 22/1, 23/2, 24/5, 25/2, 28/1, 30/2, 31/2, 32/1, 33/3, 34/1, 35/5, 37/5, 38/7, 39/1, 40/1, 41/1, 42/1, 43/4, 45/2, 47/2, 48/4, 49/3, 50/3, 51/1, 52/1, 54/2, 55/1, 56/7, 57/2, 58/2, 59/6, 60/1, 61/17, 62/9, 63/10, 64/2, 65/2. These affect citations, section/table/figure references, and equations; representative rendered text includes `Table ??`, `Section ??`, `Figure ??`, `Appendix ??`, and `Eq. (??)`.
- Page 65: the final `As authoring/editing aids` checklist bullet is isolated at the top of an otherwise almost entirely blank page.
- No additional text or layout bugs found.

### `platform_hybrid/paper/main_eai.pdf`

Pages visually inspected: **50**.

- Page 10: unresolved cross-reference `Section ??` in the cross-library scope caveat.
- Page 17: Figure 11 is missing and replaced by a large empty box containing `figures/wave6_sensitivity.pdf`.
- Page 37: unresolved cross-reference `Section ??` in the Salesforce xLAM attribution paragraph.
- Page 47: Figure 19 is missing and replaced by an empty box containing `figures/v2/old_trl_seeds.pdf`.
- No other text or layout bugs were found beyond the items listed below.

Additional cross-cutting QA:

- Page 1: the author line is clipped at the right edge after `Dhruva N M`, omitting the rest of `Dhruva N Murthy, Arumugam K`; both email lines are also clipped (`dhruva.n.m...` and `narayana.darapaneni@northwe...`).
- Page 12: Figure 6 is an empty framed box containing the raw path `figures/v2/performance_profiles.pdf`, so the intended figure is missing.
- Page 12: the artifact path is clipped at the right edge as `experiments/results/hel`; the intended source text is `experiments/results/heldout_gsm8k.json`.
- Page 12: math/text boundaries have swallowed spaces in `(p = 0.26, pairedbase - vs - GRPO, N = 200held - out prompts)`. This should read as `paired base-vs-GRPO, N = 200 held-out prompts`.
- Page 26: the run identifier is clipped as `frontier_gsm8k_nemo`; the source identifier is `frontier_gsm8k_nemotron-120b`.
- Page 27: the Hugging Face wildcard URL is clipped after `https://huggingface.co/arvindcr4/tinker-rl-bench-`, omitting the final `*`. The preceding W&B URL is also awkwardly broken after `https:` onto the next line, although no W&B characters are lost.
- Page 48: the generator path is clipped as `experiments/render_stat_r`; the source path is `experiments/render_stat_rigor_tex.py`.

### `platform_hybrid/paper/neurips_2026_variants/main_dnb.pdf`

Pages visually inspected: **31**.

- Page 1: unfinished anonymous-author metadata remains in the title block: `Anonymous Author(s)`, `Affiliation`, `Address`, and `email`.
- No other text, OCR, clipping, overlap, or layout defects were found on pages 2-31.

### `platform_hybrid/paper/neurips_2026_variants/main_workshop.pdf`

Pages visually inspected: **16**.

- Page 1: unfinished anonymous-author metadata remains in the title block: `Anonymous Author(s)`, `Affiliation`, `Address`, and `email`.
- No other text, OCR, clipping, overlap, or layout defects were found on pages 2-16.

### `platform_hybrid/paper/neurips_2026_variants/main_zvf.pdf`

Pages visually inspected: **24**.

- Page 1: unfinished anonymous-author metadata remains in the title block: `Anonymous Author(s)`, `Affiliation`, `Address`, and `email`.
- No other text, OCR, clipping, overlap, or layout defects were found on pages 2-24.

### `platform_hybrid/paper/paper_P1_scaling.pdf`

Pages visually inspected: **45**.

- No other text, OCR, clipping, overlap, or layout defects were found on pages 1-38 or 40-45.

Additional cross-cutting QA:

- Page 39: the three-part slope-constraint equation is clipped at the right edge. It visibly ends at `PLATEAU_SLOPE_MAX = 0.0`; the intended value is `0.015`.

### `platform_hybrid/paper/paper_P2_zvf.pdf`

Pages visually inspected: **44**.

- Page 20: an unreplaced figure placeholder is printed verbatim: `[figure zvf_scaling_cross_pillar.pdf pending regeneration]`.
- Page 21: an unreplaced figure placeholder is printed verbatim: `[figure zvf_library_bootstrap.pdf pending regeneration]`.
- Page 22: an unreplaced figure placeholder is printed verbatim: `[figure zvf_leadtime.pdf pending regeneration]`.
- Page 34: an unreplaced figure placeholder is printed verbatim: `[figure figures/zvf_signed_decomposition.pdf pending regeneration]`.
- No other text, OCR, clipping, overlap, or layout defects were found on pages 1-12, 14-19, 23-33, or 35-44.

Additional cross-cutting QA:

- Page 13: two consecutive headings, `5.6 Practical Diagnostic Recipe` and `5.7 Practical Diagnostic Recipe`, have the same title; section 5.6 has no intervening content.

### `platform_hybrid/paper/paper_P3_group_size.pdf`

Pages visually inspected: **61**.

- Page 9: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_extended.pdf pending regeneration]`.
- Page 14: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_advantage_variance.pdf pending regeneration]`.
- Page 16: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_iter15.pdf pending regeneration]`.
- Page 20: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_iter91.pdf pending regeneration]`.
- Page 21: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_iter95.pdf pending regeneration]`.
- Page 24: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_iter23.pdf pending regeneration]`.
- Page 26: an unreplaced figure placeholder is printed verbatim: `[figure figures/group_size_iter27.pdf pending regeneration]`.
- Page 32: an unreplaced figure placeholder is printed verbatim: `[figure group_size_iter39.pdf pending regeneration]`.
- Page 34: an unreplaced figure placeholder is printed verbatim: `[figure group_size_iter43.pdf pending regeneration]`.
- Page 35: an unreplaced figure placeholder is printed verbatim: `[figure group_size_iter47.pdf pending regeneration]`.
- No other text, OCR, clipping, overlap, or layout defects were found on pages 1-8, 10, 12-13, 15, 17-19, 22-23, 25, 27-31, 33, or 36-61.

Additional cross-cutting QA:

- Page 11: the prose contains the duplicated word/function name `the retention retention(T) = ...`; one `retention` is redundant.
- Page 47: the fourth confidence interval in the displayed delta sequence is clipped at the right edge. It visibly ends `Delta = +0.242 [+0.236, +`; the source tail is `0.248]`.

### `platform_hybrid/paper/paper_P4_length_bias.pdf`

Pages visually inspected: **44**.

- Page 19: an unreplaced figure placeholder is printed verbatim: `[figure length_bias_iter32.pdf pending regeneration]`.
- Page 24: an unreplaced figure placeholder is printed verbatim: `[figure length_bias_iter44.pdf pending regeneration]`.
- No other text, OCR, clipping, overlap, or layout defects were found on pages 1-16, 18, 20-23, or 25-44.

Additional cross-cutting QA:

- Page 17: Table 14 extends beyond the right edge. The sixth-column header `Dr.GRPO sig. (rho_betaL,betaR, p)` and the ends of all four data rows are clipped after the opening fragment / `p=`.

### `platform_hybrid/paper/paper_P5_minreport.pdf`

Pages visually inspected: **78**.

- Page 13: missing figure is rendered as the literal placeholder `[figure p5_minreport_per_item.pdf pending regeneration]`.
- Page 15: missing figure is rendered as the literal placeholder `[figure p5_field_sufficiency.pdf pending regeneration]`.
- Page 19: missing figure is rendered as the literal placeholder `[figure p5_field_discriminative_entropy.pdf pending regeneration]`.
- Page 20: missing figure is rendered as the literal placeholder `[figure p5_manifest_r2_gap.pdf pending regeneration]`.
- Page 22: missing figure is rendered as the literal placeholder `[figure p5_minreport_subfield_audit.pdf pending regeneration]`.
- Page 25: missing figure is rendered as the literal placeholder `[figure p5_mve_field_dist.pdf pending regeneration]`.
- No other text or layout bugs found after visual inspection of all 78 pages.

Additional cross-cutting QA:

- Page 34: the `Item14`-`Item17` null-excess equation is clipped at the right edge after `Item17`. The missing source text is `Delta H = +0.257 (approximately +3.6 sigma)`.

### `platform_hybrid/paper/paper_P6_registry.pdf`

Pages visually inspected: **64**.

No other text or layout bugs found after visual inspection of all 64 pages.

Additional cross-cutting QA:

- Page 35: Table 24's rightmost `Closest proxy` column extends beyond the page. Identifiers are visibly truncated (`tinker_dapo_...`, `tinker_gspo_...`, `LitePPO`, `REINF...`) instead of showing the full proxy strings / `no tinker ... rollout log` text.
- Page 39: `Prior to this iterationation, every ...` contains the typo `iterationation`; it should be `iteration`.
- Page 40: `this iterationation gives a uniform ...` contains the same typo.
- Page 41: `All 4 were closed this iterationation on real evidence` contains the same typo.
- Page 44: `a measured measured[] layer` visibly repeats `measured`; the first occurrence is redundant or should be replaced by a different descriptor.

### `platform_hybrid/paper/paper_P7_zvf_controller.pdf`

Pages visually inspected: **81**.

No other confirmed text or layout bugs found after visual inspection of all 81 pages.

Uncertain:

- Page 17: the reproduction paragraph prints the raw asset path `experiments/results/p5p8/figures/p7_per_prompt_g_distribution.pdf` followed by “Figure shows the per-method G* distribution,” but no figure is embedded on that page. This may be an intentionally external reproduction artifact, or it may be a missing figure insertion.

Additional cross-cutting QA:

- Page 22: `...are principled, buton this evidence base...` is missing a space; it should be `but on`.
- Page 52: `aer and areal swap positions` misspells the method name; it should be `aero and areal swap positions`.

### `platform_hybrid/paper/paper_P8_fraud.pdf`

Pages visually inspected: **93**.

- Page 30: missing figure is rendered as the literal placeholder `[figure p8_noisy_sensor.pdf pending regeneration]`.
- No other text or layout bugs found after visual inspection of all 93 pages.

Additional cross-cutting QA:

- Page 16: `...excluding zero in 25/25 cells (Table Table 19)` duplicates `Table`.
- Page 24: Table 29 extends beyond the right edge. The entire `sigma = 1.00` column is cut off, including the values `-0.260`, `-0.323`, `+0.086`, `-0.148`, CI `[-.221, -.075]`, and final `yes`.
- Page 28: `deterministic seed seed = 20260704` duplicates `seed`.
- Page 32: `The XGB-20raw vs XGB-pair pair is the one exception` duplicates `pair`.
- Page 45: `strictly cheaper than xgb-only ONLY at the trivial ...` duplicates `only` with inconsistent case.
- Page 68: consecutive headings `7.5.40 Falsifiable questions and operational stakes` and `7.5.41 Falsifiable questions and operational stakes` are identical; 7.5.40 has no intervening content.

### `platform_hybrid/paper/supplement.pdf`

Pages visually inspected: **18**.

- No other text or layout bugs found after visual inspection of all 18 pages. The initially suspicious references on pages 1 and 9 were verified at original resolution as valid `Table 6` and `Table 13` references.

Additional cross-cutting QA:

- Page 12: the final monospaced learning-trajectory line is clipped at the right edge. It ends at `Output="The answer is 42." (format: text with nu`; the text does not continue on page 13.

### `platform_hybrid/paper/tikz/architecture.pdf`

Pages visually inspected: **1**.

No text or layout bugs found after visual inspection of the page; the right-side W&B, HuggingFace, and GitHub nodes remain inside the canvas.

### `platform_hybrid/paper/tikz/pipeline.pdf`

Pages visually inspected: **1**.

No text or layout bugs found after visual inspection of the page.

### `platform_hybrid/paper/tikz/reward_flow.pdf`

Pages visually inspected: **1**.

No text or layout bugs found after visual inspection of the page.

### `platform_hybrid/paper/tikz/taxonomy.pdf`

Pages visually inspected: **1**.

No text or layout bugs found after visual inspection of the page.

### `platform_hybrid/sem 3 work/deliverables/group6-original-report.pdf`

Pages visually inspected: **101**.

- Page 1: the PES University logo is missing; a dashed placeholder box prints `PES University Logo (add pes_logo.png)`.
- Pages 2, 10, 11, 15, 16, 18, 24, 25, 26, 32, 33, 34, 35, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 62, 65, 79, and 80: unresolved bibliography citations are visibly rendered as `[?]`, `[? ]`, `[? ?]`, or a leading `?]`. Examples include `CogAgent [? ]` on page 10, `Source: Adapted from Wei et al. [? ] and Yao et al. [? ]` on page 18, `Source: Data from [? ? ? ]` on page 48, and `Source: Adapted from ? ]` on page 54.
- No other text or layout bugs found after visual inspection of all 101 pages. Page 96 was opened individually and its `Appendix D - Reproducibility Checklist` heading is intact.

Additional cross-cutting QA:

- Page 69: the GRPO training-pipeline diagram extends beyond the right page edge. The `Updated Policy (GRPO + ...)` node and its right-side checkpoint/arrow content are visibly cut off.
- Page 77: the Sandhya model identifier is clipped after `llm-multiturn-tool-call-gr...`; the W&B entity line below is also clipped after `arvindcr4-pes-universit...` rather than displaying the complete identifier.
- Page 78: the Hugging Face model path is clipped after `arvindcr4/skyrl-tinker-qwen3-...`; the next line begins with the Tinker run ID, so the model path never continues.

### `platform_hybrid/sem 3 work/submissions/neurips-main-track/main-track-paper-anonymous.pdf`

Pages visually inspected: **48**.

- Page 10: unresolved figure cross-reference is rendered as `Figure ??` in the “Headline findings” paragraph.
- Page 27: unresolved cross-reference is rendered as `Section ??` in the GSM8K base-model-control sentence.
- Page 29: unresolved cross-reference is rendered as `Section ??` in the xLAM 60k real-data-run sentence.
- Page 31: unresolved cross-reference is rendered as `Section ??` in the multiple-comparisons sentence.
- Page 44: unresolved cross-reference is rendered as `Section ??` in the support/evidence sentence.
- Page 46: unresolved cross-reference is rendered as `Sections ?? and 6.2` in the cost-constrained-label sentence.
- No other text or layout bugs found after visual inspection of all 48 pages. The anonymous author block is appropriate for this submission copy.

Additional cross-cutting QA:

- Page 7: Figure 1's caption says `Taxonomy of of RL libraries`; one `of` is redundant.

### `platform_hybrid/sem 3 work/submissions/neurips-main-track/main-track-paper.pdf`

Pages visually inspected: **60**.

No other text or layout bugs found after visual inspection of all 60 pages.

Additional cross-cutting QA:

- Page 7: Figure 1's caption says `Taxonomy of of RL libraries`; one `of` is redundant.

### `platform_hybrid/sem 4 work/papers/P1-scaling-laws.pdf`

Pages visually inspected: **45**.

No other text or layout bugs found after visual inspection of all 45 pages.

Additional cross-cutting QA:

- Page 39: the three-part slope-constraint equation is clipped at the right edge. It visibly ends at `PLATEAU_SLOPE_MAX = 0.0`; the intended value is `0.015`.

### `platform_hybrid/sem 4 work/papers/P2-zero-variance-fraction.pdf`

Pages visually inspected: **42**.

- Page 12: missing figure is rendered as the literal placeholder `[figure zvf_by_library.pdf pending regeneration]`.
- Page 15: missing figure is rendered as the literal placeholder `[figure zvf_antiherding_falsification.pdf pending regeneration]`.
- Page 17: two missing figures are rendered as the literal placeholders `[figure figures/zvf_dynamics_phase.pdf pending regeneration]` and `[figure figures/zvf_dynamics_leadtime.pdf pending regeneration]`.
- Page 18: missing figure is rendered as the literal placeholder `[figure zvf_scaling_cross_pillar.pdf pending regeneration]`.
- Page 19: missing figure is rendered as the literal placeholder `[figure zvf_library_bootstrap.pdf pending regeneration]`.
- Page 21: missing figure is rendered as the literal placeholder `[figure zvf_leadtime.pdf pending regeneration]`.
- Page 32: missing figure is rendered as the literal placeholder `[figure figures/zvf_signed_decomposition.pdf pending regeneration]`.
- Page 34: missing figure is rendered as the literal placeholder `[figure figures/zvf_iter62_difficulty_strata.pdf pending regeneration]`.
- No other text or layout bugs found after visual inspection of all 42 pages.

Additional cross-cutting QA:

- Page 13: two consecutive headings, `5.6 Practical Diagnostic Recipe` and `5.7 Practical Diagnostic Recipe`, have the same title; section 5.6 has no intervening content.

### `platform_hybrid/sem 4 work/papers/P3-group-size.pdf`

Pages visually inspected: **59**.

- p9: Figure 4 is replaced by the boxed placeholder `[figure figures/group_size_extended.pdf pending regeneration]`.
- p10: Figure 5 is replaced by `[figure figures/group_size.pdf pending regeneration]`.
- p13: Figure 6 is replaced by `[figure figures/group_size_advantage_variance.pdf pending regeneration]`.
- p15: Figure 7 is replaced by `[figure figures/group_size_iter15.pdf pending regeneration]`.
- p18: Figure 8 is replaced by `[figure figures/group_size_iter19.pdf pending regeneration]`.
- p20: Figure 10 is replaced by `[figure figures/group_size_iter95.pdf pending regeneration]`.
- p23: Figure 11 is replaced by `[figure figures/group_size_iter23.pdf pending regeneration]`.
- p26: Figure 12 is replaced by `[figure figures/group_size_iter27.pdf pending regeneration]`.
- p28: Figure 13 is replaced by `[figure figures/group_size_iter31.pdf pending regeneration]`.
- p30: Figure 14 is replaced by `[figure figures/group_size_iter35.pdf pending regeneration]`.
- p31: Figure 15 is replaced by `[figure group_size_iter39.pdf pending regeneration]`.
- p33: Figure 16 is replaced by `[figure group_size_iter43.pdf pending regeneration]`.
- p34: Figure 17 is replaced by `[figure group_size_iter47.pdf pending regeneration]`.
- p44: Figure 18 is replaced by `[figure figures/group_size_iter67_iaf.pdf pending regeneration]`.
- p45: Figure 19 is replaced by `[figure figures/group_size_iter107.pdf pending regeneration]`.
- p46: Figure 20 is replaced by `[figure figures/group_size.pdf pending regeneration]`.
- p48: Figure 21 is replaced by `[figure figures/group_size_iter115.pdf pending regeneration]`.

Additional cross-cutting QA:

- Page 11: the prose contains the duplicated word/function name `the retention retention(T) = ...`; one `retention` is redundant.
- Page 46: the fourth confidence interval in the displayed delta sequence is clipped at the right edge. It visibly ends `Delta = +0.242 [+0.236, +`; the source tail is `0.248]`.

### `platform_hybrid/sem 4 work/papers/P4-length-bias.pdf`

Pages visually inspected: **43**.

- p19: Figure 11 is replaced by `[figure figures/length_bias_iter32.pdf pending regeneration]`.
- p21: Figure 12 is replaced by `[figure figures/length_bias_iter36.pdf pending regeneration]`.
- p23: Figure 13 is replaced by `[figure figures/length_bias_iter40.pdf pending regeneration]`.
- p24: Figure 14 is replaced by `[figure length_bias_iter44.pdf pending regeneration]`.
- p25: Figure 15 is replaced by `[figure figures/length_plateau_slopes.pdf pending regeneration]`.
- p35: Figure 16 is replaced by `[figure figures/length_bias_iter80.pdf pending regeneration]`.

Additional cross-cutting QA:

- Page 17: Table 14 extends beyond the right edge. The sixth-column header `Dr.GRPO sig. (rho_betaL,betaR, p)` and the ends of all four data rows are clipped after the opening fragment / `p=`.

### `platform_hybrid/sem 4 work/papers/P5-report-the-stack.pdf`

Pages visually inspected: **77**.

- p13: Figure 4 is replaced by `[figure p5_minreport_per_item.pdf pending regeneration]`.
- p15: Figure 5 is replaced by `[figure p5_field_sufficiency.pdf pending regeneration]`.
- p19: Figure 6 is replaced by `[figure p5_field_discriminative_entropy.pdf pending regeneration]`.
- p20: Figure 7 is replaced by `[figure p5_manifest_r2_gap.pdf pending regeneration]`.
- p22: Figure 8 is replaced by `[figure p5_minreport_subfield_audit.pdf pending regeneration]`.
- p25: Figure 9 is replaced by `[figure p5_mve_field_dist.pdf pending regeneration]`.

Additional cross-cutting QA:

- Page 34: the `Item14`-`Item17` null-excess equation is clipped at the right edge after `Item17`. The missing source text is `Delta H = +0.257 (approximately +3.6 sigma)`.

### `platform_hybrid/sem 4 work/papers/P6-grpo-registry.pdf`

Pages visually inspected: **64**.

No other bugs found after visual inspection of all 64 pages.

Additional cross-cutting QA:

- Page 35: Table 24's rightmost `Closest proxy` column extends beyond the page. Identifiers are visibly truncated (`tinker_dapo_...`, `tinker_gspo_...`, `LitePPO`, `REINF...`) instead of showing the full proxy strings / `no tinker ... rollout log` text.
- Page 39: `Prior to this iterationation, every ...` contains the typo `iterationation`; it should be `iteration`.
- Page 40: `this iterationation gives a uniform ...` contains the same typo.
- Page 41: `All 4 were closed this iterationation on real evidence` contains the same typo.
- Page 43: `a measured measured[] layer` visibly repeats `measured`; the first occurrence is redundant or should be replaced by a different descriptor.

### `platform_hybrid/sem 4 work/papers/P7-zvf-controller.pdf`

Pages visually inspected: **81**.

No other bugs found after visual inspection of all 81 pages.

Additional cross-cutting QA:

- Page 22: `...are principled, buton this evidence base...` is missing a space; it should be `but on`.
- Page 52: `aer and areal swap positions` misspells the method name; it should be `aero and areal swap positions`.

### `platform_hybrid/sem 4 work/papers/P8-fraud.pdf`

Pages visually inspected: **93**.

- p30: Figure 3 is replaced by `[figure p8_noisy_sensor.pdf pending regeneration]`.

Additional cross-cutting QA:

- Page 16: `...excluding zero in 25/25 cells (Table Table 19)` duplicates `Table`.
- Page 24: Table 29 extends beyond the right edge. The entire `sigma = 1.00` column is cut off, including the values `-0.260`, `-0.323`, `+0.086`, `-0.148`, CI `[-.221, -.075]`, and final `yes`.
- Page 28: `deterministic seed seed = 20260704` duplicates `seed`.
- Page 32: `The XGB-20raw vs XGB-pair pair is the one exception` duplicates `pair`.
- Page 45: `strictly cheaper than xgb-only ONLY at the trivial ...` duplicates `only` with inconsistent case.
- Page 68: consecutive headings `7.5.40 Falsifiable questions and operational stakes` and `7.5.41 Falsifiable questions and operational stakes` are identical; 7.5.40 has no intervening content.

### `platform_hybrid/sem 4 work/submissions/neurips-workshop/workshop-paper-anonymous.pdf`

Pages visually inspected: **16**.

No bugs found after visual inspection of all 16 pages.

### `platform_local/blind_review/main_anon.pdf`

Pages visually inspected: **51**.

No bugs found after visual inspection of all 51 pages.

### `thesis/main.pdf`

Pages visually inspected: **31**.

No bugs found after visual inspection of all 31 pages.

### `thesis/viva/viva_slides.pdf`

Pages visually inspected: **18**.

No bugs found after visual inspection of all 18 pages.

### `zvf-program/audit/reproducibility_audit.pdf`

Pages visually inspected: **6**.

- p1: unfinished author/affiliation text (`[TODO: finalize]`) and unresolved inline `[cite: ...]` keys are visible.
- p2-p4: unresolved inline `[cite: ...]` keys are visible; p4 additionally contains two headline results tables whose numeric cells are still `[TODO: ]` / `fill from full-scale audit corpus` placeholders.
- p6: all eleven references are visible `[TODO: ...]` stubs rather than a finished bibliography.

### `zvf-program/position/min_report_rl.pdf`

Pages visually inspected: **13**.

- p1: unfinished author/affiliation block and multiple `[TODO: trace to v1 audit citation]` placeholders.
- p2-p7: unresolved `[cite: ...]` keys and/or `[TODO: trace ...]` markers remain visible in the body and tables.
- p8: both proposed audit tables are filled with `[TODO: ]` / `fill from audit corpus` placeholders rather than results.
- p9: seed count and survival threshold are explicitly left as TODOs; unresolved citation keys remain.
- p10-p11: unresolved citation/trace markers remain; p11 also contains an unfinished “extend the list” TODO.
- p13: the References section explicitly says to replace the stub with a real `.bib`, followed by unresolved TODO entries.

### `zvf-program/registry/grpo_registry.pdf`

Pages visually inspected: **12**.

- p1: unfinished author/affiliation block (`[TODO: finalize]`) and unresolved citation keys.
- p2-p3: the prose exposes `[TODO: ]` placeholders as current catalog content.
- p4-p7: the multi-page seed catalog contains many `[TODO: ]`, `[TODO: verify]`, and unresolved `[cite: ...]` cells.
- p8: unresolved citation keys remain in the tooling section.
- p9: the displayed machine-readable schema contains `"paper_doi_or_url": "TODO"`.
- p10: repository URL is still `[TODO: add repository URL]`, and TODO catalog cells remain.
- p11-p12: bibliography entries contain unresolved paper stubs and repeated `[TODO: verify final venue]` markers.

### `zvf-program/review/review_2026-07-13_slides.pdf`

Pages visually inspected: **17**.

No bugs found after visual inspection of all 17 pages.

### `zvf-program/theory/zvf_theory.pdf`

Pages visually inspected: **12**.

- p1: the title page still says `[Author]`; the document is visibly labelled `Draft / Proof Sketches` and warns that author verification is required.
- p2: unresolved `[TODO-...]` reference keys are visible throughout the introduction.
- p3: the document explicitly identifies itself as a draft skeleton and refers to inline `[GAP: ...]` / `TODO(proof-gap)` markers.
- p4-p9: theorem statements and proof sketches contain numerous visible `[GAP: ...]` and unresolved `[TODO-...]` markers; these include load-bearing independence, U-statistic framing, confidence-bound direction, learning-signal definition, global optimum, integrability, and controller-connection gaps.
- p11: unresolved `[TODO-liu-drgrpo]` and `[GAP: T1/T2/T3 sketch]` markers remain.
- p12: the heading says `References (placeholders - DO NOT cite as resolved)` and the bibliography contains unresolved TODO references.

### `platform_tinker/reports/esa_phase1/build/Phase1_Project_Report_ZVF.pdf`

Pages visually inspected: **50**.

- Page 50: the Appendix C notation table's final `AUROC` row is orphaned by itself at the top of a continuation page, leaving almost the entire page blank. The row is readable, but this is a conspicuous pagination/layout glitch.
- No other text or layout bugs found after visual inspection of all 50 pages. The spelling scan's apparent `guage`, `tage`, and `ues` matches were verified as extraction artifacts caused by line/column boundaries; the rendered words are correct.

## Rejected automated candidates (not defects)

- `platform_hybrid/paper/paper_P5_minreport.pdf` page 24 and its Semester-4 duplicate page 24: Table 23's rule reaches the edge, but the rightmost `eta^2_seed` values and brackets remain inside the page and readable.
- `platform_hybrid/paper/paper_P7_zvf_controller.pdf` page 17 and its Semester-4 duplicate: the prose prints an external figure-artifact path, but it does not assert that a figure is embedded at that location; treated as uncertain rather than a confirmed missing figure.
- P7 pages 79-80: square glyphs are intentional proof-ending QED marks, not replacement-character corruption.
- `group6-original-report.pdf` pages 96-97: square glyphs are intentional checklist boxes.
- Codespell hits such as `guage`, `ment`, `PTD`, `Meger`, `tage`, and `ues` were visually rejected as line-break, math-extraction, column-boundary, or proper-name artifacts. `vermillion`, `covert`, `trough`, `invokable`, and British `fulfilment` are legitimate context-specific words/spellings.
- Full-bleed slide backgrounds in `thesis/viva/viva_slides.pdf` and `zvf-program/review/review_2026-07-13_slides.pdf` intentionally touch the canvas edges; no slide text was clipped.

## Plot-only PDFs skipped after visual preview

Each item below is a single-page plot/figure export containing axes, labels, legends, or annotations but no narrative document text. These were previewed and skipped under the explicit graphical-plot exception.

- `platform_hybrid/experiments/results/framework_comparison.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/experiments/results/p5p8/figures/p8_asym_cost.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/comparison_bars.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/figure_lora_sparsity.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/figure_saturation_curves.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/figure_stat_rigor.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter103.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter107.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter115.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter119.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter19.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter31.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter35.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter39.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter43.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter47.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter67_iaf.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter71_decomp.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter71_g4_budget.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/group_size_iter99.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/learning_curves.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter100_var.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter104_qreg.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter108_progress_lquant.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter112_sever_reward.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter116_sever_zvf.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter124_static_vs_dynamic.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter32.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter36.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter40.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter44.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_bias_iter80.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/length_plateau_slopes.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/old_trl_seeds.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/performance_profiles.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/scaling_law_iter101.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/scaling_law_iter85.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/scaling_law_iter93.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/scaling_law_iter97.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/sensitivity_heatmap.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/comparison_bars.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/framework_comparison.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/group_size_ablation.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/kl_proxy.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/learning_curves.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/ppo_vs_grpo.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/scaling.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/sensitivity_heatmap.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/v2/zvf_correlation.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/zvf_iter62_difficulty_strata.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/zvf_iter70_quad.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/zvf_leadtime.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/zvf_library_bootstrap.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/zvf_scaling_cross_pillar.pdf` - purely graphical plot export without narrative prose.
- `platform_hybrid/paper/figures/zvf_vs_failure.pdf` - purely graphical plot export without narrative prose.

## Out-of-scope vendored dependency PDFs

The following files live inside the machine-local `.venv/` and are Matplotlib GUI toolbar icons or bundled demo artwork, not project documents:

- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/back.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/filesave.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/forward.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/hand.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/help.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/home.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/matplotlib.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/move.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/qt4_editor_options.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/subplots.pdf`
- `.venv/lib/python3.12/site-packages/matplotlib/mpl-data/images/zoom_to_rect.pdf`
