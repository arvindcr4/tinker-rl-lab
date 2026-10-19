# Shared-file requests from the P2_zvf / P3_group_size / P2_zvf_falsification_note pass (2026-10-18)

Rule 6: these files are shared; the coordinator applies changes centrally.

## 1. `platform_hybrid/paper/tikz/reward_flow.tex` — stabiliser missing from Fig. 2 (P2 #102, P3 #180)

The P2 6c172429 DISPOSITION lists "σ_r+ε in Fig. 2" as done, but the committed TikZ source still has no ε.
The runner and the text (`_shared_methods.tex`, Hyperparameter Mapping) use `(r_i - μ_G)/(σ_G + 1e-8)`; the
P3 sweep uses `+1e-6`. Proposed change (line ~122 and ~124):

- old: `$\hat{A}_i = \dfrac{r_i - \bar{r}}{\sigma_r}$`
- new: `$\hat{A}_i = \dfrac{r_i - \bar{r}}{\sigma_r + \epsilon}$`
- old: `where $\bar{r}=\mathrm{mean}(\{r_i\}),\;\sigma_r=\mathrm{std}(\{r_i\})$`
- new: `where $\bar{r}=\mathrm{mean}(\{r_i\}),\;\sigma_r=\mathrm{std}(\{r_i\})$ (population), $\epsilon>0$ small; a zero-variance group gets $\hat A_i=0$`

Reason: at σ_r = 0 (the ZVF condition) the printed formula is 0/0.

## 2. `platform_hybrid/paper/sections/_shared_methods.tex` — optional Bonferroni MDE (P2 #81)

If a fixed Bonferroni family is ever reported again, the MDE at α = 0.01, n1 = n2 = 5, power 0.8 is
d = 2.744 (`statsmodels TTestIndPower.solve_power(nobs1=5, alpha=0.01, power=0.8, ratio=1)`; the
reviewer's 2.68 is not reproduced). The current text declares no single family size, so no change is required.

## 3. `platform_hybrid/paper/tikz/{taxonomy,pipeline}.tex` — label size (P2 #106, #107)

Both figures are vector TikZ (resolution is not an issue). If the coordinator wants larger labels, raise the node
font from `\tiny`/`\scriptsize` (pipeline.tex: 4× `\tiny`, 2× `\scriptsize`; taxonomy.tex: 1× `\tiny`,
2× `\scriptsize`) to `\scriptsize`/`\footnotesize`; no content change.

**Status: done** (branch `fix/shared-tikz-figures`). Labels were raised to `\scriptsize`/`\footnotesize` and the PDFs are now tracked. P2 #106 and #107 are fixed.

## Note for the build

`*.pdf` is git-ignored, so `tikz/*.pdf` and any `figures/*.pdf` are absent from clean worktrees. The shared
preamble silently replaces a missing graphic with a "[missing figure artifact]" box instead of failing. In this pass
P2 Fig. 5 (`zvf_scaling_cross_pillar.pdf`) was missing from every clean checkout; it is now a tracked PNG
regenerated from `experiments/results/zvf_scaling_cross_pillar.tsv` by `tools/review_ledger/P2_zvf_scaling_figure.py`.
Recommend grepping built PDFs for "missing figure artifact" as part of the gate.
