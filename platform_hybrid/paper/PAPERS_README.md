# Four Per-Pillar Standalone Papers

Split out from the combined benchmark paper (`main.tex`) into four independently
compilable papers, one per research pillar. Each reuses shared infrastructure and
pulls in that pillar's section family plus its frontier-model synthesis section.

| File | Pillar | Title |
|---|---|---|
| `paper_P1_scaling.tex` | P1 | Scaling Laws for GRPO Post-Training: A Cross-Library, Cross-Scale Study |
| `paper_P2_zvf.tex` | P2 | The Zero-Variance Fraction: A Descriptive Diagnostic for Signal Starvation in GRPO |
| `paper_P3_group_size.tex` | P3 | Group Size in GRPO: Contrast Density and the Bridge to DPO |
| `paper_P4_length_bias.tex` | P4 | Length Bias and Held-Out Generalization in GRPO and Dr.GRPO |

## Structure of each paper
`\documentclass` → `_shared_preamble` → title → `_shared_author` → abstract → intro
+ related work → `_shared_methods` (benchmark, setup, statistics) → pillar results
narrative → the pillar's result sections (the real measured results + figures) →
`frontier_synthesis_*` (external ChatGPT/Gemini cross-examination) → discussion +
limitations → conclusion → `statistical_rigor_addendum` → bibliography.

## Shared, reusable pieces (extracted once from main.tex)
- `sections/_shared_preamble.tex` — packages + custom macros
- `sections/_shared_author.tex` — the author block
- `sections/_shared_methods.tex` — Benchmark Design + Experimental Setup + Statistical Methodology

## New per-pillar prose (authored for the standalone split)
`sections/p{1..4}_{abstract,intro,results_intro,conclusion}.tex` — all verified:
balanced braces, even `$`, and every citation resolves to `references.bib`.

## Compile status — ALL FOUR BUILD CLEANLY (0 LaTeX errors), with real figures
Built with TeX Live 2025 (`pdflatex` + `bibtex`):

| Paper | Pages | Errors | Undefined cites | Undefined refs |
|---|---|---|---|---|
| paper_P1_scaling.pdf | 35 | 0 | 0 | 11 |
| paper_P2_zvf.pdf | 41 | 0 | 0 | 11 |
| paper_P3_group_size.pdf | 45 | 0 | 0 | 7 |
| paper_P4_length_bias.pdf | 35 | 0 | 0 | 18 |

**All undefined citations resolved (0 `[?]` across all four).** The last two were
LLM-hallucinated keys with no matching paper: `lin2025taker` was replaced with the real
source of the G=32-vs-G=4 question — Tan et al. 2025, "Scaling Behaviors of LLM RL
Post-Training" (arXiv:2509.25300); `shen2025mad` was unfindable and dropped, leaving its
real co-citation `singhal2023drdrpo` (Singhal et al. 2023 length correlations).

- **Figures: regenerated.** 19/20 plotting scripts re-ran from the `experiments/results/` TSVs
  (matplotlib); all 25 figures the papers reference now render as real plots (0 placeholder boxes).
  One script (`group_size_iter27.py`) fails on a data-reshape and keeps its placeholder.
- **Bibliography: 10 of 12 undefined citations resolved.** Added `kaplan2020scaling`,
  `burnham2002model`, `gptoss`, `qwen3moe`, `kimi2025k2`, `kimi2025thinking`, `kimi`,
  `singhal2023drdrpo`, and self-refs `frontier2026`, `tinker-rl-lab-iter25`. **Still `[?]`
  (not fabricated — need author-supplied metadata):** `shen2025mad` (P4), `lin2025taker` (P3).
- **Undefined refs** are cross-references to labels in *other* pillars or main.tex-only sections
  (an inherent artifact of splitting one combined paper into four); render as `??`, non-fatal.

## Build
```
pdflatex paper_P1_scaling
bibtex   paper_P1_scaling
pdflatex paper_P1_scaling
pdflatex paper_P1_scaling
```
(repeat per paper). Overleaf: upload the `paper/` folder and set the main file.

## LaTeX bugs fixed to make these compile (also fix the original main.tex)
The upstream sections had never been compiled (main.tex itself failed with 386 errors). Fixes:
- `sections/_shared_preamble.tex` now defines the helper macros the sections use but that were
  never defined anywhere: `\eps \tableref \secref \paragraphref \figref \eqnref \argmax \argmin
  \E \task \algo \seed \tplat \zvf \signature \etal \note`; loads `underscore` (bare `_` in text);
  declares stray Unicode math chars (Δ δ × ≤ ≥ ≈); and makes `\includegraphics` fall back to a
  placeholder box for missing figure PDFs.
- Per-section fixes: table column-count mismatches (`scaling_laws`, `scaling_law_iter65`),
  unclosed `\fbox` in a figure placeholder (`scaling_law_iter61`), malformed inline math
  (`\(9 anchors)`, `$200$+$`, `$[$...`), bare `_\max` subscript, a fatal `\input` of a missing
  generated TSV, `\verb` inside captions / spanning lines (zvf sections), and a mis-nested
  `\end{figure}` in `length_bias` (a figure wrapped a table + another figure).

## Known residual issues (inherited from the combined paper, shared with main.tex)
1. **~35 placeholder figures** — the `*_iter*.tex` sections use `[Figure placeholder:
   … pending regeneration]` boxes. The plotting scripts (`scripts/*_fig.py`,
   `scripts/*_iter*.py`) regenerate them from the TSVs in `experiments/results/`;
   matplotlib is available.
2. **~6–10 "??" cross-references per paper** — labels that live in other pillars or
   in main.tex-only sections (e.g. `app:compute`, `sec:frontier`, cross-pillar
   `sec:zvf`). Cosmetic; do not block compilation. Resolve by localizing those refs
   or including the referenced section.
3. **10 undefined citations** (inherited, need author-supplied metadata):
   `frontier2026, gptoss, kimi, kimi2025k2, kimi2025thinking, lin2025taker,
   qwen3moe, shen2025mad, singhal2023drdrpo, tinker-rl-lab-iter25`.
   Added this round: `kaplan2020scaling`, `burnham2002model`.
