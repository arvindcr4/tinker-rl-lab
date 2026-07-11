# Per-Pillar Standalone Papers (P1–P8)

## Consolidation plan v3 — THESIS-FIRST (2026-07-11, after adversarial council review)

v2's "2+1" was reviewed adversarially by a three-model council (Gemini 3.1 Pro,
Grok, Kimi). All three independently converged on the same three objections:
(1) the 2+1 plan is a publication plan, not a degree-completion plan — its
evidence-upgrade list is ~6 months of funded work for a solo researcher;
(2) the flagship's adaptive-controller headline is pre-undermined by our own
E-R2b finding (static G=16 avoids the all-correct ZVF wall at the same rollout
budget) and by the closed Tinker/LoRA-only kernel; (3) the four-algorithm
survival audit is non-credible weeks after the unwired-loss-flag incident and
structurally hypocritical on a closed API. Full reviews: session scratchpad
`council_{agy,grok,kimi}.out`.

**v3 priority order:**

1. **M.Tech thesis (primary deliverable).** One bounded contribution: *ZVF as
   a cheap online diagnostic for signal starvation in group-relative RL, plus
   group-size-under-budget failure modes* — honestly scoped to
   Qwen3-8B / GSM8K / Tinker-LoRA, 1–3 seeds, with explicit non-claims.
   The 8 pillars map to chapters. The loss-flag incident becomes a
   methods/reproducibility chapter (postmortem + 8-item checklist + registry),
   NOT an archival standards submission.
2. **Paper 1 (post-degree, gated).** Flagship only if a pre-registered,
   compute-matched bakeoff shows the ZVF-triggered controller strictly beats
   static G=16 AND naive non-ZVF heuristics on learning or generation-compute
   cost (e.g., stays at G=2 for most of training, spikes late). If the bakeoff
   fails or is unaffordable: submit the descriptive-diagnostic scope instead
   ("ZVF tracks saturation/collapse under small G; large G mitigates it at
   fixed budget") to a workshop/short-paper venue.
3. **Paper 2 (post-degree, gated harder).** Survival audit only after (a) an
   open-stack implementation (not Tinker), (b) differential-objective test
   tooling that would have auto-caught the unwired-flag bug, (c) one method at
   a time. Until then the checklist/registry live in the thesis chapter.
4. **RL-Finetuning Bench:** versioned artifact + technical report (unchanged
   from v2). No archival claim.

Data firewall, pair canonicalization, and the v2 vehicle map below remain the
reference for which text lives where; v3 changes *sequencing and claims*, not
file ownership.

## [SUPERSEDED by v3 above] Consolidation plan v2: 17 documents → 3 submission vehicles (2026-07-11, revised after external Deep Think review)

Overall verdict from the review: proceed with revisions. The revisions below are
incorporated. Execution order: **A → B → C**, and A's terminology/notation
(ZVF bounds, G\*) must be frozen before drafting B and C.

**Vehicle A — Flagship (main track): "ZVF: Diagnostic → Theory → Controller"**
Narrative order is *diagnostic-led* (problem → solution), per review — not
controller-led as originally planned.
- Part I (diagnostic): `paper_P2_zvf` + the stratified-audit framing of
  `neurips_2026_variants/main_zvf.tex` (retired after merge)
- Part II (theory): `zvf-program/theory/zvf_theory.tex` (T1–T3, empirical validation)
- Part III (controller): `paper_P7_zvf_controller` (engineering, design rules)
- Section donations: `paper_P3` (group-size / contrast-density results incl. the
  matched-budget G2-vs-G16 panel), `paper_P4` (loss-form robustness: GRPO vs
  Dr.GRPO no-signature result)
- **Data firewall**: A exclusively owns the scientific claims and figures of the
  E-R2b (G2-vs-G16) and E-P4 (Dr.GRPO) runs. B and C cross-cite A; they do not
  republish these figures.

**Vehicle B — Position paper: "Report the Stack, Not the Label"**
Tight policy paper (review: the earlier plan was overstuffed).
- Base: `paper_P5_minreport` (eight-item standard, evidence corpus, threat model,
  toolchain)
- Appendix/linked artifact: `paper_P6_registry` (machine-readable registry)
- `zvf-program/position/min_report_rl.tex` = condensed statement until submission,
  then retires; `zvf-program/registry/grpo_registry.tex` retires
- The Survival Audit does NOT go here (moved to C, per review)

**Vehicle C — Artifact/benchmark: "RL-Finetuning Bench"**
- Base: `neurips_2026_variants/main_dnb.tex` — with the ZVF diagnostic narrative
  scrubbed out (cross-cite A instead), per review, so C does not cannibalize A
- Flagship use-case: `zvf-program/audit/reproducibility_audit.tex` (single-stack
  survival protocol — re-implement DAPO/GSPO/Dr.GRPO/MAD-GRPO on the bench and
  measure which claimed gains survive)
- Results appendix: `paper_P1_scaling`'s scaling analysis
- Retire: `main.tex` (734 errors) — after a detex/visual-diff salvage pass to
  confirm no methodology text or bib entries were stranded; `acm_main.tex` and
  `main_workshop.tex` regenerate from C on demand

**Thesis chapters (explicitly assigned, so nothing is orphaned):** P3's
bridge-to-DPO remnant, P4's held-out-generalization remnant, P1's full scaling
treatment, and `paper_P8_fraud` (parked; different program).

## Canonical map vs. the zvf-program / benchmark-variant roster (2026-07-11)

The repo carries two drafting tracks that overlap: this P1–P8 pillar series,
and a roster of benchmark variants + ZVF-program drafts ("R1–R8"). Canonical
status per duplicate pair (reciprocal scope notes are embedded in each tex):

| Concept | Canonical | Companion (kept, scoped) |
|---|---|---|
| MIN-REPORT-RL standard | `paper_P5_minreport` (evidence corpus; **eight-item** standard as of 2026-07-11 — item 8 = pass@k curves, merged from the position draft) | `zvf-program/position/min_report_rl.tex` (community-position statement) |
| GRPO-Registry | `paper_P6_registry` (schema, population, measured-evidence tiers) | `zvf-program/registry/grpo_registry.tex` (living-catalog statement) |
| ZVF theory ↔ controller | split by scope: `zvf-program/theory/zvf_theory.tex` canonical for theorems (T1–T3), `paper_P7_zvf_controller` canonical for controller engineering | cross-referenced companions |
| ZVF diagnostic | `paper_P2_zvf` (pillar paper) | `neurips_2026_variants/main_zvf.tex` (NeurIPS variant framing) |
| Benchmark | `main.tex` (+ `acm_main.tex`) | `neurips_2026_variants/main_workshop.tex`, `main_dnb.tex` (venue variants); P1–P4 are per-pillar splits |
| Survival audit | `zvf-program/audit/reproducibility_audit.tex` (no P counterpart) | — |
| Fraud detection | `paper_P8_fraud` (no R counterpart) | — |

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
