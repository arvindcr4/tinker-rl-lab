# Per-Pillar Standalone Papers (P1–P8)

## Program-wide consistency revision — 18 documents (2026-07-14)

The repository now contains 17 existing manuscripts plus the new synthesis
`unified_signal_starvation/main.tex`. They are **not 18 independent evidence
sources**: venue variants and the long compendium reuse runs, tables, and prose.
The canonical evidence hierarchy remains thesis-first:

1. Per-step ZVF is a mechanical diagnostic of contrastive group yield. For
   binary rewards, `pass@G - p^G = 1 - ZVF`; the 505-task audit reproduces the
   identity to `1.11e-16`.
2. The strongest group-size comparison is the matched-token, two-seed
   `G=2 x 160` versus `G=16 x 20` panel (2,560 rollouts per arm). It shows an
   all-correct ZVF wall for small `G`; it does not identify a universal optimum.
3. The conditional `(1-ZVF)/sqrt(G)` utility audit selects `G=4` on its
   505-task cohort. Fitted or reconstructed `G≈32` results use different
   objectives and budgets and are hypotheses, not contradictions or universal
   prescriptions.
4. Adaptive `G` is not promoted: 92.3% of logged escalation fires are on
   all-correct groups, the frozen-`p` replay is not closed-loop learning, and
   cross-prompt pooling is not a valid larger-within-prompt-group
   counterfactual. Promotion requires a seed-paired, fixed-token bakeoff against
   static `G=16` and naive boundary heuristics.
5. The `17x` same-label stack comparison also changed the base checkpoint. It
   is evidence of under-specification, not a backend-only causal effect.
6. MIN-REPORT-RL is an eight-item standard: seven run-manifest fields plus
   held-out pass@k reporting. The registry's run-start JSON intentionally has
   seven fields; pass@k belongs to the evaluation report.
7. PPO/SAO routing in `unified_signal_starvation` is a method proposal. Its GRPO
   facts reuse companion artifacts, and PPO/SAO benefits remain unmeasured.
8. P8 fraud is parked outside the ZVF thesis; it demonstrates measurement
   discipline but contributes no RL evidence.

Canonical document roster:

| ID | Source | Role after revision |
|---|---|---|
| P01 | `paper_P1_scaling.tex` | limits/identifiability audit, not a positive scaling law; workshop-short claim boundary fences multi-seed as non-claim |
| P02 | `paper_P2_zvf.tex` | descriptive ZVF diagnostic and exact accounting |
| P03 | `paper_P3_group_size.tex` | measured bounds plus explicitly reconstructed hypotheses |
| P04 | `paper_P4_length_bias.tex` | bounded null under a 200-token cap |
| P05 | `paper_P5_minreport.tex` | canonical evidence for the eight-item reporting standard |
| P06 | `paper_P6_registry.tex` | canonical seven-field registry; position-artifact resource (shared 17× exhibit is not re-claimed) |
| P07 | `paper_P7_zvf_controller.tex` | retrospective audit + prospective test plan only; adaptive G not promoted |
| P08 | `paper_P8_fraud.tex` | **ABSORBED → thesis** measurement-discipline appendix (parked; not a standalone venue paper) |
| R01 | `acm_main.tex` | **ABSORBED → R04** compact ACM regenerate of the tiered artifact vehicle |
| R02 | `neurips_2026_variants/main_zvf.tex` | **ABSORBED → P02** optional short venue vehicle of the ZVF diagnostic (not a second paper) |
| R03 | `neurips_2026_variants/main_workshop.tex` | exploratory artifact note |
| R04 | `neurips_2026_variants/main_dnb.tex` | tiered artifact paper; ZVF is bench instrumentation not flagship claim; absorbs R01 + U01 bench packaging |
| R05 | `zvf-program/theory/zvf_theory.tex` | calibration and reliability proof sketches; no adaptive set-point |
| R06 | `zvf-program/position/min_report_rl.tex` | **ABSORBED → P05** condensed position; retires at P05 submission |
| R07 | `zvf-program/registry/grpo_registry.tex` | **ABSORBED → P06** condensed catalog; retires at P06 submission |
| R08 | `zvf-program/audit/reproducibility_audit.tex` | completed 40-unit single-stack survival audit; open-stack generalization remains gated |
| U01 | `main.tex` | **ABSORBED → thesis / R04** long evidence bank; not a venue submission |
| N01 | `unified_signal_starvation/main.tex` | GRPO-grounded diagnostic + PPO/SAO evaluation contract; PPO/SAO outcomes non-claims |

### Absorption map (2026-08-02) — the former “other 6”

These six files remain on disk as satellites; they are **not** independent venue
counts. Evidence and claims live under the parent vehicle:

| Absorbed ID | Parent vehicle | Absorption rule |
|---|---|---|
| R02 | **P02** | Short stratified ZVF framing of the same diagnostic; submit P02 *or* R02, never both as independent papers |
| R06 | **P05** | Condensed community position of MIN-REPORT-RL; retire when P05 submits |
| R07 | **P06** | Condensed living-catalog of GRPO-Registry; retire when P06 submits (or ship as P05/P06 appendix) |
| R01 | **R04** | ACM-format regenerate of the tiered bench artifact; rebuild from R04 on demand |
| U01 | **thesis + R04** | Long evidence compendium for degree/thesis chapters and artifact packaging; not a conference paper |
| P08 | **thesis** | Parked cross-domain measurement side study; thesis methods/reproducibility appendix only |

**Independent venue-candidate set after absorption (12):** P01–P07, R03, R04, R05, R08, N01.  
Machine-checked in `platform_hybrid/paper/scripts/publication_worthiness_check.py`
(`ABSORPTION` / `OUT_OF_SET`).

The older plans below are retained as history. Where they disagree with this
section, this section is canonical.

**Post-E1 statistical correction (2026-08-02).** R08 is no longer a planned or
placeholder audit: all 40 frozen Qwen3-8B/GSM8K/Tinker units independently
validate. The original aggregator, however, used a large-sample MDE at eight
seeds and never called its Benjamini-Hochberg routine. Exact paired-t power puts
DAPO's MDE80 at 0.01012, above the 0.01 equivalence margin; no adjusted
difference test rejects. DAPO, GSPO, Dr.GRPO, and AERO are all `INCONCLUSIVE`.
This closes the bounded execution artifact but does not remove the main-track
gate below: a general survival claim still needs an open implementation,
objective-differential tests, longer learning curves, and prospective
external-validity evidence.

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
  survival protocol — re-implement DAPO/GSPO/Dr.GRPO/M-GRPO on the bench and
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

## Compile status — ALL FOUR BUILD CLEANLY
Reverified with TeX Live 2026 (`pdflatex` + `bibtex`):

| Paper | Pages | Errors | Undefined cites | Undefined refs |
|---|---|---|---|---|
| paper_P1_scaling.pdf | 45 | 0 | 0 | 0 |
| paper_P2_zvf.pdf | 44 | 0 | 0 | 0 |
| paper_P3_group_size.pdf | 24 | 0 | 0 | 0 |
| paper_P4_length_bias.pdf | 44 | 0 | 0 | 0 |

All active citation keys and cross-references resolve. Unverifiable inherited keys were
removed or replaced only after the cited primary paper was identified.

- **Figures:** active figure paths resolve, including the repaired group-size generators.
  Conditional fallback boxes remain in source for portability but are not rendered when the
  checked-in assets are present.
- **Bibliography:** all four roots use the canonical `references.bib`; no venue-local copy is
  needed.

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

## Remaining submission gates

The PDFs are mechanically clean, but the scientific gates remain: matched multi-seed
cross-scale evidence for P1, direct gradient geometry for P2, a direct token-matched
group-size sweep for P3, and an uncapped long-horizon mediation study for P4. These are
evidence limitations, not missing assets, references, or bibliography entries.
