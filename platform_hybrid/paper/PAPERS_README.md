# Active manuscript roster (P1-P12)

## Canonical publication queue — 12 active, 6 absorbed (2026-08-02)

The repository's publication queue contains 12 active roots. The earlier
18-file review snapshot remains useful as history, but six of those roots are
now absorbed into a parent paper or the thesis. They are not extra evidence or
extra submission vehicles.
The canonical evidence hierarchy remains thesis-first:

### Reviewer #36320 descendant boundary (2026-08-05)

`paper.tex` is a current local descendant of NeurIPS submission #36320, not
the authenticated reviewed PDF. Its correction manifest is
`REVIEWER_36320_CORRECTION_MANIFEST.md`. The 12-root roster below remains the
only live set; the six absorbed roots are audit-only history. The descendant
does not pool selected checkpoints, treats the Qwen PPO row as quarantined, and
keeps later per-run or single-stack evidence separate from the reviewed record.

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

### Pavlov's List usefulness gate for all future fine-tunes (2026-08-08)

The corrected papers may describe the historical GSM8K evidence within its
frozen boundary, but no new fine-tuned model may be presented as the program's
use-inspired or main-track model on GSM8K alone. New model training and primary
evaluation are governed by
[`../../zvf-program/flagship/PAVLOVS_LIST_TASK_CONTRACT.md`](../../zvf-program/flagship/PAVLOVS_LIST_TASK_CONTRACT.md)
and its machine-readable contract. The campaign maps the 53-company snapshot
to all 16 Pavlov domain tags and requires stateful tool, browser, artifact,
code, enterprise/finance, science, ML, security, chip-design, design, game, and
long-horizon task families. GSM8K is calibration-only and math is capped at 5%
of the training mixture.

This is a prospective experiment gate, not new evidence. Until the multi-domain
runs and held-out company-family evaluations exist, the papers must say that
the use-inspired model claim is untested. Domain-suite success would establish
task-family usefulness, not automatic readiness for every company's private
production workflow.

Canonical document roster (**active independent set only — 12**):

| ID | Source | Role after revision |
|---|---|---|
| P1 | `paper_P1_scaling.tex` | limits/identifiability audit, not a positive scaling law |
| P2 | `paper_P2_zvf.tex` | descriptive ZVF diagnostic and exact accounting; absorbs former R02 |
| P3 | `paper_P3_group_size.tex` | measured bounds plus explicitly reconstructed hypotheses |
| P4 | `paper_P4_length_bias.tex` | bounded null under a 200-token cap |
| P5 | `paper_P5_minreport.tex` | canonical eight-item reporting standard; absorbs former R06 |
| P6 | `paper_P6_registry.tex` | canonical seven-field registry; absorbs former R07 |
| P7 | `paper_P7_zvf_controller.tex` | retrospective audit and prospective test plan only |
| P8 | `neurips_2026_variants/paper_P8_workshop.tex` | exploratory workshop/artifact note; former R03 |
| P9 | `neurips_2026_variants/paper_P9_dnb.tex` | tiered artifact paper; absorbs former R01 and U01 packaging |
| P10 | `../../zvf-program/theory/paper_P10_zvf_theory.tex` | calibration and reliability proof sketches; former R05 |
| P11 | `../../zvf-program/audit/paper_P11_reproducibility_audit.tex` | bounded 40-unit single-stack audit; former R08 |
| P12 | `unified_signal_starvation/paper_P12_signal_starvation.tex` | GRPO diagnostic plus PPO/SAO evaluation contract; former N01 |

### Absorption map — independent versions **removed from live paths**

Former independent roots R01, R02, R06, R07, U01, and P08_fraud are **no longer live
manuscript paths**. They live only under `archive/absorbed/` for history:

| Former ID | Parent | Archive folder |
|---|---|---|
| R02 | **P2** | `archive/absorbed/R02_main_zvf/` |
| R06 | **P5** | `archive/absorbed/R06_min_report/` |
| R07 | **P6** | `archive/absorbed/R07_grpo_registry/` |
| R01 | **P9** | `archive/absorbed/R01_acm/` |
| U01 | **thesis + P9** | `archive/absorbed/U01_main_compendium/` |
| P08_fraud | **thesis** | `archive/absorbed/P08_fraud/` |

**Independent venue-candidate set: 12** (P1-P12).  
Do not count archived IDs as separate papers.  
Machine-checked in `platform_hybrid/paper/scripts/publication_worthiness_check.py`
(`ACTIVE_ROSTER` / `ABSORBED_ARCHIVED`).

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
| MIN-REPORT-RL standard | `paper_P5_minreport` (evidence corpus; **eight-item** standard as of 2026-07-11 — item 8 = pass@k curves, merged from the position draft) | former `min_report_rl.tex` → `archive/absorbed/R06_min_report/` |
| GRPO-Registry | `paper_P6_registry` (schema, population, measured-evidence tiers) | former `grpo_registry.tex` → `archive/absorbed/R07_grpo_registry/` |
| ZVF theory ↔ controller | split by scope: `zvf-program/theory/zvf_theory.tex` canonical for theorems (T1–T3), `paper_P7_zvf_controller` canonical for controller engineering | cross-referenced companions |
| ZVF diagnostic | `paper_P2_zvf` (pillar paper) | former `main_zvf.tex` → `archive/absorbed/R02_main_zvf/` |
| Benchmark | `neurips_2026_variants/main_dnb.tex` (R04) + workshop note R03 | former `main.tex` / `acm_main.tex` → `archive/absorbed/U01_*` / `R01_acm/` |
| Survival audit | `zvf-program/audit/reproducibility_audit.tex` (no P counterpart) | — |
| Fraud detection | thesis appendix only | former `paper_P8_fraud` → `archive/absorbed/P08_fraud/` |

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
