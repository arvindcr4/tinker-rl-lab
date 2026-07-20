# Program-wide manuscript audit (2026-07-14)

## Corpus and method

- Canonical roster: 17 existing documents plus one new GRPO/PPO/SAO synthesis.
- Expanded source roots read: 18.
- Compiled PDFs inventoried: 17 existing PDFs plus the new synthesis PDF.
- Total compiled corpus at baseline: 757 pages, approximately 510k extracted
  words. The long compendium reuses large parts of P1--P4.
- Direct reads covered each root's title, abstract, introduction, claim hierarchy,
  results framing, limitations, conclusion, and all high-risk sections identified
  by a program-wide phrase scan. Included section trees and bibliographies were
  expanded by `inventory_papers.py`.
- A requested Groq review pass was attempted with the mandated
  `kimi-k2-0905-preview` model. Groq returned HTTP 404 (model unavailable), so no
  substitute model was used. The audit continued deterministically from source,
  PDFs, and checked-in result artifacts.
- Later bounded second-opinion passes used the locally configured exact models
  `kimi-code/kimi-for-coding` through `kimi -p` and `glm-5.2` through `zai -p`.
  Both independently returned no high-confidence contradiction across the five
  targeted evidence boundaries: PPO/SAO claim status, G=32 provenance, P7
  controller status, the confounded 17x stack comparison, and MIN-REPORT's 7+1
  mapping. Prompts, execution notes, and reconciled results are recorded in
  `KIMI_REVIEW.md` and `DUAL_MODEL_REVIEW.md`.
- A final deterministic closure review expanded every root and inspected all
  328 unique included source files (56,658 lines; 3.02 MB). It checked labels,
  citation keys, input hooks, TODO/placeholder markers, claim-risk phrases, and
  exact content hashes. Results are in `FILE_REVIEW.tsv`,
  `SELF_REVIEW_FLAGS.md`, and `self_review_summary.json`.

## Canonical evidence hierarchy

1. Binary GRPO accounting: `pass@G - p^G = 1 - ZVF`, verified on 505 unique
   prompt-seed tasks to maximum error `1.11e-16`.
2. Matched-token group-size panel: `G=2 x 160` versus `G=16 x 20`, 2,560
   rollouts per arm, two seeds. Small G reaches the all-correct ZVF wall; large G
   remains mid-learning. This is a trajectory result, not a universal optimum.
3. Conditional utility: `(1-ZVF)/sqrt(G)` selects G=4 on the 505-task cohort;
   G4-G5 = `0.001767`, bootstrap CI `[0.000818, 0.002704]`. The optimum is tied
   to this objective and cohort.
4. Controller boundary audit: 1,867 fires in 2,560 observations; 1,723 (92.3%)
   are all-correct and 144 are all-wrong. An all-wrong-only rule would remove
   92.3% of fires, but performance preservation is unmeasured.

## High-confidence corrections applied

- Recast the 17x stack comparison as an under-specification exhibit because the
  managed arm silently used a different base checkpoint. Removed backend-only
  causal wording across P5, P6, P7, and the condensed position paper.
- Reconciled MIN-REPORT-RL: eight reporting items = seven run-manifest fields +
  held-out pass@k reporting. The registry schema remains intentionally seven
  run-start fields.
- Replaced the P7 cross-prompt ``empirical G'`` claim. Pooling rewards from
  different prompts is now a negative-control stress test, not a larger
  within-prompt GRPO group or controller prescription.
- Recast P7 iter-199 as a frozen-p diagnostic projection. It is no longer
  presented as closed-loop learning or empirical dominance.
- Recast R05 theory around calibration and limits. Its declared proxy objective
  yields the prior-independent tie G in {2,3}; it cannot justify adaptive G.
- Marked all P3 G=32 values by provenance: directly measured arms where they
  exist, reconstructed/fitted grids where they do not. Removed universal G=32
  guidance.
- Repaired the two P3 figure generators so they read the current canonical
  token-normalized artifact and write into the paper figure directory. Figure
  labels now say ``reconstructed'' or ``illustrative'' where appropriate.
- Repaired two venue-variant figure paths, restored the shared anonymous ethics
  appendix, and isolated N01's bibliography search path from the umbrella
  paper's `main.bbl`.
- Fixed two visible layout defects: P5's four-term entropy equation and P6's
  wide zero-evidence table/command block now fit within the page.
- Added explicit evidence-reuse notices to the long compendium and new synthesis
  so venue variants are not counted as independent replications.
- Parked P8 outside the ZVF thesis and added an actual conclusion boundary.

## Per-document revision matrix

| ID | Root | Unique role | Principal revision | Remaining gate |
|---|---|---|---|---|
| P01 | `platform_hybrid/paper/paper_P1_scaling.tex` | scaling identifiability | title and introduction now lead with limits rather than a positive law | matched multi-seed cross-scale study |
| P02 | `platform_hybrid/paper/paper_P2_zvf.tex` | ZVF diagnostic | added exact pass@G/ZVF accounting and bounded interpretation | direct gradient geometry beyond proxy |
| P03 | `platform_hybrid/paper/paper_P3_group_size.tex` | group-size evidence | separated measured G<=16 results, conditional G=4 utility, and reconstructed G=32 hypotheses | direct token-matched G sweep |
| P04 | `platform_hybrid/paper/paper_P4_length_bias.tex` | capped null test | title and introduction make the 200-token estimand explicit | uncapped/long-horizon mediation experiment |
| P05 | `platform_hybrid/paper/paper_P5_minreport.tex` | reporting-standard evidence | fixed checkpoint/backend causality and 7+1 schema wording | external corpus and released toolchain |
| P06 | `platform_hybrid/paper/paper_P6_registry.tex` | machine-readable registry | separated seven run fields from eighth evaluation item; fixed stack claim | external entries and schema adoption |
| P07 | `platform_hybrid/paper/paper_P7_zvf_controller.tex` | controller audit/proposal | removed invalid G' and closed-loop claims; added 92.3% boundary asymmetry | fixed-token bakeoff vs static G16 and naive rules |
| P08 | `platform_hybrid/paper/paper_P8_fraud.tex` | cross-domain side study | explicitly parked outside RL thesis; future work now concludes with evidence boundary | real/cross-institution fraud evaluation |
| R01 | `platform_hybrid/paper/acm_main.tex` | compact cross-library paper | scoped as venue derivative and corrected reproducibility wording | align with current artifact release |
| R02 | `platform_hybrid/paper/neurips_2026_variants/main_zvf.tex` | focused sentinel paper | added exact accounting and companion-scope firewall | within-task multi-seed prediction audit |
| R03 | `platform_hybrid/paper/neurips_2026_variants/main_workshop.tex` | artifact note | replaced universal-G framing with objective/budget-dependent question | open-stack token-matched sweep |
| R04 | `platform_hybrid/paper/neurips_2026_variants/main_dnb.tex` | Datasets & Benchmarks artifact | separated analytic from multi-seed A-grade evidence; reduced causal wording | artifact completeness and current URLs |
| R05 | `zvf-program/theory/zvf_theory.tex` | conditional theory | discharged the enumerated proof gaps, added an exact Clopper--Pearson bound, and proved the proxy optimum | prospective empirical validation; richer control objective |
| R06 | `zvf-program/position/min_report_rl.tex` | condensed position | corrected 17x interpretation, 7+1 standard, authorship, audit rules, and released-vs-planned tooling claims | controlled audit; external adoption submissions |
| R07 | `zvf-program/registry/grpo_registry.tex` | living catalog | aligned variant definitions with primary sources and distinguished implemented commands from planned adapters | external entries, unknown-field backfill, schema adoption |
| R08 | `zvf-program/audit/reproducibility_audit.tex` | survival protocol | added frozen machine contracts and a fail-closed aggregator; pilot remains descriptive | full open-stack multi-seed audit |
| U01 | `platform_hybrid/paper/main.tex` | long compendium | labeled non-independent compendium; removed unsupported group-size mechanism | split/archive the 239-page iteration history |
| N01 | `platform_hybrid/paper/unified_signal_starvation/main.tex` | GRPO/PPO/SAO synthesis | added executable PPO/SAO gates, PAM/GSR/EGM/root-ZUF metrics, trace validation, tests, and a frozen contract | execute matched-budget PPO/SAO runs |

## Program-level submission advice

- Primary thesis vehicle: P02 + matched-token P03 result + P05/P06 methods
  postmortem, with R05's proved calibration results kept within their explicit
  Bernoulli and proxy-objective assumptions.
- Best near-term paper: focused ZVF sentinel/stratification scope (R02), unless
  the adaptive-controller bakeoff succeeds.
- Do not submit P07 as an adaptive-control win before the fixed-token bakeoff.
- Do not present R08 placeholder tables as results.
- Treat U01 as an internal compendium and source archive, not a 237-page venue
  manuscript.
- Keep N01 as a methods paper until PPO and SAO outcomes exist.

## Reproducibility artifacts

- `inventory.tsv`: canonical root/page/source inventory.
- `include_map.json`: expanded TeX dependency graph.
- `similarity.tsv`: cross-document text overlap.
- `reading_packet.md`: extracted reading packet.
- `high_risk_claims.txt` / `high_risk_claims_after.txt`: before/after phrase
  scans.
- `audit_papers.py`: Groq audit driver retained with the mandated model.
- `corpus.sha256`: baseline corpus fingerprint.
- `KIMI_REVIEW.md` / `DUAL_MODEL_REVIEW.md`: exact-model bounded review records.
- `FILE_REVIEW.tsv`: one row for every unique included source file.
- `SELF_REVIEW_FLAGS.md` / `self_review_summary.json`: deterministic review
  findings and corpus totals.

## Deduplication and integrity pass

- Deleted 44 tracked legacy or duplicate sources, including 23 repeated
  group-size iteration sections, two analogy-only P3 sections, duplicated TikZ
  figures, anonymous/venue section copies, and the venue-local bibliography.
- Reduced the active closure from 353 to 328 unique source files and removed
  9,842 lines overall while preserving every canonical root.
- Parameterized the remaining shared venue content and redirected all variants
  to the canonical bibliography instead of retaining divergent copies.
- Final exact-hash scan over manuscript source, bibliography, and Markdown files
  found zero duplicate groups. All 18 roots have zero missing input hooks, zero
  unresolved citations, and zero duplicate active labels.

## Final verification

- All 18 canonical roots compile successfully with TeX Live 2026.
- Final build corpus: 864 rendered pages, including deliberate overlap among
  the umbrella compendium and venue derivatives.
- Final canonical-root logs contain no fatal TeX errors, undefined citations,
  or undefined cross-references.
- Source placeholder strings are dormant figure fallbacks or explicit evidence-
  gap prose; a rendered scan finds zero active figure fallbacks. Active gates
  are the unexecuted R08/M-GRPO and PPO/SAO experiments, public adoption
  submissions, and publication identifiers; none is an unresolved citation,
  proof hole, or hidden build failure.
- Stable new-paper artifact:
  `output/pdf/signal-starvation-grpo-ppo-sao.pdf` (11 pages).
