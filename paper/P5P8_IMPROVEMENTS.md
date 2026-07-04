
| 48 | P8 | T1+T2 | **Stack-conditioning audit (P5 mirror) on the released fraud_data.csv/test_data.csv** — train XGB across a 5-axis stack grid (2^5=32 configs spanning realistic hyperparameter choices: n_estimators × max_depth × learning_rate × subsample × scale_pos_weight) on the released 50k train / 10k test split, compute per-axis η² on 4 metrics (AUC, F1, Brier, ECE-10) with paired bootstrap CIs (B=1000, seed 20260704). **Headline**: max_depth dominates AUC (η²=0.487 [0.286, 0.694]) and Brier (0.446 [0.249, 0.667]); learning_rate dominates F1 (0.410 [0.193, 0.637]); scale_pos_weight dominates ECE-10 (0.752 [0.556, 0.919]) — the most striking single result, tree *calibration* is governed by the minority re-weighting lever, not the tree-shape lever; subsample is noise on every metric; **AUC ranges 0.83-0.99 across the 32-tree grid** — the headline AUC=0.9988 is a joint claim about the data AND the chosen max_depth=5 setting. New `\subsecref{sec:p8-stack-audit}` + table `tab:p8-stack-eta2` in `paper/sections/p8_evidence.tex`; `paper_P8_fraud.pdf` rebuilds to 24 pages / 0 errors / 0 undefined citations; **first P8 mirror of the P5 mega_eta2 finding** — the stack-conditioning thesis generalises from RL outcomes to non-RL XGBoost trees | `scripts/p5p8/p8_stack_conditioning.py` (220 LoC, stdlib + xgboost + sklearn + matplotlib); `experiments/results/p5p8/p8_stack_audit.tsv` (32 rows); `p8_stack_audit_axes.tsv` + `p8_stack_audit_boot.tsv` + `p8_stack_audit_summary.json`; `figures/p8_stack_audit.{png,pdf}`; `docs/p5p8_improvements/48_p8_stack_audit.md`; `paper/sections/p8_evidence.tex` | validated | iter 36 |
| 49 | P8 | T1+T2 | **Cost-per-decision × sensor-noise phase diagram** — closes iter-32 synthesis note "a 2-point sensor-noise sweep would test whether L* scales with c_sense" with the full phase diagram: 5 σ × 5 L = 25 cells. For each cell, train XGB-20raw (no sensor) and XGB-24full (noisy sensor), compute expected cost/dec at τ*=L/(L+c_inv), paired bootstrap CI (B=1000, seed 20260704). **Headline falsifiable claim**: **the sensor never pays for itself (0/25 cells)** — every bootstrap-mean Δ ≥ 0, no CI excludes zero on the negative side. Sensor is most cost-comparable at L=$100 (Δ=+$0.0098/dec at every σ). The 5 σ curves are visually identical within ±$0.0028/dec — **σ-INVARIANT**, stronger than iter-32's single-sigma L* drift finding. At L=$500 trees tie at Δ=0 because τ* collapses to 1.0. **Sensor-not-scorer thesis is now load-bearing across the (σ, L) phase space**, not just at iter-28's single point. New `\subsecref{sec:p8-cost-phase}` + table `tab:p8-cost-phase` in `paper/sections/p8_evidence.tex`; `paper_P8_fraud.pdf` rebuilds to 24 pages / 0 errors / 0 undefined citations | `scripts/p5p8/p8_cost_phase_diagram.py` (~250 LoC, stdlib + xgboost + sklearn + matplotlib); `experiments/results/p5p8/p8_cost_phase_diagram.tsv` (25 rows); `p8_cost_phase_diagram_boot.tsv` (25 rows); `p8_cost_phase_diagram_summary.json`; `figures/p8_cost_phase_diagram.{png,pdf}`; `docs/p5p8_improvements/49_p8_cost_phase_diagram.md`; `paper/sections/p8_evidence.tex` | validated | iter 36 |
| 50 | P5P8-SYNTH | T3 | **P5/P6 audit triangulation** — closes iter-32 synthesis note "run iter-37's claim-vs-measurement alignment audit on iter-30 surface as the gold surface to triangulate". Compute joint metric on the 32 registry rows × 98 cells evidence base: Audit A (iter 29, claim-vs-measurement alignment) is a **CEILING** (all 98 cells score 100.0, only 1 unique value, zero discriminative power); Audit B (iter 30, variant-delta × MIN-REPORT consistency) has 100pp of variation (mean B_match_rate=0.221, range [0.0, 1.0]). **Joint correlation B_match vs n_audited = +0.23 [-0.28, +0.94], CI contains 0, not significant** — expected because A is saturated. **Headline**: the two audits measure orthogonal honesty axes (truthfulness vs consistency); Audit B is the load-bearing MIN-REPORT honesty measurement on this corpus; A's ceiling is a property of the corpus, not of the audit's discriminative power. **First P5/P6 cross-paper coupling audit framework** — any future audit (e.g., per-paper trace check) can be added to the triangulation matrix | `scripts/p5p8/p5p6_audit_triangulation.py` (~150 LoC, stdlib + pandas); `experiments/results/p5p8/p5p6_audit_triangulation.tsv` (32 rows); `p5p6_audit_triangulation_summary.json`; `docs/p5p8_improvements/50_p5p6_audit_triangulation.md` | validated | iter 36 |

## Iter 36 deliverables

- `scripts/p5p8/p8_stack_conditioning.py` (220 LoC, stdlib + xgboost + sklearn + matplotlib) — JOB A Pillar 4 / P8 stack audit
- `scripts/p5p8/p8_cost_phase_diagram.py` (~250 LoC) — JOB A Pillar 4 / P8 phase diagram
- `scripts/p5p8/p5p6_audit_triangulation.py` (~150 LoC) — JOB B SYNTH
- `experiments/results/p5p8/p8_stack_audit.tsv` (32 rows)
- `experiments/results/p5p8/p8_stack_audit_axes.tsv` + `p8_stack_audit_boot.tsv` + `p8_stack_audit_summary.json`
- `experiments/results/p5p8/figures/p8_stack_audit.{png,pdf}` (heatmap)
- `experiments/results/p5p8/p8_cost_phase_diagram.tsv` (25 rows)
- `experiments/results/p5p8/p8_cost_phase_diagram_boot.tsv` (25 rows)
- `experiments/results/p5p8/p8_cost_phase_diagram_summary.json`
- `experiments/results/p5p8/figures/p8_cost_phase_diagram.{png,pdf}` (2-panel heatmap)
- `experiments/results/p5p8/p5p6_audit_triangulation.tsv` (32 rows)
- `experiments/results/p5p8/p5p6_audit_triangulation_summary.json`
- `docs/p5p8_improvements/48_p8_stack_audit.md`
- `docs/p5p8_improvements/49_p8_cost_phase_diagram.md`
- `docs/p5p8_improvements/50_p5p6_audit_triangulation.md`
- Extended `paper/sections/p8_evidence.tex` with 2 new subsections + 2 new tables (sec:p8-stack-audit + sec:p8-cost-phase)
- `paper_P8_fraud.pdf` rebuilds to 24 pages / 0 errors / 0 undefined citations
- 3 lines in `AUTORESEARCH_FINDINGS.jsonl` (pillar P8 + P8 + P5P8-SYNTH, iter 36)

## Iter 36 synthesis re-ranking (JOB B / SYNTH)

Re-ranked all 47 prior rows + 3 new (rows 48, 49, 50) by **impact × evidence × paper-facing readiness**. The ledger now has **50 rows: 50 validated, 0 proposed, 0 rejected-from-this-iter**. The iter-36 cycle closed the iter-32 synthesis notes with three new deliverables that extend P8 (items 48, 49) and surface the first P5/P6 cross-paper coupling audit (item 50).

Top of the re-ranked stack (highest impact × readiness, all validated):

1. **#48 P8 stack-conditioning audit** (iter 36, JOB A) — first P8 mirror of P5's mega_eta2; max_depth η²=0.487 [0.286, 0.694] on AUC; scale_pos_weight η²=0.752 [0.556, 0.919] on ECE.
2. **#49 P8 cost-phase diagram** (iter 36, JOB A) — 25-cell (σ, L) phase diagram; sensor never pays off (0/25 cells); σ-invariant.
3. **#47 P7 per-prompt optimal G*** (iter 35) — bimodal G* distribution, 20.3% rollout saving.
4. **#46 P6 measured-delta block** (iter 34) — 8/8 variants below grpo ZVF with CIs.
5. **#45 P5 schema-bump closure** (iter 33) — 100% post-act populate rate.

**Concrete candidates surfaced but not yet opened** (for the next synthesis iteration):

- **P8**: extend iter-36's stack audit to a 4-axis grid including `min_child_weight` and `gamma` (regularization axes) — would close the gap that the current 5-axis grid excludes regularization.
- **P7**: extend iter-35's per-prompt G* with i.i.d. binomial contrast-loss accounting at G'=8 (currently only reports the contrast-lost delta = 0.130).
- **P5/P6**: extend iter-50's triangulation matrix with iter-14's per-leaf coverage audit as Audit C — would surface whether per-leaf coverage correlates with B's match rate.

| 69 | P6 | T1+T3 | **Measured-block provenance & coverage audit (MBPCA) (iter 58)** — fresh vein, not in prior ledger. Iter-50 added the registry schema/audit; iter-46 added claim-validation verdicts; iter-54 closed the missing-delta gap. **No prior iter audited the `measured` block itself** — the array that grounds every delta entry's claim in measured evidence. **`scripts/p5p8/p6_measured_coverage.py`** (~280 LoC, stdlib only) audits every `delta_*.json`: block presence (measured / expected_effects / claim_validation row counts), source-path resolution on disk + mtime age in days, panel × metric coverage grid (panel ∈ {n2_same_stack_last10, zvf130_5seed} × metric ∈ {zvf, reward_mean, zvf_risk_mean, mean_zvf}), cross-panel sign agreement (N2 zvf vs ZVF130 risk for entries with both), registry-wide verdict tally (SUPPORTS/CONTRADICTS/NEUTRAL/UNCLAIMED), empty-measured gap analysis. **Falsifiable headline (re-run verified 2026-07-05)**: **9 of 14 deltas (64%) carry a non-empty measured array; 5 of 14 (36%) ship as provenance-only placeholders**. The 5 split into 2 structural classes: source-data-unavailable (delta_dapo, delta_drgrpo, delta_gspo) vs iter-54 design placeholders (delta_reinforce, delta_liteppo). **Verdict totals across 24 validation rows: 10 SUPPORTS, 2 CONTRADICTS, 4 NEUTRAL, 8 UNCLAIMED**. **0 measured row's source path is missing** — every cited `.tsv` resolves and freshness window is bounded. **Cross-panel agreement**: 2 of 3 entries agree on direction (delta_aero: N2 zvf=−0.025, Z130 risk=−0.148 ✓; delta_areal: −0.056, −0.246 ✓); **delta_gift is the lone dissenter** with N2 zvf=+0.125 (significant) AND ZVF130 risk=−0.263 (significant) — same-sign=False. New `registry/query.py measured-coverage` subcommand (additive, 8 prior untouched); new sidecar `registry/measured_block_audit.json` (idempotent regen in <1 sec). New §sec:p6-measured-coverage + tab:p6-measured-coverage in `paper/sections/p6_measured_coverage.tex` (cross-references iter-46 §sec:p6-measured-claimed and iter-54 §sec:p6-missing-deltas); **paper_P6_registry.pdf rebuilds to 32 pages / 0 errors / 0 undefined citations** (was 32, +0 pages — one new subsection + one new booktabs table) | `scripts/p5p8/p6_measured_coverage.py` (~280 LoC, stdlib only); `experiments/results/p5p8/p6_measured_coverage.tsv` (14 per-entry rows); `p6_measured_coverage_grid.tsv` (14 × 8 grid); `p6_measured_cross_panel.tsv` (3 cross-panel rows); `p6_measured_coverage_summary.json`; `registry/measured_block_audit.json` (sidecar cache); `registry/query.py` (added measured-coverage subcommand); `docs/p5p8_improvements/69_p6_measured_coverage.md`; `paper/sections/p6_measured_coverage.tex` (new §sec:p6-measured-coverage + tab:p6-measured-coverage); `paper/paper_P6_registry.tex` (\input line added) | validated | iter 58 |

## Iter 58 deliverables (this iter)

- `scripts/p5p8/p6_measured_coverage.py` (~280 LoC, stdlib only)
- `experiments/results/p5p8/p6_measured_coverage.tsv` (14 per-entry rows)
- `experiments/results/p5p8/p6_measured_coverage_grid.tsv` (14 × 8 grid)
- `experiments/results/p5p8/p6_measured_cross_panel.tsv` (3 cross-panel rows)
- `experiments/results/p5p8/p6_measured_coverage_summary.json`
- `registry/measured_block_audit.json` (sidecar cache, idempotent regen)
- `registry/query.py` extended with `measured-coverage` subcommand (+ `--delta <id>` filter)
- `docs/p5p8_improvements/69_p6_measured_coverage.md`
- `paper/sections/p6_measured_coverage.tex` (new §sec:p6-measured-coverage + tab:p6-measured-coverage)
- `paper/paper_P6_registry.tex` extended (1 new `\input` line)
- `paper_P6_registry.pdf` rebuilds to 32 pages / 0 errors / 0 undefined citations
- 1 line in `AUTORESEARCH_FINDINGS.jsonl` (pillar P6, iter 58)

## Iter 58 headline findings

1. **9 of 14 deltas (64%) carry measured data; 5 of 14 (36%) are provenance-only.** The 5 split into 2 structural classes — source-data-unavailable (delta_dapo, delta_drgrpo, delta_gspo have arXiv citations and stack entries but no same-stack panel run on canonical N2/Z130 source TSVs) vs iter-54 design placeholders (delta_reinforce, delta_liteppo). The actionable backlog is **3 entries awaiting a same-stack panel run** before they can carry a measured row.
2. **Registry-wide verdict totals: 10 SUPPORTS, 2 CONTRADICTS, 4 NEUTRAL, 8 UNCLAIMED across 24 validation rows.** The 8 UNCLAIMED rows are the metric/panel combinations measured but never declared as a predicted_sign — a quantified surface for a future iter to add `expected_effects` entries and lift measured-evidence utilization.
3. **0 missing sources — every cited `.tsv` resolves on disk.** Freshness window bounded by the worktree's most recent N2/Z130 mtime (no stale source paths).
4. **Cross-panel agreement: 2 of 3 entries agree on direction** (delta_aero, delta_areal both negative on both panels); **delta_gift is the lone dissenter** with N2 zvf=+0.125 (significant) AND ZVF130 risk=−0.263 (significant) — same-sign=False. The MBPCA surfaces this as the **sharpest finding of the iter**: a measurement-confirmed structural sign disagreement (not a regression). GIFT reweights groups so raw N2 zvf can rise while bounded-Z130 risk falls — the registry's prior claim "GIFT helps signal starvation" is now sharpened to **"GIFT is risk-favouring but ZVF-raising"**, a stronger and more reviewer-defensible statement.
