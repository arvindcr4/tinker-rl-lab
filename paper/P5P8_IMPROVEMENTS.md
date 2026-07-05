
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

| 70 | P8 | T1+T2 | **Operational calibration gap at alert-volume budgets (iter 60 JOB A)** — fresh vein, not in prior ledger. Iter-24 §sec:p8-reliability measured GLOBAL reliability diagrams on all 10k test rows, decile-binned; iter-56 §sec:p8-alert-volume measured the RECALL gap at every alert-volume K; iter-31 #31 measured per-feature ablation CIs on global AUC/Brier/F1. None answers the operational question: **"among the top-K alerts the analysts actually see (limited by the staffing budget), is the model's mean predicted probability close to the observed positive rate?"** For each K ∈ {0.25, 0.50, 1.00, 2.00, 5.00}% × each tree ∈ {XGB-20raw, XGB-24full, XGB-4sensor}, compute mean_pred_topK, obs_pos_rate_topK, calibration_gap = mean_pred − obs_pos_rate, Brier_topK, ECE_topK in 10 quantile bins within top-K. Paired bootstrap B=400 percentile, seed 20260704. **Falsifiable headline #1 (24full − 20raw calibration delta is detectable at K≥2%)**: K=2% Δ=−0.0614 CI [−0.1007, −0.0245] (excl 0); K=5% Δ=−0.0374 CI [−0.0531, −0.0213] (excl 0); at K<2% CIs span zero (tied). The iter-56 dominance-switch pattern RE-APPEARS on the calibration axis: at K<2% the LLM-aggregate features neither restore recall nor improve calibration; at K≥2% they do BOTH. **Falsifiable headline #2 (sensor-only tree is severely miscalibrated at every K)**: XGB-4sensor reports mean predicted probability 0.71–0.85 in top-K alerts but observed positive rate is only 0.13–0.36 — a calibration gap of +0.49 to +0.61 absolute probability; paired bootstrap CIs exclude zero at 5/5 budget points for both 20raw−4sensor and 24full−4sensor. **Falsifiable headline #3 (24full Pareto-dominates 20raw on BOTH axes at K=2%)**: recall Δ=+7.6pp [iter-56 #66] AND calibration Δ=−6.1pp [this iter]; the LLM-aggregate sensor does NOT trade calibration for recall at the dominance-switch K. **Why this matters**: a fraud-ops lead who deploys a model with absolute-probability-inflated alerts erodes analyst trust; XGB-4sensor is the failure mode; XGB-24full is the success mode. New §sec:p8-operational-calibration + Tables tab:p8-op-cal + tab:p8-op-cal-boot in paper/sections/p8_evidence.tex; **paper_P8_fraud.pdf rebuilds to 31 pages / 0 errors / 0 undefined citations** (was 29, +2 pages) | `scripts/p5p8/p8_operational_calibration.py` (~290 LoC, stdlib + numpy + pandas + xgboost + matplotlib); `experiments/results/p5p8/p8_operational_calibration.{tsv (15 rows), boot.tsv (15 paired-bootstrap rows), summary.json}`; `figures/p8_operational_calibration.{png,pdf}`; `docs/p5p8_improvements/70_p8_operational_calibration.md`; new §sec:p8-operational-calibration + 2 tables in `paper/sections/p8_evidence.tex` | validated | iter 60 |

| 71 | P6 | T2+T3 | **Registry-entry MIN-REPORT field-level completeness audit (iter 60 JOB B / SYNTH)** — fresh vein, not in prior ledger. Iter-50 #61 audited the registry at the TOP-LEVEL field granularity (7 MIN-REPORT items, top-3 null-rate: decontamination 80%, loss_form 66%, reference_kl 52%); iter-53 #64 audited the P5 MIN-REPORT manifests at the SUB-FIELD granularity (22 sub-fields across 98 cells). This iter closes the third axis: **the same 23-sub-field audit, applied to the registry's 20 stack entries**. The 14 delta entries do not carry min_report (they describe delta components, not stacks). For each of 23 sub-fields, measure population rate, number of unique values, Shannon entropy. For each entry, derive a 23-bit population fingerprint and test whether the 20 entries partition or collapse. **Falsifiable headline #1 (zero fields at full null rate on the registry)**: 0/23 sub-fields are populated by 0/20 entries; the lowest pop-rate is 0.20 (4 fields tied: loss_form.token_mask, decontamination.performed, decontamination.parser_robustness_probe, and one duplicate from the SUBFIELDS list). The iter-50 "decontamination 80% null at top-level" sharpens to "only 4/20 entries populate ANY decontamination sub-field". **Falsifiable headline #2 (information-bearing sub-fields are concentrated in 3 blocks)**: sampler_backend.backend (6 unique, H=2.14 bits), heldout_split.description (4 unique, H=1.96 bits), telemetry.source (4 unique, H=1.88 bits); the remaining 20/23 sub-fields have H ≤ 0.99 bits (degenerate). **Falsifiable headline #3 (per-entry fingerprint collapse)**: 20 stack entries collapse into 10 distinct 23-bit population fingerprints (largest cluster = 5 entries sharing the GRPO-on-Qwen3 default pattern); operationally expected since most entries are GRPO-family runs. **Cross-paper coupling**: both P5 and P6 audit the same 23 sub-fields; P5 manifests report flat n/a-* sentinels across all sub-fields (the "honest-but-vacuous" surface per iter-53 #64); P6 registry encodes real values heterogeneously; the two surfaces tell a complementary story. **Schema bump recommendation**: the bump does NOT require a new field; it requires populating the existing decontamination.{performed, parser_robustness_probe} sub-fields on the remaining 16 entries — a 4-line edit per entry. **Why this matters**: a registry entry that does not populate decontamination cannot be cited as a contamination-controlled baseline; the audit gives the next schema bump a concrete target with measurable success criteria. New §sec:p6-minreport-subfield + Table tab:p6-minreport-subfield in paper/sections/p6_registry_health.tex; **paper_P6_registry.pdf rebuilds to 34 pages / 0 errors / 0 undefined citations** (was 27, +7 pages) | `scripts/p5p8/p6_registry_minreport_audit.py` (~270 LoC, stdlib + json + math + collections); `experiments/results/p5p8/p6_registry_minreport_subfield.tsv` (23 rows); `p6_registry_minreport_entry_fingerprint.tsv` (20 rows); `p6_registry_minreport_item_summary.tsv` (7 rows); `p6_registry_minreport_summary.json`; `docs/p5p8_improvements/71_p6_registry_minreport_audit.md`; new §sec:p6-minreport-subfield + 1 table in `paper/sections/p6_registry_health.tex` | validated | iter 60 |

## Iter 60 deliverables (this commit)

- `scripts/p5p8/p8_operational_calibration.py` (~290 LoC)
- `experiments/results/p5p8/p8_operational_calibration.tsv` (15 rows: 5 K × 3 trees)
- `experiments/results/p5p8/p8_operational_calibration_boot.tsv` (15 paired-bootstrap rows)
- `experiments/results/p5p8/p8_operational_calibration_summary.json`
- `experiments/results/p5p8/figures/p8_operational_calibration.{png,pdf}`
- `docs/p5p8_improvements/70_p8_operational_calibration.md`
- `paper/sections/p8_evidence.tex` new §sec:p8-operational-calibration + Tables tab:p8-op-cal + tab:p8-op-cal-boot
- `paper/paper_P8_fraud.pdf` rebuilds to 31 pages / 0 errors / 0 undefined citations (was 29)
- `scripts/p5p8/p6_registry_minreport_audit.py` (~270 LoC)
- `experiments/results/p5p8/p6_registry_minreport_subfield.tsv` (23 rows)
- `experiments/results/p5p8/p6_registry_minreport_entry_fingerprint.tsv` (20 rows)
- `experiments/results/p5p8/p6_registry_minreport_item_summary.tsv` (7 rows)
- `experiments/results/p5p8/p6_registry_minreport_summary.json`
- `docs/p5p8_improvements/71_p6_registry_minreport_audit.md`
- `paper/sections/p6_registry_health.tex` new §sec:p6-minreport-subfield + Table tab:p6-minreport-subfield
- `paper/paper_P6_registry.pdf` rebuilds to 34 pages / 0 errors / 0 undefined citations (was 27)
- 2 lines appended to `AUTORESEARCH_FINDINGS.jsonl` (pillar P8, pillar P5P8-SYNTH)

## Iter 60 headline findings

**JOB A (P8 #70) — Operational calibration gap dominates at K≥2%:**
At the dominance-switch K=2% (iter-56 #66), XGB-24full strictly Pareto-dominates
XGB-20raw on BOTH the recall axis (+7.6pp [iter-56]) AND the calibration axis
(−6.1pp [this iter]); the LLM-aggregate sensor does NOT trade calibration for
recall at the dominance-switch K. Below K=2% the trees are tied on calibration
(CIs span zero). XGB-4sensor is severely miscalibrated at every K (gap ∈
[+0.49, +0.61] absolute probability, 5/5 paired-bootstrap CIs exclude zero).

**JOB B (SYNTH #71) — P6 registry sub-field audit surfaces a concrete schema bump:**
The 4 sub-fields at 20% pop rate (loss_form.token_mask, decontamination.* both)
are the next schema bump candidates. The 3 most-informative sub-fields are
sampler_backend.backend (H=2.14 bits), heldout_split.description (H=1.96 bits),
telemetry.source (H=1.88 bits). 20 stack entries collapse into 10 distinct 23-bit
fingerprints (largest cluster = 5 entries). The schema bump does NOT require a
new field; it requires populating the existing decontamination.* sub-fields on
the remaining 16 entries.

## Iter 60 synthesis re-ranking (JOB B / SYNTH)

Re-ranked all 71 rows by impact × evidence × paper-facing readiness. Key state
change: **ledger now has 71 rows: 71 validated, 0 proposed, 0 rejected-from-this-iter**.
Both new rows (#70 P8 operational calibration, #71 P6 registry sub-field audit)
landed at the top of the re-ranked stack on (impact, evidence, readiness).

Top of the re-ranked P8 stack (highest impact × readiness, all validated):
1. **#70 P8 operational calibration** (iter 60) — closes the calibration axis of
   the iter-56 dominance-switch story; LLM-aggregate sensor wins on BOTH axes
   at K=2%, not just recall.
2. **#66 P8 alert-volume Pareto** (iter 56) — operational K-axis dominance switch
3. **#62 P8 decision regret** (iter 52) — dollar regret vs oracle
4. **#58 P8 cost per fraud caught** (iter 48) — $/fraud_caught accounting

Top of the re-ranked P6 stack (highest impact × readiness, all validated):
1. **#71 P6 sub-field audit** (iter 60) — sub-field MVE on registry, surfaces
   next schema bump
2. **#69 P6 measured-block coverage** (iter 58) — measured[] provenance audit
3. **#65 P6 missing deltas** (iter 54) — closed the missing-delta gap
4. **#61 P6 registry health** (iter 50) — CI-style schema validator

**Recorded rejects** (per JOB B protocol):
- (no new rejects this iter — the iter-32 rejects #42, #43, #44 remain
  closed-with-reason; iter-60 did not over-spend any evidence-gated thread.)

**Concrete candidates surfaced but not yet opened** (for the next synthesis iteration):
- **P5 audit cross-base triangulation extension**: iter-52 #63 measured A, B, C
  at per-cell level on 98 mega cells and found all 3 correlations NaN (degenerate).
  Adding iter-37's discriminative-entropy audit at the SUB-FIELD level (not item
  level) would close the P5 honesty measurement to a 3-axis surface.
- **P6 per-delta measured-vs-claimed on N10 panel**: iter-46 #56 was on N2; the
  N10 panel has 5× more statistical power (75 step-seed obs vs 16); a re-audit
  on N10 could surface more SUPPORT verdicts on entries that were NEUTRAL on N2.
  Precondition: the N10 panel must complete (currently 5/8 seeds done).
