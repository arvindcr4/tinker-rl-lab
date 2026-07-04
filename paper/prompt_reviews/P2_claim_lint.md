# P2 (ZVF) Claim–Evidence Lint

**Contract:** `research_prompts/writing/claim-evidence-linter.md` — label each major claim in the
abstract + results + conclusion of `paper/paper_P2_zvf.tex` as **supported / weakly supported /
unsupported**, citing the exact artifact in `experiments/results/` that backs (or refutes) it.
Every number below was re-checked against the on-disk TSV/JSON on 2026-07-04.

**Sections linted:** `p2_abstract.tex`, `p2_results_intro.tex`, `zvf.tex`,
`zvf_cross_experiment_diagnostic.tex`, `zvf_dynamics.tex`, `zvf_gradient.tex`, `zvf_scaling.tex`,
`zvf_iter{22,30,34,38,42,46,50,58,62}.tex`, `frontier_synthesis_zvf.tex`, `p2_conclusion.tex`.

**Headline tally: 53 claims — 34 supported, 12 weakly supported, 7 unsupported.**
Worst offender: **R6** — "ZVF precedes the heldout-accuracy collapse by 5–15 rollout steps on every
trajectory" (`zvf.tex`), directly refuted by the very file it cites
(`zvf_leadtime_summary.tsv`: GRPO mean/min/max lead = **1 step**; `zvf_dynamics_leadtime.tsv`: 1–3 steps).

Support-level key:
- **S** supported — number reproduces from the cited/available artifact.
- **W** weakly supported — direction holds but the specific number, scope, or qualifier does not fully reproduce.
- **U** unsupported — number cannot be reproduced from any artifact, or is contradicted by one.

---

## 1) Claim table

### Abstract (`sections/p2_abstract.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| A1 | Benchmark of 70+ RL runs, seven libraries, five model families (0.6B–~671B), GSM8K/HumanEval/tool-use | **S** | `zvf_summary.tsv` has 80 per-seed rows; roster documented in `sections/_shared_methods.tex` (lines 39–47 "two seven-library rosters", lines 135–160 "0.6B to ~671B across 5 model families"). Caveat: `zvf_scaling_cross_pillar.tsv` lists DeepSeek-V3.1 at **685.0** params_B, vs the "~671B" ceiling. |
| A2 | ZVF correlates with catastrophic collapse, Spearman ρ ≈ 0.56–0.62 | **S** | `zvf_failure_correlation.tsv`: `mean_zvf vs is_collapse` Spearman **0.5594**, Pearson point-biserial **0.6150** (n=23). Caveat: 0.62 is the *Pearson* value; abstract labels the whole range "Spearman". |
| A3 | Only weakly correlated with final held-out outcome, ρ ≈ 0.27 | **S** | `zvf_failure_correlation.tsv`: Spearman **0.2687**, 95% CI **[−0.3704, +0.8787]**, n=23. CI includes 0 — paper correctly treats it as weak. |
| A4 | Temporally sticky: late-phase 0.87–0.97 vs early-phase 0.04–0.13 | **W** | `zvf_dynamics_phase.tsv`: late range ✓ (var-mit GRPO 0.8696; G-sweep 0.9449–0.9777). Early range holds **only** for var-mit GRPO (0.0441) and G=16 (0.1346); G=2/4/8 early phases are **0.6058 / 0.4183 / 0.2484** — well outside "0.04–0.13". Same overstatement in `p2_results_intro.tex`. |
| A5 | Lag-1 autocorrelation ≈ 0.94 | **S** | `zvf_dynamics_phase.tsv`: 9 var-mit methods mean ρ₁ = **0.939**; full range [0.774, 0.993]; all-13-cell mean 0.923. "≈0.94" fair for the method-pooled figure. |
| A6 | ZVF mechanically coupled to reward sparsity, group size, baseline accuracy | **S** | `zvf_gradient_coupling_pooled.tsv` (r(ZVF, mean_reward) = +0.79 to +0.85 per G); `zvf_partial_correlations.tsv` (r=+0.74, determinism caveat); `zvf_dynamics_phase.tsv` (early ZVF falls monotonically in G: 0.606→0.135). |
| A7 | ZVF aliases mastery with incapacity; motivates magnitude/sign-aware replacements | **S** | Definitional (ZVF = Pr(K=0)+Pr(K=G)) plus measured inversion in `zvf_signed_summary.tsv` (converged G=2 run raw ZVF 0.838 > plateaued GRPO 0.481) and `pcd_vs_zvf_summary.tsv` (jitter drives ZVF 0.1583→0.0000 while PCD invariant at 0.1538). |

### Results opener (`sections/p2_results_intro.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| RI1 | 60 measured runs spanning nine variance-mitigation methods + G-sweep | **S** | `zvf_iter42_summary.tsv`: `n_runs 60` (45 var-mit + 12 G-sweep + 3 tinker); `zvf_dynamics_summary.tsv` row count matches. |
| RI2 | Gradient-utilization identity GU = 1 − ZVF | **S** | Definitional under binary rewards; consistent with `zvf_gradient_coupling_pooled.tsv` (ZVF anti-correlates with grad_norm/advantage variance at every G). |
| RI3 | ρ ≈ 0.56–0.62 (collapse) vs ρ ≈ 0.27 (held-out) gap | **S** | Same artifact as A2/A3 (`zvf_failure_correlation.tsv`). |

### Cross-library diagnostic (`sections/zvf.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| R2 | Cross-library table: GRPO mean ZVF 0.481 / last-10 0.379; AERO 0.220 / 0.399; SCAFGRPO 0.409; MCGRPO/GIFT/AREAL/ES each 1/5 drift | **S** | `zvf_by_library.tsv` — every cell reproduces (0.4808/0.3791, 0.2203/0.3993, 0.4088, n_drift=1 for the four sub-vanilla methods). |
| R3 | AERO is "the *only* library that ties vanilla GRPO's last-10 accuracy while halving its gradient starvation" | **W** | `zvf_by_library.tsv`: halving unique to AERO ✓ (0.220 vs 0.481), but CPPO (0.392), NGRPO (0.387), SCAFGRPO (0.409) also match-or-beat GRPO's 0.379 last-10 with 34–39% ZVF cuts — "only" rests on an unstated halving criterion. |
| R4 | Naive r = 0.22, bootstrap CI [−0.29, 0.94], **n=14** | **W** | r and CI reproduce (`zvf_failure_correlation.tsv`: 0.2173 [−0.2912, +0.9362]) but at **n=23**; `zvf_iter26_residual.tsv` pooled slice is n=17. No artifact yields these numbers at n=14. |
| R5 | Residual β₂ (ZVF after mean-reward control) is **negative** on every pooled subset; a 0.1 ZVF rise tracks a 0.04 last-10 drop | **U** | Cited files (`zvf_summary.tsv` "column mean_zvf", `zvf_failure_correlation.tsv`) contain **no regression coefficient**. The residual artifact that exists, `zvf_iter26_residual.tsv`, shows the residual correlation is **positive on every slice** (+0.205, +0.795, +0.798, +0.612) — the sign of the claim is contradicted, and the −0.04-per-0.1 slope appears nowhere. Also conflicts with iter30's own citation of iter26 (r_residual = **+0.80**). |
| R6 | "The signal precedes the heldout-accuracy collapse by **5–15 rollout steps** on every trajectory (cross-validated lead-time reported in `zvf_leadtime_summary.tsv`)" | **U** | The cited file shows GRPO **mean_lead = 1.00, min = 1, max = 1**; `zvf_dynamics_leadtime.tsv` shows 3/1/1 steps. Both the paper's own dynamics section ("1–3 steps... too short to prescribe") and iter22 (lead = 1.0) contradict 5–15. |
| R7 | Iter130 AUROCs: max-fusion 0.929 [0.83, 1.00] cross-experiment, 0.805 [0.62, 0.95] within-method; per-axis values | **S** | `zvf_iter130_axis_aurocs.tsv` — all ten AUROC/CI cells reproduce exactly (0.929 [0.8314, 0.9952]; 0.8049 [0.6156, 0.9512]; magnitude within 0.0732; etc.). Panel composition (52 rows, 11 failures) matches `zvf_iter130_meta.json`. |
| R8 | Iter130 method-ranking table of mean `zvf_risk_max`: anchors 0.69/0.55, GRPO 0.53, NGRPO/CPPO/AERO 0.43–0.49, SCAFGRPO 0.21, GIFT/AREAL/ES 0.13–0.20 | **U** | Not reproducible from `zvf_iter130_method_risk.tsv` or `zvf_iter130_risk_index.tsv`. Actual mean `zvf_risk_max`: scaling anchors **0.982**, tool-use **0.993**, GRPO **0.858**, GIFT/ES **0.753**, SCAFGRPO **0.475**. The paper's 0.69/0.55 anchor values are the *weighted* `zvf_risk` column, and GRPO 0.53 / SCAFGRPO 0.21 / GIFT–ES 0.13–0.20 match **neither** column (weighted means are 0.578 / 0.225 / 0.305–0.332). MCGRPO (0.403, failure 0.2) is silently missing from the table. Qualitative ordering (GRPO top among var-mit) survives; the printed numbers do not. |
| R9 | GRPO drift slope 9.9×10⁻³/step = 24× AERO's 4.1×10⁻⁴/step; AERO CSD ≈ 0.39 | **S** | `zvf_iter130_method_risk.tsv`: grpo drift_mean 0.0099088, aero 0.0004128 (ratio 24.0); aero csd_mean 0.3927. Caveat: same section elsewhere quotes "AERO plateau ≈ 0.33" from iter126 while `zvf_iter126_lag1.tsv` AERO rolling lag-1 averages 0.393. |

### Cross-experiment diagnostic (`sections/zvf_cross_experiment_diagnostic.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| R10 | Failure-count table: collapse 3 / drift 9 / plateau 41 / converged 27; medians (plateau ZVF 0.4820; all-80: 0.4810 / 0.9870 / 0.8630) | **U** | Current `zvf_summary.tsv` (80 rows) gives **collapse 7 / drift 5 / plateau 41 / converged 27**; plateau median mean-ZVF **0.2219**; all-80 medians **0.2964 / 0.4096 / 0.4015**. The table is stale relative to the artifact it says generated it. (Collapse-row median ZVF = 1.0 does still hold on the 2 rows with ZVF data.) |
| R11 | Pooled 23 cells: ρ_Spearman = 0.27, CI [−0.37, 0.88]; point-biserial 0.62 Pearson / 0.56 Spearman | **S** | `zvf_failure_correlation.tsv` — exact (0.2687 [−0.3704, 0.8787]; 0.6150; 0.5594; n_pooled_rows=23). |
| R12 | Tool-use collapse: reward 0.0 on every rollout, ZVF_t = 1.0 for all t, heldout 0.0 over the 30-step window | **S** | `tool_code_reward_diagnostics.tsv`: both cross-tool rows have n_steps 30, reward_mean 0, zvf 1.0, peak 0, last10 0. |
| R13 | Nemotron-120B collapse: peak 0.875 → last10 0.2083 | **W** | `scaling_law_three_phase.tsv`: peak_segment_mean **0.8750** ✓ but late_segment_mean is now **0.1544**, not 0.2083. Collapse classification unaffected (both < 0.35). |
| R14 | "All four methods [MCGRPO/GIFT/AREAL/ES] reach ℓ < 0.03 from peak ≈ 0.40 … Mean-ZVF averages 0.10–0.13" | **W** | `zvf_by_library.tsv`: only **1 of 5 seeds per method** is a drift row (the drift rows do hit last10 ≈ 0.02, cf. `zvf_summary.tsv` drift median 0.0215); method-level mean last-10 is 0.316–0.328. And MCGRPO mean ZVF is **0.146**, outside "0.10–0.13". Over-generalized from the single drifting seed. |
| R15 | AERO 54% relative ZVF reduction; 3/5 GRPO seeds trip the per-step collapse flag vs 0/5 AERO; mean-ℓ within 0.02 | **S** | `zvf_by_library.tsv`: 1 − 0.2203/0.4808 = **54.2%**; per_step_collapse_rate grpo 0.60, aero 0.00; ℓ 0.3791 vs 0.3993 (Δ=0.020). |
| R16 | Recipe calibration: "four plateau cells (median ℓ = 0.38, median ZVF = 0.48)... three collapse cells all reach ZVF ≥ 0.95" | **W** | Stale/inconsistent with current `zvf_summary.tsv`: 41 plateau rows (not four) with median ZVF **0.2219**; the 0.48 figure matches only GRPO's own mean. Collapse rows with ZVF data are 2, both 1.0 ✓. |
| R17 | Anti-herding sign reversal: δ_div = +0.1224 [+0.1116, +0.1334] (frac>0 0.842) on tinker_gsm8k; −0.0668 on sweep; −0.2994 per-step agg; falsifies uniform [0.13, 0.23] band | **S** | `zvf_antiherding_falsification.tsv` — means/fractions/verdicts exact; CI endpoints differ only in the 4th decimal ([0.1115, 0.1338], [−0.0792, −0.0560], [−0.3882, −0.2150]) from bootstrap regeneration. |
| R18 | Empirical iso-G rows: tails G_iid 13 → G_emp 5 (ΔG −8) at Y=0.80, δ=+0.3436; frontier +0.0078 → ΔG +1; sweep rows; tails ΔG ∈ {−4, −8, −16} | **S** | `zvf_empirical_isog.tsv` — every quoted row reproduces exactly (0.10–0.20 bin: 7→3/13→5/23→7 for Y=0.60/0.80/0.95). |

### Dynamics, gradient, scaling (`zvf_dynamics.tex`, `zvf_gradient.tex`, `zvf_scaling.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| R19 | GRPO phase drift 0.044→0.529→0.870 (20×); AUC(0.5) 0.160 = 2.4× CPPO 0.065; ρ₁ ∈ [0.77, 0.99]; prompt-axis ρ₁ ≈ −0.005; drift positive for every method; GRPO late ≥ 0.87 (var-mit) and ≥ 0.94 (G-sweep) | **S** | `zvf_dynamics_phase.tsv` — all table cells reproduce. Minor: the "GIFT/ES/AReaL upper tail (0.001 to 0.026)" endpoint 0.026 has no counterpart (their AUC(0.5) ≤ 0.0013). |
| R20 | Lead-time: 3 collapse events, 1–3 step lead; deliberately underclaimed | **S** | `zvf_dynamics_leadtime.tsv`: seed 0 → 3 steps (θ=0.8), seeds 1–2 → 1 step (θ=0.9). Exact. |
| R22 | ZVF×gradient-proxy Pearson table (e.g. G=2: grad_norm −0.640 [−0.802, −0.513]; entropy −0.841; mean_reward +0.795) | **S** | `zvf_gradient_coupling_pooled.tsv` — all 16 rows exact. |
| R23a | Five-anchor table (Nemotron frac_below_0p1 = zero_fraction = 0.55, only collapse row) and tests T1–T5 (+0.791 / −0.707 / −0.894); separation rule | **S** | `zvf_scaling_cross_pillar.tsv` + `_summary.tsv` — exact (0.7906, −0.7071, −0.8944). |
| R23b | "ρ_Spearman(frac_below_0p1, is_collapse) = **+1.0 by construction**" (prose paragraph) | **U** | Contradicts the same section's own table and the artifact: `zvf_scaling_cross_pillar_summary.tsv` T1 = **0.7906**. (With one collapse among n=5 and ties at 0, the rank correlation is not 1.0.) |

### Iteration sections (`zvf_iter22–62.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| R24a | iter22 per-library mean-ZVF bootstrap CIs (GRPO [0.480, 0.482] disjoint from all; AERO [0.219, 0.222] between clusters) | **S** | `zvf_library_bootstrap_ci.tsv` — CIs reproduce to rounding ([0.4799, 0.4818], [0.2187, 0.2217], etc.); GRPO CI disjoint ✓. |
| R24b | iter22 collapse-rate 0.600 with bootstrap CI [0.400, 0.800], "disjoint from 0.000 of every mitigation library" | **W** | The 3/5 = 0.6 seed-level rate is real (`zvf_leadtime_summary.tsv` collapse_rate 0.6; `zvf_by_library.tsv` per_step_collapse_rate 0.6), but the cited bootstrap artifact now stores **0.1220 [0.0340, 0.2080]** (a per-step rate) — the printed CI cannot be reproduced. Disjointness from zero still holds under either version. |
| R25a | Wilcoxon pre/post: AREAL Δ=−0.099 p=0.043; GIFT Δ=−0.087 p=0.225; GRPO Δ≈−0.004 p=0.893 | **S** | `zvf_pre_post_test.tsv`: −0.0987/0.0431; −0.0867/0.2249; −0.0033/0.8927. |
| R25b | "All other libraries have \|Δ\| < 0.05 and p > 0.5" | **W** | Same file: MCGRPO Δ = **−0.0877** (>0.05) and ES p = **0.345** (<0.5) violate the sentence. |
| R26 | iter30 leading-indicator table: ROC-AUC 0.90–0.93 for level features at K ∈ {5,10,25}; slope at chance (0.487/0.267/0.665); PR-AUC 0.45→0.59→0.83; calibration top-bin 0.956→0.927; K=10 0.885-bin→0.718 | **S** | `zvf_iter30_leadindic.tsv` + `zvf_iter30_calib.tsv` — every quoted cell exact. Honest-scope caveat (3 collapse trajectories, all GRPO) is disclosed. |
| R27 | iter26 residualised correlation +0.80, CI [+0.54, +1.00] (cited in iter30) | **S** | `zvf_iter26_residual.tsv`: pooled_variance_mitigation_per_methodseed residual_r **+0.795 [+0.544, +0.997]**. Note: sign directly contradicts R5 in `zvf.tex`. |
| R28a | iter34 phase classifier: LOO 8/12 = 0.667 (chance 0.25); saturation 5/5, collapse 1/1, drift 2/3, plateau 0/3; zvf_direction top feature (+0.113); collapse-vs-rest gaps (+0.277 / +0.281 / −0.285 / −0.143); Nemotron zero_frac 0.55 / discriminator 0.516 | **S** | `zvf_iter34_summary.tsv`, `_feature_importance.tsv`, `_collapse_gap.tsv` — all exact. |
| R28b | "the 0.667 base accuracy is significantly above the permutation null distribution" | **U** | Self-contradicted: the same paragraph and `zvf_iter34_feature_importance.tsv` give smallest p_shuffled_beats = **0.195** — no significance at any conventional level. |
| R29 | iter38: δ_div = +0.122 (n=600); iso-yield savings 25%/20%; failure classifier 11/14 = 78.6% | **S** | `zvf_iter38_summary.tsv` (0.1224, 0.25/0.20, 0.7857) and `zvf_iter38_classifier.tsv` (confusion matrix identical). Minor: "SEM 0.013" is closer to the 95% CI half-width (0.011); the SEM at n=600 is ~0.006. |
| R30 | iter38: reward-only 3-NN baseline attains 9/14 = 64.3% (ZVF adds +14.3pp) | **W** | No artifact: `zvf_iter38_classifier.tsv` logs only the 3-feature run; the ablated (mean_last10, mean_peak)-only run is asserted in prose with nothing on disk. |
| R31 | iter42: cluster-LOO 58/60 = 0.967 with early-only features; burden_05/burden_07 AUC 0.975/0.978; first-passage AUC ≈ 0; pool = 45 plateau + 15 converged | **S** | `zvf_iter42_summary.tsv` — exact (0.9667, 0.9748, 0.9778, 0.0, plateau=45/converged=15; grpo cluster 0.60, all others 1.0). |
| R32 | iter42: "a researcher can abort a doomed RL run within the first **30–50%** of the trace on ZVF burden alone" | **W** | The measured features are first-half (50%) statistics; nothing on disk tests a 30% prefix. Also the pool contains only plateau-vs-converged (no collapse/drift), which the section itself concedes. |
| R33 | iter46 Iso-G: ΔG = −2.81 [−3.12, −2.52] at Y=0.80; −6.50 [−7.04, −5.94] at Y=0.95; savings 68.0/54.7/42.2/25.8/17.2%; 4/4 pre-registered predictions pass | **S** | `zvf_iter46_summary.tsv` + `zvf_iter46_predictions.tsv` — exact (−2.8139 [−3.1208, −2.5188]; −6.4950 [−7.0416, −5.9406]; P1–P4 True). |
| R34 | iter46: "the anti-herding bonus adds exactly δ_div = 0.122 to the mean yield at every fixed G ∈ {2, 4, 8, 16} where Y_iid < 1" | **W** | `zvf_iter46_summary.tsv`: uplift is **+0.1455** at G=2/4/8 and **+0.0402** at G=16 (where Y_iid = 0.9598 < 1). Neither "exactly 0.122" nor "every G ∈ {…16}" reproduces; the section's own next sentence quotes +0.145. |
| R35 | iter50: reward-leads-ZVF lag profile monotone on all 9 libraries (GRPO 0.855→0.880, AERO 0.627→0.730, peak at L=+10); phase-2 integrals GRPO 0.381 vs AERO 0.196 (~49% cut), SCAF/ES 0.000; P3 FAIL disclosed (n=3 vs n=1) | **S** | `zvf_iter50_summary.tsv` + `zvf_iter50_predictions.tsv` — all cells exact; the FAIL is honestly reported. |
| R36 | iter58 signed ZVF: ZVF⁻ separates unhealthy/healthy with AUC 1.000 (ZVF⁺ 0.000, raw 0.396); GSM8K exact split ZVF⁺ 0.105–0.160 / ZVF⁻ 0.025–0.040; G=2 inversion (raw 0.838 = 0.766 ZVF⁺ + 0.072 ZVF⁻); AERO cut is 103% ZVF⁻; iter54 anchor (0 perfect separators, CI [0.11, 0.84]) | **S** | `zvf_signed_summary.tsv`, `zvf_signed_failure_corr.tsv` (1.0000/0.0000/0.3956; n=13 vs 7; min unhealthy 0.1026 > max healthy 0.0722), `zvf_signed_aero.tsv` (−0.2605/−0.2682/+0.0076, frac 1.029), `zvf_iter54_doom_summary.tsv` (0 perfect separators, CI [0.110, 0.840]). |
| R37a | iter62 stratified table + AERO−GRPO deltas per quintile (q0 −0.006 [−0.011, −0.002] … q2 −0.480, q3 −0.476, q4 −0.230; every CI excludes 0) | **S** | `zvf_iter62_summary.tsv` + `zvf_iter62_aero_minus_grpo.tsv` — exact. |
| R37b | "The striking result … **all nine libraries** sit between 0.010 and 0.024 [at q0]" | **U** | Contradicted by the same section's own table and `zvf_iter62_summary.tsv`: SCAFGRPO q0 = **0.2352**. (The section later concedes SCAFGRPO is the exception at 0.235 — the "all nine" sentence is still wrong as written.) |

### Frontier synthesis (`sections/frontier_synthesis_zvf.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| R38a | External reasoning contributions (inelasticity theorem, PCD, LARQ, ρ ≳ 0.45 falsifiable bar, KL-distillation caveat) presented as attributed proposals, not results | **S** | Provenance is explicit ("frontier synthesis", "a falsifiable bar, not a claimed result"); no data claim to reproduce. Micro-jitter behavior is in fact corroborated on disk by `pcd_vs_zvf_summary.tsv` (uncited). |
| R38b | "consistent with the **0.845**→0.631 ZVF drop we observe as G:2→16" | **W** | `groupsize_zvf_sweep.tsv` / `zvf_dynamics_phase.tsv`: G=2 mean ZVF = **0.838**, G=16 = 0.631. The 0.845 endpoint reproduces nowhere. Also "ρ≈0.27 over 80 runs" — that ρ was computed on the 23 pooled cells, not 80 runs. |

### Discussion + conclusion (`sections/p2_conclusion.tex`)

| ID | Claim | Label | Evidence / check |
|----|-------|-------|------------------|
| C1 | ZVF reliably tracks collapse and drifts upward over training | **S** | `zvf_failure_correlation.tsv` (0.56–0.62); `zvf_dynamics_phase.tsv` (z̄_l − z̄_e > 0 for every method and every G). |
| C2 | Not a standalone predictor of final generalization | **S** | `zvf_failure_correlation.tsv` (ρ=0.27, CI spans 0). |
| C3 | Unsigned form conflates mastery and incapacity | **S** | `zvf_signed_summary.tsv` + `zvf_signed_failure_corr.tsv` (raw AUC 0.396 vs ZVF⁻ 1.000). |
| C4 | Sub-reward jitter (e.g. small length penalty) can drive ZVF to zero, falsely reporting a healthy batch | **S** | `pcd_vs_zvf_summary.tsv`: zvf_batch 0.1583 → **0.0000** after jitter while PCD unchanged (0.1538). Note: paper presents this as a frontier thought-experiment and never cites this artifact — an easy upgrade. |
| C5 | Magnitude/sign-aware replacements outlined but not yet validated at the ρ ≥ 0.45 bar | **S** | Accurate disclosure; no artifact claims otherwise. |

---

## 2) Missing / broken evidence list

Per-claim items whose evidence is absent, stale, or contradicts the text:

1. **R6 (worst offender)** — 5–15-step lead-time: cited `zvf_leadtime_summary.tsv` shows lead = 1 step (min=max=1); `zvf_dynamics_leadtime.tsv` shows 1–3. No artifact anywhere yields 5–15.
2. **R5** — negative residual β₂ and the "0.1 ZVF → 0.04 accuracy drop" slope: no regression artifact exists; the only residual artifact (`zvf_iter26_residual.tsv`) has the **opposite sign** on every slice.
3. **R8** — iter130 method-ranking table: values match neither `zvf_risk_max` nor weighted `zvf_risk` columns of `zvf_iter130_method_risk.tsv`/`zvf_iter130_risk_index.tsv`; MCGRPO row missing.
4. **R10 / R16** — failure-count table and recipe medians are stale vs the regenerated `zvf_summary.tsv` (7/5/41/27, plateau median ZVF 0.2219, all-80 medians 0.2964/0.4096/0.4015). Re-run the table generator or pin the artifact version.
5. **R23b** — "ρ = +1.0 by construction": contradicted by `zvf_scaling_cross_pillar_summary.tsv` (0.7906) and the section's own Table.
6. **R28b** — "significantly above the permutation null": own artifact p = 0.195.
7. **R37b** — "all nine libraries between 0.010 and 0.024 at q0": SCAFGRPO = 0.2352 in `zvf_iter62_summary.tsv`.
8. **R30** — reward-only 9/14 baseline: no ablation artifact; add a second block to `zvf_iter38_classifier.tsv`.
9. **R24b** — collapse-rate bootstrap CI [0.400, 0.800]: cited `zvf_library_bootstrap_ci.tsv` now stores 0.1220 [0.0340, 0.2080] under a changed definition.
10. **R4** — n=14 vs artifact n=23 (or 17); pick one and cite it.
11. **R13** — Nemotron last10 0.2083 vs current `scaling_law_three_phase.tsv` late mean 0.1544.
12. **A4/RI** — early-phase "0.04–0.13" excludes G=2/4/8 cells (0.25–0.61 in `zvf_dynamics_phase.tsv`).
13. **R38b** — 0.845 (G=2 ZVF) irreproducible; artifact says 0.838.
14. **C4** — jitter demonstration exists in `pcd_vs_zvf_summary.tsv` but is uncited; wire it in to convert a thought experiment into a measured result.

## 3) Rewrite suggestions for risky claims (semantics preserved)

- **R6** → "ZVF saturates above 0.8 before the collapse flag trips, but with only a 1–3 step first-passage lead on this corpus (`zvf_leadtime_summary.tsv`); we therefore treat late-phase ZVF level, not lead time, as the operational signal." (This matches what `zvf_dynamics.tex` and `zvf_iter22.tex` already say — delete the 5–15 sentence.)
- **R5** → either delete, or restate with the artifact's sign: "after regressing out mean reward, residualised ZVF remains *positively* associated with last-10 accuracy within the variance-mitigation slice (+0.80 [0.54, 1.00], `zvf_iter26_residual.tsv`); the naive pooled correlation is insignificant (r=0.22, CI [−0.29, 0.94], n=23)."
- **R8** → regenerate the ranking table directly from `zvf_iter130_method_risk.tsv`, label the column explicitly (weighted `zvf_risk` vs `zvf_risk_max`), and restore the MCGRPO row.
- **R10/R16** → re-run `scripts/zvf_diagnostic.py` and re-materialize the counts/medians, or state the artifact snapshot the table was built from.
- **A4** → "late-phase 0.87–0.97 versus early-phase 0.04–0.13 for variance-mitigation GRPO and the largest group size; smaller G starts higher (0.25–0.61) yet still drifts monotonically upward."
- **R23b** → "the rank correlation is 0.79 with the single collapse ranked first — maximal separation given ties at zero," dropping "+1.0 by construction".
- **R28b** → "the permutation test does not reach significance (smallest p = 0.195); with n = 12 anchors the 0.667 accuracy should be read descriptively."
- **R37b** → "eight of nine libraries sit between 0.010 and 0.024 at q0; SCAFGRPO (0.235) is the lone exception, discussed below."
- **R25b** → "all other libraries have p > 0.22, with MCGRPO showing the same directional decrease (Δ = −0.088, p = 0.50)."
- **R34** → "the anti-herding bonus adds a roughly constant +0.145 to mean yield at G ∈ {2, 4, 8}, shrinking to +0.04 at G = 16 as the iid yield saturates."
- **R3** → "AERO is the only library that *halves* ZVF while matching vanilla GRPO's last-10 accuracy; CPPO/NGRPO/SCAFGRPO achieve smaller (34–39%) reductions at comparable accuracy."
- **R14** → "one of five seeds in each of MCGRPO/GIFT/AREAL/ES drifts to ℓ < 0.03 from a peak ≈ 0.40; the surviving seeds plateau near 0.33."
- **A2** → "(Spearman ρ = 0.56; Pearson point-biserial 0.62)".

---
*Generated by the claim-evidence linter run of 2026-07-04 against worktree `tinker-rl-lab-minimax`. All labels use a strict-reviewer standard: a claim is "supported" only when its printed numbers reproduce from the on-disk artifact.*
