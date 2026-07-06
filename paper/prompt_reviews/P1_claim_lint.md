# P1 Claim–Evidence Lint (`paper_P1_scaling.tex`)

**Prompt contract:** `research_prompts/writing/claim-evidence-linter.md` — label every major claim in the abstract + results + conclusion as **supported / weakly supported / unsupported**, citing the exact TSV/JSON in `experiments/results/` (or citation) that backs it; flag any claim whose number cannot be reproduced from the data files.

**Scope reviewed:** `paper/sections/p1_abstract.tex`, `p1_results_intro.tex`, `scaling_laws.tex` (full, incl. iter 9–137 paragraphs), `scaling_law_iter{29,33,37,41,45,49,53,61,65}.tex`, `frontier_synthesis_scaling.tex`, `p1_conclusion.tex`.
**Verification method:** every quantitative claim was checked directly against the named file in `experiments/results/`; Spearman correlations were re-computed from `scaling_law_extended_frontier.tsv`.

**Tally: 58 claims — 38 supported, 12 weakly supported, 8 unsupported.**
**Worst offender:** the Δ₁T (first-minus-final) paragraph in `scaling_laws.tex` (lines 481–492), where three numbers in one paragraph contradict `scaling_law_extended_frontier.tsv` (Nemotron Δ₁T is **−0.50**, not +0.50; 7/12 — not 9/12 — anchors have |Δ₁T| ≤ 0.2; the actual +Δ outlier is Qwen3.5-27B at +0.562). Runner-up: the frontier-synthesis PPO control "paired Δ = +0.001, p = 0.75", which no data file reproduces (`samestack_ppo_grpo.json`: Δ = −0.002, p = 0.374).

---

## 1) Claim table

Legend: **S** = supported (number reproduced from cited file), **W** = weakly supported (directionally right but number/label not exactly reproducible, or unverifiable), **U** = unsupported (contradicted by, or absent from, the data files).

### Abstract (`p1_abstract.tex`)

| # | Claim (paraphrased) | Label | Evidence checked | Notes |
|---|---|---|---|---|
| A1 | Benchmark spans "70+ runs across seven RL libraries and five model families (0.6B–~671B)" | **W** | `framework_comparison.json`; `_shared_methods.tex` | No run registry in `experiments/results/` reproduces "70+". `framework_comparison.json` holds only 4 frameworks, 2 of them `mode:"dryrun"` sandbox fallbacks. "~671B" (methods) conflicts with the Pillar-1 tables, which use 685B for DeepSeek-V3.1 and include a 1T anchor (Kimi-K2). |
| A2a | Cross-scale slope of GRPO reward on log₁₀N statistically indistinguishable from zero | **S** | `scaling_law_cross_scale.tsv` (slope −0.0243, boot CI brackets 0); `scaling_law_power_law.tsv` (12 anchors, all perm p ≥ 0.46, R² ≤ 0.058) | Reproduced. |
| A2b | "...across **five orders of magnitude** in parameter count" | **U** | `scaling_law_fits.tsv`, `scaling_law_extended_frontier.tsv` | Arithmetic fails: the 5-anchor set spans 4B–685B ≈ **2.2 decades**; the 12-anchor set 4B–1000B ≈ 2.4; even 0.6B–671B (methods) ≈ 3.05. No dataset in the repo spans 5 orders of magnitude. Repeated in `frontier_synthesis_scaling.tex` ("over five orders of magnitude") and implied in the conclusion ("five orders of magnitude of single-benchmark evidence"). |
| A2c | The regressed quantity is the "GRPO reward **gain**" | **W** | `scaling_law_cross_scale.tsv` | The TSV regresses raw mean/peak reward per trace, not gain over the base policy (Δ over base is never computed in these files). Direction of the null is unaffected, but "gain" overstates what was measured. |
| A3 | Four of five frontier traces have saturation rate pinned at the fit boundary; none identifiable | **S** | `scaling_law_iter25_identifiability.tsv` (`lam_at_bound=True` 4/5, `identifiable=False` 5/5); `scaling_law_iter25_summary.tsv` (`n_identifiable 0`) | Reproduced exactly. |
| A4 | Pre-registered test geometrically falsifies three-phase hypothesis (only 2 of 12 anchors match) | **S** | `scaling_law_iter65_predictions.tsv` (P5: 2/12 PASS), `scaling_law_iter65_phase_pieces.tsv` (2 `three-phase` rows) | 2/12 reproduced. Caveat: 6 of 8 iter-65 pre-registered predictions **failed** (incl. P2 "Nemotron classified collapse"); the abstract quotes only the favorable one. |
| A5 | Nemotron-120B is a distinct collapse phase: "zero-variance fraction 0.55 versus ≤ 0.067 elsewhere" | **W** | `scaling_law_nemotron_rootcause.tsv` (`zero_fraction` 0.5500 vs next-highest 0.0667; `is_collapse=True` unique) | The **number** 0.55 is reproduced, but it is the fraction of training steps with **zero reward**, not the Pillar-2 "zero-variance fraction" (ZVF = P(K=0)+P(K=G) at rollout level). The abstract's label conflates two different diagnostics; also the paper's own geometric classifier (`scaling_law_iter65_phase_pieces.tsv`) labels Nemotron **valley**, not collapse. |
| A6 | "The defensible object is a local, stack-conditioned saturation law relating the endpoint ceiling to log₁₀N" | **W** | `scaling_law_iter133_interaction_aic.tsv` (capability-only beats params-only by 21–32 AICc; adding log₁₀N **worsens** AICc); `scaling_law_iter137_t80_scaling.tsv` (R_max ~ log₁₀N slope −0.172 ± 0.198, n.s.) | The paper's own later analyses show the ceiling-on-log₁₀N regression carries no significant signal and is dominated by the categorical capability axis — the abstract's "defensible object" is itself only weakly defensible on these files. |
| A7 | "We release all code, logs, and checkpoints" | **W** | repo tree | Code and result logs are in-repo; released checkpoints are not verifiable from `experiments/results/` (only `*_checkpoints.jsonl` training logs). |

### Results opener (`p1_results_intro.tex`)

| # | Claim | Label | Evidence checked | Notes |
|---|---|---|---|---|
| R1 | No measurable cross-scale slope / no clean phase structure / no identifiable exponent | **S** | same as A2a, A3, A4 | Reproduced. |
| R2 | Nemotron is the λ exception (λ ≈ 0.99) but likewise non-identifiable | **S** | `scaling_law_fits.tsv` (λ 0.9902), `scaling_law_iter25_identifiability.tsv` (`identifiable=False`) | Reproduced. |
| R3 | Pre-registered 12-anchor comparison falsifies three-phase (2/12 match all criteria) | **S** | `scaling_law_iter65_predictions.tsv` | Same as A4 (same caveat). |
| R4 | Nemotron "peak **held-out accuracy** 0.875 drifting to a late mean near 0.21" | **W** | `scaling_law_nemotron_rootcause.tsv` (peak_reward 0.875 @ step 3, late_mean 0.2083) | Numbers reproduce, but they are per-step **training rewards** from `frontier_gsm8k_nemotron-120b.json`, not held-out accuracy. Mislabeled. |

### Results body (`scaling_laws.tex`)

| # | Claim | Label | Evidence checked | Notes |
|---|---|---|---|---|
| S1 | Tab. `scaling-fits` per-anchor stats (mean R 0.817/0.285/0.869/0.844/0.175; λ=10 on 4 runs; Nemotron λ=0.99, R_max 0.182) | **S** | `scaling_law_fits.tsv` | Exact match, all cells. |
| S2 | Tab. `scaling-cross`: mean-R boot slope −0.114, boot 95% CI **[−1.119, +0.313]**, n_boot = **5000** | **U** | `scaling_law_cross_scale.tsv` | Not reproducible: file has boot mean −0.117, CI **[−0.796, +0.313]**, `n_boot` 983–992 (not 5000). The −1.119 lower bound matches the **R_max row of a different file** (`scaling_law_iter121_effective_compute.tsv`, boot_lo −1.1188) — an apparent copy/transposition error. Peak-R CI "[−0.830, +0.197]" also mismatches (file: [−0.623, +0.197]). Conclusion (CI brackets zero) still holds. |
| S3a | Nemotron autopsy: peak 0.875 @ step 3, late 0.208, Δ −0.667, P(R=0)=0.55, P(R>0.5)=0.05; all others P(R=0) ≤ 0.067 | **S** | `scaling_law_nemotron_rootcause.tsv` | Exact match. |
| S3b | Nemotron post-peak slope −0.0036 is "the **only negative slope** in the set" | **U** | `scaling_law_nemotron_rootcause.tsv` | Contradicted: Llama-3.1-8B-Instruct has `post_peak_decay_slope` **−0.00374** (more negative than Nemotron's −0.00361); its full-trace OLS slope in `scaling_law_fits.tsv` is also negative (−0.0037). |
| S4 | Holdout 70/30: saturation fit adds ≤ +0.0016 RMSE improvement over constant baseline | **S** | `scaling_law_holdout.tsv` | Exact match. |
| S5 | Parametric bootstrap: P(λ ≥ 9.5) = 0.47–0.60 on four runs, 0.277 Nemotron; t₈₀ CIs as printed | **S** | `scaling_law_bootstrap_ci.tsv` | Exact match. |
| S6 | Compute-adjusted slopes: +482 ± 490 (t₈₀·N) and −0.47 ± 0.87 (λ) per decade | **S** | `scaling_law_compute.tsv` (482.531 ± 490.417; −0.471 ± 0.873) | Exact match. |
| S7 | 12-anchor OLS: every slope perm p > 0.46, R² < 0.06 (all seven metrics as tabulated) | **S** | `scaling_law_power_law.tsv` | Exact match. |
| S8 | 12-anchor Spearman: ρ = −0.036 (p .91), +0.149 (p .64), −0.023 (p .94), +0.074 (p .82) | **S** | recomputed from `scaling_law_extended_frontier.tsv` | Reproduced exactly by re-computation. |
| S9 | MoE−dense mean-R gap +0.338, perm p = 0.023 (one-sided) / 0.046 (two-sided); descriptive only | **S** | `scaling_law_moe_vs_dense.tsv` (+0.3376, 0.0230, 0.0460) | Exact match; hedging appropriate. |
| S10 | Δ₁T paragraph: "9 of 12 have \|Δ₁T\| ≤ 0.2; the clear outlier is **Nemotron-120B at +0.50** (the only model with Δ₁T > +0.4); second-most-extreme is Llama at 0.00"; fig. caption "every other anchor is in [−0.50, +0.20]" | **U** | `scaling_law_extended_frontier.tsv` (`delta_first_final`) | Contradicted three ways: (i) Nemotron's Δ₁T = R(1)−R(T) = 0.0−0.5 = **−0.50**, not +0.50 (its first step is the zero; the sign is flipped); (ii) only **7/12** anchors have \|Δ₁T\| ≤ 0.2 (0.25 for Kimi & Qwen3-30B-MoE, 0.375 gpt-oss, 0.562 Qwen3.5-27B, 0.50 Nemotron); (iii) the only anchor with Δ₁T > +0.4 is **Qwen3.5-27B (+0.562)**, and "second-most-extreme … at 0.00" is not meaningful on these values. |
| S11 | Iter 17: constant model wins AIC on all 5 anchors; no significant changepoint (all perm p ≥ 0.137) | **S** | `scaling_law_iter17_aic.tsv`, `scaling_law_iter17_changepoint.tsv` | Spot-checked; matches (e.g. Qwen3.5-4B Δ_linear 1.9997 → "2.00"). |
| S12 | Iter 21: 7/12 at λ-bound; interaction perm p = 0.007; Levene p = 0.296; two-anchor MAE 4.498; compute lift share 0.643; CV +332 / −99 | **S** | `scaling_law_iter21_summary.tsv`, `_lambda_audit.tsv`, `_arch_kfold.tsv` | All reproduced (perm p 0.0072). Editorial defect: the body text contains an unfilled placeholder "$\hat\delta = $ (estimated)". |
| S13 | Iter 25: λ CI-span 0.80–0.98 of admissible range; AICc constant wins 5/5 (w_sat ≤ 0.20); T* > 200 everywhere; **0/5 identifiable** | **S** | `scaling_law_iter25_identifiability.tsv`, `_summary.tsv` | Reproduced. |
| S14 | Iter 117: BIC segmentation → 0/5 nimmaturi passes; Nemotron segment means {0.000, 0.875, 0.154}; peak−late contrast +0.721 largest | **S** | `scaling_law_three_phase.tsv` | Exact match (incl. Llama 0.9292/0.8393 → +0.090 next-largest). |
| S15 | Iter 117: t₈₀-vs-N slope +0.170, SE 0.254, degenerate (4/5 at bound); iter-109 λ-vs-N null p = 0.74 | **S** | `scaling_law_iter117_t80_scaling.tsv` (0.1699 ± 0.2538); `scaling_law_iter109b_permtest.tsv` (p_two_sided 0.7382) | Reproduced. |
| S16a | Iter 121: Spearman ρ = −0.30, perm p = 0.69; effective-compute OLS slopes −0.019 [−0.80, +0.30] and −0.018 [−1.12, +0.32]; recovery 0.17 (n=5, β=.05) and 0.07 (n=40, β=.20) | **S** | `scaling_law_iter121_late_early.tsv`, `_effective_compute.tsv`, `_synthetic_recovery.tsv` | All reproduced (0.6883, −0.0192 [−0.796,+0.304], −0.0180 [−1.119,+0.318], 0.17, 0.07). |
| S16b | Iter 121: Spearman bootstrap-95% CI "[−0.94, +0.83]" (B = 2000) | **U** | `scaling_law_iter121_late_early.tsv` | Not reproducible: file gives boot CI **[−0.80, +1.00]** with n_boot = 1000 (perm null band [−0.9, +0.9], n_perm 5000). Neither pair matches the printed interval. |
| S16c | Iter 121 bullet: "the only regime where recovery exceeds 0.5 is β ≥ 0.20 AND pool ≤ 8" | **W** | `scaling_law_iter121_power_curve.tsv` | No cell in the file exceeds 0.5 (max 0.405 at n=8, β=0.2). The paper's own table (0.40/0.41) matches the file; the bullet's ">0.5" regime does not exist in the data. |
| S17 | Iter 125: monotonicity violation rates 0.382/0.395/0.462/0.363/0.290 (all binomial p < .001); R_max bimodality gap 0.531, dip 0.522, p = 0.056; three-phase full 1/5, collapse-only 4/5 | **S** | `scaling_law_iter125_monotonicity.tsv`, `_bimodality.tsv`, `_three_phase_summary.tsv` | Exact match. |
| S18 | Iter 129: piecewise loses AICc 5/5 (Δ +1.20…+2.79, F p > .28); LOOCV cluster 5/5; log BF = −9.53 | **S** | `scaling_law_iter129_aic_compare.tsv`, `_loocv_cluster.tsv`, `_bf_capability.tsv` | Exact match (1.2008–2.7918; −9.5266). |
| S19a | Iter 133: Ward-gap perm p 0.095 → 0.042 → 0.002 → 0.002 across pools; capability-only AICc dominates by 21–32 units; 7/7 monotonicity falsification | **S** | `scaling_law_iter133_bimodality.tsv`, `_interaction_aic.tsv`, `_monotonicity.tsv` | Reproduced (0.0954/0.0420/0.0024/0.0024; violation rates incl. gpt-oss 0.3195, Kimi 0.5158). |
| S19b | Iter 133(b): "**0/7** anchors satisfy the three-phase template … **no anchor satisfies phase 1 in combination with phase 2**; 5/7 collapse-only and the remaining two are monotone-or-plateau" | **U** | `scaling_law_iter133_three_phase.tsv` | Contradicted: the file shows **Qwen3-8B and gpt-oss-20B with phase combo (1,1,1) and `three_phase_full=1`** — i.e. 2/7 satisfy the full template, and both satisfy phase 1 ∧ phase 2. (5/7 collapse-only is correct.) |
| S20 | Iter 137: 3-param fit un-binds λ 5/5 but loses AICc 5/5 (Δ +1.71…+18.18); slopes +0.507 ± 0.718 (t₈₀) and −0.172 ± 0.198 (R_max, Spearman −0.658 p .227) | **S** | `scaling_law_iter137_offset_fit.tsv`, `_t80_scaling.tsv` | Exact match, all cells of Tab. `scaling-iter137-fits`. |

### Iter subsections (`scaling_law_iter{29..65}.tex`)

| # | Claim | Label | Evidence checked | Notes |
|---|---|---|---|---|
| I29a | EEP falsified at functional-form level: GRPO AICc-best saturation 5/5 vs PPO 0/5, Fisher p = 0.0079; heldout accuracy indistinguishable (0.99 vs 0.992, p = 0.374) | **S** | `scaling_law_iter29_summary.tsv`, `_stack_compare.tsv` | Reproduced. |
| I29b | Table row "F1 bootstrap CI excludes bound … **sustained**" | **W** | `scaling_law_iter29_stack_compare.tsv` | File labels F1 `eep_status = divergent` (5/5 vs 4/5); the paper upgrades it to "sustained". Minor but a verdict-level edit of the artifact. |
| I33 | Three-phase battery: P1 9/12 (p .073), P2 5/12 (p .806), P3 7/12 (p .387), P4 ρ=0.046 (p .92) all falsified; P5 Nemotron uniqueness sustained (1/12) | **S** | `scaling_law_iter33_predictions.tsv`, `_summary.tsv` | Exact match, incl. Mann-Whitney p 0.4755 and median phase scores −0.75/−0.37. |
| I37 | Exponential not uniquely identified: linear wins AIC 9/12 (synthetic), 7/10 (dynamic); Hill n=2 wins 3/5 GRPO raw traces; extrapolation MAE 0.907 vs 0.908 | **S** | `scaling_law_iter37_summary.tsv`, `_37b_summary.tsv`, `_37c_summary.tsv`, `_37d_summary.tsv` | Reproduced (bootstrap shares 0.0963/0.7317; perm p = 1.0). Minor: text's mean w̄_sat "0.044" computes to 0.0415 from `_37_aic.tsv`. |
| I41 | Truncation stability: 7/12 stable, 2 unstable; Spearman(s₀, logP \| dense) = −0.40; 60%-trace R_max within ±10% for 9/9; CI contains 0 for 7/9 | **S** | `scaling_law_iter41_stability.tsv`, `_summary.tsv` | Reproduced. |
| I45 | Iso-compute: α_dense = 1.030 (ρ 0.714), α_MoE = 0.057 (ρ 0.143); max iso-compute gap 0.694; 3/3 pre-registered predictions pass | **S** | `scaling_law_iter45_scaling.tsv`, `_predictions.tsv` | Reproduced. Caveat inherited from upstream: the compute table carries Nemotron `R_max = 2.0`, a bound-saturated fit artifact the section itself acknowledges only later (iter49). |
| I49 | Joint fit R² = 0.18 (all), 0.78 (dense); LOO RMSE 0.504 fails the < 0.30 pre-registration; Nemotron max residual 1.202; P3 picks Qwen3.5-4B | **S** | `scaling_law_iter49_two_param.tsv`, `_predictions.tsv` | Reproduced. |
| I53 | Negative result: τ_b = 0.107, ρ = 0.112 (perm p 0.721), mean \|Δrank\| 4.0, worst 9; critic-degeneracy drop-collapse test Δ\|ρ\| = 0.001 — all three pre-registrations fail | **S** | `scaling_law_iter53_rank_summary.tsv` | Reproduced. |
| I61 | ZVF-stratified degeneracy: HIGH 7/9 vs LOW 2/3 at bound; Nemotron largest jackknife residual 0.646; step-ZVF 0.4125 vs rollout 0.1583; 5/5 predictions pass | **S** | `scaling_law_iter61_zvf_proxy.tsv`, `_stratum_fit.tsv`, `_jackknife.tsv`, `_cross_pillar.tsv`, `_predictions.tsv` | Reproduced (tables are auto-generated from these files). |
| I65a | Geometric three-phase test: only 2/12 conform; plurality valley (6/12); PCI means 1.75/1.58 | **S** | `scaling_law_iter65_phase_pieces.tsv`, `_arch_phase.tsv`, `_predictions.tsv` | Reproduced. |
| I65b | "What iter 65 proves": "**Nemotron-120B is the unique collapse anchor** (peak-then-decay shape, m₁ > 0, m₃ < 0)" | **U** | `scaling_law_iter65_phase_pieces.tsv`, `_predictions.tsv` | Contradicted by the section's own artifact: Nemotron `phase_class = valley`, `n_collapse = 0` (P2 explicitly FAILED), and its fitted m₃ = +0.0000, not < 0. The conclusion paragraph re-asserts the prediction the data rejected. |

### Frontier synthesis (`frontier_synthesis_scaling.tex`)

| # | Claim | Label | Evidence checked | Notes |
|---|---|---|---|---|
| F1 | "the paper reports a zero-reward fraction of exactly 0.55 for Nemotron-120B … against ≤ 0.067 on every well-behaved anchor" | **S** | `scaling_law_nemotron_rootcause.tsv` | Reproduced (here correctly labeled zero-*reward* fraction, unlike the abstract). |
| F2 | "Our same-stack control found GRPO and PPO statistically indistinguishable (paired **Δ = +0.001, p = 0.75**)" | **U** | `samestack_ppo_grpo.json` (`mean_diff_grpo_minus_ppo = −0.002, t = −1.0, p = 0.3739`) | Number not reproducible from any file found in `experiments/results/`. The qualitative claim (indistinguishable) holds, but both the Δ and the p-value are wrong as printed; the same control is quoted correctly (p = 0.3739) in `scaling_law_iter29.tex`. |
| F3 | "the paper's identifiability audit finds 0/5 per-trace saturation rates estimable" | **S** | `scaling_law_iter25_summary.tsv` | Reproduced. |
| F4 | External frontier-model content (gated law, C_eff axis, BEI ≥ 0.97, curve-collapse criteria), sourced to `frontier_calls/digests/frontier_P1.md` | **W** | — | Properly framed as external synthesis / falsifiable proposals, not measurements. However the cited source digest **does not exist in the repo** (`frontier_calls/` absent), so the provenance is unverifiable. |

### Conclusion (`p1_conclusion.tex`)

| # | Claim | Label | Evidence checked | Notes |
|---|---|---|---|---|
| C1 | GRPO reward gain shows no measurable monotone dependence on model size; no universal saturation exponent identifiable | **S** | `scaling_law_cross_scale.tsv`, `scaling_law_power_law.tsv`, `scaling_law_iter25_*.tsv`, `scaling_law_iter109b_permtest.tsv` | Reproduced across four independent tests (same "gain" caveat as A2c). |
| C2 | Three-phase saturation hypothesis geometrically falsified | **S** | `scaling_law_iter65_*.tsv`, `scaling_law_iter33_predictions.tsv`, `scaling_law_three_phase.tsv` | 2/12 geometric, P1–P4 falsified, 0/5 BIC passes — consistent. |
| C3 | Nemotron-120B trace best treated as a distinct collapse phase | **W** | `scaling_law_nemotron_rootcause.tsv` (`is_collapse=True`, unique) vs `scaling_law_iter65_phase_pieces.tsv` (`valley`) and `scaling_law_iter33_nemotron` battery (2/3 criteria, positive post-peak slope) | The paper's own classifiers disagree (collapse under the iter-9 criteria, valley under iter-65 geometry, 2/3 under iter-33); "best treated as collapse" is one of three in-repo readings. |
| C4 | What survives is a taxonomic, endpoint-level regression of the ceiling on log₁₀N | **W** | `scaling_law_iter133_interaction_aic.tsv`, `scaling_law_iter137_t80_scaling.tsv` | Same problem as A6: iter-133 shows the capability class, not log₁₀N, carries the signal (adding log₁₀N *worsens* AICc), and the iter-137 ceiling-on-log₁₀N slope is n.s. |
| C5 | Limitations: single-seed frontier anchors; held-out eval narrower than training coverage; short-horizon GSM8K; λ non-identifiable so rate comparisons excluded | **S** | `scaling_law_iter25_identifiability.tsv` etc. | Accurate, appropriately conservative; matches the data limitations observed throughout. |
| C6 | "The released traces, checkpoints, and scripts…" | **W** | repo | Same as A7. |

---

## 2) Missing evidence list (one item per flagged claim)

1. **A2b / F-syn / conclusion "five orders of magnitude"** — no file spans 5 decades of N. Needed: either anchors below ~0.07B or above ~400T (absurd), or correct the phrase to "≈2.4 orders of magnitude (4B–1T)" / "≈3 orders (0.6B–671B)".
2. **A1 "70+ runs across seven RL libraries"** — no machine-readable run registry under `experiments/results/` backs the count; `framework_comparison.json` covers 4 frameworks (2 dryrun). Needed: a `runs_registry.tsv` (one row per run: pillar, model, library, seed, mode real/dryrun) that sums to ≥ 70 real runs.
3. **A2c / C1 "reward gain"** — no TSV computes Δ(reward) over the base policy per anchor. Needed: a per-anchor base-policy accuracy column so "gain" is actually the regressed quantity, or rewording to "mean training reward".
4. **S2 cross-scale bootstrap CI / n_boot=5000** — `scaling_law_cross_scale.tsv` must be regenerated with n_boot = 5000 or the table re-typed from the file ([−0.796, +0.313], n_boot ≈ 990); the −1.119 bound belongs to `scaling_law_iter121_effective_compute.tsv` R_max row.
5. **S3b "only negative slope"** — contradicted by Llama's −0.00374; needs deletion or restriction ("the only negative post-peak slope among collapse candidates" is still false — Llama's is more negative).
6. **S10 Δ₁T paragraph** — every load-bearing number conflicts with `scaling_law_extended_frontier.tsv` (`delta_first_final`). Needs a full rewrite from the file (see §3), or a regenerated column if the intended definition was R(T)−R(1) restricted to post-peak windows.
7. **S16b iter-121 Spearman CI** — the printed [−0.94, +0.83] matches neither the bootstrap CI [−0.80, +1.00] nor the permutation band [−0.9, +0.9]; regenerate or re-type; B is 1000 in the file, not 2000.
8. **S19b iter-133 three-phase "0/7"** — `scaling_law_iter133_three_phase.tsv` shows 2/7 anchors (Qwen3-8B, gpt-oss-20B) with the full (1,1,1) signature. Either the TSV or the text is stale; the falsification claim must be restated as "2/7" (which still rejects the *universality* claim) or the diagnostic rerun.
9. **I65b "Nemotron unique collapse anchor"** — the iter-65 artifact classifies it valley with m₃ = +0.0000; the concluding paragraph needs to be reconciled with `scaling_law_iter65_predictions.tsv` P2 = FAIL.
10. **F2 PPO control "Δ = +0.001, p = 0.75"** — no supporting file. `samestack_ppo_grpo.json` gives Δ = −0.002, p = 0.374. Correct the numbers (they are quoted correctly in `scaling_law_iter29.tex`) or point to whichever run produced 0.75, if it exists.
11. **F4 frontier digest** — `frontier_calls/digests/frontier_P1.md` is cited as the provenance for the entire frontier-synthesis section but is absent from the repo; commit the digest or drop the pointer.
12. **A5 "zero-variance fraction" label** — no file computes rollout-level ZVF for Nemotron-120B; either compute it or relabel the abstract number as "zero-reward step fraction".
13. **I29b F1 verdict** — TSV says `divergent`; the paper table says "sustained". Align the table with `scaling_law_iter29_stack_compare.tsv`.
14. **S12 iter-21 placeholder** — "$\hat\delta = $ (estimated)" is an unfilled value; the interaction coefficient (9.576 from `scaling_law_iter21_arch_regression.tsv`) should be inserted.

## 3) Rewrite suggestions for risky claims (semantics-preserving)

- **A2b (abstract):** "across five orders of magnitude in parameter count" → **"across nearly three orders of magnitude in parameter count (4B–1T across the extended anchor pool)"**. Same fix in `frontier_synthesis_scaling.tex` ("over five orders of magnitude") and `p1_conclusion.tex` ("five orders of magnitude of single-benchmark evidence" → "roughly three orders of magnitude…").
- **A2c / C1:** "the cross-scale slope of the GRPO reward gain" → **"the cross-scale slope of the mean GRPO training reward"** (or add the base-policy Δ computation).
- **A5 (abstract):** "zero-variance fraction 0.55 versus ≤ 0.067 elsewhere" → **"a zero-reward step fraction of 0.55 versus ≤ 0.067 elsewhere (the step-level analogue of the Pillar-2 ZVF diagnostic)"**.
- **R4 (results intro):** "peak held-out accuracy 0.875" → **"peak per-step training reward 0.875"**.
- **S2 (Tab. scaling-cross):** re-type from `scaling_law_cross_scale.tsv`: boot mean −0.117, CI **[−0.796, +0.313]**; caption "n_boot ≈ 1000 effective resamples" (or regenerate at 5000).
- **S3b:** delete "(the only negative slope in the set)" or replace with **"(a negative post-peak slope, like Llama-3.1-8B-Instruct's −0.0037, but from a 0.667-unit collapse rather than a 0.156-unit drift)"**.
- **S10 (Δ₁T paragraph):** rewrite from the file: **"Across the 12 anchors, 7 of 12 have |Δ₁T| ≤ 0.2. The largest positive (collapsing) gap is Qwen3.5-27B at +0.56; Nemotron-120B's Δ₁T is −0.50 only because its trace *starts* at zero reward — the first-final contrast misses a mid-trace collapse, which is precisely why the peak-vs-late autopsy (Tab. scaling-nemotron-autopsy) is the right diagnostic for it."** (This preserves — indeed strengthens — the section's argument while matching the data.)
- **S16b:** "bootstrap-95% CI [−0.94, +0.83] (B = 2000)" → **"bootstrap-95% CI [−0.80, +1.00] (B = 1000); permutation null 95% band [−0.90, +0.90]"**.
- **S19b (iter-133 b):** "0/7 anchors satisfying the template … no anchor satisfies phase 1 in combination with phase 2" → **"only 2/7 reliable anchors (Qwen3-8B, gpt-oss-20B) satisfy the full improvement→plateau→collapse signature; the dominant pattern (5/7) is collapse-only — the template is not universal across the pool"**.
- **I65b:** "Nemotron-120B is the unique collapse anchor" → **"Nemotron-120B is the pool's structural outlier, but notably even our geometric classifier assigns it *valley* rather than collapse (P2 failed): its early collapse is followed by partial recovery, a shape outside the Nimmaturi taxonomy altogether"**.
- **F2:** "(paired Δ = +0.001, p = 0.75)" → **"(paired Δ = −0.002, p = 0.37; `samestack_ppo_grpo.json`)"**.
- **A6 / C4:** "the defensible object is a local, stack-conditioned saturation law relating the endpoint ceiling to log₁₀N" → **"the defensible object is a local, stack-conditioned taxonomy of endpoint ceilings — dominated by a categorical capability split, with log₁₀N adding no explanatory power on top of it (Tab. scaling-iter133-aic)"**.

---
*Generated by the claim-evidence linter pass on 2026-07-04. All labels verified against files under `experiments/results/`; every "U" row identifies a number that could not be reproduced from the repository data.*
