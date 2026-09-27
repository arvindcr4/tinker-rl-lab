# Statistics fixes: change log (2026-09-27)

This log responds to `statistics_audit.md`, which lists 50 issues: 15 high, 30 medium and 5 low. The edits were made to the `thesis/ch*.md`, `ch_back_run_registry.md` and `frontmatter.tex` (abstract) sources. `thesis_master.tex` was not edited, no PDF was built and nothing was committed.

**Recomputation scripts.** The fixes reuse the auditor's `recompute.py` and `recompute2.py` and add `recompute3.py`, which covers the iter-20 mechanism tests, the iter-136 test, the G-axis F-tests, the E12 app-level test, the stress-test counts and the Fisher CIs. All three scripts are in the session scratchpad and use statsmodels 0.14.6 and scipy 1.17.1.

**Line numbers** refer to the files as they stood after this pass. Other agents were editing the same files at the same time, so the numbers are approximate. Search by the quoted text.

**Test conventions** used throughout:
- Wilson intervals for proportions.
- Fisher-z intervals for correlations.
- Hanley–McNeil intervals for AUROC.
- Paired t-tests with t-intervals, alongside exact sign-flip permutation tests.
- Seed-level TOST for equivalence.
- Benjamini–Hochberg (BH) or Holm correction where a family of tests is involved.

## Summary by severity

| Severity | Fixed | Deferred / partial |
|---|---|---|
| High (15) | 14: H2–H15 | H1 (E14 denominator) was deferred to the figures agent, which owns the E14 51.29% body text. The ch01, ch09, ch10, registry and reproducibility text now carries 51.29%/51.31%. **The abstract (`frontmatter.tex` line ~221) still reads "2,271/4,428 = 51.31% reproduced", which is wrong.** |
| Medium (30) | 30. M14, M19, M21 (ch01) and M24 had already been fixed by a concurrent agent; they were verified, not re-edited. | M16 has no Holm correction, because the per-contrast p-values are not on disk. M18 has no Mantel test, because the per-pair data were not recomputed; its p-values are now labelled non-inferential. M4 has no E14 CI, which is left to the figures agent (Wilson for 2271/4428 is [0.498, 0.528]). |
| Low (5) | 5 | — |

## Conclusions that changed

1. **ZVF as an incremental predictor.** This claim is withdrawn (ch06 §6.3.4, ch10:9). The r = +0.80 came from the n = 9 synthetic variance-mitigation slice. The pooled measured residual is r = +0.205, Fisher 95% CI [−0.31, +0.62], p = 0.43, n = 17.
2. **ZVF association with the final outcome.** "ρ ≈ 0.27 real but weak" is now "not distinguishable from zero": ρ = 0.27, bootstrap 95% CI [−0.37, 0.88], Fisher CI [−0.16, 0.61], p = 0.21, n = 23. This was changed in ch04, ch06, ch10 and the abstract.
3. **ZVF risk-index AUROC 0.929.** It is now labelled a synthetic-panel exercise. All 45 variance-mitigation rows come from the dry-run projection. The 7 collapse anchors have channel values hard-coded in `zvf_diagnostic_iter130.py`. No AUROC is computed on measured runs.
4. **Nine-row ZVF-by-library panel.** The vanilla GRPO row is also synthetic, because it comes from `variance_mitigation.tsv`. ch06 had called it measured.
5. **Group-size TOST.** The step-level p < 10⁻⁹ and p < 10⁻⁴ values are withdrawn. At the seed level, equivalence to G = 2 holds at G = 4 (p = 0.012) and G = 8 (p = 0.043), and is **not** established at G = 16 (p = 0.056). "Retention" is a training-reward ratio.
6. **P4 H7 "DECISIVE".** It is downgraded to **suggestive**. The two-sided exact p is 0.0625, the floor at n = 5, and Holm correction over the 8-hypothesis family gives 0.25. ch10 now says that none of the audit's seven headlines is decisive.
7. **P3 "precise" headlines.** H1 (the GU ratio) and H2 (the retention slope) are withdrawn to hypothesis-generator status.
8. **G axis "dominates" or "is a real lever".** The η² = 0.54 is for training reward, F(3, 8) = 3.08, p = 0.090. The held-out share is η² = 0.41, p = 0.22. The ch06 §6.6 claim that "algorithm label matters much less than sampling configuration" is weakened to "no algorithm-label effect detected; sampling configuration not shown significant".
9. **Same-stack PPO vs GRPO.** Still indistinguishable, and now explicitly **not** equivalent (TOST ±0.005, p = 0.104). The "DECISIVE" equivalence verdict in the factorisation JSON is rejected.
10. **Qwen PPO/GRPO "reversal across model families".** Now described as a descriptive gap. Its d, CI, power and p are non-inferential.
11. **Pooled McNemar values in the GSM8K-CoT study.** The conclusion survives, but the evidence is weaker. With the seed as the unit, p = 0.0051 for GRPO and p = 0.0099 for Dr. GRPO, where the pooled test gave 1.1e−4 and 7.6e−4.
12. **"No baseline comparison of any kind".** Reworded to state what exists. For the campaign lanes there is no matched base arm. The paired trained-versus-base comparison in Table 9.B (14 lanes, 5–151 items each) finds no significant difference in any lane. The controlled pre-to-post held-out gains are reported in ch06.
13. **E11 provenance.** The 129/312 run used `seed809_final` and has `outstanding_blockers: []`. The step-40 caveats belong to the 4-task pilot.

## Change log

### High

| ID | File:line | Old | New | Recomputation / method |
|---|---|---|---|---|
| H1 | ch09:95, ch01:86, ch10:19 | 2271/4428 = 0.51310 | **Deferred to the figures agent.** It already changed the body text to "51.29% (errors as failures); 51.31% official over 4,426". | 2271/4428 = 0.51287; 2271/4426 = 0.51310. **The abstract (frontmatter ~221) still says 51.31% "2,271/4,428"**, which needs the same fix. |
| H2 | ch06:77 | AUROCs on the "52-row panel" presented as measured | Adds that the 45 VM rows are synthetic and the 7 anchors have imputed channels, that the index was selected post hoc, and the Hanley–McNeil CI [0.82, 1.00] for this panel only. No measured-row AUROC. | `zvf_diagnostic_iter130.py` lines 150–196 (hard-coded anchor channels); `zvf_summary.tsv` (VM rows come from `variance_mitigation.tsv`); HM CI with 11 positives and 41 negatives. |
| H3, L4 | ch06:75 | Naive r = 0.22 [−0.29, +0.94], n = 23; residual r = +0.80 [+0.54, +1.00], "strong" | Naive r = +0.22, Fisher CI [−0.29, +0.63], n = 17; residual r = +0.205 [−0.31, +0.62], p = 0.43, n = 17. The +0.80 is labelled as the synthetic n = 9 slice. The source's "not an incremental predictor" wording is quoted and kept. | `zvf_iter26_residual.tsv` (`pooled_all_per_experiment` row); Fisher z. |
| H4 | ch06:61, :63 | "The vanilla GRPO row … are the measured content"; "60 measured runs" | The whole nine-row panel is labelled synthetic. Of the 60 runs, 45 are projection rows; the measured subset includes the 12-run G sweep. | `zvf_by_library.tsv` (every row's evidence_path is `variance_mitigation.tsv`); `p2_results_intro.tex:15`. |
| H5 | ch06:89, :93; ch04:92; ch10:11 | TOST p < 10⁻⁹ / 10⁻⁴; "retention" | Seed-level paired TOST on last-10 training reward against G = 2, margin ±0.02, n = 3: G = 4 p = 0.012, G = 8 p = 0.043, G = 16 p = 0.056. The quantity is called a "training-reward ratio". The step-level p-values and intervals are withdrawn. | recompute2.py (groupsize_zvf_sweep.json). |
| H6 | ch06:159; ch10:9 | d = +2.68, one-sided p = 0.031, perm p = 0.063, "DECISIVE"; "four DECISIVE, three NULL" | d is d_z. The p = 0.031 is the exact one-sided Wilcoxon floor. Paired t(4) = 5.96, two-sided p = 0.004; exact two-sided sign-flip p = 0.0625. It is 1 of 8 family hypotheses, and Holm gives 0.25. Graded **suggestive**. ch10 now says none of the seven headlines is decisive. | recompute3.py from `length_bias_iter136_step_coupling.tsv` and `_paired_tests.tsv`; `length_bias_iter136.py:268` (d definition). |
| H7 | ch06:99 | GU ratio CI [4.16, 4.82] "precise"; slope [−0.237, −0.038] "precise" | Notes that the point estimate 5.03 lies outside the CI, which is over n = 4 budgets with mean 4.456, and quotes the audit's own "cannot support CI". The slope is fitted to 4 reconstructed budgets. Both are withdrawn to hypothesis-generator status. | `adding_error_bars_summary.json` H1/H2. |
| H8 | ch09:5, :115, :215; ch10:21, :56; ch04:124; ch01:63; abstract | "no baseline comparison of any kind" / "no controlled paired baseline run" | Campaign lanes have no matched base arm. Table 9.B is the paired trained-versus-base comparison (14 lanes, 5–151 identical items, one engine), with no significant difference and described as inconclusive. ch06 pre-to-post held-out comparisons are cited. | Table 9.B p-values verified by recompute.py/recompute2.py. |
| H9 | ch06:155; ch02:35 | McNemar pooled p = 1.1×10⁻⁴ / 7.6×10⁻⁴ | Seed-level one-sample t on the 3 per-seed deltas: p = 0.0051 (GRPO) and p = 0.0099 (Dr. GRPO), n = 3 × 200. Per-seed McNemar values are given as detail. GRPO vs Dr. GRPO paired by seed p = 0.25. The pooled values are rejected as pseudo-replicated. | recompute2.py (drgrpo_gsm8k_cot.json). |
| H10 | ch06:147 | −0.43 [−0.58, −0.28] p = 0.036; −1.11 … p = 0.047; −0.10 … p = 0.087; GSM +4.41 [−4.58, 19.84] p = 0.57 | Dr − GR paired by seed, n = 5, with paired-t CI, paired-t p and exact sign-flip p: OLS +0.43 [+0.18, +0.68] p = 0.009 (perm 0.0625); TS +1.11 [+0.33, +1.88] p = 0.017; ρ(L, R) +0.10 [−0.02, +0.22] p = 0.080; lag-1 −0.01 p = 0.94; GSM8K OLS −4.41 [−37.7, +28.9] p = 0.63 (n = 3). BH over the 14 exploratory tests gives q = 0.051 and 0.058. | recompute3.py from `length_bias_mechanism_per_run.tsv`. The sign in the source is inverted (the source's own table gives GRPO −3.65 and Dr. GRPO −3.22). |
| H11 | ch10:29 | "Each Tinker configuration ran once … no Tinker-side significance tests"; correlation "could narrow the true intervals" | "Most … ran once", with the exceptions named (the 3-seed head-to-head, the 5-seed Qwen3-8B control, the scale extension). Positive correlation makes the reported intervals too narrow, so the true intervals are wider. | — |
| H12 | ch_back_run_registry.md:115–125, :211 | A.4 labelled "Qwen/Qwen3-8B, Tinker, GSM8K held-out" | Qwen/Qwen2.5-0.5B, synthetic arithmetic_correctness, Modal A10, 40 steps, 16 prompts per step, 200 held-out problems per run, seeds 42/123/456, with a correction note. A note now says ch04 and ch06 cite A.4, not P3-C1 (Qwen3.5-4B / rlvr-openings, a different experiment). | `groupsize_zvf_sweep.json` run fields (model, task, gpu, n_steps, n_prompts). |
| H13 | ch04:82, :90; ch06:77, :167; ch10:9, :58; abstract | "ρ ≈ 0.27 real but weak" | "ρ = 0.27, 95% CI [−0.37, 0.88] (bootstrap; Fisher [−0.16, 0.61]), p = 0.21, n = 23: not distinguishable from zero". The collapse association is given as ρ = 0.56, Fisher CI [0.19, 0.79]. | Fisher z (recompute2.py/3.py); registry P2-C2 bootstrap CI. |
| H14 | ch10:41 | E11 via `step_seed809_40`; blockers outstanding | Cites `e11_full_receipt.json`: `sampler_weights/seed809_final`, `outstanding_blockers: []`. The step-40 blocker is moved to the 4-task pilot. | `e11_full_receipt.json` and `e11_trained_step40_receipt.json` fields. |
| H15 | ch01:25–27 | d = −0.14 [−1.02, 0.74], "observed power 0.06"; Llama d = 12.75 [8.49, 17.00], p < 0.001, "reverses" | A concurrent agent had already rewritten the Qwen paragraph with no CI or power. The Llama sentence now reads as a descriptive gap with non-inferential statistics and no "reversal". | n = 10 is autocorrelated last-10 steps of one run; observed power is a function of p. |

### Medium

| ID | File:line | Old | New | Method |
|---|---|---|---|---|
| M1 | ch09:85; ch05:64 | 450/1967 = 0.2288, no CI or conditional rate | Wilson [0.211, 0.248]. Among parsed responses, 450/708 = 0.636 [0.600, 0.670]. Truncation (1,086) is named as the dominant failure. | Wilson; `labbench_final_diagnostics.json`. |
| M2 | ch09:91 | 129/312, no CI | Wilson [0.360, 0.469]. Per framing: 67/156 [0.354, 0.508] and 62/156 [0.324, 0.476]. Among extracted responses, 129/162 = 0.796 [0.728, 0.851]. | Wilson. |
| M3 | ch09:93 | "roughly 0.64 pp" | "at most 0.64 pp on spec-to-RTL (1/156); 0.13 pp overall" | 129/311 − 129/312. |
| M4 | ch09:85, 87, 91, 101, 215; ch10:19; ch01:86; ch05:64; abstract | no intervals | Wilson CIs: E1 2/731 [0.08%, 0.99%]; E8; E10 88/97 [0.833, 0.950]; E11. E14 left to the figures agent. | Wilson (recompute.py). |
| M5 | ch10:19, :39; ch01:86; ch05:64; ch09:206 | "AgentDojo … 97/97" as the headline | "97/97 episodes completed, utility 88/97 = 0.907 [0.833, 0.950]" | Receipt. |
| M6 | ch09 Table 9.A E4 row; Table 9.B E4 row; E4 caveat; note after the tables | "0.068 ≈ [0.01, 0.47]"; "CI [−0.080, 0.000]" | "one nonzero task (0.409); not estimable"; "one nonzero pair (−0.160); no interval". Per-task rewards are stated. | E4 `result.json` and `paired.json` per_item. |
| M7 | ch09 Table 9.B E12 row and note | McNemar p = 0.3 on 151 clustered items; judge not disclosed | Adds an app-level exact sign-flip test, p = 0.25 (n = 6 apps; diffs −0.042, +0.030, 0, −0.160, 0, −0.042). Discloses that the judge is the same Qwen3.6-35B-A3B base model, and that judge v1 and v2 give 0.536 vs 0.808. | recompute3.py; `E12/result.json` grader and caveats. |
| M8 | ch09:195 | "…unless its test excludes zero" | Adds that none of the 17 tests is significant, that the result is inconclusive rather than evidence of equivalence, that McNemar needs ≥ 6 one-directional discordant items at n = 5–10, and that no TOST was run. | Exact McNemar: 6/0 gives p = 0.031 and 5/0 gives p = 0.0625. |
| M9 | ch06:37, :97, :165; ch01:29; abstract | p = 0.374 and "permutation p = 0.62"; "η² = 0.54 for terminal accuracy"; "DECISIVE" equivalence | The primary test is paired t, p = 0.374 (n = 5 × 200). Exact sign-flip p = 0.625. TOST ±0.005 p = 0.104 (not equivalent); ±0.010 p = 0.008. The ceiling is noted. η² = 0.54 is relabelled as `eta2_G_last10` training reward, F(3, 8) = 3.08, p = 0.09; held-out η² = 0.41, p = 0.22. The ch06 §6.6 overreach is softened. | recompute2.py; recompute3.py; `unpacking_dpo_ppo_factorization.json`. |
| M10 | ch06:87, :109, :121, :129; ch02:35 | unlabelled ± | G sweep: mean ± SE with SDs 0.0076, 0.0029, 0.0050, 0.0104 and n = 3. ch06:121 is SD over seeds. Layer-freeze is SD, n = 4. ch02 is SE, n = 5 (SDs 0.006, 0.008). The instability index is labelled "unlabelled in source; magnitude implies SD". | groupsize_zvf_sweep.json; drgrpo_vs_grpo.json; scaled_4seed_result.json. |
| M11 | ch06:71; ch04:78; ch07:49 | Residuals against the held-out-accuracy null (−0.126 … −0.073); ch04 "expected direction when signal is real" | Only the per-step-average null is used: 0.828, 0.709, 0.597, 0.522, giving residuals +0.010, +0.055, +0.094, +0.109, with measured ZVF above the null. ch04 notes that the deviation runs in opposite directions on the Qwen3-8B run and interprets neither. ch07 adds the per-step comparison. | recompute2.py (per-step mean_reward^G + (1 − mean_reward)^G). |
| M12 | ch04:78 | r = 0.92 "validated" | "checked … a shape check, not a validation; partly mechanical" | — |
| M13 | ch04:82 | pooled N = 15, within-GSM8K N = 19 | States that the N = 19 is a separate, larger Tinker GSM8K pool, not a subset of the 15 (per `paper_P8_workshop.tex:97–110`). | Source text. |
| M14 | ch07:13 | η² = 1.0000, 22×/133× | Already fixed by a concurrent agent (design called degenerate; ratios dropped). Verified. | — |
| M15 | ch07:37 | "3,230 … 13×31×10"; "bootstrap CI [0.65, 1.00]" | 13×31×10 = 4,030 generated, 800 no-ops skipped, 3,230 applied. The 7/7 interval is relabelled Wilson [0.65, 1.00], noting that a percentile bootstrap would be degenerate. | `registry_stress_per_run.tsv` (4,030 rows: 3,230 caught, 800 skipped_noop). |
| M16 | ch07:39 | "statistically significant decreases" | "within-run intervals (seed 0 only, prompt-step resampling), not replication, uncorrected". The claim that the differences exceed noise is marked untested. | Holm is not applied because per-contrast p-values are not on disk (partial). |
| M17 | ch07:63 | Hindsight "ceiling" 0.797, "tighter than Dualformer-Auto" | It is not a cost ceiling, since Dualformer at 0.657 is cheaper. The objectives differ (saturated prompts are held at G = 8). Relabelled as a constrained optimum and "bound". | Arithmetic. |
| M18 | ch07:23 | Spearman ρ = +0.529, p ≤ 10⁻³ | Notes that the unit is overlapping cell pairs, so the naive p is invalid. The ρ values are descriptive only, and no Mantel test was run. | Partial: no Mantel test run. |
| M19 | ch07:21 | In-sample R² | Already fixed by a concurrent agent (out-of-fold R², degenerate comparator). Verified. | — |
| M20 | ch06:137, :141, :147, :149 | ~15 uncorrected tests; "one fully significant cell"; "reliably" | Labelled exploratory: 3 seeds give 10 bootstrap resamples, and some units are within-run. BH across 14 tests leaves only the sign-flip cell below 0.05 (q = 0.014), and that cell's unit is the window. "Reliably" and "real" are softened. | recompute3.py BH. |
| M21 | ch06:107, :109; ch10:74; ch01:76 | "+0.125, +0.042, 0.80 → 0.85" without n; ch01 "200–500" vs ch10 "8–20" | Curriculum n = 20 (1 seed, 8 steps; +0.05 = 1 item). Loss formulations n = 12–20 per arm. ch10 now reads "Phase-1 probes' n of 8–20 (controlled comparisons already 200–500)". ch01 had already been reconciled by a concurrent agent. | curriculum_opening/FINDINGS.md; robustness_smallscale_nulls.tex. |
| M22 | ch06:129, :139 | 36% vs 71%; 37.5% as an "honest upper bound" | Wilson: 4/11 [15%, 65%]; 12/17 [47%, 87%] (the intervals overlap); 2/10 [5.7%, 51%]; 4/6 [30%, 90%]; 6/16 [18.5%, 61%]. "Upper bound" is dropped. | Wilson. |
| M23 | ch04:24; ch01:55 | "fixed in advance"; O4 "pre-registered effect-size and power accounting" | "declared … not pre-registered (206 iterations)"; O4 now says "declared (not pre-registered)". | — |
| M24 | ch04:110 | Frozen sampling 0.1/0.95/128 stated globally | Already scoped to the training-selection protocol by a concurrent agent. Per-lane settings are in ch09 §9.1. Verified. | — |
| M25 | ch06:93 | R² = 0.948, asymptote CI [0.712, 0.745] | Marked illustrative (3 parameters, 4 reconstructed points, 1 residual df). The CI is dropped. | recompute2.py R(T) fit. |
| M26 | ch08:31; ch04:102 | AUCs with no CI; accuracy 0.792 | HM CIs: LLM [0.42, 0.55] (100 positives / 400 negatives); XGB [0.75, 0.84] (144 / 9,856). The 0.800 majority baseline exceeds 0.792. Precision 34/47 [0.58, 0.83], recall 34/144 [0.17, 0.31]. | Hanley–McNeil; Wilson; `xgboost_results.json`. |
| M27 | ch04:82; ch10:11 | "0.95–0.97 at both reward extremes" | "0.95–0.97 at the low-reward extreme, 0.87–0.96 at the high-reward extreme" | ch07:15 source values. |
| M28 | ch01:29; ch_back_run_registry.md:223 | Percentile-bootstrap CIs at n = 8 | Paired-t CIs: DAPO [−0.00632, +0.00832], GSPO [−0.00357, +0.01357], Dr. GRPO [−0.01272, +0.00872], AERO [−0.01029, +0.00879]. BH-adjusted p ≥ 0.84 is stated. | recompute2.py (`zvf-program/audit/results/full`). |
| M29 | ch09:83, :113; abstract | E11 both "strictly complete replacement scope" and "original-contract full-suite" | Classified as an original-contract full-suite result that also meets the strict-completeness check, and stated to be one result. The abstract now reads "the replacement scopes LAB-Bench, AgentDojo, and the original-contract suite VerilogEval". | Table 9.A scope; ledger §2/§3 double listing. |
| M30 | ch09:197, :204, :206 | "lost seed809 adapter" | "Tinker sampler route lost; weights preserved on HF branch" | ch09:11; `ADAPTER_AVAILABILITY_CHECK_2026-09-26.json`. |

### Low

| ID | File:line | Change |
|---|---|---|
| L1 | ch06:87 | Unrounded values: 0.9817 vs 0.9783, G2 − G16 = +0.0033. |
| L2 | ch09:195 (note) | E13 1.875 is the mean of per-environment differences (recomputed 1.876), not 25.510 − 23.630. In 22 of 26 episodes the two arms are identical. The table row is unchanged, because E13 is a rerun lane. |
| L3 | ch01:17 | "asymptoting" had already been removed by a concurrent agent; my edit ("the largest group size measured") was superseded. Verified. |
| L4 | ch06:75 | Fisher intervals replace the unstable bootstrap bounds (merged into H3). |
| L5 | ch06:43 | κ values given: heuristic vs AIC +1.000 (5 anchors) and +0.625 (12 anchors); changepoint vs the others −0.029 to +0.077 (`scaling_laws.tex` Table `tab:scaling-iter17-kappa`). |

## Not touched, and risks

- **Rerun lanes.** E1, E2, E5, E6, E9 and E13 replacement-lane figures (35/300, 0/45, 20/97, 0/812, 0/34, 13/255) were left as they are, and no rerun results were inserted.
- **Generator overwrite.** The small-scale block (`<!-- small-scale:begin -->`) is generated by `outputs/e1_e14_small_scale_2026-09-26/build_ch9_section.py`. Re-running it would revert the M6, M7, M8, L2 and M30 table and caveat edits unless the generator is updated to match.
- **Table 1.1.** ch01 references a "canonical-numbers" Table 1.1 that was not found in any source file. If another agent adds it, it should carry the corrected values: ρ 0.27 not significant, the seed-level TOST, the seed-level McNemar replacement, and E10 88/97.
- **Scope overlap.** A concurrent agent other than the figures and citations agents rewrote ch01, ch07 and ch08 during this pass, and several items (M14, M19, M24, H15-Qwen, L3) were fixed there first. The edits were applied as surgical replacements against the current disk text.
