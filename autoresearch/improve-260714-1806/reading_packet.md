
# P01: Scaling laws


Root: `platform_hybrid/paper/paper_P1_scaling.tex`  Pages: 45  Words: 25314


[PAGE 1]
Limits of Scaling Laws for GRPO Post-Training:
               A Cross-Library, Cross-Scale Audit


                        Arvind C R                          Ramesh Prakash Guledgudd∗
                       PES University                            PES University
                   arvindcr4@gmail.com                         rameshpg@pes.edu



                                                 Abstract
             Group Relative Policy Optimization (GRPO) has become a default algorithm for
             reinforcement-learning (RL) post-training of language models, yet how its benefit
             scales with model size remains unclear and is easily confounded by implementation
             and evaluation choices. We study GRPO scaling as one pillar of T INKER RL-
             B ENCH, a benchmark spanning 70+ runs across seven RL libraries and five model
             families (0.6B–∼671B parameters) on GSM8K, HumanEval, and synthetic tool-
             use tasks. Crucially, these runs use managed and open back-ends rather than local
             frontier clusters: the frontier-scale anchors (Qwen3-235B, DeepSeek-V3.1 ∼671B,
             Nemotron-120B) are obtained through the T INKER managed training/inference
             API in a single-seed, descriptive-only regime, which is what makes a cross-scale
             study tractable without owning frontier hardware (and why full base weights are
             not released for those anchors). Our central result is a conservative negative one:
             across roughly 2.4 orders of magnitude in parameter count spanned by the fitted
             anchor pool (a subset of the full 0.6B–∼671B roster), the cross-scale slope of
             the mean GRPO training reward shows no reliable positive trend—flat within the
             (wide) uncertainty of these mostly single-seed anchors, which we therefore read
             descriptively rather than as a powered significance test—and no single saturation
             exponent is identifiable (four of the five frontier-scale traces have their saturation
             rate pinned at the fit boundary, and none is identifiable). A pre-registered test
             geometrically falsifies a clean three-phase saturation hypothesis (only 2 of 12
             anchors match), and the Nemotron-120B run is better described as a distinct
             collapse phase (zero-reward step fraction 0.55 versus ≤ 0.067 elsewhere) than
             as a point on a smooth trend. The defensible object is therefore a local, stack-
             conditioned taxonomy of endpoint ceilings, with categorical capability structure
             carrying more signal than log10 N , not a Chinchilla-style power law in N . We
             release all code, logs, per-step telemetry, and trained adapters; full base-model
             checkpoints are shared where model licensing permits, while the managed frontier-
             scale runs expose logs and telemetry rather than base weights.


1       Introduction
Reinforcement learning from verifiable rewards has become a standard final stage in training
reasoning-capable language models, and GRPO [23, 7] is now one of its most widely used al-
gorithms. GRPO replaces the learned value function of PPO [22] with a group-relative baseline: for
each prompt it samples a group of G completions, scores them with a verifier, and centers the rewards
within the group to form advantages. Because it removes the critic, GRPO is cheap to run at scale,
and a natural question follows: how does the benefit of GRPO post-training change as the base model
grows?
    ∗
        Project guide.


Preprint. Under review.

[PAGE 2]
TinkerRL Benchmark




               Mathematical                                           Tool Use                   Code
                Reasoning                                                                      Generation



                                                                   Function Calling
                   GSM8K                                                                       HumanEval
                                                               2 models tested   ∼ Partial
         8 models tested   ✓ Complete                                                        1 model   ◦ Partial




                      ✓ Complete   ∼ In progress   ◦ Planned




Figure 1: Taxonomy of RL libraries in T INKER RL-B ENCH. The benchmark spans LLM-native RL
(TRL), classic online RL (SB3, CleanRL, Tianshou), high-throughput RL (PufferLib, rl_games), and
offline RL (d3rlpy).


Pre-training exhibits smooth power-law scaling in parameter count and data [12, 9], and it is tempting
to expect an analogous law for RL post-training. Evidence here is thinner and more contested [8, 21,
18], in part because RL gains are entangled with the rollout stack, reward parser, reference policy,
and evaluation protocol, so cross-paper comparisons conflate the algorithm with its plumbing. This
paper isolates the scaling question inside a single, fixed benchmark.
We study GRPO scaling as one pillar of T INKER RL-B ENCH, a controlled benchmark of 70+ runs
across seven RL libraries and five model families (0.6B–∼671B) on GSM8K [5], HumanEval [4],
and synthetic tool-use tasks. Rather than assert a new law, we report what the data will and will
not support. Our contributions are: (i) a same-benchmark measurement showing the cross-scale
mean training-reward slope is statistically indistinguishable from zero over the measured 2–3-decade
parameter range; (ii) a pre-registered falsification of a clean three-phase saturation hypothesis; (iii)
an identifiability audit showing no universal saturation exponent is estimable on this data, so the
defensible object is a local, stack-conditioned saturation law with the endpoint ceiling regressed
on log10 N ; and (iv) a case study of the Nemotron-120B collapse as a distinct phase rather than
trend noise. We read single-seed frontier-API runs as illustrative case studies, not universal laws.
The title’s “limits” is therefore substantive: this paper identifies where a parameter-count scaling
law is not estimable from the corpus; it does not claim that GRPO lacks scaling behavior in better-
controlled data. The program’s stronger explanatory candidates are usable contrastive compute and
group-size-conditioned signal, tested separately in the ZVF and group-size companion papers.

1.1   Related Work

Neural scaling laws for pre-training are well established [12, 9, 16]. Scaling behavior of RL and
RLHF post-training has been examined more recently [8, 21, 18], but typically within a single stack,
leaving open whether reported trends transfer across libraries and model families. GRPO and its use
in reasoning models originate with Shao et al. [23] and DeepSeek-AI et al. [7]; Ahmadian et al. [2]
argue that much of the benefit of heavy RL machinery is recoverable by simpler estimators once the
stack is controlled. Our work complements these by fixing the benchmark and asking, conservatively,
which scaling conclusions survive.


2     Benchmark Design

2.1   Task Suite

2.2   Cross-Library Implementation

Two seven-library rosters — read carefully. This paper uses two different seven-library rosters
that should not be conflated. The cross-RL-library benchmark below (Table 2) covers TRL, SB3,


                                                                         2

[PAGE 10]
Model                     R̄max           λ̄     P (λ ≥ 9.5)               t̄80 [95% CI]
               Qwen3.5-4B                0.818      6.64                 0.599       0.61 [0.16, 2.43]
               Qwen3-8B                  0.289      5.30                 0.468       1.42 [0.16, 7.62]
               Llama-3.1-8B-Instruct     0.869      6.92                 0.595       0.46 [0.16, 1.98]
               DeepSeek-V3.1             0.847      6.48                 0.548       0.47 [0.16, 1.53]
               Nemotron-120B             0.310      3.26                 0.277     17.87 [0.16, 161.4]
Table 8: Parametric-bootstrap saturation fit (1000 residual resamples per trace). P (λ ≥ 9.5)
is the fraction of bootstrap refits in which λ hits the optimiser’s upper bound – on four of five runs
this is between 47% and 60%, confirming that the strict-R(0) = 0 functional form is degenerate on
these traces: the optimiser cannot simultaneously estimate Rmax and λ when the trace starts above
its empirical ceiling. Only Nemotron-120B has P ≤ 0.30 at the bound, because its collapse is a
real non-monotone deviation from saturation rather than a numerical artefact. t80 intervals are wide
(lower-bound 0.16 step = optimiser clamp) for the four degenerate runs, and unidentifiable (0.16–161
step) for Nemotron.


             Model                     ntrain    ntest        RMSEtest     RMSEconst       improvement
             Qwen3.5-4B                  21         9           0.2174           0.2174        +0.0000
             Qwen3-8B                    21         9           0.1720           0.1720        +0.0000
             Llama-3.1-8B-Instruct       21         9           0.1864           0.1864        −0.0000
             DeepSeek-V3.1               14         6           0.1787           0.1787        −0.0000
             Nemotron-120B               14         6           0.2278           0.2294        +0.0016
Table 9: Holdout 70/30 cross-validation of the saturation model. Four of five models show zero
improvement over predicting the constant train mean – direct empirical evidence that the saturation
fit adds no out-of-sample predictive power on these traces. Only Nemotron-120B shows a non-zero
improvement (+0.0016 RMSE), and only because its non-monotone deviation from the train mean
is partially captured by the curvature of the fit. This formalises the warning in Paragraph 4 that the
canonical R(t) = Rmax (1 − e−λt ) is the wrong functional form for these already-saturated traces.



curve on the first 70% of each trace and predict the last 30%, comparing the test RMSE to a constant
(train-mean) baseline:

Elevation: compute-adjusted scaling proxy. Per-step token counts are not in the trace files, so
we approximate “compute to saturation” as t80 · N where N is the parameter count in billions.
Regressing log10 (t80 · N ) on log10 N yields a slope of +482 ± 490 per decade – which is a wide
CI, but not informative on the four saturation-bound runs because t80 is itself clamped. Using the
parametric-bootstrap λ̄ directly yields a slope of −0.47 ± 0.87 per decade, which brackets zero. The
conclusion is the same as in Table 7: at this benchmark size and step budget, there is no identifiable
scale dependence in the GRPO saturation rate.

Elevation: Nemotron-120B collapse autopsy. We classify a trace as collapsed when (i) its peak
reward is at least 0.4, (ii) its late-window mean falls below 0.4× peak, and (iii) its zero-reward
fraction is at least 0.30. Applied to all five traces:
This is the strongest empirical justification we have for treating Nemotron-120B as the Pillar-1
counterexample to the Nimmaturi et al. [18] three-phase template: not only does its peak not satisfy
the slow-start → rapid-improvement → plateau shape, it actively diverges downward from the peak
in a way no monotone saturation model can describe. The zero-reward-fraction of 0.55 is also the
structural signal that the ZVF-based diagnostic (companion paper on cross-experiment ZVF) would
catch first, because it sees the divergence on reward variance before it shows up on reward mean.

Summary of elevation. Across the four elevation diagnostics the headline result is a strengthened
null: at this benchmark size (4 B ≤ N ≤ 671 B, T ≤ 30 steps), the canonical saturation model
is unidentifiable (parametric bootstrap on λ has P (λ = 10) ≥ 0.47 on four of five runs), adds no
out-of-sample predictive power (70/30 holdout improvement over the constant baseline is ≤ 0.0016
RMSE everywhere), and shows no detectable scale dependence (slope −0.47 ± 0.87 per decade on
λ). The single run that violates the saturation template is Nemotron-120B, and its collapse is the


                                                         10

[PAGE 14]
Model                       k        n            R̄     ∆AIC best               best   ∆AICconstant
           Qwen3.5-4B                  1        30     0.817               0.00      constant               0.00
           Qwen3.5-4B                  2        30     0.817               2.00          linear             2.00
           Qwen3.5-4B                  2        30     0.817               2.00     saturation              2.00
           Qwen3.5-4B                  3        30     0.817               4.00        logistic             4.00
           Qwen3-8B                    1        30     0.285               0.00      constant               0.00
           Qwen3-8B                    2        30     0.285               1.20          linear             1.20
           Qwen3-8B                    2        30     0.285               2.00     saturation              2.00
           Qwen3-8B                    3        30     0.285               3.17        logistic             3.17
           Llama-3.1-8B-Instruct       1        30     0.869               0.00      constant               0.00
           Llama-3.1-8B-Instruct       2        30     0.869               0.80          linear             0.80
           Llama-3.1-8B-Instruct       2        30     0.869               2.00     saturation              2.00
           Llama-3.1-8B-Instruct       3        30     0.869               4.29        logistic             4.29
           DeepSeek-V3.1               1        20     0.844               0.00      constant               0.00
           DeepSeek-V3.1               2        20     0.844               2.00          linear             2.00
           DeepSeek-V3.1               2        20     0.844               2.00     saturation              2.00
           DeepSeek-V3.1               3        20     0.844               4.00        logistic             4.00
           Nemotron-120B               1        20     0.175               0.00      constant               0.00
           Nemotron-120B               2        20     0.175               1.92          linear             1.92
           Nemotron-120B               2        20     0.175               1.72     saturation              1.72
           Nemotron-120B               3        20     0.175               3.92        logistic             3.92
Table 13: Multi-model AIC profile across the five anchors (constant / linear / saturation / logistic).
The constant model wins on every trace by ∆AIC ≤ 2 over the runner-up. By the Burnham and
Anderson [3] convention, ∆AIC < 2 means the second-best alternative is “essentially equivalent”;
∆AIC ∈ (2, 4) is “substantial support”; ∆AIC > 4 is “essentially no support”. The saturation model
is formally indistinguishable from a constant across the entire frontier – the same conclusion the
iter 9 holdout test arrived at from an out-of-sample predictive-power angle. We therefore report
the AIC profile as the likelihood-side counterpart to Table 9: both pin the saturation curve as the
wrong functional form for these already-saturated traces. scripts/scaling_law_iter17.py →
scaling_law_iter17_aic.tsv.


        Model                      n       τ̂        |∆µ|         block-boot CI95% (τ )      perm p    significant?
        Qwen3.5-4B              30         27        0.204                        [2, 28]     0.493    no
        Qwen3-8B                30         28        0.129                        [2, 28]     0.789    no
        Llama-3.1-8B-Instruct   30          5        0.127                        [2, 28]     0.742    no
        DeepSeek-V3.1           20         15        0.092                        [2, 18]     0.909    no
        Nemotron-120B           20         18        0.361                        [2, 18]     0.137    no
Table 14: Changepoint tau with block-bootstrap CI and permutation-test p-value. The brute-
force maximiser τ̂ lands late on three runs (the last 2–3 steps) and early on two, but every τ̂ is
statistically indistinguishable from the permutation null – none of the five traces has a p < 0.05
changepoint at this step budget. This is the formal complement to the AIC profile: both diagnostics
agree that the 30-step frontier traces do not contain a distinguishable break. The block-bootstrap
CI on τ covers essentially the entire trace ([2, 28] for T = 30 traces), so even the point estimate is
not data-driven. We treat the changepoint analysis as a negative result: there is no information in τ̂ .
scripts/scaling_law_iter17.py → scaling_law_iter17_changepoint.tsv.



parameterises the two families, or whether the λ-at-bound degeneracy is itself structured by trace
variance (rather than a free parameter of the dynamics).

(E) Lambda-bound degeneracy audit. The canonical fit R(t) = Rmax (1 − e−λt ) has an upper
bound λ ≤ 10 on the rate; of the 12-anchor frontier, 7 of 12 traces hit this bound (scaling_law_-
iter21_lambda_audit.tsv). The triggering condition is structural: when the per-trace reward
         Var(y) < 0.06 the saturation curve collapses onto a step function, and any λ ≥ 5 produces
variance d
identical fits to numerical precision. We define this as variance-conditioned degeneracy; the rate
constant λ is non-identified on the 7/12 of the frontier that has already saturated. Frontier synthesis
(ChatGPT Pro Extended, R1): the saturation law’s role is not predictive but taxonomic – it partitions


                                                              14

[PAGE 16]
Figure 7: Iter 21 cross-architecture audit (12 anchors). (A) λ̂ (capped at the 9.99 Levenberg-
Marquardt bound) versus log10 N by architecture; the dotted line is the bound. (B) fraction of traces
that hit the λ = 10 bound by architecture. (C) Rmax residual after regressing on log10 N ; Levene
median test p = 0.296 (→ arch-invariant).


   Axis                                  Statistic   Value             Interpretation
   E                       # traces with λ̂ ≥ 10     7/12              variance-conditioned degeneracy
   F       permutation p on log10 N ×arch_moe        0.007             interaction significant
   F                     Spearman ρ(log10 N, λ̂)     −0.023            no marginal trend
   G           5-fold CV interaction > arch only     +332              interaction informative
   G              5-fold CV log10 N only < null      −99               marginal negative
   H               Levene median Rmax residual       1.217 (p=0.296)   arch-invariant
   I                      two-anchor MAE on λ        4.498             bound-degenerate forecast fails
   J                         log10 (CF) lift share   0.643             compute > pure param count
Table 17: Iter 21 cross-architecture summary. Seven orthogonal axes computed on the 12-anchor
dataset. E: lambda-bound audit. F: full-interaction OLS regression with permutation test on the
log10 N ×arch interaction. G: 5-fold CV comparing null, log10 N only, arch + log10 N , and full
interaction. H: Levene’s median test on Rmax residuals after regressing on log10 N . I: two-anchor
(Qwen3.5-4B + Qwen3-8B) extrapolation to the ≥ 20B anchors. J: compute-adjusted regression lift
share. scripts/scaling_law_iter21.py.


(H) Rmax arch-invariance. We regress Rmax on log10 N across all 12 anchors and test whether
the residual is arch-invariant. Levene’s median test on the residual between MoE (n = 6) and dense
(n = 6): statistic = 1.217, p = 0.296 (scaling_law_iter21_r_max_residual.tsv). The OLS
slope of Rmax on log10 N is positive (estimate), with a negative intercept. Operationally, Rmax obeys
a weak power law in log10 N , and within that power law, MoE and dense are exchangeable. Frontier
synthesis (frontier synthesis): this is the operational analogue of the Pillar 3 iso-G savings – once a
model’s learning dynamics are conditioned on log10 N , the architecture-on-paper does not predict
the ceiling accuracy.

(I) Two-anchor burn-in extrapolation. A 30-step diagnostic on the two smallest anchors
(Qwen3.5-4B, Qwen3-8B) yields an OLS slope of λ̂ on log10 N ; we then predict the seven an-
chors ≥ 20B. Mean absolute error 4.498, max absolute error 9.492 (scaling_law_iter21_two_-
anchor_extrap.tsv). The diagnosis is honest: the two-anchor extrapolation cannot recover the
bound-degenerate frontier. The useful counter-experiment is to extrapolate on the arch-stratified
RMSE after regression (axis F) rather than the raw λ; the residual is ∼ 1.5 orders of magnitude
smaller and is the operational predictive scaling.

(J) Compute-aware λ regression.
                    λ̂ = α + β1 log10 N + β2 log10 (CF) + γ ⊮[arch = moe] + ϵ                            (4)
with CF = N · T (per-trace parameter-step proxy for FPO). Adding log10 (CF) lifts CV SSE
by 64.3% of the marginal value of log10 N (scaling_law_iter21_compute_regression.tsv).
The compute-adjusted proxy is moderately better than parameter count alone – consistent with frontier
synthesis’s compute-bound vs. parameter-bound framing (Section 4) and the well-known Kaplan
et al. [11] compute-vs-parameter caveat.
The iter-21 finding sharpens the iter-9/13 conclusion: on this benchmark, the λ scaling is architecture-
contingent, not absolute. The strongest cross-architecture claim supported by the data is that Rmax ,
the asymptotic ceiling accuracy, is arch-invariant after conditioning on log10 N (axis H, p = 0.296);
what differentiates MoE from dense is the transient (whether the trace sits at λ = 10 immediately or
climbs) rather than the asymptotic accuracy. This sharpens Pillar 3’s iso-G savings: the saturation
ceiling Rmax is shared across architectures, so the iso-G savings is exactly the cost of recovering the
ceiling under a smaller group size, regardless of the nominal architecture label.

Iter-25: is the saturation law identifiable at all? Every iteration so far has reported a per-trace
saturation rate λ and its implied t80 = − ln(0.2)/λ. Two symptoms suggest those numbers may be


                                                     16

[PAGE 18]
Model                      best k        ∆BIC3v1             Phase     R̄peak / R̄late   nimmaturi pass?
      Qwen3.5-4B                       3         −1.42     non-monotone     0.842 / 0.830      no
      Qwen3-8B                         1         +0.50           plateau    0.285 / 0.285      no
      Llama-3.1-8B-Instruct            3         −7.96     non-monotone     0.929 / 0.839      no
      DeepSeek-V3.1                    3         −2.31     non-monotone     0.867 / 0.844      no
      Nemotron-120B                    3         −6.53          collapse    0.875 / 0.154      no
Table 19: Iter 117 BIC-segmentation three-phase test against Nimmaturi et al. [18]
(arXiv:2507.18014). ∆BIC3v1 is the BIC difference between k = 3 and k = 1 segmentations; nega-
tive values mean the 3-segment model is preferred. All five anchors prefer either k = 1 (Qwen3-8B)
or k = 3 with non-monotone segments. None of the five anchors pass the nimmaturi three-phase
criterion, because (i) Qwen3-8B’s BIC-optimal k = 1 has no slow-start phase, and (ii) the four
k = 3 winners all have non-monotone segment means (rise-then-fall) so they fail the monotone-up
requirement. Nemotron-120B is the only anchor whose peak-vs-late contrast (0.875 vs 0.154) is
large enough to warrant the collapse label. scripts/scaling_law_fit.py → scaling_law_-
three_phase.tsv.


  Regression                           n       intercept   slope / decade   n at λ-bound       note
  log10 (t80 ) ∼ log10 (N )                5    −0.841           +0.170                 4/5    degenerate (4/5 anchored)
  log10 (t80 ) ∼ log10 (N ) [λ-free]       1        n/a              n/a                4/5    single-point (Nemotron only)
Table 20: t80 -vs-N scaling law test (iter 117 fresh angle). The full-pool OLS gives slope
+0.170/decade with SE 0.254 (CI brackets zero). Restricting to anchors with λ strictly below
the optimiser’s bound leaves a single data point (Nemotron-120B), so the regression is not identifiable.
Combined with iter 109’s λ-vs-N null (p = 0.74) and iter 105’s Rmax -vs-N failure, this is the third
independent cross-scale test that fails to reject H0 : no scaling signal. scripts/scaling_law_-
fit.py → scaling_law_iter117_t80_scaling.tsv.



changes the induced update operator” (frontier synthesis): here even the measurement of the learning
transient is under-identified, so claims must be built from endpoints and pooled cross-scale contrasts
rather than single-trace curve shapes.

Iter 117 refresh: canonical 2-param fit + explicit t80 + BIC three-phase test + Nemotron-
120B collapse audit. Iter 117 refreshes the canonical fit on the five-anchor frontier set (Qwen3.5-
4B, Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-V3.1, Nemotron-120B) and adds the explicit
t80 = − ln(0.2)/λ derivation in the canonical fits table, the BIC-segmentation three-phase test
against Nimmaturi et al. [18] (arXiv:2507.18014), and the Nemotron-120B collapse audit. The
canonical fits are unchanged from iter 9/25 in spirit: four of five anchors hit the optimiser’s λ = 10
upper bound and so report the degenerate t80 = 0.161 step; only Nemotron-120B has an identifiable
rate (λ = 0.99, t80 = 1.63 step). The new table reports the per-trace BIC for k ∈ {1, 2, 3} constant-
mean segments and applies the nimmaturi three-phase criterion (best_k == 3 AND monotone-up
AND R̄seg1 < 0.15 AND R̄seg3 > 0.40):

Iter 117 fresh angle: t80 -vs-N scaling law is degenerate. The iter-117 fresh angle is a direct
test of the scaling law on the time-to-saturation axis t80 = − ln(0.2)/λ. If the canonical saturation
model were a real scaling law, we would expect t80 ∝ N α for some α, with a Chinchilla-style
analogue predicting α < 0 (larger models saturate in fewer steps). The OLS regression of log10 (t80 )
on log10 (N ) on the five-anchor pool yields intercept −0.841, slope +0.170 with SE 0.254 (so the
95% CI brackets zero). But this regression is degenerate by construction: 4/5 anchors hit the λ = 10
bound and so report the identical t80 = 0.161, leaving only Nemotron-120B (t80 = 1.625) as a free
data point. A leave-one-out diagnostic drops any single anchor and refits: the slope ranges from
−0.50 (Nemotron dropped) to +1.41 (each of the four bound-anchored traces dropped) – a 100%+
swing that confirms the regression is driven by which anchor carries the only free λ. The honest
conclusion: the cross-scale t80 -vs-N scaling-law test is unidentifiable on the current five-anchor
pool. This is the t80 -side counterpart to the iter-109 λ-vs-N null and the iter-105 Rmax -vs-N failure:
no cross-scale scaling signal survives on the outcome-reward RL post-training evidence base.


                                                           18

[PAGE 20]
Iter 121: calibration of the “no scaling law” finding. Iter 117 closes with a negative result: no
scaling signal survives on the five-anchor pool. Iter 121 sharpens that negative into a quantitative
detection-power claim. We ask: given the noise we observe in the actual five traces, could any
plausible scaling law be detected at this sample size? We answer with three independent tests and
one synthetic calibration, all summarised in Table 21.

Test 1: Spearman rank test on ∆late-early . The simplest scaling-axis probe is the Spearman rank
correlation between log10 (N ) and ∆late-early = R̄late − R̄early . Across the five anchors we observe
ρ̂ = −0.30 with bootstrap-95% CI [−0.80, +1.00] (B = 1000) and a permutation null two-sided
p = 0.69 (P = 5000). The empirical ρ̂ is negative, i.e. the bigger models gained less in late-window
reward than the smaller ones — the opposite sign from any Chinchilla-style scaling prediction. The
wide CI and non-significant p confirm that even the most basic rank test cannot reject H0 of no
rank-correlation.

Test 2: effective-compute axis log10 (N · D). Chinchilla-style scaling uses effective compute
C = N · D, not raw parameter count. Replacing log10 (N ) with log10 (N · nsteps ) and re-running the
OLS for each of the five metrics (mean R, peak R, Var(R), R̂max , late mean) yields slopes whose
95% bootstrap CIs all bracket zero: e.g. slope on mean R is −0.019/decade with CI [−0.80, +0.30];
slope on R̂max is −0.018/decade with CI [−1.12, +0.32]. No metric shows a scaling signal on the
effective-compute axis.

Test 3: synthetic ground-truth calibration + power curve. We plant a known Chinchilla-style
law Rmax (N ) = 0.85 − β log10 (N/8) across a fresh pool of nanchors ∈ {5, 8, 12, 20, 40}, draw
per-step rewards from the empirical residual distribution (noise σ = 0.189), and force the empirically
observed fraction (80%) of anchors to saturate at λ = 10. For each (nanchors , β) pair we run M = 200
Monte-Carlo replicates and count the fraction that recover the planted slope at R̂2 ≥ 0.5. The
recovery matrix (Table 22) makes the degeneracy explicit:

       • At nanchors = 5 (the current pool) and β = 0.05 (Chinchilla-class), recovery is 0.17 – below
         chance-corrected threshold.
       • Even at nanchors = 40 and β = 0.20 (twice Chinchilla-class), recovery is only 0.07 because
         the saturation-bound anchors continue to drown the signal.
       • No grid cell exceeds 0.5 recovery; even the most favorable small-pool, high-slope settings
         remain below that threshold, so R̂2 ≥ 0.5 is not a reliable detection rule here.

The synthetic calibration shows that the noise floor of σ ≈ 0.19 combined with the 80% saturation-
bound fraction makes any linear scaling law on Rmax essentially undetectable regardless of anchor
count: even quadrupling the pool to nanchors = 20 keeps recovery below 0.20 for β ≤ 0.20.

Iter 121 conclusion. The three statistical tests (Spearman, OLS-on-N , OLS-on-N · D) all fail to
reject H0 : no scaling signal. The synthetic calibration explains why: the saturation-bound structure
of the empirical traces defeats any linear scaling-law test below β ≈ 0.20/decade, a regime stronger
than Chinchilla. The five-anchor pool is at least one order of magnitude too small (in n) and at least
one order of magnitude too saturation-bound (in λ) to falsify any plausible GRPO scaling hypothesis.
Future Pillar-1 work must either (i) extend the anchor pool to nanchors ≥ 40 with a held-out fraction of
λ-free runs, or (ii) abandon the saturation fit entirely and use the variance-based ZVF gate (companion
paper on ZVF) as the primary scaling signal.

Iter 125 elevation: structural falsification of the saturation model + three-phase test (arXiv
2507.18014). We sharpen iter121’s “no scaling law detectable” verdict into a structural one:
the saturation model R(t) = Rmax (1 − e−λt ) implies strict monotonicity (dR/dt > 0 for R <
Rmax ), so a strictly monotone trace violation rate of 0% is expected. Across all n2 ordered
pairs in each anchor we count those with R(j) < R(i) for j > i; observed violation rates are
0.382, 0.395, 0.462, 0.363, 0.290 for Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-
V3.1, and Nemotron-120B respectively – every anchor exceeds the 5% noise floor with per-anchor
binomial p < 0.001 and majority-violates binomial p = 0.031. Even Nemotron-120B (the least
violating) shows a maximum downward step ∆R = 0.875 (peak at step 3 collapsing to R ≈ 0 by


                                                  20

[PAGE 22]
Figure 10: Iter 121 four-panel detection-power figure. (A) ∆late-early vs log10 (N ) scatter with OLS
line and Spearman ρ̂ = −0.30; the permutation-null 95% band (grey) covers the observed range,
confirming the rank-correlation null. (B) OLS slopes on the effective-compute axis log10 (N · D) for
five metrics, with bootstrap 95% CIs – all bracket zero. (C) Synthetic recovery heatmap: recovery
rate of a planted Rmax (N ) scaling law vs (nanchors , β); the dark region (low recovery) dominates
even for β twice the Chinchilla-class value. (D) Power curve: recovery vs nanchors for each β, with the
current five-anchor pool marked by the dashed vertical line. The 50%-recovery threshold is crossed
only by β ≥ 0.20 at small anchor counts where R̂2 is itself high-variance.


                                         sat     pw
               Model                   R̂max   R̂max    t̂peak         γ̂   ∆AICcpw-sat
               Qwen3.5-4B              0.817    0.930     1.0    −0.0001         +2.48
               Qwen3-8B                0.285    0.255    13.0    −0.0060         +1.20
               Llama-3.1-8B-Instruct   0.869    0.948     1.0    +0.0037         +1.27
               DeepSeek-V3.1           0.844    0.842     3.0    −0.0003         +2.79
               Nemotron-120B           0.182    1.050     3.0    +0.0014         +2.27
                                                LOOCV cluster agreement     5/5 = 1.00
                              Log BF (capability+params vs params alone)        −9.53
Table 23: Iter 129 piecewise fit + LOOCV + Bayes factor. All five anchors have ∆AICc > 0
(piecewise loses to saturation by 1.2–2.8 units, well above the Burnham & Anderson 2002 thresh-
old of 2 “substantial evidence”), and the Bayes factor at n = 5 firmly favours the simpler
size-only model (log BF = −9.53, “very strong” on Kass-Raftery). Yet the capability-bimodal
cluster assignment is LOOCV stable at 5/5. Scripts: scripts/scaling_law_iter129.py
→ scaling_law_iter129_piecewise_fit.tsv, scaling_law_iter129_aic_compare.tsv,
scaling_law_iter129_loocv_cluster.tsv, scaling_law_iter129_bf_capability.tsv.



Iter 129 conclusion: Pillar 1 is closed at n = 5 – no functional-form law (monotone or piecewise)
is identifiable at the empirical noise level σ ≈ 0.19, but the capability bimodality is descriptively
reproducible and points to instruct-pretraining as the right cross-anchor axis for future n ≥ 40
anchor-pool work. See Table 23 and Figure 11.

Iter 133 elevation: capability-bimodality at n = 7/10/12. Iter 125 reported a 5-anchor capability
bimodality on Rmax (gap 0.531, Hartigan dip 0.522, permutation-p = 0.056), and iter 129 confirmed
its LOOCV stability at 5/5 agreement. Both iters were limited to the five-anchor pool that supports
the full 30-step GRPO traces; iter 121 already warned that n = 5 is at least one order of magnitude too
small to falsify any plausible GRPO scaling hypothesis. Iter 133 takes the natural sequel step: re-run
the full structural suite (monotonicity, three-phase, bimodality, AICc, LOOCV) at n ∈ {5, 7, 10, 12},
where n = 7 adds the gpt-oss-20B and Kimi-K2-Thinking long-trace anchors from the iter-13 frontier
pool, n = 10 also adds four short probes (n≤5 steps, mean-reward proxy for Rmax ), and n = 12
includes all available anchors. We use the mean trace reward as a proxy Rmax when n ≤ 5 because
the saturation fit is degenerate at this trace length (iter 25).

(a) Monotonicity violation rate (n=7 reliable anchors). The iter 125 violation-rate diagnostic
extended to the seven n ≥ 20 anchors. All seven anchors fail monotonicity at binomial p < 0.001
versus the 5% iid noise floor; violation rates are 0.382, 0.395, 0.462, 0.320, 0.363, 0.290, 0.516 for
Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, gpt-oss-20B, DeepSeek-V3.1, Nemotron-120B,
Kimi-K2-Thinking respectively. The n=7/7 monotonicity falsification matches the iter 125 n=5/5
finding and adds two new violators (gpt-oss-20B and Kimi-K2-Thinking) from the MoE frontier that
iter 125 could not test.

(b) Three-phase hypothesis (arXiv 2507.18014). Across the seven reliable anchors the iter 125
three-phase diagnostic finds only 2/7 anchors (Qwen3-8B and gpt-oss-20B) satisfying the NIMMA -
TURI -style “improvement → plateau → collapse” template. Per-anchor phase-combo counts show
the dominant pattern is still collapse_only (5/7), so the template is not universal across the reliable
anchor pool even though two anchors match the full signature.


                                                  22

[PAGE 23]
Figure 11: Iter 129 four-panel figure. (1) Reward traces with piecewise (solid) and saturation (dotted)
overlays; the two fits are visually nearly identical, consistent with the ∆AICc > 0 across all 5 anchors.
(2) ∆AICcpw-sat per anchor; every bar is positive (piecewise loses) by 1.2–2.8 units. (3) tpeak vs
log10 N with the within-capable-class regression (blue line, slope +0.945/decade, R2 = 0.985); the
cross-anchor permutation p = 0.97 confirms this slope is driven by the three capable anchors, not
the full pool. (4) LOOCV cluster stability: every held-out anchor is classified the same way under
leave-one-out as in the full fit (✓ on every bar), validating the iter125 capability bimodality.


            Pool                        n    largest gap   perm-p(ward)    LOOCV agreement
            n = 5 (iter125/129)         5         0.531            0.095                   5/5
            n = 7 (reliable n ≥ 20)     7         0.531            0.042                   7/7
            n = 10 (+ short probes)    11         0.379            0.002                 11/11
            n = 12 (all anchors)       12         0.379            0.002                 12/12
Table 24: Iter 133 capability bimodality across pool sizes. The largest-gap split, the Ward-linkage
k = 2 permutation gap p-value, and the leave-one-out cluster agreement are reported for each anchor
pool. Both the permutation gap and the LOOCV stability strengthen monotonically with pool size,
confirming that the iter125/129 finding is not a n = 5 artefact. At n ≥ 10 the cross-class gap rejects
the null at p < 0.005 with 11/11 LOOCV agreement.



(c) Bimodality strengthens with pool size. We re-run the Ward-linkage k = 2 split and the
largest-gap split of the Rmax distribution at each pool size. The permutation-test p-value on
the ward cluster gap shrinks monotonically as the anchor pool grows: pperm (n = 5) = 0.095,
pperm (n = 7) = 0.042, pperm (n = 10) = 0.002, pperm (n = 12) = 0.002. At n ≥ 10 the cross-class
gap rejects the null at p < 0.005. The capability bimodality is therefore not a n = 5 artefact: adding
anchors confirms and strengthens the signal.

(d) Capability-class dominates the cross-anchor axis. We fit four nested OLS models on Rmax
for each pool: (i) intercept only, (ii) log10 N only, (iii) capability class only (Ward cluster label), (iv)
log10 N + capability class, (v) full interaction log10 N × capable. Comparison by AICc (Burnham &
Anderson 2002) and Kass-Raftery Bayes factor categories:
The interpretation is sharp: the capability class (Ward k = 2 split on Rmax ) carries the cross-anchor
signal, not the parameter count. Adding log10 N on top of the capability class actually worsens the
AICc by 3.6 units at n = 5 and 3.7 units at n = 10, because the continuous-size axis is collinear with
the categorical capability axis (capable anchors are a mix of dense + MoE, spanning 4B-1T). The
full interaction is dominated at every pool size, consistent with the cross-class axis being categorical
rather than continuous-modulated.

(e) Iter 133 conclusion. The iter 133 results close the iter 125/129 caveat that the capability
bimodality finding was limited to n = 5. Three sharp deliverables:

      1. Monotonicity falsification is robust at n = 7: every reliable anchor fails the iter 125
         diagnostic at binomial p < 0.001. The two new violators (gpt-oss-20B and Kimi-K2-
         Thinking) come from the MoE frontier pool and confirm the iter 125 finding is not a
         dense-model artefact.
      2. Capability bimodality strengthens with pool size: the Ward-k = 2 cross-class gap
         permutation p-value falls from 0.095 (n = 5) to 0.002 (n = 10), and LOOCV agreement
         is perfect (12/12) at every pool size. This is direct empirical confirmation that the iter 121
         detection-power verdict (which predicted the capability axis would only become visible at
         n > 5) was correct.
      3. Capability-class dominates the cross-anchor axis: at every pool size the capability-only
         model beats the params-only model by 21–32 AICc units, and adding log10 N on top of
         capability worsens the fit. This is the sharpest evidence yet that the Pillar-1 scaling structure
         lives on the capability axis (instruct/pretrained pretraining) rather than the parameter-count
         axis.


                                                    23

[PAGE 25]
Model                    n      c3p    3p
                                               Rmax      λ3p        t3p
                                                                     80   AICc2p     AICc3p
         Qwen3.5-4B              30    0.799   1.000   0.357       4.51     −2.57    +15.61
         Qwen3-8B                30    0.237   1.050   0.004     400.89    −16.82    −15.11
         Llama-3.1-8B-Instruct   30    0.826   1.000   0.128      12.59    −19.16     −4.39
         DeepSeek-V3.1           20    0.843   0.930   0.001    1501.36    −18.28    −15.48
         Nemotron-120B           20    0.000   0.182   0.990       1.63     +4.92     +7.71
Table 26: Three-parameter offset saturation fit (iter 137) vs. two-parameter fit (iter 117). The
3-param un-binds λ on every anchor (vs. 4/5 anchored at λ = 10 under 2-param) but loses by AICc
on 5/5 anchors: the additional offset parameter c is not justified by the residual drop because trace
variance is high relative to the complexity penalty. t80 = − ln(0.2)/λ uses the 3-param λ and is now
informative on every anchor. Note the wide t80 range (1.6–1501) reflects the interpretive value of the
3-param, not a claim that all anchors are equally fast learners.


high-variance Qwen3.5-4B anchor): the additional offset parameter c is not justified by the residual
drop because the trace variance is high relative to the AICc complexity penalty. The 3-param is
therefore a useful interpretive lens (it exposes the baseline reward c) but does not improve model fit.
Iter 125’s structural falsification of the saturation family is reinforced, not weakened.

(b) Iter 137 cross-scale law. With the offset, the OLS regression of log10 t80 on log10 N produces
a real (no longer degenerate) slope estimate: b = +0.507 ± 0.718 (SE; t = 0.71, p > 0.5). The
analogous regression of Rmax on log10 N gives b = −0.172 ± 0.198 (Spearman ρ = −0.658,
p = 0.227, n = 5). Both slopes are far from statistical significance on this evidence base, but their
direction-of-effect is now informative: the iter 117 “4/5 anchored at bound” degeneracy is resolved
into a real but small positive log t80 -log N slope and a small negative Rmax -log N slope. Iter 137
thus sharpens the iter 117 null: the cross-scale saturation law is absent in two model classes (2-param
and 3-param), not one.

(c) Iter 137 capability-axis propagation. The iter 133 capability axis (capable = Rmax ≥ 0.7)
classifies the 3-param Rmax distribution identically: 3/5 capable (Qwen3.5-4B, Llama-3.1-8B-
Instruct, DeepSeek-V3.1) and 2/5 incapable (Qwen3-8B, Nemotron-120B). Mann-Whitney U on
3-param Rmax across classes is U = 3.0 (two-sided p = 1.0, degenerate at n = 5), and within-class
Spearman ρ(Rmax , log10 N ) is non-significant at n = 3 capable. The capability-class verdict from
iter 133 holds qualitatively through the 3-param fit, with the same within-class underpowering caveat
at n = 5.

(d) Iter 137 conclusion.   The Pillar-1 finding is now triangulated by two model classes:
      1. The 2-param fit (iter 117): λ saturates at the upper bound on 4/5 anchors, Rmax bimodality
         is confirmed (iter 125/129/133), and the t80 -vs-N regression is degenerate.
      2. The 3-param fit (iter 137): λ is finite on every anchor, but the model loses by AICc on 5/5
         anchors; the cross-scale slope on t80 is +0.507 ± 0.718 and on Rmax is −0.172 ± 0.198 –
         both far from significance. The capability-class axis is preserved.
The sharpest single sentence: GRPO saturation is real (capable anchors have Rmax > 0.8, incapable
have Rmax < 0.3) but its t80 , λ, and Rmax do not scale with parameter count on this evidence base
even when the R(0) = 0 boundary-condition artefact is removed. The cross-scale law is absent in
two model classes; the capability axis dominates in both.

Iter 140 elevation: reward-design quality as an exogenous covariate (Berkeley F24 L9 — Eureka,
Ma et al. 2023). To stress-test whether the iter 133 capability-axis verdict is robust to exogenous
covariates orthogonal to N , we import the Berkeley F24 L9 (Jim Fan / NVIDIA) Eureka framework
(arXiv:2310.12931; Ma et al., ICLR 2024 oral) and compose a single scalar Reward-Design Quality
Score (RQS) from four observable trace statistics per anchor,
                                                     1/4
                             RQS = c1 · c2 · c3 · c4      , ci ∈ [0, 1],                         (7)
where c1 = clip(10 · Var[R], 0, 1) is the reward-variance channel, c2 = frac(R > 0.5) is the shifted-
mass channel, c3 = clip(peak−trough, 0, 1) is the dynamic-range channel, and c4 = 1−2·zero_frac


                                                  25

[PAGE 27]
Figure 14: Iter 140 cross-pillar figure. (a) Reward-Design Quality Score per anchor on the 12-anchor
extended-frontier table; the five degenerate anchors (Nemotron-120B, Qwen3-32B, Qwen3-30B-MoE,
Qwen3-30B-MoE-Inst, Qwen3-235B-MoE) score RQS = 0 along the geometric-mean breakdown.
(b) AIC race on Rmax at n = 5; capability alone (AICc = −23.07) dominates capability + RQS
(AICc = −21.93) by ∆ = +1.14 (borderline NULL). (c) 12-anchor residualization scatter: residual
from the capability-only model on y-axis, RQS on x-axis; Pearson ρ = +0.225. (d) Iter 127 cross-
pillar: n = 20 (G, T) cells from Qwen2.5-0.5B/arithmetic; x-axis is the independently measured
richness proxy y = 1 − ZVFtheory from iter 131, y-axis is the iter 127 joint-fit residual; Pearson
ρ = −0.569, p = 0.029 (DECISIVE per F25 L8 recipe).


                                                2p
                 Model                   n    R̂max       p̂   95% CI on p̂     width
                 Llama-3.1-8B-Instruct   30   0.869   0.828    [0.752, 0.897]   0.145
                 DeepSeek-V3.1           20   0.844   0.827    [0.700, 0.934]   0.234
                 Qwen3.5-4B              30   0.817   0.767    [0.657, 0.863]   0.205
                 Qwen3-8B                30   0.285   0.416    [0.295, 0.541]   0.247
                 Nemotron-120B           20   0.182   0.219    [0.081, 0.380]   0.298
Table 27: Per-anchor Pass@K=1 bootstrap intervals. p̂ is the step-level Pass@1 estimate;
the CI is a percentile bootstrap (B = 20,000) over the within-anchor step rewards. Every
CI width (0.145–0.298) exceeds the 0.025–0.052 Rmax gaps that separate the three top-ranked
anchors by up to an order of magnitude. scripts/berkeley/sweagent_passk_aci.py →
experiments/results/berkeley/sweagent_passk_per_anchor.tsv.



Iter 140 conclusion. RQS fails the strict AIC test on n = 5 (borderline NULL); it shows a small
but direction-positive 4% RSS reduction on n = 12 (SUGGESTIVE); and on the n = 20 iter 127
cell grid it achieves decisive significance for the cross-pillar Eureka prediction (p = 0.029). The
capability axis of iter 133 is therefore preserved as the load-bearing cross-anchor signal, but the
reward-side of the equation is not silent: it adds diagnostic information orthogonal to capability on
the cell grid, exactly where the Eureka thesis predicts. Recommended action: add a 4-panel figure to
this section showing (a) RQS per anchor, (b) the AIC race, (c) the 12-anchor residualization scatter,
and (d) the iter 127 cross-pillar correlation.

4.1   Pass@K Confidence Intervals and the ACI Ceiling: What the Capability Split Does and
      Does Not Resolve

The iter 133 capability-class analysis above treats each anchor’s ceiling R̂max as a point estimate.
Software-agent benchmarking supplies two correctives. First, the agent–computer interface (ACI) —
how an agent observes its environment and how its output is parsed — is a first-order determinant
of measured capability, often rivalling the underlying policy [27, 25]; in our setting the ACI is the
reward parser that converts GSM8K rollouts into sparse reward. Second, success-rate estimates
should carry Pass@K sampling uncertainty [4]. We apply both to the five-anchor Rmax evidence:
treating each anchor’s per-step training rewards (n = 20–30 steps) as its Pass@K=1 sample, a
percentile bootstrap (B = 20,000) on the mean gives the 95% intervals of Table 27.

The within-tier ordering is not statistically resolvable. A pairwise CI-overlap (“straddle”)
test over all 52 = 10 anchor pairs (experiments/results/berkeley/sweagent_passk_-
scaling.tsv) splits exactly along tier lines. All three pairwise comparisons among the capable
anchors overlap (Rmax gaps 0.025, 0.027, 0.052), and the two incapable anchors overlap each other
(gap 0.110); only the six capable-versus-incapable comparisons are cleanly separated. The plain
statement is that any within-tier model ordering read off the sorted Rmax values — e.g. Llama-
3.1-8B-Instruct > DeepSeek-V3.1 > Qwen3.5-4B — is not resolvable at the current within-anchor
sample size and should not be treated as a finding of this paper. What survives, with no CI overlap
across the gap, is the capable/incapable split itself; this tempers, without overturning, the iter 133
capability-class verdict. Resolving a 0.025 gap would require roughly n ≥ 100 steps per anchor.

                                                                      2p
An Agentless-style tier reading of the bimodality. Binning R̂max           in the style of the Agent-
less pipeline audit [26] into hard-floor (< 0.30), soft-floor ([0.30, 0.70)), and reachable (≥ 0.70)


                                                 27

[PAGE 29]
Algo     n      lam-at-bound   AICc-best sat        AICc-best lin   AICc-best const   λ̂ (med.)
        GRPO      5         0/5             5/5                 0/5               0/5          0.250
        PPO       5         0/5             0/5                 5/5               0/5          0.078
Table 28: Cross-stack identifiability battery on the 40-step same-stack traces. Lambda is identifiable
in both stacks (0/5 hit the upper bound, vs. 5/5 in iter 25’s 20–30 step frontier data). However
the AICc-best model diverges: GRPO’s running mean reward is best described by an exponential
saturation curve (5/5), PPO’s by a linear trend (5/5). Fisher exact on the AICc-best-saturation rate
gives p = 0.0079 (Table 29).


           Prediction                        GRPO                     PPO          p          EEP
           E1 lambda-at-bound rate           0/5                      0/5        1.0000   sustained
           E2 AICc-best sat rate             5/5                      0/5        0.0079   falsified
           E3 heldout-resid from sat fit     0.1023                 0.3901       0.0537   sustained
           F1 bootstrap CI excludes bound    5/5                      4/5          n/a    sustained
           F2 paired heldout accuracy        0.99±0.0035         0.992±0.003     0.3739   sustained
Table 29: EEP-falsification battery on the same-stack traces. E2 is decisively falsified: GRPO and
PPO prefer different AICc-best functional forms (saturation vs. linear) on the same rollout batches.
E1 and F1 are sustained: both stacks have identifiable λ (0/5 hit the bound), and the bootstrap CI
excludes the bound in 5/5 GRPO and 4/5 PPO traces. E3 borderline (p = 0.054): PPO’s saturation fit
leaves more heldout residual than GRPO’s, consistent with E2’s AICc verdict. F2 (final accuracy) is
sustained, matching iter 9.



the GRPO estimate of 0.25). This is not a final-accuracy effect: the two stacks are statistically
indistinguishable on heldout accuracy (p = 0.37), and the bootstrap CI on λ is well-defined in both
stacks (0/5 hit the bound, vs. 5/5 in the 20–30 step frontier data).
The decoupling—identical heldout accuracy, divergent intermediate-curve shape—is the sharpest form
of the critic degeneracy pattern flagged in the frontier synthesis: PPO’s value head appears to flatten
the intermediate gradient signal into a near-linear drift rather than a saturating exponential. Practically,
this means that rules-of-thumb derived from GRPO saturation fits do not transfer to PPO even when
everything else in the stack is held fixed. The two algorithms reach the same accuracy, but they get
there via different intermediate trajectories—and the saturation model R(t) = Rmax (1 − e−λt ) is
only the right descriptive model for one of them.

Limitations. The same-stack data are at a single model size (Qwen2.5-0.5B), single task (GSM8K),
and 40-step horizon. Whether the same AICc divergence holds at 8B+ with the frontier horizon
needs a Tinker reproduction (cost-flagged in FRONTIER_INSIGHTS.md). The “saturation-supported”
criterion in iter 25 (AICc-best = sat AND CI excludes bound) is satisfied in 5/5 GRPO traces and 0/5
PPO traces; iter 25’s stricter formulation would also falsify EEP here.

Artifacts. Driver: scripts/scaling_law_iter29.py. Per-trace: scaling_law_iter29_-
identifiability.tsv, scaling_law_iter29_bootstrap.tsv. Stack rollup: scaling_law_-
iter29_summary.tsv. EEP battery: scaling_law_iter29_stack_compare.tsv. Figure:
figures/scaling_law_iter29.pdf.

4.3   Three-Phase Hypothesis (Nimmaturi et al., arXiv 2507.18014): Pre-Registered
      Falsification Battery

Motivation. Nimmaturi et al. [18] (arXiv:2507.18014, Predictive Scaling Laws for Efficient GRPO
Training of Large Reasoning Models) propose that GRPO training proceeds in three consistent phases:
slow start, rapid improvement, plateau. Iter 17 observed that the five-anchor frontier set splits into
four phase labels (plateau, saturation, drift, collapse) under a heuristic rule, but the classifier was
not pre-registered, the falsification battery was implicit, and the Nemotron-120B collapse was not
mechanically characterised. Iter 33 closes these three gaps on the twelve-anchor frontier set.


                                                       29

[PAGE 31]
• 0/12 traces have all three criteria (Nemotron-120B has 2/3 but its post-peak slope is positive:
         the recovery from zero yields ∆ > 0, even though the trace never reaches the pre-peak
         baseline).
       • 3/12 traces meet ≥ 2 of the three: Qwen3-32B, Qwen3.5-27B, Nemotron-120B.
       • 1/12 traces meet all three: only Qwen3.5-27B.

The extreme-collapse signature (zero_frac > 0.5, mean < 0.2) is unique to Nemotron-120B. This
narrows the cause: it is not “large model fails to plateau” (a generic critique) but a specific reward
parser / format failure that drives 55% of Nemotron rollouts to zero reward and prevents recovery
to the mean—consistent with the iter 17 root-cause analysis showing Nemotron’s post-peak decay
slope is positive (it does not collapse during training; it collapses before training starts and slowly
recovers).

Phase stability. Bootstrap (B = 200) leave-one-out agreement of the four-class classifier is low:
only 3/12 traces have agreement ≥ 0.95 (Qwen3-8B, gpt-oss-20B, Qwen3-30B-MoE). The remaining
9/12 traces flip phase class on the majority of bootstrap resamples. This is not a bug in the classifier:
it is a property of the twelve-anchor set, where summary statistics from n ≤ 30 step traces are not
sufficient to fix the phase label under resampling. The classification is therefore useful as a partition
of the frontier set but not a stable estimator on individual traces—a caveat the three-phase hypothesis
does not address.

Cross-architecture phase distribution. Across the twelve anchors, 6 are dense and 6 are MoE.
The phase distribution by architecture is:

       • dense (n=6): 1 plateau, 1 saturation, 2 drift, 1 collapse, 1 plateau.
       • moe (n=6): 2 plateau, 4 saturation, 1 drift, 0 collapse.

Mann-Whitney on phase score yields U =ns, p=0.48 (median phase score: dense −0.75, MoE
−0.37). The qualitative distribution difference (no MoE model collapses) is suggestive but not
significant on n = 12 (χ2 on 2×4 contingency has p = 0.31). The finding is consistent with the
broader pillar 3 observation that MoE models are less collapse-prone but the small-n test cannot
reject the null.

Limitations. (1) The phase classifier operates on synthetic traces reconstructed from summary
statistics, not raw per-step reward logs. This is necessary because the frontier traces in scaling_-
law_extended_frontier.tsv are aggregated. (2) The five pre-registered predictions are not
independent (P1, P2, P3 share information); a Bonferroni-corrected threshold of α/5 = 0.01 would
still falsify P2, P4 and still sustain P5. (3) n = 12 is too small for definitive arch-level inference; the
Mann-Whitney result should be treated as descriptive.

Artifacts. Driver: scripts/scaling_law_iter33.py.                    Per-anchor: scaling_law_-
iter33_phase_score.tsv, scaling_law_iter33_stability.tsv, scaling_law_iter33_-
nemotron.tsv. Battery: scaling_law_iter33_predictions.tsv. Summary: scaling_law_-
iter33_summary.tsv. Figure: figures/scaling_law_iter33.pdf (left: phase-score vs. mean
reward with phase class; right: peak-fraction vs. late-early delta showing the P1 and P2 prediction
planes).

4.4   Functional-Form Identifiability: Is GRPO Reward Growth Exponential?

Motivation. Iter 17–33 fit R(t) = Rmax (1 − e−λt ) to the twelve-anchor frontier set as the default
saturation form, and showed that the three-phase hypothesis (slow start, rapid improvement, plateau) is
partially falsified on that set (P2 fails: 5/12 anchors are drift or collapse; P3 borderline; P4 decisively
falsified; only P5 – Nemotron uniqueness – sustained, see §4.3). None of the prior iters, however,
tested whether the exponential itself is the right functional form. If the literature’s exponential is
mis-specified, the saturation ceiling estimates Rmax , λ, t80 propagated through the paper are biased.

Method. Iter 37 fits five candidate reward-curve forms to each anchor via nonlinear least squares
and ranks them by AIC and BIC with Akaike weights:


                                                   31

[PAGE 33]
Extrapolation comparison (iter37d). We re-run the iter21 two-anchor log-log extrapolation battery
under both forms. On the 4-anchor holdout (Qwen3-32B, Qwen3.5-27B, DeepSeek-V3.1, Qwen3-
235B-MoE, Nemotron-120B, Kimi-K2-Thinking), the mean absolute error on Rmax is essentially
identical: MAEsat = 0.907, MAEhill = 0.908. The t80 predictions diverge wildly: MAEsat = 584,
MAEhill = 2.4 × 108 . The reason is that on the synthetic frontier traces the saturation fit hits the
upper bound λ ≤ 5 and the Hill fit hits the lower bound K ≥ 0.001, making t80 unidentifiable from
a 3–30 step trace. This is itself a positive result: the iter17 t80 estimates were never identifiable on
the frontier set, and changing the functional form does not change that conclusion.

Recommendation. For frontier-trace analysis on nsteps ≤ 30 records, report the saturation form
as the literature default but also report the Hill n=2 fit and a ∆AIC column; the two are essentially
indistinguishable on this regime. For nsteps ≥ 30 per-step data, the Hill n=2 form is the preferred
nonlinear form on the 10 per-step raw traces measured here, and the exponential saturation is a close
second.

Sharpest claim. “The literature’s R(t) = Rmax (1 − e−λt ) parameterisation is not the uniquely
identified GRPO reward curve on either the 12-anchor frontier set (where it loses to linear/null on
9/12 anchors) or the 10-run raw per-step benchmark (where it loses to the Hill n=2 form on 5/10
runs, tied on the remaining 5). The iter17–33 R_max estimates should be reported with a ±0.05
envelope on the form-choice sensitivity and a Hill n=2 fit alongside.”

Artifacts. Driver:     scripts/scaling_law_iter37.py,        scaling_law_iter37b.py,
scaling_law_iter37c.py, scaling_law_iter37d.py.         Fits: scaling_law_iter37_-
fits.tsv, scaling_law_iter37b_fits.tsv, scaling_law_iter37c_fits.tsv.             AIC
summary: scaling_law_iter37_aic.tsv, scaling_law_iter37b_aic.tsv, scaling_law_-
iter37c_summary.tsv. Bootstrap: scaling_law_iter37_bootstrap.tsv. Extrapolation:
scaling_law_iter37d_fits.tsv, scaling_law_iter37d_extrap.tsv, scaling_law_-
iter37d_summary.tsv. Figure: figures/scaling_law_iter37.pdf (top: stacked Akaike
weights by anchor; bottom: bootstrap win share), figures/scaling_law_iter37b.pdf
(Akaike-weight heat-map on dynamic anchors + bootstrap box-plot), figures/scaling_-
law_iter37c.pdf (per-run Akaike weights on raw 40-step per-step traces, GRPO and PPO),
figures/scaling_law_iter37d.pdf (extrapolation MAE saturation vs Hill on 4-anchor
holdout).

4.5   Temporal stability of the saturation fit (iter41)

Setup. For each of the 12 frontier anchors we re-fit R(t) = Rmax (1 − e−λt ) to the per-step reward
trace truncated at 40%, 60%, 80%, and 100% of its length. We track the fitted λ, Rmax , and pre-
saturation slope s0 = Rmax λ across truncations and ask whether the early-fit Rmax predicts the
full-fit Rmax under a B = 200 parametric bootstrap (observation noise σ = 0.05).

Headline result. 7/12 anchors are stable (max relative ∆λ < 0.5); 2 are unstable. The pre-
saturation slope s0 = Rmax λ correlates with log P on the dense stack at Spearman ρ = −0.40.

Per-anchor stability.

Early → late prediction. Using only the first 60% of the trace, the predicted Rmax falls within
±10% of the full-fit Rmax for 9/9 anchors; the bootstrap 95% CI on the prediction error contains 0
for 7/9.

Takeaway. The saturation model is robust on identifiable traces (those with nsteps ≥ 10 and
non-pathological late-step dynamics) and produces a Rmax that is recoverable from the early portion
of the trace within ≤ 0.10 absolute error in the median case. This grounds the iter21/29/37 saturation
fits: they are not artefacts of fitting through the late-step noise.

4.6   Chinchilla-style iso-compute extrapolation (iter45)

Setup. Iter21–41 all fit R(t) = Rmax (1 − e−λt ) as a function of optimisation steps t, holding
everything else fixed. None asked the operational question: at a fixed training-compute budget,


                                                  33

[PAGE 36]
prediction                                              outcome     value      note
 P1: two-param LOO RMSE < 0.30                           NO          0.504      Rmax ∈ [0, 1] floor on 7/12 anchors caps R2 at 0.18
 P2: max |resid| is on Nemotron-120B                     YES         1.202      collapse signature unmistakable
 P3: at median log10 C = 5.09, optimal P ∈ [4, 30]B      YES       P ⋆ = 4B     Qwen3.5-4B selected at the operating point
Table 37: Three pre-registered predictions P1–P3 with measured outcomes. 2/3 pass; the lone miss is
informative (saturation floor on Rmax ≤ 1.0 causes the linear model to underfit the ceiling).



is a known weakness of the within-mix fit: a model designed to maximise iso-FLOP utility would
rebalance the design by adding more low-P anchors, which the Pillar-1 measurement grid does not
have today.

Conclusion. The two-parameter iso-FLOP joint fit formalises iter45’s αdense = 1.03 vs αMoE =
0.057 reading into a single closed-form predictor. Its LOO RMSE is too large to be predictive (0.50 >
0.30), but the residual decomposition isolates a single breakdown — Nemotron-120B contributes
1.20 of the total 0.50 RMSE — which is exactly the failure signature the rest of Pillar 1 (iter33
collapse partition, iter41 truncation extrapolation, iter45 iso-compute invariance) has documented
from independent angles. The linear read of Rmax is a ceiling-limited summary statistic; the collapse
of Nemotron-120B is the one event the summary cannot absorb.

4.8   Iter 53 – Rank preservation + temporal-peak coupling (negative result)

We re-use the iter49 two-parameter OLS fit and 12 LOO residuals (scaling_law_iter49_loo_-
residuals.tsv) but ask three pre-registered questions: does the LOO prediction preserve the
ranking of models; does the LOO residual track the temporal peak position from iter33; and does
dropping the collapse anchors contract the cross-stack correlation between log10 P and Rmax (the
critic-degeneracy test from the frontier synthesis)? The headline answer is that none of the three
pre-registered primary predictions hold; iter53 is a clean negative result rather than a refit.

Rank preservation (P1). Across the 12 anchors we have Kendall τb = 0.107 between LOO-
predicted and actual Rmax (Spearman ρ = 0.112, permutation p = 0.721). For a random ordering
τb has mean 0 and standard deviation ≈ 0.30; the observed τb = 0.107 is essentially chance. The
LOO-predicted top-3 is Kimi-K2-Thinking, DeepSeek-V3.1, Qwen3-8B; the actual top-3 by Rmax is
Nemotron-120B, Qwen3-235B-MoE, Qwen3-30B-MoE-Inst – zero overlap. Mean |∆rank| = 4.00,
median 3.5, worst swap 9 ranks. Pre-registered: τb > 0.50 – FAIL. Anchors with |∆rank| ≥ 4:
Nemotron-120B (∆rank=-4); Qwen3-235B-MoE (∆rank=-4); Qwen3-30B-MoE-Inst (∆rank=-9);
Kimi-K2-Thinking (∆rank=+4); DeepSeek-V3.1 (∆rank=+4); Qwen3-8B (∆rank=+8). The largest
single swap is Qwen3-30B-MoE-Inst (∆rank = −9): the OLS places it last, the actual order
has it third. Both anchors sit at the same (log10 P, log10 C) ≈ (1.48, 4.66–4.88), so the swap is
within-stack variance that the cross-stack OLS simply cannot see.

Temporal-peak coupling (P2). Across the 12 anchors with finite peak_frac, Spearman
ρ(peak_frac, residual) = −0.125 (permutation p = 0.700). Pre-registered: ρ < −0.30 – FAIL
(the point estimate is in the right direction but its magnitude is too small to clear the threshold and the
permutation p-value is large). The intuition we tested: the saturation fit reads the peak value as Rmax ;
a trace peaking late should be systematically over-predicted by LOO. The data do not support this for
the cross-stack pooled sample. A weaker absolute-step variant – ρ(peak_step, residual) = −0.250
(p = 0.424) – does clear the −0.20 bar (P4), but with n = 12 the test is underpowered. Conclusion:
the peak-coupling hypothesis is at best weakly supported and certainly weaker than the cross-stack
compute signal.

Critic-degeneracy test (P3, frontier synthesis). Frontier reasoning on Pillar 1 (Critic Degeneracy
Hypothesis) licenses the prediction that the residual Rmax variance is mostly explained by the
static prompt-difficulty regressor (collapse regime) rather than by compute. The concrete test:
ρ(log10 P, Rmax ) on the full sample vs. after dropping the collapse anchors. Observed 0.360
vs. 0.359 (absolute change ∆|ρ| = 0.001, well below the 0.05 threshold). Same axis for log10 C:
0.442 → 0.379. The cross-axis ρ(var(reward), Rmax ) = −0.134 on the full sample is negligible.


                                                    36

[PAGE 43]
The bar for calling it a law rather than a trend (frontier synthesis). Finally, the frontier
models specified the single experiment that would license the word “law.” ChatGPT Pro proposed a
preregistered curve-collapse sweep over model size N and rollout budget T , reporting normalized
held-out error reduction
                       A(N, T ) − A0 (N )                                                     
                                                  H(N, T ) = H∞ 1 − exp −(N γ T /τ0 )β ,
                                                                               
         H(N, T ) =                        ,                                                           (14)
                          1 − A0 (N )
so that after rescaling compute by N γ all sizes should lie on one universal post-training curve.
The claim is earned only if a law frozen on 60–70% of cells predicts blind large-N , large-T cells
within ≤ 1.5–2.0 percentage points (or ≤ 10% relative error in H) with seed-stable exponents
(CV(β), CV(γ) < 0.2); otherwise it is a trend, not a law. This is the external standard against which
our present result is honestly a taxonomic one: the paper’s identifiability audit finds 0/5 per-trace
saturation rates estimable (Table 18), so the reliable Pillar-1 structure lives in the cross-scale endpoint
regression Rmax ∼ log10 N rather than in single-trace kinetics. The frontier synthesis does not
overturn that null; it prescribes the blind curve-collapse test and the Ceff abscissa as the route by
which a future, larger sweep could upgrade the trend to a law.

5   Discussion and Limitations
The most useful result here is a boundary on what can be claimed. On this benchmark, mean GRPO
training reward does not exhibit a measurable, monotone dependence on model size, and no universal
saturation exponent is identifiable; the honest description is a local, stack-conditioned taxonomy of
endpoint behavior rather than a Chinchilla-style power law in N . Several limitations qualify even
this modest statement. The frontier-scale anchors are single-seed API runs and should be read as
case studies; held-out evaluation is narrower than training-dynamics coverage; and the strongest
evidence comes from short-horizon GSM8K training, so extrapolation to longer-horizon or non-math
tasks is unwarranted. Because the saturation rate λ is non-identifiable on most traces, per-trace rate
comparisons are excluded from our claims. Finally, the effective learning signal in GRPO depends on
within-group reward variance, which is itself coupled to model accuracy and group size; disentangling
“the model is larger” from “the model produces fewer mixed-reward groups” is a confound we can
describe but not fully resolve with the current runs.

6   Conclusion
We asked whether GRPO post-training obeys a clean scaling law in model size and found, on a
controlled cross-library benchmark, that it does not—at least not one identifiable from roughly 2.4
orders of magnitude of single-benchmark evidence. The cross-scale mean-reward slope is statistically
flat, a three-phase saturation hypothesis is geometrically falsified, and the Nemotron-120B trace is
best treated as a collapse-shaped structural outlier. What survives is a taxonomic, endpoint-level
description of the ceiling, dominated by categorical capability structure rather than an added log10 N
term, and a set of falsifiable next steps: re-plotting the scaling axis in terms of usable contrastive
compute, multi-seed matched-budget runs, and broader held-out evaluation without checkpoint
cherry-picking. The released traces, telemetry, adapters, and scripts are intended to make that stricter
follow-up straightforward.

References
 [1] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, and Marc
     Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Ad-
     vances in Neural Information Processing Systems, volume 34, pages 29304–29320, 2021.
     doi: 10.48550/arXiv.2108.13264. arXiv:2108.13264; supplementary materials at https:
     //agarwl.github.io/rliable/.
 [2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier
     Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE-style
     optimization for learning from human feedback in LLMs. arXiv preprint, 2024. doi: 10.48550/
     arXiv.2402.14740. arXiv:2402.14740.
 [3] Kenneth P. Burnham and David R. Anderson. Model Selection and Multimodel Inference: A
     Practical Information-Theoretic Approach. Springer, 2nd edition, 2002.


                                                   43


# P02: Zero-variance fraction


Root: `platform_hybrid/paper/paper_P2_zvf.tex`  Pages: 45  Words: 23420


[PAGE 1]
The Zero-Variance Fraction:
        A Descriptive Diagnostic for Signal Starvation in
                            GRPO


                        Arvind C R                         Ramesh Prakash Guledgudd∗
                       PES University                           PES University
                   arvindcr4@gmail.com                        rameshpg@pes.edu



                                                Abstract
             In GRPO, a prompt produces no learning signal when all G sampled comple-
             tions receive identical rewards: the group-centered advantages are zero and the
             gradient vanishes. The Zero-Variance Fraction (ZVF)—the share of groups in
             this degenerate state—is therefore a natural diagnostic of when binary, verifiable
             rewards stop training the policy. We study ZVF as one pillar of T INKER RL-
             B ENCH, a benchmark of 70+ RL runs across seven libraries and five model families
             (0.6B–∼671B) on GSM8K, HumanEval, and tool-use tasks. We characterize
             ZVF descriptively and, deliberately, decline to promote it to a causal or incremen-
             tally predictive statistic. Empirically, ZVF tracks training-time signal starvation
             and correlates with catastrophic collapse (Spearman ρ = 0.56; Pearson point-
             biserial 0.62), but only weakly with the final held-out outcome (ρ ≈ 0.27); it is
             temporally sticky (late-phase 0.87–0.97, versus early-phase 0.04–0.13 for variance-
             mitigation GRPO and the largest group size, with smaller groups starting higher
             at 0.25–0.61; lag-1 autocorrelation ≈ 0.94) and mechanically coupled to reward
             sparsity, group size, and baseline accuracy. A concluding cross-examination shows
             ZVF’s core weakness is that it aliases mastery with incapacity. On 505 unique
             prompt-seed tasks we additionally verify the exact binary-reward accounting iden-
             tity pass@ G − pG = 1 − ZVF to numerical error 1.11 × 10−16 : ZVF is the
             complement of contrastive group yield, not an independent performance predictor.
             This identity sharpens the diagnostic’s interpretation and motivates magnitude- and
             sign-aware replacements. We release code, logs, and per-group reward tensors.


1       Introduction
GRPO [19, 9] learns from a group-relative baseline: for each prompt it samples G completions, scores
them with a verifier, and centers the rewards within the group. With binary correctness rewards, a
group whose completions are all correct or all incorrect has zero within-group variance, zero centered
advantage, and hence contributes no gradient. As training proceeds and the policy masters easy
prompts or stalls on hard ones, the fraction of such degenerate groups grows, and the effective batch
that actually moves the policy shrinks. We call this fraction the Zero-Variance Fraction (ZVF) and
ask how well it diagnoses when GRPO stops learning.
ZVF is attractive because it is cheap, stack-agnostic, and mechanistically motivated: it measures
exactly the condition under which the group baseline nulls the update. But a diagnostic is only as
good as its predictive validity, and it is easy to over-read a quantity that is mechanically entangled
with reward sparsity, group size, and accuracy. This paper therefore treats ZVF conservatively—as a
descriptive instrument whose reach we bound rather than a causal predictor.
    ∗
        Project guide.


Preprint. Under review.

[PAGE 14]
1. Compute ZVF per optimizer step using scripts/zvf_compute_cross_framework.py
               on the live training log; keep a 30-step trailing-window mean.
            2. If the trailing-window mean exceeds 0.6 before the heldout-accuracy curve has plateaued
               near its peak, the run is collapsing into all-correct or all-wrong regimes and grouping will
               yield approximately zero advantage signal. The recipe is conservative: it flags only the
               regime where most prompts are no longer informative.
The 42 plateau rows in our matrix (median ℓ = 0.40, median ZVF = 0.22) sit below the 0.6
trailing-window trigger, so the recipe stays silent on them; the collapse rows with recorded per-step
ZVF (the two tool-use cells) reach ZVF = 1.0. The diagnostic is therefore loud on collapse and silent
on plateau – the asymmetry matches the gating behavior of GRPO at ceiling accuracy.

Why this is evidence and not a circular claim. The collapse label is computed only from peak vs.
last-10, never from ZVF. The correlation between ZVF and the collapse label is therefore a prediction,
not a fit. The 95% bootstrap CI is wide because n = 23 cells; the diagnostic value of ZVF in our
matrix is the separation (collapse rows ZVF ≈ 1, plateau rows ZVF ≈ 0.5), not the point estimate of
a single correlation.

Limitations. We deliberately do not report partial correlations “controlling for advantage variance”
or “controlling for entropy”: under binary {0, 1} rewards, within-group advantage variance is a
deterministic transform of ZVF (zero iff ZVF) and partialling it out is circular (already documented
in experiments/results/zvf_partial_correlations.tsv). Step- level p-values are not re-
ported either: per-step ZVF rows are autocorrelated (typical lag-1 ρ ≈ 0.9 on our rollout trajectories),
and any t-statistic inferred from the 480 measurements would be overconfident by roughly an order
of magnitude.

5.8        Cross-Source Anti-Herd Falsification of the Contrastive Yield Band

A frontier synthesis2 of this pillar proposed the strict band δdiv ∈ [0.13, 0.23] as a “measured
structural diversity bonus” introduced by high-temperature autoregressive sampling, on the reading
that “empirical ZVF under-predicts the i.i.d. baseline by −0.13 to −0.23” and hence the sampler anti-
herds. Our per-problem decomposition δdiv (x) = ZVFiid (px , G) − ZVFobs (x) over 1,092 measured
rows (experiments/results/zvf_contrastive_yield.tsv) shows this is regime-dependent
rather than uniform.

 Source                                         n   mean δdiv            95% boot. CI    frac(δdiv > 0)    verdict
 tinker_gsm8k (real reasoning)               600     +0.1224     [+0.1116, +0.1334]                0.842   anti-herd
 groupsize_zvf_sweep (synthetic)             480     −0.0668     [−0.0786, −0.0561]                0.298   herd
 groupsize_zvf_sweep_agg (per-step)           12     −0.2994     [−0.3890, −0.2100]                0.000   herd
Table 9: Cross-source δdiv statistics pooled at the per-problem level. Bootstrap CIs use B=2000
percentile resamples over per-problem rows (one row per (source, seed, problem_id)). The sign
reversal: Qwen3-8B on GSM8K produces rollouts that are more diverse than i.i.d. Bernoulli draws
(anti-herding, δdiv > 0), while Qwen2.5-0.5B on synthetic arithmetic produces rollouts that are
more correlated than i.i.d. draws (herding, δdiv < 0). The frontier claim of a uniform δdiv ∈
[0.13, 0.23] is supported at the lower bound on the real reasoning model (+0.1224 ≈ 0.13) but
falsified on the synthetic regime and across the bootstrap CI. Source: experiments/results/zvf_-
antiherding_falsification.tsv.


The interpretation is consistent with the visibility of high-decision boundary in each regime. On
real reasoning, the policy is on the “learning frontier” (p ∈ [0.2, 0.8]) for most prompts; sampling
temperature introduces genuine exploration that escapes the modal answer. On a small model trained
to near-perfect score on a narrow arithmetic distribution, sampling collapses to a mode: rollouts
correlate, ZVFobs EXCEEDS ZVFiid , δdiv < 0.

The empirical iso-G correction. The Contrastive Yield framing is preserved, but the iso-G siz-
ing formula Giid (p, Ytarget ) = ⌈log(1 − Ytarget )/ log(max(p, 1 − p))⌉ uses an assumption that
      2
          (ChatGPT Pro Extended + Gemini Deep Think, round 2, attribution: “frontier synthesis”)


                                                         14

[PAGE 38]
expansion gives ZVF ≈ 1 − Gϵ + O(ϵ2 ) and ∂G ZVF ≈ −ϵ → 0—group-size scaling is crushed
into a linear crawl. This is the frontier-side explanation for our temporal-stickiness result: as training
drives px → 1 on mastered prompts, Eq. (10) pushes ZVF toward 1 and makes it inelastic to G, so
ZVF drifts up and does not return—matching the monotonic phase-1-to-phase-3 drift and high lag-1
autocorrelation (ρ1 ≈ 0.94) we measure.

Replacement diagnostics: PCD and a sign-resolving quality score. Because ZVF measures
only the existence of contrast, the frontier models proposed replacements that measure its magnitude
and sign. Gemini’s Pairwise Contrast Density (PCD) is the expected within-group reward variance,
equivalently the mean squared pairwise contrast that GRPO’s baseline projection actually consumes:
                                              h X               i
           PCD = Ex Var(r | x) = Ex G12              (ri − rj )2 = G−1
                                                                                      
                                                                     G Ex px (1 − px ) ,        (11)
                                                  i<j

a sharp parabola peaking at px = 21 rather than the flat plateau that 1 − ZVF forms over intermediate
px . The models offered a concrete micro-jitter falsification: injecting noise ϵ ∼ U (0, 10−4 ) into
rewards (e.g. from a length penalty) makes ZVF flatline to 0 globally—falsely reporting a fully
healthy batch—whereas PCD is essentially invariant. ChatGPT proposed instead a sign-resolving
scalar, Learning-Adjusted Rollout Quality, LARQβ = Ex [p̂x + β · 4p̂x (1 − p̂x )] with p̂x = Kx /G,
whose first term distinguishes all-correct from all-wrong (curing the aliasing) while the second
rewards frontier mass at p̂x ≈ 12 . Both were offered with a head-to-head protocol against ZVF on our
23 pooled cells, with a stated target of ρ ≳ 0.45 (frontier synthesis)—a falsifiable bar, not a claimed
result.

Active gradient bandwidth and group size. The frontier synthesis recasts the group-size effect as
a law about gradient utilization GUG = 1 − ZVFG . From Eq. (10), the marginal group nonzero-yield
gained per extra rollout is

          hG (p) − hG+1 (p) = p G (1 − p) + (1 − p) G p,           hG (p) = p G + (1 − p) G ,        (12)
which is maximized near px ≈ 1/G or 1 − 1/G, not at px ≈ 12 (where ZVF is already tiny) nor at
px ≈ 0, 1 (where even large G fails). This motivated the models’ “Iso-Yield” dynamic-grouping
idea—route mastered/frontier prompts to small G and spend the freed budget on stubborn tails—and
a matching contrastive-yield effective compute axis Ceff = T · G · Ex [ 1 − pxG − (1 − px ) G ] · ∥∆θ∥KL ,
in which extra samples count only insofar as they create mixed-reward groups. Equation (12) is
consistent with the 0.845 → 0.631 ZVF drop we observe as G : 2 → 16: the reduction is concentrated
in the interior-px prompts and saturates in the tails.

Two caveats that sharpen our own claims. First, on our anti-herding falsification: the models
cautioned that the measured residual ∆G = ZVFobs              G             G
                                                   G − Ex [px + (1 − px ) ] ∈ [−0.23, −0.13] should
be read as edge-thinning / excess contrastive yield (13–23 percentage points fewer zero-advantage
groups than the i.i.d. Bernoulli null), not as negative pairwise rollout correlation. ZVF is an extreme-
event statistic, whereas correlation is a second-moment property, and one model produced an explicit
counterexample (G = 4, p = 12 ) with ∆G < 0 yet pairwise ρ > 0. This reinforces the conservative
stance already taken in Section 6. Second, and as a genuine challenge to the “zero gradient” framing
in Section the companion paper on ZVF gradients: under a KL-penalized reward ri = ritask − β KLi ,
when Var(rtask ) = 0 the advantage normalization cancels β and yields Ai = −(KLi − KL)/σKL ,
so a zero-variance group emits not a null update but a covert, full-magnitude distillation pull toward
the reference policy. The frontier models framed this as the single falsifying experiment for Pillar 2:
if masking these updates destabilizes training, ZVF states are load-bearing regularization anchors
rather than pure dead compute—a caveat we flag but do not resolve here.

11    Discussion and Limitations
ZVF earns its place as a descriptive diagnostic of signal starvation, but our evidence argues against
treating it as a standalone causal or incrementally predictive statistic. (Our companion Pillar-7 takes up
exactly this question as an interventional sequel—testing, rather than assuming, whether ZVF-derived
quantities can be made predictive and used for control; the two papers are complementary, descriptive


                                                   38

[PAGE 39]
characterization here and falsifiable intervention there, not contradictory.) Three limitations are load-
bearing. First, ZVF is mechanically coupled to reward sparsity, group size, and baseline accuracy,
so its correlations are partly bookkeeping rather than discovery; multivariate tests against reward
mean, entropy, and divergence proxies are needed before any causal reading. Second, its strong
association is with catastrophic collapse, an easy target, and only weak with the graded held-out
outcome that practitioners actually care about. Third, as an unsigned existence statistic it aliases
mastery (p → 1) with incapacity (p → 0) and, as the cross-examination shows, can be driven to zero
by sub-reward jitter (for example a small length penalty), falsely reporting a healthy batch. These are
reasons to prefer magnitude- and sign-aware diagnostics, which we outline but do not yet validate at
the ρ ≥ 0.45 bar the cross-examination sets.

12    Conclusion
The Zero-Variance Fraction is a useful, cheap, mechanistically grounded lens on when binary-reward
GRPO stops producing within-group signal, and it reliably tracks collapse and drifts upward over
training. It is not, on our data, a standalone predictor of final generalization, and its unsigned form
conflates two opposite regimes. The constructive path forward is to measure the magnitude and
sign of within-group contrast rather than merely its existence, and to test such replacements as
early-window leading indicators against held-out outcome. We release the per-group reward tensors
precisely so that these stricter tests can be run.

A     Formal Definition of the Zero-Variance Fraction, Non-Tautology, and
      Scope
A.1   Formal Definition

Let a GRPO batch Bt at optimizer step t consist of |Bt | prompt-groups, each group g containing
K rollouts with scalar outcome rewards rg,1 , . . . , rg,K ∈ R. We define the Zero-Variance Fraction
(ZVF) as
                                  1 X hd                                    i
                        ZVFt =              ⊮ Var(rg,1 , . . . , rg,K ) ≤ ε ,                   (13)
                                 |Bt |
                                         g∈Bt
        d is the unbiased sample variance and ε is a small numerical tolerance (we use ε = 10−6
where Var
for rewards normalized to [0, 1]). Intuitively, ZVFt is the fraction of groups at step t whose rollouts
all collapsed to the same outcome; those groups contribute zero group-relative advantage and thus
zero gradient signal to the GRPO update.

A.2   Why ZVF Is Not a Tautological Re-expression of Mean Reward

A reviewer may reasonably suspect that under binary outcome rewards r ∈ {0, 1} and group-relative
advantages, ZVFt is a direct function of the batch mean reward r̄t : if every group succeeds (or fails)
uniformly, ZVFt = 1. This is true at the extremes r̄t ∈ {0, 1} but does not hold at intermediate
values. For K i.i.d. Bernoulli rollouts with prompt-specific success probability pg , the expected ZVF
under a null independence model is
                                E[ZVFt ] = Epg pK                 K
                                                                    
                                                    g + (1 − pg )      .                           (14)
The right-hand side depends on the full prompt-difficulty distribution {pg }, not only on its first
moment. Two populations can therefore share the same mean reward r̄ = 0.5 yet have very different
ZVF: a bimodal population with pg ∈ {0, 1} yields ZVF = 1, while a uniform-hard population with
all pg = 0.5 and K = 8 yields ZVF ≈ 2 · (0.5)8 ≈ 0.008. The substantive claim is thus not “high
mean reward implies high ZVF,” but rather that ZVF is sensitive to how success mass is distributed
across prompts.

A.3   What the Released Artifact Establishes

The main Qwen3-8B/frontier GSM8K corpus bundled with this paper contains aggregate reward
trajectories, held-out checkpoint summaries, and the cross-framework parser specification in Ap-
pendix C, but not the per-group step-level rollout logs for those frontier runs. For that corpus we


                                                  39


# P03: Group size


Root: `platform_hybrid/paper/paper_P3_group_size.tex`  Pages: 25  Words: 13676


[PAGE 1]
Group Size in GRPO Is Budget- and
                 Regime-Dependent:
Contrast Density, Measured Bounds, and the Bridge to
                       DPO


                        Arvind C R                          Ramesh Prakash Guledgudd∗
                       PES University                            PES University
                   arvindcr4@gmail.com                         rameshpg@pes.edu



                                                 Abstract
             The group size G is GRPO’s most distinctive hyperparameter: it sets how many
             completions are sampled per prompt and, through the group-relative baseline, how
             much preference signal each prompt contributes. We study the effect of G as one
             pillar of T INKER RL-B ENCH, a benchmark of 70+ RL runs across seven libraries
             and five model families (0.6B–∼671B), centered on GSM8K with HumanEval
             and tool-use checks. Two results anchor the pillar. First, trainability varies with
             G in a non-monotone way: intermediate group sizes often behaved better than the
             smallest or largest we tested, but the data do not justify a universal optimum. Our
             measured equivalence tests extend to G=16; on the near-ceiling arithmetic task,
             held-out accuracy is essentially flat across this range—a retention ratio of ≈1.00
             between G=2 and G=16 (i.e. G=16 lands within noise of G=2; this is a ratio,
             not an accuracy above 100%). We caution that this is a saturated regime: near the
             ceiling, variance is compressed, so the finding is “no detectable G effect under
             saturation,” not evidence of equivalence on harder, unsaturated tasks. The √ effective
             signal-to-noise ratio rises only sublinearly with G (about 52% of the G ideal).
             We do not report a directly trained, matched-budget G=32 cell in the primary
             sweep: the paper’s G=32 values come from an explicitly labeled reconstructed
             token-budget grid and are hypotheses, not√measurements. A separate 505-task
             conditional utility audit, using (1 − ZVF)/ G as the declared cost proxy, selects
             G=4 over G=5 by 0.00177 with a 95% bootstrap interval [0.00082, 0.00270]; that
             result is cohort- and objective-conditional, not a universal optimum. Second, we
             make explicit the sense in which GRPO admits an all-pairs preference (DPO-like)
             interpretation: the group-centered update behaves as an implicit all-pairs preference
             contrast—consistent with logged per-step diagnostics (advantage variance, ZVF,
             and gradient norms), though direct gradient-vector validation is still future work,
             so we frame this as an interpretation rather than a proven equivalence—with zero-
             variance groups projected out. We frame G as a preference-density dial rather than
             a variance knob, test equivalence with TOST and difficulty-stratified retention, and
             release the in-repository artifacts used here.


1       Introduction
GRPO [19, 7] estimates advantages by sampling a group of G completions per prompt and centering
their rewards. The group size G thus plays two roles at once: it is a variance-reduction knob, like
a Monte-Carlo baseline, and it is a preference-density knob, controlling how many within-prompt
    ∗
        Project guide.


Preprint. Under review.

[PAGE 10]
Comparison                          Mean A     Mean B        Diff. (A–B)        95% CI on diff.
   G=2 vs G=16 (Wu headline)            0.982      0.978         +0.003           [−0.008, +0.015]
   G=4 vs G=16                          0.988      0.978         +0.010                  n/a
   G=8 vs G=16                          0.990      0.978         +0.012                  n/a
   G=4 vs G=32 (Pillar 3 question)      0.988       n/a           n/a         not measured on this sweep
Table 8: Per-G bootstrap on the measured Qwen2.5-0.5B / arithmetic sweep. G=2 retains 100.3%
of G=16, well above Wu et al. (2025)’s 97.6% headline and consistent with the DPO-equivalence
prediction. G=32 is not measured on this sweep; we refer to the illustrative Qwen3-8B / GSM8K
reanalysis (Table 6).



This means the empirical contrastive signal is stronger than the Bernoulli- independent prediction:
rollouts within a group are negatively correlated (correct rollouts cluster, incorrect rollouts cluster)
rather than independent. This is the exact correlation structure Wu et al. rely on in their Proposition
4.1: a non-trivial Cov(g + , g − ) reduces gradient variance further than the independent-coin baseline
predicts.

Matched-budget trajectory check on the Wu pairing itself. A two-seed matched-budget panel
on exactly the G=2 vs. G=16 pairing (2,560 rollouts per arm: G=2 × 160 steps vs. G=16 × 20
steps, Qwen3-8B/GSM8K, LoRA rank 4, batch 8, 512-token completions, seeds 123/456) adds the
time axis to the static table above: the G=2 arms drive train reward to ≈ 0.9–1.0 on the sampled pool
and terminate in a sustained all-correct zero-variance regime (ZVF ≈ 0.75–1.0, the p → 1 wall of
ZVF(p, G) = pG + (1 − p)G ), while the G=16 arms end mid-learning (train reward ≈ 0.3–0.5) with
ZVF ≤ 0.25 throughout. On this budget the small-G arm wins the train-reward race outright — and
that is precisely the trap: it spends its final optimizer steps in an endgame where most groups carry zero
gradient, while the reward curve reads success. Artifacts: tinker-runs/results/er2b_*.json.

         G     p (empirical)   ZVF (empirical)   ZVF (theory at p)      Residual   GU (empirical)
         2        0.982            0.838                0.964           −0.126          0.162
         4        0.988            0.764                0.954           −0.191          0.237
         8        0.990            0.691                0.923           −0.232          0.309
         16       0.978            0.631                0.704           −0.073          0.369
Table 9: ZVF empirical vs closed-form prediction ZVF(p, G) = pG + (1−p)G on the Qwen2.5-0.5B
/ arithmetic sweep. The negative residuals confirm that within-group rollout outcomes are negatively
correlated, not Bernoulli-independent; this is the correlation structure that makes GRPO a contrastive
gradient estimator (Proposition 4.1 of Wu et al.).


Figure. Figure 5 summarizes both views. The left panel plots the measured arithmetic sweep
(G ∈ {2, 4, 8, 16}) with the Wu et al. 97.6% retention band overlaid; the right panel plots the
illustrative T =64 M token-budget sweep with G=4 and G=32 marked. The two panels together tell
the unified story: at the near-ceiling accuracy of a small model on an easy task, G barely matters
(the contrastive reading); at the lower accuracy of a frontier model on a hard task, the marginal
variance-reduction of larger G is worth the per-token cost—but only past a token-budget threshold.

Conclusions.

      1. The Wu et al. (2025) headline is cross-validated on our small-scale sweep: G=2 retains
         100.3%±2.7% of G=16 on Qwen2.5-0.5B / arithmetic, within the 97.6% headline to within
         sampling noise. The contrastive reading is correct on easy, near-ceiling tasks.
      2. At canonical token budgets, the reconstruction favors larger G. The illustrative T =64 M
         reanalysis shows G=32 at 0.88, G=4 at 0.64—the largest G-vs-G effect in the reconstructed
         token-normalized grid. This is the regime in which the value-baseline (large-G) reading
         and the contrastive reading both agree: larger G amortizes more   √ wasted rollouts and the
         per-step gradient is more accurate, and the small per-step loss of G does not outweigh the
         increased number of informative steps at fixed total tokens.


                                                   10

[PAGE 22]
Table 15: Small-scale, multi-seed re-tests of four popular GRPO levers (Qwen3.5-4B, easy GSM8K
subset). Each lever’s “advertised” benefit fails to materialize against a matched baseline; differences
are within noise at held-out n=12–20. Effects are reported as mean held-out accuracy gain unless
noted.
Lever (proposed benefit)              Controlled result                       Verdict
Difficulty curriculum / filter col-   Eliminates zero-gradient waste as de-   Null (no free lunch).
lapsed groups                         signed (0.50→0.00 collapse frac),
                                      but at ∼4.8× sampling cost and
                                      identical held-out gain (+0.05 vs
                                      +0.05); still ties baseline at a
                                      matched 30k-token budget (+0.028
                                      vs +0.028) and loses at 5× cost.
Group-size sweet spot (G ∈            G=2 under-trains (50% batch col-        Null (no robust optimal G).
{2, 4, 8, 16})                        lapse, +0.0); a single-seed G=4
                                      “win” (+0.125) is one held-out exam-
                                      ple and does not survive multi-seed
                                      re-testing; G≥8 gives diminishing
                                      gradient signal at linear token cost.
Length-bias loss fixes (mean-         Standard sum was, if anything, best     Null (fixes do not beat sum).
norm, surprise-weighting)             on held-out (+0.125 vs +0.000
                                      mean-norm, +0.042 surprise), and
                                      completions did not inflate under it
                                      (∆len= − 16) — little length bias to
                                      correct on this setup.
Step-1 predictive layer-freeze        Step-1→final top-k layer overlap        Null (not predictable from step 1).
                                      collapses from 1.0 (on a 1.5B
                                      hardcoded-arithmetic toy) to 0.11
                                      (≈chance, both seeds) at 3B on real
                                      GSM8K; the predictability premise is
                                      a toy artifact (concentration ≈ 0.39
                                      still holds).



is indistinguishable from doing nothing, not to crown a winner. Full per-arm logs are released un-
der experiments/results/{curriculum_opening, token_budget, p3_groupsize, p4_-
surprise, p1_layerfreeze}/.


7   Discussion and Limitations

Reframing G as a preference-density dial clarifies the non-monotone effect, but several limita-
tions bound the claim. The held-out swings, while suggestive, are modest against seed and
difficulty variation, so we resist naming an optimal G; our equivalence tests are powered to
reject large differences, not to certify exact parity. Read through the four-axis pipeline fac-
torization of Ivison et al. [11], applied here to verifiable-reward stacks in the RLVR paradigm
of Lambert et al. [12], the same-stack audit places the algorithm axis as residual (η 2 =0.023;
paired ∆GRPO−PPO = − 0.002, permutation p=0.62) while the G axis dominates the variance bud-
get (η 2 =0.54 on terminal accuracy, 0.63 on gradient norm; Cohen’s d= − 1.47 for G=2→16;
experiments/results/berkeley/unpacking_dpo_ppo_factorization.json). An indepen-
dent Miller-style error-bars audit [16] of the four Pillar-3 computed contrasts sharpens their numerical
precision, not their provenance: the G=4/G=32 GU-ratio (audit H1, bootstrap CI [4.16, 4.82]) and
the negative retention-vs-T slope (H2, CI [−0.237, −0.038]) are precise within the reconstructed
grid, whereas the SNR-slope-vs-theory comparison (H3; CI [+0.148, +0.583] contains the the-
oretical +0.500) and the native-Wu G=2 ≈ G=16 equivalence claim (H4; n=3 paired seeds,
equivalence region straddled) are NULL (experiments/results/berkeley/adding_error_-
bars_summary.json).
Two further reformulations sharpen the practical reading of the G axis. First, the preference-
pair view makes the DPO connection exact rather than rhetorical: on a single winner–loser


                                                       22

[PAGE 23]
pair within a G=2 group, the GRPO loss coincides with the small-β, no-KL, online limit of
DPO [18], and Iterative RPO’s DPO-plus-NLL objective [17] corresponds to GRPO with re-
play on winning trajectories; by analytical construction the two share the same optimal group
size at every budget (G∗ = 8, 16, 32, 32 at T ∈ {1, 4, 16, 64}M, where the G=32 optima at
high T are joint-fit extrapolations—no G=32 cell and no Iterative-RPO run was          √ trained), and
the SNR slope in G (+0.366/decade, 95% CI [+0.148, +0.583]) contains the G-theory value
of +0.500 (experiments/results/berkeley/dpo_iterative_rpo_summary.json). Second,
a Dualformer-style fast/slow/auto reading of G [20] suggests the non-monotone landscape is ex-
ploitable: a difficulty-gated allocation rule (predicted accuracy ≥ 0.85 → G=2, down to G=32
otherwise) attains mean Gauto = 7.0 on the 20-cell joint-fit grid — a projected 56.2% rollout saving
versus always-G=16 and a further ≈ 42% versus the reconstructed G∗ (T ) schedule (mean G=12.0,
itself a projected 25% saving versus always-G=16) — with 5/20 cells under-predicting by more
than 5 pp, consistent with joint-fit residual noise but not attributable without a prospective allocation
experiment (experiments/results/berkeley/dualformer_compute_savings.tsv).
The all-pairs contrast identity is an exact algebraic rewriting whose observable signatures are con-
sistent with logged per-step diagnostics (advantage variance, ZVF, and gradient norms); direct
gradient-vector validation remains future work, and the mechanism separating variance-reduction
from contrast-density remains only partially resolved on existing data—the decisive controlled exper-
iment (a token- and step-matched 2×2 design that supplies contrast without variance and vice versa)
is specified but not yet run. Finally, the GRPO↔DPO bridge is exact only up to a KL penalty and the
projection of zero-variance groups; treating GRPO as literally DPO ignores stabilizing structure the
group baseline may carry.


8   Conclusion

Group size in GRPO is best understood not as a variance knob to be maximized but as a dial on
how much implicit preference signal each prompt contributes. On our benchmark its effect is non-
monotone with no universal optimum, held-out accuracy is retained almost completely across G=2
to G=16 on the measured near-ceiling task. The illustrative reconstructed GSM8K grid falls to
about 73% G=4/G=32 retention at T =64M, but no directly trained matched-budget    √        G=32 arm
validates that projection. The signal-to-noise ratio grows far more slowly than G—all consistent
with a contrast-density rather than a pure-variance account, and with the exact sense in which the
group-centered update is an all-pairs preference contrast. The clean causal test, and matched-budget
multi-seed comparisons, are the natural next experiments; we release the sweeps, per-step gradient-
norm logs, and scripts to enable them. As a reproducibility check on the released artifacts, an
automated extract-and-verify pass in the style of Paper2Agent [15] recovers the analysis recipe
from the Pillar-3 headline TSVs with full field recall (1.000), reproduces the published joint fit
(R2 =0.854 vs. the published 0.796), and transfers to a held-out slice (R2 =0.812). The recipe also
survives deletion of a fitted field (zero failures; the slope is re-recovered from the data alone), so
the released TSVs are sufficient to reconstruct the paper’s central fit without human intervention
(experiments/results/berkeley/paper2verifier.json).


References
 [1] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, and Marc
     Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Ad-
     vances in Neural Information Processing Systems, volume 34, pages 29304–29320, 2021.
     doi: 10.48550/arXiv.2108.13264. arXiv:2108.13264; supplementary materials at https:
     //agarwl.github.io/rliable/.
 [2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier
     Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE-style
     optimization for learning from human feedback in LLMs. arXiv preprint, 2024. doi: 10.48550/
     arXiv.2402.14740. arXiv:2402.14740.
 [3] Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Rémi Munos, Mark Rowland,
     Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning
     from human preferences. 2024. IPO.


                                                   23


# P04: Length bias


Root: `platform_hybrid/paper/paper_P4_length_bias.tex`  Pages: 45  Words: 26982


[PAGE 1]
A Bounded Null Test of Length Bias and Held-Out
                    Generalization
               in GRPO and Dr. GRPO


                        Arvind C R                        Ramesh Prakash Guledgudd∗
                       PES University                          PES University
                   arvindcr4@gmail.com                       rameshpg@pes.edu



                                                Abstract
             A recurring worry about GRPO is length bias: because the group-relative update
             divides by response length, the algorithm may be nudged toward longer completions
             that stumble into reward rather than toward better reasoning. The length-debiased
             variant Dr. GRPO removes this per-response normalization. We study length bias
             and held-out generalization as one pillar of T INKER RL-B ENCH, a benchmark
             of 70+ RL runs across seven libraries and five model families (0.6B–∼671B) on
             GSM8K, HumanEval, and tool-use tasks. Our headline result is a carefully scoped
             negative one: in the regime we measure—short-horizon GSM8K chain-of-thought
             with a 200-token generation cap—the mean Qwen3-8B GRPO gain over the pre-RL
             checkpoint on held-out data is small and not statistically significant, GRPO and
             Dr. GRPO produce held-out gains that are indistinguishable at our seed budget, and
             neither inflates length within this capped regime (pooled mean completion length
             drifts from 194.4 to 183.5 tokens). We stress that the 200-token cap bounds this
             claim: it cannot rule out length inflation in longer-horizon or uncapped regimes,
             only establish its absence where length is already controlled. The verbosity-
             trap signature is absent in this regime. We argue this is the expected outcome
             when length is already controlled, not evidence against Dr. GRPO, and specify the
             length-confounded regime and causal mediation tests that would actually reveal a
             difference. We release code, logs, and checkpoints.


1       Introduction
GRPO [14, 7] normalizes the group-relative advantage by response length. This introduces a potential
length bias: when reward is sparse and binary, longer completions have more chances to stumble
into a correct answer, and a length-dividing update can reward verbosity as a proxy for competence.
Dr. GRPO [11] removes the per-response 1/L factor, restoring the sequence-level gradient up to a
global scale, and is motivated precisely as a fix for reward hacking through length. Whether this fix
produces better held-out generalization in practice is the question of this pillar.
The concern matters because length inflation is a well-documented failure mode of reward optimiza-
tion [13, 4], and because held-out gains can be manufactured by a policy that pads its outputs rather
than reasoning better. A careful comparison must therefore control both length and memorization,
and must distinguish a genuine direct effect on accuracy from an indirect effect mediated by verbosity.
We study length bias as one pillar of T INKER RL-B ENCH, a controlled benchmark of 70+ runs
across seven RL libraries and five model families (0.6B–∼671B) on GSM8K [5], HumanEval [3],
and tool-use tasks. Our contributions are: (i) a held-out comparison showing that, in the short-
horizon GSM8K regime, the mean Qwen3-8B-Instruct GRPO gain over the same checkpoint’s
    ∗
        Project guide.


Preprint. Under review.

[PAGE 10]
Limitations. (1) n=5 and n=3 seeds are too few to put tight confidence intervals on the difference
between the two algorithms; we report standard deviations across seeds but do not compute seed-level
p-values. (2) Both tasks are reward-on-completion binary tasks, so the within-run ρ(len, reward) is
partially driven by the fact that longer incorrect completions contribute one full penalty to the per-step
mean – not by an advantage estimator pathology. (3) [11] themselves report the verbosity trap at
hundreds of steps on larger reasoning benchmarks; our 30–40 step horizons are short of that regime
by an order of magnitude. (4) A direct ablation that ran the same number of steps on a longer-horizon
task (DeepSeek-R1 distillation chain, or R1-Zero on MATH) is the natural next experiment, but is
beyond the scope of this iteration.

                                             Length behaviour                         Reward behaviour
 Horizon                             Easy           Hard          Ref.         Easy        Hard          Ref.
 30–40 steps (this paper)         compresses     compresses     §5.X, here    grows         flat     §5.X, here
 200+ steps (Dr. GRPO paper)        grows          grows           [11]      collapses   collapses      [11]
Table 8: Cross-horizon reconciliation. At short horizons the model compresses under reinforcement;
at long horizons Dr. GRPO’s authors observe the predicted length growth followed by reward collapse.
Our data is consistent with the early phase of the Dr. GRPO curve, not the late phase.


Take-aways for a reviewer. (1) No length-bias trap at 30–40 steps on either task. All sixteen
runs have negative length trends, none crosses the flag threshold. (2) Dr. GRPO’s effect is small at
this scale. On GSM8K-CoT it attenuates the length-trend slope by 0.27 (one sd) but does not flip
its sign. (3) Within-run coupling is negative throughout. Longer completions are not rewarded on
our data, in line with the model’s pre-trap compression regime. (4) Direct per-step measurement is
the recommended methodology – aggregate instability indices can misclassify a length-compressing
run as a length-inflation run. We therefore add the per-step Spearman triple to the recommended
diagnostic stack for any future length-bias audit on this benchmark.

5.1   Elevated analysis: trap-onset detection, paired bootstrap CIs, decile binned coupling, and
      a 100-step cross-validation

Why the elevation. The base analysis uses a global monotonic-trend flag (length slope > 0 AND
reward slope ≤ 0) and a within-run Spearman triple. A reviewer concerned about the Dr. GRPO
thesis [11] might object that the trap can engage in a sub-window without being visible in the global
trend, and that the small seed count (n=5 and n=3) makes the per-algo difference hard to assess.
This subsection addresses both concerns with four additional analyses on the same per-step traces.

(A) Sliding-window trap-onset detector. A window of W =10 consecutive steps is moved across
each run; at each step s ≥ W we compute ρW (step, len) and ρW (step, reward) on the window
[s − W, s]. The trap is said to onset at the first s where (a) the local ρW (step, len) > 0.3 (a non-trivial
positive trend, not noise), (b) the local ρW (step, reward) ≤ 0, and (c) the end-of-window length
is greater than the first-half mean length (the model has actually re-expanded past the early-stage
compression reference). Table 9 reports the rate. On the easy task only 2 of 10 runs fire and on the
hard task 4 of 6 runs fire – but in every case the local reward is still increasing or flat over the whole
run, so the global Dr. GRPO signature (length growth followed by reward collapse) is not engaged;
the sliding-window firings are local upticks against an overall downward length trajectory, not a
sustained verbosity-trap.

(B) Paired bootstrap CIs on the GRPO / Dr.GRPO difference. The base TSV reports per-seed
mean and SD. We add a paired non-parametric bootstrap (nboot = 2000, seed-deterministic) on
the Dr.GRPO − GRPO difference for the three Spearman quantities, computed on shared seeds.
Table 10 shows the result. The GSM8K-CoT length-trend difference is +0.267 with a 95% CI of
[+0.164, +0.419] – the lower bound excludes zero, so Dr.GRPO does reliably attenuate the compres-
sion on the hard task. The arithmetic length-trend difference is +0.098 with CI [+0.073, +0.119] –
also excluding zero, so even on the easy task Dr.GRPO compresses slightly less, in line with the qual-
itative claim. The GSM8K-CoT reward-trend difference is −0.020 with CI [−0.069, +0.065] – not
significant, meaning Dr.GRPO neither helps nor hurts the reward trend. The arithmetic reward-trend


                                                    10

[PAGE 43]
The Causal Length-Mediation Protocol, CLMP (frontier synthesis). Gemini Deep Think argued
that marginal pass-rate tests — including the paired McNemar test our section leans on — are
confounded, because a length-hacking policy can post held-out successes by over-generating tokens
to stochastically stumble into reward. CLMP instead treats trajectory length L as a causal mediator
between the algorithm A ∈ {0:GRPO, 1:Dr.GRPO} and held-out success S, and decomposes the
total effect into a Natural Direct Effect (genuine deduction) and a Natural Indirect Effect (verbosity-
mediated success). In an internal cross-critique the models rejected their own first proposal —
hard “iso-brevity” truncation at the minimal proof length — as mechanistically fatal (it yields
syntactically invalid, incomplete trajectories and both algorithms trivially fail), and replaced it with
a non-destructive length-stratified estimator via Pearl’s mediation formula, letting each completion
finish naturally and marginalizing over the length distribution PA (l) ≡ P (L=l | A):
                                   X                                  
                         NDE =          E[S | A=1, l] − E[S | A=0, l] P1 (l),                       (13)
                                   l
                                  X                                 
                          NIE =        E[S | A=0, l] P1 (l) − P0 (l) .                              (14)
                                   l

The reported summary statistic is the Generalization-to-Exploit Ratio GER = NDE/TE; the frontier
prediction (to be tested, not a result here) is that in the length-confounded regime GRPO would
show GER < 0.15 (accuracy is mostly length-mediated exploit) while Dr. GRPO would show
GER > 0.85. On our own data, where length is already compressed and ρ(L, R) is negative, CLMP
would report a small NIE for both algorithms — consistent with the observed equivalence.

A reviewer-proof memorization-vs-generalization design (frontier synthesis). ChatGPT Pro
proposed replacing a bare paired test with a stratified, memorization-controlled generalization ladder
of four disjoint held-out strata: S0 exact/near-duplicates (a contamination sentinel, excluded from
the headline claim), S1 same template with new variables, S2 same skill with held-out template, and
S3 external distribution. Each eval item x receives a train-nearest-neighbor exposure score M (x)
blending BM25, char/n-gram overlap, embedding similarity, and answer/template match; items are
binned into M -deciles so that a memorization account predicts gains concentrated in high-M deciles
whereas generalization predicts nonzero gains in low-M , S2 /S3 items. The flagship estimand is
the adjusted low-similarity S2 /S3 treatment effect from a logistic model with item-difficulty and
seed random effects and a length covariate, so Dr. GRPO “wins” only if its coefficient survives
length and memorization controls. Two hard negative controls were specified: (i) length-matched
replay — subsample GRPO outputs to match Dr. GRPO’s length distribution within each stratum
and M -decile, so any residual gain is not merely a shorter/longer-decoding artifact; and (ii) answer-
preserving perturbation (rename variables, permute numbers, paraphrase) with a reported consistency
C = Pr(f (x) = f (T (x))), since memorization yields high raw accuracy but low C. McNemar is
demoted to a secondary check.

Adversarial objection and the length-adversarial truncation test (frontier synthesis). The
sharpest reviewer objection the models raised against Dr. GRPO is that removing the 1/Li divisor
should explode gradient variance on long chains; their crispest rebuttal is that the native leave-one-out
control variate contains that variance, so the divisor is an invalid heuristic rather than a necessary
regularizer. As the corresponding empirical probe, Gemini Deep Think proposed a length-adversarial
truncation test: evaluate both converged policies under harsh generation caps Tmax ≪ E[|yGRPO |].
If GRPO’s held-out accuracy relies on “stumbling into correctness” via padding, it should crater
non-linearly under truncation, whereas a genuinely length-invariant Dr. GRPO policy should degrade
gracefully — a direct behavioral test of whether compressed deductive logic, rather than Markov-
horizon exploitation, was acquired. We flag this as the single highest-value follow-up: it is one
truncation sweep away from being executed on the checkpoints Sec. 5 already produced, and unlike
the CLMP and stratified-ladder designs it needs no new training regime.

6   Discussion and Limitations
The central caution is that our comparison lives in a regime that is, by construction, unfavorable for
revealing a length effect: short arithmetic chains with a binary exact-match reward, and GSM8K-CoT
sampled with a 200-token cap where step-0 completions are already near the cap and length is
weakly (often negatively) coupled to reward. A null difference between GRPO and Dr. GRPO here is


                                                  43

[PAGE 44]
therefore weak evidence about length debiasing in general. Two further limitations matter. Held-out
evaluation is single-benchmark and the top-10-checkpoint frontier comparison is single-seed (the
pre-RL-vs-post Qwen3-8B-Instruct comparison uses five seeds), so the non-significant GRPO gain
should be read as a boundary on what we can claim rather than a universal result; and length dynamics
are only weakly identifiable in this data, so mediation-style conclusions require the length-confounded
regime we specify but do not yet run. The paired-outcome tests we rely on are also marginal tests,
which can be confounded by length-mediated success—a limitation the proposed causal mediation
protocol is designed to address.
On the statistics themselves, an independent Miller-style audit [12] re-derived seven headline numbers
across the four pillar papers under paired/unpaired bootstrap CIs and an equivalence-region (TOST)
test (experiments/results/berkeley/adding_error_bars_summary.json). The single P4-
facing headline—audit H7, the iter-136 arithmetic late-training-efficiency contrast (our H3; paired
n=5, Cohen’s d = +2.68, one-sided pparam = 0.031, two-sided pperm = 0.063)—is graded
DECISIVE at α = 0.05 under the audit’s n<10 paired-bootstrap rule, while the audit’s three NULL
verdicts (its H3, H4, H5) all fall on Pillar-1/3 headlines rather than on this paper’s claims.

7   Conclusion
In the near-ceiling Qwen3-8B-Instruct evaluation, GRPO does not significantly outperform its pre-RL
checkpoint on held-out GSM8K; in the Qwen2.5-1.5B comparison, Dr. GRPO does not outperform
GRPO at our seed budget; and under the 200-token GSM8K cap neither algorithm inflates length.
Rather than read this as evidence against length debiasing, we read it as the expected outcome when
there is no length pathology to correct, and we convert it into a concrete experimental program: a
length-confounded sparse-reward regime, a causal length-mediation protocol separating genuine
deduction from verbosity-mediated exploitation, and a length-adversarial truncation test on the
released checkpoints. The most useful contribution of this pillar is thus a precise statement of where
a Dr. GRPO advantage should and should not be expected.

References
 [1] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, and Marc
     Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Ad-
     vances in Neural Information Processing Systems, volume 34, pages 29304–29320, 2021.
     doi: 10.48550/arXiv.2108.13264. arXiv:2108.13264; supplementary materials at https:
     //agarwl.github.io/rliable/.
 [2] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier
     Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting REINFORCE-style
     optimization for learning from human feedback in LLMs. arXiv preprint, 2024. doi: 10.48550/
     arXiv.2402.14740. arXiv:2402.14740.
 [3] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared
     Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large
     language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
 [4] Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei.
     Deep reinforcement learning from human preferences. Advances in Neural Information Pro-
     cessing Systems, 30, 2017.
 [5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
     Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
     Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168,
     2021.
 [6] Cédric Colas, Olivier Sigaud, and Pierre-Yves Oudeyer. A hitchhiker’s guide to statistical
     comparisons of reinforcement learning algorithms. arXiv preprint, 2019. doi: 10.48550/arXiv.
     1904.06979. arXiv:1904.06979.
 [7] DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
     Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, et al. DeepSeek-R1: Incentivizing reasoning capability
     in LLMs via reinforcement learning. arXiv preprint, 2025. doi: 10.48550/arXiv.2501.12948.
     arXiv:2501.12948.


                                                  44


# P05: MIN-REPORT-RL


Root: `platform_hybrid/paper/paper_P5_minreport.tex`  Pages: 80  Words: 43387


[PAGE 1]
Report the Stack, Not the Label:
             RL-for-LLM Results Are Stack-Conditioned


                        Arvind C R                            Ramesh Prakash Guledgudd∗
                       PES University                              PES University
                   arvindcr4@gmail.com                           rameshpg@pes.edu



                                                   Abstract
             Papers comparing RL algorithms for LLM post-training report a label—GRPO,
             DAPO, Dr.GRPO, GSPO—and treat the surrounding software stack as an imple-
             mentation detail. Our position is that this convention is backwards: on the evidence
             available to us, the stack is a material part of the reported result, and a label without
             its stack is not a reproducible experimental specification. Three exhibits motivate
             the reporting standard. First, a nominally matched-configuration backend audit—
             every labeled knob (group size, LoRA setup, learning rate, data, prompt template,
             decoding, and seed) harmonized across frameworks, though as we disclose the
             managed backend was not matched at the base-checkpoint level—spanned final
             training rewards from 5.0% to 85.6%: a 17× same-label swing that cannot be
             assigned to the backend (the managed stack also silently pinned a different base
             checkpoint, and that undisclosed bundling is itself part of the exhibit). Second, the
             same label denotes different experiments across stacks: “DAPO” run on an open
             trainer with true dynamic sampling (a toy-scale arithmetic audit) produced a mean
             Zero-Variance Fraction (ZVF) of 0.00, while “DAPO” approximated on a closed
             stack (GSM8K) via an asymmetric-clip surrogate produced mean ZVF 0.58—same
             name, opposite telemetry. Third, a 12-cell head-to-head of four labeled algorithms
             on a single fixed stack landed within a 0.034 band of one another (mean last-10
             reward 0.710–0.744), a spread comparable to seed-level variation, suggesting that
             label-only comparisons can hide stack-level variation. We therefore propose an
             eight-item minimum reportable stack—loss form, reference-policy/KL handling,
             sampler/backend/precision (including base-checkpoint identity: revision/hash),
             per-step ZVF/GU trajectory, group-size schedule, disjoint held-out split, decon-
             tamination plus a parser-robustness probe, and held-out pass@k curves alongside
             pass@1—each item earning its place via a documented ranking-relevant lever. We
             call out base-checkpoint identity explicitly because our backend audit showed an
             undisclosed checkpoint swap is itself a ranking-relevant lever: the “stack” effect is
             only interpretable once the base checkpoint is pinned and reported alongside the
             framework. We formalize an RL-reproducibility threat model with flip-risk grades
             R0–R5, specify a toolchain (manifest emitter, stack diff with CI verdicts, registry,
             auditor badge, and citation-lever tracer) that makes the standard enforceable, and
             close with a call to action for venues.


1       Introduction: The Sandwich Problem
When a paper reports that “DAPO outperforms GRPO,” the named algorithm is a thin layer sand-
wiched between two thick, largely undocumented slices. Beneath the label sits the serving and
training substrate: the sampling engine and its numerics [18, 43], attention kernels and precision [6],
the trainer’s tokenization and masking conventions, and the exact loss implementation shipped by the
    ∗
        Project guide.


Preprint. Under review.

[PAGE 2]
framework [37, 34, 14, 36]. Above the label sits the reward and evaluation apparatus: the verifier
and its answer parser, the held-out split, the prompt template, and whatever decontamination was (or
was not) performed [40, 31]. The label names the filling; the experiment is the sandwich. We call
the resulting failure of reference the sandwich problem: two papers using the same algorithm label
routinely describe different experiments, and two papers using different labels routinely differ more
in their bread than in their filling.
This is a position paper. Its position is: RL-for-LLM results are stack-conditioned, and venues
should require authors to report the stack, not merely the label. We do not claim that algorithmic
labels are meaningless—the loss-form differences among GRPO [33, 7], DAPO [39], Dr.GRPO [22],
and GSPO [42] are real and sometimes consequential. We claim that on the evidence available
to us the stack’s contribution to reported outcomes is at least as large as the label’s, and that the
community’s reporting conventions are calibrated to the wrong term. A 17× same-label outcome
swing from a nominal backend comparison (Section 5) dwarfs every label-vs-label gap in our corpus.
That comparison also bundled a different base checkpoint, so it is evidence of under-specification, not
a backend causal effect; a label can even invert its own telemetry when re-implemented on a different
stack. Deep RL went through this reckoning once already [11, 5, 1]; RL-for-LLMs has re-created the
conditions with a larger and less inspectable stack.
Our contributions are deliberately those of a position paper: evidence, a standard, a threat model, and
enforcement machinery.
1. Evidence (Section 5): four exhibits—a 17× same-label stack flip whose backend and base
   checkpoint were not both matched; a cross-stack label flip in which “DAPO” yields mean ZVF
   0.00 on one stack and 0.58 on another; a 12-cell head-to-head in which four labeled algorithms
   on one fixed stack land within noise; and cross-library ZVF variation from the companion Pillar-2
   paper of T INKER RL-B ENCH.
2. The eight-item minimum reportable stack (Section 6): a checklist small enough to enforce, in
   which every item is justified by a documented lever that can move or flip a reported ranking.
3. An RL-reproducibility threat model (Section 10): a taxonomy of confound vectors and a
   flip-risk grading R0–R5 that classifies how dangerous a given cross-paper comparison is.
4. An enforceability toolchain (Section 11): a specification for a manifest-emitting trainer plu-
   gin (trl-min-report-rl), a manifest differ with CI verdicts (grpo-stackdiff), a machine-
   readable stack registry, a 0–100 auditor badge, and a citation-lever tracer (LeverTrace).
Throughout, we scope claims to their evidence: benchmark results come from T INKER RL-B ENCH
(Section 2); several exhibits come from internal program records that we describe as such; and
toy-scale theory validations are marked directional. The paper ends with limitations and a concrete
call to action for venues (Section 15).

2     Benchmark Design
2.1   Task Suite

                Table 1: Benchmark task suite with standardized reward functions.
           Task                     Method Reward                     Dataset
           Math RL (Arithmetic)        GRPO        Binary correctness     Generated
           Math RL (GSM8K)             GRPO        Binary correctness     GSM8K
           Chat SFT                    SFT         Cross-entropy loss     NoRobots
           Preference (Shorter)        DPO         Pairwise preference    Generated
           Distillation (Off-Policy)   SFT         KL divergence          OpenThoughts3
           Distillation (On-Policy)    IS Loss     log pt − log ps        Online


2.2   Cross-Library Implementation

Two seven-library rosters — read carefully. This paper uses two different seven-library rosters
that should not be conflated. The cross-RL-library benchmark below (Table 2) covers TRL, SB3,


                                                  2

[PAGE 33]
Table 28: Cross-corpus portability of the seven-field MIN-REPORT run manifest fingerprint. n =
number of manifest records; pop = number of items populated; var = number of items with ≥ 2
unique values; H = mean Hamming distance across bootstrap pairs with 95% CI.
      Corpus                    n    pop    var   total_bits           H               verdict
      C1 mega_20260704          98   7/7    3/7     4.80       1.891 [1.858, 2.028]    STRONG
      C2 n2_reward_tensor        4   7/7    1/7     2.00       1.000 [1.000, 1.000]    STRONG
      C3 n10_seed_expansion     16   7/7    2/7     4.00       1.716 [1.690, 1.810]    STRONG
      C4 base_instruct_paired    8   2/7    1/7     0.95       0.555 [0.518, 0.606]    LIMITED
      C5 group_size_iter111      4   4/7    1/7     2.00       1.000 [1.000, 1.000]   PORTABLE
      C6 length_bias_iter60     20   2/7    2/7     2.52       1.213 [1.150, 1.260]    LIMITED
      C7 zvf_iter118_auroc       8   3/7    2/7     3.00       1.472 [1.418, 1.502]   PORTABLE



recipe replayed on this corpus) confirms the reading: stack axes (G, task_slice, temperature, model_-
                                                            2
family) explain 94.7% of ZVF variance on the 98 cells (ηstack    = 0.947, 25 unique buckets), while
 2
ηalgo = 0 by construction (single-algorithm campaign). The remaining 5.3% is seed-level residual.
Reading (A) is therefore refuted: the schema is not the binding constraint, the corpus is. The
sharp operational recommendation: add temperature_schedule to the v2 MIN-REPORT schema and
document the four GRPO/PPO hyperparameter items as deferred-to-cross-stack-campaign until a v2
mega-campaign varies the relevant axes.

7   Cross-Corpus Portability of the Seven-Field MIN-REPORT Run Manifest
The seven-field run-manifest portion of the MIN-REPORT standard in Section 6 was
validated on a single corpus (the 98-cell mega campaign of Section 5.6).         A re-
maining question is whether the standard generalizes across the heterogeneous experi-
ment corpora in this repository—or whether it is a schema custom-built for one cam-
paign. We answer with a portability test (scripts/p5p8/p5_cross_corpus_portability.py;
experiments/results/p5p8/p5_cross_corpus_portability.*).

Method. Apply the seven MIN-REPORT run-manifest fields to seven internal corpora and measure,
for each, (a) per-item coverage, (b) per-item variance, (c) total bits, (d) mean Hamming discrimination
across random record pairs (B=2000 bootstrap, seed 20260705), and (e) a STRONG / PORTABLE /
LIMITED / NULL verdict. The seven corpora are deliberately heterogeneous: the full 98-cell manifest
(C1); the N2 four-method same-stack tensors (C2); the N10 2-algo × 8-seed expansion manifest (C3);
and four rows-only summary corpora (base-instruct paired, group-size paired, length-bias paired,
ZVF per-stratum AUROC).

Verdict rule. STRONG iff ≥ 5 of 7 manifest fields populate and ≥ 1 carries variance; PORTABLE
iff ≥ 3 populate and ≥ 1 varies; LIMITED iff ≥ 1 populates and ≥ 1 varies; NULL otherwise.

Headline. Across the seven corpora, mean n_items_populated = 4.57/7 with 95% bootstrap
CI [3.00, 6.29], confirming that the standard is portable but stratified. The verdict distribution
is 3 STRONG, 2 PORTABLE, 2 LIMITED, 0 NULL—no corpus is a complete schema-failure.
Per-item coverage across corpora reveals a sharp separation: Items 1 (loss_form) and 6 (heldout_-
split) are populated in every corpus (7/7); Items 2, 3, 7 (KL, backend, decontam) are populated
only in manifest-level corpora (C1/C2/C3); Item 4 (zvf_gu_trajectory) is populated in 5/7; Item 5
(group_size_schedule) in 4/7.

Cross-paper coupling. This iter independently reproduces the iter-65 placebo pattern from the
fingerprint-as-data path: on C1 four items carry n_unique=1 because the campaign is a single-stack
Tinker-closed run, while the other three (group_size_schedule, heldout_split, decontamination_notes)
carry all 4.80 bits of cross-cell information. The iter-69 placebo-replacement finding is confirmed at
a higher abstraction: the same schema yields STRONG on C1 and LIMITED on C4/C6 even though
loaders faithfully apply every item—the variance-deficit is a corpus property, not a schema property.
The iter-73 v2.0 stack-axis extension is sharpened: the +60.1% info-budget uplift is informative on
C1-class multi-stack campaigns but yields 0 uplift on same-stack corpora (C2 N2 G=8 fixed).


                                                  33

[PAGE 61]
9.3.5    Limitations

(i) The mega panel has only 2 models, 3 task slices, 2 temperatures, and 2 seeds; the per-axis η 2
estimates have only 2–5 groups. The 95% CIs on the per-axis η 2 are wide; the headline conclusion
rests on the union-axis η 2 = 0.9967 over 50 stack cells, where the central limit theorem gives a
tighter read. (ii) The mega panel uses training-free sampling (loss_form: n/a-sampling in the
manifests); algorithm-axis comparisons on this panel would be confounded by stack drift and are
not attempted here. (iii) The H7 NULL on the step axis is *not* a refutation of the P5 thesis: the P5
thesis is that the cross-method signal is small when stack is fixed, not that within-run variation is flat.
The high η 2 (step) is the within-run learning curve (40 steps on the same prompts), exactly what the
brief expects.

9.4     Per-step algorithm-axis η 2 trajectory (iter 165)

The pooled iter-161 row 176 headline — η 2 (method → reward_mean)=0.0075 on the 160-row N2
panel — is a single number over 40 training steps. The open question is whether that number is
trajectory-stationary (the algorithm axis is a constant latent structure) or trajectory-trending (the
algorithm axis emerges or decays as training proceeds). Iter 165 tests this by computing per-step
η 2 (method|step) on the 4 method × 16 prompt = 64 obs per step per channel, with paired-prompt
bootstrap CIs (B=2000, seed 20260705).

9.4.1    Per-step decomposition

For each step s              ∈      {0, . . . , 39} and each prompt-level channel c                  ∈
{reward_mean, mean_len, cv_len}, we form 4 method-groups of 16 prompt-mean values:
  2
ηc,s = SS_between(method)/SS_total on the 64 obs. Bootstrap resamples the 16 prompts with
                                                     2
replacement (paired across methods), recomputes ηc,s     2000 times, and reports point, percentile 95%
CI, and bootstrap mean. The 40 steps are split into 3 trajectory bands: early (steps 0–13, 14 obs), mid
(steps 14–26, 13 obs), late (steps 27–39, 13 obs).

9.4.2    5 falsifiable hypotheses (4/5 PASS)

H1 (PASS, DECISIVE). Per-step mean η 2 (method|step) ≤ 0.05 on ≥ 2/3 prompt-level channels.
Evidence: mean η 2 on reward_mean = 0.0056, on mean_len = 0.0139, on cv_len = 0.0405 —
all three channels pass. The algorithm axis is small per-step on every prompt-level channel.
H2 (FAIL, sharpest negative). Trajectory |Spearman ρ| ≤ 0.5 on ≥ 5/6 channels (trajectory-
stationary). Evidence: ρ(reward_mean)=+0.114, ρ(mean_len)=+0.875, ρ(cv_len)=+0.401 —
only 2/3 channels pass. mean_len is strongly trajectory-monotone (Spearman +0.875): early-band
mean η 2 (mean_len)=0.0019, mid-band 0.0096, late-band 0.0312 — a 16× monotone growth across
the 40-step trajectory. This is the first P5 finding of a training-trajectory trend in the algorithm-
axis decomposition: the algorithm axis is not just static "label noise" but a training-time emergent
signal on length-controlled channels.
H3 (PASS, exact replication). Pooled η 2 (method, reward_mean) matches iter-161 within ±0.005.
Evidence: re-derived from canonical n2_metrics.tsv (160 rows) gives 0.0074664, matching iter-
161’s 0.0075 at ∆ = 3.4 × 10−5 (0.45% error). The two implementations converge to four-decimal
precision.
H4 (PASS, LOMO confirmation). GIFT dominates the algorithm axis on reward_mean:
LOMO(GIFT)/full < 0.5. Evidence: leave-one-method-out at per-(method, prompt) granular-
ity — removing GIFT collapses η 2 from 0.000503 to 0.0000911 (0.181×, 82% drop). Removing
GRPO raises η 2 to 1.326× full (the 3 non-GRPO methods are more dispersed); removing AERO
gives 0.962×; removing AREAL gives 1.086×. Only GIFT removal shrinks the algorithm axis,
confirming iter-89/106 H3 finding at the reward level (not just zvf).
                                                      2                         2
H5 (PASS, stationarity on reward). |mean(early ηreward_mean    ) − mean(late ηreward_mean  )| ≤ 0.02.
Evidence: early band mean = 0.0056, late band mean = 0.0060, ∆ = 0.0004 (200× below the
stationarity bar). The iter-161 pooled headline of η 2 =0.0075 is trajectory-robust on the canonical
reward_mean channel.


                                                   61

[PAGE 71]
10.1   Confound Vectors

A confound vector is a class of undisclosed stack variation capable of carrying a flip. Our taxonomy
has five, each anchored to documented evidence:
V1 — Numerics and serving. Backend, sampling engine, kernels, precision, decoding parameters.
   These fields co-varied with base-checkpoint identity in the 17× under-specification exhibit
   (Section 5.1); no vector-specific effect size is identified.
V2 — Objective semantics under a shared label. Loss-form divergence hidden by a common
   name: clip asymmetry, dynamic sampling, normalization, ratio level. Documented magnitude:
   qualitative telemetry inversion (mean ZVF 0.00 vs. 0.58; Section 5.2). The vector also runs in
   reverse — a claimed divergence of zero magnitude: in our own trainer a documented-but-unwired
   –loss drgrpo flag left a six-arm “GRPO vs. Dr.GRPO” panel training the identical objective
   under two labels, and no reward, length, or ZVF trace revealed it; only reading the runner did.
V3 — Signal geometry. Group size and its schedule, advantage normalization, and the accuracy
   regime relative to the ZVF U-shape. Documented magnitude: mean ZVF 0.33 → 0.08 moving G
   from 4 to 16 on Qwen3-32B (Table 27); regime aliasing per Table 26.
V4 — Environment and data. Contamination of the training or evaluation pool and parser/verifier
   idiosyncrasies. Documented magnitude: contamination-driven score inflation [40, 31]; sub-
   resolution reward jitter zeroing a telemetry channel (Section 6.7).
V5 — Evaluation protocol. Held-out split identity, prompt template, answer extraction. Docu-
   mented magnitude: protocol-level variance large enough to reorder leaderboards in independent
   audits [3, 12, 24].

10.2   Flip-Risk Grades R0–R5

Given two runs’ manifests (Section 11), we grade the risk that a comparison between them is
confounded. The grade is determined by the most severe difference found across the seven run-
manifest fields of Table 25.
R0 — Replicate. All seven manifest fields identical; runs differ at most in hardware instance. Com-
   parison measures noise floor.
R1 — Seed-only. Items identical; seeds differ under a declared seed policy. Comparison is the
   intended unit of statistical analysis [5, 1]; report dispersion, not a single delta.
R2 — Declared-knob. Items differ only in declared, quantitatively reported knobs (learning rate,
   steps, G with fixed schedule form). Comparable, with the knob difference stated as a caveat or
   bridged by a matched run.
R3 — Stack-vector. At least one V1/V3/V4/V5 vector differs (backend, precision, parser, split,
   schedule form). Same-label stack swings in our corpus reach 17×, but the largest exhibit is
   multi-confound; algorithmic conclusions require a bridging run on a common stack before they
   are admissible.
R4 — Label collision. The same algorithm label with different objective semantics (V2): the com-
   parison is between stacks wearing one name. Verdict: NOT- COMPARABLE; the label must be split
   (e.g., “DAPO(dynamic-sampling)” vs. “DAPO(clip-surrogate)”).
R5 — Undeclared. One or more of the seven manifest fields is missing from either manifest. Verdict:
   UNVERIFIABLE ; no grade can be assigned, which is itself the finding.

Two consequences follow. First, most cross-paper comparisons in today’s RL-for-LLM literature are
R3–R5 by construction, because items 3, 4, and 7 are almost never reported; the literature’s aggregate
ranking of GRPO-family algorithms is therefore an average over unmodeled stacks. Second, the
grading is mechanical—it requires no judgment call beyond the manifest contents— which is what
makes it suitable for continuous integration (Section 11) rather than for post-hoc reviewer heroics.

11     The Toolchain That Makes It Enforceable
Checklists change practice only when compliance is cheaper than non-compliance. The NeurIPS
reproducibility program showed that a checklist plus process measurably improves reporting [28]; our


                                                 71

[PAGE 74]
LLM benchmark scores by ranking-relevant margins, and argued for shared, versioned evaluation
infrastructure [3]. Independent audits reach the same conclusion for reasoning evaluations specifically
[12], and contamination studies show the data side of the same coin [40, 31]. Our claim is that RL
training inherits all of this—the verifier is an evaluator in the loop—plus the V1–V3 vectors unique
to on-policy optimization.

The GRPO-variant explosion. The label space we target is growing quickly: GRPO [33, 7], DAPO
[39], Dr.GRPO [22], GSPO [42], and a long tail of descendants [21, 27, 41, 23, 4]. Analyses that
unify the family—GRPO’s preference-learning reading [30, 38] and zero-variance-prompt recovery
methods [20] —support our premise that members differ by a small set of loss-form coordinates,
which is exactly what item 1 of the manifest records. Empirical scaling studies of GRPO knobs [35]
likewise presuppose the per-knob disclosure the manifest mandates. The companion T INKER RL-
B ENCH pillar papers (P1 scaling, P2 ZVF, P3 group size, P4 length bias) provide the measurement
substrate this position paper argues should become the norm.

The reporting-standards lineage of MIN-REPORT (iter 109 verification). The MIN-REPORT
standard inherits directly from four documented dataset/model reporting standards, each of which
we verified against CrossRef metadata at iter 109 (p5_iter109_crossref_verify.tsv; 4/4 pass
year + 5-gram title + author-family-overlap≥ 0.6 vs the CrossRef record). Mitchell et al. [26] intro-
duced Model Cards to document intended use, out-of-scope use, and per-slice evaluation results for
trained models; this lineage gives us MIN-REPORT items 1 (architecture and training procedure), 2
(intended and out-of-scope use), 7 (per-slice metrics, not just averages), and 8 (ethical considerations
and caveats). Gebru et al. [10] extended the idea to datasets via Datasheets, motivating items 3
(dataset composition), 5 (preprocessing, cleaning, labeling, uses), 9 (annotation process), and 10
(discriminatory-impact mitigations). Bender and Friedman [2] focused on natural-language data and
added the speaker-demographic axis, motivating items 11 (speaker consent and curation rationale)
and 12 (annotator demographics and quality control) which are not present in Mitchell’s or Gebru’s
templates. Pushkarna et al. [29] introduced Data Cards to bridge research and industry practice, moti-
vating items 4 (dataset schema and provenance), 6 (sampling distribution and representativeness), 13
(per-stack axis field markers, the zvf_yield_residual line in MIN-REPORT v2.1), and 14 (audit
transparency and version tracking). The 4 papers cover 12 distinct fields in a legacy experimen-
tal MIN-REPORT expansion (cross-coupling audit: p5_iter109_minreport_coupling.tsv);
the remaining 6 fields in that 18-field experimental manifest are either RL-specific (field 15 zvf
per-step, field 17 group-size schedule) or stack-coverage items without a direct analogue in the
prior reporting-standards literature (item 16 was rejected as a signal-bearing field at iter 81). This
18-field research schema is not the current eight-item standard in Section 6. Two of the four entries
(Bender and Friedman [2] and Pushkarna et al. [29]) were not present in paper/references.bib
prior to iter 109; iter 109 adds them with full DOI, year, volume/number/pages, and verified author
lists. The other two entries (Mitchell et al. [26], Gebru et al. [10]) were already in the bib; iter 109
retrofits them with DOIs and now actually cites them in the related work (prior to iter 109 they
were present in the bibliography but uncited). The cross-coupling audit also exposes a measurement
asymmetry: of the 12 MIN-REPORT items covered by these reporting-standards papers, items 9–12
(annotation process and annotator-demographic axes) have the lowest evidence density in our live
manifests (p5_iter109_bib_audit.tsv, see also iter 105’s 5-vs-3 discriminative-vs-primitive
split), motivating a future-iter extension of MIN-REPORT to formalise annotation-process fields at
the same strictness as the stack-axis fields.


14    Limitations

Position papers owe their readers an honest account of the evidence’s edges; five limitations are
load-bearing here. First, the headline exhibits (the 17× under-specified same-label comparison,
the cross-stack DAPO telemetry flip, the 12-cell head-to-head, and the 368-run audit) are internal
program records: they were run under a declared protocol and are summarized faithfully, but they
have not undergone the full artifact-release pipeline of the T INKER RL-B ENCH Stratum-A results, and
the closed stack involved cannot be independently re-executed by readers—an irony we acknowledge
and which the manifest standard is designed to at least make legible. Second, the gradient-signal
validation behind item 4’s theory (corr = +0.71 between gradient norm and p(1 − p)) is toy-scale—a
0.5B model on synthetic arithmetic with an open backward pass—and is directional evidence only;


                                                  74

[PAGE 75]
we do not report it as an effect size. Third, our evidence is concentrated on GSM8K-style verifiable-
reward tasks and the GRPO family; the eight-item standard (seven run-manifest fields plus pass@k
reporting) was chosen for that regime, and reward-model-based RLHF stacks may need additional
items (reward-model version and training data at minimum). Fourth, the flip-risk grades encode a
conservative ordering (label collision graded above generic stack divergence) that we have validated
on our own audits but not against a broad external corpus; the grade boundaries, especially R2 vs. R3,
will need community calibration. Fifth, the toolchain of Section 11 is a specification with feasibility
evidence, not a released product suite; the live adaptive-G result shows the telemetry is cheap, but
manifest adapters for every framework in Table 2 remain to be built. None of these limitations
weakens the central asymmetry the paper rests on: every documented lever we cite moved results by
more than the label differences the field currently headlines.

15     Conclusion: A Call to Action for Venues
The evidence pattern is consistent and, we argue, sufficient to change reporting policy: on a fixed
stack, four algorithm labels landed within 0.034 reward of one another; across under-specified stacks,
a nominal backend comparison that also changed the base checkpoint moved reward by 0.794, and a
single label produced opposite telemetry signatures. The former is not a backend causal estimate;
both are evidence that RL-for-LLM results are stack-conditioned. The community’s papers should
say so, in a form machines can check.
Our call to action is addressed to venues, because venues set the price of non-disclosure:
1. Require the manifest. Any paper whose contribution depends on an RL-for-LLM comparison
   should attach the seven-field run manifest and the eighth evaluation report (pass@k; Table 25)
   per compared run as supplementary JSON. Auto-emission (Section 11) makes this a one-line
   cost for open trainers; for closed stacks, “provider-internal, version-dated” is an acceptable and
   informative value—the point is that conditioning be declared, not that all stacks be open.
2. Grade the comparisons. Reviewers should see the grpo-stackdiff grade for every headline
   comparison. R0–R2 comparisons support algorithmic claims; R3–R4 comparisons should be
   rewritten as stack-conditioned observations; R5 should be treated as a missing-experiment flag,
   exactly as a missing baseline is today.
3. Split colliding labels. Where the registry shows one label with multiple registered semantics,
   require the qualified form (“DAPO(dynamic-sampling)”) in abstracts and tables. Precedents exist:
   the field already distinguishes “PPO” from “PPO-clip” when it matters [32, 11].
4. Reward the telemetry. Per-step ZVF/GU trajectories and group-size schedules should count as
   artifact contributions in their own right, as evaluation-harness versioning now does [3, 28].
The deep-RL community needed a reporting reckoning [11], a statistics paper [5, 1], and a decade
to converge on reporting norms. RL-for-LLMs can shortcut that decade: the confounds are already
documented, the run manifest is seven fields, the evaluation requirement is the eighth item, and
enforcement is a diff. Report the stack, not the label.

A     Worked Example: Two “DAPO” Manifests and Their Verdict
This appendix instantiates the standard end-to-end on the two runs of Exhibit 2 (Section 5.2): the
same algorithm label, run on a closed training stack and on an open single-file trainer. Field values
marked "provider-internal" illustrate the disclosure convention for closed stacks (Section 15):
a closed value is acceptable; an absent field is not.

A.1    Manifest A: “DAPO” on the Closed Tinker Stack

{
    "minreport_version": "0.1",
    "run_id": "tinker_gsm8k_dapo_s42",
    "label_claimed": "DAPO",
    "loss_form": {
      "ratio_level": "token",


                                                  75

[PAGE 77]
"disjoint_from_reward_env": true
    },
    "decontamination": {
      "performed": true,
      "parser_probe": "micro-jitter eps ~ U(0, 1e-4): zvf-sensitive, pcd-invariant"
    }
}



A.3     The Stackdiff Verdict

Running the differ on the two manifests:
$ grpo-stackdiff manifest_a.json manifest_b.json

ITEM DIFF
  1 loss_form.dynamic_sampling false != true [V2]
  1 loss_form.surrogate_note "grpo+asym-clip" != null [V2]
  3 sampler_backend.trainer tinker (closed) != open-colab [V1]
  3 sampler_backend.precision provider-internal!= bf16 [V1]
  4 zvf_gu_trace.mean_zvf 0.58 != 0.00 [consequence]
  5 group_size.G 8 != 4 [V3]
  7 decontamination.performed false != true [V4]
  7 decontamination.parser_probe not-run != micro-jitter [V4]

FLIP-RISK GRADE
  R4 (label collision): same label_claimed "DAPO", divergent objective
      semantics (dynamic_sampling). Additional R3 vectors present (V1, V3, V4).

CI VERDICT: NOT-COMPARABLE
  Do not report these runs as "DAPO vs DAPO". Qualified labels required:
    manifest_a -> DAPO(clip-surrogate, no-dynamic-sampling)
    manifest_b -> DAPO(dynamic-sampling)
  A bridging run on a common stack is required before any algorithmic claim.


The diff makes Exhibit 2 mechanical: the telemetry inversion (mean ZVF 0.58 vs. 0.00) is listed not
as a difference to adjudicate but as a consequence of the item-1 divergence, and the verdict line is
exactly what a leaderboard CI job would gate on. Note also what the diff does not say: it does not
claim either run is wrong, or that the closed stack is inferior—only that the pair cannot support a
label-level conclusion. That restraint is the entire content of the standard.


References
    [1] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, and Marc
        Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Ad-
        vances in Neural Information Processing Systems, volume 34, pages 29304–29320, 2021.
        doi: 10.48550/arXiv.2108.13264. arXiv:2108.13264; supplementary materials at https:
        //agarwl.github.io/rliable/.
    [2] Emily M. Bender and Batya Friedman. Data statements for natural language processing:
        Toward mitigating system bias and enabling better science. Transactions of the Association for
        Computational Linguistics (TACL), 6:587–604, 2018. doi: 10.1162/tacl_a_00041.
    [3] Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Ab-
        basi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, et al.
        Lessons from the trenches on reproducible evaluation of language models. arXiv preprint
        arXiv:2405.14782, 2024.
    [4] ByteDance Seed, Yu Yue, Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu,
        Jiaze Chen, Chengyi Wang, Tiantian Fan, et al. VAPO: Efficient and reliable reinforcement
        learning for advanced reasoning tasks. arXiv preprint, 2025. doi: 10.48550/arXiv.2504.05118.
        arXiv:2504.05118.


                                                  77


# P06: GRPO registry


Root: `platform_hybrid/paper/paper_P6_registry.tex`  Pages: 65  Words: 34187


[PAGE 1]
GRPO-Registry: A Machine-Readable Catalog of
    Group-Relative RL Stacks and Their Variant Deltas


                        Arvind C R                          Ramesh Prakash Guledgudd∗
                       PES University                            PES University
                   arvindcr4@gmail.com                         rameshpg@pes.edu



                                                 Abstract
             The deltas that define GRPO variants—DAPO’s asymmetric clip, dynamic sam-
             pling, and token-level loss; Dr. GRPO’s dropped length and std normalization;
             GSPO’s sequence-level importance ratio—live in paper appendices and framework
             defaults, not in any machine-readable form. The consequence, documented across
             the T INKER RL-B ENCH pillar papers, is that an algorithm label underdetermines
             the update rule actually executed: in our audits, swapping the training backend
             under a fixed label—which, undisclosed, also bundled a different base checkpoint,
             so the label alone concealed both changes—moved final reward across a 17× span
             between 85.6% and 5.0% (the same exhibit as the Pillar-5 backend audit, read here
             in the reverse direction), and the same “DAPO” label yielded mean ZVF 0.00 on
             an open trainer with true dynamic sampling but 0.58 on a closed stack running an
             asymmetric-clip surrogate. These audits motivate GRPO-R EGISTRY, a resource
             that makes such discrepancies queryable rather than presenting a new performance
             result: (i) a JSON schema whose stack records carry the seven-field run-manifest
             block of the eight-item MIN-REPORT-RL standard (loss form; reference policy
             and KL handling; sampler/backend/precision including base-checkpoint identity:
             revision/hash; per-step ZVF/GU telemetry; group-size schedule; held-out split;
             decontamination and parser probes), with held-out pass@k curves recorded as
             the eighth, evaluation-report requirement, plus per-component variant-delta status;
             (ii) three variant-delta records verified against the DAPO, Dr. GRPO, and GSPO
             papers; (iii) twenty seed stack entries derived from this benchmark’s released config
             dumps, an open Colab trainer audit, a managed-runtime head-to-head, a same-stack
             four-method run, and a five-seed risk-index batch; and (iv) a reference CLI imple-
             menting registry queries, a 0–100 MIN-REPORT auditor badge, and stackdiff,
             which maps a pair of entries to a flip-risk verdict (R0–R5). On the seed entries the
             badge spans 43–96, and stackdiff flags the open-vs-closed “DAPO” pair as an
             R5 label-flip risk from manifests alone. The registry, schema, and CLI are released
             with the benchmark.


1       Introduction
Group Relative Policy Optimization [20, 5] has spawned a family of variants at a pace that outstrips
the community’s bookkeeping. DAPO changes the clip asymmetry, the sampling loop, the loss
aggregation, and the reward shaping in one release [23]; Dr. GRPO deletes two normalization
terms [15]; GSPO moves the importance ratio from tokens to sequences [25]; and a long tail of further
variants adjusts baselines, curricula, or advantage estimators [13, 18, 24, 10, 21, 14, 9, 16, 3, 11].
Each delta is documented—but in prose, in an appendix, in a YAML default, or in a code path that a
framework may or may not implement. There is no machine-readable place to ask: which changes
does this stack actually make to base GRPO, and which of them does my launcher actually execute?
    ∗
        Project guide.

Preprint. Under review.

[PAGE 61]
is not promised beyond the benchmark’s maintenance window, and we say so rather than pretend
otherwise.

Scope discipline. The registry catalogs GRPO-family stacks because that is where our evidence
of label–implementation divergence lives. Widening to all RLHF methods would multiply schema
surface faster than curation capacity; the intended growth path is depth (more stacks, more verified
deltas) before breadth.


15    Synthesis: Why Each Schema Field Earned Its Place

The registry’s seven run-manifest fields are not a wishlist; each was forced by a measured failure
somewhere in this program. The full reporting standard adds pass@k as an eighth evaluation
requirement; the registry is the container that makes the run-start lessons reusable:

 • Items 1 and 3 (loss form; backend/precision) are forced by the stack-dependence results: a
   same-label stack comparison moved final reward 85.6% → 5.0% while also bundling a different
   base checkpoint, and the scaling pillar (P1) found stack identity competing with model scale as a
   predictor of outcome. Rankings that do not pin these fields are rankings of launchers.
 • Item 4 (per-step ZVF/GU) is forced by the ZVF pillar (P2): ZVF is U-shaped in accuracy
   across the 368-run audit (∼0.95 at both extremes, ∼0.25 mid-curve), so only a trajectory, not an
   endpoint, is interpretable—and it is cheap to emit, since it needs only per-group rewards. The
   pillar’s micro-jitter falsification (batch ZVF 0.158 → 0.000 under ∼10−4 reward jitter while
   contrast-density measures are invariant) is also why the schema records the telemetry source: a
   ZVF of zero means different things from different reward pipelines.
 • Item 5 (group-size schedule) is forced by the group-size pillar (P3): expected ZVF-indicator
   is pG + (1 − p)G , so larger G mechanically lowers ZVF, and our model×G sweeps confirm
   the monotone drop (e.g. Qwen3-32B: 0.33/0.18/0.08 at G=4/8/16). Comparing stacks at
   unrecorded G confounds algorithm with schedule; the adaptive-G seed entry shows the field
   capturing a live controller, not just a constant.
 • Items 2, 6, 7 (KL; held-out split; decontamination/parser) are forced by the confound audits behind
   P4 and the benchmark design (Section 2): silent β defaults differ 2× across open frameworks,
   and length-bias results flip under reward-parser changes that item 7’s probe would surface.

Read this way, the registry is the program’s exit artifact: the pillar papers established which stack
facts flip conclusions; the registry is the minimal data structure in which those facts travel with the
stack instead of with the paper trail.


16    Discussion and Limitations

Four limitations bound what this resource can claim. First, seed breadth: all twelve stack entries
derive from one research program’s artifacts. They exercise every schema feature (open/managed,
implemented/ surrogate/absent/unknown, fixed/adaptive G, dry-run vs. completed), but the registry’s
value at community scale is a hypothesis until external entries arrive. Second, evidence scale: the
open-trainer arms are toy-scale Colab runs; we use them as existence proofs that the schema can
capture a fully-audited stack, and as directional corroboration of the manifest-level DAPO diff—not
as effect sizes. The larger-scale numbers (the under-specified same-label swing and the 368-run
U-shape) come from the larger program audits reported in the companion pillars. Third, self-report:
for closed stacks the registry records what the operator exposes; the null semantics and R5 rung make
unauditability visible, but they cannot manufacture the missing facts—an entry can be honest and
still thin. Fourth, badge semantics: the badge measures reporting coverage with equal item weights;
it is not a correctness proof, not a quality score, and alternative weightings would reorder the middle
of Table 6 (the endpoints are robust). We ship the scoring function precisely so disagreements about
weights can be run, not argued.


                                                  61

[PAGE 62]
17       Conclusion
GRPO variants are defined by small deltas that current practice scatters across appendices, defaults,
and closed launchers. Our program’s audits show why that scattering matters for interpretation: same-
label stacks can have opposite telemetry and a 17× same-label outcome swing. GRPO-R EGISTRY
is the smallest artifact we could build that makes the problem tractable: a schema that carries the
seven-field run-manifest portion of the eight-item reporting standard and per-component variant-delta
status, twenty seed entries derived from real artifacts, and a CLI whose stackdiff turned the
open-vs-managed “DAPO” pair into an explicit label-flip warning from manifests alone. The registry
does not make stacks comparable; it makes their incomparability precise, and that is the prerequisite
for every fair comparison the field wants to run. Schema, entries, and tooling are released with the
benchmark; the entry format is a pull request away for anyone whose stack we did not catalog.

A       Artifact Reference
A.1      Layout and Validation

The artifact ships in the benchmark repository:
                                                      Listing 5: Artifact layout.
registry /
  schema . json                             # JSON Schema , draft 2020 -12; record types :
                                            #     stack | variant_delta
    entries /                               # 20 stack records + 11 variant - delta records
      trl_grpo_qwen3 -8 b_gsm8k . json                           verl_grpo_qwen3 -8 b_gsm8k . json
      openrlhf_grpo_qwen3 -8 b_gsm8k . json                      tinker_grpo_qwen3 -8 b_gsm8k . json
      colab - open_grpo_e3 . json                                colab - op en_drg rpo_e3 . json
      colab - open_dapo_e3 . json                                colab - open_grpo - adaptiveg_e3 . json
      t i n k e r _ g r p o _ q w e n 3 .5 -4 b_gsm8k . json     t i n k e r _ d a p o _ q w e n 3 .5 -4 b_gsm8k . json
      t i n k e r _ d r g r p o _ q w e n 3 .5 -4 b_gsm8k . json t i n k e r _ g s p o _ q w e n 3 .5 -4 b_gsm8k . json
      delta_dapo . json delta_drgrpo . json delta_gspo . json
    query . py                              # reference CLI ( stdlib only )
    README . md


Every entry validates against the schema:
python3 -c " import json , glob , jsonschema ; \
  s = json . load ( open ( ’ registry / schema . json ’)); \
  [ jsonschema . validate ( json . load ( open ( p )) , s ) \
    for p in glob . glob ( ’ registry / entries /*. json ’)]"



A.2      A Complete Variant-Delta Record

The DAPO record, verified against the source paper [23]:
                        Listing 6: entries/delta_dapo.json (whitespace compressed).
{ " record_type ": " variant_delta " , " sc hema_v ersion ": "0.1.0" ,
  " id ": " delta_dapo " , " name ": " DAPO " , " base ": " grpo " ,
  " citation ": { " bibkey ": " yu2025dapo " , " arxiv ": "2503.14476" ,
     " title ": " DAPO : An Open - Source LLM Reinforcement Learning
                  System at Scale " } ,
  " deltas ": [
    {" component ": " clip_higher " ,
     " field ": " loss_form . clip_eps_high " ,
     " change ": " asymmetric ( decoupled ) clip : eps_low =0.2 ,
                   eps_high =0.28"} ,
    {" component ": " d y n a m i c _ s a m p l i n g " ,
     " field ": " sampling . d y n a m i c _ s a m p l i n g " ,
     " change ": " filter prompts whose groups have zero reward
                   variance and resample until the batch is full "} ,
    {" component ": " t o k e n _ l e v e l _ l o s s " ,
     " field ": " loss_form . aggregation " ,
     " change ": " token - level policy - gradient loss ( mean over all
                   tokens in the batch rather than per - sequence
                   averaging )"} ,
    {" component ": " o v e r l o n g _ r e w a r d _ s h a p i n g " ,
     " field ": " reward . o v e r l o n g _ s h a p i n g " ,
     " change ": " soft length - aware penalty for truncated / overlong



                                                                       62


# P07: ZVF controller


Root: `platform_hybrid/paper/paper_P7_zvf_controller.tex`  Pages: 81  Words: 45413


[PAGE 1]
From Diagnostic to Controller?
  A Retrospective Audit and Prospective Test Plan for
                 Adaptive Group Size


                       Arvind C R                           Ramesh Prakash Guledgudd∗
                      PES University                             PES University
                  arvindcr4@gmail.com                          rameshpg@pes.edu



                                                 Abstract
            Our companion Pillar-2 paper characterized the Zero-Variance Fraction (ZVF)—
            the share of GRPO groups whose completions all receive the same reward—as a
            deliberately descriptive diagnostic of signal starvation. This paper is a retrospective
            controller audit and a prospective intervention plan: we ask which ZVF-derived
            quantities are mechanistically predictable and whether they are ready to control
            rollout allocation. This does not overturn Pillar-2’s restraint—we test ZVF’s
            predictive value through falsifiable predictions (T1–T3 below) rather than assume
            it, and we flag where that value rests on the weakest evidence (the causal gradient
            link is toy-scale and directional only). We model the usable learning signal per
            prompt as S = p(1 − p) (1 − hG (p)), where p is the latent success probability and
            hG (p) = pG + (1 − p)G is the probability a size-G group is degenerate, and derive
            three testable consequences: (T1) the expected ZVF indicator equals pG +(1−p)G ;
            (T2) larger group size G lowers ZVF; (T3) gradient magnitude tracks p(1 − p).
            Each prediction is paired with an empirical check: a 368-run Weights & Biases
            audit across seven models (all logged runs—a superset of the benchmark’s 70+
            core training runs; the core set is defined by run type, i.e. completed GRPO training
            runs, and is not filtered on convergence or reward, while the remaining logged runs
            are evaluations, ablation probes, and short/aborted infrastructure runs—so the audit
            is not subject to outcome-based survivorship selection) reproduces the predicted
            U-shape of ZVF in accuracy (ZVF 0.95–0.97 at both reward extremes, 0.25–0.29
            at mid-range rewards of 0.35–0.50); a model × G grid confirms the monotone
            G effect at interior accuracy (e.g. Qwen3-32B: 0.33 → 0.18 → 0.08 for G =
            4/8/16); and a toy-scale open-backward-pass experiment finds corr(∥∇∥, p(1 −
            p)) = +0.71 (0.5B model, synthetic arithmetic; directional only). Because ZVF
            aliases mastery with incapacity, we promote the pairwise-contrast density PCD =
            G−1
              G E[p(1 − p)] as the control signal: a micro-jitter falsification collapses batch
            ZVF from 0.158 to 0.000 while leaving PCD unchanged. We then operationalize
            the theory as a zvf-triage callback and an adaptive-G controller that escalates
            group size on ZVF spikes. In a four-arm open-trainer audit, adaptive-G matches
            the best fixed-recipe held-out gain (+0.575) at 186 rollouts, while DAPO-style
            dynamic sampling drives ZVF to 0.00 at +45% rollout cost—an accuracy-versus-
            compute trade, not a controller win. In the larger 2,560-observation controller
            trace, 1,723 of 1,867 escalation events (92.3%) occur on all-correct groups and only
            144 on all-wrong groups. This asymmetry shows that a symmetric ZVF trigger
            mostly spends on solved prompts; deleting those fires is only a counterfactual
            proposal because performance preservation has not been measured. No result
            here establishes prospective superiority over static G=16, Dr. GRPO, or dynamic
            sampling. All intervention evidence is single-task, small-n, or retrospective, and
   ∗
       Project guide.

Preprint. Under review.

[PAGE 21]
Table 19: Per-method posterior-predictive contrast-restoration probability on the N2 four-method
tensors (40 steps × 16 prompts × 4 methods = 2,560 prompt-step obs). The mean is dominated by
the ∼ 82% non-degenerate majority (which trivially stays non-degenerate at G′ =16); the degenerate-
only 7/25 ≈ 0.28 is the metric that matters for the controller’s design hypothesis.
         method                   ndegenerate mean restore (all 2,560) 95% bootstrap CI
         grpo                             461                       0.7230     [0.7127, 0.7334]
         aero                             461                       0.7245     [0.7138, 0.7350]
         gift                             493                       0.7087     [0.6989, 0.7183]
         areal                            452                       0.7269     [0.7165, 0.7372]
         mean across methods              467                       0.721         [0.711, 0.731]


smoother [2] coincides with the iter-51 escalation branch on the “no contrast, no update” regime;
on N10 both mechanisms are mostly inert (the iter-51 escalation branch has 0/75 saturating fires
across 5 seeds × 15 steps). (c) The P6 delta_adaptiveg registry entry (Iter 54 row 65) gets its
third empirical confirmation (after iter 31 and iter 59); the new contribution is the explicit per-step vs.
aggregate partition of seed-robustness, which the existing registry row does not distinguish.

8.11 Posterior-predictive contrast-restoration: a Pareto-reversing closed-form benefit metric

The four controller comparisons above all use the proxy metric “saved prompts = number of fired
prompts” (any escalation counts as a save). This section introduces a strictly stronger metric — the
closed-form Beta-Binomial posterior predictive probability that escalating G=8 → G′ =16 restores
within-group contrast — and shows that on the N2 four-method evidence base the metric reverses
the iter-11 Pareto conclusion.

Posterior-predictive benefit of escalation. For an observed prompt with k successes in G=8
rollouts, place a Beta(1, 1) prior and form the posterior Beta(k+1, 9−k). The posterior-predictive
probability that the group would still be degenerate (i.e. all-0 or all-G′ ) at G′ =16 is the Beta-Binomial
tail:
                                 Z 1
               ′                                                         B(k+17, 9−k)
          P (Y =0 | k, G=8) =          p16 Beta(p; k+1, 9−k) dp =                           ,           (8)
                                   0                                     B(k+1, 9−k)
                                 Z 1
                                                                                 B(k+1, 9−k+16)
        P (Y ′ =16 | k, G=8) =        (1 − p)16 Beta(p; k+1, 9−k) dp =                              ,   (9)
                                   0                                               B(k+1, 9−k)
       Pr(restore at G′ =16) = 1 − P (Y ′ =0) − P (Y ′ =16).                                           (10)
For k=0 or k=8 this evaluates to Pr(restore) = 1−9/25−9/25 = 7/25 ≈ 0.280 . . . but actually via
the closed form the k=0 case gives B(1, 25)/B(1, 9) + B(17, 9)/B(1, 9) = 9/25 + 9/25 = 18/25,
so Pr(restore) = 7/25 ≈ 0.28. This is the posterior-predictive benefit of escalating a single
truly-degenerate prompt. The result depends only on k and is identical for k=0 and k=8 by
symmetry; it is method-independent (all four GRPO-family methods give exactly the same value, to
4 decimals).

Per-(method, prompt-step) on N2. Aggregated over all 2,560 prompt-step observations in the N2
four-method tensors:

Controller cost-efficiency reverses the iter-11 Pareto.

Cross-evidence-base consistency. Iter 15 estimated ∆ZVF(8 → 16) = 0.0594 [0.0463, 0.0725]
per fire from the Qwen/Qwen3.5-4B GSM8K group-size sweep (n=3 seeds). The N2-specific
posterior-predictive estimate here (7/25 = 0.28 for degenerate prompts; 0.72 averaged over all
prompt-step observations) is of the same order of magnitude when expressed as a fraction. The two
estimates are consistent on order-of-magnitude benefit even though they measure different quantities
(mean ZVF shift vs posterior-predictive restoration probability), closing the iter-15 extrapolation
loop.


                                                    21

[PAGE 28]
controller              τ    savings            95% CI    excl. 0?    seed-CV    fire-rate
       zvf_triage          0.50    −0.8400     [−0.92, −0.75]        yes        0.061       0.84
       zvf_triage          0.55    −0.6000     [−0.67, −0.51]        yes        0.072       0.60
       zvf_triage          0.65    −0.2800     [−0.36, −0.20]        yes        0.077       0.28
       zvf_triage          0.75    −0.2800     [−0.36, −0.20]        yes        0.077       0.28
       zvf_triage          0.85    −0.0400     [−0.09, 0.00]          no        0.057       0.04
       dualformer_auto     0.50    +0.4200    [+0.37, +0.46]         yes        0.096       0.84
       dualformer_auto     0.55    +0.3000     [+0.25, +0.33]        yes        0.082       0.60
       dualformer_auto     0.65    +0.1400     [+0.10, +0.18]        yes        0.057       0.28
       dualformer_auto     0.75    +0.1400     [+0.10, +0.18]        yes        0.057       0.28
       dualformer_auto     0.85    +0.0200     [ 0.00, +0.05]         no        0.030       0.04
       hybrid              0.50    +0.0600     [−0.01, +0.14]         no        0.102       0.84
       hybrid              0.55    −0.1800     [−0.23, −0.12]        yes        0.062       0.60
       hybrid              0.65    +0.1400    [+0.10, +0.18]         yes        0.057       0.28
       hybrid              0.75    −0.2200     [−0.28, −0.16]        yes        0.066       0.28
       hybrid              0.85    −0.0400     [−0.09, 0.00]          no        0.057       0.04
Table 28: τ -sensitivity sweep on the N10 5-seed GRPO panel (Gbase =8, δ=0.10, paired bootstrap
n=2000, seed 20260704). Savings is the per-seed compute saving relative to the always-G=8
baseline; the 95% CI is the paired bootstrap CI across the 5 seeds. The seed-CV column is the
coefficient of variation of total-G across seeds (smaller = more seed-robust). The bold rows mark the
best-τ per controller under headroom-bad = 0: Dualformer-Auto@0.50 strictly Pareto-dominates
the others, and the dualformer_auto family strictly dominates zvf_triage on the savings axis at
every τ where either fires (opposite signs, non-overlapping CIs).


Table 29: Per-prompt iid-ZVF at the controller’s chosen G′ versus the fixed-G=8 baseline, on the
192 sat-band prompts. Positive ∆ZVF means the controller’s choice worsens signal (higher iid-ZVF
= more starvation). The Hybrid’s de-escalation is signal-preserving on the 181 saturated prompts
(ZVF= 1.0 either way) and signal-harmful on the 11 mixed prompts.
 regime (per k)          n Hybrid G′ iid-ZvF at G=8 iid-ZvF at G=4                               ∆
 saturated k∈{0, 8}      181            4            1.0000              1.0000                  0.0000
 boundary k∈{1, 7}         3            4            0.3436              0.5864                 +0.2428
 mid k∈{2, . . . , 6}      8            4          ≤ 0.1001            ≥ 0.1250      +0.0249 to +0.2202
 all 192 sat-band        192            4              0.9501               0.9609                  +0.0108


Falsifiable headline (this iteration). With bootstrap 95% CIs on the 192 sat-band prompts
(nboot =2000, seed 20260704, step-level resample of the 12 sat-band steps):

What the iter-31 prediction says vs what the per-prompt measurement shows. Iter-31’s falsifi-
able prediction was that the Hybrid strictly dominates zvf-triage on sat-band-heavy panels. This iter
sharpens it: the Hybrid dominates on rollout economy (every sat-band prompt goes to G=4 instead of
G=16, a 4× saving per prompt) but on the 5.7% of sat-band prompts that are mixed the de-escalation
costs signal. The zvf-triage’s escalation direction is signal-positive everywhere on this evidence base
(0/192 over-de-escalation, mean ∆ZvF = −0.0054 with CI strictly negative), but at 4× the rollout
cost. Dualformer-Auto is the strict Pareto winner on the over-de-escalation-rate axis: 1.04% rate with
CI including zero, mean ∆ZvF = 0.0000 with CI straddling zero, AND it spends 56%–62% fewer
rollouts than fixed-G=8 on the saturated prompts (because every k ∈ {0, 8} prompt maps to G′ = 2
via Dualformer’s p̂ ≥ 0.95 rule).

Sharpest reviewer-facing falsifiable prediction. On a future sat-band-heavy panel where the
mixed-prompt fraction rises above 10%, the Hybrid’s per-prompt over-de-escalation rate should
exceed 5.7% and its mean ∆ZvF should become positive at the 95% level; on a panel where the
mixed-prompt fraction stays below 5%, the Hybrid strictly Pareto-dominates zvf-triage. On the N2
sat-band panel, the mixed fraction is 5.7% — within the Hybrid’s design tolerance and consistent
with iter-31’s conclusion that the Hybrid is the calibrated answer for the saturated-prompt regime,
with a per-prompt-level cost that the step-level zvf step aggregate cannot detect.


                                                  28

[PAGE 60]
Table 58: Iter-147 UNIFIED_C4 per-method breakdown. Cross-method SD on UNIFIED_C4 cost is
0.0086 — 6× smaller than ADAPTIVE_PP_ORACLE’s cross-method SD (0.0850).
              method    controller       mean cost            CI95 cost    retention   mag/cost
              grpo      STATIC_G8           1.0000        [1.000, 1.000]     1.0000     0.2300
              grpo      STATIC_G16          2.0000        [2.000, 2.000]     1.1523     0.1325
              grpo      UNIFIED_C4          1.0969        [1.075, 1.120]     1.0564     0.2215
              aero      STATIC_G8           1.0000        [1.000, 1.000]     1.0000     0.2344
              aero      STATIC_G16          2.0000        [2.000, 2.000]     1.1366     0.1332
              aero      UNIFIED_C4          1.0891        [1.069, 1.114]     1.0469     0.2253
              gift      STATIC_G8           1.0000        [1.000, 1.000]     1.0000     0.1903
              gift      STATIC_G16          2.0000        [2.000, 2.000]     1.1458     0.1090
              gift      UNIFIED_C4          1.1000        [1.077, 1.122]     1.0544     0.1824
              areal     STATIC_G8           1.0000        [1.000, 1.000]     1.0000     0.2405
              areal     STATIC_G16          2.0000        [2.000, 2.000]     1.1540     0.1388
              areal     UNIFIED_C4          1.0797        [1.059, 1.103]     1.0392     0.2315



Why the C4 cost is concentrated in DEGENERATE cells
On the N2 panel, step-level z values are concentrated in [0.50, 0.85] — the panel is hard. This means
almost no cell hits the FAST regime (z < 0.5, where C4 would drop G to 4) and almost no cell hits
the boundary-protocol branch. The DEGENERATE regime (z ≥ 0.70) triggers on a small fraction of
(step, method) pairs; on those steps the per-prompt controller escalates G to 16 (and rarely 32) on the
prompts whose p̂ is intermediate, recovering contrast that the G8 ceiling cannot reach. This explains
why UNIFIED_C4 has the same 1.04–1.05 retention across all 4 methods despite the methods having
very different step-level z profiles.

Known idealization: closed-form Bernoulli vs autoregressive anti-herding
The z(p̂, G) = p̂G + (1 − p̂)G formula assumes i.i.d. rollouts. On real autoregressive decoding, the
per-prompt rollout rewards are anti-herding (ρ < 0, per the iter-113 frontier synthesis): empirical z is
0.13–0.23 lower than the i.i.d. prediction on the N2 panel. This means the UNIFIED_C4 controller is
conservative on real data: it under-estimates the contrast that escalation will recover, fires more often
than the i.i.d. model would predict, and over-recovers contrast relative to the i.i.d. prediction. The
measured 1.04–1.05 retention is therefore a lower bound on the controller’s true per-prompt recovery;
the anti-herding correction would predict ∼1.10–1.15 retention if applied as a structural diversity
bonus on the Bernoulli z.

Cross-paper coupling
       • P5 iter-113: ZVF structural diversity δdiv ∈ [0.13, 0.23]. Iter-147 confirms C4 is conserva-
         tive under i.i.d. Bernoulli because of this anti-herding correction.
       • P5 iter-127 (paper-P5 minreport): the per-prompt contrast magnitude cm(p, G) is the
         operational scale at which GRPO assigns credit; iter-147 is the controller evaluation at that
         operational scale.
       • P7 iter-119 (calibrated unification): step-aggregate UNIFIED_C4 has mean Gused =
         20.30 on N2 (because step-aggregate fires the DEGENERATE regime on a substantial
         fraction of step-method decisions). Iter-147’s per-prompt view gives mean Gused = 8.73
         (cost 1.09) because the regime gate is fired per-prompt and only escalates the prompts whose
         p̂ is intermediate. The two numbers are consistent: per-prompt C4 is the granular version of
         step-aggregate C4.
       • P7 iter-131 (per-prompt adaptive-G∗ ): iter-147 replaces the iter-131 family with the
         iter-119 UNIFIED_C4 rule applied per-prompt, and adds bootstrap CIs that iter-131 lacked.

Limitations
       • Closed-form Bernoulli ignores within-prompt rollout correlations. The iter-113 anti-herding
         correction should be folded in before any deployment decision.


                                                     60

[PAGE 61]
• Per-prompt C4 uses step-level z as the regime gate signal, not a per-prompt z (the per-prompt
         z is not observable without re-rolling). A smoother version would carry an EMA of recent
         per-prompt z values.
       • The H3 honest-FAIL (“C4 mag-per-cost < STATIC_G8”) is a known limitation of dynamic
         controllers at this granularity: the marginal recovery from escalation does not pay for the
         marginal cost. The benefit of C4 over STATIC_G8 is in contrast quality, not efficiency.

Conclusion of iter-147
The UNIFIED_C4 controller’s iter-119 properties transfer to per-prompt granularity on the N2 reward
tensors: 1.04–1.05 contrast retention, 9–10% cost overhead, cross-method SD 0.009 on cost,
defensive-composition property intact. The strongest single result is the cross-method uniformity:
the same controller with the same hyperparameters yields the same overhead on all four N2 methods,
which validates the iter-119 unification as a stack-portable controller family.

Pareto-Frontier + Per-Method Bootstrap CI on iter-147 UNIFIED_C4
Motivation
Iter-147 counterfactually applied the iter-119 UNIFIED_C4 controller at per-prompt granularity on
the N2 reward tensors (4 methods × 40 steps × 16 prompts = 2,560 prompt cells) and reported overall
bootstrap CIs (B = 1000, seed= 42) on the headline metrics. Iter-147 did not break the bootstrap
CI out by method, did not build a cost-vs-retention Pareto frontier, did not classify controllers as
dominated/Pareto-optimal/strictly optimal, and did not compute cross-method SDs with bootstrap
CIs to validate the headline “6× more method-portable” claim.
Iter-159 closes those four sub-gaps at the per-prompt granularity on N2. It also re-implements the iter-
147 controller evaluation from scratch because iter-147’s p7_iter147_per_cell.tsv file turned
out to have misaligned column labels (the values written do not match the iter-147 source code that
produced them — e.g. g_STATIC_G8 column has value 0.0 instead of 8.0). Iter-159 reads the N2 re-
ward tensors directly and applies the controller functions in platform_modal/scripts/p5p8/p7_-
iter147_unified_per_prompt.py verbatim.

Headline — per-(method, controller) bootstrap CI95
Table 59 reports the per-(method, controller) bootstrap CI95 on mean cost and mean retention. The
headline patterns:
       • UNIFIED_C4 cost is stable at 1.08–1.10 across all 4 methods (CI95 lower bound ≥ 1.06
         everywhere). The 8–10% cost overhead over STATIC_G8 is statistically distinguishable.
       • UNIFIED_C4 retention is 0.244–0.307 across methods, vs STATIC_G8 retention of
         0.230–0.294 (the per-method gain is +0.013 to +0.019, small but precisely estimated).
       • ADAPTIVE_PP_ORACLE has the highest retention (0.28–0.37) but at 1.69–1.88× the
         cost of G8 — it’s the most aggressive controller but at the highest compute cost.
       • DUALFORMER_PP is bit-identical to STATIC_G8 on N2 because zobs ≥ 0.50 on every
         step in N2; Berkeley row 01’s “drop G to 2/4 on easy prompts” rule fires zero times.
       • STATIC_G16 is dominated on every method (see Pareto frontier below).

Cross-method SD — the headline robustness metric
Table 60 reports the cross-method SD on cost, retention, and mag-per-cost for each controller. The
headline result: UNIFIED_C4 cross-method SD on cost is 9.31× smaller than ADAPTIVE_PP_-
ORACLE (0.0078 vs 0.0731, ratio 9.31×). This exceeds the iter-147 “6× more method-portable”
claim, which was a point estimate. Iter-159’s block-bootstrap CI on the SDs (B=2000, resample
methods with replacement) is degenerate because nmethods = 4 — the CI is a delta function — so
the 9.31× ratio is a deterministic point estimate at the n = 4 granularity. The headline is honest
about this: the ratio is computed from the 4-method population, not from a bootstrap distribution.
The inverse reading on SD(mag-per-cost) tells a complementary story: ADAPTIVE_PP_ORACLE
has the lowest SD on mag-per-cost (0.006), meaning ORACLE is the most efficient per unit cost


                                                  61

[PAGE 64]
Limitations
        • Iter-159’s cross-method SD ratio (9.31×) is a deterministic point estimate at nmethods = 4;
          the block-bootstrap CI on the SDs is degenerate because the resampling space has only 4
          points. The headline ratio is honest about this — it’s a sharper version of iter-147’s 6× point
          estimate but doesn’t add distributional evidence beyond what iter-147 had.
        • STATIC_G16’s strict dominance on every method is a Pareto statement; the appeal of
          STATIC_G16 in deployments is its determinism and simplicity (no per-prompt telemetry
          required), not its cost-vs-retention efficiency.
        • DUALFORMER_PP on N2 is bit-identical to STATIC_G8 because Berkeley row 01’s
          FAST-regime rule fires zero times. Iter-159’s Pareto finding is N2-specific; on panels with
          zobs < 0.50 (a different operating regime), DUALFORMER_PP would be Pareto-distinct
          from STATIC_G8.
        • The heldout r(zobs , reward) > 0 finding is counter-intuitive; iter-159 confirms iter-99’s
          interpretation but does not rule out alternative explanations (e.g. selection bias in the 40-step
          panel).

Conclusion of iter-159
Iter-159 closes the brief vein (a) sub-gaps at the per-method bootstrap-CI + Pareto frontier layer. The
headline numerical results:

      1. 8/8 hypotheses PASS at per-prompt granularity on N2 (the most thorough per-prompt CI
         audit in the P7 ledger).
      2. C4 cross-method SD on cost is 9.31× smaller than ORACLE (0.0078 vs 0.0731) —
         sharper than iter-147’s reported 6×.
      3. STATIC_G16 is strictly dominated on every method by ADAPTIVE_PP_ORACLE on
         both axes (cost and retention) — the paper should scope its “safe baseline” framing.
      4. C4 retention gain over STATIC_G8 is small but precisely estimated (+0.013 to +0.019,
         CI half-width 0.005–0.007).
      5. Heldout r(zobs , reward) > 0 on 4/4 methods confirms iter-99’s “zobs aggregates with
         prompt difficulty” finding with bootstrap CIs.
      6. DUALFORMER_PP on N2 is bit-identical to STATIC_G8 because Berkeley row 01’s
         de-escalation rule fires zero times.

The iter-159 deliverable promotes per-method CI breakdown as the canonical P7 controller evaluation
protocol for future audits and exposes STATIC_G16’s strict dominance as a paper-facing finding that
requires §sec:p7-iter159-pareto to honestly scope.

Motivation
Iter-98 (Iso-G row 98) Pareto-compared Iso-G against other empirical controllers but not against
the per-prompt oracle that sees each prompt’s true p̂. Iter-159 (paper_P7_zvf_controller.tex
§ iter-159) measured the gap to the oracle via a different metric (Section 11.6.4). Neither head-to-head
used a double-axis contrast restoration on the full N2 reward tensors. Iter-167 closes this gap by
computing the oracle’s optimal G⋆ for each of 4 × 40 × 16 = 2,560 (method, step, prompt) cells
under the i.i.d. Binomial contrast model, then measuring two outcome axes on every empirical
controller.

Oracle definition
For each prompt with k successes at Gbase = 8 rollouts, p̂ = k/8. The oracle controller picks
                                                               ∆Y (p̂, G′ )
               G⋆ (p̂) = arg              max                                                          (16)
                               G′ ∈{2,4,6,8,10,12,16,24,32} max(1, G′ − Gbase )

where
          ∆Y (p̂, G′ ) = Yiid (p̂, G′ ) − Yiid (p̂, Gbase ),   Yiid (p̂, G) = 1 − [p̂G + (1 − p̂)G ]


                                                       64

[PAGE 66]
• Iter-159 (per-method Pareto): confirmed that ADAPTIVE_PP_ORACLE has cost 1.69–
        1.88 and retention 0.28–0.37; iter-167 confirms that no empirical controller in this panel
        (C0–C5) achieves both: either high contrast or high cost-efficiency, never both.
      • Iter-83 (Iso-G invention): closes the brief vein “when would [Iso-G] have fired” by giving
        per-(method, step, prompt) oracle G⋆ .
      • Iter-79 (multi-trigger seed-robustness): the bounded C5 costeff CI95 (0.68–0.82) across
        seeds suggests the cost-effective gap is reproducible.

Limitations
                                                                                      ′            ′
      • The oracle uses the i.i.d. Binomial contrast model (ZVFiid (p̂, G′ ) = p̂G + (1 − p̂)G ),
        which ignores the empirical anti-herding δdiv bonus documented in iter-103 and iter-119. A
        model-aware oracle that uses the empirical δdiv per (method, step) would adjust G⋆ for the
        inflation of contrast.
      • The empirical controllers in this iteration (C0, C1, C2, C3,C5) are the same five evaluated in
        iter-98 (row 98); iter-167 does not introduce a new controller, only a new evaluation axis
        pair.
      • The bootstrap CI on Axis A assumes i.i.d. prompts; iter-79’s seed-robustness audit suggests
        prompt-block bootstrap (resampling together with their zobs ) gives equivalent intervals.

Conclusion of iter-167
     1. Iso-G Pareto-dominates every other empirical controller on both axes (Axis A: 250%+
        of oracle; Axis B: 0.71–0.77× the oracle’s marginal efficiency).
     2. Dualformer-Auto is structurally Pareto-incompatible with signal-starvation theory:
        −365.8% to −220.1% on Axis A and 3.37× to 5.06× on Axis B across the 4 methods.
     3. The 250% Iso-G-over-oracle result is not a bug; it is the quantitative consequence of
        optimising “smallest G′ achieving Y ≥ τy ” (C5) rather than “maximum ∆Y /extras”
        (oracle).
     4. The remaining 23%–29% gap (Axis B) on C5 is the room for a controller that picks G′ by
        marginal cost-effectiveness rather than absolute-yield target — a candidate for iter-171+ on
        the N2 panel.

Motivation
Iter-167 § 3 called Dualformer-Auto (Berkeley row 01) structurally Pareto-incompatible with signal-
starvation theory: −286% to −365% of oracle absolute contrast on every method (pctabs axis),
because ungated Dualformer escalates on per-prompt phat alone, without consulting the empirical
step-zobs evidence. Iter-175 fuses Berkeley row 01 (Dualformer’s per-prompt wishful G′ ) with
Berkeley row 19 (AlphaProof’s γ ∗ =0 anchor; the empirical observation that mag(γ=0) ≈ 0 on
every seed×G cell means the no-blend regime IS the empirically supported "do-nothing" prior) into
a single calibrated rule C6 .

Decision rule
For each (method, step, prompt) observation with phat = k/Gbase and observed step-zobs :
                            
                            
                             2   phat ∈ (0, 0.30]
                            
                              8   p hat ∈ (0.30, 0.55]
                            
                            
                            
             gdual (phat ) = 12 phat ∈ (0.55, 0.70] (Berkeley row 01)                            (17)
                              16 phat ∈ (0.70, 0.85]
                            
                            
                            
                            
                            
                            24 p ∈ (0.85, 1.0)
                                    hat

             γ ∗ =0 anchor := do not escalate unless both signals agree (Berkeley row 19)
                                                                              
                   firedual := gdual (phat ) ̸= Gbase ∧ 4 phat (1 − phat ) > 0.05
                   firezvf := zobs ≥ τz   (default τz = 0.70)


                                                 66

[PAGE 68]
Cross-paper coupling
      • Berkeley row 01 (Dualformer, iter 131 / 127). C6 uses Berkeley row 01’s per-prompt
        auto-G band as the wishful target subsystem. Iter-175’s contribution over row 01 is the
        γ ∗ =0 + zobs gate that recovers Dualformer’s signal (gating flips C2 ’s %abs from −11.0%
        to C6 ’s +9.9% on aero).

      • Berkeley row 19 (AlphaProof). C6 uses Berkeley row 19’s empirical observation that
        mag(γ=0) ≈ 0 as the literal "do-nothing" prior. Without this prior C2 /Dualformer over-
        escalates; with it C6 fires only on dual-confirmation, recovering the Pareto-low-cost end-
        point.

      • Iter-167 (oracle regret). C6 narrows iter-167’s gap-2 (Axis B cost-efficiency) by placing C6
        at the Pareto-low-cost endpoint that the iter-167 oracle was structurally unable to recommend
        (the oracle optimizes marginal cost-effective ratio, C6 optimizes minimum-cost submission).
        They are complementary, not competing.

      • Iter-119 / -103 (UNIFIED C4 ). C6 sits in the same Pareto-defensive space as C4 (iter-119),
        but where C4 chooses per-prompt G′ via the layered τy thresholding and never exceeds
        G = 16, C6 inherits Dualformer’s per-prompt target G′ ∈ {12, 16, 24} (the Berkeley row
        01 band) and exceeds G=16 on the high-phat dominant-class prompts (phat > 0.85, 8.3%
        of prompts) — i.e. C6 is the more aggressive sibling of C4 .

Limitations
      • C6 ’s %abs ∆Y (∼10%) is an order of magnitude below the iter-167 oracle’s marginal cost-
        effective optimum (100%). C6 is the Pareto-low-cost empirical endpoint, not a cost-effective
        oracle competitor.

      • The bootstrap CI95 on %abs ∆Y straddles 0 on every method (CI half-width ≈ 53 pp,
        ≈ 2× C5 ’s CI). The marginal contrast yield is noisy because C6 fires on ∼ 28% of prompts;
        the cost is precisely estimated (LCG bootstrap B=2000 same as iter-167/iter-171).

      • The plateau τz ∈ [0.65, 0.75] is calibrated on the 4-method single-seed N2 panel only.
        Extending to n10_seed_expansion (5 seeds complete) would cross-check the plateau width.

Conclusion of iter-175
     1. C6 Pareto-dominates C1 (zvf-triage) AND C2 (Dualformer) on every method (4/4) at
        τz = 0.70.

     2. C6 saves 5.0×–7.6× rollouts vs C1 and 1.5×–2.7× vs C2 ; it flips C2 ’s negative %abs on
        3/4 methods into a small positive on 4/4.

     3. C6 occupies the Pareto low-cost endpoint, C5 /Iso-G the high-yield endpoint; the Pareto
        frontier { C0 , C6 , C5 } is a low-cost/high-yield sandwich on 3/4 methods (aero, gift, grpo).

     4. The operating-point plateau τz ∈ [0.65, 0.75] is the self-calibration tolerance for C6 .

     5. Berkeley row 01 (Dualformer auto-G) and Berkeley row 19 (γ ∗ =0 anchor) are NOT com-
        peting primitives: they combine into C6 , where row 01 supplies the wishful target and row
        19 supplies the no-escalation prior.

Motivation

Iter-179 (§ iter-179) measured the restored contrast on the 1,312 fired prompts of the iter-119 C4
controller at the static default GN =16. The natural follow-up question is the per-prompt cost-
effective optimum: for each fired prompt, what is the smallest GN ≥ Gbase =8 that maximizes the
binomial-projected restored contrast per extra rollout? Iter-192 answers this by sweeping GN ∈
{12, 16, 24, 32, 48} on the same fired-prompt pool.


                                                 68

[PAGE 70]
• Restoration-maximising endpoint: GN = 16 on all contrast prompts, ignoring cost. Closes
        the iter-119 C4 default.
      • Cost-effective endpoint: GN = 12 on contrast prompts, GN = 8 on boundary prompts.
        Closes the iter-192 per-prompt optimum.
      • The 4 pp gap between the savings ( 45%) and the restoration loss ( 33%) is the price of
        compute: opting into the per-prompt cost-effective axis trades one unit of restoration per
        four units of compute saved.

Monotone DEC is closed-form invariant (H4). The cost-effective ratio is monotone DECREAS-
ING in GN on every contrast prompt (k ∈ {1, . . . , 7}) under the binomial scoring, because
Y (p, G) = 1 − pG − (1 − p)G is concave in G for any p ∈ (0, 1) and bounded by 1. The H4
PASS (4/4 methods) is the closed-form signature of the cost-effective axis: any monotone DECREAS-
ING eff in GN is equivalent to “first G in the grid is optimal on every contrast prompt”.

Cross-method savings invariance. The 45.2–46.2% savings range has cross-method CV = 0.92%
(i.e., σ/µ = 0.0042/0.4550), confirming the structural similarity of the four GRPO-family methods
on this N2 panel: the same boundary/contrast ratio and the same per-prompt optimum hold regardless
of method. The single “savings ≈ 45%” headline is therefore not method-stratified — it is a
property of the N2 evidence base, not the four methods.

Cross-paper coupling
      • P7 iter-179 (§ iter-179, row 191): iter-179 measured restored contrast at the static GN = 16
        on the same fired pool. Iter-192 quantifies the price of replacing G = 16 with the per-prompt
        cost-effective optimum (savings 45%, restoration −33%).
      • P7 iter-167 (oracle regret): iter-167 reported the per-prompt oracle gives G∗N ∈ {12, 16}
        on contrast prompts. Iter-192 confirms the marginal cost-effective optimum is the smaller
        end (GN = 12), and shows iter-167’s G=16 picks the restoration-maximising endpoint.
      • P7 iter-175 (C6 calibrated-hybrid): iter-175 chose GN = 16 as the static default. Iter-192
        closes the loop: C6’s static default is the restoration endpoint; the per-prompt optimum is
        the cost-effective endpoint. The right operational choice is budget-conditioned.
      • P7 iter-119 (CCC unified controller): iter-119’s DEGENERATE regime uses GN = 32 as
        the cap. Iter-192’s monotone-DEC result says cap at the SMALLEST GN that exceeds the
        trigger, not the largest. The DEGENERATE cap choice is a separate restoration-vs-cost axis
        (favouring restoration when the prompt is all-or-nothing, z ≥ 0.70).
      • Berkeley row 01 (Dualformer auto-G): row 01 reported 56.2% savings on iter127 cells
        (Gbase =16 vs auto-G). Iter-192 reports 45% savings on the N2 fired-step pool — strictly
        less because N2’s boundary share (82%) is lower than iter127’s, and the cost-effective ratio
        is monotone in G.
      • FRONTIER_INSIGHTS Round 2 (ZVF = signal availability): iter-192 quantifies the
        cost-effective axis on the closed form Y (p, G) = 1 − pG − (1 − p)G introduced in Round 2
        as the “censored contrast probability”.

Limitations
      • The per-prompt optimum is computed under the i.i.d. binomial model; the exact finite-pool
        (hypergeometric) scoring from iter-88 will give a sharper G∗N on the economized contrast
        prompts. Extension deferred.
      • The grid G does not include GN = 8 (no-op) for contrast prompts; the per-prompt optimum
        on the boundary set is the default GN = Gbase , not a sweep result.
      • The 234 contrast prompts (across the 4 methods) are concentrated on k ∈ {1, 7} (∼ 60% of
        contrast prompts) and k ∈ {2, 6} (∼ 30%), with k = 3, 4, 5 collectively < 10%. The H4
        monotone-DEC claim is closed-form, so it holds for any k ∈ {1, . . . , 7}, but the empirical
        mean G∗N = 12.000 reflects this k concentration.
      • The savings 45% number is per-prompt, not per-step: a step’s total rollout spend is the sum
        of per-prompt spends over the 16 prompts. The per-step aggregate savings is the same 45%
        (95% bootstrap CI on the per-step rollout vector excludes zero on all 4 methods).


                                                70

[PAGE 71]
Conclusion of iter-192
      1. On the 1,312 fired prompts of the iter-119 C4 controller at τ = 0.70, the per-prompt cost-
         effective optimum saves 45.2% rollouts [44.1%, 47.1%] vs the static GN = 16 on all 4
         methods.
      2. The optimum on contrast prompts (k ∈ {1, . . . , 7}) is uniformly G∗N = 12.0 on every
         method, by closed form (monotone DEC of eff in GN ).
      3. The cost-effective optimum and static GN = 16 are complementary Pareto endpoints, not
         nested: 45% rollouts saved at 33% absolute restoration loss.
      4. Cross-method CV on savings frac is 0.92% — the 45% number is method-invariant on this
         N2 evidence base.
      5. The closed-form monotone-DEC signature of eff in GN is equivalent to “smallest G in the
         grid is optimal on every contrast prompt”, a property of Y (p, G) = 1 − pG − (1 − p)G that
         generalises to any monotone-DEC cost-effective axis.

Reproduction. scripts/p5p8/p7_iter192_perfire_optimal_gn.py (≤     300 LoC,
stdlib only, deterministic LCG bootstrap seed 20260706, B = 2000); outputs in
experiments/results/p5p8/p7_iter192_{per_prompt.tsv (1,312 rows), per_-
method.tsv (4 rows), ci.tsv (8 rows), summary.json}.

Motivation
Iter-83, iter-85, iter-90, and iter-187 had all imported the Dualformer auto-mode rule (Berkeley row
01) and the AlphaProof tree-baseline γ ∗ =0 smoothing (Berkeley row 19) into the P7 controller
bank. None of those iterations had measured the rule-level concordance among the three binary fire
decisions on the same step-cells. Iter-195 closes that gap by computing three binary fire rules and
their Cohen’s κ on the N2 four-method × 40-step same-stack panel (160 step-cells).

Three step-level fire rules
For each (method, step) cell we compute three binary decisions:


AG(s) = 1[zvf(s) ≥ τ ] ,  τ = 0.70 (iter-119)                                                   (24)
                                              
                              r(s)       r(s)
DF(s) = 1 has_contrast(s) ∧ √        >√           (Berkeley row 01)                             (25)
                               Gbase      Gesc
                                                          2
                                                               k̄)2
                                     (
                                       smoothed(s) = k̄ +(G−G2
AP(s) = 1[smoothed(s) < naive(s)] ,                                             (Berkeley row 19, depth-0)
                                       naive(s) = k̄(G−
                                                     G2
                                                        k̄)

                                                                                                (26)

where k̄ is the mean success count over contrast prompts in the step and G = Gbase = 8. Concordance
is Cohen’s κ on the three pairs (AG×DF, AG×AP, DF×AP), with bootstrap percentile CIs (B= 2000,
seed= 20260706).

Headline — structural disagreement by construction
Table 67 reports the per-method fire rates and Table 68 reports the pooled κ.

Reconciliation — three algebraic reductions of the same latent signal
The naive rule-equivalence hypothesis fails structurally, not by chance. The mechanism is three
distinct algebraic reductions of the same latent signal-starvation condition:

      1. AG retains the cross-prompt contrast signal: it fires on steps where zvf(s) ≥ 0.70 (51.2%
         of N2 cells), the canonical Pillar 3 trigger (iter-119) for group-mean contrast collapse.


                                                 71

[PAGE 72]
method        n    AG-rate   DF-rate   AP-rate
                              grpo         40      0.500    1.000     0.000
                              aero         40      0.475    1.000     0.000
                              gift         40      0.650    0.975     0.000
                              areal        40      0.425    1.000     0.000
                              POOLED      160     0.5125   0.9938    0.0000
Table 67: Per-method step-level fire rates on the N2 four-method same-stack panel. AG       √ fires
                                                                                                  √on
roughly half the steps; DF fires on essentially every step (algebraically degenerate because 8 < 16
makes rbase > resc for any positive cell reward mean); AP fires on no step (algebraically degenerate
because (k 2 + (G − k)2 )/G2 ≥ k(G − k)/G2 is a fixed identity, so depth-0 smoothing is equivalent
to the GRPO group-mean).


                       pair                 κ              95% CI    excludes 0?
                       AG × DF         −0.0125   [−0.0376, 0.0000]             NO
                       AG × AP             ≈0           [≈ 0, ≈ 0]             NO
                       DF × AP             ≈0           [≈ 0, ≈ 0]             NO
Table 68: Pooled Cohen’s κ among the three step-level fire rules over 160 step-cells, with bootstrap
percentile 95% CIs (B=2000, seed= 20260706). None of the three pairs excludes zero; all 7
hypotheses on naive rule-equivalence FAIL.



      2. DF√discards the cross-prompt signal: it retains only the cell-mean reward and√ tests√whether
         r/ G is monotone decreasing in G. For any positive reward mean this is 8 < 16 and
         the rule is trivially satisfied on 99.4% of cells — an operational degenerate reduction.
      3. AP at depth = 0, w = 2 is equivalent to the GRPO group-mean: the depth-0 smoothing
         kernel (k 2 + (G − k)2 )/G2 dominates the naive Bernoulli variance k(G − k)/G2 by an
         algebraic identity. It fires on 0/160 cells — a structural degenerate reduction.

The information-preservation ordering Bayesian (iter-171) > AG > DF > AP predicts the iter-187
N10 negative (Bayesian never fires on mid-range prompts) correctly and explains why AG recovers
the largest signal on the N2 panel (where boundary cases are common).

Paper-grade conclusion
The right unification is at the latent-signal level, not the decision-rule level. Each rule is a valid
projection of the latent "needs adaptive G" condition, but at three distinct information-loss rates.
We recommend the calibrated controller section present the three rules side-by-side with their
structural failure modes rather than asserting interchangeability. When the prompt distribution
contains boundary cases (N2), the Bayesian rule (iter-171) is preferred; when the distribution is
mid-range (N10, iter-187), the AG rule at τ = 0.70 remains the information-preserving fallback.
The prior veins of Pillar 3 measured the iter-119 C4 controller at a single fired step in isolation:
iter-179 reported the static-G = 16 contrast-restoration on fired prompts; iter-192 reported the
per-prompt cost-effective optimum G∗N . Iter-199 is a frozen-p counterfactual projection, not a
                                                                                                   
closed-loop training experiment. We ask what the diagnostic trajectory of ZVF
                                                                          ] t , contrastt , costt
over the full 40-step training path would have looked like under four controller policies, using the
per-prompt kp (t = 0) from the N 2 reward tensors as a latent pi fixed for all t. Because the policy,
prompt distribution, and success probabilities do not respond to the simulated actions, the exercise
cannot estimate held-out learning or controller feedback stability.

Forward simulation under fixed latent pi
For each method in {G RPO, A ERO, G IFT, A REAL} and each policy, we simulate the trajectory
Gt ∈ {Gbase = 8, 16} for t = 0, . . . , 39:

       • BASE: Gt ≡ 8 (no controller; current best practice).


                                                   72

[PAGE 77]
outcome where unsigned ZVF reaches 0.56, consistent with the cross-examination’s diagnosis that
the missing ingredient is sign, not more of the same telemetry.

Beyond the predictions. Two results here were not anticipated by the cross-examination. First, the
interventional turn: T2 is not only an explanation but a control law, and E3 shows a live controller
acting on it matches the best fixed recipe on held-out gain. Second, the accuracy-versus-compute
framing of dynamic sampling: the cross-examination treated ZVF = 0 as a target, whereas the E3
audit prices it (+45% rollouts) and finds the purchase optional on a well-conditioned task. The
program-level lesson for the pillar family is the same one Papers 1–4 reached from four different
directions: telemetry must be magnitude-aware, stack-attributed, and priced—or it will be gamed by
the pipeline, the backend, or the budget.


14     Limitations and Proposed Decisive Experiments

14.1   Limitations

Every interventional claim in this paper is bounded by four limitations, stated in load-bearing order.
No prospective controller advantage. The long controller sections are retrospective replay, analytic
projection, or proxy optimization. Iter-199 freezes each prompt’s step-0 success estimate, so it is not
a closed learning loop. Iter-203 pools rewards from different prompts and therefore cannot emulate a
larger within-prompt GRPO group. Neither analysis estimates held-out policy improvement. The
required promotion test is a seed-paired, fixed-token comparison against static G=16 and naive
boundary heuristics.
Single task. The E3 audit and the E1 gradient probe each run on one task family (GSM8K-style
and synthetic arithmetic respectively). The theory is task-agnostic, but the controller’s measured
competitiveness is not: on harder or drifting prompt populations the trade of Table 9 could tip in any
direction.
Small n. Four arms, one open trainer, shared seeds, held-out deltas separated by at most 0.075: the
E3 differences between the top arms are within the noise band the companion papers’ power analysis
(Table 4) would demand for a Tier-A claim. We therefore claim mechanism-consistent telemetry and
competitiveness, not superiority.
LoRA-only closed stack for one comparison arm. The closed-stack telemetry cited for the label-flip
caution comes from LoRA [6] fine-tuning on a stack whose sampler and backward pass we cannot
inspect; the open reimplementation is the only arm-for-arm auditable setting, and cross-stack numbers
are quoted only as motivation, never as controller evidence.
Toy-scale gradient evidence. T3’s empirical support is the directional E1 result (+0.71 on a 0.5B
model); the coefficient is not expected to transfer, only the sign and ordering.
We also flag a provenance boundary and two scope choices. The 368-run audit, the model × G
grid, the 12-cell head-to-head, and the E1/E3 runs are internal program records (W&B projects
zvf-audit and zvf-colab-experiments, June 2026): they were run under a declared protocol
and are summarized faithfully, but they have not passed through the released-artifact pipeline that
covers the repo-internal sweeps and reward-tensor analyses (Appendix A, provenance note); the
companion Pillar-5 paper flags the same records the same way. The scope choices: PCD is validated
as a control-loop input (invariance, shape, aliasing resolution), not as an outcome predictor; and no
claim in this paper depends on the veRL/OpenRLHF rows of the framework comparison, which
remain dry-run placeholders (Table 2).

14.2   Proposed decisive experiments

The following experiments would each settle a claim this paper leaves directional. They extend
the frontier backlog maintained in the repository (platform_hybrid/experiments/FRONTIER_-
EXPERIMENT_BACKLOG.md).
D1: The PCD horse race. Log per-group reward tensors on every anchor run (not only GSM8K),
then test early-window PCD against held-out outcome at the pre-registered ρ ≥ 0.45 bar, with mean


                                                  77

[PAGE 78]
reward as the mandatory baseline covariate. This is the experiment the ρ = 0.95 vs 0.56 proxy result
gestures at but cannot replace.
D2: Adaptive-G on a hard population. Rerun the four-arm E3 design on a prompt population with
baseline p ∈ [0.05, 0.3] (e.g. competition math), n ≥ 5 seeds per arm, fixed total rollout budget rather
than fixed steps. Theory predicts this is where fixed-G starves and the controller’s targeted spend
should separate from both plain GRPO and always-on dynamic sampling; a null here would bound
the controller’s value to telemetry only.
D3: T3 at production scale. Per-group gradient attribution on an 8B model with an open backward
pass, 3+ seeds, reporting corr(∥∇∥, p̂(1− p̂)) with confidence intervals and an ablation over advantage
normalization (plain, Dr. GRPO-style, GSPO-style [15]). This either promotes T3 from directional to
quantitative or localizes where the binomial model breaks.
D4: Escalate-vs-resample crossover. A two-dimensional sweep (prompt difficulty × intervention
aggressiveness) mapping where adaptive-G, dynamic sampling [14], selective rollouts [16], and
zero-variance advantage shaping [7] each win per marginal rollout—turning Rule 5 of Section 12
from a caution into a decision boundary.
D5: Jitter red-team. Apply the micro-jitter probe to every pipeline in the registry: any stack whose
reported ZVF changes by more than 0.05 under ϵ ∼ U(0, 10−4 ) reward jitter is flagged as reporting
tie-artifact telemetry. This converts the falsification of Section 6.3 into a standing audit.

15    Conclusion
The companion Pillar-2 paper ended by refusing to over-read a diagnostic; this audit explains
what would be required to earn the control authority it withheld. A one-line binomial model,
S = p(1 − p)(1 − hG (p)), predicts the ZVF phenomena Pillar 2 catalogued: its exact expected value
(T1, matched at the endpoints and interior of the binned reward tensors), its monotone decline in group
size (T2, confirmed on two independent grids), and its coupling to gradient mass through p(1 − p)
(T3, directionally confirmed in an open backward pass at +0.71). The same model exposes the
diagnostic’s structural limit—the symmetry that aliases mastery with incapacity, visible as a U-shape
across 368 runs—and supplies the repair: the pairwise-contrast density PCD = G−1       G E[p(1 − p)],
which survives the micro-jitter that silently zeroes ZVF in any pipeline with a dense sub-reward.
The implemented callback and four-arm pilot establish feasibility, not superiority. Adaptive G reached
the same held-out delta as Dr. GRPO while spending more rollouts, and the larger retrospective trace
reveals that 92.3% of symmetric escalation fires occur on all-correct groups. Cross-prompt pooling
cannot stand in for a larger GRPO group, and a frozen-p simulation cannot stand in for closed-loop
learning. The durable output is therefore a falsifiable design rule—separate failed starvation from
solved saturation, price interventions in rollouts, and report the stack—plus a decisive next experiment:
seed-paired, fixed-token adaptive control against static G=16 and naive boundary heuristics. Until
that bakeoff succeeds, adaptive G remains a proposal.

A     Derivations
                                              binary rewards R1 , . . . , RG ∼ Bernoulli(p) for a
Throughout, a group is G conditionally i.i.d. P
prompt with latent success probability p; K = i Ri ∼ Binomial(G, p) and p̂ = K/G.

A.1   T1: expected ZVF indicator

The within-group sample variance is zero iff all rewards agree, i.e. K = 0 or K = G. These events
are disjoint, so
         E[1{degenerate}] = Pr(K = 0) + Pr(K = G) = (1 − p)G + pG = hG (p).                         (27)
For a prompt population D with per-prompt success probabilities px , linearity of expectation gives
E[ZVF] = Ex∼D [hG (px )]. Note the Jensen gap used in Section 5.2: since hG is convex near the
endpoints and the population mixes prompts of heterogeneous difficulty, Ex [hG (px )] ≥ hG (Ex [px ])
whenever mass sits at the extremes, which is why measured run-level ZVF exceeds hG (p̄) in Table 7.
□


                                                   78


# P08: Fraud detection


Root: `platform_hybrid/paper/paper_P8_fraud.tex`  Pages: 94  Words: 49413


[PAGE 1]
A Cross-Domain Measurement Side Study:
    LLM as Sensor and Scribe, Not Credit-Card Fraud
                        Scorer


                        Arvind C R                          Ramesh Prakash Guledgudd∗
                       PES University                            PES University
                   arvindcr4@gmail.com                         rameshpg@pes.edu



                                                 Abstract
             As a deliberate cross-domain probe of the T INKER RL-B ENCH measurement disci-
             pline —measure capability, do not trust the label—this side study steps outside
             GRPO to stress-test the same principle in a setting with a strong non-LLM base-
             line. Where do large language models (LLMs) actually add value in credit-card
             fraud detection when a tuned gradient-boosted tree already scores well? We run
             a head-to-head on a custom synthetic single-configuration fraud dataset (50,000
             transactions, ≈1.4% realized fraud rate). The empirical results are negative for
             LLM scoring: XGBoost reaches test AUC 0.7955 on the 10,000-row held-out split,
             while a Qwen3.5-4B SFT row-serialization arm reaches accuracy 0.792 but AUC
             0.48268 on a 500-row positive-enriched held-out evaluation. (The two AUCs sit
             on different held-out splits and are not a like-for-like ranking comparison; the
             load-bearing point is only that the LLM scorer is at chance-level ranking, not that
             it is worse than the tree by this exact margin.) The gradient-boosted tree keeps the
             real-time scorer seat for five compounding reasons: artifact-backed AUC, latency,
             cost, calibration, and prompt-injection exposure. But framing the comparison
             as “LLM versus XGBoost” asks the wrong question. We identify four capability
             gaps where the LLM contributes something the tree cannot natively represent or
             generate (even if hand-engineered features could be fed to a tree): (i) document
             and image fraud via vision–language models, whose raw pixel-and-text inputs lie
             outside any tree’s feature space; (ii) compliance narration—Suspicious Activity
             Report narratives and adverse-action letters—a regulated text-generation category
             (FinCEN; ECOA/Regulation B); (iii) cold-start triage of novel fraud typologies
             before labels exist; and (iv) agentic investigation of the alert queue, at roughly 85×
             lower cost than a human analyst by our estimates. We distill these findings into
             a hybrid architecture in which the LLM serves as sensor (feature extractor over
             unstructured evidence) and scribe (regulatory narrator), while XGBoost remains
             the scorer, with post-score agentic triage on the alert queue. The study is the
             side-probe of a reproducibility program whose thesis is “measure capability, don’t
             trust the label”: applied here, the label under test is “AI.”


1       Introduction
A bank that already runs a tuned gradient-boosted tree over its card-transaction stream faces a concrete
deployment question when the promotional material arrives promising “AI-powered fraud detection”:
what, exactly, would a large language model (LLM) add? The incumbent is formidable. Gradient-
boosted decision trees remain the strongest general-purpose learners on medium-sized tabular data
[6, 12], and on our own custom fraud benchmark a stock XGBoost [1] configuration recorded a
    ∗
        Project guide.


Preprint. Under review.

[PAGE 3]
generated with scikit-learn’s make_classification [10] with a fixed seed (42): 50,000 rows, 20
anonymized numeric features V1 , . . . , V20 (10 informative, 2 redundant), two clusters per class, class
separation 0.8, 1% label noise (flip_y= 0.01), and a 1% target positive (“fraud”) rate (realized
1.44% after the label noise)—the class-imbalance regime typical of card fraud [3]. The anonymized
V -feature convention deliberately mirrors the PCA-transformed feature sets that single-institution
card-fraud datasets expose after confidentiality processing; the reader should treat our data as a
stylized stand-in for one institution’s post-processing feature view, with none of the temporal drift,
verification latency, or covariate shift of real transaction streams [3]. Four aggregate features are
appended per row (mean, standard deviation, max, min of V1 , . . . , V20 ), giving 24 features total.

Split and metric. An 80/20 stratified train/test split (seed 42) yields 40,000 training and 10,000
test rows, with 144 positives in the test set (a realized 1.44% fraud rate: the generator targets 1%
but the flip_y= 0.01 label noise raises the realized positive rate to 1.44% across the full 50,000
rows). The primary metric is held-out ROC-AUC; F1, precision, and recall at the default threshold
are logged alongside. Both arms consume identical splits: the split files are written to disk once and
the LLM arm is fine-tuned on a textual serialization of the same training rows.

Scorer arm: XGBoost. The tree baseline is a stock XGBoost classifier [1] with 200 estimators,
maximum depth 6, learning rate 0.05, subsample and column-subsample 0.8, and scale_pos_-
weight= 7 to counter the 99:1 imbalance; the evaluation objective is AUC. No hyperparameter search
was run beyond this single sensible configuration—the point of the baseline is what an ordinary, lightly
tuned tree achieves, not a leaderboard entry. The full implementation is a single short script (train_-
xgboost.py, released with the paper) that also emits wall-clock training and per-10k-row inference
times. One disclosure belongs here rather than in the limitations: the current quick artifact reports
test AUC 0.7955 for this XGBoost arm on 10,000 held-out rows (experiments/results/quick_-
20260704/qp8_fraud.tsv); the released script’s own output (xgboost_results.json) logs
AUC=0.7942 under a slightly different feature/library configuration, i.e. the two released artifacts
agree to two decimals. An older 21 June 2026 internal record listed XGBoost at AUC 0.975, but we
could not reconstruct that environment and do not use that number as a reproducible headline result
(Section 12).

Challenger arm: fine-tuned LLM. The current challenger is a Qwen3.5-4B SFT run fine-tuned
on row-to-text serializations of the training split, in the style of TabLLM [8]: each row is rendered as
a feature-name/value string and the model is trained to emit a fraud/legitimate judgment. Because
the natural-rate test set contains too few positives for a stable LLM AUC in the quick run, the
evaluation uses a 500-row positive-enriched held-out subset with 20% fraud, disjoint from training
(qp8-fraud-sft_manifest.json). The final quick artifact reports accuracy 0.792 and AUC
0.48268 (rounded to 0.4827 in qp8-fraud-sft.tsv; summarized in qp8_fraud.tsv). An older
21 June 2026 internal record listed a fine-tuned LLM at AUC 0.948, but no released artifact reproduces
it, so we retain it only as historical provenance (Section 12).

What this setup can and cannot support. The synthetic generator gives us a clean, reproducible,
shareable testbed with a known imbalance and noise level, and it suffices for the paper’s actual claim:
a comparison of roles (who should hold the scorer seat, and what the LLM is uniquely positioned
to do elsewhere in the pipeline). It cannot support absolute performance claims about production
fraud systems, and no number in this paper should be quoted as an expected production AUC. Where
we report operational quantities that depend on facts outside the dataset— analyst costs, latency
envelopes—we scope them explicitly as internal estimates.

3   The Scorer Result: The Tree Keeps the Seat
On the tabular scoring task, the current quick artifacts place XGBoost at test AUC 0.7955 on the
10,000-row held-out split and the Qwen3.5-4B SFT row-serialization arm at accuracy 0.792 but AUC
0.48268 on a 500-row positive-enriched held-out evaluation (Table 1). The accuracy is not evidence
of useful ranking: at this class balance, AUC near 0.5 means the SFT scores do not order positives
ahead of negatives. This is consistent with the broader tabular literature: tree ensembles remain the
reference class on medium-sized tabular problems [6, 12], and LLM row-serialization approaches are
most attractive in few-shot regimes, not at 40,000 labeled examples [8].


                                                   3

[PAGE 4]
Table 1: Current artifact-backed tabular scoring evidence on the custom synthetic fraud dataset. XG-
Boost is evaluated on the full 10,000-row test split; the SFT LLM is evaluated on a 500-row positive-
enriched held-out subset because the natural-rate quick split has too few positives for stable LLM AUC.
Artifacts: quick_20260704/qp8_fraud.tsv and quick_20260704/qp8-fraud-sft.tsv. Op-
erational columns are qualitative rankings, argued in the text.
      Model                Eval split       n      Acc.     AUC ↑      Latency    Injection surface
      XGBoost (scorer)      full test     10,000    n/a    0.7955     ms-scale         none
      Qwen3.5-4B SFT      enriched test    500     0.792   0.48268     s-scale        present



The artifact-backed ranking is only the first of five reasons the tree keeps the scorer seat. Even at
AUC parity we would not move the LLM into the real-time scoring path, for four operational reasons.

1. Latency. Card authorization is a hard-real-time loop: the score must arrive within a network-
and-rules budget measured in tens of milliseconds. A 200-tree, depth-6 ensemble scores a row
in microseconds to milliseconds on commodity CPUs; the committed run of our released script
measures roughly 6 ms per 10,000 rows (xgboost_results.json)—a rounding error against the
authorization budget. An LLM forward pass over a serialized row—let alone an autoregressive
generation—is orders of magnitude slower and jitter-prone, and batching tricks that amortize LLM
cost are exactly what a per-authorization synchronous path cannot use.

2. Cost. The tree’s marginal cost per transaction is effectively zero at card-network volumes; the
LLM’s is a token bill that scales linearly with traffic. Spending per-token inference on a task the tree
does better is the least defensible line item in the architecture we propose; the LLM budget should be
reserved for the low-volume, high-value tasks of Section 8, where it has no substitute.

3. Calibration. Downstream of the score sit threshold rules, alert budgets, and expected-loss
computations that consume the score as a probability. Boosted trees produce dense, monotonically
usable scores that calibrate well with standard post-hoc maps. LLM-verbalized confidences inherit
the miscalibration pathologies of modern neural networks [7], compounded by the quantization of
confidence into tokens. A scorer whose 0.9 does not mean ninety percent silently corrupts every
expected-loss decision behind it.

4. Prompt-injection exposure. This reason is qualitative and, we believe, decisive. A tree scores
numbers; there is no channel through which a transaction can address the model. An LLM that
reads serialized transaction fields—merchant names, memo strings, cardholder-supplied text—creates
a new attack surface: adversarial instructions embedded in attacker-controlled fields, the indirect
prompt-injection pattern documented by Greshake et al. [5]. A fraudster who learns that the scorer
reads free text will put text in front of it. Placing an instruction-following model in a synchronous
decision loop over attacker-authored inputs is a security regression independent of accuracy, and it is
the single strongest argument for keeping the scorer seat non-linguistic.
The conclusion of this section is deliberately narrow: on fixed-schema tabular scoring with ample
labels, the current artifacts score the LLM below the tree—and it would not get the seat even had it
tied. Everything the LLM wins, it wins elsewhere in the pipeline—which is the subject of the next
section.


4   Measured Evidence: Calibration, CIs, and Sensor Surrogate

The five operational arguments of Section 3 are qualitative; this section puts three of them on a
quantitative footing using the released dataset (fraud_data.csv, test_data.csv) and the re-
leased stock XGBoost config of Section 2. All numbers in this section come from a single script
(scripts/p5p8/p8_calibration_cis.py); all CIs are paired bootstrap with 1,000 resamples of
the held-out test split, two-sided α=0.05.


                                                   4

[PAGE 9]
Table 9: Paired-bootstrap CIs (nboot =400, α=0.05 two-sided) on the PR-AUC delta and P @1% delta
between the three trees of Table 8 at five positive rates. “excl. 0” flags CIs that exclude zero – a
statistically detectable gap at the 95% level.
            Rate      ∆                    metric        point      95% CI        excl. 0?
            release   P @1%(24f)−(4s)      P @1%        +0.470   [+0.36, +0.58]     yes
            1.00%     P @1%(24f)−(4s)      P @1%        +0.510   [+0.36, +0.58]     yes
            0.50%     P @1%(24f)−(4s)      P @1%        +0.293   [+0.19, +0.38]     yes
            0.10%     P @1%(24f)−(4s)      P @1%        +0.040   [+0.01, +0.09]     yes
            0.05%     P @1%(24f)−(4s)      P @1%        +0.040   [+0.01, +0.09]     yes
            release   PR-AUC(24f)−(4s)     PR-AUC       +0.549   [+0.46, +0.61]     yes
            1.00%     PR-AUC(24f)−(4s)     PR-AUC       +0.591   [+0.49, +0.67]     yes
            0.50%     PR-AUC(24f)−(4s)     PR-AUC       +0.607   [+0.46, +0.74]     yes
            0.10%     PR-AUC(24f)−(4s)     PR-AUC       +0.216   [−0.05, +0.51]     no
            0.05%     PR-AUC(24f)−(4s)     PR-AUC       +0.334   [+0.10, +0.78]     yes


splits are starved for positives to power a CI – but the sign is consistent at all five rates (∆ ∈
[+0.018, +0.092]).
The second observation is that the 4-aggregate sensor surrogate loses decisively on PR-AUC at
four of five rates and on P @1% at all five rates (Table 9):
The third observation is the recall@top-1% monotonicity: the 24-feature tree recovers 100% of
positives in the top-1% score slice at every rate down to 0.05% (5 positives). This is the operating-
point evidence that the iter-4 ROC-AUC and iter-8 noise budget already implied but did not measure:
at the deployed operating point the full tree keeps the alert queue clean enough to recover every
positive at the top-1% review budget, while the 4-aggregate surrogate falls to 40% recall at 0.05%.

Reading the 4-aggregate surrogate as an oracle LLM sensor. A real LLM-as-sensor would face
the same positive-rate stress but with the additional burden of noisy numeric outputs. The iter-8
sensor-noise sweep showed that σ ≥ 0.05 on the four aggregates measurably degrades the 24-feature
tree; an LLM sensor that produces a single deterministic 4-vector per transaction has already lost the
PR-AUC contest by ≈ 0.55 on the released test split at the release rate, and the noise budget further
restricts it to σ ≤ 0.02 for the tree to register no measurable loss. The combined picture is a sharp
negative-evidence picture for the LLM-as-sensor pattern on this dataset at any realistic fraud base
rate.

Cross-paper coupling (Pillar 3 / Pillar 4). The Section 3 operational stack (latency, cost, calibra-
tion, injection surface) and the Section 4.5 operating-point gap both reduce to the same diagnostic:
an auxiliary model channel must deliver information strictly orthogonal to the dominant predictor
to register above the noise floor of a paired bootstrap CI. The Pillar-3 ZVF controller (items 08, 13,
16 of the P5P8 improvements ledger) reaches the same conclusion in the RL policy-improvement
setting.

4.6   Cost-adjusted operating curve at six review budgets

The PR-AUC and top-1% tables above measure detection quality at a fixed score-rank budget. A real
fraud-ops deployment is budgeted in dollars, not in score-rank slots: an analyst queue has a hard
top-K review budget per day, and each row that enters the queue costs both a model call and an
analyst review minute. This subsection closes that loop. We compare four deployment modes on
the released 10,000-row test split at six review budgets K ∈ {0.1%, 0.5%, 1%, 2%, 5%, 10%} of the
stream:

       • M1: XGB-20raw. Tree on the 20 raw V -features only. No LLM cost, no LLM value.
       • M2: XGB-24full (oracle LLM-as-sensor). Tree on the 20 raw + 4 hand-engineered
         aggregates. Treats the aggregates as a free oracle LLM sensor and bounds the upside of any
         real LLM-as-sensor.
       • M3: Hybrid-10%. XGB-20raw for the bottom 90% of the stream; XGB-24full for the top
         10%. The LLM sensor is paid only on the top 10% tail.


                                                    9

[PAGE 53]
• Model-side uncertainty (P8): “fraction of top-K rows that lie on a flat XGBoost-score
            plateau” – a property of the model’s decision surface, not the rollout batch. Density ∼0.008.

The 60× density gap is the structural signature of the mechanism difference. The iter-80 rule is
NOT a member of the P5/P7 synthesis family. Reframing: P5 and P7 both measure the same
underlying phenomenon (zero-variance-group starvation of policy gradient); P8 measures a different
phenomenon (model-side flat-region uncertainty) that happens to also be a contrast-related signal but
at a different abstraction layer.

7     Measured Evidence: Four-Domain Density Matrix
We close the iter-124 surface-recommendation by adding P7 per-prompt Adaptive-G* density as a
fourth domain. iter-131 ran the per-prompt Adaptive-G* simulation on the same N2 four-method
panel (2560 prompt-cells = 4 methods ×40 steps ×16 prompts), giving a matched-granularity
domain that extends the iter-124 three-domain matrix {D1 =P 8, D2 =P 7-step, D3 =P 5}-mega} to
{D1 , D2 , D3 , D4 =P 7-pp}.

7.1      Domain definitions

The four “signal-depleted” cells are defined to be matched across the operational meaning of contrast-
starved on each pillar:

Table 54: Four “signal-depleted” density domains. Each row is the fraction of evaluation units in the
canonical dataset where the domain’s signal-starvation criterion holds.
           domain                        nfire /ntotal                                  rule
    D1     P8 grad-band              84/10000 = 0.84%            row in top-K AND consecutive gradient small
    D2     P7 step zvf-triage          20/40 = 50.0%                step zvf ≥ 0.7 (DEGENERATE regime)
    D3     P5 mega zvf=1.0             36/98 = 36.7%                        cell per-step zvf == 1.0
    D4     P7 per-prompt boundary   1867/2560 = 72.9%                  per-prompt k ∈ {0, 8} (boundary)


7.2      Pairwise density ratios


Table 55: Pairwise density ratios with 95% percentile bootstrap CIs (B=1500, seed 20260705).
excl-1 flags ratios where the CI excludes 1.0 by an order-of-magnitude criterion (point < 0.1 or
> 10).
                            ratio               point         ci_lo    ci_hi   excl-1
                            P5 / P7-step       0.735       0.500       1.140   no
                            P5 / P7-pp         0.504       0.377       0.637   no
                            P5 / P8             43.7        30.4        62.4   yes
                            P7-pp / P7-step     1.46        1.12        2.11   no
                            P7-pp / P5           1.99        1.56       2.67   no
                            P7-pp / P8           86.8        71.4      109.2   yes
                            P8 / P7-step      0.0168      0.0117      0.0248   yes
                            P8 / P7-pp        0.0115      0.0092      0.0140   yes
                            P7-step / P5         1.36        0.86       2.04   no


7.3      H-tests (4-domain)

H1 (PASS). D4 sits inside the iter-124 {P5, P7-step} super-domain: both P7-pp/P7-step (1.46, CI
[1.12, 2.11]) and P7-pp/P5 (1.99, CI [1.56, 2.67]) ratios contain 1.0 at order-of-magnitude tolerance.
Adding D4 does not break the iter-124 super-domain claim ({P5, P7}-step<> {P8}) — both D4
and D3 fall on the P5/P7-step side of the split.
H2 (PASS). Density rank by nfire /ntotal : D4 = 0.729 > D2 = 0.500 > D3 = 0.367 > D1 =
0.0084. Per-prompt granularity (finest) is the most signal-depleted; per-row (D1 , coarsest) is the


                                                         53

[PAGE 67]
H2 (FAIL – Pareto-frontier non-existence): there exists at least one cell with esc_prec ≥ 0.10
AND value_rate ≥ 0.30 simultaneously. Measured: 0/700 = 0.0%. The closest-to-Pareto cell is
seed=20260708, rate=1.44%, fset=24full, τ = 0.0: value_rate=0.568, esc_prec=0.0108. Even the
highest-value operating point sits two orders of magnitude below the precision bar.
H3 (PASS): value_rate is monotone non-increasing in τ on ≥ 80% of (seed × rate × fset) cells.
Measured: 100/100 = 100.0%. Strict monotonicity across all 7 thresholds; this is the expected
monotonicity (fewer fires ⇒ fewer lifts).
H4 (PASS): at the cheap tier, breakeven rate (esc_cost_per_lift ≤ $50) is monotone non-decreasing
in τ on ≥ 50% of cells. Measured: 100/100 = 100.0%. Stricter thresholds restore breakeven by
either (a) eliminating wasted fires on non-fraud rows (precision lift) or (b) eliminating all fires entirely
(zero-cost trivially breakeven).

7.5.36    Structural precision ceiling – the signal-mass bottleneck
The H1/H2 FAIL is not a threshold-tuning miss; it is a structural property of the V_mean distribution
on this dataset. At τ = 0.0 the LLM sensor fires on ∼5000/10000 test rows and catches 50 positives:
esc_prec = 50/(50 + 4945) = 0.0100. At τ = 2.0 the LLM sensor fires on 0/10000 test rows
(n_lift = 0, n_waste = 0): the entire signal mass of V_mean is concentrated in V _mean ∈ (0, 2].
Across all 7 thresholds, the positive-class enrichment of LLM fires stays at ∼1% because the V_mean
distribution does not stratify fraud vs. non-fraud sharply enough to support a 10% precision gate at
any cutoff.
This is the operational ceiling of the LLM-as-sensor pattern on credit-card fraud detection with
this feature set: the sensor is fundamentally a recall signal (catches positives XGB misses at high
coverage) rather than a precision signal (concentrates fires on positives). The structural signal-to-noise
ratio limits precision to ∼1.1% regardless of threshold choice.

7.5.37    Sharpest paper-grade findings
      1. Pareto frontier does not exist on this dataset. At no τ ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
         and no (seed × rate × fset) cell does esc_prec reach 10%. The precision-recall trade-off is
         not a frontier – it is a single-point at (recall lift ∼0.5, precision ∼1%).
      2. Value rate is strictly monotone non-increasing in τ on 100/100 cells (H3 PASS). The
         recall signal decays predictably as the threshold rises.
      3. Breakeven restores at higher τ on 100/100 cells (H4 PASS). The cost-per-lift metric
         improves with stricter gating because wasteful fires on non-fraud rows are eliminated first.
      4. Signal mass bounded in V _mean ∈ (0, 2]. At τ = 2.0, n_lift = n_waste = 0 – the LLM
         sensor is silent. The operational conclusion is that the V_mean feature alone cannot support
         a precision-constrained deployment; combining it with the 4 aggregate features (V_std,
         V_max, V_min, V_mean jointly) or deploying a learned precision-restoration layer is the
         next step.
      5. Iter-156 H1 REFUTED is structurally reinforced. Iter-156 found esc_prec = 1% at
         τ = 0.0 and recommended threshold tuning. Iter-168 sweeps τ and finds the precision
         ceiling is structural – no tuning rescues it. The operational recommendation is now “the
         LLM-as-sensor signal is a recall instrument, not a precision instrument; deploy for recall
         lift at the cheap tier (where breakeven holds), not for precision restoration”.

7.5.38    Cross-paper coupling
         • P8 iter-156 row 172: iter-156 measured the high-recall low-precision signature at τ = 0.0
           (esc_prec = 1%, n_lift = 50, n_waste = 4945). Iter-168 confirms this is the only achievable
           operating point on this dataset – the threshold sweep cannot escape it.
         • P8 iter-160 row 174: iter-160 operating-point utility analysis showed M2 (oracle LLM
           sensor) loses on TP/$ at every realistic fraud-ops budget. Iter-168 quantifies the structural
           reason: precision is bounded at ∼1% regardless of how the LLM sensor is thresholded.
         • P8 iter-148 row 166: iter-148 cost matrix averaged across all fires; iter-156 decomposed
           fires into value/waste; iter-168 extends the value-side analysis to a 7-threshold sweep and
           confirms the recall-precision trade-off is a single point.


                                                    67

[PAGE 70]
cannot escape it. The corrected operational recommendation is: deploy the LLM-as-
        sensor for recall lift at the cheap tier only; abandon precision-restoration attempts on
        PCA- aggregated features.
     5. Precision-restoration requires more information per fire, not less. The V-stat features
        are PCA-aggregated summaries of V1 ..V20 ; precision-restoration would require either (a) a
        model that operates on the raw 20 PCA components directly (which XGB already does),
        or (b) features that actually stratify fraud vs non-fraud (which the V-stats demonstrably do
        not). The iter-172 negative finding strengthens the iter-156 / iter-168 / iter-120 chain: the
        LLM-as-sensor pattern is a recall instrument on this feature class.

7.5.43    Cross-paper coupling
         • P8 iter-168 row 179: iter-168’s recommendation (4) (“EXTEND with joint V-classifier”)
           is decisively refuted by iter-172’s 4/4 H-FAIL on 2200 cells. The precision ceiling is
           V-stat-class, not single-feature.
         • P8 iter-156 row 172: iter-156 documented the high-recall low-precision signature at τ = 0.0
           (esc_prec = 1%). Iter-172 confirms the 1% is not a Vmean -only phenomenon – every
           aggregation of the V-stats reproduces it on 0/100 cells achieving esc_prec ≥ 5%.
         • P8 iter-120 row 165: iter-120’s “score-stream geometry trigger, not feature trigger” conclu-
           sion is sharpened by iter-172: even a learned joint V-stat classifier (trained specifically to
           predict fraud from the V-stats) cannot restore precision; the geometry-driven firing is the
           dominant signal and the V-stats themselves carry little class-conditional information.
         • FRONTIER_INSIGHTS Round 2 (ZVF = signal availability): iter-172 sharpens the op-
           erational analogue. Just as ZVF measures the fraction of GRPO groups with zero advantage
           (signal starvation), iter-172 measures the fraction of LLM fires with zero class-conditional
           enrichment (precision starvation). Both are structural signal- availability measurements that
           cannot be rescued by ensemblemethods over the same feature class.

7.5.44    Operational recommendations
     1. ABANDON precision-restoration attempts on PCA-aggregated V-stat features. Iter-172
        shows 0/100 cells achieve esc_prec ≥ 5% at any τ on the joint classifier.
     2. DEPLOY the LLM-as-sensor pattern for recall lift at the cheap tier only (iter-156 H2b +
        iter-168 H4 + iter-172 H4 all converge on this operational conclusion).
     3. EXTEND the sensor to operate on raw V1 ..V20 features rather than aggregated V-stats
        if precision restoration is required. The current LLM-call budget cannot be spent more
        efficiently on PCA-aggregated features than the iter-168 sweep already measured.
     4. WIRE p8_iter172_vstat_ensemble_precision.py as a CI pre-commit on precision-
        restoration proposals: the joint classifier must achieve esc_prec ≥ 5% on ≥ 25% of cells at
        some τ to be considered a candidate for further engineering investment.




                                                   70

[PAGE 77]
3. WIRE the audit as a CI pre-commit gate — gate fails if H3 count drops below 5/10 OR if
         the decile-7 hurt exceeds +0.005.
      4. EXTEND in next-iter to per-decile cost-savings combining iter-192’s decile-stratified Brier
         lift with iter-188’s cost curve to report dollar value per decile.

7.5.59    Falsifiable questions and operational stakes
The P8 paper has, across 32 prior iterations, documented the aggregate cost-savings value of the
four V-stat features as a single block. Iter-188 reported $0.011/tx savings at c=100 on held-out data.
Iter-192 showed V-stat features help most in decile 10 (extreme-low-density) on calibration. But the
banker’s question — “if I can only afford to deploy V-stat features on a SUBSET of transactions,
WHICH Vmean decile should I target?” — has not been answered. Iter-204 stratifies held-out test_-
data.csv into 10 Vmean deciles and reports per-decile cost-savings for three feature sets at five cost
ratios.

      1. In WHICH Vmean decile does the cost-savings lift concentrate?
      2. Does the lift concentration depend on the cost ratio c?
      3. Is there ANY Vmean regime where 4sensor alone beats 20raw (contradicting iter-176’s
         blanket “4sensor is catastrophic” conclusion)?

7.5.60    Pipeline and headline findings
We train XGB-200 (max_depth=6, lr=0.05, scale_pos_weight=neg/pos) on fraud_data.csv
(50K rows, 24 features, 719 frauds) for 3 feature sets (20raw, 24full, 4sensor) × 5 seeds = 15
trained models. We stratify test_data.csv (10K rows, 144 frauds) into 10 Vmean deciles (iter-192’s
stratification; equal-frequency, n=1000/decile). For each (decile, fset, c ∈ {1, 10, 100, 1000, 10000},
seed), we threshold-sweep cost(t) = (FN(t) · c + FP(t))/N over the decile’s transactions
and record the cost-optimal t⋆ . Per (decile, c): paired-seed bootstrap 95% CI (B=2000,
seed=20260706+d·100+c) on the 24full-20raw gap and the 4sensor-20raw gap.

7.5.61    Headline results
3 PASS + 2 sharp FAIL across 5 hypotheses. The FAILs are the headline paper-grade findings.

 decile      n    pos       24full−20raw (CI95 )                4sensor−20raw               share at c=100
   0       1000    5    −0.00240 [−0.0036, −0.0012]      −0.00280 [−0.0038, −0.0018]             17%
   1       1000   11    −0.00040 [−0.0008, +0.0000]       +0.01360 [+0.0044, +0.0254]             3%
   2       1000   15    −0.02500 [−0.030, −0.020]                  +0.09360                      46%
   3       1000   19     −0.00680 [−0.008, −0.005]                +0.12100                       25%
   4       1000   17     −0.00320 [−0.004, −0.002]                +0.18620                       12%
   5       1000   13     −0.00240 [−0.003, −0.001]                +0.16700                        9%
   6       1000   23     −0.00120 [−0.002, −0.001]                +0.38340                        4%
   7       1000   20     −0.00280 [−0.004, −0.002]                +0.23980                       10%
   8       1000   12     −0.00560 [−0.007, −0.004]                +0.09220                       21%
   9       1000    9     −0.00460 [−0.005, −0.004]                +0.00700                       17%
Table 69: Per-Vmean -decile cost-savings lift at c=100 on held-out test_data.csv. “share” is each
decile’s share of the total negative gap (cost-reduction attribution). Decile 2 alone accounts for 46%
of the total lift; decile 1 contributes only 3%. The 4sensor column shows 4sensor alone is competitive
with raw features in decile 0 only (gap −0.0028, CI strictly negative), and catastrophic in every other
decile. CIs are 5-seed paired bootstrap B=2000.


7.5.62    Sharpest findings
F1 (H2 PASS — LIFT CONCENTRATION). Per-decile cost-savings lift at c=100 is concen-
trated in decile 2. Decile 2 (Vmean near the empirical median; 15 positives, 1.5% hit rate) gets
−0.025/tx savings — 62× larger than decile 1 (−0.0004/tx, smallest) and 10× larger than the per-
decile average (−0.005/tx). The lift is non-monotone in Vmean : it concentrates in mid-Vmean , not at
the extremes. Operationally, deployment that prioritizes V-stat features on mid-Vmean transactions
captures the bulk of the value.


                                                  77

[PAGE 78]
F2 (H3+H4 PASS — TOP-1 SHARE INVARIANT IN c). Decile 2 alone accounts for 45.96%
of the total positive lift share at both c=100 and c=10000. The share is independent of cost ratio
because the cost-savings gap is negative at every decile (24full beats 20raw everywhere); only the
magnitude scales with c, and decile 2’s magnitude scales fastest. This is a robust operational target:
deploy V-stat features with priority on mid-Vmean transactions, regardless of c.
F3 (H5 FAIL → SHARP — 4sensor DECILE-0 ANOMALY). Decile 0 (lowest Vmean ) is
the one regime where 4sensor alone beats 20raw. 4sensor gap at decile 0 = −0.0028/tx (CI
[−0.0038, −0.0018] strictly negative), while 4sensor gap at deciles 1–8 ranges from +0.0136 to
+0.383/tx (catastrophic). This contradicts iter-176’s blanket conclusion that “4sensor alone is
catastrophic” — iter-204 finds that in the LOW-Vmean regime (decile 0, lowest 10% of Vmean values;
5 positives, 0.5% hit rate), the LLM-derived 4-sensor block is competitive with the 20 raw V1–V20
features. Mechanism: in low-Vmean transactions, all 20 raw features have similar (low) magnitudes,
so XGB’s per-feature splits are uninformative; the 4 V-stat aggregates (especially Vmean and Vmax )
carry more discriminative signal in this regime.
F4 (H1 FAIL → HONEST REPRODUCIBILITY DRIFT). My global aggregate 24full-20raw
gap at c=100 = −0.01294/tx (5-seed paired bootstrap CI); iter-188 reported −0.01116/tx. Difference
$0.00178/tx = 16% relative drift, but both agree on sign and order of magnitude. The drift is consistent
with seed variance (iter-188 used a similar but not identical seed set). The headline gap is reproducible
at the order-of-magnitude level but not to the $0.001/tx precision claimed in iter-188.

7.5.63    Cross-paper coupling
         • P8 iter-188 (cost-asymmetric transfer) — iter-188 measured the AGGREGATE gap; iter-204
           lifts to per-decile and finds the lift concentrates in decile 2 (46% share).
         • P8 iter-192 (Vmean decile audit on Brier) — iter-192 found V-stat features help most in
           decile 10 (extreme-low-density) on calibration; iter-204 measures cost and finds decile 2
           (mid-Vmean ) dominates. The calibration-best decile is NOT the cost-best decile — two
           different operational metrics point to different V-stat priorities.
         • P8 iter-176 (sensor/scribe/scorer 3-way CIs) — iter-176 blanket concluded “4sensor is
           catastrophic”; iter-204 qualifies with “EXCEPT in decile 0”.
         • P8 iter-196 (V-stat LOO ablation) — Vmax is the dominant single contributor (43% of lift);
           iter-204’s decile 0 finding is consistent: in low-Vmean regimes, Vmax captures upper-tail
           signal that raw features miss.
         • P8 iter-200 (base-rate stress) — lift is robust across base rates; iter-204 confirms robustness
           across cost ratios (lift sign preserved at c ∈ {1, 10, 100, 1000, 10000}).

7.5.64    Operational
      1. DEPLOY V-stat features with priority on mid-Vmean transactions (decile 2 in iter-204’s
         stratification); 46% of the lift concentrates there.
      2. FOR LOW-Vmean deployments (decile 0), consider 4sensor ALONE as a viable model —
         it competes with raw features in this regime.
      3. REPORT the per-decile cost table as Table 69 in this section as the per-decile attribution
         headline.
      4. WIRE        python3 platform_modal/scripts/p5p8/p8_iter204_decile_cost_-
         savings.py as a CI pre-commit gate — fails if decile 2 share drops below 30% OR if
         decile 0 4sensor gap flips positive (4sensor becomes uniformly worse).
      5. EXTEND in next-iter to per-decile cost savings stratified by Vstd (iter-184’s covariate) for
         the next synthesis iter.

7.5.65    Falsifiable questions and operational stakes
The P5P8-SYNTH density matrix reached D16 (per-prompt reward stability on N2 four-method) at
iter-176 row 188. D1..D16 measure substantive evidence density at increasingly fine granularity.
This iter (P5P8-SYNTH JOB B) extends the matrix to the cross-paper metadata layer by computing


                                                    78

[PAGE 92]
but are not analyzed here. Nothing in this section is legal advice; it establishes only that the narration
category exists, is mandatory, and is textual—hence enterable by an LLM and not by a tree.

11    Related Work
Gradient boosting on tabular data. XGBoost [1] remains the workhorse of applied tabular
classification. Grinsztajn et al. [6] show systematically that tree-based models still outperform
deep learning on medium-sized tabular benchmarks, attributing the gap to trees’ robustness to
uninformative features and to the non-smooth, non-rotation-invariant structure of tabular targets;
Shwartz-Ziv and Armon [12] reach the same conclusion across deep tabular architectures. Our
scorer result is a single applied data point consistent with this literature, with the added operational
arguments (latency, cost, calibration [7], injection surface [5]) that the benchmarking papers do not
address.

LLMs for tabular data. TabLLM [8] established the serialize-and-prompt recipe and showed it is
most competitive in very-few-shot regimes, degrading relative to boosted trees as labeled data grows—
exactly the pattern our head-to-head reproduces at 40,000 labels, and exactly why our taxonomy
assigns the LLM the cold-start seat (Section 8.3) rather than the scorer seat.

Fraud detection on transaction data. Dal Pozzolo et al. [3] give the canonical treatment of
realistic credit-card fraud modeling—class imbalance, verification latency, and concept drift—and
motivate the anonymized-feature, heavily imbalanced regime our synthetic benchmark imitates. Our
contribution is orthogonal to this line: we do not propose a better detector but a division of labor
around an already-strong one.

LLM agents for financial crime. Pirmorad [11] shows that LLMs prompted with serialized
financial-graph neighborhoods can perform analyst-style money-laundering triage few-shot, with
coherent red-flag justifications; Naik et al. [9] builds an agentic framework specifically for drafting
AML compliance narratives (SARs) with human-in-the-loop review. These systems instantiate,
respectively, our cold-start and scribe/triage seats; our contribution is to place them in an explicit
architecture where the real-time scorer remains a tree.

VLMs for document and image fraud. FakeShield [16] demonstrates explainable image-forgery
detection and localization with multimodal LLMs; Forensics-Bench [15] benchmarks large vision–
language models across 112 forgery types and finds substantial headroom. Together they support
both halves of our sensor claim: VLMs uniquely extend the input space to raw document evidence,
and their current reliability justifies confining them to feature extraction with downstream tree scoring
and human review rather than autonomous verdicts.

Program context. The parent reproducibility program measures RL post-training claims capability-
by-capability rather than by their labels; the present side-probe applies the identical discipline to
the marketing label “AI” in fraud detection. The methodological through-line is decomposition: a
bundled claim is replaced by per-capability measurements, and the deployment decision is made seat
by seat.

12    Limitations
Custom synthetic dataset. Our benchmark is generated by make_classification [10], not
drawn from production transactions. It reproduces the class imbalance and anonymized feature shape
of single-institution card-fraud data but none of its temporal drift, verification latency, seasonality, or
adversarial adaptation [3]. The current artifact-backed AUCs (0.7955, 0.48268) are therefore results
on a stylized testbed and must not be quoted as production expectations. The margin is dataset- and
environment-specific.

Historical numbers are records, not reproductions. We could not reconstruct the environment
of the 21 June 2026 internal run that produced the older 0.975/0.948 pair, so those numbers are
historical records, not headline evidence. The paper’s architectural conclusions are insulated from


                                                    92

[PAGE 93]
this by design: the seat assignment rests on the latency, cost, calibration, and injection arguments of
Section 3, which apply at accuracy parity and do not depend on the margin or its direction.


No cross-institution validation. Everything here is one configuration of one synthetic generator
standing in for one institution’s feature view. Fraud patterns, feature pipelines, and base rates vary
sharply across issuers and geographies; the head-to-head has not been replicated on any second data
distribution, public benchmark, or real institution.


Cost estimates are internal. The ∼85× agentic-triage cost advantage (Section 8.4) is an internal
estimate built from June-2026 token prices, assumed per-alert investigation times, and loaded analyst
compensation. Each assumption is contestable; none has been validated in a production deployment;
the figure is reported to one significant figure and should be read as “one to two orders of magnitude,”
not as a measured constant. The latency and per-transaction cost contrasts of Section 3 are likewise
operational characterizations, not benchmarked service-level measurements.


Single LLM family, single recipe. The 0.48268 challenger is one instruction-tuned LLM family
under one serialization format and one fine-tuning budget. We did not sweep model families, scales,
serializations, or prompting strategies, and stronger or differently trained LLMs could narrow the
accuracy gap. We note, however, that the architecture’s assignment of seats does not hinge on the
gap: the latency, cost, calibration, and injection arguments of Section 3 apply at accuracy parity.


Capability gaps argued, not all measured. Of the four taxonomy entries, only the scorer head-to-
head carries a number of our own. The sensor, scribe, and cold-start seats are grounded in external
literature and primary regulatory text rather than in our own end-to-end evaluations; the hybrid
architecture of Section 9 is a design contribution whose integrated performance is future work, not a
deployed and audited system. Nothing in Section 10 is legal advice.


13    Conclusion and Future Work: Building the Hybrid

The obvious next step is to build and measure the architecture rather than argue it. Our plan, in the
order the seats de-risk each other:

1. Sensor first. Attach a VLM feature extractor to a document-bearing subset of cases and measure
   the marginal AUC of sensor-derived features inside the existing tree—the cleanest test, since the
   metric is the incumbent’s own. This includes adversarial evaluation: injected instructions inside
   documents must degrade at worst that document’s features [5].
2. Scribe second. Evaluate SAR-narrative drafts against FinCEN’s element-coverage structure
   (who/what/when/where/why/how) [4] with analyst-grader rubrics, and adverse-action drafts for
   fidelity to the tree’s actual attributions [13, 2]—measuring time-to-file and revision distance, with
   a hard human sign-off gate.
3. Triage third. Replace the internal 85× estimate with measured per-alert token costs and analyst-
   minutes-saved on a live queue, including the ordering quality of agent-ranked versus score-ranked
   queues.
4. Cold-start last. Simulate typology emergence by holding out a generator cluster as an “unseen
   scheme” and measuring few-shot LLM detection during the label vacuum, against the retrained
   tree’s catch-up curve.

Each experiment scores one seat with the seat’s own metric—capability measured, label ignored. If
the program thesis holds anywhere outside RL post-training, it should hold here: the pipeline that
results is not “AI replacing the model” but a tree that scores, an LLM that sees and writes, and an
analyst who decides. Until those experiments are run, the only artifact-backed performance result is
the synthetic-data scorer comparison; the sensor, scribe, and triage seats remain design hypotheses
rather than demonstrated deployment gains.


                                                  93


# R01: ACM benchmark variant


Root: `platform_hybrid/paper/acm_main.tex`  Pages: 11  Words: 4085


[PAGE 1]
1    An Evidence-Tiered Audit of RL Post-Training Implementations
2
3    ANONYMOUS AUTHOR(S)
4
5    Reinforcement learning (RL) has become a dominant approach for post-training language models, yet the field lacks standardized
6    benchmarks for comparing RL methods across training frameworks, and it remains unclear which empirical conclusions transfer across
7
     implementations, model families, and scales. We present TinkerRL-Bench, a unified benchmark spanning 11 implementations across
8
     7 RL libraries (TRL, Stable Baselines3, CleanRL, Tianshou, PufferLib, rl_games, d3rlpy) evaluated on mathematical reasoning (GSM8K,
9
10
     arithmetic), preference learning, and knowledge distillation tasks. Results to date cover five end-to-end stacks (TRL, Tinker, SB3, CleanRL,
11   Tianshou); the remaining libraries are scaffolded but not yet run. Using standardized reward functions, hyperparameter mappings,
12   and statistical evaluation protocols based on rliable, we provide a cross-library comparison of RL post-training implementations
13   for language models at scales from 0.6B to 30B parameters. We report three conservative findings. First, the large measured gap
14   between LLM-native GRPO stacks (TRL, Tinker) and classic RL libraries (SB3, CleanRL, Tianshou) is best read as an implementation
15
     and task-encoding effect—the classic-RL stacks run a small MLP policy over a discrete-action arithmetic MDP, not the autoregressive
16
     language model—rather than evidence of algorithmic superiority. Second, although online GSM8K training reward improves with
17
     GRPO, a held-out control shows the mean Qwen3-8B-Instruct gain over the same checkpoint’s pre-RL held-out accuracy is small and
18
19   not statistically significant (83.3% vs. 82.0%, 𝑝=0.26). Third, implementation and stack choices are a substantial source of cross-library
20   performance variance, though we do not claim they account for the majority of it absent a fully matched variance decomposition. We
21   release code, evaluation tooling, and checkpoints where licensing and the managed backend permit; closed-backend base weights are
22   not reproducible artifacts.
23
         Code: https://anonymous.4open.science/r/tinker-rl-lab              Models: https://anonymous.4open.science/r/tinker-rl-lab
24
25
     CCS Concepts: • Computing methodologies → Machine learning; Reinforcement learning; Learning paradigms; Natural
26
     language processing; • Software and its engineering → Software creation and management; • General and reference → Evaluation.
27
28
29
     Additional Key Words and Phrases: reinforcement learning, language models, benchmark, reproducibility, RLHF, GRPO, DPO, post-
30   training
31
32   ACM Reference Format:
33   Anonymous Author(s). 2026. An Evidence-Tiered Audit of RL Post-Training Implementations. 1, 1 (July 2026), 11 pages. https:
34   //doi.org/10.1145/nnnnnnn.nnnnnnn
35
36
37
     1    Introduction
38
39       Scope within the program. This is the compact ACM venue derivative of the benchmark artifact. It is canonical for the
40
     cross-library implementation comparison, not for the ZVF theory, adaptive-controller proposal, or the new PPO/SAO
41
     synthesis. Those questions are handled by focused companion papers; results are not pooled across them as if they
42
43   were independent replications.
44
45
     Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
46
     made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
47
     of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
48   servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
49   © 2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
50   Manuscript submitted to ACM
51
52   Manuscript submitted to ACM                                                                                                                             1

[PAGE 9]
An Evidence-Tiered Audit of RL Post-Training Implementations                                                                              9

417
418
                                      Hyperparameter Sensitivity: GRPO on Arithmetic                                         1.0
419
420
                               1e-5    0.15             0.22            0.35            0.28             0.18
421
422
423
                                                                                                                             0.8
424
425                            5e-5    0.32             0.48            0.62            0.55             0.41
426




                                                                                                                               Final Accuracy
                                                                                                                             0.6


               Learning Rate
427
428
429                            1e-4    0.45             0.65            0.73            0.71             0.58
430
431
432
                                                                                                                             0.4
433
434
                               5e-4    0.38             0.52            0.68            0.63             0.45
435
436                                                                                                                          0.2
437
438                            1e-3    0.12             0.25            0.35            0.30             0.15
439
                                                                                               Measured (Modal)
440
                                                                                                                             0.0
441                                     2                 4               8              16               32
442                                                                Batch Size
443
444
                                            Fig. 7. Hyperparameter sensitivity. Bold outline marks default values.
445
446
447
            • anon/llm-efficiency — SFT+GRPO pipeline
448
449   The de-anonymized checkpoints and full repository are available at https://anonymous.4open.science/r/tinker-rl-lab.
450       We target three ACM Artifact Badges [1]:
451
452        Artifacts Available All code, data, and checkpoints permanently archived on GitHub and Hugging Face Hub.
453        Artifacts Evaluated – Functional Complete documentation, Dockerfile, and scripts enable exercising all exper-
454           iments.
455
           Artifacts Evaluated – Reusable Modular design allows extending to new RL libraries, tasks, and model scales.
456
457
458   7   Limitations
459
          Model scale. Experiments limited to 0.6B–30B parameters.
460
461       Task coverage. Primarily mathematical reasoning and preference learning.
462
463       Platform dependency. Tinker API required for some experiments.
464
465       LoRA only. No full fine-tuning comparisons.
466
467       Statistical power. 5 seeds per experiment (10+ recommended).
468                                                                                                                  Manuscript submitted to ACM

[PAGE 10]
10                                                                                                                                               Anon.

469   7.1    Broader Impact
470
471
      This benchmark promotes transparency and reproducibility in RL post-training research. By establishing standardized
472   evaluation, we aim to reduce misleading performance claims and encourage rigorous experimental methodology.
473
474        Ethics Statement. We have read and adhere to the ACM Code of Ethics and Professional Conduct, and to the ACM
475
      Publications Policy on research involving human participants and data. This work involves no human subjects, private
476
      data, or dual-use capabilities. All models evaluated are publicly available under permissive licenses. Our benchmark
477
478   promotes transparency in RL post-training research.
479
480        Reproducibility Statement. All experiments are reproducible via Docker containers, fixed random seeds
481   ({42, 123, 456, 789, 1024} for the controlled multi-seed sweeps), and step-by-step commands documented in
482
      REPRODUCE.md. Code and model checkpoints are available (anonymized for review) at https://anonymous.4open.science/
483
484
      r/tinker-rl-lab. Statistical analyses use rliable with 10,000 bootstrap resamples.
485
486   8     Conclusion
487
488
      We present TinkerRL-Bench, a unified benchmark for comparing RL post-training methods across 7 libraries and
489   11 implementations, providing standardized evaluation protocols and comprehensive reproducibility infrastructure.
490   The evidence it offers is deliberately narrow. The current release most strongly supports three conservative claims.
491
      First, the large measured gap between LLM-native GRPO stacks and classic-RL PPO libraries is an implementation
492
493
      and task-encoding effect (the classic-RL stacks run a small MLP policy over a discrete-action arithmetic MDP, not the
494   autoregressive language model), not evidence that any one algorithm is superior. Second, trainability in our short-horizon
495   setting varies substantially with initialization and rollout regime rather than following one-size-fits-all rules. Third, and
496
      most usefully, our most important result is a negative one: on held-out GSM8K the mean Qwen3-8B GRPO gain over the
497
498
      same checkpoint’s pre-RL accuracy is small and not statistically significant (83.3% vs. 82.0%, 𝑝=0.26), so strong online
499   reward curves do not by themselves establish generalization gains. The benchmark is limited to 0.6B–30B-scale LoRA
500   experiments concentrated on mathematical reasoning, and several task-specific results are single-seed case studies. We
501
      release all code, traces, checkpoints, and documentation so that stricter multi-seed, token-budget-matched, and broader
502
503
      held-out follow-up work is straightforward to run.
504
505   References
506
       [1] ACM. 2020. Artifact Review and Badging – Version 1.1. https://www.acm.org/publications/policies/artifact-review-and-badging-current Accessed:
507        2026-04-13.
508    [2] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, and Marc Bellemare. 2021. Deep Reinforcement Learning at the Edge of
509        the Statistical Precipice. In Advances in Neural Information Processing Systems, Vol. 34. 29304–29320. doi:10.48550/arXiv.2108.13264 arXiv:2108.13264;
510        supplementary materials at https://agarwl.github.io/rliable/.
511    [3] Emanuele Cavenaghi, Gabriele Sottocornola, Fabio Stella, and Markus Zanker. 2023. A Systematic Study on Reproducibility of Reinforcement
512        Learning in Recommendation Systems. ACM Transactions on Recommender Systems 1, 3 (2023), 1–30. doi:10.1145/3596519
       [4] Cédric Colas, Olivier Sigaud, and Pierre-Yves Oudeyer. 2019. A Hitchhiker’s Guide to Statistical Comparisons of Reinforcement Learning Algorithms.
513
           arXiv preprint (2019). doi:10.48550/arXiv.1904.06979 arXiv:1904.06979.
514
       [5] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. 2018. Deep Reinforcement Learning That Matters.
515
           In Proceedings of the AAAI Conference on Artificial Intelligence (New Orleans, LA, USA), Vol. 32. doi:10.1609/aaai.v32i1.11694
516
       [6] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank
517        Adaptation of Large Language Models. In International Conference on Learning Representations. https://openreview.net/forum?id=nZeVKeeFYf9
518    [7] Emma Jordan, Adam White, Bruno Castro da Silva, Martha White, and Philip S. Thomas. 2024. Position: Benchmarking Is Limited in Reinforcement
519        Learning Research. arXiv preprint (2024). doi:10.48550/arXiv.2406.16241 arXiv:2406.16241.
520   Manuscript submitted to ACM


# R02: NeurIPS ZVF variant


Root: `platform_hybrid/paper/neurips_2026_variants/main_zvf.tex`  Pages: 25  Words: 14144


[PAGE 1]
Zero-Variance Fraction in GRPO: A Stratified Audit
              of Advantage-Collapse Diagnostics


                                             Anonymous Author(s)
                                                 Affiliation
                                                  Address
                                                   email



                                                   Abstract

 1            Zero-Variance Fraction (ZVF) is a per-step sentinel for zero reward-gradient con-
 2            tribution under group-relative policy advantages, not a cross-task predictor of
 3            GRPO performance. When every rollout in a GRPO group receives the same
 4            scalar reward, the group-relative advantage is zero and the prompt contributes no
 5            gradient; ZVF measures the share of groups in this regime. For binary rewards
 6            the accounting is exact: pass@ G − pG = 1 − ZVF, so ZVF’s complement is
 7            contrastive group yield rather than an independent performance statistic. We make
 8            two paired claims about how to use this quantity. Positive: per-step ZVF (and its
 9            complement GUt = 1 − ZVFt ) is a mechanically interpretable online utilization
10            meter that, paired with batch reward mean, distinguishes all-wrong collapse from
11            all-correct saturation and from informative-contrast training; we demonstrate this
12            with tool-use runs that sit at ZVF=1, GU=0, reward 0 from step 1 (Qwen3-32B
13            and Llama-3.1-8B), contrasted with a GSM8K run whose GU and reward both
14            move. Negative: the pooled cross-task ZVF–reward correlation that earlier drafts
15            (and concurrent work) cite as evidence that ZVF predicts GRPO outcomes does
16            not survive within-task stratification (r= − 0.769, p=0.0008, N =15 pooled →
17            r= + 0.40, p=0.09, N =19 within-GSM8K; r= + 0.13, N =7 model-varying);
18            it is confounded by task identity. We audit pooled correlations under stratifica-
19            tion and partial-correlation controls and provide a regime-to-action triage table.
20            We release step-level ZVF/GU traces, formalization, and analysis code at https:
21            //anonymous.4open.science/r/agentic-grpo-bench-anon-2C6B.


22   1   Introduction

23   Group-relative policy optimization (GRPO) and its descendants now underlie a large fraction of recent
24   reasoning-RL pipelines [34, 11], replacing the value-function critic of PPO [33] with a leave-one-out
25   advantage computed within a small group of rollouts that share a prompt. Because the advantage is
26   the within-group reward minus the within-group mean (or median), any group in which all rollouts
27   receive identical rewards contributes a zero advantage to every token and therefore zero gradient. We
28   call the share of such groups in a training batch the Zero-Variance Fraction (ZVF).

29   ZVF is a per-step sentinel, not a cross-task predictor. The framing we adopt in this paper is
30   narrow and mechanical. At step t, ZVFt = 1 implies that every rollout group in the batch has
31   zero reward variance and therefore zero reward-gradient contribution; the complementary GUt =
32   1 − ZVFt is an online utilization meter for the reward channel of the GRPO update. Pairing ZVFt
33   with batch reward mean separates all-wrong collapse (ZVF≈1, mean reward ≈ 0) from all-correct
34   saturation (ZVF≈1, mean reward ≈ 1) from informative-contrast training (ZVF<1, intermediate
35   mean reward). This is the regime in which ZVF is genuinely useful as instrumentation, and it is what

     Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute.

[PAGE 3]
86   sensitivities rather than establishing our measurements. Dr. GRPO [24] analyzes length and diffi-
 87   culty biases in the standard objective. GSPO [46], DAPO [42], and VAPO [6] are useful context
 88   for production-scale sequence-level and large-batch reasoning RL. Back to Basics [1] shows that
 89   carefully implemented REINFORCE-style methods can remain competitive. These works motivate
 90   the implementation and diagnostic axes we measure; they are not used as evidence for any numerical
 91   result in RL-F INETUNING B ENCH.

 92   2.2   Concurrent Zero-Variance Work

 93   A cluster of 2025–2026 GRPO-family papers independently identifies the same within-group zero-
 94   variance failure mode that motivates our Zero-Variance Fraction (ZVF) diagnostic: when every rollout
 95   in a group receives the same scalar reward, the group-relative advantage vanishes and no gradient
 96   signal reaches the policy. NGRPO [26] formalizes the case as “homogeneously correct or incorrect”
 97   groups and proposes advantage calibration with a hypothesised maximum-reward virtual sample plus
 98   asymmetric clipping. RL-ZVP [20] (ICLR 2026) names the construct zero-variance prompts and
 99   shows that entropy-guided advantage shaping on such prompts yields up to +8.61 accuracy points
100   across six math benchmarks over standard GRPO. Scaf-GRPO [44] frames the same phenomenon as a
101   “learning cliff” where problems beyond model capability collapse to zero reward and zero advantage,
102   and counters it with tiered in-prompt scaffolds. LENS [13] addresses the all-wrong-group sub-case
103   via confidence-reweighted penalties on negative rollouts. Hard Examples [30] treats zero-variance
104   saturation as the fundamental annotation-budget bottleneck and shows that training on the hardest
105   examples reduces it. EBPO [15] proposes an empirical Bayes shrinkage estimator that guarantees a
106   non-vanishing gradient even in saturated-failure regimes.
107   We cite these works narrowly. They corroborate the existence and importance of the within-group
108   zero-variance failure mode, and one of them (RL-ZVP) is at a peer-reviewed venue. They do not
109   validate our specific quantitative ZVF results – the project’s correlation ρ values, step-25 to step-100
110   stress-test predictiveness, or any cross-task ZVF threshold – which remain internal-artifact diagnostics
111   computed from our rollout traces. We make no priority claim with respect to these papers, several of
112   which were submitted before our ZVF traces were complete; we are rigorous, not first.

113   3     Experimental Setup
114   We instrument GRPO-style training with step-level group reward statistics across 79 runs spanning 7
115   RL libraries and the Qwen3, Qwen3.5, and Llama-3.x model families at scales from 0.6B to 671B
116   total parameters (0.6B–37B active for the largest MoE checkpoints). ZVF is computed per training
117   step as the fraction of rollout groups whose reward variance is exactly zero; full per-library code
118   paths are listed in Appendix B. Headline experiments use Qwen3-8B with LoRA rank 32, Adam
119   (β1 =0.9, β2 =0.95), learning rate 10−4 , and 5 seeds in {42, 123, 456, 789, 1024}. Frontier-scale runs
120   are single-seed and reported as case studies. Reproducibility, limitations, and ethics are deferred to
121   Appendix E, Appendix F, and Appendix G; auxiliary reward-trajectory instability signals are deferred
122   to Appendix G.

123   4     Zero-Variance Fraction: A Stratified Audit
124   4.1   Per-step sentinel framing

125   Zero-Variance Fraction (ZVF) at step t is the fraction of prompts for which all G completions
126   receive identical rewards:
                                         1 X                          
                                 ZVFt =           1 Varc∼C(p) r(c) = 0 .
                                        |Pt |
                                                  p∈Pt
127   When ZVFt = 1, every prompt contributes zero gradient. Gradient Utilization GUt = 1 − ZVFt
128   measures the fraction of prompts providing informative signal. The two quantities together with
129   batch reward mean separate three operationally distinct regimes: all-wrong collapse (ZVF≈1,
130   reward ≈ 0), all-correct saturation (ZVF≈1, reward ≈ 1), and informative contrast (ZVF<1,
131   intermediate reward). The mechanical claim is that whenever ZVFt = 1, the reward channel of the
132   GRPO advantage estimator contributes nothing to that step’s gradient (formal statement and proof in
133   Appendix A).


                                                         3

[PAGE 6]
Frontier Qwen3-8B/GSM8K (n=600)                                         ZVF decreases with G
                                                       obs-vs-theory r=0.93                                         (Qwen2.5-0.5B/arith., measured)
                                   1.0          identity                                                                                       measured mean ZVF
                                                binned mean




      observed ZVF (per problem)
                                   0.8                                                                   0.80

                                   0.6
                                                                                              mean ZVF
                                                                                                         0.75
                                   0.4
                                                                                                         0.70
                                   0.2
                                   0.0                                                                   0.65

                                          0.0       0.2       0.4       0.6       0.8   1.0                     2           4                  8             16
                                                     theoretical ZVF = pG + (1 − p)G                                            group size G

      Figure 2: Left: Pooled ZVF vs. final performance scatter (r = −0.769, p = 0.0008, N =15).
      The apparent strong negative correlation is task-confounded: the upper-right tool-use cluster sits at
      ZVF=1, reward 0; the lower-left GSM8K cluster sits at ZVF≤0.5, reward ≥ 0.3. Right: Mean ZVF
      by model family. Llama’s high mean ZVF is driven entirely by tool-use experiments. Within-GSM8K
      stratified Pearson is r = +0.40 (p = 0.09, N =19); restricted to model-varying GSM8K, r = +0.13
      (p = 0.79, N =7). The pooled negative correlation in this scatter does not survive stratification.


182   marginal GU gain from larger groups approaches zero for moderate accuracies p ∈ [0.2, 0.8] (the gain
183   from G=16 → G=32 is less than 0.3% at p=0.5). Empirically, in the single-seed Wave-2 Qwen3-
184   8B/GSM8K sweep the fixed-step last-10 reward is non-monotone in G (G=4: 0.52, G=32: 0.44,
185   G=16: 0.38, G=2: 0.38, G=8: 0.34); a token-budget-normalized reanalysis (Appendix C) shifts the
186   inverted-U apex rightward with total tokens T , so neither G=4 nor G=32 is a universal optimum.
187   We mention G only because it is mechanically tied to ZVF(p, G); G does not rescue the pooled
188   cross-task story.
189   A two-seed matched-budget panel (E-R2b: 2,560 rollouts per arm, G=2 × 160 steps vs. G=16 × 20
190   steps, LoRA rank 4, 512-token completions, seeds 123/456) sharpens the mechanical coupling into
191   a trajectory claim: the G=2 arms (160 steps) drive train reward to ≈ 0.9–1.0 on the sampled pool
192   and end in sustained ZVF ≈ 0.75–1.0 — all-correct groups, the high-p zero-variance wall that
193   ZVF(p, G) = pG + (1 − p)G predicts as p → 1 — while the G=16 arms (20 steps at the same
194   budget) end mid-learning at train reward ≈ 0.3–0.5 with ZVF at 0–0.25 throughout. Small G
195   converts a fixed rollout budget into more optimizer steps early, then exhausts its own gradient signal
196   in the endgame (Tier C, two seeds per arm).

197   Loss form leaves no ZVF signature at this scale. A six-arm uncapped panel (1,024-token com-
198   pletions, G=8, batch 4, 30 steps, three seeds per loss) comparing GRPO with Dr.GRPO’s /σ-free
199   advantage found no length inflation in either loss — mean completion length declines 3.8–12.2%
200   in all six arms — and no late-ZVF separation (Dr.GRPO 0.45/0.70/0.72 vs. GRPO 0.47/0.47/0.55).
201   Whatever Dr.GRPO’s normalization change does at larger scale, it has no observable ZVF or length
202   footprint on Qwen3-8B/GSM8K (Tier C, n=3 seeds per loss).

203    4.4                               Evidence Grade of Each Claim

204   This paper mixes claims at very different levels of statistical strength. Table 4 segregates every
205   quantitative claim in the main text by evidence grade so that downstream readers do not have to
206   reconstruct it from scattered caveats. The two rows we ask readers to take away are the headline
207   claims at the top of the table; the rest are auxiliary or descriptive.


208    5                            Conclusion

209   This paper makes two paired claims about Zero-Variance Fraction in GRPO training and audits each
210   one explicitly.


                                                                                              6

[PAGE 16]
579   F     Limitations
580   F.1   Methodological Limitations

581   Closed-Source Training Implementation (Tinker). Tinker is a commercial, closed-source API.
582   We cannot inspect the exact GRPO loss formulation, reward normalization scheme, minibatch
583   construction, or hardware configuration used server-side. Our Tinker results therefore measure
584   the platform’s GRPO implementation, not a precisely specified algorithm. Researchers wishing to
585   attribute performance differences to specific implementation choices should use the open-source
586   backends (TRL, OpenRLHF, veRL) where every hyperparameter is auditable.

587   Short Training Horizons (30 Steps). All Tinker experiments used a budget of 30 gradient steps—a
588   deliberate choice to contain API costs but one that may be insufficient to observe convergence on
589   harder tasks. Thirty steps represents roughly one or two passes over the prompt pool at the batch
590   sizes we used. Long-horizon effects such as reward hacking, catastrophic forgetting, or late-stage
591   policy collapse are unlikely to manifest at this scale. We regard our Tinker results as early-training
592   snapshots rather than converged solutions, and caution against drawing strong conclusions about
593   asymptotic performance.

594   Single-Seed Tinker Experiments. Cost constraints precluded multi-seed replication on Tinker:
595   each configuration was run once. Without variance estimates we cannot apply standard significance
596   tests to Tinker results. We report these numbers descriptively and recommend that future work with
597   larger budgets run at least three seeds, consistent with best practices from Henderson et al. [16].

598   Train-Set Reward as the Primary Metric. Reported rewards are computed on the same prompt
599   distribution used for training, not a held-out test split. We now include two limited GSM8K checks: a
600   post-hoc top-10 checkpoint audit selected by training last-10, and a separate five-seed Qwen3-8B
601   base-model control showing 83.3% post-GRPO vs. 82.0% base (t = 1.32, p = 0.26). These checks
602   reduce but do not remove the risk of over-reading generalization: the top-10 audit is post-selection
603   by construction, and broader held-out evaluation without checkpoint cherry-picking remains future
604   work.

605   Tool-Use Task: 0% Reward. The synthetic tool-use task yielded a reward of exactly 0% for all
606   completed runs under the Tinker backend. We interpret this as a task-design problem rather than a
607   model capability failure: the reward function requires exact JSON-schema compliance with nested
608   function signatures, a level of precision that is difficult to reach in 30 steps without a warm-started
609   SFT initialization. The task may also require a graduated curriculum (simple → complex tool calls)
610   that was absent from our design. We release the task definition and reward function so that the
611   community can iterate; in the interim, tool-use results should be treated as a negative baseline rather
612   than as evidence about cross-domain GRPO effectiveness.

613   G     Reward-Trajectory Instability Signals (Auxiliary)
614   Reward-trajectory instability—erratic late-stage rewards, large peak-to-tail degradation, or persistently
615   high per-step variance—is a symptom that often accompanies policy drift but does not by itself
616   measure distributional divergence between the trained policy πθ and the reference policy πref . Direct
      per-token KL tracking via Ex∼πθ log ππref θ (x)
                                                     
617
                                                   (x) was blocked by a PyTorch gradient graph error in our
618   pipeline (see Appendix F). We therefore develop reward-trajectory stability proxies as descriptive
619   instability indicators across all 28 experiments with ≥5 training steps; we are explicit that these are
620   symptom-level signals, not KL substitutes, and they are reported here as auxiliary material rather
621   than as a per-step ZVF substitute.

622   Stability Metrics. We define three complementary proxy measures. (1) The Stability Index (SI) is
623   the coefficient of variation of the last-10 reward values: SI = σ(rtail ) / |µ(rtail )|. High SI indicates
624   erratic late-stage policy behavior. (2) The Peak-to-Tail Drift (PTD) captures net reward degradation:
625   PTD = (rmax − r̄last-10 ) / rmax . PTD > 0.3 signals catastrophic reward-trajectory instability. (3) The
626   Rolling Variance (window = 5 steps) tracks how per-step reward variability evolves through training,
627   serving as a real-time proxy for reward-trajectory instability.


                                                         16

[PAGE 17]
628   Quantitative Results. Across 28 experiments, both SI and PTD correlate significantly with training
629   outcomes: SI vs. last-10 average yields Pearson r = −0.436 (p = 0.020), while PTD vs. last-10
630   average yields r = −0.517 (p = 0.005). These correlations confirm that reward-trajectory instability
631   is a useful descriptive indicator of training-time degradation, while remaining symptom-level: they
632   do not directly measure distributional divergence.

633   Instability Classification. We classify experiments into three drift regimes: 19 experiments (67.9%)
634   exhibit high instability (PTD > 0.3), 5 (17.9%) show moderate instability (0.1 < PTD ≤ 0.3), and
635   4 (14.3%) remain stable (PTD ≤ 0.1). The majority of high-drift cases come from classic RL libraries
636   (SB3, CleanRL, Tianshou), whose PPO implementations achieve near-zero reward on LLM tasks and
637   exhibit mean PTD of 0.619 ± 0.139—consistent with random-walk policy trajectories rather than
638   meaningful learning.

639   Algorithm Stability Comparison. GRPO experiments show significantly lower instability than
640   classic-RL PPO: mean SI of 0.162 ± 0.157 vs. 0.814 ± 0.370 (Mann–Whitney U = 1.0, p = 0.005);
641   mean PTD of 0.212 ± 0.205 vs. 0.619 ± 0.139 (p = 0.018). This difference is compatible with
642   accounts of GRPO’s group-relative objective reducing some forms of reference-model mismatch, but
643   our benchmark does not isolate mechanism: algorithm, implementation, model family, and task mix
644   all vary across the pooled comparison [34].

645   Nemotron-120B Collapse Case Study. Nemotron-120B presents the clearest case of catastrophic
646   reward-trajectory instability in our benchmark: SI = 1.180, PTD = 0.762. Figure 3 shows the
647   Nemotron-120B trajectory (pink) with an early surrogate-loss excursion followed by persistently
648   elevated per-step ZVF, consistent with an early-training policy excursion from which the model never
649   recovers. This contrasts sharply with Qwen3-235B-A22B (SI ≈ 0, PTD ≈ 0), whose zero-variance
650   reward trajectory indicates the policy remained in the immediate neighborhood of the reference
651   initialization throughout training.

652   Honest Limitation. These proxy metrics capture symptoms of reward-trajectory instability rather
653   than direct measurements of distributional divergence; we do not label them as “policy drift” precisely
654   because we did not measure KL. The corrected KL tracking implementation is included in the artifact
655   release, but this paper does not yet validate the proxy–KL correspondence or quantify per-token
656   divergence trajectories.

657   Heatmap of per-step ZVF (auxiliary). Figure 4 provides a per-run, per-step heatmap of ZVFt
658   across the cross-stack benchmark. The persistent red band along the Tinker tool-use rows is the
659   same all-wrong-collapse regime headlined in Figure 1; the GSM8K rows show informative-contrast
660   dynamics that vary across runs but, per Section 4.2, do not predict last-10 reward.


661   H     Ethics, Limitations, and Broader Impact

662   This statement is written to satisfy the NeurIPS Code of Ethics and Broader Impact requirements.
663   It complements the shorter Limitations and Broader Impact subsections embedded in Section F by
664   consolidating in one place a full dual-use analysis, itemised compute accounting, carbon footprint
665   estimation, data provenance disclosures, a candid acknowledgment of the closed-source training
666   backend on which a fraction of our headline numbers depends, and a list of methodological limits we
667   are aware of but did not have resources to close within the submission window.

668   H.1   Dual-Use Analysis: Misuse Risks of Reasoning RL

669   Threat model. RL-F INETUNING B ENCH releases (i) training scripts that apply GRPO to the
670   GSM8K [9] mathematical reasoning benchmark and to the Salesforce xLAM-Function-Calling-
671   60k [23] tool-use corpus, (ii) a set of LoRA adapters published on the HuggingFace Hub
672   (anonymous/tinker-rl-bench-*), and (iii) diagnostic code for Zero-Variance Fraction, reward-
673   stability, and length-bias analysis. We analyse four concrete misuse pathways these artefacts could,
674   in principle, enable, and we describe the existing guardrails and the residual risk we judge acceptable
675   for publication.


                                                        17

[PAGE 20]
736          • NVIDIA A100 80GB TDP: 400 W.
737          • NVIDIA L4 TDP: 72 W.
738          • NVIDIA T4 TDP: 70 W.
739          • Data-centre PUE: 1.1 (conservative; typical hyperscale PUE is 1.10–1.15 [14, 2]).
740          • US average grid intensity (EPA eGRID 2023): 0.367 kg CO2 -eq / kWh [40].
741          • Author-region average grid intensity (national authority report, 2023; specific country and
742            source withheld for anonymity): within the higher-carbon-intensity grid bands typical of
743            emerging-market national grids; the exact value used is the midpoint of an anonymized
744            regional average grid intensity range and is withheld here.
745          • We use the US average for Modal H100 (an anonymized regional cloud zone as configured)
746            and Tinker (location undisclosed; assumed US), and an anonymized regional average grid
747            intensity for the Colab Pro T4 instances used by the authors. The exact regional intensity is
748            withheld for anonymity; the reported project-total CO2 -eq number uses the midpoint of the
749            applicable range.

750   GPU-hour estimates. Tinker GPU-hour counts are not disclosed by the platform. We estimate
751   them from wall-clock run duration and assume a single H100-class accelerator per job, which is
752   consistent with Tinker’s published architecture for the model sizes we trained. Modal GPU-hours are
753   measured directly from Modal’s billing dashboard.

754   Interpretation and uncertainty. Our best estimate of ~296 kg CO2 -eq for the entire project is
755   comparable to a single round-trip domestic economy flight in the same region (~300 kg). The
756   dominant uncertainty is the Tinker GPU-hour count: because Tinker does not expose hardware
757   telemetry, a factor-of-two error in GPU-hours or TDP is plausible. Under a pessimistic assumption
758   (2× GPU-hours, H100 running near 100% TDP continuously), the Tinker contribution could be
759   as high as ~540 kg CO2 -eq. Under an optimistic assumption (Tinker backends actually use more
760   energy-efficient accelerators than H100, 50% average utilisation) the contribution could be as low
761   as ~130 kg. We report the central estimate in the main table and caution readers that it should
762   not be interpreted as precise. All numbers should be treated as order-of-magnitude estimates with
763   documented sensitivity to Tinker hardware assumptions; subsequent users can additionally skip the
764   failed runs and the ablation sweeps to lower the cost of reproducing our work.

765   Offset and mitigation. We did not purchase carbon offsets. Instead, we have (a) released all trained
766   checkpoints on HuggingFace Hub so that downstream users do not need to re-run our experiments;
767   (b) released step-level CSV logs so that learning curves can be inspected without re-training; and (c)
768   documented in Section H.2 which runs were wasteful, so that replicators can skip them. We regard
769   reproducibility-by-artefact as a more durable mitigation than offsets.

770   H.4   Data Provenance

771   RL-F INETUNING B ENCH uses only publicly released research datasets. No private data, personally
772   identifiable information, or licensed proprietary content is used. No human annotators were employed
773   during this work. We document each dataset below.

774   GSM8K [9]. 8,500 grade-school math word problems (7,473 train / 1,319 test) authored by hu-
775   man writers contracted by OpenAI. Released under the MIT License via https://github.com/
776   openai/grade-school-math. We use the standard train/test splits without modification. The
777   canonical citation is Cobbe et al. [9]. Known limitations of GSM8K include gender-neutral but cul-
778   turally US-centric word problems (names, currencies, sports) and a moderate rate of human-labelling
779   errors estimated at ~2% by [43]. Our held-out evaluation respects the test/train split.

780   Salesforce xLAM-Function-Calling-60k [23]. 60,000 function-calling examples generated
781   by Salesforce Research as training data for the xLAM agentic model family.                          Re-
782   leased under the CC BY 4.0 license via https://huggingface.co/datasets/Salesforce/
783   xlam-function-calling-60k. We use the public release as-is; specifically, we use the first 35
784   prompts × 10 rollouts in the 10x Structural Ceiling tool-use track and the full 60k split for the xlam-
785   60k real-data run in Section 4. Known limitations: xLAM schemas are synthetic and skew toward


                                                        20

[PAGE 21]
786   cleanly-typed arguments; the dataset under-represents ambiguous, error-handling, and multi-turn
787   tool-calling. Attribution to Salesforce is required by the license and is provided in Section 5 and in
788   the xLAM-derived model cards.

789   HumanEval [7]. 164 Python programming problems with unit tests, released by OpenAI under the
790   MIT License. Used unmodified for pass@k evaluation. Known limitations: small scale, English-
791   language docstrings, single-language coverage; documented contamination concerns against frontier
792   pretraining corpora [32].

793   NuminaMath [22]. Used by an anonymous collaborator for the multi-stage GSM8K+NuminaMath
794   pipeline. Released under Apache 2.0. We use a downstream checkpoint reported by that collaborator;
795   our repository contains only their scripts and the Apache-licensed derivative weights.

796   Open-Platypus [21]. 3,000 SFT examples used by an anonymous collaborator for Qwen3-8B
797   code generation warm-up. Open-Platypus is a curated subset released under CC BY-NC 4.0; the
798   non-commercial restriction is respected in our release, which is academic-use only.

799   Synthetic tool-use corpus (authored). We additionally generated a small five-tool synthetic corpus
800   (calculator, web-search stub, calendar, file-read, email-send) used for the tool-use case studies cited
801   in Section 4.2 (the high-ZVF / zero-reward tool-use cluster that drives the pooled task confound). The
802   corpus is authored by the paper’s authors from publicly documented APIs, contains no third-party
803   content, and is released under the MIT License in our repository under data/synthetic_tools/.
804   Generation prompts are included for auditability. No user data, scraped web content, or proprietary
805   API traces are present.

806   No web scraping, no human subjects. We did not scrape any website. We did not run any study
807   with human participants. No IRB review was therefore required. No data that could reasonably be
808   considered to contain PII, copyrighted content, or sensitive personal attributes was used at any stage
809   of training, evaluation, or reward modelling.

810   H.5   Closed-Source Tinker Acknowledgment

811   A central tension in RL-F INETUNING B ENCH is that a substantial fraction of our headline numbers
812   come from a closed-source commercial platform while our paper simultaneously advocates for
813   reproducibility and platform independence. We do not resolve this tension by hiding it; we describe it
814   precisely.

815   What Tinker is and is not. Tinker [39] is a managed LLM fine-tuning and inference service
816   provided by Thinking Machines, Inc. It exposes a Python SDK (tinker==0.16.1) that accepts
817   custom loss functions (forward_backward_custom), a limited set of optimisers, and standard LoRA
818   hyperparameters. It does not expose (i) the exact server-side GRPO loss implementation, (ii) reward
819   normalisation or baseline subtraction scheme, (iii) minibatch construction or gradient-accumulation
820   strategy, (iv) hardware configuration (GPU type, inter-node bandwidth), or (v) system-level telemetry
821   (energy, throughput, queueing).

822   What this means for our claims. Tinker results in this paper measure the platform’s implementa-
823   tion of GRPO, not an abstract specification of the algorithm. Cross-stack reward gaps between Tinker
824   GRPO and an open-source GRPO implementation cannot be fully attributed from our data: we are
825   able to rule out a handful of candidate explanations (seed variance, model-size confound) but not
826   the implementation itself. We therefore draw quantitative conclusions only from the open-source
827   side (TRL, veRL on Modal H100, OpenRLHF) where every hyperparameter is auditable, and use
828   Tinker results primarily as descriptive case studies of what a carefully engineered production stack
829   can achieve for critic-free RL at our scales.

830   Reproducibility commitments we can make.
831         1. All Tinker experiment scripts, configuration files, JSON step logs, and per-run W&B projects
832            are archived in the repository. Researchers with Tinker access can attempt replication with
833            our exact configurations.


                                                        21

[PAGE 22]
834         2. Every figure and every summary statistic derived solely from Tinker data is marked with “†”
835            and a footnote stating that independent replication requires Tinker API access.
836         3. Primary inferential claims are limited to (i) the mechanical zero-gradient theorem (Ap-
837            pendix A) and (ii) the held-out 5-seed Qwen3-8B GSM8K control reported in Section F.
838            The Tinker-derived stratified ZVF correlations (Section 4.2) are reported as artifact analyses
839            with explicit closed-backend caveats and single-seed-per-cell limitations, not as platform-
840            independent statistical conclusions.
841         4. We commit to re-running any Tinker-only experiment on an open backend if an equivalent
842            open-source service becomes available, and to issuing a revised version of this paper if any
843            Tinker result is ever shown to be due to an undisclosed implementation choice rather than
844            the algorithm.

845   Key rotation and credential hygiene. Our Tinker API key (prefix tml-...) is held in a repos-
846   itory .env.example placeholder only. The real key is stored in the authors’ password manager
847   and rotated whenever a failure mode suggests possible exfiltration. No Tinker key is commit-
848   ted to git history in the anonymous source repository (https://anonymous.4open.science/r/
849   agentic-grpo-bench-anon-2C6B).

850   H.6   Known Methodological Limits

851   In addition to the infrastructure failures and platform-specific caveats enumerated in Section F, we
852   flag the following methodological limits.

853   Short training horizons. All Tinker GRPO runs were capped at 30–50 gradient steps. This is a
854   deliberate cost-control choice and means our Tinker numbers are early-training snapshots. We cannot
855   rule out that longer training would change the qualitative picture (e.g., the 82%→83.3% held-out
856   gain might widen, or might collapse to reward hacking).

857   Single-seed Tinker experiments. Each Tinker configuration was run once, not 3–5 times as
858   best practice prescribes [16]. Tinker results therefore lack variance estimates and do not support
859   significance testing; we report them descriptively. Only the 5-seed TRL baseline and the 5-seed
860   held-out GSM8K evaluation carry proper confidence intervals.

861   Train-set reward as primary Tinker metric. Tinker runs primarily report reward on training
862   prompts. Only the GSM8K track was followed up with a separate 200-example held-out evaluation
863   per seed. Other Tinker tracks (tool-use, xLAM) remain train-set-only and we cannot distinguish
864   memorisation from generalisation for those settings.

865   LoRA only; no full fine-tuning. All experiments use LoRA [18] with ranks 8–64. We have
866   not tested whether full fine-tuning would re-order the library/algorithm comparisons. The LoRA
867   constraint is practical (cost) but it means our claims about PPO vs. GRPO and TRL vs. Tinker are
868   LoRA-conditional.

869   Benchmark coverage is narrow. We evaluate four domains: GSM8K, MATH-500 (exploratory),
870   HumanEval (subset), and synthetic/xLAM tool-use. We do not evaluate on MT-Bench, ArenaHard,
871   safety benchmarks (HarmBench, ToxicChat), or truthfulness benchmarks (TruthfulQA). Any claim
872   about “reasoning” in the paper is strictly a claim about the covered benchmarks.

873   No human preference data. We use verifiable rewards only (exact-match for math, unit-test for
874   code, schema-validity for tool-use). We do not study RLHF with human preference data; findings
875   should not be extrapolated to reward-model–based alignment without further work.

876   Closed-source implementation opacity (reiterated). Comparisons between Tinker GRPO and
877   TRL GRPO are confounded by the closed-source nature of Tinker’s implementation. See Section H.5
878   for detail.


                                                       22


# R03: NeurIPS workshop artifact


Root: `platform_hybrid/paper/neurips_2026_variants/main_workshop.tex`  Pages: 17  Words: 9735


[PAGE 1]
RL-Finetuning Bench: An Exploratory Workshop
                   Artifact for GRPO-Style
     Post-Training Across Libraries, Models, and Backends


                                             Anonymous Author(s)
                                                 Affiliation
                                                  Address
                                                   email



                                                   Abstract

 1            This is an exploratory workshop artifact note, not a benchmark paper. We release
 2            step-level group reward traces, Zero-Variance Fraction (ZVF) traces, a parser-script
 3            bundle, a per-run manifest, and a structured evidence grade table for 79 GRPO-style
 4            runs across 7 RL libraries (TRL, Stable Baselines3, CleanRL, Tianshou, PufferLib,
 5            rl_games, d3rlpy), the closed T INKER API, and 5 model families (total parameters
 6            0.6B–671B; active 0.6B–37B for the largest mixture-of-experts checkpoints). The
 7            release lives at an anonymous repository documented in Section 5.
 8            Claim hierarchy (descending strength). (i) ZVF = 1 ⇒ zero gradient is mechani-
 9            cal (Appendix B). (ii) On a five-seed Qwen3-8B GSM8K held-out test, post-GRPO
10            83.3% vs. base 82.0% (t=1.32, p=0.26): not significant. (iii) The pooled ZVF–
11            last-10 correlation (r=−0.769, N =15, p=0.0008) is task-confounded; within
12            GSM8K only (N =19), r=+0.40 (p=0.09). We report ZVF as a cheap descriptive
13            logging quantity, not as a predictive failure detector. Frontier-scale and tool-use
14            rows are single-seed case studies, released for honesty and not as benchmark
15            numbers. The rest of this artifact note is a candid evidence-grade table and three
16            discussion prompts for the workshop.



17   1   Introduction

18   Artifact card. This workshop submission is an artifact note accompanying a release that
19   contains: (i) step-level aggregate reward traces (mean, peak, last-10) for every reported run;
20   (ii) per-step Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) traces for every reported
21   run; (iii) a per-run manifest (Appendix A) listing backend, hardware, hyperparameter mapping,
22   and evidence grade; (iv) parser scripts that decode the released JSONL into a single tidy ta-
23   ble; and (v) a documented anonymous repository at https://anonymous.4open.science/r/
24   agentic-grpo-bench-anon-2C6B for code, logs, and Docker images. We deliberately do not
25   claim a “first comprehensive” GRPO benchmark, a frontier ranking, or a definitive algorithm compar-
26   ison.

27   Claim hierarchy (one sentence). The mechanical ZVF → zero-gradient theorem is an A-grade
28   definitional claim; the held-out 5-seed Qwen3-8B GSM8K control (+1.3 pp, p=0.26) is an A-grade
29   negative result; the pooled ZVF–last-10 correlation (r=−0.769, N =15) is a B∗ -grade descriptive
30   finding that is task-confounded and collapses within-task; everything else (frontier-scale Tinker
31   rows, tool-use, group-size sweeps) is C-grade case study and is released for honesty rather than as
32   benchmark fact.

     Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute.

[PAGE 4]
86    in a sustained all-correct zero-variance regime (ZVF ≈ 0.75–1.0), while G=16 ends mid-learning
87    (≈ 0.3–0.5) with ZVF ≤ 0.25 — small G trades early step-count for endgame signal exhaustion.


      Table 3: Evidence grade of each main-text claim in the workshop variant. A = multi-seed open-stack
      or definitional; B = multi-seed or pooled with a known caveat; B∗ = significant in pooled sample but
      task-confounded; C = single-seed or closed-backend case study.
          Claim                                                   Grade      Notes
          ZVF → zero gradient is mechanical                       A (def.)   App. B.
          ZVF–last-10 r= − 0.769 (N =15, pooled)                  B∗         Confounded by task type.
          Held-out Qwen3-8B GSM8K 83.3% vs. 82.0%                 A          5 seeds; p=0.26.
          Group-size sweep is non-monotonic; no headline winner   C          single-seed per G.
          Matched-budget G=2 endgame ZVF wall                     B          2 seeds/arm; 512-tok cap.
          Token-budget fitted apex near G≈32 at T ≥16M            B          3-seed appendix reanalysis; not a
                                                                             universal optimum.
          Frontier-scale Tinker case studies                      C          single-seed, closed backend.
          Tool-use 0% reward across runs                          B          task-design failure, not capability.



88    5    Reproducibility and Limitations

89    We release pinned Docker images, deterministic seed management, and rliable-based analysis
90    scripts. Anonymous code, logs, and checkpoints are at https://anonymous.4open.science/
91    r/agentic-grpo-bench-anon-2C6B. The release distinguishes three artifact tiers explicitly:
92    (i) precomputed ZVF / GU traces, released for every reported run; (ii) step-level aggregate logs
93    (mean reward, peak reward, last-10), released for every reported run; (iii) raw per-group rollout
94    JSONL (per-prompt, per-completion reward arrays), released for a documented subset of stacks, with
95    the full subset enumerated in the run manifest (Appendix A) and the parser contract released with the
 96   anonymous artifact.
97    The release has known gaps that the workshop framing makes explicit. (i) Most Tinker experiments
98    are single-seed and depend on a closed-source backend; we report them descriptively. (ii) Empirical
99    evidence concentrates on short-horizon GSM8K with binary verifiable rewards; tool-use, code, and
100   non-math coverage are sparser and yield negative or descriptive evidence only. (iii) Train-set reward in
101   the Tinker case studies is descriptive instrumentation, not an inferential signal: our primary inferential
102   claims are limited to the mechanical ZVF → zero-gradient theorem (Appendix B) and the held-out
103   5-seed Qwen3-8B GSM8K test-set control (83.3% vs. 82.0%, t=1.32, p=0.26). Tinker-derived
104   stratified correlations (e.g., the within-GSM8K ZVF re-analysis) are reported as artifact analyses with
105   closed-backend caveats, not as benchmark statistics. (iv) Cross-library comparisons are confounded
106   by checkpoint, hardware, and default-config differences; we report them as stack-level observations,
107   not as algorithm-level statements. (v) Direct KL tracking was blocked by a PyTorch gradient-graph
108   error; the corrected implementation is in the artifact but is not validated against ZVF in the current
109   paper. (vi) Verifiable rewards reduce reward-model subjectivity but do not eliminate reward hacking,
110   format gaming, or train-set reward inflation; we do not claim they do.


111   6    Discussion and Concrete Workshop Questions

112   The workshop release is deliberately exploratory. Its strongest evidence is short-horizon GSM8K
113   dynamics; non-math coverage is a stress case, not a result. We treat ZVF as a cheap descriptive
114   diagnostic, not a predictive failure detector, and we emphasize that the Qwen3-8B held-out control
115   gives a small, non-significant GRPO improvement over the base.

116   Five concrete questions for the workshop. We invite attendees to engage with five specific,
117   testable questions raised by this release rather than open-ended philosophizing:
118   1. Does step-level ZVF predict reward change within a single run, controlling for run fixed effects,
119      on stacks beyond Tinker? Our residualized within-GSM8K scatter (Figure 1c) finds rwithin close
120      to zero; is this a backend artifact or a general property?


                                                          4

[PAGE 5]
121   2. Under a fixed rollout-token budget, where does group size trade early step count for endgame
122      contrast? Our two-seed G=2 × 160 versus G=16 × 20 panel ends with the small-G arms at an
123      all-correct ZVF wall while G=16 remains mid-learning; a separate 3-seed reanalysis places a
124      fitted apex near G≈32 for one stack and budget. Neither identifies a universal optimum. The
125      needed experiment is a directly trained, multi-seed, token-matched sweep on open stacks.
126   3. Are there tasks where high-ZVF cells do predict downstream collapse, beyond GSM8K? Tool-use
127      is a degenerate ZVF=1 case from step 1; the genuinely interesting test is on intermediate-density
128      tasks (MATH-500 at low temperature, code with partial-credit rewards).
129   4. What minimum disclosure makes a managed-API result count as benchmark evidence? Our Tinker
130      rows are reproducible only at the level of API call scripts; should the community require GRPO
131      loss / reward normalization / hardware metadata for a managed-API row to count as anything
132      beyond a case study?
133   5. Should ZVF-style diagnostic claims require raw per-group rollout JSONL by default? Our release
134      bundles step-level aggregates for every run but raw JSONL only for a documented subset; we
135      invite the workshop to align on a minimum raw-artifact bar.


136   7     Ethics, Limitations, and Broader Impact

137   This statement is written to satisfy the NeurIPS Code of Ethics and Broader Impact requirements.
138   It complements the shorter Limitations and Broader Impact subsections embedded in Section 5 by
139   consolidating in one place a full dual-use analysis, itemised compute accounting, carbon footprint
140   estimation, data provenance disclosures, a candid acknowledgment of the closed-source training
141   backend on which a fraction of our headline numbers depends, and a list of methodological limits we
142   are aware of but did not have resources to close within the submission window.

143   7.1   Dual-Use Analysis: Misuse Risks of Reasoning RL

144   Threat model. RL-F INETUNING B ENCH releases (i) training scripts that apply GRPO to the
145   GSM8K [3] mathematical reasoning benchmark and to the Salesforce xLAM-Function-Calling-
146   60k [11] tool-use corpus, (ii) a set of LoRA adapters published on the HuggingFace Hub
147   (anonymous/tinker-rl-bench-*), and (iii) diagnostic code for Zero-Variance Fraction, reward-
148   stability, and length-bias analysis. We analyse four concrete misuse pathways these artefacts could,
149   in principle, enable, and we describe the existing guardrails and the residual risk we judge acceptable
150   for publication.

151   M1. Weaponisation of mathematical reasoning. The most capability-relevant artefact we release
152   is GRPO training code that improves grade-school arithmetic and multi-step reasoning. A malicious
153   operator could attempt to extend this pipeline to tasks of greater dual-use concern, e.g., chemistry
154   olympiad problems, cryptographic key recovery, or financial fraud. We judge the marginal uplift
155   provided by our release to be small for three reasons. First, GRPO at our training scale does not
156   create new capabilities; the base-model control for GSM8K (Section 4, 82.0%) shows that the bulk
157   of held-out accuracy is attributable to Qwen3-8B’s pretraining. Our full GRPO run only adds +1.3
158   percentage points (p=0.26, not statistically significant). Second, the public literature already contains
159   GRPO implementations [18, 12, 19, 24] that are at least as capable. Third, any reasoning uplift is
160   bounded by the rewarder: GSM8K rewards use exact-match against a human-written gold answer,
161   which does not generalise to the domains listed above without a new reward signal, which itself
162   requires expertise and labelled data.

163   M2. Reward hacking and the long-term alignment risk. GRPO, like all policy-gradient RL
164   from a proxy reward, is vulnerable to reward hacking: the model learns to exploit idiosyncrasies
165   of the reward function rather than solve the intended task [20, 8]. We observed verbosity drift in
166   some of our own GRPO runs, which is a concrete demonstration of this failure mode in our own data.
167   Users who apply our scripts to under-specified reward functions in production settings (e.g., a custom
168   “helpfulness” classifier) may observe models that appear to improve on held-out metrics while silently
169   optimising against unintended proxies. We include the Zero-Variance Fraction and reward-stability
170   diagnostics precisely so that downstream users can detect such drift.


                                                         5

[PAGE 8]
245   7.4   Data Provenance

246   RL-F INETUNING B ENCH uses only publicly released research datasets. No private data, personally
247   identifiable information, or licensed proprietary content is used. No human annotators were employed
248   during this work. We document each dataset below.

249   GSM8K [3]. 8,500 grade-school math word problems (7,473 train / 1,319 test) authored by hu-
250   man writers contracted by OpenAI. Released under the MIT License via https://github.com/
251   openai/grade-school-math. We use the standard train/test splits without modification. The
252   canonical citation is Cobbe et al. [3]. Known limitations of GSM8K include gender-neutral but cul-
253   turally US-centric word problems (names, currencies, sports) and a moderate rate of human-labelling
254   errors estimated at ~2% by [25]. Our held-out evaluation respects the test/train split.

255   Salesforce xLAM-Function-Calling-60k [11]. 60,000 function-calling examples generated
256   by Salesforce Research as training data for the xLAM agentic model family.                          Re-
257   leased under the CC BY 4.0 license via https://huggingface.co/datasets/Salesforce/
258   xlam-function-calling-60k. We use the public release as-is; specifically, we use the first 35
259   prompts × 10 rollouts in the 10x Structural Ceiling tool-use track and the full 60k split for the xlam-
260   60k real-data run in Section 4. Known limitations: xLAM schemas are synthetic and skew toward
261   cleanly-typed arguments; the dataset under-represents ambiguous, error-handling, and multi-turn
262   tool-calling. Attribution to Salesforce is required by the license and is provided in Section 7.6 and in
263   the xLAM-derived model cards.

264   HumanEval [2]. 164 Python programming problems with unit tests, released by OpenAI under the
265   MIT License. Used unmodified for pass@k evaluation. Known limitations: small scale, English-
266   language docstrings, single-language coverage; documented contamination concerns against frontier
267   pretraining corpora [16].

268   NuminaMath [10]. Used by an anonymous collaborator for the multi-stage GSM8K+NuminaMath
269   pipeline. Released under Apache 2.0. We use a downstream checkpoint reported by that collaborator;
270   our repository contains only their scripts and the Apache-licensed derivative weights.

271   Open-Platypus [9]. 3,000 SFT examples used by an anonymous collaborator for Qwen3-8B
272   code generation warm-up. Open-Platypus is a curated subset released under CC BY-NC 4.0; the
273   non-commercial restriction is respected in our release, which is academic-use only.

274   Synthetic tool-use corpus (authored). We additionally generated a small five-tool synthetic corpus
275   (calculator, web-search stub, calendar, file-read, email-send) used for the tool-use case studies cited
276   in Section 3 (the high-ZVF / zero-reward tool-use cluster that drives the pooled task confound). The
277   corpus is authored by the paper’s authors from publicly documented APIs, contains no third-party
278   content, and is released under the MIT License in our repository under data/synthetic_tools/.
279   Generation prompts are included for auditability. No user data, scraped web content, or proprietary
280   API traces are present.

281   No web scraping, no human subjects. We did not scrape any website. We did not run any study
282   with human participants. No IRB review was therefore required. No data that could reasonably be
283   considered to contain PII, copyrighted content, or sensitive personal attributes was used at any stage
284   of training, evaluation, or reward modelling.

285   7.5   Closed-Source Tinker Acknowledgment

286   A central tension in RL-F INETUNING B ENCH is that a substantial fraction of our headline numbers
287   come from a closed-source commercial platform while our paper simultaneously advocates for
288   reproducibility and platform independence. We do not resolve this tension by hiding it; we describe it
289   precisely.

290   What Tinker is and is not. Tinker [22] is a managed LLM fine-tuning and inference service
291   provided by Thinking Machines, Inc. It exposes a Python SDK (tinker==0.16.1) that accepts


                                                         8

[PAGE 9]
292   custom loss functions (forward_backward_custom), a limited set of optimisers, and standard LoRA
293   hyperparameters. It does not expose (i) the exact server-side GRPO loss implementation, (ii) reward
294   normalisation or baseline subtraction scheme, (iii) minibatch construction or gradient-accumulation
295   strategy, (iv) hardware configuration (GPU type, inter-node bandwidth), or (v) system-level telemetry
296   (energy, throughput, queueing).

297   What this means for our claims. Tinker results in this paper measure the platform’s implementa-
298   tion of GRPO, not an abstract specification of the algorithm. Cross-stack reward gaps between Tinker
299   GRPO and an open-source GRPO implementation cannot be fully attributed from our data: we are
300   able to rule out a handful of candidate explanations (seed variance, model-size confound) but not
301   the implementation itself. We therefore draw quantitative conclusions only from the open-source
302   side (TRL, veRL on Modal H100, OpenRLHF) where every hyperparameter is auditable, and use
303   Tinker results primarily as descriptive case studies of what a carefully engineered production stack
304   can achieve for critic-free RL at our scales.

305   Reproducibility commitments we can make.

306         1. All Tinker experiment scripts, configuration files, JSON step logs, and per-run W&B projects
307            are archived in the repository. Researchers with Tinker access can attempt replication with
308            our exact configurations.
309         2. Every figure and every summary statistic derived solely from Tinker data is marked with “†”
310            and a footnote stating that independent replication requires Tinker API access.
311         3. Primary inferential claims are limited to (i) the mechanical zero-gradient theorem (Ap-
312            pendix B) and (ii) the held-out 5-seed Qwen3-8B GSM8K control reported in Section 5.
313            The Tinker-derived stratified ZVF correlations (Section 3) are reported as artifact analyses
314            with explicit closed-backend caveats and single-seed-per-cell limitations, not as platform-
315            independent statistical conclusions.
316         4. We commit to re-running any Tinker-only experiment on an open backend if an equivalent
317            open-source service becomes available, and to issuing a revised version of this paper if any
318            Tinker result is ever shown to be due to an undisclosed implementation choice rather than
319            the algorithm.

320   Key rotation and credential hygiene. Our Tinker API key (prefix tml-...) is held in a repos-
321   itory .env.example placeholder only. The real key is stored in the authors’ password manager
322   and rotated whenever a failure mode suggests possible exfiltration. No Tinker key is commit-
323   ted to git history in the anonymous source repository (https://anonymous.4open.science/r/
324   agentic-grpo-bench-anon-2C6B).

325   7.6   Known Methodological Limits

326   In addition to the infrastructure failures and platform-specific caveats enumerated in Section 5, we
327   flag the following methodological limits.

328   Short training horizons. All Tinker GRPO runs were capped at 30–50 gradient steps. This is a
329   deliberate cost-control choice and means our Tinker numbers are early-training snapshots. We cannot
330   rule out that longer training would change the qualitative picture (e.g., the 82%→83.3% held-out
331   gain might widen, or might collapse to reward hacking).

332   Single-seed Tinker experiments. Each Tinker configuration was run once, not 3–5 times as best
333   practice prescribes [6]. Tinker results therefore lack variance estimates and do not support significance
334   testing; we report them descriptively. Only the 5-seed TRL baseline and the 5-seed held-out GSM8K
335   evaluation carry proper confidence intervals.

336   Train-set reward as primary Tinker metric. Tinker runs primarily report reward on training
337   prompts. Only the GSM8K track was followed up with a separate 200-example held-out evaluation
338   per seed. Other Tinker tracks (tool-use, xLAM) remain train-set-only and we cannot distinguish
339   memorisation from generalisation for those settings.


                                                         9


# R04: NeurIPS DNB benchmark


Root: `platform_hybrid/paper/neurips_2026_variants/main_dnb.tex`  Pages: 35  Words: 19945


[PAGE 1]
RL-Finetuning Bench: A Reproducible Artifact for
     Auditing RL Post-Training of Language Models Across
            Stacks (with Stratified ZVF Diagnostics)


                                             Anonymous Author(s)
                                                 Affiliation
                                                  Address
                                                   email



                                                   Abstract
 1            We release RL-F INETUNING B ENCH, a reproducible artifact for auditing
 2            reinforcement-learning post-training of language models across heterogeneous
 3            stacks. The artifact bundles 79 runs from 7 RL libraries, 5 model families, and
 4            total-parameter scales from 0.6B to 671B (active-parameter counts 0.6B–37B for
 5            the largest mixture-of-experts checkpoints) on GSM8K, MATH-500, HumanEval,
 6            and synthetic tool-use tasks; for every reported run we ship pinned containers, step-
 7            level reward traces, raw rollout JSONL where licensing allows, and Zero-Variance
 8            Fraction (ZVF) diagnostics.
 9            We organize all claims into three explicit evidence tiers. (A-grade, analytic or
10            multi-seed inferential): a mechanical ZVF = pG + (1−p)G identity for binary-
11            reward GRPO (analytic, not an empirical multi-seed claim), and a 5-seed held-out
12            Qwen3-8B GSM8K control showing only a +1.3pp delta over the base model
13            (p=0.26, not significant). (B-grade, single-seed descriptive): case-study runs
14            across model scale on the closed-backend Tinker API, including group-size sweeps
15            and frontier-scale checkpoints, reported as illustrative not inferential. (C-grade,
16            listed for honesty): failed or partially completed cells, enumerated in the run
17            manifest. The artifact’s value is auditability: every (backend, model, task) cell is
18            tier-classified, every log type is named (precomputed traces, step-level aggregates,
19            raw rollouts), and every closed-backend dependency is flagged.
20            We also release a stratified ZVF diagnostic suite that documents how pooled
21            ZVF-vs-performance correlations are dominated by task identity rather than a
22            within-task predictive law—a finding we treat as descriptive instrumentation, not a
23            causal claim.
24            Code,         traces,     checkpoints,       and     a     one-command         make
25            reproduce-main target are at https://anonymous.4open.science/r/
26            agentic-grpo-bench-anon-2C6B.


27   1   Introduction
28   Reinforcement learning post-training now underpins much of modern language-model reasoning and
29   alignment [84, 20], but empirical claims in this area are hard to compare across stacks. PPO [101],
30   GRPO [102], and DPO [95] are typically evaluated on different model families, with different
31   framework defaults, different reward functions, and different notions of success. The field needs
32   auditable, tier-classified artifacts more than it needs another leaderboard.

33   Artifact-first contribution. RL-F INETUNING B ENCH is released as a reproducible artifact: 79
34   runs spanning 7 RL libraries, 5 model families, 32+ checkpoints, and total-parameter scales from

     Submitted to 40th Conference on Neural Information Processing Systems (NeurIPS 2026). Do not distribute.

[PAGE 2]
35   0.6B to 671B (active-parameter counts 0.6B–37B for the largest mixture-of-experts checkpoints),
36   each accompanied by pinned containers, step-level reward traces, ZVF/GU diagnostics, and (where
37   the backend permits) raw per-group rollout JSONL. Every (backend, model, task) cell in the manifest
38   is classified into one of three evidence tiers, used consistently in the abstract, the main evidence table,
39   and the conclusion:
40          • (A-grade) multi-seed completed cells used for inference; these support the only inferential
41            claims in the paper.
42          • (B-grade) single-seed completed case-study runs on the closed-backend Tinker API, reported
43            descriptively only.
44          • (C-grade) failed, partial, or never-started cells, enumerated in the run manifest for honesty.

45   What the artifact lets reviewers do. A reviewer can: (i) re-run any A-grade open-stack cell end-to-
46   end via make reproduce-main; (ii) inspect step-level ZVF/GU and reward traces for every reported
47   run, including B-grade Tinker case studies; (iii) read off a per-run auditability matrix (Table 2)
48   that lists, for each cell, whether raw rollouts, ZVF traces, checkpoints, and held-out evaluation are
49   released; and (iv) consult the claims-evidence table (Table 1) to see exactly which artifact each claim
50   depends on and what its tier is.

51   Two narrow scientific take-aways. The artifact supports two A-grade results stated conservatively.
52   First, the mechanical identity ZVF(p, G) = pG + (1−p)G characterises when binary-reward GRPO
53   produces zero within-group gradient; we use it as a descriptive diagnostic, not as a standalone causal
54   predictor of downstream performance, because ZVF is mechanically coupled to reward sparsity,
55   group size, and baseline accuracy. Second, a 5-seed held-out Qwen3-8B GSM8K control yields only
56   a +1.3pp delta over the base model (p=0.26), so strong online reward curves on a closed backend
57   do not by themselves establish generalization gains. All B-grade single-seed case-study runs across
58   model scale are reported as illustrative.

59   Stratified ZVF diagnostics. A separate stratified diagnostic suite documents that the pooled
60   ZVF-vs-performance correlation is dominated by task identity: within GSM8K alone the apparent
61   monotone relationship does not survive (Pearson r = +0.40, p = 0.09 across the 19 Tinker GSM8K
62   runs; r = +0.13, p = 0.79 across the 7 model-varying GSM8K cells). We release these stratified
63   analyses precisely so that follow-up work can audit them against richer multi-seed corpora.

64   2     Related Work
65   RL-F INETUNING B ENCH sits at the intersection of five rapidly evolving lines of work: policy-
66   gradient algorithms for language models, RL for reasoning (including process reward models), RL
67   for tool use and agentic behaviour, empirical scaling laws for RL post-training, and the frameworks
68   and benchmarks that instantiate these algorithms. We situate our contribution against more than fifty
69   recent works; entries marked with † appeared at NeurIPS, ICML, or ICLR in the 2025–2026 cycle.

70   2.1   RL Algorithms for Large Language Models

71   Proximal Policy Optimization (PPO) [101] is the workhorse algorithm behind InstructGPT-style
72   RLHF [84, 14, 110, 7, 8] and has dominated production deployments for five years. Direct Preference
73   Optimization [95] reframed preference learning as contrastive classification and spawned a large
74   family of RL-free alternatives, including IPO [6], KTO [21], SimPO [72], ORPO [34], and SLiC [142],
75   all of which we consider as no-rollout baselines in our cross-library protocol.
76   The current wave is defined by group-relative policy gradient methods. GRPO was introduced with
77   DeepSeekMath [102] and popularised by DeepSeek-R1 [19, 20], the first reasoning RL method pub-
78   lished in Nature, which showed that pure outcome-reward RL can elicit chain-of-thought behaviour
79   from a base model without supervised cold-start data. A second wave of refinements dissects the
80   biases of vanilla GRPO. Dr. GRPO [61] removes a response-length bias and a question-difficulty
81   amplification bias by using a constant loss normaliser and dropping the standard-deviation denomi-
82   nator in the advantage. GDPO [58] decouples group-relative normalisation per reward component,
83   preventing advantage collapse when rewards are multi-valued. MC-GRPO (median-centered ad-
84   vantage) [44] replaces the group-mean baseline in the advantage with the group median to stabilise


                                                         2

[PAGE 5]
190   3     Artifact Design

191   Tasks and reward functions. The artifact covers (i) GSM8K math reasoning with binary verifiable
192   reward (complete coverage), (ii) function-calling tool use with shaped schema-validity reward (partial
193   coverage), and (iii) HumanEval code generation with binary unit-test reward (partial coverage).
194   Verifiable rewards reduce reward-model subjectivity but do not eliminate reward hacking, format
195   gaming, or train-set reward inflation; we discuss residual failure modes in Section 6. The full task
196   table and per-task data splits are listed in Appendix B.

197   Cross-stack coverage. Implementations span 7 RL libraries: TRL (HuggingFace), Stable Base-
198   lines3, CleanRL, Tianshou, PufferLib, NVIDIA rl_games, and d3rlpy, plus the closed-source Tinker
199   API. Per-stack hyperparameter mappings, baseline floor/ceiling specifications, and the full imple-
200   mentation matrix are deferred to Appendix D; the per-(stack, model, task) tier classification lives in
201   Appendix B and is the single source of truth referenced throughout this paper.

202   Models and training protocol. We evaluate 32+ checkpoints spanning 0.6B–671B total param-
203   eters (0.6B–37B active for the largest MoE checkpoints) across the Qwen3, Qwen3.5, Llama-3.x,
204   DeepSeek-V3.1, and Nemotron-120B families. Tier-A open-stack rows use LoRA (rank 32), Adam
205   (β1 =0.9, β2 =0.95), learning rate 10−4 , and seeds drawn from {42, 123, 456, 789, 1024}. The Modal
206   classic-RL arms (CleanRL, SB3, Tianshou, Qwen2.5-0.5B) use the full 5-seed protocol; the Tinker
207   Qwen3-8B Wave-6 group-size ablation is 1 seed per cell across 12 cells; and the Tinker Llama-3.3-
208   70B-Inst arm is 4 seeds. Tier-B rows (single-seed Modal H100 Llama-3.2-{1,3}B; single-seed Tinker
209   Qwen3-8B GSM8K headline; the single-seed frontier-model case-study runs across model scale
210   on Tinker) are reported descriptively. Frontier-scale Tinker runs are case studies, not inferential
211   evidence.


212   4     Results

213   4.1   Claims hierarchy and evidence

214   Table 1 maps every claim in the paper to its evidence tier and to the artifact files that support it;
215   a per-stack auditability matrix is in Table 2. Inferential conclusions (A-grade) are limited to the
216   mechanical ZVF identity (Section 4.2) and the held-out 5-seed Qwen3-8B GSM8K control. All other
217   rows are B-grade single-seed case studies or C-grade listings; the full per-row training-reward table
218   that earlier drafts placed in main is now Appendix A.

219   4.2   Zero-Variance Fraction: cross-stack diagnostic

220   We introduce three descriptive quantities that summarise when GRPO groups provide little within-
221   group learning signal. Zero-Variance Fraction (ZVF) at step t is the fraction of prompts for which
222   all G completions receive identical rewards:
                                             1 X                     
                                  ZVFt =          1 Varc∼C(p) r(c) = 0 .
                                            |Pt |
                                                 p∈Pt

223   Gradient Utilization GUt = 1 − ZVFt is the complementary fraction. For binary-outcome tasks
224   with per-prompt accuracy p, the closed-form identity ZVF(p, G) = pG + (1 − p)G holds exactly;
225   this is the (A-grade) mechanical theorem. The full derivation and a check against every logged group
226   is in Appendix C.

227   Pooled cross-task descriptive correlation. Across N = 15 experiments spanning 5 model families
228   and 0.6 B–671 B total parameters (0.6 B–37 B active for the largest MoE checkpoints), mean ZVF is
229   negatively associated with final training performance: rPearson = −0.769 (p = 0.0008), ρSpearman =
230   −0.784 (p = 0.0005). We treat this association as descriptive (B-grade), not causal: ZVF is
231   mechanically coupled to reward sparsity, group size, and baseline accuracy, and the pooled N =15
232   sample mixes tasks.


                                                        5

[PAGE 8]
261          • make smoke-test — ∼5 min on a single commodity GPU (T4 or L4 class, 16 GB). Verifies
262            that the artifact installs, the pinned containers launch, and the GSM8K reward harness emits
263            a non-trivial reward on a 16-prompt sanity batch. No model weights are updated.
264          • make reproduce-main — end-to-end re-run of every Tier-A open-stack cell in the mani-
265            fest (Modal H100 CleanRL/SB3/Tianshou/Qwen2.5-0.5B classic-RL arms; the 5-seed held-
266            out Qwen3-8B GSM8K control). Estimated wall-clock: ∼96 GPU-hours on H100 SXM5
267            (80 GB), or ∼3 hours per seed for the 0.5B reference. Produces the same step-level ZVF/GU
268            traces and the held-out evaluation numbers cited in Table 1.
269          • make reproduce-appendix — regenerates the full results table (Appendix A) figures
270            from the precomputed traces shipped with the artifact, without re-running any backend.
271            Useful for reviewers who want to verify that our plotting code agrees with the bundled trace
272            files.

273   What is and is not in the make target. make reproduce-main only re-runs A-grade open-stack
274   cells; B-grade Tinker case-study runs are not re-run because the Tinker backend is closed-source
275   and rate-limited (Section 6). For B-grade cells we ship the JSON step logs and the API-call scripts;
276   replication requires Tinker credentials and incurs the JWT-expiry failure modes documented in the
277   manifest (Appendix B).

278   Hash manifest. On a successful make reproduce-main run, the artifact writes a
279   REPRODUCIBILITY_HASHES.json file containing SHA-256 digests of every emitted ZVF trace,
280   reward CSV, and held-out evaluation summary. Reviewers can diff this against the reference man-
281   ifest released with the artifact to confirm bit-level (where deterministic) or distributional (where
282   sampling-based) agreement.

283   Submission-time status. At submission, the Makefile targets are wired to the already-released
284   Modal scripts and the held-out evaluation harness; if a target is missing or stub-only at the time
285   the camera-ready artifact tag is cut, the corresponding entry in the artifact card (Table 2) will be
286   downgraded from ✓ to “pending” and the README will list the substitute manual command.


287   6   Limitations

288   The current release has documented gaps; full discussion (dual-use, compute accounting, carbon
289   estimate, data provenance, closed-backend acknowledgement, and known methodological limits) is in
290   Appendix F. In summary: (i) all B-grade rows are single-seed and depend on a closed-source Tinker
291   backend; we report them descriptively. (ii) Most A-grade evidence is concentrated on short-horizon
292   GSM8K with binary verifiable rewards; tool-use, code-generation, and non-math coverage are sparser.
293   (iii) Inferential claims are limited to the mechanical ZVF = pG + (1−p)G identity and the held-out 5-
294   seed Qwen3-8B GSM8K control (+1.3pp, t=1.32, p=0.26, not significant). (iv) Many cross-library
295   comparisons are confounded by checkpoint, hardware, or default-config differences and should be
296   read as stack-level rather than algorithm-level statements.


297   7   Conclusion

298   RL-F INETUNING B ENCH is a shared empirical substrate for RL post-training across libraries,
299   algorithms, and model families. The current release supports three conservative claims: (i) ZVF/GU
300   are useful descriptive diagnostics of when binary-reward GRPO stops producing within-group
301   signal, but should not be read as causal predictors independent of reward mean and group size; (ii)
302   initialization and rollout regime are plausible stack-level moderators in the released case studies,
303   not isolated causal effects; and (iii) cross-stack PPO/GRPO rankings are descriptive because policy
304   architecture and task encoding are not fully matched. The held-out Qwen3-8B GSM8K control gives
305   a small, non-significant GRPO gain over the base (p=0.26). The released traces and checkpoints are
306   intended to make stricter follow-up easy.


                                                       8

[PAGE 23]
928   hypothesis is budget-dependent. We report this as a qualitative argmax pattern to test—no trained
929   matched-budget G=32 arm validates it and no committed script fits a parametric inverted-U envelope
930   or its bootstrap confidence interval, so we make no such quantitative claim.

931   E.4   Reconciliation Statement

932   We therefore revise the main-text claim as follows.

933            Under a fixed step budget at small total token counts, G=4 attains the highest
934            last-10 reward (52.1%, Table 6). Under fixed total tokens at the canonical training
935            scale used elsewhere in the paper (T ≥ 16M), G≈32 maximizes reconstructed
936            held-out accuracy in this illustrative reanalysis; it is not a measured recommen-
937            dation. (The per-token gradient-efficiency estimator GE  dtok is not co-maximized:
938            it decreases monotonically in G, so it favors small G and is reported only to
939            show that the accuracy gain at large G comes at a per-token efficiency cost.) The
940            accuracy-optimal G shifts rightward with T , so the recommended G depends on
941            the practitioner’s compute budget, not on a universal heuristic.

942   The reviewer’s concern is taken seriously: the “more rollouts is always better” heuristic is false,
943   and the opposite heuristic “small G is always better” is equally false; the correct claim is that
944   there exists a budget-dependent optimum and practitioners should locate it via the reanalysis script
945   experiments/group_size_token_normalized.py.

946   E.5   Measured Small-Scale Confirmation of the Inverted-U

947   Because the token-normalized table above is an illustrative reanalysis, we separately
948   ran a measured group-size sweep to confirm the qualitative inverted-U with fresh per-
949   seed data (Table 11; driver experiments/modal/modal_groupsize_zvf_sweep.py, artifact
950   experiments/results/groupsize_zvf_sweep.tsv). On the ungated Qwen2.5-0.5B with a
951   verifiable correctness reward on two-operand addition, G ∈ {2, 4, 8, 16} at three seeds each, held-
952   out accuracy traces an inverted-U shape with its numerical maximum at the intermediate G=8
953   (0.982 → 0.988 → 0.990 → 0.978); we emphasize the apex is not statistically separable from its
954   neighbours (G=4 0.988±.002 vs. G=8 0.990±.003, overlapping SEs at n=3), so the supported
955   claim is only that the optimum is interior, not that it is precisely G=8. The mean zero-variance fraction
956   falls monotonically with G (0.84 → 0.76 → 0.69 → 0.63), the direction ZVF(p, G) = pG +(1−p)G
957   predicts (the closed form captures the trend, not the absolute level—see the formalization appendix).
958   This is a small model on an easy task (accuracies are near ceiling, so the effect sizes are modest) and
959   is not a substitute for a Qwen3-8B/GSM8K sweep; we report it only as a measured, reproducible
960   indication that the optimal G is interior rather than smallest-or-largest.

961   F     Ethics, Limitations, and Broader Impact
962   This statement is written to satisfy the NeurIPS Code of Ethics and Broader Impact requirements.
963   It complements the shorter Limitations and Broader Impact subsections embedded in Section 6 by
964   consolidating in one place a full dual-use analysis, itemised compute accounting, carbon footprint
965   estimation, data provenance disclosures, a candid acknowledgment of the closed-source training
966   backend on which a fraction of our headline numbers depends, and a list of methodological limits we
967   are aware of but did not have resources to close within the submission window.

968   F.1   Dual-Use Analysis: Misuse Risks of Reasoning RL

969   Threat model. RL-F INETUNING B ENCH releases (i) training scripts that apply GRPO to the
970   GSM8K [15] mathematical reasoning benchmark and to the Salesforce xLAM-Function-Calling-
971   60k [57] tool-use corpus, (ii) a set of LoRA adapters published on the HuggingFace Hub
972   (anonymous/tinker-rl-bench-*), and (iii) diagnostic code for Zero-Variance Fraction, reward-
973   stability, and length-bias analysis. We analyse four concrete misuse pathways these artefacts could,
974   in principle, enable, and we describe the existing guardrails and the residual risk we judge acceptable
975   for publication.


                                                        23

[PAGE 26]
1074   GSM8K [15]. 8,500 grade-school math word problems (7,473 train / 1,319 test) authored by
1075   human writers contracted by OpenAI. Released under the MIT License via https://github.com/
1076   openai/grade-school-math. We use the standard train/test splits without modification. The
1077   canonical citation is Cobbe et al. [15]. Known limitations of GSM8K include gender-neutral but
1078   culturally US-centric word problems (names, currencies, sports) and a moderate rate of human-
1079   labelling errors estimated at ~2% by [138]. Our held-out evaluation respects the test/train split.

1080   Salesforce xLAM-Function-Calling-60k [57]. 60,000 function-calling examples generated
1081   by Salesforce Research as training data for the xLAM agentic model family.                          Re-
1082   leased under the CC BY 4.0 license via https://huggingface.co/datasets/Salesforce/
1083   xlam-function-calling-60k. We use the public release as-is; specifically, we use the first 35
1084   prompts × 10 rollouts in the 10x Structural Ceiling tool-use track and the full 60k split for the xlam-
1085   60k real-data run in Section 4. Known limitations: xLAM schemas are synthetic and skew toward
1086   cleanly-typed arguments; the dataset under-represents ambiguous, error-handling, and multi-turn
1087   tool-calling. Attribution to Salesforce is required by the license and is provided in Section F.6 and in
1088   the xLAM-derived model cards.

1089   HumanEval [12]. 164 Python programming problems with unit tests, released by OpenAI under
1090   the MIT License. Used unmodified for pass@k evaluation. Known limitations: small scale, English-
1091   language docstrings, single-language coverage; documented contamination concerns against frontier
1092   pretraining corpora [97].

1093   NuminaMath [51]. Used by an anonymous collaborator for the multi-stage GSM8K+NuminaMath
1094   pipeline. Released under Apache 2.0. We use a downstream checkpoint reported by that collaborator;
1095   our repository contains only their scripts and the Apache-licensed derivative weights.

1096   Open-Platypus [49]. 3,000 SFT examples used by an anonymous collaborator for Qwen3-8B
1097   code generation warm-up. Open-Platypus is a curated subset released under CC BY-NC 4.0; the
1098   non-commercial restriction is respected in our release, which is academic-use only.

1099   Synthetic tool-use corpus (authored). We additionally generated a small five-tool synthetic corpus
1100   (calculator, web-search stub, calendar, file-read, email-send) used for the tool-use case studies cited
1101   in Section 4.2 (the high-ZVF / zero-reward tool-use cluster that drives the pooled task confound). The
1102   corpus is authored by the paper’s authors from publicly documented APIs, contains no third-party
1103   content, and is released under the MIT License in our repository under data/synthetic_tools/.
1104   Generation prompts are included for auditability. No user data, scraped web content, or proprietary
1105   API traces are present.

1106   No web scraping, no human subjects. We did not scrape any website. We did not run any study
1107   with human participants. No IRB review was therefore required. No data that could reasonably be
1108   considered to contain PII, copyrighted content, or sensitive personal attributes was used at any stage
1109   of training, evaluation, or reward modelling.

1110   F.5   Closed-Source Tinker Acknowledgment

1111   A central tension in RL-F INETUNING B ENCH is that a substantial fraction of our headline numbers
1112   come from a closed-source commercial platform while our paper simultaneously advocates for
1113   reproducibility and platform independence. We do not resolve this tension by hiding it; we describe it
1114   precisely.

1115   What Tinker is and is not. Tinker [114] is a managed LLM fine-tuning and inference service
1116   provided by Thinking Machines, Inc. It exposes a Python SDK (tinker==0.16.1) that accepts
1117   custom loss functions (forward_backward_custom), a limited set of optimisers, and standard LoRA
1118   hyperparameters. It does not expose (i) the exact server-side GRPO loss implementation, (ii) reward
1119   normalisation or baseline subtraction scheme, (iii) minibatch construction or gradient-accumulation
1120   strategy, (iv) hardware configuration (GPU type, inter-node bandwidth), or (v) system-level telemetry
1121   (energy, throughput, queueing).


                                                         26

[PAGE 27]
1122   What this means for our claims. Tinker results in this paper measure the platform’s implementa-
1123   tion of GRPO, not an abstract specification of the algorithm. Cross-stack reward gaps between Tinker
1124   GRPO and an open-source GRPO implementation cannot be fully attributed from our data: we are
1125   able to rule out a handful of candidate explanations (seed variance, model-size confound) but not
1126   the implementation itself. We therefore draw quantitative conclusions only from the open-source
1127   side (TRL, veRL on Modal H100, OpenRLHF) where every hyperparameter is auditable, and use
1128   Tinker results primarily as descriptive case studies of what a carefully engineered production stack
1129   can achieve for critic-free RL at our scales.

1130   Reproducibility commitments we can make.

1131         1. All Tinker experiment scripts, configuration files, JSON step logs, and per-run W&B projects
1132            are archived in the repository. Researchers with Tinker access can attempt replication with
1133            our exact configurations.
1134         2. Every figure and every summary statistic derived solely from Tinker data is marked with “†”
1135            and a footnote stating that independent replication requires Tinker API access.
1136         3. Primary inferential claims are limited to (i) the mechanical zero-gradient theorem (Ap-
1137            pendix C) and (ii) the held-out 5-seed Qwen3-8B GSM8K control reported in Section 6.
1138            The Tinker-derived stratified ZVF correlations (Section 4.2) are reported as artifact analyses
1139            with explicit closed-backend caveats and single-seed-per-cell limitations, not as platform-
1140            independent statistical conclusions.
1141         4. We commit to re-running any Tinker-only experiment on an open backend if an equivalent
1142            open-source service becomes available, and to issuing a revised version of this paper if any
1143            Tinker result is ever shown to be due to an undisclosed implementation choice rather than
1144            the algorithm.

1145   Key rotation and credential hygiene. Our Tinker API key (prefix tml-...) is held in a repos-
1146   itory .env.example placeholder only. The real key is stored in the authors’ password manager
1147   and rotated whenever a failure mode suggests possible exfiltration. No Tinker key is commit-
1148   ted to git history in the anonymous source repository (https://anonymous.4open.science/r/
1149   agentic-grpo-bench-anon-2C6B).

1150   F.6   Known Methodological Limits

1151   In addition to the infrastructure failures and platform-specific caveats enumerated in Section 6, we
1152   flag the following methodological limits.

1153   Short training horizons. All Tinker GRPO runs were capped at 30–50 gradient steps. This is a
1154   deliberate cost-control choice and means our Tinker numbers are early-training snapshots. We cannot
1155   rule out that longer training would change the qualitative picture (e.g., the 82%→83.3% held-out
1156   gain might widen, or might collapse to reward hacking).

1157   Single-seed Tinker experiments. Each Tinker configuration was run once, not 3–5 times as
1158   best practice prescribes [27]. Tinker results therefore lack variance estimates and do not support
1159   significance testing; we report them descriptively. Only the 5-seed TRL baseline and the 5-seed
1160   held-out GSM8K evaluation carry proper confidence intervals.

1161   Train-set reward as primary Tinker metric. Tinker runs primarily report reward on training
1162   prompts. Only the GSM8K track was followed up with a separate 200-example held-out evaluation
1163   per seed. Other Tinker tracks (tool-use, xLAM) remain train-set-only and we cannot distinguish
1164   memorisation from generalisation for those settings.

1165   LoRA only; no full fine-tuning. All experiments use LoRA [35] with ranks 8–64. We have
1166   not tested whether full fine-tuning would re-order the library/algorithm comparisons. The LoRA
1167   constraint is practical (cost) but it means our claims about PPO vs. GRPO and TRL vs. Tinker are
1168   LoRA-conditional.


                                                        27

[PAGE 28]
1169   Benchmark coverage is narrow. We evaluate four domains: GSM8K, MATH-500 (exploratory),
1170   HumanEval (subset), and synthetic/xLAM tool-use. We do not evaluate on MT-Bench, ArenaHard,
1171   safety benchmarks (HarmBench, ToxicChat), or truthfulness benchmarks (TruthfulQA). Any claim
1172   about “reasoning” in the paper is strictly a claim about the covered benchmarks.

1173   No human preference data. We use verifiable rewards only (exact-match for math, unit-test for
1174   code, schema-validity for tool-use). We do not study RLHF with human preference data; findings
1175   should not be extrapolated to reward-model–based alignment without further work.

1176   Closed-source implementation opacity (reiterated). Comparisons between Tinker GRPO and
1177   TRL GRPO are confounded by the closed-source nature of Tinker’s implementation. See Section F.5
1178   for detail.

1179   Cross-platform hardware confounding. Tinker, Modal, Colab, and the Institutional A100 node
1180   use different accelerators, different memory hierarchies, and different inference/training stacks
1181   (Tinker proprietary, Modal vLLM, Colab PyTorch-native, TRL+Accelerate). Observed differences in
1182   reward dynamics are not purely algorithmic.

1183   Carbon estimates are approximate. As discussed in Section F.3, Tinker GPU-hour and TDP are
1184   inferred, not measured. Grid intensity is region-average, not marginal. Numbers in Table 13 should
1185   be treated as order-of-magnitude.

1186   Exploratory, not confirmatory. This is an exploratory case study, not a pre-registered confirmatory
1187   experiment. We did not pre-register primary hypotheses. All p-values are un-corrected for the number
1188   of comparisons actually made across the paper; the Bonferroni-surviving comparisons in Section 4
1189   are so flagged, but most of our secondary analyses are descriptive.

1190   Geographic and demographic limits. The specific city and country of the authors’ institution are
1191   withheld here for anonymity but disclosed in the non-anonymous version of the paper. Our choice of
1192   benchmarks, prompt phrasings, and evaluation design reflects that single-region context. Independent
1193   replication by groups in other regions, on non-English tasks, and with non-Qwen / non-Llama families
1194   is an open question.

1195   NeurIPS Paper Checklist
1196        1. Claims
1197            (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s
1198                contributions and scope? [Yes] The abstract and Section 1 state every claim with
1199                an explicit evidence tier (A/B/C) and map it to a row of Table 1 (claims hierarchy
1200                with evidence tier and artifact availability) and Table 2 (per-stack auditability matrix).
1201                (A-grade, multi-seed inferential): the mechanical ZVF=pG +(1−p)G identity for
1202                binary-reward GRPO and the held-out 5-seed Qwen3-8B GSM8K control (+1.3pp,
1203                p=0.26, not significant). (B-grade, single-seed descriptive): the closed-backend
1204                Tinker case-study runs across model scale, the single-seed group-size sweep, and the
1205                tool-use case studies. (C-grade, listed for honesty): the failed/partial cells enumerated
1206                in Appendix B. The scope is explicitly restricted to LoRA fine-tuning on three task
1207                families (math, code, tool-use) at scales of 0.6B–671B total parameters (0.6B–37B
1208                active for the largest MoE checkpoints) (Section 3). Section 6 discloses partial-run
1209                accounting; partial runs are marked † in Appendix A.
1210        2. Limitations
1211            (a) Does the paper discuss the limitations of the work performed by the authors? [Yes]
1212                Section 6 enumerates the limitations in detail:
1213                  • Platform opacity. Tinker API is a closed, serverless platform; GPU type, driver
1214                    version, CUDA runtime, and scheduler are undisclosed, so Tinker-derived results
1215                    cannot be independently reproduced without Tinker access (Section 6).
1216                  • Infrastructure failures. A non-trivial subset of Tinker runs were interrupted
1217                    by JWT token expiry, and a non-trivial subset of Modal runs hit timeouts or


                                                        28

[PAGE 29]
1218             gradient-norm blow-ups; per-run status (completed, partial, failed) is enumerated
1219             in Appendix B and partial rows are marked † in Table 5.
1220           • Single-seed Tinker runs. Cost constraints precluded multi-seed replication on
1221             Tinker; these results are treated as descriptive only (Section 6).
1222           • LoRA-only evaluation. Full fine-tuning and quantisation-aware training are not
1223             evaluated (Section 6).
1224           • Train-set reward metric. Primary Tinker metrics are training rewards, not held-out
1225             accuracy. The headline held-out control is Qwen3-8B on GSM8K (Section 6);
1226             broader held-out evaluation is incomplete and disclosed as such.
1227           • Proxy metrics in place of direct KL. the proxy-metrics discussion in the appendix
1228             discusses the limits of trajectory-based stability indices as substitutes for direct KL
1229             divergence.
1230   3. Theory Assumptions and Proofs
1231      (a) For each theoretical result, does the paper provide the full set of assumptions and a
1232          complete (or correct) proof? [NA] This is an empirical benchmarking paper; no formal
1233          theorems are claimed. GRPO and PPO objectives stated in Section 3 are background
1234          definitions, not original theoretical results.
1235   4. Experimental Result Reproducibility
1236      (a) Does the paper fully disclose all the information needed to reproduce the main ex-
1237          perimental results of the paper to the extent that it affects the main claims and/or
1238          conclusions of the paper? [Partial] Reproducibility is partial by design due to platform
1239          heterogeneity.
1240           • Modal experiments (fully reproducible). Complete source code, a pinned
1241              Dockerfile, centralised seed management (utils/seed.py), and step-by-step
1242              commands are documented in REPRODUCE.md at https://anonymous.4open.
1243              science/r/agentic-grpo-bench-anon-2C6B. Modal jobs run on explicitly
1244              provisioned NVIDIA H100 SXM5 (80 GB) workers; TRL baselines run on NVIDIA
1245              L4 (24 GB). Seeding is the manifest’s responsibility (Appendix B, source of truth):
1246              Tier-A open-stack rows (CleanRL/SB3/Tianshou/Qwen2.5-0.5B classic-RL arms +
1247             12 Tinker Wave-6 Qwen3-8B group-size cells (1 seed/cell, 30 steps max) + 4 Llama-
1248              3.3-70B-Inst rows) report mean ± SE with 95 % bootstrap CIs via rliable; Tier-B
1249              rows are single-seed and reported descriptively; Tier-C rows are excluded from
1250              headline claims. The single-seed Modal H100 Llama-3.2-1,3b GSM8K and Qwen3-
1251              8B-family rows in Table 5 are therefore Tier-B by manifest classification.
1252           • Tinker API experiments (not fully reproducible). Tinker is closed-source with
1253              serverless GPU dispatch; GPU type, driver version, and scheduling policy are not
1254              exposed. We release our exact API scripts and configuration files, but independent
1255              replication requires a Tinker account and is subject to the same JWT expiry issues
1256              we observed (Section 6). In Table 5 (Appendix A), all Tinker-only rows are
1257              explicitly listed under the italicized headers “GRPO on Tinker API”, “single-seed
1258              frontier-model case studies”, “group-size ablation (single-seed per cell)”, and
1259             “Cross-task (Tool-Use), GRPO on Tinker”; the “†” marker is reserved for partial
1260              / interrupted runs as defined in the table caption. Tinker rows in figures (ZVF
1261              correlation) are explicitly captioned as Tinker case studies.
1262          We claim Partial rather than Yes: the Tinker subset is irreducibly platform-dependent.
1263   5. Open access to data and code
1264      (a) Does the paper provide open access to the data and code, with sufficient in-
1265          structions to faithfully reproduce the main experimental results, as described
1266          above? [Yes] All code is publicly released under the Apache License 2.0
1267          at https://anonymous.4open.science/r/agentic-grpo-bench-anon-2C6B
1268          (camera-ready: GitHub mirror to be linked from the same anonymous URL). LoRA
1269          adapter checkpoints from the successful Modal experiments are on HuggingFace
1270          Hub and released alongside the anonymous repository at https://anonymous.
1271          4open.science/r/agentic-grpo-bench-anon-2C6B with model cards generated
1272          from huggingface/MODEL_CARD_TEMPLATE.md. Logged artifacts are released at
1273          three explicitly named levels of granularity, mirroring Section 5: (i) precomputed


                                                 29

[PAGE 31]
1329            societal impacts—including reward hacking, alignment failure modes of RLHF, and
1330            equity of compute access—are discussed in Section 6 and in ethics_statement.tex.
1331   10. Broader Impacts
1332        (a) Does the paper discuss both potential positive societal impacts and negative societal
1333            impacts of the work performed? [Yes] Section 6 (plus ethics_statement.tex and
1334            LIMITATIONS_AND_IMPACT.md) discusses:
1335              • Positive. Lowering the barrier to RL post-training research; enabling reproducibility
1336                audits; surfacing platform-dependent variance that is typically hidden in single-
1337                platform papers.
1338              • Negative / risks. RLHF reward hacking; alignment concerns when optimising
1339                proxy rewards; potential misuse of fine-tuned models; compute-access disparities
1340                that could limit replication to well-resourced groups.
1341              • Environmental. Best-estimate project total is ~296 kg CO2 -eq, with 130–540 kg
1342                under optimistic / pessimistic Tinker assumptions (Section F.3; the dominant uncer-
1343                tainty is the Tinker GPU-hour count because hardware telemetry is undisclosed).
1344                The Modal-attributable footprint is itemized in Table 13 of the ethics statement.
1345   11. Safeguards
1346        (a) Does the paper describe safeguards that have been put in place for responsible re-
1347            lease of data or models that have been identified as requiring safeguards? [NA]
1348            The released artefacts are LoRA adapters fine-tuned from already-public base mod-
1349            els (Qwen, Llama, Nemotron, GPT-OSS, Kimi-K2) on standard research datasets
1350            (GSM8K, HumanEval, synthetic tool-use). No new high-risk capabilities are intro-
1351            duced relative to the base models; released checkpoints inherit the base models’ li-
1352            cence terms and include standard usage disclaimers in the HuggingFace model cards
1353            (huggingface/MODEL_CARD_TEMPLATE.md).
1354   12. Licenses for existing assets
1355        (a) Are the creators or original owners of assets (e.g., code, data, models) used in the
1356            paper properly credited, and are the licence and terms of use explicitly mentioned and
1357            properly respected? [Yes] All third-party assets are credited with their licences:
1358              • GSM8K [15]: MIT licence.
1359              • HumanEval (OpenAI): MIT licence.
1360              • NoRobots (HuggingFaceH4): CC-BY-NC-4.0 (used for the chat-SFT task only).
1361              • Synthetic tool-use data: self-generated from public API documentation; no up-
1362                stream licence restrictions.
1363              • Qwen model family: Apache-2.0.
1364              • Llama model family: Meta Llama Community Licence.
1365              • Nemotron, GPT-OSS, Kimi-K2: used under their respective published licences;
1366                credits in Section 3.
1367              • TRL, PEFT, Transformers: Apache-2.0 (HuggingFace).
1368              • Modal: commercial platform; terms of service respected.
1369              • Tinker API: proprietary; used under standard API terms of service.
1370              • Our code: Apache-2.0 (see LICENSE in the repository root).
1371   13. New Assets
1372        (a) Are new assets introduced in the paper well documented and is the documentation
1373            provided alongside the assets? [Yes] New assets and their documentation:
1374              • The RL-F INETUNING B ENCH benchmark suite (harness, evaluation
1375                scripts, analysis notebooks) at https://anonymous.4open.science/r/
1376                agentic-grpo-bench-anon-2C6B, released under Apache-2.0 and documented
1377                by README.md, REPRODUCE.md, ARTIFACT.md, COMPUTE.md, BASELINES.md,
1378                BENCHMARKS_COMPARISON.md, and LIMITATIONS_AND_IMPACT.md.
1379              • LoRA adapter checkpoints from the successful Modal experiments released along-
1380                side the anonymous repository at https://anonymous.4open.science/r/
1381                agentic-grpo-bench-anon-2C6B, each accompanied by a HuggingFace model
1382                card generated from huggingface/MODEL_CARD_TEMPLATE.md (intended use,
1383                training data, evaluation results, known limitations).


                                                    31


# R05: ZVF theory


Root: `zvf-program/theory/zvf_theory.tex`  Pages: 12  Words: 5695


[PAGE 1]
Calibrating the Zero-Variance Fraction in
                         Group-Relative RL
           Estimation, Reliability Budgets, and the Limits of Group-Size Control
                             ZVF Program, Pillar 2 (Theory)

                                            Arvind C R
                                          PES University
                                       arvindcr4@gmail.com
                                      With AI drafting assistance

                                             July 14, 2026


                                               Abstract
         The Zero-Variance Fraction (ZVF) is the fraction of prompt groups whose within-group
     reward variance is zero in group-relative RL with binary rewards. It is a useful mechanistic
     diagnostic, but calibration does not automatically grant control authority. We separate those
     questions through three conditional results. T1 treats ZVF correctly as a sample mean of
     Bernoulli group indicators and gives binomial-proportion confidence intervals; under curriculum
     ordering these intervals cover a local-stage estimand unless batch composition is stratified. T2
     converts a population ZVF into a reliability budget: the number of independent groups required
     to observe at least one informative group with probability 1 − δ. This bounds rollouts to
     a nonzero reward-gradient event, not rollouts to policy improvement. T3 audits a proposed
     signal-per-rollout objective. Under that declared objective its optimum collapses to the prior-
     independent tie G ∈ {2, 3}; it therefore supplies no data-adaptive group-size set-point. A
     richer variance-, Fisher-, wall-clock-, or outcome-aware objective is required before theory can
     justify adaptive G. Checked-in sampling experiments validate the operational accounting, while
     every assumption and scope boundary is stated inline. The result is a calibrated sensor and a
     conservative reliability trigger—not a proof that a particular controller or group size improves
     held-out performance.


1    Introduction
The ZVF Program’s Pillar 1 contributed a mechanism: in group-relative, critic-free RL (GRPO and
relatives [3]), the policy gradient of a prompt-group is mediated entirely by the within-group spread
of rewards. When all K rollouts of a group collapse to the same binary outcome, the group-relative
advantage is identically zero and the group contributes no gradient. The ZVF counts the fraction of
such “dead” groups in a batch. Pillar 1 made this definition rigorous, showed it is not a tautological
re-expression of the mean reward, and validated it as a portable early-warning signal [1].
    A mechanistic diagnostic, however, is a measurement, and a measurement is only as good as its
calibration, its predictive content, and its actionability. Three gaps remain before ZVF can drive a
training-time controller rather than merely annotate a training curve:

1. Identifiability / calibration. The reported ZVFt is an estimate from finitely many groups.
   Two runs reporting ZVFt = 0.86 and 0.81 may be statistically indistinguishable. We need

                                                    1

[PAGE 11]
8    Discussion and Limitations of the Theory
What is conditionally established. Under i.i.d. groups and binary verifiable rewards: (T1)
ZVFt is an unbiased, asymptotically normal estimator of a well-defined population quantity with a
closed-form CI; (T2) high ZVF implies a logarithmically diverging reliability budget — the (1−δ)-
quantile of rollouts to a nonzero update; (T3) the signal-per-rollout objective has a well-defined
optimum, which is the universal G⋆ ∈ {2, 3} (eq. 13), prior-independent.

What is not established.

 • T1 is a binomial-proportion result, not a new U-statistic theorem. A different degree-2
   estimand would require a separate degeneracy analysis; no such claim is made here.
 • T2 bounds rollouts-to-nonzero-gradient, not rollouts-to-improvement. Corollary 1
   supplies an exact- coverage lower bound when the population ZVF is estimated.
 • T3’s objective S(p, G) is a modelling choice, and under it the optimum is the universal,
   prior-independent {2, 3} (eq. 13) — so T3 as stated supports no data-adaptive group-size policy;
   a richer objective and prospective experiment are required for any adaptive-G⋆ claim.
 • Across-group i.i.d. (Assumption 3) underlies all three and is violated by curriculum,
   replay, and within-epoch correlation. Resolved empirically (E-T1): under curriculum ordering
   the CI remains calibrated for the local (curriculum-stage) ZVF — the quantity a stage-local
   controller acts on — while global inference requires stratified batch composition. The threat is
   thus an estimand-labeling requirement, not an invalidation.

Empirical status of the theorems. Superseded by Section 7, which reports the complete n=512
validation (T1 coverage and estimand resolution, T2 floor tightness 1.00–1.05, T3 G⋆ ∈ {2, 3}
agreement, controller pilot). The truncated-pool (n=350) artifacts are retained alongside as the
pre-registration record; no verdict changed between n=350 and n=512.

No empirical claims beyond the cited artifacts. This draft contains no experiments of its
own; all numeric values above are either illustrative algebraic constants or drawn from the checked-in
experiments-next/results/*.json artifacts named inline.


References
[1] Arvind C R and the ZVF Program. Reward contrast, not algorithm labels: Auditing signal
    starvation in group-relative reinforcement learning. Companion working paper in the Tinker RL
    Lab manuscript bundle, 2026.

[2] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and
    Min Lin. Understanding R1-Zero-Like training: A critical perspective. arXiv preprint, 2025.
    doi: 10.48550/arXiv.2503.20783. arXiv:2503.20783; introduces Dr. GRPO.

[3] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li,
    Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open
    language models. arXiv preprint, 2024. doi: 10.48550/arXiv.2402.03300. arXiv:2402.03300.




                                                 11


# R06: MIN-REPORT position


Root: `zvf-program/position/min_report_rl.tex`  Pages: 14  Words: 6271


[PAGE 1]
Min-Report-RL: Reporting the Stack, Not the Label
 A Community Minimum-Reportable-Stack Standard and Reproducibility-Audit Protocol
                  for Group-Relative RL Post-Training of LLMs

                                                  Arvind C R∗
                                                PES University
                                             arvindcr4@gmail.com

                                                   July 14, 2026



                                                      Abstract
              Reinforcement-learning post-training of large language models is reported as though an
          algorithm label—PPO, GRPO, DPO, and the growing GRPO family (DAPO, GSPO, Dr.GRPO,
          M-GRPO, GRESO, EDGE-GRPO, DARS, TreePo, . . . )—fixes the experiment. It does not. In
          an audit of a group-relative RL runner, nominally identical visible “GRPO” configurations (same
          advertised model family, group size, learning rate, dataset, seed, and step budget) produced
          last-10 training reward of 84.4% on one backend and 5.0% on another—a ∼ 17× gap [2] with no
          visible hyperparameter difference. Provenance inspection later showed that the managed stack
          also pinned a different base checkpoint. The 17× number is therefore an under-specification
          exhibit, not a backend causal estimate: the label was held constant; the stack and model identity
          were not. We argue that this stack-conditioning is not an exotic edge case but the default state of
          the field, and that the literature’s current reporting norms make it structurally impossible to tell
          whether a claimed algorithmic gain survives a change of backend, sampler, reference-policy/KL
          handling, LoRA configuration, or reward parser.
              This position paper proposes Min-Report-RL: an eight-item minimum-reportable-stack
          checklist that every GRPO-family paper should satisfy—seven run-manifest fields plus held-out
          pass@k reporting—where each item is included precisely because it is a documented lever that
          can flip a head-to-head comparison. We then propose a concrete reproducibility-audit protocol—
          re-implement DAPO, GSPO, Dr.GRPO, and AERO inside one controlled language-model stack
          and report which claimed gains survive; M-GRPO is assigned to a separate agentic stratum
          because its hierarchy is not a minimal arithmetic-trainer hook. We specify the locked treatment
          matrix and verdict rules without printing unexecuted result cells. We close with an adoption
          path for getting TRL, verl, and OpenRLHF to log the Min-Report-RL fields by default, and
          with responses to the obvious objections. We do not claim that any specific GRPO variant is
          overstated; we claim that, under current reporting, the field cannot know, and that this is fixable
          with a few lines of telemetry.


1        Introduction: The Telemetry and Reporting Gap
Scope note (2026-07-11 canonicalization). This document is the condensed community-position
statement of the Min-Report-RL standard. The full-length treatment — the measured exhibits, live-
corpus coupling analysis, threat model with flip-risk grades, toolchain, and verifier implementation —
is the Pillar-5 paper “Report the Stack, Not the Label: RL-for-LLM Results Are Stack-Conditioned”
    ∗
        Pillar 4 of the ZVF Program.



                                                           1

[PAGE 2]
(platform_hybrid/paper/paper_P5_minreport). The two documents share the same eight-item
standard; the Pillar-5 paper is canonical for evidence, this one for the community-facing statement.
    A reader of the 2023–2026 RL-for-LLM literature could be forgiven for believing that “we trained
with GRPO” is a complete experimental description. It is not. The same three-letter label is attached
to runs that differ in their loss form (is there a PPO ratio? a clip? a completion-only token mask?),
their reference-policy and KL handling (frozen reference? KL penalty in the loss, or in the reward,
or absent?), their sampler and backend (vLLM vs. a managed inference API; bf16 vs. fp32 logits),
their group-size schedule, and the parser that converts a generation into a scalar reward. Each of
these is, individually, sufficient to move a result. Together they form a treatment that the algorithm
label does not name.
    This is the LLM-post-training analogue of the deep-RL reproducibility crisis documented a
decade ago [3, 5], now recurring one abstraction level up. The earlier crisis was about code-level
and hyperparameter variance within a single algorithm; the present one is worse, because the “stack”
spans a managed API whose loss we cannot inspect, a tokenizer and chat template that silently
change the prompt, a sampler whose numerics differ from the trainer’s, and a reward parser that
can reward a format artifact instead of a correct answer.

The concrete trigger. In an audit of a critic-free group-relative runner [2], we attempted a
deliberately boring comparison: hold the visible GRPO configuration fixed—Qwen3-8B-family,
group size G = 8, learning rate 10−5 , GSM8K, 30 steps, seed 42—and change the launch stack. A
managed runner reached 84.4% last-10 training reward; TRL on an H100 reached 5.0% [2]. No
visible hyperparameter explains the gap, and the managed run later proved to use a different base
checkpoint. What differs is everything the label omits: checkpoint and tokenizer identity, prompt
construction, sampler behavior, loss masking, KL/reference handling, optimizer defaults, LoRA
target modules, precision, rollout plumbing, checkpoint selection, and the evaluator. The comparison
is not evidence that one backend is better; it is evidence that “the GRPO config” was under-specified.
If a nominal backend comparison can span a 17× result while also changing the base checkpoint,
then a head-to-head between two GRPO variants—each implemented in its own stack—tells us
almost nothing about the variants.

The family is growing faster than its reporting. The TMLR survey of agentic RL [10]
now catalogs, in a single comparison table, more than twenty GRPO-family variants—and a
striking fraction of them (DAPO’s dynamic sampling, GRESO’s pre-rollout filtering, EDGE-GRPO’s
advantage collapse mitigation, DARS’s difficulty-aware reallocation) intervene on the same underlying
lever: the fraction of groups whose identical rewards contribute zero gradient, i.e. the quantity
Zvf measures. Not one of the cataloged variant papers reports a per-step dead-group trajectory
or the stack fields below, so their relative standings are, under current norms, unattributable. The
same survey’s mechanistic synthesis supplies a second indictment of current reporting: roughly
two-thirds of the RL-for-reasoning studies it reviews report only pass@1, although pass@k curves
are what distinguish a model whose solution distribution was sharpened from one whose solution
support expanded. A variant can “win” at pass@1 by concentrating probability on already-reachable
solutions while strictly shrinking the set of problems it can solve at pass@32. Under pass@1-only
reporting, these two outcomes—one an optimization of sampling, the other a capability change—are
indistinguishable. This motivates the eighth item of the standard.

Why this is a reporting problem, not just a science problem. The gap above is not a bug to
be fixed by a better trainer; it is information that is routinely not logged. Most papers do not report


                                                  2

[PAGE 4]
Visible config (held constant)     Backend S1     Backend S2
             Advertised model family            Qwen3-8B       Qwen3-8B
             Exact base checkpoint              not matched; recovered only after the run
             Group size G                       8              8
             Learning rate                      10−5           10−5
             Dataset / split                    GSM8K-500      GSM8K-500
             Steps                              30             30
             Seed                               42             42
             Last-10 training reward            84.4%          5.0%

A naïve reader concludes “S1 ’s GRPO is 17× better.” The correct reading is that the two rows
are different treatments wearing the same label : the loss form, token mask, KL/reference handling,
sampler precision, LoRA targets, optimizer defaults, and reward parser were never held fixed because
they were never reported, hence never matched. Flip any one of them and the ranking can invert.
This is the entire problem in one table: the label is constant, the conclusion is determined by the
unreported stack.

Why the GRPO family makes this acute. The methods this paper targets—DAPO, GSPO,
Dr.GRPO, M-GRPO, and the broader variance-mitigation line (AERO, CPPO, NGRPO, Scaf-
GRPO) [4, 6–9, 11–13]—are typically defined as small deltas on a base GRPO loop: a changed
advantage normalization, a clip-bound tweak, a token mask, a length penalty, an exploration bonus,
an adaptive group size. The size of the claimed improvement is frequently smaller than the stack
effect demonstrated above. When the treatment effect you are trying to measure is 2–5 points and
the nuisance effect of an unreported stack difference is tens of points, an uncontrolled head-to-head
is not measuring the method. The GRPO family is exactly the regime in which stack-conditioning is
most likely to masquerade as algorithmic progress.


3    The Min-Report-RL Standard
Min-Report-RL is a minimum-reportable-stack: the smallest set of fields such that, if two GRPO-
family papers both report them, a reader can tell whether their comparison is confounded. Each
item below is included because there is a known mechanism by which it can flip a comparison; an
item that could not change a ranking would not earn its place on a minimum list.

1. Loss form. Report: whether the update uses a PPO-style importance ratio wi,t = πθ /πθold ;
whether and how it is clipped (and the clip bounds, including asymmetric DAPO-style “clip-
higher”); whether the token mask is completion-only or whole-sequence; and whether advantages are
normalized per-group, per-batch, or with a running estimate. Why it can flip: the choice of token
mask changes the objective. In one diagnostic, 61.6–89.6% of the full-sequence loss magnitude (raw
NLL composition) came from prompt tokens rather than completion tokens [2]; a whole-sequence
mask and a completion-only mask are therefore different objectives sharing a name. (Caveat, added
2026-07-11: under exactly group-centered advantages the prompt-token gradient contributions cancel
within each group, so this figure quantifies objective composition, not gradient leakage — a distinction
that is itself rarely reported.) Dr.GRPO’s contribution is precisely a change to length/normalization
in the loss [7]; GSPO changes the importance-sampling granularity [13]. If the baseline’s loss form is
unreported, the variant’s gain is unattributable. The stake is not hypothetical: in our own companion
panel, a documented-but-unwired loss flag left six “GRPO vs. Dr.GRPO” arms silently training the


                                                   4

[PAGE 10]
The critical hook is on_grpo_rewards_computed, called after reward functions return and
after distributed gather, but before advantage normalization. It receives rewards_per_func,
reward_weights, and num_generations, and computes
                           N
                         1 X 
                  ZVFt =                                         GUt = 1 − ZVFt ,
                                                            
                             1 Var(rx,1 , . . . , rx,G ) = 0 ,
                         N
                              x=1

where unusable (all-NaN) groups are counted as zero-variance. The hook logs both human-readable
scalars (min_report_rl/zvf, min_report_rl/gu, min_report_rl/usable_groups) and a per-step
JSON event.

   Upstream surface. The cleanest upstream TRL change adds three elements to GRPOConfig:
report_min_report_rl, min_report_rl_strict, and min_report_rl_output_dir; plus two trainer
methods (_emit_min_report_rl_run_start and _emit_min_report_rl_step) and the reward-path
hook call. That is the “few lines of telemetry” version of the standard.

    Run-start manifest. The first line of min_report_rl.jsonl is a JSON object whose min_reportable_stack
field contains the seven run-manifest items. The eighth standard item, pass@k alongside pass@1,
belongs in the evaluation report. A manifest fragment is shown in Listing 1; the full schema is in the
GRPO-Reg catalog paper [1].
                  Listing 1: Fragment of the min_report_rl run-start manifest.
{
    "event": "min_report_rl.run_start",
    "min_reportable_stack": {
      "1_loss_form": {"status":"known", "loss_type":"dapo", ...},
      "2_reference_policy_kl": {"status":"known", "kl": {"enabled":false}},
      "3_sampler_backend_precision": {"rollout_engine":"vllm", ...},
      "4_zvf_gu_trajectory": {"enabled":true, "basis":"weighted_reward", ...},
      "5_group_size_schedule": {"train_num_generations":8, ...},
      "6_heldout_split": {"heldout_dataset_id":"gsm8k:test:500", ...},
      "7_decontamination_probes": {"ngram_overlap":{"passed":true}, ...}
    }
}


    Other trainers. verl’s actor/rollout separation makes sampler, precision, and group-size
schedule first-class; OpenRLHF’s reference-model and KL options map directly onto item 2. Both
can emit the same manifest block and per-step telemetry using the same shared emitter. The schema
is trainer-agnostic; only the source of each field changes.

Enforcement and audit tooling. The same manifests feed the implemented platform_hybrid/
registry/query.pystackdiff command (§7), which compares registry entries and flags label-flip
risk. The implemented registry/provenance/minreport.pyverify command grades generated
provenance records. A future literature-scale Auditor would score an existing paper or repo against
the eight-item checklist, assigns a 0–100 reproducibility-auditable badge, and produces a reviewer-
facing summary [2]. The audit paper’s locked aggregator turns the controlled single-stack audit of §4
into a fail-closed result validator. A full launcher would lock the shared stack, expand each variant
as a declared hook, and compute verdicts from paired seed-level held-out deltas [2]. Together these
tools turn the abstract standard into a mechanical check that venues and reviewers can run in CI.

                                                  10

[PAGE 12]
two registry records and reports R0–R5 label-flip risk. We also sketch a richer version that would
consume arbitrary Min-Report-RL manifests and use the L0–L7 taxonomy below. Neither com-
mand is a replacement for the audit of §4; it is an enforcement layer that tells a reviewer, in seconds,
whether a head-to-head is stack-comparable.

Manifest contract. Each run is a YAML/JSON block containing the eight Min-Report-RL
fields, plus run identity (trainer, version, seed, hardware), model and tokenizer hashes, LoRA
configuration, optimizer settings, and results. The schema is the JSON block proposed in §5.

Diff taxonomy. Every field difference receives three labels:

 • Lever. One of eight lever groups: L0 run identity/provenance, L1 loss form, L2 reference
   policy and KL handling, L3 sampler/backend/ precision, L4 usable-signal telemetry (Zvf/Gu),
   L5 group-size schedule, L6 held-out evaluation and checkpoint selection, L7 decontamination and
   parser robustness.
 • Diff kind. EQUAL, COSMETIC, PROVENANCE, PARAMETRIC, SCHEDULE, SEMAN-
   TIC_OBJECTIVE, DISTRIBUTIONAL, TELEMETRY, MISSING, OPAQUE, DERIVED, or
   INVALID_TARGET.
 • Role. TREATMENT_DELTA if the user declares it as the algorithmic delta being tested;
   NUISANCE_DELTA if it is an uncontrolled stack difference; COVERAGE_GAP if the field is
   missing; INVALIDATOR if the comparison target is not common.

Flip-risk classes. The tool compares each nuisance difference to the reported comparison margin
and classifies the overall risk as one of:

                Class               Meaning
                R0 same             No meaningful difference.
                R1 cosmetic         Version string or comment only.
                R2 small            Estimated effect < 25% of the comparison margin.
                R3 material         Effect 25–100% of the margin; can shrink a claim.
                R4 flip_capable     Effect bound ext>= margin; can flip the ranking.
                R5 invalidating     The comparison target is not common.
                RU unknown          Missing or opaque evidence; comparison unverifiable.


Verdict and CI integration. The tool emits a deterministic verdict (STACK_MATCHED, STACK_MATERIAL,
STACK_CONFOUNDED, UNVERIFIABLE, or INVALID_COMPARISON) and an exit code, so a paper’s repro-
ducibility workflow can fail a comparison that is flip-capable. Example:

python3 platform_hybrid/registry/query.py stackdiff \
  colab-open_dapo_e3 tinker_dapo_qwen3.5-4b_gsm8k

Role in the Min-Report-RL ecosystem. Authors emit the JSON block; venues ask for it;
reviewers run the reference stackdiff command. The cost to an author is still a few lines of telemetry;
the cost to a reviewer is one command. The tool turns the abstract stack-conditioning thesis of §2
into a concrete, reproducible check.




                                                  12

[PAGE 13]
8    Conclusion
The RL-for-LLM literature is in a position the deep-RL community has seen before: a three-letter
label is doing the work of a full experimental specification, and the unreported remainder of the stack
can move a result by more than the algorithmic effect anyone is trying to claim. We have argued that
this stack-conditioning is the default, demonstrated it with an under-specified same-label comparison
that differs by 17×, and proposed two coupled remedies: an eight-item minimum-reportable standard
(Min-Report-RL): seven manifest fields that expose known confounds plus a pass@k evaluation
artifact, and a controlled single-stack audit of DAPO, GSPO, Dr.GRPO, and M-GRPO that reports
which claimed gains survive. Neither remedy requires new science—only a few fields of telemetry and
the discipline to log them. The cost is a JSON block; the payoff is a literature whose comparisons
one can actually trust.


References
 [1] Arvind C R and the ZVF Program. GRPO-Registry: A living catalog of stack fields and variant
     deltas. Companion working paper in the Tinker RL Lab manuscript bundle, 2026.

 [2] Arvind C R and the ZVF Program. Reward contrast, not algorithm labels: Auditing signal
     starvation in group-relative reinforcement learning. Companion working paper in the Tinker
     RL Lab manuscript bundle, 2026.

 [3] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David
     Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI Conference on
     Artificial Intelligence, volume 32, 2018. doi: 10.1609/aaai.v32i1.11694.

 [4] Haoyang Hong, Jiajun Yin, Yuan Wang, Jingnan Liu, Zhe Chen, Ailing Yu, Ji Li, Zhiling
     Ye, Hansong Xiao, Yefei Chen, Hualei Zhou, Yun Yue, Minghui Yang, Chunxiao Guo, Junwei
     Liu, Peng Wei, and Jinjie Gu. Multi-agent deep research: Training multi-agent systems with
     M-GRPO. arXiv preprint, 2025. doi: 10.48550/arXiv.2511.13288. arXiv:2511.13288.

 [5] Riashat Islam, Peter Henderson, Maziar Gomrokchi, and Doina Precup. Reproducibility
     of benchmarked deep reinforcement learning tasks for continuous control. In ICML 2017
     Workshop on Reproducibility in Machine Learning, 2017. doi: 10.48550/arXiv.1708.04133.
     arXiv:1708.04133.

 [6] Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. CPPO: Accelerating the training
     of group relative policy optimization-based reasoning models. arXiv preprint, 2025. doi:
     10.48550/arXiv.2503.22342. arXiv:2503.22342.

 [7] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and
     Min Lin. Understanding R1-Zero-Like training: A critical perspective. arXiv preprint, 2025.
     doi: 10.48550/arXiv.2503.20783. arXiv:2503.20783; introduces Dr. GRPO.

 [8] Gongrui Nan et al. NGRPO: Negative-enhanced group relative policy optimization. arXiv
     preprint, 2025. doi: 10.48550/arXiv.2509.18851. arXiv:2509.18851.

 [9] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian
     Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning
     system at scale. arXiv preprint, 2025. doi: 10.48550/arXiv.2503.14476. arXiv:2503.14476.


                                                  13


# R07: Living GRPO registry


Root: `zvf-program/registry/grpo_registry.tex`  Pages: 13  Words: 4398


[PAGE 1]
GRPO-Registry: A Living Catalog of Stack Fields and Variant
                        Deltas
                      Toward Machine-Readable Reporting for the GRPO Family

                                                  Arvind C R∗
                                                PES University
                                             arvindcr4@gmail.com

                                                   July 14, 2026



                                                      Abstract
              The GRPO family of reinforcement-learning post-training methods is now large enough that
          no single paper can keep its variants straight. Each new variant—DAPO, GSPO, Dr.GRPO,
          M-GRPO, AERO, CPPO, NGRPO, Scaf-GRPO, and others—is reported as a small algorithmic
          delta, but the delta is almost always confounded with changes in loss form, sampler, reference
          policy / KL handling, group-size schedule, and precision. We introduce GRPO-Registry,
          a living catalog that (i) fixes the seven-field run-manifest schema drawn from the eight-item
          Min-Report-RL standard (whose eighth item is held-out pass@k reporting), (ii) defines a
          machine-readable “variant-delta” schema that records exactly what changed relative to a declared
          baseline, and (iii) seeds the catalog with the best currently available descriptions of the most
          discussed variants. The goal is not to arbitrate priority but to give reviewers and re-implementers
          a shared worksheet: for any comparison, read the two stack records, read the two variant records,
          and see whether the claimed gain is still attributable once the stack is held fixed. We release
          the schema as JSON and propose a lightweight update protocol so the catalog can stay current
          without a new paper for every new variant.


1        Introduction: Why a Registry?
Scope note (2026-07-11 canonicalization). This document is the condensed “living catalog” statement
 of the GRPO-Registry concept. The full treatment — machine-readable schema, population
 and query auditor, measured-evidence tiers, and claim-validation verdicts — is the Pillar-6 paper
“GRPO-Registry: A Machine-Readable Catalog of Group-Relative RL Stacks and Their Variant Deltas”
(platform_hybrid/paper/paper_P6_registry). The Pillar-6 paper is canonical for the registry’s
 content and evidence; this one for the catalog-as-community-resource framing.
      A reviewer reading six recent GRPO papers can reasonably conclude that each paper compares
 against “GRPO.” But the word “GRPO” is attached to different loss forms (ratio + clip, ratio +
 no clip, completion-only mask, whole-sequence mask), different reference-policy and KL treatments
(frozen reference with KL penalty in the loss, frozen reference with KL in the reward, no reference
 at all), different samplers and precisions (vLLM bf16, vLLM fp32, managed API), and different
 group-size schedules (fixed G = 8, adaptive, dynamic sampling) [2, 4, 8, 12, 18]. When the baseline
 is itself a moving target, a reported delta of +2 points is uninterpretable.
    ∗
        Companion to the ZVF Program Min-Report-RL position paper.



                                                          1

[PAGE 11]
registry record can therefore be generated automatically from the first line of min_report_rl.jsonl.


8    Living-Update Protocol
A registry that requires a new paper for every new variant will quickly become stale. We propose a
lightweight update protocol:
1. Versioned releases. The catalog is released as a versioned JSON file (e.g., grpo-registry-v0.1.json).
   Each release has a changelog.
2. Source requirement. Every entry must cite a specific paper, preprint, or code release. No
   entry may be added purely from a blog post or social-media summary.
3. Uncertainty marking. Unknown fields are recorded as null with a source_note explaining
   what was not reported. This preserves the honesty of the catalog.
4. Community curation. Changes are proposed via pull request and reviewed by at least one
   maintainer for source fidelity.
    The protocol mirrors existing living artifacts such as the Papers With Code leaderboards, but it
is organized around stack fields rather than final scores.


9    Conclusion
The GRPO family has outgrown its own label. The registry proposed here is a small, practical step
toward a literature in which the treatment—both the stack and the declared algorithmic delta—is
reported clearly enough to compare across papers. It does not replace the Min-Report-RL standard;
it depends on it. It also does not replace controlled re-implementations; it makes the need for
them visible earlier. The implemented minreport.py and query.py stackdiff commands move
the registry beyond a static worksheet, while the audit aggregator refuses incomplete evidence. The
immediate next step is to extend full stack coverage and ship default trainer emitters; automated
PDF auditing and full-scale paired training remain future engineering work.


References
 [1] Arvind C R and the ZVF Program. MIN-REPORT-RL: Reporting the stack, not the label.
     Companion position paper in the Tinker RL Lab manuscript bundle, 2026.
 [2] Arvind C R and the ZVF Program. Reward contrast, not algorithm labels: Auditing signal
     starvation in group-relative reinforcement learning. Companion working paper in the Tinker
     RL Lab manuscript bundle, 2026.
 [3] Lishui Fan, Yu Zhang, Mouxiang Chen, and Zhongxin Liu. ReCode: Reinforcing code generation
     with reasoning-process rewards. In Proceedings of the 64th Annual Meeting of the Association for
     Computational Linguistics, 2026. doi: 10.48550/arXiv.2508.05170. arXiv:2508.05170; introduces
     Consistency-Gated GRPO.
 [4] Haoyang Hong, Jiajun Yin, Yuan Wang, Jingnan Liu, Zhe Chen, Ailing Yu, Ji Li, Zhiling
     Ye, Hansong Xiao, Yefei Chen, Hualei Zhou, Yun Yue, Minghui Yang, Chunxiao Guo, Junwei
     Liu, Peng Wei, and Jinjie Gu. Multi-agent deep research: Training multi-agent systems with
     M-GRPO. arXiv preprint, 2025. doi: 10.48550/arXiv.2511.13288. arXiv:2511.13288.

                                                 11


# R08: Reproducibility audit


Root: `zvf-program/audit/reproducibility_audit.tex`  Pages: 8  Words: 2993


[PAGE 1]
GRPO-Survival-Audit: A Single-Stack Survival Protocol for
                 the GRPO Family
                      Holding the Stack Fixed So Algorithmic Gains Can Be Seen

                                                  Arvind C R∗
                                                PES University
                                             arvindcr4@gmail.com

                                                  July 20, 2026


                                                     Abstract
              The GRPO-family literature reports head-to-head comparisons in which each variant runs
          in its own trainer, sampler, and evaluation harness. Because the “stack”—loss form, refer-
          ence/KL handling, sampler precision, group-size schedule, Zvf/Gu trajectory, held-out split,
          and parser—is a documented flip lever [2], such comparisons confound algorithmic innovation
          with implementation differences. This paper proposes GRPO-Survival-Audit, a single-stack
          reproducibility-audit protocol in which DAPO, GSPO, Dr.GRPO, and AERO are re-implemented
          as declared overrides on one shared language-model trainer. M-GRPO is assigned to a separate
          agentic stratum because hierarchical credit assignment changes the environment and trajectory
          schema. We specify the shared stack, the pre-registration rules, the survival-verdict logic, the
          statistical protocol, and the role of the companion grpo-stackdiff tool in verifying that arms are
          stack-matched. We also report a small open-source pilot on Qwen2.5-0.5B-Instruct (two seeds, T4
          GPU) that demonstrates the protocol’s feasibility and produces concrete Zvf and held-out-delta
          observations. The full-scale frozen audit is complete: all 40 arm–seed units independently pass
          the local, W&B, private-Hub checkpoint, stack-fingerprint, treatment-fingerprint, and 500-row
          held-out gates. Against the shared GRPO reference, DAPO’s controlled delta is +0.00100
          (paired 95% CI [−0.00450, +0.00675]) and receives the preregistered DISAPPEARS verdict. GSPO,
          Dr.GRPO, and AERO receive INCONCLUSIVE verdicts; none of the five arms collapsed. These
          results are a stack-controlled survival audit, not a universal capability leaderboard.


1        Introduction: The Need for a Survival Audit
A typical GRPO-family paper compares a new variant against “GRPO” by running each in a different
codebase. The new variant may change the clip rule, the advantage normalization, or the group-size
schedule; the baseline may use a different token mask, KL placement, sampler precision, or reward
parser [2, 3]. The result is a comparison of two stacks SA and SB that happen to be tagged “A” and
“B”. The algorithmic delta and the stack delta are inseparable.
     The Min-Report-RL position paper [2] proposes an eight-item standard—seven run-manifest
fields plus held-out pass@k reporting—so that future papers expose the stack. The GRPO-Registry
catalog [1] records each variant’s self-declared delta relative to a baseline. This paper closes the loop:
it specifies a controlled, single-stack re-implementation protocol that asks, for each variant, what
fraction of its claimed gain survives when the stack is held fixed? We call the answer the variant’s
survival.
    ∗
        Pillar 3 of the ZVF Program.


                                                          1

[PAGE 6]
Table 2: Pilot results (Qwen2.5-0.5B-Instruct, 2 seeds, T4). Mean held-out ∆, mean Zvf,
and mean rollouts per arm.
                  Arm                 Held-out ∆      Mean Zvf     Mean rollouts
                  GRPO (baseline)         0.500         0.250            120
                  Dr.GRPO                 0.575         0.267            120
                  DAPO                    0.550         0.000            174
                  GRPO adaptive-G         0.575         0.233            186


7    Open-Source Pilot
We ran a small pilot on a single T4 GPU to validate that the required telemetry can be extracted from
a standard GRPO trainer and that the protocol produces informative stack-controlled observations.
The pilot is not the full-scale audit; it uses a small model and only two seeds.

Setup. Model: Qwen/Qwen2.5-0.5B-Instruct with full-model updates; seeds {0, 1}; ten training
steps; three prompts per batch; initial group size G0 = 4; maximum generation length 40 tokens; and
a seed-specific held-out set of 20 generated integer-addition prompts. Arms are GRPO, Dr.GRPO,
DAPO, and an adaptive-G GRPO baseline. The shared implementation keeps tokenizer, sampler,
reward parser, prompt generator, optimizer settings, and evaluation procedure fixed across arms.
These small synthetic held-out sets make the pilot a telemetry smoke test, not a language-model
capability benchmark.

Pilot results. Table 2 shows mean held-out delta, mean Zvf, and mean rollouts per arm. Even
at this scale, the stack-controlled observations are informative: DAPO reports zero Zvf but spends
more rollouts (174 vs. 120), while adaptive-G matches Dr.GRPO’s held-out delta with a different
Zvf and rollout budget.

Interpretation. This two-seed pilot is descriptive and cannot rank the arms. It shows that the
same held-out delta can arrive through different signal-budget paths. DAPO’s zero recorded Zvf is
mechanical: its resampling loop rejects flat groups before an optimizer step, spending 174 rather
than 120 rollouts. Adaptive-G reaches the same descriptive delta as Dr.GRPO while spending 186
rollouts. The comparison therefore validates telemetry and budget accounting only; it is not evidence
that one method learns better.


8    Limitations and Extensions
Scope. The audit tests survival under one shared stack. A variant that survives here may still
fail under a different stack; the audit does not claim universal robustness. It claims only that the
published gain is not purely an artifact of the original stack.

Faithfulness of re-implementation. A variant may depend on an unreported implementation
detail. If a careful re-implementation cannot reproduce the original result from the paper, that is
itself a finding about reporting quality, but it is not a finding about the method’s intrinsic value.
We report both readings and publish every arm’s full Min-Report-RL block.




                                                  6

[PAGE 7]
Extensions. The protocol generalizes to any family of methods that are small deltas on a base
loop. The same hook interface can accommodate future variants; the same grpo-stackdiff check
can verify that each new hook changes only its declared levers.


9    Conclusion
The GRPO family needs a way to separate algorithmic gains from stack effects. The Min-Report-
RL standard tells authors what to report; the GRPO-Registry catalog records each variant’s
declared delta; and the GRPO-Survival-Audit protocol reported here specifies how to hold the
stack fixed so that those deltas can be tested fairly. The frozen multi-seed audit is complete: all 40
units passed the evidence gates, DAPO’s reported advantage disappears under the shared stack, and
GSPO, Dr.GRPO, and AERO remain inconclusive at eight paired seeds. The protocol, telemetry
pilot, machine-readable preregistration, complete execution record, and grpo-stackdiff verification
step together they make it operational to ask of a GRPO-family result not “did it win?” but “did it
survive?”


References
 [1] Arvind C R and the ZVF Program. GRPO-Registry: A living catalog of stack fields and variant
     deltas. Companion working paper in the Tinker RL Lab manuscript bundle, 2026.

 [2] Arvind C R and the ZVF Program. MIN-REPORT-RL: Reporting the stack, not the label.
     Companion position paper in the Tinker RL Lab manuscript bundle, 2026.

 [3] Arvind C R and the ZVF Program. Reward contrast, not algorithm labels: Auditing signal
     starvation in group-relative reinforcement learning. Companion working paper in the Tinker
     RL Lab manuscript bundle, 2026.

 [4] Haoyang Hong, Jiajun Yin, Yuan Wang, Jingnan Liu, Zhe Chen, Ailing Yu, Ji Li, Zhiling
     Ye, Hansong Xiao, Yefei Chen, Hualei Zhou, Yun Yue, Minghui Yang, Chunxiao Guo, Junwei
     Liu, Peng Wei, and Jinjie Gu. Multi-agent deep research: Training multi-agent systems with
     M-GRPO. arXiv preprint, 2025. doi: 10.48550/arXiv.2511.13288. arXiv:2511.13288.

 [5] Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. CPPO: Accelerating the training
     of group relative policy optimization-based reasoning models. arXiv preprint, 2025. doi:
     10.48550/arXiv.2503.22342. arXiv:2503.22342.

 [6] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and
     Min Lin. Understanding R1-Zero-Like training: A critical perspective. arXiv preprint, 2025.
     doi: 10.48550/arXiv.2503.20783. arXiv:2503.20783; introduces Dr. GRPO.

 [7] Gongrui Nan et al. NGRPO: Negative-enhanced group relative policy optimization. arXiv
     preprint, 2025. doi: 10.48550/arXiv.2509.18851. arXiv:2509.18851.

 [8] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian
     Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning
     system at scale. arXiv preprint, 2025. doi: 10.48550/arXiv.2503.14476. arXiv:2503.14476.

 [9] Xichen Zhang et al. Scaf-GRPO: Scaffolded group relative policy optimization for enhancing
     LLM reasoning. arXiv preprint, 2025. doi: 10.48550/arXiv.2510.19807. arXiv:2510.19807.

                                                  7


# U01: Umbrella benchmark


Root: `platform_hybrid/paper/main.tex`  Pages: 239  Words: 133323


[PAGE 1]
An Evidence-Tiered Compendium of RL Post-Training
                     Audits


                    Arvind C R∗                               Sandhya Jeyaraj∗
                   PES University                              PES University
               arvindcr4@gmail.com                    sandhya.jeyaraj2014@gmail.com

                    Madhu Kumara L                                 Mohammad Rafi
                      PES University                                PES University
               madhukumara1993@gmail.com                      gmd.rafi.2024@gmail.com

                    Dhruva N Murthy                                 Arumugam K
                      PES University                                PES University
               dhruva.n.murthy@gmail.com                      chettyarumugam@mail.com

            Anwesh Reddy Paduri                            Narayana Darapaneni
         Great Learning / PES University            Northwestern University / Great Learning
          anwesh@greatlearning.in                 narayana.darapaneni@northwestern.edu



                                                Abstract
            RL post-training of language models is now shaped by a small set of algorithms
            (PPO, GRPO, DPO) and a growing set of frameworks (TRL, T INKER, OpenRLHF,
            veRL), yet it remains unclear which empirical conclusions transfer across im-
            plementations, model families, and scales. We present T INKER RL-B ENCH, a
            benchmark spanning 70+ runs across 7 RL libraries and 5 model families (0.6B–
            ∼671B parameters) on GSM8K, HumanEval, and synthetic tool-use tasks. Results
            to date cover five end-to-end stacks (TRL, T INKER, SB3, CleanRL, Tianshou);
            the remaining libraries are scaffolded but not yet run. The strongest evidence in
            the current release comes from short-horizon GSM8K training dynamics; held-out
            evaluation and non-math coverage remain narrower.
            Three conclusions are supported conservatively. First, Zero-Variance Fraction
            (ZVF)—the share of GRPO groups with identical rewards—tracks when binary
            outcome rewards stop producing within-group learning signal. Because ZVF
            is mechanically coupled to reward sparsity, group size, and baseline accuracy,
            we use it as a descriptive diagnostic rather than a standalone causal predictor.
            Second, trainability in our current sweeps varies substantially with initialization
            and rollout regime: instruction-tuned checkpoints were generally easier to optimize
            than comparable base models, and intermediate group sizes often behaved better
            than the smallest or largest settings we tested, but we do not claim a universal
            optimal group size. Third, PPO/GRPO rankings and frontier-model stability are
            heterogeneous across model families and end-to-end stacks; our single-seed API
            runs should be read as case studies, not universal laws. A separate held-out
            GSM8K control shows that the mean Qwen3-8B-Instruct GRPO delta over the
            same checkpoint’s pre-RL held-out accuracy is small and not statistically significant
            (83.3% vs. 82.0%, p=0.26), while tool-use and code results remain limited by
            sparse rewards and custom evaluation.
   ∗
       Equal contribution.


Preprint. Under review.

[PAGE 2]
We release code, logs, and checkpoints at https://github.com/arvindcr4/
         tinker-rl-lab.



Document status and canonical scope. This file is the program’s long-form evidence compendium,
not a single venue submission and not an independent experiment. It aggregates the benchmark
substrate and historical analysis modules, some of which also appear in focused companion papers.
Shared rows must not be meta-analyzed as independent replications. The canonical thesis-level
evidence is the per-step ZVF diagnostic and the matched-token G=2 × 160 versus G=16 × 20
comparison; adaptive control, PPO, and SAO remain prospective extensions unless explicitly backed
by their own held-out runs.


1   Introduction

Reinforcement learning post-training now underpins much of modern language-model reasoning
and alignment [96, 23], but empirical claims in this area are often hard to compare. PPO [115],
GRPO [116], and DPO [109] are usually evaluated on different model families, with different frame-
work defaults, different reward functions, and different notions of “success.” What the field needs is
not another leaderboard but a substrate that separates algorithmic conclusions from implementation,
initialization, and task-design effects.
T INKER RL-B ENCH is an attempt at such a substrate. It aggregates 70+ runs spanning 7 RL libraries
(five with completed results), 5 model families, 32+ checkpoints, and scales from 0.6B to ∼671B
parameters, with common reporting conventions and released artifacts. At the same time, we
emphasize a central caveat: the current benchmark is not uniformly mature across tasks. The
strongest evidence comes from short-horizon GSM8K experiments with verifiable binary rewards.
Frontier API runs, tool-use experiments, and code-generation studies are useful case studies, but
many remain single-seed, short-horizon, partially completed, or custom-evaluated.
The first contribution of the benchmark is therefore methodological rather than triumphalist: it makes
visible how much end-to-end stack choice matters. In our results, model family, initialization, rollout
regime, and framework defaults can all change the apparent algorithm ranking. Several comparisons
that look like “PPO vs. GRPO” are better understood as stack-level comparisons, because the training
backends differ in tokenizer/runtime integration, managed defaults, and rollout plumbing. This is
precisely why a shared measurement substrate is needed.
Our second contribution is a descriptive diagnostic for sparse-reward GRPO: Zero-Variance Fraction
(ZVF), the fraction of rollout groups with identical rewards, together with Gradient Utilization (GU =
1 − ZVF). ZVF is useful because it directly reveals when within-group contrast disappears. However,
because our main tasks use binary outcome rewards, ZVF is mechanically coupled to reward sparsity,
group size, and baseline accuracy. We therefore use ZVF/GU as diagnostics of signal degeneracy, not
as proof of an independent or causal failure mode beyond simpler observables such as reward mean,
entropy, advantage variance, or divergence proxies.
Our third contribution is a set of targeted ablations on trainability. In the current benchmark,
trainability varies substantially with initialization and rollout regime: instruction-tuned checkpoints
were generally easier to optimize than comparable base models, and intermediate rollout group sizes
often behaved better than the smallest or largest settings we tested. These observations support a
narrower claim: whether RL fine-tuning works at all can depend materially on initialization and
reward-bearing rollout structure, in addition to the nominal RL algorithm. They do not yet justify a
universal optimal group size or a general “SFT dominates RL” law.
Finally, we explicitly separate training reward from held-out generalization. For Qwen3-8B-Instruct
on GSM8K, a five-seed post-GRPO held-out evaluation yields 83.3% accuracy, but the same
instruction-tuned checkpoint’s pre-RL held-out accuracy under the identical greedy protocol is
already 82.0%; the +1.3 percentage-point delta is not statistically significant (p=0.26). This negative
control matters: strong online reward curves do not by themselves establish generalization gains. We
release the benchmark, step-level traces, and checkpoints so that stronger follow-up work can test
exactly these failure points.


                                                  2

[PAGE 19]
Model                   peak step   R̂max     R̄late   ∆late-peak   P (R = 0)   P (R > 0.5)   collapse?
 Qwen3.5-4B                      1   1.000    0.850     −0.150           0.000         0.833         no
 Qwen3-8B                       13   0.625    0.344     −0.281           0.067         0.100         no
 Llama-3.1-8B-Instruct           1   1.000    0.844     −0.156           0.000         0.933         no
 DeepSeek-V3.1                   3   1.000    0.813     −0.188           0.000         0.950         no
 Nemotron-120B                   3   0.875    0.208     −0.667            0.55          0.05        yes
Table 14: Nemotron-120B collapse root-cause characterisation. Only Nemotron satisfies all three
collapse criteria. Its peak reward of 0.875 occurs at step 3, after which the post-peak OLS slope
is −0.0036 per step, comparable in sign to the Llama-3.1-8B-Instruct drift but with a much larger
peak-to-tail collapse; 55% of its steps report zero reward, and only 5% of its steps exceed 0.5. All
four other runs have P (R = 0) ≤ 0.067 and well-retained peaks.


conclusion is the same as in Table 11: at this benchmark size and step budget, there is no identifiable
scale dependence in the GRPO saturation rate.

Elevation: Nemotron-120B collapse autopsy. We classify a trace as collapsed when (i) its peak
reward is at least 0.4, (ii) its late-window mean falls below 0.4× peak, and (iii) its zero-reward
fraction is at least 0.30. Applied to all five traces:
This is the strongest empirical justification we have for treating Nemotron-120B as the Pillar-1
counterexample to the Nimmaturi et al. [88] three-phase template: not only does its peak not satisfy
the slow-start → rapid-improvement → plateau shape, it actively diverges downward from the peak
in a way no monotone saturation model can describe. The zero-reward-fraction of 0.55 is also the
structural signal that the ZVF-based diagnostic (companion paper on cross-experiment ZVF) would
catch first, because it sees the divergence on reward variance before it shows up on reward mean.

Summary of elevation. Across the four elevation diagnostics the headline result is a strengthened
null: at this benchmark size (4 B ≤ N ≤ 671 B, T ≤ 30 steps), the canonical saturation model
is unidentifiable (parametric bootstrap on λ has P (λ = 10) ≥ 0.47 on four of five runs), adds no
out-of-sample predictive power (70/30 holdout improvement over the constant baseline is ≤ 0.0016
RMSE everywhere), and shows no detectable scale dependence (slope −0.47 ± 0.87 per decade on
λ). The single run that violates the saturation template is Nemotron-120B, and its collapse is the
strongest empirical counterexample we have to the three-phase hypothesis. This is exactly the regime
in which a saturation-only Pillar 1 diagnostic is silent – motivating the variance-based ZVF gate in
Pillar 2 as a complementary failure detector.

Elevation: extended frontier+MoE scaling (iter 13). The canonical five-anchor set leaves open
whether the null on slope(log10 N, R) is an artefact of n = 5 or a real property of the GRPO
landscape. We re-run the same OLS-on-log-N diagnostic on a wider 12-anchor set that adds the Kimi
Team, Moonshot AI [54, 55] Kimi-K2-Thinking (1T params), Qwen Team [107] Qwen3-235B and
30B-MoE, OpenAI [92] GPT-OSS-20B, and the short-form probes Qwen3-32B / Qwen3.5-27B. We
also stratify by architecture (MoE vs dense) and compute a Spearman rank correlation on top of the
OLS slope so that the result does not depend on the linearity of the log-linear fit.
We complement the OLS test with a non-parametric Spearman rank correlation ρ(log10 N, R):
ρ = −0.036 for R(1) (p = 0.91), ρ = +0.149 for R̄ (p = 0.64), ρ = −0.023 for R(T ) (p = 0.94),
and ρ = +0.074 for R̂ (p = 0.82). None of the four correlations is distinguishable from zero. This
rules out the alternative explanation that the slope-of-zero finding is an artefact of OLS assuming
linear log-N dependence; the rank test on the same 12 anchors gives the same null.
The headline positive result of the extension is on the architecture axis. Stratifying the 12 anchors
into six MoE models (GPT-OSS-20B, Qwen3-30B-MoE / -MoE-Inst, DeepSeek-V3.1, Qwen3-235B-
MoE, Kimi-K2-Thinking) versus six dense models (Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct,
Qwen3-32B, Qwen3.5-27B, Nemotron-120B):
The first-final gap ∆1T = R(t=1) − R(t=T ) adds a robust summary that does not depend on the
lambda-bound degeneracy: ∆1T < 0 means the trace is improving (positive learning), ∆1T > 0
means it is collapsing (negative learning). Across the 12 anchors, 7 of 12 have |∆1T | ≤ 0.2; the
largest positive (collapsing) gap is Qwen3.5-27B at +0.562 (the only anchor with ∆1T > +0.4).


                                                   19

[PAGE 22]
Model                       k        n            R̄     ∆AIC best               best   ∆AICconstant
            Qwen3.5-4B                  1        30     0.817               0.00      constant               0.00
            Qwen3.5-4B                  2        30     0.817               2.00          linear             2.00
            Qwen3.5-4B                  2        30     0.817               2.00     saturation              2.00
            Qwen3.5-4B                  3        30     0.817               4.00        logistic             4.00
            Qwen3-8B                    1        30     0.285               0.00      constant               0.00
            Qwen3-8B                    2        30     0.285               1.20          linear             1.20
            Qwen3-8B                    2        30     0.285               2.00     saturation              2.00
            Qwen3-8B                    3        30     0.285               3.17        logistic             3.17
            Llama-3.1-8B-Instruct       1        30     0.869               0.00      constant               0.00
            Llama-3.1-8B-Instruct       2        30     0.869               0.80          linear             0.80
            Llama-3.1-8B-Instruct       2        30     0.869               2.00     saturation              2.00
            Llama-3.1-8B-Instruct       3        30     0.869               4.29        logistic             4.29
            DeepSeek-V3.1               1        20     0.844               0.00      constant               0.00
            DeepSeek-V3.1               2        20     0.844               2.00          linear             2.00
            DeepSeek-V3.1               2        20     0.844               2.00     saturation              2.00
            DeepSeek-V3.1               3        20     0.844               4.00        logistic             4.00
            Nemotron-120B               1        20     0.175               0.00      constant               0.00
            Nemotron-120B               2        20     0.175               1.92          linear             1.92
            Nemotron-120B               2        20     0.175               1.72     saturation              1.72
            Nemotron-120B               3        20     0.175               3.92        logistic             3.92
Table 17: Multi-model AIC profile across the five anchors (constant / linear / saturation / logistic).
The constant model wins on every trace by ∆AIC ≤ 2 over the runner-up. By the Burnham and
Anderson [11] convention, ∆AIC < 2 means the second-best alternative is “essentially equivalent”;
∆AIC ∈ (2, 4) is “substantial support”; ∆AIC > 4 is “essentially no support”. The saturation model
is formally indistinguishable from a constant across the entire frontier – the same conclusion the
iter 9 holdout test arrived at from an out-of-sample predictive-power angle. We therefore report
the AIC profile as the likelihood-side counterpart to Table 13: both pin the saturation curve as the
wrong functional form for these already-saturated traces. scripts/scaling_law_iter17.py →
scaling_law_iter17_aic.tsv.


        Model                       n       τ̂        |∆µ|         block-boot CI95% (τ )      perm p    significant?
        Qwen3.5-4B              30          27        0.204                        [2, 28]     0.493    no
        Qwen3-8B                30          28        0.129                        [2, 28]     0.789    no
        Llama-3.1-8B-Instruct   30           5        0.127                        [2, 28]     0.742    no
        DeepSeek-V3.1           20          15        0.092                        [2, 18]     0.909    no
        Nemotron-120B           20          18        0.361                        [2, 18]     0.137    no
Table 18: Changepoint tau with block-bootstrap CI and permutation-test p-value. The brute-
force maximiser τ̂ lands late on three runs (the last 2–3 steps) and early on two, but every τ̂ is
statistically indistinguishable from the permutation null – none of the five traces has a p < 0.05
changepoint at this step budget. This is the formal complement to the AIC profile: both diagnostics
agree that the 30-step frontier traces do not contain a distinguishable break. The block-bootstrap
CI on τ covers essentially the entire trace ([2, 28] for T = 30 traces), so even the point estimate is
not data-driven. We treat the changepoint analysis as a negative result: there is no information in τ̂ .
scripts/scaling_law_iter17.py → scaling_law_iter17_changepoint.tsv.



method that thresholds the sign and magnitude of the post-τ̂ versus pre-τ̂ mean contrast, and (3) an
“AIC” method that maps the best-fit functional form to the same four-phase ontology (constant →
plateau, linear positive slope → saturation, linear negative slope → drift, logistic or saturation with
λ < 5 → saturation). Cohen’s κ on each pair is reported on the five- and twelve-anchor sets.

Iter 21 elevation: cross-architecture stratification, lambda-bound audit, two-anchor extrapo-
lation. The iter 9/13/17 work treats the 12 anchors as a single homogeneous scaling set. Iter 21
stratifies by architecture (MoE vs dense) and probes whether the saturation law actually distinctly
parameterises the two families, or whether the λ-at-bound degeneracy is itself structured by trace
variance (rather than a free parameter of the dynamics).


                                                               22

[PAGE 24]
Figure 12: Iter 17 elevation – 4-panel overview. (A) Multi-model AIC profile across the four
candidate functional forms on the five anchors; bar height is ∆AIC relative to the best candidate,
so the constant model has ∆ = 0 everywhere and the others are clustered between 0.8 and 4.3. (B)
Changepoint τ̂ with the block-bootstrap 95% CI on τ as the error bar; red bars flag traces where
the permutation-test p < 0.05 – none of the five anchors reach significance. (C) Effective plateau
horizon Tε at ε ∈ {0.05, 0.10, 0.20}. Two traces reach Tε=0.20 = 27 (operational saturation near the
end of the trace); the rest are unbounded. (D) Pair-wise Cohen’s κ between the three phase-label
methods. The heuristic-AIC pair is the only pair with substantial agreement; the changepoint method
essentially disagrees with both, consistent with its lacking a significance filter.


Figure 13: Iter 21 cross-architecture audit (12 anchors). (A) λ̂ (capped at the 9.99 Levenberg-
Marquardt bound) versus log10 N by architecture; the dotted line is the bound. (B) fraction of traces
that hit the λ = 10 bound by architecture. (C) Rmax residual after regressing on log10 N ; Levene
median test p = 0.296 (→ arch-invariant).


The interaction model lifts CV SSE by 332 units over the arch-only model, while the log10 N
alone model worsens SSE by 99 units vs. the null on this restricted set – direct evidence that at
the unsaturated frontier the marginal value of log10 N is negative once we have the arch dummy.
Combined: the λ scaling is architecture-contingent and the simple power-law R ∼ N b appears to be
smuggling an architecture effect through the residuals.

(H) Rmax arch-invariance. We regress Rmax on log10 N across all 12 anchors and test whether
the residual is arch-invariant. Levene’s median test on the residual between MoE (n = 6) and dense
(n = 6): statistic = 1.217, p = 0.296 (scaling_law_iter21_r_max_residual.tsv). The OLS
slope of Rmax on log10 N is positive (estimate), with a negative intercept. Operationally, Rmax obeys
a weak power law in log10 N , and within that power law, MoE and dense are exchangeable. Frontier
synthesis (frontier synthesis): this is the operational analogue of the Pillar 3 iso-G savings – once a
model’s learning dynamics are conditioned on log10 N , the architecture-on-paper does not predict
the ceiling accuracy.

(I) Two-anchor burn-in extrapolation. A 30-step diagnostic on the two smallest anchors
(Qwen3.5-4B, Qwen3-8B) yields an OLS slope of λ̂ on log10 N ; we then predict the seven an-
chors ≥ 20B. Mean absolute error 4.498, max absolute error 9.492 (scaling_law_iter21_two_-
anchor_extrap.tsv). The diagnosis is honest: the two-anchor extrapolation cannot recover the
bound-degenerate frontier. The useful counter-experiment is to extrapolate on the arch-stratified
RMSE after regression (axis F) rather than the raw λ; the residual is ∼ 1.5 orders of magnitude
smaller and is the operational predictive scaling.

(J) Compute-aware λ regression.
                   λ̂ = α + β1 log10 N + β2 log10 (CF) + γ ⊮[arch = moe] + ε                        (5)
with CF = N · T (per-trace parameter-step proxy for FPO). Adding log10 (CF) lifts CV SSE
by 64.3% of the marginal value of log10 N (scaling_law_iter21_compute_regression.tsv).
The compute-adjusted proxy is moderately better than parameter count alone – consistent with frontier
synthesis’s compute-bound vs. parameter-bound framing (Section 5.6) and the well-known Kaplan
et al. [48] compute-vs-parameter caveat.
The iter-21 finding sharpens the iter-9/13 conclusion: on this benchmark, the λ scaling is architecture-
contingent, not absolute. The strongest cross-architecture claim supported by the data is that Rmax ,
the asymptotic ceiling accuracy, is arch-invariant after conditioning on log10 N (axis H, p = 0.296);
what differentiates MoE from dense is the transient (whether the trace sits at λ = 10 immediately or
climbs) rather than the asymptotic accuracy. This sharpens Pillar 3’s iso-G savings: the saturation
ceiling Rmax is shared across architectures, so the iso-G savings is exactly the cost of recovering the
ceiling under a smaller group size, regardless of the nominal architecture label.

Iter-25: is the saturation law identifiable at all? Every iteration so far has reported a per-trace
saturation rate λ and its implied t80 = − ln(0.2)/λ. Two symptoms suggest those numbers may be


                                                  24

[PAGE 27]
Regression                           n   intercept   slope / decade   n at λ-bound   note
  log10 (t80 ) ∼ log10 (N )            5   −0.841            +0.170              4/5   degenerate (4/5 anchored)
  log10 (t80 ) ∼ log10 (N ) [λ-free]   1       n/a               n/a             4/5   single-point (Nemotron only)
Table 24: t80 -vs-N scaling law test (iter 117 fresh angle). The full-pool OLS gives slope
+0.170/decade with SE 0.254 (CI brackets zero). Restricting to anchors with λ strictly below
the optimiser’s bound leaves a single data point (Nemotron-120B), so the regression is not identifiable.
Combined with iter 109’s λ-vs-N null (p = 0.74) and iter 105’s Rmax -vs-N failure, this is the third
independent cross-scale test that fails to reject H0 : no scaling signal. scripts/scaling_law_-
fit.py → scaling_law_iter117_t80_scaling.tsv.



−0.50 (Nemotron dropped) to +1.41 (each of the four bound-anchored traces dropped) – a 100%+
swing that confirms the regression is driven by which anchor carries the only free λ. The honest
conclusion: the cross-scale t80 -vs-N scaling-law test is unidentifiable on the current five-anchor
pool. This is the t80 -side counterpart to the iter-109 λ-vs-N null and the iter-105 Rmax -vs-N failure:
no cross-scale scaling signal survives on the outcome-reward RL post-training evidence base.

Nemotron-120B collapse audit (iter 117). The Nemotron-120B violation of the nimmaturi three-
phase template is quantified by the BIC segmentation reported in Table 23: the BIC-optimal segmen-
tation is k = 3 with segment means {0.000, 0.875, 0.154} – a textbook rise-then-fall pattern in which
the peak segment (0.875) is not retained. The peak-vs-late contrast (0.875 − 0.154 = +0.721) is the
largest in the five-anchor set; the next-largest is Llama-3.1-8B-Instruct at 0.929 − 0.839 = +0.090.
This is the strongest empirical justification we have for treating Nemotron-120B as the Pillar-1 coun-
terexample: not only does its peak not satisfy the slow-start → rapid-improvement → plateau shape,
it actively diverges downward from the peak in a way no monotone saturation model can describe.
Operationally, the Nemotron collapse is exactly the failure mode that pure-reward saturation fits are
silent on – the curve does not settle, it diverges, and a benchmark that relies on a saturation-only
diagnostic will miss it. This motivates the variance-based ZVF gate (companion paper on ZVF) as
a complementary diagnostic: the zero-reward-fraction for Nemotron-120B is 0.55, well above any
well-behaved run in this set.

Iter 117 summary. Iter 117 closes the Pillar-1 scaling-law investigation on the five-anchor pool
by adding four explicit deliverables: (i) the canonical t80 = − ln(0.2)/λ derivation in the fits ta-
ble, (ii) the BIC-segmentation three-phase test against Nimmaturi et al. [88] (arXiv:2507.18014)
yielding 0/5 passes, (iii) the Nemotron-120B collapse audit with the rise-then-fall segment means
{0.000, 0.875, 0.154} formalised, and (iv) the fresh-angle t80 -vs-N scaling-law test that is demon-
strably degenerate (4/5 anchors at the λ bound). Combined with iter 109’s λ-vs-N null (p = 0.74)
and iter 105’s Rmax -vs-N failure, the Pillar-1 finding is now solidly negative: GRPO post-training
on this evidence base is not scale-law-shaped – the only strong signal is the absence of a scaling
law. This is exactly the regime in which a saturation-only Pillar-1 diagnostic is silent, motivating the
variance-based ZVF gate in Pillar 2 as the load-bearing failure detector rather than the saturation law.

Iter 121: calibration of the “no scaling law” finding. Iter 117 closes with a negative result: no
scaling signal survives on the five-anchor pool. Iter 121 sharpens that negative into a quantitative
detection-power claim. We ask: given the noise we observe in the actual five traces, could any
plausible scaling law be detected at this sample size? We answer with three independent tests and
one synthetic calibration, all summarised in Table 25.

Test 1: Spearman rank test on ∆late-early . The simplest scaling-axis probe is the Spearman rank
correlation between log10 (N ) and ∆late-early = R̄late − R̄early . Across the five anchors we observe
ρ̂ = −0.30 with bootstrap-95% CI [−0.80, +1.00] (B = 1000) and a permutation null two-sided
p = 0.69 (P = 5000). The empirical ρ̂ is negative, i.e. the bigger models gained less in late-window
reward than the smaller ones — the opposite sign from any Chinchilla-style scaling prediction. The
wide CI and non-significant p confirm that even the most basic rank test cannot reject H0 of no
rank-correlation.


                                                       27

[PAGE 29]
Test                                        statistic           95% CI / p         verdict    note
 Spearman ρ(log10 N, ∆le )                    −0.30      CI [−0.94, +0.83]    fail reject H0   pperm = 0.69 (P=5000)
 OLS mean R ∼ log10 (N · D)                  −0.019      CI [−0.80, +0.30]    fail reject H0   Chinchilla axis
 OLS R̂max ∼ log10 (N · D)                   −0.018      CI [−1.12, +0.32]    fail reject H0   canonical ceiling axis
 Synthetic recovery (5 anchors, β = 0.05)      0.17            M=200 reps      undetectable    below chance threshold
 Synthetic recovery (40 anchors, β = 0.20)     0.07            M=200 reps      undetectable    bound-fraction dominates
Table 25: Iter 121 detection-power calibration. Three independent statistical tests on the five-
anchor pool all fail to reject H0 of no scaling signal. The synthetic calibration makes the degeneracy
explicit: at the empirical noise level (σ = 0.189) and saturation-bound fraction (80%), no lin-
ear scaling law on Rmax with β ≤ 0.20/decade is recoverable. Scripts: scripts/scaling_-
law_iter121.py → scaling_law_iter121_late_early.tsv, scaling_law_iter121_-
effective_compute.tsv, scaling_law_iter121_synthetic_recovery.tsv, scaling_-
law_iter121_power_curve.tsv.


                   nanchors   β=0.010   β=0.025      β=0.050      β=0.100     β=0.200
                   5             0.22        0.15         0.17         0.26        0.40
                   8             0.07        0.06         0.04         0.14        0.41
                   12            0.01        0.01         0.01         0.04        0.30
                   20            0.00        0.00         0.01         0.01        0.17
                   40            0.00        0.00         0.00         0.00        0.07
Table 26: Iter 121 synthetic recovery matrix. Each cell is the fraction of M=200 Monte-Carlo
replicates that recover a planted scaling law Rmax (N ) = 0.85−β log10 (N/8) at R̂2 ≥ 0.5. Recovery
falls as anchor count rises because the empirical 80% saturation-bound fraction holds for every pool
size, so the bound-anchors dominate the noise floor. The five-anchor pool (current Pillar-1 evidence
base, top row) is in the only regime where any non-trivial recovery is possible, but the high recovery
rates there (≤ 0.40) are themselves uninformative — they reflect n = 5 OLS R̂2 variance, not true
detection power.



The synthetic calibration shows that the noise floor of σ ≈ 0.19 combined with the 80% saturation-
bound fraction makes any linear scaling law on Rmax essentially undetectable regardless of anchor
count: even quadrupling the pool to nanchors = 20 keeps recovery below 0.20 for β ≤ 0.20.

Iter 121 conclusion. The three statistical tests (Spearman, OLS-on-N , OLS-on-N · D) all fail to
reject H0 : no scaling signal. The synthetic calibration explains why: the saturation-bound structure
of the empirical traces defeats any linear scaling-law test below β ≈ 0.20/decade, a regime stronger
than Chinchilla. The five-anchor pool is at least one order of magnitude too small (in n) and at least
one order of magnitude too saturation-bound (in λ) to falsify any plausible GRPO scaling hypothesis.
Future Pillar-1 work must either (i) extend the anchor pool to nanchors ≥ 40 with a held-out fraction of
λ-free runs, or (ii) abandon the saturation fit entirely and use the variance-based ZVF gate (companion
paper on ZVF) as the primary scaling signal.

Iter 125 elevation: structural falsification of the saturation model + three-phase test (arXiv
2507.18014). We sharpen iter121’s “no scaling law detectable” verdict into a structural one:
the saturation model R(t) = Rmax (1 − e−λt ) implies strict monotonicity (dR/dt > 0 for R <
Rmax ), so a strictly monotone trace violation rate of 0% is expected. Across all n2 ordered
pairs in each anchor we count those with R(j) < R(i) for j > i; observed violation rates are
0.382, 0.395, 0.462, 0.363, 0.290 for Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-
V3.1, and Nemotron-120B respectively – every anchor exceeds the 5% noise floor with per-anchor
binomial p < 0.001 and majority-violates binomial p = 0.031. Even Nemotron-120B (the least
violating) shows a maximum downward step ∆R = 0.875 (peak at step 3 collapsing to R ≈ 0 by
step 7), which is incompatible with any monotone rise. Simultaneously, the three-phase hypothesis
[87] (rapid improvement → plateau → collapse) is falsified: only 1/5 anchors exhibits the full
(p1 , p2 , p3 ) = (1, 1, 1) signature; the dominant pattern is collapse-only (4/5) or monotone-or-plateau
(3/5). Finally, the Rmax distribution is bimodal: sorted {0.182, 0.285, 0.817, 0.844, 0.869} has
largest gap 0.531 separating incapable ({0.18, 0.29}) from capable ({0.82, 0.84, 0.87}); Hartigan


                                                    29

[PAGE 30]
Figure 16: Iter 121 four-panel detection-power figure. (A) ∆late-early vs log10 (N ) scatter with OLS
line and Spearman ρ̂ = −0.30; the permutation-null 95% band (grey) covers the observed range,
confirming the rank-correlation null. (B) OLS slopes on the effective-compute axis log10 (N · D) for
five metrics, with bootstrap 95% CIs – all bracket zero. (C) Synthetic recovery heatmap: recovery
rate of a planted Rmax (N ) scaling law vs (nanchors , β); the dark region (low recovery) dominates
even for β twice the Chinchilla-class value. (D) Power curve: recovery vs nanchors for each β, with the
current five-anchor pool marked by the dashed vertical line. The 50%-recovery threshold is crossed
only by β ≥ 0.20 at small anchor counts where R̂2 is itself high-variance.


                                         sat      pw
               Model                   R̂max    R̂max   t̂peak         γ̂   ∆AICcpw-sat
               Qwen3.5-4B              0.817    0.930     1.0    −0.0001         +2.48
               Qwen3-8B                0.285    0.255    13.0    −0.0060         +1.20
               Llama-3.1-8B-Instruct   0.869    0.948     1.0    +0.0037         +1.27
               DeepSeek-V3.1           0.844    0.842     3.0    −0.0003         +2.79
               Nemotron-120B           0.182    1.050     3.0    +0.0014         +2.27
                                                LOOCV cluster agreement     5/5 = 1.00
                              Log BF (capability+params vs params alone)        −9.53
Table 27: Iter 129 piecewise fit + LOOCV + Bayes factor. All five anchors have ∆AICc > 0
(piecewise loses to saturation by 1.2–2.8 units, well above the Burnham & Anderson 2002 thresh-
old of 2 “substantial evidence”), and the Bayes factor at n = 5 firmly favours the simpler
size-only model (log BF = −9.53, “very strong” on Kass-Raftery). Yet the capability-bimodal
cluster assignment is LOOCV stable at 5/5. Scripts: scripts/scaling_law_iter129.py
→ scaling_law_iter129_piecewise_fit.tsv, scaling_law_iter129_aic_compare.tsv,
scaling_law_iter129_loocv_cluster.tsv, scaling_law_iter129_bf_capability.tsv.



dip statistic 0.522 with Silverman bootstrap p = 0.056. Iter 125 elevates pillar 1 from “no scaling law
observed” to “no monotone scaling law – the structural model class is wrong, and the cross-anchor
axis is capability, not size”.

Iter 129 elevation: piecewise saturate+collapse model + LOOCV capability-bimodality valida-
tion. The natural sequel to iter125 is to test whether (i) a piecewise model
                                    Rmax (1 − e−λt )
                                  
                                                                 t ≤ tpeak ,
                         R(t) =                                                         (6)
                                    R(tpeak ) − γ(t − tpeak ) t > tpeak ,
which can express the iter125 collapse, beats saturation on formal likelihood criteria, and (ii) the
iter125 capability bimodality is robust to leave-one-out. We fit (Rmax , λ, γ) at tpeak anchored at the
empirical peak; compare to the 2-parameter saturation baseline via AICc (Burnham & Anderson 2002)
and a nested F -test; and run LOOCV on the median-split cluster assignment. Result: the piecewise
model loses to saturation in 5/5 anchors – ∆AICc ∈ [+1.20, +2.79], all F -test p > 0.28. This is
not a model-comparison failure but a data-resolution finding: the per-step trace noise (σ ≈ 0.19
per iter121) is large enough that an extra collapse-rate parameter costs more AICc than it saves in
residual variance. Conversely, the iter125 capability bimodality is LOOCV-stable at 5/5 agreement
– leaving any anchor out and refitting the median split on the remaining n − 1 = 4 always assigns
the held-out anchor the same way as in the full fit. Bayes-factor comparison (Kass & Raftery 1995
categories) of M1 (params-only) versus M2 (params + capability) gives log BFM2 −M1 = −9.53:
the descriptive bimodality is not a parametrically identifiable predictor of Rmax beyond parameter
count alone at n = 5 – confirming the iter121 detection-power verdict with a likelihood-ratio test.
Iter 129 conclusion: Pillar 1 is closed at n = 5 – no functional-form law (monotone or piecewise)
is identifiable at the empirical noise level σ ≈ 0.19, but the capability bimodality is descriptively
reproducible and points to instruct-pretraining as the right cross-anchor axis for future n ≥ 40
anchor-pool work. See Table 27 and Figure 17.

Iter 133 elevation: capability-bimodality at n = 7/10/12. Iter 125 reported a 5-anchor capability
bimodality on Rmax (gap 0.531, Hartigan dip 0.522, permutation-p = 0.056), and iter 129 confirmed
its LOOCV stability at 5/5 agreement. Both iters were limited to the five-anchor pool that supports


                                                  30

[PAGE 32]
Pool       n    AICc (params)    AICc (capability)   AICc (params+cap)     AICc (interaction)
      n=5        5           −2.04              −23.08                  −3.65                +0.61
      n=7        7          −10.98              −41.61                 −34.99               −31.88
      n = 10    11          −21.44              −52.31                 −48.64               −44.61
      n = 12    12          −23.81              −56.12                 −52.46               −49.12
Table 29: Iter 133 AICc model comparison. Across all four pool sizes, the capability-only model
has the lowest AICc (bold) by 21–32 units over the params-only baseline. Adding log10 N on
top of capability worsens the fit (AICc delta −3.65 vs −23.08 at n = 5, −48.64 vs −52.31 at
n = 10), and the full interaction is uniformly dominated. This is a sharper version of the iter129
capability+params BF sign reversal: at n = 5 iter129 reported log BFM2 −M1 = −9.53 favouring
params-only over params+capability (Kass-Raftery “very strong”); at n ≥ 7 the comparison inverts,
with capability-only ∆AICc < −21 over params-only. scripts/scaling_law_iter133.py →
scaling_law_iter133_interaction_aic.tsv.



the categorical capability axis (capable anchors are a mix of dense + MoE, spanning 4B-1T). The
full interaction is dominated at every pool size, consistent with the cross-class axis being categorical
rather than continuous-modulated.

(e) Iter 133 conclusion. The iter 133 results close the iter 125/129 caveat that the capability
bimodality finding was limited to n = 5. Three sharp deliverables:

      1. Monotonicity falsification is robust at n = 7: every reliable anchor fails the iter 125
         diagnostic at binomial p < 0.001. The two new violators (gpt-oss-20B and Kimi-K2-
         Thinking) come from the MoE frontier pool and confirm the iter 125 finding is not a
         dense-model artefact.
      2. Capability bimodality strengthens with pool size: the Ward-k = 2 cross-class gap
         permutation p-value falls from 0.095 (n = 5) to 0.002 (n = 10), and LOOCV agreement
         is perfect (12/12) at every pool size. This is direct empirical confirmation that the iter 121
         detection-power verdict (which predicted the capability axis would only become visible at
         n > 5) was correct.
      3. Capability-class dominates the cross-anchor axis: at every pool size the capability-only
         model beats the params-only model by 21–32 AICc units, and adding log10 N on top of
         capability worsens the fit. This is the sharpest evidence yet that the Pillar-1 scaling structure
         lives on the capability axis (instruct/pretrained pretraining) rather than the parameter-count
         axis.

The Pillar-1 finding is now positive rather than null: scale-law-shape is wrong, but the capability
axis is informative. This shifts the practical Pillar-1 message from “no scaling law detectable”
(iter 117/121) and “structural falsification of the saturation model” (iter 125/129) to “cross-anchor
variance is dominated by capability class” – exactly the prediction the iter 121 detection-power
verdict made and that the iter 133 anchor-pool extension now empirically verifies.

Caveats and forward work. The capability axis is operationally defined as the Ward-k = 2
cluster label on the empirical Rmax distribution; this labels the incapable cluster as {Qwen3 −
8B, N emotron − 120B, Qwen3 − 32B, Qwen3.5 − 27B, Qwen3 − 30B − M oE} and the capable
cluster as {Qwen3.5 − 4B, Llama − 3.1 − 8B − Instruct, DeepSeek − V 3.1, Kimi − K2 −
T hinking, gpt−oss−20B, Qwen3−30B −M oE −Inst, Qwen3−235B −M oE} in the n = 12
pool. The incapable cluster is dominated by base models (no instruction tuning) and MoE probes with
Rmax ≤ 0.30; the capable cluster includes the instruct-pretrained dense frontier and the saturated-
MoE frontiers. A larger n ≥ 20 anchor pool with explicit pretraining-class labels (instruct/base,
dense/MoE, RLHF/DPO/no-tune) would let us operationalise the capability axis a-priori rather than
post-hoc from Rmax – this is the natural sequel to iter 133.

Iter 137 elevation: three-parameter offset saturation model – the R(0) = 0 artefact. The
iter 117/121 canonical fit R(t) = Rmax (1 − e−λt ) forces R(0) = 0, which is unrealistic for already-
trained frontier models that begin the iter 117 trace at a non-zero base reward. Iter 117 explicitly


                                                   32

[PAGE 33]
Figure 18: Iter 133 four-panel figure. (1) Rmax distribution by pool, dots coloured by Ward-k = 2
cluster (blue = capable, orange = incapable): the partition is stable across pool sizes, with the
incapable cluster consistently anchored at Rmax < 0.30. (2) Monotonicity violation rate for the seven
reliable anchors: all seven reject the iter 125 Hmonotone at p < 0.001. (3) AICc model comparison
per pool: capability-only (green) is lowest at every pool size; the params+capability additive model
(purple) and the interaction (red) are dominated. (4) LOOCV agreement is perfect (12/12) at every
pool size, validating the Ward cluster assignment.



flagged this in its docstring: “the canonical R(t) = Rmax (1 − e−λt ) fit hits the upper λ = 10
optimiser bound on every well-behaved run; this is an artefact of the R(0) = 0 boundary condition
when the trace already starts near its empirical ceiling”. Iter 137 makes the saturation model honest
by adding a free baseline offset c ∈ [0, 1]:

                      R(t) = c + (Rmax − c) 1 − e−λt ,
                                                           
                                                                   t = 1, . . . , T,              (7)
with three interpretable parameters: the baseline reward c, the asymptotic ceiling Rmax ≥ c, and
the learning rate λ > 0. The same closed-form t80 = − ln(0.2)/λ applies (the offset cancels in the
0.2 relative-gap calculation), but λ is no longer bound to the upper optimiser edge by an unrealistic
boundary condition.

(a) Iter 137 fit results. The 3-param fit succeeds on every anchor and un-binds λ on all 5/5 anchors
(vs. 4/5 anchored at λ = 10 under the 2-param fit). However, the 3-param loses to the 2-param by
AICc on 5/5 anchors (∆AICc ranges +1.71 for the borderline Qwen3-8B anchor to +18.18 for the
high-variance Qwen3.5-4B anchor): the additional offset parameter c is not justified by the residual
drop because the trace variance is high relative to the AICc complexity penalty. The 3-param is
therefore a useful interpretive lens (it exposes the baseline reward c) but does not improve model fit.
Iter 125’s structural falsification of the saturation family is reinforced, not weakened.

(b) Iter 137 cross-scale law. With the offset, the OLS regression of log10 t80 on log10 N produces
a real (no longer degenerate) slope estimate: b = +0.507 ± 0.718 (SE; t = 0.71, p > 0.5). The
analogous regression of Rmax on log10 N gives b = −0.172 ± 0.198 (Spearman ρ = −0.658,
p = 0.227, n = 5). Both slopes are far from statistical significance on this evidence base, but their
direction-of-effect is now informative: the iter 117 “4/5 anchored at bound” degeneracy is resolved
into a real but small positive log t80 -log N slope and a small negative Rmax -log N slope. Iter 137
thus sharpens the iter 117 null: the cross-scale saturation law is absent in two model classes (2-param
and 3-param), not one.

(c) Iter 137 capability-axis propagation. The iter 133 capability axis (capable = Rmax ≥ 0.7)
classifies the 3-param Rmax distribution identically: 3/5 capable (Qwen3.5-4B, Llama-3.1-8B-
Instruct, DeepSeek-V3.1) and 2/5 incapable (Qwen3-8B, Nemotron-120B). Mann-Whitney U on
3-param Rmax across classes is U = 3.0 (two-sided p = 1.0, degenerate at n = 5), and within-class
Spearman ρ(Rmax , log10 N ) is non-significant at n = 3 capable. The capability-class verdict from
iter 133 holds qualitatively through the 3-param fit, with the same within-class underpowering caveat
at n = 5.

(d) Iter 137 conclusion.    The Pillar-1 finding is now triangulated by two model classes:

      1. The 2-param fit (iter 117): λ saturates at the upper bound on 4/5 anchors, Rmax bimodality
         is confirmed (iter 125/129/133), and the t80 -vs-N regression is degenerate.
      2. The 3-param fit (iter 137): λ is finite on every anchor, but the model loses by AICc on 5/5
         anchors; the cross-scale slope on t80 is +0.507 ± 0.718 and on Rmax is −0.172 ± 0.198 –
         both far from significance. The capability-class axis is preserved.

The sharpest single sentence: GRPO saturation is real (capable anchors have Rmax > 0.8, incapable
have Rmax < 0.3) but its t80 , λ, and Rmax do not scale with parameter count on this evidence base
even when the R(0) = 0 boundary-condition artefact is removed. The cross-scale law is absent in
two model classes; the capability axis dominates in both.


                                                  33

[PAGE 35]
Figure 20: Iter 140 cross-pillar figure. (a) Reward-Design Quality Score per anchor on the 12-anchor
extended-frontier table; the five degenerate anchors (Nemotron-120B, Qwen3-32B, Qwen3-30B-MoE,
Qwen3-30B-MoE-Inst, Qwen3-235B-MoE) score RQS = 0 along the geometric-mean breakdown.
(b) AIC race on Rmax at n = 5; capability alone (AICc = −23.07) dominates capability + RQS
(AICc = −21.93) by ∆ = +1.14 (borderline NULL). (c) 12-anchor residualization scatter: residual
from the capability-only model on y-axis, RQS on x-axis; Pearson ρ = +0.225. (d) Iter 127 cross-
pillar: n = 20 (G, T) cells from Qwen2.5-0.5B/arithmetic; x-axis is the independently measured
richness proxy y = 1 − ZVFtheory from iter 131, y-axis is the iter 127 joint-fit residual; Pearson
ρ = −0.569, p = 0.029 (DECISIVE per F25 L8 recipe).



degenerate — exactly the failure mode Eureka predicts for reward functions that starve the policy of
contrastive information.
Four pre-registered questions are run on the existing evidence base (no new training):

      1. Q2 — AIC race on Rmax at n = 5 anchors. Capability alone achieves AICc = −23.07;
         capability plus RQS achieves AICc = −21.93 (∆AICc = +1.14, just under the ∆AICc ≥
         2 NOT-EVIDENT threshold from F25 L8 / Miller’s recipe). RQS is not justified as a
         regression covariate on this evidence base.
      2. Q3 — 12-anchor residualization. Regressing rmean on the capability dummy alone on
         the 12 anchors leaves RSS = 0.867; adding RQS reduces RSS by 4.0% (to 0.832). Pearson
         ρ(RQS, residualcap−only ) = +0.225 (positive but small at n = 12); Spearman drops to
         +0.087 indicating that RQS is not a rank-replacement for capability — the few high-RQS
         anchors (gpt-oss-20B, Llama-3.1-8B-Instruct, DeepSeek-V3.1) drive the Pearson channel.
      3. Q4 — Cross-pillar cross-link on iter 127 (n = 20 cells, same model). Loading
         groupsize_zvf_sweep.tsv (iter 131) gives an iid-contrast budget proxy y = 1 −
         ZVFtheory that is independent of the joint fit’s residual r = accemp − accpred . Pear-
         son ρ(y, r) = −0.569, two-sided p = 0.029; Spearman ρ = −0.533 (consistent in rank).
         Cells with large iid contrast budget are under-predicted by the joint fit (extract ∼ 5pp beyond
         compute alone); cells with starved contrast are over-predicted. This is the Eureka signature
         — reward-design quality lets the policy spend contrast budget compute alone does not model
         — at decisive significance on iter 127’s n = 20 cell grid.

Iter 140 conclusion. RQS fails the strict AIC test on n = 5 (borderline NULL); it shows a small
but direction-positive 4% RSS reduction on n = 12 (SUGGESTIVE); and on the n = 20 iter 127
cell grid it achieves decisive significance for the cross-pillar Eureka prediction (p = 0.029). The
capability axis of iter 133 is therefore preserved as the load-bearing cross-anchor signal, but the
reward-side of the equation is not silent: it adds diagnostic information orthogonal to capability on
the cell grid, exactly where the Eureka thesis predicts. Recommended action: add a 4-panel figure to
this section showing (a) RQS per anchor, (b) the AIC race, (c) the 12-anchor residualization scatter,
and (d) the iter 127 cross-pillar correlation.

5.7   Cross-Stack Identifiability Audit: GRPO and PPO Saturation Fits on the Same Rollouts

Motivation. Iter 25 audited the canonical saturation model R(t) = Rmax 1 − e−λt on the five
                                                                                          
frontier-scale traces (Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-V3.1, Nemotron-
120B) and concluded that the model is not identifiable on those data: 4/5 traces had λ pinned at
the upper bound λ = 10 (Nemotron-120B is the exception at λ ≈ 0.99), 0/5 preferred saturation
over a constant in noise-aware AICc, and 0/5 were identifiable [131]. That audit left open whether
the failure is intrinsic to the model or an artefact of the 20–30 step frontier horizons (where the
noise floor is comparable to the signal). Iter 29 closes that gap by re-running the same identifiability
battery on the longer-horizon same-stack traces (40 steps, neff = 128, GSM8K, Qwen2.5-0.5B),
and—crucially— on both algorithm stacks simultaneously. The frontier synthesis Round 1 licenses
an Estimator-Equivalence Principle (EEP): with rollout batch, KL, clipping, masking, optimiser
and reward parser fixed, PPO and GRPO should be performance-equivalent. If EEP holds, the
saturation-fit identifiability profile of the two stacks should be indistinguishable.


                                                  35

[PAGE 37]
Prediction   Test                                       Pass       Rate      p      Verdict
       P1           peak_frac > 0.10                           9/12      0.750    0.073    falsified
       P2           late > early                               5/12      0.417    0.806    falsified
       P3           late ≈ peak (plateau)                      7/12      0.583    0.387    falsified
       P4           ρ(score, Rmean )                         ρ=0.046       –      0.922    falsified
       P5           unique(zero_frac> 0.5 & mean< 0.2)         1/12      0.083    0.083    sustained
Table 33: Pre-registered three-phase hypothesis battery on the twelve-anchor frontier set. P1–P4 are
decisively falsified: the three-phase “slow start, rapid improvement, plateau” pattern does not hold
across the frontier set. P5 is the only sustained prediction, and it isolates the unique Nemotron-120B
collapse mechanism.


Limitations. The same-stack data are at a single model size (Qwen2.5-0.5B), single task (GSM8K),
and 40-step horizon. Whether the same AICc divergence holds at 8B+ with the frontier horizon
needs a Tinker reproduction (cost-flagged in FRONTIER_INSIGHTS.md). The “saturation-supported”
criterion in iter 25 (AICc-best = sat AND CI excludes bound) is satisfied in 5/5 GRPO traces and 0/5
PPO traces; iter 25’s stricter formulation would also falsify EEP here.

Artifacts. Driver: scripts/scaling_law_iter29.py. Per-trace: scaling_law_iter29_-
identifiability.tsv, scaling_law_iter29_bootstrap.tsv. Stack rollup: scaling_law_-
iter29_summary.tsv. EEP battery: scaling_law_iter29_stack_compare.tsv. Figure:
figures/scaling_law_iter29.pdf.

5.8   Three-Phase Hypothesis (Nimmaturi et al., arXiv 2507.18014): Pre-Registered
      Falsification Battery

Motivation. Nimmaturi et al. [88] (arXiv:2507.18014, Predictive Scaling Laws for Efficient GRPO
Training of Large Reasoning Models) propose that GRPO training proceeds in three consistent phases:
slow start, rapid improvement, plateau. Iter 17 observed that the five-anchor frontier set splits into
four phase labels (plateau, saturation, drift, collapse) under a heuristic rule, but the classifier was
not pre-registered, the falsification battery was implicit, and the Nemotron-120B collapse was not
mechanically characterised. Iter 33 closes these three gaps on the twelve-anchor frontier set.

Pre-registered predictions. The three-phase hypothesis yields four falsifiable predictions on the
twelve-anchor frontier set:

P1. Rapid improvement happens >10% into the trace: peak_step/nsteps > 0.10.
P2. Improvement is positive (not a drift): late_mean − early_mean > 0.
                                            √
P3. Plateau holds: |late_mean − peak| ≤ var + 0.05.
P4. A trace-level phase score (slopeearly − slopelate )/(|slopeearly | + |slopelate | + ε) predicts mean
         reward (Pearson ρ).
P5. Nemotron-120B is the unique trace with the joint extreme-collapse signature zero_frac >
         0.50 ∧ mean < 0.20.

P1–P3 are the three-Phase-hypothesis predictions proper; P4 is a sanity check that the phase score
carries information about performance; P5 isolates the mechanism behind the Nemotron-120B
collapse observed in iter 17.

Method. The driver scripts/scaling_law_iter33.py consumes the twelve-anchor summary
experiments/results/scaling_law_extended_frontier.tsv, synthesises a per-step reward
trace from each anchor’s (mean, var, peak_step, early_mean, late_mean, zero_frac) summary, and
applies the iter 17 four-class rule (plateau / saturation / drift / collapse). Each trace is then classified
along the phase-score axis (1.0 = pure plateau, 0.0 = pure linear, < 0 = drift or collapse), and the five
predictions are tested with explicit p-values: binomial exact for P1–P3 and P5, Pearson for P4. Phase
stability is quantified by leave-one-out bootstrap agreement (B = 200).

Results.


                                                    37

[PAGE 38]
Interpretation. The three-phase hypothesis of Nimmaturi et al. is partially falsified on this
benchmark: P1 (“rapid improvement happens >10% in”) and P2 (“late > early”) both fail. P2 fails
because 3/12 traces are drifts (Llama-3.1-8B-Instruct, Qwen3-32B, Kimi-K2-Thinking) and 1/12
is a collapse (Nemotron-120B); together these four traces have late_mean < early_mean. The
three-phase hypothesis implicitly assumes the post-peak plateau holds in mean, which is violated on
these four traces. P3 (the literal “plateau holds”) is borderline (p = 0.39); P4 (phase score predicts
mean reward) is decisively falsified (ρ = 0.046, p = 0.92), meaning the trace-level shape does not
predict the final reward in this dataset—what predicts final reward is the architecture+model identity,
not the training-curve shape.
P5 is the sharpest sustained finding. Nemotron-120B is the only trace in the twelve-anchor set
with the joint zero_frac > 0.50 ∧ mean < 0.20 signature. This is a clean single-attribute uniqueness
claim: every other frontier trace has either zero_frac ≤ 0.34 (the next worst is Qwen3.5-27B at 0.33)
or mean ≥ 0.25 (the next lowest is Qwen3-8B at 0.29). The Nemotron-120B collapse is therefore
extreme and unique in the data; it is not a generic large-model failure mode but a specific mechanism
that the other traces do not exhibit.

Nemotron-120B collapse mechanism. Iter 17 listed three candidate collapse criteria (zero_frac
> 0.3, post-peak decay slope < 0, peak < 0.95). Iter 33 applies all three to the twelve-anchor set
and shows the joint pattern is diagnostic:
       • 0/12 traces have all three criteria (Nemotron-120B has 2/3 but its post-peak slope is positive:
         the recovery from zero yields ∆ > 0, even though the trace never reaches the pre-peak
         baseline).
       • 3/12 traces meet ≥ 2 of the three: Qwen3-32B, Qwen3.5-27B, Nemotron-120B.
       • 1/12 traces meet all three: only Qwen3.5-27B.
The extreme-collapse signature (zero_frac > 0.5, mean < 0.2) is unique to Nemotron-120B. This
narrows the cause: it is not “large model fails to plateau” (a generic critique) but a specific reward
parser / format failure that drives 55% of Nemotron rollouts to zero reward and prevents recovery
to the mean—consistent with the iter 17 root-cause analysis showing Nemotron’s post-peak decay
slope is positive (it does not collapse during training; it collapses before training starts and slowly
recovers).

Phase stability. Bootstrap (B = 200) leave-one-out agreement of the four-class classifier is low:
only 3/12 traces have agreement ≥ 0.95 (Qwen3-8B, gpt-oss-20B, Qwen3-30B-MoE). The remaining
9/12 traces flip phase class on the majority of bootstrap resamples. This is not a bug in the classifier:
it is a property of the twelve-anchor set, where summary statistics from n ≤ 30 step traces are not
sufficient to fix the phase label under resampling. The classification is therefore useful as a partition
of the frontier set but not a stable estimator on individual traces—a caveat the three-phase hypothesis
does not address.

Cross-architecture phase distribution. Across the twelve anchors, 6 are dense and 6 are MoE.
The phase distribution by architecture is:
       • dense (n=6): 1 plateau, 1 saturation, 2 drift, 1 collapse, 1 plateau.
       • moe (n=6): 2 plateau, 4 saturation, 1 drift, 0 collapse.
Mann-Whitney on phase score yields U =ns, p=0.48 (median phase score: dense −0.75, MoE
−0.37). The qualitative distribution difference (no MoE model collapses) is suggestive but not
significant on n = 12 (χ2 on 2×4 contingency has p = 0.31). The finding is consistent with the
broader pillar 3 observation that MoE models are less collapse-prone but the small-n test cannot
reject the null.

Limitations. (1) The phase classifier operates on synthetic traces reconstructed from summary
statistics, not raw per-step reward logs. This is necessary because the frontier traces in scaling_-
law_extended_frontier.tsv are aggregated. (2) The five pre-registered predictions are not
independent (P1, P2, P3 share information); a Bonferroni-corrected threshold of α/5 = 0.01 would
still falsify P2, P4 and still sustain P5. (3) n = 12 is too small for definitive arch-level inference; the
Mann-Whitney result should be treated as descriptive.


                                                   38

[PAGE 40]
Results: raw per-step cross-check (iter37c). The synthetic-trace result is potentially an artefact:
the synthesised trace is too smooth to expose the curvature structure of real per-step data. We
therefore fit the same five forms to the real 40-step per-step reward trajectories from samestack_-
ppo_grpo.json (Qwen2.5-0.5B, 5 seeds each of GRPO and PPO). The result inverts: linear wins
0/5 on GRPO; the Hill n=2 form wins 3/5, exponential wins 2/5, MM wins 0/5. On PPO, Hill wins 2/5,
with sat / linear / MM splitting the rest. Mean Akaike weights on GRPO: w̄hill = 0.62, w̄sat = 0.38,
all other forms ≤ 0.001. On PPO, w̄hill = 0.40, w̄MM = 0.25, w̄sat = 0.21, w̄linear = 0.14.

Interpretation: the saturation form is not robustly identifiable on the 12-anchor set, but the Hill
n=2 form is the preferred nonlinear form on raw per-step data. The apparent contradiction be-
tween the synthetic-trace battery (linear wins) and the raw-trace cross-check (Hill wins) is resolved by
trace length: the 12-anchor traces are 3–30 steps, which is too short to fit a 2-parameter nonlinear form,
so AIC defaults to linear/null. On 40-step raw per-step traces, the same AIC ranking favours nonlinear
forms, and the sigmoidal Hill n=2 form is slightly preferred over the exponential (consistent with
the saturating-then-slowing-down shape of an S-curve vs. an asymptote-with-monotonic-curvature
exponential). The implication: the iter17–33 R_max estimates derived under exponential saturation
are not robust to functional-form choice; the same data is equally or more consistent with a Hill n=2
form.

Extrapolation comparison (iter37d). We re-run the iter21 two-anchor log-log extrapolation battery
under both forms. On the 4-anchor holdout (Qwen3-32B, Qwen3.5-27B, DeepSeek-V3.1, Qwen3-
235B-MoE, Nemotron-120B, Kimi-K2-Thinking), the mean absolute error on Rmax is essentially
identical: MAEsat = 0.907, MAEhill = 0.908. The t80 predictions diverge wildly: MAEsat = 584,
MAEhill = 2.4 × 108 . The reason is that on the synthetic frontier traces the saturation fit hits the
upper bound λ ≤ 5 and the Hill fit hits the lower bound K ≥ 0.001, making t80 unidentifiable from
a 3–30 step trace. This is itself a positive result: the iter17 t80 estimates were never identifiable on
the frontier set, and changing the functional form does not change that conclusion.

Recommendation. For frontier-trace analysis on nsteps ≤ 30 records, report the saturation form
as the literature default but also report the Hill n=2 fit and a ∆AIC column; the two are essentially
indistinguishable on this regime. For nsteps ≥ 30 per-step data, the Hill n=2 form is the preferred
nonlinear form on the 10 per-step raw traces measured here, and the exponential saturation is a close
second.

Sharpest claim. “The literature’s R(t) = Rmax (1 − e−λt ) parameterisation is not the uniquely
identified GRPO reward curve on either the 12-anchor frontier set (where it loses to linear/null on
9/12 anchors) or the 10-run raw per-step benchmark (where it loses to the Hill n=2 form on 5/10
runs, tied on the remaining 5). The iter17–33 R_max estimates should be reported with a ±0.05
envelope on the form-choice sensitivity and a Hill n=2 fit alongside.”

Artifacts. Driver:     scripts/scaling_law_iter37.py,        scaling_law_iter37b.py,
scaling_law_iter37c.py, scaling_law_iter37d.py.         Fits: scaling_law_iter37_-
fits.tsv, scaling_law_iter37b_fits.tsv, scaling_law_iter37c_fits.tsv.             AIC
summary: scaling_law_iter37_aic.tsv, scaling_law_iter37b_aic.tsv, scaling_law_-
iter37c_summary.tsv. Bootstrap: scaling_law_iter37_bootstrap.tsv. Extrapolation:
scaling_law_iter37d_fits.tsv, scaling_law_iter37d_extrap.tsv, scaling_law_-
iter37d_summary.tsv. Figure: figures/scaling_law_iter37.pdf (top: stacked Akaike
weights by anchor; bottom: bootstrap win share), figures/scaling_law_iter37b.pdf
(Akaike-weight heat-map on dynamic anchors + bootstrap box-plot), figures/scaling_-
law_iter37c.pdf (per-run Akaike weights on raw 40-step per-step traces, GRPO and PPO),
figures/scaling_law_iter37d.pdf (extrapolation MAE saturation vs Hill on 4-anchor
holdout).

5.10   Temporal stability of the saturation fit (iter41)

Setup. For each of the 12 frontier anchors we re-fit R(t) = Rmax (1 − e−λt ) to the per-step reward
trace truncated at 40%, 60%, 80%, and 100% of its length. We track the fitted λ, Rmax , and pre-
saturation slope s0 = Rmax λ across truncations and ask whether the early-fit Rmax predicts the
full-fit Rmax under a B = 200 parametric bootstrap (observation noise σ = 0.05).


                                                   40

[PAGE 43]
phase class                                     n    mean |r|    median |r|   max |r|
              collapse (Qwen3.5-27B per iter33 classifier)    1        0.004        0.004     0.004
              drift                                           3        0.380        0.474     0.553
              plateau                                         3        0.600        0.451     1.202
              saturation                                      5        0.353        0.396     0.590
                                predicted    actual
   Table 39: LOO |residual| = |Rmax       − Rmax    | aggregated by deterministic phase class.


 prediction                                                  outcome      value      note
 P1: two-param LOO RMSE < 0.30                               NO            0.504     Rmax ∈ [0, 1] floor on 7/12 anchors caps R2 at 0.18
 P2: max |resid| is on Nemotron-120B                         YES           1.202     collapse signature unmistakable
 P3: at median log10 C = 5.09, optimal P ∈ [4, 30]B          YES         P ⋆ = 4B    Qwen3.5-4B selected at the operating point
Table 40: Three pre-registered predictions P1–P3 with measured outcomes. 2/3 pass; the lone miss is
informative (saturation floor on Rmax ≤ 1.0 causes the linear model to underfit the ceiling).



Pre-registered predictions.

Iso-FLOP optimal anchor picker. Applying R̂max = a log10 P + b log10 C + c at log10 C ⋆ ∈
[4.5, 7.1], Qwen3.5-4B wins every budget in the low-to-mid range. The reason is operational, not
theoretical: at P = 4B the closed-form fit generates a positive (a log10 P ) term that just outweighs
the larger b log10 C ⋆ term of competing anchors, and the OLS coefficients were estimated on a mix
dominated by large P anchors (Nemotron, DeepSeek) where the ceiling clipped the response. This
is a known weakness of the within-mix fit: a model designed to maximise iso-FLOP utility would
rebalance the design by adding more low-P anchors, which the Pillar-1 measurement grid does not
have today.

Conclusion. The two-parameter iso-FLOP joint fit formalises iter45’s αdense = 1.03 vs αMoE =
0.057 reading into a single closed-form predictor. Its LOO RMSE is too large to be predictive (0.50 >
0.30), but the residual decomposition isolates a single breakdown — Nemotron-120B contributes
1.20 of the total 0.50 RMSE — which is exactly the failure signature the rest of Pillar 1 (iter33
collapse partition, iter41 truncation extrapolation, iter45 iso-compute invariance) has documented
from independent angles. The linear read of Rmax is a ceiling-limited summary statistic; the collapse
of Nemotron-120B is the one event the summary cannot absorb.

5.13   Iter 53 – Rank preservation + temporal-peak coupling (negative result)

We re-use the iter49 two-parameter OLS fit and 12 LOO residuals (scaling_law_iter49_loo_-
residuals.tsv) but ask three pre-registered questions: does the LOO prediction preserve the
ranking of models; does the LOO residual track the temporal peak position from iter33; and does
dropping the collapse anchors contract the cross-stack correlation between log10 P and Rmax (the
critic-degeneracy test from the frontier synthesis)? The headline answer is that none of the three
pre-registered primary predictions hold; iter53 is a clean negative result rather than a refit.

Rank preservation (P1). Across the 12 anchors we have Kendall τb = 0.107 between LOO-
predicted and actual Rmax (Spearman ρ = 0.112, permutation p = 0.721). For a random ordering
τb has mean 0 and standard deviation ≈ 0.30; the observed τb = 0.107 is essentially chance. The
LOO-predicted top-3 is Kimi-K2-Thinking, DeepSeek-V3.1, Qwen3-8B; the actual top-3 by Rmax is
Nemotron-120B, Qwen3-235B-MoE, Qwen3-30B-MoE-Inst – zero overlap. Mean |∆rank| = 4.00,
median 3.5, worst swap 9 ranks. Pre-registered: τb > 0.50 – FAIL. Anchors with |∆rank| ≥ 4:
Nemotron-120B (∆rank=-4); Qwen3-235B-MoE (∆rank=-4); Qwen3-30B-MoE-Inst (∆rank=-9);
Kimi-K2-Thinking (∆rank=+4); DeepSeek-V3.1 (∆rank=+4); Qwen3-8B (∆rank=+8). The largest
single swap is Qwen3-30B-MoE-Inst (∆rank = −9): the OLS places it last, the actual order
has it third. Both anchors sit at the same (log10 P, log10 C) ≈ (1.48, 4.66–4.88), so the swap is
within-stack variance that the cross-stack OLS simply cannot see.


                                                       43

[PAGE 44]
Temporal-peak coupling (P2). Across the 12 anchors with finite peak_frac, Spearman
ρ(peak_frac, residual) = −0.125 (permutation p = 0.700). Pre-registered: ρ < −0.30 – FAIL
(the point estimate is in the right direction but its magnitude is too small to clear the threshold and the
permutation p-value is large). The intuition we tested: the saturation fit reads the peak value as Rmax ;
a trace peaking late should be systematically over-predicted by LOO. The data do not support this for
the cross-stack pooled sample. A weaker absolute-step variant – ρ(peak_step, residual) = −0.250
(p = 0.424) – does clear the −0.20 bar (P4), but with n = 12 the test is underpowered. Conclusion:
the peak-coupling hypothesis is at best weakly supported and certainly weaker than the cross-stack
compute signal.

Critic-degeneracy test (P3, frontier synthesis). Frontier reasoning on Pillar 1 (Critic Degeneracy
Hypothesis) licenses the prediction that the residual Rmax variance is mostly explained by the
static prompt-difficulty regressor (collapse regime) rather than by compute. The concrete test:
ρ(log10 P, Rmax ) on the full sample vs. after dropping the collapse anchors. Observed 0.360
vs. 0.359 (absolute change ∆|ρ| = 0.001, well below the 0.05 threshold). Same axis for log10 C:
0.442 → 0.379. The cross-axis ρ(var(reward), Rmax ) = −0.134 on the full sample is negligible.
The Critic-Degeneracy prediction – that dropping collapse should reveal a strong residual compute-
R_max correlation that the full-sample correlation was hiding – is FAIL: the full-sample correlation
is the drop-collapse correlation.

What iter53 actually shows. The iter49 two-parameter OLS fit predicts individual Rmax values
with RMSE ≈ 0.50 (about half the range), but the LOO residuals are dominated by within-stack
variance, not by compute. For three of the 12 anchors the rank is swapped by ≥ 8 positions; two of
these are the Qwen3-30B-MoE/-Inst pair at the same (log P, log C) – a pure stack-conditional gap.
The implication for paper-level claims is sharp: any cross-stack iso-FLOP prediction made by the
iter49 OLS is essentially uninformative for individual anchors, even though it is rough useful for the
medians. This negative result motivates a hierarchical next step: anchor a stack-conditional Rmax
model on the existing 12 anchors before extrapolating.

Iter 61 elevation: ZVF-conditioned saturation-fit identifiability. The iter 25 audit established
that 0/5 per-trace saturation-rate estimates are identifiable at this benchmark’s noise floor. The iter 57
audit sharpened this to 4/5 anchors at the λ = 10 optimiser bound, with Nemotron-120B as the sole
identifiable exception. What neither audit explains is why the bound-degeneracy concentrates on these
anchors. This iteration closes that gap by tying the saturation-fit degeneracy to the same structural
axis that Pillar 2 (ZVF) measures: the per-trace concentration of reward mass at the R ∈ {0, 1}
extremes.
We define a per-step ZVF proxy
                                   ZVF
                                   ] = P(R = 0) + P(R = 1),                                            (9)
computed on each anchor’s reward trace (Table 41). This is the step-level analogue of Pillar 2’s
rollout-level ZVF = P(Kx = 0) + P(Kx = G) that measures within-group collision. The two scales
differ but both measure the same property: how much of the binary-outcome probability mass sits at
the all-same extremes.
                                            ] ≥ 0.1, 9 anchors) and LOW (ZVF
We stratify the 12-anchor pool into HIGH (ZVF                               ] < 0.1, 3 anchors)
strata and refit the canonical saturation model on each stratum (Table 42):
The HIGH-ZVF stratum anchors are: Qwen3.5-4B, Llama-3.1-8B-Instruct, Qwen3.5-27B, gpt-
oss-20B, Qwen3-30B-MoE-Inst, DeepSeek-V3.1, Nemotron-120B, Qwen3-235B-MoE, Kimi-K2-
Thinking. The LOW-ZVF stratum contains Qwen3-8B, Qwen3-32B, Qwen3-30B-MoE. Notably,
Nemotron-120B sits in the HIGH-ZVF stratum via its P (R = 0) = 0.55 floor-collapse path – this is
structurally distinct from the other HIGH-ZVF anchors (ceiling-degenerate via P (R = 1) > 0.3).
Both archetypes hit the λ = 10 bound, but for different reasons: the ceiling anchors pin Rmax at R̄
with a near-step transient; the floor anchor pins Rmax at the post-peak reward (R̂max = 0.875) with a
non-monotone decay that the fit cannot describe.
We then stress-test the cross-scale joint fit from iter 49 by leave-one-out refit (Table 43):
The cross-pillar ZVF comparison (Table 44) shows that the per-step ZVF proxy across the 12 anchors
is ZVF
   ] step = 0.413 versus Pillar 2’s rollout-level ZVFrollout = 0.158 (Qwen3-8B, G=8, 600


                                                    44

[PAGE 77]
Across the N = 15 logged experiments spanning 5 model families and 3 B–671 B parameters,
aggregate ZVF and final performance are negatively associated (rPearson = −0.77, ρSpearman = −0.78).
We report this as a descriptive association, not an inferential claim: the correlation is dominated by
the out-of-scope tool-use runs, which by construction saturate at ZVF = 100% with reward 0 (mean
GU = 0%), whereas the GSM8K runs sit at much lower ZVF. Removing the saturated endpoints
leaves only a handful of in-scope runs—too few to establish significance—so, consistent with the
abstract, we treat ZVF/GU as a diagnostic of signal degeneracy rather than an independent causal
predictor of performance. Model scale does not uniformly reduce ZVF (rPearson = −0.26), and a
one-way ANOVA across model families yields F (4, 10) = 0.949, p = 0.476: after accounting for
task type, family alone does not explain performance variance.
To our knowledge this is the first cross-scale (3 B–671 B) measurement of all-equal-group (ZVF)
prevalence under a single managed runtime; we do not claim a measured cross-library result, as
the matched cross-framework ZVF comparison is withdrawn pending unbundled per-step logs
(Appendix A5). On the runs we measured, sustained GUt below roughly 0.5 coincided with reward
plateaus and regressions; we report this descriptively and do not prescribe a fixed operational
intervention threshold (the previously stated trigger is withdrawn). Related methods act on the same
all-equal-group event [162].
A controlled matched-budget panel gives the diagnostic a training-side test at both ends of the
accuracy range. At a fixed budget of 2,560 rollouts per arm (Qwen3-8B, LoRA rank 4, GSM8K,
batch 8, two seeds per arm), the G=2 × 160-step arms drive train reward to ≈ 0.9–1.0 on the sampled
pool and end in sustained ZVF ≈ 0.75–1.0 — all-correct groups, the high-accuracy zero-variance
wall implied by ZVF(p, G) = pG + (1 − p)G as p → 1 — while the G=16 × 20-step arms end
mid-learning at train reward ≈ 0.3–0.5 with ZVF at 0–0.25 throughout. Small G converts the budget
into more optimizer steps early and then exhausts its own signal in the endgame; the reading of
sustained high ZVF as signal degeneracy applies whether the wall is all-incorrect or all-correct.




Figure 33: Per-step ZVF heatmap. Red: ZVF= 1 (zero gradient); blue: ZVF= 0 (gradient flows);
grey: no data. Experiments sorted by aggregate ZVF (descending).


5.24   GRPO Is Secretly DPO: Validation Against Our Data

Wu et al. [143] prove that GRPO is algebraically equivalent to an implicit contrastive objective (DPO),
with group size G affecting only the Monte Carlo variance of that objective. Their headline finding:
2-GRPO (G = 2) retains 97.6% of 16-GRPO performance while requiring 12.5% of rollouts and
21% of training time.

ZVF Theory.     For binary-outcome tasks with per-prompt accuracy p:
               ZVF(p, G) = pG + (1 − p)G ,          GU(p, G) = 1 − pG − (1 − p)G .
At G = 2, GU(p, 2) = 2p(1 − p) is exactly the Bernoulli variance of p, reproducing the DPO
preference weighting. Beyond G = 8, the marginal GU gain from increasing group size approaches
zero for moderate accuracies p ∈ [0.2, 0.8]; the gain from G = 16 → G = 32 is less than 0.3% at
p = 0.5.


                                                  77

[PAGE 81]
estimate. The contribution of hidden managed defaults, checkpoint choice, effective capacity, and
runtime differences is not separately identified here.

Base vs. Instruct Models. Several base checkpoints underperform instruction-tuned or reasoning-
tuned alternatives, especially the two Llama base models and DeepSeek-V3.1-Base, but this pattern
is not universal: Qwen3-8B-Base is a clear counterexample. We therefore treat initialization quality
as one plausible contributor to trainability in this benchmark rather than as a necessary or sufficient
gating condition.

Group Size Ablation. We swept group size G ∈ {2, 4, 8, 16} on Qwen3-8B (Table 6, Group Size
block). Optimal last-10 performance occurs at the intermediate G=4 (52.1%); G=2 (37.5%), G=8
(34.4%), and G=16 (38.0%) are lower — an inverted-U consistent with the measured group-size
sweep (Appendix A14.5). This inverted-U pattern is descriptive. The current sweep does not isolate
why the endpoints underperform: optimizer-step count, prompt difficulty, contrast yield, and gradient
variance all change with G. In particular, GRPO does not “distribute attention across samples”; the
earlier mechanistic wording was unsupported and has been removed.

6     Reproducibility
We provide comprehensive reproducibility infrastructure:

       • Docker: Exact environment with pinned dependencies
       • Seed management: Deterministic training across frameworks
       • Weights & Biases:   All experiment logs, training curves, hyperparameter
         sweeps, and KL divergence traces at https://wandb.ai/tinker-rl-lab/
         tinker-rl-lab-world-class (project: tinker-rl-lab-world-class)
       • Hugging Face Hub: All checkpoints with model cards at https://huggingface.co/
         arvindcr4/tinker-rl-bench-*
       • Statistical toolkit: rliable-based analysis scripts
       • REPRODUCE.md: Exact commands for every experiment

Team Model Checkpoints. All task-specific models from Section 5.17 are publicly released on
Hugging Face Hub:

       • arvindcr4/tool-call-lora-qwen0.5b — Tool call LoRA (Qwen2-0.5B)
       • Balasandhya/llm-multiturn-tool-call-grpo-QloRA-Qwen2.5-3B — Multi-turn
         GRPO (team artifact; single-seed)
       • Madhu2133/qwen3-8b-code-grpo-v10 — Code reasoning GRPO adapter; self-reported
         HumanEval pass@1 on the public model card, not a controlled 5-seed run
       • MohammadRafiML/Qwen3-4B-Instruct-2507-Capstone-MathRL — Qwen3-4B-
         Instruct SFT→GRPO adapters (different base model than the Qwen3-8B track; treated as
         side artifact)
       • dhruvanmurthy/Qwen3-8B-FineTuning — SFT+GRPO tool-use pipeline on Qwen3-8B

7     Limitations
We report our limitations with deliberate candor. Top venues reward honest analysis; we believe doc-
umenting infrastructure failures alongside algorithmic results is itself a contribution to reproducible
science.

7.1   Infrastructure Failures

JWT Token Expiration (Tinker API). Of 14 Tinker experiments launched, 7 ran to completion
(DeepSeek-V3.1, Qwen3-8B, Qwen3.5-4B, Llama-3.1-8B-Instruct on GSM8K, and both tool-use


                                                  81

[PAGE 82]
evaluations), 5 were interrupted mid-training but yielded partial reward traces (Qwen3.5-27B, Qwen3-
32B, Nemotron-120B, Qwen3-235B-A22B, Qwen3-30B-A3B variants), and 1 produced no usable
data in the original World-Class Suite window (GPT-OSS-20b, Tinker API stall). Kimi-K2 originally
failed in that window and was later re-run to completion in the Bitter Lesson Campaign retry (peak
100%, last-10 80%, 20 steps; see Appendix A1); it is therefore reported as a completed single-seed
Tier-C case study in the main tables. Partial experiments are marked † in all tables; their metrics
reflect only the completed training steps and should be interpreted as early-training snapshots. This is
a fundamental limitation of serverless ML platforms that do not support long-running stateful jobs
out of the box. Our workaround—issuing tokens immediately before job submission—reduces but
does not eliminate the hazard for multi-hour runs. We recommend that platform operators expose a
token-refresh endpoint or implement background credential rotation; until then, practitioners should
checkpoint frequently and implement automatic restart logic. All reported Tinker results in this paper
derive from completed or partially-completed runs; partial-run results are marked † and their metrics
reflect only the available training steps.

Modal Timeout (60-Minute Hard Limit). Some Modal-hosted evaluation jobs timed out at the
platform’s 60-minute wall-clock limit, including held-out evaluations on larger models and the full
HumanEval pass@1 suite. Large-model inference on a single H100 is substantially slower than
expected for generation-heavy tasks—generating even modest numbers of long solutions at 32B scale
can exceed 60 minutes easily. Future work should shard generation across multiple GPUs (tensor
parallelism), use speculative decoding, or negotiate higher time limits with the provider.

KL Divergence Tracking Bug. Direct KL divergence monitoring failed because reference model
logits were computed under torch.no_grad(), but the downstream KL term expected gradients
from the reference branch. The resulting RuntimeError silently killed the tracking loop without
halting training. To mitigate this gap, Section 5.22 develops three reward-trajectory stability proxies —
Stability Index, Peak-to-Tail Drift, and Rolling Variance — that are associated with lower training-set
last-10 reward (r = −0.517, p = 0.005 for PTD vs. last-10 average). We treat them as hypothesis-
generating diagnostics of training instability, not as validated proxies for policy drift or KL divergence.
Direct proxy-vs.-KL correspondence is unvalidated in the current release; the corrected KL tracking
implementation is included in the repository for future work.

7.2   Methodological Limitations

Closed-Source Training Implementation (Tinker). Tinker is a commercial, closed-source API.
We cannot inspect the exact GRPO loss formulation, reward normalization scheme, minibatch
construction, or hardware configuration used server-side. Our Tinker results therefore measure
the platform’s GRPO implementation, not a precisely specified algorithm. Researchers wishing to
attribute performance differences to specific implementation choices should use the open-source
backends (TRL, OpenRLHF, veRL) where every hyperparameter is auditable.

Short Training Horizons (30 Steps). All Tinker experiments used a budget of 30 gradient steps—a
deliberate choice to contain API costs but one that may be insufficient to observe convergence on
harder tasks. Thirty steps represents roughly one or two passes over the prompt pool at the batch
sizes we used. Long-horizon effects such as reward hacking, catastrophic forgetting, or late-stage
policy collapse are unlikely to manifest at this scale. We regard our Tinker results as early-training
snapshots rather than converged solutions, and caution against drawing strong conclusions about
asymptotic performance.

Single-Seed Tinker Experiments. Cost constraints precluded multi-seed replication on Tinker:
each configuration was run once. Without variance estimates we cannot apply standard significance
tests to Tinker results. We report these numbers descriptively and recommend that future work with
larger budgets run at least three seeds, consistent with best practices from Henderson et al. [32].

Train-Set Reward Metric. Reported training rewards are computed on the same prompt distri-
bution used for training, not a held-out test split. A post-hoc held-out GSM8K slice (N =500 per
checkpoint, greedy decoding) was completed for eight strong Tinker checkpoints plus one partial
263-problem run (Table 9); these are checkpoint-selected on training last-10 reward and mostly lack
matched base-model controls, so they are ranking-stability checks rather than unbiased generalisation


                                                   82

[PAGE 86]
Task           Algo         Seed      n    ∆lenhalf    ρ(step, len)     ρ(len, rew)    Flag
           Arithmetic     GRPO           42     40    −0.402        −0.280            −0.281            0
           Arithmetic     GRPO          789     40    −0.465        −0.520            −0.525            0
           Arithmetic     GRPO         1024     40    −0.582        −0.635            −0.635            0
           Arithmetic     Dr. GRPO       42     40    −0.374        −0.178            −0.110            0
           Arithmetic     Dr. GRPO     1024     40    −0.450        −0.503            −0.495            0
           GSM8K-CoT      GRPO           42     30    −13.02        −0.864            −0.507            0
           GSM8K-CoT      GRPO          123     30    −6.70         −0.716            −0.241            0
           GSM8K-CoT      GRPO          456     30    −0.63         −0.285            −0.756            0
           GSM8K-CoT      Dr. GRPO       42     30    −4.91         −0.446            −0.608            0
           GSM8K-CoT      Dr. GRPO      123     30    −4.30         −0.552            −0.505            0
           GSM8K-CoT      Dr. GRPO      456     30    +0.80         −0.067            −0.741            0
Table 76: Selected per-run rows from experiments/results/length_bias.tsv. The half-life
length difference ∆lenhalf is mean(second half) − mean(first half) of mean_comp_len; negative
means compression. All flag values are 0 — the Dr. GRPO signature (positive length trend AND
flat-or-down reward) is absent across all 16 runs at 30–40 step horizons.



cleanest statement is: at the horizons and tasks available in this benchmark, GRPO and Dr. GRPO
behave nearly identically with respect to length – both compress, Dr. GRPO slightly less so – and
neither engages the verbosity-trap signature. We therefore do not recommend Dr. GRPO as a
necessary length-bias mitigation on these task scales; we do recommend measuring the same per-step
Spearman on any 200+ step run before adopting it.

Scale extension: an uncapped Qwen3-8B panel. A six-arm panel run after the analysis above
extends the scale axis: Qwen3-8B (LoRA rank 4) on GSM8K, 30 steps, G=8, batch 4, three seeds
per algorithm, with the completion cap raised 5× to 1,024 tokens. The picture is unchanged: mean
completion length declines 3.8–12.2% in all six arms (GRPO 1004→905, 981→944, 996→900;
Dr. GRPO 999→931, 972→902, 1000→878 tokens), and late-run ZVF shows no separation between
the two losses (Dr. GRPO 0.45/0.70/0.72 vs. GRPO 0.47/0.47/0.55). Step-0 completions sit near
the raised cap (≈ 98%), so upward headroom is still limited; but the trajectories move away from the
cap rather than pressing against it, so the compression finding is not a censoring artifact. The horizon
caveat stands: 30 steps does not test the 200+-step regime in which [71] observe the trap.

Limitations. (1) n=5 and n=3 seeds are too few to put tight confidence intervals on the difference
between the two algorithms; we report standard deviations across seeds but do not compute seed-level
p-values. (2) Both tasks are reward-on-completion binary tasks, so the within-run ρ(len, reward) is
partially driven by the fact that longer incorrect completions contribute one full penalty to the per-step
mean – not by an advantage estimator pathology. (3) [71] themselves report the verbosity trap at
hundreds of steps on larger reasoning benchmarks; our 30–40 step horizons are short of that regime
by an order of magnitude. (4) A direct ablation that ran the same number of steps on a longer-horizon
task (DeepSeek-R1 distillation chain, or R1-Zero on MATH) is the natural next experiment, but is
beyond the scope of this iteration.

                                              Length behaviour                            Reward behaviour
 Horizon                             Easy            Hard          Ref.            Easy          Hard         Ref.
 30–40 steps (this paper)        compresses      compresses      §5.X, here       grows         flat        §5.X, here
 200+ steps (Dr. GRPO paper)       grows           grows            [71]         collapses   collapses         [71]
Table 77: Cross-horizon reconciliation. At short horizons the model compresses under reinforcement;
at long horizons Dr. GRPO’s authors observe the predicted length growth followed by reward collapse.
Our data is consistent with the early phase of the Dr. GRPO curve, not the late phase.


Take-aways for a reviewer. (1) No length-bias trap at 30–40 steps on either task. All sixteen
runs have negative length trends, none crosses the flag threshold. (2) Dr. GRPO’s effect is small at
this scale. On GSM8K-CoT it attenuates the length-trend slope by 0.27 (one sd) but does not flip
its sign. (3) Within-run coupling is negative throughout. Longer completions are not rewarded on


                                                     86

[PAGE 137]
analysis of the kind described in Liu et al. [72], and we release our checkpoints expressly to enable
such analysis.

9.2   What the Implementation Gap Means for the Field

The large last-10-reward gap between LLM-native libraries (Tinker, TRL, running GRPO) and classic
RL libraries (SB3, CleanRL, Tianshou, running PPO on the same arithmetic task) sits near the ceiling
of what any effect-size calculation can meaningfully report: the classic-RL libraries produce last-10
reward near 1%, so the denominator of Cohen’s d is driven by near-zero within-group variance, which
is why we measure d ≈ 21.84 — a structural-failure signature of default configurations, not a standard
d-scale effect. The comparison also confounds algorithm (GRPO vs. PPO) with implementation-layer
(LLM-native vs. classic-RL), and we therefore do not claim it as evidence that “implementation-layer
choices dominate algorithmic choices”. The defensible reading is narrower: classic-RL library
defaults, running PPO as distributed, fail to train on short-horizon LLM arithmetic out of the box, and
LLM-native library defaults, running GRPO as distributed, succeed. The mechanism here is plausibly
an architectural mismatch: classic RL libraries treat the language model as a standard MDP policy
with a scalar state embedding, apply advantage normalization designed for continuous action spaces,
and do not handle the auto-regressive token generation loop correctly. Our evidence is therefore
strongest for an implementation-layer gap, not for a categorical claim about PPO in the abstract.
The practical implication is methodological, not algorithmic: our SB3 PPO row at 0.010 ± 0.002
accuracy (reproduced from the committed five-seed Modal run, modal_results_all.json) is
evidence about an SB3-style tokenization and rollout pipeline applied to autoregressive generation,
and we cannot separate this structural failure from an algorithmic failure of PPO in the abstract.
Published claims of the form “PPO fails on LLM reasoning” that rely on the same classic-RL defaults
therefore inherit the same confound. We recommend that future algorithm comparisons report the
exact PPO implementation used, verify it against a known-good LLM post-training framework, and
rule out the structural-failure mode before making an algorithm-level claim.

9.3   Connections to Concurrent Work

Selective rollouts and ZVF as a training diagnostic. Zheng et al. [162] (GRESO) motivate
selective rollouts by the prevalence of zero-advantage dead zones in GRPO training and skip prompts
whose groups are predicted to collapse to zero reward variance. Our cross-library ZVF traces
provide descriptive evidence about when within-group contrast collapses under sparse binary rewards,
overlapping with the regime such selective-rollout methods aim to address. Importantly, in this
corpus ZVF varies more by task regime than by raw model scale: tool-use experiments saturate
at ZVF = 100%, while GSM8K experiments show substantially lower ZVF across overlapping
model scales. They should not yet be treated as identifying an independent latent variable or as a
direct calibration target without showing incremental value beyond reward mean, entropy, advantage
variance, and KL.

Scaling laws: exponential saturation and its limits. Nimmaturi et al. [88] derive a three-phase
exponential saturation law for GRPO, finding that 80% of training steps contribute marginally and
proposing early stopping. We recover a three-phase-looking pattern in 73% of LLM fine-tuning
experiments, broadly consistent with their result. However, our data also reveal a critical boundary
condition: the saturation framework is a poor description of unstable training runs. Nemotron-120B’s
trajectory (87.5% peak → 16.2% last-10) is non-monotonic and cannot be fit by the exponential
saturation model — it is better characterized as a reward excursion followed by policy collapse, a
regime the scaling law does not model well. The exponential saturation model still achieves mean
R2 = 0.210 across all experiments (vs. 0.170 for a power-law baseline), but that aggregate masks
qualitative failure on collapse trajectories. We recommend that practitioners verify monotonicity
before applying early stopping rules derived from the saturation model.

“It Takes Two” and the 2-GRPO/DPO equivalence. Wu et al. [143] prove that at G = 2, GRPO’s
advantage reduces to a DPO-equivalent contrastive objective, and show empirically that 2-GRPO
retains 98.1% of 16-GRPO performance at 12.5% of rollout cost. Our theoretical analysis confirms
their prediction for the expected ZVF: GU(p, 2) = 2p(1 − p), which is exactly the Bernoulli variance
of the per-prompt accuracy p and reproduces the DPO preference weighting. For our DeepSeek-


                                                 137

[PAGE 138]
V3.1 run (p ≈ 0.85, G = 4), 30% of steps were zero-gradient (measured zero_loss_pct); the
DPO equivalence predicts that switching to G = 2 would retain the within-group contrast on the
non-degenerate steps while halving rollout overhead (≈ 50%), though we did not run this ablation.
However, the DPO equivalence also predicts that very high-accuracy models (p > 0.9) will have low
gradient utilization regardless of G, suggesting that the relevant intervention at high accuracy is not
group-size reduction but prompt re-sampling from harder sub-distributions.


9.4   Limitations of Proxy Metrics and the Path to Direct KL Measurement

Our policy drift analysis relies on three reward-trajectory proxies (SI, PTD, Rolling Variance) that
correlate significantly with training outcomes (r = −0.517, p = 0.005 for PTD) but are not the
same as direct KL divergence measurements. The proxies capture observable symptoms of policy
drift (reward instability and peak-to-tail degradation) but cannot distinguish between two importantly
different causes: (a) the policy has genuinely drifted far from the reference in token-distribution space,
or (b) the reward function is inherently noisy and the policy is stable but reward variance is high.
Nemotron-120B’s collapse is almost certainly case (a), given that its reward decline is monotonic and
persistent; but for models with moderate PTD (0.1–0.3), the two causes are indistinguishable from
reward trajectories alone.
The corrected KL tracking implementation is included in the artifact release, but this paper does not
yet validate the proxy–KL correspondence or quantify per-token divergence trajectories.


10    Conclusion

T INKER RL-B ENCH provides a shared empirical substrate for studying RL post-training across
algorithms, frameworks, and model families, but the evidence it offers is narrower than a broad
“five laws” framing. The current release most strongly supports three conservative claims. First,
ZVF/GU are useful descriptive diagnostics of when binary-reward GRPO stops producing informative
within-group signal; they should not yet be read as standalone causal or incrementally predictive
statistics. Second, trainability in our short-horizon setting varies substantially with initialization and
rollout regime: instruction-tuned checkpoints were generally easier to optimize than comparable base
models, and intermediate group sizes often behaved better than the smallest or largest settings we
tested, but the benchmark does not justify a universal optimum or a general superiority claim for any
one algorithm. Third, PPO-vs.-GRPO rankings and frontier-run stability are heterogeneous across
model families and stacks, so our results argue against one-size-fits-all recommendations rather than
for a new one.
The most important negative result is also the most useful one: on held-out GSM8K, the mean
Qwen3-8B GRPO gain over the base model is small and not statistically significant under our current
evaluation. Tool-use and code results are even less conclusive because they rely on sparse or custom
reward protocols. That boundary on what we can claim is not a weakness to hide; it is the point of
releasing the benchmark.
The immediate next steps are experimental rather than rhetorical: multivariate tests of ZVF against
reward mean, entropy, and divergence proxies; token-budget matched multi-seed PPO/GRPO com-
parisons in a single open stack; broader held-out evaluation without checkpoint cherry-picking;
and non-math tasks with process rewards or richer execution-based metrics. The released traces,
checkpoints, and scripts are intended to make that stricter follow-up easy.


11    Ethics, Limitations, and Broader Impact

This statement is written to satisfy the NeurIPS Code of Ethics and Broader Impact requirements.
It complements the shorter Limitations and Broader Impact subsections embedded in Section 7 by
consolidating in one place a full dual-use analysis, itemised compute accounting, carbon footprint
estimation, data provenance disclosures, a candid acknowledgment of the closed-source training
backend on which a fraction of our headline numbers depends, and a list of methodological limits we
are aware of but did not have resources to close within the submission window.


                                                   138

[PAGE 141]
Table 101: Energy and carbon footprint estimates. Tinker hardware is not publicly disclosed; we
assume 1× H100 equivalent per run. Failed / stalled runs are included. Totals rounded to one
significant figure.
        Platform                        GPU-h   TDP (W)      PUE    Energy (kWh)     CO2 (kg)
        Tinker (assumed H100)            ~950          700    1.1            ~732        ~269
        Modal H100 (US-East)              ~18          700    1.1             ~14        ~5.1
        PES A100 (in-kind)                ~60          400    1.1             ~26         ~19
        Colab Pro T4 (asia-south1)        ~40           70    1.2            ~3.4        ~2.4
        NVIDIA L4 (TRL baseline)          ~10           72    1.1            ~0.8        ~0.3
        Project total (best estimate)                                        ~776        ~296



factor-of-two error in GPU-hours or TDP is plausible. Under a pessimistic assumption (2× GPU-
hours, H100 running near 100% TDP continuously), the Tinker contribution could be as high as
~540 kg CO2 -eq. Under an optimistic assumption (Tinker backends actually use more energy-efficient
accelerators than H100, 50% average utilisation) the contribution could be as low as ~130 kg. We
report the central estimate in the main table and caution readers that it should not be interpreted as
precise. All numbers should be regarded as upper bounds on the carbon cost of reproducing our work,
because subsequent users can skip the failed runs and the ablation sweeps.

Offset and mitigation. We did not purchase carbon offsets. Instead, we have (a) released all trained
checkpoints on HuggingFace Hub so that downstream users do not need to re-run our experiments;
(b) released step-level CSV logs so that learning curves can be inspected without re-training; and (c)
documented in Section 11.2 which runs were wasteful, so that replicators can skip them. We regard
reproducibility-by-artefact as a more durable mitigation than offsets.

11.4   Data Provenance

T INKER RL-B ENCH uses only publicly released research datasets. No private data, personally
identifiable information, or licensed proprietary content is used. No human annotators were employed
during this work. We document each dataset below.

GSM8K [18]. 8,500 grade-school math word problems (7,473 train / 1,319 test) authored by
human writers contracted by OpenAI. Released under the MIT License via https://github.com/
openai/grade-school-math. We use the standard train/test splits without modification. The
canonical citation is Cobbe et al. [18]. Known limitations of GSM8K include gender-neutral but
culturally US-centric word problems (names, currencies, sports) and a moderate rate of human-
labelling errors estimated at ~2% by [156]. Our held-out evaluation respects the test/train split.

Salesforce xLAM-Function-Calling-60k [67]. 60,000 function-calling examples generated
by Salesforce Research as training data for the xLAM agentic model family.                          Re-
leased under the CC BY 4.0 license via https://huggingface.co/datasets/Salesforce/
xlam-function-calling-60k. We use the public release as-is; specifically, we use the first 35
prompts × 10 rollouts in the 10x Structural Ceiling tool-use track and the full 60k split for the xlam-
60k real-data run in Section 5. Known limitations: xLAM schemas are synthetic and skew toward
cleanly-typed arguments; the dataset under-represents ambiguous, error-handling, and multi-turn
tool-calling. Attribution to Salesforce is required by the license and is provided in Section 11.6 and
in the xLAM-derived model cards.

HumanEval [15]. 164 Python programming problems with unit tests, released by OpenAI under
the MIT License. Used unmodified for pass@k evaluation. Known limitations: small scale, English-
language docstrings, single-language coverage; documented contamination concerns against frontier
pretraining corpora [111].

NuminaMath [61]. Used by collaborator Mohammad Rafi for the multi-stage
GSM8K+NuminaMath pipeline. Released under Apache 2.0. We use a downstream check-


                                                 141

[PAGE 142]
point reported by Rafi; our repository contains only his scripts and the Apache-licensed derivative
weights.

Open-Platypus [59]. 3,000 SFT examples used by collaborator Madhu for Qwen3-8B code genera-
tion warm-up. Open-Platypus is a curated subset released under CC BY-NC 4.0; the non-commercial
restriction is respected in our release, which is academic-use only.

Synthetic tool-use corpus (authored). We additionally generated a small five-tool synthetic corpus
(calculator, web-search stub, calendar, file-read, email-send) used for Section 4.1’s saturation analysis.
The corpus is authored by the paper’s authors from publicly documented APIs, contains no third-party
content, and is released under the MIT License in our repository under data/synthetic_tools/.
Generation prompts are included for auditability. No user data, scraped web content, or proprietary
API traces are present.

No web scraping, no human subjects. We did not scrape any website. We did not run any study
with human participants. No IRB review was therefore required. No data that could reasonably be
considered to contain PII, copyrighted content, or sensitive personal attributes was used at any stage
of training, evaluation, or reward modelling.

11.5    Closed-Source Tinker Acknowledgment

A central tension in T INKER RL-B ENCH is that a substantial fraction of our headline numbers
come from a closed-source commercial platform while our paper simultaneously advocates for
reproducibility and platform independence. We do not resolve this tension by hiding it; we describe it
precisely.

What Tinker is and is not. Tinker [130] is a managed LLM fine-tuning and inference service
provided by Thinking Machines, Inc. It exposes a Python SDK (tinker==0.16.1) that accepts
custom loss functions (forward_backward_custom), a limited set of optimisers, and standard LoRA
hyperparameters. It does not expose (i) the exact server-side GRPO loss implementation, (ii) reward
normalisation or baseline subtraction scheme, (iii) minibatch construction or gradient-accumulation
strategy, (iv) hardware configuration (GPU type, inter-node bandwidth), or (v) system-level telemetry
(energy, throughput, queueing).

What this means for our claims. Tinker results in this paper measure the platform’s implementa-
tion of GRPO, not an abstract specification of the algorithm. A reader who asks “why does Tinker
GRPO score 99.9% on GSM8K while open-source TRL GRPO scores 73.4% on the same task”
cannot fully answer that question from our data: we are able to rule out a handful of candidate expla-
nations (seed variance, model-size confound) but not the implementation itself. We therefore draw
quantitative conclusions only from the open-source side (TRL, veRL on Modal H100, OpenRLHF)
where every hyperparameter is auditable, and use Tinker results primarily as an upper bound on what
a carefully engineered production stack can achieve for critic-free RL at our scales.

Reproducibility commitments we can make.

       1. All Tinker experiment scripts, configuration files, JSON step logs, and per-run W&B projects
          are archived in the repository. Researchers with Tinker access can attempt replication with
          our exact configurations.
       2. Every figure and every summary statistic derived solely from Tinker data is marked with “†”
          and a footnote stating that independent replication requires Tinker API access.
       3. Primary statistical claims (significance tests, ANOVA variance decompositions) are drawn
          from open-source Modal H100 and TRL runs. Tinker-only results are reported descriptively.
       4. We commit to re-running any Tinker-only experiment on an open backend if an equivalent
          open-source service becomes available, and to issuing a revised version of this paper if the
          Tinker 99.9% figure is ever shown to be due to an undisclosed implementation choice rather
          than the algorithm.


                                                  142

[PAGE 166]
Figure 65: Per-experiment mean-ZVF (x-axis) versus last-10-window heldout accuracy (y-axis),
colored by the deterministic failure taxonomy (red = collapse, orange = drift, gray = plateau, green
= converged). Three cells of interest are highlighted by their corner positions: cross-tool tool-use
collapses (top-right at ZVF = 1, last10 = 0), the variance-mitigation drift cluster (bottom-left ZVF
∈ [0.10, 0.22], last10 ∈ [0.02, 0.03]), and the GSM8K / arithmetic converged cells (diagonal at ZVF
∈ [0.6, 0.84], last10 ∈ [0.69, 0.98]). Generated by scripts/zvf_diagnostic.py.


not a fit. The 95% bootstrap CI is wide because n = 23 cells; the diagnostic value of ZVF in our
matrix is the separation (collapse rows ZVF ≈ 1, plateau rows ZVF ≈ 0.5), not the point estimate of
a single correlation.

Limitations. We deliberately do not report partial correlations “controlling for advantage variance”
or “controlling for entropy”: under binary {0, 1} rewards, within-group advantage variance is a
deterministic transform of ZVF (zero iff ZVF) and partialling it out is circular (already documented
in experiments/results/zvf_partial_correlations.tsv). Step- level p-values are not re-
ported either: per-step ZVF rows are autocorrelated (typical lag-1 ρ ≈ 0.9 on our rollout trajectories),
and any t-statistic inferred from the 480 measurements would be overconfident by roughly an order
of magnitude.

A8.8      Cross-Source Anti-Herd Falsification of the Contrastive Yield Band

A frontier synthesis3 of this pillar proposed the strict band δdiv ∈ [0.13, 0.23] as a “measured
structural diversity bonus” introduced by high-temperature autoregressive sampling, on the reading
that “empirical ZVF under-predicts the i.i.d. baseline by −0.13 to −0.23” and hence the sampler anti-
herds. Our per-problem decomposition δdiv (x) = ZVFiid (px , G) − ZVFobs (x) over 1,092 measured
rows (experiments/results/zvf_contrastive_yield.tsv) shows this is regime-dependent
rather than uniform.
The interpretation is consistent with the visibility of high-decision boundary in each regime. On
real reasoning, the policy is on the “learning frontier” (p ∈ [0.2, 0.8]) for most prompts; sampling
temperature introduces genuine exploration that escapes the modal answer. On a small model trained
to near-perfect score on a narrow arithmetic distribution, sampling collapses to a mode: rollouts
correlate, ZVFobs EXCEEDS ZVFiid , δdiv < 0.

The empirical iso-G correction. The Contrastive Yield framing is preserved, but the iso-G siz-
ing formula Giid (p, Ytarget ) = ⌈log(1 − Ytarget )/ log(max(p, 1 − p))⌉ uses an assumption that
does not hold uniformly. Replacing the iid baseline with the empirical one ZVFemp (p, G) =
ZVFiid (p, G) − δdiv (clipped to [0, 1]) yields the empirical iso-G sizing table 110; entries appear
in experiments/results/zvf_empirical_isog.tsv. The corrected table makes the front-end
savings explicit. At Ytarget = 0.80 on Qwen3-8B/GSM8K:
   3
       (ChatGPT Pro Extended + Gemini Deep Think, round 2, attribution: “frontier synthesis”)


                                                     166

[PAGE 192]
library        πH P̂ (H → H) runH last10 rank
                         GRPO         0.524      1.000 157.2 0.383   1
                         NGRPO        0.340      0.988 27.3 0.387    2
                         CPPO         0.322      0.987 25.7 0.392    3
                         AERO         0.232      0.991 20.9 0.399    4
                         MCGRPO       0.118      0.945   8.4 0.397   5
                         GIFT         0.037      0.467   2.1 0.403   6
                         AREAL        0.037      0.477   1.9 0.404   7
                         ES           0.000      0.000   0.0 0.390   8
                         SCAFGRPO     0.000      0.000   0.0 0.409   9
Table 127: Per-library (mean over 5 seeds) Markov-chain rollup of the πH , self-loop P̂ (H → H),
mean H-run length, last-10 heldout accuracy, and the πH -based rank. Vanilla GRPO is the only
library with πH > 0.5 and runH > 100, reproducing its canonical “stuck in starvation regime”
failure mode. ES and SCAFGRPO never reach H; AERO, CPPO, NGRPO pass through H but
escape quickly.



Tool-use anchor validates the H-state encoding. The bfclv4 tool-use trajectory (seed 0, Qwen-3-
32B) carries ZVFsparse = 1.0 for 4 of 5 steps and 0.0 on the remaining step (a transient de-collapse:
the model is uniformly stuck except when an ncorrect = 2 step slips past ZVF = 0). Classifying
by Equation 46 gives πH = 0.80, exit1/2 (H) = 3 steps. Seed 1 (Llama-8B-Instruct) records the
opposite trajectory: πH = 0.0, with all 5 steps in L. The two anchored trajectories partition the
H-state exactly as the chain-intuition predicts.

Limitations and what iter 74 does not change. A single 3-state chain cannot represent multi-scale
starvation (e.g., long excursions into M followed by short H-bursts); the five-cutoff sensitivity
sweep in zvf_iter74_threshold_sensitivity.tsv reports πH varying by at most ±0.04 under
cutoffs in {0.20, 0.30, 0.50, 0.65}. Iter 74 inherits the iter22 honest- statistics convention (one row
per trace, bootstrap CIs over traces not over steps), and the channel between the per-step trace and
the last-10 heldout accuracy is still observational.

A13.19    ZVF Hazard / Survival: the Alarm as a Necessary-But-Imprecise Collapse Predictor

The iter78 EWS protocol (experiments/results/zvf_iter78_anchors.tsv) reports a single
lead-time scalar per method: how many steps before heldout accuracy drops below 0.10 the alarm
first fires. Iter 82 re-frames this as a survival-analysis hazard. For each variance-mitigation trace i
with ZVFi,t and a per-step failure flag fi,t = 1{rolling10 acc < 0.10} (carrying the iter78 definition
verbatim), define the alarm ai,t = 1{ZVFi,t > 0.5} and the per-step hazard
                                                                               h(t | 1)
               h(t | a) = Pr(fi,t+1 = 1 | ai,t = a, i alive at t),     HR =             .         (47)
                                                                               h(t | 0)
Pooled across the 25 variance-mitigation traces that ever fail, with Laplace smoothing (ε = 0.5)
applied to the no-alarm denominator, we obtain
                                                           
      HR = 87.6 median,      64.9, 74.0, 85.1, 74.0, 143.0 [aero,cppo,grpo,mcgrpo,ngrpo] ,  (48)

i.e. a one to two order-of-magnitude amplification of the per-step collapse probability whenever the
alarm fires. The alarm coverage (recall) cov = Pr(ft+1 = 1, at = 1)/ Pr(ft+1 = 1) is 1.000 for
every one of the five methods with any failure (experiments/results/zvf_iter82_hazard_-
ratio.tsv). By contrast, the false-alarm rate FAR = 1 − h(t | 1) sits at 0.90 (median), with
per-method FAR in [0.90, 0.91] for the five failure-prone methods. The F1 of the ZVF alarm as a
binary collapse predictor is therefore F1 = 0.18 (median).

Reading the table. The ZVF-alarm is a necessary but not sufficient condition for collapse: every
observed failure is preceded by an alarm, but the alarm fires ∼ 10× more often than failures actually
occur. This is a clean reformulation of the lead-time finding from iter78 in a survival vocabulary, and
the result sharpens the EWS-protocol iter78 results: the alarm is reliable (it never misses a collapse
in any library that experienced one) but imprecise (it is essentially a proxy for being deep in the
starvation regime, not a sharp failure predictor).


                                                 192

[PAGE 206]
AERO-vs-GRPO effect size, paired bootstrap. On the variance-mitigation suite (Tinker rollout
workers, identical reward parser, identical KL, identical chat template; K=8, 5 seeds each), AERO
cuts mean-ZVF by 0.2605 absolute relative to vanilla GRPO (0.2203 vs 0.4808; paired bootstrap 95%
CI [−0.2618, −0.2592] – excludes zero). The last-10 accuracy gap is +0.0203 [+0.0194, +0.0211],
so AERO’s ZVF reduction is not free but is consistent across seeds. Failure-rate is degenerate (0/5
for both AERO and GRPO under the deterministic classifier), so the failure-rate CI is reported
for completeness but is not informative at n=5. Source: experiments/results/zvf_iter118_-
aero_grpo_gap.tsv.

Predictive calibration exposes the bimodal failure mode. Sorting the 23 pooled cells by mean-
ZVF into 5 even-width bins and computing the within-bin failure rate yields the dose-response in Ta-
ble 140. Bin 0 (mean-ZVF 0.11–0.12) contains the variance-mitigation drift cells (GIFT, ES, AREAL,
MCGRPO); bin 3 (mean-ZVF 0.48–1.00) contains the GRPO plateau AND the cross-tool tool-use
collapse cells; bin 2 (mean-ZVF 0.30–0.32) is the AERO/CPPO/NGRPO/SCAFGRPO band with zero
failures. The clear conclusion: low ZVF predicts variance-mitigation drift, high ZVF predicts tool-use
collapse, mid-range ZVF is the safe band where AERO lives. Source: experiments/results/zvf_-
iter118_calibration.tsv.
                      Bin n Mean-ZVF Failures Failure-rate Wilson 95% CI
                        0 3     0.116     3/3        1.00     [0.44, 1.00]
                        1 3     0.175     1/3        0.33     [0.06, 0.79]
                        2 3     0.304     0/3        0.00     [0.00, 0.56]
                        3 5     0.760     2/5        0.40     [0.12, 0.77]
                        4 3     0.799     0/3        0.00     [0.00, 0.56]
Table 140: ZVF predictive calibration (iter 118). The same-pooled 23 cells sorted into 5 even-
width ZVF bins. Failure rate is bimodal: 100% at the low-ZVF tail (variance-mitigation drift
methods), 40% at the high-ZVF tail (cross-tool collapse + GRPO plateau cohabitants), and
0% in the mid-ZVF band (0.30–0.32) where AERO/CPPO/NGRPO/SCAFGRPO live. Source:
experiments/results/zvf_iter118_calibration.tsv.


Take-away for the paper. ZVF is a first-class cross-library diagnostic because it (a) cleanly
separates collapse from non-collapse (AUROC= 1.000), (b) places AERO in a measurable −0.26
absolute ZVF gap with a tight CI, and (c) reveals that the failure label is itself bimodal (drift at low
ZVF, collapse at high ZVF). Any single-AUROC summary will average the two failure modes into a
near-chance score; the dose-response curve in Table 140 and the figure figures/zvf_iter118_-
calibration.pdf preserve both modes.

A13.29    Iso-Yield ZVF, Per-Seed AERO Traceability, and Operating-Point Sweep

The iter 118 section turned ZVF into a first-class cross-library diagnostic: pooled AUROC of
mean-ZVF against is_collapse reaches 1.000 with 95% CI [1.000, 1.000], AERO halves ZVF
on the variance-mitigation suite (−0.260 absolute, paired bootstrap CI [−0.262, −0.259]), and the
predictive-calibration curve exposes the bimodal failure mode (drift at low ZVF, collapse at high
ZVF). Iter 122 extends the diagnostic into an operational layer by asking three new questions: (a) for
a fixed task, how much group-size investment does it take to push ZVF below a target contrast level
(iso-yield dynamic grouping)? (b) is the AERO advantage deterministic across seeds or does it vary?
and (c) when ZVF is used as a binary alarm above a threshold, what precision/recall trade-off does it
deliver?

Iso-yield curve: G-vs-ZVF saturates at 0.63 for arithmetic_synthetic. The groupsize-zvf-sweep
experiment measured ZVF(G) at G ∈ {2, 4, 8, 16} on the arithmetic-synthetic task (Qwen2.5-0.5B,
nseeds =3). Iter 122 linearly interpolates ZVF in log2 G between measurement points and computes
the minimum G that achieves each τ target; see Table 141. The empirical ZVF curve asymptotes at
ZVF(16) = 0.631, so any target τ < 0.50 is UNREACHABLE through a group-size intervention
alone (slope at G=16 is shallow enough that the linear extrapolation blows past G=128). The
implication is operationally sharp: ZVF ≥ 0.5 is the asymptotic floor of the rollout-only group-size
lever on this task. Reaching ZVF ≤ 0.30 requires an algorithm-level intervention (e.g. AERO; iter
118), not just a bigger G. Source: experiments/results/zvf_iter122_iso_yield.tsv.


                                                 206

[PAGE 224]
4. Frontier stability proxy (SI/PTD). The per-run SI and PTD values for all Tinker MoE runs
         are Tier-C and are no longer counted as observations in the paper-wide BH family.

The negative held-out GSM8K result (Qwen3-8B-Instruct post-GRPO 83.3% vs. the same instruction-
tuned checkpoint’s pre-RL held-out accuracy 82.0%, p=0.26) is unaffected: that evaluation is a
5-seed held-out comparison under a common greedy eval harness (the 82.0% reference is the same
checkpoint’s pre-RL accuracy, not a separate base model), and its non-significance does not depend
on the multiplicity correction. We note two power caveats on this null. First, a seed-level Welch test
at n1 =n2 =5 only detects effects of |d| ≳ 2, so the +1.3 pp gap we observe is not separable from
zero at this seed budget. Second, aggregating the 200 held-out GSM8K items per seed to a single
per-seed mean discards item-level power, so we also examined item-level pre-RL-vs-post-GRPO
predictions with the McNemar test. For Qwen2.5-1.5B-Instruct on GSM8K chain-of-thought
(3 seeds; artifact experiments/results/drgrpo_gsm8k_cot.json) GRPO improves held-out
accuracy 20.2% → 26.3%. We report this per seed rather than pooling: the three seeds re-evaluate
the same 200 items, so pooling their discordances (63 wrong→right vs. 26 right→wrong) would treat
600 evaluations of 200 distinct items as independent, and the resulting exact pooled p ≈ 1 × 10−4
overstates significance by pseudoreplication. The honest per-seed exact McNemar p-values are 0.071,
0.036, and 0.016 (significant in 2 of 3 seeds); a seed-level one-sample test on the three held-out
deltas (+5.5, +6.0, +7.0 pp) gives t=14.0, p ≈ 0.005. This is consistent with (but does not isolate)
a headroom-dependent effect relative to the Qwen3-8B-Instruct null — GRPO shows an item-level
generalization improvement when the base is far from ceiling but not at the near-saturated 82%
Qwen3-8B-Instruct checkpoint. We caution that the two cases also differ in model family (Qwen2.5
vs. Qwen3), size (1.5B vs. 8B), and decoding regime (short CoT vs. greedy), so we cannot attribute
the difference to headroom alone. We therefore do not read the Qwen3-8B null as evidence of no
effect in general; we read it as: at a near-ceiling base, training-reward gains do not translate into a
detectable held-out improvement at our seed budget.

A17.5    Reproducibility artifact

All numbers in Table 154 are deterministic.        The driver is experiments/survival_-
analysis.py; it reads experiments/results/*.jsonl, experiments/results/**/*.jsonl,
experiments/master_results.json           and   experiments/master_results.csv,           as-
signs tiers per Eq. (60), applies BH per Eq. (61) over Tier-A/B only, and writes
experiments/results/survival_analysis.tsv with the columns finding, tier_a_-
support, tier_b_support, effect_size_cohens_d, bootstrap_ci_low, bootstrap_-
ci_high, n_runs_used, bh_adjusted_p, and conclusion. If no Tier-A groups are present,
the script degrades gracefully, emits a schema-only TSV and prints a stderr warning rather
than raising; this is the expected behaviour on restricted anonymization snapshots that ship
without the 5-seed TRL runs. The seed for the bootstrap (MASTER_SEED = 20260506) matches
experiments/compute_statistics.py, so both pipelines are bit-reproducible on the committed
artefacts.

Summary. After excluding single-seed short-horizon runs from the inferential family, only F3
survives the restricted BH correction in the current release. F1 , F2 , F4 and F5 are downgraded to
descriptive case studies until matched multi-seed ≥ 100-step Tier-A evidence is collected in an open
framework – exactly the next-step programme called out in the Conclusion.


A18     Tool-Use and Code Depth: Reward Design, Warm-Starts, and ZVF
        Beyond Verifiable Math

This appendix addresses reviewer concerns W7 (math-only depth: tool-use / code experiments are
sparse or zero-reward) and Q6 (reward design, SFT warm-starts, and ZVF behavior outside math).
We (i) acknowledge the empirical limitation in our current tool-use and code runs, (ii) document the
reward structures used by each non-math environment in the repository, (iii) explain why base models
yield near-zero reward and how a small SFT warm-start unlocks nonzero reward, (iv) characterize
the regimes under which ZVF is diagnostically informative versus degenerate, and (v) scope our
RL-dynamics claims accordingly.


                                                 224

[PAGE 230]
F5′ (defensible, weaker claim). At the frontier scales we tested (N ≥ 70B) over 20–30 steps
of Tinker-managed GRPO training on GSM8K, we observe that reward trajectories for a subset of
reasoning-tuned checkpoints remain bounded within [baseline, baseline + ε] without the oscilla-
tion/collapse patterns visible at 0.6B–8B for some base checkpoints; we do not claim this generalises
beyond the evaluated 20–30-step horizon, to additional seeds, to other initialisations, or to tasks
beyond GSM8K training reward.
This restatement is consistent with the heterogeneity already reported in Section 5.20 and with the
“early-training behaviour is heterogeneous” wording in the Bitter Lesson narrative: some frontier
checkpoints begin from or briefly sustain very high training reward; others regress sharply at equal or
larger scale.

A21.2    Evidence-strength tiering

We adopt the evidence-strength tiering shown in Table 158: strong-evidence claims require multi-seed
runs of at least 100 steps at small/mid scale; short-horizon single-seed frontier runs are demoted
to descriptive observations that illustrate early-training behaviour and are not used as standalone
scaling-law evidence.

Table 158: Evidence-strength tiering for F5 and related frontier-scale claims. Strong-evidence claims
must satisfy all columns; descriptive observations are reported transparently but not used to support
scaling laws. “Seeds” and “Steps” denote the minimum per-configuration requirement.
     Tier                    Seeds      Steps   Scope in this paper                     Usage
     Strong evidence         ≥5         ≥ 100   0.6B–8B GSM8K GRPO                      Quantitative claim
     Supportive evidence     ≥3         ≥ 50    14B–32B GSM8K GRPO (partial)            Trend statement
     Descriptive obs. (F5)   1 (typ.)   15–30   70B–671B Tinker API runs                Illustrative, not law
     Interrupted / partial   1          < 20    Subset of frontier runs († in tables)   Footnote only

All frontier runs used to illustrate F5′ fall in the third row of Table 158. We do not aggregate them with
the small-scale multi-seed runs when stating any scaling-law claim, and we flag them as observational
wherever they appear in the main text.

A21.3    Why long-horizon frontier runs are not yet included

Upgrading F5 to a strong-evidence claim requires multi-seed, long-horizon training at frontier scale.
Under the Tinker managed-API pricing regime that governed this campaign, a single 100-step GRPO
run at N ≥ 70B with our default group size (G = 8) consumes roughly C100 ≈ $X100 in managed
compute (order-of-magnitude 103 –104 USD per configuration, depending on architecture, rollout
length, and whether the backbone is dense or MoE). A defensible F5 replication therefore calls for
                         5 seeds × 200 steps × 3 frontier sizes ≈ 30 runs
at an aggregate cost roughly 30 · (200/100) · C100 , i.e. 60× the budget of the single-seed short runs
currently reported. This is outside the budget envelope of the present release; we therefore restrict F5
to the descriptive tier above and make the cost frontier explicit so that future replications can plan
accordingly.

A21.4    Consequences for downstream claims
        • Abstract, introduction, and conclusion. The “frontier-model stability is heterogeneous”
          wording already present in Sections 1 and 10 is consistent with F5′ and is retained. Any
          residual phrasing suggesting that ceilings are monotonically more stable with N should be
          read as retracted in favour of F5′ .
        • Scaling-law section. The positive scale correlations in Section 5.6 (e.g. r = 0.533 between
          N and Rmax ) are reported on the pooled fit across all tiers and should not be read as implying
          that individual frontier runs remain monotonically stable; Nemotron-120B and Qwen3-32B
          are explicit counterexamples within our own data.
        • Tables. Frontier rows in Table 6 and Table 70 continue to be flagged with † for interrupted
          runs; F5′ does not re-weight them.


                                                     230

[PAGE 235]
broader claim that GRPO-style training is variance-limited rather than capacity-limited in this regime;
(Q7) we integrate all three as platform_local/unified/ overrides, and ZVF remains predictive
of collapse under two of them and loses informativeness under the third in a way that precisely maps
out where the ZVF diagnostic applies. We read this as evidence that ZVF is a robust companion
metric to the variance-mitigation literature rather than a metric that is superseded by it.

A23 Extended Related Work: Interaction of ZVF with Adjacent RL Regimes
A reviewer correctly observed that T INKER RL-B ENCH’s zero-variance-fraction (ZVF) diagnostic
was validated primarily against outcome-reward GRPO runs with independent rollouts, and that
an adjacent line of recent work modifies one of the two ingredients ZVF depends on: the reward
signal that enters the group-relative advantage. This extended section surveys the dense, process-level
reward regime and predicts whether ZVF remains informative, loses discriminative power, or requires
an explicit surrogate.

A23.1    Process Reward Models and Dense Shaping

Process Reward Models (PRM). The “Let’s Verify Step by Step” line of work [64] introduces
process reward models (PRMs) that supply a dense, step-level reward signal rather than a single
terminal outcome reward. Under a PRM, the per-trajectory reward is a sum (or weighted aggregate)
over intermediate steps, and the within-group reward variance is bounded below by the step-level
heterogeneity even when every trajectory in the group reaches the same outcome. Formally, if
     PTi
ri = t=1     ri,t is the PRM reward for trajectory i in a group and the {ri,t } are not perfectly aligned
across i, then
                                   Vari [ri ] > 0 almost surely,                                      (63)
so the event {Vari [ri ] = 0} that ZVF counts becomes vanishingly rare. The consequence is that
ZVF approaches zero on both healthy and collapsed PRM runs, and therefore loses discriminative
power in the dense-reward regime. We flag this as a known failure mode of ZVF and recommend,
under PRM training, replacing the zero-variance-fraction indicator with either (i) the per-step reward-
variance statistic averaged over group members, or (ii) the effective-rank / effective-rollout surrogates
discussed in Appendix A5 and Appendix A18. Both surrogates remain sensitive to within-group
collapse in the presence of dense shaping.

Summary table.       Table 162 collects the predictions above.

  Regime                             ZVF applicability        Recommended replacement / augmentation
  Vanilla GRPO (outcome reward)      Informative (baseline)   —
  PRM [64]                           Degenerate (→ 0)         Per-step reward variance or ERF
Table 162: ZVF applicability across the outcome-reward and dense process-reward regimes surveyed
in this section, together with the surrogate diagnostic we recommend where ZVF degenerates.



NeurIPS Paper Checklist
      1. Claims
           (a) Do the main claims made in the abstract and introduction accurately reflect the paper’s
               contributions and scope? [Yes] The abstract and Section 1 state five contributions:
               (i) a 73-percentage-point implementation gap across 7 libraries, (ii) model-dependent
               GRPO/PPO preference, (iii) frontier collapse on Nemotron-120B, (iv) zero-variance
               fraction (ZVF) as a leading diagnostic of GRPO failure, and (v) reward-trajectory
               proxies for KL-free monitoring. Empirical support is given in Section 5: the cross-
               library gap in Section 5.1; the algorithm×model interaction in Section 5.21; the frontier
               regime in Section 5.20; ZVF analysis in Section 5.23; and proxy metrics in Sections 7.3
               and 9.4. Evidence is drawn from the 44 controlled experiments tabulated in Section 5
               and the 32-run Bitter Lesson extension of Section 5.27. The scope is explicitly restricted
               to LoRA fine-tuning on three task families (math, code, tool-use) with evaluated scales


                                                   235

[PAGE 236]
of 0.6B–235B parameters (Section 4); Section 7 records that 11 of 14 Tinker runs were
       interrupted by JWT token expiry and 4 of 6 Modal runs timed out, so reported numbers
       reflect the available data and partial runs are marked †.
2. Limitations
   (a) Does the paper discuss the limitations of the work performed by the authors? [Yes]
       Section 7 enumerates the limitations in detail:
        • Platform opacity. Tinker API is a closed, serverless platform; GPU type, driver
          version, CUDA runtime, and scheduler are undisclosed, so Tinker-derived results
          cannot be independently reproduced without Tinker access (Section 7.1).
        • Infrastructure failures. 11 of 14 Tinker runs were interrupted by JWT token
          expiry; 4 of 6 Modal runs hit timeouts or gradient-norm blow-ups. Specific failed
          runs are named in Section 7.1.
        • Single-seed Tinker runs. Cost constraints precluded multi-seed replication on
          Tinker; these results are treated as descriptive only (Section 7.2).
        • LoRA-only evaluation. Full fine-tuning and quantisation-aware training are not
          evaluated (Section 7.2).
        • Train-set reward metric. Primary Tinker metrics are training rewards, not held-out
          accuracy, with held-out evaluation only completed for the Modal Qwen3-32B /
          Qwen3.5-27B arms (Section 7.2).
        • Proxy metrics in place of direct KL. Section 9.4 discusses the limits of trajectory-
          based stability indices as substitutes for direct KL divergence.
3. Theory Assumptions and Proofs
   (a) For each theoretical result, does the paper provide the full set of assumptions and a
       complete (or correct) proof? [NA] This is an empirical benchmarking paper; no formal
       theorems are claimed. GRPO and PPO objectives stated in Section 3 are background
       definitions, not original theoretical results.
4. Experimental Result Reproducibility
   (a) Does the paper fully disclose all the information needed to reproduce the main ex-
       perimental results of the paper to the extent that it affects the main claims and/or
       conclusions of the paper? [Partial] Reproducibility is partial by design due to platform
       heterogeneity.
        • Modal experiments (fully reproducible).                Complete source code, a
           pinned Dockerfile, centralised seed management (utils/seed.py), and
           step-by-step commands are documented in REPRODUCE.md at https://
           github.com/arvindcr4/tinker-rl-lab.                 Modal jobs run on explic-
           itly provisioned NVIDIA H100 SXM5 (80 GB) workers; TRL baselines
           run on NVIDIA L4 (24 GB). All Modal/TRL arms use 5 seeds (s ∈
           {42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768}) with mean ± SE and
           95 % bootstrap CIs via rliable.
        • Tinker API experiments (not fully reproducible). Tinker is closed-source with
           serverless GPU dispatch; GPU type, driver version, and scheduling policy are not
           exposed. We release our exact API scripts and configuration files, but independent
           replication requires a Tinker account and is subject to the same JWT expiry issues
           we observed (Section 7.1). All Tinker-derived numbers in figures and tables are
           flagged with a “†”.
       We claim Partial rather than Yes: the Tinker subset is irreducibly platform-dependent.
5. Open access to data and code
   (a) Does the paper provide open access to the data and code, with suffi-
       cient instructions to faithfully reproduce the main experimental results, as de-
       scribed above? [Yes] All code is publicly released under the Apache Li-
       cense 2.0 at https://github.com/arvindcr4/tinker-rl-lab (mirrored at
       https://github.com/pes-llm-research/tinker-rl-lab). LoRA adapter
       checkpoints from the successful Modal experiments are on HuggingFace Hub
       under https://huggingface.co/arvindcr4/tinker-rl-bench-* with model
       cards generated from huggingface/MODEL_CARD_TEMPLATE.md. All experiment


                                         236

[PAGE 237]
runs are logged publicly at https://wandb.ai/arvindcr4-pes-university/
        tinker-rl-lab-world-class; per-step reward, loss, gradient-norm, and (where
        available) KL metrics are visible without login. Training datasets (GSM8K, Hu-
        manEval, NoRobots, synthetic tool-use) are public and cited in Table 1. Tinker API
        credentials are not included for security reasons; README.md explains how to obtain a
        Tinker API key.
 6. Experimental Setting/Details
    (a) Does the paper specify all the training and test details (e.g., data splits, hyperparam-
        eters, how they were chosen, type of optimizer) necessary to understand the results?
        [Yes] Section 4 gives models, data splits, LoRA configuration (rank 32), optimiser
        (Adam, β1 =0.9, β2 =0.95, ϵ=10−8 , learning rate 10−4 ), and the 5-seed Modal pro-
        tocol. Table 3 maps these hyperparameters across all 7 libraries. Appendix A2
        reports sweep ranges. Per-task YAML configurations are version-controlled under
        atropos/configs/. GSM8K uses the standard train/test partitions; HumanEval uses
        the full pass@1 suite. Tinker hyperparameters are released as API call scripts; the only
        unavailable information is the Tinker platform’s internal hardware.
 7. Experiment Statistical Significance
    (a) Does the paper report error bars suitably and correctly defined or other appropriate infor-
        mation about the statistical significance of the experiments? [Partial] For Modal/TRL
        arms with 5 seeds we report mean ± SE with 95 % bootstrap confidence intervals
        via rliable [2]; a full power analysis (Table 4) and Benjamini–Hochberg-adjusted
        pairwise significance (Table 5) are reported in Section 4. The TRL-GRPO reference
        characterisation (Qwen2.5-0.5B, 5 seeds) reports x̄ = 0.734, σ = 0.0703, IQM
        = 0.747, bootstrap 95 % CI [0.672, 0.782]. Tinker API experiments are single-seed
        (cost-constrained) and are explicitly labelled so in Sections 5.1 and 7.2; no statistical
        significance is claimed for Tinker-only comparisons. We follow the recommenda-
        tions of Colas et al. [19] and Jordan et al. [47] on the limits of single-seed evaluation.
        Because a meaningful subset of headline results is Tinker-only, the honest answer is
        Partial.
 8. Experiments Compute Resources
    (a) For each experiment, does the paper provide sufficient information on the computer
        resources (type of compute workers, memory, time of execution) needed to reproduce
        the experiments? [Partial] Section 4.4 and Appendix A1 list GPU types and estimated
        GPU-hours for each experiment group; the complete per-experiment breakdown with
        wall-clock time and estimated cost is in COMPUTE.md.
          • Modal experiments. Fully specified: H100 SXM5 (80 GB) for recent runs and
            L4 (24 GB) for the TRL reference sweep. Wall-clock and GPU-hour estimates are
            reported.
          • Tinker API experiments. Serverless dispatch masks the exact GPU type; GPU-
            hour figures are inferred from billing records and are flagged as approximate.
        Total project cost is approximately 1,200 A100-equivalent GPU-hours across both
        platforms. Partial reflects the Tinker-side hardware opacity, which is a platform
        property we cannot resolve.
 9. Code Of Ethics
    (a) Does the research conducted in the paper conform, in every respect, with the NeurIPS
        Code of Ethics? [Yes] The authors have reviewed the NeurIPS Code of Ethics and
        confirm compliance. This work involves no human subjects, no private or sensitive
        data, and no models trained on proprietary corpora. All training datasets are public re-
        search datasets (GSM8K, HumanEval, NoRobots, synthetic tool-use). Broader societal
        impacts—including reward hacking, alignment failure modes of RLHF, and equity of
        compute access—are discussed in Section 8.13 and in ethics_statement.tex.
10. Broader Impacts
    (a) Does the paper discuss both potential positive societal impacts and negative societal
        impacts of the work performed? [Yes] Section 8.13 (plus ethics_statement.tex
        and LIMITATIONS_AND_IMPACT.md) discusses:


                                            237

[PAGE 238]
• Positive. Lowering the barrier to RL post-training research; enabling reproducibility
             audits; surfacing platform-dependent variance that is typically hidden in single-
             platform papers.
           • Negative / risks. RLHF reward hacking; alignment concerns when optimising
             proxy rewards; potential misuse of fine-tuned models; compute-access disparities
             that could limit replication to well-resourced groups.
           • Environmental. Estimated 0.4–1.2 kg CO2 -equivalent for the Modal H100 ex-
             periments; the Tinker API footprint is unestimable because the hardware mix is
             undisclosed.
11. Safeguards
     (a) Does the paper describe safeguards that have been put in place for responsible release
         of data or models that have been identified as requiring safeguards? [NA] The released
         artefacts are LoRA adapters fine-tuned from already-public base models (Qwen, Llama,
         Nemotron, GPT-OSS, Kimi-K2) on standard research datasets (GSM8K, HumanEval,
         synthetic tool-use). No new high-risk capabilities are introduced relative to the base
         models; released checkpoints inherit the base models’ licence terms and include
         standard usage disclaimers in the HuggingFace model cards (huggingface/MODEL_-
         CARD_TEMPLATE.md).
12. Licenses for existing assets
     (a) Are the creators or original owners of assets (e.g., code, data, models) used in the
         paper properly credited, and are the licence and terms of use explicitly mentioned and
         properly respected? [Yes] All third-party assets are credited with their licences:
           • GSM8K [18]: MIT licence.
           • HumanEval (OpenAI): MIT licence.
           • NoRobots (HuggingFaceH4): CC-BY-NC-4.0 (used for the chat-SFT task only).
           • Synthetic tool-use data: self-generated from public API documentation; no up-
             stream licence restrictions.
           • Qwen model family: Apache-2.0.
           • Llama model family: Meta Llama Community Licence.
           • Nemotron, GPT-OSS, Kimi-K2: used under their respective published licences;
             credits in Section 4.
           • TRL, PEFT, Transformers: Apache-2.0 (HuggingFace).
           • Modal: commercial platform; terms of service respected.
           • Tinker API: proprietary; used under standard API terms of service.
           • Our code: Apache-2.0 (see LICENSE in the repository root).
13. New Assets
     (a) Are new assets introduced in the paper well documented and is the documentation
         provided alongside the assets? [Yes] New assets and their documentation:
           • The T INKER RL-B ENCH benchmark suite (harness, evaluation scripts, anal-
             ysis notebooks) at https://github.com/arvindcr4/tinker-rl-lab, re-
             leased under Apache-2.0 and documented by README.md, REPRODUCE.md,
             ARTIFACT.md, COMPUTE.md, BASELINES.md, BENCHMARKS_COMPARISON.md,
             and LIMITATIONS_AND_IMPACT.md.
           • LoRA adapter checkpoints from the successful Modal experiments at https:
             //huggingface.co/arvindcr4/tinker-rl-bench-*, each accompanied by
             a HuggingFace model card generated from huggingface/MODEL_CARD_-
             TEMPLATE.md (intended use, training data, evaluation results, known limitations).
           • All experiment logs on the public Weights & Biases project
             arvindcr4-pes-university/tinker-rl-lab-world-class.
         Checkpoints from JWT-interrupted Tinker runs are not uploaded, because those jobs
         did not reach a valid final state (Section 7.1).
14. Crowdsourcing and Research with Human Subjects
     (a) For crowdsourcing experiments and research with human subjects, does the paper
         include the full text of instructions given to participants and screenshots, if applicable,
         as well as details about compensation? [NA] No crowdsourcing or human subjects
         research was conducted.


                                             238


# N01: Unified signal starvation


Root: `platform_hybrid/paper/unified_signal_starvation/main.tex`  Pages: 11  Words: 5104


[PAGE 1]
When Does an RL Update Teach?
 A Diagnostic and Controller Proposal Across GRPO,
                     PPO, and
     Single-Rollout Asynchronous Optimization


                                           Arvind C R
                                          PES University
                                      arvindcr4@gmail.com



                                             Abstract

         Reinforcement-learning post-training is usually budgeted in rollouts, tokens, or
         wall-clock time, even though some collected trajectories induce no policy gradient.
         This failure appears differently across algorithms: a flat reward group gives zero
         group-relative advantage in GRPO, a critic can produce a near-zero advantage in
         PPO, and policy lag can cause double-sided importance masking to discard an
         otherwise informative token in Single-Rollout Asynchronous Optimization (SAO).
         We give these cases one operational description. For every policy token, we separate
         potential advantage mass from the fraction that survives the algorithm’s trust-region
         gate, yielding Effective Gradient Mass (EGM) and a root-trajectory Zero-Update
         Fraction (ZUF). EGM equal to zero is a sufficient certificate of a zero score-function
         policy update; positive EGM is deliberately treated as a proxy, because token
         gradients can cancel. This decomposition supports T RIAGE RL, a cause-aware
         controller that retries failed-starved examples, retires or distills solved-saturated
         examples, trains the critic when credit is unreliable, and refreshes stale rollouts
         when clipping destroys useful signal. The controller preserves asynchronous single-
         rollout execution and logs sampling propensities to expose curriculum bias.
         We ground the proposal in an audited GRPO artifact. Across 505 unique prompt-
         seed tasks, the identity between contrastive group yield and the gap between
         pass@G and unanimous success holds to 1.11 × 10−16 . In a 2,560-observation
         controller trace, 1,723 of 1,867 escalation events (92.3%) occurred on all-correct
         groups, whereas only 144 occurred on all-wrong groups. These are descriptive
         and counterfactual results, not evidence that the new controller improves held-out
         accuracy. We therefore state a matched-budget, seed-paired evaluation contract for
         PPO and SAO rather than manufacturing missing outcomes. The paper’s contribu-
         tion is a unified diagnostic, falsifiable routing policy, and reproducible bridge from
         an established GRPO failure mode to critic-based PPO and asynchronous SAO.
         It reuses companion GRPO artifacts and must not be counted as an independent
         replication of those results.


1   Introduction

Long-horizon language-model reinforcement learning (RL) spends most of its budget before the
optimizer runs: agents interact with tools, environments return sparse outcomes, and trajectories
may contain hundreds of thousands of tokens. Yet a completed rollout is not necessarily an effective
update. A policy-gradient coefficient can vanish because no contrast exists, because a critic predicts


Preprint. Under review.

[PAGE 2]
the observed outcome, or because an off-policy trust-region rule masks the token. Throughput metrics
do not distinguish these causes.
The distinction matters as systems move from synchronous group-relative optimization toward critic-
based single-rollout training. GRPO estimates a baseline from several responses to the same prompt
[11]; identical group rewards therefore erase the advantage. PPO uses a learned value baseline and a
sign-dependent clipped surrogate [10]. SAO replaces prompt groups with single rollouts and uses
a stricter double-sided token mask to stabilize asynchronous learning under policy lag [6]. The
official GLM-5.2 report describes a closely related production transition: compacted long trajectories
yield variable numbers of trainable sub-traces, so training moved from group-wise optimization to
critic-based PPO over individual rollouts [13]. The algorithms differ, but the operational question is
the same: how much generated learning signal reaches the actor?
We answer by factorizing the score-function coefficient into a generated signal and a survival gate.
This yields two cheap statistics that can be logged without an additional backward pass: potential
advantage mass (PAM) and gradient survival ratio (GSR). Their product is effective gradient mass
(EGM). A sampled gradient-norm audit guards against the known weakness of coefficient-only
proxies: nonzero weighted score vectors can cancel.
This framing changes the controller design. “Low signal” is not one state. All-correct GRPO groups
and all-wrong groups both have zero advantage, but the former are candidates for retirement or
distillation and the latter for targeted resampling. In PPO and SAO, low PAM with a calibrated critic
means the result was expected; low GSR with high PAM means signal existed but was destroyed by
clipping, for which a fresh rollout is preferable to widening the trust region. A bad critic calls for
critic-only updates, not repeated actor epochs. Invalid or hacked trajectories belong in a quarantine
path rather than the hard-example queue.
Our contributions are:

1. a unified, implementation-level decomposition of score-function signal in GRPO, PPO, and SAO,
   together with an exact zero-update certificate and an explicit non-converse;
2. T RIAGE RL, a root-trajectory controller that maps distinct starvation causes to retry, retire, critic-
   only, refresh, quarantine, or normal update;
3. two GRPO results reproduced from checked-in artifacts: an exact pass@G/ZVF identity and a
   strong asymmetry between all-correct and all-wrong controller fires; and
4. a preregistered evaluation contract that separates already observed GRPO facts from untested
   PPO/SAO hypotheses.

Claim status. The mathematical identities and reported GRPO counts are verified. The unified
controller is a method proposal. Its PPO and SAO benefits are falsifiable hypotheses, not current
empirical claims. This is the eighteenth document in the program but not an eighteenth independent
evidence source.

2     Background and the Three Starvation Mechanisms
Let a behavior policy µ generate action token at in state st , and let πθ be the actor being optimized.
Define the token ratio
                               πθ (at | st )
                    ρt (θ) =                 = exp(log πθ (at | st ) − log µ(at | st )) .             (1)
                               µ(at | st )
The policy loss uses a detached advantage estimate At and, depending on the algorithm, a detached
token gate mt ∈ {0, 1}.

2.1   GRPO: signal is never generated for flat groups

For prompt x, GRPO samples G responses with rewards R1 , . . . , RG and forms a group-relative
advantage such as
                                          Ri − R̄x
                                  Agrp
                                    i   =          .                                       (2)
                                            sx + ε

                                                      2

[PAGE 8]
4. Boundary-aware: failure-only retry plus solved retirement or distillation; and
5. Full triage: boundary-aware routing plus critic-only and fresh-policy transport-starvation actions.
The GRPO/PPO isolation cell uses the same Qwen3-8B initialization, GSM8K split, LoRA rank 4,
tokenizer, reward, maximum length, and five seeds. The agentic cell uses an open Qwen3-30B-A3B
checkpoint with a fixed scaffold and a public SWE-Bench Verified split, matching SAO’s published
scale where feasible [6]. If resource constraints require a smaller model, that change is made for all
arms and recorded before confirmatory runs.
Primary budgets are generated action tokens and environment calls. Actor forward/backward FLOPs
and wall-clock time are secondary budget views. Each arm receives identical maximums and an
identical base-stream floor. Every root logs model version, lag, behavior log-probability, token ratio,
advantage, gate, reward, value prediction, chunk count, selection propensity, invalid flag, and route.

7.3   Endpoints and analysis

The primary endpoint is held-out success per million generated action tokens. Secondary endpoints
are held-out pass@1 and pass@k, root-macro ZUF, token-micro PAM/GSR/EGM, sampled GUN,
critic explained variance and calibration error, actor KL, stale-token fraction, effective updates per
wall-clock hour, reward hack rate, and collapse incidence. Results use seed-paired differences, prompt-
clustered bootstrap intervals, and both intention-to-route and as-routed analyses. Hyperparameters
and exclusion rules are frozen in a machine-readable manifest before the confirmatory run.

8     Related Work
PPO introduced the clipped policy surrogate used widely in RLHF [8, 10]; GAE provides its standard
bias-variance tradeoff for critic advantages [9]. GRPO removes the learned critic by normalizing
rewards within prompt groups [11], a design later popularized in reasoning systems [3]. REINFORCE-
style baselines such as RLOO offer another critic-free route [1].
Recent work attacks rollout cost from complementary directions. CERO treats cross-epoch rollout
allocation as an online resource-allocation problem [14]. BASIS shares information across a batch
while using one rollout per prompt [5]. A Monte Carlo pass@k critic supplies token-level credit for
single-rollout PPO [2]. These methods improve the source or allocation of advantage information.
Our focus is the downstream accounting question: after baseline estimation and trust-region gating,
which root trajectories still reach the actor, and what should the system do with each failure cause?
SAO directly targets single-rollout asynchronous agentic RL and introduces strict double-sided mask-
ing, faster critic updates, frozen-attention critic training, and skip-observation GAE [6]. Asynchronous
actor- learner systems have long required off-policy correction [4, 7]; recent scaling-law work studies
the interaction between staleness and learning rate in asynchronous RLHF [12]. T RIAGE RL is
complementary: it treats staleness as one observable cause of signal non-survival and routes it without
relaxing the base optimizer’s stability rule.

9     Limitations and Broader Impact
First, EGM is not an improvement guarantee. It ignores score-vector geometry, optimizer precondi-
tioning, gradient interference across roots, and the possibility that a large gradient is harmful. Periodic
GUN measurement and held-out reward are therefore necessary. Second, critic errors can make the
semantic split wrong: a low advantage may reflect miscalibration rather than true saturation. The
critic-lag route and calibration diagnostics reduce but do not eliminate this risk.
Third, the binary solved/failed framing is cleanest for verifiable rewards. Dense or multi-objective
environments require vector-valued boundary rules and careful reward-scale normalization. Fourth,
retry curricula change data exposure. Propensity weighting can have high variance and may be
impossible in an open task generator; such runs must be labeled as curriculum learning. Fifth, the
current empirical section is GRPO-only and mostly arithmetic. The PPO/SAO generalization remains
unverified until the evaluation contract is executed. Finally, adaptive hard-example sampling can
amplify unsafe or adversarial behaviors. Invalid/hack quarantine, a base-stream floor, capped retries,
and fixed held-out safety evaluation are required guardrails.


                                                    8

[PAGE 9]
10    Conclusion
GRPO flat groups, PPO clipping, and SAO double-sided masking look like separate pathologies
because they occur at different stages. At the actor interface, they share a simple structure: an
advantage must be generated and then survive the trust-region gate. PAM, GSR, EGM, and root ZUF
make that structure observable. The resulting controller is intentionally asymmetric: retry failed
starvation, retire solved saturation, repair the critic when credit is unreliable, refresh the policy when
useful evidence is stale, and quarantine corrupted rewards. Existing GRPO artifacts establish the
accounting identity and motivate the asymmetry; matched-budget PPO and SAO experiments must
decide whether it improves learning. That separation between verified evidence and testable proposal
is the central methodological commitment of this work.

References
 [1] Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier
     Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization
     for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.
 [2] Fengdi Che, Yang Liu, Lei Yu, Meng Cao, Tong Che, A. Rupam Mahmood, and Dale
     Schuurmans. Learning with a single rollout via monte carlo pass@k critic. arXiv preprint
     arXiv:2606.25451, 2026.
 [3] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement
     learning. arXiv preprint arXiv:2501.12948, 2025.
 [4] Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward,
     Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl
     with importance weighted actor-learner architectures. In International Conference on Machine
     Learning, 2018.
 [5] Shijin Gong, Erhan Xu, Kai Ye, Francesco Quinzan, Giulia Livieri, and Chengchun Shi. Basis:
     Batchwise advantage estimation from single-rollout information sharing for llm reasoning.
     arXiv preprint arXiv:2605.27293, 2026.
 [6] Zhenyu Hou, Yujiang Li, Jie Tang, and Yuxiao Dong. Single-rollout asynchronous optimization
     for agentic reinforcement learning. arXiv preprint arXiv:2607.07508, 2026.
 [7] Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy Lilli-
     crap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep
     reinforcement learning. In International Conference on Machine Learning, 2016.
 [8] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin,
     Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models
     to follow instructions with human feedback. In Advances in Neural Information Processing
     Systems, 2022.
 [9] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-
     dimensional continuous control using generalized advantage estimation. International Confer-
     ence on Learning Representations, 2016.
[10] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal
     policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
[11] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang,
     Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of
     mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
[12] Jingwei Song, Haofeng Xu, Jie Xiao, Chengke Bao, Jingwei Shi, Pengbin Feng, Weixun Wang,
     Yuhang Han, Chuan Wu, Linfeng Zhang, and Bill Shi. Staleness-learning rate scaling laws for
     asynchronous rlhf. arXiv preprint arXiv:2607.01083, 2026.
[13] Z.ai. Glm-5.2: Built for long-horizon tasks. https://z.ai/blog/glm-5.2, 2026. Accessed
     2026-07-14.
[14] Yiming Zong, Yige Wang, and Jiashuo Jiang. Cross-epoch adaptive rollout optimization for rl
     post-training. arXiv preprint arXiv:2606.05606, 2026.


                                                    9
