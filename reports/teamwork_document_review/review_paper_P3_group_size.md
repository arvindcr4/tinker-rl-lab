**Segment scope**: platform_hybrid/paper/paper_P3_group_size.tex (Group Size in GRPO Is Budget- and Regime-Dependent: Contrast Density, Measured Bounds, and the Bridge to DPO)

# Potential Mistakes and Improvements

- **[Critical]** Parametric extrapolation and continuous thresholding from non-empirical grid points.
  In Section `sec:pillar3-iter23` (`sections/group_size_iter23.tex`), an unconstrained saturation model $|\Delta(T)| = a \cdot (1 - e^{-T/\tau})$ is fitted to four points ($T \in \{1, 4, 16, 64\}\,$M).
  From this fit, the text derives high-precision critical thresholds: $T_\text{crit}^{1\%} = 0.32\,$M, $T_\text{crit}^{3\%} = 0.99\,$M, $T_\text{crit}^{5\%} = 1.73\,$M, and $T_\text{crit}^{10\%} = 3.99\,$M tokens.
  However, the underlying values do not come from empirical training runs.
  They originate from the hardcoded `FALLBACK_ROWS` dictionary in `platform_hybrid/experiments/group_size_token_normalized.py`.
  Fitting a continuous curve to four non-empirical points creates pseudo-precision.
  It implies a physical phase transition in RL training where no empirical data exist.
  The authors should remove the sub-percent continuous threshold claims.
  They should present the transition strictly as a qualitative hypothesis.

- **[Major]** Unobserved token-level predictions in Critic-Degeneracy Hypothesis (CDH) regression.
  Section `sec:frontier-critic-degeneracy` (`sections/critic_degeneracy_hypothesis.tex`) tests five hypotheses for critic degeneracy.
  Under H5, an OLS regression reports that a 1D static regressor explains 49% of prompt reward variance ($R^2 = 0.490$, $n=7$).
  However, the dependent variable in this regression is the empirical Monte-Carlo prompt-mean reward $r_\text{mean}$.
  The actual PPO value-head predictions $V_\phi(x_{1:t})$ along token trajectories were not recorded or analyzed.
  Regressing prompt mean reward on prompt difficulty (RQS) proves only that RQS tracks task difficulty.
  It does not demonstrate that the neural critic collapsed to a static prompt-mean regressor.
  The authors must clarify that H5 evaluates prompt difficulty correlation rather than value-head state collapse.

- **[Major]** Conflation of value-head loss scale with policy gradient variance in CDH gradient norm test.
  In hypothesis H1 of Section `sec:frontier-critic-degeneracy`, PPO shows a $156\times$ larger gradient norm than GRPO (96.79 vs 0.62) and a 6% higher coefficient of variation.
  The logged `grad_norm` combines actor and critic parameters before clipping on the final inner epoch.
  PPO includes a value-head MSE loss weighted at $0.5\times$ on terminal rewards.
  Terminal value errors inflate the gradient norm independently of actor policy variance.
  GRPO has no value head and computes gradients exclusively for actor parameters.
  Comparing total gradient norm between an actor-critic model and an actor-only model introduces a structural confound.
  The authors should separate actor-head gradients from critic-head gradients before drawing variance-reduction conclusions.

- **[Major]** Confounded step budget and unadjusted learning rate in matched-rollout trajectory check.
  In Section `sec:group-size` (lines 265--288), a two-seed matched-budget panel compares $G=2$ against $G=16$ at 2,560 rollouts per arm.
  The $G=2$ arm runs 160 optimizer steps, whereas the $G=16$ arm runs only 20 optimizer steps.
  Both arms use the exact same fixed learning rate ($3 \times 10^{-5}$).
  The $G=16$ arm receives $8\times$ fewer parameter updates.
  The failure of $G=16$ to reach ceiling reward within 20 steps is confounded by optimizer step count.
  Furthermore, held-out accuracy was not measured for these runs.
  The panel demonstrates training-speed differences under step imbalance, not asymptotic algorithm quality.
  The authors should emphasize that the comparison is step-confounded and lacks held-out validation.

- **[Major]** Non-linear distortion of preference weighting by advantage standard deviation whitening.
  The mathematical bridge from GRPO to Direct Preference Optimization (DPO) rewrites centered advantages as an all-pairs contrast.
  This identity is exact for unwhitened reward differences $(r_i - r_j)$.
  In standard GRPO, the group advantage divides by sample standard deviation $\hat{\sigma}$.
  For binary rewards, dividing by $\hat{\sigma}$ upweights groups with rare outcomes ($k=1$ or $k=G-1$) by a factor of approximately $\sqrt{G}$.
  In contrast, balanced groups ($k=G/2$) receive standard weighting ($\approx 2$).
  Standard DPO applies a uniform margin without group-level standard deviation scaling.
  Section `sec:frontier-group-size` should formalize the non-linear weighting distortion introduced by $1/\hat{\sigma}$ whitening.

- **[Major]** Lack of statistical power to certify equivalence despite high retention headline.
  The Abstract and Section `sec:group-size` emphasize that $G=2$ retains 100.34% of 16-GRPO performance on arithmetic.
  However, the empirical sweep contains only $n=3$ seeds on a 40-step toy addition task.
  The minimum detectable effect size is $d = 3.07$.
  In Table `tab:groupsize-iter27-synthesis`, the two one-sided test (TOST) at $\epsilon = 0.02$ yields $p = 0.056$ on training reward and $p = 0.100$ on held-out accuracy.
  Neither test achieves statistical significance at $\alpha = 0.05$.
  Equivalence between $G=2$ and $G=16$ is not statistically established.
  The authors should temper the 100.34% headline to avoid misinterpretation as certified equivalence.

# Minor Corrections and Typos

- **[Minor]** Overfull table boxes in Section `sec:group-size`.
  Table 2 (`tab:groupsize-g4-g32`) and Table 4 (`tab:groupsize-zvf-theory`) previously exceeded line width by up to 20.7pt.
  Both tables have been enclosed in `\resizebox{\linewidth}{!}{% ... }` in the source file.

- **[Minor]** PDF string math shift warnings in section headers.
  Section titles in `sections/group_size_iter23.tex` ($T_\text{crit}$) and `sections/frontier_synthesis_group_size.tex` (GRPO$\leftrightarrow$DPO) triggered hyperref warnings.
  These titles have been wrapped using `\texorpdfstring` in the source files.

- **[Minor]** Float specifier replacement warning.
  Tables in `sections/group_size_iter23.tex` used `[h!]`, triggering LaTeX float replacement warnings.
  The specifiers were updated to `[htbp]`.

- **[Minor]** Redundant notation between gradient utilization and token efficiency.
  In Section `sec:appendix:group-size-reconcile`, $\mathrm{GU} = 1 - \mathrm{ZVF}$ is defined as a fraction, while $\mathrm{GE}_\text{tok}$ measures gradient norm per token.
  The scaling constant 55,000 in $\widehat{\mathrm{GE}}_\text{tok}$ is an empirical calibration factor.
  Explicit units should accompany $\widehat{\mathrm{GE}}_\text{tok}$ to prevent confusion with dimensionless ratios.
