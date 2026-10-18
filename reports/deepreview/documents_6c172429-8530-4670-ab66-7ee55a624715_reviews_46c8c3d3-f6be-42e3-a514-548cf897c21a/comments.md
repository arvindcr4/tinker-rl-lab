# Anchored comments

## [CRITICAL] Contradictory characterization of variance_mitigation.tsv as synthetic vs. real runtime measurements  (p.8 "We elevate ZVF from a perexperiment sanity check to a crosslibrary diagnostic by")

A major concern regarding internal consistency and scientific validity is how the 9-method variance-mitigation corpus (`experiments/results/variance_mitigation.tsv` and `zvf_by_library.tsv`, covering GRPO, AERO, CPPO, NGRPO, SCAFGRPO, MCGRPO, GIFT, AREAL, and ES) is characterized across the manuscript:

- **Declared Evidence Boundary:** The Abstract (p. 1), the caption of Table 6 (p. 8), Section 5.4 (p. 12), and the "Evidence boundary" paragraph in Section 5.5 (pp. 12–13) explicitly state that the rows in `variance_mitigation.tsv` are a synthetic *"simulation projection, not model-training runs,"* *"should be read as synthetic, not as independent real-library deployments,"* and *"cannot support an empirical ranking, mitigation claim, or statement that AERO or RL-ZVP works in this repository."*

- **Contradictory Empirical, Mechanistic, and Operational Claims:** Despite this explicit disclaimer, multiple passages in Sections 4, 5, 6, 8, 9, and 10 cross this evidence boundary by treating the exact same numbers as empirical measurements from live model-training runs on managed hardware:
  - **Section 4.1 (p. 8):** States that Table 6 aggregates *"the per-step ZVF emitted by every mitigation method we ran on a single managed runtime (Tinker rollout workers, identical reward parser, identical $\beta=0$ KL, and identical chat template)"* and claims under "Honest scope" that *"differences in Table 6 reflect the mitigation mechanism, not confounding infrastructure."*
  - **Section 4.3 (p. 9):** Asserts under "Stack-controlled confound" that *"AERO's halving of ZVF on our runtime (Table 6) is a real, same-stack measurement; we do not claim it transfers to other RL stacks without re-measurement."*
  - **Section 5.5 (p. 13):** Immediately following the "Evidence boundary" disclaimer, the bulleted list under "Reviewed-record correction" interprets Table 6 / Table 8 as empirical validation (*"Three of five GRPO seeds exhibit a per-step collapse flag... zero of five AERO seeds do... This is consistent with AERO's qualitative claim"*).
  - **Sections 6.2–6.5 (pp. 16–17):** Derive the primary training-time trajectory dynamics (monotonic signal starvation, lag-1 autocorrelation $\rho_1 \approx 0.94$, and collapse lead-time) from this 45-run corpus, describing it as *"measured runs"* (pp. 7, 20) from *"a 5-seed verifiable arithmetic correctness setting on Qwen2.5-0.5B with batch size 16 and group size 8"* (Section 6.5, p. 17).
  - **Sections 8.1–8.3 (pp. 20–23), Section 9 (pp. 23–25), and Sections 10.10, 10.14–10.16 (pp. 29–36):** Resample the $n=5$ simulated seeds to claim *"the sharpest empirical contrast in the variance-mitigation corpus"* (p. 23), apply Wilcoxon tests to claim *"AREAL's terminal ZVF decrease is a real signal"* (p. 22), fit logistic regression models to issue operational alert thresholds (p. 24), and analyze quintile mechanisms as *"genuine starvation relief"* (p. 34). Bootstrapping or fitting models on simulated trajectories only measures the programmed parameters of the simulation generator rather than empirical variance across actual LLM reinforcement learning deployments.

**Recommendation:** Reconcile these conflicting statements across Sections 4 through 10 with the evidence boundary declared in the Abstract and Section 5.5. Remove all claims that the nine methods were executed on Tinker rollout workers or `Qwen2.5-0.5B` hardware (or replace these rows with completed real model-training trajectories), and reframe all findings derived from `variance_mitigation.tsv` strictly as analytical properties of the synthetic simulation model rather than empirical algorithm evaluations.

## [CRITICAL] Retracted anti-herding analysis left intact in main text (Sections 10.7–10.13)  (p.27 "Retraction notice (2026-10-16). This subsection depends on the “antiherding” div")

Both Section 10.7 (*"Iso-Yield Curves Across Libraries"*) and Section 10.11 (*"Dynamic Iso-Yield Group Sizing (Iso-G)"*) open with bold **"Retraction notice (2026-10-16)"** disclaimers explaining that the "anti-herding" divergence $\delta_{\text{div}} = \text{ZVF}_{\text{iid}} - \text{ZVF}_{\text{obs}} = +0.122$ is an estimator artifact (evaluating the convex binomial function $h_G(p)$ at the same group's plug-in $\hat{p} = k/G$ guarantees $\delta_{\text{div}} \ge 0$ for every group by construction, with expectation $+0.105$ under purely i.i.d. rollouts) and stating that *"every number below that uses it (yield corrections, iso-$G$ savings, $\delta_{\text{div}}$-based triggers or rankings) is withdrawn."* Despite this, the manuscript leaves five full pages of invalidated prose, mathematical definitions, pre-registered prediction checks, and tables (Tables 16, 18, and 19) claiming that the rollout sampler "anti-herds" and yields 20%–25% and 17.2%–68.0% rollout cost savings.

Furthermore, **Section 10.9** (*"Headline Numbers and Cross-Reference"*, p. 28), **Section 10.10** (p. 29), and **Section 10.12** (*"What the predictions license"*, p. 31) carry no retraction warnings and continue to assert that $\delta_{\text{div}} = +0.122$ and the Iso-Yield cost savings are valid, data-driven headline findings.

**Recommendation:** Excise the withdrawn analyses, equations, and tables (Sections 10.7, 10.9 bullets 1–2, 10.11, 10.12, and 10.13) from the main body of the paper (or move them to a clearly demarcated historical errata appendix if required for repository tracking), keeping only the concise mathematical correction already provided in Section 5.8.

## [CRITICAL] Withdrawn iter130 risk index analyzed as active finding in Section 10.18  (p.37 "The earlier fused-risk AUROC and threshold analysis is withdrawn. It used synthe")

The second paragraph of Section 10.18 explicitly states: *"The earlier fused-risk AUROC and threshold analysis is withdrawn. It used synthetic/imputed anchors and cannot validate calibrated monitoring, a safe threshold, compute savings, or a stopping policy."* However, the first paragraph of Section 10.18 actively analyzes the seed-stability of that exact `iter130` per-seed risk index data to argue that $k = 1$ seed suffices for a "five-fold evaluation saving" on the top-1 method claim.

**Recommendation:** Excise the `iter130`-dependent claims in Section 10.18 (and Section 4.5) from the main body of the paper.

## [CRITICAL] Mathematical inversion and mean-reward confounding in the odds-split bound for Signed ZVF  (p.34 "We therefore treat the odds-split as a conservative lower bound on ZVF− when onl")

In Section 10.15, for runs where only aggregate $(\text{ZVF}, \bar{r}, G)$ are logged, the authors impute $\text{ZVF}^-$ using the homogeneous i.i.d. odds-split formula evaluated at the aggregate mean accuracy $\bar{r}$:
$$\widehat{\text{ZVF}}^- = \text{ZVF} \cdot \frac{(1-\bar{r})^G}{(1-\bar{r})^G + \bar{r}^G}.$$
The authors claim (pp. 33–34) that this homogeneous odds-split is a *"conservative lower bound on $\text{ZVF}^-$"* across all runs and conclude that *"$\text{ZVF}^-$ for the 15 non-GSM8K runs is the odds-split lower bound, so the true $\text{ZVF}^-$ separation is at least as clean as reported,"* yielding $\text{Mann–Whitney AUC} = 1.000$. Three mathematical problems affect this conclusion:
1. **The Odds-Split Is an Upper Bound (Over-Estimator), Not a Lower Bound, When $\bar{r} < 0.5$:** Evaluating the odds ratio at the pooled mean $\bar{r}$ under-predicts $\text{ZVF}^-$ *only* when $\bar{r} > 0.5$. Conversely, when $\bar{r} < 0.5$—which holds for **every** unhealthy `plateau` and `drift` run in Table 21 ($\bar{r} \in [0.265, 0.428]$ at $G=8$)—we have $(1-\bar{r})^8 \gg \bar{r}^8$, so the factor $\frac{(1-\bar{r})^8}{(1-\bar{r})^8 + \bar{r}^8}$ is between $0.90$ and $0.999$, assigning $90\%\text{--}99.9\%$ of raw ZVF to $\text{ZVF}^-$. Under prompt-difficulty heterogeneity, if a run with $\bar{r} = 0.391$ has mastered a subset of easy prompts ($p_x \approx 1$, which actually belong to $\text{ZVF}^+$), the homogeneous formula at $\bar{r} < 0.5$ *over-estimates* $\text{ZVF}^-$. Because it under-estimates $\text{ZVF}^-$ for all healthy runs ($\bar{r} > 0.5$) and over-estimates $\text{ZVF}^-$ for all unhealthy runs ($\bar{r} < 0.5$), the imputation artificially widens the gap between the two groups.
2. **Algebraic Confounding with Mean Reward $\bar{r}$:** Because all 7 `converged` runs have $\bar{r} > 0.5$ and all 13 `unhealthy` runs have $\bar{r} < 0.5$, multiplying ZVF by the step-like sigmoid $\frac{(1-\bar{r})^G}{(1-\bar{r})^G + \bar{r}^G}$ simply gates ZVF by whether $\bar{r} < 0.5$. On these 20 runs, mean reward $\bar{r}$ alone already achieves $\text{Mann–Whitney AUC} = 1.000$.
3. **Inconsistency Between Stated Formula and Table 21 Values:** In Table 21, the values reported for `sweep G=2` ($\bar{r} = 0.977, \text{Raw ZVF} = 0.838, \text{ZVF}^- = 0.072$) and `sweep G=16` ($\bar{r} = 0.984, \text{Raw ZVF} = 0.631, \text{ZVF}^- = 0.003$) do not match the stated power-$G$ odds-split formula: at $\bar{r} = 0.977, G = 2$, $(0.023)^2 / [(0.023)^2 + (0.977)^2] = 0.00055$, which gives $\text{ZVF}^- = 0.838 \times 0.00055 = 0.00046$ (not $0.072$), and at $G = 16, \bar{r} = 0.984$, the power-16 odds factor is $\sim 10^{-29}$ (not $0.003$).

**Recommendation:** Correct the claim that the i.i.d. odds-split is a lower bound on $\text{ZVF}^-$ when $\bar{r} < 0.5$, clarify how the `sweep` rows in Table 21 were computed, acknowledge that $\widehat{\text{ZVF}}^-$ is gated by $\bar{r} \lessgtr 0.5$, and restrict empirical separability claims for Signed ZVF to runs where exact per-group reward vectors are logged.

## [CRITICAL] Corrupted bibliographic entry and mismatched arXiv identifier in Reference [18]  (p.45 "[18] Zichen Liu et al. RL fine-tuning activates a sparse subnetwork in LLMs. arX")

Reference `[18]` is a conflated citation that combines unrelated metadata:
1. The identifier `arXiv:2510.20015` resolves to an unrelated computational physics paper on intraocular lens models (*"Comparative Analysis of Mechanical Stability and Biomarkers of Commercial and Modified Intraocular Lens (IOL) Models: A Numerical and Experimental Approach"*).
2. In Section 1.1 (page 2), `[18]` is cited alongside CPPO `[17]` and GIFT `[23]` to support the statement: *"Variants of GRPO adjust the group baseline or advantage normalization to preserve signal [17, 23, 18]."* Furthermore, the first author listed (`Zichen Liu`) is the lead author of Dr. GRPO (*"Understanding R1-Zero-Like Training: A Critical Perspective"*, `arXiv:2503.20783`), which removes GRPO's within-group standard-deviation normalization and length bias and is referenced elsewhere in the manuscript (e.g., Sections 4.1, 4.4, and Table 21) without a corresponding bibliography entry. By contrast, a title concerning sparse subnetworks during RL fine-tuning does not describe a group-baseline or advantage-normalization variant of GRPO.

Replacing or correcting `[18]` so that its author list, title, and arXiv identifier accurately correspond to the intended work is necessary for citation validity.

**Suggested edit:**

```
[18] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. arXiv preprint, 2025. doi: 10.48550/arXiv.2503.20783. arXiv:2503.20783.
```

## [MAJOR] Empty Subsections 2.1 and 2.4 omit essential task and verifier specifications  (p.3 "We standardize hyperparameters across libraries using a common configuration sch")

In Section 2, Subsection 2.1 (*Task Suite*, page 2) and Subsection 2.4 (*Reward Verification*, page 3) appear solely as empty section headings with no body text. Furthermore, Subsection 2.3 (*Hyperparameter Mapping*, Table 3, page 4) maps "Tinker PPO defaults" across six libraries without specifying the GRPO-specific parameters central to the paper (such as group size $G$, sampling temperature, or whether GRPO uses the same clip range $\epsilon = 0.2$ and target KL $= 0.01$) and omits `d3rlpy` (CQL), which is listed in Table 2 (page 3).

Because the Zero-Variance Fraction (ZVF) depends directly on how verifiable rewards are parsed and thresholded, leaving Subsections 2.1 and 2.4 unpopulated omits essential experimental specifications:

- **Synthetic Arithmetic Task Formulation:** Subsection 2.5 (page 3) states that uniform random action selection yields $\frac{1}{199} \approx 0.50\%$ accuracy on the arithmetic task and compares this to a human baseline on "two-digit addition under timed conditions" (while Appendix A.4 later mentions "two-operand addition"). Although a $\frac{1}{199}$ random baseline is mathematically consistent with selecting uniformly over the integer sums $[0, 198]$ produced by two-digit non-negative addition ($0+0$ to $99+99$), the exact operand ranges, dataset generation rules, prompt templates, and how an autoregressive LLM (such as Qwen2.5-0.5B in TRL) versus a discrete-action policy network (in SB3 or CleanRL) is constrained to a 199-way action space are never specified.

- **Deterministic Verifier Rules:** Without Subsection 2.4, there is no specification of the answer-extraction regexes (e.g., `####` or `\boxed{}`) or symbolic/floating-point equivalence rules for GSM8K, the unit-test execution sandbox and timeouts for HumanEval, or how partial versus binary rewards are assigned in Function Calling.

- **Task Completion Status Discrepancy:** Figure 1 (page 3) labels HumanEval as `"1 model ◦ Partial"` (where `◦` denotes `Planned` in the legend) and Function Calling as `"2 models tested ∼ Partial"` (`∼` denotes `In progress`), whereas Section 3.4 (page 7) states that the Modal H100 cluster experiments include *"full HumanEval benchmark evaluation"*. Populating Subsections 2.1 and 2.4 and reconciling the task completion status between Figure 1 and Section 3.4 would resolve these reproducibility gaps.

**Suggested edit:**

```
Populate Subsection 2.1 with the exact synthetic arithmetic task formulation (operand ranges, prompt templates, and action-space constraints for LLMs vs. classic RL policies) and reconcile the HumanEval completion status between Figure 1 and Section 3.4. Populate Subsection 2.4 with the deterministic answer-extraction and verification rules for GSM8K, HumanEval, and Function Calling, and include GRPO-specific hyperparameters in Subsection 2.3.
```

## [MAJOR] Structural disconnect between Section 2 benchmark apparatus and Section 4+ ZVF evaluation  (p.2 "The cross-RL-library benchmark below (Table 2) covers TRL, SB3, CleanRL, Tiansho")

Section 2.2 (pages 2–3) highlights two different seven-library rosters: (1) a cross-RL-library benchmark in Tables 2–3 (`TRL`, `Stable Baselines3`, `CleanRL`, `Tianshou`, `PufferLib`, `rl_games`, and `d3rlpy`) used as the algorithm-and-stack ablation underlying claim "$F_3$", and (2) an LLM post-training launcher roster (`Tinker`, `TRL`, `SkyRL`, `verl`, `OpenRLHF`, `Atropos`, and `HF reference launchers`), of which only `Tinker` and `TRL` produced completed runs in this release while `veRL` and `OpenRLHF` are noted as dry-run placeholders. Meanwhile, Table 1 (page 3) lists six task-method pairs including `Chat SFT` (NoRobots), `Preference (Shorter)` (DPO), and `Distillation (Off-Policy / On-Policy)` (OpenThoughts3 / IS Loss).

There is a methodological misalignment between the benchmark apparatus introduced in Section 2 and the actual ZVF analysis in the rest of the paper:

- **Applicability to Non-GRPO Tasks and Classic/Offline RL Libraries:** As Appendix A.5 (page 41) explicitly acknowledges, ZVF is ill-defined for non-outcome-reward methods (such as SFT, DPO, Off-/On-Policy Distillation in Table 1, or offline Conservative Q-Learning in `d3rlpy` in Table 2) where there are no online rollout groups. Furthermore, five of the seven libraries in Table 2 (`Stable Baselines3`, `CleanRL`, `Tianshou`, `PufferLib`, `rl_games`) are classic step-based MDP frameworks implementing actor-critic PPO with Generalized Advantage Estimation ($\text{GAE } \lambda = 0.95$, $\gamma = 0.99$ in Table 3) rather than critic-free group-relative rollout centering ($G$ completions per prompt). Section 2 does not explain what model architecture or environment wrapper is used to run these classic RL libraries on the arithmetic task or how they connect to GRPO signal starvation.

- **Omission of Evaluated Variance-Mitigation Methods:** Conversely, the nine variance-mitigation methods that form the core "cross-library" ZVF analyses from Section 4 onward (`GRPO`, `AERO`, `CPPO`, `NGRPO`, `SCAFGRPO`, `MCGRPO`, `GIFT`, `AREAL`, `ES`; e.g., Table 6 on page 8) are not introduced in Section 2 at all, and the Abstract (page 1) notes that the variance-mitigation table is a "declared simulation projection used for sensitivity analysis."

- **Undefined Cross-References (`F_3`):** Sections 2.2, 3.2, and 3.3 (pages 2, 3, 5, and 6) repeatedly reference finding "$F_3$" as well as an external *"statistical-rigor addendum"* and *"framework-configs appendix"* without defining what hypothesis "$F_3$" represents in Sections 1–3.

**Suggested edit:**

```
Clarify in Section 2 that Tables 1–3 describe the broader multi-pillar TINKERRL-BENCH suite, explicitly define hypothesis F3 upon first mention, explain how classic/offline RL libraries connect to the arithmetic benchmark, and introduce the nine variancemitigation methods and their simulation protocol in Section 2 or Section 3.
```

## [MAJOR] Table 5 includes mathematically undefined single-seed tests; planned tests unenumerated  (p.6 "Table 5: Legacy 20-test BH table – audit trail only. All hypothesis tests report")

Section 3.3 includes *Table 5: Legacy 20-test BH table – audit trail only*, while explicitly noting in the caption and text that the table *"pre-dates the Tier-A/B/C partition and silently includes single-seed Tinker rows whose Welch statistics are mathematically undefined ($n_1=n_2=1$)"* and that under the restricted Tier-A/B family computed in the addendum, *"only $F_3$ survives."* In addition, on page 5, Section 3.3 states that to address multiple comparisons *"across 5 planned tests, we apply both the conservative Bonferroni correction (family-wise error rate: $\alpha' = \alpha/k = 0.0100$) and the Benjamini–Hochberg procedure."*

While transparency regarding earlier revisions is constructive, retaining a full 20-row table of "significant" $p$-values in the main text that the paper simultaneously identifies as mathematically invalid or withdrawn raises several validity and clarity issues:

- **Invalid Test Statistics and Pseudo-Replication:** For a two-sample Welch $t$-test with $n_1=n_2=1$, sample variances and degrees of freedom ($\nu = 0$) are strictly undefined (`NaN`), yet Table 5 reports extreme floating-point $p$-values (e.g., Rank 1 `Dense vs MoE Architecture`, $\text{Raw } p = 2.357\times 10^{-21}$, $\text{BH-adj. } p = 0$; Rank 2 `PPO vs GRPO on Llama-3.1-8B`, $\text{Raw } p = 3.924\times 10^{-10}$). Similarly, a two-sample Mann–Whitney $U$ test on $n_1=n_2=5$ seeds has a minimum possible two-tailed exact $p$-value of $\frac{2}{\binom{10}{5}} = \frac{2}{252} \approx 0.00794$, making the reported Mann–Whitney $p = 3.29\times 10^{-5}$ in Rank 6 impossible for a 5-seed run-level comparison unless autocorrelated per-step trajectory rows were pooled as independent observations (which Sections 4 and 5.7 explicitly warn against due to lag-1 autocorrelation $\rho \approx 0.9$). Moreover, Ranks 8 and 9 report highly significant correlations between ZVF and Final Performance ($p = 0.00054$ and $p = 0.00081$), contradicting the Abstract, Section 4, and Section 5.4, which withdraw the ZVF–outcome correlation.

- **Missing Authoritative Test Table and Bonferroni Power:** The 5 planned Tier-A/B hypothesis tests referenced on page 5 are never enumerated in Section 3.3, nor are their test statistics, effect sizes (Cohen's $d$), or restricted-family adjusted $p$-values shown in the main text. Additionally, the power calculations in Table 4 (page 6) are evaluated at the unadjusted $\alpha = 0.05$ ($d_{\min} = 2.024$ for $n_1=n_2=5$; $d_{\min} = 1.530$ for $n_1=5, n_2=15$); under the Bonferroni-adjusted $\alpha' = 0.0100$, the minimum detectable effect size at 80% power for $n_1=n_2=5$ rises to $d_{\min} \approx 2.68$.

- **Undefined Symbols and Terms in Table 5:** Section 3.3 (page 5) defines `*` for Bonferroni survival and `†` for BH–FDR survival, but neither symbol appears in Table 5; instead, Ranks 2, 6, 8, and 9 are marked with an undefined double dagger (`‡`), and Ranks 11, 12, and 19 reference `"Last-10"` before it is defined in Section 5.1.

**Suggested edit:**

```
Replace the deprecated Table 5 in the main text with the authoritative restricted 5-test Tier-A/B hypothesis table (moving Table 5 to a changelog appendix), report the Bonferroni-adjusted MDE in Table 4, and define any footnote symbols (`‡`) and metrics (`Last-10`) upon first appearance.
```

## [MAJOR] Confounded marginal one-way ANOVAs in variance decomposition sum to 1.897  (p.7 "One-way ANOVAs across 42 experiments with valid performance observations show th")

The text reports four separate one-way ANOVAs across 42 experiments with valid performance observations: library/framework ($\eta^2 = 0.546$), training algorithm ($\eta^2 = 0.558$), model family ($\eta^2 = 0.471$), and model scale ($\eta^2 = 0.322$), concluding that library/framework and training algorithm are the dominant variance sources.

These four marginal $\eta^2$ values sum to $1.897 > 1.0$ because the experimental design across libraries, algorithms, model families, and model scales is unbalanced and non-orthogonal (collinear)—for example, classic RL libraries exclusively run PPO/CQL on synthetic arithmetic, whereas LLM launchers run GRPO on GSM8K/tool-use across specific model families and scales. Inferring that library/framework and training algorithm are the dominant variance sources from separate marginal one-way ANOVAs without accounting for factor confounding is potentially misleading, as each marginal $\eta^2$ absorbs shared variance from the other collinear factors. Furthermore, it is unclear whether the "42 experiments" pool multi-seed Tier-A runs (treating individual seeds as independent rows) alongside single-seed Tier-C runs.

**Suggested edit:**

```
Clarify the exact composition of the 42 observations (whether individual seeds are pooled with singleseed runs) and either use a multi-way variance decomposition (e.g., Type II/III partial $\eta^2$) or explicitly caveat that the one-way $\eta^2$ values reflect overlapping, confounded marginal variance across collinear factors.
```

## [MAJOR] Retention of withdrawn Fused-Risk Index analysis in Section 4.5  (p.9 "We elevate ZVF from a single-axis sanity check to a multi-channel risk index by ")

Section 4.5 (*"Iter 130: Unified ZVF Risk Index"*, pp. 9–11) presents an affirmative evaluation of a three-channel composite risk index (`zvf_risk` and `zvf_risk_max`), reporting a cross-experiment Mann–Whitney AUROC of $0.929$ $[0.83, 1.00]$, a within-methods AUROC of $0.805$ $[0.62, 0.95]$, and specific practitioner intervention thresholds (`zvf_risk_max > 0.55` as failure-positive, `< 0.30` as safe). However, later in the manuscript (Section 10.18, p. 37), this analysis is explicitly retracted: *"The earlier fused-risk AUROC and threshold analysis is withdrawn. It used synthetic/imputed anchors and cannot validate calibrated monitoring, a safe threshold, compute savings, or a stopping policy."*

Leaving Section 4.5 intact as an unflagged headline contribution on pages 9–11 while withdrawing it on page 37 creates a direct internal contradiction. Furthermore, in the second table on page 10 (*"Method-level ranking (max-fusion)"*), all five scaling-law anchors (`Qwen3-8B`, `Nemotron-120B`, `Llama-3.1-8B`, `Qwen3.5-4B`, and `DeepSeek-V3.1`) are listed with an identical `zvf_risk_max mean = 0.69 (anchor)` and `Failure rate = 1.0`, contradicting Section 5.3 (p. 12) and Table 11 (p. 19), which state that `Nemotron-120B` is the *only* scaling-law run that collapses and the other four do not.

**Recommendation:** Either remove Section 4.5 or add a prominent retraction notice at the beginning of Section 4.5 on page 9 (consistent with how withdrawn analyses are handled in Sections 5.4, 5.8, 10.7, and 10.11) so that readers do not rely on a diagnostic index and threshold policy that the manuscript later withdraws.

## [MAJOR] Non-mutually exclusive failure taxonomy equations and contradiction on tool-use collapse cells  (p.11 "collapse ≡ p > 0.7 ∧ ℓ < 0.35 (1)
drift ≡ ℓ < 0.85 p ∧ ¬collapse (2)
plateau ≡ p")

Section 5.1 states that every experiment cell is classified into *"one of four mutually exclusive buckets using only two public thresholds on the heldout-accuracy trajectory"* via Equations (1)–(4). There are two mathematical and classification inconsistencies in this formalization:

1. **Violation of Mutual Exclusivity:** As written using logical equivalence ($\equiv$) rather than an ordered `if / else if` cascade, Equations (1)–(4) are not mutually exclusive. Any trajectory with peak accuracy $p < 0.5$ simultaneously satisfies `plateau` (Eq. 3) and either `drift` (Eq. 2, when $\ell < 0.85\,p$, since $p < 0.5 \implies \neg\text{collapse}$) or `converged` (Eq. 4, when $\ell \ge 0.85\,p$). For instance, the MCGRPO, GIFT, AReaL, and ES runs in Section 5.3 (p. 12) have $p \approx 0.40$ and $\ell < 0.03$, simultaneously satisfying both Eq. (2) (`drift`) and Eq. (3) (`plateau`). Similarly, the 42 `plateau` rows in Table 7 (p. 12) have median $p = 0.4040$ and median $\ell = 0.3972$ ($0.3972 \ge 0.85 \times 0.4040 = 0.3434$), satisfying both Eq. (3) (`plateau`) and Eq. (4) (`converged`). Conversely, if Eq. (3) (`plateau`) were evaluated before Eq. (2) (`drift`) in a sequential cascade, no run with $p \approx 0.40 < 0.5$ could be assigned to `drift`, contradicting Section 5.3. Furthermore, under Eq. (3), runs that plateau at an intermediate peak accuracy $0.5 \le p \le 0.7$ with $\ell \ge 0.85\,p$ are classified as `converged` rather than `plateau`.
2. **Contradiction on Tool-Use Collapse Cells:** Section 5.3 (*"Tool-use collapse"*, p. 11) and Table 7 (p. 12) state that the two cross-tool trajectories (`cross_tool_qwen3-32b` and `cross_tool_llama-8b-inst`) have $\text{peak} = p = 0.0$ and $\text{last10} = \ell = 0.0$, and assert: *"Under the taxonomy they fall in `collapse` ($\text{peak} = 0, \text{last10} = 0$)."* However, Eq. (1) explicitly requires $p > 0.7$ for `collapse`. A trajectory with $p = 0.0$ mathematically violates $p > 0.7$ and instead satisfies Eqs. (3) and (4). It only falls into `collapse` under the alternative rules stated later in Sections 10.8 and 10.10 (pp. 28–29: *"collapse if $\bar{r}_{10} < 0.05$ or $\overline{\text{ZVF}} \ge 0.95$"*). Because these two cells account for 2 of the 3 `collapse` rows in Table 7 and drive the reported Spearman $\rho = 0.56$ / point-biserial $0.62$ association with collapse in Section 4 (p. 7), any reliance on $\overline{\text{ZVF}} \ge 0.95$ to assign the label would also conflict with the claim in Section 5.7 (p. 14) that *"The collapse label is computed only from peak vs. last-10, never from ZVF."*

**Recommendation:** Explicitly write out the exact sequential evaluation order (`if / elif / else`) and threshold conditions implemented in `scripts/zvf_diagnostic.py` so that Equations (1)–(4) are mutually exclusive, and reconcile Eq. (1) with how the $p=0, \ell=0$ tool-use cells and $p \approx 0.40$ `drift` cells are classified.

## [MAJOR] Mathematical discrepancy between text (+1.0) and Table 12 (+0.791) for Spearman rank correlation  (p.19 "Reading the table. The two ZVF-proxy columns (frac_below_0p1 and zero_fraction) ")

The text states that $\rho_{\text{Spearman}}(\text{frac\_below\_0p1}, \text{is\_collapse}) = +1.0$ by construction ($n = 5$, one collapse), and that the same holds for $\rho(\text{zero\_fraction}, \text{is\_collapse})$. However, Table 12 directly above this paragraph reports `rho= +0.791` for both `T1: Spearman(frac_below_0p1, is_collapse)` and `T2: Spearman(zero_fraction, is_collapse)`.

Mathematically, Table 12 is right and the text claim of $+1.0$ is incorrect: across the five models in Table 11, `frac_below_0p1` takes values $(0.0000, 0.1333, 0.0000, 0.0000, 0.5500)$, `zero_fraction` takes $(0.0000, 0.0667, 0.0000, 0.0000, 0.5500)$, and `is_collapse` takes $(0, 0, 0, 0, 1)$. Because three of the four non-collapse rows are tied at $0.0000$ while one non-collapse row (`Qwen3-8B`) has a strictly positive value ($0.1333$ and $0.0667$), the rank vector for each ZVF proxy is $(2, 4, 2, 2, 5)$ while the rank vector for `is_collapse` is $(2.5, 2.5, 2.5, 2.5, 5)$. Under standard tie-adjusted Spearman rank correlation, this yields $\rho = \frac{2.5}{\sqrt{6.0 \times 1.6}} = \sqrt{\frac{5}{8}} \approx +0.7906$, matching $+0.791$ in Table 12 rather than $+1.0$. Also note that on `Qwen3-8B`, `frac_below_0p1` is $0.1333$ (not $0.07$).

**Recommendation:** Update the text bridging pages 19 and 20 to state $\rho_{\text{Spearman}} = +0.791$ (matching Table 12) for both `frac_below_0p1` and `zero_fraction`, and note that it is below $+1.0$ due to rank ties among the non-collapsing models and the intermediate non-zero value on `Qwen3-8B`.

**Suggested edit:**

```
The two ZVF-proxy columns (frac_below_0p1 and zero_fraction) take their maximum value (0.55) on the Nemotron-120B row and are 0 (or 0.1333 and 0.0667 on Qwen3-8B) on every noncollapse row. ρSpearman(frac_below_0p1, is_collapse) = +0.791 (due to ties among the noncollapse rows; n = 5, one collapse), and the same holds for ρ(zero_fraction, is_collapse).
```

## [MAJOR] Inconsistent trajectory lengths (100 steps vs. 300 steps) for the variance-mitigation corpus  (p.16 "For GRPO the early-phase ZVF (phase 1 mean) is 0.044, mid-phase is 0.529, and la")

Across Sections 4 through 10, the same variance-mitigation dataset (`variance_mitigation.tsv` / `zvf_by_library.tsv`) is described with conflicting trajectory lengths:

- Table 6 (p. 8) and Section 8.1 (p. 20) state that the variance-mitigation runs consist of **100 rollout steps** per seed (and Section 6.3, p. 16, refers to a *"step-74+ heldout-accuracy collapse flag"*).

- However, Section 6.2(i) (p. 16) states that for GRPO in the same corpus, early/mid/late phase ZVF exhibits *"a $20\times$ upward drift within a **300-step run**,"* and Figure 7 (p. 18) plots `first collapse-flag step` and `first step at which ZVF crosses \theta` occurring between steps **215 and 250**.

**Recommendation:** Clarify why the step index reaches $250\text{--}300$ in Section 6 if the trajectories are $100$ steps long.

## [MAJOR] Conflicting collapse and drift tallies and max-ZVF definition across Tables 6, 8, 9, and Sections 5–10  (p.13 "Table 8: Per-library ZVF aggregation across the variancemitigation corpus (GRPO ")

Both Table 6 (p. 8) and Table 8 (p. 13) cite `experiments/results/zvf_by_library.tsv` as their source, but report conflicting failure tallies and summary metrics across the paper:

1. **Collapse Tallies (`0/5` vs. `3/5` vs. `3 collapsed / 1 surviving`):** Table 6 reports `Collapse: 0/5` for vanilla GRPO (and Section 10.10, pp. 29–30, and Table 21, p. 34, likewise state that **0 out of 45** variance-mitigation runs collapsed because their `last10_avg` hovers at $0.39$, labeling all 45 exclusively as `plateau`). In contrast, Table 8 (p. 13), Sections 8.1–8.3 (pp. 20–23, Tables 13–14), and Section 9.5 (p. 25) report `collapse-seeds: 3/5` (`collapse rate: 0.600`) and 2 surviving seeds, while Section 10.14 (p. 33, "Honest scope") states that the exact same GRPO seed pool in `variance_mitigation.tsv` *"contains 3 of 5 seeds flagged as collapsing and 1 of 5 surviving; the comparison set is degenerate at $n=1$"* (leaving the 5th seed unaccounted for). While Section 5.5 explains that `3/5` counts a mid-run per-step collapse flag (`heldout_acc` hits 0 mid-trajectory) whereas all 5 seeds end in `plateau` under the trajectory-level peak-vs-last10 taxonomy, switching between these definitions without distinction across Sections 6–10 leads Section 9 to forecast a "collapse" event on trajectories that Section 10.10 states never collapsed in their late-trace outcome (`last10_avg` $\approx 0.379$, matching AERO's $0.399$).
2. **Drift Tallies (`1/5` vs. all 4 methods):** Table 6 (p. 8) reports `Drift: 1/5` and `Plateau: 4/5` for MCGRPO, GIFT, AREAL, and ES (with 5-seed mean last-10 accuracy $\text{mean-}\ell \in [0.316, 0.328]$), whereas Section 5.3 (p. 12) states that *"All four methods in `experiments/results/variance_mitigation.tsv` reach $\ell < 0.03$ from a peak $p \approx 0.40$, so they are `drift`"* (which appears to describe only 1 of the 5 seeds per method).
3. **`max-ZVF` Definition:** Table 8 reports `max-ZVF = 0.482` for GRPO, whereas Table 9 (p. 17) reports that GRPO's late-phase mean ZVF is `0.870`. The caption of Table 8 should clarify that `max-ZVF` is the maximum of the 5 per-seed trajectory means rather than the within-trajectory maximum.

## [MAJOR] Contradictory claims on pre-collapse ZVF flatline vs. sharp jump vs. 10-step slope  (p.22 "GRPO (∆ = −0.004, p = 0.893): ZVF is statistically flat across the peak-to-termi")

Because Sections 8 through 10 are compiled by concatenating sequential research iterations (`iter22` through `iter62`) without harmonizing definitions, several subsections make mutually incompatible factual claims about the exact same underlying GRPO trajectories:

- **Section 8.2 (p. 22, Table 14):** A Wilcoxon test comparing `pre-5` ($0.883$) against `post-5` ($0.879$) on GRPO finds $\Delta = -0.004$ ($p = 0.893$), concluding: *"ZVF is statistically flat across the peak-to-terminal window. The collapse is therefore not preceded by a ZVF spike; it follows a stable high-ZVF regime (pre $\approx 0.88$). ... The lead-time extension rules out a simple 'ZVF rises, then collapse' story for GRPO."* (Note also that because Table 14 reports `mean lead` $= \text{first\_collapse\_step} - \text{first\_pass\_step} = 1.0\text{ step}$ for GRPO, the 5-step `pre_zvf_5` window $[\text{first\_pass\_step}, \text{first\_pass\_step} + 5)$ and the 5-step `post_zvf_5` window $(\text{first\_collapse\_step} - 5, \text{first\_collapse\_step}]$ overlap and invert chronological order, with `pre_zvf_5` extending 4 steps *later* than `post_zvf_5`.)

- **Section 9.2 (p. 24, Item 2 and Table 15 caption):** Directly contradicting Section 8.2, Section 9.2 states: *"Decomposing the GRPO-collapse trajectory (seed 0) into its pre- and post-collapse ZVF windows confirms the picture: ZVF is approximately flat early on and jumps sharply at the collapse step,"* and asserts that *"a sharp jump in ZVF IS the collapse signal, not a slow drift."*

- **Section 10.14 (p. 32):** Introduces a third conflicting claim on the same trajectories, asserting that *"The corollary for diagnostic deployment is that the correct detection statistic is $\Delta\text{ZVF}/\Delta t$ smoothed over a 10-step window, not the contemporaneous $\text{ZVF}_t$ alone"*—which directly contradicts Section 9.2's headline finding that the 10-step slope (`trailing_slope_10`) is *"the wrong feature"* and performs at or below chance ($\text{ROC-AUC} = 0.487$ at $K=5$ and $0.267$ at $K=10$ in Table 15).

**Recommendation:** Reconcile the opposing descriptions of GRPO's pre-collapse ZVF trajectory and slope utility across Sections 8.2, 9.2, and 10.14, and fix the overlapping/inverted window definitions when `lead_steps = 1`.

## [MAJOR] Conflicting ground-truth phase labels for Nemotron-120B and arithmetic error in zvf_discriminator  (p.26 "Collapse: 1/1 perfect. The lone collapse anchor (Qwen3.5-27B) is correctly class")

There are conflicting ground-truth phase labels for the frontier anchors and an arithmetic discrepancy in `zvf_discriminator`:
1. **Conflicting Ground-Truth Labels (Section 7 vs. Sections 10.1–10.4):** On p. 20 (concluding Section 7, matching Table 11 on p. 19 and Section 5.3 on p. 12), `Nemotron-120B` is the *only* scaling-law anchor classified as `collapse` ($\text{peak } 0.875 \to \text{late } 0.154$, `is_collapse = yes`). In Sections 10.2–10.4 (pp. 25–26), `Qwen3.5-27B` is stated to be *"the lone collapse anchor (1/1 perfect)"*, whereas `Nemotron-120B` is assigned a ground-truth label of *non-collapse* `plateau` (causing `Nemotron-120B` with `zvf_level = 0.55`, `zvf_direction = -0.81`, and `zvf_discriminator = 0.516` to exceed the `collapse` class values of $0.333, -0.281$, and $0.333$ on every metric in Section 10.4, contradicting the claim that collapse is uniquely separated by high ZVF level and negative reward direction).
2. **Discriminator Arithmetic:** Using the formula defined in Section 10.1 (`zvf_discriminator` $\equiv \text{zvf\_level} \times \max(0, 1 - \text{zvf\_direction})$), for `Qwen3.5-27B` with `zvf_level = 0.333` and `zvf_direction = -0.281`, the product is $0.333 \times (1 - (-0.281)) = 0.333 \times 1.281 = 0.427$, not $0.333$ as reported in Sections 10.2 and 10.4.

**Recommendation:** Harmonize ground-truth phase labels for `Nemotron-120B` across Sections 7 and 10.1–10.4, and verify the `zvf_discriminator` calculation.

## [MAJOR] Degenerate percentile bootstrap CIs on 0/5 binary collapse counts in Table 13  (p.21 "Table 13: Per-library iterated bootstrap (B=2000, seed-resampled). GRPO is the o")

Resampling $n=5$ observations where 0 of 5 seeds collapse produces a degenerate percentile bootstrap 95% CI of `[0.000, 0.000]` because all resamples draw only zeros. By contrast, an exact binomial (Clopper–Pearson) 95% confidence interval for $0/5$ events is `[0.000, 0.522]`, which overlaps substantially with GRPO's observed $3/5$ (`0.600`, exact 95% CI `[0.147, 0.947]`). Claiming in Section 8.1 (p. 20) that *"the collapse-rate CI for GRPO [0.400, 0.800] is entirely disjoint from the 0.000 collapse-rate CI of every mitigation library"* is an artifact of applying a non-parametric percentile bootstrap to $0/5$ binary outcomes at $n=5$.

**Recommendation:** Use exact Clopper–Pearson intervals for $0/5$ and $3/5$ binary counts in Table 13 and update the text accordingly.

## [MAJOR] Asymptotic normal approximation on n=5 Wilcoxon test understates p-value below exact 0.0625 floor  (p.22 "AREAL (∆ = −0.099, p = 0.043): ZVF DECREASES significantly over the terminal win")

Section 8.2 tests $\mathbb{E}[\text{post} - \text{pre}] = 0$ using a *"normal-approximation Wilcoxon signed-rank test ($n=5$ per library)"* and highlights `AREAL` as statistically significant at $p = 0.043 < 0.05$. At $n=5$, the discrete Wilcoxon signed-rank statistic $W^+$ has only $2^5 = 32$ possible sign configurations; the minimum possible two-sided exact $p$-value (when all 5 differences share the same sign) is $2 / 32 = 0.0625 > 0.05$. Reporting $p = 0.043$ is an artifact of applying an asymptotic continuous normal approximation ($Z \approx 2.023$) without continuity correction at $n=5$, and would not cross $\alpha = 0.05$ under an exact test or survive multiple-testing correction across the 9 libraries.

**Recommendation:** Report exact Wilcoxon signed-rank $p$-values (noting the $p = 0.0625$ floor at $n=5$) and temper the statistical significance claim for `AREAL`.

## [MAJOR] Between-method confounding, uncensored post-collapse windows, and collinearity in Table 15 forecasting CV  (p.24 "Table 15: Leave-one-(method, seed)-trajectory-out CV for each ZVF-only feature a")

Three methodological issues affect the forecasting cross-validation setup in Section 9 and Table 15:
1. **Between-Method Confounding:** The 4,500 step-rows pool all 9 libraries ($45 \text{ trajectories} \times 100 \text{ steps}$), where only 3 trajectories (all belonging to vanilla GRPO) ever set the collapse flag. Because vanilla GRPO has a persistently higher ZVF level ($0.481$, rising to $0.88$) than the 8 mitigation methods ($0.114\text{--}0.317$, none of which ever collapse), a univariate classifier trained on ZVF level across all 45 trajectories primarily separates *vanilla GRPO method identity* from the other 8 methods rather than forecasting *when* a trajectory is about to collapse within a method.
2. **Post-Collapse Window Inclusion and Base-Rate Discrepancy:** The future-collapse label $y_t(K) = \mathbb{1}[\exists s \in (t, t+K] : c_s = 1]$ does not censor out steps $t \ge t_{\text{collapse}}$ where the run has already collapsed ($c_t = 1$ and $\text{ZVF}_t$ is already saturated). Furthermore, at $K=25$, Table 15 reports a positive `base rate` of $0.264$ across $n = 4500$ steps ($1,188$ positive step-rows), which is mathematically impossible if only 3 trajectories of 100 steps ($300$ total steps across those 3 runs) ever set $c_s = 1$. Likewise, the calibration claim in Section 9.3 is computed over strongly autocorrelated ($\rho_1 \approx 0.94$) overlapping 25-step windows rather than independent pre-collapse events.
3. **Collinearity of $f_2$ and $f_3$:** On a discrete grid with unit step spacing $\Delta t = 1$, the 25-step trapezoidal area $f_3(t) = \text{AUC}(\text{zvf}_{[t-24,t]})$ differs from the 25-step arithmetic mean $f_2(t) = \overline{\text{zvf}}_{[t-24,t]}$ only by half the weight on the two boundary endpoints, making them nearly identical features.

**Recommendation:** Censor post-collapse steps ($t \ge t_{\text{collapse}}$), resolve the $0.264$ base-rate discrepancy, and evaluate forecasting ROC-AUC *within* GRPO.

## [MAJOR] Singleton class test-set leakage in LOO CV and reversed permutation p-value logic in Section 10.2–10.3  (p.26 "No shuffled-feature baseline exceeds the base accuracy at the p < 0.05 level. Th")

Two statistical issues affect the 12-anchor classifier analysis in Sections 10.2 and 10.3:
1. **Singleton Class in LOO CV (Section 10.2):** Section 10.2 claims **"Collapse: 1/1 perfect. The lone collapse anchor (Qwen3.5-27B) is correctly classified on every LOO fold"** using a 4-class nearest-centroid classifier. In strict Leave-One-Out (LOO) cross-validation, when the lone `collapse` sample ($n=1$) is held out as the test point, there are *zero* `collapse` samples remaining in the training fold to form a `collapse` class centroid—making it impossible to classify the held-out point as `collapse` unless the class centroids were computed on the full dataset prior to LOO splitting (direct test-set leakage).
2. **Misinterpreted Permutation $p$-value (Section 10.3):** Section 10.3 states: *"No shuffled-feature baseline exceeds the base accuracy at the $p < 0.05$ level. The smallest $p_{\text{shuffled} \ge \text{base}}$ is $0.195$ for `zvf_direction`, so the $0.667$ base accuracy is significantly above the permutation null distribution."* This reverses standard hypothesis-testing logic: $p_{\text{shuffled} \ge \text{base}} = 0.195 > 0.05$ means that $19.5\%$ of random feature permutations achieved an accuracy $\ge 0.667$, which *fails* to reject the null hypothesis at $\alpha = 0.05$ (and against a majority-class baseline of $5/12 = 0.417$, $8/12$ is not statistically significant).

**Recommendation:** Recompute centroids strictly within each training fold during LOO CV (noting that a singleton class cannot be recovered under strict LOO), and correct the interpretation of the permutation $p$-value ($p = 0.195$ is non-significant).

## [MAJOR] Deterministic label circularity in Section 10.8 3-NN failure-mode classifier  (p.28 "Truth labels. The truth label is deterministic on the per-library summary: colla")

In Section 10.8, the ground-truth labels for the $n=14$ pooled rows are deterministically defined as threshold rules on `mean_zvf`, `mean_last10`, and `mean_peak` (`collapse` if $\bar{r}_{10} < 0.05$ or $\overline{\text{ZVF}} \ge 0.95$; `plateau` if $\bar{r}_{10} < 0.6$ and $\overline{\text{ZVF}} \ge 0.10$; `drift` if $\text{peak} - \bar{r}_{10} > 0.10$ and $\overline{\text{ZVF}} \in [0.03, 0.30]$). Evaluating a 3-NN classifier on those exact same three variables and concluding that `mean_zvf` is non-redundant because dropping it lowers accuracy from $11/14$ to $9/14$ is circular when $\overline{\text{ZVF}}$ was explicitly used to define the truth labels (unlike the ZVF-free taxonomy in Section 5.1). Moreover, in Table 17, the 3-NN classifier achieves 0% recall on `collapse` (0/2) and 0% on `drift` (0/0), predicting `plateau` for 12 of the 14 rows.

**Recommendation:** Use the ZVF-free taxonomy from Section 5.1 for the 3-NN evaluation in Section 10.8 so the ground-truth labels do not depend on `mean_zvf`.

## [MAJOR] Potential full-trace feature leakage and task confounding in Section 10.10 early-trace analysis  (p.29 "The analysis pools the 60 (kind, method, seed) runs in experiments/results/zvf_d")

In Section 10.10, the text asks whether failure modes can be predicted from the *first half* of the trajectory, but defines `burden_05 := auc_above_zvf05` and `burden_07 := auc_above_zvf07` by reading the run-level summary fields from `zvf_dynamics.json`, which were defined in Section 6.1 (Eq. 7, p. 16) over the *full* trajectory $i = 1, \dots, T$. Unless those integrals were explicitly recomputed with an upper limit of $T/2$, they contain second-half trajectory information. Furthermore, the 60-run pool compares 45 `plateau` runs (100-step synthetic variance-mitigation trajectories) against 15 `converged` runs (12 40-step two-digit addition sweeps and 3 200-prompt inference-only GSM8K evaluations where the index is prompt ID rather than training step, cf. Table 9), confounding class labels with experimental task and trace structure.

**Recommendation:** Verify and explicitly state whether `auc_above_zvf05` and `auc_above_zvf07` are truncated at $T/2$, and control for task/dataset identity when evaluating plateau vs. converged separability.

## [MAJOR] Secular trend confounding in raw level lagged cross-correlations (Table 20)  (p.33 "Table 20: Lag profile of Pearson r(rewardt, ZVFt+L) across the nine variancemiti")

Because both mean reward and ZVF drift monotonically upward over training steps (cf. Section 6.2), computing raw level cross-correlations $r(\text{reward}_t, \text{ZVF}_{t+L})$ without detrending or first-differencing reflects shared secular trends and boundary window truncation at lag $+L$ rather than demonstrating that step-to-step changes in reward causally lead changes in ZVF (*"change in reward predicts change in ZVF five to ten steps later"*).

**Recommendation:** Compute lagged cross-correlations on first-differenced ($\Delta \text{reward}_t, \Delta \text{ZVF}_{t+L}$) or detrended series to isolate lead–lag dynamics from shared monotonic drift.

## [MAJOR] Confounding of training time/competence with intrinsic prompt difficulty in Section 10.16  (p.35 "Stratification by heldoutaccuracy quintile. We bin every per-step observation of")

In Section 10.16, the authors ask whether variance-mitigation libraries lower ZVF *"on the hard-prompt tail (the pathological starvation channel) or on the easy-prompt tail (the benign saturation channel)."* However, rather than stratifying individual prompts by their intrinsic problem difficulty $p_x$, they *"bin every per-step observation of each library's training run by its own heldout-accuracy into five equal-mass quintiles, defined per method... Quintile 0 is the hardest-prompt-dominated training window; quintile 4 is the easiest-prompt-dominated one."*

- Because `heldout-accuracy` is a global evaluation of the policy checkpoint at training step $t$ on a fixed task, and policy accuracy increases over training steps, binning optimizer steps by step-level `heldout-accuracy` stratifies the trajectory by **training time / policy competence stage** (`q0` = earliest training steps when accuracy is near its initial baseline $p \approx 0.25\text{--}0.35$; `q4` = latest training steps when accuracy is higher), not by prompt difficulty.

- In `q0` (where mean accuracy is around $0.25\text{--}0.35$), the binomial collision probability $p^8 + (1-p)^8$ is naturally near its minimum ($\approx 0.01\text{--}0.03$), which explains why all libraries have near-zero ZVF in `q0` ($0.010\text{--}0.020$, Table 22)—not because "hard-prompt groups" produce no zero-variance collisions (a truly hard prompt with $p_x \to 0$ has $(1-p_x)^8 \to 1$ and $\text{ZVF} \to 1$), but because `q0` reflects intermediate batch-level accuracy early in training.

- Additionally, Table 23 (p. 36) computes 95% confidence intervals via a *"2,000-rep step-level bootstrap"* across the 5,540 strongly autocorrelated step rows ($\rho_1 \approx 0.94$) without trajectory/block resampling, which artificially shrinks the CIs.

**Recommendation:** Reframe Section 10.16 and Tables 22–23 as **competence-stratified (or heldout-accuracy-stratified) ZVF across training stages** rather than "prompt-difficulty stratification," correct the interpretation of `q0`, and compute bootstrap CIs in Table 23 by resampling at the seed/trajectory level.

## [MAJOR] Unretracted same-group plug-in estimator residual in Appendix A.4  (p.41 "The i.i.d.-Bernoulli null is moreover only a coarse level estimate—here it over-")

Appendix A.4 states that on the Qwen3-8B GSM8K slice ($G=8$, 200 problems $\times$ 3 seeds), the i.i.d.-Bernoulli null over-predicts mean ZVF ($\approx 0.28$ vs. observed $0.16$). This $\approx 0.28$ figure ($0.1583 + 0.1224 = 0.2807$) is the average of the same-group plug-in quantity $h_G(\hat{p}_x) = \hat{p}_x^G + (1 - \hat{p}_x)^G$ evaluated at each group's empirical success rate $\hat{p}_x = k_x / G$. However, this exact quantity and its resulting $+0.1224$ residual ($\delta_{\text{div}}$) were formally retracted as an estimator artifact in the Abstract (Page 1), Section 4 (Page 7), Section 5.8 (Page 15), Sections 10.7 and 10.11 (Pages 27, 30), and Section 10.19 (Page 39).

As proven in Section 5.8, evaluating $h_G(k_x/G)$ on the same group that defines the binary all-agree indicator $\mathbf{1}\{k_x \in \{0, G\}\}$ mathematically guarantees $h_G(k_x/G) \ge \mathbf{1}\{k_x \in \{0, G\}\}$ point-wise for every group—producing an upward estimator bias of $+0.105$ in expectation even under exact i.i.d. rollouts at $\bar{p} = 0.694$. Conversely, under the corrected analysis in the Abstract and Section 5.8, the homogeneous i.i.d.-Bernoulli baseline evaluated at the pooled mean reward ($\bar{p} = 0.694$) is $h_8(\bar{p}) = 0.054$, which *under-predicts* the observed ZVF ($0.158$) by a factor of $2.9\times$ due to Jensen's inequality over heterogeneous prompt difficulties (while a beta-binomial fit with intra-prompt correlation $0.160$ predicts $0.170$).

**Suggested edit:**

```
The homogeneous i.i.d.-Bernoulli baseline evaluated at the pooled mean reward (p̄ = 0.694) is h_8(p̄) = 0.054, which under-predicts the observed ZVF (0.158) by 2.9× due to Jensen convexity over promptdifficulty dispersion, whereas a beta-binomial fit (intra-prompt correlation 0.160) predicts 0.170 (see Section 5.8).
```

## [MAJOR] Unexecuted counterfactual table with placeholder cells and threshold discrepancy  (p.42 "Table 24: Reward-shape counterfactual. ZVF is the fraction of GRPO groups with w")

1. **Unexecuted experimental table:** Appendix B presents a 12-run GRPO protocol (2 models $\times$ 2 reward versions $\times$ 3 seeds, 50 training steps each) where no runs have been executed and all 12 empirical rows in Table 24 contain placeholder dashes (`—`). Unless the target venue explicitly supports unexecuted registered-report appendices, leaving an unrun experimental table in a manuscript under review leaves the reward-coarseness vs. capacity-ceiling hypothesis unresolved. Either completing the 12 short runs to populate Table 24 or moving the unexecuted design to a future-work discussion in Section 11 would strengthen the completeness of the manuscript.
2. **Variance threshold discrepancy:** Section B.2 states that ZVF is computed with `scripts/zvf_compute_cross_framework.py`, which fixes the default variance tolerance at $\varepsilon = 10^{-6}$ in Appendix A.1 (Equation 13) and Appendix C.1/C.3 (`eps: float = 1e-6`). However, the caption of Table 24 defines ZVF as *"the fraction of GRPO groups with within-group score variance below $10^{-4}$."* While both thresholds strictly test exact homogeneity on the six-level grid $\{0.0, 0.2, 0.4, 0.6, 0.8, 1.0\}$ at $G=16$ (where the minimum nonzero unbiased variance is $\frac{0.2^2}{16} = 2.5 \times 10^{-3}$), aligning the stated threshold in Table 24 with $\varepsilon = 10^{-6}$ (or specifying whether `--eps 1e-4` is passed) would maintain consistency.

**Suggested edit:**

```
Table 24: Reward-shape counterfactual. ZVF is the fraction of GRPO groups with within-group score variance below 10^{-6}, averaged over the final ten training steps.
```

## [MAJOR] Missing bibliographic entries for evaluated variance-mitigation baselines and named methods  (p.44 "References
[1] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Cou")

While the References section includes entries for AERO `[28]`, CPPO `[17]`, GIFT `[23]`, and RL-ZVP `[16]`, several other named variance-mitigation algorithms and methods that are extensively evaluated or explicitly discussed in the main text lack entries in the References section (pages 44–46):

* **NGRPO**, **SCAFGRPO / Scaf-GRPO**, **MCGRPO**, and **AREAL / AReaL** are core baselines across the cross-library empirical tables and figures (Sections 4.1, 4.5, 5.3, 5.5, 6.2, 8, and 10), and Section 5.5 (page 13) explicitly refers to *"the original SCAFGRPO claim of suppressing all-zero groups at the reward-landscape level"* without a citation in the bibliography.

* Named statistical and dynamical methods cited inline by author-year—specifically *"the Hedges–Olkin (1985) analytical formula"* (Section 3.3, page 5) and *"the Scheffer-2009 EWS statistic"* (Section 4.5, page 9)—as well as the RL libraries benchmarked in Section 2.2 (TRL, Stable Baselines3, CleanRL, Tianshou, PufferLib, rl_games, and d3rlpy) do not have corresponding entries in the References section.

Adding complete bibliographic entries for these methods would improve scholarly completeness and traceability.

## [MINOR] Inconsistent experiment and run counts across Sections 1–3  (p.7 "Experiments are distributed across two compute platforms:
Tinker API (14 experim")

The Abstract and Section 1 (pages 1–2) state that the benchmark comprises *"70+ RL runs across seven libraries and five model families,"* Section 3.1 (page 4) mentions *"32+ model configurations,"* and Section 3.3 (page 7) analyzes *"42 experiments with valid performance observations."* However, Figure 3 (page 5) and Section 3.4 (*Compute Resources*, page 7) state that experiments are distributed across two compute platforms totaling only 20 experiments: *"Tinker API (14 experiments)"* and *"Modal H100 GPU cluster (6 experiments),"* while the *TRL baseline characterisation* on page 7 introduces a 5-seed Qwen2.5-0.5B sweep on an NVIDIA L4 GPU not listed in Section 3.4.

A reader cannot reconcile how 14 Tinker experiments, 6 Modal H100 experiments, and the L4 sweep map onto the "42 experiments" in the ANOVA or the "70+ RL runs" claimed in the Abstract (or how many of those runs are completed training runs versus inference-only evaluations on frontier models or synthetic simulation projections).

**Suggested edit:**

```
Add a concise accounting breakdown in Section 3.4 reconciling experiments versus individual seeded runs across platforms (Tinker API, Modal H100, NVIDIA L4) and tiers (Tier-A, Tier-B, Tier-C) to show how they total 42 ANOVA observations and 70+ runs.
```

## [MINOR] Ambiguous zero-variance advantage normalization formula and small text in Figure 2  (p.4 "Figure 2: Verifiable reward paradigm in TINKERRL-BENCH. Unlike shaped rewards th")

Figure 2 (page 4) defines the group-relative advantage as $\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$ where $\sigma_r = \text{std}(\{r_i\})$, without a numerical stabilizer $\varepsilon$ in the denominator or an explicit indicator for $\sigma_r = 0$ (and without specifying whether $\sigma_r$ uses sample or population standard deviation, whereas Appendix A.1 uses unbiased sample variance $\widehat{\text{Var}}$ with threshold $\varepsilon = 10^{-6}$).

When all rewards in a group are identical ($\sigma_r = 0$, the exact condition defining ZVF), the expression $\frac{r_i - \bar{r}}{\sigma_r}$ evaluates to $0/0$ (`NaN`) unless a stabilizer $\varepsilon$ is added to the denominator, standard-deviation division is omitted (unnormalized centering $r_i - \bar{r}$), or zero-variance groups are explicitly masked to zero. Moreover, Section 10.19 (page 39) later notes that if a KL penalty is folded into the reward prior to normalization ($r_i = r_i^{\text{task}} - \beta\,\text{KL}_i$), dividing by $\sigma_r$ when $\text{Var}(r^{\text{task}}) = 0$ cancels $\beta$ and amplifies the KL term into a full-magnitude update $\hat{A}_i = -(\text{KL}_i - \overline{\text{KL}})/\sigma_{\text{KL}}$.

Additionally, the internal text labels and equations inside Figure 2 are rendered at a very small font size and low resolution, and Figure 2 uses $N$ for the group rollout count ($N$ Responses, $\{x_i\}_{i=1}^N$, $\{r_i\}_{i=1}^N$), whereas the Abstract and Section 1 use $G$ (and Appendix A uses $K$).

**Suggested edit:**

```
Specify the exact stabilized advantage formula implemented in the submitted runner (including how $\sigma_r = 0$ and KL regularization are handled), standardize the rollout group size notation to $G$ instead of $N$, and increase the font size and resolution of the diagram.
```

## [MINOR] Missing citations to prior art [12, 19] and companion baseline AVSPO in Section 1.1  (p.2 "Variants of GRPO adjust the group baseline or advantage normalization to preserv")

The Abstract (page 1) and Section 5.9 (page 15) appropriately credit He et al. [12] for introducing the mathematically equivalent Advantage Collapse Rate (ACR) and Nie et al. [19] for proving the Jensen excess. However, Section 1.1 (*Related Work*, page 2) does not cite [12] or [19], and neither Section 1.1 nor Sections 4.1, 5.5, and 5.9 (pages 8, 12, and 15) mention **AVSPO (Adaptive Virtual Sample Policy Optimization)**, the companion variance-mitigation algorithm introduced alongside ACR in [12] (despite Sections 4.1, 5.5, and 5.9 comparing and discussing AERO [28], RL-ZVP [16], CPPO [17], GIFT [23], DAPO [27], GRESO [29], VCRL [15], and DARA [30]).

Because [12] is cited as the primary prior art for the diagnostic metric itself and directly uses real-time ACR/ZVF monitoring to drive AVSPO (by injecting virtual reward samples into homogeneous groups), briefly referencing [12] (including AVSPO) and [19] in Section 1.1 alongside CPPO [17] and GIFT [23], as well as discussing how AVSPO relates to the evaluated variance-mitigation mechanisms in Section 5.5 or 5.9, would make the literature positioning and synthesis complete.

**Suggested edit:**

```
Variants of GRPO adjust the group baseline, advantage normalization, or virtual sample injection to preserve signal [17, 23, 18, 12, 19]; our aim is orthogonal—not to propose a new estimator but to test how well a simple degeneracy statistic (equivalent to the Advantage Collapse Rate [12]) predicts outcomes.
```

## [MINOR] Internal repository-management directives and advising metadata in main text  (p.1 "Absorbed satellite (retired). The former short stratified NeurIPS framing (roste")

Internal repository-management directives and advising metadata appear in the main text—specifically the footnote `*Project guide` on the second author's name, the paragraph **Absorbed satellite (retired)** on page 1 (*"The former short stratified NeurIPS framing (roster R02) is retired from the live tree to `archive/absorbed/R02_main_zvf/`. Its evidence is this paper; do not count R02 as a separate submission."*), and the conversational heading **"Two seven-library rosters — read carefully"** on page 2. Moving internal repository notes to an acknowledgments or artifact appendix would improve presentation for external readers.

**Suggested edit:**

```
Remove the 'Absorbed satellite (retired)' repository directive from the title page (or move to an artifact appendix), replace '*Project guide.' with standard academic author notes if needed, and rephrase the Section 2.2 paragraph heading to 'Distinction between the two seven-library rosters.'
```

## [MINOR] Inconsistent capitalization of verl / veRL within Section 2.2  (p.2 "Tinker, TRL, SkyRL, verl, OpenRLHF, Atropos, and HF reference launchers — that t")

The capitalization of `verl` / `veRL` is inconsistent within the same paragraph (*"Tinker, TRL, SkyRL, verl, OpenRLHF..."* vs. *"Of those, only Tinker and TRL produced completed runs in this release; veRL and OpenRLHF entries..."*).

**Suggested edit:**

```
Tinker, TRL, SkyRL, veRL, OpenRLHF, Atropos, and HF reference launchers — that targets LLM post-training launchers specifically. Of those, only Tinker and TRL produced completed runs in this release; veRL
```

## [MINOR] Small font size and low resolution of internal text labels in Figure 1  (p.3 "Figure 1: Task-coverage taxonomy of TINKERRL-BENCH. The benchmark organises task")

The internal text labels inside **Figure 1** (page 3) are rendered at a very small font size and low resolution relative to the body text, making them difficult to read without high magnification.

**Suggested edit:**

```
Re-export Figure 1 in vector format with larger internal text labels matching the body font size.
```

## [MINOR] Small font size and low resolution of internal text labels in Figure 3  (p.5 "Figure 3: Experiment workflow pipeline. Multi-seed Modal/TRL runs use the centra")

The internal text labels inside **Figure 3** (page 5) are rendered at a very small font size and low resolution relative to the body text, making them difficult to read without high magnification.

**Suggested edit:**

```
Re-export Figure 3 in vector format with larger internal text labels matching the body font size.
```

## [MINOR] Clarify early-phase ZVF range (0.04–0.13) relative to Table 9 values  (p.7 "across 60 measured runs spanning nine variancemitigation methods and the group-s")

In Section 4 (second paragraph, page 7), the text states *"early-phase ZVF (0.04–0.13 for variance-mitigation GRPO and the largest group size..."* — In Table 9 (page 17), $0.044$ is vanilla `grpo` and $0.135$ is `grpo G=16`, whereas the variance-mitigation methods start at $0.012\text{–}0.015$ (except `scafgrpo` at $0.250$). Clarifying that $0.04\text{–}0.13$ refers to vanilla GRPO in the variance-mitigation suite and `grpo G=16` in the group-size sweep will prevent confusion when cross-referencing Table 9.

**Suggested edit:**

```
across 60 measured runs spanning nine variancemitigation methods and the group-size sweep, late-phase ZVF (0.87–0.98 for vanilla GRPO) far exceeds early-phase ZVF (0.044 for vanilla GRPO in the variancemitigation suite and 0.135 for G=16 in the group-size sweep; smaller groups G=2–8 start at 0.25–0.61), with lag-1 autocorrelation ≈ 0.94
```

## [MINOR] Unrelated HippoRAG information-retrieval benchmark paragraph at end of Section 4.5  (p.11 "A HippoRAG-style KG+PPR retrieval scaffold [11] over this artifact corpus (67 pa")

The end of Section 4.5 (top of p. 11) includes a standalone paragraph evaluating a *"HippoRAG-style KG+PPR retrieval scaffold [11] over this artifact corpus (67 passages, 21-entity knowledge graph)"* against BM25 (`recall@10 = 0.667`, `MRR 0.554 vs. 0.569`). This information-retrieval benchmark over the repository's documentation files has no scientific connection to GRPO or the ZVF diagnostic and should be deleted.

## [MINOR] First-passage boundary condition at i=1 and AUC normalization ambiguity in Equations (6)–(7)  (p.16 "τ (θ; z) = min{i : zi ≥ θ ∧ zi−1 < θ} (first-passage step to ZVF=θ) (6)
A(θ; z) ")

There are two definitional issues in Equations (6) and (7):

1. **Boundary condition at $i=1$ in Equation (6):** Equation (6) defines the first-passage step as $\tau(\theta; z) = \min\{i : z_i \ge \theta \land z_{i-1} < \theta\}$. Under this formula, if a trajectory starts at or above threshold at $i=1$ ($z_1 \ge \theta$), $z_0$ is undefined and the crossing fails to register (e.g., tool-use trajectories where $\text{ZVF}_t = 1.0$ for all $t$). Defining $\tau(\theta; z) = \min\{i \in \{1, \dots, T\} : z_i \ge \theta\}$ avoids this boundary issue.
2. **Normalization ambiguity in Equation (7) vs. Table 9:** Equation (7) defines $A(\theta; z) = \frac{1}{T}\sum_{i=1}^T \max(0, z_i - \theta)$ and the text below Eq. (8) states its maximum possible value is $1 - \theta$. However, the caption of Table 9 (p. 17) states that *"$A(\theta; \cdot)$ is the AUC above threshold $\theta$ normalized to $[0, 1 - \theta]$."* Please clarify whether the values in Table 9 are the raw unnormalized outputs of Eq. (7) (whose natural range is $[0, 1 - \theta]$) or rescaled by $1/(1 - \theta)$.

**Suggested edit:**

```
\tau(\theta; z) = \min\{i \in \{1, \dots, T\} : z_i \ge \theta\}
```

## [MINOR] Duplicate file path figures/zvf_vs_failure.pdf and mismatched Figure 5 description  (p.9 "• figures/zvf_vs_failure.pdf — per-(experiment, model, G) scatter of mean ZVF vs")

Section 4.4 ("Figures" bullet 1, p. 9) and Section 4.5 ("Artifacts" bullet 5, p. 11) both reference the identical file path `figures/zvf_vs_failure.pdf` for two different figures:

- Section 4.4 describes it as a single scatter plot (*"per-(experiment, model, G) scatter of mean ZVF vs heldout accuracy, colored by deterministic collapse label (Fig. 5, this section)"*—note also that Figure 5 appears in Section 5.7 on p. 14, not in Section 4).

- Section 4.5 (p. 11) describes it as a *"4-panel regenerated figure: (a) magnitude $\times$ CSD scatter... (b) per-method risk bar; (c) ROC curves... (d) risk-bin calibration."*

- Meanwhile, Figure 5 on p. 14 actually renders a 4-panel figure titled *"Pillar 2 iter 78 — failure-aware ZVF diagnostic (extends iter26)"* whose Panel D displays the withdrawn outcome correlation (`Pearson=0.276, Spearman=0.110 (n=65)`).

The figure files, internal iteration titles, and text references should be synchronized.

## [MINOR] Main-text Sections 5, 6, and 7 erroneously refer to themselves and preceding sections as 'appendices'  (p.11 "The previous two appendices formalized ZVF (Section A) and specified the per-fra")

Main-text Sections 5 (p. 11), 6 (p. 15), and 7 (p. 17) refer to themselves and preceding sections as *"appendices"*:

- Page 11 (Section 5 opening): *"The previous two appendices formalized ZVF (Section A) and specified the per-framework reward-field mapping (Section C)"*—even though Appendices A and C appear at the end of the paper on pp. 40 and 43.

- Page 15 (Section 6 opening): *"The previous appendix aggregates ZVF as a pooled per-run mean..."*

- Page 17 (Section 7 opening): *"This appendix closes the cross-pillar loop..."*

Replace *"appendix"* with *"section"* and update the references to Appendices A and C as forward references.

**Suggested edit:**

```
Appendices A and C formalize ZVF and specify the per-framework rewardfield mapping. This section closes the diagnostic loop by aggregating every ZVF stream in our measurement matrix into a single perexperiment summary, attaching a deterministic failure taxonomy, and reporting the correlation between mean-ZVF and the heldoutaccuracy outcome.
```

## [MINOR] Uncontextualized internal issue-tracker reference (#36320)  (p.13 "Reviewed-record correction. Any crosslibrary AERO/GRPO comparison below is later")

Section 5.5 ("Reviewed-record correction") contains an uncontextualized internal issue-tracker or review reference (*"not reviewed #36320 evidence"*) that should be removed or explained for external readers.

**Suggested edit:**

```
Reviewed-record correction. Any crosslibrary AERO/GRPO comparison below is later or heterogeneous evidence, not an algorithm ranking.
```

## [MINOR] Duplicate empty subsection heading '5.6 Practical Diagnostic Recipe'  (p.13 "5.6 Practical Diagnostic Recipe
5.7 Practical Diagnostic Recipe")

The subsection heading `5.6 Practical Diagnostic Recipe` is completely empty and is immediately duplicated on the next line by `5.7 Practical Diagnostic Recipe`. Delete the duplicate empty heading `5.6`.

**Suggested edit:**

```
5.6 Practical Diagnostic Recipe
```

## [MINOR] Clipped suptitle and overlapping x-axis labels in Figure 4  (p.14 "Figure 4: Cross-library ZVF diagnostic. Left: mean-ZVF per library; AERO (purple")

In Figure 4, the top suptitle is vertically clipped and illegible, and the rotated x-axis tick labels (`arithmetic_groupsize`, `drgrpo_vs_grpo_math`, `samestack_ppo_grpo`) overlap with the horizontal axis titles.

## [MINOR] Unpopulated draft placeholder in Section 5.8  (p.15 "Prospective prediction test (placeholder). [Placeholder: held-out ZVF prediction")

An unpopulated draft placeholder remains at the bottom of Section 5.8 and should either be populated with the completed experiment or removed before publication.

## [MINOR] Internal drafting notes referencing 'frontier synthesis' before Section 10.19  (p.16 "(frontier synthesis: matches Gemini Deep Think’s “structural signal bottleneck” ")

Section 6.2(i) (p. 16) and Section 8.1 (pp. 20–21) contain internal drafting notes in parentheses—such as *"(frontier synthesis: matches Gemini Deep Think's 'structural signal bottleneck' framing.)"*—that reference the external model cross-examination before it is introduced in Section 10.19 (p. 38). Rewrite these as standard prose with forward references to Section 10.19 or remove the parenthetical drafting notes.

**Suggested edit:**

```
As discussed in Section 10.19, this matches the 'structural signal bottleneck' framing.
```

## [MINOR] Incorrect reference to Table 9 column A(theta; .) as lead time  (p.16 "We report lead time as a data point (Table 9 column A(θ; ·) and Figure 7) rather")

Section 6.3 states *"We report lead time as a data point (Table 9 column $A(\theta; \cdot)$ and Figure 7)"*—however, column $A(\theta; \cdot)$ in Table 9 reports the AUC above threshold $\theta$ (sustained pressure), not the lead time $\ell = t_{\text{collapse}} - \tau(\theta; z)$.

**Suggested edit:**

```
We report lead time as a data point (Figure 7 and Table 14) rather than as an intervention prescription.
```

## [MINOR] Clipped numerical annotations above bars in Figure 6  (p.18 "Figure 6: Per-method phase-decomposed ZVF, 5 seeds pooled. Red bars (late-phase)")

In Figure 6, the $\Delta = \bar{z}_l - \bar{z}_e$ numerical annotations above each bar cluster are partially clipped by the top axis border. Increase the y-axis upper limit (e.g., to $1.10$) so the labels are fully legible.

## [MINOR] Garbled glyph for theta in Figure 7 x-axis label  (p.18 "Figure 7: First-passage step τ (θ; z) vs. first collapse-flag step tcollapse for")

In Figure 7, the x-axis label renders with a missing/garbled glyph for $\theta$ (*"first step at which ZVF crosses"* followed by a crossed-zero symbol).

## [MINOR] Uncited and misplaced Table 10 in Section 7  (p.19 "Table 10: Per-step Pearson correlation between ZVF and gradient-flow proxies, po")

Table 10 (*"Per-step Pearson correlation between ZVF and gradient-flow proxies..."*) is placed at the top of page 19 inside Section 7, but it is never cited or discussed in Section 7 (nor anywhere else in pages 7–20). Ensure Table 10 is referenced and discussed in the appropriate section.

## [MINOR] Missing minus signs on negative vertical axis tick labels in Figure 8(B)  (p.20 "Figure 8: Five-anchor ZVF-proxy diagnostic. (A) ZVF-proxy = (frac_below_0p1 + ze")

In Figure 8(B) (`late_minus_peak (negative = collapse)`), the minus signs on the negative vertical axis tick labels (`-0.7` to `0.0`) fail to render in the plot (appearing as `0.7` to `0.0`).

## [MINOR] Inconsistent definition and misnomer of 'ZVF lead-time' in Section 8.2  (p.21 "• first_pass_step = arg maxt(heldout_acc(t)), the step where the policy is most ")

Section 8.2 defines `first_pass_step = arg max_t(heldout_acc(t))` and `lead_steps = first_collapse_step - first_pass_step` (the lag between peak held-out accuracy and the first collapse flag, which does not use ZVF), whereas Section 6.3 (p. 16) defined first-passage step $\tau(\theta; z)$ and ZVF lead time as $t_{\text{collapse}} - \tau(\theta; z)$ (the lag between ZVF crossing threshold $\theta$ and collapse). Labelling the peak-accuracy-to-collapse lag in Table 14 and Figure 10 as *"ZVF lead-time"* is a misnomer.

## [MINOR] Figure 11 graphic is missing above its caption on page 24  (p.24 "Figure 11: ZVF-only features as a leading indicator of training collapse on the ")

The caption for Figure 11 is printed at the top of page 24, but the graphic itself (Left: ROC at horizon $K=5$; Right: ROC at horizon $K=10$) is missing from the rendered page.

## [MINOR] Degenerate rounded interval notation [0.93, 0.93] in Section 9.2  (p.24 "1. ZVF-level features are a strong leading indicator at every horizon. trailing_")

In Table 15, the ROC-AUC values for `trailing_mean_25` and `trailing_auc_25` lie in $[0.928, 0.932]$; rounding both endpoints to two decimal places creates a degenerate interval notation `[0.93, 0.93]`. Report three decimal places (`[0.928, 0.932]`) or write `$\approx 0.93$'.

**Suggested edit:**

```
1. ZVF-level features are a strong leading indicator at every horizon. trailing_mean_25 and trailing_auc_25 ROC-AUC lie in [0.928, 0.932] for K ∈ {5, 10, 25}
```

## [MINOR] Unformatted 'theta' in plain text instead of math symbol  (p.25 "consistently passes the threshold theta = 0.8–0.9 within 1–3 steps of collapse")

Typeset `"theta"` in math mode as `$\theta = 0.8\text{--}0.9$`.

**Suggested edit:**

```
consistently passes the threshold θ = 0.8–0.9 within 1–3 steps of collapse
```

## [MINOR] Inconsistent iteration attribution and threshold for scalar lead-time, plus typo in '(ZV F−0.5)+'  (p.32 "Iter 9 established the scalar lead-time fact: the first step at which the per-st")

Paragraph 1 states *"Iter 9 established the scalar lead-time fact: the first step at which the per-step ZVF crosses $\theta=0.5$ precedes the first step at which the heldout accuracy collapse flag trips by 1–3 steps"*, whereas earlier in Section 6.3 (p. 16) this analysis is attributed to `iter14` at thresholds $\theta = 0.8$ and $\theta = 0.9$. Additionally, in Bullet (B), remove the extraneous space inside `"(ZV F−0.5)+"` so it reads `"$(\text{ZVF}-0.5)^+$"`.

**Suggested edit:**

```
Iter 14 established the scalar lead-time fact: the first step at which the per-step ZVF crosses θ=0.8–0.9 precedes the first step at which the heldout accuracy collapse flag trips by 1–3 steps on every collapsing trajectory. We lift that scalar claim to a vector one in two directions not previously explored in the iter22/30/34/38/42/46 chain:
• (A) reward-leads-ZVF cross-correlation. We compute Pearson r(rewardt, ZVFt+L) at L ∈ {−10, −5, −1, 0, +1, +5, +10} for every (library, seed) trajectory in experiments/results/variance_mitigation.tsv. A peak of the lag profile at a positive L is the canonical signature of reward leading ZVF (reward at t is a stronger linear predictor of ZVF at t + L than of ZVF at t itself).
• (B) phase-2 (ZVF−0.5)+ integral.
```

## [MINOR] Word order typo in 'Pre-registered predictions 4'  (p.33 "Pre-registered predictions 4 are reported in experiments/results/zvf_iter50_pred")

Change `"Pre-registered predictions 4 are reported in..."` to `"Four pre-registered predictions are reported in..."`.

**Suggested edit:**

```
Four pre-registered predictions are reported in experiments/results/zvf_iter50_predictions.tsv
```

## [MINOR] Rounding discrepancy in ZVF+ difference (+0.008 vs +0.007 in Table 21)  (p.34 "Against matched-stack GRPO, AERO reduces raw ZVF by −0.261; the signed decomposi")

In Table 21 on the same page, GRPO has $\text{ZVF}^+ = 0.011$ and AERO has $\text{ZVF}^+ = 0.018$, which is a difference of $+0.007$ ($-0.268 + 0.007 = -0.261$), rather than $+0.008$.

**Suggested edit:**

```
Against matched-stack GRPO, AERO reduces raw ZVF by −0.261; the signed decomposition attributes −0.268 of that to ZVF− (103% – ZVF+ moves the other way, +0.007).
```

## [MINOR] Missing minus superscript on ZVF- and combined boxplot in Figure 12  (p.35 "Figure 12: Signed ZVF decomposition. Left: per-run stacked ZVF− (starvation, red")

In Figure 12's plot title (`Raw ZVF = ZVF (pathological) + ZVF+ (benign)`), legend (`ZVF (all-wrong, starvation)`), and right-panel x-axis label (`ZVF AUC=1.00`), the minus superscript on $\text{ZVF}^-$ fails to render, making $\text{ZVF}^-$ visually indistinguishable from raw `ZVF`. Additionally, the right panel plots a single combined boxplot per metric rather than separating the `healthy` (green) and `unhealthy` (red) distributions indicated by the legend.

## [MINOR] Repeated number in '−0.48 to −0.48 reductions' in Section 10.16  (p.36 "is revealed to be a misleading framing: a −0.006 ZVF change at the hard end is d")

The text states `"dwarfed by −0.48 to −0.48 reductions"`, repeating `-0.48` (Table 23 gives `-0.476` and `-0.480`). Write `"−0.476 to −0.480 reductions"` or `"≈ −0.48 reductions"`.

**Suggested edit:**

```
is revealed to be a misleading framing: a −0.006 ZVF change at the hard end is dwarfed by −0.476 to −0.480 reductions in the middle quintiles (q2 and q3),
```

## [MINOR] Missing space after '(ii)' in Section 10.16  (p.36 "at the end of training; (ii)flattener SCAFGRPO with ∆4−0=0.16,")

Add a space between `(ii)` and `flattener` in `"; (ii)flattener SCAFGRPO"`.

**Suggested edit:**

```
at the end of training; (ii) flattener SCAFGRPO with ∆4−0=0.16,
```

## [MINOR] Missing minus signs on negative vertical axis tick labels in Figure 13 (right panel)  (p.37 "Figure 13: Difficulty-stratified ZVF, 9 libraries. Left: mean ZVF vs heldoutaccu")

In the right panel of Figure 13 (`AERO - GRPO ZVF (per-step mean)`), the minus signs on the negative vertical axis tick labels (`-0.5` to `0.0`) fail to render in the plot (appearing as `0.5` to `0.0`).

## [MINOR] Distinguish biased vs. unbiased within-group variance in Equation (11)  (p.38 "PCD = Ex[Var(r | x)] = Ex[1/G2 ∑i<j(ri − rj )2] = G−1/G Ex[px(1 − px)], (11)")

Equation (11) defines Pairwise Contrast Density (PCD) via the chain of equalities:
$$\text{PCD} = \mathbb{E}_x\big[\text{Var}(r \mid x)\big] = \mathbb{E}_x\left[\frac{1}{G^2}\sum_{i<j}(r_i - r_j)^2\right] = \frac{G-1}{G}\mathbb{E}_x\big[p_x(1 - p_x)\big].$$
For conditionally i.i.d. Bernoulli rollouts $r_1, \dots, r_G \sim \text{Bernoulli}(p_x)$, the middle expression $\frac{1}{G^2}\sum_{i<j}(r_i - r_j)^2 = \frac{1}{G}\sum_{i=1}^G (r_i - \bar{r})^2$ is the *biased* (`ddof=0`) empirical within-group sample variance, whose conditional expectation is indeed $\frac{G-1}{G}p_x(1 - p_x)$. However, writing $\text{Var}(r \mid x)$ in the first term is ambiguous because the true conditional population variance of a single Bernoulli draw given prompt $x$ is $\text{Var}(r \mid x) = p_x(1 - p_x)$ (without the $\frac{G-1}{G}$ factor), and the expectation of the *unbiased* sample variance $\widehat{\text{Var}}(r_{g,1}, \dots, r_{g,G}) = \frac{1}{G-1}\sum_{i=1}^G (r_i - \bar{r})^2$ defined in Equation (13) (and implemented with `ddof=1` in Appendix C.1) is also $\mathbb{E}_x[p_x(1 - p_x)]$.

Clarify that the first term in Equation (11) refers to the biased empirical group-sample variance $\widehat{\text{Var}}_G(\mathbf{r} \mid x) = \frac{1}{G}\sum_{i=1}^G (r_i - \bar{r})^2$ of $G$ rollouts.

**Suggested edit:**

```
\text{PCD} = \mathbb{E}_x\big[\widehat{\text{Var}}_G(\mathbf{r} \mid x)\big] = \mathbb{E}_x\left[\frac{1}{G^2}\sum_{i<j}(r_i - r_j)^2\right] = \frac{G-1}{G}\mathbb{E}_x\big[p_x(1 - p_x)\big],
```

## [MINOR] Clarify architectural and logging assumptions for KL-penalized zero-variance caveat  (p.39 "under a KL-penalized reward ri = rtaski − β KLi, when Var(rtask) = 0 the advanta")

While the derivation $A_i = -(\text{KL}_i - \overline{\text{KL}})/\sigma_{\text{KL}}$ is algebraically valid when $\beta > 0$ is folded into the scalar reward prior to standard-deviation normalization (assuming $\sigma_{\text{KL}} > 0$ and negligible numerical stabilization $\varepsilon_{\text{std}} \ll \beta\sigma_{\text{KL}}$), this formulation relies on two specific assumptions:
1. **Loss-level vs. reward-level KL regularization:** In canonical GRPO [21, 10] and standard implementations (such as TRL's `GRPOTrainer` and veRL), the reference KL divergence is typically applied as an additive per-token penalty in the surrogate loss *outside* the group-standardized advantage rather than subtracted from the scalar outcome reward prior to dividing by $\sigma_r$. Under loss-level KL regularization (or when $\beta = 0$, as in the paper's managed runtime experiments in Sections 4.1 and 10.7), $\text{Var}(r^{\text{task}}) = 0$ leaves the group-normalized advantage identically zero, and any external KL gradient scales with $\beta$ rather than canceling $\beta$.
2. **Task-reward vs. total-reward ZVF logging:** Even when $-\beta\,\text{KL}_i$ is folded into the pre-normalized scalar reward $r_i$, if ZVF is computed on the total reward $r_i$ via Equation (13), then $\widehat{\text{Var}}(r_1, \dots, r_G) = \beta^2 \widehat{\text{Var}}(\text{KL}_1, \dots, \text{KL}_G)$, which is nonzero whenever completion KL divergences vary within the group.

Explicitly state that this caveat applies specifically to implementations where $\beta > 0$ is folded into the scalar reward prior to group-standardization while ZVF is tracked on the task reward $r^{\text{task}}$ alone.

**Suggested edit:**

```
under a reward-level KL penalty r_i = r_i^{task} - β KL_i (where β > 0 is folded into the scalar reward prior to group standardization while ZVF is logged on r^{task} alone), when Var(r^{task}) = 0 the advantage normalization cancels β and yields A_i = -(KL_i - KL̄)/σ_{KL}, so a task-zerovariance group emits not a null update but a covert, full-magnitude distillation pull toward the reference policy.
```

## [MINOR] Unspecified reward scaling in canonical reference implementation  (p.43 "Rewards not in [0, 1] are scaled to that range before thresholding.")

Section C.3 states: *"We fix $\varepsilon = 10^{-6}$ for rewards normalized to $[0, 1]$... Rewards not in $[0, 1]$ are scaled to that range before thresholding."* However, the reference implementation `zvf(rewards_2d: np.ndarray, eps: float = 1e-6)` in Section C.1 computes `rewards_2d.var(axis=-1, ddof=1) <= eps` directly on the input array without any range normalization, and empirical min-max normalization on a batch whose rewards are constant outside $[0, 1]$ (e.g., all $-1$) would result in division by zero ($0/0$). Specify in Sections C.1 and C.3 that `rewards_2d` is pre-scaled using known, fixed task reward bounds $[R_{\min}, R_{\max}]$ prior to calling `zvf()` (or include the bound-based scaling step in the pseudocode).

**Suggested edit:**

```
Rewards not in [0, 1] are affine-scaled using fixed, known task reward bounds [R_min, R_max] before calling zvf().
```

## [MINOR] Inconsistent G=2 ZVF figure (0.845 vs. 0.838 in Table 21)  (p.39 "Equation (12) is consistent with the 0.845 → 0.631 ZVF drop we observe as G : 2 ")

The text refers to *"the $0.845 \to 0.631$ ZVF drop we observe as $G : 2 \to 16$,"* whereas Table 21 (Page 34) and Appendix A.4 (Page 41) report the $G=2$ ZVF on this sweep as $0.838$ (rounded to $0.84$ on Page 41) rather than $0.845$.

**Suggested edit:**

```
Equation (12) is consistent with the 0.838 → 0.631 ZVF drop we observe as G : 2 → 16:
```

## [MINOR] Broken cross-reference phrasing ('in Section the companion paper')  (p.39 "Second, and as a genuine challenge to the “zero gradient” framing in Section the")

There is a broken cross-reference phrasing in *"to the 'zero gradient' framing in Section the companion paper on ZVF gradients"*. Either remove *"Section"* or insert the intended section number.

**Suggested edit:**

```
Second, and as a genuine challenge to the “zero gradient” framing in the companion paper on ZVF gradients:
```

## [MINOR] Remove 'et al.' from single-author paper in Reference [23]  (p.45 "[23] Zhichao Wang et al. GIFT: Group-relative implicit fine-tuning integrates GR")

In Reference `[23]`, the author list is written as *"Zhichao Wang et al."*, whereas `arXiv:2510.23868` (*"GIFT: Group-Relative Implicit Fine-Tuning Integrates GRPO with DPO and UNA"*) is a single-author paper by Zhichao Wang; *"et al."* should be removed.

**Suggested edit:**

```
[23] Zhichao Wang. GIFT: Group-relative implicit fine-tuning integrates GRPO with DPO and UNA. arXiv preprint, 2025. doi: 10.48550/arXiv.2510.23868. arXiv:2510.23868.
```

## [MINOR] Duplicate conference designation and inconsistent formatting in Reference [16]  (p.45 "[16] Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Lai, and Eunho Yang. No promp")

In Reference `[16]`, the venue string reads *"International Conference on Learning Representations, September 2025. ICLR 2026; also referred to as RL-ZVP; arXiv:2509.21880."* Cleaning up the duplicated conference designation (*"International Conference on Learning Representations, September 2025. ICLR 2026"*) and matching the `doi` / `arXiv preprint` formatting used in the surrounding entries would improve consistency.

**Suggested edit:**

```
[16] Thanh-Long V. Le, Myeongho Jeon, Kim Vu, Viet Lai, and Eunho Yang. No prompt left behind: Exploiting zerovariance prompts in LLM reinforcement learning via entropy-guided advantage shaping. In International Conference on Learning Representations (ICLR), 2026. doi: 10.48550/arXiv.2509.21880. arXiv:2509.21880; also referred to as RL-ZVP.
```

## [MINOR] Abbreviated first name for final co-author in Reference [19]  (p.45 "[19] Wen-Hua Nie, Jia-Nan Wu, Junlin Liu, Zi-Wei Li, Zheng Lin, Zi-Jian Zhang, Y")

In Reference `[19]`, the final co-author is abbreviated as *"J. Jang"* (for Jyh-Shing Roger Jang), whereas full first names are used for all other authors in `[19]` and throughout the rest of the bibliography.

**Suggested edit:**

```
[19] Wen-Hua Nie, Jia-Nan Wu, Junlin Liu, Zi-Wei Li, Zheng Lin, Zi-Jian Zhang, Yi-Long Fan, Haoran Zheng, and Jyh-Shing Roger Jang. Gradient starvation in binary-reward GRPO: Why group-mean centering fails and why the simplest fix works. arXiv preprint, 2026. doi: 10.48550/arXiv.2605.07689. arXiv:2605.07689.
```

## [SUGGESTION] Standardize group-size notation between K and G across appendices  (p.40 "Let a GRPO batch Bt at optimizer step t consist of |Bt| prompt-groups, each grou")

Appendices A.1, A.2, A.5, and C use $K$ to denote the rollout group size (`[|B_t|, K]`), whereas the main text, Section 10.19, Appendix A.4, and Appendix B use $G$ for group size (while using $K_x$ in Section 10.19 for the within-group success count, $K$ in Section 9.1 for the forecasting horizon, and $K$ in Section 10.7 for the number of prompts). Standardizing the group-size symbol to $G$ (or explicitly noting $K \equiv G$ at the start of Appendix A.1) would prevent symbol collision across sections.

**Suggested edit:**

```
Let a GRPO batch B_t at optimizer step t consist of |B_t| prompt-groups, each group g containing K ≡ G rollouts with scalar outcome rewards r_{g,1}, . . . , r_{g,K} ∈ ℝ.
```
