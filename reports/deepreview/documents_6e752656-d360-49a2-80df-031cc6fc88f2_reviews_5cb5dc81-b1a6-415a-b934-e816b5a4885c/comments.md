# Anchored comments

## [CRITICAL] Post-hoc circularity (endogeneity) in capability-class regression and AICc comparison  (p.23 "The interpretation is sharp: the capability class (Ward k = 2 split on Rmax) car")

In **Iter 133** (and propagated into **Iters 137** and **140**), four nested OLS models are fitted with the empirical asymptotic ceiling $R_{\max}$ as the dependent variable and compared via $\text{AIC}_c$. However, the explanatory variable `capability class` is defined *post-hoc* as the Ward-linkage $k = 2$ cluster assignment computed directly on the empirical distribution of the response variable $R_{\max}$ itself (partitioning $R_{\max} \le 0.30$ from $R_{\max} \ge 0.70$, as noted on p. 24). Partitioning a 1D response variable $Y = R_{\max}$ into two clusters by minimizing within-cluster variance (Ward's criterion) and then regressing $Y$ on that exact cluster indicator introduces target leakage (endogeneity) and guarantees a large reduction in residual sum of squares (RSS) by construction. Moreover, the standard $\text{AIC}_c$ penalty ($2k + \frac{2k(k+1)}{n-k-1}$) treats the data-derived cluster indicator as a pre-specified exogenous covariate and does not penalize for the degrees of freedom consumed by optimizing the split threshold on the response variable. Similarly, the LOOCV cluster agreement tests (Tables 23 and 24) only verify that holding out one value of $R_{\max}$ and re-clustering the remaining $n - 1$ values assigns the held-out $R_{\max}$ to the same cluster across a $0.531$ gap in the response distribution; they do not validate an independent predictive feature.

**Actionable Recommendation:** Define the capability indicator *a priori* from exogenous model metadata independent of the GSM8K RL reward traces before running the OLS and $\text{AIC}_c$ comparisons—for example, using an explicit binary indicator for whether the checkpoint underwent instruction/reasoning tuning prior to GRPO post-training vs. a base pretrained checkpoint, or using zero-shot baseline evaluation accuracy $R(0)$ prior to RL step 1.

## [CRITICAL] Violation of nested model likelihood monotonicity in Table 26 (∆AICc up to +18.18) and Rmax > 1.0  (p.25 "Table 26: Three-parameter offset saturation fit (iter 137) vs. two-parameter fit")

Table 26 exhibits two mathematical and optimization issues:
1. **Violation of Nested Model Likelihood Monotonicity:** The 3-parameter offset saturation model $R(t) = c + (R_{\max} - c)(1 - e^{-\lambda t})$ with $c \in [0, 1]$ (Eq. 6, p. 24) strictly nests the 2-parameter canonical model $R(t) = R_{\max}(1 - e^{-\lambda t})$ at the boundary $c = 0$. Under least-squares / Gaussian MLE, adding a free parameter to a nested model cannot increase the minimized residual sum of squares ($\text{RSS}_{3\text{p}} \le \text{RSS}_{2\text{p}}$). At $n = 30$, adding one parameter (moving from $k=2$ to $k=3$, or $k=3$ to $k=4$ including noise variance $\hat{\sigma}^2$) increases the small-sample $\text{AIC}_c$ penalty $2k + \frac{2k(k+1)}{n-k-1}$ by at most $\approx 2.34$ to $2.57$ units, meaning $\Delta\text{AIC}_c = \text{AICc}_{3\text{p}} - \text{AICc}_{2\text{p}}$ mathematically cannot exceed $\approx +2.57$ if both models are globally optimized under the same likelihood formula. Yet Table 26 reports $\Delta\text{AIC}_c = +18.18$ on `Qwen3.5-4B` ($-2.57$ vs. $+15.61$) and $+14.77$ on `Llama-3.1-8B-Instruct` ($-19.16$ vs. $-4.39$). This indicates that the 3-parameter fitting routine either fixed $c_{3\text{p}}$ heuristically rather than jointly optimizing $(c, R_{\max}, \lambda)$, constrained $\lambda_{3\text{p}}$ away from the $\lambda = 10$ boundary, or used a different likelihood definition between Iter 117 and Iter 137.
2. **Asymptotic Ceiling Exceeding $[0, 1]$ Support:** Per-step GSM8K rewards are means of binary correctness rollouts strictly bounded in $[0, 1]$. However, Table 26 reports $R_{\max}^{3\text{p}} = 1.050$ for `Qwen3-8B` (despite `Qwen3-8B` having an empirical peak of $0.625$ and mean of $0.285$). Similar unphysical ceiling estimates exceeding $1.0$ appear in **Table 23** (p. 22, $\hat{R}_{\max}^{\text{pw}} = 1.050$ for `Nemotron-120B`) and **Figure 4(a)** (p. 9, bootstrap 95% CI extending to $1.5$). Allowing $R_{\max} > 1.0$ creates collinearity between $(R_{\max} - c)$ and $\lambda$ in the near-linear regime; enforcing $R_{\max} \in [c, 1.0]$ during optimization is recommended.

## [CRITICAL] Mathematical discrepancy in compute-adjusted scaling proxy slope (+482 vs. +1.170)  (p.10 "Regressing log10(t80 · N) on log10 N yields a slope of +482 ± 490 per decade – w")

By logarithm properties, $\log_{10}(t_{80} \cdot N) = \log_{10}(t_{80}) + \log_{10} N$, so the OLS slope of $\log_{10}(t_{80} \cdot N)$ on $\log_{10} N$ must equal $1 + \text{slope}(\log_{10} t_{80}, \log_{10} N)$. In **Table 20** (p. 19), the OLS slope of $\log_{10}(t_{80})$ on $\log_{10} N$ across the same 5 anchors is reported as $+0.170 \pm 0.254$ per decade. Consequently, the slope of $\log_{10}(t_{80} \cdot N)$ on $\log_{10} N$ must be $1 + 0.170 = +1.170$ per decade, not $+482 \pm 490$. A slope of $+482$ indicates that the unlogged product $t_{80} \cdot N$ was regressed on $\log_{10} N$ while being mislabeled in the text as $\log_{10}(t_{80} \cdot N)$.

**Suggested edit:**

```
Regressing t80 · N on log10 N yields a slope of +482 ± 490 per decade (while regressing log10(t80 · N) on log10 N yields a slope of +1.170 ± 0.254 per decade)
```

## [CRITICAL] Invalid functional-form and phase fitting on synthetically reconstructed traces  (p.30 "The driver scripts/scaling_law_iter33.py consumes the twelve-anchor summary expe")

In Section 4.3 (`scaling_law_iter33.py`), Section 4.4 (`scaling_law_iter37.py`), and Section 4.8 (`iter65`, pp. 39–41), the analyses do not operate on the raw per-step reward trajectories for the 12-anchor frontier set. Instead, they synthesize a per-step reward trace from six aggregate summary statistics (`mean`, `var`, `peak_step`, `early_mean`, `late_mean`, `zero_frac`). Furthermore, in Iteration 65 (Tables 42–44), three-segment OLS regressions with two changepoints (requiring $n \ge 6$ steps and fitting 3 slopes, 3 intercepts, and 2 changepoints) are fit to all 12 anchors, even though Table 32 (p. 34) indicates that five of the 12 anchors have $n_{\text{steps}} < 6$: `Qwen3-32B` ($n=3$), `Qwen3.5-27B` ($n=3$), `Qwen3-30B-MoE-Inst` ($n=3$), `Qwen3-235B-MoE` ($n=4$), and `Qwen3-30B-MoE` ($n=5$).

Reconstructing a time series of length $T \in [3, 30]$ from six aggregate summary statistics cannot recover the true step-by-step curvature, autocorrelation, or higher-order temporal structure of the empirical trajectory; any intermediate curvature in the reconstructed series is an artifact of the synthesis interpolation rule itself. Evaluating whether GRPO reward growth is exponential, Michaelis-Menten, Hill-shaped, or linear (Section 4.4), testing three-phase learning dynamics and $B=200$ bootstrap phase-label stability (Section 4.3), or fitting three-piece OLS segments in `iter65` measures the functional properties of the trace synthesizer rather than the empirical dynamics of GRPO. This explains why the linear baseline wins AIC on 9/12 anchors in `iter37` and 7/10 in `iter37b`, whereas the 10-trace raw per-step cross-check (`iter37c`, p. 32) inverts the result (linear wins 0/5 on GRPO). Because raw per-step telemetry is released for all runs, all trajectory-shape, phase-segmentation, and functional-form analyses in Sections 4.3–4.5 and 4.8 should either be recomputed directly on the raw per-step telemetry logs or restricted to anchors with full raw per-step traces.

**Suggested edit:**

```
Recompute the phase-classification, functional-form AIC/BIC, and piecewise-slope analyses in Sections 4.3, 4.4, and 4.8 directly on the raw per-step telemetry logs, and exclude anchors with $n_{\text{steps}} < 6$ from multi-segment regressions.
```

## [CRITICAL] Unphysical asymptotic ceiling estimates (R_max = 2.000) distort downstream regressions  (p.34 "Table 31: Stability of the saturation-model fit under trace truncation. max |∆λ|")

Because the GSM8K verifiable reward is binary correctness averaged over rollouts, per-step rewards and any physical asymptotic accuracy ceiling must satisfy $R_{\max} \in [0, 1]$ (as noted on p. 4 and in Table 37 on p. 36). In earlier tables (e.g., Tables 6, 10, 23, 26, and 27), `Nemotron-120B` has a fitted saturation ceiling of $R_{\max} = 0.182$ (with empirical mean reward $\bar{R} = 0.175$ and peak $0.875$). However, starting in Section 4.5 (Table 31) and continuing through Sections 4.6–4.7 (Table 32, p. 34; p. 35), the nonlinear least-squares solver uses an upper box constraint of $2.0$, assigning unphysical fitted ceilings to three models:

- `Nemotron-120B`: $R_{\max} = 2.0000$ (with $\lambda_{\text{full}} = 0.0105$ and $t_{80,\text{full}} = 153.40$, described on p. 35 as *"fit-saturating at the upper bound"*),

- `Qwen3-30B-MoE-Inst`: $R_{\max} = 1.002$ (Table 32),

- `Qwen3-235B-MoE`: $R_{\max} = 1.0017$ (Table 31) / $1.002$ (Table 32).

In addition, three short-horizon anchors ($n_{\text{steps}} \in \{3, 5\}$: `Qwen3-32B`, `Qwen3.5-27B`, `Qwen3-30B-MoE-Inst`) have `nan` saturation fits in Table 31, yet Table 32 populates numerical $R_{\max}$ values for them ($0.250$, $0.598$, and $1.002$) without explaining how those values were imputed when the `iter41` fit was `nan`.

This unphysical $R_{\max} = 2.000$ boundary artifact is propagated directly as the ground-truth dependent variable ($R_{\max}^{\text{actual}}$) into the Chinchilla iso-compute OLS regressions (`iter45`, Tables 32–34), the two-parameter iso-FLOP joint fit and leave-one-out (LOO) residual decomposition (`iter49`, Tables 35–37), and the rank-preservation audit (`iter53`, Section 4.8):
1. **Compute-Exponent Distortion (Tables 33 & 35):** In Table 32, the other five models labeled `dense` span $\log_{10} C \in [4.618, 5.090]$ with $R_{\max} \in [0.250, 0.869]$, while `Nemotron-120B` sits at the extreme upper end of compute ($\log_{10} C = 6.090$) with the unphysical $R_{\max} = 2.000$. This single $(6.090, 2.000)$ leverage point artificially drives the strong positive dense compute slope ($\alpha_{\text{dense}} = 1.030$ in Table 33; $\alpha_{\log C} = 0.980, R^2 = 0.779$ in Table 35).
2. **LOO Residual and Rank-Preservation Distortion (pp. 35–36):** On p. 35 and in Table 37 (Prediction P2), the headline finding that `Nemotron-120B` has an absolute LOO error $> 1.2$ units ($2.0 - 0.80$) penalizes a valid linear prediction ($0.80 \in [0, 1]$) against an unphysical solver bound ($2.0$). Similarly, in Section 4.8 (p. 36, "Rank preservation (P1)"), ranking the collapsed `Nemotron-120B` run (actual mean reward $0.175$) as the #1 highest-performing model in the benchmark solely because its diverging exponential fit hit $2.000$ invalidates the rank-correlation test ($\tau_b = 0.107$).
3. **Inconsistent Observed $R_{\max}$ Across Tables (p. 34 vs. p. 38):** In Table 40 (p. 38), `R_max obs` for the exact same `iter49` model changes without explanation to `0.281` for `Nemotron-120B` (with `R_max pred = 0.928` and `|residual| = 0.646`, contradicting Table 36's `1.202`), `0.375` for `Qwen3-8B` (vs. `0.331` in Table 32), `0.312` for `Qwen3-32B` (vs. `0.250`), `0.450` for `Qwen3-30B-MoE` (vs. `0.333`), and `1.000` for the top anchors.

**Suggested edit:**

```
Constrain $R_{\max} \in [0, 1]$ during optimization, document how short-horizon ($n_{\text{steps}} \le 5$) $R_{\max}$ values are imputed, and recompute Tables 31–40 using a single consistent $R_{\max}$ vector.
```

## [CRITICAL] Re-introduction of withdrawn 'dense' misclassification for Nemotron-120B and total-parameter FLOP inflation  (p.34 "Table 32: Per-anchor compute proxy and fitted Rmax.")

Earlier in the manuscript (Abstract p. 1 and Table 12 on p. 12), the authors explicitly documented and withdrew an architecture classification defect: `Nemotron-3-Super-120B-A12B` is a **LatentMoE** model (12B active parameters), not a dense model, which changed the 12-anchor architecture split from 6-vs-6 to **7 MoE vs. 5 dense** and led to the formal withdrawal of the architecture stratification claim.

However, throughout Sections 4.3, 4.6, 4.7, and 4.8 (Iterations 33, 45, 49, 61, and 65), the tables and statistical tests revert to the withdrawn 6-vs-6 split with `Nemotron-120B` misclassified as `dense`:

- **Table 32 (p. 34) and Table 38 (p. 37):** Both tables explicitly list `Nemotron-120B` with `arch = dense`.

- **Table 33 (p. 34) and Table 35 (p. 35):** The within-stack compute regressions stratify the 12 anchors into `dense` ($n=6$) and `moe` ($n=6$). Placing `Nemotron-120B` (and its $R_{\max} = 2.000$ outlier) into the `dense` stack directly confounds the headline comparison between $\alpha_{\text{dense}} = 1.030$ and $\alpha_{\text{MoE}} = 0.057$.

- **Total vs. Active Parameters in Compute Proxy:** The compute proxy $C_{\text{proxy}} = P \cdot n_{\text{steps}} \cdot L_{\text{bar}}$ in Table 32 uses *total* parameters $P$ for MoE models ($685\text{B}$ for DeepSeek-V3.1, $1000\text{B}$ for Kimi-K2, $235\text{B}$ for Qwen3-235B-MoE) rather than *active* parameters per token ($\sim 37\text{B}, 32\text{B}, 22\text{B}$), inflating MoE per-token FLOPs by an order of magnitude.

- **Section 4.3 (p. 31) and Table 44 (p. 41):** Both stratify the 12 anchors into `dense (n=6)` and `moe (n=6)`. On p. 31, the text concludes that *"no MoE model collapses"* because the collapsing `Nemotron-120B` run was placed in the `dense` group; under the corrected LatentMoE classification from Table 12, the sole collapse anchor is in fact an MoE model.

All cross-architecture analyses in Sections 4.3–4.8 should be updated to use the corrected 7 MoE / 5 dense split and active parameter counts for MoE compute proxies.

**Suggested edit:**

```
Reclassify `Nemotron-120B` as `moe` (7 MoE / 5 dense) across Tables 32, 33, 35, 38, and 44 and Section 4.3, and compute $C_{\text{proxy}}$ using active parameters per token for MoE architectures.
```

## [CRITICAL] Unverified architectural claims and algorithmic mechanisms in Section 7  (p.46 "Traditional RL post-training often relies on superficial baseline subtraction wi")

Section 7 is appended *after* the References section (pp. 45–47) and asserts a series of implemented architectural mechanisms and dynamic mitigation rules that contradict or have no basis in the experimental setup, methodology, or empirical analyses presented in Sections 1–6:

- **Section 7.1** states that *"P1 introduces closed-loop control dynamics to stabilize policy gradients,"* specifically employing *"streaming token-level credit assignment, and non-parametric quantile filtering,"* and trading *"marginal rollout compute ($G = 4 \to G = 16$) for guaranteed variance non-degeneracy."*

- **Section 7.2** specifies three *"AntiVibe Mitigation Rules"*: (i) *"Dynamic group scaling $G \leftarrow \min(2G, 64)$ or prompt injection"* for ZVF collapse, (ii) *"Length-normalized advantage centering $\tilde{A}_i = A_i / \sqrt{L_i}$"* for length bias distortion, and (iii) *"Adaptive reward scaling sentinel with automatic learning-rate boost"* for gradient starvation.

None of these dynamic control loops, length-normalized advantages, or adaptive learning-rate schedules appear anywhere in the experimental setup (Section 3, which specifies standard static-hyperparameter GRPO and PPO runs with fixed learning rates of $10^{-4}$ or $10^{-5}$ in Section 3.2, Table 3, and standard group-normalized advantages $\hat{A}_i = (r_i - \bar{r})/\sigma_r$ in Figure 2) or in the scaling analyses (Section 4). On the contrary, Sections 4 and 5 explicitly treat high zero-variance fractions (ZVF) and static group sizes as *unmitigated* empirical confounders observed in the static traces (e.g., Table 38, p. 37; p. 42; p. 43) rather than conditions actively resolved by a closed-loop controller.

**Recommendation:** Remove Section 7 and Figures 17–26 entirely if they are automated rubric boilerplate or artifacts from an external codebase generator (`mohi-devhub/antivibe`). If any of these mechanisms are intended as conceptual proposals for future work rather than implemented components of the current benchmark, relocate them to a dedicated future-work subsection or appendix and explicitly label them as proposed designs rather than implemented features.

## [CRITICAL] Unverified automated verification gates contradicting single-seed design  (p.46 "To audit this implementation against vibe-coding, every claim in GRPO Scaling La")

Section 7.3 asserts that every claim in the audit must satisfy *"(1) Seed-matched baseline reproducibility within $\pm 1\%$, (2) Monotonic ZVF reduction under adaptive control, and (3) Zero unhandled numerical exceptions during zero-variance rollout states."* However, Sections 1, 3, and 5 explicitly emphasize that the frontier scaling runs are single-seed ($n = 1$) descriptive runs where seed-matched reproducibility cannot be computed, and no adaptive ZVF control experiment is reported anywhere in the paper. This section should be removed or explicitly reframed as future verification criteria.

## [MAJOR] Discrepancy between 'cross-library' GRPO framing and actual library coverage  (p.1 "a benchmark spanning 70+ runs across seven RL libraries and five model families ")

There is a discrepancy between the paper's Title (*"Limits of Scaling Laws for GRPO Post-Training: A Cross-Library, Cross-Scale Audit"*), Abstract, and Section 1—which frame this study as a cross-library audit of GRPO scaling across *"seven RL libraries"*—and the operational coverage described in Section 2.2 (Page 3). Section 1.1 further motivates the study by noting that prior RL scaling work is conducted *"typically within a single stack, leaving open whether reported trends transfer across libraries and model families."* However, Section 2.2 clarifies that there are two separate seven-library rosters, neither of which evaluates cross-scale GRPO across seven libraries:

1. The first roster (Table 2: TRL, Stable Baselines3, CleanRL, Tianshou, PufferLib, `rl_games`, and `d3rlpy`) consists of only **one** LLM-native framework that implements GRPO (HuggingFace TRL), alongside five classic online PPO libraries and one offline CQL library (`d3rlpy`). As noted in Section 2.2 and Page 6 (Table 5), this roster is used for a small-scale synthetic arithmetic task ablation underlying claim $F_3$, rather than cross-scale LLM GRPO post-training.
2. The second roster (Tinker, TRL, SkyRL, verl, OpenRLHF, Atropos, and HF reference launchers) targets LLM post-training launchers, but Section 2.2 explicitly states that *"only Tinker and TRL produced completed runs in this release; veRL and OpenRLHF entries in the `framework_comparison.json` artefact are dry-run placeholders."*
3. Furthermore, the core cross-scale GRPO scaling analyses in Section 4 (both the 5-anchor and 12-anchor pools) are executed entirely within a single managed API backend (Tinker).

To prevent readers from misinterpreting the cross-scale GRPO audit as having been replicated across seven LLM RL frameworks, consider revising the Title, Abstract, and Section 1 to state upfront that the 7-library comparison applies to the classic-RL/arithmetic baseline ($F_3$), whereas the LLM post-training runs were completed on two backends (TRL and the managed Tinker API) and the cross-scale frontier GRPO audit is conducted within a single backend (Tinker).

## [MAJOR] Inconsistent task-suite descriptions across Abstract, Section 1, Table 1, and Figure 1  (p.2 "Table 1: Benchmark task suite with standardized reward functions.")

There is a consistency gap across the task descriptions on Pages 1–3:

- The Abstract and Section 1 state that the benchmark studies GRPO scaling across 70+ runs *"on GSM8K, HumanEval, and synthetic tool-use tasks."*

- However, Figure 1 (Page 3) indicates that only `GSM8K` (*"8 models tested ✓ Complete"*) is complete, whereas `Function Calling` (*"2 models tested ∼ Partial"*) is marked as in progress and `HumanEval` (*"1 model ◦ Partial"*) is marked with the symbol `◦` (which the legend defines as *"Planned"*).

- Simultaneously, Table 1 (*"Benchmark task suite with standardized reward functions"*, Page 2) lists a different set of tasks and methods—`Math RL (Arithmetic)` with GRPO, `Math RL (GSM8K)` with GRPO, `Chat SFT` (NoRobots), `Preference (Shorter)` with DPO, and Off-/On-Policy `Distillation` (SFT / IS Loss)—while omitting both `HumanEval` (Code Generation) and `Tool Use / Function Calling` entirely.

- Later in the document (Section 4), all cross-scale GRPO scaling analyses are performed exclusively on GSM8K traces.

Consider clarifying in the Abstract and Section 1 that while the overarching `TINKERRL-BENCH` suite encompasses synthetic arithmetic, GSM8K, SFT/DPO/distillation, and partial/planned probes on HumanEval and tool use, the empirical cross-scale GRPO scaling audit in this manuscript is evaluated on GSM8K (with classic-RL library ablations on synthetic arithmetic). Additionally, reconciling Table 1 and Figure 1 so that both reflect a consistent task taxonomy would substantially improve clarity.

## [MAJOR] Nemotron-120B misclassified as dense in Section 3.1  (p.4 "Frontier scale (70B–∼671B total / ∼37B active). Qwen3-235B-A22B (MoE, 22B active")

On page 4, under *Frontier scale* and *Model family coverage*, Nemotron-120B is explicitly labeled as `"Nemotron-120B (120B dense)"` and `"NVIDIA's dense frontier model."` This directly contradicts the Abstract (p. 1, `"Nemotron-120B LatentMoE"`) and Table 12 (p. 12), which explicitly withdraws an earlier MoE-vs.-dense stratification claim on the grounds that `"Nemotron-3-Super-120B-A12B is LatentMoE (12B active), not dense; it was the sole collapse anchor in the 'dense' set"` and notes that an earlier draft misclassified it as dense. Updating Section 3.1 to reflect the LatentMoE architecture (as well as correcting the legacy `"dense"` labels that persist in later tables such as Tables 32, 33, 35, and 38) is necessary for internal consistency.

**Suggested edit:**

```
Frontier scale (70B–1T total / ~37B active). Qwen3-235B-A22B (MoE, 22B active), DeepSeek-V3.1 (~671B total / ~37B active MoE), Nemotron-120B (120B LatentMoE, 12B active), Kimi-K2 (1T total / 32B active MoE).
```

## [MAJOR] Kimi-K2 scale tier placement and total parameter range inconsistency  (p.4 "Mid scale (14B–32B). Qwen3-14B, Qwen3-30B-A3B (MoE), Qwen3-32B, Qwen3.5-27B, GPT")

Section 3.1 states that the roster spans `"0.6B to ~671B total parameters (up to ~37B active for MoE checkpoints)"` and places `"Kimi-K2 (selected scale points)"` under *Mid scale (14B–32B)* while omitting it from the *Frontier scale* paragraph. Elsewhere in the manuscript (e.g., Abstract, p. 1; Section 1, p. 2; Section 4, p. 11; Table 32, p. 34), the benchmark range is stated as `"0.6B–1T parameters"` with Kimi-K2 / Kimi-K2-Thinking treated as a 1-trillion-parameter (`1000.0B` total, ~32B active) frontier MoE anchor. Clarifying whether the 14B–32B placement refers to Kimi-K2's 32B active parameter count rather than its 1T total parameter count, moving it to the frontier MoE tier, and updating the total parameter upper bound in Section 3.1 from `~671B` to `1T` would resolve this ambiguity.

**Suggested edit:**

```
Mid scale (14B–32B). Qwen3-14B, Qwen3-30B-A3B (MoE), Qwen3-32B, Qwen3.5-27B, GPT-OSS-20b.
```

## [MAJOR] Empty Section 2.4 and unexplained 1/199 arithmetic baseline bound  (p.3 "2.4 Reward Verification
2.5 Baselines: Floor and Ceiling
Following best practice")

Section 2.4 (`"2.4 Reward Verification"`) consists solely of a bare section heading with no body text, and is immediately followed by Section 2.5 (`"2.5 Baselines: Floor and Ceiling"`), which states: *"Floor (random baseline): Uniform random action selection yields $\frac{1}{199} \approx 0.50\%$ accuracy on the arithmetic task."* Because the formulation of the generated arithmetic task (e.g., two-digit addition with an action space of $[-99, 99]$ or $[0, 198]$ yielding 199 discrete outcomes) and the reward verification protocol are not described in Section 2.1 or Section 2.4—and because the paper later argues in Section 4.1 (pp. 27–28) and Section 4.3 (p. 31) that reward-parser formatting and Agent–Computer Interface (ACI) constraints are a first-order determinant of measured capability ceilings (such as for Qwen3-8B) and the root cause of the Nemotron-120B collapse, while Figure 2 highlights verifiable rewards as a core design pillar—omitting the specification of the reward verifier, answer-extraction rules, and the 199-choice arithmetic action space leaves a critical implementation gap. Furthermore, Section 2.5 defines Floor, Ceiling, and Human (`~99.5%`) baselines exclusively for two-digit addition, whereas the scaling-law audit in Section 4 is conducted on GSM8K. Populate Section 2.4 with the verifier specifications and include GSM8K baseline bounds in Section 2.5.

## [MAJOR] Missing GRPO hyperparameters in Table 3 and Section 3.2  (p.4 "Table 3: Hyperparameter mapping across libraries (PPO defaults).")

Section 2.3 and Table 3 only document hyperparameter mappings for *PPO* defaults (`Clip range` $\epsilon = 0.2$, `Entropy coeff.` $0.01$, `GAE` $\lambda = 0.95$, `Discount` $\gamma = 0.99$, `Target KL` $0.01$) across six RL libraries (excluding Tinker). Neither Section 2.3 nor Section 3.2 specifies the core GRPO-specific hyperparameters required to reproduce the paper's central LLM scaling experiments—most notably the group size $G$ (which has to be inferred retroactively in Section 4, p. 16, as $n=16$ or $n=8$ from the least common multiple of reward denominators), rollout temperature, maximum generation token length, KL penalty coefficient $\beta$, or number of prompts per step. Furthermore, Section 3.2 (p. 5) defers per-run configurations for Tinker/API, Task-4, Wave-6, and frontier case studies to a `"framework-configs appendix"` that is not included in the manuscript.

## [MAJOR] Confounded marginal one-way ANOVAs where eta-squared values sum to 1.897  (p.7 "Variance decomposition. One-way ANOVAs across 42 experiments with valid performa")

Under *Variance decomposition* (Section 3.3, p. 7), the authors report separate one-way ANOVAs across 42 experiments with valid performance observations yielding $\eta^2 = 0.546$ for library/framework, $\eta^2 = 0.558$ for training algorithm, $\eta^2 = 0.471$ for model family, and $\eta^2 = 0.322$ for model scale. Because these four $\eta^2$ values sum to $1.897 > 1.0$, they are computed from four separate marginal one-way ANOVAs on an unbalanced, non-orthogonal experimental grid where factors are heavily collinear (for instance, Table 2 on p. 3 shows that classic RL libraries only run PPO or CQL on the arithmetic task, while frontier LLM families exclusively run GRPO via the Tinker API on GSM8K). Marginal one-way $\eta^2$ values in a confounded design absorb shared variance across collinear factors and cannot isolate the unique variance contribution of library/algorithm versus model scale. Replacing or supplementing these marginal figures with a multi-factor ANOVA (e.g., Type II/III partial $\eta^2$ or sequential variance partitioning) and explicitly noting the collinearity of the experimental design is needed to support the claim that library and algorithm dominate model scale.

## [MAJOR] Undefined single-seed test p-values, missing Tier-A/B table, and undefined dagger in Table 5  (p.6 "Table 5: Legacy 20-test BH table – audit trail only.")

There are three issues regarding Table 5 (`"Legacy 20-test BH table – audit trail only"`) and the multiple-comparison reporting in Section 3.3:
1. **Undefined Single-Seed Test Statistics:** Table 5 reports 20 hypothesis tests with finite, extreme raw $p$-values down to $2.357 \times 10^{-21}$ (Rank 1, Dense vs. MoE Architecture Welch $t$-test) and $3.924 \times 10^{-10}$ (Rank 2, PPO vs. GRPO on Llama-3.1-8B Welch $t$-test). While the caption disclaims that this table predates the Tier-A/B/C partition and `"silently includes single-seed Tinker rows whose Welch statistics are mathematically undefined (n1=n2=1)"`, if $n_1=n_2=1$, a two-sample Welch $t$-test has zero degrees of freedom and returns `NaN` (division by zero) rather than a finite numeric $p$-value; the reported extreme $p$-values indicate that the legacy script likely pooled non-independent per-step reward telemetry ($T=20\text{--}30$ steps) or per-problem rollouts as independent pseudo-replicates.
2. **Missing Authoritative Restricted Tier-A/B Table and Unexplained Markers:** The text on page 6 states that `"The headline BH result the paper now stands behind is the restricted Tier-A/B family computed in the addendum, where only F3 survives"` and page 5 states that results across `"5 planned tests"` surviving Bonferroni are marked $^*$ and those surviving BH–FDR are marked $^\dagger$. However, the 5 planned tests and the authoritative restricted Tier-A/B test table (including the specific test statistic, degrees of freedom, sample size, and $p$-value for $F_3$) are not presented in Section 3.3, and the markers $^*$ and $^\dagger$ are not used in Table 5. Replacing Table 5 in the main text with the authoritative restricted Tier-A/B test table (and moving Table 5 to an appendix with an explicit note on what observational unit generated those legacy numbers) would make the statistical methodology self-contained.
3. **Undefined Double Dagger (`‡`):** Ranks 2, 6, 8, 9, and 20 in Table 5 are marked with a double dagger (`‡`), which is never defined in the table caption or body text.

## [MAJOR] Withdrawn MoE-vs.-dense stratification still presented as positive headline and reused downstream  (p.12 "The headline positive result of the extension is on the architecture axis. Strat")

Because Section 4 is structured as an append-only sequence of iteration logs, withdrawn claims remain presented as positive results in the main text and downstream tables:
1. Immediately *below* **Table 12** (which is titled *"Withdrawn MoE-vs-dense stratification"* and states *"Disposition: Withdrawn. Reclassification removes the architecture headline"* because `Nemotron-3-Super-120B-A12B` is a LatentMoE model rather than dense, changing the one-sided permutation $p$-value to $\approx 0.178$), this paragraph still begins: *"The headline positive result of the extension is on the architecture axis."* before stating in the final sentence that an earlier draft misclassified Nemotron and withdrawing the split.
2. **Figure 5(c)** (p. 13) still plots the withdrawn 6-vs-6 boxplot ($+0.338, p = 0.023$).
3. Despite the reclassification into 7 MoE and 5 dense models in Table 12, **Iter 21** on p. 15 (Paragraph H) and **Table 17** (p. 17) revert to reporting Levene's median test *"between MoE ($n=6$) and dense ($n=6$)"*, re-introducing the misclassified partition.

**Actionable Recommendation:** Consolidate the chronological `iter` logs into a unified, self-consistent narrative where superseded intermediate claims and withdrawn splits are either revised using the corrected 7-vs-5 classification or moved to a historical audit appendix.

## [MAJOR] Conflicting claims across iterations on whether Rmax scales with log10 N  (p.17 "This is a constructive correction, not a nihilistic one. It relocates Pillar 1’s")

There is a direct contradiction across iterations regarding whether the endpoint ceiling $R_{\max}$ scales positively with $\log_{10} N$:

- **Page 7** (opening paragraph of Section 4), **pp. 15–16** (Iter 21, Paragraph H: *"The OLS slope of $R_{\max}$ on $\log_{10} N$ is positive (estimate), with a negative intercept. Operationally, $R_{\max}$ obeys a weak power law in $\log_{10} N$"*), and **p. 17** (Iter 25, anchored here) assert that the reliable scaling structure lives in the positive cross-scale endpoint regression $R_{\max} \sim \log_{10} N$.

- Conversely, **Iter 117** (p. 18: *"the iter-105 $R_{\max}$-vs-$N$ failure: no cross-scale scaling signal survives"*), **Iter 121** (p. 20: *"no scaling signal survives on the five-anchor pool"*), and **Iter 137** (p. 25, Subsection (b): *"regression of $R_{\max}$ on $\log_{10} N$ gives $b = -0.172 \pm 0.198$ ... GRPO saturation is real ... but its $t_{80}$, $\lambda$, and $R_{\max}$ do not scale with parameter count"*) state the exact opposite.

**Actionable Recommendation:** Reconcile the earlier Section 4 intro, Iter 21, and Iter 25 conclusions with the negative findings of Iters 117, 121, and 137 so the manuscript presents a consistent position on $R_{\max}$ scaling.

## [MAJOR] Contradictory heuristic phase labels and three-phase definitions across iterations  (p.15 "Table 16: Pair-wise Cohen’s κ on phase labels. On the 5-anchor set the heuristic")

There are multiple internal inconsistencies across iterations regarding phase classifications and the definition of the Nimmaturi et al. (`arXiv:2507.18014`) three-phase template:
1. On **p. 8** (Table 6 and "Three-phase learning dynamics"), the heuristic classifier assigns the 5 anchors to 2 `plateau`, 1 `saturation`, 1 `drift`, and 1 `collapse`. Yet **Table 16**'s caption (p. 15) states that *"On the 5-anchor set the heuristic and AIC methods agree perfectly ($\kappa = 1.0$) because both label all five traces as 'plateau'"*.
2. On **pp. 7–8** and in **Iter 117** (Table 19, p. 18), the Nimmaturi et al. [18] three-phase template is defined as *"slow start $\to$ rapid improvement $\to$ plateau"* with $0/5$ anchors passing (and `Qwen3-8B` labeled `plateau` at best $k = 1$ with no slow-start phase). However, in **Iter 125** (p. 21, citing duplicate bibliography entry [17] for the same arXiv preprint) and **Iter 133(b)** (p. 23), the Nimmaturi template is redefined as *"rapid improvement $\to$ plateau $\to$ collapse"*, under which $1/5$ (p. 21) and $2/7$ (p. 23, `Qwen3-8B` and `gpt-oss-20B`) are reported as satisfying the full signature.
3. In **Figure 9(B)**'s caption (p. 19), it states *"only Nemotron-120B's three segments satisfy the nimmaturi monotone-up criterion and even then it is a rise-then-fall (collapse)"*, which is self-contradictory because a rise-then-fall sequence $\{0.000, 0.875, 0.154\}$ is not monotonically increasing (as correctly noted in Table 19 on p. 18).

## [MAJOR] Clarify boundary clamping in AIC and continuity/parameter counting in piecewise model (Eq. 5)  (p.21 "R(t) = 
Rmax(1 − e−λt) t ≤ tpeak,
R(tpeak) − γ(t − tpeak) t > tpeak,
(5)")

In Equation (5) (Iter 129), please clarify:
1. Whether $R(t_{\text{peak}})$ in the second branch denotes the fitted value from the first branch $R_{\max}(1 - e^{-\lambda t_{\text{peak}}})$ (enforcing continuity at $t_{\text{peak}}$) or the empirical peak reward $\hat{R}_{\text{peak}}$.
2. Whether selecting $t_{\text{peak}}$ from the empirical trace is counted as an estimated parameter in the $\text{AIC}_c$ degrees of freedom.

Additionally, in **Table 13** (p. 14) and **Table 23** (p. 22), when $\lambda$ is clamped at the upper boundary $\lambda = 10$ (4 of 5 traces), the saturation model collapses to a constant step function for $t \ge 1$ and is assigned $k = 2$ with $\Delta\text{AIC} = 2.00$ relative to the constant model ($k = 1$); note that standard regularity conditions for AIC require an interior maximum likelihood estimate rather than a boundary-clamped parameter.

## [MAJOR] Inverted sign in planted scaling law for synthetic calibration (Test 3 and Table 22)  (p.20 "We plant a known Chinchilla-style law Rmax(N) = 0.85 − β log10(N/8) across a fre")

Here in **Test 3** (and in the caption of **Table 22** on p. 21), the planted Chinchilla-style scaling law for the synthetic recovery simulation is written with a minus sign as $R_{\max}(N) = 0.85 - \beta \log_{10}(N/8)$ for $\beta > 0$ ($\beta \in \{0.010, \dots, 0.200\}$). With a minus sign and $\beta > 0$, this equation models the asymptotic reward ceiling $R_{\max}$ *decreasing* with model size $N$ (and yields negative $R_{\max}$ for large $N$ when $\beta = 0.20$), whereas a positive scaling law for reward ceiling would have $R_{\max}$ increasing with $N$ (or error $1 - R_{\max}$ decreasing with $N$). Clarify whether a plus sign was intended or if the synthetic law intentionally planted a negative slope.

**Suggested edit:**

```
We plant a known Chinchilla-style law Rmax(N) = 0.85 + β log10(N/8) across a fresh pool of nanchors ∈ {5, 8, 12, 20, 40}
```

## [MAJOR] Arithmetic mismatch between 'n = 10' pool label and n = 11 sample count  (p.22 "Iter 133 takes the natural sequel step: re-run the full structural suite (monoto")

The text states that the structural suite is run at $n \in \{5, 7, 10, 12\}$, explaining that *"$n = 7$ adds the gpt-oss-20B and Kimi-K2-Thinking long-trace anchors from the iter-13 frontier pool, $n = 10$ also adds four short probes ($n \le 5$ steps, mean-reward proxy for $R_{\max}$), and $n = 12$ includes all available anchors."* Arithmetically, adding 4 short probes to $n = 7$ yields $7 + 4 = 11$ anchors, not 10. Consistently, **Table 24** (p. 23) and **Table 25** (p. 24) label the third row `n = 10 (+ short probes)` in the `Pool` column, but list `11` in the sample-size column `n` and report `11/11` in the `LOOCV agreement` column, while the surrounding prose on pp. 23–24 refers to this pool as $n = 10$. Similarly, on **p. 11** ("Elevation: extended frontier+MoE scaling (iter 13)"), the text lists only 6 additions to the 5 core anchors (totaling 11 models), omitting `Qwen3-30B-MoE-Inst` which appears on p. 12 and in Figure 5(d). Reconcile the pool labels, sample counts, and complete 12-anchor roster.

**Suggested edit:**

```
re-run the full structural suite (monotonicity, threephase, bimodality, AICc, LOOCV) at n ∈ {5, 7, 11, 12}, where n = 7 adds the gptoss-20B and Kimi-K2-Thinking long-trace anchors from the iter-13 frontier pool, n = 11 also adds four short probes (n≤5 steps, mean-reward proxy for Rmax), and n = 12 includes all available anchors.
```

## [MAJOR] Contradictory 7/12 count for both saturated and unsaturated subsets in Iter 21  (p.15 "on the 7/12 unsaturated anchors. The interaction coefficient δ is the sharpest t")

On **p. 14**, Paragraph (E) states that *"of the 12-anchor frontier, 7 of 12 traces hit this bound ($\lambda \le 10$)... the rate constant $\lambda$ is non-identified on the 7/12 of the frontier that has already saturated"*, which is repeated in **Table 17** (p. 17, Axis E: `# traces with \hat{\lambda} >= 10: 7/12`). Immediately following on **p. 15**, Paragraph (F) states that Equation (3) is fit *"on the 7/12 unsaturated anchors."* If 7 of 12 traces hit the upper bound and are saturated, only $12 - 7 = 5$ anchors can be unsaturated (or vice versa). Additionally, line 3 of Paragraph (F) and line 4 of Paragraph (H) contain unedited template placeholders (`"\hat{\delta} = (estimated)"` and `"is positive (estimate)"`) where numerical values are missing.

## [MAJOR] Contradictory collapse classifications and post-peak slope signs across iterations 33, 49, and 65  (p.31 "• 0/12 traces have all three criteria (Nemotron-120B has 2/3 but its post-peak s")

There are multiple internal contradictions regarding which model collapses and the sign of `Nemotron-120B`'s post-peak slope across Sections 4.3, 4.7, and 4.8:
1. **Collapse Identity (`iter33` vs. `iter49` vs. `iter65`):** In Section 4.3 (p. 30, *Interpretation*), the text states that under the `iter33` classifier, *"3/12 traces are drifts (Llama-3.1-8B-Instruct, Qwen3-32B, Kimi-K2-Thinking) and 1/12 is a collapse (Nemotron-120B)."* Yet on p. 31 (*Nemotron-120B collapse mechanism*), the text states that under the three candidate collapse criteria, *"1/12 traces meet all three: only Qwen3.5-27B."* Then, in Section 4.7 (Table 36 and text, p. 35), the text states: *"The phase classifier in iter33 labels only Qwen3.5-27B as 'collapse'"* ($n=1$, with residual $0.004$), while moving `Nemotron-120B` into `plateau` (where `max |r| = 1.202` appears). Finally, in Table 42 (p. 40), the `iter65` three-piece OLS classifier reports `drift = 0` and `collapse = 0` across all 12 anchors, labeling `Nemotron-120B` as `valley`.
2. **Post-Peak Slope Sign:** Earlier in Table 10 (p. 11), `Nemotron-120B` peaks at step 3 ($0.875$) with a negative post-peak OLS slope of $-0.0036$/step. Here on p. 31, the text claims the opposite: *"its post-peak slope is positive: the recovery from zero yields $\Delta > 0$... showing Nemotron's post-peak decay slope is positive (it does not collapse during training; it collapses before training starts and slowly recovers)."*

**Suggested edit:**

```
Reconcile the definition and assignment of the `collapse` class across `iter10`, `iter33`, `iter49`, and `iter65`, and verify whether `Nemotron-120B`'s post-peak slope is $-0.0036$/step (as in Table 10 on raw telemetry) or positive (an artifact of synthetic trace reconstruction in `iter33`).
```

## [MAJOR] Conclusion of iter 65 directly contradicts Table 42, Prediction P2, and m_3 slope  (p.41 "What iter 65 proves. The threephase hypothesis is geometrically falsified: only ")

In Iteration 65 (Table 42, p. 40), the caption states: *"No anchor is classified as collapse or drift: Nemotron-120B, despite its late-trace decay, is classified valley ($n_{\text{collapse}} = 0$),"* and on p. 40 (*Per-phase slopes*) the text notes that for `Nemotron-120B`, *"m3 = +0.0000 reflects the late-trace partial recovery after the early collapse."* Likewise, prediction `P2_nemotron_classified_collapse` on p. 40 is marked `FAIL (nemotron_class=valley, n_collapse=0)`.

However, this concluding paragraph on p. 41 asserts the exact opposite: *"the falsification is concentrated on the collapse and drift classes. Nemotron-120B is the unique collapse anchor (peak-then-decay shape, $m_1 > 0, m_3 < 0$)"*—directly contradicting Table 42 ($n_{\text{collapse}} = 0, n_{\text{drift}} = 0$), prediction P2, and the $m_3 = +0.0000$ slope reported on p. 40.

**Suggested edit:**

```
What iter 65 proves. The threephase hypothesis is geometrically falsified: only 2/12 anchors have the canonical (slow, fast, plateau) OLS-slope pattern, and the plurality (6/12, including Nemotron-120B) fall into the valley class (early dip with partial late recovery, $n_{\text{collapse}} = 0, n_{\text{drift}} = 0$).
```

## [MAJOR] Table 43 sorting, |m_3| threshold, and phase-class label mismatches  (p.40 "Table 43: Top-6 traces by |m3|. Only anchors with strong late-trace drift appear")

Table 43 and its accompanying text on p. 40 contain several internal discrepancies:
1. **Sorting and Threshold Mismatch:** Table 43 is captioned *"Top-6 traces by $|m_3|$..."* and the text states *"Five of the six have $|m_3| \ge 0.005$."* However, the rows are not sorted monotonically by $|m_3|$ (`+0.0875`, `-0.0620`, `-0.0250`, `-0.0054`, `+0.0054`, `+0.0037`), and `Qwen3.5-4B` has $|m_3| = 0.0037 < 0.005$.
2. **Phase-Class Label Contradictions:** The first row (`DeepSeek-V3.1`) is explicitly labeled `three-phase` despite having $m_2 = -0.0125$ and $m_3 = +0.0875$, which violates two of the three canonical three-phase slope targets in Equation 9 ($|m_2| \ge 0.040$ and $|m_3| < 0.015$). Similarly, `Qwen3-30B-MoE` is labeled `monotonic-rising` in Table 43 despite having strongly negative slopes $m_2 = -0.3760$ and $m_3 = -0.0620$.

**Suggested edit:**

```
Verify the phase-classification logic in `scaling_law_iter65.py` so that `threephase` requires satisfying Eq. (9) ($|m_1| < 0.020, m_2 \ge 0.040, |m_3| < 0.015$) and `monotonic-rising` requires non-negative segment slopes, and sort Table 43 strictly by $|m_3|$.
```

## [MAJOR] Impossible negative PCI range and inverted verdict on P3_pci_below_3_for_most  (p.39 "PCI = X
3
k=1
⊮[constraintk] · (1 + 0.5 · wk),
with wk ∈ [0, 1] the proximity-to")

There are two mathematical issues with the Phase-Conformity Index (PCI) definition and prediction check:
1. **Formula Range (p. 39):** PCI is defined as $\text{PCI} = \sum_{k=1}^3 \mathbb{1}[\text{constraint}_k] \cdot (1 + 0.5 \cdot w_k)$ with $w_k \in [0, 1]$. Because the indicator $\mathbb{1}[\text{constraint}_k] \in \{0, 1\}$ and $1 + 0.5 w_k \ge 1$, a trace failing all three constraints has $\mathbb{1}[\text{constraint}_k] = 0$ for all $k$ and therefore must have $\text{PCI} = 0$, not $\text{PCI} \in [-1.5, 0]$ as stated in the text.
2. **Inverted Verdict on P3 (p. 40):** On p. 40, `P3_pci_below_3_for_most` is marked `FAIL (pci_mean=1.667)` even though all 12 traces in `all_pci=[1.0, 1.5, 1.0, 2.5, 2.5, 1.5, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0]` are strictly below $3.0$ (12/12 have $\text{PCI} < 3$, with mean $1.667 < 3$).

**Suggested edit:**

```
with $w_k \in [0, 1]$ the proximity-to-bound weight. A trace matching all three constraints has $\text{PCI} \in [3.0, 4.5]$ (or $3.0$ if normalized); a trace failing all three has $\text{PCI} = 0$.
```

## [MAJOR] Confounding of non-stationary training steps with Pass@1 CIs and Qwen3-8B mean mismatch  (p.27 "Table 27: Per-anchor Pass@K=1 bootstrap intervals. pˆ is the step-level Pass@1 e")

In Section 4.1 (p. 27), 95% confidence intervals on Pass@$K=1$ are computed by *"treating each anchor's per-step training rewards ($n = 20\text{--}30$ steps) as its Pass@$K=1$ sample"* and bootstrapping across steps. This raises three issues:
1. **Non-Stationarity Confounding:** Because policy weights are updated across online RL steps $t = 1, \dots, T$ and each step's reward is already an average over $n_{\text{eff}} \in \{8, 16\}$ rollouts on training prompts (p. 16), resampling across training steps conflates non-stationary learning dynamics and batch-to-batch variance with static evaluation-time Pass@1 sampling uncertainty on a fixed checkpoint (Miller, 2024 [15]; Chen et al., 2021 [4]).
2. **Estimand Mismatch:** Comparing the CI width of the *full-trajectory mean* $\hat{p}$ against the point gaps of the *asymptotic ceiling* $\hat{R}_{\max}^{2p}$ compares two different estimands.
3. **Numerical Contradiction for `Qwen3-8B`:** In Table 27, the step-level Pass@1 estimate $\hat{p}$ (the mean over the 30-step training rewards) for `Qwen3-8B` is reported as $0.416$ with a 95% CI of $[0.295, 0.541]$, whereas in Table 6 (p. 8), Table 13 (p. 14), and Table 38 (p. 37), the exact same 30-step trace for `Qwen3-8B` has a mean reward $\bar{R} = 0.285$—which falls completely outside the reported $[0.295, 0.541]$ bootstrap CI of its own mean.

**Suggested edit:**

```
Verify why `Qwen3-8B` has $\hat{p} = 0.416$ in Table 27 versus $\bar{R} = 0.285$ in Tables 6, 13, and 38, and clarify that within-trace step bootstrapping measures trajectory-mean variability across training batches rather than held-out Pass@1 evaluation uncertainty.
```

## [MAJOR] Circular ACI proxy decomposition inflates low-performing models via endogenous RQS  (p.28 "A proxy decomposition of the observed ceiling by ACI quality, Rpolicy max = Robs")

The authors decompose the observed ceiling by Agent–Computer Interface (ACI) quality via $R_{\max}^{\text{policy}} = \min(1, R_{\max}^{\text{obs}} / \max(\text{RQS}, 0.05))$, concluding that for `Qwen3-8B` ($R_{\max}^{\text{obs}} = 0.285$, $\text{RQS} = 0.353$), *"part of the 'incapable' verdict for Qwen3-8B is a reward-parser ceiling rather than a policy deficit"* because $0.285 / 0.353 \approx 0.808$.

However, per Equation 7 (p. 26), $\text{RQS} = (c_1 c_2 c_3 c_4)^{1/4}$ is not an exogenous measure of reward-parser fidelity; it is computed directly from the policy's own reward trajectory, including $c_2 = \text{frac}(R > 0.5)$ and $c_4 = 1 - 2\cdot\text{zero\_frac}$. A model with genuinely low problem-solving accuracy on GSM8K ($R_{\max}^{\text{obs}} = 0.285$, $\text{frac}(R > 0.5) = 0.100$ in Table 10) automatically receives a small $c_2$ and thus a small $\text{RQS}$ even under a flawless reward parser. Dividing $R_{\max}^{\text{obs}}$ by a monotonically increasing function of the trace's own reward algebraically inflates any low-performing model's "implied policy ceiling" toward $1.0$ without isolating parser errors.

**Suggested edit:**

```
Explicitly note that because $\text{RQS}$ is endogenously derived from the reward trajectory ($c_2 = \text{frac}(R > 0.5)$ and $c_4 = 1 - 2\cdot\text{zero\_frac}$), dividing $R_{\max}^{\text{obs}}$ by $\text{RQS}$ cannot distinguish between a reward-parser formatting failure and a genuine policy capability deficit without an independent parser audit on raw model completions.
```

## [MAJOR] Missing signal-to-noise term in detectability formula T* and GCD vs. LCM typo  (p.28 "(b) effective binomial n inferred from the GCD of reward denominators; (c) AICc ")

There are two issues in the Method description of Section 4.2:
1. **Omitted Variance/Effect-Size Term in $T^\star$ (item e):** The minimum trace length for detectability is written as $T^\star = (z_{1-\alpha/2} + z_{\text{power}})^2$. At $\alpha = 0.05$ and power $0.80$, $(1.96 + 0.84)^2 \approx 7.84$, which is a dimensionless constant that omits the binomial variance and effect-size denominator ($\sigma_{\text{step}}^2 / \Delta^2$) needed to produce the $T_{0.8}^\star > 200$ step thresholds reported in Table 18 (p. 17).
2. **GCD vs. LCM Typo (item b):** Item (b) states *"effective binomial $n$ inferred from the GCD of reward denominators"*, whereas p. 16 correctly states *"LCM of the reward denominators"*. For rational fractions reduced to lowest terms, the common denominator is the least common multiple (LCM) of the denominators, not the greatest common divisor (GCD).

**Suggested edit:**

```
(b) effective binomial $n$ inferred from the LCM of reward denominators; (c) AICc model selection over {constant, linear, saturation}; (d) parametric bootstrap ($B = 400$) on $\text{Binomial}(n_{\text{eff}}, p_t)$ for the lambda CI; (e) detectability $T^\star = (z_{1-\alpha/2} + z_{\text{power}})^2 \sigma_{\text{step}}^2 / \Delta^2$ benchmark against the actual trace length.
```

## [MAJOR] Unreconciled functional-form rankings, task ambiguity (0.99 vs. 0.734 accuracy), and critic warm-up lag  (p.29 "Table 29: EEP-falsification battery on the samestack traces. E2 is decisively fa")

Comparing Section 4.2 (`iter29`, Tables 28–29) with Section 3.3 (p. 7) and Section 4.4 (`iter37c`, p. 32) on the exact same 10 trajectories (`samestack_ppo_grpo.json`, 5 GRPO + 5 PPO seeds on `Qwen2.5-0.5B` over 40 steps) reveals three issues that should be reconciled:
1. **Divergent Functional-Form Rankings (`iter29` vs. `iter37c`):** In Table 28, when comparing `{constant, linear, saturation}`, `GRPO` prefers `saturation` 5/5 and `PPO` prefers `linear` 5/5 (Fisher exact $p = 0.0079$, Table 29). Yet in `iter37c` (p. 32), when Hill ($n=2$) and Michaelis-Menten are added to the candidate set on the exact same 10 runs, `GRPO` prefers Hill ($n=2$) on 3/5 runs (saturation wins only 2/5), and on `PPO` the mean Akaike weight for `linear` is $\bar{w}_{\text{linear}} = 0.14$—lower than both `saturation` ($\bar{w}_{\text{sat}} = 0.21$) and `MM` ($\bar{w}_{\text{MM}} = 0.25$).
2. **Task Scope Discrepancy ($0.99$ vs. $0.734$ Accuracy):** Table 29 reports that `Qwen2.5-0.5B` reaches paired held-out accuracy of $0.99 \pm 0.0035$ (GRPO) and $0.992 \pm 0.003$ (PPO) in 40 steps, whereas Section 3.3 (p. 7) reported that the 5-seed `Qwen2.5-0.5B` TRL GRPO baseline on GSM8K achieves a mean accuracy of $\bar{x} = 0.734$. This indicates that `samestack_ppo_grpo.json` was evaluated on the simplified synthetic arithmetic task (or an easy subset) rather than the full GSM8K test set, which should be clarified in Section 4.2.
3. **Critic Warm-Up Horizon:** Because 40 gradient steps on a 0.5B model is a short warm-up horizon where PPO's randomly initialized value critic is still fitting the return scale, the claim that PPO fundamentally follows a linear trajectory rather than saturating should be qualified as potentially reflecting critic warm-up lag.

**Suggested edit:**

```
Clarify whether `samestack_ppo_grpo.json` was evaluated on the synthetic arithmetic task (explaining the $0.99$ accuracy vs. $0.734$ on GSM8K in Section 3.3), cross-reference the 5-model `iter37c` comparison where PPO's mean Akaike weight on `saturation` ($0.21$) exceeds `linear` ($0.14$), and note that a 40-step horizon may reflect value-critic warm-up lag.
```

## [MAJOR] Unverified external LLM conjectures presented as formal synthesis and missing RLVR baselines  (p.41 "4.9 Frontier-Model Cross-Examination: Scaling Laws
To stress-test the Pillar-1 f")

Section 4.9 (*Frontier-Model Cross-Examination: Scaling Laws*, pp. 41–43) devotes nearly three pages and five numbered display equations (Equations 10–14) to reporting mathematical conjectures generated by prompting commercial LLMs (*"ChatGPT Pro Extended and Gemini Deep Think"*), which are also cited in earlier subsections (pp. 28, 29, 36, 39) as authoritative theoretical justifications (*"frontier synthesis"*). While the text transparently notes that these are not additional experiments, presenting unverified LLM-generated equations and ad-hoc numerical thresholds (e.g., $\text{BEI} \ge 0.97$ and gradient-norm ratios in $[0.7, 1.3]$ in Eq. 13, or unmeasured covariates like `GU`, `KLspike`, and `lengthdrift` in Eq. 11) as formal numbered equations in the main results section weakens the scientific rigor of the paper.

Additionally, the manuscript's ZVF and scaling findings would be substantially strengthened by condensing these LLM conjectures into the Discussion or an appendix (taking direct intellectual ownership of the hypotheses) and contextualizing the results against contemporary RLVR literature designed to address zero-variance groups and normalization biases—such as **DAPO** (dynamic sampling for zero-variance groups), **Dr. GRPO** (removing length and std-normalization bias), and **ScaleRL**—as well as harder, un-saturated reasoning benchmarks (e.g., MATH-500 or AIME).

**Suggested edit:**

```
Move the external LLM crossexamination transcripts/equations (Section 4.9) to an appendix or condense them into the Discussion section, and add citations and discussion comparing the ZVF/scaling findings against contemporary RLVR methods (e.g., DAPO, Dr. GRPO, ScaleRL).
```

## [MAJOR] Unconsolidated chronological 'Iter' log structure with shifting parameter bounds  (p.27 "Recommended action: add a 4-panel figure to this section showing (a) RQS per anc")

Sections 4.1–4.8 (pp. 27–41) are structured as an append-only concatenation of automated research iterations (jumping backward from `Iter 140` on p. 27 to `Iter 29`, `Iter 33`, `Iter 37`, `Iter 41`, `Iter 45`, `Iter 49`, `Iter 53`, `Iter 61`, and `Iter 65` on pp. 28–41) where later-presented iterations contradict or overwrite earlier ones using shifting parameter bounds ($\lambda \le 5.0$ in Table 31 vs. $\lambda \le 10$ in Table 39; $R_{\max} \le 1$ vs. $R_{\max} \le 2$), raw prompt instructions (e.g., *"Recommended action: add a 4-panel figure to this section..."* on p. 27), and references to missing external figure files. Consolidating these iterations into a single self-consistent narrative is strongly recommended.

**Suggested edit:**

```
Remove internal prompt instructions (such as 'Recommended action: add a 4-panel figure...') and consolidate the chronological `Iter` blocks across Sections 4.1–4.8 into a unified, self-consistent results narrative with uniform parameter bounds.
```

## [MAJOR] Ten unreferenced generic flowcharts appended after References (Figures 17–26)  (p.45 "Figure 17: GRPO Rollout Scaling Grid. Multi-seed rollout batching architecture.")

Ten generic three-box TikZ flowcharts (*GRPO Rollout Scaling Grid*, *Pass@k Advantage Factorization*, *Group-Size Noise Boundary*, *Token-Matched Sweep Matrix*, *Multi-Seed Variance Reduction*, *Empirical Pass@1 Scaling Frontier*, *Token-Matched Rollout Allocation*, *Exploration-Exploitation Tradeoff Curve*, *Task Difficulty Stratification Matrix*, and *End-to-End GRPO Scaling Benchmark Flow*) are placed across pages 45–47 without a single in-text citation or explanation anywhere in the manuscript. Several depict pipelines (such as Pass@k advantage factorization or dynamic resource schedulers) that are not part of this study. Remove Figures 17–26 unless they are integrated into and justified by the main text.

## [MINOR] Unexplained references to companion papers, internal artifacts, and claim identifier F3  (p.2 "The program’s stronger explanatory candidates are usable contrastive compute and")

Several key concepts, internal claim identifiers, and external references are introduced in Sections 1 and 2 without sufficient self-contained context:

- Section 1 (Page 2) references *"the ZVF and group-size companion papers"* without bibliographic citations or an explanation of the multi-pillar structure of `TINKERRL-BENCH`.

- Section 2.2 (Page 3) repeatedly references **$F_3$** (*"the algorithm-and-stack ablation underlying $F_3$"*, *"the cross-RL-library evidence in $F_3$ uses the first"*), as well as *"the framework-gap discussion (framework-configurations material)"* and *"Supplementary presentation materials,"* without defining what hypothesis or finding $F_3$ represents at this point in the text or where those external materials are located.

Briefly summarize the multi-pillar structure of `TINKERRL-BENCH`, provide formal citations (or self-contained summaries) for the ZVF and group-size companion papers in Section 1, and explicitly define claim $F_3$ upon its first appearance in Section 2.2 so that the early sections can be understood independently.

## [MINOR] Inconsistency between Figure 1 legend symbol '◦ Planned' and HumanEval node label '◦ Partial'  (p.3 "Figure 1: Task-coverage taxonomy of TINKERRL-BENCH. The benchmark organises task")

In Figure 1, the bottom legend defines the symbol `◦` as *"Planned"* (`✓ Complete  ∼ In progress  ◦ Planned`), but inside the `HumanEval` node the symbol is paired with the text *"Partial"* (`1 model  ◦ Partial`), creating a label inconsistency with `Function Calling` (`2 models tested  ∼ Partial`).

## [MINOR] Inconsistent capitalization of 'verl' vs. 'veRL' in Section 2.2  (p.3 "Of those, only Tinker and TRL produced completed runs in this release; veRL and ")

The library `verl` is spelled in lowercase (`verl`) in line 6 of Section 2.2 (`"SkyRL, verl, OpenRLHF..."`) and in mixed case (`veRL`) in line 8 of the same paragraph (`"veRL and OpenRLHF entries..."`). Please standardize the capitalization.

**Suggested edit:**

```
Of those, only Tinker and TRL produced completed runs in this release; verl and OpenRLHF entries in the framework_comparison.json artefact are dry-run placeholders.
```

## [MINOR] Inconsistent DeepSeek-V3.1 parameter count (671B vs. 685B) and Tier-A dense subset bound  (p.4 "Tier-A inferential claims are restricted to the dense-reasoning subset at 0.6B–2")

There are two parameter-scale inconsistencies in the opening paragraph of Section 3.1:

- **DeepSeek-V3.1 Parameter Count:** Section 3.1 lists DeepSeek-V3.1 as `"~671B total / ~37B active MoE"` (which is also used as `671.0B` in Tables 6 and 7, and `671B` in the text on p. 42), whereas the Abstract (p. 1) describes it as `"at the 685B class"` and Tables 32 (p. 34), 38 (p. 37), and 43 (p. 40) use `685.0B`. Standardizing the nominal parameter count across all sections, tables, and regressions would ensure clarity on the exact $\log_{10} N$ values entering the regressions.

- **Tier-A Dense Subset Upper Bound:** Section 3.1 states that `"Tier-A inferential claims are restricted to the dense-reasoning subset at 0.6B–235B"`, yet in the same section `Qwen3-235B-A22B` is explicitly an MoE model (`"MoE, 22B active"`) and the dense Qwen3 family is described as extending only from `"0.6B to 32B"` (with Nemotron-120B also being LatentMoE). Furthermore, Section 3.2 (p. 5) states that the multi-seed Tier-A sweeps were conducted on classic-RL libraries ($F_3$) and `Qwen2.5-0.5B`. Reconciling the actual parameter range and model membership of the Tier-A inferential subset is recommended.

## [MINOR] Unreconciled experiment and run counts across Sections 1–3.4  (p.7 "Experiments are distributed across two compute platforms:
Tinker API (14 experim")

There is a clarity issue in reconciling the experiment counts reported across the setup sections:

- Figure 1 (p. 3) lists `"8 models tested"` on GSM8K, `"2 models tested"` on Function Calling, and `"1 model"` on HumanEval (which also differs from the 5-anchor and 12-anchor GSM8K pools in Section 4).

- Figure 3 (p. 5) and Section 3.4 (p. 7) tally `"14 experiments"` on the Tinker API and `"6 experiments"` on the Modal H100 cluster (20 experiments total), plus a 5-seed TRL GRPO baseline on an NVIDIA L4 GPU mentioned on page 7.

- Section 3.3 (p. 7) states that the variance decomposition is computed across `"42 experiments with valid performance observations"`.

- The Abstract and Section 1 state that the benchmark spans `"70+ runs"`, and Section 3.1 (p. 4) refers to `"32+ model configurations"`.

Providing a unified summary ledger or breakdown table in Section 3 that explicitly maps how the 14 Tinker runs, 6 Modal H100 runs, L4 baseline seeds, and classic RL library sweeps aggregate to the 42 valid ANOVA observations and 70+ total runs—and clarifying whether multi-seed sweeps are counted as single "experiments" or individual "runs"—would make the experimental accounting transparent.

## [MINOR] Clarify equal-variance approximation in Table 4 power calculation for Welch t-test  (p.6 "Table 4: Statistical power for two-sample Welch t-tests at α = 0.05, power targe")

The two-sample $t$-test power analysis and minimum detectable effect sizes in Table 4 ($d_{\min} = 2.024$ for $n_1=n_2=5$ and $d_{\min} = 1.530$ for $n_1=5, n_2=15$ at $\alpha=0.05, 1-\beta=0.80$) are computed via `statsmodels.stats.power.TTestIndPower`. Note that `TTestIndPower` assumes pooled equal variances with Student $t$-distribution degrees of freedom $\text{df} = n_1 + n_2 - 2$, whereas the text specifies Welch's unequal-variances $t$-test, which can have smaller Satterthwaite degrees of freedom when variances differ; noting that the power calculation serves as an equal-variance approximation would make this exact.

## [MINOR] Small diagram font size in Figure 1  (p.3 "Figure 1: Task-coverage taxonomy of TINKERRL-BENCH.")

The text inside the diagram boxes of Figure 1 (`"8 models tested"`, `"2 models tested"`, `"1 model"`) is rendered at a very small font size and low resolution; enlarging the node labels would improve legibility.

## [MINOR] Small font size and notation collision (N vs. G for group size) in Figure 2  (p.4 "Figure 2: Verifiable reward paradigm in TINKERRL-BENCH.")

The text inside the diagram boxes of Figure 2 (`"Generate N Responses"`, `"Reward Function"`, etc., and the bottom advantage equation) is rendered at a very small font size and low resolution; enlarging the node labels would improve legibility. Additionally, in the advantage formula inside Figure 2 ($\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$), the rollouts are indexed from $i=1$ to $N$ (`"Generate N Responses"`), whereas Section 1 (p. 2) denotes group size by $G$ and Section 4 uses $N$ for model parameter count in billions; using $G$ in Figure 2 would avoid notation collision with parameter count $N$.

## [MINOR] Small diagram font size and low resolution in Figure 3  (p.5 "Figure 3: Experiment workflow pipeline.")

The text inside the diagram boxes of Figure 3 is rendered at a very small font size and low resolution; enlarging the node labels would improve legibility.

## [MINOR] Missing bibliography entry for Hedges–Olkin (1985)  (p.5 "confidence intervals follow the Hedges–Olkin (1985) analytical formula and are r")

`"Hedges–Olkin (1985)"` is cited in the text by author-year, but it does not have a corresponding numbered entry in the References section.

## [MINOR] Typo: 'a internally' -> 'an internally'  (p.7 "We then run a internally recorded model-selection comparison")

Grammatical typo: replace `"a internally"` with `"an internally"`.

**Suggested edit:**

```
We then run an internally recorded model-selection comparison
```

## [MINOR] Truncated sentence at end of 'Cross-scale law' paragraph  (p.8 "We regress each per-trace statistic on log10 N (N = parameter count in billions)")

The paragraph "Cross-scale law" ends abruptly mid-sentence at the bottom of the page (`"The fitted coefficients are"`) with no equation, values, or table reference before the next heading (`"Three-phase learning dynamics."`).

**Suggested edit:**

```
The fitted coefficients are reported in Table 7.
```

## [MINOR] Dangling colons where referenced tables floated to top of page  (p.10 "The headline finding is that the four “well-behaved” runs sit at the optimiser’s")

Paragraphs 3, 4, and 6 on page 10 (as well as paragraph 2 on p. 12 and paragraph (d) on p. 23) end with dangling colons (`"...far more often than the point estimate suggests:"`, `"...comparing the test RMSE to a constant (train-mean) baseline:"`, `"...Applied to all five traces:"`) even though the corresponding tables (Tables 8, 9, 10, 12, and 25) floated to the tops of pages 10, 11, 12, and 24 rather than appearing inline below the text. Replace these colons with explicit table references.

**Suggested edit:**

```
The headline finding is that the four “well-behaved” runs sit at the optimiser’s λ = 10 upper bound far more often than the point estimate suggests (Table 8).
```

## [MINOR] Undefined 'Frontier synthesis' and duplicated parenthetical text  (p.16 "Frontier synthesis (frontier synthesis): this is the operational analogue of the")

On **p. 14** (Paragraph E), `"Frontier synthesis (ChatGPT Pro Extended, R1):"` introduces `"Frontier synthesis"` and the abbreviation `"R1"` (for Round 1 of the external cross-examination) long before Section 4.9 (p. 41) defines what the frontier synthesis is. Additionally, here on **p. 16** (line 2), there is duplicated parenthetical text: `"Frontier synthesis (frontier synthesis):"`.

**Suggested edit:**

```
Frontier synthesis (see Section 4.9): this is the operational analogue of the Pillar 3 iso-G savings
```

## [MINOR] Unedited template placeholder '(estimate)' in Paragraph (H)  (p.15 "The OLS slope of Rmax on log10 N is positive (estimate), with a negative interce")

Paragraph (H) contains an unedited template placeholder `"is positive (estimate)"` where the numerical slope estimate is missing.

## [MINOR] Lowercase author surname 'nimmaturi', duplicate reference [17]/[18], and inline script paths  (p.18 "Table 19: Iter 117 BIC-segmentation threephase test against Nimmaturi et al. [18")

1. The author surname `"Nimmaturi"` is repeatedly typeset in lowercase as `"nimmaturi pass?"` (Table 19 header) and `"the nimmaturi three-phase criterion"` (Table 19 caption and body text on p. 18).
2. Page 21 cites `Nimmaturi et al. [17]` whereas pp. 7, 8, 10, 17, 18, and 19 cite `Nimmaturi et al. [18]` (both [17] and [18] in the References section on pp. 44–45 are duplicate entries for `arXiv:2507.18014`).
3. Throughout Section 4 (pp. 11–27), internal repository script invocations and local TSV file paths (e.g., `scripts/scaling_law_fit.py -> scaling_law_three_phase.tsv`) are printed directly inside body paragraphs and table captions; moving these to a reproducibility appendix or code README would improve readability.

## [MINOR] Typo 'fourof', mismatched 'Iter113' supertitle, and raw BibTeX key in Figure 9 (and Figure 13)  (p.19 "Figure 9: Iter 117 four-panel overview. (A) Saturation fit R(t) = Rmax(1 − e−λt)")

1. In the Figure 9 caption (line 3), fix the spacing typo `"fourof five anchors"` $\to$ `"four of five anchors"`.
2. The supertitle embedded at the top of **Figure 9** (and identically in **Figure 13** on p. 26) reads `"Pillar 1 Iter113"` (whereas the text and captions refer to `Iter 117` and `Iter 137`) and contains the raw concatenated BibTeX key `"0/5 anchors satisfy nimmaturi2025predictive three-phase"`.

**Suggested edit:**

```
Figure 9: Iter 117 four-panel overview. (A) Saturation fit R(t) = Rmax(1 − e−λt) versus observed reward for each of the five anchors; dashed lines show the fitted curve. (B) BIC bar chart for k ∈ {1, 2, 3} constant-mean segments per anchor: four of five anchors prefer k = 3
```

## [MINOR] Math-mode formatting artifacts in model names inside set braces  (p.24 "this labels the incapable cluster as {Qwen3 − 8B, Nemotron−120B, Qwen3−32B, Qwen")

Model names inside set braces are formatted in LaTeX math mode without `\text{...}` or `\texttt{...}` wrappers, causing italicization, minus signs in place of hyphens, and spacing artifacts (e.g., `{Qwen3 − 8B, Nemotron − 120B, ...}` and `T hinking`).

**Suggested edit:**

```
this labels the incapable cluster as {Qwen3-8B, Nemotron-120B, Qwen3-32B, Qwen3.5-27B, Qwen3-30B-MoE} and the capable cluster as {Qwen3.5-4B, Llama-3.1-8B-Instruct, DeepSeek-V3.1, Kimi-K2-Thinking, gptoss-20B, Qwen3-30B-MoE-Inst, Qwen3-235B-MoE} in the n = 12 pool.
```

## [MINOR] Unexplained course lecture shorthand, missing bibliography entry for Ma et al. (2023), and leftover editing note  (p.27 "Recommended action: add a 4-panel figure to this section showing (a) RQS per anc")

In "Iter 140 elevation" (pp. 25–27):
1. The text contains unexplained course lecture shorthand (`"Berkeley F24 L9 — Eureka"` on p. 25, `"F25 L8 / Miller's recipe"` on p. 26, and `"DECISIVE per F25 L8 recipe"` in Figure 14's caption on p. 27).
2. It cites `(arXiv:2310.12931; Ma et al., ICLR 2024 oral)` inline on p. 26 without a corresponding entry in the References section.
3. It leaves an internal editing directive at the end of the section here on p. 27 (`"Recommended action: add a 4-panel figure to this section..."`), which duplicates **Figure 14** already placed at the top of p. 27.

**Suggested edit:**

```
The capability axis of iter 133 is therefore preserved as the load-bearing cross-anchor signal, but the reward-side of the equation is not silent: it adds diagnostic information orthogonal to capability on the cell grid, exactly where the Eureka thesis predicts (Figure 14).
```

## [MINOR] Undefined optimal scale P* = 3.46B in Table 34  (p.35 "Table 34: Internally recorded iso-compute predictions P1–P3.")

In Table 34, `P3_chinchilla_P_star_in_range` reports `YES` with `P*=3.46B vs P_median=30.0B; ratio=0.115`, but Section 4.6 only fits a univariate regression $\log R_{\max} \sim \alpha \log C$ (Table 33) and never defines the objective function or closed-form equation used to compute $P^\star = 3.46\text{B}$.

**Suggested edit:**

```
Define the formula or optimization procedure used to derive $P^\star = 3.46\text{B}$ in Section 4.6.
```

## [MINOR] Orphaned empty subheadings across pages 28, 30, 33, 34, and 35  (p.28 "Results.")

Several bold subheadings appear as standalone lines with no body text underneath them across pages 28–35:

- **Results.** (p. 28, bottom)

- **Results.** (p. 30, middle)

- **Per-anchor stability.** (p. 33)

- **Compute-proxy table.**, **Within-stack scaling.**, and **Predictions.** (p. 34)

- **Joint OLS coefficients.** and **Internally recorded predictions.** (p. 35)

Remove these orphaned subheadings or populate them with the intended text.

## [MINOR] References to missing external PDF figure paths (iter29, iter33, iter37)  (p.29 "Artifacts. Driver: scripts/scaling_law_iter29.py. Per-trace: scaling_law_iter29_")

Sections 4.2, 4.3, and 4.4 reference internal file paths `figures/scaling_law_iter29.pdf` (p. 29), `figures/scaling_law_iter33.pdf` (p. 31), and `figures/scaling_law_iter37.pdf` through `iter37d.pdf` (p. 33), none of which are embedded in the manuscript PDF. Either embed the corresponding figures or remove the dangling file references.

**Suggested edit:**

```
Artifacts. Driver: scripts/scaling_law_iter29.py. Per-trace: scaling_law_iter29_identifiability.tsv, scaling_law_iter29_bootstrap.tsv. Stack rollup: scaling_law_iter29_summary.tsv. EEP battery: scaling_law_iter29_stack_compare.tsv.
```

## [MINOR] Clarify exact binomial test null parameter and p = 0.083 threshold for P5 in Table 30  (p.30 "Table 30: Internally recorded threephase hypothesis battery on the twelve-anchor")

In Table 30, Prediction P5 reports `Pass = 1/12`, `Rate = 0.083`, `p = 0.083`, and `Verdict = sustained`. A $p$-value of $0.083$ exceeds $\alpha = 0.05$ (and the $\alpha/5 = 0.01$ Bonferroni threshold noted on p. 31), and $1/12 \approx 0.0833$ is simply the raw sample fraction. Clarify the null hypothesis parameter $p_0$ used for the exact binomial test on P5 and why $p = 0.083$ is labeled `sustained`.

**Suggested edit:**

```
Clarify the binomial null parameter $p_0$ for P5 or note that P5 is a descriptive uniqueness check ($1/12$ traces) rather than a test rejecting $H_0$ at $\alpha = 0.05$.
```

## [MINOR] Duplicate '1 plateau' category and MoE phase counts summing to 7 instead of n=6  (p.31 "• dense (n=6): 1 plateau, 1 saturation, 2 drift, 1 collapse, 1 plateau.
• moe (n")

In the first bullet point (`dense (n=6): 1 plateau, 1 saturation, 2 drift, 1 collapse, 1 plateau`), the category `1 plateau` is listed twice instead of `2 plateau`. In the second bullet point (`moe (n=6): 2 plateau, 4 saturation, 1 drift, 0 collapse`), the counts sum to $2 + 4 + 1 + 0 = 7$, contradicting `n=6` (or reflecting the 7 MoE / 5 dense split while keeping the `n=6` header).

**Suggested edit:**

```
• dense (n=5): 2 plateau, 1 saturation, 2 drift, 0 collapse.
• moe (n=7): 2 plateau, 3 saturation, 1 drift, 1 collapse.
```

## [MINOR] Holdout list names 6 models instead of 4  (p.33 "On the 4-anchor holdout (Qwen3-32B, Qwen3.5-27B, DeepSeek-V3.1, Qwen3- 235B-MoE,")

The text states *"On the 4-anchor holdout (Qwen3-32B, Qwen3.5-27B, DeepSeek-V3.1, Qwen3-235B-MoE, Nemotron-120B, Kimi-K2-Thinking)"*, which lists 6 models instead of 4.

**Suggested edit:**

```
On the 6-anchor holdout (Qwen3-32B, Qwen3.5-27B, DeepSeek-V3.1, Qwen3-235B-MoE, Nemotron-120B, Kimi-K2-Thinking), the mean absolute error on Rmax is essentially identical:
```

## [MINOR] Unexplained denominator shift from 12 to 9 anchors in Section 4.5  (p.33 "Headline result. 7/12 anchors are stable (max relative ∆λ < 0.5); 2 are unstable")

The text states *"7/12 anchors are stable... 2 are unstable"*, omitting the 3 anchors that are `nan` / `degenerate` in Table 31, and subsequently switches the denominator to `9/9` and `7/9` under *Early -> late prediction* without explicitly noting that the 3 degenerate short-trace anchors were excluded.

**Suggested edit:**

```
Headline result. Across the 12 anchors, 7/12 are stable (max relative ∆λ < 0.5), 2/12 are unstable, and 3/12 short-horizon traces (n_steps ≤ 5) are degenerate (nan).

Early → late prediction. Restricting to the 9 non-degenerate anchors and using only the first 60% of the trace, the predicted Rmax falls within ±10% of the full-fit Rmax for 9/9 anchors; the bootstrap 95% CI on the prediction error contains 0 for 7/9.
```

## [MINOR] Misaligned wide tilde on ZVF, 'residuall' typo, and unformatted float literals on pp. 37-39  (p.38 "• P3_nemotron_largest_jackknife_residual: PASS. Nemotron-120B has the largest |L")

Across pages 37–39:
1. In Table 38 and Equation (8) (p. 37), on p. 38, and in Figure 15's caption (p. 39), the wide tilde over `ZVF` (`\widetilde{\text{ZVF}}`) renders shifted to the right of the letters.
2. In Table 40 (p. 38, column 4 header) and Bullet P3, `|residual|` is misspelled as `|residuall|` / `|LOO residuall`.
3. In Bullets P4 and P5 on p. 38, unrounded floating-point literals (`observed=-0.03542003029503481` and `observed=0.41250000000000003`) and raw comparison syntax (`expected=> rollout-level`) are left in the text.

**Suggested edit:**

```
• P3_nemotron_largest_jackknife_residual: PASS. Nemotron-120B has the largest |LOO residual| in the crossscale fit (observed = Nemotron-120B, expected = Nemotron-120B).
• P4_nemotron_alpha_logP_shift: PASS. Removing Nemotron-120B shifts α_logP in the same direction as the full-fit residual (observed = -0.035, expected: sign(residual)·sign(shift) > 0).
• P5_step_zvf_above_rollout_zvf: PASS. Mean per-step ZVF proxy exceeds Pillar 2 rolloutlevel mean_zvf (observed = 0.413 > 0.158).
```

## [MINOR] Equation (9) overflows right margin and clips threshold value  (p.39 "|m1| < SLOW_SLOPE_MAX = 0.020, |m2| ≥ FAST_SLOPE_MIN = 0.040, |m3| < PLATEAU_SLO")

The right-hand side of Equation (9) (`|m_3| < PLATEAU_SLOPE_MAX = 0.015,`) overflows the right margin and is partially clipped (`= 0.0`) in the rendered PDF. Wrap the equation across two lines or shorten the variable names.

**Suggested edit:**

```
|m_1| < 0.020,\quad |m_2| \ge 0.040,\quad |m_3| < 0.015,
```

## [MINOR] Class label mismatch between Figure 16(B) legend and Table 42  (p.41 "Figure 16: Iter 65 phase-conformity geometry. (A) per-phase OLS slopes (m1, m2, ")

The legend in Figure 16(B) labels one class `incomplete-saturation`, whereas Table 42 (p. 40) labels it `monotonic-rising`. Align the class labels between the figure and table.

## [MINOR] Undefined variables N_0, (+), GU, and unindexed KL_t in Equations (10)-(12)  (p.42 "∆GRPO(N) = z(N) ∆max 1 − e −λ(log N−log N0)+ − 1 − z(N) Ccollapse, (10)
where ∆G")

In Equations (10)–(12) on p. 42:
1. Define the positive-part subscript $(\log N - \log N_0)_+ = \max(0, \log N - \log N_0)$ and the threshold parameter $N_0$ in Eq. (10).
2. Define the variable `GU` (and clarify `KLspike` and `lengthdrift`) in Eq. (11).
3. Clarify the time index $t$ on $\text{KL}_t$ in $C_{\text{eff}}(G) = T \cdot G \cdot \mathbb{E}_x[Y_G(p_x)] \cdot \text{KL}_t$ in Eq. (12), where $C_{\text{eff}}(G)$ is written as a function of $G$ multiplied by total steps $T$ rather than summed over step index $t = 1, \dots, T$.

## [MINOR] Consolidate duplicate bibliography entries [11] and [12] (Kaplan et al., 2020)  (p.44 "[11] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, R")

Jared Kaplan et al., *"Scaling laws for neural language models"* (arXiv:2001.08361, 2020) is listed twice in the References section—once as [11] and immediately again as [12]. These should be consolidated into a single bibliographic entry and all in-text citations unified.

## [MINOR] Consistent capitalization and complete metadata for Kimi K2 references [13] and [14]  (p.44 "[13] Kimi Team, Moonshot AI. Kimi k2. https://moonshotai.github.io/Kimi-K2/, 202")

In References [13] and [14] (*"Kimi k2"* and *"Kimi k2 thinking"*), capitalize the model names consistently as *"Kimi K2"* and *"Kimi K2 Thinking"*, and provide complete bibliographic metadata (such as the URL or technical report identifier) for [14].

**Suggested edit:**

```
[13] Kimi Team, Moonshot AI. Kimi K2. about:sanitized, 2025.
[14] Kimi Team, Moonshot AI. Kimi K2 Thinking. about:sanitized, 2025.
```

## [MINOR] Consolidate duplicate bibliography entries [17] and [18] (Nimmaturi et al., 2025)  (p.44 "[17] Datta Nimmaturi, Vaishnavi Bhargava, Rajat Ghosh, and Johnu George. Predict")

Datta Nimmaturi et al., *"Predictive scaling laws for efficient GRPO training of large reasoning models"* (arXiv:2507.18014, 2025) is listed twice—once as [17] (omitting the fifth author, Debojyoti Dutta) and again on p. 45 as [18] (including Debojyoti Dutta and an inline annotation *"identifies three consistent training phases: slow start, rapid improvement, plateau"*). Merge these into a single, complete bibliographic entry without inline notes.

**Suggested edit:**

```
[17] Datta Nimmaturi, Vaishnavi Bhargava, Rajat Ghosh, Johnu George, and Debojyoti Dutta. Predictive scaling laws for efficient GRPO training of large reasoning models. arXiv preprint arXiv:2507.18014, 2025. doi: 10.48550/arXiv.2507.18014.
```

## [MINOR] Remove internal iteration self-citation [24] and use section cross-reference  (p.45 "[24] TinkerRL-Bench Team. TinkerRL-Bench: Iteration-25 scaling-law identifiabili")

Reference [24] cites an internal iteration of the very manuscript under review (cited on p. 28 to reference Iteration 25, which is already presented in Section 4 on pp. 16–17). Replace the citation on p. 28 with an internal section/table cross-reference (e.g., Section 4, Table 18) and remove [24] from the bibliography.

## [MINOR] Standardize author list formatting in Reference [25] (OpenHands)  (p.45 "[25] Xingyao Wang, Graham Neubig, et al. OpenHands: An open platform for AI soft")

The author list for OpenHands (*"Xingyao Wang, Graham Neubig, et al."*) truncates the author list with *"et al."* after only two authors (and out of standard author order, as Boxuan Li, Yufan Song, Frank F. Xu, etc. precede Graham Neubig on the paper), whereas other entries list the full author roster or standard truncation. Ensure consistent author formatting across all bibliography entries.

**Suggested edit:**

```
[25] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations (ICLR), 2025. doi: 10.48550/arXiv.2407.16741. arXiv:2407.16741.
```

## [SUGGESTION] Contextualize GSM8K null scaling result against recent RLVR scaling and GRPO literature  (p.2 "Scaling behavior of RL and RLHF post-training has been examined more recently [8")

In Section 1 and Section 1.1 (Page 2), the related work on RL post-training scaling cites only Hilton et al. [8] (single-agent RL in game/visual environments), Rafailov et al. [21] (reward model overoptimization in direct alignment algorithms), and Nimmaturi et al. [18]. The paper's central negative finding—that 4 of 5 frontier models start at step-1 rewards of $R(1) \in [0.875, 1.0]$ on GSM8K and show flat trajectories over $T \le 30$ steps—is closely tied to benchmark ceiling saturation on grade-school math and zero-variance group rollouts. Recent RLVR scaling and GRPO optimization studies—such as *DAPO* (Yu et al., 2025) and *Dr. GRPO* (Liu et al., 2025), which directly address zero-variance prompt filtering and length/standard-deviation advantage normalization biases in GRPO, as well as positive RL post-training scaling studies on harder mathematical reasoning benchmarks (e.g., sigmoidal compute-performance scaling in *ScaleRL*, Khatri et al., 2025; power-law RL scaling in Tan et al., 2025)—are not discussed in Section 1.1.

Consider expanding Section 1.1 to briefly situate the paper's identifiability audit relative to recent RLVR literature on zero-variance group filtering (e.g., *DAPO* / *Dr. GRPO*) and contemporary RL scaling laws (e.g., *ScaleRL*), and discuss how evaluating modern frontier models (up to 671B–1T parameters) on GSM8K induces immediate step-1 ceiling saturation compared to competition-level benchmarks (such as MATH-500 or AIME).

## [SUGGESTION] Contextualize pre-training benchmark saturation confound on GSM8K vs. intrinsic scaling limits  (p.8 "The cause is the functional-form mismatch: a strict-R(0) = 0 model assumes the t")

As documented here and in **Table 6** (p. 8) and **Table 26** (p. 25), 3 of the 5 core anchors begin GSM8K training at or near the $1.0$ ceiling at step 1 ($R(1) = 1.000$ for `Qwen3.5-4B` and `Llama-3.1-8B-Instruct`; $R(1) = 0.875$ for `DeepSeek-V3.1`, with baseline offset $c_{3\text{p}} \in [0.799, 0.843]$), and training runs are restricted to $T \le 30$ steps (with four probes in the 12-anchor extension having only $n \le 5$ steps, p. 22). Because modern instruction-tuned and reasoning models largely saturate grade-school math (GSM8K) prior to RL post-training, the remaining headroom ($1 - c \le 0.20$) is comparable to the step-level binomial sampling noise floor ($\sigma_{\text{step}} \in [0.074, 0.119]$ for batch sizes $n \in \{8, 16\}$, Table 18, p. 17). Thus, the flat slopes and unidentifiable saturation rates ($\lambda$ clamping at the upper bound) are heavily driven by pre-training ceiling effects on an already-saturated benchmark over a short horizon.

**Actionable Recommendation:** Explicitly distinguish benchmark-ceiling saturation from intrinsic algorithmic scaling limits in Section 4, and contextualize these findings alongside contemporary RLVR scaling studies on harder competition-level benchmarks with substantial headroom (e.g., MATH-500, AIME 2024/2025) over longer horizons, as well as GRPO variants (such as dynamic zero-variance filtering in DAPO or unnormalized advantages in Dr. GRPO) that directly address zero-variance group degeneracy.

## [SUGGESTION] Explicitly discuss GSM8K pre-saturation and synthetic trace reconstruction in Limitations  (p.43 "The frontier-scale anchors are single-seed API runs and should be read as case s")

While Section 5 maintains strict claim integrity regarding the paper's core negative findings, two key methodological constraints established earlier in the manuscript should be explicitly synthesized here to strengthen the scope boundary:

1. **Benchmark Pre-Saturation on GSM8K as the Primary Driver of Ceiling Degeneracy:** Section 5 notes that *"the strongest evidence comes from short-horizon GSM8K training"* and mentions within-group reward variance as a confound, but does not explicitly state *why* $\lambda$ and the cross-scale slope are unidentifiable on GSM8K—namely, that capable modern instruction-tuned and reasoning checkpoints (such as Qwen3.5-4B, Llama-3.1-8B-Instruct, and DeepSeek-V3.1) initialize at step 1 already near the empirical accuracy ceiling on GSM8K ($R(1) \in [0.80, 1.00]$; Table 6, p. 8; p. 24; Table 38, p. 37). Explicitly stating that GSM8K pre-saturation leaves minimal headroom for measuring post-training learning kinetics—and that evaluating on harder, unsaturated reasoning benchmarks (such as competition-level MATH-500 or AIME) is necessary before concluding that GRPO trajectories are flat across model scales—would make the limitations discussion much more informative.
2. **Reliance on Reconstructed Synthetic Traces and Ultra-Short Horizons:** Several of the 12-anchor analyses (such as the three-phase falsification and functional-form batteries in Sections 4.3–4.4) operated on synthetic traces reconstructed from summary statistics rather than raw per-step reward logs (as documented in Section 4.3, p. 31), and four of the 12 anchors had very short horizons ($n \le 5$ steps, Table 32, p. 34). Explicitly noting this trace-resolution constraint alongside the single-seed caveat in Section 5 would strengthen methodological transparency.

## [SUGGESTION] Contextualize roadmap against contemporary RLVR variance-mitigation literature  (p.43 "What survives is a taxonomic, endpoint-level description of the ceiling, dominat")

Sections 5 and 6 frame the next steps around *"usable contrastive compute"* and disentangling model scale from *"within-group reward variance"* (ZVF), contrasting their null scaling finding primarily against pre-training Chinchilla laws [9, 12] and Nimmaturi et al. [18]. Briefly situating these findings relative to directly relevant RLVR methods that explicitly intervene on zero-variance rollout groups and advantage normalization biases—such as dynamic sampling in DAPO (Yu et al., 2025) or unnormalized advantages in Dr. GRPO (Liu et al., 2025), as well as recent RL compute-scaling studies on harder benchmarks—would better ground the conclusion's roadmap in current domain practice.
