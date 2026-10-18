**Segment scope**: platform_hybrid/paper/paper_P1_scaling.tex (51-page manuscript on GRPO Scaling Laws, execution-substrate variance, and cross-scale horizons up to 1T parameters)

# Potential Mistakes and Improvements

### [Major] Step-Horizon Discrepancy Between Text and Checked-In Artifacts in 12-Anchor Panel
- **Observation**: In Section~\ref{par:scaling-extended} (lines 400--406 of `sections/scaling_laws.tex`), the prose asserts specific trajectory horizons for several probe models in the 12-anchor pool:
  - `gpt-oss-20B`: text asserts $T=24$, but checked-in artifact `experiments/results/scaling_law_extended_frontier.tsv` (row 7) records `n_steps = 30`.
  - `Qwen3.5-27B`: text asserts $T=4$, but `scaling_law_extended_frontier.tsv` (row 6) records `n_steps = 3`.
  - `Qwen3-32B`: text asserts $T=5$, but `scaling_law_extended_frontier.tsv` (row 5) records `n_steps = 3`.
  - `Qwen3-235B-MoE`: text asserts $T=3$, but `scaling_law_extended_frontier.tsv` (row 12) records `n_steps = 4`.
- **Concern**: When pooling traces across models spanning 4B to 1T parameters to test cross-scale scaling laws ($\log_{10} N$ regressions and rank tests), discrepancies in trajectory lengths undermine denominator hygiene and data reproducibility. Furthermore, using extremely short runs ($T \in [3, 4]$ steps) alongside 20-to-30-step runs to evaluate first-to-final reward differences ($\Delta_{1T} = R(1) - R(T)$) conflates initialization noise with training dynamics.
- **Remediation**: Align the text descriptions in `sections/scaling_laws.tex` with the exact values in `scaling_law_extended_frontier.tsv`. Explicitly flag that the ultra-short anchors ($T \le 5$) serve strictly as qualitative sanity checks rather than completed post-training trajectories.

### [Major] Saturation Model Functional Form Mismatch Enforces Boundary Pinning
- **Observation**: In Table~\ref{tab:scaling-fits} and Section~\ref{par:saturation-discussion}, the two-parameter canonical saturation model $R(t) = R_{\max}(1 - e^{-\lambda t})$ hits the artificial optimizer upper bound $\lambda = 10.0$ on four of five core frontier traces (Qwen3.5-4B, Qwen3-8B, Llama-3.1-8B-Instruct, DeepSeek-V3.1).
- **Concern**: The mathematical formulation imposes a strict initial condition $R(0) = 0$. However, instruction-tuned frontier models evaluate on GSM8K with initial step rewards $R(1) \in [0.28, 1.00]$ (e.g., Qwen3.5-4B starts at $1.00$, DeepSeek-V3.1 at $0.875$). Fitting a curve constrained to pass through the origin forces $\lambda \to \infty$ (clamping at the upper bound $\lambda=10.0$), yielding a trivial saturation time $t_{80} = -\ln(0.2)/\lambda \approx 0.16$ steps. Consequently, parameter estimates $\lambda$ and $t_{80}$ are mathematical artifacts of an inappropriate boundary condition rather than identifiable empirical properties of GRPO dynamics.
- **Remediation**: The manuscript properly identifies this defect in `par:saturation-discussion` and subsequent elevation iterations (iter 25, iter 29, iter 109). To maintain scientific clarity throughout the long draft, ensure that early sections do not present $\lambda$ or $t_{80}$ as physical convergence constants before the unidentifiability audit is introduced.

### [Major] Structural Coherence and Deprecated Exploration Blocks in 51-Page Draft
- **Observation**: `paper_P1_scaling.tex` is 51 pages long and concatenates 16 distinct section files representing an iterative research journal spanning iter 9 to iter 142.
- **Concern**: Several intermediate sections record hypotheses and preliminary statistical tests that are subsequently superseded, qualified, or withdrawn in later sections (for instance, early positive architecture splits vs.\ iter 13 withdrawal of the MoE-vs-dense gap; initial saturation law fits vs.\ subsequent failure of identifiability). A reader inspecting earlier sections without reading through to the conclusion risks taking early exploratory findings as defended results.
- **Remediation**: The companion note (`paper_P1_identifiability_note.tex`) correctly extracts the clean, shippable negative audit. For the long thesis manuscript, add explicit cross-referencing banners at the head of exploratory sections pointing to their final disposition in the synthesis sections.

### [Minor] Permutation Test Interpretability Under Extreme Parameter Heterogeneity
- **Observation**: In Table~\ref{tab:scaling-powerlaw}, OLS regressions on $\log_{10} N$ for seven reward metrics report non-significant permutation $p$-values ($p \in [0.46, 0.85]$) across $n=12$ anchors.
- **Concern**: The lack of significance ($p > 0.46$) reflects not only the absence of a simple monotonic relationship between parameter count and GRPO reward gain, but also extreme cross-checkpoint variance: the 12 models encompass different base training mixtures, varying degrees of supervised instruction tuning, diverse active-to-total parameter ratios (MoE vs dense), and disparate rollout budgets ($T=3$ to $T=30$).
- **Remediation**: Note cautiously that the failure to reject the zero-slope null should not be interpreted as proven scale-invariance or practical equivalence across all architectures, but rather as an identifiability failure due to checkpoint and recipe confounding in this empirical panel.

# Minor Corrections and Typos

1. **Undefined Reference Remediation (`sec:exp-design`)**:
   - In `sections/scaling_law_iter29.tex` line 23, the text referenced `\secref{sec:exp-design}`, but the corresponding label in `sections/_shared_methods.tex` was `\label{sec:setup}`. This generated an undefined reference warning during compilation.
   - *Status*: Remediated by adding `\label{sec:exp-design}` to `sections/_shared_methods.tex`. Clean compilation verified.

2. **Hyperref PDF String Math Warning**:
   - In `sections/scaling_passk_ci.tex` line 22, the subsection title contained `Pass@$K$`, causing `Package hyperref Warning: Token not allowed in a PDF string (Unicode): removing 'math shift'`.
   - *Status*: Remediated using `\texorpdfstring{Pass@$K$}{Pass@K}`.

3. **Overfull Horizontal Boxes**:
   - Minor overfull boxes occurred at `sections/scaling_laws:132` (2.77pt) and `sections/scaling_laws:360` (3.32pt) due to inline mathematical terms at paragraph ends. These do not disrupt layout significantly but can be tightened with discretionary hyphens if strict margin conformance is required.
