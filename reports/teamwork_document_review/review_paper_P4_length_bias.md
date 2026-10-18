**Segment scope**: Comprehensive review of `platform_hybrid/paper/paper_P4_length_bias.tex` and its associated sections, experimental artifacts, and cross-cutting Segment 1 manuscripts.

# Potential Mistakes and Improvements

- **[Critical] Potential Confounding of Length Compression by Generation Ceiling Censoring.**
  The paper claims that GRPO and Dr.\,GRPO induce length compression rather than length expansion during training.
  However, the experimental setups initialize rollouts against the generation ceilings.
  On GSM8K-CoT, the generation cap is set to $200$ tokens, and step-0 mean completion length is already $194.4$ tokens ($97.2\%$ of the ceiling).
  In the uncapped Qwen3-8B panel, the cap is set to $1{,}024$ tokens, and step-0 completions average $999$--$1{,}004$ tokens ($97.6\%$--$98.0\%$ of the ceiling).
  Because completions start within $2\%$--$3\%$ of the hard cap, any exploratory generation that extends length is immediately truncated.
  Truncation breaks solution syntax and forfeits terminal reward.
  This introduces a mechanical penalty against longer trajectories.
  Therefore, observed length reduction may be an artifact of boundary avoidance rather than unconstrained policy compression.
  The authors should explicitly state that initialization at $\approx 98\%$ of the ceiling limits the validity of the compression finding.

- **[Critical] Discrepancy Between Harness Loss Implementation and Canonical Dr.\,GRPO Formulation.**
  Dr.\,GRPO introduces two specific mathematical modifications to standard GRPO:
  (1) replacing per-sequence length normalization $1/|o_i|$ with a constant global sequence length factor ($1/L_{\max}$), and
  (2) removing the group standard deviation denominator $\sigma_r$ from advantage estimation ($A_i = r_i - \bar{r}_g$).
  Inspection of the test harness (`live_zvf_probe.py` at commit `1959effe`) reveals that `--loss drgrpo` only altered reward standardization.
  It did not implement canonical fixed-token loss normalization.
  Furthermore, repository audit records show that the three Dr.\,GRPO arms and three GRPO arms in the uncapped panel were executed across different batch environments.
  Historical runs were contaminated by mislabeled executions preserved as `.invalid_actually_grpo.json`.
  Evaluating Dr.\,GRPO without its canonical token denominator leaves the length-debiasing mechanism untested.
  The manuscript should clarify that the evaluated variant tested reward de-standardization rather than canonical token-denominator normalization.

- **[Major] Severe Statistical Underpowering from Small Seed Allocations.**
  Key comparative findings rely on very small sample sizes.
  On GSM8K-CoT, comparisons between GRPO and Dr.\,GRPO use only $n = 3$ seeds per algorithm.
  Computing correlation coefficients across $n = 3$ points leaves only $df = 1$ degree of freedom.
  At $df = 1$, statistical significance requires an extreme correlation ($|r| > 0.997$).
  Reporting $p$-values for seed correlations (e.g., $r = -0.60, p = 0.59$) provides negligible diagnostic power.
  Bootstrap confidence intervals constructed on three observations are also vulnerable to sample instability.
  The authors should present these metrics as descriptive case studies rather than inferential statistical tests.

- **[Major] Task Degeneracy of Arithmetic Baseline for Reasoned Length Bias.**
  The arithmetic benchmark uses Qwen2.5-0.5B on single-step arithmetic problems.
  On this task, completion lengths average approximately $4.5$ tokens throughout training.
  Reward quickly reaches a saturation ceiling above $0.988$.
  A sequence of $4.5$ tokens cannot exhibit reasoning verbosity traps or complex chain-of-thought exploration.
  Interpreting a reduction from $5$ tokens to $4.5$ tokens as evidence of policy compression overstates the relevance of the task.
  The authors should restrict length-bias conclusions to genuine multi-step chain-of-thought workloads.

- **[Major] Overclaiming on ReAct Intermediate-Credit Shaping Diagnostic.**
  Section~\ref{sec:length-bias-react-shaping} presents an evaluation entitled "ReAct-Style Intermediate-Credit Shaping".
  This evaluation does not execute an interactive agent with interleaved thought and action tokens.
  Instead, it evaluates post-hoc reward column transformations on $n = 2$ rollout seeds from BFCLv4.
  Labeling a static two-seed reward reweighting as a ReAct study risks misrepresenting the empirical basis.
  The section should be explicitly titled and described as a preliminary reward-densification check on static rollouts.

- **[Minor] Ambiguous Attribution in Citation Key for Dr.\,GRPO.**
  The bibliography entry `tong2025drgrpo` lists Zichen Liu as the first author (`Understanding R1-Zero-Like Training: A Critical Perspective`).
  The citation key `tong2025drgrpo` appears to be a clerical remnant from earlier drafts.
  While BibTeX resolves the citation text correctly as Liu et al. (2025), updating the key avoids confusion during collaborative editing.

# Minor Corrections and Typos

- **[Remediated] Dangling Internal Drafting References `§5.X`.**
  In `sections/length_bias.tex`, five occurrences of the draft placeholder `§5.X` were present in the body text and in Table~\ref{tab:length-bias-horizon}.
  These references pointed to an unnumbered section from a prior monolithic draft.
  These placeholders were remediated into explicit benchmark and section references (`Sec.~\ref{sec:length-bias-pillar}`).

- **[Remediated] Undefined Reference `sec:frontier-critic-degeneracy` Under Standalone Compilation.**
  `sections/cdh_echo_synthesis.tex` referenced `\ref{sec:frontier-critic-degeneracy}`, which is defined in `sections/critic_degeneracy_hypothesis.tex`.
  When compiling `paper_P4_length_bias.tex` standalone, `critic_degeneracy_hypothesis.tex` was not included, generating two undefined reference warnings.
  The cross-reference was safeguarded with conditional macro logic (`\ifcsname r@sec:frontier-critic-degeneracy\endcsname`), enabling clean compilation across both P3 and P4.

- **[Remediated] Table and Equation Overfull Hbox Warnings.**
  Overfull horizontal box warnings occurred in Table~\ref{tab:length-iter36-cross} (`length_bias_iter36.tex`), the displayed chain in `length_bias_iter132.tex`, and the deliverables list in `length_bias_iter136.tex`.
  All three instances were remediated using proper font sizing, `aligned` math splitting, and standard text formatting.
