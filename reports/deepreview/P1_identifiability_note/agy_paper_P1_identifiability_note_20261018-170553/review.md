## Summary

This two-page note reports a negative audit of size-scaling laws for group-relative policy optimization (GRPO) post-training on GSM8K across a 12-anchor multi-family model panel (0.6B to 1T parameters). The author finds no statistically significant cross-scale reward slope ($p = 0.74$ for $\lambda$–$N$ and $p = 0.51$ for $R_\infty$–$N$ under $10^4$ permutations), observes boundary-pinned saturation rates ($\lambda$) on four of five frontier traces, and notes that constant models win corrected Akaike Information Criterion ($\text{AIC}_c$) selection across tested anchor families. The manuscript also documents a training collapse on `Nemotron-3-Super-120B-A12B` and formally withdraws two earlier claims from an associated 47-page draft: a false-positive Mixture-of-Experts (MoE) versus dense advantage ($+0.338$, $p = 0.023$, invalidated because Nemotron-120B is a LatentMoE model rather than dense, shifting one-sided $p$ to $\approx 0.178$) and an unverified leave-Nemotron sensitivity check. The author concludes that cross-scale differences reflect a stack-conditioned taxonomy of plateau, drift, and collapse rather than a Chinchilla-style scaling law in parameter count $N$.

---

## Strengths

- **Transparent self-correction and negative reporting:** The note explicitly retracts an earlier architecture comparison ($+0.338$ gap, $p = 0.023$) after correcting the classification of `Nemotron-3-Super-120B-A12B` from dense to LatentMoE (120B total / 12B active), and withdraws an unrepeatable leave-Nemotron sensitivity claim.
- **Clear scope boundaries:** Section 1 explicitly separates what is claimed (lack of identifiable size scaling in this specific anchor panel) from non-claims (no multi-seed power, no causal architecture ranking, and no universal ban on post-training scaling laws).
- **Concrete collapse diagnostics:** Rather than treating `Nemotron-120B` as a smooth scaling point, Table 1 quantifies its training instability using exact trajectory metrics (peak reward $0.875$ at step 3, last-10 mean reward $0.1625$, exact zero-reward step fraction $0.55$ or $11/20$, and $5/20$ steps with reward $> 0.25$).

---

## Weaknesses

- **Not self-contained as a scientific publication unit:** Despite claiming in Section 5 that "This short note is the publication unit," the manuscript omits the list of 12 anchor models, the mathematical form of the parametric scaling laws fitted ($\lambda$, $R_\infty$), the GRPO hyperparameter recipes, and the full statistical tables, relying instead on local file paths (`platform_hybrid/experiments/`, `frontier_gsm8k_*.json`, and `paper_P1_scaling.pdf`).
- **Severely under-powered and confounded experimental design:** Frontier runs are single-seed ("mostly single-seed", "Frontier anchors are single-seed") and at least `Nemotron-120B` runs for only 20 steps ("11/20", "5/20 steps"). Furthermore, the 12-anchor panel mixes dense and MoE architectures across different model families without defining whether $N$ measures total or active parameters, and Section 4 admits that endpoints "co-move with recipe, horizon, and architecture."
- **Unresolved contradiction in the withdrawn leave-Nemotron analysis:** Table 1 states that for the leave-Nemotron sensitivity check, "recomputation reverses the earlier non-significant story," which implies that removing the collapsed `Nemotron-120B` outlier makes the scaling or architecture relationship statistically significant. The note does not report this recomputed statistic or explain why a significant scaling trend on the remaining 11 non-collapsed models does not contradict the headline negative conclusion.
- **Complete absence of bibliography and formal citations:** The manuscript contains zero bibliographic references. It mentions GRPO, GSM8K, Chinchilla, Kimi-K2, DeepSeek-V3.1, Nemotron-3-Super-120B-A12B, and "Related public scaling-law work" without citing a single paper or model release.
- **Internal editorial memo tone:** Sections 2 and 5 read like an internal repository README or thesis erratum (e.g., "Do not submit the long PDF as a scaling-law paper") rather than a peer-reviewed conference submission.

---

## Anchored comments

- **[CRITICAL] p.1: "cross-scale slope of mean training reward is consistent with zero ... λ is pinned at the fit boundary on four of five frontier traces"** - The parametric model defining the saturation rate $\lambda$, asymptotic reward $R_\infty$, and boundary constraints is never written down, making the statistical claims impossible to evaluate mathematically. Write out the exact functional form fitted to the reward trajectories (e.g., $R(t) = R_\infty - \Delta R \exp(-\lambda t)$), state the optimization bounds for $\lambda$ and $R_\infty$, and define how "mean training reward" is aggregated across steps.
- **[CRITICAL] p.2: "Leave-Nemotron “sensitivity” / Withdrawn — no trusted checked-in TSV; recomputation reverses the earlier non-significant story"** - If recomputing the analysis without the collapsed 20-step `Nemotron-120B` run "reverses the earlier non-significant story," then excluding the single collapsed run yields a statistically significant result, which directly challenges the paper's central thesis. Report the exact recomputed slope, test statistic, and $p$-value for the 11-anchor leave-Nemotron panel and discuss whether the headline null result is driven entirely by a single 20-step training collapse.
- **[MAJOR] p.1: "on a multi-family GSM8K anchor panel (0.6B–1T, managed Tinker runs, mostly single-seed)"** - The 12 anchor models in the panel are never listed, nor is the definition of model size $N$ (total vs. active parameters for MoE models like Kimi-K2, DeepSeek-V3.1, and Nemotron-120B) specified. Add a table listing all 12 anchor models, their family, architecture class (dense vs. MoE), total parameters, active parameters, number of training steps, number of seeds, and fitted $(R_\infty, \lambda)$ values.
- **[MAJOR] p.1: "5 of 20 steps above reward 0.25"** - A training horizon of 20 steps is extremely short for post-training reinforcement learning and may reflect warmup or learning-rate miscalibration rather than an intrinsic scaling property. Report the step count, batch size, rollout group size $G$, and learning rate schedule for every anchor run, and justify why 20 steps is sufficient to estimate asymptotic reward $R_\infty$ and saturation rate $\lambda$.
- **[MAJOR] p.1: "a constant model wins AICc on every anchor family tested in the elevation suite"** - The candidate model set in the "elevation suite" and the sample size $n$ used to compute the small-sample correction in $\text{AIC}_c$ are omitted. Specify the competing functional forms, the number of data points $n$ per family (noting that $\text{AIC}_c$ carries a heavy penalty when $n$ is small, such as $2$–$4$ models per family), and report the $\Delta \text{AIC}_c$ values in Table 1.
- **[MAJOR] p.1: "Source traces live under platform_hybrid/experiments/ (scaling-law TSV suite and tinker-runs/results/frontier_gsm8k_*.json)."** - Referencing local, unlinked filesystem paths does not make the paper reproducible or self-contained for external readers. Replace local directory references with self-contained summary tables in the manuscript and provide an anonymized, public repository URL or supplementary archive containing the TSV and JSON artifacts.
- **[MINOR] p.1: "3 Negative results (recomputed)"** - Section 3 is an empty heading with no body text because Table 1 floated to the top of page 2. Add introductory prose in Section 3 that walks the reader through each row of Table 1 and connects the statistical tests to the claims in Section 1.
- **[MINOR] p.2: "The 47-page paper_P1_scaling.pdf retains elevation history for the thesis archive. This short note is the publication unit. Do not submit the long PDF as a scaling-law paper."** - This reads like private instructions to the author or lab group rather than academic prose. Rewrite Section 5 to objectively summarize what additional ablations or historical baselines exist in the extended technical report, and remove the self-directive sentence.
- **[SUGGESTION] p.2: "Related public scaling-law work occupies the broad positive claim space"** - This statement is vague and lacks scholarly context. Add a brief related-work paragraph citing foundational pre-training scaling laws (e.g., Kaplan et al., 2020; Hoffmann et al., 2022) and recent post-training/RL scaling analyses, explaining how your setup differs.

---

## Questions for the authors

1. **Leave-Nemotron recomputation:** In Table 1, you state that recomputing the leave-Nemotron sensitivity check "reverses the earlier non-significant story." What are the exact slope and $p$-value of the cross-scale reward trend and the MoE-vs-dense gap when `Nemotron-120B` is excluded? If a significant trend emerges after removing a single 20-step collapsed run, why is the overall conclusion framed as an identifiability failure of scaling rather than hyperparameter instability on one model?
2. **Definition of scale $N$ for MoE models:** In the cross-scale permutation tests ($\lambda$–$N$ $p=0.74$, $R_\infty$–$N$ $p=0.51$), do you use total parameters (e.g., 120B for Nemotron, 685B for DeepSeek-V3.1, 1T for Kimi-K2) or active parameters per token (e.g., 12B for Nemotron)? Does the flat-slope conclusion change if active parameters are used instead of total parameters?
3. **Complete 12-anchor specification and step horizons:** What are the exact 12 models in the panel, how many models belong to each family in the $\text{AIC}_c$ elevation suite, and how many training steps were run for each model? Were all frontier traces restricted to 20 steps like `Nemotron-120B`?
4. **Statistical power of the 12-anchor permutation test:** Given that endpoints "co-move with recipe, horizon, and architecture" across only 12 mostly single-seed points, what minimum effect size (slope in reward per $\log_{10} N$) was the permutation test powered to detect at $\alpha = 0.05$?

---

## Citations to verify

The manuscript contains **no References section and zero formal citations**, despite referencing several methods, datasets, scaling laws, and frontier models that require proper attribution and verification:

- **Missing foundational citations:** Group-Relative Policy Optimization / GRPO (Shao et al., 2024, *DeepSeekMath*), GSM8K (Cobbe et al., 2021), and "Chinchilla-style law" (Hoffmann et al., 2022) are central to the paper but uncited.
- **Unverified / potentially misstated model specifications:**
  - `Nemotron-3-Super-120B-A12B` ("LatentMoE: 120B total / 12B active"): Requires an official technical report or HuggingFace model card citation to verify the exact model naming and "LatentMoE" 120B/12B architecture specification.
  - `Kimi-K2 (1T)` and `DeepSeek-V3.1 (685B class)`: Require citations to their respective technical reports/releases (e.g., DeepSeek-V3 is typically cited as 671B total parameters; the "685B class" and "V3.1" designation should be verified and cited).
- **Unverifiable internal artifacts:** `paper_P1_scaling.pdf` (47-page draft), `platform_hybrid/experiments/`, and `tinker-runs/results/frontier_gsm8k_*.json` are local files not accessible to reviewers or readers.
- **Future manuscript date:** The manuscript is dated "August 2, 2026" on page 1.

---

## Overall score (1-10) and confidence (1-5)

- **Overall Score:** **2 / 10** (Strong Reject — While the transparency in retracting false-positive results is commendable, the manuscript is an incomplete internal erratum memo rather than a self-contained scientific paper. It lacks equations, data tables for the 12 anchors, hyperparameter details, and a bibliography, relies on 20-step single-seed confounded trajectories, and leaves an unexplained contradiction where removing the single collapsed run "reverses the earlier non-significant story.")
- **Confidence:** **5 / 5** (Absolutely certain — The manuscript is two pages long, and all technical omissions, statistical tensions, and formatting issues are directly verifiable from the text provided.)
