### Summary

This 3-page correction note retracts a prior claim that the i.i.d. Bernoulli model for the Zero-Variance Fraction ($\text{ZVF}_{\text{iid}} = p^G + (1-p)^G$) in Group-Relative Policy Optimization (GRPO) is "sign-wrong." The author demonstrates that two previously reported divergences with opposite signs ($+0.1224$ on Qwen3-8B/GSM8K and $-0.0703$ on synthetic arithmetic) were methodological artefacts caused by using two different estimators:
1. **Per-group plug-in bias:** Estimating $\hat{p}_i = k_i/G$ from the same group guarantees $\delta_i = h_G(k_i/G) - \mathbf{1}\{k_i \in \{0, G\}\} \ge 0$ for every group, yielding $+0.105$ in expectation even under exact i.i.d. Bernoulli rollouts at $\bar{p} = 0.694$.
2. **Step-mean Jensen gap:** Comparing $h_G(\bar{p}_t)$ against batch $\text{ZVF}_t$ across heterogeneous prompts yields $\mathbb{E}[\delta_t] = h_G(\bar{p}_t) - \mathbb{E}_i[h_G(p_i)] \le 0$ by convexity of $h_G$.

On 600 GSM8K groups ($G=8$, 3 seeds $\times$ 200 problems), the observed ZVF ($0.158$) exceeds the homogeneous-binomial baseline ($0.054$) and is closely approximated ($0.170$) by a two-parameter Beta-Binomial model ($\alpha=3.61, \beta=1.62$). The note explicitly disclaims any novel theoretical contribution or new diagnostic, crediting equivalent formulations to He et al. [1] and Nie et al. [2].

---

### Strengths

- **Exemplary scientific transparency:** The note directly diagnoses and retracts the author's prior mathematical and empirical errors, clearly separating estimator bias from physical rollout dynamics.
- **Crisp mathematical diagnosis:** The explanation of why the per-group plug-in estimator $\delta_i = h_G(k_i/G) - \mathbf{1}\{k_i \in \{0,G\}\}$ is deterministically non-negative and why the step-mean estimator is non-positive via Jensen's inequality is exact, elementary, and pedagogical.
- **Explicit scope and non-claims:** Sections 4 and 5 clearly state what is withdrawn (including an earlier pooled Spearman $\rho = 0.27$ over simulated/mixed cells) and accurately credit prior literature rather than overclaiming a corrected baseline as a new method.
- **Reproducibility commitment:** The manuscript links the exact script and checked-in JSON artefacts needed to regenerate every number without GPU compute.

---

### Weaknesses

- **No novel contribution for a main-track ML venue:** As the author explicitly states in the Abstract and Section 4 ("Nothing here is new... we claim no new diagnostic or theorem"), this manuscript is a retraction/erratum note for an earlier preprint rather than a standalone research paper suitable for NeurIPS or ICLR main proceedings.
- **Incomplete manuscript with raw placeholder text:** Section 3 contains an unfinished, margin-clipped placeholder (`[Placeholder — to be filled from experiments/results/publicatio`) for a pending held-out prediction experiment.
- **Unidentifiability in the empirical dataset:** Because the three GSM8K seeds sampled different problem sets without repeated group sampling per prompt, the dataset cannot distinguish between across-prompt difficulty dispersion ($\text{Var}(p_i)$) and true conditional within-prompt rollout correlation.
- **Terminological conflation of marginal ICC and within-prompt correlation:** The Abstract and Table 1 label the Beta-Binomial parameter $\rho = 1/(a+b+1) = 0.160$ as "intra-prompt correlation" / "ICC", while Section 3 states that the data "cannot separate prompt-difficulty dispersion from within-prompt rollout correlation." Under a Beta-Binomial mixture of conditionally i.i.d. Bernoulli trials, $\rho = 0.160$ *is* the marginal intra-class correlation induced entirely by prompt-difficulty dispersion.

---

### Anchored comments

- **[CRITICAL] p.2:** `"[Placeholder — to be ﬁlled from experiments/results/publicatio"` — The manuscript is incomplete and contains a literal placeholder that is also clipped at the right page margin. Fix: Complete the held-out prediction experiment before submission, or remove the "Prospective test (results pending)" paragraph entirely so the note is self-contained.
- **[CRITICAL] p.2:** `"The only contribution of this note is the retraction and the GSM8K numbers."` — A retraction of the author's own unpublished/preprint claim with no new theorem, diagnostic, or completed empirical validation does not meet the novelty threshold for NeurIPS/ICLR main proceedings. Fix: Submit this document as an arXiv replacement/erratum linked to the original note, or expand it into a full empirical study once the multi-sample held-out validation is complete.
- **[MAJOR] p.1 & p.2:** `"a two-parameter beta-binomial (intra-prompt correlation 0.16) predicts 0.170"` vs. `"we cannot separate prompt-diﬃculty dispersion from within-prompt rollout correlation with these artefacts"` — Calling the Beta-Binomial overdispersion parameter $\rho = (a+b+1)^{-1} = 0.160$ "intra-prompt correlation" in the Abstract confuses readers when Section 3 says within-prompt rollout correlation cannot be separated from prompt-difficulty dispersion. In a hierarchical model $p_i \sim \text{Beta}(a,b)$ with conditionally i.i.d. rollouts, marginal correlation between two rollouts on the same prompt equals $\text{Var}(p_i)/[\bar{p}(1-\bar{p})] = 0.160$ purely due to across-prompt variance in $p_i$. Fix: Clarify that $\text{ICC} = 0.160$ measures *marginal* intra-class correlation (equivalent to prompt-difficulty dispersion under conditional independence), which is distinct from *conditional* within-prompt rollout correlation given $p_i$.
- **[MAJOR] p.2:** `"χ2 of k-histogram vs. Binomial(8, ¯p) | 8119 (df = 7)"` — At $\bar{p} = 0.694$ and $N = 600$, the expected counts under $\text{Binomial}(8, 0.694)$ for $k=0$ and $k=1$ are $600 \times 0.306^8 \approx 0.046$ and $600 \times 8 \times 0.694 \times 0.306^7 \approx 0.83$. Because these expected counts are well below 1, a few all-zero groups inflate Pearson's $\chi^2$ to $8119$, violating the standard asymptotic assumptions of the $\chi^2(7)$ test. Fix: Report the log-likelihood ratio ($G$-test / deviance) or an exact Monte Carlo $p$-value alongside the raw $k$-histogram counts, or pool the low-count tail bins ($k \le 3$) before computing $\chi^2$.
- **[MINOR] p.1 & p.2:** `"−0.0703 on synthetic arithmetic"` (p.1, Abstract) vs. `"a negative mean (−0.067, n = 480 steps; 95% CI [−0.079, −0.056])"` (p.2, Section 2) — The two numbers for the synthetic arithmetic divergence do not match and the discrepancy is not explained. Fix: State explicitly why the retracted headline number was $-0.0703$ while the recomputed 480-step mean is $-0.067$ (e.g., subset of steps, seed filtering, or rounding error in the retracted draft).
- **[MINOR] p.2:** `"Mean success rate ¯p | 0.694"` and `"Beta-binomial ﬁt (a=3.61, b=1.62; ICC 0.160)"` — A Beta-Binomial with $a=3.61, b=1.62$ has mean $a/(a+b) = 3.61/5.23 = 0.6902 \neq 0.694$. Fix: Specify the fitting method used (e.g., maximum likelihood vs. method of moments, which would match $\bar{p} = 0.694$ exactly).
- **[SUGGESTION] p.1:** `"The relation pass@G −pG = 1 −hG(p) is an algebraic identity under the model"` — Since $\text{pass@}G = 1 - (1-p)^G$ under i.i.d. Bernoulli trials, substituting $h_G(p) = p^G + (1-p)^G$ yields $\text{pass@}G + p^G = 1 + p^G - (1-p)^G$, whereas $1 - h_G(p) = (1 - (1-p)^G) - p^G = \text{pass@}G - p^G$. Fix: Write out $\text{pass@}G = 1 - (1-p)^G$ inline so the identity $\text{pass@}G - p^G = 1 - h_G(p)$ (the probability of a non-degenerate group) is immediately transparent.

---

### Questions for the authors

1. **Synthetic arithmetic discrepancy:** Why does the Abstract cite $-0.0703$ for the synthetic-arithmetic divergence while Section 2 reports $-0.067$ ($n=480$ steps)?
2. ** GSM8K prompt overlap across seeds:** Do the three JSON artefacts (`tinker_gsm8k_zvf_s{42,123,456}.json`, 200 problems each from the 1,319-problem GSM8K test set) share any overlapping prompt IDs across seeds? Even partial overlap (~30–40 prompts by birthday-paradox expectation if sampled uniformly without replacement per seed) would allow an initial repeated-measures estimate of $p_i$ without waiting for new runs.
3. **Beta-Binomial goodness of fit:** While $\text{Binomial}(8, \bar{p})$ is rejected with $\chi^2 = 8119$, what is the goodness-of-fit $\chi^2$ (or log-likelihood) of the fitted $\text{Beta-Binomial}(8, 3.61, 1.62)$ model against the empirical $k$-histogram?

---

### Citations to verify

- **Manuscript date & future-dated arXiv identifiers (p.1, p.3):** The manuscript is dated `October 16, 2026`, and references [1], [2], and [3] carry 2026 arXiv identifiers (`arXiv:2605.21125`, `arXiv:2605.07689`, `arXiv:2610.00574`). Notably, `arXiv:2605.21125` ([1] He et al., 2026) has a sequence number (`21125`) that is unusually high for a single monthly arXiv volume, and all three 2026 preprints should be verified for exact title, author list, and identifier accuracy.
- **Reference [6] (p.3):** G. Jiang et al., *"VCRL: Variance-based curriculum reinforcement learning for large language models"*, `arXiv:2509.19803`, 2025 — Verify that the author list and arXiv ID match the published/archived record.

---

### Overall score and confidence

- **Overall Score:** **2 / 10** (Strong Reject for a main conference track such as NeurIPS/ICLR due to being an incomplete erratum/retraction note with a cut-off placeholder and no claimed novel contribution; however, it is a commendable and rigorous **erratum** once the placeholder is resolved).
- **Confidence:** **5 / 5** (The mathematical derivations, binomial/beta-binomial calculations, and statistical arguments in the text are brief, self-contained, and directly verifiable from first principles).
