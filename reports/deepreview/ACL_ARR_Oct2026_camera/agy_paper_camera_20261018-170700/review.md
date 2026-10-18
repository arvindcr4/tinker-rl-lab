### Summary

This four-page paper presents negative and cautionary empirical results regarding algorithm attribution in reinforcement-learning post-training for language models (specifically the GRPO family). Across 42 experiments grouped into seven exhibits (E-1 through E-7), the authors evaluate how differences in software frameworks, managed backends, hidden base-checkpoint changes, and implementation surrogates affect outcomes on GSM8K exact match. The experiments show that:
1. Marginal one-way ANOVAs yield comparable $\eta^2$ for framework (0.546) and algorithm (0.558) across an observational, collinear 42-run corpus (E-1).
2. Same-stack comparisons between PPO and GRPO on Qwen2.5-0.5B (E-2) and among four GRPO variants on a frozen stack (E-4) yield non-significant, inconclusive results.
3. Cross-backend comparisons suffer from provenance conflicts, autocorrelation, or extreme confounding ($d = 12.75$, E-3; 5.0% to 85.6% reward shift due to silent base-checkpoint pinning, E-5).
4. Identical method labels ("DAPO") produce opposite zero-variance fraction (ZVF) telemetry across open and closed stacks (0.00 vs. 0.58, E-6), whereas four distinct labels on a fixed stack compress into a 0.034 reward band (E-7).

To address these ambiguities, the authors propose an eight-item minimum reportable stack, accompanied by a JSON schema, a 20-record seed registry, an outcome-blind auditor, and a manifest diff tool that flags cross-stack risks on a six-rung ladder (R0–R5).

---

### Strengths

- **Clear distinction between null results and equivalence:** The manuscript consistently distinguishes between failing to detect an effect and proving equivalence (e.g., E-2 $p = 0.374$, and E-4 where the exact MDE of 0.0101 exceeded the 0.010 pre-registered design margin). Retracting an earlier aggregator's "differences disappear" claim in E-4 demonstrates high statistical integrity.
- **Concrete mechanistic exhibits of stack confounding:** Exhibits E-5 (silent base-checkpoint bundling causing a 17$\times$ reward change) and E-6 (the label "DAPO" denoting true dynamic sampling with ZVF 0.00 on one stack vs. an asymmetric-clip surrogate with ZVF 0.58 on another) provide concrete evidence of how label-only comparisons mislead readers.
- **Actionable reporting schema:** Separating unreported fields (`null`) from explicitly absent features (e.g., "no KL term") in the JSON schema turns reporting omissions into mechanically auditable R5 risks via the diff tool.
- **Candid limitations:** The authors explicitly state the constraints of their study: short training horizons (30–50 steps), small-to-mid model scales, closed managed-service internals, collinearity in E-1, and coarseness in the 20-record registry.

---

### Weaknesses

- **Missing experimental specifications in core exhibits:** A paper advocating an eight-item minimum reportable stack omits basic experimental details in its own exhibits. Specifically, E-4, E-5, and E-6 do not state which base model(s) were trained, and E-2, E-3, E-5, E-6, and E-7 do not report their exact MDEs or observed power despite Section 2 promising them for all tests.
- **Undefined baseline and variants:** The variant "AERO" in E-4 is introduced without citation, mathematical definition, or explanation of how it differs from GRPO.
- **Severely underpowered regimes and short horizons:** Running Qwen2.5-0.5B for only 40 steps on GSM8K (E-2) or managed runs for 30–50 steps (Limitations) is far shorter than standard RL post-training regimes. Failure to separate algorithms at 40 steps on a 0.5B model provides weak evidence about whether algorithm differences survive in realistic training regimes.
- **Information density vs. completeness:** Because the paper is compressed into four pages without an appendix, the reader cannot inspect the 42 experiments in E-1, the exact mathematical definitions of the eight reporting items, or the 20 seed records in the registry.

---

### Anchored comments

- **[CRITICAL] p.2: "Holding a frozen stack fingerprint, we compared DAPO, GSPO, Dr. GRPO and AERO against GRPO over eight paired seeds, evaluating each unit on 500 held-out GSM8K items."** - Exhibit E-4 is the flagship controlled variant audit of the paper, but it omits the base model identity, step budget, learning rate, group size $G$, and framework used. Furthermore, "AERO" is never defined or cited anywhere in the manuscript. **Fix:** Explicitly state the base model, framework, group size, and training horizon for E-4 in Section 3 and Table 1, and provide a citation or formal definition for AERO.
- **[MAJOR] p.1: "We use Welch or paired t-tests with Benjamini–Hochberg correction across the test family, report effect sizes with 95% confidence intervals, and state observed power and the minimum detectable effect (MDE)."** - The text promises 95% CIs, observed power, and MDE across comparisons, but E-2 only reports $\Delta = -0.002, p = 0.374$ (missing 95% CI, observed power, and MDE), E-3 (Qwen3-8B) only reports $p = 0.7605$ (missing effect size and CI), and E-4 omits the point estimates, 95% CIs, and $p$-values for DAPO and Dr. GRPO in the prose (Table 1 gives point estimates $+0.00100$ and $-0.00200$ without CIs or $p$-values). **Fix:** Add a complete statistical table reporting $n$, point estimate, 95% CI, raw and BH-adjusted $p$-values, observed power, and MDE for every contrast in E-2, E-3, E-4, and E-7.
- **[MAJOR] p.2: "Under a fixed label, changing the backend moved final training reward from 5.0% to 85.6%. We do not report this as a backend effect, because the same inspection found that the managed backend had silently pinned a different base checkpoint."** - E-5 and E-6 omit the specific method label, model family, and backends involved in E-5, as well as the model and step count in E-6. **Fix:** Specify which method label, nominal model name, and actual pinned base checkpoints produced the 5.0% vs. 85.6% result in E-5, and state the model and step count for the two DAPO stacks in E-6.
- **[MAJOR] p.2: "Across 42 experiments, four separate one-way ANOVAs give marginal $\eta^2 = 0.546$ for library or framework, 0.558 for training algorithm, 0.471 for model family and 0.322 for model scale."** - While the authors rightly note that these marginal $\eta^2$ values sum to 1.897 and do not partition variance due to collinearity, reporting unadjusted one-way ANOVAs on a convenience corpus where framework and model/backend are confounded can still mislead readers into thinking 0.546 vs. 0.558 is a meaningful quantitative comparison. **Fix:** Report Type II or Type III partial $\eta^2$ (or sequential variance decomposition bounds across factor orderings) alongside the marginal values, and provide a summary cross-tabulation of the 42 experiments across framework, algorithm, and model family.
- **[MINOR] p.2: "Running GRPO, DAPO, Dr. GRPO and GSPO on one managed stack (Qwen3.5-4B, GSM8K, three seeds per arm), mean last-10 training reward spans 0.710–0.744... On this corpus the algorithm term explains $\eta^2 = 0.0075$ of reward variance, while the group-size axis explains far more."** - Table 1 describes E-7 as "12 cells" (which matches 4 algorithms $\times$ 3 seeds, or 4 algorithms $\times$ 3 group sizes), and the text mentions a "group-size axis" without stating what group sizes were swept or what $\eta^2$ the group-size axis actually achieved. **Fix:** Clarify the exact factorial design of E-7 (number of group sizes $\times$ algorithms $\times$ seeds), and report the exact $\eta^2$ for the group-size factor.
- **[MINOR] p.2: "its $n = 10$ is the last ten autocorrelated steps of one run per arm. On Llama-3.1-8B-Instruct the cross-back-end contrast gives $d = 12.75$ ([8.49, 17.00], $p < 0.001$)."** - It is unclear whether the Llama-3.1-8B-Instruct contrast ($d = 12.75$) also used $n = 10$ autocorrelated training steps from a single run per arm, which would artificially deflate variance and inflate Cohen's $d$. **Fix:** Explicitly state the sample unit and sample size $n$ for the Llama-3.1-8B-Instruct contrast in E-3.
- **[SUGGESTION] p.3: "An outcome-blind auditor scores each record from 0 to 100 against the items; on the twenty seed records, badges span 43–96... A diff tool maps a pair of records onto a six-rung risk ladder, from identical manifests (R0) to unauditable (R5, a differing item is null on one side)."** - Rungs R1 through R4 of the risk ladder and the weighting scheme of the 0–100 auditor score are never defined in the text. **Fix:** Add a compact table or inline list defining rungs R0–R5 and the exact point weights assigned to the eight reporting items.

---

### Questions for the authors

1. **Details of Exhibit E-4:** What base model, framework, group size $G$, and training step budget were used in E-4, and what is the exact definition or citation for the "AERO" variant?
2. **Sample units in Exhibit E-3 and E-1:** In E-3, is the Llama-3.1-8B-Instruct Cohen's $d = 12.75$ computed across independent random seeds on held-out accuracy, or across the last 10 autocorrelated training steps of a single run like the Qwen3-8B row? In E-1, what is the exact dependent variable across all 42 experiments (held-out accuracy or last-10 training reward)?
3. **Factorial structure of Exhibit E-7:** How are the "12 cells" and "three seeds per arm" structured across the four algorithms and the "group-size axis" in E-7, and what is the $\eta^2$ for group size?
4. **Risk ladder definitions:** How are rungs R1, R2, R3, and R4 defined in the manifest diff tool?

---

### Citations to verify

- **Zheng et al. (2025) — Group Sequence Policy Optimization (`ArXiv:2507.18071`):** Verify the arXiv identifier and metadata. Note that the manuscript also cites `Qwen3.5-4B` (p.2, E-7) and `Thinking Machines Lab (2025)` (Tinker API); verify that the exact model naming (`Qwen3.5-4B` vs. Qwen2.5/Qwen3 series) matches the official release artifacts used in the registry.
- **von Werra et al. (2022) — TRL: Transformer reinforcement learning:** TRL was initially released in 2020; verify the preferred citation year and metadata for the version of TRL evaluated in the study.
- **"AERO" (p.2, Table 1 and E-4):** Used as one of the four audited GRPO variants in E-4, but has no corresponding entry in the References section.

---

### Overall score and confidence

- **Overall Score:** **6 / 10** (Marginally above the acceptance threshold. The paper provides a clear, statistically honest message and practical tooling for a timely problem in LLM RL post-training, but it must fill in the missing experimental details in its own exhibits—such as the missing model identities in E-4/E-5/E-6, the undefined AERO variant, and the missing power/MDE statistics promised in Section 2.)
- **Confidence:** **4 / 5** (Confident in the evaluation of the experimental design, statistical methodology, and RL post-training literature based on the provided manuscript text.)
