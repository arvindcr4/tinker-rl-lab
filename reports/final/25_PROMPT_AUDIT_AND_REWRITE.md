# 25-Prompt Comprehensive Audit: TinkerRL Paper

**Paper:** "When Does GRPO Have a Learning Signal? Reward Diversity Diagnostics and Stack Effects in LLM Post-Training"

**Date:** 2026-04-22

---

## PROMPT 1: Main Upgrade — Read and Improve

### Current Thesis
The paper argues three central claims:
1. **Reward contrast**: ZVF/GU (Zero-Variance Fraction / Gradient Utilization) are useful diagnostics for GRPO training signal
2. **Stack identifiability**: GRPO hyperparameters alone don't fully specify the training process—backend, sampler, LoRA config, etc. matter
3. **Reward-vs-capability separation**: Training reward conflates with held-out benchmark accuracy in most prior work

### Key Improvement Opportunities

**1. Stronger Central Claim**
The paper's thesis is sound but too broad. The strongest defensible claim is:
> *"GRPO training reward is non-identifiable across frameworks: nominally identical configurations produce wildly different outcomes due to hidden stack variables. ZVF/GU provide actionable triage diagnostics for this non-identifiability."*

This is demonstrable, falsifiable, and addresses a real gap in the literature.

**2. Claims Requiring Deflation**
- The held-out evaluation (Table 4) is checkpoint-selected, not a clean controlled experiment. Reading it as "GRPO improves GSM8K by X%" is overclaim.
- ZVF correlation with final reward ($r = -0.769$) is largely tautological on binary rewards (Section 6.4 explicitly caveats this)
- Cross-library PPO comparisons are stack-mismatch evidence, not PPO failure evidence

**3. Structural Improvement**
The paper would benefit from leading with the non-identifiability thesis, then showing ZVF/GU as a diagnostic response. Currently it buries the non-identifiability point in results and over-indexes on ZVF statistics.

---

## PROMPT 2: Claim Audit

### Claim 1: "ZVF has a large zero-order association with final performance ($r = -0.769$, $p = 0.0008$)"

| Aspect | Rating |
|--------|--------|
| Evidence level | Correlation on 15 experiments, single-seed heterogeneity |
| Overclaim risk | **HIGH** — ZVF is a monotone function of per-prompt accuracy $p$: $ZVF(p,G) = p^G + (1-p)^G$. Correlating ZVF with reward is correlating $f(p)$ with $p$. |
| Exact rewrite | "ZVF correlates with reward on binary tasks because both measure per-prompt accuracy. The partial correlation controlling for $R_t$ at lag 5 is $r = 0.059$ ($p = 0.096$): ZVF has marginal independent signal at longer horizons. ZVF's practical value is as an early-failure triage diagnostic." |

### Claim 2: "GRPO achieves competitive performance across model scales from 0.6B to 235B"

| Aspect | Rating |
|--------|--------|
| Evidence level | Single-seed training reward, mostly partial runs |
| Overclaim risk | **HIGH** — "Performance" conflates training reward with capability. Most Tinker runs are single-seed with no variance estimate. |
| Exact rewrite | "GRPO produces training reward trajectories spanning 0.6B to 235B models under the GSM8K rollout parser. Held-out accuracy evaluation covers 10 checkpoints only. Generalization claims require multi-seed held-out evaluation." |

### Claim 3: "PPO vs GRPO evidence shows algorithm-label is insufficient"

| Aspect | Rating |
|--------|--------|
| Evidence level | Single-seed, artifact-sensitive (PPO last-10 differs between ledger 22.5% and statistics summary 35.0%) |
| Overclaim risk | **MEDIUM** — The evidence supports stack-sensitivity, not algorithm comparison |
| Exact rewrite | "GRPO and PPO labels are under-specified unless model family, backend, reward parser, sampler, LoRA config, optimizer, checkpoint selection, and evaluator are reported. The PPO vs GRPO rows are evidence for stack-conditioned reporting, not an algorithm ranking." |

### Claim 4: "TRL-GRPO on Qwen3-8B reaches 84.4% last-10 vs TRL's 5.0%"

| Aspect | Rating |
|--------|--------|
| Evidence level | Single-seed runs on different backends |
| Overclaim risk | **MEDIUM** — Demonstrates stack sensitivity but does not prove Tinker is better than TRL |
| Exact rewrite | "Under nominally matched visible hyperparameters (Qwen3-8B, G=8, lr=10⁻⁵, GSM8K-500, 30 steps), Tinker reaches 84.4% last-10 reward while TRL-GRPO on Modal H100 reaches 5.0%. This gap is evidence that visible GRPO hyperparameters alone do not fully specify training dynamics." |

### Claim 5: "GRPO is secretly DPO at G=2 (Wu et al. 2025)"

| Aspect | Rating |
|--------|--------|
| Evidence level | Theoretical reference + empirical validation on 10 experiments |
| Overclaim risk | **LOW** — This is a theoretical result from cited work |
| Exact rewrite | "Following Wu et al. (2025), GRPO at G=2 is algebraically equivalent to DPO, where GU(p,2) = 2p(1-p) reproduces the DPO preference likelihood weighting. For our DeepSeek-V3.1 run (p≈0.85, G=8), 27% of steps were zero-gradient; switching to G=2 would reduce rollouts by 75% while preserving informative contrastive signal." |

---

## PROMPT 3: A-Level Thesis Check

### Is the central thesis NeurIPS/ICML/ICLR-worthy?

**Verdict: YES, with refinements**

### Strengths
- Non-identifiability of GRPO across frameworks is a real and important problem
- ZVF/GU as triage diagnostics is actionable and falsifiable
- The empirical demonstration (nominally matched configs → divergent outcomes) is compelling
- Connects to concurrent work (AERO, 2-GRPO/DPO equivalence, Dr. GRPO) cleanly

### What Must Be Demoted
1. **Generalization claims** — "GRPO improves reasoning" → "GRPO produces training reward gains that may or may not generalize"
2. **ZVF as predictor** → "ZVF as early triage diagnostic"
3. **PPO vs GRPO ranking** → "PPO and GRPO labels are under-specified treatment labels"
4. **Scaling law claims** → "Descriptive exponential saturation fits, not validated scaling laws"

### Single Strongest Thesis
> *"GRPO training dynamics are non-identifiable from hyperparameter labels alone: framework, backend, reward parser, sampler, LoRA configuration, and evaluator jointly determine outcomes in ways that hyperparameter tables do not capture. We introduce ZVF/GU as actionable diagnostics for this non-identifiability."*

This thesis is:
- Defensible with existing evidence
- Novel (no prior work systematically demonstrates stack non-identifiability)
- Actionable (practitioners can use ZVF/GU immediately)
- Falsifiable (if someone shows hyperparameter labels are sufficient, the thesis fails)

---

## PROMPT 4: Trust Calibration

### Finding every place where the paper asks for more trust than evidence supports

**1. Abstract: "demonstrates significant cross-library performance variance"**
- Risk: "Significant" implies statistical significance, but variance is driven by single-seed Tinker runs with no variance estimate
- Fix: "demonstrates substantial cross-library performance divergence under nominally matched configurations"

**2. "GRPO achieves near-perfect training reward on frontier models"**
- Risk: Peak reward 100% on Qwen3-235B-A22B is early-training snapshot (15 steps, partial)
- Fix: "reaches 100% peak reward in early training on Qwen3-235B-A22B (15 steps)"

**3. "The capacity ceiling of our 30-step GRPO recipe on GSM8K is model-dependent: 95.0% for Llama-3.3-70B-Instruct"**
- Risk: This is a single-seed, checkpoint-selected observation, not a controlled ceiling estimate
- Fix: "In our single-seed 30-step runs, Llama-3.3-70B-Instruct reached 95.0% held-out accuracy on the top-10 checkpoint selection"

**4. "ZVF provides an early-warning signal for training collapse"**
- Risk: The "early-warning" framing implies predictive validity, but the lagged regression shows marginal independent signal only at lag 10 ($p=0.045$)
- Fix: "ZVF ≥ 80% sustained over 5+ steps with reward ≤ 5% correctly identifies all collapsed runs (precision 1.0, recall 1.0) in our corpus, but this should be interpreted as regime identification, not prediction"

**5. "This paper introduces TinkerRL-Bench"**
- Risk: TinkerRL-Bench is a loose collection of tasks and libraries, not a standardized benchmark with published evaluation protocol
- Fix: "This paper reports experiments across TinkerRL-Bench, a benchmark design spanning 8+ RL libraries, 3 task domains, and 0.6B–235B model scales"

---

## PROMPT 5: Contribution Ranking

### Rank contributions by originality and defensibility

| Rank | Contribution | Originality | Defensibility | Recommendation |
|------|--------------|-------------|---------------|----------------|
| 1 | Stack non-identifiability: nominally matched configs → divergent outcomes | **HIGH** | **HIGH** | Foreground in abstract, intro, results |
| 2 | ZVF/GU as triage diagnostics | **MEDIUM** | **MEDIUM-HIGH** | Foreground with honest caveats on tautology |
| 3 | Cross-library empirical survey (8+ libraries, 0.6B–235B) | **MEDIUM** | **MEDIUM** | Foreground with single-seed caveats |
| 4 | GRPO-as-DPO validation (G=2 equivalence, group size ablation) | **MEDIUM** | **HIGH** | Foreground with theoretical backing |
| 5 | Policy drift proxy analysis (SI, PTD, Rolling Variance) | **LOW-MEDIUM** | **MEDIUM** | Appendix with honest proxy caveats |
| 6 | Exponential saturation fits | **LOW** | **LOW** | Appendix only |
| 7 | Cross-task tool-use results (0% reward) | **LOW** | **LOW** | Appendix as negative baseline |
| 8 | Dense vs MoE comparison | **LOW** | **LOW** | Appendix (confounded by base vs instruct) |

**What to Foreground:**
- Stack non-identifiability (Contributions 1)
- ZVF/GU triage diagnostics (Contribution 2)
- Cross-library empirical survey (Contribution 3)

**What to Move to Appendix:**
- Exponential saturation fits
- Cross-task tool-use results (0% reward baseline)
- Dense vs MoE comparison (confounded)
- Policy drift proxies (SI, PTD)

---

## PROMPTS 6-7: Warm-Up Then Hard Problem

### Prompt 6: Evidence Hierarchy (10 bullets)

1. **Strongest evidence:** Framework comparison under nominally matched config (Tinker vs TRL on Qwen3-8B, G=8, lr=10⁻⁵, GSM8K, 30 steps) → 84.4% vs 5.0% last-10 reward
2. **Strong evidence:** Cross-library PPO implementations (SB3, CleanRL, Tianshou) achieve near-random on tokenized arithmetic task → stack-mismatch, not PPO failure
3. **Strong evidence:** ZVF ≥ 80% + reward ≤ 5% for 5+ steps identifies all collapsed runs (precision 1.0, recall 1.0) in corpus
4. **Moderate evidence:** ZVF correlation with final reward ($r = -0.769$) is largely tautological; partial correlation controlling for $R_t$ at lag 5 is $r = 0.059$ ($p = 0.096$)
5. **Moderate evidence:** Held-out evaluation of top-10 checkpoints (87.2%–95.0% band, mean 91.6%) — but checkpoint-selected
6. **Moderate evidence:** Group size ablation (G=2,4,8,16) shows inverted-U on Qwen3-8B, peaks at G=8 — single seed
7. **Weak evidence:** Cross-architecture scaling (0.6B–235B) — single-seed training reward only
8. **Weak evidence:** Dense vs MoE at matched active parameters — confounded by base vs instruct initialization
9. **Weak evidence:** Tool-use task (0% reward) — task design problem, not capability failure
10. **Weak evidence:** Exponential saturation fits (mean $R^2 = 0.210$) — short traces, poor fit quality

### Prompt 7: Simplified Mixed-Reward-Group Mechanism

**Simple explanation:**
GRPO trains by generating G completions per prompt and rewarding the ones that beat their group average. When all G completions get the same reward (e.g., all wrong), the group contributes zero gradient — this is the Zero-Variance Fraction (ZVF). ZVF = 1 means no learning signal. ZVF = 0 means full gradient utilization.

**Strongest theorem-like statement the paper can safely claim:**

> *For binary-reward tasks with per-prompt accuracy $p$ and group size $G$: $ZVF(p,G) = p^G + (1-p)^G$ and $GU(p,G) = 1 - p^G - (1-p)^G$. When $p \in [0.2, 0.8]$ and $G \geq 8$, marginal GU gain from increasing $G$ approaches zero. At $G=2$, $GU(p,2) = 2p(1-p)$, algebraically equivalent to DPO preference weighting (Wu et al., 2025).*

**What the paper CANNOT safely claim:**
- ZVF predicts final performance (it's largely confounded by current reward)
- GRPO broadly improves reasoning (held-out evidence is checkpoint-selected)
- PPO is worse than GRPO (evidence is single-seed and backend-confounded)

---

## PROMPTS 8-12: One-Step Editing Prompts

### Prompt 8: Abstract Rewrite

**Original abstract framing:** Multi-platform benchmark, cross-library variance, model-dependent algorithm preference

**Rewritten abstract (thesis: reward contrast, stack identifiability, reward-vs-capability separation):**

> Group Relative Policy Optimization (GRPO) is widely adopted for LLM post-training, yet training dynamics remain poorly understood and non-identifiable across implementations. We present TinkerRL-Bench, a cross-library benchmarking framework spanning 8+ RL libraries and models from 0.6B to 235B parameters, and identify two structural phenomena that govern whether GRPO has a learning signal at all.

> First, we introduce Zero-Variance Fraction (ZVF) and Gradient Utilization (GU) as triage diagnostics for GRPO training signal. ZVF measures the fraction of prompts where all G completions receive identical rewards, producing zero gradient. Across 15 experiments, ZVF ≥ 80% sustained over 5+ steps with reward ≤ 5% correctly identifies all collapsed runs (precision 1.0, recall 1.0); ZVF is task-driven, not model-scale-driven. Second, we demonstrate that GRPO hyperparameters are insufficient to specify training dynamics: nominally matched configurations (Qwen3-8B, G=8, lr=10⁻⁵, GSM8K) produce 84.4% last-10 reward on Tinker and 5.0% on TRL, proving that backend, sampler, reward parser, LoRA configuration, and evaluator are co-determinants of outcome. Third, we show that training reward conflates with held-out capability in most prior work: held-out GSM8K evaluation of the top-10 checkpoints (selected by training reward) yields an 87.2%–95.0% accuracy band, demonstrating that checkpoint-selected training reward does not cleanly predict held-out generalization.

> We do not claim GRPO universally improves reasoning, nor that our hyperparameter configurations are globally optimal. Our contribution is diagnostic: characterizing the structural boundaries of reward variance and stack identifiability that determine whether critic-free RL produces learning signal.

### Prompt 9: Introduction Rewrite (first two pages)

**Rewritten intro structure (claim-to-evidence audit):**

**Paragraph 1 (Problem statement):** GRPO is widely adopted but its training dynamics are poorly understood and non-identifiable across implementations.

**Paragraph 2 (Central thesis):** We identify two structural phenomena governing GRPO learning signal: (1) reward contrast via ZVF/GU, and (2) stack non-identifiability.

**Paragraph 3 (Evidence for claim 1):** ZVF = 1 means zero gradient on a prompt; ZVF is task-driven, not model-scale-driven. Tool-use experiments saturate at ZVF = 100% regardless of model size; GSM8K experiments average ZVF = 8.5%. ZVF ≥ 80% + reward ≤ 5% for 5+ steps identifies all collapsed runs in our corpus (precision 1.0, recall 1.0).

**Paragraph 4 (Evidence for claim 2):** GRPO hyperparameters are insufficient. Nominated matched config (Qwen3-8B, G=8, lr=10⁻⁵, GSM8K) → Tinker: 84.4% last-10, TRL: 5.0% last-10. This gap is not a framework ranking; it proves the treatment label is under-specified.

**Paragraph 5 (Reward-vs-capability separation):** Training reward conflates with held-out accuracy in prior work. Held-out evaluation of top-10 checkpoints (selected by training reward) yields 87.2%–95.0% band. Checkpoint selection by training reward is unreliable: four Llama-3.3-70B-Instruct seeds range from 95.0% to 98.1% last-10 but all land on 95.0% held-out.

**Paragraph 6 (Contribution statement):** We contribute: (1) ZVF/GU triage diagnostics, (2) stack non-identifiability evidence, (3) honest held-out evaluation with checkpoint-selection caveats, (4) cross-library benchmark spanning 8+ libraries and 0.6B–235B models.

### Prompt 10: Results Rewrite (no conflation)

**Rewritten Results section structure:**

**Key principle:** Every result sentence must specify whether it reports (a) training reward, (b) held-out accuracy, or (c) proxy reward.

**Section 1: Training-reward ledger (Table 1)**
"These rows report online reward under the GSM8K rollout/evaluation parser, not held-out benchmark accuracy. Peak reward and last-10-step mean reward are training-environment summaries. Tinker runs are single-seed; Modal runs are single-seed on H100."

**Section 2: Held-out evaluation (Table 4)**
"These rows report greedy accuracy on a fixed N=500 slice of the GSM8K test split, evaluated after training. Checkpoints are selected by ranking all ~70 campaign runs by training last-10 reward and taking the top ten. This is a checkpoint-selection analysis, not evidence that GRPO caused a significant held-out capability gain."

**Section 3: Proxy rewards (tool-use, code)**
"The tool-use reward scores JSON well-formedness, tool-name selection, and argument-key presence without executing tools. It measures tool-call schema compliance, not tool use. The code row measures behavior under a broken verifier that rejects legitimate completions using len, range, or sum."

**Section 4: ZVF/GU diagnostics**
"ZVF and GU are training-signal diagnostics, not performance metrics. ZVF = 1 means zero gradient; GU = 1 - ZVF measures the fraction of prompts providing informative signal."

### Prompt 11: Related Work Rewrite

**Position against key related work:**

**vs. DeepSeekMath (Shao et al., 2024):**
DeepSeekMath uses GRPO with reference-policy KL penalties and shows strong GSM8K/MATH improvements. Our work does not contradict this: we show that GRPO produces learning signal on GSM8K, but that training reward confounds with held-out accuracy in checkpoint-selection analyses. We additionally show that stack variables (backend, sampler, LoRA config) modulate the effective learning signal in ways that hyperparameter tables do not capture.

**vs. RL-ZVP (Zhang et al., 2026):**
AERO motivates pre-screening prompts by the prevalence of zero-advantage dead zones in GRPO. Our ZVF measurement ($r = -0.769$ with final performance) provides empirical calibration for the phenomenon AERO targets. Importantly, our data reveal ZVF is task-driven, not model-scale-driven: tool-use experiments saturate at ZVF = 100% regardless of model size, suggesting AERO's compute savings will be largest when task–model alignment is poor.

**vs. Online Difficulty Filtering:**
Difficulty-filtering approaches select for solvable prompts to maintain gradient flow. Our ZVF/GU diagnostics complement this: ZVF monitors gradient flow directly rather than filtering prompts by estimated difficulty. ZVF is regime-identifying (cold-start collapse vs. saturation vs. healthy contrast), not a difficulty filter.

**vs. Spurious Rewards (Kirk et al., 2023; related literature):**
Spurious reward hacking occurs when models maximize proxy rewards without genuine capability gain. Our held-out evaluation partially addresses this: checkpoint selection by training reward is unreliable (four Llama-3.3-70B-Instruct seeds range 95.0%–98.1% last-10 but all land on 95.0% held-out). We recommend held-out evaluation as a standard safeguard against spurious reward conflation.

**vs. Deep-RL Reproducibility (Henderson et al., 2018; Islam & Elmek, 2017):**
Henderson et al. (2018) show that deep-RL algorithm comparisons are often underpowered and non-reproducible due to high variance and sensitivity to hyperparameters. Our work replicates this pattern in LLM post-training: single-seed comparisons are unreliable (MDE $d = 2.024$ at 80% power for n=5), and nominally identical configurations produce divergent outcomes. We recommend multi-seed replication and explicit stack reporting as reproducibility minimums.

### Prompt 12: Limitations Rewrite

**Harder-to-attack limitations while preserving strongest contribution:**

**Strengthened limitations:**

1. **Closed-source training implementation:** "Tinker is a commercial, closed-source API. We cannot inspect the exact GRPO loss formulation, reward normalization scheme, or hardware configuration. Our Tinker results measure the platform's implementation, not a precisely specified algorithm. Researchers attributing performance to specific implementation choices should use open-source backends (TRL, OpenRLHF, veRL) where every hyperparameter is auditable."

2. **Short training horizons:** "All Tinker experiments used 30 gradient steps—a budget choice to contain API costs. Long-horizon effects (reward hacking, catastrophic forgetting, late-stage collapse) are unlikely to manifest at this horizon. We regard our results as early-training snapshots rather than converged solutions."

3. **Train-set reward as primary metric:** "Reported rewards are computed on the same prompt distribution used for training, not a held-out test split. Held-out evaluation covers only the top-10 checkpoints on GSM8K; tool-use and xLAM results remain train-set-only. Training-reward gains may partly reflect prompt-level memorization."

4. **ZVF correlation is partially tautological:** "ZVF's zero-order correlation with final reward ($r = -0.769$) is largely explained by current reward level on binary-reward tasks: the partial correlation controlling for $R_t$ at lag 5 is only $r = 0.059$ ($p = 0.096$). ZVF is useful for triage, not as an independent causal predictor."

5. **Algorithm-label effects are not identified:** "PPO vs GRPO comparisons are single-seed and backend-confounded. We cannot detect medium or large effects (MDE $d = 2.024$ at 80% power for n=5). We use these comparisons to motivate stack-conditioned reporting, not to claim algorithmic superiority."

---

## PROMPTS 13-16: Reviewer Simulation

### Prompt 13: Hostile NeurIPS Reviewer — Top 10 Rejection Reasons

**1. Insufficient statistical power**
"The paper reports single-seed Tinker experiments with no variance estimate. With n=1, no significance test applies. The minimum detectable effect at 80% power for n=5 per arm is Cohen's $d = 2.024$—an enormous effect. Most claimed differences in the paper fall below this threshold."

**2. Checkpoint selection invalidates held-out claims**
"Table 4 (held-out GSM8K) selects the top-10 checkpoints by training last-10 reward. This is selection bias: the paper evaluates checkpoints it already knows performed well on training. This is not a held-out generalization result."

**3. Tautological ZVF correlation**
"ZVF = $p^G + (1-p)^G$ for binary rewards. Correlating ZVF with reward is correlating $f(p)$ with $p$. The lagged regression (Section 6.4) confirms: partial correlation controlling for $R_t$ at lag 5 is $r = 0.059$ ($p = 0.096$)—not significant. The paper acknowledges this but still foregrounds ZVF as a key contribution."

**4. Missing comparison to strong baselines**
"The paper compares to TRL-GRPO, PPO from classical RL libraries (SB3, CleanRL, Tianshou), and self-reported prior work. There is no comparison to DeepSeekMath, Qwen-Math, or other state-of-the-art GRPO implementations that achieve 50%+ GSM8K improvement. The claimed 'competitive performance' is against a weak baseline."

**5. Tool-use task is not evaluated**
"The tool-use task yields 0% reward for both Qwen3-32B and Llama-3.1-8B. This is a failed experiment, not a negative result. The paper does not demonstrate learning on any tool-use task."

**6. Concrete algorithm recommendation missing**
"The paper provides diagnostics (ZVF/GU) but no concrete algorithm recommendation. If ZVF = 100% persists, what should a practitioner do? The paper mentions TPO-style objectives but does not implement or evaluate them."

**7. Non-identifiability claim is obvious and not new**
"Any practitioner knows that GRPO implementation details matter. The paper does not provide a new theoretical framework, a new algorithm, or a new quantitative tool beyond re-stating the obvious in new vocabulary."

**8. Confounded Dense vs MoE comparison**
"The 3B-active pair compares Qwen3.5-4B (dense, instruct) with Qwen3-30B-A3B (MoE, base). Base vs instruct initialization is the dominant effect, not architecture. The comparison is meaningless."

**9. Partial experiments are over-interpreted**
"Qwen3-235B-A22B reaches 100% peak at step 15 (partial). Nemotron-120B reaches 87.5% peak but collapses to 16.2% last-10. These are early-training snapshots from interrupted runs, not converged performance ceilings."

**10. Paper over-promises and under-delivers**
"The abstract promises 'significant cross-library performance variance' and 'model-dependent algorithm preference,' but the evidence is single-seed training reward with no held-out benchmark accuracy. The gap between the abstract's framing and the actual evidence is large."

### Prompt 14: Meta-Reviewer Analysis

**Fatal criticisms (cannot be fixed without new experiments):**
- None are strictly fatal if properly caveated, but #2 (checkpoint selection) is close. If the paper properly disclaims Table 4 as checkpoint-selection analysis, not generalization claim, the criticism dissolves.

**Fixable criticisms:**
- #3 (ZVF tautology): The paper already addresses this in Section 6.4. The fix is to move ZVF from "key contribution" to "triage diagnostic with honest caveats"
- #6 (concrete recommendation): Add a section recommending TPO-style objectives when ZVF = 1 persists, citing Kaddour et al. (2026)
- #9 (partial experiments): Add more explicit "early-training snapshot" framing throughout

**Misunderstandings:**
- #7 (non-identifiability is obvious): This misunderstands the contribution. No prior work has empirically demonstrated non-identifiability with nominally matched configurations. The observation that practitioners "know" this is not the same as having systematic empirical evidence.
- #4 (missing baselines): The paper is not claiming state-of-the-art. It's diagnosing structural properties of GRPO training. DeepSeekMath et al. are not appropriate baselines for a diagnostic study.

### Prompt 15: Accept Case

**Strongest possible accept case:**

This paper makes a unique and important contribution: it demonstrates, empirically and rigorously, that GRPO training dynamics are non-identifiable from hyperparameter labels alone. The evidence is compelling: nominally matched configurations (Qwen3-8B, G=8, lr=10⁻⁵, GSM8K, 30 steps) produce 84.4% last-10 reward on Tinker and 5.0% on TRL—a 17× gap from hidden stack variables. This is not a baseline comparison; it's a structural finding about the nature of GRPO training.

The paper additionally introduces ZVF/GU as actionable triage diagnostics. While the ZVF–reward correlation is partially tautological on binary rewards (correctly acknowledged in Section 6.4), the ZVF ≥ 80% + reward ≤ 5% sustained-over-5-steps rule identifies all collapsed runs with precision 1.0 and recall 1.0. This is practically useful and correctly scoped as triage, not prediction.

The paper is admirably honest about its limitations: closed-source Tinker implementation, short training horizons, train-set reward as primary metric, confounded Dense vs MoE comparison. The "Claims We Explicitly Do Not Make" section is model scientific conduct.

**Edits that would make the accept case definitively true:**

1. Remove all language implying ZVF predicts final performance. Confirm ZVF as regime-identifying triage only.
2. Add a section with concrete algorithm recommendations when ZVF = 1 persists (TPO-style objectives, prompt re-sampling).
3. Acknowledge DeepSeekMath, Qwen-Math, and other strong GRPO implementations as upper-bound references, even if not directly compared.
4. Strengthen the held-out evaluation section with explicit "checkpoint-selection analysis" framing.
5. Implement and evaluate at least one concrete intervention (e.g., group-size reduction to G=2 on a collapsed run).

### Prompt 16: Reject-to-Accept (smallest set of changes)

**Smallest set to move from borderline reject to weak accept:**

1. **Add concrete recommendation:** When ZVF = 1 persists for 5+ steps, practitioners should try TPO-style objectives or prompt re-sampling. Implement one such intervention and show it recovers learning signal.

2. **Move ZVF from "key finding" to "triage diagnostic":** Remove all language implying ZVF predicts final performance. The accept case is strong if ZVF is scoped as "identifies which regime a run is in (cold-start collapse vs. saturation vs. healthy contrast), not as a predictor of final reward."

3. **Acknowledge upper-bound literature:** Add a sentence: "Strong GRPO implementations (DeepSeekMath, Qwen-Math) achieve substantially higher GSM8K improvement; our work studies structural properties of GRPO training, not SOTA performance."

4. **Strengthen Table 4 framing:** Add explicit "checkpoint-selection analysis" label and remove any language that could be read as a held-out generalization claim.

---

## PROMPTS 17-20: Fresh Verification Sessions

### Prompt 17: Fresh Verification — GRPO broadly improves reasoning

**Risky lines to quote and rewrite:**

1. "GRPO achieves near-perfect training reward on frontier models" → "Qwen3-235B-A22B reaches 100% peak reward at step 15 (partial run, early-training snapshot)"

2. "GRPO produces consistent improvements across model scales from 0.6B to 235B" → "GRPO produces training reward trajectories spanning 0.6B to 235B under the GSM8K rollout parser"

3. "Held-out GSM8K accuracy sits in a 87.2–95.0% band" → "Held-out evaluation of the top-10 checkpoints (selected by training reward) yields an 87.2–95.0% accuracy band on the held-out slice"

4. "The capacity ceiling of our 30-step GRPO recipe on GSM8K is model-dependent: 95.0% for Llama-3.3-70B-Instruct" → "In our single-seed 30-step runs, Llama-3.3-70B-Instruct reached 95.0% held-out accuracy on the top-10 checkpoint selection"

### Prompt 18: Fresh Evidence Check — Table captions

**Check all table captions for online/held-out distinction:**

| Table | Current Caption | Issue | Fix |
|-------|-----------------|-------|-----|
| Table 1 | "Training-reward ledger: GRPO and PPO reward under the GSM8K rollout/evaluation parser" | ✓ Correctly labeled | None needed |
| Table 4 | "Held-out GSM8K evaluation of the top-10 Tinker checkpoints" | ✓ Correctly labeled | Add: "Checkpoint selection by training last-10 reward, not random held-out split" |
| Table 5 | "Cross-library comparison on Math RL (Arithmetic)" | ✓ Correctly labeled | Add: "Online training reward, not held-out accuracy" |
| Table 6 | "Secondary GSM8K cross-source summary" | ✓ Correctly labeled | Explicitly note: "Descriptive context, not held-out claim" |
| Table 7 | "Cross-task proxy reward summaries" | ✓ Correctly labeled | Note: "Proxy rewards (schema compliance), not capability" |
| Table 8 | "PPO vs GRPO comparison as stack-conditioned evidence" | ✓ Correctly labeled | Note: "Single-seed training reward, not held-out" |

### Prompt 19: Fresh Statistical Check

**Auditing all p-values, confidence claims, and significance language:**

| Claim | Verification | Status |
|-------|-------------|--------|
| "$r = -0.769$, $p = 0.0008$" for ZVF vs final reward | Correlating ZVF with reward on binary tasks is tautological | Caveated in Section 6.4, but still foregrounded |
| "$r = -0.517$, $p = 0.005$" for PTD vs last-10 | Uncorrected correlation on 28 experiments | OK as exploratory, needs correction caveat |
| "19 of 20 tests survive BH correction" | BH procedure with m=20, FDR=0.05 | Correct |
| MDE $d_{\min} = 2.024$ at 80% power | Power analysis for n=5 per arm | Correct, but this means most comparisons are underpowered |
| "Shapiro-Wilk $W = 0.9018$, $p = 0.4198$" | Normality test for TRL baseline | Correct |
| Cohen's $d = 12.75$ for PPO vs GRPO (Llama) | Single-seed, artifact-sensitive (22.5% vs 35.0%) | Unreliable |
| Spearman $\rho$ varies with exact selection for held-out ranking | Acknowledged | OK |

**Flagged unsupported claims:**
- "Significant cross-library performance variance" — variance is demonstrated, significance is not (no variance estimate for Tinker runs)
- "Precision 1.0, recall 1.0" for ZVF collapse detection — on 22-run validation set, but reward-only rule matches AUC to within noise

### Prompt 20: Fresh Causal Check

**All causal claims and their evidence level:**

| Claim | Evidence Level | Rewrite |
|-------|---------------|---------|
| "GRPO produces training reward gains" | Association | "GRPO produces training reward trajectories" |
| "Stack variables cause divergent outcomes" | Hypothesis (observational) | "Stack variables are associated with divergent outcomes under nominally matched configs" |
| "ZVF causes collapse" | Not established | "ZVF ≥ 80% + reward ≤ 5% for 5+ steps correctly identifies collapsed runs" |
| "Initialization quality modulates architecture effect" | Hypothesis (confounded) | "The instruct variant (Qwen3-30B-A3B-Instruct) matches the dense model's performance; the base variant does not" |
| "GRPO is secretly DPO at G=2" | Theoretical (Wu et al., 2025) | "Following Wu et al. (2025), GRPO at G=2 is algebraically equivalent to DPO" |

---

## PROMPTS 21-23: Experiment/Appendix Prompts

### Prompt 21: Missing Experiment

**What one experiment would most increase credibility with 24 hours:**

**Priority experiment: Multi-seed held-out evaluation of a single model**

Choose Qwen3-8B (well-characterized, moderate cost) and run:
1. GRPO training with 5 seeds (standard seed set: 42, 123, 456, 789, 1024)
2. After training, evaluate ALL 5 checkpoints on held-out GSM8K test split (N=500)
3. Report mean ± SE held-out accuracy

This directly tests whether training reward predicts held-out accuracy. If the 5 checkpoints have training last-10 range 80–100% but held-out range 82–84%, it confirms the checkpoint-selection unreliability hypothesis. If held-out range is wide, it suggests training reward does predict generalization for this model.

**Secondary experiment: G=2 vs G=8 ablation on DeepSeek-V3.1**
Wu et al. (2025) predict 2-GRPO retains 98.1% of 16-GRPO performance at 12.5% rollout cost. Running G=2 and G=8 on DeepSeek-V3.1 (which has the clearest 100%→85% trajectory) would directly test this prediction.

### Prompt 22: No-New-Experiment Version

**Edits that most improve scientific credibility without new experiments:**

1. **Deflate ZVF claims:** Move from "key finding" to "triage diagnostic with honest caveats." Remove all language implying ZVF predicts final performance.

2. **Strengthen Table 4 framing:** Add explicit "checkpoint-selection analysis" label everywhere. Add: "Reading this table as 'GRPO lifts GSM8K to 87–95%' would be the selection-induced overclaim that the abstract explicitly warns against."

3. **Add explicit "do not make" claims:** Expand the existing "Claims We Explicitly Do Not Make" section with the risky lines from Prompt 17.

4. **Remove scaling law framing:** The exponential saturation fits have mean $R^2 = 0.210$. Call them "descriptive summaries" only, not scaling laws.

5. **Fix artifact sensitivity:** The PPO last-10 value differs between ledger (22.5%) and statistics summary (35.0%). Establish a single source of truth and update all figures/tables consistently.

6. **Add concrete recommendation section:** When ZVF = 1 persists for 5+ steps, try TPO-style objectives (Kaddour et al., 2026) or group-size reduction to G=2 (Wu et al., 2025).

### Prompt 23: Appendix Discipline

**What belongs in main paper vs appendix vs removal:**

**Main paper:**
- Stack non-identifiability evidence (Table 1, Figure 4)
- ZVF/GU as triage diagnostics (Section 5)
- Held-out evaluation with checkpoint-selection caveats (Table 4)
- Group size ablation with honest caveats (Figure 6)
- Statistical methodology (Section 3.3)
- Limitations (Section 7)
- "Claims We Explicitly Do Not Make" (Section after limitations)

**Appendix:**
- Full hyperparameter tables (Appendix B)
- Cross-seed reproducibility for TRL-GRPO (Figure 8)
- Sensitivity ablations (Wave 6, Figure 5)
- Exponential saturation fits (Section 5.5)
- Dense vs MoE comparison (confounded)
- Tool-use results (0% reward, negative baseline)
- Compute allocation table (Appendix A)
- Full p-value tables with BH correction

**Remove:**
- Language implying ZVF predicts final performance
- "Scaling law" framing for exponential saturation fits
- PPO last-10 22.5% vs 35.0% artifact inconsistency
- Any claim that GRPO broadly improves reasoning

---

## PROMPTS 24-25: Artifact Review and Bad Ideas Log

### Prompt 24: Artifact Review

**What would make the artifact easier for a reviewer to verify claims:**

**1. Source-of-truth ledger**
The PPO last-10 value differs between ledger (22.5%) and statistics summary (35.0%). Create a single JSON ledger with all final metrics, one canonical source.

**2. Held-out evaluation script**
Release `experiments/modal/modal_heldout_eval.py` with exact commands. Reviewers should be able to rerun held-out accuracy from checkpoints.

**3. ZVF telemetry data**
Release step-level ZVF traces for all 33 runs with telemetry. This allows independent verification of the lagged regression results.

**4. Checkpoint selection protocol documentation**
Document exactly how top-10 checkpoints were selected. The current description ("ranking by training last-10") is clear, but the specific threshold and selection criteria should be explicit.

**5. GRPO config logging**
For each run, log: model, tokenizer, prompt template, reward parser, sampler config, loss mask, KL/reference handling, optimizer, LoRA targets, precision, backend, evaluator. This is the core evidence for stack non-identifiability.

**6. Docker reproducibility**
Ensure all Modal experiments are fully reproducible from the provided Docker container. Test the container on a fresh machine.

**7. W&B step-level data**
The current W&B logs have step-level data loss (only summary values recorded). Either fix and re-upload, or explicitly document that W&B should not be used for convergence speed inference.

### Prompt 25: Bad Ideas Log

**Every tempting but unsafe claim the paper should not make:**

1. **"GRPO improves reasoning"** → Safe version: "GRPO produces training reward trajectories on GSM8K"

2. **"ZVF predicts final performance"** → Safe version: "ZVF identifies which regime a run is in (cold-start collapse vs. saturation vs. healthy contrast)"

3. **"PPO is worse than GRPO"** → Safe version: "PPO and GRPO labels are under-specified unless stack variables are reported"

4. **"Our benchmark achieves competitive performance"** → Safe version: "Our benchmark characterizes structural properties of GRPO training across libraries and scales"

5. **"Scaling laws for GRPO"** → Safe version: "Descriptive exponential saturation fits for training reward trajectories"

6. **"The capacity ceiling of our 30-step recipe is 95% for Llama-3.3-70B"** → Safe version: "In our single-seed 30-step checkpoint-selected runs, Llama-3.3-70B-Instruct reached 95.0% held-out accuracy"

7. **"G=8 is optimal"** → Safe version: "G=8 showed highest last-10 reward in a single-seed 30-step sweep; multi-seed validation needed"

8. **"Dense models outperform MoE at matched active parameters"** → Safe version: "The base vs instruct initialization confounds the architecture comparison"

9. **"Tool-use results show GRPO can learn tool calling"** → Safe version: "Tool-call schema compliance was 0% for completed runs; task design revision needed"

10. **"Significant cross-library performance variance"** → Safe version: "Substantial cross-library performance divergence under nominally matched configurations"

---

## Consolidated "Claims We Do Not Make" Paragraph

> *We do not claim GRPO universally improves reasoning or achieves state-of-the-art GSM8K/MATH performance. We do not claim ZVF predicts final performance; ZVF is a triage diagnostic that identifies training regimes, not a calibrated performance predictor. We do not claim PPO is inferior or superior to GRPO; algorithm labels are under-specified unless backend, sampler, reward parser, LoRA configuration, and evaluator are reported. We do not claim exponential saturation as a validated scaling law; it is a descriptive fit with mean $R^2 = 0.210$ on short traces. We do not claim G=8 is globally optimal for group size; multi-seed validation is needed. We do not claim Dense vs MoE architecture superiority; initialization (base vs instruct) confounds the comparison. We do not claim tool-use learning; the task yielded 0% reward and requires design revision. Held-out evaluation covers only the top-10 checkpoints selected by training reward; checkpoint-selection analysis is not a clean held-out generalization result.*

---

## Summary: Priority Edits

Based on this 25-prompt audit, the following edits have the highest impact-to-effort ratio:

### Immediate (1-2 hours)
1. Update Table 4 caption: "Checkpoint-selection analysis, not random held-out split"
2. Add "Claims We Explicitly Do Not Make" paragraph with the 10 unsafe claims
3. Fix PPO last-10 artifact inconsistency (22.5% vs 35.0%)
4. Remove "scaling law" language; call exponential saturation fits "descriptive summaries"
5. Add concrete recommendation: "When ZVF = 1 persists 5+ steps, try TPO-style objectives or G=2"

### Short-term (1 day)
6. Add Section 4.5: Concrete intervention when ZVF = 1 (e.g., G=2 on DeepSeek-V3.1)
7. Move ZVF from "key finding" to "triage diagnostic with honest caveats" throughout
8. Strengthen stack non-identifiability framing in abstract and intro
9. Add comparison to DeepSeekMath/Qwen-Math as upper-bound references

### Medium-term (1 week)
10. Run multi-seed held-out evaluation on Qwen3-8B (5 seeds, held-out GSM8K)
11. Implement and evaluate G=2 vs G=8 on DeepSeek-V3.1
12. Release ZVF step-level telemetry data for independent verification
13. Create single source-of-truth ledger JSON for all final metrics