# ChatGPT Pro (Extended Pro) Review Feedback
**Date:** 2026-04-04
**Model:** ChatGPT Pro with Extended Pro deep thinking
**Context:** Paper improvement advice for tonight's submission

---

## 1. PRIORITIZATION OF WEAKNESSES

### 1.1 GSM8K results are training-set reward, NOT held-out test accuracy (Fatal)

- **Why it's critical**: Peer reviewers at top venues value generalization performance over training-set rewards. If your evaluation is overfitting or overly optimistic (training-set reward is higher than test accuracy), it would likely be flagged as a major issue. Given that GSM8K is a well-known benchmark, reviewers will expect the evaluation to be generalizable, i.e., held-out test performance.
- **Severity**: **Fatal**. Without held-out test accuracy, your conclusions regarding the generalization ability of your method are compromised.

### 1.2 No comparison baselines against PPO, REINFORCE, or SFT-only on same models/data (Fatal)

- **Why it's critical**: NeurIPS and ICML reviewers are extremely sensitive to comparative baselines. If you claim that GRPO-based optimization is superior, but you lack comparisons to well-established methods (like PPO or REINFORCE), your claims will be questioned. This gap diminishes the strength of your contribution.
- **Severity**: **Fatal**. In the absence of baselines, reviewers may question the validity of your method's effectiveness and may view the paper as incomplete.

### 1.3 Tool-calling scores from a custom internal setup, not standardized benchmark (Acceptable with caveats)

- **Why it's critical**: The lack of a standardized benchmark may be problematic, but it's not necessarily a fatal flaw. Custom setups are common in exploratory work, but they reduce the paper's external validity. Top venues would prefer benchmark comparisons to verify your findings, but as long as your experimental setup is sufficiently detailed and reproducible, it may still pass with caveats.
- **Severity**: **Moderate**. This could be flagged, but it's less critical than the other two issues.

### 1.4 HumanEval based on a 50-problem subset (Acceptable with caveats)

- **Why it's critical**: Using a subset of HumanEval reduces the completeness of your results, but 50 problems are still a non-negligible sample size, and many studies use subsets in preliminary experiments. Reviewers may ask for more comprehensive results, but this will not be flagged as fatal unless your claim hinges entirely on this benchmark.
- **Severity**: **Moderate**. This would be an issue if the subset is central to your claims, but it's likely to pass if framed appropriately as a preliminary experiment.

### 1.5 Paper exceeds the NeurIPS 9-page limit (Fatal)

- **Why it's critical**: This is a hard limit at NeurIPS, and exceeding it would almost certainly get your paper desk-rejected. ICML and EMNLP are less strict, but it's still an issue.
- **Severity**: **Fatal**. Immediate action is required to cut down to 9 pages.

---

## 2. STRATEGIC IMPROVEMENTS

### 2.1 Mitigation Strategy for GSM8K Results

- **Action**: Run the evaluation on the held-out test set for GSM8K as soon as possible. If the results aren't ready before submission, explicitly state in the paper that the results are based on training-set reward and that held-out accuracy is pending, with an expected update after submission. This transparency is key. You can still include preliminary results if you frame them as such, but reviewers need to understand that the full evaluation is not yet complete.
- **Impact**: This will allow you to address the most critical issue without additional GPU resources.

### 2.2 Mitigation Strategy for Baseline Comparisons

- **Action**: Use publicly available implementations for PPO and REINFORCE with pre-trained models that match your setup (e.g., Qwen3-8B). You don't need to train them from scratch but can show that your method outperforms existing methods in terms of both tool-calling and code generation tasks. Focus on a single baseline comparison for each domain (i.e., one PPO comparison for tool-calling, one REINFORCE for code generation). This can be done using Tinker API for inference (with appropriate baselines available on platforms like Hugging Face).
- **Impact**: This would give your paper much-needed rigor and make your comparisons much more defensible.

### 2.3 Mitigation Strategy for Tool-Calling Setup

- **Action**: Even without a standardized benchmark, make sure your custom tool-calling setup is well-documented and reproducible. Consider including a supplementary material or appendix with the details of your internal setup and any relevant datasets.
- **Impact**: This would raise your paper's credibility even without the use of a formal benchmark.

### 2.4 Mitigation Strategy for HumanEval

- **Action**: Acknowledge the limitation of using a 50-problem subset and propose future work that includes the full set. Mention that you are currently working on evaluating the full set but chose a subset due to resource constraints.
- **Impact**: This shows that you're aware of the limitation but are still making meaningful contributions.

### 2.5 Mitigation Strategy for Paper Length

- **Action**: Condense the background, literature review, and methodology sections by removing redundancy. Focus more on experimental results and key findings. Move in-depth technical details to the supplementary material. If necessary, shorten or combine findings that are less novel.
- **Impact**: Bringing the paper within NeurIPS's 9-page limit is a hard requirement.

---

## 3. FRAMING ANALYSIS

**Current Framing**: "Empirical Characterization Study"

- **Pros**: This framing is generally safe and aligns with the exploratory nature of the work. However, it may not carry the same weight as a more solution-oriented or reproducibility-based framing.
- **Cons**: Given the gaps in your evaluation (e.g., no full evaluation, no PPO or REINFORCE comparisons), this framing could come off as lacking robustness. Reviewers might expect a higher level of generalization.

**Reframing**: "Reproducibility/Scaling Study"

- **Pros**: This could help frame your work as a broad attempt to understand the scalability and reproducibility of fine-tuning techniques across different agentic tasks. The experimental results would be positioned as contributions toward understanding how generalizable the technique is.
- **Cons**: This reframing reduces the novelty of your claims (as the focus would shift more toward empirical validation rather than new methods), but it might be safer for surviving peer review.

**Recommendation**: Reframe slightly toward "Scaling and Reproducibility" while still emphasizing empirical insights. Acknowledge that your results are preliminary and are focused on understanding generalizable principles in fine-tuning LLMs with GRPO-based techniques.

---

## 4. NOVELTY ASSESSMENT

**Genuinely Novel**: Findings 1, 2, 4, and 7
- These findings contribute to the understanding of GRPO and agentic LLM fine-tuning, particularly within tool-calling, code generation, and math reasoning.

**Empirical Confirmations (not novel)**: Findings 3, 5, and 6
- Finding 3 ("two-phase learning progression") could be new if the progression is distinctive, but two-phase training is not novel in deep learning literature.
- Finding 5 ("synthetic vs real tool schema gap") is interesting, but synthetic-to-real gaps are well-documented in RL and tool use.
- Finding 6 ("LoRA rank scaling speed, not ceiling") might be a confirmation rather than a groundbreaking discovery. However, it can be reframed as an interesting empirical result.

**Action**: Reframe Findings 3, 5, and 6 as empirical confirmations of known patterns in the context of GRPO and agentic LLM fine-tuning, which adds value by validating them in the specific setup you used.

---

## 5. NON-OBVIOUS IMPROVEMENTS

### Structural and Presentation
- Break paper into distinct sections with clear hypothesis, setup, and evaluation metrics per experiment
- Use tables or graphs to summarize experimental setups and results
- Provide full details in supplementary material

### Related Work
- Include more related work on GRPO, REINFORCE, PPO, LoRA, and other fine-tuning methods
- Discuss how findings align or diverge from previous studies
- If claiming novelty, differentiate clearly

### Alternative Interpretations
- Consider alternative interpretations: Do the two-phase learning findings imply a specific architectural choice or just the nature of the task?
- Could the synthetic-to-real gap be attributed to dataset size or model architecture rather than schema complexity?

### Summary
With these improvements, particularly addressing the fatal weaknesses, you can still submit a competitive paper. Key focus: address generalization issue (GSM8K), add baseline comparisons, and frame the paper to mitigate evaluation gaps.
