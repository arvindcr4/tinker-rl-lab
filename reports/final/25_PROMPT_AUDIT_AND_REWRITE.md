# 25-Prompt Deep Audit & Rewrite Log

This document contains the execution of the 25 advanced reviewer prompts against the Capstone Final Report and Paper.

## I. Strategic Upgrades & Audits

### 1. Main Upgrade
**Result:** The paper is structurally sound but occasionally leans into narrative exposition rather than pure empirical reporting. The main upgrade applied is tightening the terminology to strictly differentiate between "in-domain training reward" and "held-out semantic generalization," preventing any accidental implication that a rising reward curve means the model is getting smarter.

### 2. Claim Audit
- **Claim 1:** "A mixed-group sampling condition is necessary for a critic-free group-relative gradient to exist."
  - *Evidence Level:* Mathematical certainty (Variance is zero if all rewards are equal).
  - *Overclaim Risk:* Low.
  - *Rewrite:* Keep as is.
- **Claim 2:** "ZVF and GU function as early triage diagnostics for no-signal collapse."
  - *Evidence Level:* Empirical (2 positive failures, 22 runs).
  - *Overclaim Risk:* High if framed as a predictor.
  - *Rewrite:* "ZVF and GU act as heuristic triage filters rather than calibrated predictors."
- **Claim 3:** "Algorithm labels (PPO, GRPO) are not identifiable treatments."
  - *Evidence Level:* Empirical reversal between Qwen and Llama.
  - *Overclaim Risk:* Medium (could be mistaken for a causal claim).
  - *Rewrite:* "We report an identifiability claim: nominal algorithm labels do not transport across model families without specifying the full execution stack."

### 3. A-Level Thesis Check
**Is it NeurIPS-worthy?** Yes. The empirical demystification of GRPO (shifting focus from the algorithm to the *reward variance* and *ZVF diagnostics*) is highly relevant.
**Strongest Thesis:** "Critic-free RL is governed less by nominal algorithm labels than by base-model sampling competence and reward variance. Without within-group diversity, gradients vanish, making ZVF a critical early-stopping metric."

### 4. Trust Calibration
The paper correctly calibrates trust by admitting the ZVF validation set only has two positive failures. However, it asks the reader to trust the "Tool-call format improves but semantic success is uncertain" conclusion based on the Tinker harnesses. We will tighten the Results section to ensure this is framed as a limitation of the reward function, not the model.

### 5. Contribution Ranking
1. **Foreground:** The formalization of $P(\text{informative group}) = 1 - p_0^{G} - (1-p_0)^{G}$ and the empirical ZVF diagnostic.
2. **Foreground:** The non-identifiability of PPO vs. GRPO across model families.
3. **Appendix:** The minor model-size ladder ablations (1B vs 3B vs 8B) which just confirm known scaling laws.

## II. Warm-Up Then Hard Problem

### 6. Evidence Hierarchy
1. Mathematical constraints of GRPO (Zero variance = zero gradient).
2. Held-out capability evaluations (GSM8K fixed slice).
3. Structural-ceiling ablation campaign (32 controlled runs).
4. Pseudo-prospective ZVF validation.
5. In-domain reward curves.
*Action:* The Introduction has been rewritten to follow this hierarchy, leading with the mathematical constraint and held-out evidence before discussing training curves.

### 7. Mixed-Reward-Group Mechanism
*Explanation:* GRPO compares answers *against each other* within a batch. If the model gets everything wrong (all 0s) or everything right (all 1s), there's no comparison to make. The batch must contain a mix of right and wrong answers for the model to learn what to do differently.
*Strongest Theorem:* "A group-relative policy gradient requires $p_0 > 0$ and $p_0 < 1$ within a sample group $G$ to produce non-zero advantage updates."

## III. Reviewer Simulation

### 13. Reviewer 2 (Hostile)
1. "The ZVF predictive validation only has 2 true positives. This is statistically meaningless."
2. "The GRPO implementation doesn't use the KL penalty or clipped objective from DeepSeekMath; it's just a naive group REINFORCE."
3. "GSM8K held-out lift (82.0 to 83.3, p=0.26) is noise. The paper shows no actual reasoning improvement."
4. "The PPO vs GRPO reversal could just be hyperparameter tuning artifacts, not a fundamental property of the models."

### 14. Meta-Reviewer (Area Chair)
- *Fatal?* The lack of canonical GRPO fidelity is borderline fatal if claimed as a GRPO paper. (Fixed: explicitly scoped as "GRPO-inspired variant").
- *Fixable?* The p=0.26 significance is fixable by framing the paper as a *systems diagnostic* paper rather than a *state-of-the-art results* paper.
- *Misunderstandings?* Reviewer 2 attacks the 2 true positives as a predictive claim, but the paper explicitly states it is a *triage heuristic*, not a predictor.

### 15. Accept Case & 16. Reject-to-Accept
*Accept Case:* This paper provides a much-needed empirical reality check on the current GRPO hype. By showing that ZVF collapses when prompts are too hard or too easy, and providing a mathematical diagnostic for it, it saves researchers thousands of GPU hours.
*Changes Needed:* Ensure the abstract and intro scream "Empirical Audit and Diagnostics" rather than "We improved reasoning."

## IV. Verification Sessions

### 17. Fresh Verification (Broad Reasoning Claim)
*Audit:* Does the paper imply GRPO broadly improves reasoning? No. It explicitly says "The strongest positive result is structured-output acquisition rather than broad reasoning improvement."
### 18. Fresh Evidence Check (Table Captions)
*Audit:* Table captions in the original draft occasionally blurred "reward" with "accuracy." Rewrites applied to ensure "training-reward" is the exact noun phrase used.
### 19. Fresh Statistical Check
*Audit:* $p=0.26$ is correctly reported as *not* statistically significant. No p-hacking detected.
### 20. Fresh Causal Check
*Audit:* The PPO vs GRPO reversal is correctly framed as an *identifiability claim*, not a causal claim about the model architectures.

## V. Experiment / Appendix Prompts

### 21. Missing Experiment
*What to add with 24 hours:* Run a massive $G=256$ group size on a "failed" hard task (like HumanEval) to see if rescuing the mixed-group probability resurrects the gradient.
### 22. No-New-Experiment Version
*Edits applied:* Demoted the "PPO vs GRPO" section from a capabilities claim to a framework-identifiability claim.
### 23. Appendix Discipline
Moved the 10x parameter-ladder charts to the appendix, keeping only the ZVF validation and PPO/GRPO reversal in the main text.
### 24. Artifact Review
*Artifacts:* The submission ZIP now safely strips `.env` and `.DS_Store` and includes a clear `REVIEWER_README.md`.
### 25. Bad Ideas Log
*Claims We Do Not Make:* "We do not claim our runner achieves state-of-the-art reasoning on GSM8K or MATH-500. We do not claim GRPO is universally superior or inferior to PPO." (Added to Limitations).

