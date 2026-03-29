# Paper Improvement Plan

## Rationale
This plan re-centers the paper around claims the repository can support now, then stages the missing experiments needed to make stronger, standardized, reviewer-proof claims without overfitting to any single benchmark.

## 1. Re-scope the paper immediately

### What to change now
- Treat the current draft as a **training-dynamics and methodology paper**, not yet a generalization paper.
- Remove or downgrade any claim that depends on:
  - held-out GSM8K generalization that has not been measured,
  - standardized tool-calling evaluation that has not been run,
  - full HumanEval/MBPP benchmark coverage that has not been completed.
- Use precise labels everywhere:
  - **training-set reward**,
  - **training-prompt accuracy under fresh rollouts**,
  - **preliminary/custom internal evaluation**,
  - **planned standardized evaluation**.

### Revised core claim
> Under a lightweight PEFT setup, GRPO reliably optimizes verifiable training rewards on small-model math tasks above a capability threshold, but stronger claims about cross-benchmark generalization and agentic tool use require standardized held-out evaluation.

## 2. Highest-priority experiments before submission

### Priority A — math generalization
This is the single most important gap.

#### Required actions
1. Recover or regenerate the trained LoRA adapters/checkpoints used in the GSM8K and MATH runs.
2. Run **held-out GSM8K test evaluation** on the full 1,319-example test set.
3. Run **held-out MATH evaluation** with the same decoding protocol.
4. Report:
   - pass@1 / exact-match accuracy,
   - 95% bootstrap confidence intervals,
   - seed count,
   - decoding settings,
   - checkpoint identity,
   - evaluation script version.
5. Add a direct **train-vs-test table** for every reported model.

#### Minimum acceptable paper outcome
- If full held-out math is completed: keep the broader math section.
- If not: explicitly narrow the paper to **training optimization dynamics** and move generalization claims to future work.

### Priority B — tool-calling standardization
The custom tool metrics need external grounding.

#### Required actions
1. Separate the current tool results into:
   - **internal synthetic single-turn results**,
   - **internal multi-turn scenario results**.
2. For every current tool result, publish:
   - tool schemas,
   - prompt templates,
   - argument distributions,
   - reward/scoring rubric,
   - sample prompts and failures.
3. Add at least one standardized evaluation path:
   - **FC-RewardBench / ToolRM-style outcome evaluation** for single-turn function calls,
   - **proxy-state / final-state judged multi-turn evaluation** for agentic workflows.
4. Report judge reliability:
   - inter-rater agreement if human judging is used,
   - judge-model details and agreement checks if LLM judging is used.
5. Replace unsupported phrasing like “reliable tool caller” with narrower language until standard benchmarks are run.

### Priority C — code generation standardization
The 50-problem HumanEval subset is not enough for a benchmark claim.

#### Required actions
1. Run the **canonical full HumanEval harness**.
2. Run **MBPP** with the standard evaluation harness.
3. Report:
   - full-set pass@1,
   - pass@k when appropriate,
   - 95% confidence intervals,
   - exact decoding parameters,
   - seed averaging,
   - number of sampled completions per problem.
4. Keep the 50-problem subset only as a pilot result in appendix/supplement.

## 3. Critical ablations to support the central scientific claims

## A. Capacity vs exploration vs reward sparsity
The current “3B fails, 8B succeeds” story is promising but under-identified.

### Required ablations
- **Group size:** G in {16, 32, 64, 128}
- **Temperature:** T in {0.7, 1.0, 1.3, 1.5}
- **KL regularization:** beta in {0.01, 0.05, 0.1}
- **Entropy regularization:** small sweep if KL alone is insufficient
- **Curriculum:** easy-to-hard or short-to-long GSM8K subsets
- **Reward densification:** partial credit / step-aware shaping where verifier design allows it

### Main question to answer
Is the below-threshold failure truly a capacity issue, or mostly a failure to sample at least one rewarding trajectory per GRPO group?

### Metrics to add
- zero-reward step rate,
- zero-advantage / zero-loss step rate,
- fraction of all-bad / mixed / all-good groups,
- reward variance within group,
- entropy trajectory,
- KL-to-SFT trajectory.

## B. MoE volatility
The MoE finding is interesting but currently descriptive.

### Required diagnostics
- router entropy,
- per-expert load balance,
- expert utilization coefficient of variation,
- router shift ratio,
- clip fraction,
- gradient norm statistics,
- reward and loss variance over time.

### Required comparison
Compare dense vs MoE under matched decoding, reward, group size, and update budget.

## 4. Baselines the paper should add or position explicitly

## Online RL baselines
- **RLOO**
- **REINFORCE++**
- **PPO** if infrastructure cost is manageable
- **step-wise GRPO variants** such as S-GRPO / StepGRPO when available

## Offline / preference baselines
- **SFT-only**
- **DPO**
- **Step-DPO** if step labels/preferences are available

## Why these matter
These baselines answer whether GRPO is uniquely effective here, or just one workable option among several lightweight post-training methods.

## 5. PEFT and efficiency claims
The current LoRA/QLoRA story is plausible but underspecified.

### Required actions
1. Keep **QLoRA + LoRA rank 32** as the main practical baseline.
2. Add bounded PEFT ablations:
   - rank {8, 16, 32, 64},
   - adapter placement if easy to vary.
3. Add a small contextual comparison section for:
   - **QR-Adaptor**,
   - **LoTA-QAF**,
   - **QR-LoRA**.
4. Report efficiency with actual numbers:
   - trainable parameter count,
   - GPU type/count,
   - wall-clock time,
   - GPU-hours,
   - peak memory,
   - tokens processed.

### Revised efficiency claim
> GRPO + QLoRA appears practically lightweight for small-model verifiable RL, but efficiency should be framed relative to matched baselines and explicit compute accounting.

## 6. Reporting changes that should happen even before new experiments

### Add one methods table with
- dataset names,
- train/validation/test splits,
- effective training set sizes,
- prompt templates,
- output format requirements,
- decoding settings,
- group size,
- learning rate,
- clip range,
- LoRA rank,
- number of update steps,
- compute budget.

### Add one evaluation table with
- task,
- metric,
- split,
- judge/verifier type,
- deterministic vs sampled decoding,
- number of examples,
- confidence interval method.

### Add one failure taxonomy per domain
- wrong function name,
- malformed JSON,
- wrong argument type,
- wrong argument value,
- unnecessary repeated tool call,
- correct format but incorrect math answer,
- parse failure,
- timeout / no answer.

### Add one limitations box
Explicitly list:
- training-set-only math results where applicable,
- non-standard code evaluation where applicable,
- custom tool benchmarks,
- missing inter-rater reliability,
- missing released checkpoints/prompts where applicable.

## 7. Suggested paper restructuring

## Title / positioning
Avoid a title that implies standardized cross-domain superiority unless those experiments are complete.

## Recommended structure
1. **Introduction**
   - practical question: when does lightweight GRPO work for small models on verifiable tasks?
2. **Method**
   - GRPO setup, PEFT recipe, verifiers, and telemetry
3. **Experimental Protocol**
   - datasets, splits, decoding, budgets, evaluation rules
4. **Math Results**
   - training dynamics first, held-out accuracy second
5. **Mechanistic Analysis**
   - capacity threshold, reward sparsity, zero-advantage groups, entropy/KL
6. **MoE Stability Analysis**
   - volatility plus routing telemetry
7. **Tool/Code Results**
   - either standardized results, or clearly marked preliminary appendix material
8. **Limitations and Reproducibility**
9. **Conclusion**

## 8. Submission strategy by timeline

## If you have 3–5 days
- Fix all wording and labeling.
- Add methods/evaluation tables.
- Remove unsupported tool/code headline claims from abstract and intro.
- Add a reviewer-facing “what is preliminary vs standardized” subsection.
- Run at least one full held-out GSM8K evaluation if checkpoints are available.

## If you have 1–2 weeks
- Complete full held-out GSM8K.
- Add matched SFT baseline.
- Add 3B rescue ablations: group size + temperature + curriculum.
- Add KL/entropy trajectories and zero-advantage/group-composition plots.

## If you have 3–6 weeks
- Complete standardized tool benchmark.
- Complete full HumanEval/MBPP harness.
- Add RLOO/REINFORCE++ and DPO/Step-DPO comparisons.
- Add MoE routing diagnostics.
- Prepare reproducibility release package.

## 9. Concrete writing changes for each reviewer concern

| Reviewer concern | Paper action |
|---|---|
| GSM8K is training-set only | Relabel all math plots/tables; add held-out eval section and train-vs-test comparison |
| Tool scores are custom | Add rubric, judge details, schema release, and standardized benchmark plan/results |
| HumanEval uses only 50 problems | Move subset result to pilot appendix; run full harness for main text |
| 0% JSON SFT validity is suspicious | Document SFT data, prompt format, decoding, parser strictness, and baseline reproduction |
| 3B vs 8B may reflect exploration | Add group-size, temperature, curriculum, and reward-density ablations |
| No KL anchor / drift telemetry | Log and plot KL-to-SFT, entropy, zero-advantage, and reward variance |
| MoE volatility is anecdotal | Add router entropy/load metrics and matched dense-vs-MoE comparison |
| Efficiency claims are under-positioned | Add compute table and matched PEFT / baseline discussion |
| Reproducibility is incomplete | Release prompts, configs, eval scripts, and checkpoint access instructions |

## 10. Final recommended claim boundary

### Safe claim now
- GRPO is effective at optimizing **verifiable training rewards** in small-model math settings above a capability threshold, with informative failure modes tied to reward sparsity and exploration.

### Claim only after new experiments
- GRPO improves **held-out mathematical generalization**.
- GRPO yields **standardized tool-calling gains**.
- GRPO improves **canonical code-generation benchmark performance**.
- GRPO is **more efficient than competing critic-free or preference-based methods**.

## 11. Bottom line

If only one thing gets done before submission, do the full held-out GSM8K evaluation and rewrite the abstract/introduction/results so every metric is honestly labeled. If three things get done, add standardized code evaluation and at least one standardized tool-calling protocol. If the team has enough time for a stronger paper, the most valuable scientific package is: held-out math + 3B rescue ablations + MoE telemetry + matched baselines.
