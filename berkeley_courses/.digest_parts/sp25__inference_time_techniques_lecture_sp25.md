### Inference-Time Techniques for LLM Reasoning
*Speaker:* Xinyun Chen

#### Key claims / techniques
- Chain-of-thought (CoT) prompting increases reasoning performance by eliciting step-by-step intermediate reasoning; CoT gains scale strongly with model size and enable strategies such as decomposition and planning.
- Zero-shot CoT instructions like “Let’s think step by step” can trigger CoT generation without labeled exemplars, but still lag behind few-shot CoT.
- Analogical prompting instructs the model to self-generate relevant exemplars and high-level knowledge for each problem, improving over both zero-shot CoT and manually designed few-shot CoT.
- LLM-driven prompt engineering—generating candidate prompts, scoring them on a small validation set, and iteratively optimizing instructions—can match or exceed few-shot CoT accuracy.
- Task-specific decomposition methods (least-to-most prompting, Self-Discover) improve compositional generalization by breaking complex problems into subproblems and composing reasoning structures.
- Self-consistency improves over single-sample decoding by sampling multiple reasoning paths and selecting the most common final answer; performance scales with sample count and benefits from diverse sampling.
- Consistency-based selection extends beyond math reasoning to code generation (AlphaCode execution-based clustering) and open-ended generation (Universal Self-Consistency), though the latter depends on long-context capability.
- Trained LLM verifiers can outperform consistency-based selection; process-supervised reward models (PRM) scale better with more samples than outcome-supervised reward models (ORM), but verifier quality and task transfer are limiting factors.
- Tree-of-thoughts and related search methods integrate step-level state evaluation with tree search (BFS, MCTS) to explore partial solution spaces and scale inference compute more effectively than single-path CoT.
- Iterative self-improvement and reflection can help when reliable external feedback is available (e.g., code execution, task-specific heuristics), but without oracle feedback LLMs often fail to self-correct and can degrade performance; multi-agent debate does not outperform self-consistency without a good evaluator.
- The optimal allocation of inference-time compute—parallel vs. sequential sampling, model size vs. number of samples—is model- and task-specific and empirically determined.

#### Relevance hooks
- Closely related to RL post-training benchmarking: the lecture explicitly contrasts outcome-supervised vs. process-supervised reward models and discusses verifier quality and scaling, which are central to RL/GRPO-style post-training evaluation.
- Relevant to evals-with-error-bars: self-consistency, verifier-based selection, and search results depend heavily on sample budget, sampling diversity, and selection method, so reported benchmark numbers should be interpreted with those design choices in mind.
- Relevant to agent evaluation methodology: tree-of-thoughts, self-reflection, self-debugging, and multi-agent debate are core inference-time agent workflows; the lecture cautions that their benefits require reliable external evaluation or strong self-evaluation.

#### Cited paper titles (verbatim only)
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Show Your Work: Scratchpads for Intermediate Computation with Language Models
- Emergent Abilities of Large Language Models
- Large Language Models are Zero-Shot Reasoners
- Large Language Models as Analogical Reasoners
- How to solve it
- Large Language Models are Human-Level Prompt Engineers
- Large Language Models as Optimizers
- Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
- Compositional Semantic Parsing with Large Language Models
- Measuring Compositional Generalization: A Comprehensive Method on Realistic Data
- SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures
- Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Competition-level Code Generation with AlphaCode
- Universal Self-Consistency for Large Language Model Generation
- Training Verifiers to Solve Math Word Problems
- Let’s Verify Step by Step
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- Reflexion: Language Agents with Verbal Reinforcement Learning
- Self-Refine: Iterative Refinement with Self-Feedback
- Teaching Large Language Models to Self-Debug
- Language Models can Solve Computer Tasks
- Large Language Models Cannot Self-Correct Reasoning Yet
- Improving Factuality and Reasoning in Language Models through Multiagent Debate
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving

Index row: sp25 | inference_time_techniques_lecture_sp25.pdf | Xinyun Chen | Inference-time reasoning improves through CoT prompting, multi-sample consistency, and search/self-reflection when grounded by reliable evaluation. | ok
