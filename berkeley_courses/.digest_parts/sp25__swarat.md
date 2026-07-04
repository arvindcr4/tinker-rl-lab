### Abstraction and Discovery with Large Language Model Agents
*Speaker:* Swarat Chaudhuri

#### Key claims / techniques
- LLM agents for discovery should systematically search spaces of hypotheses, conjectures, and proofs; use prior knowledge to prioritize search; learn from experience; and discover reusable abstract concepts/tools.
- Frontier tool-integrated math LLMs can solve competition-style problems with Python/SymPy, but the neural-only approach faces data scarcity and lack of verifiability beyond high-school/competition settings.
- Formal representations (e.g., Lean/Coq) provide a verifiable alternative path: informal statements are autoformalized, then a neural prover searches for a formal proof checked by the proof assistant.
- Copra is an in-context-learning agent for formal theorem proving that prompts a frontier LLM, parses tactics, executes them in the proof environment, augments the prompt with error/goal feedback, backtracks, and queries a lemma database.
- Copra can integrate informal natural-language reasoning with formal reasoning hierarchically (e.g., ask for an informal solution, split into subgoals, then solve each subgoal formally), and can repair informal LLM hallucinations using proof-assistant feedback.
- AlphaProof uses reinforcement learning for theorem proving, learning from both successful proofs and failed disproofs, including misformalized problems, and complements training-time RL with test-time RL on problem variants.
- LaSR extends symbolic regression with a learned concept library: an LLM-driven evolutionary loop performs mutation, crossover, initialization, and abstraction over a population of programmatic hypotheses, producing symbolic abstractions such as "power-law trend" or "exponential growth/decay."
- LaSR outperforms PySR even when using local language models and limited LLM budget, and user-provided or LLM-generated concepts can accelerate hypothesis search, though concepts are not guaranteed to be factual or correct.
- LLM scaling laws can be discovered from benchmark data by treating law discovery as symbolic regression; the recovered laws reveal interactions between training hyperparameters and test-time shots.
- A self-evolving visual concept library uses VLMs to score textual concept descriptors contrastively against images, then refines the descriptors through evolution to enable zero-shot visual classification.

#### Relevance hooks
- Agent evaluation methodology: Copra, LaSR, and the visual-concept library are concrete agent systems evaluated against proof assistants, symbolic-regression benchmarks, and zero-shot classification tasks, respectively, using external, task-specific feedback rather than human judgment alone.
- RL post-training benchmarking: AlphaProof is presented as a reinforcement-learning system trained on verifiable proof/disproof signals and test-time search, making formal theorem proving a relevant domain for RL post-training and reasoning benchmarks.
- RL reproducibility standards: The emphasis on formal proof assistants provides deterministic, checkable correctness criteria, which is a model for reproducible evaluation of RL-trained reasoning agents.

#### Cited paper titles (verbatim only)
- How NuminaMath Won the 1st AIMO Progress Prize
- Formal Mathematical Reasoning: A New Frontier in AI.
- An In-Context Learning Agent for Formal Theorem-Proving
- LaSR: Symbolic Regression with a Learned Concept Library
- Training Compute-Optimal Large Language Models
- Self-Evolving Visual Concept Library Using Vision-Language Critics
- Alphageometry
- Proofwala
- Process-Driven Autoformalization
- Minimo

`Index row: sp25 | swarat.pdf | Swarat Chaudhuri | LLM agents can combine formal proof-assistant feedback and LLM-guided evolutionary search to automate mathematical reasoning and empirical discovery. | ok`
