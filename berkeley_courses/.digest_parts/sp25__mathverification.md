### Formal Reasoning Meets LLMs: Towards AI for Mathematics and Verification
*Speaker:* Kaiyu Yang

#### Key claims / techniques
- Math and coding serve as proxies for complex reasoning and planning and are relatively easy to evaluate: math answers can be checked and code can be run through unit tests.
- State-of-the-art math LLMs are built from strong pretrained models plus supervised finetuning (SFT), reinforcement learning (RL), and substantial engineering.
- SFT on mathematical data uses problems paired with human- and LLM-curated step-by-step or tool-integrated solutions (e.g., using `sympy`); the largest public datasets contain roughly 900K examples.
- RL on verifiable math problems compares final answers against ground truth and optimizes rewards; GRPO is highlighted as the RL algorithm behind DeepSeek-R1.
- RL’s verifiability requirement limits it to problems with numeric or otherwise checkable answers and does not directly apply to open-ended proofs.
- Two major gaps are identified: moving from pre-college math (AIME/IMO) to advanced mathematical research, and moving from answer guessing to valid proof generation.
- Formal mathematical reasoning grounds reasoning in formal systems (first/higher-order logic, dependent type theory, programs with formal specifications) so proof assistants can verify proofs automatically and provide feedback, mitigating data scarcity.
- Proof assistants such as Lean allow theorems and proofs to be represented formally; AlphaProof is cited as an example of large-scale search and RL using Lean feedback.
- LLMs can be trained for theorem proving to generate either next tactics or complete proofs, with individual tactics assembled into full proofs via search algorithms.
- LeanDojo provides open-source training data, model checkpoints, and interaction tools for Lean, containing 98,641 theorems, 217,639 tactics, and 129,162 premises.
- ReProver (Retrieval-Augmented Prover) retrieves accessible premises, concatenates them with the current proof state, and uses the combined context for tactic generation.
- LIPS (LLM-based Inequality Prover with Symbolic Reasoning) synergizes neural and symbolic reasoning for Olympiad inequalities, categorizing proof steps as scaling (applying lemmas such as Cauchy-Schwarz or AM-GM) or rewriting (equivalent transformations), and prunes scaling tactics symbolically.
- LIPS solves 16/20 Olympiad-level inequality problems, exceeding IMO gold medalists (15/20) and DeepSeek-R1 (4/20), and discovers novel proof paths using AM-GM that human experts considered hopeless.
- Autoformalization faces two core challenges: evaluating whether an informal theorem was formalized correctly is hard (equivalence checking is infeasible, human evaluation is expensive, proxy metrics are inaccurate), and informal proofs contain explicit and implicit reasoning gaps that formal proofs must fill.
- LeanEuclid is introduced as a benchmark for autoformalizing Euclidean geometry, with 48 problems from Euclid’s Elements and 125 from UniGeo, including the first faithful formalization of Euclid’s proofs; diagrammatic reasoning gaps are addressed via a formal system E implemented in Lean and automated with SMT solvers.

#### Relevance hooks
- GRPO/RL post-training benchmarking: the lecture explicitly describes GRPO as the RL algorithm used to optimize verifiable math rewards, popularized by DeepSeek-R1, and contrasts this with the need for formal proof verification when numeric rewards are unavailable.
- Evals-with-error-bars / agent evaluation methodology: formal proof assistants provide automatic, rigorous verification of reasoning steps, offering a pathway for more trustworthy evaluation of LLM reasoning than answer-key matching alone.
- RL reproducibility standards: LeanDojo is presented as an open-source resource releasing data, model checkpoints, and interaction tools to support reproducible research on neural theorem proving.

#### Cited paper titles (verbatim only)
- "Can AI do maths yet? Thoughts from a mathematician"
- "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad"
- "Formal Mathematical Reasoning: A New Frontier in AI"
- "AI achieves silver-medal standard solving International Mathematical Olympiad problems"
- "TacticToe: Learning to Prove with Tactics"
- "GamePad: A Learning Environment for Theorem Proving"
- "Learning to Prove Theorems via Interacting with Proof Assistants"
- "HOList: An Environment for Machine Learning of Higher-Order Theorem Proving"
- "Generative Language Modeling for Automated Theorem Proving"
- "HyperTree Proof Search for Neural Theorem Proving"
- "Proof Artifact Co-training for Theorem Proving with Language Models"
- "LeanDojo: Theorem Proving in Lean using Language Models"
- "Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning"
- "Autoformalization with Large Language Models"
- "A formal system for Euclid's Elements"

Index row: sp25 | mathverification.pdf | Kaiyu Yang | Survey of SFT/RL (GRPO/DeepSeek-R1) math training and the shift to formal reasoning in Lean for proofs, with case studies in LeanDojo/ReProver, LIPS inequalities, and LeanEuclid geometry autoformalization | ok
