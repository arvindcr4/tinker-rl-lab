### Compound AI Systems & Natural Language Programming with DSPy
*Speaker:* Omar Khattab

#### Key claims / techniques
- Compound AI Systems are modular programs that use language models as specialized components, yielding gains in quality, control, transparency, efficiency, and inference-time scaling (e.g., retrieval-augmented generation, multi-hop RAG, compositional report generation).
- Current compound systems are often “stringly-typed,” coupling architecture with incidental prompt text; DSPy proposes instead to program them with fuzzy natural-language-typed modules that learn their behavior.
- A DSPy program separates concerns into Modules, Signatures, Adapters, Predictors, Metrics/Assertions, and Optimizers.
- Signatures specify the input/output mapping (e.g., `context, question -> query`) rather than the prompt text; modules such as `dspy.ChainOfThought` provide the inference-time strategy.
- Example multi-hop RAG program (`MultiHop`) iteratively generates search queries, retrieves passages, and generates an answer, all expressed as a Python class with DSPy modules.
- Adapters translate signatures into basic prompts; Optimizers then tune them, e.g., raising a HotPotQA multi-hop QA score from 33% to 55% with GPT-3.5, 50% with Llama2-13B, and 39% with T5-770M after `BootstrapFinetune` from 200 answers.
- `BootstrapFewShot` uses rejection sampling to build task demonstrations and can search over demonstration sets (`BootstrapFewShotWithRandomSearch`).
- OPRO is extended from single-prompt optimization to multi-stage programs; Module-Level OPRO updates multiple module prompts in parallel, and grounding the proposer with bootstrapped demos, dataset summaries, program-code summaries, and generation tips improves instruction proposals.
- MIPRO (Multi-prompt Instruction PRoposal Optimizer) bootstraps demonstrations, proposes instruction candidates via an LM program, and jointly optimizes instructions and few-shot demonstration sets with Bayesian optimization using a surrogate model for credit assignment.
- LangProBe benchmark experiments show that bootstrapped demonstrations are critical, instruction optimization helps most on tasks with many conditional rules, and MIPRO (combining both) is often the most effective approach.
- DSPy has been used in production systems and SoTA research systems (PATH, IReRa, STORM, EDEN, Efficient Agents, ECG-Chat) and a U of Toronto MEDIQA competition win.

#### Relevance hooks
- Agent evaluation methodology: the lecture frames multi-hop RAG, tool use, and ReAct-style reasoning as modular LM programs and introduces LangProBe as a benchmark for language-model programs.
- RL reproducibility standards: LangProBe results are reported as averages over 5 runs with Wilcoxon signed-rank significance tests, an example of statistical rigor in optimizer benchmarking.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, or length bias.

#### Cited paper titles (verbatim only)
- "DSPY: COMPILING DECLARATIVE LANGUAGE MODEL CALLS INTO SELF-IMPROVING PIPELINES"
- "Large Language Models as Optimizers"
- "STORM: Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models"

Index row: f24 | dspy_lec.pdf | Omar Khattab | DSPy compiles modular LM programs into optimized prompts via signatures, few-shot bootstrapping, and grounded instruction optimizers such as MIPRO | ok
