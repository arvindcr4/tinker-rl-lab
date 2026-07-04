### On Memory, Reasoning, and Planning of Language Agents
*Speaker:* Yu Su

#### Key claims / techniques
- Defines language agents as contemporary AI agents with integrated LLMs that use language as a vehicle for reasoning and communication, contrasting an "LLM-first" scaffolding view with an "Agent-first" view that retains classic agent challenges (perception, world models, planning).
- Argues that token generation serves as a unified mechanism for perception, intuitive inference, and symbolic reasoning in LLMs, with self-reflection as a meta-reasoning action operating over an internal "inner monologue" environment.
- Presents HippoRAG, a neurobiologically inspired non-parametric long-term memory for LLMs modeled on hippocampal indexing theory; it partitions memory into neocortex (perception/language/reasoning), parahippocampus (working-memory bridge), and hippocampus (indexing/auto-associative memory) to improve pattern separation and completion over dense RAG.
- Notes that parametric continual learning in LLMs suffers from catastrophic forgetting and ripple effects, making structured non-parametric memory (e.g., HippoRAG, GraphRAG) a promising direction for sensemaking and multi-hop associativity.
- Presents Grokked Transformers as a mechanistic study of implicit (non-CoT) reasoning: standard GPT-2-scale transformers can learn implicit compositional reasoning, but only through a grokking phase transition from rote memorization to generalization.
- Finds that systematic generalization varies by reasoning type and depends more on data distribution than on raw scale; identifies parallel versus staged generalizing circuits via logit lens and causal tracing, and improves systematicity with cross-layer parameter sharing.
- Proposes WebDreamer, a model-based planner for web agents that uses LLM-predicted state transitions to simulate action outcomes before execution, avoiding irreversible real-world interactions and costly tree search.
- Reports that model-based planning (WebDreamer) is more accurate than reactive planning and more efficient than tree search on VisualWebArena.
- Frames modern language-agent planning as facing open-ended action spaces, fuzzy natural-language goals, and difficult or non-binary goal tests, illustrated by web agents (Mind2Web/SeeAct/UGround) and travel planning (TravelPlanner).
- Identifies open questions: reliable rewards for o1/R1-style reasoning, better world models, balancing reactive vs. model-based planning, endogenous vs. exogenous safety risks, and applications such as agentic search and science agents.

#### Relevance hooks
- Agent evaluation methodology: discusses web-agent benchmarks (Mind2Web, VisualWebArena, TravelPlanner) and contrasts reactive, tree-search, and model-based planning paradigms.
- RL post-training / GRPO relevance: hypothesizes that o1/R1-style long CoT emerges when a capable base model already knows basic reasoning constructs and RL learns to combine them, raising the question of how to obtain reliable rewards for such reasoning.
- None directly supported by the extracted text for ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"
- "Grokking of Implicit Relations in Transformers: A Mechanistic Journey to the Edge of Generalization"
- "Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization"
- "Is Your LLM Secretly a World Model of the Internet? Model-based Planning for Web Agents"
- "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency"
- "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks"
- "Mind2Web: Towards a Generalist Agent for the Web"
- "TravelPlanner: A Benchmark for Real-World Planning with Language Agents"
- "Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts"
- "Why Does New Knowledge Create Messy Ripple Effects in LLMs?"
- "Human-level control through deep reinforcement learning."
- "In Search of Memory: The Emergence of a New Science of Mind"
- "Language Agents: Foundations, Prospects, and Risks"

Index row: sp25 | language_agents_YuSu_Berkeley.pdf | Yu Su | Lecture surveys memory (HippoRAG), implicit reasoning via grokking, and model-based web planning (WebDreamer) for language agents. | ok
