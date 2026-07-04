### Multimodal Autonomous AI Agents
*Speaker:* Russ Salakhutdinov

#### Key claims / techniques
- **VisualWebArena** is introduced as a benchmark of realistic, visually grounded web tasks for multimodal agents, moving beyond text-and-HTML WebArena tasks.
- Raw HTML is insufficient for web agents: it is often minified/compressed, interactive elements render incorrectly, spatial layout is lost, and pages can exceed 100k tokens.
- Multimodal agents use a **VLM + Set-of-Marks (SoM)** representation over interactable elements to simplify cluttered accessibility trees.
- Agent architecture decomposes into high-level planning/reasoning, observation parsing, and low-level action generation (click, type, hover, stop).
- Agent performance suffers from **exponential error compounding**: even 99% single-step accuracy falls to ~73% at 30 steps and ~60.5% at 50 steps.
- A **best-first tree search** method improves agents by combining a baseline policy, environment backtracking, and a prompted multimodal LLM (GPT-4o) as a value function to score intermediate states.
- Search yields clear gains on both VisualWebArena and WebArena for Llama-3-70B and GPT-4o baselines; ablations show both the policy and value function have substantial headroom.
- **InSTA (Internet-Scale Training For Agents)** proposes a scalable pipeline where Llama generates synthetic agentic tasks from ~150k live web domains, then verifies task success automatically.
- Synthetic Llama-generated data outperforms human demonstrations on step accuracy (+89.5% on Mind2Web, +122.1% on WebLINX) and generalization (+149% WebLINX, +156.3% Mind2Web).
- **Plan-Seq-Learn (PSL)** extends the same planning/parsing/learning architecture to long-horizon robotic manipulation, using structured language plans and a single RL policy trained with local observations.

#### Relevance hooks
- Agent evaluation methodology: VisualWebArena defines execution-based, visually grounded web-agent benchmarks with realistic POMDP tasks.
- RL reproducibility / scaling: InSTA automates large-scale synthetic task generation and verification, and PSL trains RL policies for long-horizon robotics from language plans.
- Inference-time scaling: tree search demonstrates how test-time compute (value-guided backtracking) mitigates error accumulation in sequential agents.

#### Cited paper titles (verbatim only)
- REACT Synergizing Reasoning and Acting in Language Models
- Chain of Thought Prompting Elicits Reasoning in Large Language Models
- WebGPT: Browser-assisted Question-Answering with Human Feedback
- MIND2WEB: Towards a Generalist Agent for the Web
- Toolformer: Language Models can Teach Themselves to Use Tools
- ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings
- SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
- WebArena: A Realistic Web Environment for Building Autonomous Agents
- Autonomous Evaluation and Refinement of Digital Agents
- VisualWebArena -- Evaluating Multimodal Agents on Realistic Visual Web Tasks
- Tree Search for Language Model Agents
- Towards Internet-Scale Training For Agents
- Plan-Seq-Learn (PSL): Language Model Guided RL for Solving Long Horizon Robotics

Index row: sp25 | ruslan-multimodal.pdf | Russ Salakhutdinov | Lecture covers VisualWebArena, tree search for web agents, and internet-scale synthetic task generation for agent training. | ok
